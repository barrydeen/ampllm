use crate::glm5::state::Glm5State;
use crate::glm5::weights::{DevWeight, Glm5LayerWeights, Glm5ModelWeights};
use ampllm_cuda_kernels::ffi::{launch_linear_q8_1, launch_quantize_q8_1};
use ampllm_cuda_kernels::{copy_d2d_raw_async, glm5, set_device, CudaGraphExec, CudaStream};
use ampllm_gguf::{dequantize_row_to_f32, GgmlType, GgufFile};
use anyhow::Result;
use std::ffi::c_void;

pub fn enqueue_layer_m1(
    weights: &Glm5ModelWeights,
    state: &Glm5State,
    l: usize,
    phase_slot: usize,
    fuse_prev_ffn: *const f32,
    defer_hc_post: bool,
    has_prev_pf: bool,
    next_pf_layer: Option<usize>,
    stream: &CudaStream,
) -> Result<()> {
    let cfg = &weights.cfg;
    let lw: &Glm5LayerWeights = &weights.layers[l];
    let sc = if lw.device == 0 { &state.sc0 } else { &state.sc1 };
    let db_dev = if lw.device == 0 {
        state.doorbell.db0_dev_ptr
    } else {
        state.doorbell.db1_dev_ptr
    };
    let layer_evs = state.phase_prof.as_ref().map(|pp| {
        if lw.device == 0 {
            &pp.ev0[phase_slot & 1][l].ev
        } else {
            &pp.ev1[l - weights.split_layer].ev
        }
    });

    let n_embd = cfg.n_embd as i32;
    let hc = cfg.hc as i32;
    let norm_eps = cfg.rms_norm_eps;
    let hc_eps = cfg.hc_eps;
    let skip_dev: *const i32 = if lw.device == 0 {
        sc.skip_flag_dev.ptr.cast()
    } else {
        std::ptr::null()
    };

    unsafe {
        if let Some(ev) = layer_evs {
            ev[0].record(stream)?;
        }
        if !skip_dev.is_null() {
            glm5::skip_gate(db_dev, sc.skip_flag_dev.ptr.cast(), stream);
        }
        // 1. HC pre (with optional fused previous-layer HC post) + attn_norm + x_q8
        let (r_old_ptr, r_new_ptr, post_in_ptr, comb_in_ptr): (
            *const f32,
            *mut f32,
            *const f32,
            *const f32,
        ) = if fuse_prev_ffn.is_null() {
            (
                sc.r_buf[0].ptr.cast(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
            )
        } else {
            (
                sc.r_buf[1].ptr.cast(),
                sc.r_buf[0].ptr.cast(),
                sc.post.ptr.cast(),
                sc.comb.ptr.cast(),
            )
        };
        let hca = lw.hc_attn_fn.as_ref().unwrap();
        glm5::hc_fused_m1(
            fuse_prev_ffn,
            r_old_ptr,
            r_new_ptr,
            post_in_ptr,
            comb_in_ptr,
            sc.post.ptr.cast(),
            sc.comb.ptr.cast(),
            hca.ptr as *const c_void,
            hca.ggml_type as i32,
            lw.hc_attn_scale.unwrap() as *const f32,
            lw.hc_attn_base.unwrap() as *const f32,
            lw.attn_norm as *const f32,
            n_embd,
            hc,
            cfg.sinkhorn_iters as i32,
            norm_eps,
            hc_eps,
            sc.x_f32.ptr.cast(),
            sc.x_q8.ptr,
            sc.hc_part_dev.ptr.cast(),
            sc.hc_ctr_dev.ptr.cast(),
            skip_dev,
            stream,
        );
        if let Some(ev) = layer_evs {
            ev[1].record(stream)?;
        }

        // 2. Mixer
        if lw.is_kda {
            let kst = state.kda[l].as_ref().unwrap();
            let aq = lw.attn_q.as_ref().unwrap();
            let ak = lw.attn_k.as_ref().unwrap();
            let av = lw.attn_v.as_ref().unwrap();
            let fa = lw.ssm_f_a.as_ref().unwrap();
            let ga = lw.ssm_g_a.as_ref().unwrap();
            let sb = lw.ssm_beta.as_ref().unwrap();

            let w_ptrs = [
                aq.ptr as *const c_void,
                ak.ptr as *const c_void,
                av.ptr as *const c_void,
                fa.ptr as *const c_void,
                ga.ptr as *const c_void,
                sb.ptr as *const c_void,
            ];
            let y_ptrs = [
                sc.q_p.ptr.cast::<f32>(),
                sc.k_p.ptr.cast::<f32>(),
                sc.v_p.ptr.cast::<f32>(),
                sc.f_a.ptr.cast::<f32>(),
                sc.g_a.ptr.cast::<f32>(),
                sc.beta.ptr.cast::<f32>(),
            ];
            let w_types = [
                aq.ggml_type as i32,
                ak.ggml_type as i32,
                av.ggml_type as i32,
                fa.ggml_type as i32,
                ga.ggml_type as i32,
                sb.ggml_type as i32,
            ];
            let n_outs = [
                cfg.d_inner() as i32,
                cfg.d_inner() as i32,
                cfg.d_inner() as i32,
                cfg.kda_head_dim as i32,
                cfg.kda_head_dim as i32,
                cfg.n_head as i32,
            ];
            glm5::mv_batch_skip(
                &w_ptrs,
                &y_ptrs,
                &w_types,
                &n_outs,
                None,
                n_embd,
                sc.x_q8.ptr,
                sc.x_f32.ptr.cast(),
                skip_dev,
                stream,
            );

            let fb = lw.ssm_f_b.as_ref().unwrap();
            let gb = lw.ssm_g_b.as_ref().unwrap();
            if fb.ggml_type == GgmlType::Q8_0
                && gb.ggml_type == GgmlType::Q8_0
                && cfg.kda_head_dim == 128
                && cfg.d_conv == 4
            {
                glm5::kda_prep_m1(
                    sc.q_p.ptr.cast(),
                    sc.k_p.ptr.cast(),
                    sc.v_p.ptr.cast(),
                    lw.ssm_conv_q.unwrap() as *const f32,
                    lw.ssm_conv_k.unwrap() as *const f32,
                    lw.ssm_conv_v.unwrap() as *const f32,
                    kst.conv_q.ptr.cast(),
                    kst.conv_k.ptr.cast(),
                    kst.conv_v.ptr.cast(),
                    sc.qc.ptr.cast(),
                    sc.kc.ptr.cast(),
                    sc.vc.ptr.cast(),
                    sc.f_a.ptr.cast(),
                    sc.g_a.ptr.cast(),
                    fb.ptr as *const c_void,
                    gb.ptr as *const c_void,
                    lw.ssm_dt_bias.unwrap() as *const f32,
                    lw.ssm_a.unwrap() as *const f32,
                    cfg.kda_gate_lb,
                    cfg.n_head as i32,
                    sc.g1.ptr.cast(),
                    sc.g2.ptr.cast(),
                    skip_dev,
                    stream,
                );
                let out_f32_opt = if matches!(
                    lw.attn_output.ggml_type,
                    GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16
                ) {
                    sc.gated.ptr.cast()
                } else {
                    std::ptr::null_mut()
                };
                glm5::kda_rec_gate_q8_m1(
                    sc.qc.ptr.cast(),
                    sc.kc.ptr.cast(),
                    sc.vc.ptr.cast(),
                    sc.g1.ptr.cast(),
                    sc.beta.ptr.cast(),
                    kst.s_mat.ptr.cast(),
                    sc.g2.ptr.cast(),
                    lw.ssm_norm.unwrap() as *const f32,
                    norm_eps,
                    cfg.n_head as i32,
                    out_f32_opt,
                    sc.gated_q8.ptr,
                    skip_dev,
                    stream,
                );
            } else {
                glm5::kda_conv_silu_l2(
                    sc.q_p.ptr.cast(),
                    lw.ssm_conv_q.unwrap() as *const f32,
                    kst.conv_q.ptr.cast(),
                    cfg.d_inner() as i32,
                    cfg.d_conv as i32,
                    cfg.kda_head_dim as i32,
                    cfg.n_head as i32,
                    true,
                    1,
                    sc.qc.ptr.cast(),
                    stream,
                );
                glm5::kda_conv_silu_l2(
                    sc.k_p.ptr.cast(),
                    lw.ssm_conv_k.unwrap() as *const f32,
                    kst.conv_k.ptr.cast(),
                    cfg.d_inner() as i32,
                    cfg.d_conv as i32,
                    cfg.kda_head_dim as i32,
                    cfg.n_head as i32,
                    true,
                    1,
                    sc.kc.ptr.cast(),
                    stream,
                );
                glm5::kda_conv_silu_l2(
                    sc.v_p.ptr.cast(),
                    lw.ssm_conv_v.unwrap() as *const f32,
                    kst.conv_v.ptr.cast(),
                    cfg.d_inner() as i32,
                    cfg.d_conv as i32,
                    cfg.kda_head_dim as i32,
                    cfg.n_head as i32,
                    false,
                    1,
                    sc.vc.ptr.cast(),
                    stream,
                );

                // f_b & g_b (Q8_0, K = kda_head_dim = 128)
                launch_quantize_q8_1(
                    sc.f_a.ptr.cast(),
                    sc.qr_q8.ptr,
                    1,
                    cfg.kda_head_dim as i32,
                    stream.raw,
                );
                launch_quantize_q8_1(
                    sc.g_a.ptr.cast(),
                    sc.h_q8.ptr,
                    1,
                    cfg.kda_head_dim as i32,
                    stream.raw,
                );
                glm5::linear_q8_1_skip(
                    fb.ggml_type as i32,
                    fb.ptr as *const c_void,
                    sc.qr_q8.ptr,
                    sc.f_a.ptr.cast(),
                    sc.g_raw.ptr.cast(),
                    cfg.d_inner() as i32,
                    cfg.kda_head_dim as i32,
                    1,
                    skip_dev,
                    stream,
                );
                glm5::linear_q8_1_skip(
                    gb.ggml_type as i32,
                    gb.ptr as *const c_void,
                    sc.h_q8.ptr,
                    sc.g_a.ptr.cast(),
                    sc.g2.ptr.cast(),
                    cfg.d_inner() as i32,
                    cfg.kda_head_dim as i32,
                    1,
                    skip_dev,
                    stream,
                );

                glm5::kda_gate_beta(
                    sc.g_raw.ptr.cast(),
                    lw.ssm_dt_bias.unwrap() as *const f32,
                    lw.ssm_a.unwrap() as *const f32,
                    cfg.kda_gate_lb,
                    cfg.kda_head_dim as i32,
                    cfg.n_head as i32,
                    1,
                    sc.g1.ptr.cast(),
                    sc.beta.ptr.cast(),
                    stream,
                );
                glm5::kda_recurrence_skip(
                    sc.qc.ptr.cast(),
                    sc.kc.ptr.cast(),
                    sc.vc.ptr.cast(),
                    sc.g1.ptr.cast(),
                    sc.beta.ptr.cast(),
                    kst.s_mat.ptr.cast(),
                    cfg.n_head as i32,
                    cfg.kda_head_dim as i32,
                    1,
                    sc.scan.ptr.cast(),
                    skip_dev,
                    stream,
                );
                glm5::kda_out_gate_q8(
                    sc.scan.ptr.cast(),
                    lw.ssm_norm.unwrap() as *const f32,
                    sc.g2.ptr.cast(),
                    cfg.kda_head_dim as i32,
                    cfg.n_head as i32,
                    1,
                    norm_eps,
                    sc.gated.ptr.cast(),
                    sc.gated_q8.ptr,
                    stream,
                );
            }
            glm5::linear_q8_1_skip(
                lw.attn_output.ggml_type as i32,
                lw.attn_output.ptr as *const c_void,
                sc.gated_q8.ptr,
                sc.gated.ptr.cast(),
                sc.mixer.ptr.cast(),
                n_embd,
                cfg.d_inner() as i32,
                1,
                skip_dev,
                stream,
            );
        } else {
            let dst = state.dsa[l].as_ref().unwrap();
            let qa = lw.attn_q_a.as_ref().unwrap();
            let kva = lw.attn_kv_a_mqa.as_ref().unwrap();
            let ik = lw.idx_k.as_ref().unwrap();
            let ig = lw.idx_gate.as_ref().unwrap();
            let ip = lw.idx_proj.as_ref().unwrap();

            let prescale = 1.0f32 / ((cfg.idx_key_dim * cfg.idx_heads) as f32).sqrt();
            let w_ptrs = [
                qa.ptr as *const c_void,
                kva.ptr as *const c_void,
                ik.ptr as *const c_void,
                ig.ptr as *const c_void,
                ip.ptr as *const c_void,
            ];
            let y_ptrs = [
                sc.qr.ptr.cast::<f32>(),
                sc.kv.ptr.cast::<f32>(),
                sc.ik_tmp.ptr.cast::<f32>(),
                sc.ig_tmp.ptr.cast::<f32>(),
                sc.iw.ptr.cast::<f32>(),
            ];
            let w_types = [
                qa.ggml_type as i32,
                kva.ggml_type as i32,
                ik.ggml_type as i32,
                ig.ggml_type as i32,
                ip.ggml_type as i32,
            ];
            let n_outs = [
                cfg.q_lora_rank as i32,
                cfg.kv_lora_rank as i32,
                cfg.idx_key_dim as i32,
                cfg.idx_key_dim as i32,
                cfg.idx_heads as i32,
            ];
            let scales = [1.0f32, 1.0, 1.0, 1.0, prescale];
            glm5::mv_batch_skip(
                &w_ptrs,
                &y_ptrs,
                &w_types,
                &n_outs,
                Some(&scales),
                n_embd,
                sc.x_q8.ptr,
                sc.x_f32.ptr.cast(),
                skip_dev,
                stream,
            );

            let ram_lat_dev = dst
                .ram_lat8
                .as_ref()
                .map_or(std::ptr::null_mut(), |b| {
                    b.dev_ptr_for_device(lw.device).unwrap_or(std::ptr::null_mut())
                });
            glm5::dsa_prep_m1_dyn(
                sc.qr.ptr.cast(),
                lw.attn_q_a_norm.unwrap() as *const f32,
                sc.qr_q8.ptr,
                cfg.q_lora_rank as i32,
                sc.kv.ptr.cast(),
                lw.attn_kv_a_norm.unwrap() as *const f32,
                dst.vram_lat8.ptr,
                ram_lat_dev,
                weights.dsa_res_cap as i32,
                cfg.kv_lora_rank as i32,
                sc.ik_tmp.ptr.cast(),
                lw.idx_k_norm_w.unwrap() as *const f32,
                lw.idx_k_norm_b.unwrap() as *const f32,
                dst.ik_ring.ptr.cast(),
                sc.ig_tmp.ptr.cast(),
                dst.ig_ring.ptr.cast(),
                lw.idx_ape.unwrap() as *const f32,
                cfg.idx_key_dim as i32,
                cfg.idx_kpool as i32,
                516,
                sc.pos_arr_dev.ptr.cast(),
                dst.pooled_f16.ptr,
                norm_eps,
                skip_dev,
                stream,
            );

            let qb = lw.attn_q_b.as_ref().unwrap();
            let iqb = lw.idx_q_b.as_ref().unwrap();
            let q_w_ptrs = [qb.ptr as *const c_void, iqb.ptr as *const c_void];
            let q_y_ptrs = [sc.q_full.ptr.cast::<f32>(), sc.iq.ptr.cast::<f32>()];
            let q_wt = [qb.ggml_type as i32, iqb.ggml_type as i32];
            let q_no = [
                (cfg.n_head * cfg.qk_nope_head_dim) as i32,
                (cfg.idx_heads * cfg.idx_key_dim) as i32,
            ];
            glm5::mv_batch_skip(
                &q_w_ptrs,
                &q_y_ptrs,
                &q_wt,
                &q_no,
                None,
                cfg.q_lora_rank as i32,
                sc.qr_q8.ptr,
                sc.qr.ptr.cast(),
                skip_dev,
                stream,
            );

            glm5::dsa_q_absorb(
                lw.attn_k_b_f16.unwrap() as *const c_void,
                GgmlType::F16 as i32,
                sc.q_full.ptr.cast(),
                sc.q_abs.ptr.cast(),
                cfg.qk_nope_head_dim as i32,
                cfg.kv_lora_rank as i32,
                cfg.n_head as i32,
                1,
                stream,
            );

            let top_pools_cap = cfg.idx_top_k / cfg.idx_kpool;
            let n_sel = cfg.idx_top_k + cfg.idx_kpool - 1;
            glm5::dsa_score_select_dyn(
                sc.iq.ptr.cast(),
                dst.pooled_f16.ptr,
                sc.iw.ptr.cast(),
                cfg.idx_key_dim as i32,
                cfg.idx_heads as i32,
                cfg.idx_kpool as i32,
                top_pools_cap as i32,
                cfg.idx_select_tail,
                n_sel as i32,
                weights.pool_cap as i32,
                sc.pos_arr_dev.ptr.cast(),
                sc.dsa_scores.ptr.cast(),
                sc.dsa_cells.ptr.cast(),
                stream,
            );

            let attn_out_f32 = if matches!(
                lw.attn_output.ggml_type,
                GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16
            ) {
                sc.dsa_attn.ptr.cast()
            } else {
                std::ptr::null_mut()
            };
            glm5::dsa_mla_vproj_q8_m1(
                sc.q_abs.ptr.cast(),
                dst.vram_lat8.ptr,
                ram_lat_dev,
                weights.dsa_res_cap as i32,
                sc.dsa_cells.ptr.cast(),
                lw.attn_v_b_f16.unwrap() as *const c_void,
                GgmlType::F16 as i32,
                cfg.kv_lora_rank as i32,
                cfg.n_head as i32,
                cfg.v_head_dim as i32,
                cfg.qk_nope_head_dim as i32,
                n_sel as i32,
                sc.q_abs.ptr.cast(),
                sc.mma_b_f16.ptr.cast(),
                attn_out_f32,
                sc.dsa_attn_q8.ptr,
                skip_dev,
                stream,
            );

            let v_tot = (cfg.n_head * cfg.v_head_dim) as i32;
            glm5::linear_q8_1_skip(
                lw.attn_output.ggml_type as i32,
                lw.attn_output.ptr as *const c_void,
                sc.dsa_attn_q8.ptr,
                sc.dsa_attn.ptr.cast(),
                sc.mixer.ptr.cast(),
                n_embd,
                v_tot,
                1,
                skip_dev,
                stream,
            );
        }
        if let Some(ev) = layer_evs {
            ev[2].record(stream)?;
        }

        // 3+4. Fused HC post (r_buf[0] -> r_buf[1]) + HC pre (ffn half) + ffn_norm + x_q8
        let hcf = lw.hc_ffn_fn.as_ref().unwrap();
        glm5::hc_fused_m1(
            sc.mixer.ptr.cast(),
            sc.r_buf[0].ptr.cast(),
            sc.r_buf[1].ptr.cast(),
            sc.post.ptr.cast(),
            sc.comb.ptr.cast(),
            sc.post.ptr.cast(),
            sc.comb.ptr.cast(),
            hcf.ptr as *const c_void,
            hcf.ggml_type as i32,
            lw.hc_ffn_scale.unwrap() as *const f32,
            lw.hc_ffn_base.unwrap() as *const f32,
            lw.ffn_norm as *const f32,
            n_embd,
            hc,
            cfg.sinkhorn_iters as i32,
            norm_eps,
            hc_eps,
            sc.x_f32.ptr.cast(),
            sc.x_q8.ptr,
            sc.hc_part_dev.ptr.cast(),
            sc.hc_ctr_dev.ptr.cast(),
            skip_dev,
            stream,
        );
        if let Some(ev) = layer_evs {
            ev[3].record(stream)?;
        }

        // 5. FFN
        if !lw.is_moe {
            let fg = lw.ffn_gate.as_ref().unwrap();
            let fu = lw.ffn_up.as_ref().unwrap();
            let fd = lw.ffn_down.as_ref().unwrap();
            let w_ptrs = [fg.ptr as *const c_void, fu.ptr as *const c_void];
            let y_ptrs = [sc.gate.ptr.cast::<f32>(), sc.up.ptr.cast::<f32>()];
            let wt = [fg.ggml_type as i32, fu.ggml_type as i32];
            let no = [cfg.n_ff_dense as i32, cfg.n_ff_dense as i32];
            glm5::mv_batch_skip(
                &w_ptrs,
                &y_ptrs,
                &wt,
                &no,
                None,
                n_embd,
                sc.x_q8.ptr,
                sc.x_f32.ptr.cast(),
                skip_dev,
                stream,
            );
            glm5::swiglu_clamp_q8(
                sc.gate.ptr.cast(),
                sc.up.ptr.cast(),
                cfg.swiglu_clamp_shexp,
                cfg.n_ff_dense as i32,
                std::ptr::null_mut(),
                sc.h_q8.ptr,
                1,
                stream,
            );
            glm5::linear_q8_1_skip(
                fd.ggml_type as i32,
                fd.ptr as *const c_void,
                sc.h_q8.ptr,
                sc.h_f32.ptr.cast(),
                sc.sh_out.ptr.cast(),
                n_embd,
                cfg.n_ff_dense as i32,
                1,
                skip_dev,
                stream,
            );
            if !defer_hc_post {
                glm5::hc_post_add(
                    sc.r_buf[1].ptr.cast(),
                    sc.post.ptr.cast(),
                    sc.comb.ptr.cast(),
                    sc.sh_out.ptr.cast(),
                    std::ptr::null(),
                    std::ptr::null(),
                    sc.r_buf[0].ptr.cast(),
                    n_embd,
                    hc,
                    1,
                    stream,
                );
            }
            if let Some(ev) = layer_evs {
                ev[4].record(stream)?;
            }
        } else {
            let ginp = lw.ffn_gate_inp.as_ref().unwrap();
            let gs = lw.ffn_gate_sh.as_ref().unwrap();
            let us = lw.ffn_up_sh.as_ref().unwrap();
            let ds = lw.ffn_down_sh.as_ref().unwrap();
            let n_ff_sh = (cfg.n_ff_shexp * cfg.n_shared_expert) as i32;

            let mut w_moe_in = [
                ginp.ptr as *const c_void,
                gs.ptr as *const c_void,
                us.ptr as *const c_void,
                std::ptr::null(),
            ];
            let mut y_moe_in = [
                sc.moe_logits.ptr.cast::<f32>(),
                sc.gate.ptr.cast::<f32>(),
                sc.up.ptr.cast::<f32>(),
                std::ptr::null_mut(),
            ];
            let mut wt_moe_in = [
                ginp.ggml_type as i32,
                gs.ggml_type as i32,
                us.ggml_type as i32,
                0,
            ];
            let mut no_moe_in = [cfg.n_expert as i32, n_ff_sh, n_ff_sh, 0];
            let mut n_moe_jobs = 3usize;
            if let Some(nl) = next_pf_layer {
                let ginp_n = weights.layers[nl].ffn_gate_inp.as_ref().unwrap();
                w_moe_in[3] = ginp_n.ptr as *const c_void;
                y_moe_in[3] = sc.pf_logits.ptr.cast::<f32>();
                wt_moe_in[3] = ginp_n.ggml_type as i32;
                no_moe_in[3] = cfg.n_expert as i32;
                n_moe_jobs = 4;
            }
            glm5::mv_batch_skip(
                &w_moe_in[..n_moe_jobs],
                &y_moe_in[..n_moe_jobs],
                &wt_moe_in[..n_moe_jobs],
                &no_moe_in[..n_moe_jobs],
                None,
                n_embd,
                sc.x_q8.ptr,
                sc.x_f32.ptr.cast(),
                skip_dev,
                stream,
            );
            if let Some(ev) = layer_evs {
                ev[4].record(stream)?;
            }

            let exp_seq_ptr = sc.db_expected_seq.ptr.cast::<u64>().add(l);
            let spares_ptr = lw
                .spares_dev
                .as_ref()
                .map_or(std::ptr::null_mut(), |b| b.ptr.cast());
            let (pf_in_eids, pf_in_ptrs): (*const i32, *const u64) = if has_prev_pf {
                (
                    sc.pf_eids[l & 1].ptr.cast(),
                    sc.pf_ptrs[l & 1].ptr.cast(),
                )
            } else {
                (std::ptr::null(), std::ptr::null())
            };
            let (
                pf_logits_next,
                pf_bias_next,
                vram_ptrs_next,
                ram_ptrs_next,
                max_pf,
                pf_out_src,
                pf_out_dst,
                pf_out_eids,
                pf_out_ptrs,
                pf_out_n,
            ): (
                *const f32,
                *const f32,
                *const u64,
                *const u64,
                i32,
                *mut u64,
                *mut u64,
                *mut i32,
                *mut u64,
                *mut i32,
            ) = if let Some(nl) = next_pf_layer {
                let nlw = &weights.layers[nl];
                let slot = nl & 1;
                (
                    sc.pf_logits.ptr.cast(),
                    nlw.exp_probs_bias.unwrap() as *const f32,
                    nlw.vram_ptrs_dev.as_ref().unwrap().ptr.cast(),
                    nlw.ram_ptrs_dev.as_ref().unwrap().ptr.cast(),
                    state.max_pf,
                    sc.pf_src[slot].ptr.cast(),
                    sc.pf_dst[slot].ptr.cast(),
                    sc.pf_eids[slot].ptr.cast(),
                    sc.pf_ptrs[slot].ptr.cast(),
                    sc.pf_n[slot].ptr.cast(),
                )
            } else {
                (
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            glm5::router_dual_lane_pf(
                sc.moe_logits.ptr.cast(),
                lw.exp_probs_bias.unwrap() as *const f32,
                cfg.n_expert as i32,
                cfg.n_expert_used as i32,
                cfg.expert_weights_scale,
                cfg.expert_weights_norm,
                lw.vram_ptrs_dev.as_ref().unwrap().ptr.cast(),
                lw.ram_ptrs_dev.as_ref().unwrap().ptr.cast(),
                spares_ptr,
                lw.k_spares as i32,
                lw.dcnt_dev.as_ref().unwrap().ptr.cast(),
                sc.moe_route_cnt_dev.ptr.cast(),
                sc.moe_scratch_ptrs_dev.ptr.cast(),
                state.cpu_plan,
                state.promote_min,
                cfg.n_expert_used as i32,
                sc.moe_plan_ptrs_dev.ptr.cast(),
                sc.moe_fetch_src_dev.ptr.cast(),
                sc.x_f32.ptr.cast(),
                n_embd,
                cfg.n_ff_exp as i32,
                l as i32,
                lw.gu_type as i32,
                lw.d_type as i32,
                cfg.swiglu_clamp_exp,
                lw.u_off as u32,
                lw.d_off as u32,
                sc.gate.ptr.cast(),
                sc.up.ptr.cast(),
                cfg.swiglu_clamp_shexp,
                n_ff_sh,
                sc.h_q8.ptr,
                sc.moe_ids.ptr.cast(),
                sc.moe_rw.ptr.cast(),
                db_dev,
                sc.db_seq_state.ptr.cast(),
                exp_seq_ptr,
                sc.pos_arr_dev.ptr.cast(),
                skip_dev,
                pf_in_eids,
                pf_in_ptrs,
                pf_logits_next,
                pf_bias_next,
                vram_ptrs_next,
                ram_ptrs_next,
                max_pf,
                pf_out_src,
                pf_out_dst,
                pf_out_eids,
                pf_out_ptrs,
                pf_out_n,
                stream,
            );
            if let Some(ev) = layer_evs {
                ev[5].record(stream)?;
            }

            let fork_prefetch = |nl: usize| -> Result<()> {
                let slot = nl & 1;
                sc.ev_pf_start.record(stream)?;
                sc.pf_stream.wait_event(&sc.ev_pf_start)?;
                glm5::moe_prefetch(
                    sc.pf_src[slot].ptr.cast(),
                    sc.pf_dst[slot].ptr.cast(),
                    sc.pf_n[slot].ptr.cast(),
                    weights.layers[nl].blob_stride,
                    &sc.pf_stream,
                );
                sc.ev_pf_done[slot].record(&sc.pf_stream)?;
                Ok(())
            };

            let gu_rb = lw.gu_type.row_bytes(cfg.n_embd);
            let d_rb = lw.d_type.row_bytes(cfg.n_ff_exp);
            let ds_rb = ds.ggml_type.row_bytes(n_ff_sh as usize);
            let fetch_overlap = {
                static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                state.max_pf == 0
                    && *ON.get_or_init(|| {
                        std::env::var("AMPLLM_FETCH_OVERLAP")
                            .map(|v| v != "0")
                            .unwrap_or(true)
                    })
            };
            let fetch_blocks = {
                static NB: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
                *NB.get_or_init(|| {
                    std::env::var("AMPLLM_FETCH_BLOCKS")
                        .ok()
                        .and_then(|s| s.parse::<i32>().ok())
                        .unwrap_or(16)
                        .clamp(4, 64)
                })
            };

            if fetch_overlap {
                sc.ev_pf_start.record(stream)?;
                sc.pf_stream.wait_event(&sc.ev_pf_start)?;
                glm5::moe_fetch_nblocks(
                    sc.moe_fetch_src_dev.ptr.cast(),
                    sc.moe_plan_ptrs_dev.ptr.cast(),
                    cfg.n_expert_used as i32,
                    lw.blob_stride,
                    fetch_blocks,
                    &sc.pf_stream,
                );
                sc.ev_pf_done[0].record(&sc.pf_stream)?;

                glm5::moe_plan_filtered(
                    lw.gu_type as i32,
                    lw.d_type as i32,
                    sc.moe_plan_ptrs_dev.ptr.cast(),
                    lw.u_off as u32,
                    lw.d_off as u32,
                    gu_rb,
                    d_rb,
                    sc.x_q8.ptr,
                    sc.moe_rw.ptr.cast(),
                    sc.moe_inter_q8.ptr,
                    ds.ggml_type as i32,
                    ds.ptr as *const c_void,
                    ds_rb,
                    sc.h_q8.ptr,
                    n_ff_sh,
                    sc.sh_out.ptr.cast(),
                    sc.gpu_moe_out.ptr.cast(),
                    n_embd,
                    cfg.n_ff_exp as i32,
                    cfg.n_expert_used as i32,
                    1,
                    cfg.swiglu_clamp_exp,
                    skip_dev,
                    sc.moe_fetch_src_dev.ptr.cast(),
                    1,
                    stream,
                );

                stream.wait_event(&sc.ev_pf_done[0])?;
                if let Some(ev) = layer_evs {
                    ev[6].record(stream)?;
                }

                glm5::moe_plan_filtered(
                    lw.gu_type as i32,
                    lw.d_type as i32,
                    sc.moe_plan_ptrs_dev.ptr.cast(),
                    lw.u_off as u32,
                    lw.d_off as u32,
                    gu_rb,
                    d_rb,
                    sc.x_q8.ptr,
                    sc.moe_rw.ptr.cast(),
                    sc.moe_inter_q8.ptr,
                    ds.ggml_type as i32,
                    ds.ptr as *const c_void,
                    ds_rb,
                    sc.h_q8.ptr,
                    n_ff_sh,
                    sc.sh_out.ptr.cast(),
                    sc.gpu_moe_out.ptr.cast(),
                    n_embd,
                    cfg.n_ff_exp as i32,
                    cfg.n_expert_used as i32,
                    1,
                    cfg.swiglu_clamp_exp,
                    skip_dev,
                    sc.moe_fetch_src_dev.ptr.cast(),
                    2,
                    stream,
                );
                if let Some(ev) = layer_evs {
                    ev[7].record(stream)?;
                }
            } else {
                glm5::moe_fetch(
                    sc.moe_fetch_src_dev.ptr.cast(),
                    sc.moe_plan_ptrs_dev.ptr.cast(),
                    cfg.n_expert_used as i32,
                    lw.blob_stride,
                    stream,
                );
                if let Some(ev) = layer_evs {
                    ev[6].record(stream)?;
                }
                if state.pf_at_fetch {
                    if let Some(nl) = next_pf_layer {
                        fork_prefetch(nl)?;
                    }
                }

                if has_prev_pf {
                    stream.wait_event(&sc.ev_pf_done[l & 1])?;
                }

                glm5::moe_plan(
                    lw.gu_type as i32,
                    lw.d_type as i32,
                    sc.moe_plan_ptrs_dev.ptr.cast(),
                    lw.u_off as u32,
                    lw.d_off as u32,
                    gu_rb,
                    d_rb,
                    sc.x_q8.ptr,
                    sc.moe_rw.ptr.cast(),
                    sc.moe_inter_q8.ptr,
                    ds.ggml_type as i32,
                    ds.ptr as *const c_void,
                    ds_rb,
                    sc.h_q8.ptr,
                    n_ff_sh,
                    sc.sh_out.ptr.cast(),
                    sc.gpu_moe_out.ptr.cast(),
                    n_embd,
                    cfg.n_ff_exp as i32,
                    cfg.n_expert_used as i32,
                    1,
                    cfg.swiglu_clamp_exp,
                    skip_dev,
                    stream,
                );
                if let Some(ev) = layer_evs {
                    ev[7].record(stream)?;
                }
            }

            if defer_hc_post {
                glm5::doorbell_wait_add(
                    db_dev,
                    exp_seq_ptr,
                    sc.gpu_moe_out.ptr.cast(),
                    n_embd,
                    stream,
                );
            } else {
                glm5::doorbell_wait_hc_post(
                    db_dev,
                    exp_seq_ptr,
                    sc.r_buf[1].ptr.cast(),
                    sc.post.ptr.cast(),
                    sc.comb.ptr.cast(),
                    sc.gpu_moe_out.ptr.cast(),
                    std::ptr::null(),
                    sc.r_buf[0].ptr.cast(),
                    n_embd,
                    hc,
                    1,
                    stream,
                );
            }
            if let Some(ev) = layer_evs {
                ev[8].record(stream)?;
            }
            if !state.pf_at_fetch {
                if let Some(nl) = next_pf_layer {
                    fork_prefetch(nl)?;
                }
            }
        }
    }
    Ok(())
}

pub fn dequant_embed_to_host(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    token: u32,
    dst: &mut [f32],
) -> Result<()> {
    let raw = gguf.tensor_bytes("token_embd.weight")?;
    let rb = weights.token_embd_row_bytes;
    let off = (token as usize) * rb;
    dequantize_row_to_f32(weights.token_embd_type, &raw[off..off + rb], dst);
    Ok(())
}

fn enqueue_gpu0_body_m1(weights: &Glm5ModelWeights, state: &Glm5State, phase_slot: usize) -> Result<()> {
    unsafe {
        glm5::embed_replicate(
            state.sc0.emb.ptr.cast(),
            state.sc0.r_buf[0].ptr.cast(),
            weights.cfg.n_embd as i32,
            weights.cfg.hc as i32,
            1,
            &state.sc0.stream,
        );
    }
    if let Some(ref pp) = state.phase_prof {
        pp.ev0_emb_end[phase_slot & 1].record(&state.sc0.stream)?;
    }
    enqueue_layer_range_m1(weights, state, 0, weights.split_layer, phase_slot & 1, true)?;
    Ok(())
}

/// Dispatch `[lb, le)` M=1 layers with fused cross-layer HC-post -> next-layer HC-pre and lookahead MoE prefetch.
fn enqueue_layer_range_m1(
    weights: &Glm5ModelWeights,
    state: &Glm5State,
    lb: usize,
    le: usize,
    phase_slot: usize,
    is_gpu0: bool,
) -> Result<()> {
    let sc = if is_gpu0 { &state.sc0 } else { &state.sc1 };
    for l in lb..le {
        let lw = &weights.layers[l];
        let fuse_prev_ffn: *const f32 = if l > lb {
            if weights.layers[l - 1].is_moe {
                sc.gpu_moe_out.ptr.cast()
            } else {
                sc.sh_out.ptr.cast()
            }
        } else {
            std::ptr::null()
        };
        let defer_hc_post = l + 1 < le;
        let has_prev_pf = lw.is_moe && state.max_pf > 0 && l > lb && weights.layers[l - 1].is_moe;
        let next_pf_layer =
            if lw.is_moe && state.max_pf > 0 && l + 1 < le && weights.layers[l + 1].is_moe {
                Some(l + 1)
            } else {
                None
            };
        enqueue_layer_m1(
            weights,
            state,
            l,
            phase_slot,
            fuse_prev_ffn,
            defer_hc_post,
            has_prev_pf,
            next_pf_layer,
            &sc.stream,
        )?;
    }
    Ok(())
}

#[inline]
fn prof_enabled() -> bool {
    static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROF.get_or_init(|| std::env::var_os("AMPLLM_PROFILE").is_some())
}

pub fn run_gpu0_half_m1(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    token: u32,
    pos: usize,
    use_cuda_graph: bool,
) -> Result<()> {
    let n_embd = weights.cfg.n_embd;
    let slot = pos & 1;
    let emb_off = slot * n_embd;
    dequant_embed_to_host(gguf, weights, token, &mut state.hop_emb_pin0.as_mut_slice()[emb_off..emb_off + n_embd])?;
    state.pos_pin0.as_mut_slice()[slot] = pos as i32;

    set_device(0)?;
    let prof = prof_enabled();
    if prof {
        state.prof_g0_start[slot].record(&state.sc0.stream)?;
    }
    if let Some(ref pp) = state.phase_prof {
        pp.ev0_h2d_start[slot].record(&state.sc0.stream)?;
    }
    let pos_bytes = unsafe {
        std::slice::from_raw_parts(state.pos_pin0.host_ptr().add(slot).cast::<u8>(), 4)
    };
    state.sc0.pos_arr_dev.copy_from_host_async(pos_bytes, &state.sc0.stream)?;
    let emb_bytes = unsafe {
        std::slice::from_raw_parts(state.hop_emb_pin0.host_ptr().add(emb_off).cast::<u8>(), n_embd * 4)
    };
    state.sc0.emb.copy_from_host_async(emb_bytes, &state.sc0.stream)?;
    if let Some(ref pp) = state.phase_prof {
        pp.ev0_h2d_end[slot].record(&state.sc0.stream)?;
    }

    if use_cuda_graph {
        if state.decode_graph0.is_none() {
            let g = CudaGraphExec::capture(&state.sc0.stream, || enqueue_gpu0_body_m1(weights, state, 0))?;
            state.decode_graph0 = Some(g);
        }
        state.decode_graph0.as_ref().unwrap().launch(&state.sc0.stream)?;
    } else {
        enqueue_gpu0_body_m1(weights, state, slot)?;
    }

    if state.p2p_ok {
        if let Some(ref pp) = state.phase_prof {
            pp.ev0_hop_end[slot].record(&state.sc0.stream)?;
        }
        if prof {
            state.prof_g0_end[slot].record(&state.sc0.stream)?;
        }
        state.ev_g0_done[slot].record(&state.sc0.stream)?;
    } else {
        unsafe {
            glm5::copy_to_mapped_f32(
                state.sc0.r_buf[0].ptr.cast(),
                state.hop_pin_dev0[slot] as *mut f32,
                (weights.cfg.hc * n_embd) as i32,
                &state.sc0.stream,
            );
        }
        if let Some(ref pp) = state.phase_prof {
            pp.ev0_hop_end[slot].record(&state.sc0.stream)?;
        }
        if prof {
            state.prof_g0_end[slot].record(&state.sc0.stream)?;
        }
        state.ev_g0_done[slot].record(&state.sc0.stream)?;
    }
    Ok(())
}

pub fn hop_gpu0_to_gpu1_r0(
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    m: usize,
    hop_slot: usize,
) -> Result<()> {
    let hop_bytes = m * weights.cfg.hc * weights.cfg.n_embd * 4;
    let slot = hop_slot & 1;
    set_device(1)?;
    state.sc1.stream.wait_event(&state.ev_g0_done[slot])?;
    if m == 1 {
        if let Some(ref pp) = state.phase_prof {
            pp.ev1_hop_start.record(&state.sc1.stream)?;
        }
    }
    if state.p2p_ok {
        state
            .sc1
            .r_buf[0]
            .copy_from_device_async(&state.sc0.r_buf[0], hop_bytes, &state.sc1.stream)?;
        if m == 1 {
            if let Some(ref pp) = state.phase_prof {
                pp.ev1_hop_end.record(&state.sc1.stream)?;
            }
        }
        state.ev_g1_done.record(&state.sc1.stream)?;
    } else {
        let h_slice = unsafe {
            std::slice::from_raw_parts(state.hop_pin[slot].host_ptr().cast::<u8>(), hop_bytes)
        };
        state.sc1.r_buf[0].copy_from_host_async(h_slice, &state.sc1.stream)?;
        if m == 1 {
            if let Some(ref pp) = state.phase_prof {
                pp.ev1_hop_end.record(&state.sc1.stream)?;
            }
        }
    }
    Ok(())
}

fn enqueue_gpu1_body_m1(
    weights: &Glm5ModelWeights,
    state: &Glm5State,
    need_logits: bool,
) -> Result<()> {
    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;

    enqueue_layer_range_m1(weights, state, weights.split_layer, cfg.n_layers, 0, false)?;

    unsafe {
        if let Some(ref pp) = state.phase_prof {
            pp.ev1_head_start.record(&state.sc1.stream)?;
        }
        glm5::mean_streams_norm_q8(
            state.sc1.r_buf[0].ptr.cast(),
            weights.output_norm_dev.ptr.cast(),
            cfg.rms_norm_eps,
            n_embd as i32,
            cfg.hc as i32,
            state.head_x.ptr.cast(),
            state.head_xq.ptr,
            1,
            &state.sc1.stream,
        );
        if let Some(ref pp) = state.phase_prof {
            pp.ev1_norm_end.record(&state.sc1.stream)?;
        }
        launch_linear_q8_1(
            weights.output_weight_type as i32,
            weights.output_weight_dev.ptr,
            state.head_xq.ptr,
            state.head_x.ptr.cast(),
            state.logits_dev.ptr.cast(),
            cfg.n_vocab as i32,
            n_embd as i32,
            1,
            state.sc1.stream.raw,
        );
        if let Some(ref pp) = state.phase_prof {
            pp.ev1_logits_end.record(&state.sc1.stream)?;
        }
        if !need_logits {
            glm5::argmax(
                state.logits_dev.ptr.cast(),
                cfg.n_vocab as i32,
                cfg.n_vocab as i32,
                std::ptr::null(),
                state.argmax_pv.ptr.cast(),
                state.argmax_pi.ptr.cast(),
                state.tok_host_dev1 as *mut i32,
                &state.sc1.stream,
            );
        }
        if let Some(ref pp) = state.phase_prof {
            pp.ev1_head_end.record(&state.sc1.stream)?;
        }
    }
    Ok(())
}

pub fn enqueue_gpu1_half_and_head_m1(
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    pos: usize,
    need_logits: bool,
    use_cuda_graph: bool,
) -> Result<()> {
    let cfg = &weights.cfg;
    hop_gpu0_to_gpu1_r0(weights, state, 1, pos & 1)?;

    let pos_slot = pos & 1;
    state.pos_pin1.as_mut_slice()[pos_slot] = pos as i32;
    set_device(1)?;
    let prof = prof_enabled();
    if prof {
        state.prof_g1_start.record(&state.sc1.stream)?;
    }
    let pos_bytes = unsafe {
        std::slice::from_raw_parts(state.pos_pin1.host_ptr().add(pos_slot).cast::<u8>(), 4)
    };
    state.sc1.pos_arr_dev.copy_from_host_async(pos_bytes, &state.sc1.stream)?;
    if let Some(ref pp) = state.phase_prof {
        pp.ev1_pos_end.record(&state.sc1.stream)?;
    }

    if use_cuda_graph {
        if need_logits {
            if state.decode_graph1_logits.is_none() {
                let g = CudaGraphExec::capture(&state.sc1.stream, || enqueue_gpu1_body_m1(weights, state, true))?;
                state.decode_graph1_logits = Some(g);
            }
            state.decode_graph1_logits.as_ref().unwrap().launch(&state.sc1.stream)?;
        } else {
            if state.decode_graph1_greedy.is_none() {
                let g = CudaGraphExec::capture(&state.sc1.stream, || enqueue_gpu1_body_m1(weights, state, false))?;
                state.decode_graph1_greedy = Some(g);
            }
            state.decode_graph1_greedy.as_ref().unwrap().launch(&state.sc1.stream)?;
        }
    } else {
        enqueue_gpu1_body_m1(weights, state, need_logits)?;
    }

    if need_logits {
        unsafe {
            let lh_slice = std::slice::from_raw_parts_mut(
                state.target_logits_pin.host_ptr().cast::<u8>(),
                cfg.n_vocab * 4,
            );
            state.logits_dev.copy_to_host_async(lh_slice, &state.sc1.stream)?;
        }
    }
    if let Some(ref pp) = state.phase_prof {
        pp.ev1_d2h_end.record(&state.sc1.stream)?;
    }
    if prof {
        state.prof_g1_end.record(&state.sc1.stream)?;
    }
    Ok(())
}

pub fn finish_gpu1_head_sync_m1(state: &mut Glm5State, pos: usize) -> Result<u32> {
    set_device(1)?;
    state.sc1.stream.sync()?;
    state.head_x_pos = Some(pos);
    Ok(state.tok_host.as_slice()[0] as u32)
}

pub fn finish_gpu1_head_m1(
    _weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    pos: usize,
    need_logits: bool,
) -> Result<(u32, Option<Vec<f32>>)> {
    set_device(1)?;
    state.sc1.stream.sync()?;
    state.head_x_pos = Some(pos);

    if need_logits {
        let logits = state.target_logits_pin.as_slice().to_vec();
        let mut best_id = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_id = i;
            }
        }
        Ok((best_id as u32, Some(logits)))
    } else {
        Ok((state.tok_host.as_slice()[0] as u32, None))
    }
}

pub fn run_gpu1_half_and_head_m1(
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    pos: usize,
    need_logits: bool,
    use_cuda_graph: bool,
) -> Result<(u32, Option<Vec<f32>>)> {
    enqueue_gpu1_half_and_head_m1(weights, state, pos, need_logits, use_cuda_graph)?;
    finish_gpu1_head_m1(weights, state, pos, need_logits)
}

pub fn decode_token_m1(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    token: u32,
    need_logits: bool,
) -> Result<(u32, Option<Vec<f32>>)> {
    let pos = state.cur_pos;
    run_gpu0_half_m1(gguf, weights, state, token, pos, false)?;
    let res = run_gpu1_half_and_head_m1(weights, state, pos, need_logits, false)?;
    state.cur_pos = pos + 1;
    Ok(res)
}

fn enqueue_snapshot_gpu0_kda(weights: &Glm5ModelWeights, state: &Glm5State) -> Result<()> {
    for l in 0..weights.split_layer {
        if let (Some(k), Some(Some(s))) = (&state.kda[l], state.snap_kda_s0.get(l)) {
            s.copy_from_device_async(&k.s_mat, k.s_mat.bytes, &state.sc0.stream)?;
            state.snap_conv_q0[l].as_ref().unwrap().copy_from_device_async(&k.conv_q, k.conv_q.bytes, &state.sc0.stream)?;
            state.snap_conv_k0[l].as_ref().unwrap().copy_from_device_async(&k.conv_k, k.conv_k.bytes, &state.sc0.stream)?;
            state.snap_conv_v0[l].as_ref().unwrap().copy_from_device_async(&k.conv_v, k.conv_v.bytes, &state.sc0.stream)?;
        }
    }
    Ok(())
}

fn enqueue_restore_gpu0_kda(weights: &Glm5ModelWeights, state: &Glm5State) -> Result<()> {
    for l in 0..weights.split_layer {
        if let (Some(k), Some(Some(s))) = (&state.kda[l], state.snap_kda_s0.get(l)) {
            k.s_mat.copy_from_device_async(s, s.bytes, &state.sc0.stream)?;
            k.conv_q.copy_from_device_async(state.snap_conv_q0[l].as_ref().unwrap(), k.conv_q.bytes, &state.sc0.stream)?;
            k.conv_k.copy_from_device_async(state.snap_conv_k0[l].as_ref().unwrap(), k.conv_k.bytes, &state.sc0.stream)?;
            k.conv_v.copy_from_device_async(state.snap_conv_v0[l].as_ref().unwrap(), k.conv_v.bytes, &state.sc0.stream)?;
        }
    }
    Ok(())
}

pub fn snapshot_gpu0_kda(
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    use_cuda_graph: bool,
) -> Result<()> {
    set_device(0)?;
    if use_cuda_graph {
        if state.snap_kda0_graph.is_none() {
            let g = CudaGraphExec::capture(&state.sc0.stream, || enqueue_snapshot_gpu0_kda(weights, state))?;
            state.snap_kda0_graph = Some(g);
        }
        state.snap_kda0_graph.as_ref().unwrap().launch(&state.sc0.stream)?;
    } else {
        enqueue_snapshot_gpu0_kda(weights, state)?;
    }
    Ok(())
}

pub fn restore_gpu0_kda(
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    use_cuda_graph: bool,
) -> Result<()> {
    set_device(0)?;
    if use_cuda_graph {
        if state.restore_kda0_graph.is_none() {
            let g = CudaGraphExec::capture(&state.sc0.stream, || enqueue_restore_gpu0_kda(weights, state))?;
            state.restore_kda0_graph = Some(g);
        }
        state.restore_kda0_graph.as_ref().unwrap().launch(&state.sc0.stream)?;
    } else {
        enqueue_restore_gpu0_kda(weights, state)?;
    }
    Ok(())
}

/// Write blk.45 MTP DSA cache entry at position `pos`: pairs `head_x_dev` with `next_tok`.
pub fn mtp_cache_entry_m1(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    pos: usize,
    head_x_dev: *const f32,
    next_tok: u32,
) -> Result<()> {
    let Some(mtp_l) = weights.cfg.mtp_layer else {
        return Ok(());
    };
    let cfg = &weights.cfg;
    let lw = &weights.layers[mtp_l];
    let dst = state.dsa[mtp_l].as_ref().unwrap();
    let n_embd = cfg.n_embd;

    dequant_embed_to_host(gguf, weights, next_tok, &mut state.hop_emb_pin1.as_mut_slice()[..n_embd])?;
    set_device(1)?;
    let stream = &state.sc1.stream;
    let emb_bytes = unsafe {
        std::slice::from_raw_parts(state.hop_emb_pin1.host_ptr().cast::<u8>(), n_embd * 4)
    };
    state.sc1.emb.copy_from_host_async(emb_bytes, stream)?;

    unsafe {
        glm5::mtp_in_q8(
            state.sc1.emb.ptr.cast(),
            head_x_dev,
            lw.nextn_enorm.unwrap() as *const f32,
            lw.nextn_hnorm.unwrap() as *const f32,
            cfg.rms_norm_eps,
            n_embd as i32,
            state.mtp_cat_q8.ptr,
            stream,
        );
        let eh = lw.nextn_eh_proj.as_ref().unwrap();
        launch_linear_q8_1(
            eh.ggml_type as i32,
            eh.ptr as *const c_void,
            state.mtp_cat_q8.ptr,
            std::ptr::null(),
            state.mtp_h.ptr.cast(),
            n_embd as i32,
            (2 * n_embd) as i32,
            1,
            stream.raw,
        );
        glm5::rms_norm_q8(
            state.mtp_h.ptr.cast(),
            std::ptr::null(),
            lw.attn_norm as *const f32,
            cfg.rms_norm_eps,
            n_embd as i32,
            false,
            state.sc1.x_f32.ptr.cast(),
            state.sc1.x_q8.ptr,
            1,
            stream,
        );
        let kva = lw.attn_kv_a_mqa.as_ref().unwrap();
        let ik = lw.idx_k.as_ref().unwrap();
        let ig = lw.idx_gate.as_ref().unwrap();
        let w_ptrs = [kva.ptr as *const c_void, ik.ptr as *const c_void, ig.ptr as *const c_void];
        let y_ptrs = [
            state.sc1.kv.ptr.cast::<f32>(),
            state.sc1.ik_tmp.ptr.cast::<f32>(),
            state.sc1.ig_tmp.ptr.cast::<f32>(),
        ];
        let wt = [kva.ggml_type as i32, ik.ggml_type as i32, ig.ggml_type as i32];
        let no = [cfg.kv_lora_rank as i32, cfg.idx_key_dim as i32, cfg.idx_key_dim as i32];
        glm5::mv_batch(
            &w_ptrs,
            &y_ptrs,
            &wt,
            &no,
            None,
            n_embd as i32,
            state.sc1.x_q8.ptr,
            state.sc1.x_f32.ptr.cast(),
            stream,
        );
        let ram_lat_dev = dst
            .ram_lat8
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.dev_ptr_for_device(1).unwrap_or(std::ptr::null_mut()));
        glm5::dsa_prep_kv_ik(
            state.sc1.kv.ptr.cast(),
            lw.attn_kv_a_norm.unwrap() as *const f32,
            dst.vram_lat8.ptr,
            ram_lat_dev,
            weights.dsa_res_cap as i32,
            state.sc1.ik_tmp.ptr.cast(),
            lw.idx_k_norm_w.unwrap() as *const f32,
            lw.idx_k_norm_b.unwrap() as *const f32,
            dst.ik_ring.ptr.cast(),
            state.sc1.ig_tmp.ptr.cast(),
            dst.ig_ring.ptr.cast(),
            cfg.kv_lora_rank as i32,
            cfg.idx_key_dim as i32,
            pos as i32,
            516,
            cfg.rms_norm_eps,
            1,
            stream,
        );
        if (pos + 1) % cfg.idx_kpool == 0 {
            let pi = pos / cfg.idx_kpool;
            glm5::dsa_pool(
                dst.ik_ring.ptr.cast(),
                dst.ig_ring.ptr.cast(),
                lw.idx_ape.unwrap() as *const f32,
                cfg.idx_key_dim as i32,
                cfg.idx_kpool as i32,
                pi as i32,
                1,
                516,
                dst.pooled_f16.ptr,
                stream,
            );
        }
    }
    if state.mtp_cached_pos <= pos {
        state.mtp_cached_pos = pos + 1;
    }
    Ok(())
}

#[inline]
unsafe fn linear_q8_m(
    w: &DevWeight,
    x_q8: *const c_void,
    x_f32: *const f32,
    y: *mut f32,
    n_out: usize,
    k_in: usize,
    m: usize,
    stream: &CudaStream,
) {
    launch_linear_q8_1(
        w.ggml_type as i32,
        w.ptr as *const c_void,
        x_q8,
        x_f32,
        y,
        n_out as i32,
        k_in as i32,
        m as i32,
        stream.raw,
    );
}

pub fn enqueue_layer_batch_m(
    weights: &Glm5ModelWeights,
    state: &Glm5State,
    l: usize,
    pos0: usize,
    m: usize,
    stream: &CudaStream,
) -> Result<()> {
    let cfg = &weights.cfg;
    let lw = &weights.layers[l];
    let sc = if lw.device == 0 { &state.sc0 } else { &state.sc1 };
    let db_dev = if lw.device == 0 {
        state.doorbell.db0_dev_ptr
    } else {
        state.doorbell.db1_dev_ptr
    };

    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let norm_eps = cfg.rms_norm_eps;
    let hc_eps = cfg.hc_eps;

    unsafe {
        let hca = lw.hc_attn_fn.as_ref().unwrap();
        glm5::hc_pre_norm_q8(
            sc.r_buf[0].ptr.cast(),
            hca.ptr as *const c_void,
            hca.ggml_type as i32,
            lw.hc_attn_scale.unwrap() as *const f32,
            lw.hc_attn_base.unwrap() as *const f32,
            lw.attn_norm as *const f32,
            n_embd as i32,
            hc as i32,
            cfg.sinkhorn_iters as i32,
            norm_eps,
            hc_eps,
            sc.post.ptr.cast(),
            sc.comb.ptr.cast(),
            sc.x_f32.ptr.cast(),
            sc.x_q8.ptr,
            m as i32,
            stream,
        );

        if lw.is_kda {
            let kst = state.kda[l].as_ref().unwrap();
            let ktr = state.kda_trace[l].as_ref().unwrap();
            linear_q8_m(lw.attn_q.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), ktr.q_p.ptr.cast(), cfg.d_inner(), n_embd, m, stream);
            linear_q8_m(lw.attn_k.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), ktr.k_p.ptr.cast(), cfg.d_inner(), n_embd, m, stream);
            linear_q8_m(lw.attn_v.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), ktr.v_p.ptr.cast(), cfg.d_inner(), n_embd, m, stream);
            linear_q8_m(lw.ssm_f_a.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.f_a.ptr.cast(), cfg.kda_head_dim, n_embd, m, stream);
            linear_q8_m(lw.ssm_g_a.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.g_a.ptr.cast(), cfg.kda_head_dim, n_embd, m, stream);
            linear_q8_m(lw.ssm_beta.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), ktr.beta.ptr.cast(), cfg.n_head, n_embd, m, stream);

            glm5::kda_conv_silu_l2(
                ktr.q_p.ptr.cast(), lw.ssm_conv_q.unwrap() as *const f32, kst.conv_q.ptr.cast(),
                cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                true, m as i32, sc.qc.ptr.cast(), stream,
            );
            glm5::kda_conv_silu_l2(
                ktr.k_p.ptr.cast(), lw.ssm_conv_k.unwrap() as *const f32, kst.conv_k.ptr.cast(),
                cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                true, m as i32, sc.kc.ptr.cast(), stream,
            );
            glm5::kda_conv_silu_l2(
                ktr.v_p.ptr.cast(), lw.ssm_conv_v.unwrap() as *const f32, kst.conv_v.ptr.cast(),
                cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                false, m as i32, sc.vc.ptr.cast(), stream,
            );

            launch_quantize_q8_1(sc.f_a.ptr.cast(), sc.qr_q8.ptr, m as i32, cfg.kda_head_dim as i32, stream.raw);
            launch_quantize_q8_1(sc.g_a.ptr.cast(), sc.h_q8.ptr, m as i32, cfg.kda_head_dim as i32, stream.raw);
            linear_q8_m(lw.ssm_f_b.as_ref().unwrap(), sc.qr_q8.ptr, sc.f_a.ptr.cast(), sc.g_raw.ptr.cast(), cfg.d_inner(), cfg.kda_head_dim, m, stream);
            linear_q8_m(lw.ssm_g_b.as_ref().unwrap(), sc.h_q8.ptr, sc.g_a.ptr.cast(), sc.g2.ptr.cast(), cfg.d_inner(), cfg.kda_head_dim, m, stream);

            glm5::kda_gate_beta(
                sc.g_raw.ptr.cast(), lw.ssm_dt_bias.unwrap() as *const f32, lw.ssm_a.unwrap() as *const f32,
                cfg.kda_gate_lb, cfg.kda_head_dim as i32, cfg.n_head as i32, m as i32,
                ktr.g1.ptr.cast(), ktr.beta.ptr.cast(), stream,
            );
            glm5::kda_recurrence(
                sc.qc.ptr.cast(), sc.kc.ptr.cast(), sc.vc.ptr.cast(), ktr.g1.ptr.cast(), ktr.beta.ptr.cast(),
                kst.s_mat.ptr.cast(), cfg.n_head as i32, cfg.kda_head_dim as i32, m as i32,
                sc.scan.ptr.cast(), stream,
            );
            glm5::kda_out_gate_q8(
                sc.scan.ptr.cast(), lw.ssm_norm.unwrap() as *const f32, sc.g2.ptr.cast(),
                cfg.kda_head_dim as i32, cfg.n_head as i32, m as i32, norm_eps,
                sc.gated.ptr.cast(), sc.gated_q8.ptr, stream,
            );
            linear_q8_m(&lw.attn_output, sc.gated_q8.ptr, sc.gated.ptr.cast(), sc.mixer.ptr.cast(), n_embd, cfg.d_inner(), m, stream);
        } else {
            let dst = state.dsa[l].as_ref().unwrap();
            let prescale = 1.0f32 / ((cfg.idx_key_dim * cfg.idx_heads) as f32).sqrt();
            linear_q8_m(lw.attn_q_a.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.qr.ptr.cast(), cfg.q_lora_rank, n_embd, m, stream);
            linear_q8_m(lw.attn_kv_a_mqa.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.kv.ptr.cast(), cfg.kv_lora_rank, n_embd, m, stream);
            linear_q8_m(lw.idx_k.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.ik_tmp.ptr.cast(), cfg.idx_key_dim, n_embd, m, stream);
            linear_q8_m(lw.idx_gate.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.ig_tmp.ptr.cast(), cfg.idx_key_dim, n_embd, m, stream);
            linear_q8_m(lw.idx_proj.as_ref().unwrap(), sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.iw.ptr.cast(), cfg.idx_heads, n_embd, m, stream);
            glm5::scale_f32(sc.iw.ptr.cast(), prescale, (m * cfg.idx_heads) as i32, stream);

            glm5::rms_norm_q8(
                sc.qr.ptr.cast(), std::ptr::null(), lw.attn_q_a_norm.unwrap() as *const f32,
                norm_eps, cfg.q_lora_rank as i32, false, sc.qr.ptr.cast(), sc.qr_q8.ptr, m as i32, stream,
            );

            let ram_lat_dev = dst
                .ram_lat8
                .as_ref()
                .map_or(std::ptr::null_mut(), |b| b.dev_ptr_for_device(lw.device).unwrap_or(std::ptr::null_mut()));
            glm5::dsa_prep_kv_ik(
                sc.kv.ptr.cast(), lw.attn_kv_a_norm.unwrap() as *const f32,
                dst.vram_lat8.ptr, ram_lat_dev, weights.dsa_res_cap as i32,
                sc.ik_tmp.ptr.cast(), lw.idx_k_norm_w.unwrap() as *const f32, lw.idx_k_norm_b.unwrap() as *const f32,
                dst.ik_ring.ptr.cast(), sc.ig_tmp.ptr.cast(), dst.ig_ring.ptr.cast(),
                cfg.kv_lora_rank as i32, cfg.idx_key_dim as i32, pos0 as i32, 516, norm_eps, m as i32, stream,
            );

            let pool0 = pos0 / cfg.idx_kpool;
            let pool_end = (pos0 + 1) / cfg.idx_kpool;
            let n_new_pools = pool_end.saturating_sub(pool0);
            if n_new_pools > 0 {
                glm5::dsa_pool(
                    dst.ik_ring.ptr.cast(), dst.ig_ring.ptr.cast(), lw.idx_ape.unwrap() as *const f32,
                    cfg.idx_key_dim as i32, cfg.idx_kpool as i32, pool0 as i32, n_new_pools as i32, 516,
                    dst.pooled_f16.ptr, stream,
                );
            }

            linear_q8_m(lw.attn_q_b.as_ref().unwrap(), sc.qr_q8.ptr, sc.qr.ptr.cast(), sc.q_full.ptr.cast(), cfg.n_head * cfg.qk_nope_head_dim, cfg.q_lora_rank, m, stream);
            linear_q8_m(lw.idx_q_b.as_ref().unwrap(), sc.qr_q8.ptr, sc.qr.ptr.cast(), sc.iq.ptr.cast(), cfg.idx_heads * cfg.idx_key_dim, cfg.q_lora_rank, m, stream);

            glm5::dsa_q_absorb(
                lw.attn_k_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32,
                sc.q_full.ptr.cast(), sc.q_abs.ptr.cast(),
                cfg.qk_nope_head_dim as i32, cfg.kv_lora_rank as i32, cfg.n_head as i32, m as i32, stream,
            );

            let top_pools_cap = cfg.idx_top_k / cfg.idx_kpool;
            let n_sel = cfg.idx_top_k + cfg.idx_kpool - 1;
            glm5::dsa_score_select(
                sc.iq.ptr.cast(), dst.pooled_f16.ptr, sc.iw.ptr.cast(),
                cfg.idx_key_dim as i32, cfg.idx_heads as i32, cfg.idx_kpool as i32,
                top_pools_cap as i32, cfg.idx_select_tail, n_sel as i32,
                sc.pos_arr_dev.ptr.cast(), sc.dsa_scores.ptr.cast(), sc.dsa_cells.ptr.cast(),
                m as i32, pool_end as i32, stream,
            );
            glm5::dsa_mla_attn(
                sc.q_abs.ptr.cast(), dst.vram_lat8.ptr, ram_lat_dev, weights.dsa_res_cap as i32,
                sc.dsa_cells.ptr.cast(), lw.attn_v_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32,
                cfg.kv_lora_rank as i32, cfg.n_head as i32, cfg.v_head_dim as i32,
                cfg.qk_nope_head_dim as i32, n_sel as i32, sc.dsa_attn.ptr.cast(), m as i32, stream,
            );

            let v_tot = cfg.n_head * cfg.v_head_dim;
            launch_quantize_q8_1(sc.dsa_attn.ptr.cast(), sc.dsa_attn_q8.ptr, m as i32, v_tot as i32, stream.raw);
            linear_q8_m(&lw.attn_output, sc.dsa_attn_q8.ptr, sc.dsa_attn.ptr.cast(), sc.mixer.ptr.cast(), n_embd, v_tot, m, stream);
        }

        glm5::hc_post_add(
            sc.r_buf[0].ptr.cast(), sc.post.ptr.cast(), sc.comb.ptr.cast(), sc.mixer.ptr.cast(),
            std::ptr::null(), std::ptr::null(), sc.r_buf[1].ptr.cast(), n_embd as i32, hc as i32, m as i32, stream,
        );

        let hcf = lw.hc_ffn_fn.as_ref().unwrap();
        glm5::hc_pre_norm_q8(
            sc.r_buf[1].ptr.cast(), hcf.ptr as *const c_void, hcf.ggml_type as i32,
            lw.hc_ffn_scale.unwrap() as *const f32, lw.hc_ffn_base.unwrap() as *const f32,
            lw.ffn_norm as *const f32, n_embd as i32, hc as i32, cfg.sinkhorn_iters as i32,
            norm_eps, hc_eps, sc.post.ptr.cast(), sc.comb.ptr.cast(),
            sc.x_f32.ptr.cast(), sc.x_q8.ptr, m as i32, stream,
        );

        if !lw.is_moe {
            let fg = lw.ffn_gate.as_ref().unwrap();
            let fu = lw.ffn_up.as_ref().unwrap();
            let fd = lw.ffn_down.as_ref().unwrap();
            linear_q8_m(fg, sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.gate.ptr.cast(), cfg.n_ff_dense, n_embd, m, stream);
            linear_q8_m(fu, sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.up.ptr.cast(), cfg.n_ff_dense, n_embd, m, stream);
            glm5::swiglu_clamp_q8(
                sc.gate.ptr.cast(), sc.up.ptr.cast(), cfg.swiglu_clamp_shexp, cfg.n_ff_dense as i32,
                std::ptr::null_mut(), sc.h_q8.ptr, m as i32, stream,
            );
            linear_q8_m(fd, sc.h_q8.ptr, sc.h_f32.ptr.cast(), sc.sh_out.ptr.cast(), n_embd, cfg.n_ff_dense, m, stream);
            glm5::hc_post_add(
                sc.r_buf[1].ptr.cast(), sc.post.ptr.cast(), sc.comb.ptr.cast(), sc.sh_out.ptr.cast(),
                std::ptr::null(), std::ptr::null(), sc.r_buf[0].ptr.cast(), n_embd as i32, hc as i32, m as i32, stream,
            );
        } else {
            let ginp = lw.ffn_gate_inp.as_ref().unwrap();
            linear_q8_m(ginp, sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.moe_logits.ptr.cast(), cfg.n_expert, n_embd, m, stream);
            sc.moe_rows_dev.copy_from_host_async(&state.expert_cache.layers[l].res_mask_host, stream)?;
            let exp_seq_ptr = sc.db_expected_seq.ptr.cast::<u64>().add(l);
            glm5::router_sigmoid_probe(
                sc.moe_logits.ptr.cast(), lw.exp_probs_bias.unwrap() as *const f32,
                cfg.n_expert as i32, cfg.n_expert_used as i32, cfg.expert_weights_scale, cfg.expert_weights_norm,
                sc.moe_rows_dev.ptr.cast(), sc.x_f32.ptr.cast(),
                n_embd as i32, cfg.n_ff_exp as i32, l as i32, lw.gu_type as i32, lw.d_type as i32,
                cfg.swiglu_clamp_exp, sc.moe_ids.ptr.cast(), sc.moe_rw.ptr.cast(),
                db_dev, sc.db_seq_state.ptr.cast(), exp_seq_ptr, m as i32, stream,
            );
            let mut ids_h = vec![0i32; m * cfg.n_expert_used];
            ampllm_cuda_kernels::copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut ids_h),
                sc.moe_ids.ptr,
                stream,
            )?;
            stream.sync()?;
            let vp = &state.expert_cache.layers[l].vram_ptrs_host;
            let mut plan_ptrs_h = vec![0u64; m * cfg.n_expert_used];
            for (i, &eid) in ids_h.iter().enumerate() {
                if eid >= 0 && (eid as usize) < vp.len() {
                    plan_ptrs_h[i] = vp[eid as usize];
                }
            }
            sc.moe_plan_ptrs_dev
                .copy_from_host_async(bytemuck::cast_slice(&plan_ptrs_h), stream)?;

            let gs = lw.ffn_gate_sh.as_ref().unwrap();
            let us = lw.ffn_up_sh.as_ref().unwrap();
            let ds = lw.ffn_down_sh.as_ref().unwrap();
            let n_ff_sh = cfg.n_ff_shexp * cfg.n_shared_expert;
            linear_q8_m(gs, sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.gate.ptr.cast(), n_ff_sh, n_embd, m, stream);
            linear_q8_m(us, sc.x_q8.ptr, sc.x_f32.ptr.cast(), sc.up.ptr.cast(), n_ff_sh, n_embd, m, stream);
            glm5::swiglu_clamp_q8(
                sc.gate.ptr.cast(), sc.up.ptr.cast(), cfg.swiglu_clamp_shexp, n_ff_sh as i32,
                std::ptr::null_mut(), sc.h_q8.ptr, m as i32, stream,
            );
            linear_q8_m(ds, sc.h_q8.ptr, sc.h_f32.ptr.cast(), sc.sh_out.ptr.cast(), n_embd, n_ff_sh, m, stream);

            let gu_rb = lw.gu_type.row_bytes(n_embd);
            let d_rb = lw.d_type.row_bytes(cfg.n_ff_exp);
            glm5::moe_plan(
                lw.gu_type as i32, lw.d_type as i32,
                sc.moe_plan_ptrs_dev.ptr.cast(),
                lw.u_off as u32, lw.d_off as u32,
                gu_rb, d_rb, sc.x_q8.ptr, sc.moe_rw.ptr.cast(),
                sc.moe_inter_q8.ptr, 0, std::ptr::null(), 0, std::ptr::null(), 0,
                sc.sh_out.ptr.cast(), sc.gpu_moe_out.ptr.cast(),
                n_embd as i32, cfg.n_ff_exp as i32, cfg.n_expert_used as i32, m as i32,
                cfg.swiglu_clamp_exp, std::ptr::null(), stream,
            );
            glm5::doorbell_wait_hc_post(
                db_dev, exp_seq_ptr, sc.r_buf[1].ptr.cast(), sc.post.ptr.cast(), sc.comb.ptr.cast(),
                sc.gpu_moe_out.ptr.cast(), std::ptr::null(), sc.r_buf[0].ptr.cast(),
                n_embd as i32, hc as i32, m as i32, stream,
            );
        }
    }
    Ok(())
}

pub fn snapshot_all_kda(weights: &Glm5ModelWeights, state: &mut Glm5State) -> Result<()> {
    set_device(0)?;
    for l in 0..weights.split_layer {
        if let (Some(k), Some(Some(s))) = (&state.kda[l], state.snap_kda_s0.get(l)) {
            s.copy_from_device_async(&k.s_mat, k.s_mat.bytes, &state.sc0.stream)?;
            state.snap_conv_q0[l].as_ref().unwrap().copy_from_device_async(&k.conv_q, k.conv_q.bytes, &state.sc0.stream)?;
            state.snap_conv_k0[l].as_ref().unwrap().copy_from_device_async(&k.conv_k, k.conv_k.bytes, &state.sc0.stream)?;
            state.snap_conv_v0[l].as_ref().unwrap().copy_from_device_async(&k.conv_v, k.conv_v.bytes, &state.sc0.stream)?;
        }
    }
    set_device(1)?;
    for l in weights.split_layer..weights.cfg.n_layers {
        if let (Some(k), Some(Some(s))) = (&state.kda[l], state.snap_kda_s1.get(l)) {
            s.copy_from_device_async(&k.s_mat, k.s_mat.bytes, &state.sc1.stream)?;
            state.snap_conv_q1[l].as_ref().unwrap().copy_from_device_async(&k.conv_q, k.conv_q.bytes, &state.sc1.stream)?;
            state.snap_conv_k1[l].as_ref().unwrap().copy_from_device_async(&k.conv_k, k.conv_k.bytes, &state.sc1.stream)?;
            state.snap_conv_v1[l].as_ref().unwrap().copy_from_device_async(&k.conv_v, k.conv_v.bytes, &state.sc1.stream)?;
        }
    }
    Ok(())
}

pub fn rollback_kda_from_trace(
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    n_commit: usize,
) -> Result<()> {
    if n_commit == 0 {
        return Ok(());
    }
    let cfg = &weights.cfg;
    set_device(0)?;
    for l in 0..weights.split_layer {
        if let (Some(k), Some(Some(s)), Some(ktr)) =
            (&state.kda[l], state.snap_kda_s0.get(l), &state.kda_trace[l])
        {
            let lw = &weights.layers[l];
            let sc = &state.sc0;
            k.s_mat.copy_from_device_async(s, s.bytes, &sc.stream)?;
            k.conv_q.copy_from_device_async(state.snap_conv_q0[l].as_ref().unwrap(), k.conv_q.bytes, &sc.stream)?;
            k.conv_k.copy_from_device_async(state.snap_conv_k0[l].as_ref().unwrap(), k.conv_k.bytes, &sc.stream)?;
            k.conv_v.copy_from_device_async(state.snap_conv_v0[l].as_ref().unwrap(), k.conv_v.bytes, &sc.stream)?;
            unsafe {
                glm5::kda_conv_silu_l2(
                    ktr.q_p.ptr.cast(), lw.ssm_conv_q.unwrap() as *const f32, k.conv_q.ptr.cast(),
                    cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                    true, n_commit as i32, sc.qc.ptr.cast(), &sc.stream,
                );
                glm5::kda_conv_silu_l2(
                    ktr.k_p.ptr.cast(), lw.ssm_conv_k.unwrap() as *const f32, k.conv_k.ptr.cast(),
                    cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                    true, n_commit as i32, sc.kc.ptr.cast(), &sc.stream,
                );
                glm5::kda_conv_silu_l2(
                    ktr.v_p.ptr.cast(), lw.ssm_conv_v.unwrap() as *const f32, k.conv_v.ptr.cast(),
                    cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                    false, n_commit as i32, sc.vc.ptr.cast(), &sc.stream,
                );
                glm5::kda_recurrence(
                    sc.qc.ptr.cast(), sc.kc.ptr.cast(), sc.vc.ptr.cast(),
                    ktr.g1.ptr.cast(), ktr.beta.ptr.cast(), k.s_mat.ptr.cast(),
                    cfg.n_head as i32, cfg.kda_head_dim as i32, n_commit as i32,
                    sc.scan.ptr.cast(), &sc.stream,
                );
            }
        }
    }
    set_device(1)?;
    for l in weights.split_layer..cfg.n_layers {
        if let (Some(k), Some(Some(s)), Some(ktr)) =
            (&state.kda[l], state.snap_kda_s1.get(l), &state.kda_trace[l])
        {
            let lw = &weights.layers[l];
            let sc = &state.sc1;
            k.s_mat.copy_from_device_async(s, s.bytes, &sc.stream)?;
            k.conv_q.copy_from_device_async(state.snap_conv_q1[l].as_ref().unwrap(), k.conv_q.bytes, &sc.stream)?;
            k.conv_k.copy_from_device_async(state.snap_conv_k1[l].as_ref().unwrap(), k.conv_k.bytes, &sc.stream)?;
            k.conv_v.copy_from_device_async(state.snap_conv_v1[l].as_ref().unwrap(), k.conv_v.bytes, &sc.stream)?;
            unsafe {
                glm5::kda_conv_silu_l2(
                    ktr.q_p.ptr.cast(), lw.ssm_conv_q.unwrap() as *const f32, k.conv_q.ptr.cast(),
                    cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                    true, n_commit as i32, sc.qc.ptr.cast(), &sc.stream,
                );
                glm5::kda_conv_silu_l2(
                    ktr.k_p.ptr.cast(), lw.ssm_conv_k.unwrap() as *const f32, k.conv_k.ptr.cast(),
                    cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                    true, n_commit as i32, sc.kc.ptr.cast(), &sc.stream,
                );
                glm5::kda_conv_silu_l2(
                    ktr.v_p.ptr.cast(), lw.ssm_conv_v.unwrap() as *const f32, k.conv_v.ptr.cast(),
                    cfg.d_inner() as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                    false, n_commit as i32, sc.vc.ptr.cast(), &sc.stream,
                );
                glm5::kda_recurrence(
                    sc.qc.ptr.cast(), sc.kc.ptr.cast(), sc.vc.ptr.cast(),
                    ktr.g1.ptr.cast(), ktr.beta.ptr.cast(), k.s_mat.ptr.cast(),
                    cfg.n_head as i32, cfg.kda_head_dim as i32, n_commit as i32,
                    sc.scan.ptr.cast(), &sc.stream,
                );
            }
        }
    }
    Ok(())
}

pub fn verify_batch_m(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    tokens: &[u32],
    pos0: usize,
    need_logits: bool,
) -> Result<Vec<u32>> {
    let m = tokens.len();
    debug_assert!((1..=4).contains(&m));
    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;

    if m > 1 {
        snapshot_all_kda(weights, state)?;
    }

    {
        let emb_dst = &mut state.hop_emb_pin0.as_mut_slice()[..m * n_embd];
        let p0 = &mut state.pos_pin0.as_mut_slice()[..m];
        let p1 = &mut state.pos_pin1.as_mut_slice()[..m];
        for (j, &tok) in tokens.iter().enumerate() {
            dequant_embed_to_host(gguf, weights, tok, &mut emb_dst[j * n_embd..(j + 1) * n_embd])?;
            p0[j] = (pos0 + j) as i32;
            p1[j] = (pos0 + j) as i32;
        }
    }

    set_device(0)?;
    let pos_bytes = unsafe { std::slice::from_raw_parts(state.pos_pin0.host_ptr().cast::<u8>(), m * 4) };
    state.sc0.pos_arr_dev.copy_from_host_async(pos_bytes, &state.sc0.stream)?;
    let emb_bytes = unsafe { std::slice::from_raw_parts(state.hop_emb_pin0.host_ptr().cast::<u8>(), m * n_embd * 4) };
    state.sc0.emb.copy_from_host_async(emb_bytes, &state.sc0.stream)?;
    unsafe {
        glm5::embed_replicate(
            state.sc0.emb.ptr.cast(),
            state.sc0.r_buf[0].ptr.cast(),
            n_embd as i32,
            cfg.hc as i32,
            m as i32,
            &state.sc0.stream,
        );
    }

    if m == 1 {
        enqueue_layer_range_m1(weights, state, 0, weights.split_layer, pos0, true)?;
    } else {
        for l in 0..weights.split_layer {
            enqueue_layer_batch_m(weights, state, l, pos0, m, &state.sc0.stream)?;
        }
    }

    if state.p2p_ok {
        state.ev_g0_done[0].record(&state.sc0.stream)?;
    } else {
        let hop_bytes = m * cfg.hc * n_embd * 4;
        let h_slice = unsafe { std::slice::from_raw_parts_mut(state.hop_pin[0].host_ptr().cast::<u8>(), hop_bytes) };
        state.sc0.r_buf[0].copy_to_host_async(h_slice, &state.sc0.stream)?;
        state.ev_g0_done[0].record(&state.sc0.stream)?;
    }

    hop_gpu0_to_gpu1_r0(weights, state, m, 0)?;

    set_device(1)?;
    let pos1_bytes = unsafe { std::slice::from_raw_parts(state.pos_pin1.host_ptr().cast::<u8>(), m * 4) };
    state.sc1.pos_arr_dev.copy_from_host_async(pos1_bytes, &state.sc1.stream)?;

    if m == 1 {
        enqueue_layer_range_m1(weights, state, weights.split_layer, cfg.n_layers, pos0, false)?;
    } else {
        for l in weights.split_layer..cfg.n_layers {
            enqueue_layer_batch_m(weights, state, l, pos0, m, &state.sc1.stream)?;
        }
    }

    unsafe {
        glm5::mean_streams_norm_q8(
            state.sc1.r_buf[0].ptr.cast(),
            weights.output_norm_dev.ptr.cast(),
            cfg.rms_norm_eps,
            n_embd as i32,
            cfg.hc as i32,
            state.head_x.ptr.cast(),
            state.head_xq.ptr,
            m as i32,
            &state.sc1.stream,
        );
        launch_linear_q8_1(
            weights.output_weight_type as i32,
            weights.output_weight_dev.ptr,
            state.head_xq.ptr,
            state.head_x.ptr.cast(),
            state.logits_dev.ptr.cast(),
            cfg.n_vocab as i32,
            n_embd as i32,
            m as i32,
            state.sc1.stream.raw,
        );
        if !need_logits {
            for j in 0..m {
                glm5::argmax(
                    state.logits_dev.ptr.cast::<f32>().add(j * cfg.n_vocab),
                    cfg.n_vocab as i32,
                    cfg.n_vocab as i32,
                    std::ptr::null(),
                    state.argmax_pv.ptr.cast(),
                    state.argmax_pi.ptr.cast(),
                    state.tok_dev.ptr.cast::<i32>().add(j),
                    &state.sc1.stream,
                );
            }
            let th_slice = std::slice::from_raw_parts_mut(state.tok_host.host_ptr().cast::<u8>(), m * 4);
            state.tok_dev.copy_to_host_async(th_slice, &state.sc1.stream)?;
        } else {
            let lh_slice = std::slice::from_raw_parts_mut(
                state.target_logits_pin.host_ptr().cast::<u8>(),
                m * cfg.n_vocab * 4,
            );
            state.logits_dev.copy_to_host_async(lh_slice, &state.sc1.stream)?;
        }
    }

    state.sc1.stream.sync()?;
    if need_logits {
        Ok(Vec::new())
    } else {
        Ok(state.tok_host.as_slice()[..m]
            .iter()
            .map(|&t| t as u32)
            .collect())
    }
}

pub fn copy_head_x_row_to_base(state: &mut Glm5State, src_row: usize, n_embd: usize) -> Result<()> {
    if src_row == 0 {
        return Ok(());
    }
    set_device(1)?;
    unsafe {
        let src = state.head_x.ptr.cast::<f32>().add(src_row * n_embd);
        copy_d2d_raw_async(state.head_x.ptr, src.cast(), n_embd * 4, &state.sc1.stream)?;
    }
    Ok(())
}

pub fn enqueue_mtp_draft_chained_on_gpu1(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    pos: usize,
    prev_draft_tok: u32,
    use_draft_vocab: bool,
    need_draft_logits: bool,
    use_cuda_graph: bool,
) -> Result<()> {
    set_device(1)?;
    unsafe {
        copy_d2d_raw_async(
            state.mtp_chain_h.ptr,
            state.mtp_h.ptr,
            weights.cfg.n_embd * 4,
            &state.sc1.stream,
        )?;
    }
    let ch_ptr = state.mtp_chain_h.ptr.cast();
    enqueue_mtp_draft_on_gpu1(
        gguf,
        weights,
        state,
        pos,
        ch_ptr,
        prev_draft_tok,
        use_draft_vocab,
        need_draft_logits,
        use_cuda_graph,
    )
}

fn enqueue_mtp_draft_body_m1(
    weights: &Glm5ModelWeights,
    state: &Glm5State,
    head_x_dev: *const f32,
    use_dv: bool,
    need_draft_logits: bool,
) {
    let mtp_l = weights.cfg.mtp_layer.unwrap();
    let cfg = &weights.cfg;
    let lw = &weights.layers[mtp_l];
    let dst = state.dsa[mtp_l].as_ref().unwrap();
    let n_embd = cfg.n_embd;
    let stream = &state.sc1.stream;

    unsafe {
        glm5::mtp_in_q8(
            state.sc1.emb.ptr.cast(),
            head_x_dev,
            lw.nextn_enorm.unwrap() as *const f32,
            lw.nextn_hnorm.unwrap() as *const f32,
            cfg.rms_norm_eps,
            n_embd as i32,
            state.mtp_cat_q8.ptr,
            stream,
        );
        let eh = lw.nextn_eh_proj.as_ref().unwrap();
        launch_linear_q8_1(
            eh.ggml_type as i32,
            eh.ptr as *const c_void,
            state.mtp_cat_q8.ptr,
            std::ptr::null(),
            state.mtp_h.ptr.cast(),
            n_embd as i32,
            (2 * n_embd) as i32,
            1,
            stream.raw,
        );
        glm5::rms_norm_q8(
            state.mtp_h.ptr.cast(),
            std::ptr::null(),
            lw.attn_norm as *const f32,
            cfg.rms_norm_eps,
            n_embd as i32,
            false,
            state.sc1.x_f32.ptr.cast(),
            state.sc1.x_q8.ptr,
            1,
            stream,
        );

        let qa = lw.attn_q_a.as_ref().unwrap();
        let kva = lw.attn_kv_a_mqa.as_ref().unwrap();
        let ik = lw.idx_k.as_ref().unwrap();
        let ig = lw.idx_gate.as_ref().unwrap();
        let ip = lw.idx_proj.as_ref().unwrap();
        let prescale = 1.0f32 / ((cfg.idx_key_dim * cfg.idx_heads) as f32).sqrt();
        let w_ptrs = [qa.ptr as *const c_void, kva.ptr as *const c_void, ik.ptr as *const c_void, ig.ptr as *const c_void, ip.ptr as *const c_void];
        let y_ptrs = [state.sc1.qr.ptr.cast::<f32>(), state.sc1.kv.ptr.cast::<f32>(), state.sc1.ik_tmp.ptr.cast::<f32>(), state.sc1.ig_tmp.ptr.cast::<f32>(), state.sc1.iw.ptr.cast::<f32>()];
        let wt = [qa.ggml_type as i32, kva.ggml_type as i32, ik.ggml_type as i32, ig.ggml_type as i32, ip.ggml_type as i32];
        let no = [cfg.q_lora_rank as i32, cfg.kv_lora_rank as i32, cfg.idx_key_dim as i32, cfg.idx_key_dim as i32, cfg.idx_heads as i32];
        let scs = [1.0f32, 1.0, 1.0, 1.0, prescale];
        glm5::mv_batch(&w_ptrs, &y_ptrs, &wt, &no, Some(&scs), n_embd as i32, state.sc1.x_q8.ptr, state.sc1.x_f32.ptr.cast(), stream);

        let ram_lat_dev = dst.ram_lat8.as_ref().map_or(std::ptr::null_mut(), |b| b.dev_ptr_for_device(1).unwrap_or(std::ptr::null_mut()));
        glm5::dsa_prep_m1_dyn(
            state.sc1.qr.ptr.cast(), lw.attn_q_a_norm.unwrap() as *const f32, state.sc1.qr_q8.ptr, cfg.q_lora_rank as i32,
            state.sc1.kv.ptr.cast(), lw.attn_kv_a_norm.unwrap() as *const f32, dst.vram_lat8.ptr, ram_lat_dev, weights.dsa_res_cap as i32, cfg.kv_lora_rank as i32,
            state.sc1.ik_tmp.ptr.cast(), lw.idx_k_norm_w.unwrap() as *const f32, lw.idx_k_norm_b.unwrap() as *const f32, dst.ik_ring.ptr.cast(),
            state.sc1.ig_tmp.ptr.cast(), dst.ig_ring.ptr.cast(), lw.idx_ape.unwrap() as *const f32,
            cfg.idx_key_dim as i32, cfg.idx_kpool as i32, 516, state.sc1.mtp_pos_dev.ptr.cast(), dst.pooled_f16.ptr,
            cfg.rms_norm_eps, std::ptr::null(), stream,
        );

        let qb = lw.attn_q_b.as_ref().unwrap();
        let iqb = lw.idx_q_b.as_ref().unwrap();
        glm5::mv_batch(
            &[qb.ptr as *const c_void, iqb.ptr as *const c_void],
            &[state.sc1.q_full.ptr.cast::<f32>(), state.sc1.iq.ptr.cast::<f32>()],
            &[qb.ggml_type as i32, iqb.ggml_type as i32],
            &[(cfg.n_head * cfg.qk_nope_head_dim) as i32, (cfg.idx_heads * cfg.idx_key_dim) as i32],
            None, cfg.q_lora_rank as i32, state.sc1.qr_q8.ptr, state.sc1.qr.ptr.cast(), stream,
        );
        glm5::dsa_q_absorb(lw.attn_k_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32, state.sc1.q_full.ptr.cast(), state.sc1.q_abs.ptr.cast(), cfg.qk_nope_head_dim as i32, cfg.kv_lora_rank as i32, cfg.n_head as i32, 1, stream);

        let top_pools_cap = cfg.idx_top_k / cfg.idx_kpool;
        let n_sel = cfg.idx_top_k + cfg.idx_kpool - 1;
        glm5::dsa_score_select_dyn(
            state.sc1.iq.ptr.cast(), dst.pooled_f16.ptr, state.sc1.iw.ptr.cast(),
            cfg.idx_key_dim as i32, cfg.idx_heads as i32, cfg.idx_kpool as i32, top_pools_cap as i32,
            cfg.idx_select_tail, n_sel as i32, weights.pool_cap as i32, state.sc1.mtp_pos_dev.ptr.cast(),
            state.sc1.dsa_scores.ptr.cast(), state.sc1.dsa_cells.ptr.cast(), stream,
        );
        let mtp_out_f32 = if matches!(lw.attn_output.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16) {
            state.sc1.dsa_attn.ptr.cast()
        } else {
            std::ptr::null_mut()
        };
        glm5::dsa_mla_vproj_q8_m1(
            state.sc1.q_abs.ptr.cast(), dst.vram_lat8.ptr, ram_lat_dev, weights.dsa_res_cap as i32, state.sc1.dsa_cells.ptr.cast(),
            lw.attn_v_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32, cfg.kv_lora_rank as i32, cfg.n_head as i32, cfg.v_head_dim as i32, cfg.qk_nope_head_dim as i32, n_sel as i32,
            state.sc1.q_abs.ptr.cast(), state.sc1.mma_b_f16.ptr.cast(), mtp_out_f32, state.sc1.dsa_attn_q8.ptr, std::ptr::null(), stream,
        );

        let v_tot = (cfg.n_head * cfg.v_head_dim) as i32;
        launch_linear_q8_1(lw.attn_output.ggml_type as i32, lw.attn_output.ptr as *const c_void, state.sc1.dsa_attn_q8.ptr, state.sc1.dsa_attn.ptr.cast(), state.sc1.mixer.ptr.cast(), n_embd as i32, v_tot, 1, stream.raw);

        glm5::rms_norm_q8(
            state.mtp_h.ptr.cast(),
            state.sc1.mixer.ptr.cast(),
            lw.ffn_norm as *const f32,
            cfg.rms_norm_eps,
            n_embd as i32,
            true,
            state.sc1.x_f32.ptr.cast(),
            state.sc1.x_q8.ptr,
            1,
            stream,
        );

        let ginp = lw.ffn_gate_inp.as_ref().unwrap();
        let gs = lw.ffn_gate_sh.as_ref().unwrap();
        let us = lw.ffn_up_sh.as_ref().unwrap();
        let ds = lw.ffn_down_sh.as_ref().unwrap();
        let n_ff_sh = (cfg.n_ff_shexp * cfg.n_shared_expert) as i32;
        glm5::mv_batch(
            &[ginp.ptr as *const c_void, gs.ptr as *const c_void, us.ptr as *const c_void],
            &[state.sc1.moe_logits.ptr.cast::<f32>(), state.sc1.gate.ptr.cast::<f32>(), state.sc1.up.ptr.cast::<f32>()],
            &[ginp.ggml_type as i32, gs.ggml_type as i32, us.ggml_type as i32],
            &[cfg.n_expert as i32, n_ff_sh, n_ff_sh],
            None, n_embd as i32, state.sc1.x_q8.ptr, state.sc1.x_f32.ptr.cast(), stream,
        );
        let exp_seq_ptr = state.sc1.db_expected_seq.ptr.cast::<u64>().add(mtp_l);
        let spares_ptr = lw.spares_dev.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr.cast());
        glm5::router_dual_lane(
            state.sc1.moe_logits.ptr.cast(), lw.exp_probs_bias.unwrap() as *const f32,
            cfg.n_expert as i32, cfg.n_expert_used as i32, cfg.expert_weights_scale, cfg.expert_weights_norm,
            lw.vram_ptrs_dev.as_ref().unwrap().ptr.cast(), lw.ram_ptrs_dev.as_ref().unwrap().ptr.cast(),
            spares_ptr, lw.k_spares as i32, lw.dcnt_dev.as_ref().unwrap().ptr.cast(),
            state.sc1.moe_route_cnt_dev.ptr.cast(), state.sc1.moe_scratch_ptrs_dev.ptr.cast(),
            state.cpu_plan, state.promote_min, state.mtp_keep,
            state.sc1.moe_plan_ptrs_dev.ptr.cast(), state.sc1.moe_fetch_src_dev.ptr.cast(),
            state.sc1.x_f32.ptr.cast(), n_embd as i32, cfg.n_ff_exp as i32, mtp_l as i32,
            lw.gu_type as i32, lw.d_type as i32, cfg.swiglu_clamp_exp,
            lw.u_off as u32, lw.d_off as u32,
            state.sc1.gate.ptr.cast(), state.sc1.up.ptr.cast(), cfg.swiglu_clamp_shexp, n_ff_sh, state.sc1.h_q8.ptr,
            state.sc1.moe_ids.ptr.cast(), state.sc1.moe_rw.ptr.cast(), state.doorbell.db1_dev_ptr,
            state.sc1.db_seq_state.ptr.cast(), exp_seq_ptr, std::ptr::null(), std::ptr::null(), stream,
        );
        glm5::moe_fetch(
            state.sc1.moe_fetch_src_dev.ptr.cast(), state.sc1.moe_plan_ptrs_dev.ptr.cast(),
            cfg.n_expert_used as i32, lw.blob_stride, stream,
        );

        let gu_rb = lw.gu_type.row_bytes(n_embd);
        let d_rb = lw.d_type.row_bytes(cfg.n_ff_exp);
        let ds_rb = ds.ggml_type.row_bytes(n_ff_sh as usize);
        glm5::moe_plan(
            lw.gu_type as i32, lw.d_type as i32,
            state.sc1.moe_plan_ptrs_dev.ptr.cast(), lw.u_off as u32, lw.d_off as u32,
            gu_rb, d_rb, state.sc1.x_q8.ptr, state.sc1.moe_rw.ptr.cast(),
            state.sc1.moe_inter_q8.ptr,
            ds.ggml_type as i32, ds.ptr as *const c_void, ds_rb, state.sc1.h_q8.ptr, n_ff_sh,
            state.sc1.sh_out.ptr.cast(), state.sc1.gpu_moe_out.ptr.cast(),
            n_embd as i32, cfg.n_ff_exp as i32, cfg.n_expert_used as i32, 1, cfg.swiglu_clamp_exp,
            std::ptr::null(), stream,
        );

        glm5::doorbell_wait_hc_post(
            state.doorbell.db1_dev_ptr, exp_seq_ptr, std::ptr::null(), std::ptr::null(), std::ptr::null(),
            state.sc1.gpu_moe_out.ptr.cast(), std::ptr::null(), state.mtp_h.ptr.cast(),
            n_embd as i32, 1, 1, stream,
        );

        glm5::rms_norm_q8(
            state.mtp_h.ptr.cast(),
            std::ptr::null(),
            lw.nextn_shnorm.unwrap() as *const f32,
            cfg.rms_norm_eps,
            n_embd as i32,
            false,
            state.sc1.x_f32.ptr.cast(),
            state.sc1.x_q8.ptr,
            1,
            stream,
        );

        let dv = &weights.draft_vocab;
        if use_dv && dv.n_lead < cfg.n_vocab {
            let n_lead = dv.n_lead as i32;
            launch_linear_q8_1(
                weights.output_weight_type as i32,
                weights.output_weight_dev.ptr,
                state.sc1.x_q8.ptr,
                state.sc1.x_f32.ptr.cast(),
                state.logits_dev.ptr.cast(),
                n_lead,
                n_embd as i32,
                1,
                stream.raw,
            );
            let mut total_rows = n_lead;
            if let Some(ref ew) = dv.extra_w_dev {
                let n_ex = dv.extra_ids.len() as i32;
                launch_linear_q8_1(
                    weights.output_weight_type as i32,
                    ew.ptr,
                    state.sc1.x_q8.ptr,
                    state.sc1.x_f32.ptr.cast(),
                    state.logits_dev.ptr.cast::<f32>().add(dv.n_lead).cast(),
                    n_ex,
                    n_embd as i32,
                    1,
                    stream.raw,
                );
                total_rows += n_ex;
            }
            if !need_draft_logits {
                let ex_ids = dv.extra_ids_dev.as_ref().map_or(std::ptr::null(), |b| b.ptr.cast());
                glm5::argmax(
                    state.logits_dev.ptr.cast(),
                    total_rows,
                    n_lead,
                    ex_ids,
                    state.argmax_pv.ptr.cast(),
                    state.argmax_pi.ptr.cast(),
                    state.mtp_tok_host_dev1 as *mut i32,
                    stream,
                );
            }
        } else {
            launch_linear_q8_1(
                weights.output_weight_type as i32,
                weights.output_weight_dev.ptr,
                state.sc1.x_q8.ptr,
                state.sc1.x_f32.ptr.cast(),
                state.logits_dev.ptr.cast(),
                cfg.n_vocab as i32,
                n_embd as i32,
                1,
                stream.raw,
            );
            if !need_draft_logits {
                glm5::argmax(
                    state.logits_dev.ptr.cast(),
                    cfg.n_vocab as i32,
                    cfg.n_vocab as i32,
                    std::ptr::null(),
                    state.argmax_pv.ptr.cast(),
                    state.argmax_pi.ptr.cast(),
                    state.mtp_tok_host_dev1 as *mut i32,
                    stream,
                );
            }
        }
    }
}

/// Queue full blk.45 MTP forward on GPU 1 at position `pos`, writing drafted token into `mtp_tok_host`
/// (or full logits into `mtp_logits_pin` when `need_draft_logits` is true).
pub fn enqueue_mtp_draft_on_gpu1(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    pos: usize,
    head_x_dev: *const f32,
    next_tok: u32,
    use_draft_vocab: bool,
    need_draft_logits: bool,
    use_cuda_graph: bool,
) -> Result<()> {
    if weights.cfg.mtp_layer.is_none() {
        return Ok(());
    }
    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;

    let mslot = 2 + (pos & 1);
    dequant_embed_to_host(gguf, weights, next_tok, &mut state.hop_emb_pin1.as_mut_slice()[..n_embd])?;
    state.pos_pin1.as_mut_slice()[mslot] = pos as i32;
    set_device(1)?;

    let pos_bytes = unsafe {
        std::slice::from_raw_parts(state.pos_pin1.host_ptr().add(mslot).cast::<u8>(), 4)
    };
    state.sc1.mtp_pos_dev.copy_from_host_async(pos_bytes, &state.sc1.stream)?;
    let emb_bytes = unsafe {
        std::slice::from_raw_parts(state.hop_emb_pin1.host_ptr().cast::<u8>(), n_embd * 4)
    };
    state.sc1.emb.copy_from_host_async(emb_bytes, &state.sc1.stream)?;

    let use_dv = !need_draft_logits && use_draft_vocab && weights.draft_vocab.n_lead < cfg.n_vocab;
    let hx_base = state.head_x.ptr.cast::<f32>();
    let prof = prof_enabled();

    if use_cuda_graph && head_x_dev == hx_base {
        if need_draft_logits {
            if state.mtp_graph_full_logits.is_none() {
                let g = CudaGraphExec::capture(&state.sc1.stream, || {
                    enqueue_mtp_draft_body_m1(weights, state, hx_base, false, true);
                    Ok(())
                })?;
                state.mtp_graph_full_logits = Some(g);
            }
            if prof {
                state.prof_mtp_start.record(&state.sc1.stream)?;
            }
            state.mtp_graph_full_logits.as_ref().unwrap().launch(&state.sc1.stream)?;
        } else if use_dv {
            if state.mtp_graph_dv.is_none() {
                let g = CudaGraphExec::capture(&state.sc1.stream, || {
                    enqueue_mtp_draft_body_m1(weights, state, hx_base, true, false);
                    Ok(())
                })?;
                state.mtp_graph_dv = Some(g);
            }
            if prof {
                state.prof_mtp_start.record(&state.sc1.stream)?;
            }
            state.mtp_graph_dv.as_ref().unwrap().launch(&state.sc1.stream)?;
        } else {
            if state.mtp_graph_full_greedy.is_none() {
                let g = CudaGraphExec::capture(&state.sc1.stream, || {
                    enqueue_mtp_draft_body_m1(weights, state, hx_base, false, false);
                    Ok(())
                })?;
                state.mtp_graph_full_greedy = Some(g);
            }
            if prof {
                state.prof_mtp_start.record(&state.sc1.stream)?;
            }
            state.mtp_graph_full_greedy.as_ref().unwrap().launch(&state.sc1.stream)?;
        }
    } else {
        if prof {
            state.prof_mtp_start.record(&state.sc1.stream)?;
        }
        enqueue_mtp_draft_body_m1(weights, state, head_x_dev, use_dv, need_draft_logits);
    }

    if need_draft_logits {
        unsafe {
            let lh = std::slice::from_raw_parts_mut(
                state.mtp_logits_pin.host_ptr().cast::<u8>(),
                cfg.n_vocab * 4,
            );
            state.logits_dev.copy_to_host_async(lh, &state.sc1.stream)?;
        }
    }
    if prof {
        state.prof_mtp_end.record(&state.sc1.stream)?;
    }
    state.ev_mtp_done.record(&state.sc1.stream)?;
    if state.mtp_cached_pos <= pos {
        state.mtp_cached_pos = pos + 1;
    }
    Ok(())
}
