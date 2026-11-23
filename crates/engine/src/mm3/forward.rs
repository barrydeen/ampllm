use crate::mm3::dspark::Mm3Dspark;
use crate::mm3::state::{
    Mm3GpuPromoState, Mm3GpuScratch, Mm3LayerKv, Mm3ProfStats, Mm3State, MM3_ATTN_SPLITS,
    MM3_DMA_BATCH, MM3_DMA_RING, MM3_MAX_VERIFY, MM3_META_SLOTS, MM3_PREFILL_GRP_WIN, MM3_SUB,
};
use crate::mm3::vision::Mm3ImageSpan;
use crate::mm3::weights::{
    Mm3FfnWeights, Mm3KvFormat, Mm3LayerExperts, Mm3LayerWeights, Mm3MoeMeta, Mm3Weights,
    QuantMatrixSpec,
};
use ampllm_cpu_kernels::mm3_cpu::{
    q6_k_row_to_f32_cpu, run_mm3_cpu_moe_grouped_prequant, run_mm3_cpu_moe_m1, Mm3CpuExpertJob,
    Mm3CpuGroupedExpert, Mm3CpuScratch,
};
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::{
    copy_d2d_raw_async, copy_d2h_raw_async, copy_h2d_raw_async, copy_peer_raw_async, ffi, glm5,
    mm3, set_device, CudaStream, PinnedHostBuffer,
};
use ampllm_gguf::{GgmlType, Mm3Config};
use anyhow::{anyhow, bail, Result};
use std::ffi::c_void;
use std::time::Instant;

const PROMO_DECAY: f32 = 0.95;
const PROMO_THRESH: f32 = 1.8;
const PROMO_MAX_PENDING: usize = 4;

#[inline]
unsafe fn linear_mm3(
    mat: QuantMatrixSpec,
    x_q8_1: *const c_void,
    x_f32_opt: *const f32,
    y_f32: *mut f32,
    t: usize,
    stream: &CudaStream,
) {
    if t <= MM3_MAX_VERIFY {
        if mat.ggml_type == GgmlType::Q6K && mat.n_in % 1024 == 0 {
            mm3::gv_q6k_multi(
                1,
                mat.ptr as *const c_void,
                y_f32,
                mat.n_out as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                x_q8_1,
                mat.n_in as i32,
                t as i32,
                stream,
            );
        } else if mat.ggml_type == GgmlType::Q8_0 && mat.n_in % 128 == 0 {
            mm3::gv_q8_0_multi(
                1,
                mat.ptr as *const c_void,
                y_f32,
                mat.n_out as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                x_q8_1,
                mat.n_in as i32,
                t as i32,
                stream,
            );
        } else {
            ffi::launch_linear_q8_1(
                mat.ggml_type as i32,
                mat.ptr as *const c_void,
                x_q8_1,
                x_f32_opt,
                y_f32,
                mat.n_out as i32,
                mat.n_in as i32,
                t as i32,
                stream.raw,
            );
        }
    } else {
        ffi::launch_gemm_mmq_mma(
            mat.ggml_type as i32,
            x_q8_1,
            mat.ptr as *const c_void,
            mat.row_bytes,
            y_f32,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            t as i32,
            mat.n_out as i32,
            mat.n_in as i32,
            1.0,
            0.0,
            stream.raw,
        );
    }
}

fn embed_chunk_host(
    weights: &Mm3Weights,
    pool: &CpuThreadPool,
    tokens: &[u32],
    pos0: usize,
    images: &[Mm3ImageSpan],
    dst: &mut [f32],
) -> Result<()> {
    let t = tokens.len();
    let d = weights.cfg.n_embd;
    if t == 1 {
        weights.embed_row_f32(tokens[0], &mut dst[..d])?;
    } else {
        let raw = weights.gguf.tensor_bytes("token_embd.weight")?;
        let rb = weights.cfg.n_embd;
        let row_bytes = ampllm_gguf::GgmlType::Q6K.row_bytes(rb);
        let n_vocab = weights.cfg.vocab_size;
        let tok_addr = tokens.as_ptr() as usize;
        let dst_addr = dst.as_ptr() as usize;
        let raw_addr = raw.as_ptr() as usize;
        pool.parallel_for(t, |i0, i1| {
            let toks = unsafe { std::slice::from_raw_parts(tok_addr as *const u32, t) };
            for i in i0..i1 {
                let tid = (toks[i] as usize).min(n_vocab.saturating_sub(1));
                let row_raw = unsafe {
                    std::slice::from_raw_parts(
                        (raw_addr + tid * row_bytes) as *const u8,
                        row_bytes,
                    )
                };
                let out_row = unsafe {
                    std::slice::from_raw_parts_mut((dst_addr as *mut f32).add(i * d), d)
                };
                q6_k_row_to_f32_cpu(row_raw, out_row);
            }
        });
    }

    for sp in images {
        let ov_s = sp.start_pos.max(pos0);
        let ov_e = (sp.start_pos + sp.n_tokens).min(pos0 + t);
        if ov_s < ov_e {
            for p in ov_s..ov_e {
                let src_off = (p - sp.start_pos) * d;
                let dst_off = (p - pos0) * d;
                dst[dst_off..dst_off + d].copy_from_slice(&sp.rows[src_off..src_off + d]);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_attention_block(
    lw: &Mm3LayerWeights,
    kv: &Mm3LayerKv,
    g: &Mm3GpuScratch,
    cfg: &Mm3Config,
    kv_fmt: Mm3KvFormat,
    rope_inv_freq: *const f32,
    t: usize,
    max_pos_inclusive: usize,
) -> Result<()> {
    let d = cfg.n_embd;
    let hq = cfg.n_heads;
    let hkv = cfg.n_kv_heads;
    let hd = cfg.head_dim;
    let q_dim = hq * hd;
    let eps = cfg.rms_norm_eps;
    let rot_dim = cfg.rot_dim;
    let kv_fmt_i32 = kv_fmt as i32;
    let rot_mask = if kv_fmt == Mm3KvFormat::Int8HadV { 2 } else { 0 };
    let scale = 1.0f32 / (hd as f32).sqrt();
    let nblk = (max_pos_inclusive + 1).div_ceil(cfg.ix_block_size);
    let splits = if t <= MM3_MAX_VERIFY {
        MM3_ATTN_SPLITS
    } else {
        1
    };
    let part_ptr = if t <= MM3_MAX_VERIFY {
        g.attn_part.ptr as *mut f32
    } else {
        std::ptr::null_mut()
    };

    unsafe {
        let h_ptr = g.h.ptr as *mut f32;
        let xn_ptr = g.xn.ptr as *mut f32;
        let xq8_1 = g.x_q8_1.ptr;
        let q_ptr = g.q.ptr as *mut f32;
        let k_ptr = g.k.ptr as *mut f32;
        let v_ptr = g.v.ptr as *mut f32;
        let iq_ptr = g.iq.ptr as *mut f32;
        let ik_ptr = g.ik.ptr as *mut f32;
        let pos_ptr = g.pos_dev.ptr as *const i32;
        let out_ptr = g.attn_out.ptr as *mut f32;
        let out_q8 = g.attn_q8_1.ptr;
        let block_out_ptr = g.block_out.ptr as *mut f32;

        // GGUF text backbone norms already have +1.0 baked in (`add_one = false`).
        mm3::rms_norm_f32(
            h_ptr,
            lw.attn_norm.ptr as *const f32,
            t as i32,
            d as i32,
            eps,
            false,
            xn_ptr,
            xq8_1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &g.stream,
        );
        g.norm_ev.record(&g.stream)?;
        g.side_stream.wait_event(&g.norm_ev)?;

        // Overlap MSA indexer projections + scoring on side_stream while QKV + RoPE + KV store run on main stream.
        if t <= MM3_MAX_VERIFY
            && lw.index_q.ggml_type == GgmlType::Q8_0
            && lw.index_k.ggml_type == GgmlType::Q8_0
            && lw.index_q.n_in == d
            && lw.index_k.n_in == d
            && d % 128 == 0
        {
            mm3::gv_q8_0_multi(
                2,
                lw.index_q.buf.ptr,
                iq_ptr,
                lw.index_q.n_out as i32,
                lw.index_k.buf.ptr,
                ik_ptr,
                lw.index_k.n_out as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                xq8_1,
                d as i32,
                t as i32,
                &g.side_stream,
            );
        } else {
            linear_mm3(lw.index_q.as_spec(), xq8_1, xn_ptr, iq_ptr, t, &g.side_stream);
            linear_mm3(lw.index_k.as_spec(), xq8_1, xn_ptr, ik_ptr, t, &g.side_stream);
        }
        mm3::index_norm_rope_store(
            iq_ptr,
            ik_ptr,
            lw.index_q_norm.ptr as *const f32,
            lw.index_k_norm.ptr as *const f32,
            pos_ptr,
            rope_inv_freq,
            t as i32,
            cfg.ix_heads as i32,
            cfg.ix_head_dim as i32,
            rot_dim as i32,
            eps,
            kv.ic_bf16.ptr as *mut u16,
            &g.side_stream,
        );
        mm3::msa_block_scores(
            iq_ptr,
            kv.ic_bf16.ptr as *const u16,
            pos_ptr,
            t as i32,
            cfg.ix_heads as i32,
            nblk as i32,
            cfg.ix_block_size as i32,
            g.idx_scores.ptr as *mut f32,
            &g.side_stream,
        );
        mm3::msa_select_topk(
            g.idx_scores.ptr as *const f32,
            pos_ptr,
            t as i32,
            cfg.ix_heads as i32,
            nblk as i32,
            cfg.ix_block_size as i32,
            cfg.ix_top_k as i32,
            cfg.ix_local_blocks as i32,
            g.idx_kept.ptr as *mut i32,
            &g.side_stream,
        );
        g.idx_done_ev.record(&g.side_stream)?;

        if t <= MM3_MAX_VERIFY
            && lw.attn_q.ggml_type == GgmlType::Q8_0
            && lw.attn_k.ggml_type == GgmlType::Q8_0
            && lw.attn_v.ggml_type == GgmlType::Q8_0
            && lw.attn_q.n_in == d
            && lw.attn_k.n_in == d
            && lw.attn_v.n_in == d
            && d % 128 == 0
        {
            mm3::gv_q8_0_multi(
                3,
                lw.attn_q.buf.ptr,
                q_ptr,
                lw.attn_q.n_out as i32,
                lw.attn_k.buf.ptr,
                k_ptr,
                lw.attn_k.n_out as i32,
                lw.attn_v.buf.ptr,
                v_ptr,
                lw.attn_v.n_out as i32,
                xq8_1,
                d as i32,
                t as i32,
                &g.stream,
            );
        } else {
            linear_mm3(lw.attn_q.as_spec(), xq8_1, xn_ptr, q_ptr, t, &g.stream);
            linear_mm3(lw.attn_k.as_spec(), xq8_1, xn_ptr, k_ptr, t, &g.stream);
            linear_mm3(lw.attn_v.as_spec(), xq8_1, xn_ptr, v_ptr, t, &g.stream);
        }

        let kc_bf16 = kv
            .kc_bf16
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.ptr as *mut u16);
        let vc_bf16 = kv
            .vc_bf16
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.ptr as *mut u16);
        let kc_i8 = kv
            .kc_i8
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.ptr as *mut i8);
        let ks_i8 = kv
            .ks_i8
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.ptr as *mut f32);
        let vc_i8 = kv
            .vc_i8
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.ptr as *mut i8);
        let vs_i8 = kv
            .vs_i8
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.ptr as *mut f32);

        mm3::qk_norm_rope_kv_store(
            q_ptr,
            k_ptr,
            v_ptr,
            lw.attn_q_norm.ptr as *const f32,
            lw.attn_k_norm.ptr as *const f32,
            pos_ptr,
            rope_inv_freq,
            t as i32,
            hq as i32,
            hkv as i32,
            hd as i32,
            rot_dim as i32,
            eps,
            kv_fmt_i32,
            rot_mask,
            0,
            kc_bf16,
            vc_bf16,
            kc_i8,
            ks_i8,
            vc_i8,
            vs_i8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &g.stream,
        );

        g.stream.wait_event(&g.idx_done_ev)?;

        mm3::msa_sparse_attn(
            q_ptr,
            kc_bf16,
            vc_bf16,
            kc_i8,
            ks_i8,
            vc_i8,
            vs_i8,
            g.idx_kept.ptr as *const i32,
            pos_ptr,
            t as i32,
            hq as i32,
            hkv as i32,
            cfg.ix_top_k as i32,
            cfg.ix_block_size as i32,
            kv_fmt_i32,
            rot_mask,
            0,
            scale,
            part_ptr,
            splits,
            out_ptr,
            &g.stream,
        );

        ffi::launch_quantize_q8_1(out_ptr, out_q8, t as i32, q_dim as i32, g.stream.raw);
        linear_mm3(lw.attn_o.as_spec(), out_q8, out_ptr, block_out_ptr, t, &g.stream);
        mm3::vision_residual_bias(
            h_ptr,
            block_out_ptr,
            std::ptr::null(),
            t * d,
            d as i32,
            &g.stream,
        );
    }
    Ok(())
}

fn run_dense_mlp_block(lw: &Mm3LayerWeights, g: &Mm3GpuScratch, cfg: &Mm3Config, t: usize) {
    let Mm3FfnWeights::Dense(ref dw) = lw.ffn else {
        unreachable!();
    };
    let d = cfg.n_embd;
    let ff = cfg.n_dense_ff;
    unsafe {
        let h_ptr = g.h.ptr as *mut f32;
        let xn_ptr = g.xn.ptr as *mut f32;
        let xq8_1 = g.x_q8_1.ptr;
        let gp = g.mlp_gate.ptr as *mut f32;
        let up = g.mlp_up.ptr as *mut f32;
        let iq8 = g.mlp_inter_q8_1.ptr;
        let bo = g.block_out.ptr as *mut f32;

        mm3::rms_norm_f32(
            h_ptr,
            lw.ffn_norm.ptr as *const f32,
            t as i32,
            d as i32,
            cfg.rms_norm_eps,
            false,
            xn_ptr,
            xq8_1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &g.stream,
        );
        if t <= MM3_MAX_VERIFY
            && dw.gate.ggml_type == GgmlType::Q6K
            && dw.up.ggml_type == GgmlType::Q6K
            && dw.gate.n_in == d
            && dw.up.n_in == d
            && d % 1024 == 0
        {
            mm3::gv_q6k_multi(
                2,
                dw.gate.buf.ptr,
                gp,
                dw.gate.n_out as i32,
                dw.up.buf.ptr,
                up,
                dw.up.n_out as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                xq8_1,
                d as i32,
                t as i32,
                &g.stream,
            );
        } else {
            linear_mm3(dw.gate.as_spec(), xq8_1, xn_ptr, gp, t, &g.stream);
            linear_mm3(dw.up.as_spec(), xq8_1, xn_ptr, up, t, &g.stream);
        }
        mm3::swigluoai(
            gp,
            up,
            t as i32,
            ff as i32,
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
            std::ptr::null_mut(),
            iq8,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &g.stream,
        );
        linear_mm3(dw.down.as_spec(), iq8, std::ptr::null(), bo, t, &g.stream);
        mm3::vision_residual_bias(h_ptr, bo, std::ptr::null(), t * d, d as i32, &g.stream);
    }
}

#[inline]
fn score_expert(
    use_score: &mut [f32],
    use_last_fwd: &mut [u64],
    fwd_id: u64,
    key: usize,
    add: bool,
) -> f32 {
    let last = use_last_fwd[key];
    if fwd_id > last {
        use_score[key] *= PROMO_DECAY.powi((fwd_id - last) as i32);
        use_last_fwd[key] = fwd_id;
    }
    if add {
        use_score[key] += 1.0;
    }
    use_score[key]
}

fn consider_promotion(
    il: usize,
    eu: usize,
    n_exp: usize,
    promo: &mut Mm3GpuPromoState,
    use_score: &mut [f32],
    use_last_fwd: &mut [u64],
    fwd_id: u64,
    ram_host_ptr: usize,
) {
    let sc = score_expert(use_score, use_last_fwd, fwd_id, il * n_exp + eu, true);
    if promo.n_slots == 0
        || ram_host_ptr == 0
        || sc < PROMO_THRESH
        || promo.promo_queue.len() >= PROMO_MAX_PENDING
    {
        return;
    }
    let mut best: Option<usize> = None;
    let mut best_sc = sc;
    for s in 0..promo.n_slots {
        if promo.slot_pend[s].is_some() || promo.slot_used_fwd[s] == fwd_id {
            continue;
        }
        let owner_sc = match promo.slot_owner[s] {
            None => -1.0f32,
            Some((oil, oe)) => score_expert(use_score, use_last_fwd, fwd_id, oil * n_exp + oe, false),
        };
        if owner_sc < best_sc {
            best_sc = owner_sc;
            best = Some(s);
        }
    }
    if let Some(s) = best {
        promo.slot_pend[s] = Some((il, eu));
        promo.promo_queue.push((s, il, eu));
    }
}

fn issue_gpu_promotions(weights: &Mm3Weights, state: &mut Mm3State, gidx: usize) -> Result<()> {
    let promo = &mut state.promo[gidx];
    if promo.promo_queue.is_empty() {
        return Ok(());
    }
    let g = &state.gpus[gidx];
    set_device(g.device)?;
    g.promo_gate_ev.record(&g.stream)?;
    g.copy_stream.wait_event(&g.promo_gate_ev)?;
    let slab_ptr = weights.promo_slabs[gidx]
        .as_ref()
        .ok_or_else(|| anyhow!("GPU {gidx} missing promo slab"))?
        .ptr;
    let stride = weights.promo_slot_strides[gidx];
    for (slot, il, e) in promo.promo_queue.drain(..) {
        let Mm3FfnWeights::Moe(ref mw) = weights.layers[il].ffn else {
            continue;
        };
        let Some(ref es) = weights.experts[il] else {
            continue;
        };
        let ram_ptr = es.ram_host_ptrs[e];
        if ram_ptr == 0 {
            continue;
        }
        unsafe {
            copy_h2d_raw_async(
                slab_ptr.add(slot * stride),
                std::slice::from_raw_parts(ram_ptr as *const u8, mw.blob_bytes),
                &g.copy_stream,
            )?;
        }
        promo.slot_ev[slot].record(&g.copy_stream)?;
        state.cache_stats.promotions += 1;
        state.prof.promos_issued += 1;
    }
    Ok(())
}

pub fn commit_gpu_promotions(
    weights: &mut Mm3Weights,
    state: &mut Mm3State,
    gidx: usize,
) -> Result<()> {
    issue_gpu_promotions(weights, state, gidx)?;
    let promo = &mut state.promo[gidx];
    if promo.n_slots == 0 {
        return Ok(());
    }
    if !promo.slot_pend.iter().any(|x| x.is_some()) {
        return Ok(());
    }
    let t_sync0 = state.prof_enabled.then(Instant::now);
    let dev = weights.gpu_ids[gidx];
    set_device(dev)?;
    let slab_base = weights.promo_slabs[gidx]
        .as_ref()
        .ok_or_else(|| anyhow!("GPU {gidx} missing promo slab"))?
        .as_u64();
    let stride = weights.promo_slot_strides[gidx];
    for s in 0..promo.n_slots {
        if let Some((new_il, new_e)) = promo.slot_pend[s].take() {
            promo.slot_ev[s].sync()?;
            if let Some((old_il, old_e)) = promo.slot_owner[s].take() {
                if let Some(ref mut old_es) = weights.experts[old_il] {
                    if !old_es.is_static_resident[old_e] {
                        old_es.is_resident[old_e] = false;
                        old_es.promo_slot_of[old_e] = -1;
                        old_es.gate_ptrs_host[old_e] = 0;
                        old_es.up_ptrs_host[old_e] = 0;
                        old_es.down_ptrs_host[old_e] = 0;
                    }
                }
            }
            let (u_off, d_off) = match weights.layers[new_il].ffn {
                Mm3FfnWeights::Moe(ref mw) => (mw.u_off, mw.d_off),
                _ => unreachable!(),
            };
            let addr = slab_base + (s * stride) as u64;
            if let Some(ref mut new_es) = weights.experts[new_il] {
                new_es.is_resident[new_e] = true;
                new_es.promo_slot_of[new_e] = s as i16;
                new_es.gate_ptrs_host[new_e] = addr;
                new_es.up_ptrs_host[new_e] = addr + u_off as u64;
                new_es.down_ptrs_host[new_e] = addr + d_off as u64;
            }
            promo.slot_owner[s] = Some((new_il, new_e));
            promo.slot_used_fwd[s] = state.fwd_id;
        }
    }
    if let Some(t0) = t_sync0 {
        state.prof.promo_sync_us += t0.elapsed().as_secs_f64() * 1e6;
    }
    Ok(())
}

pub fn commit_all_promotions(weights: &mut Mm3Weights, state: &mut Mm3State) -> Result<()> {
    for gidx in 0..state.promo.len() {
        commit_gpu_promotions(weights, state, gidx)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_decode_moe_block(
    il: usize,
    lw: &Mm3LayerWeights,
    es_ro: Option<&Mm3LayerExperts>,
    mut experts_rw: Option<&mut [Option<Mm3LayerExperts>]>,
    promo_slab_info: Option<(u64, usize)>,
    mw: &Mm3MoeMeta,
    g: &mut Mm3GpuScratch,
    pool: &CpuThreadPool,
    cpu_scratch: &mut Mm3CpuScratch,
    cfg: &Mm3Config,
    t: usize,
    profile_counts: &mut [u32],
    hits_acc: &mut u64,
    misses_acc: &mut u64,
    stage_split: bool,
    mut promo_opt: Option<&mut Mm3GpuPromoState>,
    mut score_state: Option<(&mut [f32], &mut [u64], u64)>,
    mut prof_opt: Option<&mut Mm3ProfStats>,
) -> Result<()> {
    let d = cfg.n_embd;
    let ff = cfg.expert_ff;
    let sh_ff = cfg.n_dense_ff.min(mw.sh_gate.n_out);
    let n_exp = cfg.n_experts;
    let topk = cfg.n_experts_used;
    let nb_x = d.div_ceil(256);
    let q8k_tok_bytes = nb_x * mm3::Q8K_BLOCK_BYTES;
    let q8k_exp_rb = ff.div_ceil(256) * mm3::Q8K_BLOCK_BYTES;
    let pairs = t * topk;
    if pairs > 32 {
        bail!("MM3 decode MoE supports at most 32 (token, slot) pairs, got {pairs}");
    }

    unsafe {
        mm3::rms_norm_f32(
            g.h.ptr as *const f32,
            lw.ffn_norm.ptr as *const f32,
            t as i32,
            d as i32,
            cfg.rms_norm_eps,
            false,
            g.xn.ptr as *mut f32,
            g.x_q8_1.ptr,
            g.x_q8_k.ptr,
            std::ptr::null_mut(),
            &g.stream,
        );
        g.norm_ev.record(&g.stream)?;
        g.side_stream.wait_event(&g.norm_ev)?;
        mm3::router_f32(
            g.xn.ptr as *const f32,
            mw.gate_inp.ptr as *const f32,
            g.router_logits.ptr as *mut f32,
            mw.exp_probs_bias.ptr as *const f32,
            t as i32,
            d as i32,
            n_exp as i32,
            topk as i32,
            cfg.routed_scaling_factor,
            g.router_ids.ptr as *mut i32,
            g.router_wts.ptr as *mut f32,
            &g.side_stream,
        );
        g.router_ev.record(&g.side_stream)?;
        g.d2h_stream.wait_event(&g.router_ev)?;
        copy_d2h_raw_async(
            &mut g.h_ids.as_mut_slice()[..pairs * 4],
            g.router_ids.ptr,
            &g.d2h_stream,
        )?;
        copy_d2h_raw_async(
            &mut g.h_wts.as_mut_slice()[..pairs * 4],
            g.router_wts.ptr,
            &g.d2h_stream,
        )?;
        copy_d2h_raw_async(
            &mut g.h_x_q8k.as_mut_slice()[..t * q8k_tok_bytes],
            g.x_q8_k.ptr,
            &g.d2h_stream,
        )?;
        g.d2h_ev.record(&g.d2h_stream)?;

        // Launch shared expert on g.stream while router + D2H run concurrently.
        if t <= MM3_MAX_VERIFY
            && mw.sh_gate.ggml_type == GgmlType::Q6K
            && mw.sh_up.ggml_type == GgmlType::Q6K
            && mw.sh_gate.n_in == d
            && mw.sh_up.n_in == d
            && d % 1024 == 0
        {
            mm3::gv_q6k_multi(
                2,
                mw.sh_gate.buf.ptr,
                g.mlp_gate.ptr as *mut f32,
                mw.sh_gate.n_out as i32,
                mw.sh_up.buf.ptr,
                g.mlp_up.ptr as *mut f32,
                mw.sh_up.n_out as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                g.x_q8_1.ptr,
                d as i32,
                t as i32,
                &g.stream,
            );
        } else {
            linear_mm3(
                mw.sh_gate.as_spec(),
                g.x_q8_1.ptr,
                g.xn.ptr as *const f32,
                g.mlp_gate.ptr as *mut f32,
                t,
                &g.stream,
            );
            linear_mm3(
                mw.sh_up.as_spec(),
                g.x_q8_1.ptr,
                g.xn.ptr as *const f32,
                g.mlp_up.ptr as *mut f32,
                t,
                &g.stream,
            );
        }
        mm3::swigluoai(
            g.mlp_gate.ptr as *const f32,
            g.mlp_up.ptr as *const f32,
            t as i32,
            sh_ff as i32,
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
            std::ptr::null_mut(),
            g.mlp_inter_q8_1.ptr,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &g.stream,
        );
        linear_mm3(
            mw.sh_down.as_spec(),
            g.mlp_inter_q8_1.ptr,
            std::ptr::null(),
            g.sh_out.ptr as *mut f32,
            t,
            &g.stream,
        );
    }

    let t_d2h0 = prof_opt.as_ref().map(|_| Instant::now());
    g.d2h_ev.sync()?;
    if let (Some(t0), Some(prof)) = (t_d2h0, prof_opt.as_deref_mut()) {
        prof.moe_d2h_us += t0.elapsed().as_secs_f64() * 1e6;
    }

    let ids = bytemuck::cast_slice::<u8, i32>(&g.h_ids.as_slice()[..pairs * 4]);
    let wts = bytemuck::cast_slice::<u8, f32>(&g.h_wts.as_slice()[..pairs * 4]);

    let mut grp_eid = [0usize; 32];
    let mut grp_is_vram = [false; 32];
    let mut grp_gate_ptr = [0u64; 32];
    let mut grp_ram_ptr = [0usize; 32];
    let mut grp_head = [-1i8; 32];
    let mut grp_tail = [-1i8; 32];
    let mut grp_len = [0usize; 32];
    let mut pair_next = [-1i8; 32];
    let mut grp_cnt = 0usize;
    let mut vram_grp_cnt = 0usize;

    {
        let es = if let Some(ref arr) = experts_rw {
            arr[il]
                .as_ref()
                .ok_or_else(|| anyhow!("layer {il} missing expert storage"))?
        } else {
            es_ro.ok_or_else(|| anyhow!("layer {il} missing read-only expert storage"))?
        };

        for idx in 0..pairs {
            let e = ids[idx];
            if e < 0 || (e as usize) >= n_exp {
                continue;
            }
            let eu = e as usize;
            profile_counts[eu] += 1;
            let is_v = es.is_resident[eu];
            if is_v {
                *hits_acc += 1;
                if let Some(ref mut prof) = prof_opt {
                    if es.is_static_resident[eu] {
                        prof.static_hits += 1;
                    } else {
                        prof.promo_hits += 1;
                    }
                }
            } else {
                *misses_acc += 1;
            }

            if let Some(gi) = grp_eid[..grp_cnt].iter().position(|&x| x == eu) {
                pair_next[grp_tail[gi] as usize] = idx as i8;
                grp_tail[gi] = idx as i8;
                grp_len[gi] += 1;
            } else {
                let gi = grp_cnt;
                grp_eid[gi] = eu;
                grp_is_vram[gi] = is_v;
                grp_gate_ptr[gi] = es.gate_ptrs_host[eu];
                grp_ram_ptr[gi] = es.ram_host_ptrs[eu];
                grp_head[gi] = idx as i8;
                grp_tail[gi] = idx as i8;
                grp_len[gi] = 1;
                grp_cnt += 1;
                if is_v {
                    vram_grp_cnt += 1;
                    if !es.is_static_resident[eu] {
                        let ps = es.promo_slot_of[eu];
                        if ps >= 0 {
                            if let (Some(ref mut promo), Some((ref mut us, ref mut ul, fid))) =
                                (&mut promo_opt, &mut score_state)
                            {
                                let p_idx = ps as usize;
                                if p_idx < promo.slot_used_fwd.len() {
                                    promo.slot_used_fwd[p_idx] = *fid;
                                }
                                score_expert(us, ul, *fid, il * n_exp + eu, true);
                            }
                        }
                    }
                } else if let (Some(ref mut promo), Some((ref mut us, ref mut ul, fid))) =
                    (&mut promo_opt, &mut score_state)
                {
                    consider_promotion(il, eu, n_exp, promo, us, ul, *fid, es.ram_host_ptrs[eu]);
                }
            }
        }
    }

    let mut cold_order = [0usize; 32];
    let mut n_cold = 0usize;
    for gi in 0..grp_cnt {
        if !grp_is_vram[gi] {
            let eu = grp_eid[gi];
            if grp_ram_ptr[gi] == 0 {
                bail!("layer {il} expert {eu} missing from both VRAM and pinned RAM");
            }
            cold_order[n_cold] = gi;
            n_cold += 1;
        }
    }
    if n_cold > 1 {
        cold_order[..n_cold].sort_unstable_by(|&a, &b| {
            grp_len[b]
                .cmp(&grp_len[a])
                .then_with(|| grp_eid[a].cmp(&grp_eid[b]))
        });
    }

    let n_stage = if !stage_split || n_cold == 0 {
        0
    } else if t == 1 {
        if n_cold >= 6 { 1 } else { 0 }
    } else {
        let copy_ms = 1.10f64;
        let gpu_exp_ms = 0.14f64;
        let vram_est = (vram_grp_cnt as f64) * gpu_exp_ms;
        let max_s = n_cold.min(MM3_DMA_BATCH).min(4);
        let extra_all: f64 = cold_order[..n_cold]
            .iter()
            .map(|&gi| (grp_len[gi] - 1) as f64)
            .sum();
        let base_t = (0.34 + 0.28 * ((n_cold - 1) as f64) + 0.04 * extra_all).max(vram_est);
        let mut best_s = 0usize;
        let mut best_t = base_t;
        for s in 1..=max_s {
            let rem = n_cold - s;
            let cpu_rem = if rem == 0 {
                0.0
            } else {
                let extra_rem: f64 = cold_order[s..n_cold]
                    .iter()
                    .map(|&gi| (grp_len[gi] - 1) as f64)
                    .sum();
                0.34 + 0.28 * ((rem - 1) as f64) + 0.04 * extra_rem
            };
            let gpu_t = (vram_est + (s as f64) * gpu_exp_ms).max((s as f64) * copy_ms);
            let tot_t = cpu_rem.max(gpu_t);
            if tot_t < best_t {
                best_s = s;
                best_t = tot_t;
            }
        }
        if best_t < base_t * 0.90 {
            best_s
        } else {
            0
        }
    };

    if let Some(ref mut prof) = prof_opt {
        prof.staged_exps += n_stage as u64;
        prof.cpu_exps += (n_cold - n_stage) as u64;
    }

    // Build plan0 (VRAM experts) & plan1 (staged cold experts) in pinned buffers.
    let (ng0, ne0, max_nt0) = {
        let p0 = unsafe { &mut *(g.h_decode_plan[0].as_mut_ptr() as *mut mm3::Mm3DecodePlanC) };
        let mut ng = 0usize;
        let mut ne = 0usize;
        let mut max_nt = 0usize;
        for gi in 0..grp_cnt {
            if !grp_is_vram[gi] {
                continue;
            }
            p0.base_ptrs[ng] = grp_gate_ptr[gi];
            p0.grp_off[ng] = ne as i32;
            let start_ne = ne;
            let mut cur = grp_head[gi];
            while cur >= 0 {
                let idx = cur as usize;
                p0.entry_tok[ne] = (idx / topk) as i32;
                p0.entry_wt[ne] = wts[idx];
                ne += 1;
                cur = pair_next[idx];
            }
            max_nt = max_nt.max(ne - start_ne);
            ng += 1;
        }
        p0.grp_off[ng] = ne as i32;
        (ng, ne, max_nt)
    };

    let plan_bytes = std::mem::size_of::<mm3::Mm3DecodePlanC>();
    if ne0 > 0 {
        unsafe {
            copy_h2d_raw_async(
                g.decode_plan_dev[0].ptr,
                &g.h_decode_plan[0].as_slice()[..plan_bytes],
                &g.stream,
            )?;
            mm3::moe_grp_gate_up_f32(
                mw.gate_type as i32,
                mw.up_type as i32,
                g.decode_plan_dev[0].ptr,
                mw.u_off,
                mw.gate_row_bytes,
                mw.up_row_bytes,
                g.x_q8_k.ptr,
                g.moe_inter_f32.ptr as *mut f32,
                d as i32,
                ff as i32,
                ng0 as i32,
                max_nt0 as i32,
                cfg.swiglu_alpha,
                cfg.swiglu_limit,
                &g.stream,
            );
            mm3::quantize_q8_k(
                g.moe_inter_f32.ptr as *const f32,
                g.moe_inter_q8k.ptr,
                ne0 as i32,
                ff as i32,
                &g.stream,
            );
            mm3::moe_grp_down_q8k(
                mw.down_type as i32,
                g.decode_plan_dev[0].ptr,
                mw.d_off,
                mw.down_row_bytes,
                g.moe_inter_q8k.ptr,
                g.moe_expert_out.ptr as *mut f32,
                ff as i32,
                d as i32,
                ng0 as i32,
                max_nt0 as i32,
                &g.stream,
            );
        }
    }

    let (ng1, ne1, max_nt1, dma_r, used_dma_stage) = if n_stage > 0 {
        let r = g.decode_dma_cursor % MM3_DMA_RING;
        set_device(g.device)?;
        let dma_base = g.dma_stage.ptr as usize;
        let slot_stride = g.slot_stride;
        let bsz = mw.blob_bytes;
        let base_s = r * MM3_DMA_BATCH;
        let p1 = unsafe { &mut *(g.h_decode_plan[1].as_mut_ptr() as *mut mm3::Mm3DecodePlanC) };
        let mut ne = 0usize;
        let mut max_nt = 0usize;
        let mut dma_synced = false;

        for s in 0..n_stage {
            let gi = cold_order[s];
            let eu = grp_eid[gi];
            let src_ptr = grp_ram_ptr[gi] as *const u8;
            let mut retained_addr: Option<u64> = None;

            if let (Some(ref mut exp_arr), Some(ref mut promo), Some((slab_base, stride))) =
                (&mut experts_rw, &mut promo_opt, promo_slab_info)
            {
                if promo.n_slots > 0 {
                    let fid = score_state.as_ref().map_or(0, |(_, _, id)| *id);
                    let mut picked = promo.slot_pend.iter().position(|&x| x == Some((il, eu)));
                    if let Some(qs) = picked {
                        promo.slot_pend[qs] = None;
                        promo
                            .promo_queue
                            .retain(|&(_, qil, qe)| !(qil == il && qe == eu));
                    } else {
                        let cand_sc = if let Some((ref mut us, ref mut ul, fid_v)) = score_state {
                            score_expert(us, ul, fid_v, il * n_exp + eu, false)
                        } else {
                            0.0f32
                        };
                        let mut best_sc = cand_sc;
                        for ps in 0..promo.n_slots {
                            if promo.slot_pend[ps].is_some() || promo.slot_used_fwd[ps] == fid {
                                continue;
                            }
                            let owner_sc = match promo.slot_owner[ps] {
                                None => -1.0f32,
                                Some((oil, oe)) => {
                                    if let Some((ref mut us, ref mut ul, fid_v)) = score_state {
                                        score_expert(us, ul, fid_v, oil * n_exp + oe, false)
                                    } else {
                                        0.0f32
                                    }
                                }
                            };
                            if owner_sc < best_sc {
                                best_sc = owner_sc;
                                picked = Some(ps);
                            }
                        }
                    }
                    if let Some(ps) = picked {
                        if let Some((old_il, old_e)) = promo.slot_owner[ps].take() {
                            if let Some(ref mut old_es) = exp_arr[old_il] {
                                if !old_es.is_static_resident[old_e] {
                                    old_es.is_resident[old_e] = false;
                                    old_es.promo_slot_of[old_e] = -1;
                                    old_es.gate_ptrs_host[old_e] = 0;
                                    old_es.up_ptrs_host[old_e] = 0;
                                    old_es.down_ptrs_host[old_e] = 0;
                                }
                            }
                        }
                        let addr = slab_base + (ps * stride) as u64;
                        unsafe {
                            copy_h2d_raw_async(
                                addr as *mut c_void,
                                std::slice::from_raw_parts(src_ptr, bsz),
                                &g.copy_stream,
                            )?;
                        }
                        if let Some(ref mut cur_es) = exp_arr[il] {
                            cur_es.is_resident[eu] = true;
                            cur_es.promo_slot_of[eu] = ps as i16;
                            cur_es.gate_ptrs_host[eu] = addr;
                            cur_es.up_ptrs_host[eu] = addr + mw.u_off as u64;
                            cur_es.down_ptrs_host[eu] = addr + mw.d_off as u64;
                        }
                        promo.slot_owner[ps] = Some((il, eu));
                        promo.slot_used_fwd[ps] = fid;
                        if let Some(ref mut prof) = prof_opt {
                            prof.promos_issued += 1;
                        }
                        retained_addr = Some(addr);
                    }
                }
            }

            let dst_addr = if let Some(addr) = retained_addr {
                addr as usize
            } else {
                if !dma_synced {
                    g.copy_stream.wait_event(&g.compute_events[r])?;
                    g.decode_dma_cursor += 1;
                    dma_synced = true;
                }
                let stage_addr = dma_base + (base_s + s) * slot_stride;
                unsafe {
                    copy_h2d_raw_async(
                        stage_addr as *mut c_void,
                        std::slice::from_raw_parts(src_ptr, bsz),
                        &g.copy_stream,
                    )?;
                }
                stage_addr
            };

            p1.base_ptrs[s] = dst_addr as u64;
            p1.grp_off[s] = ne as i32;
            let start_ne = ne;
            let mut cur = grp_head[gi];
            while cur >= 0 {
                let idx = cur as usize;
                p1.entry_tok[ne] = (idx / topk) as i32;
                p1.entry_wt[ne] = wts[idx];
                ne += 1;
                cur = pair_next[idx];
            }
            max_nt = max_nt.max(ne - start_ne);
        }
        p1.grp_off[n_stage] = ne as i32;
        g.dma_events[r].record(&g.copy_stream)?;
        (n_stage, ne, max_nt, r, dma_synced)
    } else {
        (0, 0, 0, 0, false)
    };

    let cpu_rem = n_cold - n_stage;
    let cpu_wait_val = unsafe {
        if ne1 > 0 {
            g.stream.wait_event(&g.dma_events[dma_r])?;
            copy_h2d_raw_async(
                g.decode_plan_dev[1].ptr,
                &g.h_decode_plan[1].as_slice()[..plan_bytes],
                &g.stream,
            )?;
            mm3::moe_grp_gate_up_f32(
                mw.gate_type as i32,
                mw.up_type as i32,
                g.decode_plan_dev[1].ptr,
                mw.u_off,
                mw.gate_row_bytes,
                mw.up_row_bytes,
                g.x_q8_k.ptr,
                (g.moe_inter_f32.ptr as *mut f32).add(ne0 * ff),
                d as i32,
                ff as i32,
                ng1 as i32,
                max_nt1 as i32,
                cfg.swiglu_alpha,
                cfg.swiglu_limit,
                &g.stream,
            );
            mm3::quantize_q8_k(
                (g.moe_inter_f32.ptr as *const f32).add(ne0 * ff),
                g.moe_inter_q8k.ptr.add(ne0 * q8k_exp_rb),
                ne1 as i32,
                ff as i32,
                &g.stream,
            );
            mm3::moe_grp_down_q8k(
                mw.down_type as i32,
                g.decode_plan_dev[1].ptr,
                mw.d_off,
                mw.down_row_bytes,
                g.moe_inter_q8k.ptr.add(ne0 * q8k_exp_rb),
                (g.moe_expert_out.ptr as *mut f32).add(ne0 * d),
                ff as i32,
                d as i32,
                ng1 as i32,
                max_nt1 as i32,
                &g.stream,
            );
            if used_dma_stage {
                g.compute_events[dma_r].record(&g.stream)?;
            }
        }
        let wait_opt = if cpu_rem > 0 {
            g.cpu_wait_seq = g.cpu_wait_seq.wrapping_add(1);
            if g.cpu_wait_seq == 0 {
                g.cpu_wait_seq = 1;
            }
            let wait_val = g.cpu_wait_seq;
            mm3::wait_u32(
                g.h_cpu_wait_mapped.dev_ptr() as *const u32,
                wait_val,
                &g.stream,
            );
            Some(wait_val)
        } else {
            None
        };
        let cpu_out_dev_ptr = if wait_opt.is_some() {
            g.h_cpu_out_mapped.dev_ptr() as *const f32
        } else {
            std::ptr::null()
        };
        mm3::moe_grp_combine(
            g.h.ptr as *const f32,
            g.sh_out.ptr as *const f32,
            g.moe_expert_out.ptr as *const f32,
            if ne0 > 0 { g.decode_plan_dev[0].ptr } else { std::ptr::null() },
            ne0 as i32,
            if ne1 > 0 { g.decode_plan_dev[1].ptr } else { std::ptr::null() },
            ne1 as i32,
            cpu_out_dev_ptr,
            t as i32,
            d as i32,
            g.h.ptr as *mut f32,
            &g.stream,
        );
        wait_opt
    };

    // Run remaining cold experts on CPU concurrently with GPU MoE!
    if let Some(wait_val) = cpu_wait_val {
        struct WaitRelease {
            ptr: *mut u32,
            val: u32,
        }
        impl Drop for WaitRelease {
            fn drop(&mut self) {
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                unsafe {
                    std::ptr::write_volatile(self.ptr, self.val);
                }
            }
        }
        let _wait_guard = WaitRelease {
            ptr: g.h_cpu_wait_mapped.host_ptr(),
            val: wait_val,
        };
        let t_cpu0 = prof_opt.as_ref().map(|_| Instant::now());
        if t == 1 {
            let mut jobs: [Mm3CpuExpertJob; 16] = [Mm3CpuExpertJob {
                gate_ptr: std::ptr::null(),
                up_ptr: std::ptr::null(),
                down_ptr: std::ptr::null(),
                weight: 0.0,
            }; 16];
            let mut nj = 0usize;
            for &gi in &cold_order[n_stage..n_cold] {
                let base = grp_ram_ptr[gi];
                let mut cur = grp_head[gi];
                while cur >= 0 {
                    let idx = cur as usize;
                    jobs[nj] = Mm3CpuExpertJob {
                        gate_ptr: base as *const u8,
                        up_ptr: (base + mw.u_off) as *const u8,
                        down_ptr: (base + mw.d_off) as *const u8,
                        weight: wts[idx],
                    };
                    nj += 1;
                    cur = pair_next[idx];
                }
            }
            if nj > 0 {
                let x_pre = &g.h_x_q8k.as_slice()[..q8k_tok_bytes];
                let cpu_out = &mut g.h_cpu_out_mapped.as_mut_slice()[..d];
                run_mm3_cpu_moe_m1(
                    pool,
                    cpu_scratch,
                    mw.gate_type,
                    mw.up_type,
                    mw.down_type,
                    d,
                    ff,
                    cfg.swiglu_alpha,
                    cfg.swiglu_limit,
                    &[],
                    Some(x_pre),
                    &jobs[..nj],
                    cpu_out,
                );
            } else {
                g.h_cpu_out_mapped.as_mut_slice()[..d].fill(0.0);
            }
        } else {
            let need_x = t * nb_x;
            if cpu_scratch.x_q8k.len() < need_x {
                cpu_scratch.x_q8k.resize(need_x, Default::default());
            }
            let x_bytes = &g.h_x_q8k.as_slice()[..t * q8k_tok_bytes];
            for tok in 0..t {
                unsafe {
                    let dst = cpu_scratch.x_q8k.as_mut_ptr().add(tok * nb_x) as *mut u8;
                    std::ptr::copy_nonoverlapping(
                        x_bytes[tok * q8k_tok_bytes..(tok + 1) * q8k_tok_bytes].as_ptr(),
                        dst,
                        q8k_tok_bytes,
                    );
                }
            }
            let mut toks_flat = [0usize; 32];
            let mut wts_flat = [0.0f32; 32];
            let mut grp_bounds = [(0usize, 0usize, 0usize); 32];
            let mut flat_n = 0usize;
            for (ci, &gi) in cold_order[n_stage..n_cold].iter().enumerate() {
                let s0 = flat_n;
                let mut cur = grp_head[gi];
                while cur >= 0 {
                    let idx = cur as usize;
                    toks_flat[flat_n] = idx / topk;
                    wts_flat[flat_n] = wts[idx];
                    flat_n += 1;
                    cur = pair_next[idx];
                }
                grp_bounds[ci] = (grp_ram_ptr[gi], s0, flat_n);
            }
            let mut groups: [Mm3CpuGroupedExpert<'_>; 32] = std::array::from_fn(|_| Mm3CpuGroupedExpert {
                gate_ptr: std::ptr::null(),
                up_ptr: std::ptr::null(),
                down_ptr: std::ptr::null(),
                token_indices: &[],
                token_weights: &[],
            });
            for (ci, &(base, s0, s1)) in grp_bounds[..cpu_rem].iter().enumerate() {
                groups[ci] = Mm3CpuGroupedExpert {
                    gate_ptr: base as *const u8,
                    up_ptr: (base + mw.u_off) as *const u8,
                    down_ptr: (base + mw.d_off) as *const u8,
                    token_indices: &toks_flat[s0..s1],
                    token_weights: &wts_flat[s0..s1],
                };
            }
            let cpu_out = &mut g.h_cpu_out_mapped.as_mut_slice()[..t * d];
            run_mm3_cpu_moe_grouped_prequant(
                pool,
                cpu_scratch,
                mw.gate_type,
                mw.up_type,
                mw.down_type,
                d,
                ff,
                cfg.swiglu_alpha,
                cfg.swiglu_limit,
                t,
                &groups[..cpu_rem],
                cpu_out,
                true,
            );
        }
        if let (Some(t0), Some(prof)) = (t_cpu0, prof_opt.as_deref_mut()) {
            prof.moe_cpu_us += t0.elapsed().as_secs_f64() * 1e6;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_mm3_moe_group_mmq_class(
    g: &Mm3GpuScratch,
    mw: &Mm3MoeMeta,
    cfg: &Mm3Config,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)],
    tile_mode: i32,
    bm: usize,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let stream = &g.stream;
    let d = cfg.n_embd;
    let ff = cfg.expert_ff;
    let n_exp_max = cfg.n_experts;

    let dispatch_one_window =
        |win: &[(u64, usize, usize)], meta_cursor: &mut usize| -> Result<()> {
            if win.is_empty() {
                return Ok(());
            }
            let slot = *meta_cursor % MM3_META_SLOTS;
            if *meta_cursor >= MM3_META_SLOTS {
                g.meta_events[slot].sync()?;
            }
            *meta_cursor += 1;

            let base_row_off = win[0].1;
            let n_exp = win.len();
            let ms = &g.meta_slots[slot];
            let ptrs_pin = std::slice::from_raw_parts_mut(ms.h_exp_ptrs.as_ptr() as *mut u64, n_exp);
            let off_pin = std::slice::from_raw_parts_mut(ms.h_exp_row_off.as_ptr() as *mut i32, n_exp);
            let nt_pin = std::slice::from_raw_parts_mut(ms.h_exp_nt.as_ptr() as *mut i32, n_exp);
            let te_pin = std::slice::from_raw_parts_mut(ms.h_tile_exp_id.as_ptr() as *mut i32, g.max_tiles);
            let tm_pin = std::slice::from_raw_parts_mut(ms.h_tile_m_off.as_ptr() as *mut i32, g.max_tiles);

            let mut w_rows = 0usize;
            let mut n_tiles = 0usize;
            for (ei, &(ptr, row_off, nt)) in win.iter().enumerate() {
                debug_assert_eq!(row_off - base_row_off, w_rows);
                ptrs_pin[ei] = ptr;
                off_pin[ei] = w_rows as i32;
                nt_pin[ei] = nt as i32;
                w_rows += nt;
                let mut m_off = 0usize;
                while m_off < nt {
                    te_pin[n_tiles] = ei as i32;
                    tm_pin[n_tiles] = m_off as i32;
                    n_tiles += 1;
                    m_off += bm;
                }
            }

            let d_ptrs = ms.exp_ptrs.ptr as *const u64;
            let d_off = ms.exp_row_off.ptr as *const i32;
            let d_nt = ms.exp_nt.ptr as *const i32;
            let d_te = ms.tile_exp_id.ptr as *const i32;
            let d_tm = ms.tile_m_off.ptr as *const i32;

            copy_h2d_raw_async(
                ms.exp_ptrs.ptr,
                std::slice::from_raw_parts(ptrs_pin.as_ptr() as *const u8, n_exp * 8),
                stream,
            )?;
            copy_h2d_raw_async(
                ms.exp_row_off.ptr,
                std::slice::from_raw_parts(off_pin.as_ptr() as *const u8, n_exp * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                ms.exp_nt.ptr,
                std::slice::from_raw_parts(nt_pin.as_ptr() as *const u8, n_exp * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                ms.tile_exp_id.ptr,
                std::slice::from_raw_parts(te_pin.as_ptr() as *const u8, n_tiles * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                ms.tile_m_off.ptr,
                std::slice::from_raw_parts(tm_pin.as_ptr() as *const u8, n_tiles * 4),
                stream,
            )?;

            let rows_dev = (g.grp_rows_dev.ptr as *const i32).add(base_row_off);
            let wts_dev = (g.grp_wts_dev.ptr as *const f32).add(base_row_off);
            let x_q8_1 = g.x_q8_1.ptr;
            let gate_f32 = g.moe_grp_gate.ptr as *mut f32;
            let up_f32 = g.moe_grp_up.ptr as *mut f32;
            let inter_q8_1 = g.moe_grp_inter_q8_1.ptr;
            let block_out = g.block_out.ptr as *mut f32;

            // Independent gate & up launches preserve correctness across mixed-quant layers.
            ffi::launch_gemm_mmq_mma_grouped(
                mw.gate_type as i32,
                x_q8_1,
                d_ptrs,
                0,
                mw.gate_row_bytes,
                gate_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                ff as i32,
                d as i32,
                1.0,
                0,
                tile_mode,
                stream.raw,
            );
            ffi::launch_gemm_mmq_mma_grouped(
                mw.up_type as i32,
                x_q8_1,
                d_ptrs,
                mw.u_off,
                mw.up_row_bytes,
                up_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                ff as i32,
                d as i32,
                1.0,
                0,
                tile_mode,
                stream.raw,
            );
            mm3::swigluoai(
                gate_f32,
                up_f32,
                w_rows as i32,
                ff as i32,
                cfg.swiglu_alpha,
                cfg.swiglu_limit,
                std::ptr::null_mut(),
                inter_q8_1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                stream,
            );
            ffi::launch_gemm_mmq_mma_grouped(
                mw.down_type as i32,
                inter_q8_1,
                d_ptrs,
                mw.d_off,
                mw.down_row_bytes,
                block_out,
                std::ptr::null(),
                rows_dev,
                wts_dev,
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                d as i32,
                ff as i32,
                1.0,
                1,
                tile_mode,
                stream.raw,
            );

            g.meta_events[slot].record(stream)?;
            Ok(())
        };

    let mut idx = 0usize;
    while idx < jobs.len() {
        let mut end = idx;
        let mut rows = 0usize;
        let mut tiles = 0usize;
        let mut exps = 0usize;
        while end < jobs.len() {
            let (_, _, nt) = jobs[end];
            let jt = nt.div_ceil(bm);
            if end > idx
                && (rows + nt > MM3_PREFILL_GRP_WIN || tiles + jt > g.max_tiles || exps + 1 > n_exp_max)
            {
                break;
            }
            rows += nt;
            tiles += jt;
            exps += 1;
            end += 1;
        }
        dispatch_one_window(&jobs[idx..end], meta_cursor)?;
        idx = end;
    }
    Ok(())
}

unsafe fn run_mm3_moe_group_mmq(
    g: &Mm3GpuScratch,
    mw: &Mm3MoeMeta,
    cfg: &Mm3Config,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)],
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let s1 = jobs.iter().take_while(|j| j.2 > 96).count();
    let s2 = s1 + jobs[s1..].iter().take_while(|j| j.2 > 32).count();
    run_mm3_moe_group_mmq_class(g, mw, cfg, meta_cursor, &jobs[..s1], 1, 128)?;
    run_mm3_moe_group_mmq_class(g, mw, cfg, meta_cursor, &jobs[s1..s2], 2, 64)?;
    run_mm3_moe_group_mmq_class(g, mw, cfg, meta_cursor, &jobs[s2..], 3, 32)?;
    Ok(())
}

struct ColdBucket {
    eid: usize,
    row_off: usize,
    tok_indices: Vec<usize>,
    tok_wts: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_moe_block(
    il: usize,
    lw: &Mm3LayerWeights,
    es: &Mm3LayerExperts,
    mw: &Mm3MoeMeta,
    g: &mut Mm3GpuScratch,
    pool: &CpuThreadPool,
    cpu_scratch: &mut Mm3CpuScratch,
    cfg: &Mm3Config,
    t: usize,
    profile_counts: &mut [u32],
    hits_acc: &mut u64,
    misses_acc: &mut u64,
    stage_split: bool,
) -> Result<()> {
    if t <= MM3_MAX_VERIFY {
        return run_decode_moe_block(
            il,
            lw,
            Some(es),
            None,
            None,
            mw,
            g,
            pool,
            cpu_scratch,
            cfg,
            t,
            profile_counts,
            hits_acc,
            misses_acc,
            stage_split,
            None,
            None,
            None,
        );
    }

    let d = cfg.n_embd;
    let ff = cfg.expert_ff;
    let sh_ff = cfg.n_dense_ff.min(mw.sh_gate.n_out);
    let n_exp = cfg.n_experts;
    let topk = cfg.n_experts_used;
    let nb_x = d.div_ceil(256);
    let q8k_tok_bytes = nb_x * mm3::Q8K_BLOCK_BYTES;
    let pairs = t * topk;

    unsafe {
        mm3::rms_norm_f32(
            g.h.ptr as *const f32,
            lw.ffn_norm.ptr as *const f32,
            t as i32,
            d as i32,
            cfg.rms_norm_eps,
            false,
            g.xn.ptr as *mut f32,
            g.x_q8_1.ptr,
            g.x_q8_k.ptr,
            std::ptr::null_mut(),
            &g.stream,
        );
        g.norm_ev.record(&g.stream)?;
        g.side_stream.wait_event(&g.norm_ev)?;
        mm3::router_f32(
            g.xn.ptr as *const f32,
            mw.gate_inp.ptr as *const f32,
            g.router_logits.ptr as *mut f32,
            mw.exp_probs_bias.ptr as *const f32,
            t as i32,
            d as i32,
            n_exp as i32,
            topk as i32,
            cfg.routed_scaling_factor,
            g.router_ids.ptr as *mut i32,
            g.router_wts.ptr as *mut f32,
            &g.side_stream,
        );
        g.router_ev.record(&g.side_stream)?;
        g.d2h_stream.wait_event(&g.router_ev)?;
        copy_d2h_raw_async(
            &mut g.h_ids.as_mut_slice()[..pairs * 4],
            g.router_ids.ptr,
            &g.d2h_stream,
        )?;
        copy_d2h_raw_async(
            &mut g.h_wts.as_mut_slice()[..pairs * 4],
            g.router_wts.ptr,
            &g.d2h_stream,
        )?;
        copy_d2h_raw_async(
            &mut g.h_x_q8k.as_mut_slice()[..t * q8k_tok_bytes],
            g.x_q8_k.ptr,
            &g.d2h_stream,
        )?;
        g.d2h_ev.record(&g.d2h_stream)?;

        // Initialize block_out with the shared expert output while router D2H finishes.
        linear_mm3(
            mw.sh_gate.as_spec(),
            g.x_q8_1.ptr,
            g.xn.ptr as *const f32,
            g.mlp_gate.ptr as *mut f32,
            t,
            &g.stream,
        );
        linear_mm3(
            mw.sh_up.as_spec(),
            g.x_q8_1.ptr,
            g.xn.ptr as *const f32,
            g.mlp_up.ptr as *mut f32,
            t,
            &g.stream,
        );
        mm3::swigluoai(
            g.mlp_gate.ptr as *const f32,
            g.mlp_up.ptr as *const f32,
            t as i32,
            sh_ff as i32,
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
            std::ptr::null_mut(),
            g.mlp_inter_q8_1.ptr,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &g.stream,
        );
        linear_mm3(
            mw.sh_down.as_spec(),
            g.mlp_inter_q8_1.ptr,
            std::ptr::null(),
            g.block_out.ptr as *mut f32,
            t,
            &g.stream,
        );
    }

    g.d2h_ev.sync()?;
    let ids = bytemuck::cast_slice::<u8, i32>(&g.h_ids.as_slice()[..pairs * 4]);
    let wts = bytemuck::cast_slice::<u8, f32>(&g.h_wts.as_slice()[..pairs * 4]);

    let mut raw_buckets: Vec<(Vec<usize>, Vec<f32>)> =
        (0..n_exp).map(|_| (Vec::new(), Vec::new())).collect();
    for idx in 0..pairs {
        let e = ids[idx];
        if e >= 0 && (e as usize) < n_exp {
            let eu = e as usize;
            profile_counts[eu] += 1;
            raw_buckets[eu].0.push(idx / topk);
            raw_buckets[eu].1.push(wts[idx]);
            if es.is_resident[eu] {
                *hits_acc += 1;
            } else {
                *misses_acc += 1;
            }
        }
    }

    let mut vram_raw: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut cold_list: Vec<ColdBucket> = Vec::new();
    for (eu, (toks, wt_vec)) in raw_buckets.into_iter().enumerate() {
        if toks.is_empty() {
            continue;
        }
        let target = if es.gate_ptrs_host[eu] != 0 {
            &mut vram_raw
        } else {
            // Cold list split below into ColdBucket windows.
            let mut s = 0usize;
            while s < toks.len() {
                let end = (s + MM3_PREFILL_GRP_WIN).min(toks.len());
                cold_list.push(ColdBucket {
                    eid: eu,
                    row_off: 0,
                    tok_indices: toks[s..end].to_vec(),
                    tok_wts: wt_vec[s..end].to_vec(),
                });
                s = end;
            }
            continue;
        };
        let mut s = 0usize;
        while s < toks.len() {
            let end = (s + MM3_PREFILL_GRP_WIN).min(toks.len());
            target.push((eu, toks[s..end].to_vec(), wt_vec[s..end].to_vec()));
            s = end;
        }
    }

    vram_raw.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    cold_list.sort_by(|a, b| {
        b.tok_indices
            .len()
            .cmp(&a.tok_indices.len())
            .then_with(|| a.eid.cmp(&b.eid))
    });

    // Balance cold tail onto overlapped AVX-512 CPU vs double-buffered PCIe DMA -> GPU MMQ.
    let k_cold = cold_list.len();
    let n_gpu_cold = if k_cold == 0 {
        0
    } else {
        let cpu_max_nt = (t / 12).clamp(16, 128);
        let cpu_scale = 8.0 / (pool.num_threads().max(1) as f64);
        let blob_us = (mw.blob_bytes as f64) / 11_500.0;
        let comp_cost = |nt: usize| -> f64 { 20.0 + 1.1 * (nt as f64) };
        let vram_us: f64 = vram_raw.iter().map(|v| comp_cost(v.1.len())).sum();
        let out_h2d_us = ((t * d * 4) as f64) / 11_500.0;

        let mut pref_gpu_comp = vec![0.0f64; k_cold + 1];
        for i in 0..k_cold {
            pref_gpu_comp[i + 1] = pref_gpu_comp[i] + comp_cost(cold_list[i].tok_indices.len());
        }
        let mut suff_cpu = vec![0.0f64; k_cold + 1];
        let mut suff_ok = vec![true; k_cold + 1];
        for i in (0..k_cold).rev() {
            let nt = cold_list[i].tok_indices.len();
            suff_ok[i] = suff_ok[i + 1] && nt <= cpu_max_nt;
            suff_cpu[i] = suff_cpu[i + 1] + (120.0 + 28.0 * (nt as f64)) * cpu_scale;
        }
        let mut split = k_cold;
        for cand in (0..k_cold).rev() {
            if !suff_ok[cand] {
                break;
            }
            let cpu_est = suff_cpu[cand] + out_h2d_us;
            let gpu_est = (vram_us + pref_gpu_comp[cand]).max((cand as f64) * blob_us);
            if cpu_est <= 0.95 * gpu_est {
                split = cand;
            } else {
                break;
            }
        }
        split
    };

    let rows_pin = bytemuck::cast_slice_mut::<u8, i32>(&mut g.h_grp_rows.as_mut_slice()[..pairs * 4]);
    let wts_pin = bytemuck::cast_slice_mut::<u8, f32>(&mut g.h_grp_wts.as_mut_slice()[..pairs * 4]);
    let mut cur_off = 0usize;
    let mut vram_jobs: Vec<(u64, usize, usize)> = Vec::with_capacity(vram_raw.len());
    for (eu, toks, wt_vec) in &vram_raw {
        let nt = toks.len();
        vram_jobs.push((es.gate_ptrs_host[*eu], cur_off, nt));
        for k in 0..nt {
            rows_pin[cur_off + k] = toks[k] as i32;
            wts_pin[cur_off + k] = wt_vec[k];
        }
        cur_off += nt;
    }

    let mut cold_groups: Vec<(usize, usize)> = Vec::new();
    {
        let mut i = 0usize;
        while i < n_gpu_cold {
            let gs = i;
            let mut g_rows = 0usize;
            while i < n_gpu_cold
                && (i - gs) < MM3_DMA_BATCH
                && (i == gs || g_rows + cold_list[i].tok_indices.len() <= MM3_PREFILL_GRP_WIN)
            {
                let nt = cold_list[i].tok_indices.len();
                cold_list[i].row_off = cur_off;
                for k in 0..nt {
                    rows_pin[cur_off + k] = cold_list[i].tok_indices[k] as i32;
                    wts_pin[cur_off + k] = cold_list[i].tok_wts[k];
                }
                cur_off += nt;
                g_rows += nt;
                i += 1;
            }
            cold_groups.push((gs, i));
        }
    }

    if cur_off > 0 {
        unsafe {
            copy_h2d_raw_async(
                g.grp_rows_dev.ptr,
                std::slice::from_raw_parts(rows_pin.as_ptr() as *const u8, cur_off * 4),
                &g.stream,
            )?;
            copy_h2d_raw_async(
                g.grp_wts_dev.ptr,
                std::slice::from_raw_parts(wts_pin.as_ptr() as *const u8, cur_off * 4),
                &g.stream,
            )?;
        }
    }

    let cpu_jobs = &cold_list[n_gpu_cold..];
    for c in cpu_jobs {
        if es.ram_host_ptrs[c.eid] == 0 {
            bail!(
                "layer {il} expert {} missing from both VRAM and pinned RAM",
                c.eid
            );
        }
    }
    for c in &cold_list[..n_gpu_cold] {
        if es.ram_host_ptrs[c.eid] == 0 {
            bail!(
                "layer {il} expert {} missing from both VRAM and pinned RAM",
                c.eid
            );
        }
    }

    // Prepare CPU prequantized Q8_K activations before overlapping CPU || GPU.
    let need_x = t * nb_x;
    if !cpu_jobs.is_empty() {
        if cpu_scratch.x_q8k.len() < need_x {
            cpu_scratch.x_q8k.resize(need_x, Default::default());
        }
        let x_bytes = &g.h_x_q8k.as_slice()[..t * q8k_tok_bytes];
        for tok in 0..t {
            unsafe {
                let dst = cpu_scratch.x_q8k.as_mut_ptr().add(tok * nb_x) as *mut u8;
                std::ptr::copy_nonoverlapping(
                    x_bytes[tok * q8k_tok_bytes..(tok + 1) * q8k_tok_bytes].as_ptr(),
                    dst,
                    q8k_tok_bytes,
                );
            }
        }
    }

    let dev_id = g.device;
    let cpu_out_addr = g.h_cpu_out.as_mut_ptr() as usize;
    let ram_host_ptrs = &es.ram_host_ptrs;
    let dma_base = g.dma_stage.ptr as usize;
    let slot_stride = g.slot_stride;
    let u_off = mw.u_off;
    let d_off = mw.d_off;
    let blob_bytes = mw.blob_bytes;

    let kick_dma_group = |g_idx: usize, r_slot: usize, wait_compute: bool| -> Result<()> {
        set_device(dev_id)?;
        if wait_compute {
            g.copy_stream.wait_event(&g.compute_events[r_slot])?;
        } else {
            g.copy_stream.wait_event(&g.router_ev)?;
        }
        let (gs, ge) = cold_groups[g_idx];
        let base_s = r_slot * MM3_DMA_BATCH;
        for i in 0..(ge - gs) {
            let eid = cold_list[gs + i].eid;
            let src_ptr = ram_host_ptrs[eid] as *const u8;
            let dst_ptr = unsafe { (dma_base as *mut u8).add((base_s + i) * slot_stride) as *mut c_void };
            unsafe {
                copy_h2d_raw_async(
                    dst_ptr,
                    std::slice::from_raw_parts(src_ptr, blob_bytes),
                    &g.copy_stream,
                )?;
            }
        }
        g.dma_events[r_slot].record(&g.copy_stream)?;
        Ok(())
    };

    for r_slot in 0..MM3_DMA_RING.min(cold_groups.len()) {
        kick_dma_group(r_slot, r_slot, false)?;
    }

    let (cpu_res, gpu_res) = std::thread::scope(|scope| {
        let h_cpu = (!cpu_jobs.is_empty()).then(|| {
            scope.spawn(move || -> Result<bool> {
                let cpu_out =
                    unsafe { std::slice::from_raw_parts_mut(cpu_out_addr as *mut f32, t * d) };
                let mut did_any = false;
                let mut idx = 0usize;
                while idx < cpu_jobs.len() {
                    let mut end = idx;
                    let mut batch_rows = 0usize;
                    let mut count = 0usize;
                    while end < cpu_jobs.len() && count < 16 {
                        let next_nt = cpu_jobs[end].tok_indices.len();
                        if count > 0 && batch_rows + next_nt > 1024 {
                            break;
                        }
                        batch_rows += next_nt;
                        end += 1;
                        count += 1;
                    }
                    let groups: Vec<Mm3CpuGroupedExpert<'_>> = cpu_jobs[idx..end]
                        .iter()
                        .map(|job| {
                            let base = ram_host_ptrs[job.eid];
                            Mm3CpuGroupedExpert {
                                gate_ptr: base as *const u8,
                                up_ptr: (base + u_off) as *const u8,
                                down_ptr: (base + d_off) as *const u8,
                                token_indices: &job.tok_indices,
                                token_weights: &job.tok_wts,
                            }
                        })
                        .collect();
                    run_mm3_cpu_moe_grouped_prequant(
                        pool,
                        cpu_scratch,
                        mw.gate_type,
                        mw.up_type,
                        mw.down_type,
                        d,
                        ff,
                        cfg.swiglu_alpha,
                        cfg.swiglu_limit,
                        t,
                        &groups,
                        cpu_out,
                        !did_any,
                    );
                    did_any = true;
                    idx = end;
                }
                Ok(did_any)
            })
        });

        let gpu_closure = (|| -> Result<()> {
            set_device(dev_id)?;
            let mut meta_cursor = 0usize;
            if !vram_jobs.is_empty() {
                unsafe {
                    run_mm3_moe_group_mmq(g, mw, cfg, &mut meta_cursor, &vram_jobs)?;
                }
            }
            let mut grp_jobs: Vec<(u64, usize, usize)> = Vec::with_capacity(MM3_DMA_BATCH);
            for (g_idx, &(gs, ge)) in cold_groups.iter().enumerate() {
                let r_slot = g_idx % MM3_DMA_RING;
                let base_s = r_slot * MM3_DMA_BATCH;
                g.stream.wait_event(&g.dma_events[r_slot])?;
                grp_jobs.clear();
                for i in 0..(ge - gs) {
                    let b = &cold_list[gs + i];
                    grp_jobs.push((
                        (dma_base + (base_s + i) * slot_stride) as u64,
                        b.row_off,
                        b.tok_indices.len(),
                    ));
                }
                unsafe {
                    run_mm3_moe_group_mmq(g, mw, cfg, &mut meta_cursor, &grp_jobs)?;
                }
                g.compute_events[r_slot].record(&g.stream)?;
                let next_g = g_idx + MM3_DMA_RING;
                if next_g < cold_groups.len() {
                    kick_dma_group(next_g, r_slot, true)?;
                }
            }
            Ok(())
        })();

        let c_res = match h_cpu {
            Some(h) => h
                .join()
                .unwrap_or_else(|_| Err(anyhow!("MM3 CPU prefill MoE thread panicked"))),
            None => Ok(false),
        };
        (c_res, gpu_closure)
    });

    gpu_res?;
    if cpu_res? {
        unsafe {
            copy_h2d_raw_async(
                g.xn.ptr,
                &g.h_cpu_out.as_slice()[..t * d * 4],
                &g.stream,
            )?;
            mm3::vision_residual_bias(
                g.block_out.ptr as *mut f32,
                g.xn.ptr as *const f32,
                std::ptr::null(),
                t * d,
                d as i32,
                &g.stream,
            );
        }
    }
    unsafe {
        mm3::vision_residual_bias(
            g.h.ptr as *mut f32,
            g.block_out.ptr as *const f32,
            std::ptr::null(),
            t * d,
            d as i32,
            &g.stream,
        );
    }
    Ok(())
}

fn capture_tap_if_needed(
    dspark: Option<&Mm3Dspark>,
    g: &Mm3GpuScratch,
    il: usize,
    t: usize,
    stride: usize,
    d: usize,
) -> Result<()> {
    let Some(ds) = dspark else {
        return Ok(());
    };
    let bytes = t * d * 4;
    if g.is_head_gpu {
        if let Some(tap_j) = ds.cfg.target_layer_ids.iter().position(|&l| l == il) {
            if let Some(ref taps_dev) = g.taps_dev {
                unsafe {
                    copy_d2d_raw_async(
                        taps_dev.ptr.add(tap_j * stride * d * 4),
                        g.h.ptr,
                        bytes,
                        &g.stream,
                    )?;
                }
            }
        }
    } else if let Some(local_idx) = g.local_tap_indices.iter().position(|&(l, _)| l == il) {
        if let Some(ref local_taps) = g.local_taps_dev {
            unsafe {
                copy_d2d_raw_async(
                    local_taps.ptr.add(local_idx * stride * d * 4),
                    g.h.ptr,
                    bytes,
                    &g.stream,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handoff_gpu0_to_head_refs(
    p2p_enabled: bool,
    d: usize,
    g0: &Mm3GpuScratch,
    g1: &Mm3GpuScratch,
    xfer_h_pin: &mut PinnedHostBuffer,
    taps_xfer_pin: Option<&mut PinnedHostBuffer>,
    t: usize,
    stride: usize,
    copy_taps: bool,
) -> Result<()> {
    let bytes = t * d * 4;
    if p2p_enabled {
        set_device(g0.device)?;
        g0.handoff_ev.record(&g0.stream)?;
        set_device(g1.device)?;
        g1.stream.wait_event(&g0.handoff_ev)?;
        unsafe {
            copy_peer_raw_async(g1.h.ptr, g1.device, g0.h.ptr, g0.device, bytes, &g1.stream)?;
            if copy_taps {
                if let (Some(ref local_taps), Some(ref taps_dev)) =
                    (&g0.local_taps_dev, &g1.taps_dev)
                {
                    for (li, &(_, tap_j)) in g0.local_tap_indices.iter().enumerate() {
                        copy_peer_raw_async(
                            taps_dev.ptr.add(tap_j * stride * d * 4),
                            g1.device,
                            local_taps.ptr.add(li * stride * d * 4),
                            g0.device,
                            bytes,
                            &g1.stream,
                        )?;
                    }
                }
            }
        }
        g1.p2p_done_ev.record(&g1.stream)?;
        set_device(g0.device)?;
        g0.stream.wait_event(&g1.p2p_done_ev)?;
        return Ok(());
    }

    set_device(g0.device)?;
    let mut taps_pin_opt = taps_xfer_pin;
    unsafe {
        copy_d2h_raw_async(&mut xfer_h_pin.as_mut_slice()[..bytes], g0.h.ptr, &g0.stream)?;
        if copy_taps {
            if let (Some(ref local_taps), Some(ref mut taps_pin)) =
                (&g0.local_taps_dev, taps_pin_opt.as_deref_mut())
            {
                for (li, _) in g0.local_tap_indices.iter().enumerate() {
                    let src_off = li * stride * d * 4;
                    let dst_off = li * bytes;
                    copy_d2h_raw_async(
                        &mut taps_pin.as_mut_slice()[dst_off..dst_off + bytes],
                        local_taps.ptr.add(src_off),
                        &g0.stream,
                    )?;
                }
            }
        }
    }
    g0.stream.sync()?;

    set_device(g1.device)?;
    unsafe {
        copy_h2d_raw_async(g1.h.ptr, &xfer_h_pin.as_slice()[..bytes], &g1.stream)?;
        if copy_taps {
            if let (Some(ref taps_dev), Some(ref taps_pin)) = (&g1.taps_dev, taps_pin_opt) {
                for (li, &(_, tap_j)) in g0.local_tap_indices.iter().enumerate() {
                    let src_off = li * bytes;
                    let dst_off = tap_j * stride * d * 4;
                    copy_h2d_raw_async(
                        taps_dev.ptr.add(dst_off),
                        &taps_pin.as_slice()[src_off..src_off + bytes],
                        &g1.stream,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn handoff_gpu0_to_head(
    weights: &Mm3Weights,
    state: &mut Mm3State,
    t: usize,
    stride: usize,
    copy_taps: bool,
) -> Result<()> {
    let (left, right) = state.gpus.split_at_mut(1);
    handoff_gpu0_to_head_refs(
        weights.p2p_enabled,
        weights.cfg.n_embd,
        &left[0],
        &right[0],
        &mut state.xfer_h_pin,
        state.taps_xfer_pin.as_mut(),
        t,
        stride,
        copy_taps,
    )
}

fn emit_decode_logits_and_argmax(
    weights: &Mm3Weights,
    state: &mut Mm3State,
    t: usize,
    greedy_argmax: bool,
) -> Result<Vec<u32>> {
    let cfg = &weights.cfg;
    let d = cfg.n_embd;
    let vocab = cfg.vocab_size;
    let eps = cfg.rms_norm_eps;
    let head_idx = state.gpus.len() - 1;
    let hg = &mut state.gpus[head_idx];
    set_device(hg.device)?;

    let logits_buf = hg
        .logits
        .as_ref()
        .ok_or_else(|| anyhow!("head GPU missing logits buffer"))?;

    unsafe {
        mm3::rms_norm_f32(
            hg.h.ptr as *const f32,
            weights.output_norm.ptr as *const f32,
            t as i32,
            d as i32,
            eps,
            false,
            hg.xn.ptr as *mut f32,
            hg.x_q8_1.ptr,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &hg.stream,
        );
        linear_mm3(
            weights.lm_head.as_spec(),
            hg.x_q8_1.ptr,
            hg.xn.ptr as *const f32,
            logits_buf.ptr as *mut f32,
            t,
            &hg.stream,
        );
    }

    if greedy_argmax {
        let vals_p = hg.argmax_vals.as_ref().unwrap().ptr as *mut f32;
        let ids_p = hg.argmax_ids.as_ref().unwrap().ptr as *mut i32;
        let out_p = hg.argmax_out.as_ref().unwrap().ptr as *mut i32;
        let h_arg = hg.h_argmax.as_mut().unwrap();
        unsafe {
            for r in 0..t {
                glm5::argmax(
                    (logits_buf.ptr as *const f32).add(r * vocab),
                    vocab as i32,
                    vocab as i32,
                    std::ptr::null(),
                    vals_p,
                    ids_p,
                    out_p.add(r),
                    &hg.stream,
                );
            }
            copy_d2h_raw_async(&mut h_arg.as_mut_slice()[..t * 4], out_p as *mut c_void, &hg.stream)?;
        }
        hg.stream.sync()?;
        let toks = bytemuck::cast_slice::<u8, i32>(&h_arg.as_slice()[..t * 4]);
        Ok(toks.iter().map(|&v| v.max(0) as u32).collect())
    } else {
        unsafe {
            copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut state.logits_host[..t * vocab]),
                logits_buf.ptr,
                &hg.stream,
            )?;
        }
        hg.stream.sync()?;
        Ok(Vec::new())
    }
}

pub struct Mm3LayerTrace {
    pub after_attn: Vec<Vec<f32>>,
    pub after_ffn: Vec<Vec<f32>>,
    pub router_last: Vec<Vec<(i32, f32, bool)>>,
    pub l03_all_ffn: Vec<f32>,
}

impl Mm3LayerTrace {
    pub fn new(n_layers: usize) -> Self {
        Self {
            after_attn: vec![Vec::new(); n_layers],
            after_ffn: vec![Vec::new(); n_layers],
            router_last: vec![Vec::new(); n_layers],
            l03_all_ffn: Vec::new(),
        }
    }
}

fn copy_last_row_h(g: &Mm3GpuScratch, t: usize, d: usize) -> Result<Vec<f32>> {
    set_device(g.device)?;
    g.stream.sync()?;
    let mut out = vec![0.0f32; d];
    unsafe {
        let src = (g.h.ptr as *const u8).add((t - 1) * d * 4) as *const c_void;
        copy_d2h_raw_async(bytemuck::cast_slice_mut::<f32, u8>(&mut out), src, &g.stream)?;
    }
    g.stream.sync()?;
    Ok(out)
}

fn copy_all_rows_h(g: &Mm3GpuScratch, t: usize, d: usize) -> Result<Vec<f32>> {
    set_device(g.device)?;
    g.stream.sync()?;
    let mut out = vec![0.0f32; t * d];
    unsafe {
        copy_d2h_raw_async(
            bytemuck::cast_slice_mut::<f32, u8>(&mut out),
            g.h.ptr as *const c_void,
            &g.stream,
        )?;
    }
    g.stream.sync()?;
    Ok(out)
}

fn record_router_last(
    weights: &Mm3Weights,
    g: &Mm3GpuScratch,
    il: usize,
    t: usize,
    tr: &mut Mm3LayerTrace,
) {
    if let Mm3FfnWeights::Moe(_) = weights.layers[il].ffn {
        let topk = weights.cfg.n_experts_used;
        let base = (t - 1) * topk;
        let ids = bytemuck::cast_slice::<u8, i32>(&g.h_ids.as_slice()[..t * topk * 4]);
        let wts = bytemuck::cast_slice::<u8, f32>(&g.h_wts.as_slice()[..t * topk * 4]);
        let es = weights.experts[il].as_ref();
        tr.router_last[il] = (0..topk)
            .map(|k| {
                let id = ids[base + k];
                let wt = wts[base + k];
                let res = es.is_some_and(|e| {
                    id >= 0 && (id as usize) < e.is_resident.len() && e.is_resident[id as usize]
                });
                (id, wt, res)
            })
            .collect();
    }
}

/// Evaluate `1 <= tokens.len() <= MM3_MAX_VERIFY` tokens from `pos0`.
/// Returns greedy argmax token IDs when `compute_logits && greedy_argmax`,
/// or writes full logits into `state.logits_host[..t * vocab]` when `compute_logits && !greedy_argmax`.
pub fn forward_decode_verify(
    weights: &mut Mm3Weights,
    state: &mut Mm3State,
    tokens: &[u32],
    pos0: usize,
    feed_dspark: bool,
    compute_logits: bool,
    greedy_argmax: bool,
) -> Result<Vec<u32>> {
    forward_decode_verify_traced(
        weights,
        state,
        tokens,
        pos0,
        feed_dspark,
        compute_logits,
        greedy_argmax,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_decode_verify_traced(
    weights: &mut Mm3Weights,
    state: &mut Mm3State,
    tokens: &[u32],
    pos0: usize,
    feed_dspark: bool,
    compute_logits: bool,
    greedy_argmax: bool,
    mut trace: Option<&mut Mm3LayerTrace>,
) -> Result<Vec<u32>> {
    let t = tokens.len();
    if t == 0 || t > MM3_MAX_VERIFY {
        bail!("forward_decode_verify expects 1..={MM3_MAX_VERIFY} tokens, got {t}");
    }
    if pos0 + t > state.max_ctx {
        bail!(
            "MM3 sequence length {} exceeds max_ctx {}",
            pos0 + t,
            state.max_ctx
        );
    }

    if feed_dspark {
        if let Some(ref ds) = weights.dspark {
            set_device(ds.device)?;
            ds.stream.sync()?;
        }
    }

    state.fwd_id += 1;
    let fwd_id = state.fwd_id;
    commit_gpu_promotions(weights, state, 0)?;

    let cfg = weights.cfg.clone();
    let d = cfg.n_embd;
    let n_exp = cfg.n_experts;
    let kv_fmt = weights.kv_fmt;
    let layer_split = weights.layer_split;
    let has_two_gpus = state.gpus.len() >= 2;
    let stage_split = state.stage_split;
    let prof_enabled = state.prof_enabled;
    if prof_enabled {
        state.prof.steps += 1;
        state.prof.tokens += t as u64;
    }

    {
        let emb_f32 = bytemuck::cast_slice_mut::<u8, f32>(&mut state.emb_pin.as_mut_slice()[..t * d * 4]);
        embed_chunk_host(weights, &state.pool, tokens, pos0, &[], emb_f32)?;
    }

    for g in &mut state.gpus {
        set_device(g.device)?;
        unsafe {
            mm3::iota(g.pos_dev.ptr as *mut i32, t as i32, pos0 as i32, &g.stream);
        }
    }

    {
        let g0 = &state.gpus[0];
        set_device(g0.device)?;
        unsafe {
            copy_h2d_raw_async(g0.h.ptr, &state.emb_pin.as_slice()[..t * d * 4], &g0.stream)?;
        }
    }

    let max_pos_inc = pos0 + t - 1;
    let mut total_hits = 0u64;
    let mut total_misses = 0u64;

    for il in 0..cfg.n_layers {
        if has_two_gpus && il == layer_split {
            commit_gpu_promotions(weights, state, 1)?;
        }
        let lw = &weights.layers[il];
        let gidx = lw.gpu_idx;
        let g = &mut state.gpus[gidx];
        set_device(g.device)?;
        let inv_freq = weights.rope_inv_freq[gidx].ptr as *const f32;
        let t_att0 = prof_enabled.then(Instant::now);
        run_attention_block(lw, &state.kv[il], g, &cfg, kv_fmt, inv_freq, t, max_pos_inc)?;
        if let Some(t0) = t_att0 {
            if prof_enabled {
                state.prof.attn_us += t0.elapsed().as_secs_f64() * 1e6;
            }
        }
        if let Some(tr) = trace.as_deref_mut() {
            tr.after_attn[il] = copy_last_row_h(g, t, d)?;
        }

        match lw.ffn {
            Mm3FfnWeights::Dense(_) => run_dense_mlp_block(lw, g, &cfg, t),
            Mm3FfnWeights::Moe(ref mw) => {
                let promo_slab_info = weights.promo_slabs[gidx]
                    .as_ref()
                    .map(|b| (b.as_u64(), weights.promo_slot_strides[gidx]));
                run_decode_moe_block(
                    il,
                    lw,
                    None,
                    Some(&mut weights.experts),
                    promo_slab_info,
                    mw,
                    g,
                    &state.pool,
                    &mut state.cpu_scratch,
                    &cfg,
                    t,
                    &mut state.profile_counts[il],
                    &mut total_hits,
                    &mut total_misses,
                    stage_split,
                    state.promo.get_mut(gidx),
                    Some((
                        &mut state.use_score[..cfg.n_layers * n_exp],
                        &mut state.use_last_fwd[..cfg.n_layers * n_exp],
                        fwd_id,
                    )),
                    prof_enabled.then_some(&mut state.prof),
                )?;
            }
        }
        if let Some(tr) = trace.as_deref_mut() {
            tr.after_ffn[il] = copy_last_row_h(g, t, d)?;
            record_router_last(weights, g, il, t, tr);
        }
        if feed_dspark {
            capture_tap_if_needed(weights.dspark.as_ref(), g, il, t, MM3_SUB, d)?;
        }

        if has_two_gpus && il + 1 == layer_split {
            let t_ho0 = prof_enabled.then(Instant::now);
            handoff_gpu0_to_head(weights, state, t, MM3_SUB, feed_dspark)?;
            issue_gpu_promotions(weights, state, 0)?;
            if let Some(t0) = t_ho0 {
                state.prof.handoff_us += t0.elapsed().as_secs_f64() * 1e6;
            }
        }
    }

    state.cache_stats.hits += total_hits;
    state.cache_stats.misses += total_misses;

    let head_idx = state.gpus.len() - 1;
    let t_lm0 = prof_enabled.then(Instant::now);
    let res = if compute_logits {
        emit_decode_logits_and_argmax(weights, state, t, greedy_argmax)?
    } else {
        Vec::new()
    };
    if let Some(t0) = t_lm0 {
        state.prof.lm_head_us += t0.elapsed().as_secs_f64() * 1e6;
    }

    if feed_dspark && weights.dspark.is_some() {
        let hg = &state.gpus[head_idx];
        set_device(hg.device)?;
        if !compute_logits {
            hg.taps_ready_ev.record(&hg.stream)?;
        }
        let taps_ptr = hg
            .taps_dev
            .as_ref()
            .ok_or_else(|| anyhow!("head GPU missing taps_dev"))?
            .ptr as *const f32;
        if let Some(ref mut ds) = weights.dspark {
            set_device(ds.device)?;
            if !compute_logits {
                ds.stream.wait_event(&hg.taps_ready_ev)?;
            }
            ds.context_from_taps(taps_ptr, t, pos0, MM3_SUB)?;
        }
    }

    issue_gpu_promotions(weights, state, head_idx)?;

    state.pos = pos0 + t;
    Ok(res)
}

/// Evaluate an arbitrary prompt slice `[tokens]` starting at `pos0`, supporting multimodal `Mm3ImageSpan`s.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill(
    weights: &mut Mm3Weights,
    state: &mut Mm3State,
    tokens: &[u32],
    pos0: usize,
    images: &[Mm3ImageSpan],
    feed_dspark: bool,
    compute_last_logits: bool,
    greedy_last: bool,
) -> Result<Option<u32>> {
    forward_prefill_traced(
        weights,
        state,
        tokens,
        pos0,
        images,
        feed_dspark,
        compute_last_logits,
        greedy_last,
        None,
    )
}

/// Helper to execute layers `layer_start..layer_end` on GPU `gidx` for prefill chunk `t`.
#[allow(clippy::too_many_arguments)]
fn run_prefill_gpu_stage(
    weights: &Mm3Weights,
    dspark: Option<&Mm3Dspark>,
    g: &mut Mm3GpuScratch,
    kv_slice: &mut [Mm3LayerKv],
    profile_slice: &mut [Vec<u32>],
    pool: &CpuThreadPool,
    cpu_scratch: &mut Mm3CpuScratch,
    gidx: usize,
    layer_start: usize,
    layer_end: usize,
    t: usize,
    max_pos_inc: usize,
    feed_dspark: bool,
    stage_split: bool,
    hits_acc: &mut u64,
    misses_acc: &mut u64,
) -> Result<()> {
    let cfg = &weights.cfg;
    let d = cfg.n_embd;
    let kv_fmt = weights.kv_fmt;
    set_device(g.device)?;
    let inv_freq = weights.rope_inv_freq[gidx].ptr as *const f32;
    for il in layer_start..layer_end {
        let lw = &weights.layers[il];
        let local_i = il - layer_start;
        run_attention_block(lw, &kv_slice[local_i], g, cfg, kv_fmt, inv_freq, t, max_pos_inc)?;
        match lw.ffn {
            Mm3FfnWeights::Dense(_) => run_dense_mlp_block(lw, g, cfg, t),
            Mm3FfnWeights::Moe(ref mw) => {
                let es = weights.experts[il]
                    .as_ref()
                    .ok_or_else(|| anyhow!("layer {il} missing expert storage"))?;
                run_prefill_moe_block(
                    il,
                    lw,
                    es,
                    mw,
                    g,
                    pool,
                    cpu_scratch,
                    cfg,
                    t,
                    &mut profile_slice[local_i],
                    hits_acc,
                    misses_acc,
                    stage_split,
                )?;
            }
        }
        if feed_dspark {
            capture_tap_if_needed(dspark, g, il, t, MM3_SUB, d)?;
        }
    }
    Ok(())
}

/// Evaluate an arbitrary prompt slice `[tokens]` starting at `pos0`, supporting multimodal `Mm3ImageSpan`s and optional tracing.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_traced(
    weights: &mut Mm3Weights,
    state: &mut Mm3State,
    tokens: &[u32],
    pos0: usize,
    images: &[Mm3ImageSpan],
    feed_dspark: bool,
    compute_last_logits: bool,
    greedy_last: bool,
    mut trace: Option<&mut Mm3LayerTrace>,
) -> Result<Option<u32>> {
    if tokens.is_empty() {
        return Ok(None);
    }
    if pos0 + tokens.len() > state.max_ctx {
        bail!(
            "MM3 prompt {} + {} exceeds max_ctx {}",
            pos0,
            tokens.len(),
            state.max_ctx
        );
    }

    commit_all_promotions(weights, state)?;

    let cfg = &weights.cfg;
    let d = cfg.n_embd;
    let kv_fmt = weights.kv_fmt;
    let layer_split = weights.layer_split;
    let has_two_gpus = state.gpus.len() >= 2;
    let stage_split = state.stage_split;
    let mut total_hits = 0u64;
    let mut total_misses = 0u64;

    // 2-GPU chunk-pipelined prefill when `tokens.len() > MM3_SUB` and not tracing.
    if has_two_gpus && tokens.len() > MM3_SUB && trace.is_none() {
        let mut chunks: Vec<(usize, usize)> = Vec::with_capacity(tokens.len().div_ceil(MM3_SUB));
        let mut c_cur = 0usize;
        while c_cur < tokens.len() {
            let len = (tokens.len() - c_cur).min(MM3_SUB);
            chunks.push((c_cur, len));
            c_cur += len;
        }

        let mut ds_opt = weights.dspark.take();
        if let Some(ref ds) = ds_opt {
            set_device(ds.device)?;
            ds.stream.sync()?;
        }

        let n_layers = cfg.n_layers;
        let mut run_pipe = || -> Result<Option<u32>> {
            let (left_g, right_g) = state.gpus.split_at_mut(1);
            let g0 = &mut left_g[0];
            let g1 = &mut right_g[0];
            let (kv0, kv1) = state.kv.split_at_mut(layer_split);
            let (pc0, pc1) = state.profile_counts.split_at_mut(layer_split);
            let pool_full = &state.pool;
            let (pool0, pool1) = state
                .pipe_pools
                .as_ref()
                .ok_or_else(|| anyhow!("2-GPU pipe pools missing"))?;
            let cs0 = &mut state.cpu_scratch;
            let cs1 = state
                .cpu_scratch_pipe1
                .as_mut()
                .ok_or_else(|| anyhow!("2-GPU pipe1 scratch missing"))?;
            let emb_pin = &mut state.emb_pin;

            // Chunk 0 GPU0 warm start
            {
                let (off0, len0) = chunks[0];
                let cpos0 = pos0 + off0;
                let max_p0 = cpos0 + len0 - 1;
                let emb_f32 = bytemuck::cast_slice_mut::<u8, f32>(
                    &mut emb_pin.as_mut_slice()[..len0 * d * 4],
                );
                embed_chunk_host(weights, pool_full, &tokens[off0..off0 + len0], cpos0, images, emb_f32)?;
                set_device(g0.device)?;
                let pos_sl =
                    bytemuck::cast_slice_mut::<u8, i32>(&mut g0.h_pos.as_mut_slice()[..len0 * 4]);
                for (i, slot) in pos_sl.iter_mut().enumerate() {
                    *slot = (cpos0 + i) as i32;
                }
                unsafe {
                    copy_h2d_raw_async(g0.pos_dev.ptr, &g0.h_pos.as_slice()[..len0 * 4], &g0.stream)?;
                    copy_h2d_raw_async(g0.h.ptr, &emb_pin.as_slice()[..len0 * d * 4], &g0.stream)?;
                }
                run_prefill_gpu_stage(
                    weights,
                    ds_opt.as_ref(),
                    g0,
                    kv0,
                    pc0,
                    pool_full,
                    cs0,
                    0,
                    0,
                    layer_split,
                    len0,
                    max_p0,
                    feed_dspark,
                    stage_split,
                    &mut total_hits,
                    &mut total_misses,
                )?;
            }

            let mut final_tok = None;
            for c in 0..chunks.len() {
                let (c_off, c_len) = chunks[c];
                let c_pos = pos0 + c_off;
                let c_max_pos = c_pos + c_len - 1;
                let is_last = c + 1 == chunks.len();

                if feed_dspark && c > 0 {
                    if let Some(ref ds) = ds_opt {
                        set_device(ds.device)?;
                        ds.stream.sync()?;
                    }
                }

                handoff_gpu0_to_head_refs(
                    weights.p2p_enabled,
                    d,
                    g0,
                    g1,
                    &mut state.xfer_h_pin,
                    state.taps_xfer_pin.as_mut(),
                    c_len,
                    MM3_SUB,
                    feed_dspark,
                )?;

                if !is_last {
                    let (n_off, n_len) = chunks[c + 1];
                    let n_pos = pos0 + n_off;
                    let n_max_pos = n_pos + n_len - 1;
                    let ds_ro = ds_opt.as_ref();
                    let w_ro = &*weights;

                    let (res0, res1) = std::thread::scope(|scope| {
                        let h0 = scope.spawn(|| -> Result<(u64, u64)> {
                            let mut h0_cnt = 0u64;
                            let mut m0_cnt = 0u64;
                            {
                                let emb_f32 = bytemuck::cast_slice_mut::<u8, f32>(
                                    &mut emb_pin.as_mut_slice()[..n_len * d * 4],
                                );
                                embed_chunk_host(
                                    w_ro,
                                    pool0,
                                    &tokens[n_off..n_off + n_len],
                                    n_pos,
                                    images,
                                    emb_f32,
                                )?;
                            }
                            set_device(g0.device)?;
                            let pos_sl = bytemuck::cast_slice_mut::<u8, i32>(
                                &mut g0.h_pos.as_mut_slice()[..n_len * 4],
                            );
                            for (i, slot) in pos_sl.iter_mut().enumerate() {
                                *slot = (n_pos + i) as i32;
                            }
                            unsafe {
                                copy_h2d_raw_async(
                                    g0.pos_dev.ptr,
                                    &g0.h_pos.as_slice()[..n_len * 4],
                                    &g0.stream,
                                )?;
                                copy_h2d_raw_async(
                                    g0.h.ptr,
                                    &emb_pin.as_slice()[..n_len * d * 4],
                                    &g0.stream,
                                )?;
                            }
                            run_prefill_gpu_stage(
                                w_ro,
                                ds_ro,
                                g0,
                                kv0,
                                pc0,
                                pool0,
                                cs0,
                                0,
                                0,
                                layer_split,
                                n_len,
                                n_max_pos,
                                feed_dspark,
                                stage_split,
                                &mut h0_cnt,
                                &mut m0_cnt,
                            )?;
                            Ok((h0_cnt, m0_cnt))
                        });

                        let r1 = (|| -> Result<(u64, u64)> {
                            let mut h1_cnt = 0u64;
                            let mut m1_cnt = 0u64;
                            set_device(g1.device)?;
                            let pos_sl = bytemuck::cast_slice_mut::<u8, i32>(
                                &mut g1.h_pos.as_mut_slice()[..c_len * 4],
                            );
                            for (i, slot) in pos_sl.iter_mut().enumerate() {
                                *slot = (c_pos + i) as i32;
                            }
                            unsafe {
                                copy_h2d_raw_async(
                                    g1.pos_dev.ptr,
                                    &g1.h_pos.as_slice()[..c_len * 4],
                                    &g1.stream,
                                )?;
                            }
                            run_prefill_gpu_stage(
                                w_ro,
                                ds_ro,
                                g1,
                                kv1,
                                pc1,
                                pool1,
                                cs1,
                                1,
                                layer_split,
                                n_layers,
                                c_len,
                                c_max_pos,
                                feed_dspark,
                                stage_split,
                                &mut h1_cnt,
                                &mut m1_cnt,
                            )?;
                            Ok((h1_cnt, m1_cnt))
                        })();

                        let r0 = h0
                            .join()
                            .unwrap_or_else(|_| Err(anyhow!("MM3 GPU0 pipe thread panicked")));
                        (r0, r1)
                    });

                    let (h0, m0) = res0?;
                    let (h1, m1) = res1?;
                    total_hits += h0 + h1;
                    total_misses += m0 + m1;

                    if feed_dspark {
                        let taps_opt = g1.taps_dev.as_ref().map(|b| b.ptr as *const f32);
                        if let (Some(ref mut ds), Some(taps_ptr)) = (&mut ds_opt, taps_opt) {
                            set_device(g1.device)?;
                            g1.taps_ready_ev.record(&g1.stream)?;
                            set_device(ds.device)?;
                            ds.stream.wait_event(&g1.taps_ready_ev)?;
                            ds.context_from_taps(taps_ptr, c_len, c_pos, MM3_SUB)?;
                        }
                    }
                } else {
                    set_device(g1.device)?;
                    let pos_sl = bytemuck::cast_slice_mut::<u8, i32>(
                        &mut g1.h_pos.as_mut_slice()[..c_len * 4],
                    );
                    for (i, slot) in pos_sl.iter_mut().enumerate() {
                        *slot = (c_pos + i) as i32;
                    }
                    unsafe {
                        copy_h2d_raw_async(
                            g1.pos_dev.ptr,
                            &g1.h_pos.as_slice()[..c_len * 4],
                            &g1.stream,
                        )?;
                    }
                    run_prefill_gpu_stage(
                        weights,
                        ds_opt.as_ref(),
                        g1,
                        kv1,
                        pc1,
                        pool_full,
                        cs0,
                        1,
                        layer_split,
                        n_layers,
                        c_len,
                        c_max_pos,
                        feed_dspark,
                        stage_split,
                        &mut total_hits,
                        &mut total_misses,
                    )?;

                    if feed_dspark && ds_opt.is_some() {
                        set_device(g1.device)?;
                        g1.taps_ready_ev.record(&g1.stream)?;
                        let taps_ptr = g1
                            .taps_dev
                            .as_ref()
                            .ok_or_else(|| anyhow!("head GPU missing taps_dev"))?
                            .ptr as *const f32;
                        if let Some(ref mut ds) = ds_opt {
                            set_device(ds.device)?;
                            ds.stream.wait_event(&g1.taps_ready_ev)?;
                            ds.context_from_taps(taps_ptr, c_len, c_pos, MM3_SUB)?;
                        }
                    }

                    if compute_last_logits {
                        set_device(g1.device)?;
                        let vocab = cfg.vocab_size;
                        let eps = cfg.rms_norm_eps;
                        let logits_buf = g1
                            .logits
                            .as_ref()
                            .ok_or_else(|| anyhow!("head GPU missing logits buffer"))?;
                        unsafe {
                            let last_h = (g1.h.ptr as *const f32).add((c_len - 1) * d);
                            mm3::rms_norm_f32(
                                last_h,
                                weights.output_norm.ptr as *const f32,
                                1,
                                d as i32,
                                eps,
                                false,
                                g1.xn.ptr as *mut f32,
                                g1.x_q8_1.ptr,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                                &g1.stream,
                            );
                            linear_mm3(
                                weights.lm_head.as_spec(),
                                g1.x_q8_1.ptr,
                                g1.xn.ptr as *const f32,
                                logits_buf.ptr as *mut f32,
                                1,
                                &g1.stream,
                            );
                        }
                        if greedy_last {
                            let vals_p = g1.argmax_vals.as_ref().unwrap().ptr as *mut f32;
                            let ids_p = g1.argmax_ids.as_ref().unwrap().ptr as *mut i32;
                            let out_p = g1.argmax_out.as_ref().unwrap().ptr as *mut i32;
                            let h_arg = g1.h_argmax.as_mut().unwrap();
                            unsafe {
                                glm5::argmax(
                                    logits_buf.ptr as *const f32,
                                    vocab as i32,
                                    vocab as i32,
                                    std::ptr::null(),
                                    vals_p,
                                    ids_p,
                                    out_p,
                                    &g1.stream,
                                );
                                copy_d2h_raw_async(
                                    &mut h_arg.as_mut_slice()[..4],
                                    out_p as *mut c_void,
                                    &g1.stream,
                                )?;
                            }
                            g1.stream.sync()?;
                            let tid = bytemuck::cast_slice::<u8, i32>(&h_arg.as_slice()[..4])[0];
                            final_tok = Some(tid.max(0) as u32);
                        } else {
                            unsafe {
                                copy_d2h_raw_async(
                                    bytemuck::cast_slice_mut(&mut state.logits_host[..vocab]),
                                    logits_buf.ptr,
                                    &g1.stream,
                                )?;
                            }
                            g1.stream.sync()?;
                        }
                    }
                    if let Some(ref ds) = ds_opt {
                        set_device(ds.device)?;
                        ds.stream.sync()?;
                    }
                }
            }
            Ok(final_tok)
        };

        let pipe_res = run_pipe();
        weights.dspark = ds_opt;
        let final_tok = pipe_res?;
        state.cache_stats.hits += total_hits;
        state.cache_stats.misses += total_misses;
        state.pos = pos0 + tokens.len();
        return Ok(final_tok);
    }

    let mut c0 = 0usize;
    while c0 < tokens.len() {
        let t = (tokens.len() - c0).min(MM3_SUB);
        let chunk_pos = pos0 + c0;
        let max_pos_inc = chunk_pos + t - 1;
        let is_last_chunk = c0 + t == tokens.len();

        if let Some(ref ds) = weights.dspark {
            set_device(ds.device)?;
            ds.stream.sync()?;
        }

        {
            let emb_f32 =
                bytemuck::cast_slice_mut::<u8, f32>(&mut state.emb_pin.as_mut_slice()[..t * d * 4]);
            embed_chunk_host(
                weights,
                &state.pool,
                &tokens[c0..c0 + t],
                chunk_pos,
                images,
                emb_f32,
            )?;
        }

        for g in &mut state.gpus {
            set_device(g.device)?;
            let pos_sl = bytemuck::cast_slice_mut::<u8, i32>(&mut g.h_pos.as_mut_slice()[..t * 4]);
            for (i, slot) in pos_sl.iter_mut().enumerate() {
                *slot = (chunk_pos + i) as i32;
            }
            unsafe {
                copy_h2d_raw_async(g.pos_dev.ptr, &g.h_pos.as_slice()[..t * 4], &g.stream)?;
            }
        }

        {
            let g0 = &state.gpus[0];
            set_device(g0.device)?;
            unsafe {
                copy_h2d_raw_async(g0.h.ptr, &state.emb_pin.as_slice()[..t * d * 4], &g0.stream)?;
            }
        }

        for il in 0..cfg.n_layers {
            let lw = &weights.layers[il];
            let gidx = lw.gpu_idx;
            let g = &mut state.gpus[gidx];
            set_device(g.device)?;
            let inv_freq = weights.rope_inv_freq[gidx].ptr as *const f32;
            run_attention_block(lw, &state.kv[il], g, cfg, kv_fmt, inv_freq, t, max_pos_inc)?;
            if is_last_chunk {
                if let Some(tr) = trace.as_deref_mut() {
                    tr.after_attn[il] = copy_last_row_h(g, t, d)?;
                }
            }

            match lw.ffn {
                Mm3FfnWeights::Dense(_) => run_dense_mlp_block(lw, g, cfg, t),
                Mm3FfnWeights::Moe(ref mw) => {
                    let es = weights.experts[il]
                        .as_ref()
                        .ok_or_else(|| anyhow!("layer {il} missing expert storage"))?;
                    run_prefill_moe_block(
                        il,
                        lw,
                        es,
                        mw,
                        g,
                        &state.pool,
                        &mut state.cpu_scratch,
                        cfg,
                        t,
                        &mut state.profile_counts[il],
                        &mut total_hits,
                        &mut total_misses,
                        stage_split,
                    )?;
                }
            }
            if is_last_chunk {
                if let Some(tr) = trace.as_deref_mut() {
                    tr.after_ffn[il] = copy_last_row_h(g, t, d)?;
                    if il == 3 {
                        tr.l03_all_ffn = copy_all_rows_h(g, t, d)?;
                    }
                    record_router_last(weights, g, il, t, tr);
                }
            }
            if feed_dspark {
                capture_tap_if_needed(weights.dspark.as_ref(), g, il, t, MM3_SUB, d)?;
            }

            if has_two_gpus && il + 1 == layer_split {
                handoff_gpu0_to_head(weights, state, t, MM3_SUB, feed_dspark)?;
            }
        }

        let head_idx = state.gpus.len() - 1;
        let mut last_tok = None;

        if feed_dspark && weights.dspark.is_some() {
            let hg = &state.gpus[head_idx];
            set_device(hg.device)?;
            hg.taps_ready_ev.record(&hg.stream)?;
        }

        if is_last_chunk && compute_last_logits {
            // Compute output norm and lm_head only on the final token row `t - 1`.
            let hg = &mut state.gpus[head_idx];
            set_device(hg.device)?;
            let vocab = cfg.vocab_size;
            let eps = cfg.rms_norm_eps;
            let logits_buf = hg
                .logits
                .as_ref()
                .ok_or_else(|| anyhow!("head GPU missing logits buffer"))?;
            unsafe {
                let last_h = (hg.h.ptr as *const f32).add((t - 1) * d);
                mm3::rms_norm_f32(
                    last_h,
                    weights.output_norm.ptr as *const f32,
                    1,
                    d as i32,
                    eps,
                    false,
                    hg.xn.ptr as *mut f32,
                    hg.x_q8_1.ptr,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &hg.stream,
                );
                linear_mm3(
                    weights.lm_head.as_spec(),
                    hg.x_q8_1.ptr,
                    hg.xn.ptr as *const f32,
                    logits_buf.ptr as *mut f32,
                    1,
                    &hg.stream,
                );
            }
            if greedy_last {
                let vals_p = hg.argmax_vals.as_ref().unwrap().ptr as *mut f32;
                let ids_p = hg.argmax_ids.as_ref().unwrap().ptr as *mut i32;
                let out_p = hg.argmax_out.as_ref().unwrap().ptr as *mut i32;
                let h_arg = hg.h_argmax.as_mut().unwrap();
                unsafe {
                    glm5::argmax(
                        logits_buf.ptr as *const f32,
                        vocab as i32,
                        vocab as i32,
                        std::ptr::null(),
                        vals_p,
                        ids_p,
                        out_p,
                        &hg.stream,
                    );
                    copy_d2h_raw_async(&mut h_arg.as_mut_slice()[..4], out_p as *mut c_void, &hg.stream)?;
                }
                hg.stream.sync()?;
                let tid = bytemuck::cast_slice::<u8, i32>(&h_arg.as_slice()[..4])[0];
                last_tok = Some(tid.max(0) as u32);
            } else {
                unsafe {
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut state.logits_host[..vocab]),
                        logits_buf.ptr,
                        &hg.stream,
                    )?;
                }
                hg.stream.sync()?;
            }
        }

        if feed_dspark {
            let taps_opt = state.gpus[head_idx].taps_dev.as_ref().map(|b| b.ptr as *const f32);
            if let (Some(ref mut ds), Some(taps_ptr)) = (&mut weights.dspark, taps_opt) {
                set_device(ds.device)?;
                ds.stream.wait_event(&state.gpus[head_idx].taps_ready_ev)?;
                ds.context_from_taps(taps_ptr, t, chunk_pos, MM3_SUB)?;
                if !is_last_chunk {
                    ds.stream.sync()?;
                }
            }
        }

        if is_last_chunk && compute_last_logits {
            if let Some(ref ds) = weights.dspark {
                set_device(ds.device)?;
                ds.stream.sync()?;
            }
            state.cache_stats.hits += total_hits;
            state.cache_stats.misses += total_misses;
            state.pos = pos0 + tokens.len();
            return Ok(last_tok);
        }

        c0 += t;
    }

    if feed_dspark {
        if let Some(ref ds) = weights.dspark {
            set_device(ds.device)?;
            ds.stream.sync()?;
        }
    }
    state.cache_stats.hits += total_hits;
    state.cache_stats.misses += total_misses;
    state.pos = pos0 + tokens.len();
    Ok(None)
}
