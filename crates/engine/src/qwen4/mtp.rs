use crate::qwen4::forward::{embed_tokens_raw, hc_read_gpu, linear_q8};
use crate::qwen4::state::Qwen4State;
use crate::qwen4::weights::{MtpWeights, Qwen4Weights};
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::ffi::launch_quantize_q8_1;
use ampllm_cuda_kernels::{
    copy_d2d_raw_async, copy_h2d_raw_async, glm5, hy4, qwen4, set_device, CudaEvent, CudaGraphExec,
    CudaStream,
};
use ampllm_gguf::{GgmlType, Qwen4Config};
use anyhow::Result;

/// Pre-populate MTP drafter Q8 KV cache on GPU 1 across prefill cells `cell0 .. cell0 + next_tokens.len()`
/// without borrowing the full `Qwen4Weights` or `Qwen4State` (enables concurrent dual-GPU prefill pipelining).
#[allow(clippy::too_many_arguments)]
pub fn mtp_prefill_kv_parts(
    cfg: &Qwen4Config,
    mw: &MtpWeights,
    tok_embd: (&[u8], usize, GgmlType),
    ms: &mut crate::qwen4::state::MtpState,
    stream: &CudaStream,
    side: Option<(&CudaStream, &CudaEvent, &CudaEvent)>,
    mrope_opt: *const i32,
    pool: &CpuThreadPool,
    next_tokens: &[u32],
    r_rows_ptr: *const f32,
    cell0: usize,
    can_graph: bool,
    shift_r_last: bool,
) -> Result<()> {
    if next_tokens.is_empty() {
        return Ok(());
    }

    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let hc_dim = hc * n_embd;
    let hc_lr = cfg.hc_low_rank;
    let eps = cfg.rms_norm_eps;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.head_dim;

    set_device(mw.device)?;
    let batch_cap = ms.batch_cap.max(1);

    let emit_kv_batch = |ms: &crate::qwen4::state::MtpState, off: usize, nb: usize| -> Result<()> {
        let r_sub = unsafe { r_rows_ptr.add(off * hc_dim) };
        if let Some((side_s, fork_ev, join_ev)) = side {
            fork_ev.record(stream)?;
            side_s.wait_event(fork_ev)?;
            unsafe {
                qwen4::rmsnorm_f32(
                    ms.emb.ptr.cast::<f32>(),
                    mw.pre_fc_norm_embedding.ptr.cast::<f32>(),
                    nb as i32,
                    n_embd as i32,
                    eps,
                    ms.emb_norm.ptr.cast::<f32>(),
                    ms.emb_norm_q8.ptr,
                    side_s,
                );
                linear_q8(
                    mw.fc_embedding.as_spec(),
                    ms.emb_norm_q8.ptr,
                    ms.emb_norm.ptr.cast::<f32>(),
                    ms.emb_fc.ptr.cast::<f32>(),
                    nb,
                    side_s,
                );
                qwen4::rmsnorm_f32(
                    r_sub,
                    mw.pre_fc_norm_hidden.ptr.cast::<f32>(),
                    nb as i32,
                    hc_dim as i32,
                    eps,
                    ms.hid_norm.ptr.cast::<f32>(),
                    std::ptr::null_mut(),
                    stream,
                );
                launch_quantize_q8_1(
                    ms.hid_norm.ptr.cast::<f32>(),
                    ms.hid_norm_q8.ptr,
                    (nb * hc) as i32,
                    n_embd as i32,
                    stream.raw,
                );
                linear_q8(
                    mw.fc_hidden.as_spec(),
                    ms.hid_norm_q8.ptr,
                    ms.hid_norm.ptr.cast::<f32>(),
                    ms.hid_fc.ptr.cast::<f32>(),
                    nb * hc,
                    stream,
                );
            }
            join_ev.record(side_s)?;
            stream.wait_event(join_ev)?;
        } else {
            unsafe {
                qwen4::rmsnorm_f32(
                    ms.emb.ptr.cast::<f32>(),
                    mw.pre_fc_norm_embedding.ptr.cast::<f32>(),
                    nb as i32,
                    n_embd as i32,
                    eps,
                    ms.emb_norm.ptr.cast::<f32>(),
                    ms.emb_norm_q8.ptr,
                    stream,
                );
                linear_q8(
                    mw.fc_embedding.as_spec(),
                    ms.emb_norm_q8.ptr,
                    ms.emb_norm.ptr.cast::<f32>(),
                    ms.emb_fc.ptr.cast::<f32>(),
                    nb,
                    stream,
                );
                qwen4::rmsnorm_f32(
                    r_sub,
                    mw.pre_fc_norm_hidden.ptr.cast::<f32>(),
                    nb as i32,
                    hc_dim as i32,
                    eps,
                    ms.hid_norm.ptr.cast::<f32>(),
                    std::ptr::null_mut(),
                    stream,
                );
                launch_quantize_q8_1(
                    ms.hid_norm.ptr.cast::<f32>(),
                    ms.hid_norm_q8.ptr,
                    (nb * hc) as i32,
                    n_embd as i32,
                    stream.raw,
                );
                linear_q8(
                    mw.fc_hidden.as_spec(),
                    ms.hid_norm_q8.ptr,
                    ms.hid_norm.ptr.cast::<f32>(),
                    ms.hid_fc.ptr.cast::<f32>(),
                    nb * hc,
                    stream,
                );
            }
        }
        unsafe {
            qwen4::add_streams_broadcast(
                ms.hid_fc.ptr.cast::<f32>(),
                ms.emb_fc.ptr.cast::<f32>(),
                nb as i32,
                hc as i32,
                n_embd as i32,
                ms.r_mtp.ptr.cast::<f32>(),
                stream,
            );

            hc_read_gpu(
                &mw.hc_attn,
                ms.r_mtp.ptr.cast::<f32>(),
                nb,
                hc,
                n_embd,
                hc_lr,
                eps,
                ms.xn.ptr.cast::<f32>(),
                ms.xn_q8.ptr,
                ms.lo.ptr.cast::<f32>(),
                ms.lo_q8.ptr,
                ms.hc_up.ptr.cast::<f32>(),
                ms.mixed.ptr.cast::<f32>(),
                ms.mixed_q8.ptr,
                std::ptr::null_mut(),
                false,
                stream,
            );
            if nb <= 8
                && mw.k_proj.ggml_type == GgmlType::Q8_0
                && mw.v_proj.ggml_type == GgmlType::Q8_0
            {
                qwen4::mmvq_q8_0_multi2(
                    mw.k_proj.buf.ptr,
                    ms.k.ptr.cast::<f32>(),
                    mw.k_proj.n_out as i32,
                    mw.v_proj.buf.ptr,
                    ms.v.ptr.cast::<f32>(),
                    mw.v_proj.n_out as i32,
                    ms.mixed_q8.ptr,
                    n_embd as i32,
                    nb as i32,
                    stream,
                );
            } else {
                linear_q8(
                    mw.k_proj.as_spec(),
                    ms.mixed_q8.ptr,
                    ms.mixed.ptr.cast::<f32>(),
                    ms.k.ptr.cast::<f32>(),
                    nb,
                    stream,
                );
                linear_q8(
                    mw.v_proj.as_spec(),
                    ms.mixed_q8.ptr,
                    ms.mixed.ptr.cast::<f32>(),
                    ms.v.ptr.cast::<f32>(),
                    nb,
                    stream,
                );
            }
            qwen4::qsa_norm_imrope(
                ms.k.ptr.cast::<f32>(),
                hd as i32,
                mw.k_norm.ptr.cast::<f32>(),
                ms.k.ptr.cast::<f32>(),
                nb as i32,
                n_kv as i32,
                hd as i32,
                cfg.rope_dim as i32,
                eps,
                cfg.rope_base,
                ms.pos_dev.ptr.cast::<i32>(),
                mrope_opt,
                stream,
            );
            qwen4::kv_append_q8(
                ms.k.ptr.cast::<f32>(),
                ms.v.ptr.cast::<f32>(),
                ms.pos_dev.ptr.cast::<i32>(),
                nb as i32,
                n_kv as i32,
                hd as i32,
                ms.k_q.ptr.cast::<i8>(),
                ms.v_q.ptr.cast::<i8>(),
                ms.k_sc.ptr.cast::<u16>(),
                ms.v_sc.ptr.cast::<u16>(),
                stream,
            );
        }
        Ok(())
    };

    if can_graph && shift_r_last && next_tokens.len() <= 7 {
        let nb = next_tokens.len();
        {
            let h_pos = &mut ms.decode_pos_pin.as_mut_slice()[1..1 + nb];
            for i in 0..nb {
                h_pos[i] = (cell0 + i) as i32;
            }
            let h_emb = &mut ms.decode_emb_pin.as_mut_slice()[n_embd..(1 + nb) * n_embd];
            embed_tokens_raw(cfg, tok_embd.0, tok_embd.1, tok_embd.2, pool, next_tokens, 0, &[], h_emb);
        }
        if !ms.prefill_graphs.contains_key(&nb) {
            let pos_src = (ms.decode_pos_dev + 4) as *const std::ffi::c_void;
            let emb_src = (ms.decode_emb_dev + (n_embd * 4) as u64) as *const std::ffi::c_void;
            let graph = {
                let ms_ref: &crate::qwen4::state::MtpState = &*ms;
                CudaGraphExec::capture(stream, || {
                    unsafe {
                        qwen4::copy_u32_sm(ms_ref.pos_dev.ptr, pos_src, nb as i32, false, stream);
                        qwen4::copy_u32_sm(
                            ms_ref.emb.ptr,
                            emb_src,
                            (nb * n_embd) as i32,
                            false,
                            stream,
                        );
                    }
                    emit_kv_batch(ms_ref, 0, nb)?;
                    unsafe {
                        copy_d2d_raw_async(
                            r_rows_ptr as *mut std::ffi::c_void,
                            r_rows_ptr.add(nb * hc_dim) as *const std::ffi::c_void,
                            hc_dim * 4,
                            stream,
                        )?;
                    }
                    Ok(())
                })?
            };
            ms.prefill_graphs.insert(nb, graph);
        }
        ms.prefill_graphs[&nb].launch(stream)?;
        ms.kv_pos = cell0 + nb;
        return Ok(());
    }

    if next_tokens.len() <= 8 {
        let nb = next_tokens.len();
        {
            let h_pos = &mut ms.decode_pos_pin.as_mut_slice()[..nb];
            for i in 0..nb {
                h_pos[i] = (cell0 + i) as i32;
            }
            let h_emb = &mut ms.decode_emb_pin.as_mut_slice()[..nb * n_embd];
            embed_tokens_raw(cfg, tok_embd.0, tok_embd.1, tok_embd.2, pool, next_tokens, 0, &[], h_emb);
        }
        unsafe {
            qwen4::copy_u32_sm(
                ms.pos_dev.ptr,
                ms.decode_pos_dev as *const std::ffi::c_void,
                nb as i32,
                false,
                stream,
            );
            qwen4::copy_u32_sm(
                ms.emb.ptr,
                ms.decode_emb_dev as *const std::ffi::c_void,
                (nb * n_embd) as i32,
                false,
                stream,
            );
            ms.h2d_done.record(stream)?;
        }
        emit_kv_batch(ms, 0, nb)?;
        if shift_r_last {
            unsafe {
                copy_d2d_raw_async(
                    r_rows_ptr as *mut std::ffi::c_void,
                    r_rows_ptr.add(nb * hc_dim) as *const std::ffi::c_void,
                    hc_dim * 4,
                    stream,
                )?;
            }
        }
        ms.h2d_done.sync()?;
        ms.kv_pos = cell0 + nb;
        return Ok(());
    }

    let mut off = 0usize;
    while off < next_tokens.len() {
        if off > 0 {
            ms.h2d_done.sync()?;
        }
        let nb = (next_tokens.len() - off).min(batch_cap);
        let sub_toks = &next_tokens[off..off + nb];

        let h_pos_slice = unsafe {
            std::slice::from_raw_parts_mut(ms.h_pos.as_mut_ptr().cast::<i32>(), nb)
        };
        for i in 0..nb {
            h_pos_slice[i] = (cell0 + off + i) as i32;
        }
        let h_emb_slice = unsafe {
            std::slice::from_raw_parts_mut(ms.h_emb.as_mut_ptr().cast::<f32>(), nb * n_embd)
        };
        embed_tokens_raw(cfg, tok_embd.0, tok_embd.1, tok_embd.2, pool, sub_toks, 0, &[], h_emb_slice);

        unsafe {
            copy_h2d_raw_async(
                ms.pos_dev.ptr,
                std::slice::from_raw_parts(ms.h_pos.as_ptr(), nb * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                ms.emb.ptr,
                std::slice::from_raw_parts(ms.h_emb.as_ptr(), nb * n_embd * 4),
                stream,
            )?;
            ms.h2d_done.record(stream)?;
        }
        emit_kv_batch(ms, off, nb)?;
        off += nb;
    }

    if shift_r_last {
        unsafe {
            copy_d2d_raw_async(
                r_rows_ptr as *mut std::ffi::c_void,
                r_rows_ptr.add(next_tokens.len() * hc_dim) as *const std::ffi::c_void,
                hc_dim * 4,
                stream,
            )?;
        }
    }
    ms.h2d_done.sync()?;
    ms.kv_pos = cell0 + next_tokens.len();
    Ok(())
}

/// Pre-populate MTP drafter Q8 KV cache on GPU 1 across prefill cells `cell0 .. cell0 + next_tokens.len()`.
pub fn mtp_prefill_kv(
    weights: &Qwen4Weights,
    state: &mut Qwen4State,
    pool: &CpuThreadPool,
    next_tokens: &[u32],
    r_rows_ptr: *const f32,
    cell0: usize,
) -> Result<()> {
    let Some(ref mw) = weights.mtp else {
        return Ok(());
    };
    let can_graph = !state.mrope_active
        && state.use_cuda_graph
        && std::ptr::eq(r_rows_ptr, state.r_last.ptr.cast::<f32>());
    let Some(ref mut ms) = state.mtp else {
        return Ok(());
    };
    let head_gidx = state.gpus.len() - 1;
    let hg = &state.gpus[head_gidx];
    let mrope_opt = if state.mrope_active {
        hg.ptr::<i32>(hg.off_mrope_pos3_dev).cast_const()
    } else {
        std::ptr::null()
    };
    let tbl = weights.token_embd_bytes()?;
    mtp_prefill_kv_parts(
        &weights.cfg,
        mw,
        (tbl, weights.token_embd_row_bytes, weights.token_embd_type),
        ms,
        &hg.stream,
        Some((&hg.side_stream, &hg.side_fork_ev, &hg.side_join_ev)),
        mrope_opt,
        pool,
        next_tokens,
        r_rows_ptr,
        cell0,
        can_graph,
        true,
    )
}

/// Run one single-token MTP draft step from `(tok, r_in_ptr)` at `cell_pos`.
/// Writes updated hidden state into `ms.r_mtp` (usable as `r_in_ptr` for chaining the next draft)
/// and either returns argmax token directly (`compute_argmax = true`) or leaves `ms.logits` ready on GPU.
pub fn mtp_draft_step(
    weights: &Qwen4Weights,
    state: &mut Qwen4State,
    pool: &CpuThreadPool,
    tok: u32,
    r_in_ptr: *const f32,
    cell_pos: usize,
    compute_argmax: bool,
) -> Result<(u32, f32)> {
    let mw = weights
        .mtp
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("MTP weights not loaded"))?;
    let ms = state
        .mtp
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("MTP state not allocated"))?;

    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let hc_dim = hc * n_embd;
    let hc_lr = cfg.hc_low_rank;
    let eps = cfg.rms_norm_eps;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let head_gidx = state.gpus.len() - 1;
    let hg = &state.gpus[head_gidx];
    let stream = &hg.stream;
    let side_s = &hg.side_stream;
    let fork_ev = &hg.side_fork_ev;
    let join_ev = &hg.side_join_ev;
    set_device(mw.device)?;

    let decode_pos = ms.decode_pos_pin.as_mut_slice();
    decode_pos[0] = cell_pos as i32;
    let h_emb_slice = ms.decode_emb_pin.as_mut_slice();
    let tbl = weights.token_embd_bytes()?;
    embed_tokens_raw(
        cfg,
        tbl,
        weights.token_embd_row_bytes,
        weights.token_embd_type,
        pool,
        &[tok],
        0,
        &[],
        &mut h_emb_slice[..n_embd],
    );

    let mrope_opt = if state.mrope_active {
        hg.ptr::<i32>(hg.off_mrope_pos3_dev).cast_const()
    } else {
        std::ptr::null()
    };

    let is_first = std::ptr::eq(r_in_ptr, state.r_last.ptr.cast::<f32>());
    let is_chained = std::ptr::eq(r_in_ptr, ms.r_mtp.ptr.cast::<f32>());
    let cap_bucket = if cell_pos + 1 <= 2048 {
        2048usize.min(state.max_ctx)
    } else {
        state.max_ctx
    };
    let can_graph = compute_argmax && !state.mrope_active && state.use_cuda_graph && (is_first || is_chained);

    let emit_body = |ms_ref: &crate::qwen4::state::MtpState, max_n_ids: usize| -> Result<()> {
        unsafe {
            qwen4::copy_u32_sm(
                ms_ref.pos_dev.ptr,
                ms_ref.decode_pos_dev as *const std::ffi::c_void,
                1,
                false,
                stream,
            );
            qwen4::copy_u32_sm(
                ms_ref.emb.ptr,
                ms_ref.decode_emb_dev as *const std::ffi::c_void,
                n_embd as i32,
                false,
                stream,
            );
            qwen4::fill_dense_widths(
                ms_ref.pos_dev.ptr.cast::<i32>(),
                ms_ref.width_dev.ptr.cast::<i32>(),
                1,
                stream,
            );
        }

        fork_ev.record(stream)?;
        side_s.wait_event(fork_ev)?;
        unsafe {
            qwen4::rmsnorm_f32(
                ms_ref.emb.ptr.cast::<f32>(),
                mw.pre_fc_norm_embedding.ptr.cast::<f32>(),
                1,
                n_embd as i32,
                eps,
                ms_ref.emb_norm.ptr.cast::<f32>(),
                ms_ref.emb_norm_q8.ptr,
                side_s,
            );
            linear_q8(
                mw.fc_embedding.as_spec(),
                ms_ref.emb_norm_q8.ptr,
                ms_ref.emb_norm.ptr.cast::<f32>(),
                ms_ref.emb_fc.ptr.cast::<f32>(),
                1,
                side_s,
            );
            qwen4::rmsnorm_f32(
                r_in_ptr,
                mw.pre_fc_norm_hidden.ptr.cast::<f32>(),
                1,
                hc_dim as i32,
                eps,
                ms_ref.hid_norm.ptr.cast::<f32>(),
                std::ptr::null_mut(),
                stream,
            );
            launch_quantize_q8_1(
                ms_ref.hid_norm.ptr.cast::<f32>(),
                ms_ref.hid_norm_q8.ptr,
                hc as i32,
                n_embd as i32,
                stream.raw,
            );
            linear_q8(
                mw.fc_hidden.as_spec(),
                ms_ref.hid_norm_q8.ptr,
                ms_ref.hid_norm.ptr.cast::<f32>(),
                ms_ref.hid_fc.ptr.cast::<f32>(),
                hc,
                stream,
            );
        }
        join_ev.record(side_s)?;
        stream.wait_event(join_ev)?;

        unsafe {
            qwen4::add_streams_broadcast(
                ms_ref.hid_fc.ptr.cast::<f32>(),
                ms_ref.emb_fc.ptr.cast::<f32>(),
                1,
                hc as i32,
                n_embd as i32,
                ms_ref.r_mtp.ptr.cast::<f32>(),
                stream,
            );

            // Attention HC read & dense QSA over [0 ..= cell_pos]
            hc_read_gpu(
                &mw.hc_attn,
                ms_ref.r_mtp.ptr.cast::<f32>(),
                1,
                hc,
                n_embd,
                hc_lr,
                eps,
                ms_ref.xn.ptr.cast::<f32>(),
                ms_ref.xn_q8.ptr,
                ms_ref.lo.ptr.cast::<f32>(),
                ms_ref.lo_q8.ptr,
                ms_ref.hc_up.ptr.cast::<f32>(),
                ms_ref.mixed.ptr.cast::<f32>(),
                ms_ref.mixed_q8.ptr,
                ms_ref.inject.ptr.cast::<f32>(),
                true,
                stream,
            );
            if mw.q_proj.ggml_type == GgmlType::Q8_0
                && mw.k_proj.ggml_type == GgmlType::Q8_0
                && mw.v_proj.ggml_type == GgmlType::Q8_0
            {
                qwen4::mmvq_q8_0_multi3(
                    mw.q_proj.buf.ptr,
                    ms_ref.qfull.ptr.cast::<f32>(),
                    mw.q_proj.n_out as i32,
                    mw.k_proj.buf.ptr,
                    ms_ref.k.ptr.cast::<f32>(),
                    mw.k_proj.n_out as i32,
                    mw.v_proj.buf.ptr,
                    ms_ref.v.ptr.cast::<f32>(),
                    mw.v_proj.n_out as i32,
                    ms_ref.mixed_q8.ptr,
                    n_embd as i32,
                    1,
                    stream,
                );
            } else {
                linear_q8(
                    mw.q_proj.as_spec(),
                    ms_ref.mixed_q8.ptr,
                    ms_ref.mixed.ptr.cast::<f32>(),
                    ms_ref.qfull.ptr.cast::<f32>(),
                    1,
                    stream,
                );
                linear_q8(
                    mw.k_proj.as_spec(),
                    ms_ref.mixed_q8.ptr,
                    ms_ref.mixed.ptr.cast::<f32>(),
                    ms_ref.k.ptr.cast::<f32>(),
                    1,
                    stream,
                );
                linear_q8(
                    mw.v_proj.as_spec(),
                    ms_ref.mixed_q8.ptr,
                    ms_ref.mixed.ptr.cast::<f32>(),
                    ms_ref.v.ptr.cast::<f32>(),
                    1,
                    stream,
                );
            }
            qwen4::qsa_norm_imrope(
                ms_ref.qfull.ptr.cast::<f32>(),
                (hd * 2) as i32,
                mw.q_norm.ptr.cast::<f32>(),
                ms_ref.q.ptr.cast::<f32>(),
                1,
                n_head as i32,
                hd as i32,
                cfg.rope_dim as i32,
                eps,
                cfg.rope_base,
                ms_ref.pos_dev.ptr.cast::<i32>(),
                mrope_opt,
                stream,
            );
            qwen4::qsa_norm_imrope(
                ms_ref.k.ptr.cast::<f32>(),
                hd as i32,
                mw.k_norm.ptr.cast::<f32>(),
                ms_ref.k.ptr.cast::<f32>(),
                1,
                n_kv as i32,
                hd as i32,
                cfg.rope_dim as i32,
                eps,
                cfg.rope_base,
                ms_ref.pos_dev.ptr.cast::<i32>(),
                mrope_opt,
                stream,
            );
            qwen4::kv_append_q8(
                ms_ref.k.ptr.cast::<f32>(),
                ms_ref.v.ptr.cast::<f32>(),
                ms_ref.pos_dev.ptr.cast::<i32>(),
                1,
                n_kv as i32,
                hd as i32,
                ms_ref.k_q.ptr.cast::<i8>(),
                ms_ref.v_q.ptr.cast::<i8>(),
                ms_ref.k_sc.ptr.cast::<u16>(),
                ms_ref.v_sc.ptr.cast::<u16>(),
                stream,
            );
            qwen4::qsa_decode_attn_q8(
                ms_ref.q.ptr.cast::<f32>(),
                ms_ref.qfull.ptr.cast::<f32>(),
                ms_ref.k_q.ptr.cast::<i8>(),
                ms_ref.v_q.ptr.cast::<i8>(),
                ms_ref.k_sc.ptr.cast::<u16>(),
                ms_ref.v_sc.ptr.cast::<u16>(),
                std::ptr::null(),
                ms_ref.width_dev.ptr.cast::<i32>(),
                max_n_ids as i32,
                max_n_ids as i32,
                1,
                n_kv as i32,
                scale,
                ms_ref.qsa_ws.ptr.cast::<f32>(),
                std::ptr::null_mut(),
                ms_ref.attn_q8.ptr,
                stream,
            );
            linear_q8(
                mw.o_proj.as_spec(),
                ms_ref.attn_q8.ptr,
                std::ptr::null(),
                ms_ref.block_out.ptr.cast::<f32>(),
                1,
                stream,
            );
            qwen4::hc_write_strided(
                ms_ref.r_mtp.ptr.cast::<f32>(),
                ms_ref.block_out.ptr.cast::<f32>(),
                std::ptr::null(),
                std::ptr::null(),
                ms_ref.inject.ptr.cast::<f32>(),
                hc as i32,
                1,
                hc as i32,
                n_embd as i32,
                stream,
            );

            // MLP HC + MoE (512 resident Q2_0 experts + shared expert)
            hc_read_gpu(
                &mw.hc_mlp,
                ms_ref.r_mtp.ptr.cast::<f32>(),
                1,
                hc,
                n_embd,
                hc_lr,
                eps,
                ms_ref.xn.ptr.cast::<f32>(),
                ms_ref.xn_q8.ptr,
                ms_ref.lo.ptr.cast::<f32>(),
                ms_ref.lo_q8.ptr,
                ms_ref.hc_up.ptr.cast::<f32>(),
                ms_ref.mixed.ptr.cast::<f32>(),
                ms_ref.mixed_q8.ptr,
                ms_ref.inject.ptr.cast::<f32>(),
                true,
                stream,
            );
        }

        fork_ev.record(stream)?;
        side_s.wait_event(fork_ev)?;
        unsafe {
            hy4::gemv_f32(
                mw.router_gate_f32.ptr.cast::<f32>(),
                cfg.n_expert as i32,
                n_embd as i32,
                ms_ref.mixed.ptr.cast::<f32>(),
                1,
                ms_ref.router_logits.ptr.cast::<f32>(),
                1.0,
                stream,
            );
            qwen4::router_top10(
                ms_ref.router_logits.ptr.cast::<f32>(),
                1,
                cfg.n_expert as i32,
                cfg.n_expert_used as i32,
                ms_ref.topk_ids.ptr.cast::<i32>(),
                ms_ref.topk_w.ptr.cast::<f32>(),
                stream,
            );
            glm5::moe_resident(
                mw.gu_type as i32,
                mw.d_type as i32,
                mw.dev_gate_ptrs.ptr.cast::<u64>(),
                mw.dev_up_ptrs.ptr.cast::<u64>(),
                mw.dev_down_ptrs.ptr.cast::<u64>(),
                mw.gu_row_bytes,
                mw.d_row_bytes,
                ms_ref.mixed_q8.ptr,
                ms_ref.topk_ids.ptr.cast::<i32>(),
                ms_ref.topk_w.ptr.cast::<f32>(),
                ms_ref.moe_inter_q8.ptr,
                std::ptr::null_mut(),
                ms_ref.block_out.ptr.cast::<f32>(),
                n_embd as i32,
                cfg.n_ff_exp as i32,
                cfg.n_expert_used as i32,
                1,
                0.0,
                stream,
            );

            if mw.sh_gate.ggml_type == GgmlType::Q8_0 && mw.sh_up.ggml_type == GgmlType::Q8_0 {
                qwen4::sh_gu_swiglu_q8(
                    mw.sh_gate.buf.ptr,
                    mw.sh_up.buf.ptr,
                    ms_ref.mixed_q8.ptr,
                    ms_ref.sh_gate.ptr.cast::<f32>(),
                    cfg.n_ff_shared as i32,
                    n_embd as i32,
                    1,
                    side_s,
                );
                launch_quantize_q8_1(
                    ms_ref.sh_gate.ptr.cast::<f32>(),
                    ms_ref.sh_hq.ptr,
                    1,
                    cfg.n_ff_shared as i32,
                    side_s.raw,
                );
            } else {
                linear_q8(
                    mw.sh_gate.as_spec(),
                    ms_ref.mixed_q8.ptr,
                    ms_ref.mixed.ptr.cast::<f32>(),
                    ms_ref.sh_gate.ptr.cast::<f32>(),
                    1,
                    side_s,
                );
                linear_q8(
                    mw.sh_up.as_spec(),
                    ms_ref.mixed_q8.ptr,
                    ms_ref.mixed.ptr.cast::<f32>(),
                    ms_ref.sh_up.ptr.cast::<f32>(),
                    1,
                    side_s,
                );
                glm5::swiglu_clamp_q8(
                    ms_ref.sh_gate.ptr.cast::<f32>(),
                    ms_ref.sh_up.ptr.cast::<f32>(),
                    0.0,
                    cfg.n_ff_shared as i32,
                    std::ptr::null_mut(),
                    ms_ref.sh_hq.ptr,
                    1,
                    side_s,
                );
            }
            linear_q8(
                mw.sh_down.as_spec(),
                ms_ref.sh_hq.ptr,
                std::ptr::null(),
                ms_ref.sh_out.ptr.cast::<f32>(),
                1,
                side_s,
            );
            qwen4::shared_gate_dot(
                ms_ref.mixed.ptr.cast::<f32>(),
                mw.sh_gate_scalar_f32.ptr,
                GgmlType::F32 as i32,
                1,
                n_embd as i32,
                ms_ref.sh_gate_logit.ptr.cast::<f32>(),
                side_s,
            );
        }
        join_ev.record(side_s)?;
        stream.wait_event(join_ev)?;

        unsafe {
            qwen4::hc_write_strided(
                ms_ref.r_mtp.ptr.cast::<f32>(),
                ms_ref.block_out.ptr.cast::<f32>(),
                ms_ref.sh_out.ptr.cast::<f32>(),
                ms_ref.sh_gate_logit.ptr.cast::<f32>(),
                ms_ref.inject.ptr.cast::<f32>(),
                hc as i32,
                1,
                hc as i32,
                n_embd as i32,
                stream,
            );

            // Final mixer -> shared output LM head
            hc_read_gpu(
                &mw.hc_mixer,
                ms_ref.r_mtp.ptr.cast::<f32>(),
                1,
                hc,
                n_embd,
                hc_lr,
                eps,
                ms_ref.xn.ptr.cast::<f32>(),
                ms_ref.xn_q8.ptr,
                ms_ref.lo.ptr.cast::<f32>(),
                ms_ref.lo_q8.ptr,
                ms_ref.hc_up.ptr.cast::<f32>(),
                ms_ref.mixed.ptr.cast::<f32>(),
                ms_ref.mixed_q8.ptr,
                std::ptr::null_mut(),
                false,
                stream,
            );
            if compute_argmax
                && mw.draft_vocab.is_some()
                && (mw.draft_head_q4.is_some() || weights.output_head.ggml_type == GgmlType::Q8_0)
            {
                let dv = mw.draft_vocab.as_ref().unwrap();
                let n_sub = mw.draft_vocab_len as i32;
                if let Some(ref dh) = mw.draft_head_q4 {
                    linear_q8(
                        dh.as_spec(),
                        ms_ref.mixed_q8.ptr,
                        std::ptr::null(),
                        ms_ref.logits.ptr.cast::<f32>(),
                        1,
                        stream,
                    );
                } else {
                    qwen4::mmvq_q8_0_indexed_m1(
                        weights.output_head.buf.ptr,
                        dv.ptr.cast::<i32>(),
                        ms_ref.mixed_q8.ptr,
                        ms_ref.logits.ptr.cast::<f32>(),
                        n_sub,
                        n_embd as i32,
                        stream,
                    );
                }
                qwen4::argmax_prob_indexed(
                    ms_ref.logits.ptr.cast::<f32>(),
                    n_sub,
                    dv.ptr.cast::<i32>(),
                    ms_ref.argmax_vals.ptr.cast::<f32>(),
                    ms_ref.argmax_ids.ptr.cast::<i32>(),
                    ms_ref.argmax_sums.ptr.cast::<f32>(),
                    ms_ref.argmax_dev_ptr as *mut i32,
                    (ms_ref.argmax_dev_ptr + 4) as *mut f32,
                    stream,
                );
            } else {
                linear_q8(
                    weights.output_head.as_spec(),
                    ms_ref.mixed_q8.ptr,
                    ms_ref.mixed.ptr.cast::<f32>(),
                    ms_ref.logits.ptr.cast::<f32>(),
                    1,
                    stream,
                );
                if compute_argmax {
                    qwen4::argmax_prob(
                        ms_ref.logits.ptr.cast::<f32>(),
                        cfg.n_vocab as i32,
                        ms_ref.argmax_vals.ptr.cast::<f32>(),
                        ms_ref.argmax_ids.ptr.cast::<i32>(),
                        ms_ref.argmax_sums.ptr.cast::<f32>(),
                        ms_ref.argmax_dev_ptr as *mut i32,
                        (ms_ref.argmax_dev_ptr + 4) as *mut f32,
                        stream,
                    );
                }
            }
        }
        Ok(())
    };

    if can_graph {
        let gkey = (is_first, cap_bucket);
        if !ms.draft_graphs.contains_key(&gkey) {
            let graph = {
                let ms_ref: &crate::qwen4::state::MtpState = &*ms;
                CudaGraphExec::capture(stream, || emit_body(ms_ref, cap_bucket))?
            };
            ms.draft_graphs.insert(gkey, graph);
        }
        ms.draft_graphs[&gkey].launch(stream)?;
    } else {
        emit_body(&*ms, cell_pos + 1)?;
    }

    stream.sync()?;
    ms.kv_pos = cell_pos + 1;
    let (tok_out, prob_out) = if compute_argmax {
        let s = ms.argmax_out.as_slice();
        (s[0] as u32, f32::from_bits(s[1] as u32))
    } else {
        (0, 1.0f32)
    };
    Ok((tok_out, prob_out))
}
