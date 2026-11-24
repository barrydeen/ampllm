use crate::qwen4::mtp::mtp_prefill_kv_parts;
use crate::qwen4::ple::{PleCursor, PleEngine};
use crate::qwen4::state::{
    GpuStageScratch, PleGpuState, Qwen4LayerState, Qwen4State, QWEN4_DMA_BATCH, QWEN4_DMA_RING,
    QWEN4_META_SLOTS, QWEN4_PREFILL_GRP_WIN,
};
use crate::qwen4::{
    drain_async_adapt_state, poll_async_adapt_state, promote_gpu_slice_sync,
    select_prompt_adapt_swaps, start_prompt_adapt_gpu1,
};
use crate::qwen4::vision::Qwen4ImageSpan;
use crate::qwen4::weights::{
    AttnLayerWeights, HcWeights, LayerExpertsStorage, LayerMoeDenseWeights, QuantMatrixSpec,
    Qwen4LayerWeights, Qwen4Weights,
};
use ampllm_cpu_kernels::glm5_cpu::{
    quantize_glm5_cpu_x_q8_1, quantize_glm5_cpu_x_q8k, run_glm5_cpu_moe_grouped_prequant,
    Glm5CpuGroupedExpert, Glm5CpuScratch,
};
use ampllm_cpu_kernels::{active_cpu_isa, tok_row_to_f32_cpu, CpuIsaTier, CpuThreadPool};
use ampllm_cuda_kernels::ffi::launch_quantize_q8_1;
use ampllm_cuda_kernels::{
    copy_d2d_raw_async, copy_d2h_raw_async, copy_h2d_raw_async, copy_peer_raw_async, glm5, hy4,
    qwen4, set_device, zero_raw_async, CudaGraphExec, CudaStream,
};
use ampllm_gguf::{GgmlType, Qwen4Config};
use anyhow::Result;
use std::ffi::c_void;
use std::time::Instant;

#[derive(Debug, Default, Clone)]
pub struct Qwen4StepProfile {
    pub steps: usize,
    pub tokens: usize,
    pub embed_ms: f64,
    pub ple_ms: f64,
    pub hc_ms: f64,
    pub gdn_ms: f64,
    pub qsa_ms: f64,
    pub moe_router_sh_ms: f64,
    pub moe_routed_ms: f64,
    pub head_ms: f64,
    pub p2p_ms: f64,
}

impl Qwen4StepProfile {
    pub fn print_summary(&self, tag: &str) {
        if self.steps == 0 {
            return;
        }
        let tot = self.embed_ms
            + self.ple_ms
            + self.hc_ms
            + self.gdn_ms
            + self.qsa_ms
            + self.moe_router_sh_ms
            + self.moe_routed_ms
            + self.head_ms
            + self.p2p_ms;
        eprintln!(
            "[QWEN4_PROFILE {tag}] steps={} tok={} tot={:.2}ms | emb={:.2} ple={:.2} hc={:.2} gdn={:.2} qsa={:.2} router_sh={:.2} moe_routed={:.2} head={:.2} p2p={:.2}",
            self.steps,
            self.tokens,
            tot,
            self.embed_ms,
            self.ple_ms,
            self.hc_ms,
            self.gdn_ms,
            self.qsa_ms,
            self.moe_router_sh_ms,
            self.moe_routed_ms,
            self.head_ms,
            self.p2p_ms,
        );
    }
}

#[inline]
pub(crate) unsafe fn linear_q8(
    mat: QuantMatrixSpec,
    x_q8: *const c_void,
    x_f32_opt: *const f32,
    y_f32: *mut f32,
    t: usize,
    stream: &CudaStream,
) {
    linear_q8_tiled(mat, x_q8, x_f32_opt, y_f32, t, 0, stream);
}

#[inline]
pub(crate) unsafe fn linear_q8_tiled(
    mat: QuantMatrixSpec,
    x_q8: *const c_void,
    x_f32_opt: *const f32,
    y_f32: *mut f32,
    t: usize,
    force_tile: i32,
    stream: &CudaStream,
) {
    if t <= 8 {
        ampllm_cuda_kernels::ffi::launch_linear_q8_1(
            mat.ggml_type as i32,
            mat.ptr as *const c_void,
            x_q8,
            x_f32_opt,
            y_f32,
            mat.n_out as i32,
            mat.n_in as i32,
            t as i32,
            stream.raw,
        );
    } else {
        glm5::gemm_mmq_mma_tiled(
            mat.ggml_type as i32,
            x_q8,
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
            force_tile,
            stream,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) unsafe fn linear_f32_or_bf16_tc(
    g: &GpuStageScratch,
    w_type: GgmlType,
    w_ptr: *const c_void,
    x_f32: *const f32,
    y_f32: *mut f32,
    t: usize,
    n_out: usize,
    k_in: usize,
    stream: &CudaStream,
) {
    if t <= 32 || n_out * k_in > g.max_f16_elems_w || t > g.sub_max {
        match w_type {
            GgmlType::Bf16 => hy4::gemv_bf16(
                w_ptr.cast::<u16>(),
                n_out as i32,
                k_in as i32,
                x_f32,
                t as i32,
                y_f32,
                1.0,
                stream,
            ),
            _ => hy4::gemv_f32(
                w_ptr.cast::<f32>(),
                n_out as i32,
                k_in as i32,
                x_f32,
                t as i32,
                y_f32,
                1.0,
                stream,
            ),
        }
    } else {
        let f16_x = g.ptr::<c_void>(g.off_f16_x);
        let f16_w = g.ptr::<c_void>(g.off_f16_w);
        glm5::f32_to_f16(x_f32, f16_x, (t * k_in) as i32, stream);
        glm5::dequant_to_f16(w_type as i32, w_ptr, f16_w, (n_out * k_in) as i64, stream);
        glm5::gemm_f16_mma_nt(
            f16_x,
            f16_w,
            y_f32,
            t as i32,
            n_out as i32,
            k_in as i32,
            1.0,
            0.0,
            stream,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn hc_read_gpu(
    hc: &HcWeights,
    r: *const f32,
    t: usize,
    hc_streams: usize,
    n_embd: usize,
    hc_lr: usize,
    eps: f32,
    xn: *mut f32,
    xn_q8: *mut c_void,
    lo: *mut f32,
    lo_q8: *mut c_void,
    up: *mut f32,
    mixed: *mut f32,
    mixed_q8: *mut c_void,
    inject_opt: *mut f32,
    fuse_inject: bool,
    stream: &CudaStream,
) {
    let fused_inj =
        fuse_inject && t <= 8 && hc_streams <= 4 && !inject_opt.is_null() && hc.inject.is_some();
    if fused_inj {
        let inj_w = hc.inject.as_ref().unwrap();
        qwen4::hc_rmsnorm_inject_q8(
            r,
            hc.norm.ptr.cast::<f32>(),
            inj_w.ptr,
            hc.inject_type as i32,
            t as i32,
            hc_streams as i32,
            n_embd as i32,
            eps,
            xn,
            xn_q8,
            inject_opt,
            stream,
        );
    } else {
        qwen4::hc_rmsnorm_q8(
            r,
            hc.norm.ptr.cast::<f32>(),
            t as i32,
            hc_streams as i32,
            n_embd as i32,
            eps,
            xn,
            xn_q8,
            stream,
        );
    }
    let down_tile = if t <= 8 {
        0
    } else if t <= 1024 {
        3
    } else {
        2
    };
    linear_q8_tiled(hc.down.as_spec(), xn_q8, xn, lo, t, down_tile, stream);
    qwen4::hc_down_silu_q8(
        lo,
        t as i32,
        hc_streams as i32,
        hc_lr as i32,
        std::ptr::null_mut(),
        lo_q8,
        stream,
    );
    if t <= 16 && hc.up.ggml_type == GgmlType::Q8_0 && hc_lr <= 512 {
        qwen4::hc_up_mix_q8(
            hc.up.buf.ptr,
            lo_q8,
            xn,
            t as i32,
            hc_streams as i32,
            n_embd as i32,
            hc_lr as i32,
            mixed,
            mixed_q8,
            stream,
        );
    } else {
        let up_tile = if t <= 512 { 3 } else { 0 };
        linear_q8_tiled(
            hc.up.as_spec(),
            lo_q8,
            std::ptr::null(),
            up,
            t,
            up_tile,
            stream,
        );
        qwen4::hc_mix_q8(
            xn,
            up,
            t as i32,
            hc_streams as i32,
            n_embd as i32,
            mixed,
            mixed_q8,
            stream,
        );
    }
    if !fused_inj && !inject_opt.is_null() {
        if let Some(ref inj_w) = hc.inject {
            qwen4::hc_inject_dot(
                xn,
                inj_w.ptr,
                hc.inject_type as i32,
                t as i32,
                hc_streams as i32,
                n_embd as i32,
                inject_opt,
                stream,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn embed_tokens_raw(
    cfg: &Qwen4Config,
    tbl: &[u8],
    rb: usize,
    ty: GgmlType,
    pool: &CpuThreadPool,
    tokens: &[u32],
    pos_start: usize,
    image_spans: &[Qwen4ImageSpan],
    out_f32: &mut [f32],
) {
    let n_embd = cfg.n_embd;
    let nt = tokens.len();
    let n_vocab = cfg.n_vocab;

    if nt < 24 || pool.num_threads() <= 1 {
        for (i, &tok) in tokens.iter().enumerate() {
            let tid = (tok as usize).min(n_vocab - 1);
            tok_row_to_f32_cpu(
                ty,
                &tbl[tid * rb..(tid + 1) * rb],
                &mut out_f32[i * n_embd..(i + 1) * n_embd],
            );
        }
    } else {
        let out_addr = out_f32.as_mut_ptr() as usize;
        pool.parallel_for(nt, |start, end| {
            for i in start..end {
                let tid = (tokens[i] as usize).min(n_vocab - 1);
                let dst = unsafe {
                    std::slice::from_raw_parts_mut((out_addr as *mut f32).add(i * n_embd), n_embd)
                };
                tok_row_to_f32_cpu(ty, &tbl[tid * rb..(tid + 1) * rb], dst);
            }
        });
    }

    if !image_spans.is_empty() {
        for span in image_spans {
            let span_end = span.start_pos + span.nx * span.ny;
            let chunk_end = pos_start + nt;
            let ov_start = span.start_pos.max(pos_start);
            let ov_end = span_end.min(chunk_end);
            for p in ov_start..ov_end {
                let local_tok = p - pos_start;
                let local_row = p - span.start_pos;
                out_f32[local_tok * n_embd..(local_tok + 1) * n_embd]
                    .copy_from_slice(&span.rows[local_row * n_embd..(local_row + 1) * n_embd]);
            }
        }
    }
}

pub(crate) fn embed_tokens_host(
    weights: &Qwen4Weights,
    pool: &CpuThreadPool,
    tokens: &[u32],
    pos_start: usize,
    image_spans: &[Qwen4ImageSpan],
    out_f32: &mut [f32],
) -> Result<()> {
    let tbl = weights.token_embd_bytes()?;
    embed_tokens_raw(
        &weights.cfg,
        tbl,
        weights.token_embd_row_bytes,
        weights.token_embd_type,
        pool,
        tokens,
        pos_start,
        image_spans,
        out_f32,
    );
    Ok(())
}

struct ColdBucket {
    e: usize,
    row_off: usize,
    nt: usize,
    tok_indices: Vec<usize>,
    pair_indices: Vec<usize>,
    tok_wts: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_qwen4_moe_group_mmq_class(
    g: &GpuStageScratch,
    mw: &LayerMoeDenseWeights,
    n_embd: usize,
    n_ff: usize,
    n_exp_max: usize,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)],
    tile_mode: i32,
    bm: usize,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let stream = &g.stream;
    let rel = &g.meta_pin_rel;
    let max_tiles = QWEN4_PREFILL_GRP_WIN.div_ceil(32) + n_exp_max;
    let fused_gu = mw.u_off as usize == mw.gate_bytes;

    let dispatch_one_window = |win: &[(u64, usize, usize)], meta_cursor: &mut usize| -> Result<()> {
        if win.is_empty() {
            return Ok(());
        }
        let slot = *meta_cursor % QWEN4_META_SLOTS;
        if *meta_cursor >= QWEN4_META_SLOTS {
            g.meta_events[slot].sync()?;
        }
        *meta_cursor += 1;

        let base_row_off = win[0].1;
        let n_exp = win.len();
        let meta_base = g.meta_pin.as_ptr().add(slot * rel.slot_bytes) as *mut u8;
        let ptrs_pin = std::slice::from_raw_parts_mut(meta_base.add(rel.exp_ptrs).cast::<u64>(), n_exp);
        let off_pin = std::slice::from_raw_parts_mut(meta_base.add(rel.exp_row_off).cast::<i32>(), n_exp);
        let nt_pin = std::slice::from_raw_parts_mut(meta_base.add(rel.exp_nt).cast::<i32>(), n_exp);
        let te_pin = std::slice::from_raw_parts_mut(meta_base.add(rel.tile_exp_id).cast::<i32>(), max_tiles);
        let tm_pin = std::slice::from_raw_parts_mut(meta_base.add(rel.tile_m_off).cast::<i32>(), max_tiles);

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

        let mdev = &g.meta_dev[slot];
        let d_ptrs = g.ptr::<u64>(mdev.exp_ptrs);
        let d_off = g.ptr::<i32>(mdev.exp_row_off);
        let d_nt = g.ptr::<i32>(mdev.exp_nt);
        let d_te = g.ptr::<i32>(mdev.tile_exp_id);
        let d_tm = g.ptr::<i32>(mdev.tile_m_off);

        copy_h2d_raw_async(
            d_ptrs.cast(),
            std::slice::from_raw_parts(ptrs_pin.as_ptr() as *const u8, n_exp * 8),
            stream,
        )?;
        copy_h2d_raw_async(
            d_off.cast(),
            std::slice::from_raw_parts(off_pin.as_ptr() as *const u8, n_exp * 4),
            stream,
        )?;
        copy_h2d_raw_async(
            d_nt.cast(),
            std::slice::from_raw_parts(nt_pin.as_ptr() as *const u8, n_exp * 4),
            stream,
        )?;
        copy_h2d_raw_async(
            d_te.cast(),
            std::slice::from_raw_parts(te_pin.as_ptr() as *const u8, n_tiles * 4),
            stream,
        )?;
        copy_h2d_raw_async(
            d_tm.cast(),
            std::slice::from_raw_parts(tm_pin.as_ptr() as *const u8, n_tiles * 4),
            stream,
        )?;

        let rows_dev = g.ptr::<i32>(g.off_small_rows).add(base_row_off);
        let wts_dev = g.ptr::<f32>(g.off_small_wts).add(base_row_off);
        let x_q8 = g.ptr::<c_void>(g.off_mixed_q8);
        let gu_f32 = g.ptr::<f32>(g.off_moe_grp_gu);
        let inter_q8 = g.ptr::<c_void>(g.off_moe_inter_q8);
        let block_out = g.ptr::<f32>(g.off_block_out);

        if fused_gu {
            glm5::gemm_mmq_mma_grouped(
                mw.gu_type as i32,
                x_q8,
                d_ptrs,
                0,
                mw.gu_row_bytes,
                gu_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                (2 * n_ff) as i32,
                n_embd as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::swiglu_clamp_interleaved_q8(
                gu_f32,
                0.0,
                w_rows as i32,
                n_ff as i32,
                inter_q8,
                stream,
            );
        } else {
            let up_f32 = gu_f32.add(w_rows * n_ff);
            glm5::gemm_mmq_mma_grouped(
                mw.gu_type as i32,
                x_q8,
                d_ptrs,
                0,
                mw.gu_row_bytes,
                gu_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                n_ff as i32,
                n_embd as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::gemm_mmq_mma_grouped(
                mw.gu_type as i32,
                x_q8,
                d_ptrs,
                mw.u_off as usize,
                mw.gu_row_bytes,
                up_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                n_ff as i32,
                n_embd as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::swiglu_clamp_q8(
                gu_f32,
                up_f32,
                0.0,
                n_ff as i32,
                std::ptr::null_mut(),
                inter_q8,
                w_rows as i32,
                stream,
            );
        }

        glm5::gemm_mmq_mma_grouped(
            mw.d_type as i32,
            inter_q8,
            d_ptrs,
            mw.d_off as usize,
            mw.d_row_bytes,
            block_out,
            std::ptr::null(),
            rows_dev,
            wts_dev,
            d_off,
            d_nt,
            d_te,
            d_tm,
            n_tiles as i32,
            n_embd as i32,
            n_ff as i32,
            1.0,
            true,
            tile_mode,
            stream,
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
                && (rows + nt > QWEN4_PREFILL_GRP_WIN
                    || tiles + jt > max_tiles
                    || exps + 1 > n_exp_max)
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

#[allow(clippy::too_many_arguments)]
unsafe fn run_qwen4_moe_group_mmq(
    g: &GpuStageScratch,
    mw: &LayerMoeDenseWeights,
    n_embd: usize,
    n_ff: usize,
    n_exp_max: usize,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)],
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let s1 = jobs.iter().take_while(|j| j.2 > 96).count();
    let s2 = s1 + jobs[s1..].iter().take_while(|j| j.2 > 32).count();
    run_qwen4_moe_group_mmq_class(g, mw, n_embd, n_ff, n_exp_max, meta_cursor, &jobs[..s1], 1, 128)?;
    run_qwen4_moe_group_mmq_class(g, mw, n_embd, n_ff, n_exp_max, meta_cursor, &jobs[s1..s2], 2, 64)?;
    run_qwen4_moe_group_mmq_class(g, mw, n_embd, n_ff, n_exp_max, meta_cursor, &jobs[s2..], 3, 32)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_moe_layer(
    lw: &Qwen4LayerWeights,
    es: &mut LayerExpertsStorage,
    g: &mut GpuStageScratch,
    cpu_pool: &CpuThreadPool,
    cpu_scratch: &mut Glm5CpuScratch,
    cfg: &Qwen4Config,
    t: usize,
    pos_start: usize,
    plen: usize,
    step_counter: u64,
    cpu_share_override: Option<f64>,
    mixed_prefetched: bool,
) -> Result<(u64, u64)> {
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let n_exp = cfg.n_expert;
    let topk = cfg.n_expert_used;
    let n_ff = cfg.n_ff_exp;
    let mw = &lw.moe;
    let total_pairs = t * topk;
    let det_prefill =
        t <= 64 && (total_pairs * n_embd * 4) <= (QWEN4_PREFILL_GRP_WIN * 2 * n_ff * 4);
    let small_max_nt = if det_prefill { 64usize } else { 8usize };

    unsafe {
        // Sync-free wait via event from stream -> d2h_stream; start zeroing block_out on stream cleanly.
        g.p2p_done.record(&g.stream)?;
        zero_raw_async(g.ptr::<c_void>(g.off_block_out), t * n_embd * 4, &g.stream)?;
        if det_prefill && total_pairs > 0 {
            zero_raw_async(
                g.ptr::<c_void>(g.off_moe_grp_gu),
                total_pairs * n_embd * 4,
                &g.stream,
            )?;
        }
        g.d2h_stream.wait_event(&g.p2p_done)?;
        copy_d2h_raw_async(
            std::slice::from_raw_parts_mut(g.h_ids.as_mut_ptr(), total_pairs * 4),
            g.ptr::<c_void>(g.off_router_ids),
            &g.d2h_stream,
        )?;
        copy_d2h_raw_async(
            std::slice::from_raw_parts_mut(g.h_w.as_mut_ptr(), total_pairs * 4),
            g.ptr::<c_void>(g.off_router_w),
            &g.d2h_stream,
        )?;
        g.d2h_done.record(&g.d2h_stream)?;
        g.d2h_done.sync()?;
    }

    let ids_slice = unsafe { std::slice::from_raw_parts(g.h_ids.as_ptr().cast::<i32>(), total_pairs) };
    let w_slice = unsafe { std::slice::from_raw_parts(g.h_w.as_ptr().cast::<f32>(), total_pairs) };

    let mut buckets: Vec<(Vec<usize>, Vec<usize>, Vec<f32>)> =
        (0..n_exp).map(|_| (Vec::new(), Vec::new(), Vec::new())).collect();
    let mut hits = 0u64;
    let mut misses = 0u64;

    for idx in 0..total_pairs {
        let tok = idx / topk;
        let eid = ids_slice[idx] as usize;
        let wt = w_slice[idx];
        if eid < n_exp {
            buckets[eid].0.push(tok);
            buckets[eid].1.push(idx);
            buckets[eid].2.push(wt);
            es.heat[eid] += 1.0;
            es.prompt_hits[eid] += 1;
            if pos_start + tok >= plen.saturating_sub(2048) {
                es.prompt_tail_hits[eid] += 1;
            }
            es.last_used[eid] = step_counter;
            if es.is_resident[eid] {
                hits += 1;
            } else {
                misses += 1;
            }
        }
    }

    let mut vram_small_raw: Vec<(usize, Vec<usize>, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut vram_large_raw: Vec<(usize, Vec<usize>, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut cold_list: Vec<ColdBucket> = Vec::new();

    for (eid, (toks, pairs, wts)) in buckets.into_iter().enumerate() {
        let nt = toks.len();
        if nt == 0 {
            continue;
        }
        if es.vram_ptrs_host[eid] != 0 {
            if nt <= small_max_nt {
                vram_small_raw.push((eid, toks, pairs, wts));
            } else if nt <= QWEN4_PREFILL_GRP_WIN {
                vram_large_raw.push((eid, toks, pairs, wts));
            } else {
                let mut pos = 0usize;
                while pos < nt {
                    let end = (pos + QWEN4_PREFILL_GRP_WIN).min(nt);
                    vram_large_raw.push((
                        eid,
                        toks[pos..end].to_vec(),
                        pairs[pos..end].to_vec(),
                        wts[pos..end].to_vec(),
                    ));
                    pos = end;
                }
            }
        } else if nt <= QWEN4_PREFILL_GRP_WIN {
            cold_list.push(ColdBucket {
                e: eid,
                row_off: 0,
                nt,
                tok_indices: toks,
                pair_indices: pairs,
                tok_wts: wts,
            });
        } else {
            let mut pos = 0usize;
            while pos < nt {
                let end = (pos + QWEN4_PREFILL_GRP_WIN).min(nt);
                cold_list.push(ColdBucket {
                    e: eid,
                    row_off: 0,
                    nt: end - pos,
                    tok_indices: toks[pos..end].to_vec(),
                    pair_indices: pairs[pos..end].to_vec(),
                    tok_wts: wts[pos..end].to_vec(),
                });
                pos = end;
            }
        }
    }

    vram_large_raw.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    cold_list.sort_by(|a, b| b.nt.cmp(&a.nt).then_with(|| a.e.cmp(&b.e)));

    let k_cold = cold_list.len();
    let n_gpu_cold = if k_cold == 0 {
        0
    } else if let Some(cpu_frac) = cpu_share_override {
        let cpu_cnt = ((k_cold as f64) * cpu_frac.clamp(0.0, 1.0)).round() as usize;
        k_cold.saturating_sub(cpu_cnt)
    } else {
        // Calibrated PCIe vs CPU cost model for Qwen4 (n_embd=2560, n_ff=640, blob~3.1-4.0MB)
        let cpu_max_nt = (t / 12).clamp(32, 256);
        let isa_mult = match active_cpu_isa() {
            CpuIsaTier::Avx512 => 1.0,
            CpuIsaTier::Avx2 => 2.0,
            CpuIsaTier::Scalar => 8.0,
        };
        let cpu_scale = (8.0 / (cpu_pool.num_threads().max(1) as f64)) * isa_mult;
        let blob_us = (mw.blob_bytes as f64) / 11_500.0;
        let comp_cost = |nt: usize| -> f64 {
            if nt <= small_max_nt {
                12.0 + 0.90 * (nt as f64)
            } else {
                16.0 + 0.75 * (nt as f64)
            }
        };
        let vram_us: f64 = (vram_small_raw.len() as f64) * 12.0
            + vram_small_raw.iter().map(|v| 0.90 * v.1.len() as f64).sum::<f64>()
            + vram_large_raw.iter().map(|v| comp_cost(v.1.len())).sum::<f64>();
        let cpu_quant_us = 45.0 + 0.5 * (t as f64);
        let out_h2d_us = ((t * n_embd * 4) as f64) / 11_500.0;
        let mut pref_gpu_comp = vec![0.0f64; k_cold + 1];
        for i in 0..k_cold {
            pref_gpu_comp[i + 1] = pref_gpu_comp[i] + comp_cost(cold_list[i].nt);
        }
        let mut suff_cpu = vec![0.0f64; k_cold + 1];
        let mut suff_ok = vec![true; k_cold + 1];
        for i in (0..k_cold).rev() {
            let nt = cold_list[i].nt;
            suff_ok[i] = suff_ok[i + 1] && nt <= cpu_max_nt;
            suff_cpu[i] = suff_cpu[i + 1] + (100.0 + 20.0 * (nt as f64)) * cpu_scale;
        }
        let mut split = k_cold;
        for cand in (0..k_cold).rev() {
            if !suff_ok[cand] {
                break;
            }
            let cpu_est = cpu_quant_us + suff_cpu[cand] + out_h2d_us;
            let gpu_est = (vram_us + pref_gpu_comp[cand]).max((cand as f64) * blob_us);
            if cpu_est <= 0.95 * gpu_est {
                split = cand;
            } else {
                break;
            }
        }
        split
    };

    // Flatten GPU rows & wts into pinned staging buffers
    let rows_pin = unsafe { std::slice::from_raw_parts_mut(g.h_small_rows.as_mut_ptr().cast::<i32>(), total_pairs) };
    let mut pair_pin = det_prefill.then(|| unsafe {
        std::slice::from_raw_parts_mut(g.h_small_pair.as_mut_ptr().cast::<i32>(), total_pairs)
    });
    let wts_pin = unsafe { std::slice::from_raw_parts_mut(g.h_small_wts.as_mut_ptr().cast::<f32>(), total_pairs) };
    let ptrs_pin = unsafe { std::slice::from_raw_parts_mut(g.h_small_ptrs.as_mut_ptr().cast::<u64>(), n_exp) };
    let off_pin = unsafe { std::slice::from_raw_parts_mut(g.h_small_off.as_mut_ptr().cast::<i32>(), n_exp) };
    let nt_pin = unsafe { std::slice::from_raw_parts_mut(g.h_small_nt.as_mut_ptr().cast::<i32>(), n_exp) };

    let mut cur_off = 0usize;
    let n_vram_small = vram_small_raw.len();
    for (si, (eid, toks, pairs, wts)) in vram_small_raw.iter().enumerate() {
        let nt = toks.len();
        ptrs_pin[si] = es.vram_ptrs_host[*eid];
        off_pin[si] = cur_off as i32;
        nt_pin[si] = nt as i32;
        for k in 0..nt {
            rows_pin[cur_off + k] = toks[k] as i32;
            if let Some(ref mut pp) = pair_pin {
                pp[cur_off + k] = pairs[k] as i32;
            }
            wts_pin[cur_off + k] = wts[k];
        }
        cur_off += nt;
    }

    let mut vram_large: Vec<(u64, usize, usize)> = Vec::with_capacity(vram_large_raw.len());
    for (eid, toks, pairs, wts) in &vram_large_raw {
        let nt = toks.len();
        vram_large.push((es.vram_ptrs_host[*eid], cur_off, nt));
        for k in 0..nt {
            rows_pin[cur_off + k] = toks[k] as i32;
            if let Some(ref mut pp) = pair_pin {
                pp[cur_off + k] = pairs[k] as i32;
            }
            wts_pin[cur_off + k] = wts[k];
        }
        cur_off += nt;
    }

    for job in &mut cold_list[..n_gpu_cold] {
        job.row_off = cur_off;
        for k in 0..job.nt {
            rows_pin[cur_off + k] = job.tok_indices[k] as i32;
            if let Some(ref mut pp) = pair_pin {
                pp[cur_off + k] = job.pair_indices[k] as i32;
            }
            wts_pin[cur_off + k] = job.tok_wts[k];
        }
        cur_off += job.nt;
    }

    let cold_groups: Vec<(usize, usize)> = {
        let mut v = Vec::with_capacity(n_gpu_cold.div_ceil(QWEN4_DMA_BATCH));
        let mut g0 = 0usize;
        while g0 < n_gpu_cold {
            let g1 = (g0 + QWEN4_DMA_BATCH).min(n_gpu_cold);
            v.push((g0, g1));
            g0 = g1;
        }
        v
    };

    // Prepare slot_off/nt for small jobs inside cold DMA batches (each batch relative to its own first small job)
    let mut n_small_meta = n_vram_small;
    let mut cold_small_meta_bases = Vec::with_capacity(cold_groups.len());
    for &(gs, ge) in &cold_groups {
        let k_g_large = cold_list[gs..ge].iter().take_while(|j| j.nt > small_max_nt).count();
        let k_g_small = (ge - gs) - k_g_large;
        cold_small_meta_bases.push(n_small_meta);
        if k_g_small > 0 {
            let mut local_off = 0i32;
            for j in &cold_list[gs + k_g_large..ge] {
                off_pin[n_small_meta] = local_off;
                nt_pin[n_small_meta] = j.nt as i32;
                local_off += j.nt as i32;
                n_small_meta += 1;
            }
        }
    }

    let cpu_jobs = &cold_list[n_gpu_cold..];
    unsafe {
        if !cpu_jobs.is_empty() && !mixed_prefetched {
            copy_d2h_raw_async(
                std::slice::from_raw_parts_mut(g.h_mixed.as_mut_ptr(), t * n_embd * 4),
                g.ptr::<c_void>(g.off_mixed),
                &g.d2h_stream,
            )?;
            g.d2h_done.record(&g.d2h_stream)?;
        }

        if cur_off > 0 {
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_small_rows),
                std::slice::from_raw_parts(rows_pin.as_ptr() as *const u8, cur_off * 4),
                &g.stream,
            )?;
            if let Some(ref pp) = pair_pin {
                copy_h2d_raw_async(
                    g.ptr::<c_void>(g.off_small_pair),
                    std::slice::from_raw_parts(pp.as_ptr() as *const u8, cur_off * 4),
                    &g.stream,
                )?;
            }
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_small_wts),
                std::slice::from_raw_parts(wts_pin.as_ptr() as *const u8, cur_off * 4),
                &g.stream,
            )?;
        }
        if n_vram_small > 0 {
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_small_ptrs),
                std::slice::from_raw_parts(ptrs_pin.as_ptr() as *const u8, n_vram_small * 8),
                &g.stream,
            )?;
        }
        if n_small_meta > 0 {
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_small_off),
                std::slice::from_raw_parts(off_pin.as_ptr() as *const u8, n_small_meta * 4),
                &g.stream,
            )?;
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_small_nt),
                std::slice::from_raw_parts(nt_pin.as_ptr() as *const u8, n_small_meta * 4),
                &g.stream,
            )?;
        }
    }

    let kick_group = |g_idx: usize, r_slot: usize| -> Result<()> {
        let (gs, ge) = cold_groups[g_idx];
        let base_s = r_slot * QWEN4_DMA_BATCH;
        let dma_base = g.dma_stage.as_u64();
        for (i, c_idx) in (gs..ge).enumerate() {
            let eid = cold_list[c_idx].e;
            let hp = es.ram_host_ptrs[eid] as *const u8;
            anyhow::ensure!(!hp.is_null(), "cold expert {eid} missing host backing ptr");
            let dst_ptr = (dma_base + ((base_s + i) * g.slot_stride) as u64) as *mut c_void;
            unsafe {
                copy_h2d_raw_async(
                    dst_ptr,
                    std::slice::from_raw_parts(hp, mw.blob_bytes),
                    &g.copy_stream,
                )?;
            }
        }
        g.dma_events[r_slot].record(&g.copy_stream)?;
        Ok(())
    };

    let init_g = QWEN4_DMA_RING.min(cold_groups.len());
    for g_idx in 0..init_g {
        kick_group(g_idx, g_idx)?;
    }

    let dev_id = lw.device;
    let h_mixed_addr = g.h_mixed.as_ptr() as usize;
    let h_cpu_addr = g.h_cpu.as_mut_ptr() as usize;
    let d2h_done_ev = if mixed_prefetched {
        &g.fetch_done_ev
    } else {
        &g.d2h_done
    };
    let ram_ptrs_ref = &es.ram_host_ptrs;

    let (cpu_res, gpu_res) = std::thread::scope(|scope| {
        let h_cpu = (!cpu_jobs.is_empty()).then(|| {
            scope.spawn(move || -> Result<bool> {
                set_device(dev_id)?;
                d2h_done_ev.sync()?;
                let host_x = unsafe { std::slice::from_raw_parts(h_mixed_addr as *const f32, t * n_embd) };
                if matches!(mw.gu_type, GgmlType::Q4K | GgmlType::Q5K) {
                    quantize_glm5_cpu_x_q8_1(cpu_pool, cpu_scratch, n_embd, t, host_x);
                } else {
                    quantize_glm5_cpu_x_q8k(cpu_pool, cpu_scratch, n_embd, t, host_x);
                }
                let cpu_out = unsafe { std::slice::from_raw_parts_mut(h_cpu_addr as *mut f32, t * n_embd) };
                let mut did_any = false;
                let mut idx = 0usize;
                let u_off = mw.u_off as usize;
                let d_off = mw.d_off as usize;
                while idx < cpu_jobs.len() {
                    let mut end = idx;
                    let mut batch_rows = 0usize;
                    let mut count = 0usize;
                    while end < cpu_jobs.len() && count < 16 {
                        let next_nt = cpu_jobs[end].nt;
                        if count > 0 && batch_rows + next_nt > 1024 {
                            break;
                        }
                        batch_rows += next_nt;
                        end += 1;
                        count += 1;
                    }
                    let groups: Vec<Glm5CpuGroupedExpert<'_>> = cpu_jobs[idx..end]
                        .iter()
                        .filter_map(|job| {
                            let base = ram_ptrs_ref[job.e];
                            if base == 0 {
                                None
                            } else {
                                Some(Glm5CpuGroupedExpert {
                                    gate_ptr: base as *const u8,
                                    up_ptr: (base + u_off) as *const u8,
                                    down_ptr: (base + d_off) as *const u8,
                                    token_indices: &job.tok_indices,
                                    token_weights: &job.tok_wts,
                                })
                            }
                        })
                        .collect();
                    if !groups.is_empty() {
                        run_glm5_cpu_moe_grouped_prequant(
                            cpu_pool,
                            cpu_scratch,
                            mw.gu_type,
                            mw.d_type,
                            n_embd,
                            n_ff,
                            0.0,
                            t,
                            &groups,
                            cpu_out,
                            !did_any,
                        );
                        did_any = true;
                    }
                    idx = end;
                }
                Ok(did_any)
            })
        });

        let gpu_closure = (|| -> Result<()> {
            let run_small = |ptrs_dev: *const u64,
                             off_dev: *const i32,
                             nt_dev: *const i32,
                             rows_dev: *const i32,
                             pairs_dev: *const i32,
                             wts_dev: *const f32,
                             nj: usize| unsafe {
                glm5::moe_prefill_small_gu(
                    mw.gu_type as i32,
                    ptrs_dev,
                    off_dev,
                    nt_dev,
                    rows_dev,
                    mw.u_off,
                    mw.gu_row_bytes,
                    g.ptr::<c_void>(g.off_mixed_q8),
                    g.ptr::<c_void>(g.off_moe_inter_q8),
                    n_embd as i32,
                    n_ff as i32,
                    nj as i32,
                    0.0,
                    &g.stream,
                );
                if det_prefill {
                    glm5::moe_prefill_small_down_det(
                        mw.d_type as i32,
                        ptrs_dev,
                        off_dev,
                        nt_dev,
                        pairs_dev,
                        wts_dev,
                        mw.d_off,
                        mw.d_row_bytes,
                        g.ptr::<c_void>(g.off_moe_inter_q8),
                        g.ptr::<f32>(g.off_moe_grp_gu),
                        n_embd as i32,
                        n_ff as i32,
                        nj as i32,
                        &g.stream,
                    );
                } else {
                    glm5::moe_prefill_small_down(
                        mw.d_type as i32,
                        ptrs_dev,
                        off_dev,
                        nt_dev,
                        rows_dev,
                        wts_dev,
                        mw.d_off,
                        mw.d_row_bytes,
                        g.ptr::<c_void>(g.off_moe_inter_q8),
                        g.ptr::<f32>(g.off_block_out),
                        n_embd as i32,
                        n_ff as i32,
                        nj as i32,
                        &g.stream,
                    );
                }
            };

            if n_vram_small > 0 {
                run_small(
                    g.ptr::<u64>(g.off_small_ptrs),
                    g.ptr::<i32>(g.off_small_off),
                    g.ptr::<i32>(g.off_small_nt),
                    g.ptr::<i32>(g.off_small_rows),
                    g.ptr::<i32>(g.off_small_pair),
                    g.ptr::<f32>(g.off_small_wts),
                    n_vram_small,
                );
            }

            let mut meta_cursor = 0usize;
            if !vram_large.is_empty() {
                unsafe {
                    run_qwen4_moe_group_mmq(
                        g,
                        mw,
                        n_embd,
                        n_ff,
                        n_exp,
                        &mut meta_cursor,
                        &vram_large,
                    )?;
                }
            }

            let dma_base = g.dma_stage.as_u64();
            let mut grp_jobs: Vec<(u64, usize, usize)> = Vec::with_capacity(QWEN4_DMA_BATCH);
            for (g_idx, &(gs, ge)) in cold_groups.iter().enumerate() {
                let r_slot = g_idx % QWEN4_DMA_RING;
                let base_s = r_slot * QWEN4_DMA_BATCH;
                g.stream.wait_event(&g.dma_events[r_slot])?;

                let k_g_large = cold_list[gs..ge].iter().take_while(|j| j.nt > small_max_nt).count();
                let k_g_small = (ge - gs) - k_g_large;

                if k_g_large > 0 {
                    grp_jobs.clear();
                    for i in 0..k_g_large {
                        let j = &cold_list[gs + i];
                        grp_jobs.push((
                            dma_base + ((base_s + i) * g.slot_stride) as u64,
                            j.row_off,
                            j.nt,
                        ));
                    }
                    unsafe {
                        run_qwen4_moe_group_mmq(
                            g,
                            mw,
                            n_embd,
                            n_ff,
                            n_exp,
                            &mut meta_cursor,
                            &grp_jobs,
                        )?;
                    }
                }

                if k_g_small > 0 {
                    let meta_base = cold_small_meta_bases[g_idx];
                    let row_base = cold_list[gs + k_g_large].row_off;
                    unsafe {
                        let ptrs_dev = g.ptr::<u64>(g.off_dma_ptrs_dev).add(base_s + k_g_large);
                        let off_dev = g.ptr::<i32>(g.off_small_off).add(meta_base);
                        let nt_dev = g.ptr::<i32>(g.off_small_nt).add(meta_base);
                        let rows_dev = g.ptr::<i32>(g.off_small_rows).add(row_base);
                        let pairs_dev = g.ptr::<i32>(g.off_small_pair).add(row_base);
                        let wts_dev = g.ptr::<f32>(g.off_small_wts).add(row_base);
                        run_small(
                            ptrs_dev, off_dev, nt_dev, rows_dev, pairs_dev, wts_dev, k_g_small,
                        );
                    }
                }

                g.compute_events[r_slot].record(&g.stream)?;
                let next_g = g_idx + QWEN4_DMA_RING;
                if next_g < cold_groups.len() {
                    g.copy_stream.wait_event(&g.compute_events[r_slot])?;
                    kick_group(next_g, r_slot)?;
                }
            }
            Ok(())
        })();

        let c_res = match h_cpu {
            Some(h) => h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("Qwen4 CPU MoE thread panicked"))),
            None => Ok(false),
        };
        (c_res, gpu_closure)
    });

    gpu_res?;
    if mixed_prefetched && cpu_jobs.is_empty() {
        g.stream.wait_event(&g.fetch_done_ev)?;
    }
    unsafe {
        if det_prefill {
            qwen4::reduce_pairs_f32(
                g.ptr::<f32>(g.off_moe_grp_gu),
                g.ptr::<f32>(g.off_block_out),
                t as i32,
                topk as i32,
                n_embd as i32,
                &g.stream,
            );
        }
        if cpu_res? {
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_mixed),
                std::slice::from_raw_parts(g.h_cpu.as_ptr(), t * n_embd * 4),
                &g.stream,
            )?;
            hy4::add_f32(
                g.ptr::<f32>(g.off_block_out),
                g.ptr::<f32>(g.off_mixed),
                t * n_embd,
                &g.stream,
            );
        }
        qwen4::hc_write(
            g.ptr::<f32>(g.off_r_buf),
            g.ptr::<f32>(g.off_block_out),
            std::ptr::null(),
            std::ptr::null(),
            g.ptr::<f32>(g.off_inject),
            t as i32,
            hc as i32,
            n_embd as i32,
            &g.stream,
        );
    }

    Ok((hits, misses))
}

#[allow(clippy::too_many_arguments)]
fn forward_half_prefill(
    cfg: &Qwen4Config,
    layers_all: &[Qwen4LayerWeights],
    tok_embd: (&[u8], usize, GgmlType),
    ple_bytes: &[u8],
    layer_range: std::ops::Range<usize>,
    experts_slice: &mut [LayerExpertsStorage],
    layer_states: &mut [Qwen4LayerState],
    g: &mut GpuStageScratch,
    mut ple_opt: Option<&mut PleGpuState>,
    ple_engine: Option<&PleEngine>,
    ple_cursor: &mut PleCursor,
    cpu_pool: &CpuThreadPool,
    cpu_scratch: &mut Glm5CpuScratch,
    tokens: &[u32],
    pos_start: usize,
    image_spans: &[Qwen4ImageSpan],
    mrope_active: bool,
    emb_host_opt: Option<&mut [f32]>,
    is_gpu0: bool,
    plen: usize,
    max_ctx: usize,
    max_idx_cap: usize,
    step_counter: u64,
    cpu_share_override: Option<f64>,
    mut profile: Option<&mut Qwen4StepProfile>,
) -> Result<(u64, u64)> {
    let t = tokens.len();
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let hc_dim = hc * n_embd;
    let hc_lr = cfg.hc_low_rank;
    let eps = cfg.rms_norm_eps;
    let gdn_hk = cfg.ssm_group_count;
    let gdn_hv = cfg.ssm_time_step_rank;
    let gdn_qkv_ch = cfg.gdn_qkv_dim();
    let gdn_qk_heads = 2 * gdn_hk;
    let qsa_nh = cfg.n_head;
    let qsa_nkv = cfg.n_head_kv;
    let qsa_hd = cfg.head_dim;
    let rope_dim = cfg.rope_dim;
    let rope_base = cfg.rope_base;
    let qsa_scale = 1.0f32 / (qsa_hd as f32).sqrt();
    let max_blocks = (max_ctx / 4) + 1;
    let n_exp = cfg.n_expert;
    let topk = cfg.n_expert_used;
    let n_ff_sh = cfg.n_ff_shared;
    let q8_row_x = (n_embd / 32) * 36;
    let prof_on = profile.is_some();

    set_device(g.device)?;
    let h_pos_slice = unsafe { std::slice::from_raw_parts_mut(g.h_pos.as_mut_ptr().cast::<i32>(), t) };
    for i in 0..t {
        h_pos_slice[i] = (pos_start + i) as i32;
    }
    unsafe {
        copy_h2d_raw_async(
            g.ptr::<c_void>(g.off_pos_dev),
            std::slice::from_raw_parts(g.h_pos.as_ptr(), t * 4),
            &g.stream,
        )?;
    }

    if is_gpu0 {
        let t_emb = if prof_on { Some(Instant::now()) } else { None };
        let emb_buf = emb_host_opt.expect("GPU0 requires emb staging buffer");
        embed_tokens_raw(
            cfg,
            tok_embd.0,
            tok_embd.1,
            tok_embd.2,
            cpu_pool,
            tokens,
            pos_start,
            image_spans,
            &mut emb_buf[..t * n_embd],
        );
        unsafe {
            copy_h2d_raw_async(
                g.ptr::<c_void>(g.off_mixed),
                bytemuck::cast_slice(&emb_buf[..t * n_embd]),
                &g.stream,
            )?;
            qwen4::hc_expand(
                g.ptr::<f32>(g.off_mixed),
                t as i32,
                hc as i32,
                n_embd as i32,
                g.ptr::<f32>(g.off_r_buf),
                &g.stream,
            );
        }
        if let Some(ts) = t_emb {
            g.stream.sync()?;
            if let Some(ref mut p) = profile {
                p.embed_ms += ts.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    let sub = g.sub_max.max(1);
    let mrope_opt = if mrope_active {
        g.ptr::<i32>(g.off_mrope_pos3_dev).cast_const()
    } else {
        std::ptr::null()
    };

    for (loc_il, il) in layer_range.enumerate() {
        let lw = &layers_all[il];

        // Sub-chunked PLE + Attention (`sub = 1024`)
        let mut s0 = 0usize;
        let mut sub_idx = 0usize;
        while s0 < t {
            let sb = (t - s0).min(sub);
            let r_sub = unsafe { g.ptr::<f32>(g.off_r_buf).add(s0 * hc_dim) };
            let pos_sub = unsafe { g.ptr::<i32>(g.off_pos_dev).add(s0) };
            let inj_sub = unsafe { g.ptr::<f32>(g.off_inject).add(s0 * hc) };
            let mixed_sub = g.ptr::<f32>(g.off_mixed);
            let mixed_q8_sub = g.ptr::<c_void>(g.off_mixed_q8);
            let block_out_sub = g.ptr::<f32>(g.off_block_out);

            if let Some(ref ple_w) = lw.ple {
                let t_ple = if prof_on { Some(Instant::now()) } else { None };
                let ple_st = ple_opt
                    .as_deref_mut()
                    .ok_or_else(|| anyhow::anyhow!("missing Qwen4 PLE GPU state"))?;
                let pe = ple_engine.ok_or_else(|| anyhow::anyhow!("missing Qwen4 PleEngine"))?;
                let slot = sub_idx & 1;
                if sub_idx >= 2 {
                    ple_st.h2d_events[slot].sync()?;
                }
                pe.lookup_batch(
                    ple_bytes,
                    &tokens[s0..s0 + sb],
                    ple_cursor,
                    ple_st.host_f32_slot_mut(slot, sb * n_embd),
                )?;
                unsafe {
                    copy_h2d_raw_async(
                        ple_st.emb.ptr,
                        ple_st.host_slot_bytes(slot, sb * n_embd),
                        &g.stream,
                    )?;
                    ple_st.h2d_events[slot].record(&g.stream)?;
                    launch_quantize_q8_1(
                        ple_st.emb.ptr.cast::<f32>(),
                        mixed_q8_sub,
                        sb as i32,
                        n_embd as i32,
                        g.stream.raw,
                    );
                    linear_q8(
                        ple_w.key.as_spec(),
                        mixed_q8_sub,
                        ple_st.emb.ptr.cast::<f32>(),
                        ple_st.key.ptr.cast::<f32>(),
                        sb,
                        &g.stream,
                    );
                    linear_q8(
                        ple_w.value.as_spec(),
                        mixed_q8_sub,
                        ple_st.emb.ptr.cast::<f32>(),
                        ple_st.val.ptr.cast::<f32>(),
                        sb,
                        &g.stream,
                    );
                    qwen4::ple_postops(
                        ple_st.key.ptr.cast::<f32>(),
                        r_sub,
                        ple_st.val.ptr.cast::<f32>(),
                        ple_st.history.ptr.cast::<f32>(),
                        ple_w.norm_key.ptr.cast::<f32>(),
                        ple_w.norm_query.ptr.cast::<f32>(),
                        ple_w.norm_conv.ptr.cast::<f32>(),
                        ple_w.conv1d.ptr.cast::<f32>(),
                        ple_st.scratch_qn.ptr.cast::<f32>(),
                        ple_st.scratch_gated.ptr.cast::<f32>(),
                        ple_st.scratch_gate.ptr.cast::<f32>(),
                        sb as i32,
                        n_embd as i32,
                        hc as i32,
                        eps,
                        std::ptr::null_mut(),
                        &g.stream,
                    );
                }
                if let Some(ts) = t_ple {
                    g.stream.sync()?;
                    if let Some(ref mut p) = profile {
                        p.ple_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            // Attention HC read
            {
                let t_hc = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    hc_read_gpu(
                        &lw.hc_attn,
                        r_sub,
                        sb,
                        hc,
                        n_embd,
                        hc_lr,
                        eps,
                        g.ptr::<f32>(g.off_xn),
                        g.ptr::<c_void>(g.off_xn_q8),
                        g.ptr::<f32>(g.off_hc_lo),
                        g.ptr::<c_void>(g.off_hc_lo_q8),
                        g.ptr::<f32>(g.off_hc_up),
                        mixed_sub,
                        mixed_q8_sub,
                        inj_sub,
                        false,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_hc {
                    g.stream.sync()?;
                    if let Some(ref mut p) = profile {
                        p.hc_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            // Sequence mixer: GDN or QSA
            match (&lw.attn, &mut layer_states[loc_il]) {
                (AttnLayerWeights::Gdn(gw), Qwen4LayerState::Gdn(gs)) => {
                    let t_gdn = if prof_on { Some(Instant::now()) } else { None };
                    unsafe {
                        let qkv_ptr = g.ptr::<f32>(g.off_gdn_qkv);
                        let z_ptr = g.ptr::<f32>(g.off_gdn_z_gate);
                        let h_qkv_ptr = g.ptr::<f32>(g.off_gdn_h_qkv);
                        let gate_ptr = g.ptr::<f32>(g.off_gdn_gate);
                        let beta_ptr = g.ptr::<f32>(g.off_gdn_beta);
                        let y_ptr = g.ptr::<f32>(g.off_gdn_y);
                        let y_q8_ptr = g.ptr::<c_void>(g.off_gdn_y_q8);

                        linear_q8(gw.attn_qkv.as_spec(), mixed_q8_sub, mixed_sub, qkv_ptr, sb, &g.stream);
                        linear_q8(gw.attn_gate.as_spec(), mixed_q8_sub, mixed_sub, z_ptr, sb, &g.stream);
                        if sb <= 32 {
                            qwen4::gdn_ab_f32(
                                mixed_sub,
                                gw.ssm_alpha.ptr.cast::<f32>(),
                                gw.ssm_beta.ptr.cast::<f32>(),
                                gw.ssm_dt_bias.ptr.cast::<f32>(),
                                gw.ssm_a.ptr.cast::<f32>(),
                                gate_ptr,
                                beta_ptr,
                                sb as i32,
                                n_embd as i32,
                                gdn_hv as i32,
                                &g.stream,
                            );
                        } else {
                            let alpha_raw = y_ptr;
                            let beta_raw = y_ptr.add(sb * gdn_hv);
                            linear_f32_or_bf16_tc(
                                g,
                                GgmlType::F32,
                                gw.ssm_alpha.ptr,
                                mixed_sub,
                                alpha_raw,
                                sb,
                                gdn_hv,
                                n_embd,
                                &g.stream,
                            );
                            linear_f32_or_bf16_tc(
                                g,
                                GgmlType::F32,
                                gw.ssm_beta.ptr,
                                mixed_sub,
                                beta_raw,
                                sb,
                                gdn_hv,
                                n_embd,
                                &g.stream,
                            );
                            qwen4::gdn_gate_beta_postop(
                                alpha_raw,
                                beta_raw,
                                gw.ssm_dt_bias.ptr.cast::<f32>(),
                                gw.ssm_a.ptr.cast::<f32>(),
                                gate_ptr,
                                beta_ptr,
                                sb as i32,
                                gdn_hv as i32,
                                &g.stream,
                            );
                        }
                        qwen4::gdn_conv_l2(
                            gs.conv_hist.ptr.cast::<f32>(),
                            qkv_ptr,
                            gw.ssm_conv1d.ptr.cast::<f32>(),
                            h_qkv_ptr,
                            sb as i32,
                            gdn_qkv_ch as i32,
                            gdn_qk_heads as i32,
                            eps,
                            std::ptr::null_mut(),
                            &g.stream,
                        );
                        if sb == 1 {
                            qwen4::gdn_decode_step_norm_q8(
                                gs.state.ptr.cast::<f32>(),
                                h_qkv_ptr,
                                gate_ptr,
                                beta_ptr,
                                z_ptr,
                                gw.ssm_norm.ptr.cast::<f32>(),
                                eps,
                                std::ptr::null_mut(),
                                y_q8_ptr,
                                gdn_hk as i32,
                                gdn_hv as i32,
                                &g.stream,
                            );
                        } else if sb <= 8 {
                            qwen4::gdn_verify_steps_norm_q8(
                                gs.state.ptr.cast::<f32>(),
                                h_qkv_ptr,
                                gate_ptr,
                                beta_ptr,
                                z_ptr,
                                gw.ssm_norm.ptr.cast::<f32>(),
                                eps,
                                std::ptr::null_mut(),
                                y_q8_ptr,
                                std::ptr::null_mut(),
                                sb as i32,
                                gdn_hk as i32,
                                gdn_hv as i32,
                                &g.stream,
                            );
                        } else {
                            qwen4::gdn_tc_prefill(
                                gs.state.ptr.cast::<f32>(),
                                h_qkv_ptr,
                                gate_ptr,
                                beta_ptr,
                                z_ptr,
                                gw.ssm_norm.ptr.cast::<f32>(),
                                eps,
                                y_ptr,
                                std::ptr::null_mut(),
                                g.ptr::<c_void>(g.off_gdn_tc_scratch),
                                sb as i32,
                                gdn_hk as i32,
                                gdn_hv as i32,
                                &g.stream,
                            );
                            launch_quantize_q8_1(
                                y_ptr,
                                y_q8_ptr,
                                sb as i32,
                                cfg.ssm_inner_size as i32,
                                g.stream.raw,
                            );
                        }
                        linear_q8(gw.ssm_out.as_spec(), y_q8_ptr, std::ptr::null(), block_out_sub, sb, &g.stream);
                    }
                    if let Some(ts) = t_gdn {
                        g.stream.sync()?;
                        if let Some(ref mut p) = profile {
                            p.gdn_ms += ts.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
                (AttnLayerWeights::Qsa(qw), Qwen4LayerState::Qsa(qs)) => {
                    let t_qsa = if prof_on { Some(Instant::now()) } else { None };
                    unsafe {
                        let qfull_p = g.ptr::<f32>(g.off_qfull);
                        let q_p = g.ptr::<f32>(g.off_q);
                        let k_p = g.ptr::<f32>(g.off_k);
                        let v_p = g.ptr::<f32>(g.off_v);
                        let q_idx_p = g.ptr::<f32>(g.off_q_idx);
                        let k_idx_p = g.ptr::<f32>(g.off_k_idx);
                        let widths_p = g.ptr::<i32>(g.off_widths_dev);
                        let attn_out_p = g.ptr::<f32>(g.off_attn_out);
                        let attn_q8_p = g.ptr::<c_void>(g.off_attn_q8);

                        linear_q8(qw.attn_q.as_spec(), mixed_q8_sub, mixed_sub, qfull_p, sb, &g.stream);
                        linear_q8(qw.attn_k.as_spec(), mixed_q8_sub, mixed_sub, k_p, sb, &g.stream);
                        linear_q8(qw.attn_v.as_spec(), mixed_q8_sub, mixed_sub, v_p, sb, &g.stream);

                        qwen4::qsa_norm_imrope(
                            qfull_p,
                            (qsa_hd * 2) as i32,
                            qw.attn_q_norm.ptr.cast::<f32>(),
                            q_p,
                            sb as i32,
                            qsa_nh as i32,
                            qsa_hd as i32,
                            rope_dim as i32,
                            eps,
                            rope_base,
                            pos_sub,
                            mrope_opt,
                            &g.stream,
                        );
                        qwen4::qsa_norm_imrope(
                            k_p,
                            qsa_hd as i32,
                            qw.attn_k_norm.ptr.cast::<f32>(),
                            k_p,
                            sb as i32,
                            qsa_nkv as i32,
                            qsa_hd as i32,
                            rope_dim as i32,
                            eps,
                            rope_base,
                            pos_sub,
                            mrope_opt,
                            &g.stream,
                        );
                        qwen4::kv_append_q8(
                            k_p,
                            v_p,
                            pos_sub,
                            sb as i32,
                            qsa_nkv as i32,
                            qsa_hd as i32,
                            qs.k_q.ptr.cast::<i8>(),
                            qs.v_q.ptr.cast::<i8>(),
                            qs.k_sc.ptr.cast::<u16>(),
                            qs.v_sc.ptr.cast::<u16>(),
                            &g.stream,
                        );

                        linear_f32_or_bf16_tc(
                            g,
                            GgmlType::Bf16,
                            qw.idx_q_proj_bf16.ptr,
                            mixed_sub,
                            q_idx_p,
                            sb,
                            cfg.idx_heads * cfg.idx_key_dim,
                            n_embd,
                            &g.stream,
                        );
                        qwen4::qsa_norm_imrope(
                            q_idx_p,
                            cfg.idx_key_dim as i32,
                            qw.idx_q_norm.ptr.cast::<f32>(),
                            q_idx_p,
                            sb as i32,
                            cfg.idx_heads as i32,
                            cfg.idx_key_dim as i32,
                            rope_dim as i32,
                            eps,
                            rope_base,
                            pos_sub,
                            mrope_opt,
                            &g.stream,
                        );
                        linear_f32_or_bf16_tc(
                            g,
                            GgmlType::Bf16,
                            qw.idx_k_proj_bf16.ptr,
                            mixed_sub,
                            k_idx_p,
                            sb,
                            cfg.idx_key_dim,
                            n_embd,
                            &g.stream,
                        );

                        if sb <= 8 {
                            qwen4::idx_append_steps(
                                k_idx_p,
                                pos_sub,
                                sb as i32,
                                qw.idx_k_norm.ptr.cast::<f32>(),
                                eps,
                                rope_base,
                                mrope_opt,
                                qs.tail.ptr.cast::<f32>(),
                                qs.dead.ptr.cast::<f32>(),
                                qs.pooled.ptr.cast::<f32>(),
                                max_ctx as i32,
                                &g.stream,
                            );
                        } else {
                            qwen4::idx_append_batch(
                                k_idx_p,
                                sb as i32,
                                (pos_start + s0) as i32,
                                qw.idx_k_norm.ptr.cast::<f32>(),
                                eps,
                                rope_base,
                                mrope_opt,
                                qs.tail.ptr.cast::<f32>(),
                                qs.dead.ptr.cast::<f32>(),
                                qs.pooled.ptr.cast::<f32>(),
                                max_ctx as i32,
                                &g.stream,
                            );
                        }

                        let max_n_kv = pos_start + s0 + sb;
                        let sparse = max_n_kv > max_idx_cap;
                        let (ids_ptr, max_n_ids, id_cap) = if sparse {
                            let used_blocks = (max_n_kv / 4 + 1).min(max_blocks);
                            qwen4::idx_score_select(
                                qs.pooled.ptr.cast::<f32>(),
                                qs.dead.ptr.cast::<f32>(),
                                q_idx_p,
                                pos_sub,
                                sb as i32,
                                used_blocks as i32,
                                cfg.idx_top_k as i32,
                                g.ptr::<f32>(g.off_idx_scores),
                                g.ptr::<i32>(g.off_idx_ids),
                                widths_p,
                                &g.stream,
                            );
                            (g.ptr::<i32>(g.off_idx_ids).cast_const(), cfg.idx_top_k, cfg.idx_top_k)
                        } else {
                            qwen4::fill_dense_widths(pos_sub, widths_p, sb as i32, &g.stream);
                            (std::ptr::null(), max_n_kv, max_idx_cap)
                        };

                        if sb <= 8 {
                            qwen4::qsa_decode_attn_q8(
                                q_p,
                                qfull_p,
                                qs.k_q.ptr.cast::<i8>(),
                                qs.v_q.ptr.cast::<i8>(),
                                qs.k_sc.ptr.cast::<u16>(),
                                qs.v_sc.ptr.cast::<u16>(),
                                ids_ptr,
                                widths_p,
                                max_n_ids as i32,
                                id_cap as i32,
                                sb as i32,
                                qsa_nkv as i32,
                                qsa_scale,
                                g.ptr::<f32>(g.off_qsa_ws),
                                std::ptr::null_mut(),
                                attn_q8_p,
                                &g.stream,
                            );
                        } else {
                            qwen4::qsa_prefill_attn_q8(
                                q_p,
                                qfull_p,
                                qs.k_q.ptr.cast::<i8>(),
                                qs.v_q.ptr.cast::<i8>(),
                                qs.k_sc.ptr.cast::<u16>(),
                                qs.v_sc.ptr.cast::<u16>(),
                                ids_ptr,
                                widths_p,
                                id_cap as i32,
                                sb as i32,
                                qsa_nkv as i32,
                                qsa_scale,
                                attn_out_p,
                                &g.stream,
                            );
                            launch_quantize_q8_1(
                                attn_out_p,
                                attn_q8_p,
                                sb as i32,
                                (qsa_nh * qsa_hd) as i32,
                                g.stream.raw,
                            );
                        }
                        linear_q8(qw.attn_output.as_spec(), attn_q8_p, std::ptr::null(), block_out_sub, sb, &g.stream);
                    }
                    if let Some(ts) = t_qsa {
                        g.stream.sync()?;
                        if let Some(ref mut p) = profile {
                            p.qsa_ms += ts.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
                _ => anyhow::bail!("layer {il} attn weight/state mismatch"),
            }

            // Write attention block_out into R_sub
            {
                let t_hc = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    qwen4::hc_write(
                        r_sub,
                        block_out_sub,
                        std::ptr::null(),
                        std::ptr::null(),
                        inj_sub,
                        sb as i32,
                        hc as i32,
                        n_embd as i32,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_hc {
                    g.stream.sync()?;
                    if let Some(ref mut p) = profile {
                        p.hc_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            s0 += sb;
            sub_idx += 1;
        }

        // Sub-chunked MLP HC read + Router + Shared Expert (shared expert folds directly into R per sub-chunk)
        let mw = &lw.moe;
        let prefetch_mixed = t >= 128
            && experts_slice[loc_il].n_resident < cfg.n_expert
            && cpu_share_override != Some(0.0);
        let mut s0 = 0usize;
        while s0 < t {
            let sb = (t - s0).min(sub);
            let r_sub = unsafe { g.ptr::<f32>(g.off_r_buf).add(s0 * hc_dim) };
            let mixed_sub = unsafe { g.ptr::<f32>(g.off_mixed).add(s0 * n_embd) };
            let mixed_q8_sub = unsafe { g.ptr::<u8>(g.off_mixed_q8).add(s0 * q8_row_x).cast::<c_void>() };
            let inj_sub = unsafe { g.ptr::<f32>(g.off_inject).add(s0 * hc) };
            let rlogits_sub = unsafe { g.ptr::<f32>(g.off_router_logits).add(s0 * n_exp) };
            let rids_sub = unsafe { g.ptr::<i32>(g.off_router_ids).add(s0 * topk) };
            let rw_sub = unsafe { g.ptr::<f32>(g.off_router_w).add(s0 * topk) };

            {
                let t_hc = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    hc_read_gpu(
                        &lw.hc_ffn,
                        r_sub,
                        sb,
                        hc,
                        n_embd,
                        hc_lr,
                        eps,
                        g.ptr::<f32>(g.off_xn),
                        g.ptr::<c_void>(g.off_xn_q8),
                        g.ptr::<f32>(g.off_hc_lo),
                        g.ptr::<c_void>(g.off_hc_lo_q8),
                        g.ptr::<f32>(g.off_hc_up),
                        mixed_sub,
                        mixed_q8_sub,
                        inj_sub,
                        false,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_hc {
                    g.stream.sync()?;
                    if let Some(ref mut p) = profile {
                        p.hc_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            if prefetch_mixed {
                unsafe {
                    g.fetch_start_ev.record(&g.stream)?;
                    g.fetch_stream.wait_event(&g.fetch_start_ev)?;
                    let byte_off = s0 * n_embd * 4;
                    let byte_len = sb * n_embd * 4;
                    copy_d2h_raw_async(
                        std::slice::from_raw_parts_mut(g.h_mixed.as_mut_ptr().add(byte_off), byte_len),
                        mixed_sub.cast(),
                        &g.fetch_stream,
                    )?;
                }
            }

            {
                let t_rsh = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    linear_f32_or_bf16_tc(
                        g,
                        GgmlType::F32,
                        mw.gate_inp.ptr,
                        mixed_sub,
                        rlogits_sub,
                        sb,
                        n_exp,
                        n_embd,
                        &g.stream,
                    );
                    qwen4::router_top10(
                        rlogits_sub,
                        sb as i32,
                        n_exp as i32,
                        topk as i32,
                        rids_sub,
                        rw_sub,
                        &g.stream,
                    );

                    let sh_g = g.ptr::<f32>(g.off_sh_gate);
                    let sh_u = g.ptr::<f32>(g.off_sh_up);
                    let sh_hq = g.ptr::<c_void>(g.off_sh_hq);
                    let sh_out = g.ptr::<f32>(g.off_sh_out);
                    let sh_gl = g.ptr::<f32>(g.off_sh_gate_logit);

                    linear_q8(mw.sh_gate.as_spec(), mixed_q8_sub, mixed_sub, sh_g, sb, &g.stream);
                    linear_q8(mw.sh_up.as_spec(), mixed_q8_sub, mixed_sub, sh_u, sb, &g.stream);
                    glm5::swiglu_clamp_q8(sh_g, sh_u, 0.0, n_ff_sh as i32, std::ptr::null_mut(), sh_hq, sb as i32, &g.stream);
                    linear_q8(mw.sh_down.as_spec(), sh_hq, std::ptr::null(), sh_out, sb, &g.stream);
                    qwen4::shared_gate_dot(
                        mixed_sub,
                        mw.gate_inp_shexp.ptr,
                        GgmlType::F32 as i32,
                        sb as i32,
                        n_embd as i32,
                        sh_gl,
                        &g.stream,
                    );
                    qwen4::hc_write(
                        r_sub,
                        std::ptr::null(),
                        sh_out,
                        sh_gl,
                        inj_sub,
                        sb as i32,
                        hc as i32,
                        n_embd as i32,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_rsh {
                    g.stream.sync()?;
                    if let Some(ref mut p) = profile {
                        p.moe_router_sh_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }
            s0 += sb;
        }
        if prefetch_mixed {
            g.fetch_done_ev.record(&g.fetch_stream)?;
        }

        // Routed MoE across full chunk `t`
        let t_moe = if prof_on { Some(Instant::now()) } else { None };
        let (lh, lm) = run_prefill_moe_layer(
            lw,
            &mut experts_slice[loc_il],
            g,
            cpu_pool,
            cpu_scratch,
            cfg,
            t,
            pos_start,
            plen,
            step_counter,
            cpu_share_override,
            prefetch_mixed,
        )?;
        total_hits += lh;
        total_misses += lm;
        if let Some(ts) = t_moe {
            g.stream.sync()?;
            if let Some(ref mut p) = profile {
                p.moe_routed_ms += ts.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    Ok((total_hits, total_misses))
}

#[inline]
unsafe fn emit_head_logits_and_argmax(
    weights: &Qwen4Weights,
    hg: &GpuStageScratch,
    r_last_ptr: *mut c_void,
    t: usize,
    keep_r_last_rows: usize,
    n_logit_rows: usize,
    greedy_argmax: bool,
) -> Result<()> {
    let cfg = &weights.cfg;
    let hc = cfg.hc;
    let n_embd = cfg.n_embd;
    let hc_dim = hc * n_embd;
    let hc_lr = cfg.hc_low_rank;
    let eps = cfg.rms_norm_eps;
    let n_vocab = cfg.n_vocab;

    if keep_r_last_rows > 0 {
        let kr = keep_r_last_rows.min(t);
        let src_off = (t - kr) * hc_dim * 4;
        copy_d2d_raw_async(
            r_last_ptr,
            hg.ptr::<u8>(hg.off_r_buf).add(src_off).cast(),
            kr * hc_dim * 4,
            &hg.stream,
        )?;
    }

    if n_logit_rows > 0 {
        let nr = n_logit_rows.min(t);
        let r_src = hg.ptr::<u8>(hg.off_r_buf).add((t - nr) * hc_dim * 4).cast::<f32>();
        hc_read_gpu(
            &weights.output_hc,
            r_src,
            nr,
            hc,
            n_embd,
            hc_lr,
            eps,
            hg.ptr::<f32>(hg.off_xn),
            hg.ptr::<c_void>(hg.off_xn_q8),
            hg.ptr::<f32>(hg.off_hc_lo),
            hg.ptr::<c_void>(hg.off_hc_lo_q8),
            hg.ptr::<f32>(hg.off_hc_up),
            hg.ptr::<f32>(hg.off_mixed),
            hg.ptr::<c_void>(hg.off_mixed_q8),
            std::ptr::null_mut(),
            false,
            &hg.stream,
        );
        linear_q8(
            weights.output_head.as_spec(),
            hg.ptr::<c_void>(hg.off_mixed_q8),
            hg.ptr::<f32>(hg.off_mixed),
            hg.ptr::<f32>(hg.off_logits),
            nr,
            &hg.stream,
        );
        if greedy_argmax {
            let logits_p = hg.ptr::<f32>(hg.off_logits);
            let vals_p = hg.ptr::<f32>(hg.off_argmax_vals);
            let ids_p = hg.ptr::<i32>(hg.off_argmax_ids);
            let out_p = hg.ptr::<i32>(hg.off_argmax_out);
            for r in 0..nr {
                glm5::argmax(
                    logits_p.add(r * n_vocab),
                    n_vocab as i32,
                    n_vocab as i32,
                    std::ptr::null(),
                    vals_p,
                    ids_p,
                    out_p.add(r),
                    &hg.stream,
                );
            }
        }
    }
    Ok(())
}

/// Dual-GPU pipelined prefill across arbitrary `tokens.len()` prompt length.
/// Populates MTP KV directly from `g1.off_r_buf` when `use_mtp == true` and computes final token logits.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_pipelined(
    weights: &mut Qwen4Weights,
    state: &mut Qwen4State,
    ple_engine: Option<&PleEngine>,
    tokens: &[u32],
    pos_offset: usize,
    image_spans: &[Qwen4ImageSpan],
    use_mtp: bool,
    greedy_first: bool,
    mut profile: Option<&mut Qwen4StepProfile>,
) -> Result<Option<u32>> {
    let plen = tokens.len();
    if plen == 0 {
        return Ok(None);
    }
    let total_len = pos_offset + plen;
    drain_async_adapt_state(weights, state)?;
    let cfg = &weights.cfg;
    let hc_dim = cfg.hc * cfg.n_embd;
    let n_layer = cfg.n_layer;
    let split = weights.split_layer.min(n_layer);
    let n_gpus = state.gpus.len();
    let max_t = state.max_t;
    let max_ctx = state.max_ctx;
    let max_idx_cap = state.max_idx_cap;
    let mrope_active = state.mrope_active;
    state.step_counter += 1;
    let step_id = state.step_counter;

    let cpu_share_override = std::env::var("AMPLLM_QWEN4_CPU_SHARE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());

    let chunk_env_set = std::env::var("AMPLLM_QWEN4_CHUNK").is_ok();
    let chunk_size = if n_gpus >= 2 && !chunk_env_set && profile.is_none() {
        if plen <= 2048 {
            plen
        } else if plen <= 6144 {
            plen.div_ceil(2).min(max_t)
        } else {
            max_t
        }
    } else {
        max_t
    };

    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut c0 = 0usize;
    while c0 < plen {
        let c1 = (c0 + chunk_size).min(plen);
        chunks.push((c0, c1));
        c0 = c1;
    }

    let tok_embd = (
        weights.gguf.tensor_bytes("token_embd.weight")?,
        weights.token_embd_row_bytes,
        weights.token_embd_type,
    );
    let ple_bytes = match weights.ple_shard_idx {
        Some(s) => &weights.gguf.shard_bytes(s)[weights.ple_offset..weights.ple_offset + weights.ple_byte_len],
        None => &[],
    };
    let cfg_ref = &weights.cfg;
    let layers_ref = &weights.layers;
    let prof_on = profile.is_some();

    if n_gpus == 1 || chunks.len() == 1 || prof_on {
        // Sequential execution (single GPU, single chunk, or exact layer profiling) using full 16-core pool
        for (ci, &(cs, ce)) in chunks.iter().enumerate() {
            let t = ce - cs;
            let is_last = ci + 1 == chunks.len();
            let emb_slice = unsafe {
                std::slice::from_raw_parts_mut(state.h_emb_stage.as_mut_ptr().cast::<f32>(), t * cfg_ref.n_embd)
            };
            let (h0, m0) = forward_half_prefill(
                cfg_ref,
                layers_ref,
                tok_embd,
                ple_bytes,
                0..split,
                &mut weights.experts[0..split],
                &mut state.layers[0..split],
                &mut state.gpus[0],
                state.ple.as_mut(),
                ple_engine,
                &mut state.ple_cursor,
                &state.pool,
                &mut state.cpu_scratch_full,
                &tokens[cs..ce],
                pos_offset + cs,
                image_spans,
                mrope_active,
                Some(emb_slice),
                true,
                total_len,
                max_ctx,
                max_idx_cap,
                step_id,
                cpu_share_override,
                profile.as_deref_mut(),
            )?;
            state.cache_stats.hits += h0;
            state.cache_stats.misses += m0;

            if n_gpus >= 2 {
                let t_p2p = if prof_on { Some(Instant::now()) } else { None };
                set_device(state.gpus[0].device)?;
                state.gpus[0].p2p_done.record(&state.gpus[0].stream)?;
                set_device(state.gpus[1].device)?;
                state.gpus[1].stream.wait_event(&state.gpus[0].p2p_done)?;
                unsafe {
                    copy_peer_raw_async(
                        state.gpus[1].ptr::<c_void>(state.gpus[1].off_r_buf),
                        state.gpus[1].device,
                        state.gpus[0].ptr::<c_void>(state.gpus[0].off_r_buf),
                        state.gpus[0].device,
                        t * hc_dim * 4,
                        &state.gpus[1].stream,
                    )?;
                }
                if let Some(ts) = t_p2p {
                    state.gpus[1].stream.sync()?;
                    if let Some(ref mut p) = profile {
                        p.p2p_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                let (h1, m1) = forward_half_prefill(
                    cfg_ref,
                    layers_ref,
                    tok_embd,
                    ple_bytes,
                    split..n_layer,
                    &mut weights.experts[split..n_layer],
                    &mut state.layers[split..n_layer],
                    &mut state.gpus[1],
                    None,
                    None,
                    &mut state.ple_cursor,
                    &state.pool,
                    &mut state.cpu_scratch_full,
                    &tokens[cs..ce],
                    pos_offset + cs,
                    image_spans,
                    mrope_active,
                    None,
                    false,
                    total_len,
                    max_ctx,
                    max_idx_cap,
                    step_id,
                    cpu_share_override,
                    profile.as_deref_mut(),
                )?;
                state.cache_stats.hits += h1;
                state.cache_stats.misses += m1;
            }

            let hg_idx = state.gpus.len() - 1;
            let hg = &state.gpus[hg_idx];
            set_device(hg.device)?;
            let keep_r = if use_mtp && is_last { 1 } else { 0 };
            let n_log = if is_last { 1 } else { 0 };
            let t_head = if prof_on && is_last { Some(Instant::now()) } else { None };
            unsafe {
                emit_head_logits_and_argmax(
                    weights,
                    hg,
                    state.r_last.ptr,
                    t,
                    keep_r,
                    n_log,
                    greedy_first && is_last,
                )?;
            }
            if is_last && greedy_first {
                unsafe {
                    copy_d2h_raw_async(
                        std::slice::from_raw_parts_mut(hg.small_pin.as_mut_ptr(), 4),
                        hg.ptr::<c_void>(hg.off_argmax_out),
                        &hg.stream,
                    )?;
                }
            }
            hg.stream.sync()?;
            if let Some(ts) = t_head {
                if let Some(ref mut p) = profile {
                    p.head_ms += ts.elapsed().as_secs_f64() * 1000.0;
                }
            }

            if use_mtp {
                let mtp_end = ce.min(plen - 1);
                if mtp_end > cs {
                    let hg = &state.gpus[hg_idx];
                    let r_ptr = hg.ptr::<f32>(hg.off_r_buf).cast_const();
                    let mrope_ptr = if mrope_active {
                        hg.ptr::<i32>(hg.off_mrope_pos3_dev).cast_const()
                    } else {
                        std::ptr::null()
                    };
                    if let (Some(ref mw), Some(ref mut ms)) = (&weights.mtp, &mut state.mtp) {
                        mtp_prefill_kv_parts(
                            cfg_ref,
                            mw,
                            tok_embd,
                            ms,
                            &hg.stream,
                            Some((&hg.side_stream, &hg.side_fork_ev, &hg.side_join_ev)),
                            mrope_ptr,
                            &state.pool,
                            &tokens[cs + 1..mtp_end + 1],
                            r_ptr,
                            pos_offset + cs,
                            false,
                            false,
                        )?;
                    }
                }
            }
            if let Some(ref mut p) = profile {
                p.steps += 1;
                p.tokens += t;
            }
        }
    } else {
        // Dual-GPU multi-chunk pipelining using CCD-split pools (`pool0` / `pool1`)
        let pool0 = state.pool0.as_ref().unwrap_or(&state.pool);
        let pool1 = state.pool1.as_ref().unwrap_or(&state.pool);
        let mut sc0_opt = state.cpu_scratch0.take();
        let mut sc1_opt = state.cpu_scratch1.take();
        let sc0 = sc0_opt.get_or_insert_with(|| Glm5CpuScratch::new(cfg_ref.n_embd, cfg_ref.n_ff_exp, max_t.max(64)));
        let sc1 = sc1_opt.get_or_insert_with(|| Glm5CpuScratch::new(cfg_ref.n_embd, cfg_ref.n_ff_exp, max_t.max(64)));

        let (cs0, ce0) = chunks[0];
        let t0 = ce0 - cs0;
        let emb_slice0 = unsafe {
            std::slice::from_raw_parts_mut(state.h_emb_stage.as_mut_ptr().cast::<f32>(), t0 * cfg_ref.n_embd)
        };
        let (h0, m0) = forward_half_prefill(
            cfg_ref,
            layers_ref,
            tok_embd,
            ple_bytes,
            0..split,
            &mut weights.experts[0..split],
            &mut state.layers[0..split],
            &mut state.gpus[0],
            state.ple.as_mut(),
            ple_engine,
            &mut state.ple_cursor,
            &state.pool,
            &mut state.cpu_scratch_full,
            &tokens[cs0..ce0],
            pos_offset + cs0,
            image_spans,
            mrope_active,
            Some(emb_slice0),
            true,
            total_len,
            max_ctx,
            max_idx_cap,
            step_id,
            cpu_share_override,
            None,
        )?;
        state.cache_stats.hits += h0;
        state.cache_stats.misses += m0;

        set_device(state.gpus[0].device)?;
        state.gpus[0].p2p_done.record(&state.gpus[0].stream)?;
        set_device(state.gpus[1].device)?;
        state.gpus[1].stream.wait_event(&state.gpus[0].p2p_done)?;
        unsafe {
            copy_peer_raw_async(
                state.gpus[1].ptr::<c_void>(state.gpus[1].off_r_buf),
                state.gpus[1].device,
                state.gpus[0].ptr::<c_void>(state.gpus[0].off_r_buf),
                state.gpus[0].device,
                t0 * hc_dim * 4,
                &state.gpus[1].stream,
            )?;
        }
        state.gpus[1].p2p_done.record(&state.gpus[1].stream)?;

        for ci in 0..chunks.len() - 1 {
            let (cs_prev, ce_prev) = chunks[ci];
            let (cs_next, ce_next) = chunks[ci + 1];
            let t_prev = ce_prev - cs_prev;
            let t_next = ce_next - cs_next;

            // GPU0 waits only for P2P read of g0.r_buf before overwriting g0.r_buf with chunk `ci + 1`
            set_device(state.gpus[0].device)?;
            state.gpus[0].stream.wait_event(&state.gpus[1].p2p_done)?;

            let (g0_s, g1_s) = state.gpus.split_at_mut(1);
            let g0 = &mut g0_s[0];
            let g1 = &mut g1_s[0];
            let (exp0, exp1) = weights.experts.split_at_mut(split);
            let (layers0, layers1) = state.layers.split_at_mut(split);
            let emb_slice_next = unsafe {
                std::slice::from_raw_parts_mut(state.h_emb_stage.as_mut_ptr().cast::<f32>(), t_next * cfg_ref.n_embd)
            };

            let (r0, r1) = std::thread::scope(|s| {
                let h_g0 = s.spawn(|| -> Result<(u64, u64)> {
                    forward_half_prefill(
                        cfg_ref,
                        layers_ref,
                        tok_embd,
                        ple_bytes,
                        0..split,
                        exp0,
                        layers0,
                        g0,
                        state.ple.as_mut(),
                        ple_engine,
                        &mut state.ple_cursor,
                        pool0,
                        sc0,
                        &tokens[cs_next..ce_next],
                        pos_offset + cs_next,
                        image_spans,
                        mrope_active,
                        Some(emb_slice_next),
                        true,
                        total_len,
                        max_ctx,
                        max_idx_cap,
                        step_id,
                        cpu_share_override,
                        None,
                    )
                });
                let res1 = (|| -> Result<(u64, u64)> {
                    let mut dummy_ple = PleCursor::default();
                    let hm = forward_half_prefill(
                        cfg_ref,
                        layers_ref,
                        tok_embd,
                        ple_bytes,
                        split..n_layer,
                        exp1,
                        layers1,
                        g1,
                        None,
                        None,
                        &mut dummy_ple,
                        pool1,
                        sc1,
                        &tokens[cs_prev..ce_prev],
                        pos_offset + cs_prev,
                        image_spans,
                        mrope_active,
                        None,
                        false,
                        total_len,
                        max_ctx,
                        max_idx_cap,
                        step_id,
                        cpu_share_override,
                        None,
                    )?;
                    if use_mtp {
                        let mtp_end = ce_prev.min(plen - 1);
                        if mtp_end > cs_prev {
                            let r_ptr = g1.ptr::<f32>(g1.off_r_buf).cast_const();
                            let mrope_ptr = if mrope_active {
                                g1.ptr::<i32>(g1.off_mrope_pos3_dev).cast_const()
                            } else {
                                std::ptr::null()
                            };
                            if let (Some(ref mw), Some(ref mut ms)) = (&weights.mtp, &mut state.mtp) {
                                mtp_prefill_kv_parts(
                                    cfg_ref,
                                    mw,
                                    tok_embd,
                                    ms,
                                    &g1.stream,
                                    Some((&g1.side_stream, &g1.side_fork_ev, &g1.side_join_ev)),
                                    mrope_ptr,
                                    pool1,
                                    &tokens[cs_prev + 1..mtp_end + 1],
                                    r_ptr,
                                    pos_offset + cs_prev,
                                    false,
                                    false,
                                )?;
                            }
                        }
                    } else {
                        g1.stream.sync()?;
                    }
                    let _ = t_prev;
                    Ok(hm)
                })();
                (h_g0.join().expect("gpu0 prefill panicked"), res1)
            });
            let (h0_n, m0_n) = r0?;
            let (h1_p, m1_p) = r1?;
            state.cache_stats.hits += h0_n + h1_p;
            state.cache_stats.misses += m0_n + m1_p;

            set_device(state.gpus[0].device)?;
            state.gpus[0].p2p_done.record(&state.gpus[0].stream)?;
            set_device(state.gpus[1].device)?;
            state.gpus[1].stream.wait_event(&state.gpus[0].p2p_done)?;
            unsafe {
                copy_peer_raw_async(
                    state.gpus[1].ptr::<c_void>(state.gpus[1].off_r_buf),
                    state.gpus[1].device,
                    state.gpus[0].ptr::<c_void>(state.gpus[0].off_r_buf),
                    state.gpus[0].device,
                    t_next * hc_dim * 4,
                    &state.gpus[1].stream,
                )?;
            }
            state.gpus[1].p2p_done.record(&state.gpus[1].stream)?;
        }

        state.cpu_scratch0 = sc0_opt;
        state.cpu_scratch1 = sc1_opt;

        // Drain final chunk on GPU 1 using full 16-core pool while overlapping GPU 0 prompt promotion
        let (cs_last, ce_last) = *chunks.last().unwrap();
        let t_last = ce_last - cs_last;
        let has_profile = weights.has_profile;
        let max_prompt_slots = std::env::var("AMPLLM_QWEN4_PROMPT_ADAPT_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if has_profile { 0 } else { 16 })
            .min(16);
        let layer_bbs: Vec<usize> = layers_ref.iter().map(|l| l.moe.blob_bytes).collect();
        let host_atomic = weights.layers_host_atomic.clone();
        let (exp0, exp1) = weights.experts.split_at_mut(split);
        let (g0_s, g1_s) = state.gpus.split_at_mut(1);
        let g0 = &mut g0_s[0];
        let g1 = &mut g1_s[0];
        let p0_per_layer = if has_profile {
            max_prompt_slots.min(8)
        } else {
            max_prompt_slots
        };
        let p0_max_swaps = if has_profile { 160 } else { 384 };
        let swaps0 = if plen >= 512 {
            select_prompt_adapt_swaps(
                has_profile,
                cfg_ref.n_expert,
                &layer_bbs,
                0,
                0,
                exp0,
                p0_per_layer,
                p0_max_swaps,
            )
        } else {
            Vec::new()
        };
        let (r_p0, r_g1) = std::thread::scope(|s| {
            let h_p0 = (!swaps0.is_empty()).then(|| {
                s.spawn(|| promote_gpu_slice_sync(swaps0, 0, exp0, &host_atomic, g0))
            });
            let res1 = forward_half_prefill(
                cfg_ref,
                layers_ref,
                tok_embd,
                ple_bytes,
                split..n_layer,
                exp1,
                &mut state.layers[split..n_layer],
                g1,
                None,
                None,
                &mut state.ple_cursor,
                &state.pool,
                &mut state.cpu_scratch_full,
                &tokens[cs_last..ce_last],
                pos_offset + cs_last,
                image_spans,
                mrope_active,
                None,
                false,
                total_len,
                max_ctx,
                max_idx_cap,
                step_id,
                cpu_share_override,
                None,
            );
            let p0_res = match h_p0 {
                Some(h) => h
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("GPU0 prompt adapt thread panicked"))),
                None => Ok(0),
            };
            (p0_res, res1)
        });
        let promos0 = r_p0?;
        let (h1_f, m1_f) = r_g1?;
        state.cache_stats.promotions += promos0;
        state.cache_stats.hits += h1_f;
        state.cache_stats.misses += m1_f;

        let hg = &state.gpus[1];
        set_device(hg.device)?;
        let keep_r = if use_mtp { 1 } else { 0 };
        unsafe {
            emit_head_logits_and_argmax(
                weights,
                hg,
                state.r_last.ptr,
                t_last,
                keep_r,
                1,
                greedy_first,
            )?;
            if greedy_first {
                copy_d2h_raw_async(
                    std::slice::from_raw_parts_mut(hg.small_pin.as_mut_ptr(), 4),
                    hg.ptr::<c_void>(hg.off_argmax_out),
                    &hg.stream,
                )?;
            }
        }
        hg.stream.sync()?;

        let gpu1_adapt_max = std::env::var("AMPLLM_QWEN4_PROMPT_ADAPT_GPU1_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if has_profile { 0 } else { 192 });
        if plen >= 512 && gpu1_adapt_max > 0 {
            let p1_per_layer = if has_profile {
                max_prompt_slots.min(8)
            } else {
                max_prompt_slots
            };
            let _ = start_prompt_adapt_gpu1(
                has_profile,
                cfg_ref.n_expert,
                &layer_bbs,
                &mut weights.experts[split..],
                state,
                split,
                p1_per_layer,
            )?;
        }

        if use_mtp {
            let mtp_end = ce_last.min(plen - 1);
            if mtp_end > cs_last {
                let hg = &state.gpus[1];
                let r_ptr = hg.ptr::<f32>(hg.off_r_buf).cast_const();
                let mrope_ptr = if mrope_active {
                    hg.ptr::<i32>(hg.off_mrope_pos3_dev).cast_const()
                } else {
                    std::ptr::null()
                };
                if let (Some(ref mw), Some(ref mut ms)) = (&weights.mtp, &mut state.mtp) {
                    mtp_prefill_kv_parts(
                        cfg_ref,
                        mw,
                        tok_embd,
                        ms,
                        &hg.stream,
                        Some((&hg.side_stream, &hg.side_fork_ev, &hg.side_join_ev)),
                        mrope_ptr,
                        &state.pool,
                        &tokens[cs_last + 1..mtp_end + 1],
                        r_ptr,
                        pos_offset + cs_last,
                        false,
                        false,
                    )?;
                }
            }
        }
        poll_async_adapt_state(weights, state)?;
    }

    if greedy_first {
        let hg = state.gpus.last().unwrap();
        let tok = unsafe { *hg.small_pin.as_ptr().cast::<i32>() } as u32;
        Ok(Some(tok))
    } else {
        Ok(None)
    }
}

/// Graphed Doorbell decode / speculative verify step for `1 <= tokens.len() <= state.max_verify`.
/// When `greedy_argmax == true`, returns the `t` argmax token IDs without transferring logits to host.
#[allow(clippy::too_many_arguments)]
pub fn forward_decode_verify_step(
    weights: &mut Qwen4Weights,
    state: &mut Qwen4State,
    ple_engine: Option<&PleEngine>,
    tokens: &[u32],
    pos_start: usize,
    save_snap: bool,
    greedy_argmax: bool,
    mut profile: Option<&mut Qwen4StepProfile>,
) -> Result<Vec<u32>> {
    let t = tokens.len();
    if t == 0 {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        t <= state.max_verify,
        "forward_decode_verify_step: t={t} exceeds max_verify={}",
        state.max_verify
    );

    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let hc_dim = hc * n_embd;
    let hc_lr = cfg.hc_low_rank;
    let eps = cfg.rms_norm_eps;
    let gdn_hk = cfg.ssm_group_count;
    let gdn_hv = cfg.ssm_time_step_rank;
    let gdn_qkv_ch = cfg.gdn_qkv_dim();
    let gdn_qk_heads = 2 * gdn_hk;
    let qsa_nh = cfg.n_head;
    let qsa_nkv = cfg.n_head_kv;
    let qsa_hd = cfg.head_dim;
    let rope_dim = cfg.rope_dim;
    let rope_base = cfg.rope_base;
    let qsa_scale = 1.0f32 / (qsa_hd as f32).sqrt();
    let max_blocks = (state.max_ctx / 4) + 1;
    let n_exp = cfg.n_expert;
    let topk = cfg.n_expert_used;
    let n_ff = cfg.n_ff_exp;
    let n_ff_sh = cfg.n_ff_shared;
    let n_layer = cfg.n_layer;
    let split = weights.split_layer.min(n_layer);
    let n_gpus = state.gpus.len();
    let prof_on = profile.is_some();

    let env_pcie_div = std::env::var("AMPLLM_QWEN4_PCIE_DIV")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .map(|v| v.max(0));
    let cpu_isa = active_cpu_isa();
    let pcie_div = env_pcie_div.unwrap_or(match cpu_isa {
        CpuIsaTier::Avx512 => 0,
        CpuIsaTier::Avx2 => 2,
        CpuIsaTier::Scalar => 1,
    });
    let pcie_div_t1 = env_pcie_div.unwrap_or(match cpu_isa {
        CpuIsaTier::Avx512 | CpuIsaTier::Avx2 => 0,
        CpuIsaTier::Scalar => 2,
    });
    let pcie_max = std::env::var("AMPLLM_QWEN4_PCIE_MAX")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(2)
        .max(0);

    state.step_counter += 1;
    let step_id = state.step_counter;

    // Pre-graph host prep into PinnedMappedBuffer (SM-read inside graph; zero H2D copy engine usage)
    let t_emb0 = Instant::now();
    let emb_slice = &mut state.decode_emb_pin.as_mut_slice()[..t * n_embd];
    embed_tokens_host(weights, &state.pool, tokens, pos_start, &[], emb_slice)?;
    if save_snap {
        state.ple_cursor_snap = Some(state.ple_cursor);
    }

    for g in &mut state.gpus {
        let hp = &mut g.decode_pos_pin.as_mut_slice()[..t];
        for i in 0..t {
            hp[i] = (pos_start + i) as i32;
        }
    }

    if let Some(ref mut ple_st) = state.ple {
        let pe = ple_engine.ok_or_else(|| anyhow::anyhow!("missing Qwen4 PleEngine"))?;
        pe.lookup_batch(
            weights.ple_table_bytes(),
            tokens,
            &mut state.ple_cursor,
            &mut ple_st.decode_emb_pin.as_mut_slice()[..t * n_embd],
        )?;
    }
    if let Some(ref mut p) = profile {
        p.embed_ms += t_emb0.elapsed().as_secs_f64() * 1000.0;
    }

    let pos_end = pos_start + t;
    let sparse = pos_end > state.max_idx_cap;
    let can_graph = !state.mrope_active && !prof_on && state.use_cuda_graph;
    let bucket_qsa_blocks = if can_graph {
        max_blocks
    } else {
        ((pos_end / 4) + 1).min(max_blocks)
    };
    let bucket_dense_n = if can_graph { state.max_idx_cap } else { pos_end };

    let emit_gpu_half = |gidx: usize,
                         g: &GpuStageScratch,
                         layer_range: std::ops::Range<usize>,
                         layers_slice: &[Qwen4LayerState],
                         ple_ref: Option<&PleGpuState>,
                         is_first_gpu: bool,
                         is_head_gpu: bool,
                         prof_opt: &mut Option<&mut Qwen4StepProfile>|
     -> Result<()> {
        set_device(g.device)?;
        let db_dev = state.doorbell.dev_ptr(gidx);
        let mrope_opt = if state.mrope_active {
            g.ptr::<i32>(g.off_mrope_pos3_dev).cast_const()
        } else {
            std::ptr::null()
        };

        unsafe {
            qwen4::copy_u32_sm(
                g.ptr::<c_void>(g.off_pos_dev),
                g.decode_pos_dev as *const c_void,
                t as i32,
                false,
                &g.stream,
            );
        }

        if is_first_gpu {
            unsafe {
                qwen4::copy_u32_sm(
                    g.ptr::<c_void>(g.off_mixed),
                    state.decode_emb_dev as *const c_void,
                    (t * n_embd) as i32,
                    false,
                    &g.stream,
                );
                if let Some(ple_st) = ple_ref {
                    qwen4::copy_u32_sm(
                        ple_st.emb.ptr,
                        ple_st.decode_emb_dev as *const c_void,
                        (t * n_embd) as i32,
                        false,
                        &g.stream,
                    );
                }
                qwen4::hc_expand(
                    g.ptr::<f32>(g.off_mixed),
                    t as i32,
                    hc as i32,
                    n_embd as i32,
                    g.ptr::<f32>(g.off_r_buf),
                    &g.stream,
                );
            }
        }

        let r_buf = g.ptr::<f32>(g.off_r_buf);
        let mixed = g.ptr::<f32>(g.off_mixed);
        let mixed_q8 = g.ptr::<c_void>(g.off_mixed_q8);
        let block_out = g.ptr::<f32>(g.off_block_out);
        let inject = g.ptr::<f32>(g.off_inject);
        let pos_dev = g.ptr::<i32>(g.off_pos_dev);
        let widths_dev = g.ptr::<i32>(g.off_widths_dev);

        for (loc_il, il) in layer_range.enumerate() {
            let lw = &weights.layers[il];
            let es = &weights.experts[il];

            if let (Some(ref ple_w), Some(ple_st)) = (&lw.ple, ple_ref) {
                let t_ple = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    launch_quantize_q8_1(
                        ple_st.emb.ptr.cast::<f32>(),
                        mixed_q8,
                        t as i32,
                        n_embd as i32,
                        g.stream.raw,
                    );
                    linear_q8(
                        ple_w.key.as_spec(),
                        mixed_q8,
                        ple_st.emb.ptr.cast::<f32>(),
                        ple_st.key.ptr.cast::<f32>(),
                        t,
                        &g.stream,
                    );
                    linear_q8(
                        ple_w.value.as_spec(),
                        mixed_q8,
                        ple_st.emb.ptr.cast::<f32>(),
                        ple_st.val.ptr.cast::<f32>(),
                        t,
                        &g.stream,
                    );
                    let snap_ptr = if save_snap {
                        ple_st
                            .snap
                            .as_ref()
                            .map_or(std::ptr::null_mut(), |b| b.ptr.cast::<f32>())
                    } else {
                        std::ptr::null_mut()
                    };
                    qwen4::ple_postops(
                        ple_st.key.ptr.cast::<f32>(),
                        r_buf,
                        ple_st.val.ptr.cast::<f32>(),
                        ple_st.history.ptr.cast::<f32>(),
                        ple_w.norm_key.ptr.cast::<f32>(),
                        ple_w.norm_query.ptr.cast::<f32>(),
                        ple_w.norm_conv.ptr.cast::<f32>(),
                        ple_w.conv1d.ptr.cast::<f32>(),
                        ple_st.scratch_qn.ptr.cast::<f32>(),
                        ple_st.scratch_gated.ptr.cast::<f32>(),
                        ple_st.scratch_gate.ptr.cast::<f32>(),
                        t as i32,
                        n_embd as i32,
                        hc as i32,
                        eps,
                        snap_ptr,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_ple {
                    g.stream.sync()?;
                    if let Some(ref mut p) = prof_opt {
                        p.ple_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            // Attention HC read
            {
                let t_hc = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    hc_read_gpu(
                        &lw.hc_attn,
                        r_buf,
                        t,
                        hc,
                        n_embd,
                        hc_lr,
                        eps,
                        g.ptr::<f32>(g.off_xn),
                        g.ptr::<c_void>(g.off_xn_q8),
                        g.ptr::<f32>(g.off_hc_lo),
                        g.ptr::<c_void>(g.off_hc_lo_q8),
                        g.ptr::<f32>(g.off_hc_up),
                        mixed,
                        mixed_q8,
                        inject,
                        true,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_hc {
                    g.stream.sync()?;
                    if let Some(ref mut p) = prof_opt {
                        p.hc_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            // Sequence mixer
            match (&lw.attn, &layers_slice[loc_il]) {
                (AttnLayerWeights::Gdn(gw), Qwen4LayerState::Gdn(gs)) => {
                    let t_gdn = if prof_on { Some(Instant::now()) } else { None };
                    unsafe {
                        let qkv_p = g.ptr::<f32>(g.off_gdn_qkv);
                        let z_p = g.ptr::<f32>(g.off_gdn_z_gate);
                        let h_qkv_p = g.ptr::<f32>(g.off_gdn_h_qkv);
                        let gate_p = g.ptr::<f32>(g.off_gdn_gate);
                        let beta_p = g.ptr::<f32>(g.off_gdn_beta);
                        let y_q8_p = g.ptr::<c_void>(g.off_gdn_y_q8);
                        let gdn_ab_s = if !prof_on {
                            g.side_fork_ev.record(&g.stream)?;
                            g.side_stream.wait_event(&g.side_fork_ev)?;
                            &g.side_stream
                        } else {
                            &g.stream
                        };

                        if t <= 8
                            && gw.attn_qkv.ggml_type == GgmlType::Q8_0
                            && gw.attn_gate.ggml_type == GgmlType::Q8_0
                        {
                            qwen4::mmvq_q8_0_multi2(
                                gw.attn_qkv.buf.ptr,
                                qkv_p,
                                gw.attn_qkv.n_out as i32,
                                gw.attn_gate.buf.ptr,
                                z_p,
                                gw.attn_gate.n_out as i32,
                                mixed_q8,
                                n_embd as i32,
                                t as i32,
                                &g.stream,
                            );
                        } else {
                            linear_q8(gw.attn_qkv.as_spec(), mixed_q8, mixed, qkv_p, t, &g.stream);
                            linear_q8(gw.attn_gate.as_spec(), mixed_q8, mixed, z_p, t, &g.stream);
                        }
                        qwen4::gdn_ab_f32(
                            mixed,
                            gw.ssm_alpha.ptr.cast::<f32>(),
                            gw.ssm_beta.ptr.cast::<f32>(),
                            gw.ssm_dt_bias.ptr.cast::<f32>(),
                            gw.ssm_a.ptr.cast::<f32>(),
                            gate_p,
                            beta_p,
                            t as i32,
                            n_embd as i32,
                            gdn_hv as i32,
                            gdn_ab_s,
                        );
                        let hsnap_ptr = if save_snap {
                            gs.hist_snap
                                .as_ref()
                                .map_or(std::ptr::null_mut(), |b| b.ptr.cast::<f32>())
                        } else {
                            std::ptr::null_mut()
                        };
                        qwen4::gdn_conv_l2(
                            gs.conv_hist.ptr.cast::<f32>(),
                            qkv_p,
                            gw.ssm_conv1d.ptr.cast::<f32>(),
                            h_qkv_p,
                            t as i32,
                            gdn_qkv_ch as i32,
                            gdn_qk_heads as i32,
                            eps,
                            hsnap_ptr,
                            &g.stream,
                        );
                        if save_snap {
                            if let Some(ref hq) = gs.h_qkv_snap {
                                copy_d2d_raw_async(hq.ptr, h_qkv_p.cast(), t * gdn_qkv_ch * 4, &g.stream)?;
                            }
                            if let Some(ref gt) = gs.gate_snap {
                                copy_d2d_raw_async(gt.ptr, gate_p.cast(), t * gdn_hv * 4, gdn_ab_s)?;
                            }
                            if let Some(ref bt) = gs.beta_snap {
                                copy_d2d_raw_async(bt.ptr, beta_p.cast(), t * gdn_hv * 4, gdn_ab_s)?;
                            }
                        }
                        if !prof_on {
                            g.side_join_ev.record(&g.side_stream)?;
                            g.stream.wait_event(&g.side_join_ev)?;
                        }
                        let ssnap_ptr = if save_snap {
                            gs.state_snap
                                .as_ref()
                                .map_or(std::ptr::null_mut(), |b| b.ptr.cast::<f32>())
                        } else {
                            std::ptr::null_mut()
                        };
                        qwen4::gdn_steps_split_norm_q8(
                            gs.state.ptr.cast::<f32>(),
                            h_qkv_p,
                            gate_p,
                            beta_p,
                            z_p,
                            gw.ssm_norm.ptr.cast::<f32>(),
                            eps,
                            g.ptr::<f32>(g.off_gdn_y),
                            std::ptr::null_mut(),
                            y_q8_p,
                            ssnap_ptr,
                            t as i32,
                            gdn_hk as i32,
                            gdn_hv as i32,
                            &g.stream,
                        );
                        linear_q8(gw.ssm_out.as_spec(), y_q8_p, std::ptr::null(), block_out, t, &g.stream);
                    }
                    if let Some(ts) = t_gdn {
                        g.stream.sync()?;
                        if let Some(ref mut p) = prof_opt {
                            p.gdn_ms += ts.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
                (AttnLayerWeights::Qsa(qw), Qwen4LayerState::Qsa(qs)) => {
                    let t_qsa = if prof_on { Some(Instant::now()) } else { None };
                    unsafe {
                        let qfull_p = g.ptr::<f32>(g.off_qfull);
                        let q_p = g.ptr::<f32>(g.off_q);
                        let k_p = g.ptr::<f32>(g.off_k);
                        let v_p = g.ptr::<f32>(g.off_v);
                        let q_idx_p = g.ptr::<f32>(g.off_q_idx);
                        let k_idx_p = g.ptr::<f32>(g.off_k_idx);
                        let attn_q8_p = g.ptr::<c_void>(g.off_attn_q8);
                        let idx_s = if !prof_on {
                            g.side_fork_ev.record(&g.stream)?;
                            g.side_stream.wait_event(&g.side_fork_ev)?;
                            &g.side_stream
                        } else {
                            &g.stream
                        };

                        if t <= 8
                            && qw.attn_q.ggml_type == GgmlType::Q8_0
                            && qw.attn_k.ggml_type == GgmlType::Q8_0
                            && qw.attn_v.ggml_type == GgmlType::Q8_0
                        {
                            qwen4::mmvq_q8_0_multi3(
                                qw.attn_q.buf.ptr,
                                qfull_p,
                                qw.attn_q.n_out as i32,
                                qw.attn_k.buf.ptr,
                                k_p,
                                qw.attn_k.n_out as i32,
                                qw.attn_v.buf.ptr,
                                v_p,
                                qw.attn_v.n_out as i32,
                                mixed_q8,
                                n_embd as i32,
                                t as i32,
                                &g.stream,
                            );
                        } else {
                            linear_q8(qw.attn_q.as_spec(), mixed_q8, mixed, qfull_p, t, &g.stream);
                            linear_q8(qw.attn_k.as_spec(), mixed_q8, mixed, k_p, t, &g.stream);
                            linear_q8(qw.attn_v.as_spec(), mixed_q8, mixed, v_p, t, &g.stream);
                        }

                        qwen4::qsa_norm_imrope(
                            qfull_p,
                            (qsa_hd * 2) as i32,
                            qw.attn_q_norm.ptr.cast::<f32>(),
                            q_p,
                            t as i32,
                            qsa_nh as i32,
                            qsa_hd as i32,
                            rope_dim as i32,
                            eps,
                            rope_base,
                            pos_dev,
                            mrope_opt,
                            &g.stream,
                        );
                        qwen4::qsa_norm_imrope(
                            k_p,
                            qsa_hd as i32,
                            qw.attn_k_norm.ptr.cast::<f32>(),
                            k_p,
                            t as i32,
                            qsa_nkv as i32,
                            qsa_hd as i32,
                            rope_dim as i32,
                            eps,
                            rope_base,
                            pos_dev,
                            mrope_opt,
                            &g.stream,
                        );
                        qwen4::kv_append_q8(
                            k_p,
                            v_p,
                            pos_dev,
                            t as i32,
                            qsa_nkv as i32,
                            qsa_hd as i32,
                            qs.k_q.ptr.cast::<i8>(),
                            qs.v_q.ptr.cast::<i8>(),
                            qs.k_sc.ptr.cast::<u16>(),
                            qs.v_sc.ptr.cast::<u16>(),
                            &g.stream,
                        );

                        hy4::gemv_bf16(
                            qw.idx_q_proj_bf16.ptr.cast::<u16>(),
                            (cfg.idx_heads * cfg.idx_key_dim) as i32,
                            n_embd as i32,
                            mixed,
                            t as i32,
                            q_idx_p,
                            1.0,
                            idx_s,
                        );
                        qwen4::qsa_norm_imrope(
                            q_idx_p,
                            cfg.idx_key_dim as i32,
                            qw.idx_q_norm.ptr.cast::<f32>(),
                            q_idx_p,
                            t as i32,
                            cfg.idx_heads as i32,
                            cfg.idx_key_dim as i32,
                            rope_dim as i32,
                            eps,
                            rope_base,
                            pos_dev,
                            mrope_opt,
                            idx_s,
                        );
                        hy4::gemv_bf16(
                            qw.idx_k_proj_bf16.ptr.cast::<u16>(),
                            cfg.idx_key_dim as i32,
                            n_embd as i32,
                            mixed,
                            t as i32,
                            k_idx_p,
                            1.0,
                            idx_s,
                        );

                        if save_snap {
                            if let (Some(ref ts), Some(ref ds), Some(ref ks)) =
                                (&qs.tail_snap, &qs.dead_snap, &qs.k_idx_snap)
                            {
                                copy_d2d_raw_async(ts.ptr, qs.tail.ptr, 4 * cfg.idx_key_dim * 4, idx_s)?;
                                copy_d2d_raw_async(ds.ptr, qs.dead.ptr, cfg.idx_key_dim * 4, idx_s)?;
                                copy_d2d_raw_async(ks.ptr, k_idx_p.cast(), t * cfg.idx_key_dim * 4, idx_s)?;
                            }
                        }

                        qwen4::idx_append_steps(
                            k_idx_p,
                            pos_dev,
                            t as i32,
                            qw.idx_k_norm.ptr.cast::<f32>(),
                            eps,
                            rope_base,
                            mrope_opt,
                            qs.tail.ptr.cast::<f32>(),
                            qs.dead.ptr.cast::<f32>(),
                            qs.pooled.ptr.cast::<f32>(),
                            state.max_ctx as i32,
                            idx_s,
                        );

                        let (ids_ptr, max_n_ids, id_cap) = if sparse {
                            qwen4::idx_score_select(
                                qs.pooled.ptr.cast::<f32>(),
                                qs.dead.ptr.cast::<f32>(),
                                q_idx_p,
                                pos_dev,
                                t as i32,
                                bucket_qsa_blocks as i32,
                                cfg.idx_top_k as i32,
                                g.ptr::<f32>(g.off_idx_scores),
                                g.ptr::<i32>(g.off_idx_ids),
                                widths_dev,
                                idx_s,
                            );
                            (g.ptr::<i32>(g.off_idx_ids).cast_const(), cfg.idx_top_k, cfg.idx_top_k)
                        } else {
                            qwen4::fill_dense_widths(pos_dev, widths_dev, t as i32, idx_s);
                            (std::ptr::null(), bucket_dense_n, bucket_dense_n)
                        };
                        if !prof_on {
                            g.side_join_ev.record(&g.side_stream)?;
                            g.stream.wait_event(&g.side_join_ev)?;
                        }

                        qwen4::qsa_decode_attn_q8(
                            q_p,
                            qfull_p,
                            qs.k_q.ptr.cast::<i8>(),
                            qs.v_q.ptr.cast::<i8>(),
                            qs.k_sc.ptr.cast::<u16>(),
                            qs.v_sc.ptr.cast::<u16>(),
                            ids_ptr,
                            widths_dev,
                            max_n_ids as i32,
                            id_cap as i32,
                            t as i32,
                            qsa_nkv as i32,
                            qsa_scale,
                            g.ptr::<f32>(g.off_qsa_ws),
                            std::ptr::null_mut(),
                            attn_q8_p,
                            &g.stream,
                        );
                        linear_q8(qw.attn_output.as_spec(), attn_q8_p, std::ptr::null(), block_out, t, &g.stream);
                    }
                    if let Some(ts) = t_qsa {
                        g.stream.sync()?;
                        if let Some(ref mut p) = prof_opt {
                            p.qsa_ms += ts.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
                _ => anyhow::bail!("layer {il} attn weight/state mismatch"),
            }

            // Attn hc_write + FFN hc_read
            {
                let t_hc = if prof_on { Some(Instant::now()) } else { None };
                unsafe {
                    qwen4::hc_write_strided(
                        r_buf,
                        block_out,
                        std::ptr::null(),
                        std::ptr::null(),
                        inject,
                        hc as i32,
                        t as i32,
                        hc as i32,
                        n_embd as i32,
                        &g.stream,
                    );
                    hc_read_gpu(
                        &lw.hc_ffn,
                        r_buf,
                        t,
                        hc,
                        n_embd,
                        hc_lr,
                        eps,
                        g.ptr::<f32>(g.off_xn),
                        g.ptr::<c_void>(g.off_xn_q8),
                        g.ptr::<f32>(g.off_hc_lo),
                        g.ptr::<c_void>(g.off_hc_lo_q8),
                        g.ptr::<f32>(g.off_hc_up),
                        mixed,
                        mixed_q8,
                        inject,
                        true,
                        &g.stream,
                    );
                }
                if let Some(ts) = t_hc {
                    g.stream.sync()?;
                    if let Some(ref mut p) = prof_opt {
                        p.hc_ms += ts.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            // Router + Doorbell launch + Shared Expert + Deterministic(t==1)/Deduplicated(t>=2) MoE + Doorbell wait HC write
            let mw = &lw.moe;
            let use_pcie_fetch = t >= 2 && es.all_pinned_cold && pcie_div > 0 && pcie_max > 0;
            let fetch_cap = if use_pcie_fetch {
                pcie_max.min(QWEN4_DMA_BATCH as i32).min(8)
            } else {
                0
            };
            {
                let t_rsh = if prof_on { Some(Instant::now()) } else { None };
                let pair_out_p = g.ptr::<f32>(g.off_moe_pair_out);
                let gpu_pair_mask_p = g.ptr::<u16>(g.off_gpu_pair_mask);
                unsafe {
                    let rlogits_p = g.ptr::<f32>(g.off_router_logits);
                    let rids_p = g.ptr::<i32>(g.off_router_ids);
                    let rw_p = g.ptr::<f32>(g.off_router_w);
                    let plan_p = g.ptr::<u64>(g.off_plan_ptrs_dev);
                    let seq_state_p = g.ptr::<u64>(g.off_db_seq_state);
                    let exp_seq_p = g.ptr::<u64>(g.off_db_exp_seq);
                    let ram_p = if es.all_pinned_cold {
                        es.dev_ram_ptrs.ptr.cast::<u64>()
                    } else {
                        std::ptr::null()
                    };
                    let res_ptrs_p = g.ptr::<u64>(g.off_small_ptrs);
                    let res_off_p = g.ptr::<i32>(g.off_small_off);
                    let res_nt_p = g.ptr::<i32>(g.off_small_nt);
                    let res_rows_p = g.ptr::<i32>(g.off_small_rows);
                    let res_pair_p = g.ptr::<i32>(g.off_small_pair);
                    let res_wts_p = g.ptr::<f32>(g.off_small_wts);
                    let fetch_src_p = g.ptr::<u64>(g.off_fetch_src_ptrs);
                    let fetch_dst_p = g.ptr::<u64>(g.off_fetch_dst_ptrs);
                    let fetch_off_p = g.ptr::<i32>(g.off_fetch_off);
                    let fetch_nt_p = g.ptr::<i32>(g.off_fetch_nt);
                    let fetch_rows_p = g.ptr::<i32>(g.off_fetch_rows);
                    let fetch_pair_p = g.ptr::<i32>(g.off_fetch_pair);
                    let fetch_wts_p = g.ptr::<f32>(g.off_fetch_wts);

                    hy4::gemv_f32(
                        mw.gate_inp.ptr.cast::<f32>(),
                        n_exp as i32,
                        n_embd as i32,
                        mixed,
                        t as i32,
                        rlogits_p,
                        1.0,
                        &g.stream,
                    );
                    if t == 1 {
                        let (t1_div, t1_max) =
                            if es.all_pinned_cold && pcie_div_t1 > 0 && pcie_max > 0 {
                                (pcie_div_t1, pcie_max)
                            } else {
                                (0, 0)
                            };
                        qwen4::router_doorbell_plan(
                            rlogits_p,
                            t as i32,
                            n_exp as i32,
                            topk as i32,
                            es.dev_vram_ptrs.ptr.cast::<u64>(),
                            ram_p,
                            t1_div,
                            t1_max,
                            mixed,
                            n_embd as i32,
                            n_ff as i32,
                            il as i32,
                            mw.gu_type as i32,
                            mw.d_type as i32,
                            rids_p,
                            rw_p,
                            plan_p,
                            db_dev as *mut c_void,
                            seq_state_p,
                            exp_seq_p,
                            &g.stream,
                        );
                    } else {
                        qwen4::router_doorbell_plan_dedup(
                            rlogits_p,
                            t as i32,
                            n_exp as i32,
                            topk as i32,
                            es.dev_vram_ptrs.ptr.cast::<u64>(),
                            ram_p,
                            if use_pcie_fetch { pcie_div } else { 0 },
                            fetch_cap,
                            g.dma_stage.as_u64(),
                            g.slot_stride,
                            mixed,
                            n_embd as i32,
                            n_ff as i32,
                            il as i32,
                            mw.gu_type as i32,
                            mw.d_type as i32,
                            rids_p,
                            rw_p,
                            res_ptrs_p,
                            res_off_p,
                            res_nt_p,
                            res_rows_p,
                            res_pair_p,
                            res_wts_p,
                            fetch_src_p,
                            fetch_dst_p,
                            fetch_off_p,
                            fetch_nt_p,
                            fetch_rows_p,
                            fetch_pair_p,
                            fetch_wts_p,
                            gpu_pair_mask_p,
                            db_dev as *mut c_void,
                            seq_state_p,
                            exp_seq_p,
                            &g.stream,
                        );
                        if use_pcie_fetch {
                            g.fetch_start_ev.record(&g.stream)?;
                            g.fetch_stream.wait_event(&g.fetch_start_ev)?;
                            qwen4::moe_fetch_unique(
                                fetch_src_p,
                                fetch_dst_p,
                                mw.blob_bytes,
                                fetch_cap,
                                &g.fetch_stream,
                            );
                            g.fetch_done_ev.record(&g.fetch_stream)?;
                        }
                    }

                    let sh_s = if !prof_on {
                        g.side_fork_ev.record(&g.stream)?;
                        g.side_stream.wait_event(&g.side_fork_ev)?;
                        &g.side_stream
                    } else {
                        &g.stream
                    };
                    let sh_g = g.ptr::<f32>(g.off_sh_gate);
                    let sh_u = g.ptr::<f32>(g.off_sh_up);
                    let sh_hq = g.ptr::<c_void>(g.off_sh_hq);
                    let sh_out = g.ptr::<f32>(g.off_sh_out);
                    let sh_gl = g.ptr::<f32>(g.off_sh_gate_logit);

                    if t <= 8
                        && mw.sh_gate.ggml_type == GgmlType::Q8_0
                        && mw.sh_up.ggml_type == GgmlType::Q8_0
                    {
                        qwen4::sh_gu_swiglu_q8(
                            mw.sh_gate.buf.ptr,
                            mw.sh_up.buf.ptr,
                            mixed_q8,
                            sh_g,
                            n_ff_sh as i32,
                            n_embd as i32,
                            t as i32,
                            sh_s,
                        );
                        launch_quantize_q8_1(sh_g, sh_hq, t as i32, n_ff_sh as i32, sh_s.raw);
                    } else {
                        linear_q8(mw.sh_gate.as_spec(), mixed_q8, mixed, sh_g, t, sh_s);
                        linear_q8(mw.sh_up.as_spec(), mixed_q8, mixed, sh_u, t, sh_s);
                        glm5::swiglu_clamp_q8(sh_g, sh_u, 0.0, n_ff_sh as i32, std::ptr::null_mut(), sh_hq, t as i32, sh_s);
                    }
                    linear_q8(mw.sh_down.as_spec(), sh_hq, std::ptr::null(), sh_out, t, sh_s);
                    qwen4::shared_gate_dot(
                        mixed,
                        mw.gate_inp_shexp.ptr,
                        GgmlType::F32 as i32,
                        t as i32,
                        n_embd as i32,
                        sh_gl,
                        sh_s,
                    );

                    if let Some(ts) = t_rsh {
                        g.stream.sync()?;
                        if let Some(ref mut p) = prof_opt {
                            p.moe_router_sh_ms += ts.elapsed().as_secs_f64() * 1000.0;
                        }
                    }

                    let t_moe = if prof_on { Some(Instant::now()) } else { None };
                    let moe_inter_p = g.ptr::<c_void>(g.off_moe_inter_q8);
                    if t == 1 {
                        glm5::moe_plan_filtered(
                            mw.gu_type as i32,
                            mw.d_type as i32,
                            plan_p,
                            mw.u_off,
                            mw.d_off,
                            mw.gu_row_bytes,
                            mw.d_row_bytes,
                            mixed_q8,
                            rw_p,
                            moe_inter_p,
                            0,
                            std::ptr::null(),
                            0,
                            std::ptr::null(),
                            0,
                            std::ptr::null_mut(),
                            block_out,
                            n_embd as i32,
                            n_ff as i32,
                            topk as i32,
                            t as i32,
                            0.0,
                            std::ptr::null(),
                            std::ptr::null(),
                            0,
                            &g.stream,
                        );
                    } else {
                        let n_jobs_res = (t * topk) as i32;
                        glm5::moe_prefill_small_gu(
                            mw.gu_type as i32,
                            res_ptrs_p,
                            res_off_p,
                            res_nt_p,
                            res_rows_p,
                            mw.u_off,
                            mw.gu_row_bytes,
                            mixed_q8,
                            moe_inter_p,
                            n_embd as i32,
                            n_ff as i32,
                            n_jobs_res,
                            0.0,
                            &g.stream,
                        );
                        glm5::moe_prefill_small_down_det(
                            mw.d_type as i32,
                            res_ptrs_p,
                            res_off_p,
                            res_nt_p,
                            res_pair_p,
                            res_wts_p,
                            mw.d_off,
                            mw.d_row_bytes,
                            moe_inter_p,
                            pair_out_p,
                            n_embd as i32,
                            n_ff as i32,
                            n_jobs_res,
                            &g.stream,
                        );
                        if use_pcie_fetch {
                            g.stream.wait_event(&g.fetch_done_ev)?;
                            glm5::moe_prefill_small_gu(
                                mw.gu_type as i32,
                                fetch_dst_p,
                                fetch_off_p,
                                fetch_nt_p,
                                fetch_rows_p,
                                mw.u_off,
                                mw.gu_row_bytes,
                                mixed_q8,
                                moe_inter_p,
                                n_embd as i32,
                                n_ff as i32,
                                fetch_cap,
                                0.0,
                                &g.stream,
                            );
                            glm5::moe_prefill_small_down_det(
                                mw.d_type as i32,
                                fetch_dst_p,
                                fetch_off_p,
                                fetch_nt_p,
                                fetch_pair_p,
                                fetch_wts_p,
                                mw.d_off,
                                mw.d_row_bytes,
                                moe_inter_p,
                                pair_out_p,
                                n_embd as i32,
                                n_ff as i32,
                                fetch_cap,
                                &g.stream,
                            );
                        }
                    }
                    qwen4::doorbell_wait(db_dev as *const c_void, exp_seq_p, &g.stream);
                    if !prof_on {
                        g.side_join_ev.record(&g.side_stream)?;
                        g.stream.wait_event(&g.side_join_ev)?;
                    }
                    if t == 1 {
                        qwen4::moe_reduce_hc_write(
                            db_dev as *const c_void,
                            exp_seq_p,
                            r_buf,
                            block_out,
                            std::ptr::null(),
                            std::ptr::null(),
                            0,
                            sh_out,
                            sh_gl,
                            inject,
                            hc as i32,
                            t as i32,
                            hc as i32,
                            n_embd as i32,
                            &g.stream,
                        );
                    } else {
                        qwen4::moe_reduce_hc_write(
                            db_dev as *const c_void,
                            exp_seq_p,
                            r_buf,
                            std::ptr::null(),
                            pair_out_p,
                            gpu_pair_mask_p,
                            topk as i32,
                            sh_out,
                            sh_gl,
                            inject,
                            hc as i32,
                            t as i32,
                            hc as i32,
                            n_embd as i32,
                            &g.stream,
                        );
                    }
                    if let Some(ts) = t_moe {
                        g.stream.sync()?;
                        if let Some(ref mut p) = prof_opt {
                            p.moe_routed_ms += ts.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
            }
        }

        if is_head_gpu {
            let t_head = if prof_on { Some(Instant::now()) } else { None };
            unsafe {
                emit_head_logits_and_argmax(
                    weights,
                    g,
                    state.r_last.ptr,
                    t,
                    t,
                    t,
                    greedy_argmax,
                )?;
                if greedy_argmax {
                    qwen4::copy_u32_sm(
                        g.argmax_out_dev as *mut c_void,
                        g.ptr::<c_void>(g.off_argmax_out),
                        t as i32,
                        true,
                        &g.stream,
                    );
                }
            }
            if let Some(ts) = t_head {
                g.stream.sync()?;
                if let Some(ref mut p) = prof_opt {
                    p.head_ms += ts.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }
        Ok(())
    };

    let db_guard = state.doorbell.active_guard();
    let eff_pcie_div = if t == 1 { pcie_div_t1 } else { pcie_div };
    let gkey = (t, sparse, greedy_argmax, save_snap, eff_pcie_div, pcie_max);

    // GPU 0 execution / graph launch
    {
        let g0 = &state.gpus[0];
        set_device(g0.device)?;
        if can_graph {
            if !state.trunk_graphs[0].contains_key(&gkey) {
                let graph = CudaGraphExec::capture(&g0.stream, || {
                    let mut no_prof = None;
                    emit_gpu_half(
                        0,
                        g0,
                        0..split,
                        &state.layers[0..split],
                        state.ple.as_ref(),
                        true,
                        n_gpus == 1,
                        &mut no_prof,
                    )
                })?;
                state.trunk_graphs[0].insert(gkey, graph);
            }
            state.trunk_graphs[0][&gkey].launch(&g0.stream)?;
        } else {
            emit_gpu_half(
                0,
                g0,
                0..split,
                &state.layers[0..split],
                state.ple.as_ref(),
                true,
                n_gpus == 1,
                &mut profile,
            )?;
        }
    }

    // P2P + GPU 1 execution / graph launch
    if n_gpus >= 2 {
        let g0 = &state.gpus[0];
        let g1 = &state.gpus[1];
        let t_p2p = if prof_on { Some(Instant::now()) } else { None };
        set_device(g0.device)?;
        g0.p2p_done.record(&g0.stream)?;
        set_device(g1.device)?;
        g1.stream.wait_event(&g0.p2p_done)?;
        unsafe {
            copy_peer_raw_async(
                g1.ptr::<c_void>(g1.off_r_buf),
                g1.device,
                g0.ptr::<c_void>(g0.off_r_buf),
                g0.device,
                t * hc_dim * 4,
                &g1.stream,
            )?;
        }
        if let Some(ts) = t_p2p {
            g1.stream.sync()?;
            if let Some(ref mut p) = profile {
                p.p2p_ms += ts.elapsed().as_secs_f64() * 1000.0;
            }
        }

        if can_graph {
            if !state.trunk_graphs[1].contains_key(&gkey) {
                let graph = CudaGraphExec::capture(&g1.stream, || {
                    let mut no_prof = None;
                    emit_gpu_half(
                        1,
                        g1,
                        split..n_layer,
                        &state.layers[split..n_layer],
                        None,
                        false,
                        true,
                        &mut no_prof,
                    )
                })?;
                state.trunk_graphs[1].insert(gkey, graph);
            }
            state.trunk_graphs[1][&gkey].launch(&g1.stream)?;
        } else {
            emit_gpu_half(
                1,
                g1,
                split..n_layer,
                &state.layers[split..n_layer],
                None,
                false,
                true,
                &mut profile,
            )?;
        }
    }

    let hg = state.gpus.last().unwrap();
    set_device(hg.device)?;
    hg.stream.sync()?;
    let mut out_toks = Vec::new();
    if greedy_argmax {
        out_toks.extend(hg.argmax_out_pin.as_slice()[..t].iter().map(|&v| v as u32));
    }
    drop(db_guard);

    // Record expert heat + stats from mapped doorbell trace (after stream.sync & doorbell completion)
    {
        let db0 = &state.doorbell.db0.as_slice()[0];
        for il in 0..split {
            let np = (db0.routed_n_pairs[il] as usize).min(80);
            let es = &mut weights.experts[il];
            for k in 0..np {
                let eid = db0.routed_ids[il][k];
                if eid >= 0 && (eid as usize) < n_exp {
                    let u = eid as usize;
                    es.heat[u] += 1.0;
                    es.decode_usage[u] += 1.0;
                    es.last_used[u] = step_id;
                    if es.is_resident[u] {
                        state.cache_stats.hits += 1;
                        state.decode_cache_stats.hits += 1;
                    } else {
                        state.cache_stats.misses += 1;
                        state.decode_cache_stats.misses += 1;
                    }
                }
            }
        }
        if n_gpus >= 2 {
            if let Some(ref db1_buf) = state.doorbell.db1 {
                let db1 = &db1_buf.as_slice()[0];
                for il in split..n_layer {
                    let np = (db1.routed_n_pairs[il] as usize).min(80);
                    let es = &mut weights.experts[il];
                    for k in 0..np {
                        let eid = db1.routed_ids[il][k];
                        if eid >= 0 && (eid as usize) < n_exp {
                            let u = eid as usize;
                            es.heat[u] += 1.0;
                            es.decode_usage[u] += 1.0;
                            es.last_used[u] = step_id;
                            if es.is_resident[u] {
                                state.cache_stats.hits += 1;
                                state.decode_cache_stats.hits += 1;
                            } else {
                                state.cache_stats.misses += 1;
                                state.decode_cache_stats.misses += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(ref mut p) = profile {
        p.steps += 1;
        p.tokens += t;
    }

    Ok(out_toks)
}

/// Restore PLE history/cursor, GDN states/conv_hist, and QSA indexer state after speculative rejection.
pub fn rollback_verify(
    weights: &Qwen4Weights,
    state: &mut Qwen4State,
    accepted_tokens: &[u32],
    accept_count: usize,
    pos_before: usize,
) -> Result<()> {
    let cfg = &weights.cfg;
    let hc_dim = cfg.hc * cfg.n_embd;

    if let Some(mut cursor) = state.ple_cursor_snap {
        for &tok in accepted_tokens.iter().take(accept_count + 1) {
            cursor.prev[0] = cursor.prev[1];
            cursor.prev[1] = tok as i32;
        }
        state.ple_cursor = cursor;
    }

    let n_acc = accept_count + 1;
    let _ = pos_before;
    let ch_bytes = cfg.gdn_qkv_dim() * 3 * 4;
    let n_layer = cfg.n_layer;
    let split = weights.split_layer.min(n_layer);
    let n_gpus = state.gpus.len();
    let can_graph = !state.mrope_active && state.use_cuda_graph;
    let max_ctx = state.max_ctx;

    for gidx in 0..n_gpus {
        let layer_range = if gidx == 0 { 0..split } else { split..n_layer };
        let g = &state.gpus[gidx];
        let layers_slice = &state.layers[layer_range.clone()];
        let ple_ref = if gidx == 0 { state.ple.as_ref() } else { None };
        let mrope_opt = if state.mrope_active {
            g.ptr::<i32>(g.off_mrope_pos3_dev).cast_const()
        } else {
            std::ptr::null()
        };
        set_device(g.device)?;

        let emit_rollback_gpu = || -> Result<()> {
            if let Some(ple_st) = ple_ref {
                if let Some(ref snap) = ple_st.snap {
                    let hist_bytes = hc_dim * 9 * 4;
                    unsafe {
                        copy_d2d_raw_async(
                            ple_st.history.ptr,
                            snap.ptr.add(accept_count * hist_bytes),
                            hist_bytes,
                            &g.stream,
                        )?;
                    }
                }
            }
            for (loc_il, il) in layer_range.clone().enumerate() {
                let lw = &weights.layers[il];
                match (&lw.attn, &layers_slice[loc_il]) {
                    (AttnLayerWeights::Gdn(_), Qwen4LayerState::Gdn(gs)) => {
                        if let (
                            Some(ref ss),
                            Some(ref hs),
                            Some(ref hq),
                            Some(ref gt),
                            Some(ref bt),
                        ) = (
                            &gs.state_snap,
                            &gs.hist_snap,
                            &gs.h_qkv_snap,
                            &gs.gate_snap,
                            &gs.beta_snap,
                        ) {
                            unsafe {
                                qwen4::gdn_replay_steps(
                                    ss.ptr.cast::<f32>(),
                                    gs.state.ptr.cast::<f32>(),
                                    hq.ptr.cast::<f32>(),
                                    gt.ptr.cast::<f32>(),
                                    bt.ptr.cast::<f32>(),
                                    n_acc as i32,
                                    cfg.ssm_group_count as i32,
                                    cfg.ssm_time_step_rank as i32,
                                    &g.stream,
                                );
                                copy_d2d_raw_async(
                                    gs.conv_hist.ptr,
                                    hs.ptr.add(accept_count * ch_bytes),
                                    ch_bytes,
                                    &g.stream,
                                )?;
                            }
                        }
                    }
                    (AttnLayerWeights::Qsa(qw), Qwen4LayerState::Qsa(qs)) => {
                        if let (Some(ref ts), Some(ref ds), Some(ref ks)) =
                            (&qs.tail_snap, &qs.dead_snap, &qs.k_idx_snap)
                        {
                            unsafe {
                                copy_d2d_raw_async(
                                    qs.tail.ptr,
                                    ts.ptr,
                                    4 * cfg.idx_key_dim * 4,
                                    &g.stream,
                                )?;
                                copy_d2d_raw_async(
                                    qs.dead.ptr,
                                    ds.ptr,
                                    cfg.idx_key_dim * 4,
                                    &g.stream,
                                )?;
                                qwen4::idx_append_steps(
                                    ks.ptr.cast::<f32>(),
                                    g.ptr::<i32>(g.off_pos_dev),
                                    n_acc as i32,
                                    qw.idx_k_norm.ptr.cast::<f32>(),
                                    cfg.rms_norm_eps,
                                    cfg.rope_base,
                                    mrope_opt,
                                    qs.tail.ptr.cast::<f32>(),
                                    qs.dead.ptr.cast::<f32>(),
                                    qs.pooled.ptr.cast::<f32>(),
                                    max_ctx as i32,
                                    &g.stream,
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        };

        if can_graph {
            if !state.rollback_graphs[gidx].contains_key(&accept_count) {
                let graph = CudaGraphExec::capture(&g.stream, emit_rollback_gpu)?;
                state.rollback_graphs[gidx].insert(accept_count, graph);
            }
            state.rollback_graphs[gidx][&accept_count].launch(&g.stream)?;
        } else {
            emit_rollback_gpu()?;
        }
    }

    Ok(())
}
