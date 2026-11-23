use crate::hy4::engram::gather_engram_window;
use crate::hy4::state::{
    hy4_profile_enabled, Hy4GpuScratch, Hy4LayerCache, Hy4ProfileStats, Hy4State, HY4_ATTN_SUB,
    HY4_DMA_GROUP_CAP, HY4_DMA_RING_GROUPS, HY4_K_SUB, HY4_META_MAX_EXPS, HY4_META_MAX_TILES,
    HY4_META_SLOTS, HY4_PREFILL_GRP_WIN, HY4_SMALL_MOE_CAP, HY4_W_TOKEN,
};
use crate::hy4::weights::{GpuMat, Hy4LayerWeights, Hy4ModelWeights};
use ampllm_cpu_kernels::{
    quantize_glm5_cpu_x_q8k, run_glm5_cpu_moe_grouped_prequant, run_glm5_cpu_moe_m1,
    tok_row_to_f32_cpu, CpuThreadPool, Glm5CpuExpertJob, Glm5CpuGroupedExpert, Glm5CpuScratch,
};
use ampllm_cuda_kernels::ffi::{launch_linear_q8_1, launch_quantize_q8_1};
use ampllm_cuda_kernels::{
    copy_d2h_raw_async, copy_h2d_raw_async, copy_peer_raw_async, glm5, hy4, set_device,
};
use ampllm_gguf::GgmlType;
use anyhow::{bail, Result};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Instant;

/// Image token embedding span overlapping prompt token positions `start_pos..start_pos + rows/E`.
#[derive(Debug, Clone)]
pub struct Hy4ImageSpan {
    pub start_pos: usize,
    pub rows: Vec<f32>,
}

#[inline]
pub fn hy4_dump_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("AMPLLM_HY4_DUMP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub unsafe fn dbg_sum(
    sc: &Hy4GpuScratch,
    name: &str,
    il: isize,
    ptr: *const f32,
    rows: usize,
    ld: usize,
    width: usize,
    off: usize,
) {
    if !hy4_dump_enabled() || rows == 0 {
        return;
    }
    let w = if width == 0 { ld } else { width };
    sc.stream.sync().expect("sync before dump");
    let mut host = vec![0.0f32; rows * ld];
    copy_d2h_raw_async(bytemuck::cast_slice_mut(&mut host), ptr.cast(), &sc.stream)
        .expect("dump d2h");
    sc.stream.sync().expect("dump d2h sync");
    let mut s = 0.0f64;
    for r in 0..rows {
        for c in 0..w {
            s += host[r * ld + off + c] as f64;
        }
    }
    let label = format!("{name}-{il}");
    eprintln!("DUMP {label:<24} sum = {s:.6}");
}

#[inline]
fn mmq_ok(ty: GgmlType, k: usize) -> bool {
    let aligned_k = match ty {
        GgmlType::Q8_0 => k % 128 == 0,
        GgmlType::Q4K
        | GgmlType::Q5K
        | GgmlType::Q6K
        | GgmlType::Iq4Xs
        | GgmlType::Iq2S
        | GgmlType::Iq3Xxs
        | GgmlType::Iq3S => k % 256 == 0,
        _ => false,
    };
    aligned_k
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn mm_hy4(
    mat: &GpuMat,
    x_f32: *const f32,
    x_q8_opt: *const c_void,
    t: usize,
    y_f32: *mut f32,
    alpha: f32,
    sc: &Hy4GpuScratch,
) {
    if t == 0 || mat.n == 0 {
        return;
    }
    let stream = &sc.stream;
    match mat.ggml_type {
        GgmlType::F32 => {
            if t <= HY4_W_TOKEN {
                hy4::gemv_f32(
                    mat.ptr().cast(),
                    mat.n as i32,
                    mat.k as i32,
                    x_f32,
                    t as i32,
                    y_f32,
                    alpha,
                    stream,
                );
            } else {
                glm5::f32_to_f16(x_f32, sc.x_f16_tmp, (t * mat.k) as i32, stream);
                glm5::f32_to_f16(mat.ptr().cast(), sc.w_f16_tmp, (mat.n * mat.k) as i32, stream);
                glm5::gemm_f16_mma_nt(
                    sc.x_f16_tmp,
                    sc.w_f16_tmp,
                    y_f32,
                    t as i32,
                    mat.n as i32,
                    mat.k as i32,
                    alpha,
                    0.0,
                    stream,
                );
            }
        }
        GgmlType::Bf16 => {
            if t <= HY4_W_TOKEN {
                hy4::gemv_bf16(
                    mat.ptr().cast(),
                    mat.n as i32,
                    mat.k as i32,
                    x_f32,
                    t as i32,
                    y_f32,
                    alpha,
                    stream,
                );
            } else {
                glm5::f32_to_f16(x_f32, sc.x_f16_tmp, (t * mat.k) as i32, stream);
                glm5::dequant_to_f16(
                    GgmlType::Bf16 as i32,
                    mat.ptr(),
                    sc.w_f16_tmp,
                    (mat.n * mat.k) as i64,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    sc.x_f16_tmp,
                    sc.w_f16_tmp,
                    y_f32,
                    t as i32,
                    mat.n as i32,
                    mat.k as i32,
                    alpha,
                    0.0,
                    stream,
                );
            }
        }
        _ => {
            let xq = if !x_q8_opt.is_null() {
                x_q8_opt
            } else {
                launch_quantize_q8_1(x_f32, sc.x_q8, t as i32, mat.k as i32, stream.raw);
                sc.x_q8 as *const c_void
            };
            if t <= HY4_W_TOKEN || !mmq_ok(mat.ggml_type, mat.k) {
                launch_linear_q8_1(
                    mat.ggml_type as i32,
                    mat.ptr(),
                    xq,
                    x_f32,
                    y_f32,
                    mat.n as i32,
                    mat.k as i32,
                    t as i32,
                    stream.raw,
                );
                if (alpha - 1.0).abs() > 1e-7 {
                    glm5::scale_f32(y_f32, alpha, (t * mat.n) as i32, stream);
                }
            } else {
                glm5::gemm_mmq_mma_tiled(
                    mat.ggml_type as i32,
                    xq,
                    mat.ptr(),
                    mat.row_bytes,
                    y_f32,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    t as i32,
                    mat.n as i32,
                    mat.k as i32,
                    alpha,
                    0.0,
                    0,
                    stream,
                );
            }
        }
    }
}

pub fn decode_tokens_to_host(
    weights: &Hy4ModelWeights,
    ids: &[u32],
    p0: usize,
    imgs: &[Hy4ImageSpan],
    dst: &mut [f32],
) -> Result<()> {
    let e = weights.cfg.n_embd;
    let nv = weights.cfg.n_vocab;
    for (t, &id) in ids.iter().enumerate() {
        if (id as usize) >= nv {
            bail!("token id {id} outside vocab {nv}");
        }
        let src = unsafe {
            std::slice::from_raw_parts(
                weights
                    .tok_embd_ptr
                    .add((id as usize) * weights.tok_row_bytes),
                weights.tok_row_bytes,
            )
        };
        tok_row_to_f32_cpu(weights.tok_type, src, &mut dst[t * e..(t + 1) * e]);
    }
    let t_len = ids.len();
    for im in imgs {
        let im_toks = im.rows.len() / e;
        let a = p0.max(im.start_pos);
        let b = (p0 + t_len).min(im.start_pos + im_toks);
        if a < b {
            let count = (b - a) * e;
            let dst_off = (a - p0) * e;
            let src_off = (a - im.start_pos) * e;
            dst[dst_off..dst_off + count].copy_from_slice(&im.rows[src_off..src_off + count]);
        }
    }
    Ok(())
}

/// Core attention block shared by trunk layers and MTP `blk.42`.
/// Reads `sc.xa[tok_off..tok_off+t_len, E]`, writes `sc.y[tok_off..tok_off+t_len, E]` (unless `cache_only`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_core_layer(
    weights: &Hy4ModelWeights,
    lw: &Hy4LayerWeights,
    cache: &Hy4LayerCache,
    sc: &Hy4GpuScratch,
    t_len: usize,
    tok_off: usize,
    p0: usize,
    cache_only: bool,
) -> Result<()> {
    let cfg = &weights.cfg;
    let e = cfg.n_embd;
    let xa_p = sc.xa.add(tok_off * e);
    let y_p = sc.y.add(tok_off * e);
    let pos_p = sc.pos.add(tok_off);
    let lo_p = sc.lo.add(tok_off);
    let nk_p = sc.nk.add(tok_off);
    let h = lw.heads;
    let kl = cfg.kv_lora_rank;
    let r = cfg.rope_dim;
    let kr = cfg.kv_row();
    let nope = cfg.nope();
    let eps = cfg.rms_norm_eps;
    let stream = &sc.stream;
    let kv_ring_cap = if lw.swa {
        weights.swa_ring_cap as i32
    } else {
        0
    };
    let ivg_ring_cap = weights.ivg_ring_cap as i32;
    let kv_is_q8 = weights.kv_is_q8;
    let kv_rb = weights.kv_rb as i32;

    launch_quantize_q8_1(xa_p, sc.x_q8, t_len as i32, e as i32, stream.raw);

    if cache_only {
        mm_hy4(&lw.kv_a, xa_p, sc.x_q8, t_len, sc.kvr, 1.0, sc);
        hy4::kv_store(
            sc.kvr,
            t_len as i32,
            kl as i32,
            r as i32,
            lw.kv_a_norm.ptr.cast(),
            eps,
            pos_p,
            lw.rope_base,
            cache.kv.ptr,
            kv_ring_cap,
            kv_is_q8,
            kv_rb,
            stream,
        );
        if let Some(ref ixw) = lw.indexer {
            let d = cfg.ix_dim;
            let kp = cfg.ix_kpool;
            mm_hy4(&ixw.k, xa_p, sc.x_q8, t_len, sc.ik, 1.0, sc);
            hy4::rmsnorm_strided(
                sc.ik,
                d,
                ixw.k_norm.ptr.cast(),
                sc.ik,
                d,
                d as i32,
                t_len as i32,
                eps,
                stream,
            );
            mm_hy4(&ixw.vg, sc.ik, std::ptr::null(), t_len, sc.ivt, 1.0, sc);
            let ivg_ptr = cache.ivg.as_ref().unwrap().ptr;
            hy4::store_rows(
                sc.ivt,
                t_len as i32,
                (2 * d) as i32,
                ivg_ptr,
                ivg_ring_cap,
                pos_p,
                kv_is_q8,
                stream,
            );
            let last = p0 + t_len - 1;
            let g0 = p0 / kp;
            let g1 = (last + 1) / kp;
            if g1 > g0 {
                let pooled_ptr = cache.pooled.as_ref().unwrap().ptr;
                hy4::ix_pool(
                    ivg_ptr,
                    ivg_ring_cap,
                    g0 as i32,
                    (g1 - g0) as i32,
                    kp as i32,
                    d as i32,
                    ixw.ape.ptr.cast(),
                    ixw.pnorm.ptr.cast(),
                    eps,
                    r as i32,
                    lw.rope_base,
                    pooled_ptr,
                    kv_is_q8,
                    stream,
                );
            }
        }
        return Ok(());
    }

    mm_hy4(&lw.q_a, xa_p, sc.x_q8, t_len, sc.qa, 1.0, sc);
    mm_hy4(&lw.kv_a, xa_p, sc.x_q8, t_len, sc.kvr, 1.0, sc);
    mm_hy4(&lw.gate, xa_p, sc.x_q8, t_len, sc.g, 1.0, sc);
    if let Some(ref ixw) = lw.indexer {
        mm_hy4(&ixw.k, xa_p, sc.x_q8, t_len, sc.ik, 1.0, sc);
    }

    hy4::rmsnorm_strided(
        sc.qa,
        cfg.q_lora_rank,
        lw.q_a_norm.ptr.cast(),
        sc.qr,
        cfg.q_lora_rank,
        cfg.q_lora_rank as i32,
        t_len as i32,
        eps,
        stream,
    );
    dbg_sum(sc, "q_resid", lw.il as isize, sc.qr, t_len, cfg.q_lora_rank, cfg.q_lora_rank, 0);

    launch_quantize_q8_1(
        sc.qr,
        sc.x_q8,
        t_len as i32,
        cfg.q_lora_rank as i32,
        stream.raw,
    );
    mm_hy4(&lw.q_b, sc.qr, sc.x_q8, t_len, sc.q, 1.0, sc);

    hy4::kv_store(
        sc.kvr,
        t_len as i32,
        kl as i32,
        r as i32,
        lw.kv_a_norm.ptr.cast(),
        eps,
        pos_p,
        lw.rope_base,
        cache.kv.ptr,
        kv_ring_cap,
        kv_is_q8,
        kv_rb,
        stream,
    );

    hy4::rope(
        sc.q,
        t_len as i32,
        h as i32,
        h * cfg.k_mla,
        cfg.k_mla,
        nope as i32,
        r as i32,
        pos_p,
        lw.rope_base,
        stream,
    );
    dbg_sum(
        sc,
        "q_pe",
        lw.il as isize,
        sc.q,
        t_len * h,
        cfg.k_mla,
        r,
        nope,
    );

    if t_len <= HY4_W_TOKEN {
        hy4::quantize_heads_q8_1(
            sc.q,
            sc.x_q8,
            t_len as i32,
            h as i32,
            nope as i32,
            h * cfg.k_mla,
            cfg.k_mla,
            stream,
        );
        hy4::batched_heads_q8(
            lw.k_b.ptr(),
            sc.x_q8,
            sc.qcat,
            t_len as i32,
            h as i32,
            kl as i32,
            nope as i32,
            h * (nope / 32),
            h * kr,
            kr,
            stream,
        );
    } else {
        hy4::batched_heads_q8_mma(
            lw.k_b.ptr(),
            sc.q,
            sc.qcat,
            t_len as i32,
            h as i32,
            kl as i32,
            nope as i32,
            h * cfg.k_mla,
            cfg.k_mla,
            h * kr,
            kr,
            stream,
        );
    }
    hy4::copy_cols(
        sc.q.add(nope),
        cfg.k_mla,
        sc.qcat.add(kl),
        kr,
        (t_len * h) as i32,
        r as i32,
        stream,
    );

    let attn_scale = 1.0f32 / (cfg.k_mla as f32).sqrt();

    if let Some(ref ixw) = lw.indexer {
        let d = cfg.ix_dim;
        let ih = cfg.ix_heads;
        let kp = cfg.ix_kpool;
        let top = (cfg.ix_top_k / kp) - 1;
        let maxk = cfg.ix_top_k + kp;
        let npool_ld = weights.max_ctx / kp + 1;

        mm_hy4(&ixw.q_b, sc.qr, std::ptr::null(), t_len, sc.iq, 1.0, sc);
        hy4::rope(
            sc.iq,
            t_len as i32,
            ih as i32,
            ih * d,
            d,
            (d - r) as i32,
            r as i32,
            pos_p,
            lw.rope_base,
            stream,
        );
        hy4::rmsnorm_strided(
            sc.ik,
            d,
            ixw.k_norm.ptr.cast(),
            sc.ik,
            d,
            d as i32,
            t_len as i32,
            eps,
            stream,
        );
        dbg_sum(sc, "indexer_k", lw.il as isize, sc.ik, t_len, d, d, 0);

        mm_hy4(&ixw.vg, sc.ik, std::ptr::null(), t_len, sc.ivt, 1.0, sc);
        dbg_sum(sc, "indexer_gate", lw.il as isize, sc.ivt, t_len, 2 * d, d, d);
        dbg_sum(sc, "indexer_q", lw.il as isize, sc.iq, t_len, ih * d, ih * d, 0);

        let ivg_ptr = cache.ivg.as_ref().unwrap().ptr;
        let pooled_ptr = cache.pooled.as_ref().unwrap().ptr;
        hy4::store_rows(
            sc.ivt,
            t_len as i32,
            (2 * d) as i32,
            ivg_ptr,
            ivg_ring_cap,
            pos_p,
            kv_is_q8,
            stream,
        );
        let last = p0 + t_len - 1;
        let g0 = p0 / kp;
        let g1 = (last + 1) / kp;
        if g1 > g0 {
            hy4::ix_pool(
                ivg_ptr,
                ivg_ring_cap,
                g0 as i32,
                (g1 - g0) as i32,
                kp as i32,
                d as i32,
                ixw.ape.ptr.cast(),
                ixw.pnorm.ptr.cast(),
                eps,
                r as i32,
                lw.rope_base,
                pooled_ptr,
                kv_is_q8,
                stream,
            );
        }
        mm_hy4(
            &ixw.proj,
            xa_p,
            std::ptr::null(),
            t_len,
            sc.wt,
            1.0f32 / ((d * ih) as f32).sqrt(),
            sc,
        );
        dbg_sum(sc, "indexer_weights", lw.il as isize, sc.wt, t_len, ih, ih, 0);
        if !kv_is_q8 && g1 > 0 {
            dbg_sum(sc, "indexer_pool_k", lw.il as isize, pooled_ptr.cast(), g1, d, d, 0);
        }

        for t0 in (0..t_len).step_by(HY4_K_SUB) {
            let ts = (t_len - t0).min(HY4_K_SUB);
            let n_pool = (p0 + t0 + ts - 1) / kp;
            hy4::ix_score(
                sc.iq.add(t0 * ih * d),
                sc.wt.add(t0 * ih),
                ts as i32,
                ih as i32,
                d as i32,
                pooled_ptr,
                pos_p.add(t0),
                kp as i32,
                n_pool as i32,
                sc.ix_score,
                npool_ld,
                kv_is_q8,
                stream,
            );
            hy4::ix_select(
                sc.ix_score,
                npool_ld,
                ts as i32,
                pos_p.add(t0),
                kp as i32,
                top as i32,
                sc.keys.add(t0 * maxk),
                maxk as i32,
                sc.knk.add(t0),
                stream,
            );
            if t_len <= HY4_W_TOKEN {
                hy4::mla_decode(
                    sc.qcat.add(t0 * h * kr),
                    ts as i32,
                    h as i32,
                    cache.kv.ptr,
                    0,
                    sc.keys.add(t0 * maxk),
                    maxk as i32,
                    std::ptr::null(),
                    sc.knk.add(t0),
                    maxk as i32,
                    attn_scale,
                    lw.sinks.ptr.cast(),
                    sc.mla_ws,
                    sc.o.add(t0 * h * kl),
                    kv_is_q8,
                    kv_rb,
                    stream,
                );
            } else {
                hy4::mla_prefill(
                    sc.qcat.add(t0 * h * kr),
                    ts as i32,
                    h as i32,
                    cache.kv.ptr,
                    0,
                    sc.keys.add(t0 * maxk),
                    maxk as i32,
                    std::ptr::null(),
                    sc.knk.add(t0),
                    attn_scale,
                    lw.sinks.ptr.cast(),
                    sc.o.add(t0 * h * kl),
                    kv_is_q8,
                    kv_rb,
                    stream,
                );
            }
        }
    } else {
        // SWA layer
        for t0 in (0..t_len).step_by(HY4_K_SUB) {
            let ts = (t_len - t0).min(HY4_K_SUB);
            if t_len <= HY4_W_TOKEN {
                hy4::mla_decode(
                    sc.qcat.add(t0 * h * kr),
                    ts as i32,
                    h as i32,
                    cache.kv.ptr,
                    kv_ring_cap,
                    std::ptr::null(),
                    0,
                    lo_p.add(t0),
                    nk_p.add(t0),
                    cfg.n_swa as i32,
                    attn_scale,
                    lw.sinks.ptr.cast(),
                    sc.mla_ws,
                    sc.o.add(t0 * h * kl),
                    kv_is_q8,
                    kv_rb,
                    stream,
                );
            } else {
                hy4::mla_prefill(
                    sc.qcat.add(t0 * h * kr),
                    ts as i32,
                    h as i32,
                    cache.kv.ptr,
                    kv_ring_cap,
                    std::ptr::null(),
                    0,
                    lo_p.add(t0),
                    nk_p.add(t0),
                    attn_scale,
                    lw.sinks.ptr.cast(),
                    sc.o.add(t0 * h * kl),
                    kv_is_q8,
                    kv_rb,
                    stream,
                );
            }
        }
    }

    if t_len <= HY4_W_TOKEN {
        hy4::quantize_heads_q8_1(
            sc.o,
            sc.x_q8,
            t_len as i32,
            h as i32,
            kl as i32,
            h * kl,
            kl,
            stream,
        );
        hy4::batched_heads_q8(
            lw.v_b.ptr(),
            sc.x_q8,
            sc.ob,
            t_len as i32,
            h as i32,
            cfg.v_mla as i32,
            kl as i32,
            h * (kl / 32),
            h * cfg.v_mla,
            cfg.v_mla,
            stream,
        );
    } else {
        hy4::batched_heads_q8_mma(
            lw.v_b.ptr(),
            sc.o,
            sc.ob,
            t_len as i32,
            h as i32,
            cfg.v_mla as i32,
            kl as i32,
            h * kl,
            kl,
            h * cfg.v_mla,
            cfg.v_mla,
            stream,
        );
    }
    dbg_sum(sc, "attn_kqv", lw.il as isize, sc.ob, t_len, h * cfg.v_mla, h * cfg.v_mla, 0);

    hy4::sigmoid_mul(sc.ob, sc.g, t_len * h * cfg.v_mla, stream);
    dbg_sum(sc, "attn_gated", lw.il as isize, sc.ob, t_len, h * cfg.v_mla, h * cfg.v_mla, 0);

    mm_hy4(&lw.wo, sc.ob, std::ptr::null(), t_len, y_p, 1.0, sc);
    Ok(())
}

struct ColdPrefillJob {
    e: usize,
    row_off: usize,
    nt: usize,
    tok_indices: Vec<usize>,
    tok_wts: Vec<f32>,
}

unsafe fn run_moe_group_mmq_class(
    sc: &Hy4GpuScratch,
    lw_moe: &crate::hy4::weights::Hy4MoeWeights,
    lat: usize,
    n_ff_exp: usize,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)],
    tile_mode: i32,
    bm: usize,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let stream = &sc.stream;
    let one_meta_bytes = HY4_META_MAX_EXPS * 16 + HY4_META_MAX_TILES * 8;
    let fused_gu = lw_moe.u_off == n_ff_exp * lw_moe.gu_row_bytes;

    let dispatch_one_window = |win: &[(u64, usize, usize)], meta_cursor: &mut usize| -> Result<()> {
        if win.is_empty() {
            return Ok(());
        }
        let slot = *meta_cursor % HY4_META_SLOTS;
        if *meta_cursor >= HY4_META_SLOTS {
            sc.ev_meta_done[slot].sync()?;
        }
        *meta_cursor += 1;

        let base_row_off = win[0].1;
        let n_exp = win.len();
        let meta_base = sc.meta_pin.as_mut_ptr().add(slot * one_meta_bytes);
        let ptrs_pin = std::slice::from_raw_parts_mut(meta_base.cast::<u64>(), n_exp);
        let off_pin = std::slice::from_raw_parts_mut(
            meta_base.add(HY4_META_MAX_EXPS * 8).cast::<i32>(),
            n_exp,
        );
        let nt_pin = std::slice::from_raw_parts_mut(
            meta_base.add(HY4_META_MAX_EXPS * 12).cast::<i32>(),
            n_exp,
        );
        let te_pin = std::slice::from_raw_parts_mut(
            meta_base.add(HY4_META_MAX_EXPS * 16).cast::<i32>(),
            HY4_META_MAX_TILES,
        );
        let tm_pin = std::slice::from_raw_parts_mut(
            meta_base
                .add(HY4_META_MAX_EXPS * 16 + HY4_META_MAX_TILES * 4)
                .cast::<i32>(),
            HY4_META_MAX_TILES,
        );

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

        let d_ptrs = sc.prefill_grp_ptrs_dev.add(slot * HY4_META_MAX_EXPS);
        let d_off = sc.prefill_grp_off_dev.add(slot * HY4_META_MAX_EXPS);
        let d_nt = sc.prefill_grp_nt_dev.add(slot * HY4_META_MAX_EXPS);
        let d_te = sc.prefill_tile_exp_dev.add(slot * HY4_META_MAX_TILES);
        let d_tm = sc.prefill_tile_moff_dev.add(slot * HY4_META_MAX_TILES);

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

        let rows_dev = sc.prefill_rows_dev.add(base_row_off);
        let wts_dev = sc.prefill_wts_dev.add(base_row_off);
        if fused_gu {
            glm5::gemm_mmq_mma_grouped(
                lw_moe.tg as i32,
                sc.x_q8,
                d_ptrs,
                0,
                lw_moe.gu_row_bytes,
                sc.prefill_gu_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                (2 * n_ff_exp) as i32,
                lat as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::swiglu_clamp_interleaved_q8(
                sc.prefill_gu_f32,
                lw_moe.clamp_lim,
                w_rows as i32,
                n_ff_exp as i32,
                sc.prefill_grp_q8,
                stream,
            );
        } else {
            // Separate gate and up into prefill_gu_f32[0..] and prefill_gu_f32[w_rows*n_ff_exp..]
            let up_f32 = sc.prefill_gu_f32.add(w_rows * n_ff_exp);
            glm5::gemm_mmq_mma_grouped(
                lw_moe.tg as i32,
                sc.x_q8,
                d_ptrs,
                0,
                lw_moe.gu_row_bytes,
                sc.prefill_gu_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                n_ff_exp as i32,
                lat as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::gemm_mmq_mma_grouped(
                lw_moe.tg as i32,
                sc.x_q8,
                d_ptrs,
                lw_moe.u_off,
                lw_moe.gu_row_bytes,
                up_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off,
                d_nt,
                d_te,
                d_tm,
                n_tiles as i32,
                n_ff_exp as i32,
                lat as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::swiglu_clamp_q8(
                sc.prefill_gu_f32,
                up_f32,
                lw_moe.clamp_lim,
                n_ff_exp as i32,
                std::ptr::null_mut(),
                sc.prefill_grp_q8,
                w_rows as i32,
                stream,
            );
        }

        glm5::gemm_mmq_mma_grouped(
            lw_moe.td as i32,
            sc.prefill_grp_q8,
            d_ptrs,
            lw_moe.d_off,
            lw_moe.dn_row_bytes,
            sc.latout,
            std::ptr::null(),
            rows_dev,
            wts_dev,
            d_off,
            d_nt,
            d_te,
            d_tm,
            n_tiles as i32,
            lat as i32,
            n_ff_exp as i32,
            1.0,
            true,
            tile_mode,
            stream,
        );
        sc.ev_meta_done[slot].record(stream)?;
        Ok(())
    };

    let mut idx = 0usize;
    while idx < jobs.len() {
        let (ptr0, roff0, nt0) = jobs[idx];
        if nt0 > HY4_PREFILL_GRP_WIN {
            for t0 in (0..nt0).step_by(HY4_PREFILL_GRP_WIN) {
                let slice_job = [(ptr0, roff0 + t0, (nt0 - t0).min(HY4_PREFILL_GRP_WIN))];
                dispatch_one_window(&slice_job, meta_cursor)?;
            }
            idx += 1;
            continue;
        }
        let mut end = idx;
        let mut sum_rows = 0usize;
        let mut sum_tiles = 0usize;
        while end < jobs.len() && (end - idx) < HY4_META_MAX_EXPS {
            let nt = jobs[end].2;
            let t_exp = nt.div_ceil(bm);
            if end > idx
                && (sum_rows + nt > HY4_PREFILL_GRP_WIN || sum_tiles + t_exp > HY4_META_MAX_TILES)
            {
                break;
            }
            sum_rows += nt;
            sum_tiles += t_exp;
            end += 1;
        }
        dispatch_one_window(&jobs[idx..end], meta_cursor)?;
        idx = end;
    }
    Ok(())
}

unsafe fn run_moe_group_mmq(
    sc: &Hy4GpuScratch,
    lw_moe: &crate::hy4::weights::Hy4MoeWeights,
    lat: usize,
    n_ff_exp: usize,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)], // sorted desc by nt, contiguous in prefill_rows_dev
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let s1 = jobs.iter().take_while(|j| j.2 > 96).count();
    let s2 = s1 + jobs[s1..].iter().take_while(|j| j.2 > 32).count();
    run_moe_group_mmq_class(sc, lw_moe, lat, n_ff_exp, meta_cursor, &jobs[..s1], 1, 128)?;
    run_moe_group_mmq_class(sc, lw_moe, lat, n_ff_exp, meta_cursor, &jobs[s1..s2], 2, 64)?;
    run_moe_group_mmq_class(sc, lw_moe, lat, n_ff_exp, meta_cursor, &jobs[s2..], 3, 32)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn ffn_core_layer(
    weights: &Hy4ModelWeights,
    lw: &Hy4LayerWeights,
    sc: &Hy4GpuScratch,
    t_len: usize,
    h_ids_buf: &mut [i32],
    h_w_buf: &mut [f32],
    h_lat_buf: &mut [f32],
    h_cpu_buf: &mut [f32],
    cpu_pool: &CpuThreadPool,
    cpu_scratch: &mut Glm5CpuScratch,
    prof: &mut Hy4ProfileStats,
) -> Result<()> {
    let prof_on = hy4_profile_enabled();
    let cfg = &weights.cfg;
    let e = cfg.n_embd;
    let stream = &sc.stream;

    if let Some(ref df) = lw.dense_ffn {
        let ff = cfg.n_ff;
        let sub = HY4_ATTN_SUB.min(t_len);
        for a0 in (0..t_len).step_by(sub) {
            let ts = (t_len - a0).min(sub);
            let xa_p = sc.xa.add(a0 * e);
            let y_p = sc.y.add(a0 * e);
            launch_quantize_q8_1(xa_p, sc.x_q8, ts as i32, e as i32, stream.raw);
            mm_hy4(&df.gate, xa_p, sc.x_q8, ts, sc.gu, 1.0, sc);
            mm_hy4(&df.up, xa_p, sc.x_q8, ts, sc.gu.add(ts * ff), 1.0, sc);
            hy4::swiglu_f32(
                sc.gu,
                sc.gu.add(ts * ff),
                ff,
                ts as i32,
                ff as i32,
                0.0,
                sc.hd,
                ff,
                stream,
            );
            mm_hy4(&df.down, sc.hd, std::ptr::null(), ts, y_p, 1.0, sc);
        }
        return Ok(());
    }

    let lm = lw.moe.as_ref().expect("MoE layer missing moe weights");
    let lat = cfg.n_expert_latent;
    let ffe = cfg.n_ff_exp;
    let sf = cfg.n_ff_exp * cfg.n_expert_shared;
    let topk = cfg.n_expert_used;
    let n_exp = cfg.n_expert;

    mm_hy4(&lm.router, sc.xa, std::ptr::null(), t_len, sc.rl, 1.0, sc);
    dbg_sum(sc, "ffn_moe_logits", lw.il as isize, sc.rl, t_len, n_exp, n_exp, 0);

    hy4::router_sigmoid_topk(
        sc.rl,
        t_len as i32,
        n_exp as i32,
        lm.router_b.ptr.cast(),
        topk as i32,
        cfg.expert_weights_norm,
        cfg.expert_weights_scale,
        sc.ids,
        sc.rw,
        stream,
    );
    if hy4_dump_enabled() {
        stream.sync()?;
        copy_d2h_raw_async(
            bytemuck::cast_slice_mut(&mut h_ids_buf[..t_len * topk]),
            sc.ids.cast(),
            stream,
        )?;
        stream.sync()?;
        let sum: i64 = h_ids_buf[..t_len * topk].iter().map(|&v| v as i64).sum();
        let label = format!("ffn_moe_topk-{}", lw.il);
        eprintln!("DUMP {label:<24} sum = {sum}");
    }

    let need_routes_host = lm.is_partial || t_len > 4;
    let kick_routes_d2h = |h_ids_buf: &mut [i32],
                           h_w_buf: &mut [f32],
                           h_lat_buf: &mut [f32]|
     -> Result<()> {
        if need_routes_host {
            sc.ev_d2h_x.record(stream)?;
            sc.d2h_stream.wait_event(&sc.ev_d2h_x)?;
            copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut h_ids_buf[..t_len * topk]),
                sc.ids.cast(),
                &sc.d2h_stream,
            )?;
            copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut h_w_buf[..t_len * topk]),
                sc.rw.cast(),
                &sc.d2h_stream,
            )?;
            if lm.is_partial {
                copy_d2h_raw_async(
                    bytemuck::cast_slice_mut(&mut h_lat_buf[..t_len * lat]),
                    sc.lat.cast(),
                    &sc.d2h_stream,
                )?;
            }
            sc.ev_routes_ready.record(&sc.d2h_stream)?;
        }
        Ok(())
    };

    if t_len <= HY4_ATTN_SUB {
        // Quantize xa once for rdown + sh_gate + sh_up, then immediately overlap ALL of Shared Expert on GPU!
        launch_quantize_q8_1(sc.xa, sc.x_q8, t_len as i32, e as i32, stream.raw);
        mm_hy4(&lm.rdown, sc.xa, sc.x_q8, t_len, sc.lat, 1.0, sc);
        kick_routes_d2h(h_ids_buf, h_w_buf, h_lat_buf)?;
        mm_hy4(&lm.sh_gate, sc.xa, sc.x_q8, t_len, sc.gu, 1.0, sc);
        mm_hy4(&lm.sh_up, sc.xa, sc.x_q8, t_len, sc.gu.add(t_len * sf), 1.0, sc);
        hy4::swiglu_f32(
            sc.gu,
            sc.gu.add(t_len * sf),
            sf,
            t_len as i32,
            sf as i32,
            0.0,
            sc.hd,
            sf,
            stream,
        );
        mm_hy4(&lm.sh_down, sc.hd, std::ptr::null(), t_len, sc.cur, 1.0, sc);
    } else {
        for a0 in (0..t_len).step_by(HY4_ATTN_SUB) {
            let ts = (t_len - a0).min(HY4_ATTN_SUB);
            let xa_p = sc.xa.add(a0 * e);
            launch_quantize_q8_1(xa_p, sc.x_q8, ts as i32, e as i32, stream.raw);
            mm_hy4(&lm.rdown, xa_p, sc.x_q8, ts, sc.lat.add(a0 * lat), 1.0, sc);
        }
        kick_routes_d2h(h_ids_buf, h_w_buf, h_lat_buf)?;
        for a0 in (0..t_len).step_by(HY4_ATTN_SUB) {
            let ts = (t_len - a0).min(HY4_ATTN_SUB);
            let xa_p = sc.xa.add(a0 * e);
            launch_quantize_q8_1(xa_p, sc.x_q8, ts as i32, e as i32, stream.raw);
            mm_hy4(&lm.sh_gate, xa_p, sc.x_q8, ts, sc.gu, 1.0, sc);
            mm_hy4(&lm.sh_up, xa_p, sc.x_q8, ts, sc.gu.add(ts * sf), 1.0, sc);
            hy4::swiglu_f32(
                sc.gu,
                sc.gu.add(ts * sf),
                sf,
                ts as i32,
                sf as i32,
                0.0,
                sc.hd,
                sf,
                stream,
            );
            mm_hy4(&lm.sh_down, sc.hd, std::ptr::null(), ts, sc.cur.add(a0 * e), 1.0, sc);
        }
    }

    // Execute routed experts
    if t_len <= 4 {
        launch_quantize_q8_1(sc.lat, sc.x_q8, t_len as i32, lat as i32, stream.raw);
        let vram_base = lm.vram_experts.as_ref().map_or(0u64, |b| b.as_u64());
        hy4::build_moe_plan(
            sc.ids,
            lm.slot_d.ptr.cast(),
            vram_base,
            lm.blob_bytes,
            (t_len * topk) as i32,
            sc.plan_ptrs,
            stream,
        );
        glm5::moe_plan_filtered(
            lm.tg as i32,
            lm.td as i32,
            sc.plan_ptrs,
            lm.u_off as u32,
            lm.d_off as u32,
            lm.gu_row_bytes,
            lm.dn_row_bytes,
            sc.x_q8,
            sc.rw,
            sc.dec_inter_q8,
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            sc.latout,
            lat as i32,
            ffe as i32,
            topk as i32,
            t_len as i32,
            lm.clamp_lim,
            std::ptr::null(),
            std::ptr::null(),
            0,
            stream,
        );

        if lm.is_partial {
            let t_rw0 = prof_on.then(Instant::now);
            sc.ev_routes_ready.sync()?;
            if let Some(t0) = t_rw0 {
                prof.routes_wait_ms += t0.elapsed().as_secs_f64() * 1e3;
            }
            if t_len == 1 {
                let mut jobs: Vec<Glm5CpuExpertJob> = Vec::with_capacity(topk);
                for j in 0..topk {
                    let eid = h_ids_buf[j] as usize;
                    if eid < n_exp && lm.slot_of[eid] < 0 {
                        jobs.push(Glm5CpuExpertJob {
                            gate_ptr: lm.cold_gate_ptrs[eid],
                            up_ptr: lm.cold_up_ptrs[eid],
                            down_ptr: lm.cold_down_ptrs[eid],
                            weight: h_w_buf[j],
                        });
                    }
                }
                if prof_on {
                    let c_cnt = jobs.len() as u64;
                    prof.n_cpu_experts += c_cnt;
                    prof.n_vram_experts += (topk as u64).saturating_sub(c_cnt);
                }
                if !jobs.is_empty() {
                    let t_cm0 = prof_on.then(Instant::now);
                    run_glm5_cpu_moe_m1(
                        cpu_pool,
                        cpu_scratch,
                        lm.tg,
                        lm.td,
                        lat,
                        ffe,
                        lm.clamp_lim,
                        &h_lat_buf[..lat],
                        None,
                        &jobs,
                        &mut h_cpu_buf[..lat],
                    );
                    copy_h2d_raw_async(
                        sc.cpo.cast(),
                        bytemuck::cast_slice(&h_cpu_buf[..lat]),
                        stream,
                    )?;
                    hy4::add_f32(sc.latout, sc.cpo, lat, stream);
                    if let Some(t0) = t_cm0 {
                        prof.cpu_moe_ms += t0.elapsed().as_secs_f64() * 1e3;
                    }
                    if prof_on {
                        let t_gt0 = Instant::now();
                        stream.sync()?;
                        prof.gpu_moe_ms += t_gt0.elapsed().as_secs_f64() * 1e3;
                    }
                }
            } else {
                // t_len = 2..4 (MTP verify): group cold experts so each cold expert is loaded once!
                let mut counts = vec![0usize; n_exp];
                let mut c_slots = 0u64;
                for &raw in &h_ids_buf[..t_len * topk] {
                    let eid = raw as usize;
                    if eid < n_exp && lm.slot_of[eid] < 0 {
                        counts[eid] += 1;
                        c_slots += 1;
                    }
                }
                let mut tok_lists: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
                for (eid, &cnt) in counts.iter().enumerate() {
                    if cnt > 0 {
                        tok_lists.push((eid, Vec::with_capacity(cnt), Vec::with_capacity(cnt)));
                    }
                }
                if prof_on {
                    prof.n_cpu_experts += c_slots;
                    prof.n_vram_experts += (t_len * topk) as u64 - c_slots;
                }
                if !tok_lists.is_empty() {
                    let mut eid_to_idx = vec![usize::MAX; n_exp];
                    for (idx, &(eid, _, _)) in tok_lists.iter().enumerate() {
                        eid_to_idx[eid] = idx;
                    }
                    for t in 0..t_len {
                        for j in 0..topk {
                            let eid = h_ids_buf[t * topk + j] as usize;
                            let idx = if eid < n_exp { eid_to_idx[eid] } else { usize::MAX };
                            if idx != usize::MAX {
                                tok_lists[idx].1.push(t);
                                tok_lists[idx].2.push(h_w_buf[t * topk + j]);
                            }
                        }
                    }
                    let groups: Vec<Glm5CpuGroupedExpert<'_>> = tok_lists
                        .iter()
                        .map(|(eid, ti, tw)| Glm5CpuGroupedExpert {
                            gate_ptr: lm.cold_gate_ptrs[*eid],
                            up_ptr: lm.cold_up_ptrs[*eid],
                            down_ptr: lm.cold_down_ptrs[*eid],
                            token_indices: ti,
                            token_weights: tw,
                        })
                        .collect();
                    let t_cm0 = prof_on.then(Instant::now);
                    ampllm_cpu_kernels::run_glm5_cpu_moe_grouped(
                        cpu_pool,
                        cpu_scratch,
                        lm.tg,
                        lm.td,
                        lat,
                        ffe,
                        lm.clamp_lim,
                        t_len,
                        &h_lat_buf[..t_len * lat],
                        &groups,
                        &mut h_cpu_buf[..t_len * lat],
                        true,
                    );
                    copy_h2d_raw_async(
                        sc.cpo.cast(),
                        bytemuck::cast_slice(&h_cpu_buf[..t_len * lat]),
                        stream,
                    )?;
                    hy4::add_f32(sc.latout, sc.cpo, t_len * lat, stream);
                    if let Some(t0) = t_cm0 {
                        prof.cpu_moe_ms += t0.elapsed().as_secs_f64() * 1e3;
                    }
                    if prof_on {
                        let t_gt0 = Instant::now();
                        stream.sync()?;
                        prof.gpu_moe_ms += t_gt0.elapsed().as_secs_f64() * 1e3;
                    }
                }
            }
        }
    } else {
        // Prefill / medium batch (T > 4): grouped MMQ Tensor Cores + triple-buffered DMA + AVX-512 overlap
        launch_quantize_q8_1(sc.lat, sc.x_q8, t_len as i32, lat as i32, stream.raw);
        ampllm_cuda_kernels::zero_raw_async(sc.latout.cast(), t_len * lat * 4, stream)?;

        let t_rw0 = prof_on.then(Instant::now);
        sc.ev_routes_ready.sync()?;
        if let Some(t0) = t_rw0 {
            prof.routes_wait_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        let total_routes = t_len * topk;
        let host_ids = &h_ids_buf[..total_routes];
        let host_rw = &h_w_buf[..total_routes];

        let mut counts = vec![0usize; n_exp];
        for &raw in host_ids {
            let eid = raw as usize;
            if eid < n_exp {
                counts[eid] += 1;
            }
        }
        let mut offsets = vec![0usize; n_exp];
        let mut run_off = 0usize;
        for eid in 0..n_exp {
            offsets[eid] = run_off;
            run_off += counts[eid];
        }
        let mut cursors = offsets.clone();
        let mut bucket_rows = vec![0i32; run_off];
        let mut bucket_wts = vec![0.0f32; run_off];
        for t in 0..t_len {
            for j in 0..topk {
                let eid = host_ids[t * topk + j] as usize;
                if eid < n_exp {
                    let dst = cursors[eid];
                    cursors[eid] = dst + 1;
                    bucket_rows[dst] = t as i32;
                    bucket_wts[dst] = host_rw[t * topk + j];
                }
            }
        }

        let small_max_nt: usize = std::env::var("AMPLLM_HY4_SMALL_MOE_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(1, HY4_SMALL_MOE_CAP);

        let rows_pin = std::slice::from_raw_parts_mut(sc.small_pin.as_mut_ptr().cast::<i32>(), total_routes);
        let wts_pin = std::slice::from_raw_parts_mut(
            sc.small_pin.as_mut_ptr().add(total_routes * 4).cast::<f32>(),
            total_routes,
        );
        let meta_start = sc.small_pin.as_mut_ptr().add(total_routes * 8);
        let exp_ptrs_pin = std::slice::from_raw_parts_mut(meta_start.cast::<u64>(), n_exp);
        let slot_off_pin = std::slice::from_raw_parts_mut(meta_start.add(n_exp * 8).cast::<i32>(), n_exp);
        let slot_nt_pin = std::slice::from_raw_parts_mut(meta_start.add(n_exp * 12).cast::<i32>(), n_exp);

        let vram_base = lm.vram_experts.as_ref().map_or(0u64, |b| b.as_u64());
        let mut n_vram_small = 0usize;
        let mut vram_small_raw: Vec<(u64, usize, usize)> = Vec::new();
        let mut vram_large_raw: Vec<(u64, usize, usize)> = Vec::new();
        let mut cold_raw: Vec<(usize, usize, usize)> = Vec::new();
        let mut cur_off = 0usize;

        for eid in 0..n_exp {
            let nt = counts[eid];
            if nt == 0 {
                continue;
            }
            let b_off = offsets[eid];
            let slot = lm.slot_of[eid];
            if slot >= 0 {
                let ptr = vram_base + (slot as usize * lm.blob_bytes) as u64;
                if nt <= small_max_nt {
                    rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
                    wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
                    vram_small_raw.push((ptr, cur_off, nt));
                    cur_off += nt;
                } else {
                    vram_large_raw.push((ptr, b_off, nt));
                }
            } else {
                cold_raw.push((eid, b_off, nt));
            }
        }

        for (i, &(ptr, roff, nt)) in vram_small_raw.iter().enumerate() {
            exp_ptrs_pin[i] = ptr;
            slot_off_pin[i] = roff as i32;
            slot_nt_pin[i] = nt as i32;
            n_vram_small += 1;
        }

        cold_raw.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        let k_cold = cold_raw.len();
        let k_cold_large = cold_raw.iter().take_while(|x| x.2 > small_max_nt).count();
        let mut cold_row_offs = vec![0usize; k_cold];
        for (c_idx, &(_, b_off, nt)) in cold_raw.iter().enumerate() {
            if nt <= small_max_nt {
                rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
                wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
                cold_row_offs[c_idx] = cur_off;
                cur_off += nt;
            }
        }

        vram_large_raw.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        let mut vram_large: Vec<(u64, usize, usize)> = Vec::with_capacity(vram_large_raw.len());
        for &(ptr, b_off, nt) in &vram_large_raw {
            rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
            wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
            vram_large.push((ptr, cur_off, nt));
            cur_off += nt;
        }

        for (c_idx, &(_, b_off, nt)) in cold_raw[..k_cold_large].iter().enumerate() {
            rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
            wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
            cold_row_offs[c_idx] = cur_off;
            cur_off += nt;
        }

        let mut cold_list: Vec<ColdPrefillJob> = Vec::with_capacity(k_cold);
        for (c_idx, &(eid, b_off, nt)) in cold_raw.iter().enumerate() {
            let rs = &bucket_rows[b_off..b_off + nt];
            let ws = &bucket_wts[b_off..b_off + nt];
            let roff = cold_row_offs[c_idx];
            if c_idx >= k_cold_large {
                let sm_idx = n_vram_small + (c_idx - k_cold_large);
                slot_off_pin[sm_idx] = roff as i32;
                slot_nt_pin[sm_idx] = nt as i32;
            }
            cold_list.push(ColdPrefillJob {
                e: eid,
                row_off: roff,
                nt,
                tok_indices: rs.iter().map(|&r| r as usize).collect(),
                tok_wts: ws.to_vec(),
            });
        }
        let n_slot_meta = n_vram_small + (k_cold - k_cold_large);

        // Decide split between GPU DMA streaming and AVX-512 CPU
        let n_gpu_cold = if !lm.cold_is_pinned || sc.dma_stage.is_none() || t_len <= 16 {
            0usize
        } else if let Ok(v) = std::env::var("AMPLLM_HY4_DMA_MIN_NT") {
            let min_nt = v.parse::<usize>().unwrap_or(6);
            cold_list.iter().take_while(|j| j.nt >= min_nt).count()
        } else {
            // Balanced PCIe DMA vs 16-core AVX-512 cost model (lat=1536, ffe=1536, blob ~3.6-4.6MB)
            let cpu_max_nt = (t_len / 12).clamp(40, 256);
            let cpu_scale = 8.0 / (cpu_pool.num_threads().max(1) as f64);
            let blob_us = (lm.blob_bytes as f64) / 11_000.0;
            let comp_cost = |nt: usize| -> f64 {
                if nt <= small_max_nt {
                    14.0 + 1.15 * (nt as f64)
                } else {
                    18.0 + 1.00 * (nt as f64)
                }
            };
            let vram_us: f64 = (n_vram_small as f64) * 14.0
                + vram_small_raw.iter().map(|&(_, _, nt)| 1.15 * nt as f64).sum::<f64>()
                + vram_large.iter().map(|&(_, _, nt)| comp_cost(nt)).sum::<f64>();
            let cpu_quant_us = 45.0 + 0.5 * (t_len as f64);
            let out_h2d_us = ((t_len * lat * 4) as f64) / 11_500.0;
            let mut pref_gpu_comp = vec![0.0f64; k_cold + 1];
            for i in 0..k_cold {
                pref_gpu_comp[i + 1] = pref_gpu_comp[i] + comp_cost(cold_list[i].nt);
            }
            let mut suff_cpu = vec![0.0f64; k_cold + 1];
            let mut suff_ok = vec![true; k_cold + 1];
            for i in (0..k_cold).rev() {
                let nt = cold_list[i].nt;
                suff_ok[i] = suff_ok[i + 1] && nt <= cpu_max_nt;
                suff_cpu[i] = suff_cpu[i + 1] + (140.0 + 28.0 * (nt as f64)) * cpu_scale;
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

        if cur_off > 0 {
            copy_h2d_raw_async(
                sc.prefill_rows_dev.cast(),
                std::slice::from_raw_parts(rows_pin.as_ptr() as *const u8, cur_off * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                sc.prefill_wts_dev.cast(),
                std::slice::from_raw_parts(wts_pin.as_ptr() as *const u8, cur_off * 4),
                stream,
            )?;
        }
        if n_vram_small > 0 {
            copy_h2d_raw_async(
                sc.prefill_exp_ptrs_dev.cast(),
                std::slice::from_raw_parts(exp_ptrs_pin.as_ptr() as *const u8, n_vram_small * 8),
                stream,
            )?;
        }
        if n_slot_meta > 0 {
            copy_h2d_raw_async(
                sc.prefill_slot_off_dev.cast(),
                std::slice::from_raw_parts(slot_off_pin.as_ptr() as *const u8, n_slot_meta * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                sc.prefill_slot_nt_dev.cast(),
                std::slice::from_raw_parts(slot_nt_pin.as_ptr() as *const u8, n_slot_meta * 4),
                stream,
            )?;
        }

        let cold_groups: Vec<(usize, usize)> = {
            let mut v = Vec::with_capacity(n_gpu_cold.div_ceil(HY4_DMA_GROUP_CAP));
            let mut g0 = 0usize;
            while g0 < n_gpu_cold {
                let g1 = (g0 + HY4_DMA_GROUP_CAP).min(n_gpu_cold);
                v.push((g0, g1));
                g0 = g1;
            }
            v
        };

        let kick_group = |g_idx: usize, r_slot: usize| -> Result<()> {
            let (g_start, g_end) = cold_groups[g_idx];
            let base_s = r_slot * HY4_DMA_GROUP_CAP;
            for (i, c_idx) in (g_start..g_end).enumerate() {
                let eid = cold_list[c_idx].e;
                let hp = lm.cold_blob_ptrs[eid];
                let host_slice = std::slice::from_raw_parts(hp, lm.blob_bytes);
                let dst_ptr = sc.dma_slot_ptrs[base_s + i] as *mut c_void;
                copy_h2d_raw_async(dst_ptr, host_slice, &sc.copy_stream)?;
            }
            sc.ev_h2d_group_done[r_slot].record(&sc.copy_stream)?;
            Ok(())
        };

        let init_g = HY4_DMA_RING_GROUPS.min(cold_groups.len());
        for g_idx in 0..init_g {
            kick_group(g_idx, g_idx)?;
        }

        let cpu_jobs = &cold_list[n_gpu_cold..];
        let dev_id = lw.device;
        let h_lat_addr = h_lat_buf.as_ptr() as usize;
        let h_cpu_addr = h_cpu_buf.as_mut_ptr() as usize;

        let (cpu_res, gpu_res) = std::thread::scope(|scope| {
            let h_cpu = (!cpu_jobs.is_empty()).then(|| {
                scope.spawn(move || -> Result<bool> {
                    set_device(dev_id)?;
                    let host_x = std::slice::from_raw_parts(h_lat_addr as *const f32, t_len * lat);
                    quantize_glm5_cpu_x_q8k(cpu_pool, cpu_scratch, lat, t_len, host_x);
                    let cpu_out = std::slice::from_raw_parts_mut(h_cpu_addr as *mut f32, t_len * lat);
                    let mut did_any = false;
                    let mut idx = 0usize;
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
                            .map(|job| Glm5CpuGroupedExpert {
                                gate_ptr: lm.cold_gate_ptrs[job.e],
                                up_ptr: lm.cold_up_ptrs[job.e],
                                down_ptr: lm.cold_down_ptrs[job.e],
                                token_indices: &job.tok_indices,
                                token_weights: &job.tok_wts,
                            })
                            .collect();
                        run_glm5_cpu_moe_grouped_prequant(
                            cpu_pool,
                            cpu_scratch,
                            lm.tg,
                            lm.td,
                            lat,
                            ffe,
                            lm.clamp_lim,
                            t_len,
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
                let run_small = |ptrs_dev: *const u64, off_dev: *const i32, nt_dev: *const i32, nj: usize| {
                    glm5::moe_prefill_small_gu(
                        lm.tg as i32,
                        ptrs_dev,
                        off_dev,
                        nt_dev,
                        sc.prefill_rows_dev,
                        lm.u_off as u32,
                        lm.gu_row_bytes,
                        sc.x_q8,
                        sc.prefill_grp_q8,
                        lat as i32,
                        ffe as i32,
                        nj as i32,
                        lm.clamp_lim,
                        stream,
                    );
                    glm5::moe_prefill_small_down(
                        lm.td as i32,
                        ptrs_dev,
                        off_dev,
                        nt_dev,
                        sc.prefill_rows_dev,
                        sc.prefill_wts_dev,
                        lm.d_off as u32,
                        lm.dn_row_bytes,
                        sc.prefill_grp_q8,
                        sc.latout,
                        lat as i32,
                        ffe as i32,
                        nj as i32,
                        stream,
                    );
                };

                if n_vram_small > 0 {
                    run_small(
                        sc.prefill_exp_ptrs_dev,
                        sc.prefill_slot_off_dev,
                        sc.prefill_slot_nt_dev,
                        n_vram_small,
                    );
                }

                let mut meta_cursor = 0usize;
                if !vram_large.is_empty() {
                    run_moe_group_mmq(sc, lm, lat, ffe, &mut meta_cursor, &vram_large)?;
                }

                let mut grp_jobs: Vec<(u64, usize, usize)> = Vec::with_capacity(HY4_DMA_GROUP_CAP);
                for (g_idx, &(g_start, g_end)) in cold_groups.iter().enumerate() {
                    let r_slot = g_idx % HY4_DMA_RING_GROUPS;
                    let base_s = r_slot * HY4_DMA_GROUP_CAP;
                    stream.wait_event(&sc.ev_h2d_group_done[r_slot])?;

                    let k_g_large = cold_list[g_start..g_end]
                        .iter()
                        .take_while(|j| j.nt > small_max_nt)
                        .count();
                    let k_g_small = (g_end - g_start) - k_g_large;

                    if k_g_large > 0 {
                        grp_jobs.clear();
                        for i in 0..k_g_large {
                            let j = &cold_list[g_start + i];
                            grp_jobs.push((sc.dma_slot_ptrs[base_s + i], j.row_off, j.nt));
                        }
                        run_moe_group_mmq(sc, lm, lat, ffe, &mut meta_cursor, &grp_jobs)?;
                    }
                    if k_g_small > 0 {
                        let ptrs_dev = sc.dma_ptrs_dev.add(base_s + k_g_large);
                        let meta_idx = g_start + k_g_large - k_cold_large;
                        let off_dev = sc.prefill_slot_off_dev.add(n_vram_small + meta_idx);
                        let nt_dev = sc.prefill_slot_nt_dev.add(n_vram_small + meta_idx);
                        run_small(ptrs_dev, off_dev, nt_dev, k_g_small);
                    }

                    sc.ev_comp_group_free[r_slot].record(stream)?;
                    let next_g = g_idx + HY4_DMA_RING_GROUPS;
                    if next_g < cold_groups.len() {
                        sc.copy_stream.wait_event(&sc.ev_comp_group_free[r_slot])?;
                        kick_group(next_g, r_slot)?;
                    }
                }
                Ok(())
            })();

            let c_res = match h_cpu {
                Some(h) => h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("CPU MoE worker panicked"))),
                None => Ok(false),
            };
            (c_res, gpu_closure)
        });

        gpu_res?;
        if cpu_res? {
            copy_h2d_raw_async(
                sc.cpo.cast(),
                bytemuck::cast_slice(&h_cpu_buf[..t_len * lat]),
                stream,
            )?;
            hy4::add_f32(sc.latout, sc.cpo, t_len * lat, stream);
        }
    }

    dbg_sum(sc, "ffn_moe_out", lw.il as isize, sc.latout, t_len, lat, lat, 0);

    // Routed up projection (quantize latout -> x_q8)
    mm_hy4(&lm.rup, sc.latout, std::ptr::null(), t_len, sc.y, 1.0, sc);
    dbg_sum(sc, "ffn_moe_up", lw.il as isize, sc.y, t_len, e, e, 0);

    // Shared expert output already lives in `sc.cur`; add into `sc.y`
    dbg_sum(sc, "ffn_shexp", lw.il as isize, sc.cur, t_len, e, e, 0);
    hy4::add_f32(sc.y, sc.cur, t_len * e, stream);
    Ok(())
}

/// Forward a full trunk window of `T` tokens (`ids[0..T]`) starting at position `p0`,
/// leaving final normed states in `hring` and (if `want > 0`) `want` rows of logits on `head` + `st.h_logits`.
pub unsafe fn forward_trunk_window(
    weights: &Hy4ModelWeights,
    st: &mut Hy4State,
    ids: &[u32],
    p0: usize,
    want: usize,
    imgs: &[Hy4ImageSpan],
) -> Result<()> {
    let t_len = ids.len();
    let cfg = &weights.cfg;
    let e = cfg.n_embd;
    let hc = cfg.hc;
    let eps = cfg.rms_norm_eps;
    let dump = hy4_dump_enabled();
    let prof_on = hy4_profile_enabled();
    let t_trunk0 = prof_on.then(Instant::now);
    let tc = weights.tc;

    // 1. Decode token embeddings into pinned h_x (with image span substitution)
    {
        let h_x = std::slice::from_raw_parts_mut(st.h_x.as_mut_ptr().cast::<f32>(), t_len * e);
        decode_tokens_to_host(weights, ids, p0, imgs, h_x)?;
    }

    // 2. Gather Engram F8_E4M3 rows either before Layer 0 (only if Layer 0 has Engram)
    //    or right after Layer 0 is asynchronously queued on GPU 0 (hiding Engram lookup behind Layer 0 compute!).
    let has_engram = cfg.engram.is_some() && weights.engram_table.is_some();
    let need_eg_before_l0 = cfg
        .engram
        .as_ref()
        .is_some_and(|ec| ec.layers.contains(&0));
    if has_engram {
        if st.history.len() < p0 + t_len {
            st.history.resize(p0 + t_len, 0);
        }
        st.history[p0..p0 + t_len].copy_from_slice(ids);
    }
    let mut did_engram = false;
    if has_engram && need_eg_before_l0 {
        let eg_cfg = cfg.engram.as_ref().unwrap();
        let eg_tbl = weights.engram_table.as_ref().unwrap();
        let h_eg_s = std::slice::from_raw_parts_mut(st.h_eg.as_mut_ptr(), st.h_eg.bytes());
        let h_val_s = std::slice::from_raw_parts_mut(st.h_valid.as_mut_ptr().cast::<f32>(), t_len);
        let h_egs_s = st.h_egs.as_ref().map(|b| {
            std::slice::from_raw_parts_mut(b.as_mut_ptr().cast::<f32>(), b.bytes() / 4)
        });
        let t_eg0 = prof_on.then(Instant::now);
        gather_engram_window(
            eg_cfg,
            eg_tbl,
            &st.cpu_pool,
            &st.history,
            p0,
            t_len,
            tc,
            h_eg_s,
            h_val_s,
            h_egs_s,
        );
        if let Some(t0) = t_eg0 {
            st.profile.engram_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        did_engram = true;
    }

    // 3. Prepare SWA positions `[pos | lo | nk]` in pinned h_pos
    let h_pos_s = std::slice::from_raw_parts_mut(st.h_pos.as_mut_ptr().cast::<i32>(), 3 * tc);
    for t in 0..t_len {
        let p = p0 + t;
        let lo = p.saturating_sub(cfg.n_swa - 1);
        h_pos_s[t] = p as i32;
        h_pos_s[tc + t] = lo as i32;
        h_pos_s[2 * tc + t] = (p - lo + 1) as i32;
    }

    let mut cur_gpu: Option<usize> = None;
    for (il, lw) in weights.layers.iter().enumerate() {
        if has_engram && !did_engram && lw.engram.is_some() {
            let eg_cfg = cfg.engram.as_ref().unwrap();
            let eg_tbl = weights.engram_table.as_ref().unwrap();
            let h_eg_s = std::slice::from_raw_parts_mut(st.h_eg.as_mut_ptr(), st.h_eg.bytes());
            let h_val_s = std::slice::from_raw_parts_mut(st.h_valid.as_mut_ptr().cast::<f32>(), t_len);
            let h_egs_s = st.h_egs.as_ref().map(|b| {
                std::slice::from_raw_parts_mut(b.as_mut_ptr().cast::<f32>(), b.bytes() / 4)
            });
            let t_eg0 = prof_on.then(Instant::now);
            gather_engram_window(
                eg_cfg,
                eg_tbl,
                &st.cpu_pool,
                &st.history,
                p0,
                t_len,
                tc,
                h_eg_s,
                h_val_s,
                h_egs_s,
            );
            if let Some(t0) = t_eg0 {
                st.profile.engram_ms += t0.elapsed().as_secs_f64() * 1e3;
            }
            did_engram = true;
        }

        let h_ids_s = std::slice::from_raw_parts_mut(st.h_ids.as_mut_ptr().cast::<i32>(), st.h_ids.bytes() / 4);
        let h_w_s = std::slice::from_raw_parts_mut(st.h_w.as_mut_ptr().cast::<f32>(), st.h_w.bytes() / 4);
        let h_lat_s = std::slice::from_raw_parts_mut(st.h_lat.as_mut_ptr().cast::<f32>(), st.h_lat.bytes() / 4);
        let h_cpu_s = std::slice::from_raw_parts_mut(st.h_cpu.as_mut_ptr().cast::<f32>(), st.h_cpu.bytes() / 4);

        let gi = lw.gpu_idx;
        let sc = &st.sc[gi];
        let cache = &st.caches[il];
        if cur_gpu != Some(gi) {
            set_device(sc.device)?;
            copy_h2d_raw_async(sc.pos.cast(), std::slice::from_raw_parts(h_pos_s.as_ptr() as *const u8, t_len * 4), &sc.stream)?;
            copy_h2d_raw_async(sc.lo.cast(), std::slice::from_raw_parts(h_pos_s[tc..].as_ptr() as *const u8, t_len * 4), &sc.stream)?;
            copy_h2d_raw_async(sc.nk.cast(), std::slice::from_raw_parts(h_pos_s[2 * tc..].as_ptr() as *const u8, t_len * 4), &sc.stream)?;
            if let Some(prev_gi) = cur_gpu {
                let t_p0 = prof_on.then(Instant::now);
                let psc = &st.sc[prev_gi];
                let nbytes = t_len * hc * e * 4;
                if weights.p2p_ok {
                    psc.ev_p2p_ready.record(&psc.stream)?;
                    sc.stream.wait_event(&psc.ev_p2p_ready)?;
                    copy_peer_raw_async(sc.x.cast(), sc.device, psc.x.cast(), psc.device, nbytes, &sc.stream)?;
                } else {
                    psc.stream.sync()?;
                    let mut tmp = vec![0u8; nbytes];
                    copy_d2h_raw_async(&mut tmp, psc.x.cast(), &psc.stream)?;
                    psc.stream.sync()?;
                    copy_h2d_raw_async(sc.x.cast(), &tmp, &sc.stream)?;
                }
                if let Some(t0) = t_p0 {
                    st.profile.p2p_ms += t0.elapsed().as_secs_f64() * 1e3;
                }
            } else {
                copy_h2d_raw_async(
                    sc.cur.cast(),
                    std::slice::from_raw_parts(st.h_x.as_mut_ptr(), t_len * e * 4),
                    &sc.stream,
                )?;
                hy4::expand_streams(sc.cur, t_len as i32, hc as i32, e as i32, sc.x, &sc.stream);
            }
            cur_gpu = Some(gi);
        }

        // Engram injection (sub-chunked by HY4_ATTN_SUB)
        if let Some(ref eg_lw) = lw.engram {
            let eg_cfg = cfg.engram.as_ref().unwrap();
            let cols = eg_cfg.cols();
            let hd = eg_cfg.head_dim;
            let row_b = cols * hd;
            let sub = HY4_ATTN_SUB.min(t_len);
            for a0 in (0..t_len).step_by(sub) {
                let ts = (t_len - a0).min(sub);
                let src_eg = st.h_eg.as_mut_ptr().add((eg_lw.module_idx * tc + a0) * row_b);
                copy_h2d_raw_async(sc.eg_rows_u8.cast(), std::slice::from_raw_parts(src_eg, ts * row_b), &sc.stream)?;
                copy_h2d_raw_async(
                    sc.eg_valid.cast(),
                    std::slice::from_raw_parts(st.h_valid.as_mut_ptr().add(a0 * 4), ts * 4),
                    &sc.stream,
                )?;
                let sc_opt = if let Some(ref b) = st.h_egs {
                    let p = b.as_mut_ptr().cast::<f32>().add((eg_lw.module_idx * tc + a0) * cols);
                    copy_h2d_raw_async(sc.eg_sc.cast(), std::slice::from_raw_parts(p as *const u8, ts * cols * 4), &sc.stream)?;
                    sc.eg_sc as *const f32
                } else {
                    std::ptr::null()
                };
                hy4::engram_rows(
                    sc.eg_rows_u8,
                    sc.eg_valid,
                    ts as i32,
                    cols as i32,
                    hd as i32,
                    eg_lw.lookup_norm.ptr.cast(),
                    eg_cfg.engram_eps,
                    sc_opt,
                    sc.eg_r,
                    &sc.stream,
                );
                dbg_sum(sc, "engram_rows_normed", il as isize, sc.eg_r, ts, row_b, row_b, 0);
                mm_hy4(&eg_lw.kv, sc.eg_r, std::ptr::null(), ts, sc.eg_kvo, 1.0, sc);
                dbg_sum(sc, "engram_kv(no bias)", il as isize, sc.eg_kvo, ts, (hc + 1) * e, (hc + 1) * e, 0);
                hy4::engram_mix(
                    sc.x.add(a0 * hc * e),
                    sc.eg_kvo,
                    eg_lw.kv_b.ptr.cast(),
                    ts as i32,
                    hc as i32,
                    e as i32,
                    eg_lw.gate.ptr.cast(),
                    eg_lw.pre_gamma.ptr.cast(),
                    eg_lw.hc_scale.ptr.cast(),
                    sc.eg_valid,
                    eg_cfg.engram_eps,
                    &sc.stream,
                );
            }
            dbg_sum(sc, "engram_out", il as isize, sc.x, t_len, hc * e, hc * e, 0);
        }

        // Attention block
        let hca = lw.hc_attn.as_ref().unwrap();
        hy4::hc_pre(
            sc.x,
            t_len as i32,
            hc as i32,
            e as i32,
            hca.fn_w.ptr.cast(),
            true,
            hca.scale.ptr.cast(),
            hca.base.ptr.cast(),
            eps,
            cfg.hc_eps,
            cfg.hc_magnitude,
            lw.attn_norm.ptr.cast(),
            eps,
            false,
            sc.hc_part,
            sc.xa,
            sc.post,
            &sc.stream,
        );
        dbg_sum(sc, "attn_norm", il as isize, sc.xa, t_len, e, e, 0);

        let sub = HY4_ATTN_SUB.min(t_len);
        for a0 in (0..t_len).step_by(sub) {
            let ts = (t_len - a0).min(sub);
            attn_core_layer(weights, lw, cache, sc, ts, a0, p0 + a0, false)?;
        }

        hy4::hc_post_norm(
            sc.x,
            t_len as i32,
            hc as i32,
            e as i32,
            sc.y,
            sc.post,
            lw.post_attn.ptr.cast(),
            eps,
            if dump { sc.cur } else { std::ptr::null_mut() },
            &sc.stream,
        );
        if dump {
            dbg_sum(sc, "attn_out", il as isize, sc.cur, t_len, e, e, 0);
            dbg_sum(sc, "hc_attn_out", il as isize, sc.x, t_len, hc * e, hc * e, 0);
        }

        // FFN block
        let hcf = lw.hc_ffn.as_ref().unwrap();
        hy4::hc_pre(
            sc.x,
            t_len as i32,
            hc as i32,
            e as i32,
            hcf.fn_w.ptr.cast(),
            true,
            hcf.scale.ptr.cast(),
            hcf.base.ptr.cast(),
            eps,
            cfg.hc_eps,
            cfg.hc_magnitude,
            lw.ffn_norm.ptr.cast(),
            eps,
            false,
            sc.hc_part,
            sc.xa,
            sc.post,
            &sc.stream,
        );
        dbg_sum(sc, "ffn_norm", il as isize, sc.xa, t_len, e, e, 0);

        ffn_core_layer(
            weights,
            lw,
            sc,
            t_len,
            h_ids_s,
            h_w_s,
            h_lat_s,
            h_cpu_s,
            &st.cpu_pool,
            &mut st.cpu_scratch,
            &mut st.profile,
        )?;

        hy4::hc_post_norm(
            sc.x,
            t_len as i32,
            hc as i32,
            e as i32,
            sc.y,
            sc.post,
            lw.post_ffn.ptr.cast(),
            eps,
            if dump { sc.cur } else { std::ptr::null_mut() },
            &sc.stream,
        );
        if dump {
            dbg_sum(sc, "ffn_out", il as isize, sc.cur, t_len, e, e, 0);
            dbg_sum(sc, "l_out", il as isize, sc.x, t_len, hc * e, hc * e, 0);
        }
    }

    // 4. Output Head on `head_gpu_idx`
    let hg_idx = weights.head_gpu_idx;
    let hsc = &st.sc[hg_idx];
    if cur_gpu != Some(hg_idx) {
        set_device(hsc.device)?;
        copy_h2d_raw_async(hsc.pos.cast(), std::slice::from_raw_parts(h_pos_s.as_ptr() as *const u8, t_len * 4), &hsc.stream)?;
        if let Some(prev_gi) = cur_gpu {
            let t_p0 = prof_on.then(Instant::now);
            let psc = &st.sc[prev_gi];
            let nbytes = t_len * hc * e * 4;
            if weights.p2p_ok {
                psc.ev_p2p_ready.record(&psc.stream)?;
                hsc.stream.wait_event(&psc.ev_p2p_ready)?;
                copy_peer_raw_async(hsc.x.cast(), hsc.device, psc.x.cast(), psc.device, nbytes, &hsc.stream)?;
            } else {
                psc.stream.sync()?;
                let mut tmp = vec![0u8; nbytes];
                copy_d2h_raw_async(&mut tmp, psc.x.cast(), &psc.stream)?;
                psc.stream.sync()?;
                copy_h2d_raw_async(hsc.x.cast(), &tmp, &hsc.stream)?;
            }
            if let Some(t0) = t_p0 {
                st.profile.p2p_ms += t0.elapsed().as_secs_f64() * 1e3;
            }
        }
    }

    let head = &weights.head;
    hy4::hc_pre(
        hsc.x,
        t_len as i32,
        hc as i32,
        e as i32,
        head.hh.fn_w.ptr.cast(),
        true,
        head.hh.scale.ptr.cast(),
        head.hh.base.ptr.cast(),
        eps,
        cfg.hc_eps,
        cfg.hc_magnitude,
        std::ptr::null(),
        eps,
        true,
        hsc.hc_part,
        hsc.hcur,
        std::ptr::null_mut(),
        &hsc.stream,
    );
    dbg_sum(hsc, "hc_head", -1, hsc.hcur.add((t_len - 1) * e), 1, e, e, 0);

    hy4::rmsnorm_strided(
        hsc.hcur,
        e,
        head.out_norm.ptr.cast(),
        hsc.hcur,
        e,
        e as i32,
        t_len as i32,
        eps,
        &hsc.stream,
    );
    dbg_sum(hsc, "result_norm", -1, hsc.hcur.add((t_len - 1) * e), 1, e, e, 0);

    hy4::store_rows(
        hsc.hcur,
        t_len as i32,
        e as i32,
        st.hring.ptr,
        weights.hring_cap as i32,
        hsc.pos,
        false,
        &hsc.stream,
    );

    if want > 0 {
        mm_hy4(
            &head.out_w,
            hsc.hcur.add((t_len - want) * e),
            std::ptr::null(),
            want,
            hsc.logits,
            1.0,
            hsc,
        );
        if dump {
            dbg_sum(hsc, "result_output", -1, hsc.logits, want, cfg.n_vocab, cfg.n_vocab, 0);
            hsc.stream.sync()?;
            let mut last_row = vec![0.0f32; cfg.n_vocab];
            copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut last_row),
                hsc.logits.add((want - 1) * cfg.n_vocab).cast(),
                &hsc.stream,
            )?;
            hsc.stream.sync()?;
            let nv = last_row.len();
            eprintln!(
                "DUMP logits[last] {:.4} {:.4} {:.4} ... {:.4} {:.4} {:.4}",
                last_row[0], last_row[1], last_row[2], last_row[nv - 3], last_row[nv - 2], last_row[nv - 1]
            );
        }
        copy_d2h_raw_async(
            std::slice::from_raw_parts_mut(st.h_logits.as_mut_ptr(), want * cfg.n_vocab * 4),
            hsc.logits.cast(),
            &hsc.stream,
        )?;
    }

    hsc.stream.sync()?;
    if let Some(t0) = t_trunk0 {
        st.profile.trunk_ms += t0.elapsed().as_secs_f64() * 1e3;
        st.profile.n_rounds += 1;
    }
    Ok(())
}
