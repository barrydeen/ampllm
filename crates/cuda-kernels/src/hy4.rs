use crate::{ffi, CudaStream};
use std::ffi::c_void;

pub const HY4_HC_BLOCKS: usize = 24;
pub const HY4_MAX_M: usize = 16;
pub const HY4_TOP1_SCRATCH_FLOATS: usize = 386;

#[inline]
pub fn hc_part_floats(t: usize) -> usize {
    t * HY4_HC_BLOCKS * (HY4_MAX_M + 1)
}

#[inline]
pub fn kv_q8_row_bytes(kv_lora: usize, n_rot: usize) -> usize {
    ((kv_lora + kv_lora / 16 + 2 * n_rot + 15) / 16) * 16
}

#[inline]
pub fn mla_decode_ws_floats(t: usize, h: usize, max_keys: usize) -> usize {
    let kc = if t <= 4 { 32 } else { 128 };
    let nchunk = ((max_keys + kc - 1) / kc).max(1);
    t * h * nchunk * 514
}

pub unsafe fn expand_streams(
    emb: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    x_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_expand_streams(emb, t, hc, e, x_out, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_pre(
    x: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    fn_w: *const f32,
    fn_by_el: bool,
    scale: *const f32,
    base: *const f32,
    eps: f32,
    hc_eps: f32,
    mag: f32,
    norm_w_opt: *const f32,
    norm_eps: f32,
    is_head: bool,
    part_buf: *mut f32,
    cur: *mut f32,
    post_opt: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_hc_pre(
        x,
        t,
        hc,
        e,
        fn_w,
        i32::from(fn_by_el),
        scale,
        base,
        eps,
        hc_eps,
        mag,
        norm_w_opt,
        norm_eps,
        i32::from(is_head),
        part_buf,
        cur,
        post_opt,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_post_norm(
    x: *mut f32,
    t: i32,
    hc: i32,
    e: i32,
    y: *const f32,
    post: *const f32,
    w_opt: *const f32,
    eps: f32,
    y_norm_opt: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_hc_post_norm(x, t, hc, e, y, post, w_opt, eps, y_norm_opt, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn rope(
    x: *mut f32,
    t: i32,
    nh: i32,
    ld_t: usize,
    ld_h: usize,
    off: i32,
    n_rot: i32,
    pos: *const i32,
    base: f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_rope(x, t, nh, ld_t, ld_h, off, n_rot, pos, base, stream.raw);
}

pub unsafe fn sigmoid_mul(x: *mut f32, g: *const f32, n: usize, stream: &CudaStream) {
    ffi::launch_hy4_sigmoid_mul(x, g, n, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn swiglu_f32(
    gate: *const f32,
    up: *const f32,
    ld_gu: usize,
    rows: i32,
    n: i32,
    clamp_lim: f32,
    out: *mut f32,
    ld_out: usize,
    stream: &CudaStream,
) {
    ffi::launch_hy4_swiglu_f32(gate, up, ld_gu, rows, n, clamp_lim, out, ld_out, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn store_rows(
    src: *const f32,
    t: i32,
    dim: i32,
    ring: *mut c_void,
    ring_cap: i32,
    pos: *const i32,
    is_f16: bool,
    stream: &CudaStream,
) {
    ffi::launch_hy4_store_rows(src, t, dim, ring, ring_cap, pos, i32::from(is_f16), stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_f32(
    w: *const f32,
    n: i32,
    k: i32,
    x: *const f32,
    m: i32,
    y: *mut f32,
    alpha: f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_gemv_f32(w, n, k, x, m, y, alpha, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_bf16(
    w: *const u16,
    n: i32,
    k: i32,
    x: *const f32,
    m: i32,
    y: *mut f32,
    alpha: f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_gemv_bf16(w, n, k, x, m, y, alpha, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_heads_q8_1(
    x: *const f32,
    x_q8_1: *mut c_void,
    t: i32,
    h: i32,
    k: i32,
    ld_t: usize,
    ld_h: usize,
    stream: &CudaStream,
) {
    ffi::launch_hy4_quantize_heads_q8_1(x, x_q8_1, t, h, k, ld_t, ld_h, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn batched_heads_q8(
    w_q8_0: *const c_void,
    x_q8_1: *const c_void,
    y: *mut f32,
    t: i32,
    h: i32,
    n: i32,
    k: i32,
    x_tok_stride_blk: usize,
    y_tok_stride: usize,
    y_head_stride: usize,
    stream: &CudaStream,
) {
    ffi::launch_hy4_batched_heads_q8(
        w_q8_0,
        x_q8_1,
        y,
        t,
        h,
        n,
        k,
        x_tok_stride_blk,
        y_tok_stride,
        y_head_stride,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn batched_heads_q8_mma(
    w_q8_0: *const c_void,
    x_f32: *const f32,
    y: *mut f32,
    t: i32,
    h: i32,
    n: i32,
    k: i32,
    x_tok_stride: usize,
    x_head_stride: usize,
    y_tok_stride: usize,
    y_head_stride: usize,
    stream: &CudaStream,
) {
    ffi::launch_hy4_batched_heads_q8_mma(
        w_q8_0,
        x_f32,
        y,
        t,
        h,
        n,
        k,
        x_tok_stride,
        x_head_stride,
        y_tok_stride,
        y_head_stride,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_store(
    kv: *const f32,
    t: i32,
    kv_lora: i32,
    n_rot: i32,
    w: *const f32,
    eps: f32,
    pos: *const i32,
    base: f32,
    cache: *mut c_void,
    ring_cap: i32,
    is_q8: bool,
    rb: i32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_kv_store(
        kv,
        t,
        kv_lora,
        n_rot,
        w,
        eps,
        pos,
        base,
        cache,
        ring_cap,
        i32::from(is_q8),
        rb,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_decode(
    q: *const f32,
    t: i32,
    h: i32,
    cache: *const c_void,
    ring_cap: i32,
    keys_opt: *const i32,
    ldk: i32,
    lo_opt: *const i32,
    nk: *const i32,
    max_keys: i32,
    scale: f32,
    sinks_opt: *const f32,
    ws: *mut f32,
    out: *mut f32,
    is_q8: bool,
    rb: i32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_mla_decode(
        q,
        t,
        h,
        cache,
        ring_cap,
        keys_opt,
        ldk,
        lo_opt,
        nk,
        max_keys,
        scale,
        sinks_opt,
        ws,
        out,
        i32::from(is_q8),
        rb,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_prefill(
    q: *const f32,
    t: i32,
    h: i32,
    cache: *const c_void,
    ring_cap: i32,
    keys_opt: *const i32,
    ldk: i32,
    lo_opt: *const i32,
    nk: *const i32,
    scale: f32,
    sinks_opt: *const f32,
    out: *mut f32,
    is_q8: bool,
    rb: i32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_mla_prefill(
        q,
        t,
        h,
        cache,
        ring_cap,
        keys_opt,
        ldk,
        lo_opt,
        nk,
        scale,
        sinks_opt,
        out,
        i32::from(is_q8),
        rb,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn ix_pool(
    ivg: *const c_void,
    ring_cap: i32,
    g0: i32,
    n_new: i32,
    kpool: i32,
    dim: i32,
    ape: *const f32,
    w: *const f32,
    eps: f32,
    n_rot: i32,
    base: f32,
    pooled: *mut c_void,
    is_f16: bool,
    stream: &CudaStream,
) {
    ffi::launch_hy4_ix_pool(
        ivg,
        ring_cap,
        g0,
        n_new,
        kpool,
        dim,
        ape,
        w,
        eps,
        n_rot,
        base,
        pooled,
        i32::from(is_f16),
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn ix_score(
    q: *const f32,
    wt: *const f32,
    t: i32,
    heads: i32,
    dim: i32,
    pooled: *const c_void,
    pos: *const i32,
    kpool: i32,
    n_pool: i32,
    score: *mut f32,
    ld: usize,
    is_f16: bool,
    stream: &CudaStream,
) {
    ffi::launch_hy4_ix_score(
        q,
        wt,
        t,
        heads,
        dim,
        pooled,
        pos,
        kpool,
        n_pool,
        score,
        ld,
        i32::from(is_f16),
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn ix_select(
    score: *const f32,
    ld: usize,
    t: i32,
    pos: *const i32,
    kpool: i32,
    top: i32,
    keys: *mut i32,
    ldk: i32,
    nk: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_ix_select(score, ld, t, pos, kpool, top, keys, ldk, nk, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn router_sigmoid_topk(
    logits: *const f32,
    t: i32,
    n_exp: i32,
    bias_opt: *const f32,
    n_used: i32,
    norm: bool,
    scale: f32,
    ids: *mut i32,
    wout: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_router_sigmoid_topk(
        logits,
        t,
        n_exp,
        bias_opt,
        n_used,
        i32::from(norm),
        scale,
        ids,
        wout,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn engram_rows(
    rows: *const u8,
    valid: *const f32,
    t: i32,
    cols: i32,
    dim: i32,
    ln: *const f32,
    eps: f32,
    scale_opt: *const f32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_engram_rows(rows, valid, t, cols, dim, ln, eps, scale_opt, out, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn engram_mix(
    x: *mut f32,
    kv: *const f32,
    kv_b: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    egate: *const f32,
    gamma: *const f32,
    hcs: *const f32,
    valid: *const f32,
    eps: f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_engram_mix(x, kv, kv_b, t, hc, e, egate, gamma, hcs, valid, eps, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn vis_layernorm_f16(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    n: i32,
    d: i32,
    eps: f32,
    y16: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_layernorm_f16(x, w, b, n, d, eps, y16, stream.raw);
}

pub unsafe fn vis_rmsnorm_inplace(
    x: *mut f32,
    w: *const f32,
    n: i32,
    d: i32,
    eps: f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_rmsnorm_inplace(x, w, n, d, eps, stream.raw);
}

pub unsafe fn vis_add_bias(y: *mut f32, bias: *const f32, n: usize, m: i32, stream: &CudaStream) {
    ffi::launch_hy4_vis_add_bias(y, bias, n, m, stream.raw);
}

pub unsafe fn vis_bias_act_f16(
    y: *const f32,
    bias: *const f32,
    n: usize,
    m: i32,
    act: i32,
    y16: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_bias_act_f16(y, bias, n, m, act, y16, stream.raw);
}

pub unsafe fn vis_residual_bias(
    h: *mut f32,
    t: *const f32,
    bias: *const f32,
    n: usize,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_residual_bias(h, t, bias, n, m, stream.raw);
}

pub unsafe fn vis_merge_gather_f16(
    xn: *const f32,
    gw: i32,
    d: i32,
    ow: i32,
    oh: i32,
    m16: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_merge_gather_f16(xn, gw, d, ow, oh, m16, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn vis_place_rows(
    y: *const f32,
    bias: *const f32,
    nl: *const f32,
    ow: i32,
    oh: i32,
    o: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_place_rows(y, bias, nl, ow, oh, o, out, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn vis_flash_attn(
    qkv: *const f32,
    n: i32,
    heads: i32,
    hd: i32,
    scale: f32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_vis_flash_attn(qkv, n, heads, hd, scale, out, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn copy_cols(
    src: *const f32,
    ld_src: usize,
    dst: *mut f32,
    ld_dst: usize,
    rows: i32,
    width: i32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_copy_cols(src, ld_src, dst, ld_dst, rows, width, stream.raw);
}

pub unsafe fn add_f32(dst: *mut f32, src: *const f32, n: usize, stream: &CudaStream) {
    ffi::launch_hy4_add_f32(dst, src, n, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn rmsnorm_strided(
    x: *const f32,
    ld_x: usize,
    w: *const f32,
    y: *mut f32,
    ld_y: usize,
    dim: i32,
    rows: i32,
    eps: f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_rmsnorm_strided(x, ld_x, w, y, ld_y, dim, rows, eps, stream.raw);
}

pub unsafe fn top1_prob(
    logits: *const f32,
    n_vocab: i32,
    scratch_and_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_hy4_top1_prob(logits, n_vocab, scratch_and_out, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn build_moe_plan(
    ids: *const i32,
    slot_d: *const i32,
    vram_base: u64,
    blob_bytes: usize,
    count: i32,
    plan_ptrs_out: *mut u64,
    stream: &CudaStream,
) {
    ffi::launch_hy4_build_moe_plan(ids, slot_d, vram_base, blob_bytes, count, plan_ptrs_out, stream.raw);
}
