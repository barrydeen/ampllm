//! Thin Rust wrappers for MiniMax M3.1 Flash (`mm3`), DSpark, and Vision CUDA kernels.

use crate::{ffi, CudaStream};
use std::ffi::c_void;

/// Size in bytes of one packed CUDA/CPU `block_q8_K` (4 + 256 + 32 = 292 bytes).
pub const Q8K_BLOCK_BYTES: usize = 292;

/// Number of floats per `(token, kv_head, split, gqa_group)` scratch slice for MSA/DSpark split-K attention.
pub const ATTN_PART_STRIDE: usize = 130;

/// Number of floats required in the Markov reduction scratch buffer (`640` values + `640` indices).
pub const MARKOV_PART_FLOATS: usize = 640 * 2;

/// Device-side grouped decode/verify MoE plan (`<= 32` distinct experts and `<= 32` `(tok, slot)` entries).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mm3DecodePlanC {
    pub base_ptrs: [u64; 32],
    pub grp_off: [i32; 33],
    pub entry_tok: [i32; 32],
    pub entry_wt: [f32; 32],
    pub _pad: i32,
}

impl Default for Mm3DecodePlanC {
    fn default() -> Self {
        Self {
            base_ptrs: [0; 32],
            grp_off: [0; 33],
            entry_tok: [0; 32],
            entry_wt: [0.0; 32],
            _pad: 0,
        }
    }
}

/// Fused RMSNorm with optional `+1` weight offset and multi-format outputs (`F32`, `Q8_1`, `Q8_K`, `BF16`).
pub unsafe fn rms_norm_f32(
    x: *const f32,
    w: *const f32,
    rows: i32,
    dim: i32,
    eps: f32,
    add_one: bool,
    y_f32_opt: *mut f32,
    y_q8_1_opt: *mut c_void,
    y_q8_k_opt: *mut c_void,
    y_bf16_opt: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_rms_norm_f32(
        x,
        w,
        rows,
        dim,
        eps,
        if add_one { 1 } else { 0 },
        y_f32_opt,
        y_q8_1_opt,
        y_q8_k_opt,
        y_bf16_opt,
        stream.raw,
    );
}

/// Quantizes `m` rows of `k` `f32` values into packed 292-byte `block_q8_K` blocks.
pub unsafe fn quantize_q8_k(
    x: *const f32,
    out_q8k: *mut c_void,
    m: i32,
    k: i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_quantize_q8_k(x, out_q8k, m, k, stream.raw);
}

/// Computes `swigluoai(gate, up)` and writes any combination of `F32`, `Q8_1`, `Q8_K`, or `BF16` outputs.
pub unsafe fn swigluoai(
    gate: *const f32,
    up: *const f32,
    rows: i32,
    ff: i32,
    alpha: f32,
    limit: f32,
    y_f32_opt: *mut f32,
    y_q8_1_opt: *mut c_void,
    y_q8_k_opt: *mut c_void,
    y_bf16_opt: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_swigluoai(
        gate,
        up,
        rows,
        ff,
        alpha,
        limit,
        y_f32_opt,
        y_q8_1_opt,
        y_q8_k_opt,
        y_bf16_opt,
        stream.raw,
    );
}

/// Applies partial RoPE + Q/K RMSNorm (and optional Had128+INT8 rotation) and stores K/V into GPU/host ring caches.
#[allow(clippy::too_many_arguments)]
pub unsafe fn qk_norm_rope_kv_store(
    q: *mut f32,
    k: *const f32,
    v: *const f32,
    q_norm_w: *const f32,
    k_norm_w: *const f32,
    pos: *const i32,
    inv_freq: *const f32,
    t: i32,
    hq: i32,
    hkv: i32,
    hd: i32,
    rot_dim: i32,
    eps: f32,
    kv_fmt: i32,
    rot_mask: i32,
    ring: i32,
    k_bf16_opt: *mut u16,
    v_bf16_opt: *mut u16,
    k_i8_opt: *mut i8,
    k_scale_opt: *mut f32,
    v_i8_opt: *mut i8,
    v_scale_opt: *mut f32,
    host_k_opt: *mut c_void,
    host_k_scale_opt: *mut f32,
    host_v_opt: *mut c_void,
    host_v_scale_opt: *mut f32,
    host_tag_opt: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_qk_norm_rope_kv_store(
        q,
        k,
        v,
        q_norm_w,
        k_norm_w,
        pos,
        inv_freq,
        t,
        hq,
        hkv,
        hd,
        rot_dim,
        eps,
        kv_fmt,
        rot_mask,
        ring,
        k_bf16_opt,
        v_bf16_opt,
        k_i8_opt,
        k_scale_opt,
        v_i8_opt,
        v_scale_opt,
        host_k_opt,
        host_k_scale_opt,
        host_v_opt,
        host_v_scale_opt,
        host_tag_opt,
        stream.raw,
    );
}

/// Applies RMSNorm and partial RoPE to MSA indexer Q/K and stores BF16 indexer keys at `pos`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_norm_rope_store(
    iq: *mut f32,
    ik: *const f32,
    iq_norm_w: *const f32,
    ik_norm_w: *const f32,
    pos: *const i32,
    inv_freq: *const f32,
    t: i32,
    hi: i32,
    hd: i32,
    rot_dim: i32,
    eps: f32,
    ic_bf16: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_index_norm_rope_store(
        iq,
        ik,
        iq_norm_w,
        ik_norm_w,
        pos,
        inv_freq,
        t,
        hi,
        hd,
        rot_dim,
        eps,
        ic_bf16,
        stream.raw,
    );
}

/// Computes blockwise max-pooled `dot(iq, ic_bf16)` indexer scores `[hi, t, nblk]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn msa_block_scores(
    iq: *const f32,
    ic_bf16: *const u16,
    pos: *const i32,
    t: i32,
    hi: i32,
    nblk: i32,
    blk: i32,
    bs: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_msa_block_scores(iq, ic_bf16, pos, t, hi, nblk, blk, bs, stream.raw);
}

/// Selects the top-`topk` blocks per `(hi, t)` including own-block forcing and local window retention.
#[allow(clippy::too_many_arguments)]
pub unsafe fn msa_select_topk(
    bs: *const f32,
    pos: *const i32,
    t: i32,
    hi: i32,
    nblk: i32,
    blk: i32,
    topk: i32,
    local_blocks: i32,
    kept: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_msa_select_topk(
        bs,
        pos,
        t,
        hi,
        nblk,
        blk,
        topk,
        local_blocks,
        kept,
        stream.raw,
    );
}

/// Runs block-sparse MSA GQA attention over kept blocks using BF16 or Had128-rotated INT8 KV caches.
#[allow(clippy::too_many_arguments)]
pub unsafe fn msa_sparse_attn(
    q: *const f32,
    kc_bf16_opt: *const u16,
    vc_bf16_opt: *const u16,
    kc_i8_opt: *const i8,
    k_scale_opt: *const f32,
    vc_i8_opt: *const i8,
    v_scale_opt: *const f32,
    kept: *const i32,
    pos: *const i32,
    t: i32,
    hq: i32,
    hkv: i32,
    topk: i32,
    blk: i32,
    kv_fmt: i32,
    rot_mask: i32,
    ring: i32,
    scale: f32,
    part_opt: *mut f32,
    splits: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_msa_sparse_attn(
        q,
        kc_bf16_opt,
        vc_bf16_opt,
        kc_i8_opt,
        k_scale_opt,
        vc_i8_opt,
        v_scale_opt,
        kept,
        pos,
        t,
        hq,
        hkv,
        topk,
        blk,
        kv_fmt,
        rot_mask,
        ring,
        scale,
        part_opt,
        splits,
        out,
        stream.raw,
    );
}

/// Computes MoE router logits (when `x_f32_opt` and `gate_w_opt` are non-null) and sigmoid top-k normalized weights.
#[allow(clippy::too_many_arguments)]
pub unsafe fn router_f32(
    x_f32_opt: *const f32,
    gate_w_opt: *const f32,
    logits_buf: *mut f32,
    bias_opt: *const f32,
    t: i32,
    d: i32,
    e: i32,
    topk: i32,
    routed_scaling_factor: f32,
    ids_out: *mut i32,
    weights_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_router_f32(
        x_f32_opt,
        gate_w_opt,
        logits_buf,
        bias_opt,
        t,
        d,
        e,
        topk,
        routed_scaling_factor,
        ids_out,
        weights_out,
        stream.raw,
    );
}

/// Fused GPU MoE gate+up GEMV with independent `gate_type`/`up_type`, `swigluoai`, and `Q8_K` requantization.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_gate_up_q8k(
    gate_type: i32,
    up_type: i32,
    gate_ptrs: *const u64,
    up_ptrs: *const u64,
    gate_row_bytes: usize,
    up_row_bytes: usize,
    x_q8k: *const c_void,
    expert_ids: *const i32,
    inter_q8k: *mut c_void,
    inter_f32_opt: *mut f32,
    m: i32,
    n_embd: i32,
    n_ff: i32,
    topk: i32,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_moe_gate_up_q8k(
        gate_type,
        up_type,
        gate_ptrs,
        up_ptrs,
        gate_row_bytes,
        up_row_bytes,
        x_q8k,
        expert_ids,
        inter_q8k,
        inter_f32_opt,
        m,
        n_embd,
        n_ff,
        topk,
        swiglu_alpha,
        swiglu_limit,
        skip_dev,
        stream.raw,
    );
}

/// GPU MoE down projection from `Q8_K` intermediate activations to `[m, topk, n_embd]` `f32`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_down_q8k(
    down_type: i32,
    down_ptrs: *const u64,
    down_row_bytes: usize,
    inter_q8k: *const c_void,
    expert_ids: *const i32,
    expert_out: *mut f32,
    m: i32,
    n_ff: i32,
    n_embd: i32,
    topk: i32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_moe_down_q8k(
        down_type,
        down_ptrs,
        down_row_bytes,
        inter_q8k,
        expert_ids,
        expert_out,
        m,
        n_ff,
        n_embd,
        topk,
        skip_dev,
        stream.raw,
    );
}

/// Combines weighted routed GPU expert outputs with optional residual and shared expert output.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_combine(
    res_opt: *const f32,
    expert_out: *const f32,
    weights: *const f32,
    expert_ids: *const i32,
    vram_ptrs_opt: *const u64,
    shared_opt: *const f32,
    m: i32,
    topk: i32,
    n_embd: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_moe_combine(
        res_opt,
        expert_out,
        weights,
        expert_ids,
        vram_ptrs_opt,
        shared_opt,
        m,
        topk,
        n_embd,
        out,
        stream.raw,
    );
}

/// High-occupancy grouped GPU MoE gate+up GEMV with SwiGLU-OAI activation (`F32` intermediate output).
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_grp_gate_up_f32(
    gate_type: i32,
    up_type: i32,
    plan_dev: *const c_void,
    u_off: usize,
    gate_row_bytes: usize,
    up_row_bytes: usize,
    x_q8k: *const c_void,
    inter_f32: *mut f32,
    n_embd: i32,
    n_ff: i32,
    n_grp: i32,
    max_nt: i32,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_moe_grp_gate_up_f32(
        gate_type,
        up_type,
        plan_dev,
        u_off,
        gate_row_bytes,
        up_row_bytes,
        x_q8k,
        inter_f32,
        n_embd,
        n_ff,
        n_grp,
        max_nt,
        swiglu_alpha,
        swiglu_limit,
        stream.raw,
    );
}

/// High-occupancy grouped GPU MoE down projection from `Q8_K` intermediate activations to `[total_entries, n_embd]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_grp_down_q8k(
    down_type: i32,
    plan_dev: *const c_void,
    d_off: usize,
    down_row_bytes: usize,
    inter_q8k: *const c_void,
    expert_out: *mut f32,
    n_ff: i32,
    n_embd: i32,
    n_grp: i32,
    max_nt: i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_moe_grp_down_q8k(
        down_type,
        plan_dev,
        d_off,
        down_row_bytes,
        inter_q8k,
        expert_out,
        n_ff,
        n_embd,
        n_grp,
        max_nt,
        stream.raw,
    );
}

/// Combines residual, shared expert output, optional CPU output, and up to two grouped GPU MoE plan outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_grp_combine(
    res_opt: *const f32,
    shared_opt: *const f32,
    expert_out: *const f32,
    plan0_dev_opt: *const c_void,
    n_entries0: i32,
    plan1_dev_opt: *const c_void,
    n_entries1: i32,
    cpu_out_opt: *const f32,
    m: i32,
    n_embd: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_moe_grp_combine(
        res_opt,
        shared_opt,
        expert_out,
        plan0_dev_opt,
        n_entries0,
        plan1_dev_opt,
        n_entries1,
        cpu_out_opt,
        m,
        n_embd,
        out,
        stream.raw,
    );
}

/// Custom SM86 BF16 Tensor Core GEMM `C = alpha * A * B^T + beta * C` (`A[M, K]`, `B[N, K]`, `C[M, N]`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn bf16_gemm_mma(
    a: *const u16,
    ld_a: i32,
    b: *const u16,
    ld_b: i32,
    c: *mut f32,
    ld_c: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_bf16_gemm_mma(a, ld_a, b, ld_b, c, ld_c, m, n, k, alpha, beta, stream.raw);
}

/// Dequantizes an MXFP8 (`F8_E4M3` + `U8` UE8M0 block-32 scale) `[N, K]` matrix to `BF16`.
pub unsafe fn mxfp8_to_bf16(
    w: *const u8,
    scale: *const u8,
    n: i32,
    k: i32,
    out_bf16: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_mxfp8_to_bf16(w, scale, n, k, out_bf16, stream.raw);
}

/// Computes `y = x * W^T + beta * y` for MXFP8 weights `[N, K]` using fused MXFP8 decode or BF16 MMA scratch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mxfp8_linear(
    w: *const u8,
    scale: *const u8,
    n: i32,
    k: i32,
    x: *const f32,
    ld_x: i32,
    rows: i32,
    y: *mut f32,
    ld_y: i32,
    beta: f32,
    part_opt: *mut f32,
    w_bf16_scratch_opt: *mut u16,
    x_bf16_scratch_opt: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_mxfp8_linear(
        w,
        scale,
        n,
        k,
        x,
        ld_x,
        rows,
        y,
        ld_y,
        beta,
        part_opt,
        w_bf16_scratch_opt,
        x_bf16_scratch_opt,
        stream.raw,
    );
}

/// DSpark BF16 sliding-window GQA attention over `[lo[t], hi]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_window_attn(
    q: *const f32,
    kc_bf16: *const u16,
    vc_bf16: *const u16,
    ring: i32,
    lo: *const i32,
    hi: i32,
    t: i32,
    hq: i32,
    hkv: i32,
    scale: f32,
    part_opt: *mut f32,
    splits: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_dspark_window_attn(
        q,
        kc_bf16,
        vc_bf16,
        ring,
        lo,
        hi,
        t,
        hq,
        hkv,
        scale,
        part_opt,
        splits,
        out,
        stream.raw,
    );
}

/// DSpark rank-256 Markov transition bias addition, greedy argmax token selection, and optional acceptance confidence.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_markov_greedy(
    logits: *const f32,
    w2_bf16: *const u16,
    w1_bf16: *const u16,
    v: i32,
    r: i32,
    prev: *mut i32,
    out: *mut i32,
    cw_opt: *const f32,
    cb_opt: *const f32,
    hidden_opt: *const f32,
    d: i32,
    conf_opt: *mut f32,
    part: *mut f32,
    bias_opt: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_dspark_markov_greedy(
        logits,
        w2_bf16,
        w1_bf16,
        v,
        r,
        prev,
        out,
        cw_opt,
        cb_opt,
        hidden_opt,
        d,
        conf_opt,
        part,
        bias_opt,
        stream.raw,
    );
}

/// Vision patch embedding linear projection `out[N, D] = px[N, P] * w_patch[D, P]^T`.
pub unsafe fn vision_patch_embed_f32(
    px: *const f32,
    w_patch: *const f32,
    n: i32,
    d: i32,
    p: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_vision_patch_embed_f32(px, w_patch, n, d, p, out, stream.raw);
}

/// Vision LayerNorm with affine weight+bias and optional `F32` / `BF16` outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vision_layernorm(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    rows: i32,
    dim: i32,
    eps: f32,
    y_f32_opt: *mut f32,
    y_bf16_opt: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_vision_layernorm(
        x, w, b, rows, dim, eps, y_f32_opt, y_bf16_opt, stream.raw,
    );
}

/// Vision fused QKV bias addition and 3D (T, H, W) interleaved-pair partial RoPE.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vision_qkv_rope_3d(
    qkv: *mut f32,
    bias_opt: *const f32,
    pos3d: *const i32,
    inv_freq: *const f32,
    n: i32,
    dim: i32,
    heads: i32,
    hd: i32,
    nf: i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_vision_qkv_rope_3d(
        qkv, bias_opt, pos3d, inv_freq, n, dim, heads, hd, nf, stream.raw,
    );
}

/// Vision non-causal full bidirectional multi-head self-attention.
pub unsafe fn vision_full_attn(
    qkv: *const f32,
    n: i32,
    dim: i32,
    heads: i32,
    hd: i32,
    scale: f32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_vision_full_attn(qkv, n, dim, heads, hd, scale, out, stream.raw);
}

/// Vision elementwise optional bias add and optional exact `erff` GELU with `F32`/`BF16` outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vision_bias_gelu(
    y: *const f32,
    bias_opt: *const f32,
    n: usize,
    m: i32,
    apply_gelu: bool,
    y_f32_opt: *mut f32,
    y_bf16_opt: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_mm3_vision_bias_gelu(
        y,
        bias_opt,
        n,
        m,
        if apply_gelu { 1 } else { 0 },
        y_f32_opt,
        y_bf16_opt,
        stream.raw,
    );
}

/// Vision in-place residual add with optional bias `h[i] += t[i] (+ bias[i % m])`.
pub unsafe fn vision_residual_bias(
    h: *mut f32,
    t: *const f32,
    bias_opt: *const f32,
    n: usize,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_vision_residual_bias(h, t, bias_opt, n, m, stream.raw);
}

/// Coalesced multi-job `Q6_K` decode/verify GEMV (`M = 1..8`) sharing `x_q8_1`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gv_q6k_multi(
    n_jobs: i32,
    w0: *const c_void,
    y0: *mut f32,
    n0: i32,
    w1: *const c_void,
    y1: *mut f32,
    n1: i32,
    w2: *const c_void,
    y2: *mut f32,
    n2: i32,
    x_q8_1: *const c_void,
    k: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_gv_q6k_multi(
        n_jobs, w0, y0, n0, w1, y1, n1, w2, y2, n2, x_q8_1, k, m, stream.raw,
    );
}

/// Coalesced multi-job `Q8_0` decode/verify GEMV (`M = 1..8`) sharing `x_q8_1`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gv_q8_0_multi(
    n_jobs: i32,
    w0: *const c_void,
    y0: *mut f32,
    n0: i32,
    w1: *const c_void,
    y1: *mut f32,
    n1: i32,
    w2: *const c_void,
    y2: *mut f32,
    n2: i32,
    x_q8_1: *const c_void,
    k: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_gv_q8_0_multi(
        n_jobs, w0, y0, n0, w1, y1, n1, w2, y2, n2, x_q8_1, k, m, stream.raw,
    );
}

/// Populates device integer positions `[pos0 .. pos0 + n)`.
pub unsafe fn iota(out: *mut i32, n: i32, pos0: i32, stream: &CudaStream) {
    ffi::launch_mm3_iota(out, n, pos0, stream.raw);
}

/// Copy-engine-free DSpark draft block initialization from mapped anchor embedding + device mask row.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_draft_block(
    anc_mapped_f32: *const f32,
    mask_f32: *const f32,
    d: i32,
    b: i32,
    pos0: i32,
    window: i32,
    anchor_id: i32,
    h_out: *mut f32,
    pos_out: *mut i32,
    lo_out: *mut i32,
    prev_out: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_mm3_dspark_draft_block(
        anc_mapped_f32,
        mask_f32,
        d,
        b,
        pos0,
        window,
        anchor_id,
        h_out,
        pos_out,
        lo_out,
        prev_out,
        stream.raw,
    );
}

/// Spin-waits on mapped host `u32` word until it equals `target_val`, then executes `__threadfence_system()`.
pub unsafe fn wait_u32(wait_dev_ptr: *const u32, target_val: u32, stream: &CudaStream) {
    ffi::launch_mm3_wait_u32(wait_dev_ptr, target_val, stream.raw);
}
