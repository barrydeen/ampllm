#![allow(clippy::too_many_arguments)]

use crate::cuda_rt::CudaStream;
use crate::ffi;
use std::ffi::c_void;

// ---- HC-4 Gated Residual & RMSNorm ----

pub unsafe fn hc_expand(
    emb: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    r_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_expand(emb, t, hc, e, r_out, stream.raw);
}

pub unsafe fn hc_rmsnorm_q8(
    r: *const f32,
    w_norm: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    eps: f32,
    xn_out: *mut f32,
    xn_q8_opt: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_rmsnorm_q8(r, w_norm, t, hc, e, eps, xn_out, xn_q8_opt, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_rmsnorm_inject_q8(
    r: *const f32,
    w_norm: *const f32,
    w_inject: *const c_void,
    inject_w_type: i32,
    t: i32,
    hc: i32,
    e: i32,
    eps: f32,
    xn_out: *mut f32,
    xn_q8_opt: *mut c_void,
    inject_partials_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_rmsnorm_inject_q8(
        r,
        w_norm,
        w_inject,
        inject_w_type,
        t,
        hc,
        e,
        eps,
        xn_out,
        xn_q8_opt,
        inject_partials_out,
        stream.raw,
    );
}

pub unsafe fn hc_down_silu_q8(
    lo_in: *const f32,
    t: i32,
    hc: i32,
    lr: i32,
    lo_out_opt: *mut f32,
    lo_q8_opt: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_down_silu_q8(lo_in, t, hc, lr, lo_out_opt, lo_q8_opt, stream.raw);
}

pub unsafe fn hc_mix_q8(
    xn: *const f32,
    up_logits: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    mixed_out: *mut f32,
    mixed_q8_opt: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_mix_q8(xn, up_logits, t, hc, e, mixed_out, mixed_q8_opt, stream.raw);
}

pub unsafe fn hc_inject_dot(
    xn: *const f32,
    w_inject: *const c_void,
    w_type: i32,
    t: i32,
    hc: i32,
    e: i32,
    inject_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_inject_dot(xn, w_inject, w_type, t, hc, e, inject_out, stream.raw);
}

pub unsafe fn hc_write(
    r: *mut f32,
    block_out: *const f32,
    sh_out_opt: *const f32,
    sh_gate_logit_opt: *const f32,
    inject: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_write(
        r,
        block_out,
        sh_out_opt,
        sh_gate_logit_opt,
        inject,
        t,
        hc,
        e,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_write_strided(
    r: *mut f32,
    block_out: *const f32,
    sh_out_opt: *const f32,
    sh_gate_logit_opt: *const f32,
    inject: *const f32,
    inject_stride: i32,
    t: i32,
    hc: i32,
    e: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_write_strided(
        r,
        block_out,
        sh_out_opt,
        sh_gate_logit_opt,
        inject,
        inject_stride,
        t,
        hc,
        e,
        stream.raw,
    );
}

pub unsafe fn add_streams_broadcast(
    h: *const f32,
    e: *const f32,
    t: i32,
    hc: i32,
    e_dim: i32,
    r_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_add_streams_broadcast(h, e, t, hc, e_dim, r_out, stream.raw);
}

pub unsafe fn rmsnorm_f32(
    x: *const f32,
    w: *const f32,
    rows: i32,
    cols: i32,
    eps: f32,
    y_out: *mut f32,
    y_q8_opt: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_rmsnorm_f32(x, w, rows, cols, eps, y_out, y_q8_opt, stream.raw);
}

// ---- PLE Post-Ops ----

pub unsafe fn ple_postops(
    key: *mut f32,
    hidden_r: *mut f32,
    value: *const f32,
    history: *mut f32,
    w_norm_key: *const f32,
    w_norm_query: *const f32,
    w_norm_conv: *const f32,
    w_conv1d_f32: *const f32,
    scratch_qn: *mut f32,
    scratch_gated: *mut f32,
    scratch_gate: *mut f32,
    t: i32,
    n: i32,
    h: i32,
    eps: f32,
    snap_opt: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_ple_postops(
        key,
        hidden_r,
        value,
        history,
        w_norm_key,
        w_norm_query,
        w_norm_conv,
        w_conv1d_f32,
        scratch_qn,
        scratch_gated,
        scratch_gate,
        t,
        n,
        h,
        eps,
        snap_opt,
        stream.raw,
    );
}

// ---- GDN Linear Attention ----

pub unsafe fn gdn_ab_f32(
    x: *const f32,
    w_alpha: *const f32,
    w_beta: *const f32,
    dt_bias: *const f32,
    ssm_a: *const f32,
    gate_out: *mut f32,
    beta_out: *mut f32,
    t: i32,
    n_embd: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_ab_f32(
        x,
        w_alpha,
        w_beta,
        dt_bias,
        ssm_a,
        gate_out,
        beta_out,
        t,
        n_embd,
        h_v,
        stream.raw,
    );
}

pub unsafe fn gdn_gate_beta_postop(
    alpha_raw: *const f32,
    beta_raw: *const f32,
    dt_bias: *const f32,
    ssm_a: *const f32,
    gate_out: *mut f32,
    beta_out: *mut f32,
    t: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_gate_beta_postop(
        alpha_raw,
        beta_raw,
        dt_bias,
        ssm_a,
        gate_out,
        beta_out,
        t,
        h_v,
        stream.raw,
    );
}

pub unsafe fn gdn_conv_l2(
    conv_hist: *mut f32,
    qkv: *const f32,
    conv_w: *const f32,
    h_out: *mut f32,
    t: i32,
    channels: i32,
    qk_heads: i32,
    eps: f32,
    hist_snap_opt: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_conv_l2(
        conv_hist,
        qkv,
        conv_w,
        h_out,
        t,
        channels,
        qk_heads,
        eps,
        hist_snap_opt,
        stream.raw,
    );
}

pub unsafe fn gdn_decode_step_norm_q8(
    state: *mut f32,
    h_qkv: *const f32,
    gate: *const f32,
    beta: *const f32,
    z_gate: *const f32,
    ssm_norm_w: *const f32,
    eps: f32,
    y_f32_opt: *mut f32,
    y_q8_opt: *mut c_void,
    h_k: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_decode_step_norm_q8(
        state,
        h_qkv,
        gate,
        beta,
        z_gate,
        ssm_norm_w,
        eps,
        y_f32_opt,
        y_q8_opt,
        h_k,
        h_v,
        stream.raw,
    );
}

pub unsafe fn gdn_verify_steps_norm_q8(
    state: *mut f32,
    h_qkv: *const f32,
    gate: *const f32,
    beta: *const f32,
    z_gate: *const f32,
    ssm_norm_w: *const f32,
    eps: f32,
    y_f32_opt: *mut f32,
    y_q8_opt: *mut c_void,
    state_snap_opt: *mut f32,
    t: i32,
    h_k: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_verify_steps_norm_q8(
        state,
        h_qkv,
        gate,
        beta,
        z_gate,
        ssm_norm_w,
        eps,
        y_f32_opt,
        y_q8_opt,
        state_snap_opt,
        t,
        h_k,
        h_v,
        stream.raw,
    );
}

pub fn gdn_tc_scratch_bytes(h_k: i32, h_v: i32) -> usize {
    unsafe { ffi::qwen4_gdn_tc_scratch_bytes(h_k, h_v) }
}

pub unsafe fn gdn_tc_prefill(
    state: *mut f32,
    h_qkv: *const f32,
    gate: *const f32,
    beta: *const f32,
    z_gate: *const f32,
    ssm_norm_w: *const f32,
    eps: f32,
    y_f32_opt: *mut f32,
    y_f16_opt: *mut c_void,
    tc_scratch: *mut c_void,
    t: i32,
    h_k: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_tc_prefill(
        state,
        h_qkv,
        gate,
        beta,
        z_gate,
        ssm_norm_w,
        eps,
        y_f32_opt,
        y_f16_opt,
        tc_scratch,
        t,
        h_k,
        h_v,
        stream.raw,
    );
}

// ---- QSA Attention, Indexer, IMRoPE & INT8 KV ----

pub unsafe fn qsa_norm_imrope(
    src: *const f32,
    in_stride: i32,
    gamma: *const f32,
    out: *mut f32,
    t: i32,
    n_heads: i32,
    head_dim: i32,
    n_rot: i32,
    eps: f32,
    freq_base: f32,
    pos_dev: *const i32,
    mrope_pos3_opt: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_qsa_norm_imrope(
        src,
        in_stride,
        gamma,
        out,
        t,
        n_heads,
        head_dim,
        n_rot,
        eps,
        freq_base,
        pos_dev,
        mrope_pos3_opt,
        stream.raw,
    );
}

pub unsafe fn idx_append_steps(
    raw_k: *const f32,
    pos_dev: *const i32,
    n_steps: i32,
    gamma: *const f32,
    eps: f32,
    freq_base: f32,
    mrope_pos3_opt: *const i32,
    tail: *mut f32,
    dead: *mut f32,
    pooled: *mut f32,
    max_cells: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_idx_append_steps(
        raw_k,
        pos_dev,
        n_steps,
        gamma,
        eps,
        freq_base,
        mrope_pos3_opt,
        tail,
        dead,
        pooled,
        max_cells,
        stream.raw,
    );
}

pub unsafe fn idx_append_batch(
    raw_k: *const f32,
    n_tokens: i32,
    p0: i32,
    gamma: *const f32,
    eps: f32,
    freq_base: f32,
    mrope_pos3_opt: *const i32,
    tail: *mut f32,
    dead: *mut f32,
    pooled: *mut f32,
    max_cells: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_idx_append_batch(
        raw_k,
        n_tokens,
        p0,
        gamma,
        eps,
        freq_base,
        mrope_pos3_opt,
        tail,
        dead,
        pooled,
        max_cells,
        stream.raw,
    );
}

pub unsafe fn idx_score_select(
    pooled: *const f32,
    dead: *const f32,
    q_idx: *const f32,
    pos_dev: *const i32,
    t: i32,
    max_blocks: i32,
    top_k_cap: i32,
    scores_ws: *mut f32,
    ids_out: *mut i32,
    width_out: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_idx_score_select(
        pooled,
        dead,
        q_idx,
        pos_dev,
        t,
        max_blocks,
        top_k_cap,
        scores_ws,
        ids_out,
        width_out,
        stream.raw,
    );
}

pub unsafe fn kv_append_q8(
    k: *const f32,
    v: *const f32,
    pos_dev: *const i32,
    t: i32,
    n_kv_heads: i32,
    head_dim: i32,
    k_q: *mut i8,
    v_q: *mut i8,
    k_sc: *mut u16,
    v_sc: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_kv_append_q8(
        k,
        v,
        pos_dev,
        t,
        n_kv_heads,
        head_dim,
        k_q,
        v_q,
        k_sc,
        v_sc,
        stream.raw,
    );
}

pub fn qsa_decode_ws_floats(t: i32, n_kv_heads: i32, max_n_ids: i32) -> usize {
    unsafe { ffi::qwen4_qsa_decode_ws_floats(t, n_kv_heads, max_n_ids) }
}

pub unsafe fn qsa_decode_attn_q8(
    q: *const f32,
    q_full_gate_opt: *const f32,
    k_q: *const i8,
    v_q: *const i8,
    k_sc: *const u16,
    v_sc: *const u16,
    ids_opt: *const i32,
    n_ids_dev: *const i32,
    max_n_ids: i32,
    id_cap: i32,
    t: i32,
    n_kv_heads: i32,
    scale: f32,
    ws_buf: *mut f32,
    out_f32_opt: *mut f32,
    out_q8_opt: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_qsa_decode_attn_q8(
        q,
        q_full_gate_opt,
        k_q,
        v_q,
        k_sc,
        v_sc,
        ids_opt,
        n_ids_dev,
        max_n_ids,
        id_cap,
        t,
        n_kv_heads,
        scale,
        ws_buf,
        out_f32_opt,
        out_q8_opt,
        stream.raw,
    );
}

pub unsafe fn qsa_prefill_attn_q8(
    q: *const f32,
    q_full_gate_opt: *const f32,
    k_q: *const i8,
    v_q: *const i8,
    k_sc: *const u16,
    v_sc: *const u16,
    ids_opt: *const i32,
    n_ids_dev: *const i32,
    id_cap: i32,
    t: i32,
    n_kv_heads: i32,
    scale: f32,
    out_f32: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_qsa_prefill_attn_q8(
        q,
        q_full_gate_opt,
        k_q,
        v_q,
        k_sc,
        v_sc,
        ids_opt,
        n_ids_dev,
        id_cap,
        t,
        n_kv_heads,
        scale,
        out_f32,
        stream.raw,
    );
}

// ---- Router & Shared Gate ----

pub unsafe fn router_top10(
    logits: *const f32,
    t: i32,
    n_exp: i32,
    topk: i32,
    ids_out: *mut i32,
    weights_out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_router_top10(logits, t, n_exp, topk, ids_out, weights_out, stream.raw);
}

pub unsafe fn shared_gate_dot(
    x: *const f32,
    w_sh_gate: *const c_void,
    w_type: i32,
    t: i32,
    n_embd: i32,
    out_logits: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_shared_gate_dot(x, w_sh_gate, w_type, t, n_embd, out_logits, stream.raw);
}

// ---- Vision Helpers ----

pub unsafe fn vis_rope2d(
    qkv: *mut f32,
    pos_yx: *const i32,
    n_patches: i32,
    heads: i32,
    hd: i32,
    base: f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_vis_rope2d(qkv, pos_yx, n_patches, heads, hd, base, stream.raw);
}

pub unsafe fn vis_merge_2x2_f16(
    x_ln: *const f32,
    n_merged: i32,
    d: i32,
    out_f16: *mut u16,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_vis_merge_2x2_f16(x_ln, n_merged, d, out_f16, stream.raw);
}

// ---- Phase-4 Decode/Verify & Fusion Helpers ----

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Qwen4Doorbell {
    pub req_seq: u64,
    pub done_seq: u64,
    pub layer_idx: i32,
    pub miss_count: i32,
    pub m_batch: i32,
    pub gu_type: i32,
    pub d_type: i32,
    pub n_embd: i32,
    pub n_ff: i32,
    pub _pad: i32,
    pub miss_tok: [i32; 80],
    pub miss_ids: [i32; 80],
    pub miss_weights: [f32; 80],
    pub x_in: [f32; 8 * 2560],
    pub out: [f32; 8 * 2560],
    pub routed_ids: [[i32; 80]; 48],
    pub routed_n_pairs: [i32; 48],
}

const _: () = assert!(std::mem::size_of::<Qwen4Doorbell>() == 180_416);

impl Default for Qwen4Doorbell {
    fn default() -> Self {
        Self {
            req_seq: 0,
            done_seq: 0,
            layer_idx: 0,
            miss_count: 0,
            m_batch: 1,
            gu_type: 0,
            d_type: 0,
            n_embd: 0,
            n_ff: 0,
            _pad: 0,
            miss_tok: [0; 80],
            miss_ids: [0; 80],
            miss_weights: [0.0; 80],
            x_in: [0.0; 8 * 2560],
            out: [0.0; 8 * 2560],
            routed_ids: [[-1; 80]; 48],
            routed_n_pairs: [0; 48],
        }
    }
}

pub unsafe fn fill_dense_widths(
    pos_dev: *const i32,
    width_out: *mut i32,
    t: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_fill_dense_widths(pos_dev, width_out, t, stream.raw);
}

pub unsafe fn hc_up_mix_q8(
    w_up_q8_0: *const c_void,
    lo_q8: *const c_void,
    xn: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    lr: i32,
    mixed_out: *mut f32,
    mixed_q8_opt: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_hc_up_mix_q8(
        w_up_q8_0,
        lo_q8,
        xn,
        t,
        hc,
        e,
        lr,
        mixed_out,
        mixed_q8_opt,
        stream.raw,
    );
}

pub unsafe fn router_doorbell_plan(
    logits: *const f32,
    t: i32,
    n_exp: i32,
    topk: i32,
    vram_ptrs: *const u64,
    ram_ptrs: *const u64,
    gpu_pcie_div: i32,
    gpu_pcie_max: i32,
    mixed_f32: *const f32,
    n_embd: i32,
    n_ff: i32,
    layer_idx: i32,
    gu_type: i32,
    d_type: i32,
    ids_out: *mut i32,
    weights_out: *mut f32,
    plan_ptrs_out: *mut u64,
    doorbell_mapped: *mut c_void,
    seq_state_dev: *mut u64,
    expected_seq_out: *mut u64,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_router_doorbell_plan(
        logits,
        t,
        n_exp,
        topk,
        vram_ptrs,
        ram_ptrs,
        gpu_pcie_div,
        gpu_pcie_max,
        mixed_f32,
        n_embd,
        n_ff,
        layer_idx,
        gu_type,
        d_type,
        ids_out,
        weights_out,
        plan_ptrs_out,
        doorbell_mapped,
        seq_state_dev,
        expected_seq_out,
        stream.raw,
    );
}

pub unsafe fn doorbell_wait_hc_write(
    doorbell_mapped: *const c_void,
    expected_seq_ptr: *const u64,
    r: *mut f32,
    block_out: *const f32,
    sh_out_opt: *const f32,
    sh_gate_logit_opt: *const f32,
    inject: *const f32,
    t: i32,
    hc: i32,
    e: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_doorbell_wait_hc_write(
        doorbell_mapped,
        expected_seq_ptr,
        r,
        block_out,
        sh_out_opt,
        sh_gate_logit_opt,
        inject,
        t,
        hc,
        e,
        stream.raw,
    );
}

pub unsafe fn doorbell_wait(
    doorbell_mapped: *const c_void,
    expected_seq_ptr: *const u64,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_doorbell_wait(doorbell_mapped, expected_seq_ptr, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_reduce_hc_write(
    doorbell_mapped: *const c_void,
    expected_seq_ptr: *const u64,
    r: *mut f32,
    block_out_opt: *const f32,
    pair_out_opt: *const f32,
    gpu_pair_mask_opt: *const u16,
    topk: i32,
    sh_out_opt: *const f32,
    sh_gate_logit_opt: *const f32,
    inject: *const f32,
    inject_stride: i32,
    t: i32,
    hc: i32,
    e: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_moe_reduce_hc_write(
        doorbell_mapped,
        expected_seq_ptr,
        r,
        block_out_opt,
        pair_out_opt,
        gpu_pair_mask_opt,
        topk,
        sh_out_opt,
        sh_gate_logit_opt,
        inject,
        inject_stride,
        t,
        hc,
        e,
        stream.raw,
    );
}

pub unsafe fn argmax_prob(
    logits: *const f32,
    n_vocab: i32,
    partial_max: *mut f32,
    partial_idx: *mut i32,
    partial_sum: *mut f32,
    out_id: *mut i32,
    out_prob: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_argmax_prob(
        logits,
        n_vocab,
        partial_max,
        partial_idx,
        partial_sum,
        out_id,
        out_prob,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn argmax_prob_indexed(
    logits: *const f32,
    n_vocab: i32,
    sub_to_id_opt: *const i32,
    partial_max: *mut f32,
    partial_idx: *mut i32,
    partial_sum: *mut f32,
    out_id: *mut i32,
    out_prob: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_argmax_prob_indexed(
        logits,
        n_vocab,
        sub_to_id_opt,
        partial_max,
        partial_idx,
        partial_sum,
        out_id,
        out_prob,
        stream.raw,
    );
}

pub unsafe fn moe_fetch_unique(
    fetch_src_ptrs: *const u64,
    fetch_dst_ptrs: *const u64,
    blob_bytes: usize,
    fetch_cap: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_moe_fetch_unique(
        fetch_src_ptrs,
        fetch_dst_ptrs,
        blob_bytes,
        fetch_cap,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn router_doorbell_plan_dedup(
    logits: *const f32,
    t: i32,
    n_exp: i32,
    topk: i32,
    vram_ptrs: *const u64,
    ram_ptrs: *const u64,
    gpu_pcie_div: i32,
    fetch_cap: i32,
    dma_stage_base: u64,
    slot_stride: usize,
    mixed_f32: *const f32,
    n_embd: i32,
    n_ff: i32,
    layer_idx: i32,
    gu_type: i32,
    d_type: i32,
    ids_out: *mut i32,
    weights_out: *mut f32,
    res_ptrs_out: *mut u64,
    res_off_out: *mut i32,
    res_nt_out: *mut i32,
    res_rows_out: *mut i32,
    res_pair_out: *mut i32,
    res_wts_out: *mut f32,
    fetch_src_out: *mut u64,
    fetch_dst_out: *mut u64,
    fetch_off_out: *mut i32,
    fetch_nt_out: *mut i32,
    fetch_rows_out: *mut i32,
    fetch_pair_out: *mut i32,
    fetch_wts_out: *mut f32,
    gpu_pair_mask_out: *mut u16,
    doorbell_mapped: *mut c_void,
    seq_state_dev: *mut u64,
    expected_seq_out: *mut u64,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_router_doorbell_plan_dedup(
        logits,
        t,
        n_exp,
        topk,
        vram_ptrs,
        ram_ptrs,
        gpu_pcie_div,
        fetch_cap,
        dma_stage_base,
        slot_stride,
        mixed_f32,
        n_embd,
        n_ff,
        layer_idx,
        gu_type,
        d_type,
        ids_out,
        weights_out,
        res_ptrs_out,
        res_off_out,
        res_nt_out,
        res_rows_out,
        res_pair_out,
        res_wts_out,
        fetch_src_out,
        fetch_dst_out,
        fetch_off_out,
        fetch_nt_out,
        fetch_rows_out,
        fetch_pair_out,
        fetch_wts_out,
        gpu_pair_mask_out,
        doorbell_mapped,
        seq_state_dev,
        expected_seq_out,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn mmvq_q8_0_multi2(
    w0: *const c_void,
    y0: *mut f32,
    n0: i32,
    w1: *const c_void,
    y1: *mut f32,
    n1: i32,
    x_q8: *const c_void,
    k: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_mmvq_q8_0_multi2(w0, y0, n0, w1, y1, n1, x_q8, k, m, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn mmvq_q8_0_multi3(
    w0: *const c_void,
    y0: *mut f32,
    n0: i32,
    w1: *const c_void,
    y1: *mut f32,
    n1: i32,
    w2: *const c_void,
    y2: *mut f32,
    n2: i32,
    x_q8: *const c_void,
    k: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_mmvq_q8_0_multi3(
        w0, y0, n0, w1, y1, n1, w2, y2, n2, x_q8, k, m, stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn sh_gu_swiglu_q8(
    w_gate: *const c_void,
    w_up: *const c_void,
    x_q8: *const c_void,
    y_act: *mut f32,
    n: i32,
    k: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_sh_gu_swiglu_q8(w_gate, w_up, x_q8, y_act, n, k, m, stream.raw);
}

pub unsafe fn mmvq_q8_0_indexed_m1(
    w: *const c_void,
    subset_ids: *const i32,
    x_q8: *const c_void,
    y: *mut f32,
    n_sub: i32,
    k: i32,
    stream: &CudaStream,
) {
    ffi::launch_mmvq_q8_0_indexed_m1(w, subset_ids, x_q8, y, n_sub, k, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_steps_split_norm_q8(
    state: *mut f32,
    h_qkv: *const f32,
    gate: *const f32,
    beta: *const f32,
    z_gate: *const f32,
    ssm_norm_w: *const f32,
    eps: f32,
    y_ws_f32: *mut f32,
    y_f32_opt: *mut f32,
    y_q8_opt: *mut c_void,
    state_before_opt: *mut f32,
    t: i32,
    h_k: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_steps_split_norm_q8(
        state,
        h_qkv,
        gate,
        beta,
        z_gate,
        ssm_norm_w,
        eps,
        y_ws_f32,
        y_f32_opt,
        y_q8_opt,
        state_before_opt,
        t,
        h_k,
        h_v,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_replay_steps(
    state_before: *const f32,
    state_out: *mut f32,
    h_qkv: *const f32,
    gate: *const f32,
    beta: *const f32,
    n_steps: i32,
    h_k: i32,
    h_v: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_gdn_replay_steps(
        state_before,
        state_out,
        h_qkv,
        gate,
        beta,
        n_steps,
        h_k,
        h_v,
        stream.raw,
    );
}

pub unsafe fn reduce_pairs_f32(
    pair_out: *const f32,
    block_out: *mut f32,
    t: i32,
    topk: i32,
    n_embd: i32,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_reduce_pairs_f32(pair_out, block_out, t, topk, n_embd, stream.raw);
}

pub unsafe fn patch_u64(mapped_pairs: *const u64, n: i32, stream: &CudaStream) {
    ffi::launch_qwen4_patch_u64(mapped_pairs, n, stream.raw);
}

pub unsafe fn copy_u32_sm(
    dst: *mut c_void,
    src: *const c_void,
    n_u32: i32,
    system_fence: bool,
    stream: &CudaStream,
) {
    ffi::launch_qwen4_copy_u32_sm(dst, src, n_u32, i32::from(system_fence), stream.raw);
}
