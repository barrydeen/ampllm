use crate::{ffi, CudaStream};
use std::ffi::c_void;

/// Shared mapped host-device doorbell structure for overlapping GPU dense/resident work
/// with 15-core AVX-512 CPU expert execution without D2H sync overhead.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Glm5Doorbell {
    pub req_seq: u64,
    pub done_seq: u64,
    pub layer_idx: i32,
    pub miss_count: i32,
    pub m_batch: i32,
    pub gu_type: i32,
    pub d_type: i32,
    pub n_embd: i32,
    pub n_ff: i32,
    pub clamp_lim: f32,
    pub cancel_flag: i32,
    pub x_q8k_valid: i32,
    pub u_off: u32,
    pub d_off: u32,
    pub x_q8k: [u8; 16 * 292],
    pub miss_tok: [i32; 32],
    pub miss_ids: [i32; 32],
    pub miss_weights: [f32; 32],
    pub x_in: [f32; 16384],
    pub out: [f32; 16384],
    pub routed_ids: [[i32; 32]; 48],
    pub promo_eid: [[i32; 8]; 48],
    pub promo_vram_ptr: [[u64; 8]; 48],
    pub pci_fetch_total: u64,
}

impl Default for Glm5Doorbell {
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
            clamp_lim: 10.0,
            cancel_flag: 0,
            x_q8k_valid: 0,
            u_off: 0,
            d_off: 0,
            x_q8k: [0; 16 * 292],
            miss_tok: [0; 32],
            miss_ids: [0; 32],
            miss_weights: [0.0; 32],
            x_in: [0.0; 16384],
            out: [0.0; 16384],
            routed_ids: [[-1; 32]; 48],
            promo_eid: [[-1; 8]; 48],
            promo_vram_ptr: [[0; 8]; 48],
            pci_fetch_total: 0,
        }
    }
}

pub unsafe fn skip_gate(doorbell_mapped: *const c_void, skip_dev: *mut i32, stream: &CudaStream) {
    ffi::launch_glm5_skip_gate(doorbell_mapped, skip_dev, stream.raw);
}

pub unsafe fn linear_q8_1_skip(
    w_type: i32,
    w: *const c_void,
    x_q8: *const c_void,
    x_f32: *const f32,
    y: *mut f32,
    n: i32,
    k: i32,
    m: i32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_linear_q8_1_skip(w_type, w, x_q8, x_f32, y, n, k, m, skip_dev, stream.raw);
}

pub unsafe fn moe_resident_skip(
    gu_type: i32,
    d_type: i32,
    gate_ptrs: *const u64,
    up_ptrs: *const u64,
    down_ptrs: *const u64,
    gu_row_bytes: usize,
    d_row_bytes: usize,
    x_q8: *const c_void,
    expert_ids: *const i32,
    router_w: *const f32,
    inter_q8: *mut c_void,
    inter_f32_opt: *mut f32,
    out_f32: *mut f32,
    n_embd: i32,
    n_ff: i32,
    topk: i32,
    m: i32,
    clamp_lim: f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_resident(
        gu_type, d_type, gate_ptrs, up_ptrs, down_ptrs, gu_row_bytes, d_row_bytes, x_q8,
        expert_ids, router_w, inter_q8, inter_f32_opt, out_f32, n_embd, n_ff, topk, m, clamp_lim,
        skip_dev, stream.raw,
    );
}

pub unsafe fn moe_resident(
    gu_type: i32,
    d_type: i32,
    gate_ptrs: *const u64,
    up_ptrs: *const u64,
    down_ptrs: *const u64,
    gu_row_bytes: usize,
    d_row_bytes: usize,
    x_q8: *const c_void,
    expert_ids: *const i32,
    router_w: *const f32,
    inter_q8: *mut c_void,
    inter_f32_opt: *mut f32,
    out_f32: *mut f32,
    n_embd: i32,
    n_ff: i32,
    topk: i32,
    m: i32,
    clamp_lim: f32,
    stream: &CudaStream,
) {
    moe_resident_skip(
        gu_type, d_type, gate_ptrs, up_ptrs, down_ptrs, gu_row_bytes, d_row_bytes, x_q8,
        expert_ids, router_w, inter_q8, inter_f32_opt, out_f32, n_embd, n_ff, topk, m, clamp_lim,
        std::ptr::null(), stream,
    );
}

pub unsafe fn moe_plan(
    gu_type: i32,
    d_type: i32,
    plan_ptrs: *const u64,
    u_off: u32,
    d_off: u32,
    gu_row_bytes: usize,
    d_row_bytes: usize,
    x_q8: *const c_void,
    router_w: *const f32,
    inter_q8: *mut c_void,
    sh_down_type: i32,
    sh_down_w: *const c_void,
    sh_down_row_bytes: usize,
    sh_hq: *const c_void,
    n_ff_sh: i32,
    sh_out_opt: *mut f32,
    out_f32: *mut f32,
    n_embd: i32,
    n_ff: i32,
    topk: i32,
    m: i32,
    clamp_lim: f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_plan(
        gu_type, d_type, plan_ptrs, u_off, d_off, gu_row_bytes, d_row_bytes, x_q8, router_w,
        inter_q8, sh_down_type, sh_down_w, sh_down_row_bytes, sh_hq, n_ff_sh, sh_out_opt,
        out_f32, n_embd, n_ff, topk, m, clamp_lim, skip_dev, stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_plan_filtered(
    gu_type: i32,
    d_type: i32,
    plan_ptrs: *const u64,
    u_off: u32,
    d_off: u32,
    gu_row_bytes: usize,
    d_row_bytes: usize,
    x_q8: *const c_void,
    router_w: *const f32,
    inter_q8: *mut c_void,
    sh_down_type: i32,
    sh_down_w: *const c_void,
    sh_down_row_bytes: usize,
    sh_hq: *const c_void,
    n_ff_sh: i32,
    sh_out_opt: *mut f32,
    out_f32: *mut f32,
    n_embd: i32,
    n_ff: i32,
    topk: i32,
    m: i32,
    clamp_lim: f32,
    skip_dev: *const i32,
    fetch_src_filter: *const u64,
    filter_mode: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_plan_filtered(
        gu_type, d_type, plan_ptrs, u_off, d_off, gu_row_bytes, d_row_bytes, x_q8, router_w,
        inter_q8, sh_down_type, sh_down_w, sh_down_row_bytes, sh_hq, n_ff_sh, sh_out_opt,
        out_f32, n_embd, n_ff, topk, m, clamp_lim, skip_dev, fetch_src_filter, filter_mode,
        stream.raw,
    );
}

pub unsafe fn router_dual_lane(
    logits: *const f32,
    bias: *const f32,
    n_expert: i32,
    topk: i32,
    w_scale: f32,
    norm_w: bool,
    vram_ptrs: *mut u64,
    ram_ptrs: *const u64,
    spares: *mut u64,
    k_spares: i32,
    dcnt: *mut u32,
    route_cnt_dev: *mut u32,
    scratch_ptrs: *const u64,
    cpu_plan: u64,
    promote_min: i32,
    skip_from: i32,
    plan_ptrs_out: *mut u64,
    fetch_src_out: *mut u64,
    x_f32: *const f32,
    n_embd: i32,
    n_ff: i32,
    layer_idx: i32,
    gu_type: i32,
    d_type: i32,
    clamp_lim: f32,
    u_off: u32,
    d_off: u32,
    sh_gate: *const f32,
    sh_up: *const f32,
    sh_clamp_lim: f32,
    n_ff_sh: i32,
    sh_hq: *mut c_void,
    ids_out: *mut i32,
    rw_out: *mut f32,
    doorbell_mapped: *mut c_void,
    seq_state_dev: *mut u64,
    expected_seq_out: *mut u64,
    pos_arr_dev: *const i32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_router_dual_lane(
        logits, bias, n_expert, topk, w_scale, i32::from(norm_w), vram_ptrs, ram_ptrs, spares,
        k_spares, dcnt, route_cnt_dev, scratch_ptrs, cpu_plan, promote_min, skip_from,
        plan_ptrs_out, fetch_src_out, x_f32, n_embd, n_ff, layer_idx, gu_type, d_type, clamp_lim,
        u_off, d_off, sh_gate, sh_up, sh_clamp_lim, n_ff_sh, sh_hq, ids_out, rw_out,
        doorbell_mapped, seq_state_dev, expected_seq_out, pos_arr_dev, skip_dev, stream.raw,
    );
}

pub unsafe fn moe_fetch(
    fetch_src: *const u64,
    plan_ptrs: *const u64,
    topk: i32,
    blob_bytes: usize,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_fetch(fetch_src, plan_ptrs, topk, blob_bytes, stream.raw);
}

pub unsafe fn moe_fetch_nblocks(
    fetch_src: *const u64,
    plan_ptrs: *const u64,
    topk: i32,
    blob_bytes: usize,
    n_blocks: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_fetch_nblocks(fetch_src, plan_ptrs, topk, blob_bytes, n_blocks, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn router_dual_lane_pf(
    logits: *const f32,
    bias: *const f32,
    n_expert: i32,
    topk: i32,
    w_scale: f32,
    norm_w: bool,
    vram_ptrs: *mut u64,
    ram_ptrs: *const u64,
    spares: *mut u64,
    k_spares: i32,
    dcnt: *mut u32,
    route_cnt_dev: *mut u32,
    scratch_ptrs: *const u64,
    cpu_plan: u64,
    promote_min: i32,
    skip_from: i32,
    plan_ptrs_out: *mut u64,
    fetch_src_out: *mut u64,
    x_f32: *const f32,
    n_embd: i32,
    n_ff: i32,
    layer_idx: i32,
    gu_type: i32,
    d_type: i32,
    clamp_lim: f32,
    u_off: u32,
    d_off: u32,
    sh_gate: *const f32,
    sh_up: *const f32,
    sh_clamp_lim: f32,
    n_ff_sh: i32,
    sh_hq: *mut c_void,
    ids_out: *mut i32,
    rw_out: *mut f32,
    doorbell_mapped: *mut c_void,
    seq_state_dev: *mut u64,
    expected_seq_out: *mut u64,
    pos_arr_dev: *const i32,
    skip_dev: *const i32,
    pf_in_eids: *const i32,
    pf_in_ptrs: *const u64,
    pf_logits_next: *const f32,
    pf_bias_next: *const f32,
    vram_ptrs_next: *const u64,
    ram_ptrs_next: *const u64,
    max_pf: i32,
    pf_out_src: *mut u64,
    pf_out_dst: *mut u64,
    pf_out_eids: *mut i32,
    pf_out_ptrs: *mut u64,
    pf_out_n: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_router_dual_lane_pf(
        logits, bias, n_expert, topk, w_scale, i32::from(norm_w), vram_ptrs, ram_ptrs, spares,
        k_spares, dcnt, route_cnt_dev, scratch_ptrs, cpu_plan, promote_min, skip_from,
        plan_ptrs_out, fetch_src_out, x_f32, n_embd, n_ff, layer_idx, gu_type, d_type, clamp_lim,
        u_off, d_off, sh_gate, sh_up, sh_clamp_lim, n_ff_sh, sh_hq, ids_out, rw_out,
        doorbell_mapped, seq_state_dev, expected_seq_out, pos_arr_dev, skip_dev, pf_in_eids,
        pf_in_ptrs, pf_logits_next, pf_bias_next, vram_ptrs_next, ram_ptrs_next, max_pf,
        pf_out_src, pf_out_dst, pf_out_eids, pf_out_ptrs, pf_out_n, stream.raw,
    );
}

pub unsafe fn moe_prefetch(
    pf_src: *const u64,
    pf_dst: *const u64,
    pf_n: *const i32,
    blob_bytes: usize,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_prefetch(pf_src, pf_dst, pf_n, blob_bytes, stream.raw);
}

pub unsafe fn mv_batch_skip(
    w_ptrs: &[*const c_void],
    y_ptrs: &[*mut f32],
    w_types: &[i32],
    n_outs: &[i32],
    scales: Option<&[f32]>,
    k: i32,
    x_q8: *const c_void,
    x_f32: *const f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    debug_assert_eq!(w_ptrs.len(), y_ptrs.len());
    debug_assert_eq!(w_ptrs.len(), w_types.len());
    debug_assert_eq!(w_ptrs.len(), n_outs.len());
    let sc_ptr = scales.map_or(std::ptr::null(), |s| s.as_ptr());
    ffi::launch_glm5_mv_batch(
        w_ptrs.as_ptr(), y_ptrs.as_ptr(), w_types.as_ptr(), n_outs.as_ptr(), sc_ptr,
        w_ptrs.len() as i32, k, x_q8, x_f32, skip_dev, stream.raw,
    );
}

pub unsafe fn mv_batch(
    w_ptrs: &[*const c_void],
    y_ptrs: &[*mut f32],
    w_types: &[i32],
    n_outs: &[i32],
    scales: Option<&[f32]>,
    k: i32,
    x_q8: *const c_void,
    x_f32: *const f32,
    stream: &CudaStream,
) {
    mv_batch_skip(w_ptrs, y_ptrs, w_types, n_outs, scales, k, x_q8, x_f32, std::ptr::null(), stream);
}

pub unsafe fn embed_replicate(
    emb: *const f32,
    r_out: *mut f32,
    n_embd: i32,
    hc: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_embed_replicate(emb, r_out, n_embd, hc, m, stream.raw);
}

pub unsafe fn hc_pre_norm_q8(
    r_in: *const f32,
    w_fn: *const c_void,
    w_fn_type: i32,
    w_scale: *const f32,
    w_base: *const f32,
    norm_w: *const f32,
    n_embd: i32,
    hc: i32,
    iters: i32,
    norm_eps: f32,
    hc_eps: f32,
    post_out: *mut f32,
    comb_out: *mut f32,
    x_f32_out: *mut f32,
    x_q8_out: *mut c_void,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_hc_pre_norm_q8(
        r_in, w_fn, w_fn_type, w_scale, w_base, norm_w, n_embd, hc, iters,
        norm_eps, hc_eps, post_out, comb_out, x_f32_out, x_q8_out, m, stream.raw,
    );
}

pub unsafe fn hc_post_add(
    r_in: *const f32,
    post: *const f32,
    comb: *const f32,
    a: *const f32,
    b_opt: *const f32,
    c_opt: *const f32,
    r_out: *mut f32,
    n_embd: i32,
    hc: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_hc_post_add(
        r_in, post, comb, a, b_opt, c_opt, r_out, n_embd, hc, m, stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_fused_m1(
    block_out: *const f32,
    r_old: *const f32,
    r_new: *mut f32,
    post_in: *const f32,
    comb_in: *const f32,
    post_out: *mut f32,
    comb_out: *mut f32,
    w_fn: *const c_void,
    w_fn_type: i32,
    w_scale: *const f32,
    w_base: *const f32,
    norm_w: *const f32,
    n_embd: i32,
    hc: i32,
    iters: i32,
    norm_eps: f32,
    hc_eps: f32,
    x_f32_out: *mut f32,
    x_q8_out: *mut c_void,
    hc_part: *mut f32,
    hc_ctr: *mut u64,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_hc_fused_m1(
        block_out, r_old, r_new, post_in, comb_in, post_out, comb_out, w_fn, w_fn_type,
        w_scale, w_base, norm_w, n_embd, hc, iters, norm_eps, hc_eps, x_f32_out,
        x_q8_out, hc_part, hc_ctr, skip_dev, stream.raw,
    );
}

pub unsafe fn mean_streams_norm_q8(
    r_in: *const f32,
    norm_w: *const f32,
    eps: f32,
    n_embd: i32,
    hc: i32,
    out_f32: *mut f32,
    out_q8: *mut c_void,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_mean_streams_norm_q8(
        r_in, norm_w, eps, n_embd, hc, out_f32, out_q8, m, stream.raw,
    );
}

pub unsafe fn rms_norm_q8(
    base_res: *mut f32,
    add_opt: *const f32,
    norm_w: *const f32,
    eps: f32,
    dim: i32,
    write_back_base: bool,
    out_f32: *mut f32,
    out_q8: *mut c_void,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_rms_norm_q8(
        base_res, add_opt, norm_w, eps, dim, i32::from(write_back_base), out_f32, out_q8, m,
        stream.raw,
    );
}

pub unsafe fn swiglu_clamp_q8(
    gate: *const f32,
    up: *const f32,
    lim: f32,
    n_ff: i32,
    out_f32_opt: *mut f32,
    out_q8: *mut c_void,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_swiglu_clamp_q8(gate, up, lim, n_ff, out_f32_opt, out_q8, m, stream.raw);
}

pub unsafe fn kda_conv_silu_l2(
    x_proj: *const f32,
    conv_w: *const f32,
    conv_state: *mut f32,
    d_inner: i32,
    d_conv: i32,
    head_dim: i32,
    n_head: i32,
    apply_l2: bool,
    t: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_kda_conv_silu_l2(
        x_proj, conv_w, conv_state, d_inner, d_conv, head_dim, n_head,
        i32::from(apply_l2), t, out, stream.raw,
    );
}

pub unsafe fn kda_gate_beta(
    g_in: *const f32,
    dt_bias: *const f32,
    ssm_a: *const f32,
    lower_bound: f32,
    head_dim: i32,
    n_head: i32,
    t: i32,
    g1_out: *mut f32,
    beta_inout: *mut f32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_kda_gate_beta(
        g_in, dt_bias, ssm_a, lower_bound, head_dim, n_head, t, g1_out, beta_inout, stream.raw,
    );
}

pub unsafe fn prepare_kda_smem(head_dim: i32) {
    ffi::launch_glm5_prepare_kda_smem(head_dim);
}

pub unsafe fn kda_recurrence_skip(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g1: *const f32,
    beta: *const f32,
    state: *mut f32,
    n_head: i32,
    head_dim: i32,
    t: i32,
    out: *mut f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_kda_recurrence(
        q, k, v, g1, beta, state, n_head, head_dim, t, out, skip_dev, stream.raw,
    );
}

pub unsafe fn kda_recurrence(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g1: *const f32,
    beta: *const f32,
    state: *mut f32,
    n_head: i32,
    head_dim: i32,
    t: i32,
    out: *mut f32,
    stream: &CudaStream,
) {
    kda_recurrence_skip(q, k, v, g1, beta, state, n_head, head_dim, t, out, std::ptr::null(), stream);
}

pub unsafe fn kda_out_gate_q8(
    o: *const f32,
    norm_w: *const f32,
    g2: *const f32,
    head_dim: i32,
    n_head: i32,
    t: i32,
    eps: f32,
    gated_f32: *mut f32,
    gated_q8: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_kda_out_gate_q8(
        o, norm_w, g2, head_dim, n_head, t, eps, gated_f32, gated_q8, stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_prep_m1(
    q_proj: *const f32,
    k_proj: *const f32,
    v_proj: *const f32,
    conv_w_q: *const f32,
    conv_w_k: *const f32,
    conv_w_v: *const f32,
    conv_st_q: *mut f32,
    conv_st_k: *mut f32,
    conv_st_v: *mut f32,
    q_out: *mut f32,
    k_out: *mut f32,
    v_out: *mut f32,
    f_a: *const f32,
    g_a: *const f32,
    f_b: *const c_void,
    g_b: *const c_void,
    dt_bias: *const f32,
    ssm_a: *const f32,
    lower_bound: f32,
    n_head: i32,
    g1_out: *mut f32,
    g2_out: *mut f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_kda_prep_m1(
        q_proj, k_proj, v_proj, conv_w_q, conv_w_k, conv_w_v, conv_st_q, conv_st_k,
        conv_st_v, q_out, k_out, v_out, f_a, g_a, f_b, g_b, dt_bias, ssm_a, lower_bound,
        n_head, g1_out, g2_out, skip_dev, stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_rec_gate_q8_m1(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g1: *const f32,
    beta_raw: *const f32,
    state: *mut f32,
    g2: *const f32,
    norm_w: *const f32,
    eps: f32,
    n_head: i32,
    gated_f32_opt: *mut f32,
    gated_q8: *mut c_void,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_kda_rec_gate_q8_m1(
        q, k, v, g1, beta_raw, state, g2, norm_w, eps, n_head, gated_f32_opt, gated_q8,
        skip_dev, stream.raw,
    );
}

pub unsafe fn dsa_prep_kv_ik(
    kv_raw: *const f32,
    kv_norm_w: *const f32,
    vram_lat8: *mut c_void,
    ram_lat8: *mut c_void,
    res_cap: i32,
    ik_raw: *const f32,
    ik_norm_w: *const f32,
    ik_norm_b: *const f32,
    ik_cache: *mut f32,
    ig_raw: *const f32,
    ig_cache: *mut f32,
    kv_lora: i32,
    idx_key: i32,
    pos0: i32,
    ring_cap: i32,
    eps: f32,
    t: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_prep_kv_ik(
        kv_raw, kv_norm_w, vram_lat8, ram_lat8, res_cap,
        ik_raw, ik_norm_w, ik_norm_b, ik_cache, ig_raw, ig_cache,
        kv_lora, idx_key, pos0, ring_cap, eps, t, stream.raw,
    );
}

pub unsafe fn dsa_prep_kv_ik_dyn(
    kv_raw: *const f32,
    kv_norm_w: *const f32,
    vram_lat8: *mut c_void,
    ram_lat8: *mut c_void,
    res_cap: i32,
    ik_raw: *const f32,
    ik_norm_w: *const f32,
    ik_norm_b: *const f32,
    ik_cache: *mut f32,
    ig_raw: *const f32,
    ig_cache: *mut f32,
    kv_lora: i32,
    idx_key: i32,
    pos_arr: *const i32,
    ring_cap: i32,
    eps: f32,
    t: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_prep_kv_ik_dyn(
        kv_raw, kv_norm_w, vram_lat8, ram_lat8, res_cap,
        ik_raw, ik_norm_w, ik_norm_b, ik_cache, ig_raw, ig_cache,
        kv_lora, idx_key, pos_arr, ring_cap, eps, t, stream.raw,
    );
}

pub unsafe fn dsa_pool(
    ik_cache: *const f32,
    ig_cache: *const f32,
    ape: *const f32,
    key_dim: i32,
    kpool: i32,
    pool0: i32,
    n_new_pools: i32,
    ring_cap: i32,
    pooled_f16: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_pool(
        ik_cache, ig_cache, ape, key_dim, kpool, pool0, n_new_pools, ring_cap,
        pooled_f16, stream.raw,
    );
}

pub unsafe fn dsa_pool_m1_dyn(
    ik_cache: *const f32,
    ig_cache: *const f32,
    ape: *const f32,
    key_dim: i32,
    kpool: i32,
    pos_arr: *const i32,
    ring_cap: i32,
    pooled_f16: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_pool_m1_dyn(
        ik_cache, ig_cache, ape, key_dim, kpool, pos_arr, ring_cap, pooled_f16, stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_prep_m1_dyn(
    qr_inout: *mut f32,
    q_norm_w: *const f32,
    qr_q8_out: *mut c_void,
    q_lora: i32,
    kv_raw: *const f32,
    kv_norm_w: *const f32,
    vram_lat8: *mut c_void,
    ram_lat8: *mut c_void,
    res_cap: i32,
    kv_lora: i32,
    ik_raw: *const f32,
    ik_norm_w: *const f32,
    ik_norm_b: *const f32,
    ik_cache: *mut f32,
    ig_raw: *const f32,
    ig_cache: *mut f32,
    ape: *const f32,
    idx_key: i32,
    kpool: i32,
    ring_cap: i32,
    pos_arr: *const i32,
    pooled_f16: *mut c_void,
    eps: f32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_prep_m1_dyn(
        qr_inout, q_norm_w, qr_q8_out, q_lora, kv_raw, kv_norm_w, vram_lat8, ram_lat8,
        res_cap, kv_lora, ik_raw, ik_norm_w, ik_norm_b, ik_cache, ig_raw, ig_cache, ape,
        idx_key, kpool, ring_cap, pos_arr, pooled_f16, eps, skip_dev, stream.raw,
    );
}

pub unsafe fn dsa_q_absorb(
    wk_b: *const c_void,
    wk_type: i32,
    q: *const f32,
    q_abs: *mut f32,
    qk_nope: i32,
    kv_lora: i32,
    n_head: i32,
    t: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_q_absorb(
        wk_b, wk_type, q, q_abs, qk_nope, kv_lora, n_head, t, stream.raw,
    );
}

pub unsafe fn dsa_score_select(
    iq: *const f32,
    pooled_f16: *const c_void,
    weights: *const f32,
    key_dim: i32,
    idx_heads: i32,
    kpool: i32,
    top_pools_cap: i32,
    select_tail: bool,
    n_sel: i32,
    pos_arr: *const i32,
    scores_buf: *mut f32,
    cells_out: *mut i32,
    t: i32,
    cur_pool_done: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_score_select(
        iq, pooled_f16, weights, key_dim, idx_heads, kpool, top_pools_cap,
        i32::from(select_tail), n_sel, pos_arr, scores_buf, cells_out, t,
        cur_pool_done, stream.raw,
    );
}

pub unsafe fn dsa_score_select_dyn(
    iq: *const f32,
    pooled_f16: *const c_void,
    weights: *const f32,
    key_dim: i32,
    idx_heads: i32,
    kpool: i32,
    top_pools_cap: i32,
    select_tail: bool,
    n_sel: i32,
    pool_cap: i32,
    pos_arr: *const i32,
    scores_buf: *mut f32,
    cells_out: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_score_select_dyn(
        iq, pooled_f16, weights, key_dim, idx_heads, kpool, top_pools_cap,
        i32::from(select_tail), n_sel, pool_cap, pos_arr, scores_buf, cells_out, stream.raw,
    );
}

pub unsafe fn dsa_mla_attn_skip(
    q_abs: *const f32,
    vram_lat8: *const c_void,
    ram_lat8: *const c_void,
    res_cap: i32,
    cells: *const i32,
    wv_b: *const c_void,
    wv_type: i32,
    kv_lora: i32,
    n_head: i32,
    v_head: i32,
    qk_nope: i32,
    n_sel: i32,
    out: *mut f32,
    t: i32,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_mla_attn(
        q_abs, vram_lat8, ram_lat8, res_cap, cells, wv_b, wv_type,
        kv_lora, n_head, v_head, qk_nope, n_sel, out, t, skip_dev, stream.raw,
    );
}

pub unsafe fn dsa_mla_attn(
    q_abs: *const f32,
    vram_lat8: *const c_void,
    ram_lat8: *const c_void,
    res_cap: i32,
    cells: *const i32,
    wv_b: *const c_void,
    wv_type: i32,
    kv_lora: i32,
    n_head: i32,
    v_head: i32,
    qk_nope: i32,
    n_sel: i32,
    out: *mut f32,
    t: i32,
    stream: &CudaStream,
) {
    dsa_mla_attn_skip(
        q_abs, vram_lat8, ram_lat8, res_cap, cells, wv_b, wv_type,
        kv_lora, n_head, v_head, qk_nope, n_sel, out, t, std::ptr::null(), stream,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_mla_vproj_q8_m1(
    q_abs: *const f32,
    vram_lat8: *const c_void,
    ram_lat8: *const c_void,
    res_cap: i32,
    cells: *const i32,
    wv_b: *const c_void,
    wv_type: i32,
    kv_lora: i32,
    n_head: i32,
    v_head: i32,
    qk_nope: i32,
    n_sel: i32,
    ctx_scratch: *mut f32,
    split_scratch_opt: *mut f32,
    out_f32_opt: *mut f32,
    out_q8: *mut c_void,
    skip_dev: *const i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_mla_vproj_q8_m1(
        q_abs, vram_lat8, ram_lat8, res_cap, cells, wv_b, wv_type, kv_lora, n_head,
        v_head, qk_nope, n_sel, ctx_scratch, split_scratch_opt, out_f32_opt, out_q8,
        skip_dev, stream.raw,
    );
}

pub unsafe fn router_sigmoid_probe_skip(
    logits: *const f32,
    bias: *const f32,
    n_expert: i32,
    topk: i32,
    w_scale: f32,
    norm_w: bool,
    res_mask: *const u8,
    x_f32: *const f32,
    n_embd: i32,
    n_ff: i32,
    layer_idx: i32,
    gu_type: i32,
    d_type: i32,
    clamp_lim: f32,
    ids_out: *mut i32,
    rw_out: *mut f32,
    doorbell_mapped: *mut c_void,
    seq_state_dev: *mut u64,
    expected_seq_out: *mut u64,
    skip_dev: *const i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_router_sigmoid_probe(
        logits, bias, n_expert, topk, w_scale, i32::from(norm_w), res_mask,
        x_f32, n_embd, n_ff, layer_idx, gu_type, d_type, clamp_lim,
        ids_out, rw_out, doorbell_mapped, seq_state_dev, expected_seq_out, skip_dev, m, stream.raw,
    );
}

pub unsafe fn router_sigmoid_probe(
    logits: *const f32,
    bias: *const f32,
    n_expert: i32,
    topk: i32,
    w_scale: f32,
    norm_w: bool,
    res_mask: *const u8,
    x_f32: *const f32,
    n_embd: i32,
    n_ff: i32,
    layer_idx: i32,
    gu_type: i32,
    d_type: i32,
    clamp_lim: f32,
    ids_out: *mut i32,
    rw_out: *mut f32,
    doorbell_mapped: *mut c_void,
    seq_state_dev: *mut u64,
    expected_seq_out: *mut u64,
    m: i32,
    stream: &CudaStream,
) {
    router_sigmoid_probe_skip(
        logits, bias, n_expert, topk, w_scale, norm_w, res_mask,
        x_f32, n_embd, n_ff, layer_idx, gu_type, d_type, clamp_lim,
        ids_out, rw_out, doorbell_mapped, seq_state_dev, expected_seq_out,
        std::ptr::null(), m, stream,
    );
}

pub unsafe fn doorbell_wait_hc_post(
    doorbell_mapped: *const c_void,
    expected_seq_ptr: *const u64,
    r_in: *const f32,
    post: *const f32,
    comb: *const f32,
    sh_out: *const f32,
    gpu_moe: *const f32,
    r_out: *mut f32,
    n_embd: i32,
    hc: i32,
    m: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_doorbell_wait_hc_post(
        doorbell_mapped, expected_seq_ptr, r_in, post, comb, sh_out, gpu_moe,
        r_out, n_embd, hc, m, stream.raw,
    );
}

pub unsafe fn doorbell_wait_add(
    doorbell_mapped: *const c_void,
    expected_seq_ptr: *const u64,
    inout_ffn: *mut f32,
    n_embd: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_doorbell_wait_add(
        doorbell_mapped, expected_seq_ptr, inout_ffn, n_embd, stream.raw,
    );
}

pub unsafe fn mtp_in_q8(
    emb: *const f32,
    h: *const f32,
    enorm: *const f32,
    hnorm: *const f32,
    eps: f32,
    n_embd: i32,
    cat_q8: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_mtp_in_q8(emb, h, enorm, hnorm, eps, n_embd, cat_q8, stream.raw);
}

pub unsafe fn argmax(
    logits: *const f32,
    n_rows: i32,
    n_lead: i32,
    extra_ids: *const i32,
    part_vals_scratch: *mut f32,
    part_ids_scratch: *mut i32,
    out_tok: *mut i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_argmax(
        logits, n_rows, n_lead, extra_ids, part_vals_scratch, part_ids_scratch, out_tok, stream.raw,
    );
}

pub unsafe fn copy_to_mapped_f32(
    src: *const f32,
    dst_mapped: *mut f32,
    n: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_copy_to_mapped_f32(src, dst_mapped, n, stream.raw);
}

pub unsafe fn f32_to_f16(src: *const f32, dst_f16: *mut c_void, n: i32, stream: &CudaStream) {
    ffi::launch_f32_to_f16(src, dst_f16, n, stream.raw);
}

pub unsafe fn dequant_to_f16(
    w_type: i32,
    src: *const c_void,
    dst_f16: *mut c_void,
    n_elems: i64,
    stream: &CudaStream,
) {
    ffi::launch_dequant_to_f16(w_type, src, dst_f16, n_elems, stream.raw);
}

pub unsafe fn gemm_f16_mma_nt(
    a_f16: *const c_void,
    b_f16: *const c_void,
    c_f32: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    stream: &CudaStream,
) {
    ffi::launch_gemm_f16_mma_nt(a_f16, b_f16, c_f32, m, n, k, alpha, beta, stream.raw);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_mmq_mma(
    w_type: i32,
    a_q8: *const c_void,
    b_quant: *const c_void,
    b_row_bytes: usize,
    c_f32: *mut f32,
    rows_in_opt: *const i32,
    rows_out_opt: *const i32,
    wts_out_opt: *const f32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    stream: &CudaStream,
) {
    ffi::launch_gemm_mmq_mma(
        w_type,
        a_q8,
        b_quant,
        b_row_bytes,
        c_f32,
        rows_in_opt,
        rows_out_opt,
        wts_out_opt,
        m,
        n,
        k,
        alpha,
        beta,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_mmq_mma_tiled(
    w_type: i32,
    a_q8: *const c_void,
    b_quant: *const c_void,
    b_row_bytes: usize,
    c_f32: *mut f32,
    rows_in_opt: *const i32,
    rows_out_opt: *const i32,
    wts_out_opt: *const f32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    force_tile: i32,
    stream: &CudaStream,
) {
    ffi::launch_gemm_mmq_mma_tiled(
        w_type,
        a_q8,
        b_quant,
        b_row_bytes,
        c_f32,
        rows_in_opt,
        rows_out_opt,
        wts_out_opt,
        m,
        n,
        k,
        alpha,
        beta,
        force_tile,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_mmq_mma_grouped(
    w_type: i32,
    a_q8: *const c_void,
    exp_ptrs: *const u64,
    b_byte_offset: usize,
    b_row_bytes: usize,
    c_f32: *mut f32,
    rows_in_opt: *const i32,
    rows_out_opt: *const i32,
    wts_out_opt: *const f32,
    exp_row_off: *const i32,
    exp_nt: *const i32,
    tile_exp_id: *const i32,
    tile_m_off: *const i32,
    n_m_tiles: i32,
    n: i32,
    k: i32,
    alpha: f32,
    use_atomic_scatter: bool,
    tile_mode: i32,
    stream: &CudaStream,
) {
    ffi::launch_gemm_mmq_mma_grouped(
        w_type,
        a_q8,
        exp_ptrs,
        b_byte_offset,
        b_row_bytes,
        c_f32,
        rows_in_opt,
        rows_out_opt,
        wts_out_opt,
        exp_row_off,
        exp_nt,
        tile_exp_id,
        tile_m_off,
        n_m_tiles,
        n,
        k,
        alpha,
        i32::from(use_atomic_scatter),
        tile_mode,
        stream.raw,
    );
}

pub unsafe fn swiglu_clamp_interleaved_q8(
    gu: *const f32,
    lim: f32,
    nt: i32,
    n_ff: i32,
    out_q8: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_swiglu_clamp_interleaved_q8(gu, lim, nt, n_ff, out_q8, stream.raw);
}

pub unsafe fn gather_rows_f16(
    src: *const f32,
    rows: *const i32,
    dst_f16: *mut c_void,
    nt: i32,
    k: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_gather_rows_f16(src, rows, dst_f16, nt, k, stream.raw);
}

pub unsafe fn swiglu_clamp_to_f16(
    gate: *const f32,
    up: *const f32,
    lim: f32,
    n: i32,
    out_f16: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_swiglu_clamp_to_f16(gate, up, lim, n, out_f16, stream.raw);
}

pub unsafe fn swiglu_clamp_interleaved_to_f16(
    gu: *const f32,
    lim: f32,
    nt: i32,
    n_ff: i32,
    out_f16: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_swiglu_clamp_interleaved_to_f16(gu, lim, nt, n_ff, out_f16, stream.raw);
}

pub unsafe fn scatter_weighted_add_f32(
    src: *const f32,
    rows: *const i32,
    weights: *const f32,
    dst: *mut f32,
    nt: i32,
    k: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_scatter_weighted_add_f32(src, rows, weights, dst, nt, k, stream.raw);
}

pub unsafe fn scale_f32(x: *mut f32, scale: f32, n: i32, stream: &CudaStream) {
    ffi::launch_glm5_scale_f32(x, scale, n, stream.raw);
}

pub unsafe fn mtp_in_batch_f16(
    emb: *const f32,
    h: *const f32,
    enorm: *const f32,
    hnorm: *const f32,
    eps: f32,
    n_embd: i32,
    m: i32,
    cat_f16: *mut c_void,
    stream: &CudaStream,
) {
    ffi::launch_glm5_mtp_in_batch_f16(emb, h, enorm, hnorm, eps, n_embd, m, cat_f16, stream.raw);
}

pub unsafe fn moe_prefill_small_gu(
    gu_type: i32,
    exp_ptrs: *const u64,
    slot_off: *const i32,
    slot_nt: *const i32,
    rows_all: *const i32,
    u_off: u32,
    gu_row_bytes: usize,
    x_q8: *const c_void,
    inter_q8: *mut c_void,
    n_embd: i32,
    n_ff: i32,
    n_jobs: i32,
    clamp_lim: f32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_prefill_small_gu(
        gu_type,
        exp_ptrs,
        slot_off,
        slot_nt,
        rows_all,
        u_off,
        gu_row_bytes,
        x_q8,
        inter_q8,
        n_embd,
        n_ff,
        n_jobs,
        clamp_lim,
        stream.raw,
    );
}

pub unsafe fn moe_prefill_small_down(
    d_type: i32,
    exp_ptrs: *const u64,
    slot_off: *const i32,
    slot_nt: *const i32,
    rows_all: *const i32,
    wts_all: *const f32,
    d_off: u32,
    d_row_bytes: usize,
    inter_q8: *const c_void,
    out_f32: *mut f32,
    n_embd: i32,
    n_ff: i32,
    n_jobs: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_moe_prefill_small_down(
        d_type,
        exp_ptrs,
        slot_off,
        slot_nt,
        rows_all,
        wts_all,
        d_off,
        d_row_bytes,
        inter_q8,
        out_f32,
        n_embd,
        n_ff,
        n_jobs,
        stream.raw,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_mla_prefill_mma(
    q_abs: *const f32,
    vram_lat8: *const c_void,
    ram_lat8: *const c_void,
    res_cap: i32,
    cells: *const i32,
    wv_b: *const c_void,
    wv_type: i32,
    kv_lora: i32,
    n_head: i32,
    v_head: i32,
    qk_nope: i32,
    n_sel: i32,
    ctx_scratch: *mut f32,
    out: *mut f32,
    t: i32,
    stream: &CudaStream,
) {
    ffi::launch_glm5_dsa_mla_prefill_mma(
        q_abs,
        vram_lat8,
        ram_lat8,
        res_cap,
        cells,
        wv_b,
        wv_type,
        kv_lora,
        n_head,
        v_head,
        qk_nope,
        n_sel,
        ctx_scratch,
        out,
        t,
        stream.raw,
    );
}
