use crate::cuda_rt::CudaStreamHandle;
use libc::c_void;

extern "C" {
    pub fn launch_quantize_q8_1(
        x: *const f32,
        y_q8_1: *mut c_void,
        m: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mv_q8_planar_m1(
        q0: *const c_void, d0: *const c_void, n0: i32, x0: *const f32, y0: *mut f32,
        q1: *const c_void, d1: *const c_void, n1: i32, x1: *const f32, y1: *mut f32,
        conv_w1: *const f32, conv_off1: i32,
        q2: *const c_void, d2: *const c_void, n2: i32, x2: *const f32, y2: *mut f32,
        conv_w2: *const f32, conv_off2: i32,
        q3: *const c_void, d3: *const c_void, n3: i32, x3: *const f32, y3: *mut f32,
        q4: *const c_void, d4: *const c_void, n4: i32, x4: *const f32, y4: *mut f32,
        add_res0: *const f32,
        conv_history: *mut f32,
        n_conv: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_fused_qkvrg_m1(
        w_type: i32,
        w0: *const c_void, n0: i32, x0: *const c_void, y0: *mut f32,
        w1: *const c_void, n1: i32, x1: *const c_void, y1: *mut f32,
        conv_w1: *const f32, conv_off1: i32,
        w2: *const c_void, n2: i32, x2: *const c_void, y2: *mut f32,
        conv_w2: *const f32, conv_off2: i32,
        w3: *const c_void, n3: i32, x3: *const c_void, y3: *mut f32,
        w4: *const c_void, n4: i32, x4: *const c_void, y4: *mut f32,
        conv_history: *mut f32,
        n_conv: i32,
        k: i32,
        row_bytes: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_linear_q8_1(
        w_type: i32,
        w: *const c_void,
        x_q8: *const c_void,
        x_f32: *const f32,
        y: *mut f32,
        n: i32,
        k: i32,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_linear_q8_1_skip(
        w_type: i32,
        w: *const c_void,
        x_q8: *const c_void,
        x_f32: *const f32,
        y: *mut f32,
        n: i32,
        k: i32,
        m: i32,
        skip_dev: *const i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_linear_q8_1_conv_add_m1(
        w_type: i32,
        w: *const c_void,
        x_q8: *const c_void,
        y: *mut f32,
        conv_w: *const f32,
        add_res: *const f32,
        conv_history: *mut f32,
        conv_offset: i32,
        n_conv: i32,
        n: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_embed_lookup_rmsnorm(
        table: *const c_void,
        table_type: i32,
        token_ids: *const i32,
        norm_weight: *const f32,
        out: *mut f32,
        n_embd: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mmvq_q8_0_multi2(
        w0: *const c_void,
        y0: *mut f32,
        n0: i32,
        w1: *const c_void,
        y1: *mut f32,
        n1: i32,
        x_q8: *const c_void,
        k: i32,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mmvq_q8_0_multi3(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_sh_gu_swiglu_q8(
        w_gate: *const c_void,
        w_up: *const c_void,
        x_q8: *const c_void,
        y_act: *mut f32,
        n: i32,
        k: i32,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mmvq_q8_0_indexed_m1(
        w: *const c_void,
        subset_ids: *const i32,
        x_q8: *const c_void,
        y: *mut f32,
        n_sub: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_rmsnorm_f32(
        x: *const f32,
        gamma: *const f32,
        y: *mut f32,
        dim: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_l2_clip_scale_f32(
        x: *mut f32,
        scale: *const f32,
        tau: *const f32,
        head_dim: i32,
        n_head: i32,
        m: i32,
        threshold: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_attn_res_mix_f32(
        state_ptrs: *const *const f32,
        n_states: i32,
        queries: *const f32,
        out: *mut f32,
        n_embd: i32,
        n_q: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_attn_res_mix_jobs_f32(
        state_ptrs: *const *const f32,
        n_states: i32,
        queries: *const f32,
        out: *mut f32,
        n_embd: i32,
        n_q: i32,
        n_jobs: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_attn_res_mix_rms_f32(
        state_ptrs: *const *const f32,
        n_states: i32,
        queries: *const f32,
        norm_w0: *const f32,
        norm_w1: *const f32,
        norm_w2: *const f32,
        norm_y0: *mut f32,
        norm_y1: *mut f32,
        norm_y2: *mut f32,
        n_embd: i32,
        n_q: i32,
        m: i32,
        res_eps: f32,
        rms_eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_rmsnorm_multi_f32(
        x0: *const f32, x1: *const f32, x2: *const f32,
        w0: *const f32, w1: *const f32, w2: *const f32,
        y0: *mut f32, y1: *mut f32, y2: *mut f32,
        n_jobs: i32,
        dim: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_loop_gates_fused_f32(
        x0: *const f32, x1: *const f32, x2: *const f32,
        m0: *const f32, m1: *const f32, m2: *const f32,
        d0: *const f32, d1: *const f32, d2: *const f32,
        f0: *const f32, f1: *const f32, f2: *const f32,
        g0: *const f32, g1: *const f32, g2: *const f32,
        y0: *mut f32, y1: *mut f32, y2: *mut f32,
        scratch_sc: *mut f32,
        n_gates: i32,
        m: i32,
        c: i32,
        r: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_concat_two_rmsnorm_f32(
        cur: *const f32,
        msg: *const f32,
        out: *mut f32,
        n_embd: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_apply_loop_ctrl_f32(
        cur: *mut f32,
        msg: *const f32,
        forget_raw: *const f32,
        gain_raw: *const f32,
        n_total: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_short_conv_f32(
        cur: *const f32,
        static_w: *const f32,
        dyn_proj: *const f32,
        history: *mut f32,
        out: *mut f32,
        n_channels: i32,
        n_heads: i32,
        m: i32,
        conv_offset: i32,
        n_conv: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_short_conv_add_f32(
        cur: *const f32,
        static_w: *const f32,
        dyn_proj: *const f32,
        add_res: *const f32,
        history: *mut f32,
        out: *mut f32,
        n_channels: i32,
        n_heads: i32,
        m: i32,
        conv_offset: i32,
        n_conv: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_sigmoid_mul_f32(a: *mut f32, gate: *const f32, n_total: i32, stream: CudaStreamHandle);

    pub fn launch_add_f32(a: *mut f32, b: *const f32, n_total: i32, stream: CudaStreamHandle);

    pub fn launch_compute_tau_f32(
        pos: *const i32,
        tau: *mut f32,
        m: i32,
        n_floor: f32,
        alpha: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_router_topk_sigmoid_f32(
        logits: *const f32,
        bias: *const f32,
        out_ids: *mut i32,
        out_weights: *mut f32,
        n_expert: i32,
        topk: i32,
        m: i32,
        route_scale: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_router_topk_and_f16_limbs_m1(
        logits: *const f32,
        bias: *const f32,
        out_ids: *mut i32,
        out_weights: *mut f32,
        x_f32: *const f32,
        x_limbs: *mut c_void,
        n_expert: i32,
        topk: i32,
        n_embd: i32,
        route_scale: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_softcap_tanh_f32(x: *mut f32, n_total: i32, cap: f32, stream: CudaStreamHandle);

    pub fn launch_argmax_f32(
        logits: *const f32,
        n_vocab: i32,
        out_id: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_argmax_val_id_f32(
        logits: *const f32,
        n_vocab: i32,
        out_val: *mut f32,
        out_id: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_moe_grouped_f32(
        gate_up_ptrs: *const u64,
        down_ptrs: *const u64,
        gate_up_scales: *const f32,
        down_scales: *const f32,
        gate_up_row_bytes: usize,
        down_row_bytes: usize,
        gate_up_type: i32,
        down_type: i32,
        x_q8: *const c_void,
        x_f32: *const f32,
        expert_ids: *const i32,
        expert_weights: *const f32,
        inter_f32: *mut f32,
        inter_q8: *mut c_void,
        slot_out: *mut f32,
        out: *mut f32,
        n_embd: i32,
        n_ff: i32,
        topk: i32,
        m: i32,
        situ_beta: f32,
        situ_linear_beta: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_moe_reduce_conv_add_m1_f32(
        slot_out: *const f32,
        static_w: *const f32,
        history: *mut f32,
        ffn_inp: *const f32,
        cur_out: *mut f32,
        n_embd: i32,
        topk: i32,
        conv_offset: i32,
        n_conv: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_prep_qkv_r_and_kv_store(
        q_cur: *mut f32,
        q_clip: *const f32,
        k_cur: *mut f32,
        v_cur: *mut f32,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        r_cur: *mut f32,
        r_clip: *const f32,
        tau: *const f32,
        pos: *const i32,
        sink_w: *const f32,
        sink_x: *const f32,
        sinks_out: *mut f32,
        kp_w: *const f32,
        kp_x: *const f32,
        k_conv_w: *const f32,
        vp_w: *const f32,
        vp_x: *const f32,
        v_conv_w: *const f32,
        conv_history: *mut f32,
        n_conv: i32,
        n_embd: i32,
        n_head: i32,
        n_head_kv: i32,
        head_dim: i32,
        d_rel: i32,
        m: i32,
        ring_cap: i32,
        kv_is_q8: i32,
        qk_clip_th: f32,
        rel_clip_th: f32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_kv_store(
        k_cur: *const f32,
        v_cur: *const f32,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        pos: *const i32,
        n_head_kv: i32,
        head_dim: i32,
        m: i32,
        ring_cap: i32,
        kv_is_q8: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_flash_attn_rel_decode_m1(
        q_cur: *const f32,
        r_cur: *const f32,
        attn_rel_bias: *const f32,
        sinks: *const f32,
        attn_gate: *const f32,
        k_cache: *const c_void,
        v_cache: *const c_void,
        q_pos: *const i32,
        split_out_scratch: *mut f32,
        split_ml_scratch: *mut f32,
        out: *mut f32,
        n_head: i32,
        n_head_kv: i32,
        head_dim: i32,
        d_rel: i32,
        rel_extent: i32,
        swa_window: i32,
        ring_cap: i32,
        kv_is_q8: i32,
        max_kv_len: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_flash_attn_rel(
        q_cur: *const f32,
        r_cur: *const f32,
        attn_rel_bias: *const f32,
        sinks: *const f32,
        k_cache: *const c_void,
        v_cache: *const c_void,
        q_pos: *const i32,
        rel_table_scratch: *mut f32,
        split_out_scratch: *mut f32,
        split_ml_scratch: *mut f32,
        out: *mut f32,
        n_head: i32,
        n_head_kv: i32,
        head_dim: i32,
        d_rel: i32,
        rel_extent: i32,
        swa_window: i32,
        ring_cap: i32,
        kv_is_q8: i32,
        max_kv_len: i32,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_doorbell_signal(req_seq_dev: *mut u64, seq_val: u64, stream: CudaStreamHandle);

    pub fn launch_doorbell_wait_and_add(
        done_seq_dev: *const u64,
        expected_seq: u64,
        partial_a: *const f32,
        partial_b: *const f32,
        out: *mut f32,
        n_total: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_p2p_push(
        local_partial: *const f32,
        peer_recv: *mut f32,
        peer_flag: *mut u32,
        seq: u32,
        n_total: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_p2p_wait_reduce(
        local_buf: *mut f32,
        recv_buf: *const f32,
        flag_buf: *const u32,
        expected_seq: u32,
        n_total: i32,
        stream: CudaStreamHandle,
    );

    // GLM-5.3 kernels (moe_grouped.cu, glm5_ops.cu, glm5_gemm_mma.cu)
    pub fn launch_glm5_moe_resident(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_plan(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_plan_filtered(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_mv_batch(
        w_ptrs: *const *const c_void,
        y_ptrs: *const *mut f32,
        w_types: *const i32,
        n_outs: *const i32,
        scales: *const f32,
        n_jobs: i32,
        k: i32,
        x_q8: *const c_void,
        x_f32: *const f32,
        skip_dev: *const i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_skip_gate(
        doorbell_mapped: *const c_void,
        skip_dev: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_embed_replicate(
        emb: *const f32,
        r_out: *mut f32,
        n_embd: i32,
        hc: i32,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_hc_pre_norm_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_hc_post_add(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_hc_fused_m1(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_mean_streams_norm_q8(
        r_in: *const f32,
        norm_w: *const f32,
        eps: f32,
        n_embd: i32,
        hc: i32,
        out_f32: *mut f32,
        out_q8: *mut c_void,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_rms_norm_q8(
        base_res: *mut f32,
        add_opt: *const f32,
        norm_w: *const f32,
        eps: f32,
        dim: i32,
        write_back_base: i32,
        out_f32: *mut f32,
        out_q8: *mut c_void,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_swiglu_clamp_q8(
        gate: *const f32,
        up: *const f32,
        lim: f32,
        n_ff: i32,
        out_f32_opt: *mut f32,
        out_q8: *mut c_void,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_kda_conv_silu_l2(
        x_proj: *const f32,
        conv_w: *const f32,
        conv_state: *mut f32,
        d_inner: i32,
        d_conv: i32,
        head_dim: i32,
        n_head: i32,
        apply_l2: i32,
        t: i32,
        out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_kda_gate_beta(
        g_in: *const f32,
        dt_bias: *const f32,
        ssm_a: *const f32,
        lower_bound: f32,
        head_dim: i32,
        n_head: i32,
        t: i32,
        g1_out: *mut f32,
        beta_inout: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_prepare_kda_smem(head_dim: i32);

    pub fn launch_glm5_kda_recurrence(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_kda_out_gate_q8(
        o: *const f32,
        norm_w: *const f32,
        g2: *const f32,
        head_dim: i32,
        n_head: i32,
        t: i32,
        eps: f32,
        gated_f32: *mut f32,
        gated_q8: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_kda_prep_m1(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_kda_rec_gate_q8_m1(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_prep_kv_ik(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_prep_kv_ik_dyn(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_pool(
        ik_cache: *const f32,
        ig_cache: *const f32,
        ape: *const f32,
        key_dim: i32,
        kpool: i32,
        pool0: i32,
        n_new_pools: i32,
        ring_cap: i32,
        pooled_f16: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_pool_m1_dyn(
        ik_cache: *const f32,
        ig_cache: *const f32,
        ape: *const f32,
        key_dim: i32,
        kpool: i32,
        pos_arr: *const i32,
        ring_cap: i32,
        pooled_f16: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_prep_m1_dyn(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_q_absorb(
        wk_b: *const c_void,
        wk_type: i32,
        q: *const f32,
        q_abs: *mut f32,
        qk_nope: i32,
        kv_lora: i32,
        n_head: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_score_select(
        iq: *const f32,
        pooled_f16: *const c_void,
        weights: *const f32,
        key_dim: i32,
        idx_heads: i32,
        kpool: i32,
        top_pools_cap: i32,
        select_tail: i32,
        n_sel: i32,
        pos_arr: *const i32,
        scores_buf: *mut f32,
        cells_out: *mut i32,
        t: i32,
        cur_pool_done: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_score_select_dyn(
        iq: *const f32,
        pooled_f16: *const c_void,
        weights: *const f32,
        key_dim: i32,
        idx_heads: i32,
        kpool: i32,
        top_pools_cap: i32,
        select_tail: i32,
        n_sel: i32,
        pool_cap: i32,
        pos_arr: *const i32,
        scores_buf: *mut f32,
        cells_out: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_mla_attn(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_mla_vproj_q8_m1(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_router_sigmoid_probe(
        logits: *const f32,
        bias: *const f32,
        n_expert: i32,
        topk: i32,
        w_scale: f32,
        norm_w: i32,
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_router_dual_lane(
        logits: *const f32,
        bias: *const f32,
        n_expert: i32,
        topk: i32,
        w_scale: f32,
        norm_w: i32,
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_fetch(
        fetch_src: *const u64,
        plan_ptrs: *const u64,
        topk: i32,
        blob_bytes: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_fetch_nblocks(
        fetch_src: *const u64,
        plan_ptrs: *const u64,
        topk: i32,
        blob_bytes: usize,
        n_blocks: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_router_dual_lane_pf(
        logits: *const f32,
        bias: *const f32,
        n_expert: i32,
        topk: i32,
        w_scale: f32,
        norm_w: i32,
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_prefetch(
        pf_src: *const u64,
        pf_dst: *const u64,
        pf_n: *const i32,
        blob_bytes: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_doorbell_wait_hc_post(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_doorbell_wait_add(
        doorbell_mapped: *const c_void,
        expected_seq_ptr: *const u64,
        inout_ffn: *mut f32,
        n_embd: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_mtp_in_q8(
        emb: *const f32,
        h: *const f32,
        enorm: *const f32,
        hnorm: *const f32,
        eps: f32,
        n_embd: i32,
        cat_q8: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_argmax(
        logits: *const f32,
        n_rows: i32,
        n_lead: i32,
        extra_ids: *const i32,
        part_vals_scratch: *mut f32,
        part_ids_scratch: *mut i32,
        out_tok: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_copy_to_mapped_f32(
        src: *const f32,
        dst_mapped: *mut f32,
        n: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_f32_to_f16(src: *const f32, dst_f16: *mut c_void, n: i32, stream: CudaStreamHandle);

    pub fn launch_dequant_to_f16(
        w_type: i32,
        src: *const c_void,
        dst_f16: *mut c_void,
        n_elems: i64,
        stream: CudaStreamHandle,
    );

    pub fn launch_gemm_f16_mma_nt(
        a_f16: *const c_void,
        b_f16: *const c_void,
        c_f32: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_gemm_mmq_mma(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_gemm_mmq_mma_tiled(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_gemm_mmq_mma_grouped(
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
        use_atomic_scatter: i32,
        tile_mode: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_swiglu_clamp_interleaved_q8(
        gu: *const f32,
        lim: f32,
        nt: i32,
        n_ff: i32,
        out_q8: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_gather_rows_f16(
        src: *const f32,
        rows: *const i32,
        dst_f16: *mut c_void,
        nt: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_swiglu_clamp_to_f16(
        gate: *const f32,
        up: *const f32,
        lim: f32,
        n: i32,
        out_f16: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_swiglu_clamp_interleaved_to_f16(
        gu: *const f32,
        lim: f32,
        nt: i32,
        n_ff: i32,
        out_f16: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_scatter_weighted_add_f32(
        src: *const f32,
        rows: *const i32,
        weights: *const f32,
        dst: *mut f32,
        nt: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_scale_f32(x: *mut f32, scale: f32, n: i32, stream: CudaStreamHandle);

    pub fn launch_glm5_mtp_in_batch_f16(
        emb: *const f32,
        h: *const f32,
        enorm: *const f32,
        hnorm: *const f32,
        eps: f32,
        n_embd: i32,
        m: i32,
        cat_f16: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_prefill_small_gu(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_prefill_small_down(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_moe_prefill_small_down_det(
        d_type: i32,
        exp_ptrs: *const u64,
        slot_off: *const i32,
        slot_nt: *const i32,
        pair_all: *const i32,
        wts_all: *const f32,
        d_off: u32,
        d_row_bytes: usize,
        inter_q8: *const c_void,
        pair_out_f32: *mut f32,
        n_embd: i32,
        n_ff: i32,
        n_jobs: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_glm5_dsa_mla_prefill_mma(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_expand_streams(
        emb: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        x_out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_hc_pre(
        x: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        fn_w: *const f32,
        fn_by_el: i32,
        scale: *const f32,
        base: *const f32,
        eps: f32,
        hc_eps: f32,
        mag: f32,
        norm_w_opt: *const f32,
        norm_eps: f32,
        is_head: i32,
        part_buf: *mut f32,
        cur: *mut f32,
        post_opt: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_hc_post_norm(
        x: *mut f32,
        t: i32,
        hc: i32,
        e: i32,
        y: *const f32,
        post: *const f32,
        w_opt: *const f32,
        eps: f32,
        y_norm_opt: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_rope(
        x: *mut f32,
        t: i32,
        nh: i32,
        ld_t: usize,
        ld_h: usize,
        off: i32,
        n_rot: i32,
        pos: *const i32,
        base: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_sigmoid_mul(
        x: *mut f32,
        g: *const f32,
        n: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_swiglu_f32(
        gate: *const f32,
        up: *const f32,
        ld_gu: usize,
        rows: i32,
        n: i32,
        clamp_lim: f32,
        out: *mut f32,
        ld_out: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_store_rows(
        src: *const f32,
        t: i32,
        dim: i32,
        ring: *mut c_void,
        ring_cap: i32,
        pos: *const i32,
        is_f16: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_gemv_f32(
        w: *const f32,
        n: i32,
        k: i32,
        x: *const f32,
        m: i32,
        y: *mut f32,
        alpha: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_gemv_bf16(
        w: *const u16,
        n: i32,
        k: i32,
        x: *const f32,
        m: i32,
        y: *mut f32,
        alpha: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_quantize_heads_q8_1(
        x: *const f32,
        x_q8_1: *mut c_void,
        t: i32,
        h: i32,
        k: i32,
        ld_t: usize,
        ld_h: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_batched_heads_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_batched_heads_q8_mma(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_kv_store(
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
        is_q8: i32,
        rb: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_mla_decode(
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
        is_q8: i32,
        rb: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_mla_prefill(
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
        is_q8: i32,
        rb: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_ix_pool(
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
        is_f16: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_ix_score(
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
        is_f16: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_ix_select(
        score: *const f32,
        ld: usize,
        t: i32,
        pos: *const i32,
        kpool: i32,
        top: i32,
        keys: *mut i32,
        ldk: i32,
        nk: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_router_sigmoid_topk(
        logits: *const f32,
        t: i32,
        n_exp: i32,
        bias_opt: *const f32,
        n_used: i32,
        norm: i32,
        scale: f32,
        ids: *mut i32,
        wout: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_engram_rows(
        rows: *const u8,
        valid: *const f32,
        t: i32,
        cols: i32,
        dim: i32,
        ln: *const f32,
        eps: f32,
        scale_opt: *const f32,
        out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_engram_mix(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_layernorm_f16(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        n: i32,
        d: i32,
        eps: f32,
        y16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_rmsnorm_inplace(
        x: *mut f32,
        w: *const f32,
        n: i32,
        d: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_add_bias(
        y: *mut f32,
        bias: *const f32,
        n: usize,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_bias_act_f16(
        y: *const f32,
        bias: *const f32,
        n: usize,
        m: i32,
        act: i32,
        y16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_residual_bias(
        h: *mut f32,
        t: *const f32,
        bias: *const f32,
        n: usize,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_merge_gather_f16(
        xn: *const f32,
        gw: i32,
        d: i32,
        ow: i32,
        oh: i32,
        m16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_place_rows(
        y: *const f32,
        bias: *const f32,
        nl: *const f32,
        ow: i32,
        oh: i32,
        o: i32,
        out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_vis_flash_attn(
        qkv: *const f32,
        n: i32,
        heads: i32,
        hd: i32,
        scale: f32,
        out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_copy_cols(
        src: *const f32,
        ld_src: usize,
        dst: *mut f32,
        ld_dst: usize,
        rows: i32,
        width: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_add_f32(
        dst: *mut f32,
        src: *const f32,
        n: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_rmsnorm_strided(
        x: *const f32,
        ld_x: usize,
        w: *const f32,
        y: *mut f32,
        ld_y: usize,
        dim: i32,
        rows: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_top1_prob(
        logits: *const f32,
        n_vocab: i32,
        scratch_and_out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_hy4_build_moe_plan(
        ids: *const i32,
        slot_d: *const i32,
        vram_base: u64,
        blob_bytes: usize,
        count: i32,
        plan_ptrs_out: *mut u64,
        stream: CudaStreamHandle,
    );

    // ---- Qwen3.8-Flash-Next (qwen4exp) kernels ----

    pub fn launch_qwen4_hc_expand(
        emb: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        r_out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_rmsnorm_q8(
        r: *const f32,
        w_norm: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        eps: f32,
        xn_out: *mut f32,
        xn_q8_opt: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_rmsnorm_inject_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_down_silu_q8(
        lo_in: *const f32,
        t: i32,
        hc: i32,
        lr: i32,
        lo_out_opt: *mut f32,
        lo_q8_opt: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_mix_q8(
        xn: *const f32,
        up_logits: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        mixed_out: *mut f32,
        mixed_q8_opt: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_inject_dot(
        xn: *const f32,
        w_inject: *const c_void,
        w_type: i32,
        t: i32,
        hc: i32,
        e: i32,
        inject_out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_write(
        r: *mut f32,
        block_out: *const f32,
        sh_out_opt: *const f32,
        sh_gate_logit_opt: *const f32,
        inject: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_write_strided(
        r: *mut f32,
        block_out: *const f32,
        sh_out_opt: *const f32,
        sh_gate_logit_opt: *const f32,
        inject: *const f32,
        inject_stride: i32,
        t: i32,
        hc: i32,
        e: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_add_streams_broadcast(
        h: *const f32,
        e: *const f32,
        t: i32,
        hc: i32,
        e_dim: i32,
        r_out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_rmsnorm_f32(
        x: *const f32,
        w: *const f32,
        rows: i32,
        cols: i32,
        eps: f32,
        y_out: *mut f32,
        y_q8_opt: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_ple_postops(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_ab_f32(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_gate_beta_postop(
        alpha_raw: *const f32,
        beta_raw: *const f32,
        dt_bias: *const f32,
        ssm_a: *const f32,
        gate_out: *mut f32,
        beta_out: *mut f32,
        t: i32,
        h_v: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_conv_l2(
        conv_hist: *mut f32,
        qkv: *const f32,
        conv_w: *const f32,
        h_out: *mut f32,
        t: i32,
        channels: i32,
        qk_heads: i32,
        eps: f32,
        hist_snap_opt: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_decode_step_norm_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_verify_steps_norm_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn qwen4_gdn_tc_scratch_bytes(h_k: i32, h_v: i32) -> usize;

    pub fn launch_qwen4_gdn_tc_prefill(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_qsa_norm_imrope(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_idx_append_steps(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_idx_append_batch(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_idx_score_select(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_kv_append_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn qwen4_qsa_decode_ws_floats(t: i32, n_kv_heads: i32, max_n_ids: i32) -> usize;

    pub fn launch_qwen4_qsa_decode_attn_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_qsa_prefill_attn_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_router_top10(
        logits: *const f32,
        t: i32,
        n_exp: i32,
        topk: i32,
        ids_out: *mut i32,
        weights_out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_shared_gate_dot(
        x: *const f32,
        w_sh_gate: *const c_void,
        w_type: i32,
        t: i32,
        n_embd: i32,
        out_logits: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_vis_rope2d(
        qkv: *mut f32,
        pos_yx: *const i32,
        n_patches: i32,
        heads: i32,
        hd: i32,
        base: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_vis_merge_2x2_f16(
        x_ln: *const f32,
        n_merged: i32,
        d: i32,
        out_f16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_fill_dense_widths(
        pos_dev: *const i32,
        width_out: *mut i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_hc_up_mix_q8(
        w_up_q8_0: *const c_void,
        lo_q8: *const c_void,
        xn: *const f32,
        t: i32,
        hc: i32,
        e: i32,
        lr: i32,
        mixed_out: *mut f32,
        mixed_q8_opt: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_router_doorbell_plan(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_doorbell_wait_hc_write(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_doorbell_wait(
        doorbell_mapped: *const c_void,
        expected_seq_ptr: *const u64,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_moe_reduce_hc_write(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_argmax_prob(
        logits: *const f32,
        n_vocab: i32,
        partial_max: *mut f32,
        partial_idx: *mut i32,
        partial_sum: *mut f32,
        out_id: *mut i32,
        out_prob: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_argmax_prob_indexed(
        logits: *const f32,
        n_vocab: i32,
        sub_to_id_opt: *const i32,
        partial_max: *mut f32,
        partial_idx: *mut i32,
        partial_sum: *mut f32,
        out_id: *mut i32,
        out_prob: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_moe_fetch_unique(
        fetch_src_ptrs: *const u64,
        fetch_dst_ptrs: *const u64,
        blob_bytes: usize,
        fetch_cap: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_router_doorbell_plan_dedup(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_steps_split_norm_q8(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_gdn_replay_steps(
        state_before: *const f32,
        state_out: *mut f32,
        h_qkv: *const f32,
        gate: *const f32,
        beta: *const f32,
        n_steps: i32,
        h_k: i32,
        h_v: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_reduce_pairs_f32(
        pair_out: *const f32,
        block_out: *mut f32,
        t: i32,
        topk: i32,
        n_embd: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_qwen4_patch_u64(mapped_pairs: *const u64, n: i32, stream: CudaStreamHandle);

    pub fn launch_qwen4_copy_u32_sm(
        dst: *mut c_void,
        src: *const c_void,
        n_u32: i32,
        system_fence: i32,
        stream: CudaStreamHandle,
    );

    // -----------------------------------------------------------------------
    // MiniMax M3.1 Flash (mm3) kernels
    // -----------------------------------------------------------------------

    pub fn launch_mm3_rms_norm_f32(
        x: *const f32,
        w: *const f32,
        rows: i32,
        dim: i32,
        eps: f32,
        add_one: i32,
        y_f32_opt: *mut f32,
        y_q8_1_opt: *mut c_void,
        y_q8_k_opt: *mut c_void,
        y_bf16_opt: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_quantize_q8_k(
        x: *const f32,
        out_q8k: *mut c_void,
        m: i32,
        k: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_swigluoai(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_qk_norm_rope_kv_store(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_index_norm_rope_store(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_msa_block_scores(
        iq: *const f32,
        ic_bf16: *const u16,
        pos: *const i32,
        t: i32,
        hi: i32,
        nblk: i32,
        blk: i32,
        bs: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_msa_select_topk(
        bs: *const f32,
        pos: *const i32,
        t: i32,
        hi: i32,
        nblk: i32,
        blk: i32,
        topk: i32,
        local_blocks: i32,
        kept: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_msa_sparse_attn(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_router_f32(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_moe_gate_up_q8k(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_moe_down_q8k(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_moe_combine(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_moe_grp_gate_up_f32(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_moe_grp_down_q8k(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_moe_grp_combine(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_bf16_gemm_mma(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_mxfp8_to_bf16(
        w: *const u8,
        scale: *const u8,
        n: i32,
        k: i32,
        out_bf16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_mxfp8_linear(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_dspark_window_attn(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_dspark_markov_greedy(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_vision_patch_embed_f32(
        px: *const f32,
        w_patch: *const f32,
        n: i32,
        d: i32,
        p: i32,
        out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_vision_layernorm(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        rows: i32,
        dim: i32,
        eps: f32,
        y_f32_opt: *mut f32,
        y_bf16_opt: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_vision_qkv_rope_3d(
        qkv: *mut f32,
        bias_opt: *const f32,
        pos3d: *const i32,
        inv_freq: *const f32,
        n: i32,
        dim: i32,
        heads: i32,
        hd: i32,
        nf: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_vision_full_attn(
        qkv: *const f32,
        n: i32,
        dim: i32,
        heads: i32,
        hd: i32,
        scale: f32,
        out: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_vision_bias_gelu(
        y: *const f32,
        bias_opt: *const f32,
        n: usize,
        m: i32,
        apply_gelu: i32,
        y_f32_opt: *mut f32,
        y_bf16_opt: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_vision_residual_bias(
        h: *mut f32,
        t: *const f32,
        bias_opt: *const f32,
        n: usize,
        m: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_gv_q6k_multi(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_gv_q8_0_multi(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_iota(
        out: *mut i32,
        n: i32,
        pos0: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_dspark_draft_block(
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
        stream: CudaStreamHandle,
    );

    pub fn launch_mm3_wait_u32(
        wait_dev_ptr: *const u32,
        target_val: u32,
        stream: CudaStreamHandle,
    );

    // Inkling-2 fast decode + batched verify block kernels (`inkling2_fast.cu`)
    pub fn launch_f32_to_f16_fast(
        x: *const f32,
        y: *mut u16,
        n: usize,
        stream: CudaStreamHandle,
    );

    pub fn launch_rmsnorm_f16_multi(
        x0: *const f32,
        x1: *const f32,
        x2: *const f32,
        w0: *const f32,
        w1: *const f32,
        w2: *const f32,
        y32_0: *mut f32,
        y32_1: *mut f32,
        y32_2: *mut f32,
        y16_0: *mut u16,
        y16_1: *mut u16,
        y16_2: *mut u16,
        n_jobs: i32,
        dim: i32,
        m: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_attn_res_mix_rms_f16(
        state_ptrs: *const *const f32,
        n_states: i32,
        queries: *const f32,
        norm_w0: *const f32,
        norm_w1: *const f32,
        norm_w2: *const f32,
        norm_y0: *mut f32,
        norm_y1: *mut f32,
        norm_y2: *mut f32,
        norm_y16_0: *mut u16,
        norm_y16_1: *mut u16,
        norm_y16_2: *mut u16,
        n_embd: i32,
        n_q: i32,
        m: i32,
        res_eps: f32,
        rms_eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_q8_mma_tile(
        q: *const i8,
        d: *const u16,
        n: i32,
        k: i32,
        x16: *const u16,
        x_ld: i32,
        t: i32,
        out: *mut f32,
        y_ld: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_q8_mma_qkvrg(
        q0: *const i8,
        d0: *const u16,
        n0: i32,
        x0: *const u16,
        y0: *mut f32,
        q1: *const i8,
        d1: *const u16,
        n1: i32,
        x1: *const u16,
        y1: *mut f32,
        q2: *const i8,
        d2: *const u16,
        n2: i32,
        x2: *const u16,
        y2: *mut f32,
        q3: *const i8,
        d3: *const u16,
        n3: i32,
        x3: *const u16,
        y3: *mut f32,
        q4: *const i8,
        d4: *const u16,
        n4: i32,
        x4: *const u16,
        y4: *mut f32,
        k: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_q8_mma_argmax_batch(
        q: *const i8,
        d: *const u16,
        n: i32,
        k: i32,
        x16: *const u16,
        x_ld: i32,
        t: i32,
        part_vals: *mut f32,
        part_ids: *mut i32,
        out_ids: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_short_conv_rec_f32(
        cur: *const f32,
        static_w: *const f32,
        dyn_proj: *const f32,
        add_res: *const f32,
        history: *mut f32,
        out: *mut f32,
        rec_old_hist: *mut f32,
        rec_x: *mut f32,
        n_sconv_channels: i32,
        slot_idx: i32,
        n_channels: i32,
        n_heads: i32,
        m: i32,
        conv_offset: i32,
        n_conv: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_short_conv_kv_pair_rec_f32(
        k_in: *const f32,
        k_static_w: *const f32,
        k_dyn_proj: *const f32,
        k_out: *mut f32,
        v_in: *const f32,
        v_static_w: *const f32,
        v_dyn_proj: *const f32,
        v_out: *mut f32,
        history: *mut f32,
        rec_old_hist: *mut f32,
        rec_x: *mut f32,
        n_sconv_channels: i32,
        slot_idx: i32,
        kv_dim: i32,
        n_heads: i32,
        m: i32,
        n_conv: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mv_q8p_colsx_qkvrg(
        q0: *const i8, d0: *const u16, n0: i32, x0: *const f32, y0: *mut f32,
        q1: *const i8, d1: *const u16, n1: i32, x1: *const f32, y1: *mut f32,
        q2: *const i8, d2: *const u16, n2: i32, x2: *const f32, y2: *mut f32,
        q3: *const i8, d3: *const u16, n3: i32, x3: *const f32, y3: *mut f32,
        q4: *const i8, d4: *const u16, n4: i32, x4: *const f32, y4: *mut f32,
        k: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mv_q8p_colsx(
        q: *const i8,
        d: *const u16,
        n: i32,
        k: i32,
        x: *const f32,
        x_ld: i32,
        t: i32,
        y: *mut f32,
        y_ld: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mv_q8p_colsx_argmax(
        q: *const i8,
        d: *const u16,
        n: i32,
        k: i32,
        x: *const f32,
        x_ld: i32,
        t: i32,
        part_vals: *mut f32,
        part_ids: *mut i32,
        out_ids: *mut i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mv_f16_batch(
        w_f16: *const u16,
        x_f32: *const f32,
        out_f32: *mut f32,
        n: i32,
        k: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_mv_f16_pair_batch(
        w0_f16: *const u16,
        x0_f32: *const f32,
        out0_f32: *mut f32,
        w1_f16: *const u16,
        x1_f32: *const f32,
        out1_f32: *mut f32,
        n: i32,
        k: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_conv_rollback_all(
        histories: *const *mut f32,
        rec_old_hist: *const f32,
        rec_x: *const f32,
        n_slots: i32,
        n_sconv_channels: i32,
        accept_n: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_loop_gates_fast_f32(
        x0: *const f32,
        x1: *const f32,
        x2: *const f32,
        m0: *const f32,
        m1: *const f32,
        m2: *const f32,
        d0: *const f32,
        d1: *const f32,
        d2: *const f32,
        f0: *const f32,
        f1: *const f32,
        f2: *const f32,
        g0: *const f32,
        g1: *const f32,
        g2: *const f32,
        y0: *mut f32,
        y1: *mut f32,
        y2: *mut f32,
        scratch_sc: *mut f32,
        n_gates: i32,
        m: i32,
        c: i32,
        r: i32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_prep_qkv_r_block(
        q_cur: *mut f32,
        q16: *mut u16,
        q_clip: *const f32,
        k_cur: *mut f32,
        k_cache: *mut c_void,
        v_cur: *mut f32,
        v_cache: *mut c_void,
        r_cur: *mut f32,
        r_clip: *const f32,
        tau: *const f32,
        pos: *const i32,
        sink_w: *const f32,
        sink_w16: *const u16,
        sink_x: *const f32,
        sinks_out: *mut f32,
        n_embd: i32,
        n_head: i32,
        n_head_kv: i32,
        head_dim: i32,
        d_rel: i32,
        m: i32,
        ring_cap: i32,
        kv_mode: i32,
        qk_clip_th: f32,
        rel_clip_th: f32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_attn_block_mma(
        q16: *const u16,
        r_cur: *const f32,
        attn_rel_bias: *const f32,
        sinks: *const f32,
        attn_gate: *const f32,
        k_cache: *const c_void,
        v_cache: *const c_void,
        pos_d: *const i32,
        split_out: *mut f32,
        split_ml: *mut f32,
        out32: *mut f32,
        out16: *mut u16,
        n_head: i32,
        n_head_kv: i32,
        d_rel: i32,
        rel_extent: i32,
        swa_window: i32,
        ring_cap: i32,
        kv_mode: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_moe_block_fast(
        gate_up_ptrs: *const u64,
        down_ptrs: *const u64,
        gate_up_scales: *const f32,
        down_scales: *const f32,
        w_type: i32,
        x_f32: *const f32,
        expert_ids: *const i32,
        expert_weights: *const f32,
        x_limbs: *mut u16,
        inter_f16: *mut u16,
        slot_out: *mut f32,
        grp_gu_ptrs: *mut u64,
        grp_dn_ptrs: *mut u64,
        grp_n: *mut i32,
        grp_tok: *mut i32,
        grp_row: *mut i32,
        slot_sg: *mut f32,
        slot_su: *mut f32,
        slot_wt: *mut f32,
        ffn_post_conv_w: *const f32,
        conv_history: *mut f32,
        ffn_inp: *const f32,
        cur_out: *mut f32,
        rec_old_hist: *mut f32,
        rec_x: *mut f32,
        n_sconv_channels: i32,
        slot_idx: i32,
        conv_offset: i32,
        n_conv: i32,
        n_embd: i32,
        n_ff: i32,
        topk: i32,
        t: i32,
        situ_beta: f32,
        situ_linear_beta: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_route_and_moe_block_fast(
        router_logits: *const f32,
        router_bias: *const f32,
        n_expert: i32,
        route_scale: f32,
        gate_up_ptrs: *const u64,
        down_ptrs: *const u64,
        gate_up_scales: *const f32,
        down_scales: *const f32,
        w_type: i32,
        x_f32: *const f32,
        expert_ids: *mut i32,
        expert_weights: *mut f32,
        x_limbs: *mut u16,
        inter_f16: *mut u16,
        slot_out: *mut f32,
        grp_gu_ptrs: *mut u64,
        grp_dn_ptrs: *mut u64,
        grp_n: *mut i32,
        grp_tok: *mut i32,
        grp_row: *mut i32,
        slot_sg: *mut f32,
        slot_su: *mut f32,
        slot_wt: *mut f32,
        ffn_post_conv_w: *const f32,
        conv_history: *mut f32,
        ffn_inp: *const f32,
        cur_out: *mut f32,
        rec_old_hist: *mut f32,
        rec_x: *mut f32,
        n_sconv_channels: i32,
        slot_idx: i32,
        conv_offset: i32,
        n_conv: i32,
        n_embd: i32,
        n_ff: i32,
        topk: i32,
        t: i32,
        situ_beta: f32,
        situ_linear_beta: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_router_f32_batch(
        w_f32: *const f32,
        x_f32: *const f32,
        out_f32: *mut f32,
        n_expert: i32,
        n_embd: i32,
        t: i32,
        stream: CudaStreamHandle,
    );

    // Inkling-2 DSpark drafter kernels (`inkling2_draft.cu`)
    pub fn launch_dspark_fold_chunk(
        lm_q: *const i8,
        lm_qd: *const u16,
        lm_f16: *const u16,
        dht: *const f32,
        dhb: *const f32,
        dlm_tmp: *mut f32,
        r0: i32,
        m: i32,
        n_rows: i32,
        r: i32,
        c: i32,
        fold_w_f32: *mut f32,
        fold_b: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_fold_quant(
        fold_w_f32: *const f32,
        r: i32,
        n_rows: i32,
        fold_q: *mut i8,
        fold_sc: *mut f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_mix_rms(
        taps: *const f32,
        n_tap: i32,
        tap_stride_elems: usize,
        r0: i32,
        rows: i32,
        c: i32,
        alpha: *const f32,
        w: *const f32,
        eps: f32,
        out_f32: *mut f32,
        out_f16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_ctx_rope_kv_store(
        k_in: *const f32,
        v_in: *const f32,
        k_norm_w: *const f32,
        n_rows: i32,
        n_kv: i32,
        pos0: i32,
        cap: i32,
        theta: f32,
        eps: f32,
        is_fp8: i32,
        k_scale: f32,
        v_scale: f32,
        kc_dst: *mut c_void,
        vc_dst: *mut c_void,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_init_block(
        anchor_h: *const f32,
        mask: *const f32,
        res: *mut f32,
        state: *mut f32,
        pos: *mut i32,
        toks: *mut i32,
        par: *const i32,
        p: i32,
        anchor: i32,
        t: i32,
        c: i32,
        r: i32,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_add_rms_f16(
        src_x: *const f32,
        res: *mut f32,
        w: *const f32,
        eps: f32,
        t: i32,
        c: i32,
        h32: *mut f32,
        h16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_rms_neox_qk(
        q: *mut f32,
        k: *mut f32,
        qnw: *const f32,
        knw: *const f32,
        pos: *const i32,
        t: i32,
        h: i32,
        kv: i32,
        theta: f32,
        eps: f32,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_attn_mma(
        q: *const f32,
        kc: *const c_void,
        vc: *const c_void,
        kb: *const f32,
        vb: *const f32,
        k_lo: i32,
        k_hi: i32,
        par: *const i32,
        span: i32,
        cap: i32,
        is_fp8: i32,
        k_inv: f32,
        v_inv: f32,
        nq: i32,
        h: i32,
        hkv: i32,
        scratch: *mut f32,
        out16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_silu_mul_f16(
        gu: *const f32,
        t: i32,
        f: i32,
        out16: *mut u16,
        stream: CudaStreamHandle,
    );

    pub fn launch_dspark_rnn_step_argmax(
        rnn_w_bf16: *const u16,
        rnn_b: *const f32,
        w1_bf16: *const u16,
        fold_q: *const i8,
        fold_sc: *const f32,
        fold_b: *const u16,
        base: *const f32,
        h_step: *const f32,
        f_rnn: *const f32,
        r: i32,
        c: i32,
        n_rows: i32,
        n_vocab: i32,
        state: *mut f32,
        proj_tmp: *mut f32,
        fold_xs: *mut f32,
        part_vals: *mut f32,
        part_ids: *mut i32,
        prev_tok: *const i32,
        out_tok: *mut i32,
        stream: CudaStreamHandle,
    );
}
