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
}
