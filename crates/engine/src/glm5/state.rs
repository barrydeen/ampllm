use crate::glm5::cache::{DynamicExpertCache, GpuRegion};
use crate::glm5::weights::{CpuColdArena, Glm5ModelWeights};
use ampllm_cpu_kernels::glm5_cpu::{
    run_glm5_cpu_moe_grouped, run_glm5_cpu_moe_m1, Glm5CpuExpertJob, Glm5CpuGroupedExpert,
    Glm5CpuScratch,
};
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::{
    enable_peer_access, glm5::Glm5Doorbell, set_device, CudaEvent, CudaGraphExec, CudaStream,
    DeviceBuffer, PinnedMappedBuffer,
};
use ampllm_gguf::{GgmlType, Glm5Config};
use anyhow::{Context, Result};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

pub struct DsaLayerState {
    pub vram_lat8: DeviceBuffer,
    pub ram_lat8: Option<PinnedMappedBuffer<u8>>,
    pub ik_ring: DeviceBuffer,      // [4, idx_key] F32 (or prefill chunked)
    pub ig_ring: DeviceBuffer,      // [4, idx_key] F32
    pub pooled_f16: DeviceBuffer,   // [pool_cap, idx_key] F16
}

pub struct KdaLayerState {
    pub s_mat: DeviceBuffer,      // [n_head, head_dim, head_dim] F32
    pub conv_q: DeviceBuffer,     // [d_inner, 3] F32
    pub conv_k: DeviceBuffer,     // [d_inner, 3] F32
    pub conv_v: DeviceBuffer,     // [d_inner, 3] F32
}

pub struct GpuScratch {
    pub device: i32,
    pub max_batch: usize,
    pub max_prefill_chunk: usize,
    pub stream: CudaStream,
    // Residual [0]: [max_prefill_chunk, hc, n_embd]; [1]: [max_batch, hc, n_embd]
    pub r_buf: [DeviceBuffer; 2],
    // Persistent cross-phase buffers (sized for max_prefill_chunk)
    pub emb: DeviceBuffer,        // [max_prefill_chunk, n_embd]
    pub post: DeviceBuffer,       // [max_prefill_chunk, hc]
    pub comb: DeviceBuffer,       // [max_prefill_chunk, hc * hc]
    pub x_f32: DeviceBuffer,      // [max_prefill_chunk, n_embd]
    pub x_q8: DeviceBuffer,       // [max_prefill_chunk, (n_embd / 32) * 36]
    pub mixer: DeviceBuffer,      // [max_prefill_chunk, n_embd]
    pub sh_out: DeviceBuffer,     // [max_prefill_chunk, n_embd]
    pub gpu_moe_out: DeviceBuffer,// [max_prefill_chunk, n_embd]
    pub pos_arr_dev: DeviceBuffer,// [max_prefill_chunk] i32
    pub skip_flag_dev: DeviceBuffer,// [1] i32 SPEC_CANCEL device mirror
    pub mtp_pos_dev: DeviceBuffer,// [1] i32 dedicated blk.45 MTP position
    pub db_seq_state: DeviceBuffer,// [1] u64
    pub db_expected_seq: DeviceBuffer,// [n_layer_total] u64
    pub hc_part_dev: DeviceBuffer,// [26 * 32] f32 multi-block HC reduction scratch
    pub hc_ctr_dev: DeviceBuffer,// [1] u64 HC grid-barrier counter
    pub pf_stream: CudaStream,    // side stream for lookahead/prefill expert PCIe streaming
    pub d2h_stream: CudaStream,   // dedicated D2H DMA stream so prefill x_f32 D2H never waits behind H2D
    pub promo_stream: CudaStream, // async promotion DMA stream overlapped with next layers
    pub ev_promo_start: CudaEvent,
    pub ev_promo_done: CudaEvent,
    pub promo_src_dev: DeviceBuffer, // [48] u64 per-layer async promotion RAM src
    pub promo_dst_dev: DeviceBuffer, // [48] u64 per-layer async promotion VRAM dst
    pub ev_pf_start: CudaEvent,
    pub ev_pf_done: [CudaEvent; 2],
    pub ev_stream_copy: CudaEvent,
    pub ev_stream_comp: CudaEvent,
    pub ev_ring_ready: Vec<CudaEvent>,
    pub ev_ring_free: Vec<CudaEvent>,
    pub ev_meta_done: Vec<CudaEvent>,
    pub pf_logits: DeviceBuffer,  // [n_expert] f32 next-layer lookahead router logits
    pub pf_eids: [DeviceBuffer; 2], // [2] i32 double-buffered prefetched expert IDs
    pub pf_ptrs: [DeviceBuffer; 2], // [2] u64 double-buffered prefetched VRAM ptrs
    pub pf_src: [DeviceBuffer; 2],  // [2] u64 double-buffered prefetch source RAM ptrs
    pub pf_dst: [DeviceBuffer; 2],  // [2] u64 double-buffered prefetch dst VRAM ptrs
    pub pf_n: [DeviceBuffer; 2],    // [1] i32 double-buffered prefetch count
    pub mma_a_f16: DeviceBuffer,  // [max_batch, 16384] F16
    // Dual-lane PCIe + scratch MoE staging
    pub moe_scratch_pool: DeviceBuffer,     // [8 * max_blob_stride]
    pub moe_scratch_ptrs_dev: DeviceBuffer, // [16] u64 (8 default slots; rewritten to 16 when lent)
    pub moe_plan_ptrs_dev: DeviceBuffer,    // [32] u64 (m * topk, up to M=4)
    pub moe_fetch_src_dev: DeviceBuffer,    // [8] u64
    pub moe_fetch_dst_dev: DeviceBuffer,    // [8] u64
    pub moe_route_cnt_dev: DeviceBuffer,    // [1] u32
    // Shared phase-aliased scratch backing KDA / DSA / FFN+MoE / mma_b_f16 (~140 MiB instead of ~285 MiB)
    pub _phase_arena: DeviceBuffer,
    // KDA per-step scratch (GpuRegion views inside _phase_arena)
    pub q_p: GpuRegion,
    pub k_p: GpuRegion,
    pub v_p: GpuRegion,
    pub qc: GpuRegion,
    pub kc: GpuRegion,
    pub vc: GpuRegion,
    pub f_a: GpuRegion,
    pub g_raw: GpuRegion,
    pub g1: GpuRegion,
    pub beta: GpuRegion,
    pub g_a: GpuRegion,
    pub g2: GpuRegion,
    pub scan: GpuRegion,
    pub gated: GpuRegion,
    pub gated_q8: GpuRegion,
    // DSA per-step scratch (GpuRegion views inside _phase_arena)
    pub qr: GpuRegion,
    pub qr_q8: GpuRegion,
    pub kv: GpuRegion,
    pub q_full: GpuRegion,
    pub q_abs: GpuRegion,
    pub ik_tmp: GpuRegion,
    pub ig_tmp: GpuRegion,
    pub iq: GpuRegion,
    pub iw: GpuRegion,
    pub dsa_scores: GpuRegion,
    pub dsa_cells: GpuRegion,
    pub dsa_attn: GpuRegion,
    pub dsa_attn_q8: GpuRegion,
    // FFN / MoE scratch (GpuRegion views inside _phase_arena)
    pub gate: GpuRegion,
    pub up: GpuRegion,
    pub h_f32: GpuRegion,
    pub h_q8: GpuRegion,
    pub moe_logits: GpuRegion,
    pub moe_ids: GpuRegion,
    pub moe_rw: GpuRegion,
    pub moe_inter_q8: GpuRegion,
    pub mma_b_f16: GpuRegion,
    pub moe_rows_dev: GpuRegion,
    pub moe_wts_dev: GpuRegion,
    pub moe_tok_dn: GpuRegion,
    pub moe_exp_ptrs_dev: GpuRegion,
    pub moe_slot_off_dev: GpuRegion,
    pub moe_slot_nt_dev: GpuRegion,
    // Rotating grouped-MoE prefill metadata slots (16 deep, ~336 KiB total)
    pub moe_grp_stride_exps: usize,
    pub moe_tile_stride: usize,
    pub moe_grp_ptrs_dev: DeviceBuffer,
    pub moe_grp_off_dev: DeviceBuffer,
    pub moe_grp_nt_dev: DeviceBuffer,
    pub moe_tile_exp_dev: DeviceBuffer,
    pub moe_tile_moff_dev: DeviceBuffer,
    pub prefill_grp_ptrs_pin: PinnedMappedBuffer<u64>,
    pub prefill_grp_off_pin: PinnedMappedBuffer<i32>,
    pub prefill_grp_nt_pin: PinnedMappedBuffer<i32>,
    pub prefill_tile_exp_pin: PinnedMappedBuffer<i32>,
    pub prefill_tile_moff_pin: PinnedMappedBuffer<i32>,
    // Pinned host staging buffers for prefill MoE (independent per GPU; aliased io_pin saves 256 MiB)
    pub prefill_ids_pin: PinnedMappedBuffer<i32>,
    pub prefill_rw_pin: PinnedMappedBuffer<f32>,
    pub prefill_io_pin: PinnedMappedBuffer<f32>,
    pub prefill_exp_ptrs_pin: PinnedMappedBuffer<u64>,
    pub prefill_slot_off_pin: PinnedMappedBuffer<i32>,
    pub prefill_slot_nt_pin: PinnedMappedBuffer<i32>,
    pub prefill_rows_pin: PinnedMappedBuffer<i32>,
    pub prefill_wts_pin: PinnedMappedBuffer<f32>,
}

impl GpuScratch {
    pub fn alloc(
        dev: i32,
        cfg: &Glm5Config,
        pool_cap: usize,
        max_batch: usize,
        max_prefill_chunk: usize,
        max_lend_chunk: usize,
        max_blob_stride: usize,
        max_f16_w_elems: usize,
    ) -> Result<Self> {
        set_device(dev)?;
        let stream = CudaStream::new(dev)?;
        let n_sel = cfg.idx_top_k + cfg.idx_kpool - 1;
        let qk_out = cfg.n_head * cfg.qk_nope_head_dim;
        let v_out = cfg.n_head * cfg.v_head_dim;
        let q8 = |elems: usize| (elems / 32) * 36;
        let align256 = |x: usize| (x + 255) & !255;

        let min_f16_w_elems = (cfg.n_head * 32 * 514 * 2).max(4 * 1024 * 1024);
        let max_w_f16_elems = max_f16_w_elems.max(min_f16_w_elems);

        // Phase-aliased arena offsets:
        // [0 .. mma_b_off): tail mixer outputs (dsa_attn/dsa_attn_q8 | gated/gated_q8 | gate/up)
        let sz_dsa_attn = max_batch * v_out * 4;
        let sz_dsa_attn_q8 = max_batch * q8(v_out);
        let sz_gated = max_batch * cfg.d_inner() * 4;
        let sz_gated_q8 = max_batch * q8(cfg.d_inner());
        let sz_gate = max_batch * cfg.n_ff_dense * 4;
        let sz_up = max_batch * cfg.n_ff_dense * 4;

        let end_dsa_tail = align256(sz_dsa_attn) + align256(sz_dsa_attn_q8);
        let end_kda_tail = align256(sz_gated) + align256(sz_gated_q8);
        let end_ffn_head = align256(sz_gate) + align256(sz_up);
        let mma_b_off = end_dsa_tail.max(end_kda_tail).max(end_ffn_head);
        let sz_mma_b = max_w_f16_elems * 2;

        // DSA pre-attn intermediates sit cleanly after max pre-attn DSA F16 weight (and FlashMLA split scratch)
        let dsa_pre_max_w = (cfg.n_head * cfg.qk_nope_head_dim * cfg.q_lora_rank * 2)
            .max(cfg.n_embd * cfg.q_lora_rank * 2)
            .max(cfg.idx_heads * cfg.idx_key_dim * cfg.q_lora_rank * 2)
            .min(sz_mma_b)
            .max(cfg.n_head * 32 * 514 * 4);
        debug_assert!(
            sz_mma_b >= cfg.n_head * 32 * 514 * 4,
            "FlashMLA split scratch must fit in mma_b_f16"
        );
        let mut cur_dsa = align256(mma_b_off + dsa_pre_max_w);
        let mut alloc_dsa = |sz: usize| {
            let o = cur_dsa;
            cur_dsa += align256(sz);
            (o, sz)
        };
        let (o_qr, sz_qr) = alloc_dsa(max_batch * cfg.q_lora_rank * 4);
        let (o_qr_q8, sz_qr_q8) = alloc_dsa(max_batch * q8(cfg.q_lora_rank));
        let (o_kv, sz_kv) = alloc_dsa(max_batch * cfg.kv_lora_rank * 4);
        let (o_q_full, sz_q_full) = alloc_dsa(max_batch * qk_out * 4);
        let (o_q_abs, sz_q_abs) = alloc_dsa(max_batch * cfg.n_head * cfg.kv_lora_rank * 4);
        let (o_ik_tmp, sz_ik_tmp) = alloc_dsa(max_batch * cfg.idx_key_dim * 4);
        let (o_ig_tmp, sz_ig_tmp) = alloc_dsa(max_batch * cfg.idx_key_dim * 4);
        let (o_iq, sz_iq) = alloc_dsa(max_batch * cfg.idx_heads * cfg.idx_key_dim * 4);
        let (o_iw, sz_iw) = alloc_dsa(max_batch * cfg.idx_heads * 4);
        let (o_dsa_scores, sz_dsa_scores) = alloc_dsa(max_batch * pool_cap * 4);
        let (o_dsa_cells, sz_dsa_cells) = alloc_dsa(max_batch * n_sel * 4);

        // KDA intermediates sit cleanly after max KDA F16 weight
        let kda_max_w = (cfg.n_embd * cfg.d_inner() * 2)
            .max(cfg.kda_head_dim * cfg.d_inner() * 2)
            .min(sz_mma_b);
        let mut cur_kda = align256(mma_b_off + kda_max_w);
        let mut alloc_kda = |sz: usize| {
            let o = cur_kda;
            cur_kda += align256(sz);
            (o, sz)
        };
        let (o_qp, sz_qp) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_kp, sz_kp) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_vp, sz_vp) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_qc, sz_qc) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_kc, sz_kc) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_vc, sz_vc) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_fa, sz_fa) = alloc_kda(max_batch * cfg.kda_head_dim * 4);
        let (o_g_raw, sz_g_raw) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_g1, sz_g1) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_beta, sz_beta) = alloc_kda(max_batch * cfg.n_head * 4);
        let (o_ga, sz_ga) = alloc_kda(max_batch * cfg.kda_head_dim * 4);
        let (o_g2, sz_g2) = alloc_kda(max_batch * cfg.d_inner() * 4);
        let (o_scan, sz_scan) = alloc_kda(max_batch * cfg.d_inner() * 4);

        // Dense FFN (h_f32/h_q8) sits after max FFN F16 weight AND after cur_kda (for KDA M=1 h_q8 reuse)
        let ffn_max_w = (cfg.n_embd * cfg.n_ff_dense * 2).min(sz_mma_b);
        let mut cur_ffn = align256(mma_b_off + ffn_max_w).max(cur_kda);
        let mut alloc_ffn = |sz: usize| {
            let o = cur_ffn;
            cur_ffn += align256(sz);
            (o, sz)
        };
        let (o_h_f32, sz_h_f32) = alloc_ffn(max_batch * cfg.n_ff_dense * 4);
        let (o_h_q8, sz_h_q8) = alloc_ffn(max_batch * q8(cfg.n_ff_dense));

        // MoE scratch sits after max MoE F16 weight AND after cur_ffn (so h_f32/h_q8 remain disjoint)
        let moe_max_w = (cfg.n_embd * cfg.n_ff_shexp * cfg.n_shared_expert * 2)
            .max(cfg.n_embd * 2 * cfg.n_ff_exp * 2)
            .min(sz_mma_b);
        let mut cur_moe = align256(mma_b_off + moe_max_w).max(cur_ffn);
        let mut alloc_moe = |sz: usize| {
            let o = cur_moe;
            cur_moe += align256(sz);
            (o, sz)
        };
        let (o_moe_logits, sz_moe_logits) = alloc_moe(max_prefill_chunk * cfg.n_expert * 4);
        let (o_moe_ids, sz_moe_ids) = alloc_moe(max_prefill_chunk * cfg.n_expert_used * 4);
        let (o_moe_rw, sz_moe_rw) = alloc_moe(max_prefill_chunk * cfg.n_expert_used * 4);
        let (o_moe_inter_q8, sz_moe_inter_q8) = alloc_moe(max_prefill_chunk * cfg.n_expert_used * q8(cfg.n_ff_exp));
        let (o_moe_rows_dev, sz_moe_rows_dev) = alloc_moe(max_prefill_chunk * cfg.n_expert_used * 4);
        let (o_moe_wts_dev, sz_moe_wts_dev) = alloc_moe(max_prefill_chunk * cfg.n_expert_used * 4);
        let (o_moe_tok_dn, sz_moe_tok_dn) = alloc_moe(max_prefill_chunk * cfg.n_embd * 4);
        let (o_moe_exp_ptrs_dev, sz_moe_exp_ptrs_dev) = alloc_moe(cfg.n_expert * 8);
        let (o_moe_slot_off_dev, sz_moe_slot_off_dev) = alloc_moe(cfg.n_expert * 4);
        let (o_moe_slot_nt_dev, sz_moe_slot_nt_dev) = alloc_moe(cfg.n_expert * 4);

        let phase_bytes = (mma_b_off + sz_mma_b)
            .max(cur_dsa)
            .max(cur_kda)
            .max(cur_ffn)
            .max(cur_moe);
        let phase_arena = DeviceBuffer::alloc(dev, phase_bytes)?;

        let slot_stride = (max_blob_stride.max(4096) + 15) & !15;
        let moe_scratch_pool = DeviceBuffer::alloc_zeroed(dev, 8 * slot_stride)?;
        let scratch_base = moe_scratch_pool.as_u64();
        let scratch_ptrs_h: [u64; 8] =
            std::array::from_fn(|i| scratch_base + (i * slot_stride) as u64);
        let moe_scratch_ptrs_dev = DeviceBuffer::alloc(dev, 16 * 8)?;
        moe_scratch_ptrs_dev.copy_from_host_async(bytemuck::cast_slice(&scratch_ptrs_h), &stream)?;
        let moe_plan_ptrs_dev = DeviceBuffer::alloc_zeroed(dev, 32 * 8)?;
        let moe_fetch_src_dev = DeviceBuffer::alloc_zeroed(dev, 8 * 8)?;
        let moe_fetch_dst_dev = DeviceBuffer::alloc_zeroed(dev, 8 * 8)?;
        let moe_route_cnt_dev = DeviceBuffer::alloc_zeroed(dev, 4)?;
        let pf_stream = CudaStream::new(dev)?;
        let d2h_stream = CudaStream::new(dev)?;
        let promo_stream = CudaStream::new(dev)?;
        let ev_promo_start = CudaEvent::new(dev)?;
        let ev_promo_done = CudaEvent::new(dev)?;
        let promo_src_dev = DeviceBuffer::alloc_zeroed(dev, 48 * 8)?;
        let promo_dst_dev = DeviceBuffer::alloc_zeroed(dev, 48 * 8)?;
        let ev_pf_start = CudaEvent::new(dev)?;
        let ev_pf_done = [CudaEvent::new(dev)?, CudaEvent::new(dev)?];
        let ev_stream_copy = CudaEvent::new(dev)?;
        let ev_stream_comp = CudaEvent::new(dev)?;
        let mut ev_ring_ready = Vec::with_capacity(16);
        let mut ev_ring_free = Vec::with_capacity(16);
        let mut ev_meta_done = Vec::with_capacity(16);
        for _ in 0..16 {
            ev_ring_ready.push(CudaEvent::new(dev)?);
            ev_ring_free.push(CudaEvent::new(dev)?);
            ev_meta_done.push(CudaEvent::new(dev)?);
        }
        let meta_slots = 16usize;
        let moe_grp_stride_exps = cfg.n_expert.max(1);
        let moe_tile_stride = 2048usize;
        let moe_grp_ptrs_dev = DeviceBuffer::alloc(dev, meta_slots * moe_grp_stride_exps * 8)?;
        let moe_grp_off_dev = DeviceBuffer::alloc(dev, meta_slots * moe_grp_stride_exps * 4)?;
        let moe_grp_nt_dev = DeviceBuffer::alloc(dev, meta_slots * moe_grp_stride_exps * 4)?;
        let moe_tile_exp_dev = DeviceBuffer::alloc(dev, meta_slots * moe_tile_stride * 4)?;
        let moe_tile_moff_dev = DeviceBuffer::alloc(dev, meta_slots * moe_tile_stride * 4)?;
        let prefill_grp_ptrs_pin = PinnedMappedBuffer::<u64>::alloc(meta_slots * moe_grp_stride_exps)?;
        let prefill_grp_off_pin = PinnedMappedBuffer::<i32>::alloc(meta_slots * moe_grp_stride_exps)?;
        let prefill_grp_nt_pin = PinnedMappedBuffer::<i32>::alloc(meta_slots * moe_grp_stride_exps)?;
        let prefill_tile_exp_pin = PinnedMappedBuffer::<i32>::alloc(meta_slots * moe_tile_stride)?;
        let prefill_tile_moff_pin = PinnedMappedBuffer::<i32>::alloc(meta_slots * moe_tile_stride)?;
        let pf_logits = DeviceBuffer::alloc_zeroed(dev, cfg.n_expert * 4)?;
        let pf_eid_init = [-1i32; 2];
        let pf_eid0 = DeviceBuffer::alloc(dev, 8)?;
        pf_eid0.copy_from_host_async(bytemuck::cast_slice(&pf_eid_init), &stream)?;
        let pf_eid1 = DeviceBuffer::alloc(dev, 8)?;
        pf_eid1.copy_from_host_async(bytemuck::cast_slice(&pf_eid_init), &stream)?;
        let pf_eids = [pf_eid0, pf_eid1];
        let pf_ptrs = [DeviceBuffer::alloc_zeroed(dev, 16)?, DeviceBuffer::alloc_zeroed(dev, 16)?];
        let pf_src = [DeviceBuffer::alloc_zeroed(dev, 16)?, DeviceBuffer::alloc_zeroed(dev, 16)?];
        let pf_dst = [DeviceBuffer::alloc_zeroed(dev, 16)?, DeviceBuffer::alloc_zeroed(dev, 16)?];
        let pf_n = [DeviceBuffer::alloc_zeroed(dev, 4)?, DeviceBuffer::alloc_zeroed(dev, 4)?];

        let max_pin_toks = max_lend_chunk.max(max_prefill_chunk);
        let prefill_ids_pin = PinnedMappedBuffer::<i32>::alloc(max_pin_toks * cfg.n_expert_used)?;
        let prefill_rw_pin = PinnedMappedBuffer::<f32>::alloc(max_pin_toks * cfg.n_expert_used)?;
        let prefill_io_pin = PinnedMappedBuffer::<f32>::alloc(max_pin_toks * cfg.n_embd)?;
        let prefill_exp_ptrs_pin = PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?;
        let prefill_slot_off_pin = PinnedMappedBuffer::<i32>::alloc(cfg.n_expert)?;
        let prefill_slot_nt_pin = PinnedMappedBuffer::<i32>::alloc(cfg.n_expert)?;
        let prefill_rows_pin = PinnedMappedBuffer::<i32>::alloc(max_pin_toks * cfg.n_expert_used)?;
        let prefill_wts_pin = PinnedMappedBuffer::<f32>::alloc(max_pin_toks * cfg.n_expert_used)?;

        Ok(Self {
            device: dev,
            max_batch,
            max_prefill_chunk,
            stream,
            r_buf: [
                DeviceBuffer::alloc_zeroed(dev, max_prefill_chunk * cfg.hc * cfg.n_embd * 4)?,
                DeviceBuffer::alloc_zeroed(dev, max_batch * cfg.hc * cfg.n_embd * 4)?,
            ],
            emb: DeviceBuffer::alloc(dev, max_prefill_chunk * cfg.n_embd * 4)?,
            post: DeviceBuffer::alloc(dev, max_prefill_chunk * cfg.hc * 4)?,
            comb: DeviceBuffer::alloc(dev, max_prefill_chunk * cfg.hc * cfg.hc * 4)?,
            x_f32: DeviceBuffer::alloc(dev, max_prefill_chunk * cfg.n_embd * 4)?,
            x_q8: DeviceBuffer::alloc(dev, max_prefill_chunk * q8(cfg.n_embd))?,
            mixer: DeviceBuffer::alloc(dev, max_prefill_chunk * cfg.n_embd * 4)?,
            sh_out: DeviceBuffer::alloc(dev, max_prefill_chunk * cfg.n_embd * 4)?,
            gpu_moe_out: DeviceBuffer::alloc_zeroed(dev, max_prefill_chunk * cfg.n_embd * 4)?,
            pos_arr_dev: DeviceBuffer::alloc_zeroed(dev, max_prefill_chunk * 4)?,
            skip_flag_dev: DeviceBuffer::alloc_zeroed(dev, 4)?,
            mtp_pos_dev: DeviceBuffer::alloc_zeroed(dev, 4)?,
            db_seq_state: DeviceBuffer::alloc_zeroed(dev, 16)?,
            db_expected_seq: DeviceBuffer::alloc_zeroed(dev, cfg.n_layer_total * 8)?,
            hc_part_dev: DeviceBuffer::alloc_zeroed(dev, 26 * 32 * 4)?,
            hc_ctr_dev: DeviceBuffer::alloc_zeroed(dev, 8)?,
            pf_stream,
            d2h_stream,
            promo_stream,
            ev_promo_start,
            ev_promo_done,
            promo_src_dev,
            promo_dst_dev,
            ev_pf_start,
            ev_pf_done,
            ev_stream_copy,
            ev_stream_comp,
            ev_ring_ready,
            ev_ring_free,
            ev_meta_done,
            pf_logits,
            pf_eids,
            pf_ptrs,
            pf_src,
            pf_dst,
            pf_n,
            mma_a_f16: DeviceBuffer::alloc(dev, max_batch * 16384 * 2)?,

            moe_scratch_pool,
            moe_scratch_ptrs_dev,
            moe_plan_ptrs_dev,
            moe_fetch_src_dev,
            moe_fetch_dst_dev,
            moe_route_cnt_dev,

            q_p: GpuRegion::sub(&phase_arena, o_qp, sz_qp),
            k_p: GpuRegion::sub(&phase_arena, o_kp, sz_kp),
            v_p: GpuRegion::sub(&phase_arena, o_vp, sz_vp),
            qc: GpuRegion::sub(&phase_arena, o_qc, sz_qc),
            kc: GpuRegion::sub(&phase_arena, o_kc, sz_kc),
            vc: GpuRegion::sub(&phase_arena, o_vc, sz_vc),
            f_a: GpuRegion::sub(&phase_arena, o_fa, sz_fa),
            g_raw: GpuRegion::sub(&phase_arena, o_g_raw, sz_g_raw),
            g1: GpuRegion::sub(&phase_arena, o_g1, sz_g1),
            beta: GpuRegion::sub(&phase_arena, o_beta, sz_beta),
            g_a: GpuRegion::sub(&phase_arena, o_ga, sz_ga),
            g2: GpuRegion::sub(&phase_arena, o_g2, sz_g2),
            scan: GpuRegion::sub(&phase_arena, o_scan, sz_scan),
            gated: GpuRegion::sub(&phase_arena, 0, sz_gated),
            gated_q8: GpuRegion::sub(&phase_arena, align256(sz_gated), sz_gated_q8),

            qr: GpuRegion::sub(&phase_arena, o_qr, sz_qr),
            qr_q8: GpuRegion::sub(&phase_arena, o_qr_q8, sz_qr_q8),
            kv: GpuRegion::sub(&phase_arena, o_kv, sz_kv),
            q_full: GpuRegion::sub(&phase_arena, o_q_full, sz_q_full),
            q_abs: GpuRegion::sub(&phase_arena, o_q_abs, sz_q_abs),
            ik_tmp: GpuRegion::sub(&phase_arena, o_ik_tmp, sz_ik_tmp),
            ig_tmp: GpuRegion::sub(&phase_arena, o_ig_tmp, sz_ig_tmp),
            iq: GpuRegion::sub(&phase_arena, o_iq, sz_iq),
            iw: GpuRegion::sub(&phase_arena, o_iw, sz_iw),
            dsa_scores: GpuRegion::sub(&phase_arena, o_dsa_scores, sz_dsa_scores),
            dsa_cells: GpuRegion::sub(&phase_arena, o_dsa_cells, sz_dsa_cells),
            dsa_attn: GpuRegion::sub(&phase_arena, 0, sz_dsa_attn),
            dsa_attn_q8: GpuRegion::sub(&phase_arena, align256(sz_dsa_attn), sz_dsa_attn_q8),

            gate: GpuRegion::sub(&phase_arena, 0, sz_gate),
            up: GpuRegion::sub(&phase_arena, align256(sz_gate), sz_up),
            h_f32: GpuRegion::sub(&phase_arena, o_h_f32, sz_h_f32),
            h_q8: GpuRegion::sub(&phase_arena, o_h_q8, sz_h_q8),
            moe_logits: GpuRegion::sub(&phase_arena, o_moe_logits, sz_moe_logits),
            moe_ids: GpuRegion::sub(&phase_arena, o_moe_ids, sz_moe_ids),
            moe_rw: GpuRegion::sub(&phase_arena, o_moe_rw, sz_moe_rw),
            moe_inter_q8: GpuRegion::sub(&phase_arena, o_moe_inter_q8, sz_moe_inter_q8),
            mma_b_f16: GpuRegion::sub(&phase_arena, mma_b_off, sz_mma_b),
            moe_rows_dev: GpuRegion::sub(&phase_arena, o_moe_rows_dev, sz_moe_rows_dev),
            moe_wts_dev: GpuRegion::sub(&phase_arena, o_moe_wts_dev, sz_moe_wts_dev),
            moe_tok_dn: GpuRegion::sub(&phase_arena, o_moe_tok_dn, sz_moe_tok_dn),
            moe_exp_ptrs_dev: GpuRegion::sub(&phase_arena, o_moe_exp_ptrs_dev, sz_moe_exp_ptrs_dev),
            moe_slot_off_dev: GpuRegion::sub(&phase_arena, o_moe_slot_off_dev, sz_moe_slot_off_dev),
            moe_slot_nt_dev: GpuRegion::sub(&phase_arena, o_moe_slot_nt_dev, sz_moe_slot_nt_dev),

            moe_grp_stride_exps,
            moe_tile_stride,
            moe_grp_ptrs_dev,
            moe_grp_off_dev,
            moe_grp_nt_dev,
            moe_tile_exp_dev,
            moe_tile_moff_dev,
            prefill_grp_ptrs_pin,
            prefill_grp_off_pin,
            prefill_grp_nt_pin,
            prefill_tile_exp_pin,
            prefill_tile_moff_pin,

            _phase_arena: phase_arena,

            prefill_ids_pin,
            prefill_rw_pin,
            prefill_io_pin,
            prefill_exp_ptrs_pin,
            prefill_slot_off_pin,
            prefill_slot_nt_pin,
            prefill_rows_pin,
            prefill_wts_pin,
        })
    }
}

pub struct DoorbellWorker {
    pub active: Arc<AtomicBool>,
    pub stop: Arc<AtomicBool>,
    pub miss_experts_total: Arc<AtomicU64>,
    pub unique_cold_experts_total: Arc<AtomicU64>,
    pub miss_calls_total: Arc<AtomicU64>,
    pub cpu_busy_ns: Arc<AtomicU64>,
    pub cpu_busy_ns_g0: Arc<AtomicU64>,
    pub cpu_busy_ns_g1: Arc<AtomicU64>,
    pub db0: PinnedMappedBuffer<Glm5Doorbell>,
    pub db1: PinnedMappedBuffer<Glm5Doorbell>,
    pub db0_dev_ptr: *mut c_void,
    pub db1_dev_ptr: *mut c_void,
    pub pool: Arc<CpuThreadPool>,
    pub pool0: Arc<CpuThreadPool>,
    pub pool1: Arc<CpuThreadPool>,
    pub handle0: Option<JoinHandle<()>>,
    pub handle1: Option<JoinHandle<()>>,
}
unsafe impl Send for DoorbellWorker {}
unsafe impl Sync for DoorbellWorker {}

impl Drop for DoorbellWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle0.take() {
            let _ = h.join();
        }
        if let Some(h) = self.handle1.take() {
            let _ = h.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_db_worker(
    name: &str,
    is_db0: bool,
    db_addr: usize,
    pool_w: Arc<CpuThreadPool>,
    pool_all: Arc<CpuThreadPool>,
    active_w: Arc<AtomicBool>,
    stop_w: Arc<AtomicBool>,
    miss_exp_w: Arc<AtomicU64>,
    uniq_exp_w: Arc<AtomicU64>,
    miss_calls_w: Arc<AtomicU64>,
    cpu_ns_w: Arc<AtomicU64>,
    cpu_half_ns_w: Arc<AtomicU64>,
    layers_atomic: Arc<Vec<Vec<std::sync::atomic::AtomicUsize>>>,
    layer_u_off: Vec<usize>,
    layer_d_off: Vec<usize>,
) -> Result<JoinHandle<()>> {
    let handle = std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            if let Some(&c0) = pool_w.physical_cores().first() {
                ampllm_cpu_kernels::pool::pin_current_thread(c0);
            }
            let mut scratch = Glm5CpuScratch::new(4096, 2048, 32);
            let mut x_stage = vec![0.0f32; 4 * 4096];
            let mut out_stage = vec![0.0f32; 4 * 4096];
            let mut grp_eids = [0usize; 32];
            let mut grp_toks = [[0usize; 4]; 32];
            let mut grp_wts = [[0.0f32; 4]; 32];
            let mut grp_len = [0usize; 32];
            let mut seen = 0u64;

            while !stop_w.load(Ordering::Relaxed) {
                let did = unsafe {
                    let db = &mut *(db_addr as *mut Glm5Doorbell);
                    let req = std::ptr::read_volatile(&db.req_seq);
                    if req <= seen {
                        false
                    } else {
                        std::sync::atomic::fence(Ordering::Acquire);
                        if is_db0 && std::ptr::read_volatile(&db.cancel_flag) != 0 {
                            std::sync::atomic::fence(Ordering::Release);
                            std::ptr::write_volatile(&mut db.done_seq, req);
                            seen = req;
                            true
                        } else {
                            let t0 = Instant::now();
                            let mb = (db.m_batch as usize).clamp(1, 4);
                            let l = db.layer_idx as usize;
                            let gu_type =
                                GgmlType::from_u32(db.gu_type as u32).unwrap_or(GgmlType::Iq2S);
                            let d_type =
                                GgmlType::from_u32(db.d_type as u32).unwrap_or(GgmlType::Iq3Xxs);
                            let n_embd = db.n_embd as usize;
                            let n_ff = db.n_ff as usize;

                            if mb == 1 {
                                let mc = (db.miss_count as usize).min(8);
                                if mc > 0 {
                                    let u_off = layer_u_off.get(l).copied().unwrap_or(0);
                                    let d_off = layer_d_off.get(l).copied().unwrap_or(0);
                                    let mut jobs_arr = [Glm5CpuExpertJob {
                                        gate_ptr: std::ptr::null(),
                                        up_ptr: std::ptr::null(),
                                        down_ptr: std::ptr::null(),
                                        weight: 0.0,
                                    }; 8];
                                    let mut n_jobs = 0usize;
                                    for i in 0..mc {
                                        let eid = db.miss_ids[i] as usize;
                                        if let Some(atom) =
                                            layers_atomic.get(l).and_then(|r| r.get(eid))
                                        {
                                            let base = atom.load(Ordering::Acquire);
                                            if base != 0 {
                                                jobs_arr[n_jobs] = Glm5CpuExpertJob {
                                                    gate_ptr: base as *const u8,
                                                    up_ptr: (base + u_off) as *const u8,
                                                    down_ptr: (base + d_off) as *const u8,
                                                    weight: db.miss_weights[i],
                                                };
                                                n_jobs += 1;
                                            }
                                        }
                                    }
                                    let x_q8k_pre = if db.x_q8k_valid != 0 {
                                        let bytes = (n_embd / 256)
                                            * ampllm_cpu_kernels::glm5_cpu::Q8K_BLOCK_BYTES;
                                        Some(&db.x_q8k[..bytes])
                                    } else {
                                        None
                                    };
                                    run_glm5_cpu_moe_m1(
                                        &pool_w,
                                        &mut scratch,
                                        gu_type,
                                        d_type,
                                        n_embd,
                                        n_ff,
                                        db.clamp_lim,
                                        &db.x_in[..n_embd],
                                        x_q8k_pre,
                                        &jobs_arr[..n_jobs],
                                        &mut db.out[..n_embd],
                                    );
                                    uniq_exp_w.fetch_add(n_jobs as u64, Ordering::Relaxed);
                                }
                                miss_calls_w.fetch_add(1, Ordering::Relaxed);
                                miss_exp_w.fetch_add(mc as u64, Ordering::Relaxed);
                            } else {
                                let mc = (db.miss_count as usize).min(32);
                                if mc > 0 {
                                    let x_len = mb * n_embd;
                                    x_stage[..x_len].copy_from_slice(&db.x_in[..x_len]);
                                    let mut n_groups = 0usize;
                                    for i in 0..mc {
                                        let eid = db.miss_ids[i] as usize;
                                        let tok = db.miss_tok[i] as usize;
                                        let wt = db.miss_weights[i];
                                        if tok >= mb {
                                            continue;
                                        }
                                        if let Some(g) =
                                            (0..n_groups).find(|&g| grp_eids[g] == eid)
                                        {
                                            let k = grp_len[g];
                                            if k < 4 {
                                                grp_toks[g][k] = tok;
                                                grp_wts[g][k] = wt;
                                                grp_len[g] = k + 1;
                                            }
                                        } else if n_groups < 32 {
                                            grp_eids[n_groups] = eid;
                                            grp_toks[n_groups][0] = tok;
                                            grp_wts[n_groups][0] = wt;
                                            grp_len[n_groups] = 1;
                                            n_groups += 1;
                                        }
                                    }
                                    let u_off = layer_u_off.get(l).copied().unwrap_or(0);
                                    let d_off = layer_d_off.get(l).copied().unwrap_or(0);
                                    let mut groups = Vec::with_capacity(n_groups);
                                    for g in 0..n_groups {
                                        let eid = grp_eids[g];
                                        let k = grp_len[g];
                                        if let Some(atom) =
                                            layers_atomic.get(l).and_then(|r| r.get(eid))
                                        {
                                            let base = atom.load(Ordering::Acquire);
                                            if base != 0 {
                                                groups.push(Glm5CpuGroupedExpert {
                                                    gate_ptr: base as *const u8,
                                                    up_ptr: (base + u_off) as *const u8,
                                                    down_ptr: (base + d_off) as *const u8,
                                                    token_indices: &grp_toks[g][..k],
                                                    token_weights: &grp_wts[g][..k],
                                                });
                                            }
                                        }
                                    }
                                    run_glm5_cpu_moe_grouped(
                                        &pool_all,
                                        &mut scratch,
                                        gu_type,
                                        d_type,
                                        n_embd,
                                        n_ff,
                                        db.clamp_lim,
                                        mb,
                                        &x_stage[..x_len],
                                        &groups,
                                        &mut out_stage[..x_len],
                                        true,
                                    );
                                    db.out[..x_len].copy_from_slice(&out_stage[..x_len]);
                                    uniq_exp_w.fetch_add(groups.len() as u64, Ordering::Relaxed);
                                }
                                miss_calls_w.fetch_add(mb as u64, Ordering::Relaxed);
                                miss_exp_w.fetch_add(mc as u64, Ordering::Relaxed);
                            }

                            let elapsed = t0.elapsed().as_nanos() as u64;
                            cpu_ns_w.fetch_add(elapsed, Ordering::Relaxed);
                            cpu_half_ns_w.fetch_add(elapsed, Ordering::Relaxed);
                            std::sync::atomic::fence(Ordering::Release);
                            std::ptr::write_volatile(&mut db.done_seq, req);
                            seen = req;
                            true
                        }
                    }
                };
                if !did {
                    if active_w.load(Ordering::Relaxed) {
                        std::hint::spin_loop();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
            }
        })?;
    Ok(handle)
}

pub(crate) struct DoorbellActiveGuard(Arc<AtomicBool>);

impl DoorbellActiveGuard {
    pub(crate) fn new(flag: &Arc<AtomicBool>) -> Self {
        flag.store(true, Ordering::Release);
        Self(Arc::clone(flag))
    }
}

impl Drop for DoorbellActiveGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl DoorbellWorker {
    pub fn spawn(pool: Arc<CpuThreadPool>, arena: &CpuColdArena) -> Result<Self> {
        let mut db0 = PinnedMappedBuffer::<Glm5Doorbell>::alloc(1)?;
        let mut db1 = PinnedMappedBuffer::<Glm5Doorbell>::alloc(1)?;
        db0.as_mut_slice()[0] = Glm5Doorbell::default();
        db1.as_mut_slice()[0] = Glm5Doorbell::default();
        let db0_dev_ptr = db0.dev_ptr_for_device(0)?;
        let db1_dev_ptr = db1.dev_ptr_for_device(1)?;

        let (cores0, cores1) = if std::env::var("AMPLLM_CPU_TOPOLOGY").as_deref() == Ok("cross") {
            ampllm_cpu_kernels::pool::discover_cross_ccd_interleaved_cores()
        } else if std::env::var("AMPLLM_CPU_SMT").as_deref() == Ok("1") {
            ampllm_cpu_kernels::pool::discover_ccd_split_cores_smt()
        } else {
            ampllm_cpu_kernels::pool::discover_ccd_split_cores()
        };
        tracing::info!("GLM-5.3 CPU pools: GPU0={cores0:?}, GPU1={cores1:?}");
        let c0_pin = cores0.first().copied();
        let c1_pin = cores1.first().copied();
        let mut batch_cores = Vec::with_capacity(cores0.len() + cores1.len());
        if let Some(c) = c0_pin.or(c1_pin) {
            batch_cores.push(c);
        }
        for &c in cores0.iter().chain(cores1.iter()) {
            let start = 1.min(batch_cores.len());
            if Some(c) != c0_pin && Some(c) != c1_pin && !batch_cores[start..].contains(&c) {
                batch_cores.push(c);
            }
        }
        tracing::info!(
            "GLM-5.3 batch CPU pool ({} threads): workers={:?}",
            batch_cores.len(),
            &batch_cores[1.min(batch_cores.len())..],
        );
        let batch_pool = Arc::new(CpuThreadPool::from_cores(batch_cores));
        let pool0 = Arc::new(CpuThreadPool::from_cores(cores0));
        let pool1 = Arc::new(CpuThreadPool::from_cores(cores1));

        let active = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let miss_experts_total = Arc::new(AtomicU64::new(0));
        let unique_cold_experts_total = Arc::new(AtomicU64::new(0));
        let miss_calls_total = Arc::new(AtomicU64::new(0));
        let cpu_busy_ns = Arc::new(AtomicU64::new(0));
        let cpu_busy_ns_g0 = Arc::new(AtomicU64::new(0));
        let cpu_busy_ns_g1 = Arc::new(AtomicU64::new(0));

        let handle0 = spawn_db_worker(
            "glm5-cpu-db0",
            true,
            db0.host_ptr() as usize,
            Arc::clone(&pool0),
            Arc::clone(&batch_pool),
            Arc::clone(&active),
            Arc::clone(&stop),
            Arc::clone(&miss_experts_total),
            Arc::clone(&unique_cold_experts_total),
            Arc::clone(&miss_calls_total),
            Arc::clone(&cpu_busy_ns),
            Arc::clone(&cpu_busy_ns_g0),
            Arc::clone(&arena.layers_atomic),
            arena.layer_u_off.clone(),
            arena.layer_d_off.clone(),
        )?;
        let handle1 = spawn_db_worker(
            "glm5-cpu-db1",
            false,
            db1.host_ptr() as usize,
            Arc::clone(&pool1),
            Arc::clone(&batch_pool),
            Arc::clone(&active),
            Arc::clone(&stop),
            Arc::clone(&miss_experts_total),
            Arc::clone(&unique_cold_experts_total),
            Arc::clone(&miss_calls_total),
            Arc::clone(&cpu_busy_ns),
            Arc::clone(&cpu_busy_ns_g1),
            Arc::clone(&arena.layers_atomic),
            arena.layer_u_off.clone(),
            arena.layer_d_off.clone(),
        )?;

        Ok(Self {
            active,
            stop,
            miss_experts_total,
            unique_cold_experts_total,
            miss_calls_total,
            cpu_busy_ns,
            cpu_busy_ns_g0,
            cpu_busy_ns_g1,
            db0,
            db1,
            db0_dev_ptr,
            db1_dev_ptr,
            pool,
            pool0,
            pool1,
            handle0: Some(handle0),
            handle1: Some(handle1),
        })
    }
}

pub struct KdaTraceBuffers {
    pub q_p: DeviceBuffer,
    pub k_p: DeviceBuffer,
    pub v_p: DeviceBuffer,
    pub g1: DeviceBuffer,
    pub beta: DeviceBuffer,
}

pub struct LayerPhaseEvents {
    pub ev: [CudaEvent; 9],
}

impl LayerPhaseEvents {
    fn new(dev: i32) -> Result<Self> {
        Ok(Self {
            ev: [
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
                CudaEvent::new_timed(dev)?,
            ],
        })
    }
}

pub struct PhaseProfiler {
    pub ev0: [Vec<LayerPhaseEvents>; 2],
    pub ev0_h2d_start: [CudaEvent; 2],
    pub ev0_h2d_end: [CudaEvent; 2],
    pub ev0_emb_end: [CudaEvent; 2],
    pub ev0_hop_end: [CudaEvent; 2],
    pub ev1: Vec<LayerPhaseEvents>,
    pub ev1_hop_start: CudaEvent,
    pub ev1_hop_end: CudaEvent,
    pub ev1_pos_end: CudaEvent,
    pub ev1_head_start: CudaEvent,
    pub ev1_norm_end: CudaEvent,
    pub ev1_logits_end: CudaEvent,
    pub ev1_head_end: CudaEvent,
    pub ev1_d2h_end: CudaEvent,
    // [hc_attn, kda, dsa, hc_ffn, dense_ffn, moe_in, router, fetch, moe_plan, db_wait]
    pub g0_layers_ms: [f64; 10],
    pub g0_extra_ms: [f64; 3], // [h2d, emb_rep, hop_d2h]
    pub g1_layers_ms: [f64; 10],
    pub g1_extra_ms: [f64; 6], // [hop_h2d, pos_h2d, head_norm, head_gemv, head_argmax, d2h]
    pub steps: usize,
}

impl PhaseProfiler {
    pub fn alloc(split_layer: usize, n_layers: usize) -> Result<Self> {
        let mut ev0_a = Vec::with_capacity(split_layer);
        let mut ev0_b = Vec::with_capacity(split_layer);
        for _ in 0..split_layer {
            ev0_a.push(LayerPhaseEvents::new(0)?);
            ev0_b.push(LayerPhaseEvents::new(0)?);
        }
        let n1 = n_layers.saturating_sub(split_layer);
        let mut ev1 = Vec::with_capacity(n1);
        for _ in 0..n1 {
            ev1.push(LayerPhaseEvents::new(1)?);
        }
        Ok(Self {
            ev0: [ev0_a, ev0_b],
            ev0_h2d_start: [CudaEvent::new_timed(0)?, CudaEvent::new_timed(0)?],
            ev0_h2d_end: [CudaEvent::new_timed(0)?, CudaEvent::new_timed(0)?],
            ev0_emb_end: [CudaEvent::new_timed(0)?, CudaEvent::new_timed(0)?],
            ev0_hop_end: [CudaEvent::new_timed(0)?, CudaEvent::new_timed(0)?],
            ev1,
            ev1_hop_start: CudaEvent::new_timed(1)?,
            ev1_hop_end: CudaEvent::new_timed(1)?,
            ev1_pos_end: CudaEvent::new_timed(1)?,
            ev1_head_start: CudaEvent::new_timed(1)?,
            ev1_norm_end: CudaEvent::new_timed(1)?,
            ev1_logits_end: CudaEvent::new_timed(1)?,
            ev1_head_end: CudaEvent::new_timed(1)?,
            ev1_d2h_end: CudaEvent::new_timed(1)?,
            g0_layers_ms: [0.0; 10],
            g0_extra_ms: [0.0; 3],
            g1_layers_ms: [0.0; 10],
            g1_extra_ms: [0.0; 6],
            steps: 0,
        })
    }

    pub fn reset(&mut self) {
        self.g0_layers_ms = [0.0; 10];
        self.g0_extra_ms = [0.0; 3];
        self.g1_layers_ms = [0.0; 10];
        self.g1_extra_ms = [0.0; 6];
        self.steps = 0;
    }

    pub fn accumulate_step(&mut self, weights: &Glm5ModelWeights, slot0: usize) -> Result<()> {
        let split_l = weights.split_layer;
        let n_layers = weights.cfg.n_layers;
        let s0 = slot0 & 1;

        set_device(0)?;
        self.g0_extra_ms[0] += self.ev0_h2d_start[s0].elapsed_ms(&self.ev0_h2d_end[s0])? as f64;
        self.g0_extra_ms[1] += self.ev0_h2d_end[s0].elapsed_ms(&self.ev0_emb_end[s0])? as f64;
        for l in 0..split_l {
            let lw = &weights.layers[l];
            let le = &self.ev0[s0][l].ev;
            self.g0_layers_ms[0] += le[0].elapsed_ms(&le[1])? as f64;
            let mix_idx = if lw.is_kda { 1 } else { 2 };
            self.g0_layers_ms[mix_idx] += le[1].elapsed_ms(&le[2])? as f64;
            self.g0_layers_ms[3] += le[2].elapsed_ms(&le[3])? as f64;
            if !lw.is_moe {
                self.g0_layers_ms[4] += le[3].elapsed_ms(&le[4])? as f64;
            } else {
                self.g0_layers_ms[5] += le[3].elapsed_ms(&le[4])? as f64;
                self.g0_layers_ms[6] += le[4].elapsed_ms(&le[5])? as f64;
                self.g0_layers_ms[7] += le[5].elapsed_ms(&le[6])? as f64;
                self.g0_layers_ms[8] += le[6].elapsed_ms(&le[7])? as f64;
                self.g0_layers_ms[9] += le[7].elapsed_ms(&le[8])? as f64;
            }
        }
        if split_l > 0 {
            let last_ev = if weights.layers[split_l - 1].is_moe {
                &self.ev0[s0][split_l - 1].ev[8]
            } else {
                &self.ev0[s0][split_l - 1].ev[4]
            };
            self.g0_extra_ms[2] += last_ev.elapsed_ms(&self.ev0_hop_end[s0])? as f64;
        }

        set_device(1)?;
        self.g1_extra_ms[0] += self.ev1_hop_start.elapsed_ms(&self.ev1_hop_end)? as f64;
        self.g1_extra_ms[1] += self.ev1_hop_end.elapsed_ms(&self.ev1_pos_end)? as f64;
        for (idx, l) in (split_l..n_layers).enumerate() {
            let lw = &weights.layers[l];
            let le = &self.ev1[idx].ev;
            self.g1_layers_ms[0] += le[0].elapsed_ms(&le[1])? as f64;
            let mix_idx = if lw.is_kda { 1 } else { 2 };
            self.g1_layers_ms[mix_idx] += le[1].elapsed_ms(&le[2])? as f64;
            self.g1_layers_ms[3] += le[2].elapsed_ms(&le[3])? as f64;
            if !lw.is_moe {
                self.g1_layers_ms[4] += le[3].elapsed_ms(&le[4])? as f64;
            } else {
                self.g1_layers_ms[5] += le[3].elapsed_ms(&le[4])? as f64;
                self.g1_layers_ms[6] += le[4].elapsed_ms(&le[5])? as f64;
                self.g1_layers_ms[7] += le[5].elapsed_ms(&le[6])? as f64;
                self.g1_layers_ms[8] += le[6].elapsed_ms(&le[7])? as f64;
                self.g1_layers_ms[9] += le[7].elapsed_ms(&le[8])? as f64;
            }
        }
        self.g1_extra_ms[2] += self.ev1_head_start.elapsed_ms(&self.ev1_norm_end)? as f64;
        self.g1_extra_ms[3] += self.ev1_norm_end.elapsed_ms(&self.ev1_logits_end)? as f64;
        self.g1_extra_ms[4] += self.ev1_logits_end.elapsed_ms(&self.ev1_head_end)? as f64;
        self.g1_extra_ms[5] += self.ev1_head_end.elapsed_ms(&self.ev1_d2h_end)? as f64;
        self.steps += 1;
        Ok(())
    }

    pub fn print_and_reset(&mut self) {
        if self.steps == 0 {
            return;
        }
        let n = self.steps as f64;
        let sum0: f64 = (self.g0_layers_ms.iter().sum::<f64>() + self.g0_extra_ms.iter().sum::<f64>()) / n;
        let sum1: f64 = (self.g1_layers_ms.iter().sum::<f64>() + self.g1_extra_ms.iter().sum::<f64>()) / n;
        eprintln!(
            "AMPLLM_PHASE_BREAKDOWN steps={} dev=0 total={sum0:.2}ms | h2d={:.2} emb_rep={:.2} hc_attn={:.2} kda={:.2} dsa={:.2} hc_ffn={:.2} dense_ffn={:.2} moe_in={:.2} router={:.2} fetch={:.2} moe_plan={:.2} db_wait={:.2} hop_d2h={:.2}",
            self.steps,
            self.g0_extra_ms[0] / n,
            self.g0_extra_ms[1] / n,
            self.g0_layers_ms[0] / n,
            self.g0_layers_ms[1] / n,
            self.g0_layers_ms[2] / n,
            self.g0_layers_ms[3] / n,
            self.g0_layers_ms[4] / n,
            self.g0_layers_ms[5] / n,
            self.g0_layers_ms[6] / n,
            self.g0_layers_ms[7] / n,
            self.g0_layers_ms[8] / n,
            self.g0_layers_ms[9] / n,
            self.g0_extra_ms[2] / n,
        );
        eprintln!(
            "AMPLLM_PHASE_BREAKDOWN steps={} dev=1 total={sum1:.2}ms | hop_h2d={:.2} pos_h2d={:.2} hc_attn={:.2} kda={:.2} dsa={:.2} hc_ffn={:.2} dense_ffn={:.2} moe_in={:.2} router={:.2} fetch={:.2} moe_plan={:.2} db_wait={:.2} head_norm={:.2} head_gemv={:.2} head_argmax={:.2} d2h={:.2}",
            self.steps,
            self.g1_extra_ms[0] / n,
            self.g1_extra_ms[1] / n,
            self.g1_layers_ms[0] / n,
            self.g1_layers_ms[1] / n,
            self.g1_layers_ms[2] / n,
            self.g1_layers_ms[3] / n,
            self.g1_layers_ms[4] / n,
            self.g1_layers_ms[5] / n,
            self.g1_layers_ms[6] / n,
            self.g1_layers_ms[7] / n,
            self.g1_layers_ms[8] / n,
            self.g1_layers_ms[9] / n,
            self.g1_extra_ms[2] / n,
            self.g1_extra_ms[3] / n,
            self.g1_extra_ms[4] / n,
            self.g1_extra_ms[5] / n,
        );
        self.reset();
    }
}

pub struct Glm5State {
    pub cur_pos: usize,
    pub head_x_pos: Option<usize>,
    pub mtp_cached_pos: usize,
    pub p2p_ok: bool,
    pub sc0: GpuScratch,
    pub sc1: GpuScratch,
    pub kda: Vec<Option<KdaLayerState>>,
    pub dsa: Vec<Option<DsaLayerState>>,
    // GPU 0 <-> GPU 1 double-buffered pinned hop buffer (when P2P unavailable or for async staging)
    pub hop_pin: [PinnedMappedBuffer<f32>; 2],
    pub hop_pin_dev0: [u64; 2],
    pub hop_emb_pin0: PinnedMappedBuffer<f32>,
    pub hop_emb_pin1: PinnedMappedBuffer<f32>,
    pub pos_pin0: PinnedMappedBuffer<i32>,
    pub pos_pin1: PinnedMappedBuffer<i32>,
    // Head + MTP scratch on GPU 1
    pub head_x: DeviceBuffer,       // [max_batch, n_embd] F32 (post output_norm)
    pub head_xq: DeviceBuffer,      // [max_batch, q8_bytes(n_embd)]
    pub logits_dev: DeviceBuffer,   // [n_vocab] F32
    pub target_logits_pin: PinnedMappedBuffer<f32>,
    pub mtp_logits_pin: PinnedMappedBuffer<f32>,
    pub argmax_pv: DeviceBuffer,
    pub argmax_pi: DeviceBuffer,
    pub tok_dev: DeviceBuffer,
    pub tok_host: PinnedMappedBuffer<i32>,
    pub tok_host_dev1: u64,
    pub mtp_cat_q8: DeviceBuffer,   // [(2 * n_embd / 32) * 36]
    pub mtp_h: DeviceBuffer,        // [n_embd]
    pub mtp_chain_h: DeviceBuffer,  // [n_embd]
    pub mtp_tok_host: PinnedMappedBuffer<i32>,
    pub mtp_tok_host_dev1: u64,
    // KDA snapshot + trace buffers on GPU 0 & GPU 1 for M <= 4 speculative rollback
    pub kda_trace: Vec<Option<KdaTraceBuffers>>,
    pub snap_kda_s0: Vec<Option<DeviceBuffer>>,
    pub snap_conv_q0: Vec<Option<DeviceBuffer>>,
    pub snap_conv_k0: Vec<Option<DeviceBuffer>>,
    pub snap_conv_v0: Vec<Option<DeviceBuffer>>,
    pub snap_kda_s1: Vec<Option<DeviceBuffer>>,
    pub snap_conv_q1: Vec<Option<DeviceBuffer>>,
    pub snap_conv_k1: Vec<Option<DeviceBuffer>>,
    pub snap_conv_v1: Vec<Option<DeviceBuffer>>,
    // Events for async GPU0 <-> GPU1 pipelining and MTP completion
    pub ev_g0_done: [CudaEvent; 2],
    pub ev_g1_done: CudaEvent,
    pub ev_mtp_done: CudaEvent,
    // Timed profiling events for GPU 0 / GPU 1 / MTP execution attribution
    pub prof_g0_start: [CudaEvent; 2],
    pub prof_g0_end: [CudaEvent; 2],
    pub prof_g1_start: CudaEvent,
    pub prof_g1_end: CudaEvent,
    pub prof_mtp_start: CudaEvent,
    pub prof_mtp_end: CudaEvent,
    // Captured CUDA graphs for steady-state M=1 decode & MTP drafting
    pub decode_graph0: Option<CudaGraphExec>,
    pub decode_graph1_greedy: Option<CudaGraphExec>,
    pub decode_graph1_logits: Option<CudaGraphExec>,
    pub mtp_graph_dv: Option<CudaGraphExec>,
    pub mtp_graph_dv_logits: Option<CudaGraphExec>,
    pub last_mtp_used_dv: bool,
    pub mtp_graph_full_greedy: Option<CudaGraphExec>,
    pub mtp_graph_full_logits: Option<CudaGraphExec>,
    pub snap_kda0_graph: Option<CudaGraphExec>,
    pub restore_kda0_graph: Option<CudaGraphExec>,
    // Online dynamic VRAM expert cache & async H2D mover
    pub expert_cache: DynamicExpertCache,
    // Persistent CPU scratch for chunked prefill MoE (independent per GPU worker)
    pub max_prefill_chunk: usize,
    pub max_lend_chunk: usize,
    pub prefill_cpu_scratch0: Glm5CpuScratch,
    pub prefill_cpu_scratch1: Glm5CpuScratch,
    // Dual-lane router / spare-slot configuration
    pub cpu_plan: u64,
    pub promote_min: i32,
    pub mtp_keep: i32,
    pub max_pf: i32,
    pub pf_at_fetch: bool,
    pub batch_pci: i32,
    pub async_promo: bool,
    // Optional per-phase CUDA-event profiler (AMPLLM_PHASE_PROFILE=1)
    pub phase_prof: Option<PhaseProfiler>,
    // Doorbell worker (must drop AFTER GPU finishes, BEFORE arena!)
    pub doorbell: DoorbellWorker,
}

impl Glm5State {
    pub fn alloc(weights: &Glm5ModelWeights, cpu_pool: Arc<CpuThreadPool>, max_ctx: usize) -> Result<Self> {
        let cfg = &weights.cfg;
        let max_batch = 128usize;
        let max_prefill_chunk = std::env::var("AMPLLM_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(512)
            .clamp(max_batch, 1024);
        let max_lend_chunk = std::env::var("AMPLLM_PREFILL_MAX_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8192)
            .clamp(max_prefill_chunk, 8192);
        let p2p_01 = enable_peer_access(0, 1).unwrap_or(false);
        let p2p_10 = enable_peer_access(1, 0).unwrap_or(false);
        let p2p_ok = p2p_01 && p2p_10;
        let max_blob_stride = weights
            .layers
            .iter()
            .map(|lw| lw.blob_stride)
            .max()
            .unwrap_or(0);
        let mut cpu_plan = std::env::var("AMPLLM_CPU_PLAN")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x654432210u64)
            & !(1u64 << 63);
        if std::env::var("AMPLLM_DIRECT_PCIE").is_ok_and(|v| v == "1") {
            cpu_plan |= 1u64 << 63;
        }
        let promote_min = std::env::var("AMPLLM_PROMOTE_MIN")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(1);
        let mtp_keep = std::env::var("AMPLLM_MTP_KEEP")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(2)
            .clamp(0, 8);
        let max_pf = std::env::var("AMPLLM_MAX_PF")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0)
            .clamp(0, 2);
        let pf_at_fetch = std::env::var("AMPLLM_PF_AT")
            .map(|s| s.eq_ignore_ascii_case("fetch"))
            .unwrap_or(false);
        let batch_pci = std::env::var("AMPLLM_BATCH_PCI")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(2)
            .clamp(0, 6);
        let async_promo = std::env::var("AMPLLM_ASYNC_PROMO").is_ok_and(|v| v == "1");

        let mmq_ok_w = |w: &crate::glm5::weights::DevWeight| -> bool {
            matches!(
                w.ggml_type,
                GgmlType::Iq2S
                    | GgmlType::Iq3Xxs
                    | GgmlType::Iq3S
                    | GgmlType::Q6K
                    | GgmlType::Q8_0
                    | GgmlType::Q4K
            ) && w.dims[0].is_multiple_of(32)
        };
        let max_f16_elems_for_dev = |dev: i32| -> usize {
            let mut max_e = 0usize;
            for lw in weights.layers.iter().filter(|lw| lw.device == dev) {
                let projs = [
                    lw.attn_q.as_ref(),
                    lw.attn_k.as_ref(),
                    lw.attn_v.as_ref(),
                    lw.ssm_beta.as_ref(),
                    lw.ssm_f_a.as_ref(),
                    lw.ssm_f_b.as_ref(),
                    lw.ssm_g_a.as_ref(),
                    lw.ssm_g_b.as_ref(),
                    lw.attn_q_a.as_ref(),
                    lw.attn_q_b.as_ref(),
                    lw.attn_kv_a_mqa.as_ref(),
                    lw.idx_k.as_ref(),
                    lw.idx_q_b.as_ref(),
                    lw.idx_proj.as_ref(),
                    lw.idx_gate.as_ref(),
                    Some(&lw.attn_output),
                    lw.ffn_gate.as_ref(),
                    lw.ffn_up.as_ref(),
                    lw.ffn_down.as_ref(),
                    lw.ffn_gate_inp.as_ref(),
                    lw.ffn_gate_sh.as_ref(),
                    lw.ffn_up_sh.as_ref(),
                    lw.ffn_down_sh.as_ref(),
                ];
                for w in projs.into_iter().flatten() {
                    if !mmq_ok_w(w) {
                        max_e = max_e.max(w.dims[0] * w.dims[1]);
                    }
                }
                if let Some(ref eh) = lw.nextn_eh_proj {
                    // mtp_cache_chunk_split dequants nextn_eh_proj into mma_b_f16 unconditionally
                    max_e = max_e.max(eh.dims[0] * eh.dims[1]);
                }
            }
            max_e
        };
        let max_f16_0 = max_f16_elems_for_dev(0);
        let max_f16_1 = max_f16_elems_for_dev(1);

        let (f0_s, _) = ampllm_cuda_kernels::mem_info(0)?;
        let (f1_s, _) = ampllm_cuda_kernels::mem_info(1)?;
        tracing::info!("Glm5State::alloc entry: GPU0 free={} MiB, GPU1 free={} MiB", f0_s >> 20, f1_s >> 20);
        let sc0 = GpuScratch::alloc(0, cfg, weights.pool_cap, max_batch, max_prefill_chunk, max_lend_chunk, max_blob_stride, max_f16_0)
            .with_context(|| format!("GpuScratch::alloc(0, max_batch={max_batch}) free0={} MiB", f0_s >> 20))?;
        let sc1 = GpuScratch::alloc(1, cfg, weights.pool_cap, max_batch, max_prefill_chunk, max_lend_chunk, max_blob_stride, max_f16_1)
            .with_context(|| format!("GpuScratch::alloc(1, max_batch={max_batch}) free1={} MiB", f1_s >> 20))?;
        let (f0_a, _) = ampllm_cuda_kernels::mem_info(0)?;
        let (f1_a, _) = ampllm_cuda_kernels::mem_info(1)?;
        tracing::info!("Glm5State::alloc post-scratch: GPU0 free={} MiB, GPU1 free={} MiB", f0_a >> 20, f1_a >> 20);

        let mut kda = Vec::with_capacity(cfg.n_layer_total);
        let mut dsa = Vec::with_capacity(cfg.n_layer_total);
        let mut kda_trace = Vec::with_capacity(cfg.n_layer_total);
        let mut snap_kda_s0 = Vec::with_capacity(weights.split_layer);
        let mut snap_conv_q0 = Vec::with_capacity(weights.split_layer);
        let mut snap_conv_k0 = Vec::with_capacity(weights.split_layer);
        let mut snap_conv_v0 = Vec::with_capacity(weights.split_layer);
        let mut snap_kda_s1 = Vec::with_capacity(cfg.n_layer_total);
        let mut snap_conv_q1 = Vec::with_capacity(cfg.n_layer_total);
        let mut snap_conv_k1 = Vec::with_capacity(cfg.n_layer_total);
        let mut snap_conv_v1 = Vec::with_capacity(cfg.n_layer_total);

        let lat_stride = cfg.kv_lora_rank + cfg.kv_lora_rank / 16;
        let use_ram_lat = max_ctx > weights.dsa_res_cap;
        let verify_m = std::env::var("AMPLLM_VERIFY_M")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 4);
        let max_verify_m = if verify_m >= 2 { 4usize } else { 1usize };

        for l in 0..cfg.n_layer_total {
            let dev = weights.layers[l].device;
            set_device(dev)?;
            if weights.layers[l].is_kda {
                let s_bytes = cfg.n_head * cfg.kda_head_dim * cfg.kda_head_dim * 4;
                let c_bytes = cfg.d_inner() * (cfg.d_conv - 1) * 4;
                let di_bytes = max_verify_m * cfg.d_inner() * 4;
                let bt_bytes = max_verify_m * cfg.n_head * 4;
                kda.push(Some(KdaLayerState {
                    s_mat: DeviceBuffer::alloc_zeroed(dev, s_bytes)?,
                    conv_q: DeviceBuffer::alloc_zeroed(dev, c_bytes)?,
                    conv_k: DeviceBuffer::alloc_zeroed(dev, c_bytes)?,
                    conv_v: DeviceBuffer::alloc_zeroed(dev, c_bytes)?,
                }));
                if max_verify_m >= 2 {
                    kda_trace.push(Some(KdaTraceBuffers {
                        q_p: DeviceBuffer::alloc(dev, di_bytes)?,
                        k_p: DeviceBuffer::alloc(dev, di_bytes)?,
                        v_p: DeviceBuffer::alloc(dev, di_bytes)?,
                        g1: DeviceBuffer::alloc(dev, di_bytes)?,
                        beta: DeviceBuffer::alloc(dev, bt_bytes)?,
                    }));
                } else {
                    kda_trace.push(None);
                }
                dsa.push(None);
                if dev == 0 {
                    snap_kda_s0.push(Some(DeviceBuffer::alloc_zeroed(0, s_bytes)?));
                    snap_conv_q0.push(Some(DeviceBuffer::alloc_zeroed(0, c_bytes)?));
                    snap_conv_k0.push(Some(DeviceBuffer::alloc_zeroed(0, c_bytes)?));
                    snap_conv_v0.push(Some(DeviceBuffer::alloc_zeroed(0, c_bytes)?));
                    snap_kda_s1.push(None);
                    snap_conv_q1.push(None);
                    snap_conv_k1.push(None);
                    snap_conv_v1.push(None);
                } else if max_verify_m >= 2 {
                    snap_kda_s1.push(Some(DeviceBuffer::alloc_zeroed(1, s_bytes)?));
                    snap_conv_q1.push(Some(DeviceBuffer::alloc_zeroed(1, c_bytes)?));
                    snap_conv_k1.push(Some(DeviceBuffer::alloc_zeroed(1, c_bytes)?));
                    snap_conv_v1.push(Some(DeviceBuffer::alloc_zeroed(1, c_bytes)?));
                } else {
                    snap_kda_s1.push(None);
                    snap_conv_q1.push(None);
                    snap_conv_k1.push(None);
                    snap_conv_v1.push(None);
                }
            } else {
                kda.push(None);
                kda_trace.push(None);
                let ram_lat8 = if use_ram_lat {
                    Some(PinnedMappedBuffer::<u8>::alloc(max_ctx * lat_stride)?)
                } else {
                    None
                };
                dsa.push(Some(DsaLayerState {
                    vram_lat8: DeviceBuffer::alloc_zeroed(dev, weights.dsa_res_cap * lat_stride)?,
                    ram_lat8,
                    ik_ring: DeviceBuffer::alloc_zeroed(dev, 516 * cfg.idx_key_dim * 4)?,
                    ig_ring: DeviceBuffer::alloc_zeroed(dev, 516 * cfg.idx_key_dim * 4)?,
                    pooled_f16: DeviceBuffer::alloc_zeroed(dev, weights.pool_cap * cfg.idx_key_dim * 2)?,
                }));
                if dev == 0 {
                    snap_kda_s0.push(None);
                    snap_conv_q0.push(None);
                    snap_conv_k0.push(None);
                    snap_conv_v0.push(None);
                }
                snap_kda_s1.push(None);
                snap_conv_q1.push(None);
                snap_conv_k1.push(None);
                snap_conv_v1.push(None);
            }
        }

        set_device(0)?;
        unsafe { ampllm_cuda_kernels::glm5::prepare_kda_smem(cfg.kda_head_dim as i32); }
        set_device(1)?;
        unsafe { ampllm_cuda_kernels::glm5::prepare_kda_smem(cfg.kda_head_dim as i32); }
        let q8 = |e: usize| (e / 32) * 36;
        let head_x = DeviceBuffer::alloc(1, max_prefill_chunk * cfg.n_embd * 4)?;
        let head_xq = DeviceBuffer::alloc(1, max_batch * q8(cfg.n_embd))?;
        let logits_dev = DeviceBuffer::alloc(1, max_verify_m * cfg.n_vocab * 4)?;
        let target_logits_pin = PinnedMappedBuffer::<f32>::alloc(max_verify_m * cfg.n_vocab)?;
        let mtp_logits_pin = PinnedMappedBuffer::<f32>::alloc(cfg.n_vocab)?;
        let argmax_pv = DeviceBuffer::alloc(1, 128 * 4)?;
        let argmax_pi = DeviceBuffer::alloc(1, 128 * 4)?;
        let tok_dev = DeviceBuffer::alloc(1, max_verify_m * 4)?;
        let tok_host = PinnedMappedBuffer::<i32>::alloc(max_verify_m)?;
        let tok_host_dev1 = tok_host.dev_ptr_for_device(1)? as u64;
        let mtp_cat_q8 = DeviceBuffer::alloc(1, q8(2 * cfg.n_embd))?;
        let mtp_h = DeviceBuffer::alloc(1, cfg.n_embd * 4)?;
        let mtp_chain_h = DeviceBuffer::alloc(1, cfg.n_embd * 4)?;
        let mtp_tok_host = PinnedMappedBuffer::<i32>::alloc(1)?;
        let mtp_tok_host_dev1 = mtp_tok_host.dev_ptr_for_device(1)? as u64;

        let hop_pin = [
            PinnedMappedBuffer::<f32>::alloc(max_prefill_chunk * cfg.hc * cfg.n_embd)?,
            PinnedMappedBuffer::<f32>::alloc(max_prefill_chunk * cfg.hc * cfg.n_embd)?,
        ];
        let hop_pin_dev0 = [
            hop_pin[0].dev_ptr_for_device(0)? as u64,
            hop_pin[1].dev_ptr_for_device(0)? as u64,
        ];
        let hop_emb_pin0 = PinnedMappedBuffer::<f32>::alloc(max_prefill_chunk * cfg.n_embd)?;
        let hop_emb_pin1 = PinnedMappedBuffer::<f32>::alloc(max_prefill_chunk * cfg.n_embd)?;
        let pos0_cap = max_verify_m.max(max_lend_chunk).max(2);
        let pos1_cap = max_verify_m.max(max_lend_chunk).max(4);
        let pos_pin0 = PinnedMappedBuffer::<i32>::alloc(pos0_cap)?;
        let pos_pin1 = PinnedMappedBuffer::<i32>::alloc(pos1_cap)?;

        let ev_g0_done = [CudaEvent::new(0)?, CudaEvent::new(0)?];
        let ev_g1_done = CudaEvent::new(1)?;
        let ev_mtp_done = CudaEvent::new(1)?;
        let prof_g0_start = [CudaEvent::new_timed(0)?, CudaEvent::new_timed(0)?];
        let prof_g0_end = [CudaEvent::new_timed(0)?, CudaEvent::new_timed(0)?];
        let prof_g1_start = CudaEvent::new_timed(1)?;
        let prof_g1_end = CudaEvent::new_timed(1)?;
        let prof_mtp_start = CudaEvent::new_timed(1)?;
        let prof_mtp_end = CudaEvent::new_timed(1)?;

        let expert_cache = DynamicExpertCache::from_weights(weights)?;
        let doorbell = DoorbellWorker::spawn(cpu_pool, &weights.cold_arena)?;
        let prefill_cpu_scratch0 = Glm5CpuScratch::new(cfg.n_embd, cfg.n_ff_exp, 1024);
        let prefill_cpu_scratch1 = Glm5CpuScratch::new(cfg.n_embd, cfg.n_ff_exp, 1024);
        let phase_prof = if std::env::var_os("AMPLLM_PHASE_PROFILE").is_some() {
            Some(PhaseProfiler::alloc(weights.split_layer, cfg.n_layers)?)
        } else {
            None
        };

        Ok(Self {
            cur_pos: 0,
            head_x_pos: None,
            mtp_cached_pos: 0,
            p2p_ok,
            sc0,
            sc1,
            kda,
            dsa,
            hop_pin,
            hop_pin_dev0,
            hop_emb_pin0,
            hop_emb_pin1,
            pos_pin0,
            pos_pin1,
            head_x,
            head_xq,
            logits_dev,
            target_logits_pin,
            mtp_logits_pin,
            argmax_pv,
            argmax_pi,
            tok_dev,
            tok_host,
            tok_host_dev1,
            mtp_cat_q8,
            mtp_h,
            mtp_chain_h,
            mtp_tok_host,
            mtp_tok_host_dev1,
            kda_trace,
            snap_kda_s0,
            snap_conv_q0,
            snap_conv_k0,
            snap_conv_v0,
            snap_kda_s1,
            snap_conv_q1,
            snap_conv_k1,
            snap_conv_v1,
            ev_g0_done,
            ev_g1_done,
            ev_mtp_done,
            prof_g0_start,
            prof_g0_end,
            prof_g1_start,
            prof_g1_end,
            prof_mtp_start,
            prof_mtp_end,
            decode_graph0: None,
            decode_graph1_greedy: None,
            decode_graph1_logits: None,
            mtp_graph_dv: None,
            mtp_graph_dv_logits: None,
            last_mtp_used_dv: false,
            mtp_graph_full_greedy: None,
            mtp_graph_full_logits: None,
            snap_kda0_graph: None,
            restore_kda0_graph: None,
            expert_cache,
            max_prefill_chunk,
            max_lend_chunk,
            prefill_cpu_scratch0,
            prefill_cpu_scratch1,
            cpu_plan,
            promote_min,
            mtp_keep,
            max_pf,
            pf_at_fetch,
            batch_pci,
            async_promo,
            phase_prof,
            doorbell,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.cur_pos = 0;
        self.head_x_pos = None;
        self.mtp_cached_pos = 0;
        self.last_mtp_used_dv = false;
        {
            let db0 = &mut self.doorbell.db0.as_mut_slice()[0];
            db0.cancel_flag = 0;
            db0.routed_ids = [[-1; 32]; 48];
            db0.promo_eid = [[-1; 8]; 48];
            db0.promo_vram_ptr = [[0; 8]; 48];
        }
        {
            let db1 = &mut self.doorbell.db1.as_mut_slice()[0];
            db1.cancel_flag = 0;
            db1.routed_ids = [[-1; 32]; 48];
            db1.promo_eid = [[-1; 8]; 48];
            db1.promo_vram_ptr = [[0; 8]; 48];
        }
        set_device(0)?;
        self.sc0.skip_flag_dev.zero()?;
        set_device(1)?;
        self.sc1.skip_flag_dev.zero()?;
        for (l, k_opt) in self.kda.iter().enumerate() {
            if let Some(k) = k_opt {
                set_device(k.s_mat.device)?;
                k.s_mat.zero()?;
                k.conv_q.zero()?;
                k.conv_k.zero()?;
                k.conv_v.zero()?;
            }
            if let Some(Some(s)) = self.snap_kda_s0.get(l) {
                set_device(0)?;
                s.zero()?;
                self.snap_conv_q0[l].as_ref().unwrap().zero()?;
                self.snap_conv_k0[l].as_ref().unwrap().zero()?;
                self.snap_conv_v0[l].as_ref().unwrap().zero()?;
            }
            if let Some(Some(s)) = self.snap_kda_s1.get(l) {
                set_device(1)?;
                s.zero()?;
                self.snap_conv_q1[l].as_ref().unwrap().zero()?;
                self.snap_conv_k1[l].as_ref().unwrap().zero()?;
                self.snap_conv_v1[l].as_ref().unwrap().zero()?;
            }
        }
        for d_opt in &self.dsa {
            if let Some(d) = d_opt {
                set_device(d.ik_ring.device)?;
                d.ik_ring.zero()?;
                d.ig_ring.zero()?;
            }
        }
        Ok(())
    }
}
