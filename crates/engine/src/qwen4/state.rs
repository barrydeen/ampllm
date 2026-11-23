use crate::config::EngineConfig;
use crate::qwen4::ple::PleCursor;
use crate::qwen4::weights::Qwen4Weights;
use ampllm_cache::CacheStats;
use ampllm_cpu_kernels::glm5_cpu::{
    run_glm5_cpu_moe_grouped_small, run_glm5_cpu_moe_m1, Glm5CpuExpertJob, Glm5CpuGroupedExpert,
    Glm5CpuScratch,
};
use ampllm_cpu_kernels::pool::discover_ccd_split_cores;
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::qwen4::Qwen4Doorbell;
use ampllm_cuda_kernels::{
    qwen4, set_device, CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, PinnedHostBuffer,
    PinnedMappedBuffer,
};
use ampllm_gguf::{GgmlType, Qwen4Config};
use anyhow::Result;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

pub const QWEN4_DEFAULT_CHUNK: usize = 4096;
pub const QWEN4_SUB: usize = 1024;
pub const QWEN4_MAX_VERIFY: usize = 8;
pub const QWEN4_DMA_RING: usize = 3;
pub const QWEN4_DMA_BATCH: usize = 8;
pub const QWEN4_DMA_SLOTS: usize = QWEN4_DMA_RING * QWEN4_DMA_BATCH;
pub const QWEN4_PREFILL_GRP_WIN: usize = 4096;
pub const QWEN4_META_SLOTS: usize = 4;

#[inline]
const fn align256(n: usize) -> usize {
    (n + 255) & !255
}

#[inline]
pub fn next_cap_bucket(pos: usize, sparse: bool, max_ctx: usize) -> usize {
    if !sparse {
        pos.max(256).next_power_of_two().min(max_ctx)
    } else {
        let blocks = (pos / 4).saturating_add(1);
        let bucket_blocks = ((blocks + 255) / 256) * 256;
        (bucket_blocks * 4).max(1024).min(max_ctx + 4)
    }
}

pub struct GdnLayerState {
    pub state: DeviceBuffer,
    pub conv_hist: DeviceBuffer,
    pub state_snap: Option<DeviceBuffer>,
    pub hist_snap: Option<DeviceBuffer>,
    pub h_qkv_snap: Option<DeviceBuffer>,
    pub gate_snap: Option<DeviceBuffer>,
    pub beta_snap: Option<DeviceBuffer>,
}

pub struct QsaLayerState {
    pub k_q: DeviceBuffer,
    pub v_q: DeviceBuffer,
    pub k_sc: DeviceBuffer,
    pub v_sc: DeviceBuffer,
    pub tail: DeviceBuffer,
    pub dead: DeviceBuffer,
    pub pooled: DeviceBuffer,
    pub tail_snap: Option<DeviceBuffer>,
    pub dead_snap: Option<DeviceBuffer>,
    pub k_idx_snap: Option<DeviceBuffer>,
}

pub enum Qwen4LayerState {
    Gdn(GdnLayerState),
    Qsa(QsaLayerState),
}

#[derive(Clone, Copy)]
pub struct MetaSlotOffsets {
    pub exp_ptrs: usize,
    pub rows: usize,
    pub wts: usize,
    pub exp_row_off: usize,
    pub exp_nt: usize,
    pub tile_exp_id: usize,
    pub tile_m_off: usize,
}

#[derive(Clone, Copy)]
pub struct MetaPinRel {
    pub slot_bytes: usize,
    pub exp_ptrs: usize,
    pub rows: usize,
    pub wts: usize,
    pub exp_row_off: usize,
    pub exp_nt: usize,
    pub tile_exp_id: usize,
    pub tile_m_off: usize,
}

pub struct GpuStageScratch {
    pub device: i32,
    pub stream: CudaStream,
    pub copy_stream: CudaStream,
    pub fetch_stream: CudaStream,
    pub side_stream: CudaStream,
    pub d2h_stream: CudaStream,
    pub dma_events: [CudaEvent; QWEN4_DMA_RING],
    pub compute_events: [CudaEvent; QWEN4_DMA_RING],
    pub meta_events: [CudaEvent; QWEN4_META_SLOTS],
    pub d2h_done: CudaEvent,
    pub p2p_done: CudaEvent,
    pub fetch_start_ev: CudaEvent,
    pub fetch_done_ev: CudaEvent,
    pub side_fork_ev: CudaEvent,
    pub side_join_ev: CudaEvent,
    pub adapt_d2h_ev: CudaEvent,
    pub adapt_h2d_ev: CudaEvent,

    pub sub_max: usize,
    pub qsa_ws_floats: usize,
    pub max_f16_elems_w: usize,
    pub slot_stride: usize,
    pub arena: DeviceBuffer,
    pub dma_stage: DeviceBuffer,
    pub off_dma_ptrs_dev: usize,

    // Persistent chunk offsets in `arena`
    pub off_r_buf: usize,
    pub off_mixed: usize,
    pub off_mixed_q8: usize,
    pub off_inject: usize,
    pub off_router_logits: usize,
    pub off_router_ids: usize,
    pub off_router_w: usize,
    pub off_block_out: usize,
    pub off_pos_dev: usize,
    pub off_mrope_pos3_dev: usize,
    pub off_widths_dev: usize,
    pub off_plan_ptrs_dev: usize,
    pub off_db_seq_state: usize,
    pub off_db_exp_seq: usize,
    pub off_argmax_vals: usize,
    pub off_argmax_ids: usize,
    pub off_argmax_out: usize,
    pub off_logits: usize,
    pub off_sh_out: usize,
    pub off_sh_gate_logit: usize,
    pub off_small_ptrs: usize,
    pub off_small_off: usize,
    pub off_small_nt: usize,
    pub off_small_rows: usize,
    pub off_small_pair: usize,
    pub off_small_wts: usize,
    pub off_fetch_src_ptrs: usize,
    pub off_fetch_dst_ptrs: usize,
    pub off_fetch_off: usize,
    pub off_fetch_nt: usize,
    pub off_fetch_rows: usize,
    pub off_fetch_pair: usize,
    pub off_fetch_wts: usize,
    pub off_gpu_pair_mask: usize,
    pub off_moe_pair_out: usize,
    pub meta_dev: [MetaSlotOffsets; QWEN4_META_SLOTS],

    // Phase A1 (HC read) - aliased at `phase_base`
    pub off_xn: usize,
    pub off_xn_q8: usize,
    pub off_hc_lo: usize,
    pub off_hc_lo_q8: usize,
    pub off_hc_up: usize,

    // Phase A2 (F16 TC + GDN/QSA/Shared) - aliased at `phase_base`
    pub off_f16_x: usize,
    pub off_f16_w: usize,
    pub off_gdn_qkv: usize,
    pub off_gdn_z_gate: usize,
    pub off_gdn_h_qkv: usize,
    pub off_gdn_gate: usize,
    pub off_gdn_beta: usize,
    pub off_gdn_y: usize,
    pub off_gdn_y_q8: usize,
    pub off_gdn_tc_scratch: usize,

    pub off_qfull: usize,
    pub off_q: usize,
    pub off_k: usize,
    pub off_v: usize,
    pub off_q_idx: usize,
    pub off_k_idx: usize,
    pub off_idx_scores: usize,
    pub off_idx_ids: usize,
    pub off_qsa_ws: usize,
    pub off_attn_out: usize,
    pub off_attn_q8: usize,

    pub off_sh_gate: usize,
    pub off_sh_up: usize,
    pub off_sh_hq: usize,

    // Phase B (Routed MoE) - aliased at `phase_base`
    pub off_moe_grp_gu: usize,
    pub off_moe_inter_q8: usize,

    // Pinned host staging
    pub h_pos: PinnedHostBuffer,
    pub h_ids: PinnedHostBuffer,
    pub h_w: PinnedHostBuffer,
    pub h_mixed: PinnedHostBuffer,
    pub h_cpu: PinnedHostBuffer,
    pub h_small_ptrs: PinnedHostBuffer,
    pub h_small_off: PinnedHostBuffer,
    pub h_small_nt: PinnedHostBuffer,
    pub h_small_rows: PinnedHostBuffer,
    pub h_small_pair: PinnedHostBuffer,
    pub h_small_wts: PinnedHostBuffer,
    pub meta_pin: PinnedHostBuffer,
    pub meta_pin_rel: MetaPinRel,
    pub small_pin: PinnedHostBuffer,
    pub warmup_bounce: PinnedHostBuffer,
    pub adapt_patch_pin: PinnedMappedBuffer<u64>,
    pub adapt_patch_dev: u64,
    pub adapt_patch_cursor: usize,
    pub decode_pos_pin: PinnedMappedBuffer<i32>,
    pub decode_pos_dev: u64,
    pub argmax_out_pin: PinnedMappedBuffer<i32>,
    pub argmax_out_dev: u64,
}

impl GpuStageScratch {
    pub fn alloc(
        dev: i32,
        cfg: &Qwen4Config,
        max_ctx: usize,
        max_t: usize,
        max_verify: usize,
        max_idx_cap: usize,
        max_blob_bytes: usize,
        is_head_gpu: bool,
    ) -> Result<Self> {
        set_device(dev)?;
        let stream = CudaStream::new(dev)?;
        let copy_stream = CudaStream::new(dev)?;
        let fetch_stream = CudaStream::new(dev)?;
        let side_stream = CudaStream::new(dev)?;
        let d2h_stream = CudaStream::new(dev)?;
        let dma_events = [
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
        ];
        let compute_events = [
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
        ];
        let meta_events = [
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
        ];
        let d2h_done = CudaEvent::new(dev)?;
        let p2p_done = CudaEvent::new(dev)?;
        let fetch_start_ev = CudaEvent::new(dev)?;
        let fetch_done_ev = CudaEvent::new(dev)?;
        let side_fork_ev = CudaEvent::new(dev)?;
        let side_join_ev = CudaEvent::new(dev)?;
        let adapt_d2h_ev = CudaEvent::new(dev)?;
        let adapt_h2d_ev = CudaEvent::new(dev)?;
        let slot_stride = align256(max_blob_bytes.max(4096));

        let hc = cfg.hc;
        let n_embd = cfg.n_embd;
        let hc_dim = hc * n_embd;
        let hc_lr = cfg.hc_low_rank;
        let q8_row_x = (n_embd / 32) * 36;
        let q8_row_hc = (hc_dim / 32) * 36;
        let q8_row_lr = (hc_lr / 32) * 36;

        let sub = QWEN4_SUB.min(max_t);
        let grp_win = QWEN4_PREFILL_GRP_WIN;
        let max_tiles = grp_win.div_ceil(32) + cfg.n_expert;

        let gdn_qkv_dim = cfg.gdn_qkv_dim();
        let gdn_z_dim = cfg.ssm_inner_size;
        let gdn_y_q8_row = (gdn_z_dim / 32) * 36;
        let gdn_tc_bytes = qwen4::gdn_tc_scratch_bytes(
            cfg.ssm_group_count as i32,
            cfg.ssm_time_step_rank as i32,
        );

        let qfull_dim = cfg.n_head * cfg.head_dim * 2;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let q_idx_dim = cfg.idx_heads * cfg.idx_key_dim;
        let k_idx_dim = cfg.idx_key_dim;
        let max_blocks = (max_ctx / 4) + 1;
        let qsa_ws_floats = qwen4::qsa_decode_ws_floats(
            max_verify as i32,
            cfg.n_head_kv as i32,
            max_idx_cap as i32,
        );
        let attn_q8_row = (q_dim / 32) * 36;

        let max_pairs = max_t * cfg.n_expert_used;
        let ff_q8_row = (cfg.n_ff_exp / 32) * 36;
        let sh_q8_row = (cfg.n_ff_shared / 32) * 36;
        let max_f16_n_out = cfg
            .n_expert
            .max(q_idx_dim)
            .max(k_idx_dim)
            .max(cfg.ssm_time_step_rank);
        let max_f16_elems_w = max_f16_n_out * n_embd;

        let mut cursor = 0usize;
        let mut alloc_off = |bytes: usize| -> usize {
            let off = cursor;
            cursor += align256(bytes.max(16));
            off
        };

        // Persistent buffers
        let off_r_buf = alloc_off(max_t * hc_dim * 4);
        let off_mixed = alloc_off(max_t * n_embd * 4);
        let off_mixed_q8 = alloc_off(max_t * q8_row_x);
        let off_inject = alloc_off(max_t * hc * hc * 4);
        let off_router_logits = alloc_off(max_t * cfg.n_expert * 4);
        let off_router_ids = alloc_off(max_pairs * 4);
        let off_router_w = alloc_off(max_pairs * 4);
        let off_block_out = alloc_off(max_t * n_embd * 4);
        let off_pos_dev = alloc_off(max_t * 4);
        let off_mrope_pos3_dev = alloc_off(max_ctx * 3 * 4);
        let off_widths_dev = alloc_off(max_t * 4);
        let off_plan_ptrs_dev = alloc_off(max_verify * cfg.n_expert_used * 8);
        let off_db_seq_state = alloc_off(16);
        let off_db_exp_seq = alloc_off(16);
        let off_argmax_vals = alloc_off(512 * 4);
        let off_argmax_ids = alloc_off(512 * 4);
        let off_argmax_out = alloc_off(max_verify * 4);
        let off_logits = if is_head_gpu {
            alloc_off(max_verify * cfg.n_vocab * 4)
        } else {
            0
        };
        let off_sh_out = alloc_off(sub * n_embd * 4);
        let off_sh_gate_logit = alloc_off(sub * 4);

        let off_small_ptrs = alloc_off(cfg.n_expert * 8);
        let off_small_off = alloc_off(cfg.n_expert * 4);
        let off_small_nt = alloc_off(cfg.n_expert * 4);
        let off_small_rows = alloc_off(max_pairs * 4);
        let off_small_pair = alloc_off(64usize.max(max_verify) * cfg.n_expert_used * 4);
        let off_small_wts = alloc_off(max_pairs * 4);
        let off_fetch_src_ptrs = alloc_off(QWEN4_DMA_BATCH * 8);
        let off_fetch_dst_ptrs = alloc_off(QWEN4_DMA_BATCH * 8);
        let off_fetch_off = alloc_off(QWEN4_DMA_BATCH * 4);
        let off_fetch_nt = alloc_off(QWEN4_DMA_BATCH * 4);
        let off_fetch_rows = alloc_off(max_pairs * 4);
        let off_fetch_pair = alloc_off(max_verify * cfg.n_expert_used * 4);
        let off_fetch_wts = alloc_off(max_pairs * 4);
        let off_gpu_pair_mask = alloc_off(max_verify * 2);
        let off_moe_pair_out = alloc_off(max_verify * cfg.n_expert_used * n_embd * 4);
        let off_dma_ptrs_dev = alloc_off(QWEN4_DMA_SLOTS * 8);

        let mut meta_dev = [MetaSlotOffsets {
            exp_ptrs: 0,
            rows: 0,
            wts: 0,
            exp_row_off: 0,
            exp_nt: 0,
            tile_exp_id: 0,
            tile_m_off: 0,
        }; QWEN4_META_SLOTS];
        for slot in &mut meta_dev {
            slot.exp_ptrs = alloc_off(cfg.n_expert * 8);
            slot.rows = alloc_off(grp_win * 4);
            slot.wts = alloc_off(grp_win * 4);
            slot.exp_row_off = alloc_off(cfg.n_expert * 4);
            slot.exp_nt = alloc_off(cfg.n_expert * 4);
            slot.tile_exp_id = alloc_off(max_tiles * 4);
            slot.tile_m_off = alloc_off(max_tiles * 4);
        }

        let phase_base = cursor;
        let mut max_end = phase_base;
        let bump = |cur: &mut usize, max_e: &mut usize, bytes: usize| -> usize {
            let off = *cur;
            *cur += align256(bytes.max(16));
            if *cur > *max_e {
                *max_e = *cur;
            }
            off
        };

        // Phase A1: HC read
        let mut p_a1 = phase_base;
        let off_xn = bump(&mut p_a1, &mut max_end, sub * hc_dim * 4);
        let off_xn_q8 = bump(&mut p_a1, &mut max_end, sub * q8_row_hc);
        let off_hc_lo = bump(&mut p_a1, &mut max_end, sub * hc_lr * 4);
        let off_hc_lo_q8 = bump(&mut p_a1, &mut max_end, sub * q8_row_lr);
        let off_hc_up = bump(&mut p_a1, &mut max_end, sub * hc_dim * 4);

        // Phase A2 base: F16 TC scratch shared across GDN/QSA/Router
        let mut p_a2 = phase_base;
        let off_f16_x = bump(&mut p_a2, &mut max_end, sub * n_embd * 2);
        let off_f16_w = bump(&mut p_a2, &mut max_end, max_f16_elems_w * 2);
        let a2_after_f16 = p_a2;

        // GDN branch
        let mut p_gdn = a2_after_f16;
        let off_gdn_qkv = bump(&mut p_gdn, &mut max_end, sub * gdn_qkv_dim * 4);
        let off_gdn_z_gate = bump(&mut p_gdn, &mut max_end, sub * gdn_z_dim * 4);
        let off_gdn_h_qkv = bump(&mut p_gdn, &mut max_end, sub * gdn_qkv_dim * 4);
        let off_gdn_gate = bump(&mut p_gdn, &mut max_end, sub * cfg.ssm_time_step_rank * 4);
        let off_gdn_beta = bump(&mut p_gdn, &mut max_end, sub * cfg.ssm_time_step_rank * 4);
        let off_gdn_y = bump(&mut p_gdn, &mut max_end, sub * gdn_z_dim * 4);
        let off_gdn_y_q8 = bump(&mut p_gdn, &mut max_end, sub * gdn_y_q8_row);
        let off_gdn_tc_scratch = bump(&mut p_gdn, &mut max_end, gdn_tc_bytes);

        // QSA branch
        let mut p_qsa = a2_after_f16;
        let off_qfull = bump(&mut p_qsa, &mut max_end, sub * qfull_dim * 4);
        let off_q = bump(&mut p_qsa, &mut max_end, sub * q_dim * 4);
        let off_k = bump(&mut p_qsa, &mut max_end, sub * kv_dim * 4);
        let off_v = bump(&mut p_qsa, &mut max_end, sub * kv_dim * 4);
        let off_q_idx = bump(&mut p_qsa, &mut max_end, sub * q_idx_dim * 4);
        let off_k_idx = bump(&mut p_qsa, &mut max_end, sub * k_idx_dim * 4);
        let off_idx_scores = bump(&mut p_qsa, &mut max_end, sub * max_blocks * 4);
        let off_idx_ids = bump(&mut p_qsa, &mut max_end, sub * max_idx_cap * 4);
        let off_qsa_ws = bump(&mut p_qsa, &mut max_end, qsa_ws_floats * 4);
        let off_attn_out = bump(&mut p_qsa, &mut max_end, sub * q_dim * 4);
        let off_attn_q8 = bump(&mut p_qsa, &mut max_end, sub * attn_q8_row);

        // Shared MoE branch
        let mut p_sh = a2_after_f16;
        let off_sh_gate = bump(&mut p_sh, &mut max_end, sub * cfg.n_ff_shared * 4);
        let off_sh_up = bump(&mut p_sh, &mut max_end, sub * cfg.n_ff_shared * 4);
        let off_sh_hq = bump(&mut p_sh, &mut max_end, sub * sh_q8_row);

        // Phase B (Routed MoE)
        let mut p_b = phase_base;
        let off_moe_grp_gu = bump(&mut p_b, &mut max_end, grp_win * 2 * cfg.n_ff_exp * 4);
        let off_moe_inter_q8 = bump(
            &mut p_b,
            &mut max_end,
            grp_win.max(max_verify * cfg.n_expert_used) * ff_q8_row,
        );

        let arena = DeviceBuffer::alloc(dev, max_end)?;
        // Zero doorbell seq_state ONCE (must never be reset across prompts)
        let db_seq_ptr = unsafe { arena.ptr.cast::<u8>().add(off_db_seq_state).cast::<c_void>() };
        unsafe {
            ampllm_cuda_kernels::zero_raw_async(db_seq_ptr, 16, &stream)?;
        }
        stream.sync()?;

        let dma_stage = DeviceBuffer::alloc(dev, QWEN4_DMA_SLOTS * slot_stride)?;
        let dma_base = dma_stage.as_u64();
        let dma_ptrs_init: Vec<u64> = (0..QWEN4_DMA_SLOTS)
            .map(|s| dma_base + (s * slot_stride) as u64)
            .collect();
        let dma_ptrs_dev_raw = unsafe { arena.ptr.cast::<u8>().add(off_dma_ptrs_dev).cast::<c_void>() };
        unsafe {
            ampllm_cuda_kernels::copy_h2d_raw_async(
                dma_ptrs_dev_raw,
                bytemuck::cast_slice(&dma_ptrs_init),
                &stream,
            )?;
        }
        stream.sync()?;

        // Meta pinned host layout
        let mut mc = 0usize;
        let mut m_alloc = |bytes: usize| -> usize {
            let off = mc;
            mc += align256(bytes.max(16));
            off
        };
        let meta_pin_rel = MetaPinRel {
            exp_ptrs: m_alloc(cfg.n_expert * 8),
            rows: m_alloc(grp_win * 4),
            wts: m_alloc(grp_win * 4),
            exp_row_off: m_alloc(cfg.n_expert * 4),
            exp_nt: m_alloc(cfg.n_expert * 4),
            tile_exp_id: m_alloc(max_tiles * 4),
            tile_m_off: m_alloc(max_tiles * 4),
            slot_bytes: mc,
        };

        let meta_pin = PinnedHostBuffer::alloc(QWEN4_META_SLOTS * meta_pin_rel.slot_bytes)?;
        let h_pos = PinnedHostBuffer::alloc(max_t * 4)?;
        let h_ids = PinnedHostBuffer::alloc(max_pairs * 4)?;
        let h_w = PinnedHostBuffer::alloc(max_pairs * 4)?;
        let h_mixed = PinnedHostBuffer::alloc(max_t * n_embd * 4)?;
        let h_cpu = PinnedHostBuffer::alloc(max_t * n_embd * 4)?;
        let h_small_ptrs = PinnedHostBuffer::alloc(cfg.n_expert * 8)?;
        let h_small_off = PinnedHostBuffer::alloc(cfg.n_expert * 4)?;
        let h_small_nt = PinnedHostBuffer::alloc(cfg.n_expert * 4)?;
        let h_small_rows = PinnedHostBuffer::alloc(max_pairs * 4)?;
        let h_small_pair = PinnedHostBuffer::alloc(64usize.max(max_verify) * cfg.n_expert_used * 4)?;
        let h_small_wts = PinnedHostBuffer::alloc(max_pairs * 4)?;
        let small_pin = PinnedHostBuffer::alloc(4096)?;
        let warmup_bounce = PinnedHostBuffer::alloc(16 * slot_stride)?;
        let mut adapt_patch_pin = PinnedMappedBuffer::<u64>::alloc(8 * 1024 * 2)?;
        adapt_patch_pin.as_mut_slice().fill(0);
        let adapt_patch_dev = adapt_patch_pin.dev_ptr_for_device(dev)? as u64;
        let mut decode_pos_pin = PinnedMappedBuffer::<i32>::alloc(max_verify)?;
        decode_pos_pin.as_mut_slice().fill(0);
        let decode_pos_dev = decode_pos_pin.dev_ptr_for_device(dev)? as u64;
        let mut argmax_out_pin = PinnedMappedBuffer::<i32>::alloc(max_verify)?;
        argmax_out_pin.as_mut_slice().fill(0);
        let argmax_out_dev = argmax_out_pin.dev_ptr_for_device(dev)? as u64;

        Ok(Self {
            device: dev,
            stream,
            copy_stream,
            fetch_stream,
            side_stream,
            d2h_stream,
            dma_events,
            compute_events,
            meta_events,
            d2h_done,
            p2p_done,
            fetch_start_ev,
            fetch_done_ev,
            side_fork_ev,
            side_join_ev,
            adapt_d2h_ev,
            adapt_h2d_ev,
            sub_max: sub,
            qsa_ws_floats,
            max_f16_elems_w,
            slot_stride,
            arena,
            dma_stage,
            off_dma_ptrs_dev,
            off_r_buf,
            off_mixed,
            off_mixed_q8,
            off_inject,
            off_router_logits,
            off_router_ids,
            off_router_w,
            off_block_out,
            off_pos_dev,
            off_mrope_pos3_dev,
            off_widths_dev,
            off_plan_ptrs_dev,
            off_db_seq_state,
            off_db_exp_seq,
            off_argmax_vals,
            off_argmax_ids,
            off_argmax_out,
            off_logits,
            off_sh_out,
            off_sh_gate_logit,
            off_small_ptrs,
            off_small_off,
            off_small_nt,
            off_small_rows,
            off_small_pair,
            off_small_wts,
            off_fetch_src_ptrs,
            off_fetch_dst_ptrs,
            off_fetch_off,
            off_fetch_nt,
            off_fetch_rows,
            off_fetch_pair,
            off_fetch_wts,
            off_gpu_pair_mask,
            off_moe_pair_out,
            meta_dev,
            off_xn,
            off_xn_q8,
            off_hc_lo,
            off_hc_lo_q8,
            off_hc_up,
            off_f16_x,
            off_f16_w,
            off_gdn_qkv,
            off_gdn_z_gate,
            off_gdn_h_qkv,
            off_gdn_gate,
            off_gdn_beta,
            off_gdn_y,
            off_gdn_y_q8,
            off_gdn_tc_scratch,
            off_qfull,
            off_q,
            off_k,
            off_v,
            off_q_idx,
            off_k_idx,
            off_idx_scores,
            off_idx_ids,
            off_qsa_ws,
            off_attn_out,
            off_attn_q8,
            off_sh_gate,
            off_sh_up,
            off_sh_hq,
            off_moe_grp_gu,
            off_moe_inter_q8,
            h_pos,
            h_ids,
            h_w,
            h_mixed,
            h_cpu,
            h_small_ptrs,
            h_small_off,
            h_small_nt,
            h_small_rows,
            h_small_pair,
            h_small_wts,
            meta_pin,
            meta_pin_rel,
            small_pin,
            warmup_bounce,
            adapt_patch_pin,
            adapt_patch_dev,
            adapt_patch_cursor: 0,
            decode_pos_pin,
            decode_pos_dev,
            argmax_out_pin,
            argmax_out_dev,
        })
    }

    #[inline]
    pub fn ptr<T>(&self, off: usize) -> *mut T {
        unsafe { self.arena.ptr.cast::<u8>().add(off).cast::<T>() }
    }

    #[inline]
    pub fn dev_ptr(&self, off: usize) -> u64 {
        self.arena.as_u64() + off as u64
    }
}

pub struct PleGpuState {
    pub sub_cap: usize,
    pub n_embd: usize,
    pub emb: DeviceBuffer,
    pub key: DeviceBuffer,
    pub val: DeviceBuffer,
    pub scratch_qn: DeviceBuffer,
    pub scratch_gated: DeviceBuffer,
    pub scratch_gate: DeviceBuffer,
    pub history: DeviceBuffer,
    pub snap: Option<DeviceBuffer>,
    pub host_f32: PinnedHostBuffer,
    pub h2d_events: [CudaEvent; 2],
    pub decode_emb_pin: PinnedMappedBuffer<f32>,
    pub decode_emb_dev: u64,
}

impl PleGpuState {
    pub fn alloc(dev: i32, cfg: &Qwen4Config, sub_cap: usize, max_verify: usize) -> Result<Self> {
        set_device(dev)?;
        let n = cfg.n_embd;
        let h = cfg.hc;
        let d = h * n;
        let snap = if max_verify > 0 {
            Some(DeviceBuffer::alloc_zeroed(dev, max_verify * d * 9 * 4)?)
        } else {
            None
        };
        let mut decode_emb_pin = PinnedMappedBuffer::<f32>::alloc(max_verify.max(1) * n)?;
        decode_emb_pin.as_mut_slice().fill(0.0);
        let decode_emb_dev = decode_emb_pin.dev_ptr_for_device(dev)? as u64;
        Ok(Self {
            sub_cap,
            n_embd: n,
            emb: DeviceBuffer::alloc(dev, sub_cap * n * 4)?,
            key: DeviceBuffer::alloc(dev, sub_cap * d * 4)?,
            val: DeviceBuffer::alloc(dev, sub_cap * n * 4)?,
            scratch_qn: DeviceBuffer::alloc(dev, sub_cap * d * 4)?,
            scratch_gated: DeviceBuffer::alloc(dev, sub_cap * d * 4)?,
            scratch_gate: DeviceBuffer::alloc(dev, sub_cap * h * 4)?,
            history: DeviceBuffer::alloc_zeroed(dev, d * 9 * 4)?,
            snap,
            host_f32: PinnedHostBuffer::alloc(2 * sub_cap * n * 4)?,
            h2d_events: [CudaEvent::new(dev)?, CudaEvent::new(dev)?],
            decode_emb_pin,
            decode_emb_dev,
        })
    }

    #[inline]
    pub fn host_f32_slot_mut(&mut self, slot: usize, len: usize) -> &mut [f32] {
        let base = (slot & 1) * self.sub_cap * self.n_embd;
        unsafe {
            std::slice::from_raw_parts_mut(self.host_f32.as_mut_ptr().cast::<f32>().add(base), len)
        }
    }

    #[inline]
    pub fn host_slot_bytes(&self, slot: usize, floats: usize) -> &[u8] {
        let base_bytes = (slot & 1) * self.sub_cap * self.n_embd * 4;
        unsafe {
            std::slice::from_raw_parts(self.host_f32.as_ptr().add(base_bytes), floats * 4)
        }
    }
}

pub struct MtpState {
    pub batch_cap: usize,
    pub h2d_done: CudaEvent,
    pub k_q: DeviceBuffer,
    pub v_q: DeviceBuffer,
    pub k_sc: DeviceBuffer,
    pub v_sc: DeviceBuffer,
    pub pos_dev: DeviceBuffer,
    pub width_dev: DeviceBuffer,
    pub emb: DeviceBuffer,
    pub emb_norm: DeviceBuffer,
    pub emb_norm_q8: DeviceBuffer,
    pub emb_fc: DeviceBuffer,
    pub hid_norm: DeviceBuffer,
    pub hid_norm_q8: DeviceBuffer,
    pub hid_fc: DeviceBuffer,
    pub r_mtp: DeviceBuffer,
    pub xn: DeviceBuffer,
    pub xn_q8: DeviceBuffer,
    pub lo: DeviceBuffer,
    pub lo_q8: DeviceBuffer,
    pub hc_up: DeviceBuffer,
    pub mixed: DeviceBuffer,
    pub mixed_q8: DeviceBuffer,
    pub inject: DeviceBuffer,
    pub qfull: DeviceBuffer,
    pub q: DeviceBuffer,
    pub k: DeviceBuffer,
    pub v: DeviceBuffer,
    pub qsa_ws: DeviceBuffer,
    pub qsa_ws_floats: usize,
    pub attn_out: DeviceBuffer,
    pub attn_q8: DeviceBuffer,
    pub block_out: DeviceBuffer,
    pub router_logits: DeviceBuffer,
    pub topk_ids: DeviceBuffer,
    pub topk_w: DeviceBuffer,
    pub moe_inter_q8: DeviceBuffer,
    pub sh_gate: DeviceBuffer,
    pub sh_up: DeviceBuffer,
    pub sh_hq: DeviceBuffer,
    pub sh_out: DeviceBuffer,
    pub sh_gate_logit: DeviceBuffer,
    pub logits: DeviceBuffer,
    pub argmax_vals: DeviceBuffer,
    pub argmax_ids: DeviceBuffer,
    pub argmax_sums: DeviceBuffer,
    pub argmax_out: PinnedMappedBuffer<i32>,
    pub argmax_dev_ptr: u64,
    pub h_pos: PinnedHostBuffer,
    pub h_emb: PinnedHostBuffer,
    pub decode_pos_pin: PinnedMappedBuffer<i32>,
    pub decode_pos_dev: u64,
    pub decode_emb_pin: PinnedMappedBuffer<f32>,
    pub decode_emb_dev: u64,
    pub draft_graphs: HashMap<(bool, usize), CudaGraphExec>,
    pub prefill_graphs: HashMap<usize, CudaGraphExec>,
    pub kv_pos: usize,
}

impl MtpState {
    pub fn alloc(dev: i32, cfg: &Qwen4Config, max_ctx: usize, batch_cap: usize) -> Result<Self> {
        set_device(dev)?;
        let n_embd = cfg.n_embd;
        let hc = cfg.hc;
        let hc_dim = hc * n_embd;
        let hc_lr = cfg.hc_low_rank;
        let qfull_dim = cfg.n_head * cfg.head_dim * 2;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let qsa_ws_floats = qwen4::qsa_decode_ws_floats(1, cfg.n_head_kv as i32, max_ctx as i32);
        let mut argmax_out = PinnedMappedBuffer::alloc(2)?;
        argmax_out.as_mut_slice().fill(0);
        let argmax_dev_ptr = argmax_out.dev_ptr_for_device(dev)? as u64;
        let mut decode_pos_pin = PinnedMappedBuffer::<i32>::alloc(8)?;
        decode_pos_pin.as_mut_slice().fill(0);
        let decode_pos_dev = decode_pos_pin.dev_ptr_for_device(dev)? as u64;
        let mut decode_emb_pin = PinnedMappedBuffer::<f32>::alloc(8 * n_embd)?;
        decode_emb_pin.as_mut_slice().fill(0.0);
        let decode_emb_dev = decode_emb_pin.dev_ptr_for_device(dev)? as u64;

        Ok(Self {
            batch_cap,
            h2d_done: CudaEvent::new(dev)?,
            k_q: DeviceBuffer::alloc_zeroed(dev, max_ctx * kv_dim)?,
            v_q: DeviceBuffer::alloc_zeroed(dev, max_ctx * kv_dim)?,
            k_sc: DeviceBuffer::alloc_zeroed(dev, max_ctx * cfg.n_head_kv * 4 * 2)?,
            v_sc: DeviceBuffer::alloc_zeroed(dev, max_ctx * cfg.n_head_kv * 4 * 2)?,
            pos_dev: DeviceBuffer::alloc(dev, batch_cap * 4)?,
            width_dev: DeviceBuffer::alloc(dev, 4)?,
            emb: DeviceBuffer::alloc(dev, batch_cap * n_embd * 4)?,
            emb_norm: DeviceBuffer::alloc(dev, batch_cap * n_embd * 4)?,
            emb_norm_q8: DeviceBuffer::alloc(dev, batch_cap * (n_embd / 32) * 36)?,
            emb_fc: DeviceBuffer::alloc(dev, batch_cap * n_embd * 4)?,
            hid_norm: DeviceBuffer::alloc(dev, batch_cap * hc_dim * 4)?,
            hid_norm_q8: DeviceBuffer::alloc(dev, batch_cap * (hc_dim / 32) * 36)?,
            hid_fc: DeviceBuffer::alloc(dev, batch_cap * hc_dim * 4)?,
            r_mtp: DeviceBuffer::alloc(dev, batch_cap * hc_dim * 4)?,
            xn: DeviceBuffer::alloc(dev, batch_cap * hc_dim * 4)?,
            xn_q8: DeviceBuffer::alloc(dev, batch_cap * (hc_dim / 32) * 36)?,
            lo: DeviceBuffer::alloc(dev, batch_cap * hc_lr * 4)?,
            lo_q8: DeviceBuffer::alloc(dev, batch_cap * (hc_lr / 32) * 36)?,
            hc_up: DeviceBuffer::alloc(dev, batch_cap * hc_dim * 4)?,
            mixed: DeviceBuffer::alloc(dev, batch_cap * n_embd * 4)?,
            mixed_q8: DeviceBuffer::alloc(dev, batch_cap * (n_embd / 32) * 36)?,
            inject: DeviceBuffer::alloc(dev, batch_cap * hc * hc * 4)?,
            qfull: DeviceBuffer::alloc(dev, qfull_dim * 4)?,
            q: DeviceBuffer::alloc(dev, q_dim * 4)?,
            k: DeviceBuffer::alloc(dev, batch_cap * kv_dim * 4)?,
            v: DeviceBuffer::alloc(dev, batch_cap * kv_dim * 4)?,
            qsa_ws: DeviceBuffer::alloc(dev, qsa_ws_floats * 4)?,
            qsa_ws_floats,
            attn_out: DeviceBuffer::alloc(dev, q_dim * 4)?,
            attn_q8: DeviceBuffer::alloc(dev, (q_dim / 32) * 36)?,
            block_out: DeviceBuffer::alloc(dev, n_embd * 4)?,
            router_logits: DeviceBuffer::alloc(dev, cfg.n_expert * 4)?,
            topk_ids: DeviceBuffer::alloc(dev, cfg.n_expert_used * 4)?,
            topk_w: DeviceBuffer::alloc(dev, cfg.n_expert_used * 4)?,
            moe_inter_q8: DeviceBuffer::alloc(dev, cfg.n_expert_used * (cfg.n_ff_exp / 32) * 36)?,
            sh_gate: DeviceBuffer::alloc(dev, cfg.n_ff_shared * 4)?,
            sh_up: DeviceBuffer::alloc(dev, cfg.n_ff_shared * 4)?,
            sh_hq: DeviceBuffer::alloc(dev, (cfg.n_ff_shared / 32) * 36)?,
            sh_out: DeviceBuffer::alloc(dev, n_embd * 4)?,
            sh_gate_logit: DeviceBuffer::alloc(dev, 4)?,
            logits: DeviceBuffer::alloc(dev, cfg.n_vocab * 4)?,
            argmax_vals: DeviceBuffer::alloc(dev, 128 * 4)?,
            argmax_ids: DeviceBuffer::alloc(dev, 128 * 4)?,
            argmax_sums: DeviceBuffer::alloc(dev, 128 * 4)?,
            argmax_out,
            argmax_dev_ptr,
            h_pos: PinnedHostBuffer::alloc(batch_cap * 4)?,
            h_emb: PinnedHostBuffer::alloc(batch_cap * n_embd * 4)?,
            decode_pos_pin,
            decode_pos_dev,
            decode_emb_pin,
            decode_emb_dev,
            draft_graphs: HashMap::new(),
            prefill_graphs: HashMap::new(),
            kv_pos: 0,
        })
    }
}

struct SendDbPtr(*mut Qwen4Doorbell);
unsafe impl Send for SendDbPtr {}

pub struct Qwen4DoorbellGuard {
    active: Arc<AtomicUsize>,
}

impl Drop for Qwen4DoorbellGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::Release);
    }
}

pub struct Qwen4DoorbellWorker {
    pub db0: PinnedMappedBuffer<Qwen4Doorbell>,
    pub db1: Option<PinnedMappedBuffer<Qwen4Doorbell>>,
    pub db0_dev: u64,
    pub db1_dev: u64,
    active: Arc<AtomicUsize>,
    wake: Arc<(Mutex<bool>, Condvar)>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

unsafe impl Send for Qwen4DoorbellWorker {}
unsafe impl Sync for Qwen4DoorbellWorker {}

impl Drop for Qwen4DoorbellWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        {
            let (lock, cv) = &*self.wake;
            if let Ok(mut g) = lock.lock() {
                *g = true;
                cv.notify_all();
            }
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Qwen4DoorbellWorker {
    pub fn spawn(pool: Arc<CpuThreadPool>, weights: &Qwen4Weights) -> Result<Self> {
        let mut db0 = PinnedMappedBuffer::<Qwen4Doorbell>::alloc(1)?;
        db0.as_mut_slice()[0] = Qwen4Doorbell::default();
        let db0_dev = db0.dev_ptr_for_device(weights.gpu_ids[0])? as u64;

        let (mut db1, db1_dev) = if weights.gpu_ids.len() >= 2 {
            let mut buf = PinnedMappedBuffer::<Qwen4Doorbell>::alloc(1)?;
            buf.as_mut_slice()[0] = Qwen4Doorbell::default();
            let dev_ptr = buf.dev_ptr_for_device(weights.gpu_ids[1])? as u64;
            (Some(buf), dev_ptr)
        } else {
            (None, 0u64)
        };

        let db0_ptr = SendDbPtr(db0.as_mut_slice().as_mut_ptr());
        let db1_ptr = db1
            .as_mut()
            .map(|b| SendDbPtr(b.as_mut_slice().as_mut_ptr()));

        let layers_atomic = Arc::clone(&weights.layers_host_atomic);
        let layer_u_off = Arc::clone(&weights.layer_u_off);
        let layer_d_off = Arc::clone(&weights.layer_d_off);
        let active = Arc::new(AtomicUsize::new(0));
        let wake = Arc::new((Mutex::new(false), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let active_w = Arc::clone(&active);
        let wake_w = Arc::clone(&wake);
        let stop_w = Arc::clone(&stop);
        let cfg_embd = weights.cfg.n_embd;
        let cfg_ff = weights.cfg.n_ff_exp;
        let cfg_exp = weights.cfg.n_expert;

        let handle = std::thread::Builder::new()
            .name("qwen4-doorbell".to_owned())
            .spawn(move || {
                let db0_raw = db0_ptr;
                let db1_raw = db1_ptr;
                let mut scratch = Glm5CpuScratch::new(cfg_embd, cfg_ff, cfg_exp.max(80));
                let mut seen0 = 0u64;
                let mut seen1 = 0u64;

                let mut grp_eids = [0usize; 80];
                let mut grp_toks = [[0usize; 8]; 80];
                let mut grp_wts = [[0.0f32; 8]; 80];
                let mut grp_len = [0usize; 80];
                let mut x_stage = vec![0.0f32; 8 * cfg_embd];
                let mut out_stage = vec![0.0f32; 8 * cfg_embd];

                let process_db = |db: &mut Qwen4Doorbell,
                                  seen: &mut u64,
                                  scratch: &mut Glm5CpuScratch,
                                  x_stage: &mut [f32],
                                  out_stage: &mut [f32],
                                  grp_eids: &mut [usize; 80],
                                  grp_toks: &mut [[usize; 8]; 80],
                                  grp_wts: &mut [[f32; 8]; 80],
                                  grp_len: &mut [usize; 80]| {
                    let req = unsafe { std::ptr::read_volatile(&db.req_seq) };
                    if req <= *seen {
                        return false;
                    }
                    std::sync::atomic::fence(Ordering::Acquire);
                    let mb = (db.m_batch as usize).clamp(1, 8);
                    let l = db.layer_idx as usize;
                    let mc = (db.miss_count as usize).min(80);
                    let gu_type = GgmlType::from_u32(db.gu_type as u32).unwrap_or(GgmlType::Q4K);
                    let d_type = GgmlType::from_u32(db.d_type as u32).unwrap_or(GgmlType::Q5_1);
                    let n_embd = db.n_embd as usize;
                    let n_ff = db.n_ff as usize;
                    let u_off = layer_u_off.get(l).copied().unwrap_or(0);
                    let d_off = layer_d_off.get(l).copied().unwrap_or(0);

                    if mc == 0 {
                        db.out[..mb * n_embd].fill(0.0);
                    } else if mb == 1 {
                        let mut jobs = Vec::with_capacity(mc);
                        for i in 0..mc {
                            let eid = db.miss_ids[i] as usize;
                            if let Some(atom) = layers_atomic.get(l).and_then(|r| r.get(eid)) {
                                let base = atom.load(Ordering::Acquire);
                                if base != 0 {
                                    jobs.push(Glm5CpuExpertJob {
                                        gate_ptr: base as *const u8,
                                        up_ptr: (base + u_off) as *const u8,
                                        down_ptr: (base + d_off) as *const u8,
                                        weight: db.miss_weights[i],
                                    });
                                }
                            }
                        }
                        run_glm5_cpu_moe_m1(
                            &pool,
                            scratch,
                            gu_type,
                            d_type,
                            n_embd,
                            n_ff,
                            0.0,
                            &db.x_in[..n_embd],
                            None,
                            &jobs,
                            &mut db.out[..n_embd],
                        );
                    } else {
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
                            if let Some(g) = (0..n_groups).find(|&g| grp_eids[g] == eid) {
                                let k = grp_len[g];
                                if k < 8 {
                                    grp_toks[g][k] = tok;
                                    grp_wts[g][k] = wt;
                                    grp_len[g] = k + 1;
                                }
                            } else if n_groups < 80 {
                                grp_eids[n_groups] = eid;
                                grp_toks[n_groups][0] = tok;
                                grp_wts[n_groups][0] = wt;
                                grp_len[n_groups] = 1;
                                n_groups += 1;
                            }
                        }
                        let mut groups_raw: [std::mem::MaybeUninit<Glm5CpuGroupedExpert<'_>>; 80] =
                            std::array::from_fn(|_| std::mem::MaybeUninit::uninit());
                        let mut valid_groups = 0usize;
                        for g in 0..n_groups {
                            let eid = grp_eids[g];
                            if let Some(atom) = layers_atomic.get(l).and_then(|r| r.get(eid)) {
                                let base = atom.load(Ordering::Acquire);
                                if base != 0 {
                                    let len = grp_len[g];
                                    groups_raw[valid_groups].write(Glm5CpuGroupedExpert {
                                        gate_ptr: base as *const u8,
                                        up_ptr: (base + u_off) as *const u8,
                                        down_ptr: (base + d_off) as *const u8,
                                        token_indices: &grp_toks[g][..len],
                                        token_weights: &grp_wts[g][..len],
                                    });
                                    valid_groups += 1;
                                }
                            }
                        }
                        let groups = unsafe {
                            std::slice::from_raw_parts(
                                groups_raw.as_ptr().cast::<Glm5CpuGroupedExpert<'_>>(),
                                valid_groups,
                            )
                        };
                        run_glm5_cpu_moe_grouped_small(
                            &pool,
                            scratch,
                            gu_type,
                            d_type,
                            n_embd,
                            n_ff,
                            0.0,
                            mb,
                            &x_stage[..x_len],
                            groups,
                            &mut out_stage[..x_len],
                            true,
                        );
                        db.out[..x_len].copy_from_slice(&out_stage[..x_len]);
                    }
                    std::sync::atomic::fence(Ordering::Release);
                    unsafe {
                        std::ptr::write_volatile(&mut db.done_seq, req);
                    }
                    *seen = req;
                    true
                };

                while !stop_w.load(Ordering::Acquire) {
                    if active_w.load(Ordering::Acquire) == 0 {
                        let (lock, cv) = &*wake_w;
                        if let Ok(mut g) = lock.lock() {
                            while active_w.load(Ordering::Acquire) == 0
                                && !stop_w.load(Ordering::Acquire)
                            {
                                *g = false;
                                g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
                            }
                        }
                        continue;
                    }
                    let d0 = unsafe { &mut *db0_raw.0 };
                    let mut did = process_db(
                        d0,
                        &mut seen0,
                        &mut scratch,
                        &mut x_stage,
                        &mut out_stage,
                        &mut grp_eids,
                        &mut grp_toks,
                        &mut grp_wts,
                        &mut grp_len,
                    );
                    if let Some(ref p1) = db1_raw {
                        let d1 = unsafe { &mut *p1.0 };
                        if process_db(
                            d1,
                            &mut seen1,
                            &mut scratch,
                            &mut x_stage,
                            &mut out_stage,
                            &mut grp_eids,
                            &mut grp_toks,
                            &mut grp_wts,
                            &mut grp_len,
                        ) {
                            did = true;
                        }
                    }
                    if !did {
                        std::hint::spin_loop();
                    }
                }
            })?;

        Ok(Self {
            db0,
            db1,
            db0_dev,
            db1_dev,
            active,
            wake,
            stop,
            handle: Some(handle),
        })
    }

    pub fn active_guard(&self) -> Qwen4DoorbellGuard {
        self.active.fetch_add(1, Ordering::Release);
        let (lock, cv) = &*self.wake;
        if let Ok(mut g) = lock.lock() {
            *g = true;
            cv.notify_all();
        }
        Qwen4DoorbellGuard {
            active: Arc::clone(&self.active),
        }
    }

    #[inline]
    pub fn dev_ptr(&self, gidx: usize) -> u64 {
        if gidx == 0 {
            self.db0_dev
        } else {
            self.db1_dev
        }
    }
}

pub type TrunkGraphKey = (usize, bool, bool, bool, i32, i32);

#[derive(Clone, Copy, Debug)]
pub struct AdaptSwap {
    pub layer: usize,
    pub gidx: usize,
    pub ev_e: usize,
    pub in_e: usize,
    pub slot: usize,
    pub v_ptr: u64,
    pub free_h: usize,
    pub free_d: u64,
    pub old_in_h: usize,
    pub old_in_d: u64,
    pub blob_bytes: usize,
}

pub enum AsyncAdaptState {
    Idle,
    D2hInFlight { swaps: Vec<AdaptSwap>, gpu_mask: u32 },
    H2dInFlight { swaps: Vec<AdaptSwap>, gpu_mask: u32 },
}

pub struct Qwen4State {
    pub max_ctx: usize,
    pub max_t: usize,
    pub max_verify: usize,
    pub max_idx_cap: usize,
    pub pos: usize,
    pub mrope_pos: usize,
    pub mrope_active: bool,
    pub mrope_table_host: Vec<i32>,
    pub prev_tokens: Vec<u32>,
    pub ple_cursor: PleCursor,
    pub ple_cursor_snap: Option<PleCursor>,
    pub gpus: Vec<GpuStageScratch>,
    pub layers: Vec<Qwen4LayerState>,
    pub ple: Option<PleGpuState>,

    // Head + MTP + CPU/Graph resources
    pub head_dev: i32,
    pub r_last: DeviceBuffer,
    pub logits_host: Vec<f32>,
    pub h_emb_stage: PinnedHostBuffer,
    pub decode_emb_pin: PinnedMappedBuffer<f32>,
    pub decode_emb_dev: u64,
    pub mtp: Option<MtpState>,

    pub pool: Arc<CpuThreadPool>,
    pub pool0: Option<Arc<CpuThreadPool>>,
    pub pool1: Option<Arc<CpuThreadPool>>,
    pub cpu_scratch_full: Glm5CpuScratch,
    pub cpu_scratch0: Option<Glm5CpuScratch>,
    pub cpu_scratch1: Option<Glm5CpuScratch>,
    pub doorbell: Qwen4DoorbellWorker,

    pub use_cuda_graph: bool,
    pub trunk_graphs: Vec<HashMap<TrunkGraphKey, CudaGraphExec>>,
    pub rollback_graphs: Vec<HashMap<usize, CudaGraphExec>>,
    pub cache_stats: CacheStats,
    pub decode_cache_stats: CacheStats,
    pub step_counter: u64,
    pub adapt_state: AsyncAdaptState,
}

impl Qwen4State {
    pub fn alloc(
        weights: &Qwen4Weights,
        engine_cfg: &EngineConfig,
        pool: Arc<CpuThreadPool>,
        mtp_k: usize,
    ) -> Result<Self> {
        let cfg = &weights.cfg;
        let max_ctx = engine_cfg.max_ctx.min(cfg.context_length).max(256);
        let has_mtp = weights.mtp.is_some() && mtp_k > 0;
        let max_verify = if has_mtp { QWEN4_MAX_VERIFY } else { 1 };
        let chunk_env = std::env::var("AMPLLM_QWEN4_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(QWEN4_DEFAULT_CHUNK);
        let max_t = chunk_env.clamp(128, 8192).max(max_verify).min(max_ctx).max(16);
        let max_idx_cap = cfg.idx_top_k + 3;

        let n_gpus = weights.gpu_ids.len();
        let mut gpus = Vec::with_capacity(n_gpus);
        for (idx, &dev) in weights.gpu_ids.iter().enumerate() {
            let is_head = idx + 1 == n_gpus;
            gpus.push(GpuStageScratch::alloc(
                dev,
                cfg,
                max_ctx,
                max_t,
                QWEN4_MAX_VERIFY,
                max_idx_cap,
                weights.max_blob_bytes,
                is_head,
            )?);
        }

        let max_blocks = (max_ctx / 4) + 1;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let gdn_qkv_dim = cfg.gdn_qkv_dim();
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for (il, lw) in weights.layers.iter().enumerate() {
            let dev = lw.device;
            set_device(dev)?;
            if cfg.is_qsa(il) {
                layers.push(Qwen4LayerState::Qsa(QsaLayerState {
                    k_q: DeviceBuffer::alloc_zeroed(dev, max_ctx * kv_dim)?,
                    v_q: DeviceBuffer::alloc_zeroed(dev, max_ctx * kv_dim)?,
                    k_sc: DeviceBuffer::alloc_zeroed(dev, max_ctx * cfg.n_head_kv * 4 * 2)?,
                    v_sc: DeviceBuffer::alloc_zeroed(dev, max_ctx * cfg.n_head_kv * 4 * 2)?,
                    tail: DeviceBuffer::alloc_zeroed(dev, 4 * cfg.idx_key_dim * 4)?,
                    dead: DeviceBuffer::alloc_zeroed(dev, cfg.idx_key_dim * 4)?,
                    pooled: DeviceBuffer::alloc_zeroed(dev, max_blocks * cfg.idx_key_dim * 4)?,
                    tail_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, 4 * cfg.idx_key_dim * 4)?)
                    } else {
                        None
                    },
                    dead_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, cfg.idx_key_dim * 4)?)
                    } else {
                        None
                    },
                    k_idx_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, QWEN4_MAX_VERIFY * cfg.idx_key_dim * 4)?)
                    } else {
                        None
                    },
                }));
            } else {
                let s_elems = cfg.ssm_time_step_rank * cfg.ssm_state_size * cfg.ssm_state_size;
                layers.push(Qwen4LayerState::Gdn(GdnLayerState {
                    state: DeviceBuffer::alloc_zeroed(dev, s_elems * 4)?,
                    conv_hist: DeviceBuffer::alloc_zeroed(dev, gdn_qkv_dim * 3 * 4)?,
                    state_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, s_elems * 4)?)
                    } else {
                        None
                    },
                    hist_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, QWEN4_MAX_VERIFY * gdn_qkv_dim * 3 * 4)?)
                    } else {
                        None
                    },
                    h_qkv_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, QWEN4_MAX_VERIFY * gdn_qkv_dim * 4)?)
                    } else {
                        None
                    },
                    gate_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, QWEN4_MAX_VERIFY * cfg.ssm_time_step_rank * 4)?)
                    } else {
                        None
                    },
                    beta_snap: if has_mtp {
                        Some(DeviceBuffer::alloc_zeroed(dev, QWEN4_MAX_VERIFY * cfg.ssm_time_step_rank * 4)?)
                    } else {
                        None
                    },
                }));
            }
        }

        let sub = QWEN4_SUB.min(max_t);
        let ple = if weights.ple_shard_idx.is_some() {
            Some(PleGpuState::alloc(
                weights.gpu_ids[0],
                cfg,
                sub,
                if has_mtp { QWEN4_MAX_VERIFY } else { 0 },
            )?)
        } else {
            None
        };

        let head_dev = *weights.gpu_ids.last().unwrap();
        set_device(head_dev)?;
        let r_last = DeviceBuffer::alloc(head_dev, QWEN4_MAX_VERIFY * cfg.hc * cfg.n_embd * 4)?;
        let logits_host = vec![0.0f32; QWEN4_MAX_VERIFY * cfg.n_vocab];
        let h_emb_stage = PinnedHostBuffer::alloc(max_t * cfg.n_embd * 4)?;
        set_device(weights.gpu_ids[0])?;
        let mut decode_emb_pin = PinnedMappedBuffer::<f32>::alloc(QWEN4_MAX_VERIFY * cfg.n_embd)?;
        decode_emb_pin.as_mut_slice().fill(0.0);
        let decode_emb_dev = decode_emb_pin.dev_ptr_for_device(weights.gpu_ids[0])? as u64;
        set_device(head_dev)?;

        let mtp = if has_mtp {
            Some(MtpState::alloc(head_dev, cfg, max_ctx, sub)?)
        } else {
            None
        };

        let doorbell = Qwen4DoorbellWorker::spawn(Arc::clone(&pool), weights)?;
        let (pool0, pool1, cpu_scratch0, cpu_scratch1) = if n_gpus >= 2 {
            let (c0, c1) = discover_ccd_split_cores();
            (
                Some(Arc::new(CpuThreadPool::from_cores(c0))),
                Some(Arc::new(CpuThreadPool::from_cores(c1))),
                Some(Glm5CpuScratch::new(cfg.n_embd, cfg.n_ff_exp, max_t.max(64))),
                Some(Glm5CpuScratch::new(cfg.n_embd, cfg.n_ff_exp, max_t.max(64))),
            )
        } else {
            (None, None, None, None)
        };
        let cpu_scratch_full = Glm5CpuScratch::new(cfg.n_embd, cfg.n_ff_exp, max_t.max(64));

        let use_cuda_graph = engine_cfg.use_cuda_graph
            && std::env::var("AMPLLM_QWEN4_GRAPH").map_or(true, |v| v != "0");
        let trunk_graphs = (0..n_gpus).map(|_| HashMap::new()).collect();
        let rollback_graphs = (0..n_gpus).map(|_| HashMap::new()).collect();

        Ok(Self {
            max_ctx,
            max_t,
            max_verify,
            max_idx_cap,
            pos: 0,
            mrope_pos: 0,
            mrope_active: false,
            mrope_table_host: Vec::new(),
            prev_tokens: Vec::new(),
            ple_cursor: PleCursor::default(),
            ple_cursor_snap: None,
            gpus,
            layers,
            ple,
            head_dev,
            r_last,
            logits_host,
            h_emb_stage,
            decode_emb_pin,
            decode_emb_dev,
            mtp,
            pool,
            pool0,
            pool1,
            cpu_scratch_full,
            cpu_scratch0,
            cpu_scratch1,
            doorbell,
            use_cuda_graph,
            trunk_graphs,
            rollback_graphs,
            cache_stats: CacheStats::default(),
            decode_cache_stats: CacheStats::default(),
            step_counter: 0,
            adapt_state: AsyncAdaptState::Idle,
        })
    }

    pub fn gpu_idx_for_device(&self, dev: i32) -> usize {
        self.gpus
            .iter()
            .position(|g| g.device == dev)
            .unwrap_or(0)
    }

    pub fn sync_all(&self) -> Result<()> {
        for g in &self.gpus {
            set_device(g.device)?;
            g.stream.sync()?;
            g.copy_stream.sync()?;
            g.d2h_stream.sync()?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        for ls in &self.layers {
            match ls {
                Qwen4LayerState::Gdn(g) => {
                    g.state.zero()?;
                    g.conv_hist.zero()?;
                }
                Qwen4LayerState::Qsa(q) => {
                    q.tail.zero()?;
                    q.dead.zero()?;
                    q.pooled.zero()?;
                }
            }
        }
        if let Some(ref ple) = self.ple {
            ple.history.zero()?;
        }
        if let Some(ref mut mtp) = self.mtp {
            mtp.kv_pos = 0;
        }
        self.pos = 0;
        self.mrope_pos = 0;
        self.mrope_active = false;
        self.mrope_table_host.clear();
        self.prev_tokens.clear();
        self.ple_cursor.reset();
        self.ple_cursor_snap = None;
        Ok(())
    }
}
