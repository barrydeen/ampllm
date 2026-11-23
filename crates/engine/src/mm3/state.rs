use crate::config::EngineConfig;
use crate::mm3::weights::{Mm3FfnWeights, Mm3KvFormat, Mm3Weights};
use ampllm_cache::{align_up_256, CacheStats};
use ampllm_cpu_kernels::mm3_cpu::Mm3CpuScratch;
use ampllm_cpu_kernels::pool::discover_cross_ccd_interleaved_cores;
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::{
    mm3, set_device, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer, PinnedMappedBuffer,
};
use anyhow::Result;
use std::sync::Arc;

pub const MM3_SUB: usize = 1024;
pub const MM3_MAX_VERIFY: usize = 8;
pub const MM3_PREFILL_GRP_WIN: usize = 2048;
pub const MM3_DMA_RING: usize = 2;
pub const MM3_DMA_BATCH: usize = 8;
pub const MM3_META_SLOTS: usize = 4;
pub const MM3_ATTN_SPLITS: i32 = 16;

pub struct Mm3LayerKv {
    pub device: i32,
    pub ic_bf16: DeviceBuffer,
    pub kc_bf16: Option<DeviceBuffer>,
    pub vc_bf16: Option<DeviceBuffer>,
    pub kc_i8: Option<DeviceBuffer>,
    pub vc_i8: Option<DeviceBuffer>,
    pub ks_i8: Option<DeviceBuffer>,
    pub vs_i8: Option<DeviceBuffer>,
}

impl Mm3LayerKv {
    pub fn alloc(weights: &Mm3Weights, max_ctx: usize) -> Result<Vec<Self>> {
        let cfg = &weights.cfg;
        let ic_bytes = max_ctx * cfg.ix_head_dim * 2;
        let kv_elems = max_ctx * cfg.n_kv_heads * cfg.head_dim;
        let scale_bytes = max_ctx * cfg.n_kv_heads * 4;
        let mut out = Vec::with_capacity(cfg.n_layers);

        for lw in &weights.layers {
            set_device(lw.device)?;
            let ic_bf16 = DeviceBuffer::alloc_zeroed(lw.device, ic_bytes)?;
            let (kc_bf16, vc_bf16, kc_i8, vc_i8, ks_i8, vs_i8) = match weights.kv_fmt {
                Mm3KvFormat::Bf16 => (
                    Some(DeviceBuffer::alloc_zeroed(lw.device, kv_elems * 2)?),
                    Some(DeviceBuffer::alloc_zeroed(lw.device, kv_elems * 2)?),
                    None,
                    None,
                    None,
                    None,
                ),
                Mm3KvFormat::Int8HadV => (
                    None,
                    None,
                    Some(DeviceBuffer::alloc_zeroed(lw.device, kv_elems)?),
                    Some(DeviceBuffer::alloc_zeroed(lw.device, kv_elems)?),
                    Some(DeviceBuffer::alloc_zeroed(lw.device, scale_bytes)?),
                    Some(DeviceBuffer::alloc_zeroed(lw.device, scale_bytes)?),
                ),
            };
            out.push(Self {
                device: lw.device,
                ic_bf16,
                kc_bf16,
                vc_bf16,
                kc_i8,
                vc_i8,
                ks_i8,
                vs_i8,
            });
        }
        Ok(out)
    }

    pub fn zero_all(&self) -> Result<()> {
        set_device(self.device)?;
        self.ic_bf16.zero()?;
        if let Some(ref b) = self.kc_bf16 {
            b.zero()?;
        }
        if let Some(ref b) = self.vc_bf16 {
            b.zero()?;
        }
        if let Some(ref b) = self.kc_i8 {
            b.zero()?;
        }
        if let Some(ref b) = self.vc_i8 {
            b.zero()?;
        }
        if let Some(ref b) = self.ks_i8 {
            b.zero()?;
        }
        if let Some(ref b) = self.vs_i8 {
            b.zero()?;
        }
        Ok(())
    }
}

pub struct Mm3MetaSlot {
    pub exp_ptrs: DeviceBuffer,
    pub exp_row_off: DeviceBuffer,
    pub exp_nt: DeviceBuffer,
    pub tile_exp_id: DeviceBuffer,
    pub tile_m_off: DeviceBuffer,
    pub h_exp_ptrs: PinnedHostBuffer,
    pub h_exp_row_off: PinnedHostBuffer,
    pub h_exp_nt: PinnedHostBuffer,
    pub h_tile_exp_id: PinnedHostBuffer,
    pub h_tile_m_off: PinnedHostBuffer,
}

impl Mm3MetaSlot {
    fn alloc(device: i32, n_exp: usize, max_tiles: usize) -> Result<Self> {
        Ok(Self {
            exp_ptrs: DeviceBuffer::alloc(device, n_exp * 8)?,
            exp_row_off: DeviceBuffer::alloc(device, n_exp * 4)?,
            exp_nt: DeviceBuffer::alloc(device, n_exp * 4)?,
            tile_exp_id: DeviceBuffer::alloc(device, max_tiles * 4)?,
            tile_m_off: DeviceBuffer::alloc(device, max_tiles * 4)?,
            h_exp_ptrs: PinnedHostBuffer::alloc(n_exp * 8)?,
            h_exp_row_off: PinnedHostBuffer::alloc(n_exp * 4)?,
            h_exp_nt: PinnedHostBuffer::alloc(n_exp * 4)?,
            h_tile_exp_id: PinnedHostBuffer::alloc(max_tiles * 4)?,
            h_tile_m_off: PinnedHostBuffer::alloc(max_tiles * 4)?,
        })
    }
}

pub struct Mm3GpuScratch {
    pub device: i32,
    pub is_head_gpu: bool,
    pub stream: CudaStream,
    pub side_stream: CudaStream,
    pub copy_stream: CudaStream,
    pub d2h_stream: CudaStream,
    pub norm_ev: CudaEvent,
    pub idx_done_ev: CudaEvent,
    pub router_ev: CudaEvent,
    pub d2h_ev: CudaEvent,
    pub handoff_ev: CudaEvent,
    pub p2p_done_ev: CudaEvent,
    pub promo_gate_ev: CudaEvent,
    pub taps_ready_ev: CudaEvent,
    pub dma_events: [CudaEvent; MM3_DMA_RING],
    pub compute_events: [CudaEvent; MM3_DMA_RING],
    pub meta_events: [CudaEvent; MM3_META_SLOTS],

    // Core activations
    pub h: DeviceBuffer,
    pub xn: DeviceBuffer,
    pub block_out: DeviceBuffer,
    pub sh_out: DeviceBuffer,
    pub x_q8_1: DeviceBuffer,
    pub x_q8_k: DeviceBuffer,

    // Attention + MSA indexer
    pub q: DeviceBuffer,
    pub k: DeviceBuffer,
    pub v: DeviceBuffer,
    pub attn_out: DeviceBuffer,
    pub attn_q8_1: DeviceBuffer,
    pub iq: DeviceBuffer,
    pub ik: DeviceBuffer,
    pub idx_scores: DeviceBuffer,
    pub idx_kept: DeviceBuffer,
    pub attn_part: DeviceBuffer,
    pub pos_dev: DeviceBuffer,

    // Dense & Shared MLP
    pub mlp_gate: DeviceBuffer,
    pub mlp_up: DeviceBuffer,
    pub mlp_inter_q8_1: DeviceBuffer,

    // Router & Decode MoE
    pub router_logits: DeviceBuffer,
    pub router_ids: DeviceBuffer,
    pub router_wts: DeviceBuffer,
    pub moe_inter_f32: DeviceBuffer,
    pub moe_inter_q8k: DeviceBuffer,
    pub moe_expert_out: DeviceBuffer,
    pub decode_plan_dev: [DeviceBuffer; 2],
    pub h_decode_plan: [PinnedHostBuffer; 2],
    pub decode_dma_cursor: usize,

    // Prefill grouped MMQ + double-buffered DMA staging
    pub moe_grp_gate: DeviceBuffer,
    pub moe_grp_up: DeviceBuffer,
    pub moe_grp_inter_q8_1: DeviceBuffer,
    pub grp_rows_dev: DeviceBuffer,
    pub grp_wts_dev: DeviceBuffer,
    pub meta_slots: [Mm3MetaSlot; MM3_META_SLOTS],
    pub dma_stage: DeviceBuffer,
    pub slot_stride: usize,
    pub max_tiles: usize,

    // Head GPU logits & argmax
    pub logits: Option<DeviceBuffer>,
    pub argmax_vals: Option<DeviceBuffer>,
    pub argmax_ids: Option<DeviceBuffer>,
    pub argmax_out: Option<DeviceBuffer>,
    pub h_argmax: Option<PinnedHostBuffer>,

    // DSpark taps
    pub taps_dev: Option<DeviceBuffer>,
    pub local_taps_dev: Option<DeviceBuffer>,
    pub local_tap_indices: Vec<(usize, usize)>,

    // Pinned host staging
    pub h_pos: PinnedHostBuffer,
    pub h_ids: PinnedHostBuffer,
    pub h_wts: PinnedHostBuffer,
    pub h_x_q8k: PinnedHostBuffer,
    pub h_cpu_out: PinnedHostBuffer,
    pub h_cpu_out_mapped: PinnedMappedBuffer<f32>,
    pub h_cpu_wait_mapped: PinnedMappedBuffer<u32>,
    pub cpu_wait_seq: u32,
    pub h_grp_rows: PinnedHostBuffer,
    pub h_grp_wts: PinnedHostBuffer,
}

impl Mm3GpuScratch {
    pub fn alloc(
        device: i32,
        gpu_idx: usize,
        weights: &Mm3Weights,
        max_ctx: usize,
    ) -> Result<Self> {
        set_device(device)?;
        let cfg = &weights.cfg;
        let is_head_gpu = gpu_idx + 1 == weights.gpu_ids.len();
        let max_t = MM3_SUB;
        let n_embd = cfg.n_embd;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let n_exp = cfg.n_experts;
        let topk = cfg.n_experts_used;
        let max_blocks = max_ctx.div_ceil(cfg.ix_block_size).max(1);
        let max_ff = cfg.n_dense_ff.max(cfg.expert_ff);
        let q8_1_x_rb = (n_embd / 32) * 36;
        let q8_1_q_rb = (q_dim / 32) * 36;
        let q8_1_ff_rb = (max_ff / 32) * 36;
        let q8_k_x_rb = n_embd.div_ceil(256) * mm3::Q8K_BLOCK_BYTES;
        let q8_k_exp_rb = cfg.expert_ff.div_ceil(256) * mm3::Q8K_BLOCK_BYTES;

        let max_blob_bytes = weights
            .layers
            .iter()
            .filter_map(|lw| match lw.ffn {
                Mm3FfnWeights::Moe(ref m) if lw.gpu_idx == gpu_idx => Some(m.blob_bytes),
                _ => None,
            })
            .max()
            .unwrap_or(4096);
        let slot_stride = align_up_256(max_blob_bytes.max(4096));
        let max_tiles = MM3_PREFILL_GRP_WIN.div_ceil(32) + n_exp;
        let has_local_moe = weights
            .layers
            .iter()
            .any(|lw| lw.gpu_idx == gpu_idx && matches!(lw.ffn, Mm3FfnWeights::Moe(_)));

        let apart_floats = MM3_MAX_VERIFY
            * cfg.n_kv_heads
            * (MM3_ATTN_SPLITS as usize)
            * (cfg.n_heads / cfg.n_kv_heads)
            * mm3::ATTN_PART_STRIDE;

        let local_tap_indices: Vec<(usize, usize)> = if let Some(ref ds) = weights.dspark {
            ds.cfg
                .target_layer_ids
                .iter()
                .enumerate()
                .filter_map(|(j, &il)| {
                    if !is_head_gpu && weights.layers.get(il).is_some_and(|l| l.gpu_idx == gpu_idx) {
                        Some((il, j))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let taps_dev = if is_head_gpu {
            weights
                .dspark
                .as_ref()
                .map(|ds| {
                    DeviceBuffer::alloc(device, ds.cfg.target_layer_ids.len() * max_t * n_embd * 4)
                })
                .transpose()?
        } else {
            None
        };
        let local_taps_dev = if !local_tap_indices.is_empty() {
            Some(DeviceBuffer::alloc(
                device,
                local_tap_indices.len() * max_t * n_embd * 4,
            )?)
        } else {
            None
        };

        let plan_bytes = std::mem::size_of::<mm3::Mm3DecodePlanC>();
        Ok(Self {
            device,
            is_head_gpu,
            stream: CudaStream::new(device)?,
            side_stream: CudaStream::new(device)?,
            copy_stream: CudaStream::new(device)?,
            d2h_stream: CudaStream::new(device)?,
            norm_ev: CudaEvent::new(device)?,
            idx_done_ev: CudaEvent::new(device)?,
            router_ev: CudaEvent::new(device)?,
            d2h_ev: CudaEvent::new(device)?,
            handoff_ev: CudaEvent::new(device)?,
            p2p_done_ev: CudaEvent::new(device)?,
            promo_gate_ev: CudaEvent::new(device)?,
            taps_ready_ev: CudaEvent::new(device)?,
            dma_events: [CudaEvent::new(device)?, CudaEvent::new(device)?],
            compute_events: [CudaEvent::new(device)?, CudaEvent::new(device)?],
            meta_events: [
                CudaEvent::new(device)?,
                CudaEvent::new(device)?,
                CudaEvent::new(device)?,
                CudaEvent::new(device)?,
            ],

            h: DeviceBuffer::alloc(device, max_t * n_embd * 4)?,
            xn: DeviceBuffer::alloc(device, max_t * n_embd * 4)?,
            block_out: DeviceBuffer::alloc(device, max_t * n_embd * 4)?,
            sh_out: DeviceBuffer::alloc(device, max_t * n_embd * 4)?,
            x_q8_1: DeviceBuffer::alloc(device, max_t * q8_1_x_rb)?,
            x_q8_k: DeviceBuffer::alloc(device, max_t * q8_k_x_rb)?,

            q: DeviceBuffer::alloc(device, max_t * q_dim * 4)?,
            k: DeviceBuffer::alloc(device, max_t * kv_dim * 4)?,
            v: DeviceBuffer::alloc(device, max_t * kv_dim * 4)?,
            attn_out: DeviceBuffer::alloc(device, max_t * q_dim * 4)?,
            attn_q8_1: DeviceBuffer::alloc(device, max_t * q8_1_q_rb)?,
            iq: DeviceBuffer::alloc(device, max_t * cfg.ix_heads * cfg.ix_head_dim * 4)?,
            ik: DeviceBuffer::alloc(device, max_t * cfg.ix_head_dim * 4)?,
            idx_scores: DeviceBuffer::alloc(device, cfg.ix_heads * max_t * max_blocks * 4)?,
            idx_kept: DeviceBuffer::alloc(device, cfg.ix_heads * max_t * cfg.ix_top_k * 4)?,
            attn_part: DeviceBuffer::alloc(device, apart_floats * 4)?,
            pos_dev: DeviceBuffer::alloc(device, max_t * 4)?,

            mlp_gate: DeviceBuffer::alloc(device, max_t * max_ff * 4)?,
            mlp_up: DeviceBuffer::alloc(device, max_t * max_ff * 4)?,
            mlp_inter_q8_1: DeviceBuffer::alloc(device, max_t * q8_1_ff_rb)?,

            router_logits: DeviceBuffer::alloc(device, max_t * n_exp * 4)?,
            router_ids: DeviceBuffer::alloc(device, max_t * topk * 4)?,
            router_wts: DeviceBuffer::alloc(device, max_t * topk * 4)?,
            moe_inter_f32: DeviceBuffer::alloc(device, MM3_MAX_VERIFY * topk * cfg.expert_ff * 4)?,
            moe_inter_q8k: DeviceBuffer::alloc(device, MM3_MAX_VERIFY * topk * q8_k_exp_rb)?,
            moe_expert_out: DeviceBuffer::alloc(device, MM3_MAX_VERIFY * topk * n_embd * 4)?,
            decode_plan_dev: [
                DeviceBuffer::alloc(device, plan_bytes)?,
                DeviceBuffer::alloc(device, plan_bytes)?,
            ],
            h_decode_plan: [
                PinnedHostBuffer::alloc(plan_bytes)?,
                PinnedHostBuffer::alloc(plan_bytes)?,
            ],
            decode_dma_cursor: 0,

            moe_grp_gate: DeviceBuffer::alloc(
                device,
                if has_local_moe {
                    MM3_PREFILL_GRP_WIN * cfg.expert_ff * 4
                } else {
                    16
                },
            )?,
            moe_grp_up: DeviceBuffer::alloc(
                device,
                if has_local_moe {
                    MM3_PREFILL_GRP_WIN * cfg.expert_ff * 4
                } else {
                    16
                },
            )?,
            moe_grp_inter_q8_1: DeviceBuffer::alloc(
                device,
                if has_local_moe {
                    MM3_PREFILL_GRP_WIN * (cfg.expert_ff / 32) * 36
                } else {
                    16
                },
            )?,
            grp_rows_dev: DeviceBuffer::alloc(device, max_t * topk * 4)?,
            grp_wts_dev: DeviceBuffer::alloc(device, max_t * topk * 4)?,
            meta_slots: [
                Mm3MetaSlot::alloc(device, n_exp, max_tiles)?,
                Mm3MetaSlot::alloc(device, n_exp, max_tiles)?,
                Mm3MetaSlot::alloc(device, n_exp, max_tiles)?,
                Mm3MetaSlot::alloc(device, n_exp, max_tiles)?,
            ],
            dma_stage: DeviceBuffer::alloc(
                device,
                if has_local_moe {
                    MM3_DMA_RING * MM3_DMA_BATCH * slot_stride
                } else {
                    16
                },
            )?,
            slot_stride,
            max_tiles,

            logits: is_head_gpu
                .then(|| DeviceBuffer::alloc(device, MM3_MAX_VERIFY * cfg.vocab_size * 4))
                .transpose()?,
            argmax_vals: is_head_gpu
                .then(|| DeviceBuffer::alloc(device, 512 * 4))
                .transpose()?,
            argmax_ids: is_head_gpu
                .then(|| DeviceBuffer::alloc(device, 512 * 4))
                .transpose()?,
            argmax_out: is_head_gpu
                .then(|| DeviceBuffer::alloc(device, MM3_MAX_VERIFY * 4))
                .transpose()?,
            h_argmax: is_head_gpu
                .then(|| PinnedHostBuffer::alloc(MM3_MAX_VERIFY * 4))
                .transpose()?,

            taps_dev,
            local_taps_dev,
            local_tap_indices,

            h_pos: PinnedHostBuffer::alloc(max_t * 4)?,
            h_ids: PinnedHostBuffer::alloc(max_t * topk * 4)?,
            h_wts: PinnedHostBuffer::alloc(max_t * topk * 4)?,
            h_x_q8k: PinnedHostBuffer::alloc(max_t * q8_k_x_rb)?,
            h_cpu_out: PinnedHostBuffer::alloc(max_t * n_embd * 4)?,
            h_cpu_out_mapped: {
                set_device(device)?;
                PinnedMappedBuffer::alloc(max_t * n_embd)?
            },
            h_cpu_wait_mapped: {
                let mut w = PinnedMappedBuffer::alloc(1)?;
                w.as_mut_slice()[0] = 0;
                w
            },
            cpu_wait_seq: 0,
            h_grp_rows: PinnedHostBuffer::alloc(max_t * topk * 4)?,
            h_grp_wts: PinnedHostBuffer::alloc(max_t * topk * 4)?,
        })
    }
}

pub struct Mm3GpuPromoState {
    pub n_slots: usize,
    pub slot_owner: Vec<Option<(usize, usize)>>,
    pub slot_used_fwd: Vec<u64>,
    pub slot_pend: Vec<Option<(usize, usize)>>,
    pub slot_ev: Vec<CudaEvent>,
    pub promo_queue: Vec<(usize, usize, usize)>,
}

impl Mm3GpuPromoState {
    pub fn empty() -> Self {
        Self {
            n_slots: 0,
            slot_owner: Vec::new(),
            slot_used_fwd: Vec::new(),
            slot_pend: Vec::new(),
            slot_ev: Vec::new(),
            promo_queue: Vec::new(),
        }
    }

    pub fn new(device: i32, n_slots: usize) -> Result<Self> {
        set_device(device)?;
        let mut slot_ev = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slot_ev.push(CudaEvent::new(device)?);
        }
        Ok(Self {
            n_slots,
            slot_owner: vec![None; n_slots],
            slot_used_fwd: vec![0u64; n_slots],
            slot_pend: vec![None; n_slots],
            slot_ev,
            promo_queue: Vec::with_capacity(8),
        })
    }
}

#[derive(Default)]
pub struct Mm3ProfStats {
    pub steps: u64,
    pub tokens: u64,
    pub attn_us: f64,
    pub moe_d2h_us: f64,
    pub moe_cpu_us: f64,
    pub handoff_us: f64,
    pub lm_head_us: f64,
    pub dspark_prop_us: f64,
    pub promo_sync_us: f64,
    pub static_hits: u64,
    pub promo_hits: u64,
    pub staged_exps: u64,
    pub cpu_exps: u64,
    pub promos_issued: u64,
}

impl Mm3ProfStats {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

pub struct Mm3State {
    pub max_ctx: usize,
    pub pos: usize,
    pub history: Vec<u32>,
    pub kv: Vec<Mm3LayerKv>,
    pub gpus: Vec<Mm3GpuScratch>,
    pub pool: Arc<CpuThreadPool>,
    pub pipe_pools: Option<(Arc<CpuThreadPool>, Arc<CpuThreadPool>)>,
    pub cpu_scratch: Mm3CpuScratch,
    pub cpu_scratch_pipe1: Option<Mm3CpuScratch>,
    pub emb_pin: PinnedHostBuffer,
    pub xfer_h_pin: PinnedHostBuffer,
    pub taps_xfer_pin: Option<PinnedHostBuffer>,
    pub logits_host: Vec<f32>,
    pub profile_counts: Vec<Vec<u32>>,
    pub cache_stats: CacheStats,
    pub fwd_id: u64,
    pub use_score: Vec<f32>,
    pub use_last_fwd: Vec<u64>,
    pub promo: Vec<Mm3GpuPromoState>,
    pub stage_split: bool,
    pub prof_enabled: bool,
    pub prof: Mm3ProfStats,
}

impl Mm3State {
    pub fn alloc(
        weights: &Mm3Weights,
        engine_cfg: &EngineConfig,
        pool: Arc<CpuThreadPool>,
    ) -> Result<Self> {
        let max_ctx = engine_cfg.max_ctx;
        let cfg = &weights.cfg;
        let kv = Mm3LayerKv::alloc(weights, max_ctx)?;

        let mut gpus = Vec::with_capacity(weights.gpu_ids.len());
        for (gidx, &dev) in weights.gpu_ids.iter().enumerate() {
            gpus.push(Mm3GpuScratch::alloc(dev, gidx, weights, max_ctx)?);
        }

        let max_local_taps = gpus
            .iter()
            .map(|g| g.local_tap_indices.len())
            .max()
            .unwrap_or(0);
        let taps_xfer_pin = if max_local_taps > 0 {
            Some(PinnedHostBuffer::alloc(
                max_local_taps * MM3_SUB * cfg.n_embd * 4,
            )?)
        } else {
            None
        };

        let cpu_scratch = Mm3CpuScratch::new(
            cfg.n_embd,
            cfg.expert_ff,
            MM3_SUB * cfg.n_experts_used,
        );
        let (pipe_pools, cpu_scratch_pipe1) = if weights.gpu_ids.len() >= 2 {
            let (c0, c1) = discover_cross_ccd_interleaved_cores();
            (
                Some((
                    Arc::new(CpuThreadPool::from_cores(c0)),
                    Arc::new(CpuThreadPool::from_cores(c1)),
                )),
                Some(Mm3CpuScratch::new(
                    cfg.n_embd,
                    cfg.expert_ff,
                    MM3_SUB * cfg.n_experts_used,
                )),
            )
        } else {
            (None, None)
        };
        let promo = (0..weights.gpu_ids.len())
            .map(|_| Mm3GpuPromoState::empty())
            .collect();
        let stage_split = std::env::var("AMPLLM_MM3_STAGE_SPLIT").as_deref() != Ok("0");
        let prof_enabled = std::env::var("AMPLLM_MM3_PROF").as_deref() == Ok("1");

        Ok(Self {
            max_ctx,
            pos: 0,
            history: Vec::with_capacity(max_ctx),
            kv,
            gpus,
            pool,
            pipe_pools,
            cpu_scratch,
            cpu_scratch_pipe1,
            emb_pin: PinnedHostBuffer::alloc(MM3_SUB * cfg.n_embd * 4)?,
            xfer_h_pin: PinnedHostBuffer::alloc(MM3_SUB * cfg.n_embd * 4)?,
            taps_xfer_pin,
            logits_host: vec![0.0f32; MM3_MAX_VERIFY * cfg.vocab_size],
            profile_counts: vec![vec![0u32; cfg.n_experts]; cfg.n_layers],
            cache_stats: CacheStats::default(),
            fwd_id: 0,
            use_score: vec![0.0f32; cfg.n_layers * cfg.n_experts],
            use_last_fwd: vec![0u64; cfg.n_layers * cfg.n_experts],
            promo,
            stage_split,
            prof_enabled,
            prof: Mm3ProfStats::default(),
        })
    }

    pub fn init_promo_from_weights(&mut self, weights: &Mm3Weights) -> Result<()> {
        self.promo.clear();
        for (gidx, &dev) in weights.gpu_ids.iter().enumerate() {
            let n_slots = weights.promo_slot_counts.get(gidx).copied().unwrap_or(0);
            self.promo.push(Mm3GpuPromoState::new(dev, n_slots)?);
        }
        Ok(())
    }

    pub fn sync_all(&self) -> Result<()> {
        for g in &self.gpus {
            set_device(g.device)?;
            g.stream.sync()?;
            g.side_stream.sync()?;
            g.copy_stream.sync()?;
            g.d2h_stream.sync()?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.sync_all()?;
        for k in &self.kv {
            k.zero_all()?;
        }
        self.pos = 0;
        self.history.clear();
        self.cache_stats = CacheStats::default();
        self.prof.reset();
        Ok(())
    }
}
