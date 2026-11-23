use crate::hy4::weights::{align256, Hy4ModelWeights};
use ampllm_cpu_kernels::{CpuThreadPool, Glm5CpuScratch};
use ampllm_cuda_kernels::{hy4, set_device, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use anyhow::Result;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

#[inline]
pub fn hy4_profile_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("AMPLLM_HY4_PROFILE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[derive(Default, Clone, Debug)]
pub struct Hy4ProfileStats {
    pub engram_ms: f64,
    pub routes_wait_ms: f64,
    pub cpu_moe_ms: f64,
    pub gpu_moe_ms: f64,
    pub p2p_ms: f64,
    pub mtp_draft_ms: f64,
    pub trunk_ms: f64,
    pub n_cpu_experts: u64,
    pub n_vram_experts: u64,
    pub n_rounds: u64,
}

impl Hy4ProfileStats {
    pub fn print(&self, label: &str) {
        let tot_exp = (self.n_cpu_experts + self.n_vram_experts).max(1) as f64;
        let cpu_pct = 100.0 * (self.n_cpu_experts as f64) / tot_exp;
        eprintln!(
            "PROFILE [{label}]: rounds={} | trunk={:.1}ms engram={:.1}ms routes_wait={:.1}ms cpu_moe={:.1}ms gpu_tail/moe={:.1}ms p2p={:.1}ms mtp_draft={:.1}ms | experts cpu={}/{} ({:.1}%)",
            self.n_rounds,
            self.trunk_ms,
            self.engram_ms,
            self.routes_wait_ms,
            self.cpu_moe_ms,
            self.gpu_moe_ms,
            self.p2p_ms,
            self.mtp_draft_ms,
            self.n_cpu_experts,
            self.n_cpu_experts + self.n_vram_experts,
            cpu_pct,
        );
    }
}

pub const HY4_W_TOKEN: usize = 16;
pub const HY4_ATTN_SUB: usize = 512;
pub const HY4_K_SUB: usize = 64;
pub const HY4_SMALL_MOE_CAP: usize = 32;
pub const HY4_DMA_RING_GROUPS: usize = 3;
pub const HY4_DMA_GROUP_CAP: usize = 8;
pub const HY4_DMA_SLOTS: usize = HY4_DMA_RING_GROUPS * HY4_DMA_GROUP_CAP;
pub const HY4_PREFILL_GRP_WIN: usize = 4096;
pub const HY4_META_SLOTS: usize = 4;
pub const HY4_META_MAX_EXPS: usize = 512;
pub const HY4_META_MAX_TILES: usize = 2048;

#[derive(Clone, Copy)]
pub struct Hy4ScratchLayout {
    pub sub: usize,
    pub s_x: usize,
    pub s_pos: usize,
    pub s_lo: usize,
    pub s_nk: usize,
    pub s_keys: usize,
    pub s_knk: usize,
    pub s_hcur: usize,
    pub s_logits: usize,
    pub s_top1: usize,
    pub s_cur: usize,
    pub s_xa: usize,
    pub s_y: usize,
    pub s_post: usize,
    pub s_hcp: usize,
    pub s_qa: usize,
    pub s_qr: usize,
    pub s_q: usize,
    pub s_qcat: usize,
    pub s_kvr: usize,
    pub s_o: usize,
    pub s_ob: usize,
    pub s_g: usize,
    pub s_ws: usize,
    pub s_iq: usize,
    pub s_ik: usize,
    pub s_ivt: usize,
    pub s_wt: usize,
    pub s_ixsc: usize,
    pub s_egr: usize,
    pub s_egv: usize,
    pub s_egsc: usize,
    pub s_egrm: usize,
    pub s_egkvo: usize,
    pub s_gu: usize,
    pub s_hd: usize,
    pub s_rl: usize,
    pub s_ids: usize,
    pub s_rw: usize,
    pub s_lat: usize,
    pub s_latout: usize,
    pub s_cpo: usize,
    pub s_plan: usize,
    pub s_iq8: usize,
    pub s_pgu: usize,
    pub s_pgq: usize,
    pub shared_phase_bytes: usize,
    pub s_xq8: usize,
    pub s_xf16: usize,
    pub s_wf16: usize,
    pub s_pr: usize,
    pub s_pw: usize,
    pub s_ep: usize,
    pub s_so: usize,
    pub s_sn: usize,
    pub s_gp: usize,
    pub s_go: usize,
    pub s_gn: usize,
    pub s_te: usize,
    pub s_tm: usize,
    pub s_dp: usize,
    pub total_arena: usize,
}

impl Hy4ScratchLayout {
    pub fn compute(cfg: &ampllm_gguf::Hy4Config, tc: usize, max_ctx: usize, h_max: usize) -> Self {
        let sub = HY4_ATTN_SUB.min(tc);
        let e = cfg.n_embd;
        let hc = cfg.hc;
        let maxk = cfg.ix_top_k + cfg.ix_kpool;
        let kv_row = cfg.kv_row();
        let ff_dense = cfg.n_ff.max(cfg.n_ff_exp * cfg.n_expert_shared);
        let eg_cols = cfg.engram.as_ref().map_or(1, |c| c.cols());
        let eg_hd = cfg.engram.as_ref().map_or(1, |c| c.head_dim);
        let eg_dim = eg_cols * eg_hd;
        let n_pools = max_ctx / cfg.ix_kpool + 1;
        let q8_ff = (cfg.n_ff_exp / 32) * 36;
        let max_q8_k_sub = (h_max * cfg.kv_lora_rank)
            .max(e)
            .max(2 * e)
            .max(cfg.q_lora_rank)
            .max(eg_dim)
            .max(ff_dense)
            .max(h_max * cfg.v_mla);
        let max_q8_k_full = e.max(cfg.n_expert_latent);

        let s_x = align256(tc * hc * e * 4);
        let s_pos = align256(tc * 4);
        let s_lo = align256(tc * 4);
        let s_nk = align256(tc * 4);
        let s_keys = align256(sub * maxk * 4);
        let s_knk = align256(sub * 4);
        let s_hcur = align256(tc * e * 4);
        let s_logits = align256(HY4_W_TOKEN * cfg.n_vocab * 4);
        let s_top1 = align256(hy4::HY4_TOP1_SCRATCH_FLOATS * 4);
        let s_cur = align256(tc * e * 4);
        let s_xa = align256(tc * e * 4);
        let s_y = align256(tc * e * 4);
        let s_post = align256(tc * hc * 4);
        let s_hcp = align256(hy4::hc_part_floats(tc) * 4);

        // Phase A (Attention + Indexer): sub-chunked to `HY4_ATTN_SUB` tokens!
        let s_qa = align256(sub * cfg.q_lora_rank * 4);
        let s_qr = align256(sub * cfg.q_lora_rank * 4);
        let s_q = align256(sub * h_max * cfg.k_mla * 4);
        let s_qcat = align256(sub * h_max * kv_row * 4);
        let s_kvr = align256(sub * kv_row * 4);
        let s_o = align256(sub * h_max * cfg.kv_lora_rank * 4);
        let s_ob = align256(sub * h_max * cfg.v_mla * 4);
        let s_g = align256(sub * h_max * cfg.v_mla * 4);
        let s_ws = align256(hy4::mla_decode_ws_floats(HY4_W_TOKEN, h_max, maxk) * 4);
        let s_iq = align256(sub * cfg.ix_heads * cfg.ix_dim * 4);
        let s_ik = align256(sub * cfg.ix_dim * 4);
        let s_ivt = align256(sub * 2 * cfg.ix_dim * 4);
        let s_wt = align256(sub * cfg.ix_heads * 4);
        let s_ixsc = align256(HY4_K_SUB * n_pools * 4);
        let phase_a_bytes = s_qa
            + s_qr
            + s_q
            + s_qcat
            + s_kvr
            + s_o
            + s_ob
            + s_g
            + s_ws
            + s_iq
            + s_ik
            + s_ivt
            + s_wt
            + s_ixsc;

        // Phase B: Engram/Dense/Shared sub-chunked (`sub`), routed MoE full (`tc`)
        let s_egr = align256(sub * eg_dim);
        let s_egv = align256(sub * 4);
        let s_egsc = align256(sub * eg_cols * 4);
        let s_egrm = align256(sub * eg_dim * 4);
        let s_egkvo = align256(sub * (hc + 1) * e * 4);
        let s_gu = align256(2 * sub * ff_dense * 4);
        let s_hd = align256(sub * ff_dense * 4);
        let s_rl = align256(tc * cfg.n_expert * 4);
        let s_ids = align256(tc * cfg.n_expert_used * 4);
        let s_rw = align256(tc * cfg.n_expert_used * 4);
        let s_lat = align256(tc * cfg.n_expert_latent * 4);
        let s_latout = align256(tc * cfg.n_expert_latent * 4);
        let s_cpo = align256(tc * cfg.n_expert_latent * 4);
        let s_plan = align256(HY4_W_TOKEN * cfg.n_expert_used * 8);
        let s_iq8 = align256(HY4_W_TOKEN * cfg.n_expert_used * q8_ff);
        let s_pgu = align256(HY4_PREFILL_GRP_WIN * 2 * cfg.n_ff_exp * 4);
        let max_small_rows = (cfg.n_expert * HY4_SMALL_MOE_CAP).min(tc * cfg.n_expert_used);
        let s_pgq = align256(HY4_PREFILL_GRP_WIN.max(max_small_rows) * q8_ff);
        let phase_b_bytes = s_egr
            + s_egv
            + s_egsc
            + s_egrm
            + s_egkvo
            + s_gu
            + s_hd
            + s_rl
            + s_ids
            + s_rw
            + s_lat
            + s_latout
            + s_cpo
            + s_plan
            + s_iq8
            + s_pgu
            + s_pgq;

        let shared_phase_bytes = phase_a_bytes.max(phase_b_bytes);

        let x_q8_elems = (sub * max_q8_k_sub).max(tc * max_q8_k_full);
        let s_xq8 = align256((x_q8_elems / 32) * 36);
        let s_xf16 = align256(tc * e.max(cfg.n_expert_latent) * 2);
        let s_wf16 = align256((cfg.n_expert * e).max(e * cfg.q_lora_rank) * 2);
        let s_pr = align256(tc * cfg.n_expert_used * 4);
        let s_pw = align256(tc * cfg.n_expert_used * 4);
        let s_ep = align256(cfg.n_expert * 8);
        let s_so = align256(cfg.n_expert * 4);
        let s_sn = align256(cfg.n_expert * 4);
        let s_gp = align256(HY4_META_SLOTS * HY4_META_MAX_EXPS * 8);
        let s_go = align256(HY4_META_SLOTS * HY4_META_MAX_EXPS * 4);
        let s_gn = align256(HY4_META_SLOTS * HY4_META_MAX_EXPS * 4);
        let s_te = align256(HY4_META_SLOTS * HY4_META_MAX_TILES * 4);
        let s_tm = align256(HY4_META_SLOTS * HY4_META_MAX_TILES * 4);
        let s_dp = align256(HY4_DMA_SLOTS * 8);

        let total_arena = (s_x
            + s_pos
            + s_lo
            + s_nk
            + s_keys
            + s_knk
            + s_hcur
            + s_logits
            + s_top1
            + s_cur
            + s_xa
            + s_y
            + s_post
            + s_hcp
            + shared_phase_bytes
            + s_xq8
            + s_xf16
            + s_wf16
            + s_pr
            + s_pw
            + s_ep
            + s_so
            + s_sn
            + s_gp
            + s_go
            + s_gn
            + s_te
            + s_tm
            + s_dp)
            .max(384usize << 20);

        Self {
            sub,
            s_x,
            s_pos,
            s_lo,
            s_nk,
            s_keys,
            s_knk,
            s_hcur,
            s_logits,
            s_top1,
            s_cur,
            s_xa,
            s_y,
            s_post,
            s_hcp,
            s_qa,
            s_qr,
            s_q,
            s_qcat,
            s_kvr,
            s_o,
            s_ob,
            s_g,
            s_ws,
            s_iq,
            s_ik,
            s_ivt,
            s_wt,
            s_ixsc,
            s_egr,
            s_egv,
            s_egsc,
            s_egrm,
            s_egkvo,
            s_gu,
            s_hd,
            s_rl,
            s_ids,
            s_rw,
            s_lat,
            s_latout,
            s_cpo,
            s_plan,
            s_iq8,
            s_pgu,
            s_pgq,
            shared_phase_bytes,
            s_xq8,
            s_xf16,
            s_wf16,
            s_pr,
            s_pw,
            s_ep,
            s_so,
            s_sn,
            s_gp,
            s_go,
            s_gn,
            s_te,
            s_tm,
            s_dp,
            total_arena,
        }
    }
}

pub struct Hy4LayerCache {
    pub kv: DeviceBuffer,
    pub ivg: Option<DeviceBuffer>,
    pub pooled: Option<DeviceBuffer>,
}

pub struct Hy4GpuScratch {
    pub device: i32,
    pub tc: usize,
    pub stream: CudaStream,
    pub copy_stream: CudaStream,
    pub d2h_stream: CudaStream,
    pub ev_routes_ready: CudaEvent,
    pub ev_h2d_group_done: [CudaEvent; HY4_DMA_RING_GROUPS],
    pub ev_comp_group_free: [CudaEvent; HY4_DMA_RING_GROUPS],
    pub ev_d2h_x: CudaEvent,
    pub ev_p2p_ready: CudaEvent,
    /// Single contiguous GPU scratch arena (also lent directly to `Hy4Vision::encode_image` on GPU 0 when idle!).
    pub arena: DeviceBuffer,
    /// Double/triple-buffered DMA expert staging slots (`HY4_DMA_SLOTS * max_blob_bytes`).
    pub dma_stage: Option<DeviceBuffer>,
    // Views inside `arena`
    pub x: *mut f32,
    pub pos: *mut i32,
    pub lo: *mut i32,
    pub nk: *mut i32,
    pub keys: *mut i32,
    pub knk: *mut i32,
    pub hcur: *mut f32,
    pub logits: *mut f32,
    pub top1_scratch: *mut f32,
    pub cur: *mut f32,
    pub xa: *mut f32,
    pub y: *mut f32,
    pub post: *mut f32,
    pub hc_part: *mut f32,
    // Attention / Indexer views (Phase A)
    pub qa: *mut f32,
    pub qr: *mut f32,
    pub q: *mut f32,
    pub qcat: *mut f32,
    pub kvr: *mut f32,
    pub o: *mut f32,
    pub ob: *mut f32,
    pub g: *mut f32,
    pub mla_ws: *mut f32,
    pub iq: *mut f32,
    pub ik: *mut f32,
    pub ivt: *mut f32,
    pub wt: *mut f32,
    pub ix_score: *mut f32,
    // Engram / FFN / MoE views (Phase B, shares arena segment with Phase A)
    pub eg_rows_u8: *mut u8,
    pub eg_valid: *mut f32,
    pub eg_sc: *mut f32,
    pub eg_r: *mut f32,
    pub eg_kvo: *mut f32,
    pub gu: *mut f32,
    pub hd: *mut f32,
    pub rl: *mut f32,
    pub ids: *mut i32,
    pub rw: *mut f32,
    pub lat: *mut f32,
    pub latout: *mut f32,
    pub cpo: *mut f32,
    pub plan_ptrs: *mut u64,
    pub dec_inter_q8: *mut c_void,
    pub prefill_gu_f32: *mut f32,
    pub prefill_grp_q8: *mut c_void,
    // Shared scratch at tail of arena
    pub x_q8: *mut c_void,
    pub x_f16_tmp: *mut c_void,
    pub w_f16_tmp: *mut c_void,
    pub prefill_rows_dev: *mut i32,
    pub prefill_wts_dev: *mut f32,
    pub prefill_exp_ptrs_dev: *mut u64,
    pub prefill_slot_off_dev: *mut i32,
    pub prefill_slot_nt_dev: *mut i32,
    pub prefill_grp_ptrs_dev: *mut u64,
    pub prefill_grp_off_dev: *mut i32,
    pub prefill_grp_nt_dev: *mut i32,
    pub prefill_tile_exp_dev: *mut i32,
    pub prefill_tile_moff_dev: *mut i32,
    pub dma_ptrs_dev: *mut u64,
    // Pinned host staging for prefill grouped MoE metadata (async H2D without overwriting in-flight slots)
    pub ev_meta_done: [CudaEvent; HY4_META_SLOTS],
    pub meta_pin: PinnedHostBuffer,
    pub small_pin: PinnedHostBuffer,
    pub dma_slot_ptrs: Vec<u64>,
}

unsafe impl Send for Hy4GpuScratch {}
unsafe impl Sync for Hy4GpuScratch {}

impl Hy4GpuScratch {
    fn alloc_for_gpu(weights: &Hy4ModelWeights, gpu_idx: usize) -> Result<Self> {
        let cfg = &weights.cfg;
        let dev = weights.gpus[gpu_idx];
        set_device(dev)?;
        let tc = weights.tc;
        let h_max = cfg.heads.iter().copied().max().unwrap_or(64);
        let l = Hy4ScratchLayout::compute(cfg, tc, weights.max_ctx, h_max);

        let arena = DeviceBuffer::alloc_zeroed(dev, l.total_arena)?;
        let base = arena.ptr as *mut u8;
        let mut off = 0usize;
        let take = |off: &mut usize, sz: usize| -> *mut u8 {
            let p = unsafe { base.add(*off) };
            *off += sz;
            p
        };

        let x = take(&mut off, l.s_x).cast();
        let pos = take(&mut off, l.s_pos).cast();
        let lo = take(&mut off, l.s_lo).cast();
        let nk = take(&mut off, l.s_nk).cast();
        let keys = take(&mut off, l.s_keys).cast();
        let knk = take(&mut off, l.s_knk).cast();
        let hcur = take(&mut off, l.s_hcur).cast();
        let logits = take(&mut off, l.s_logits).cast();
        let top1_scratch = take(&mut off, l.s_top1).cast();
        let cur = take(&mut off, l.s_cur).cast();
        let xa = take(&mut off, l.s_xa).cast();
        let y = take(&mut off, l.s_y).cast();
        let post = take(&mut off, l.s_post).cast();
        let hc_part = take(&mut off, l.s_hcp).cast();

        let shared_start = off;
        let qa = take(&mut off, l.s_qa).cast();
        let qr = take(&mut off, l.s_qr).cast();
        let q = take(&mut off, l.s_q).cast();
        let qcat = take(&mut off, l.s_qcat).cast();
        let kvr = take(&mut off, l.s_kvr).cast();
        let o = take(&mut off, l.s_o).cast();
        let ob = take(&mut off, l.s_ob).cast();
        let g = take(&mut off, l.s_g).cast();
        let mla_ws = take(&mut off, l.s_ws).cast();
        let iq = take(&mut off, l.s_iq).cast();
        let ik = take(&mut off, l.s_ik).cast();
        let ivt = take(&mut off, l.s_ivt).cast();
        let wt = take(&mut off, l.s_wt).cast();
        let ix_score = take(&mut off, l.s_ixsc).cast();

        off = shared_start;
        let eg_rows_u8 = take(&mut off, l.s_egr).cast();
        let eg_valid = take(&mut off, l.s_egv).cast();
        let eg_sc = take(&mut off, l.s_egsc).cast();
        let eg_r = take(&mut off, l.s_egrm).cast();
        let eg_kvo = take(&mut off, l.s_egkvo).cast();
        let gu = take(&mut off, l.s_gu).cast();
        let hd = take(&mut off, l.s_hd).cast();
        let rl = take(&mut off, l.s_rl).cast();
        let ids = take(&mut off, l.s_ids).cast();
        let rw = take(&mut off, l.s_rw).cast();
        let lat = take(&mut off, l.s_lat).cast();
        let latout = take(&mut off, l.s_latout).cast();
        let cpo = take(&mut off, l.s_cpo).cast();
        let plan_ptrs = take(&mut off, l.s_plan).cast();
        let dec_inter_q8 = take(&mut off, l.s_iq8).cast();
        let prefill_gu_f32 = take(&mut off, l.s_pgu).cast();
        let prefill_grp_q8 = take(&mut off, l.s_pgq).cast();

        off = shared_start + l.shared_phase_bytes;
        let x_q8 = take(&mut off, l.s_xq8).cast();
        let x_f16_tmp = take(&mut off, l.s_xf16).cast();
        let w_f16_tmp = take(&mut off, l.s_wf16).cast();
        let prefill_rows_dev = take(&mut off, l.s_pr).cast();
        let prefill_wts_dev = take(&mut off, l.s_pw).cast();
        let prefill_exp_ptrs_dev = take(&mut off, l.s_ep).cast();
        let prefill_slot_off_dev = take(&mut off, l.s_so).cast();
        let prefill_slot_nt_dev = take(&mut off, l.s_sn).cast();
        let prefill_grp_ptrs_dev = take(&mut off, l.s_gp).cast();
        let prefill_grp_off_dev = take(&mut off, l.s_go).cast();
        let prefill_grp_nt_dev = take(&mut off, l.s_gn).cast();
        let prefill_tile_exp_dev = take(&mut off, l.s_te).cast();
        let prefill_tile_moff_dev = take(&mut off, l.s_tm).cast();
        let dma_ptrs_dev = take(&mut off, l.s_dp).cast::<u64>();

        let has_partial = weights
            .layers
            .iter()
            .any(|l| l.gpu_idx == gpu_idx && l.moe.as_ref().is_some_and(|m| m.is_partial));
        let mut dma_slot_ptrs = Vec::new();
        let dma_stage = if has_partial && weights.max_blob_bytes > 0 {
            let db = DeviceBuffer::alloc(dev, HY4_DMA_SLOTS * weights.max_blob_bytes)?;
            for i in 0..HY4_DMA_SLOTS {
                dma_slot_ptrs.push(db.as_u64() + (i * weights.max_blob_bytes) as u64);
            }
            let st_tmp = CudaStream::new(dev)?;
            unsafe {
                ampllm_cuda_kernels::copy_h2d_raw_async(
                    dma_ptrs_dev.cast(),
                    bytemuck::cast_slice(&dma_slot_ptrs),
                    &st_tmp,
                )?;
            }
            st_tmp.sync()?;
            Some(db)
        } else {
            None
        };

        let stream = CudaStream::new(dev)?;
        let copy_stream = CudaStream::new(dev)?;
        let d2h_stream = CudaStream::new(dev)?;
        let ev_routes_ready = CudaEvent::new(dev)?;
        let ev_d2h_x = CudaEvent::new(dev)?;
        let ev_p2p_ready = CudaEvent::new(dev)?;
        let ev_h2d_group_done = [CudaEvent::new(dev)?, CudaEvent::new(dev)?, CudaEvent::new(dev)?];
        let ev_comp_group_free = [CudaEvent::new(dev)?, CudaEvent::new(dev)?, CudaEvent::new(dev)?];
        let ev_meta_done = [
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
            CudaEvent::new(dev)?,
        ];
        // Per-slot meta bytes: grp_ptrs[512]*8 + grp_off[512]*4 + grp_nt[512]*4 + te[2048]*4 + tm[2048]*4
        let one_meta_bytes = HY4_META_MAX_EXPS * 16 + HY4_META_MAX_TILES * 8;
        let meta_pin = PinnedHostBuffer::alloc(HY4_META_SLOTS * one_meta_bytes)?;
        // Pinned small-job + rows/wts staging: 2 * tc * topk * 4 + n_expert * (8 + 4 + 4)
        let small_pin_bytes = tc * cfg.n_expert_used * 8 + cfg.n_expert * 16;
        let small_pin = PinnedHostBuffer::alloc(small_pin_bytes)?;

        Ok(Self {
            device: dev,
            tc,
            stream,
            copy_stream,
            d2h_stream,
            ev_routes_ready,
            ev_d2h_x,
            ev_p2p_ready,
            ev_h2d_group_done,
            ev_comp_group_free,
            arena,
            dma_stage,
            x,
            pos,
            lo,
            nk,
            keys,
            knk,
            hcur,
            logits,
            top1_scratch,
            cur,
            xa,
            y,
            post,
            hc_part,
            qa,
            qr,
            q,
            qcat,
            kvr,
            o,
            ob,
            g,
            mla_ws,
            iq,
            ik,
            ivt,
            wt,
            ix_score,
            eg_rows_u8,
            eg_valid,
            eg_sc,
            eg_r,
            eg_kvo,
            gu,
            hd,
            rl,
            ids,
            rw,
            lat,
            latout,
            cpo,
            plan_ptrs,
            dec_inter_q8,
            prefill_gu_f32,
            prefill_grp_q8,
            x_q8,
            x_f16_tmp,
            w_f16_tmp,
            prefill_rows_dev,
            prefill_wts_dev,
            prefill_exp_ptrs_dev,
            prefill_slot_off_dev,
            prefill_slot_nt_dev,
            prefill_grp_ptrs_dev,
            prefill_grp_off_dev,
            prefill_grp_nt_dev,
            prefill_tile_exp_dev,
            prefill_tile_moff_dev,
            dma_ptrs_dev,
            ev_meta_done,
            meta_pin,
            small_pin,
            dma_slot_ptrs,
        })
    }
}

pub struct Hy4State {
    pub sc: Vec<Hy4GpuScratch>,
    pub caches: Vec<Hy4LayerCache>,
    pub mtp_cache: Option<Hy4LayerCache>,
    // Head & MTP state buffers (on head GPU)
    pub hring: DeviceBuffer,
    pub mtp_in: Option<DeviceBuffer>,
    pub mtp_cur: Option<DeviceBuffer>,
    pub mtp_hin: Option<DeviceBuffer>,
    pub mtp_out: Option<DeviceBuffer>,
    // Pinned host staging (CPU WB cacheable!)
    pub h_x: PinnedHostBuffer,
    pub h_pos: PinnedHostBuffer,
    pub h_eg: PinnedHostBuffer,
    pub h_valid: PinnedHostBuffer,
    pub h_egs: Option<PinnedHostBuffer>,
    pub h_ids: PinnedHostBuffer,
    pub h_w: PinnedHostBuffer,
    pub h_lat: PinnedHostBuffer,
    pub h_cpu: PinnedHostBuffer,
    pub h_logits: PinnedHostBuffer,
    pub h_emb_mtp: PinnedHostBuffer,
    pub h_top1: PinnedHostBuffer,
    // CPU execution runtime
    pub cpu_pool: Arc<CpuThreadPool>,
    pub cpu_scratch: Glm5CpuScratch,
    // Sequence tracking
    pub end_pos: usize,
    pub mtp_done: usize,
    pub history: Vec<u32>,
    pub profile: Hy4ProfileStats,
}

impl Hy4State {
    pub fn alloc(weights: &Hy4ModelWeights, cpu_pool: Arc<CpuThreadPool>) -> Result<Self> {
        let cfg = &weights.cfg;
        let mut sc = Vec::with_capacity(weights.gpus.len());
        for gi in 0..weights.gpus.len() {
            sc.push(Hy4GpuScratch::alloc_for_gpu(weights, gi)?);
        }

        let alloc_cache = |dev: i32, swa: bool, ix: bool| -> Result<Hy4LayerCache> {
            set_device(dev)?;
            let kv_rows = if swa {
                weights.swa_ring_cap
            } else {
                weights.max_ctx
            };
            let kv = DeviceBuffer::alloc_zeroed(dev, kv_rows * weights.kv_rb)?;
            let (ivg, pooled) = if ix {
                let pe = if weights.kv_is_q8 { 2 } else { 4 };
                let ivg_b = weights.ivg_ring_cap * 2 * cfg.ix_dim * pe;
                let pool_b = (weights.max_ctx / cfg.ix_kpool + 1) * cfg.ix_dim * pe;
                (
                    Some(DeviceBuffer::alloc_zeroed(dev, ivg_b)?),
                    Some(DeviceBuffer::alloc_zeroed(dev, pool_b)?),
                )
            } else {
                (None, None)
            };
            Ok(Hy4LayerCache { kv, ivg, pooled })
        };

        let mut caches = Vec::with_capacity(weights.layers.len());
        for lw in &weights.layers {
            caches.push(alloc_cache(lw.device, lw.swa, lw.ix)?);
        }
        let mtp_cache = if let Some(ref ml) = weights.mtp_layer {
            Some(alloc_cache(ml.device, ml.swa, ml.ix)?)
        } else {
            None
        };

        let hdev = weights.head.device;
        set_device(hdev)?;
        let hring = DeviceBuffer::alloc_zeroed(hdev, weights.hring_cap * cfg.n_embd * 4)?;
        let (mtp_in, mtp_cur, mtp_hin, mtp_out) = if weights.mtp_layer.is_some() {
            let e = cfg.n_embd;
            let sub = HY4_ATTN_SUB.min(weights.tc);
            (
                Some(DeviceBuffer::alloc_zeroed(hdev, sub * 2 * e * 4)?),
                Some(DeviceBuffer::alloc_zeroed(hdev, sub * e * 4)?),
                Some(DeviceBuffer::alloc_zeroed(hdev, sub * e * 4)?),
                Some(DeviceBuffer::alloc_zeroed(hdev, sub * e * 4)?),
            )
        } else {
            (None, None, None, None)
        };

        let tc = weights.tc;
        let sub = HY4_ATTN_SUB.min(tc);
        let e = cfg.n_embd;
        let n_eg_mod = cfg.engram.as_ref().map_or(1, |c| c.layers.len().max(1));
        let eg_cols = cfg.engram.as_ref().map_or(1, |c| c.cols());
        let eg_hd = cfg.engram.as_ref().map_or(1, |c| c.head_dim);
        let has_eg_scales = weights
            .engram_table
            .as_ref()
            .is_some_and(|t| t.scales_mmap.is_some());

        let h_x = PinnedHostBuffer::alloc(tc * e * 4)?;
        let h_pos = PinnedHostBuffer::alloc(3 * tc * 4)?;
        let h_eg = PinnedHostBuffer::alloc(n_eg_mod * tc * eg_cols * eg_hd)?;
        let h_valid = PinnedHostBuffer::alloc(tc * 4)?;
        let h_egs = if has_eg_scales {
            Some(PinnedHostBuffer::alloc(n_eg_mod * tc * eg_cols * 4)?)
        } else {
            None
        };
        let h_ids = PinnedHostBuffer::alloc(tc * cfg.n_expert_used * 4)?;
        let h_w = PinnedHostBuffer::alloc(tc * cfg.n_expert_used * 4)?;
        let h_lat = PinnedHostBuffer::alloc(tc * cfg.n_expert_latent * 4)?;
        let h_cpu = PinnedHostBuffer::alloc(tc * cfg.n_expert_latent * 4)?;
        let h_logits = PinnedHostBuffer::alloc(HY4_W_TOKEN * cfg.n_vocab * 4)?;
        let h_emb_mtp = PinnedHostBuffer::alloc(sub * e * 4)?;
        let h_top1 = PinnedHostBuffer::alloc(2 * 4)?;

        let cpu_scratch = Glm5CpuScratch::new(cfg.n_expert_latent, cfg.n_ff_exp, cfg.n_expert_used);

        Ok(Self {
            sc,
            caches,
            mtp_cache,
            hring,
            mtp_in,
            mtp_cur,
            mtp_hin,
            mtp_out,
            h_x,
            h_pos,
            h_eg,
            h_valid,
            h_egs,
            h_ids,
            h_w,
            h_lat,
            h_cpu,
            h_logits,
            h_emb_mtp,
            h_top1,
            cpu_pool,
            cpu_scratch,
            end_pos: 0,
            mtp_done: 0,
            history: Vec::with_capacity(weights.max_ctx),
            profile: Hy4ProfileStats::default(),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.end_pos = 0;
        self.mtp_done = 0;
        self.history.clear();
        self.profile = Hy4ProfileStats::default();
        for c in &self.caches {
            set_device(c.kv.device)?;
            c.kv.zero()?;
            if let Some(ref ivg) = c.ivg {
                ivg.zero()?;
            }
            if let Some(ref pooled) = c.pooled {
                pooled.zero()?;
            }
        }
        if let Some(ref mc) = self.mtp_cache {
            set_device(mc.kv.device)?;
            mc.kv.zero()?;
            if let Some(ref ivg) = mc.ivg {
                ivg.zero()?;
            }
            if let Some(ref pooled) = mc.pooled {
                pooled.zero()?;
            }
        }
        set_device(self.hring.device)?;
        self.hring.zero()?;
        Ok(())
    }
}
