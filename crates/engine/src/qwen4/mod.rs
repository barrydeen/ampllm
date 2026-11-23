pub mod forward;
pub mod mtp;
pub mod ple;
pub mod state;
pub mod vision;
pub mod weights;

pub use vision::Qwen4ImageSpan;

use crate::config::EngineConfig;
use crate::inkling2::StreamGenerateResult;
use crate::qwen4::forward::{
    forward_decode_verify_step, forward_prefill_pipelined, rollback_verify, Qwen4StepProfile,
};
use crate::snapshot::{
    d2h_full, d2h_prefix, h2d_bytes, EngineSnapshotData, QuantumGenerateResult, QuantumOptions,
    Qwen4GdnSnap, Qwen4LayerSnap, Qwen4MtpSnap, Qwen4QsaSnap, Qwen4Snapshot, SequenceSnapshot,
};
use ampllm_cache::CacheStats;
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::{
    copy_d2h_raw_async, copy_h2d_raw_async, glm5, qwen4, set_device, CudaStream, PinnedHostBuffer,
};
use ampllm_gguf::{GgufFile, Qwen4Config, Qwen4VisionConfig};
use anyhow::{bail, Result};
use ple::PleEngine;
use state::{AdaptSwap, AsyncAdaptState, GpuStageScratch, Qwen4State};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use weights::{LayerExpertsStorage, Qwen4Weights};

pub(crate) fn patch_adapt_ptrs_for_gpu(
    g: &mut GpuStageScratch,
    pairs: &[(u64, u64)],
) -> Result<()> {
    if pairs.is_empty() {
        return Ok(());
    }
    set_device(g.device)?;
    let cap = g.adapt_patch_pin.as_slice().len() / 2;
    anyhow::ensure!(cap > 0, "Qwen4 adapt_patch_pin empty");
    for chunk in pairs.chunks(cap) {
        if g.adapt_patch_cursor + chunk.len() > cap {
            g.adapt_patch_cursor = 0;
        }
        let start = g.adapt_patch_cursor;
        {
            let dst =
                &mut g.adapt_patch_pin.as_mut_slice()[start * 2..(start + chunk.len()) * 2];
            for (i, &(addr, val)) in chunk.iter().enumerate() {
                dst[i * 2] = addr;
                dst[i * 2 + 1] = val;
            }
        }
        unsafe {
            qwen4::patch_u64(
                (g.adapt_patch_dev as *const u64).add(start * 2),
                chunk.len() as i32,
                &g.stream,
            );
        }
        g.adapt_patch_cursor = start + chunk.len();
    }
    Ok(())
}

fn commit_adapt_h2d_state(
    weights: &mut Qwen4Weights,
    state: &mut Qwen4State,
    swaps: Vec<AdaptSwap>,
) -> Result<()> {
    let n_gpus = state.gpus.len();
    let mut gpu_pairs: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n_gpus];
    for sw in &swaps {
        let es = &mut weights.experts[sw.layer];
        es.slot_of[sw.in_e] = sw.slot as i32;
        es.expert_of_slot[sw.slot] = sw.in_e;
        es.is_resident[sw.in_e] = true;
        es.vram_ptrs_host[sw.in_e] = sw.v_ptr;
        es.ram_host_ptrs[sw.in_e] = 0;
        es.ram_dev_ptrs[sw.in_e] = 0;
        es.free_ram_slots.push((sw.old_in_h, sw.old_in_d));
        weights.layers_host_atomic[sw.layer][sw.in_e].store(0, Ordering::Release);
        gpu_pairs[sw.gidx].push((
            es.dev_vram_ptrs.ptr as u64 + (sw.in_e * 8) as u64,
            sw.v_ptr,
        ));
        gpu_pairs[sw.gidx].push((es.dev_ram_ptrs.ptr as u64 + (sw.in_e * 8) as u64, 0));
    }
    for (gidx, g) in state.gpus.iter_mut().enumerate() {
        if !gpu_pairs[gidx].is_empty() {
            patch_adapt_ptrs_for_gpu(g, &gpu_pairs[gidx])?;
        }
    }
    state.cache_stats.promotions += swaps.len() as u64;
    Ok(())
}

pub(crate) fn poll_async_adapt_state(
    weights: &mut Qwen4Weights,
    state: &mut Qwen4State,
) -> Result<()> {
    let cur = std::mem::replace(&mut state.adapt_state, AsyncAdaptState::Idle);
    match cur {
        AsyncAdaptState::Idle => Ok(()),
        AsyncAdaptState::D2hInFlight { swaps, gpu_mask } => {
            for (gidx, g) in state.gpus.iter().enumerate() {
                if (gpu_mask & (1u32 << gidx)) != 0 {
                    set_device(g.device)?;
                    if !g.adapt_d2h_ev.is_ready()? {
                        state.adapt_state = AsyncAdaptState::D2hInFlight { swaps, gpu_mask };
                        return Ok(());
                    }
                }
            }
            let n_gpus = state.gpus.len();
            let mut gpu_pairs: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n_gpus];
            for sw in &swaps {
                let es = &mut weights.experts[sw.layer];
                es.slot_of[sw.ev_e] = -1;
                es.is_resident[sw.ev_e] = false;
                es.vram_ptrs_host[sw.ev_e] = 0;
                es.ram_host_ptrs[sw.ev_e] = sw.free_h;
                es.ram_dev_ptrs[sw.ev_e] = sw.free_d;
                weights.layers_host_atomic[sw.layer][sw.ev_e].store(sw.free_h, Ordering::Release);
                gpu_pairs[sw.gidx].push((es.dev_vram_ptrs.ptr as u64 + (sw.ev_e * 8) as u64, 0));
                gpu_pairs[sw.gidx].push((
                    es.dev_ram_ptrs.ptr as u64 + (sw.ev_e * 8) as u64,
                    sw.free_d,
                ));
            }
            for (gidx, g) in state.gpus.iter_mut().enumerate() {
                if !gpu_pairs[gidx].is_empty() {
                    patch_adapt_ptrs_for_gpu(g, &gpu_pairs[gidx])?;
                }
                if (gpu_mask & (1u32 << gidx)) != 0 {
                    set_device(g.device)?;
                    g.adapt_d2h_ev.record(&g.stream)?;
                    g.copy_stream.wait_event(&g.adapt_d2h_ev)?;
                }
            }
            for sw in &swaps {
                let g = &state.gpus[sw.gidx];
                set_device(g.device)?;
                let src = unsafe {
                    std::slice::from_raw_parts(sw.old_in_h as *const u8, sw.blob_bytes)
                };
                unsafe {
                    copy_h2d_raw_async(sw.v_ptr as *mut c_void, src, &g.copy_stream)?;
                }
            }
            for (gidx, g) in state.gpus.iter().enumerate() {
                if (gpu_mask & (1u32 << gidx)) != 0 {
                    set_device(g.device)?;
                    g.adapt_h2d_ev.record(&g.copy_stream)?;
                }
            }
            state.adapt_state = AsyncAdaptState::H2dInFlight { swaps, gpu_mask };
            Ok(())
        }
        AsyncAdaptState::H2dInFlight { swaps, gpu_mask } => {
            for (gidx, g) in state.gpus.iter().enumerate() {
                if (gpu_mask & (1u32 << gidx)) != 0 {
                    set_device(g.device)?;
                    if !g.adapt_h2d_ev.is_ready()? {
                        state.adapt_state = AsyncAdaptState::H2dInFlight { swaps, gpu_mask };
                        return Ok(());
                    }
                }
            }
            commit_adapt_h2d_state(weights, state, swaps)
        }
    }
}

pub(crate) fn drain_async_adapt_state(
    weights: &mut Qwen4Weights,
    state: &mut Qwen4State,
) -> Result<()> {
    let cur = std::mem::replace(&mut state.adapt_state, AsyncAdaptState::Idle);
    match cur {
        AsyncAdaptState::Idle => Ok(()),
        AsyncAdaptState::D2hInFlight { swaps, gpu_mask } => {
            for (gidx, g) in state.gpus.iter().enumerate() {
                if (gpu_mask & (1u32 << gidx)) != 0 {
                    set_device(g.device)?;
                    g.adapt_d2h_ev.sync()?;
                }
            }
            for sw in &swaps {
                weights.experts[sw.layer]
                    .free_ram_slots
                    .push((sw.free_h, sw.free_d));
            }
            Ok(())
        }
        AsyncAdaptState::H2dInFlight { swaps, gpu_mask } => {
            for (gidx, g) in state.gpus.iter().enumerate() {
                if (gpu_mask & (1u32 << gidx)) != 0 {
                    set_device(g.device)?;
                    g.adapt_h2d_ev.sync()?;
                }
            }
            commit_adapt_h2d_state(weights, state, swaps)?;
            state.sync_all()
        }
    }
}

pub(crate) fn select_prompt_adapt_swaps(
    has_profile: bool,
    n_exp: usize,
    layer_bbs: &[usize],
    split_offset: usize,
    gidx: usize,
    experts: &mut [LayerExpertsStorage],
    max_per_layer: usize,
    max_total_swaps: usize,
) -> Vec<AdaptSwap> {
    if max_per_layer == 0
        || max_total_swaps == 0
        || std::env::var("AMPLLM_QWEN4_ADAPT").is_ok_and(|v| v == "0")
    {
        return Vec::new();
    }
    let min_score = if has_profile { 4.0 } else { 1.0 };
    let min_diff = if has_profile { 2.0 } else { 0.5 };
    let score = |es: &LayerExpertsStorage, e: usize| -> f32 {
        (es.prompt_tail_hits[e] as f32) * 3.0 + (es.prompt_hits[e] as f32)
    };
    // (gain, cold_score, il, loc_il, ev_e, in_e)
    let mut candidates: Vec<(f32, f32, usize, usize, usize, usize)> = Vec::new();
    for (loc_il, es) in experts.iter().enumerate() {
        if !es.all_pinned_cold
            || es.n_resident == 0
            || es.n_resident >= n_exp
            || es.free_ram_slots.is_empty()
        {
            continue;
        }
        let il = split_offset + loc_il;
        let cap = es.free_ram_slots.len().min(max_per_layer);
        let mut cold: Vec<(f32, u16, usize)> = (0..n_exp)
            .filter(|&e| !es.is_resident[e] && es.ram_host_ptrs[e] != 0)
            .filter_map(|e| {
                let s = score(es, e);
                (s >= min_score).then_some((s, es.profile_rank[e], e))
            })
            .collect();
        if cold.is_empty() {
            continue;
        }
        cold.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut res: Vec<(f32, u16, usize)> = es
            .expert_of_slot
            .iter()
            .copied()
            .map(|e| (score(es, e), es.profile_rank[e], e))
            .collect();
        res.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        let pairs = cold.len().min(res.len()).min(cap);
        for k in 0..pairs {
            let gain = cold[k].0 - res[k].0;
            if gain <= min_diff {
                break;
            }
            candidates.push((gain, cold[k].0, il, loc_il, res[k].2, cold[k].2));
        }
    }
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.5.cmp(&b.5))
    });
    candidates.truncate(max_total_swaps);

    let mut swaps = Vec::with_capacity(candidates.len());
    for (_, _, il, loc_il, ev_e, in_e) in candidates {
        let es = &mut experts[loc_il];
        let Some((free_h, free_d)) = es.free_ram_slots.pop() else {
            continue;
        };
        swaps.push(AdaptSwap {
            layer: il,
            gidx,
            ev_e,
            in_e,
            slot: es.slot_of[ev_e] as usize,
            v_ptr: es.vram_ptrs_host[ev_e],
            free_h,
            free_d,
            old_in_h: es.ram_host_ptrs[in_e],
            old_in_d: es.ram_dev_ptrs[in_e],
            blob_bytes: layer_bbs[il],
        });
    }
    swaps
}

pub(crate) fn promote_gpu_slice_sync(
    swaps: Vec<AdaptSwap>,
    split_offset: usize,
    experts: &mut [LayerExpertsStorage],
    layers_host_atomic: &[Vec<AtomicUsize>],
    g: &mut GpuStageScratch,
) -> Result<u64> {
    if swaps.is_empty() || std::env::var("AMPLLM_QWEN4_ADAPT").is_ok_and(|v| v == "0") {
        return Ok(0);
    }
    set_device(g.device)?;
    g.adapt_h2d_ev.record(&g.stream)?;
    g.d2h_stream.wait_event(&g.adapt_h2d_ev)?;
    for sw in &swaps {
        let dst = unsafe { std::slice::from_raw_parts_mut(sw.free_h as *mut u8, sw.blob_bytes) };
        unsafe {
            copy_d2h_raw_async(dst, sw.v_ptr as *const c_void, &g.d2h_stream)?;
        }
    }
    g.adapt_d2h_ev.record(&g.d2h_stream)?;
    g.adapt_d2h_ev.sync()?;

    let mut ev_pairs = Vec::with_capacity(swaps.len() * 2);
    for sw in &swaps {
        let loc = sw.layer - split_offset;
        let es = &mut experts[loc];
        es.slot_of[sw.ev_e] = -1;
        es.is_resident[sw.ev_e] = false;
        es.vram_ptrs_host[sw.ev_e] = 0;
        es.ram_host_ptrs[sw.ev_e] = sw.free_h;
        es.ram_dev_ptrs[sw.ev_e] = sw.free_d;
        layers_host_atomic[sw.layer][sw.ev_e].store(sw.free_h, Ordering::Release);
        ev_pairs.push((es.dev_vram_ptrs.ptr as u64 + (sw.ev_e * 8) as u64, 0));
        ev_pairs.push((
            es.dev_ram_ptrs.ptr as u64 + (sw.ev_e * 8) as u64,
            sw.free_d,
        ));
    }
    patch_adapt_ptrs_for_gpu(g, &ev_pairs)?;
    g.adapt_d2h_ev.record(&g.stream)?;
    g.copy_stream.wait_event(&g.adapt_d2h_ev)?;

    for sw in &swaps {
        let src = unsafe { std::slice::from_raw_parts(sw.old_in_h as *const u8, sw.blob_bytes) };
        unsafe {
            copy_h2d_raw_async(sw.v_ptr as *mut c_void, src, &g.copy_stream)?;
        }
    }
    g.adapt_h2d_ev.record(&g.copy_stream)?;
    g.adapt_h2d_ev.sync()?;

    let mut in_pairs = Vec::with_capacity(swaps.len() * 2);
    for sw in &swaps {
        let loc = sw.layer - split_offset;
        let es = &mut experts[loc];
        es.slot_of[sw.in_e] = sw.slot as i32;
        es.expert_of_slot[sw.slot] = sw.in_e;
        es.is_resident[sw.in_e] = true;
        es.vram_ptrs_host[sw.in_e] = sw.v_ptr;
        es.ram_host_ptrs[sw.in_e] = 0;
        es.ram_dev_ptrs[sw.in_e] = 0;
        es.free_ram_slots.push((sw.old_in_h, sw.old_in_d));
        layers_host_atomic[sw.layer][sw.in_e].store(0, Ordering::Release);
        in_pairs.push((
            es.dev_vram_ptrs.ptr as u64 + (sw.in_e * 8) as u64,
            sw.v_ptr,
        ));
        in_pairs.push((es.dev_ram_ptrs.ptr as u64 + (sw.in_e * 8) as u64, 0));
    }
    patch_adapt_ptrs_for_gpu(g, &in_pairs)?;
    g.stream.sync()?;
    Ok(swaps.len() as u64)
}

pub(crate) fn start_prompt_adapt_gpu1(
    has_profile: bool,
    n_exp: usize,
    layer_bbs: &[usize],
    experts_tail: &mut [LayerExpertsStorage],
    state: &mut Qwen4State,
    split: usize,
    max_per_layer: usize,
) -> Result<usize> {
    if max_per_layer == 0
        || !matches!(state.adapt_state, AsyncAdaptState::Idle)
        || state.gpus.len() < 2
        || std::env::var("AMPLLM_QWEN4_ADAPT").is_ok_and(|v| v == "0")
    {
        return Ok(0);
    }
    let max_total_swaps = std::env::var("AMPLLM_QWEN4_PROMPT_ADAPT_GPU1_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(if has_profile { 0 } else { 192 });
    let swaps = select_prompt_adapt_swaps(
        has_profile,
        n_exp,
        layer_bbs,
        split,
        1,
        experts_tail,
        max_per_layer,
        max_total_swaps,
    );
    if swaps.is_empty() {
        return Ok(0);
    }
    let g = &state.gpus[1];
    set_device(g.device)?;
    g.adapt_h2d_ev.record(&g.stream)?;
    g.d2h_stream.wait_event(&g.adapt_h2d_ev)?;
    for sw in &swaps {
        let dst = unsafe { std::slice::from_raw_parts_mut(sw.free_h as *mut u8, sw.blob_bytes) };
        unsafe {
            copy_d2h_raw_async(dst, sw.v_ptr as *const c_void, &g.d2h_stream)?;
        }
    }
    g.adapt_d2h_ev.record(&g.d2h_stream)?;
    let n = swaps.len();
    state.adapt_state = AsyncAdaptState::D2hInFlight {
        swaps,
        gpu_mask: 1u32 << 1,
    };
    Ok(n)
}

pub struct Qwen4Engine {
    pub cfg: Qwen4Config,
    pub max_ctx: usize,
    pub committed_tokens: Vec<u32>,
    pub prefix_reuse_tokens_total: u64,
    weights: Qwen4Weights,
    state: Qwen4State,
    ple: Option<PleEngine>,
    pool: Arc<CpuThreadPool>,
    mtp_k: usize,
    mtp_k_adaptive: bool,
    spec_min_p: f32,
    save_profile_path: Option<PathBuf>,
    profile_enabled: bool,
    pub draft_n_total: usize,
    pub draft_accepted_total: usize,
}

impl Drop for Qwen4Engine {
    fn drop(&mut self) {
        if self.save_profile_path.is_some() {
            let _ = self.save_profile();
        }
    }
}

impl Qwen4Engine {
    pub fn load(gguf: GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let cfg = Qwen4Config::from_gguf(&gguf)?;
        let pool = Arc::new(CpuThreadPool::new(engine_cfg.cpu_threads));
        let mut weights = Qwen4Weights::load_dense(gguf, cfg.clone(), &engine_cfg)?;

        let env_mtp = std::env::var("AMPLLM_QWEN4_MTP_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let spec_min_p = std::env::var("AMPLLM_QWEN4_SPEC_MIN_P")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let (mtp_k, mtp_k_adaptive) = if !engine_cfg.use_spec || weights.mtp.is_none() {
            (0, false)
        } else if let Some(k) = env_mtp {
            (k.min(7), false)
        } else {
            (4, spec_min_p <= 0.0)
        };

        let state = Qwen4State::alloc(&weights, &engine_cfg, Arc::clone(&pool), mtp_k)?;
        weights.allocate_experts(&engine_cfg, &pool)?;

        let ple = if let Some(ref pcfg) = cfg.ple {
            Some(PleEngine::new(pcfg.clone(), Arc::clone(&pool))?)
        } else {
            None
        };

        let profile_enabled = std::env::var("AMPLLM_QWEN4_PROFILE")
            .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));

        let res_total: usize = weights.experts.iter().map(|e| e.n_resident).sum();
        let exp_total = cfg.n_layer * cfg.n_expert;
        let all_pinned = weights.experts.iter().all(|e| e.all_pinned_cold);
        eprintln!(
            "qwen4exp loaded: gpus={:?} split={} ctx={} chunk={} mtp_k={} (adapt={}, min_p={:.2}) graph={} vision={} experts={}/{} ({:.1}%) pinned_cold={}",
            weights.gpu_ids,
            weights.split_layer,
            state.max_ctx,
            state.max_t,
            mtp_k,
            mtp_k_adaptive,
            spec_min_p,
            state.use_cuda_graph,
            weights.vision.is_some(),
            res_total,
            exp_total,
            100.0 * (res_total as f64) / (exp_total.max(1) as f64),
            all_pinned,
        );

        let mut engine = Self {
            cfg,
            max_ctx: state.max_ctx,
            committed_tokens: Vec::new(),
            prefix_reuse_tokens_total: 0,
            weights,
            state,
            ple,
            pool,
            mtp_k,
            mtp_k_adaptive,
            spec_min_p,
            save_profile_path: engine_cfg.save_profile.clone(),
            profile_enabled,
            draft_n_total: 0,
            draft_accepted_total: 0,
        };
        if engine.state.use_cuda_graph
            && std::env::var("AMPLLM_QWEN4_PRECAPTURE").map_or(true, |v| v != "0")
        {
            engine.precapture_decode_graphs()?;
        }
        Ok(engine)
    }

    pub fn precapture_decode_graphs(&mut self) -> Result<()> {
        if !self.state.use_cuda_graph {
            return Ok(());
        }
        let saved_heat: Vec<Vec<f32>> = self
            .weights
            .experts
            .iter()
            .map(|e| e.heat.clone())
            .collect();
        let sparse_pos = self.state.max_idx_cap;

        let _ = forward_decode_verify_step(
            &mut self.weights,
            &mut self.state,
            self.ple.as_ref(),
            &[0u32],
            0,
            false,
            true,
            None,
        )?;
        if sparse_pos + 1 <= self.state.max_ctx {
            let _ = forward_decode_verify_step(
                &mut self.weights,
                &mut self.state,
                self.ple.as_ref(),
                &[0u32],
                sparse_pos,
                false,
                true,
                None,
            )?;
        }

        if self.weights.mtp.is_some() && self.state.mtp.is_some() {
            let max_v = self.state.max_verify.min(6);
            for t in 2..=max_v {
                let toks = vec![0u32; t];
                let _ = forward_decode_verify_step(
                    &mut self.weights,
                    &mut self.state,
                    self.ple.as_ref(),
                    &toks,
                    0,
                    true,
                    true,
                    None,
                )?;
                if sparse_pos + t <= self.state.max_ctx {
                    let _ = forward_decode_verify_step(
                        &mut self.weights,
                        &mut self.state,
                        self.ple.as_ref(),
                        &toks,
                        sparse_pos,
                        true,
                        true,
                        None,
                    )?;
                }
            }
            let r_last_ptr = self.state.r_last.ptr.cast::<f32>();
            let r_mtp_ptr = self.state.mtp.as_ref().unwrap().r_mtp.ptr.cast::<f32>();
            for &r_in in &[r_last_ptr, r_mtp_ptr] {
                let _ = mtp::mtp_draft_step(
                    &self.weights,
                    &mut self.state,
                    &self.pool,
                    0,
                    r_in,
                    0,
                    true,
                )?;
                if self.state.max_ctx > 2048 {
                    let _ = mtp::mtp_draft_step(
                        &self.weights,
                        &mut self.state,
                        &self.pool,
                        0,
                        r_in,
                        2048,
                        true,
                    )?;
                }
            }
            let dummy_acc = vec![0u32; max_v];
            for acc in 0..max_v.saturating_sub(1) {
                rollback_verify(&self.weights, &mut self.state, &dummy_acc, acc, 0)?;
            }
            let hg_dev = self.state.gpus.last().unwrap().device;
            set_device(hg_dev)?;
            for nb in 1..=max_v.saturating_sub(1) {
                mtp::mtp_prefill_kv(
                    &self.weights,
                    &mut self.state,
                    &self.pool,
                    &dummy_acc[..nb],
                    r_last_ptr,
                    0,
                )?;
                self.state.gpus.last().unwrap().stream.sync()?;
            }
        }

        self.state.sync_all()?;
        self.reset()?;
        self.reset_cache_stats();
        for (es, h) in self.weights.experts.iter_mut().zip(saved_heat) {
            es.heat = h;
            es.decode_usage.fill(0.0);
            es.prompt_hits.fill(0);
            es.prompt_tail_hits.fill(0);
            es.last_used.fill(0);
        }
        self.state.step_counter = 0;
        Ok(())
    }

    pub fn has_vision(&self) -> bool {
        self.weights.vision.is_some()
    }

    pub fn vision_config(&self) -> Option<&Qwen4VisionConfig> {
        self.weights.vision.as_ref().map(|v| &v.cfg)
    }

    pub fn set_mtp_k(&mut self, k: usize) {
        self.mtp_k = if self.weights.mtp.is_some() {
            k.min(7)
        } else {
            0
        };
        self.mtp_k_adaptive = false;
    }

    pub fn set_adaptive_mtp(&mut self, enabled: bool) {
        if self.weights.mtp.is_some()
            && std::env::var("AMPLLM_QWEN4_MTP_K").is_err()
        {
            self.mtp_k_adaptive = enabled && self.spec_min_p <= 0.0;
        }
    }

    pub fn mtp_k(&self) -> usize {
        self.mtp_k
    }

    pub fn reset_cache_stats(&mut self) {
        self.state.cache_stats = CacheStats::default();
        self.state.decode_cache_stats = CacheStats::default();
    }

    pub fn encode_image(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<(usize, usize, Vec<f32>)> {
        let Some(ref vis) = self.weights.vision else {
            bail!("Qwen4 model has no mmproj vision encoder loaded");
        };
        let stream = CudaStream::new(vis.device)?;
        vis.encode_image(rgb, width, height, &stream)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.drain_async_adapt()?;
        self.committed_tokens.clear();
        self.state.reset()
    }

    pub fn capture_snapshot(&mut self, has_images: bool) -> Result<SequenceSnapshot> {
        self.drain_async_adapt()?;
        self.state.sync_all()?;
        let pos = self.state.pos;
        anyhow::ensure!(
            self.committed_tokens.len() == pos,
            "Qwen4 snapshot invariant violated: committed={} != pos={pos}",
            self.committed_tokens.len()
        );
        let cfg = &self.cfg;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let q_blocks = (pos / 4) + 1;
        let mut layers = Vec::with_capacity(self.state.layers.len());
        let mut byte_len = 0usize;
        for ls in &self.state.layers {
            match ls {
                state::Qwen4LayerState::Gdn(g) => {
                    let st = d2h_full(&g.state)?;
                    let ch = d2h_full(&g.conv_hist)?;
                    byte_len += st.len() + ch.len();
                    layers.push(Qwen4LayerSnap::Gdn(Qwen4GdnSnap {
                        state: st,
                        conv_hist: ch,
                    }));
                }
                state::Qwen4LayerState::Qsa(q) => {
                    let k_q = d2h_prefix(&q.k_q, pos * kv_dim)?;
                    let v_q = d2h_prefix(&q.v_q, pos * kv_dim)?;
                    let k_sc = d2h_prefix(&q.k_sc, pos * cfg.n_head_kv * 8)?;
                    let v_sc = d2h_prefix(&q.v_sc, pos * cfg.n_head_kv * 8)?;
                    let tail = d2h_full(&q.tail)?;
                    let dead = d2h_full(&q.dead)?;
                    let pooled = d2h_prefix(&q.pooled, q_blocks * cfg.idx_key_dim * 4)?;
                    byte_len += k_q.len()
                        + v_q.len()
                        + k_sc.len()
                        + v_sc.len()
                        + tail.len()
                        + dead.len()
                        + pooled.len();
                    layers.push(Qwen4LayerSnap::Qsa(Qwen4QsaSnap {
                        k_q,
                        v_q,
                        k_sc,
                        v_sc,
                        tail,
                        dead,
                        pooled,
                    }));
                }
            }
        }
        let ple_history = match self.state.ple {
            Some(ref p) => {
                let h = d2h_full(&p.history)?;
                byte_len += h.len();
                Some(h)
            }
            None => None,
        };
        let r_last_row0 = d2h_prefix(&self.state.r_last, cfg.hc * cfg.n_embd * 4)?;
        byte_len += r_last_row0.len();

        let mtp = match self.state.mtp {
            Some(ref ms) => {
                let kp = ms.kv_pos.min(pos);
                let k_q = d2h_prefix(&ms.k_q, kp * kv_dim)?;
                let v_q = d2h_prefix(&ms.v_q, kp * kv_dim)?;
                let k_sc = d2h_prefix(&ms.k_sc, kp * cfg.n_head_kv * 8)?;
                let v_sc = d2h_prefix(&ms.v_sc, kp * cfg.n_head_kv * 8)?;
                byte_len += k_q.len() + v_q.len() + k_sc.len() + v_sc.len();
                Some(Qwen4MtpSnap {
                    kv_pos: kp,
                    k_q,
                    v_q,
                    k_sc,
                    v_sc,
                })
            }
            None => None,
        };
        let mrope_table_host = if self.state.mrope_active {
            byte_len += self.state.mrope_table_host.len() * 4;
            self.state.mrope_table_host.clone()
        } else {
            Vec::new()
        };

        Ok(SequenceSnapshot {
            tokens: self.committed_tokens.clone(),
            has_images,
            allow_lcp_truncate: false,
            max_safe_rollback_delta: None,
            rollback_align: 1,
            byte_len,
            inner: EngineSnapshotData::Qwen4(Qwen4Snapshot {
                layers,
                ple_history,
                ple_cursor: self.state.ple_cursor,
                r_last_row0,
                mtp,
                mrope_active: self.state.mrope_active,
                mrope_pos: self.state.mrope_pos,
                mrope_table_host,
            }),
        })
    }

    pub fn restore_snapshot(
        &mut self,
        snap: &SequenceSnapshot,
        _truncate_to: Option<usize>,
    ) -> Result<()> {
        let EngineSnapshotData::Qwen4(ref qs) = snap.inner else {
            bail!("expected Qwen4 snapshot");
        };
        self.drain_async_adapt()?;
        self.state.sync_all()?;
        anyhow::ensure!(
            qs.layers.len() == self.state.layers.len(),
            "Qwen4 snapshot layer count mismatch"
        );
        for (dst, src) in self.state.layers.iter().zip(qs.layers.iter()) {
            match (dst, src) {
                (state::Qwen4LayerState::Gdn(g), Qwen4LayerSnap::Gdn(gs)) => {
                    h2d_bytes(&g.state, &gs.state)?;
                    h2d_bytes(&g.conv_hist, &gs.conv_hist)?;
                }
                (state::Qwen4LayerState::Qsa(q), Qwen4LayerSnap::Qsa(qss)) => {
                    h2d_bytes(&q.k_q, &qss.k_q)?;
                    h2d_bytes(&q.v_q, &qss.v_q)?;
                    h2d_bytes(&q.k_sc, &qss.k_sc)?;
                    h2d_bytes(&q.v_sc, &qss.v_sc)?;
                    h2d_bytes(&q.tail, &qss.tail)?;
                    h2d_bytes(&q.dead, &qss.dead)?;
                    h2d_bytes(&q.pooled, &qss.pooled)?;
                }
                _ => bail!("Qwen4 snapshot layer kind mismatch"),
            }
        }
        if let (Some(ref ple), Some(ref ph)) = (&self.state.ple, &qs.ple_history) {
            h2d_bytes(&ple.history, ph)?;
        }
        self.state.ple_cursor = qs.ple_cursor;
        self.state.ple_cursor_snap = None;
        h2d_bytes(&self.state.r_last, &qs.r_last_row0)?;

        if let (Some(ref mut ms), Some(ref ms_snap)) = (&mut self.state.mtp, &qs.mtp) {
            h2d_bytes(&ms.k_q, &ms_snap.k_q)?;
            h2d_bytes(&ms.v_q, &ms_snap.v_q)?;
            h2d_bytes(&ms.k_sc, &ms_snap.k_sc)?;
            h2d_bytes(&ms.v_sc, &ms_snap.v_sc)?;
            ms.kv_pos = ms_snap.kv_pos;
        } else if let Some(ref mut ms) = self.state.mtp {
            ms.kv_pos = 0;
        }

        self.state.pos = snap.tokens.len();
        self.committed_tokens.clear();
        self.committed_tokens.extend_from_slice(&snap.tokens);
        self.state.mrope_active = qs.mrope_active;
        self.state.mrope_pos = qs.mrope_pos;
        if qs.mrope_active {
            self.state.mrope_table_host = qs.mrope_table_host.clone();
            for g in &self.state.gpus {
                set_device(g.device)?;
                unsafe {
                    copy_h2d_raw_async(
                        g.ptr::<c_void>(g.off_mrope_pos3_dev),
                        bytemuck::cast_slice(&self.state.mrope_table_host),
                        &g.stream,
                    )?;
                }
                g.stream.sync()?;
            }
        } else {
            self.state.mrope_table_host.clear();
        }
        Ok(())
    }

    pub fn cache_summary(&self) -> String {
        let res: usize = self.weights.experts.iter().map(|e| e.n_resident).sum();
        let tot = self.cfg.n_layer * self.cfg.n_expert;
        let st = &self.state.cache_stats;
        let lookups = st.total_lookups();
        let hit_pct = if lookups > 0 {
            100.0 * st.vram_hit_rate()
        } else {
            100.0
        };
        let dst = &self.state.decode_cache_stats;
        let dec_lookups = dst.total_lookups();
        if dec_lookups > 0 {
            let dec_pct = 100.0 * dst.vram_hit_rate();
            format!(
                "resident={res}/{tot} vram_hit={hit_pct:.1}% ({}/{}) dec_hit={dec_pct:.1}% ({}/{}) promos={} mtp_k={}",
                st.hits, lookups, dst.hits, dec_lookups, st.promotions, self.mtp_k
            )
        } else {
            format!(
                "resident={res}/{tot} vram_hit={hit_pct:.1}% ({}/{}) promos={} mtp_k={}",
                st.hits, lookups, st.promotions, self.mtp_k
            )
        }
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.state.cache_stats.clone()
    }

    pub fn save_profile(&self) -> Result<()> {
        if let Some(ref path) = self.save_profile_path {
            let prof = self.weights.export_profile();
            prof.save(path)?;
        }
        Ok(())
    }

    fn poll_async_adapt(&mut self) -> Result<()> {
        poll_async_adapt_state(&mut self.weights, &mut self.state)
    }

    fn drain_async_adapt(&mut self) -> Result<()> {
        drain_async_adapt_state(&mut self.weights, &mut self.state)
    }

    fn start_async_adapt(&mut self, max_total_swaps: usize, min_usage: f32) -> Result<usize> {
        if max_total_swaps == 0
            || !matches!(self.state.adapt_state, state::AsyncAdaptState::Idle)
            || std::env::var("AMPLLM_QWEN4_ADAPT").is_ok_and(|v| v == "0")
        {
            return Ok(0);
        }
        let n_layer = self.weights.experts.len();
        let n_exp = self.cfg.n_expert;
        let mut cands: Vec<(f32, usize, usize, usize)> = Vec::new();
        for il in 0..n_layer {
            let es = &self.weights.experts[il];
            if !es.all_pinned_cold
                || es.n_resident == 0
                || es.n_resident >= n_exp
                || es.free_ram_slots.is_empty()
            {
                continue;
            }
            let cap = es.free_ram_slots.len().min(4);
            let mut cold: Vec<(f32, usize)> = (0..n_exp)
                .filter(|&e| !es.is_resident[e] && es.ram_host_ptrs[e] != 0 && es.decode_usage[e] >= min_usage)
                .map(|e| (es.decode_usage[e], e))
                .collect();
            if cold.is_empty() {
                continue;
            }
            cold.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut res: Vec<(f32, u16, usize)> = es
                .expert_of_slot
                .iter()
                .copied()
                .map(|e| (es.decode_usage[e], es.profile_rank[e], e))
                .collect();
            res.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            let pairs = cold.len().min(res.len()).min(cap);
            let min_gain = if self.weights.has_profile { 2.5 } else { 0.25 };
            for k in 0..pairs {
                let gain = cold[k].0 - res[k].0;
                if gain >= min_gain {
                    cands.push((gain, il, res[k].2, cold[k].1));
                }
            }
        }
        if cands.is_empty() {
            return Ok(0);
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        cands.truncate(max_total_swaps);

        let mut swaps = Vec::with_capacity(cands.len());
        let mut gpu_mask = 0u32;
        for &(_, il, ev_e, in_e) in &cands {
            let dev = self.weights.layers[il].device;
            let gidx = self.state.gpu_idx_for_device(dev);
            let bb = self.weights.layers[il].moe.blob_bytes;
            let es = &mut self.weights.experts[il];
            let Some((free_h, free_d)) = es.free_ram_slots.pop() else {
                continue;
            };
            let slot = es.slot_of[ev_e] as usize;
            let v_ptr = es.vram_ptrs_host[ev_e];
            let old_in_h = es.ram_host_ptrs[in_e];
            let old_in_d = es.ram_dev_ptrs[in_e];
            swaps.push(state::AdaptSwap {
                layer: il,
                gidx,
                ev_e,
                in_e,
                slot,
                v_ptr,
                free_h,
                free_d,
                old_in_h,
                old_in_d,
                blob_bytes: bb,
            });
            gpu_mask |= 1u32 << gidx;
        }
        if swaps.is_empty() {
            return Ok(0);
        }
        for sw in &swaps {
            let g = &self.state.gpus[sw.gidx];
            set_device(g.device)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(sw.free_h as *mut u8, sw.blob_bytes) };
            unsafe {
                copy_d2h_raw_async(dst, sw.v_ptr as *const c_void, &g.d2h_stream)?;
            }
        }
        for (gidx, g) in self.state.gpus.iter().enumerate() {
            if (gpu_mask & (1u32 << gidx)) != 0 {
                set_device(g.device)?;
                g.adapt_d2h_ev.record(&g.d2h_stream)?;
            }
        }
        let n = swaps.len();
        self.state.adapt_state = state::AsyncAdaptState::D2hInFlight { swaps, gpu_mask };
        Ok(n)
    }

    fn run_warmup(&mut self, max_swaps: usize, min_heat: f32) -> Result<usize> {
        self.drain_async_adapt()?;
        if max_swaps == 0 || self.state.gpus.is_empty() {
            return Ok(0);
        }
        let swaps = if self.state.gpus.len() >= 2 {
            let (g0_s, g1_s) = self.state.gpus.split_at_mut(1);
            let g0 = &mut g0_s[0];
            let g1 = &mut g1_s[0];
            let streams = [
                (&g0.copy_stream, &g0.d2h_stream, &g0.dma_events, &g0.compute_events),
                (&g1.copy_stream, &g1.d2h_stream, &g1.dma_events, &g1.compute_events),
            ];
            let mut bounces: [&mut PinnedHostBuffer; 2] =
                [&mut g0.warmup_bounce, &mut g1.warmup_bounce];
            self.weights
                .warmup_from_heat(&streams, &mut bounces, max_swaps, min_heat)?
        } else {
            let g0 = &mut self.state.gpus[0];
            let streams = [(&g0.copy_stream, &g0.d2h_stream, &g0.dma_events, &g0.compute_events)];
            let mut bounces: [&mut PinnedHostBuffer; 1] = [&mut g0.warmup_bounce];
            self.weights
                .warmup_from_heat(&streams, &mut bounces, max_swaps, min_heat)?
        };
        self.state.cache_stats.promotions += swaps as u64;
        Ok(swaps)
    }

    fn setup_imrope_for_prompt(
        &mut self,
        prompt_len: usize,
        images: &[Qwen4ImageSpan],
    ) -> Result<()> {
        if images.is_empty() {
            self.state.mrope_active = false;
            self.state.mrope_pos = prompt_len;
            return Ok(());
        }

        let cap = self.state.max_ctx;
        let mut tab = vec![0i32; cap * 3];
        let mut rp = 0i32;
        let mut span_idx = 0usize;
        let mut sorted_spans: Vec<&Qwen4ImageSpan> = images.iter().collect();
        sorted_spans.sort_by_key(|s| s.start_pos);

        let mut c = 0usize;
        while c < cap {
            while span_idx < sorted_spans.len() && sorted_spans[span_idx].start_pos < c {
                span_idx += 1;
            }
            if span_idx < sorted_spans.len() && sorted_spans[span_idx].start_pos == c {
                let sp = sorted_spans[span_idx];
                let count = sp.nx * sp.ny;
                for i in 0..count {
                    if c >= cap {
                        break;
                    }
                    let iy = (i / sp.nx) as i32;
                    let ix = (i % sp.nx) as i32;
                    tab[c * 3 + 0] = rp;
                    tab[c * 3 + 1] = rp + iy;
                    tab[c * 3 + 2] = rp + ix;
                    c += 1;
                }
                rp += (sp.nx.max(sp.ny)) as i32;
                if c == prompt_len {
                    self.state.mrope_pos = rp as usize;
                }
            } else {
                tab[c * 3 + 0] = rp;
                tab[c * 3 + 1] = rp;
                tab[c * 3 + 2] = rp;
                rp += 1;
                c += 1;
                if c == prompt_len {
                    self.state.mrope_pos = rp as usize;
                }
            }
        }

        for g in &self.state.gpus {
            set_device(g.device)?;
            unsafe {
                copy_h2d_raw_async(
                    g.ptr::<c_void>(g.off_mrope_pos3_dev),
                    bytemuck::cast_slice(&tab),
                    &g.stream,
                )?;
            }
            g.stream.sync()?;
        }
        self.state.mrope_active = true;
        self.state.mrope_table_host = tab;
        Ok(())
    }

    fn argmax_on_head(&self) -> Result<u32> {
        let hg = self.state.gpus.last().unwrap();
        set_device(hg.device)?;
        unsafe {
            glm5::argmax(
                hg.ptr::<f32>(hg.off_logits),
                self.cfg.n_vocab as i32,
                self.cfg.n_vocab as i32,
                std::ptr::null(),
                hg.ptr::<f32>(hg.off_argmax_vals),
                hg.ptr::<i32>(hg.off_argmax_ids),
                hg.ptr::<i32>(hg.off_argmax_out),
                &hg.stream,
            );
            copy_d2h_raw_async(
                std::slice::from_raw_parts_mut(hg.small_pin.as_ptr() as *mut u8, 4),
                hg.ptr::<c_void>(hg.off_argmax_out),
                &hg.stream,
            )?;
        }
        hg.stream.sync()?;
        let tok = unsafe { *hg.small_pin.as_ptr().cast::<i32>() };
        Ok(tok as u32)
    }

    fn read_logits_rows(&mut self, n_rows: usize) -> Result<&[f32]> {
        let n_vocab = self.cfg.n_vocab;
        let hg = self.state.gpus.last().unwrap();
        set_device(hg.device)?;
        unsafe {
            copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut self.state.logits_host[..n_rows * n_vocab]),
                hg.ptr::<c_void>(hg.off_logits),
                &hg.stream,
            )?;
        }
        hg.stream.sync()?;
        Ok(&self.state.logits_host[..n_rows * n_vocab])
    }

    /// Evaluate `tokens` from clean state and return final-token `[n_vocab]` FP32 logits (for parity tests).
    pub fn forward_logits_debug(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.reset()?;
        self.setup_imrope_for_prompt(tokens.len(), &[])?;
        forward_prefill_pipelined(
            &mut self.weights,
            &mut self.state,
            self.ple.as_ref(),
            tokens,
            0,
            &[],
            false,
            false,
            None,
        )?;
        self.state.pos = tokens.len();
        self.committed_tokens = tokens.to_vec();
        Ok(self.read_logits_rows(1)?.to_vec())
    }

    /// Evaluate `tokens` one token at a time (`t = 1` decode path) and return final-token `[n_vocab]` FP32 logits.
    pub fn forward_logits_debug_stepwise(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.reset()?;
        self.setup_imrope_for_prompt(tokens.len(), &[])?;
        for i in 0..tokens.len() {
            let _ = forward_decode_verify_step(
                &mut self.weights,
                &mut self.state,
                self.ple.as_ref(),
                &tokens[i..i + 1],
                i,
                false,
                false,
                None,
            )?;
            self.state.pos = i + 1;
            self.committed_tokens.push(tokens[i]);
        }
        Ok(self.read_logits_rows(1)?.to_vec())
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        params: &crate::sampling::SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        self.generate_stream_with_images(prompt_tokens, &[], params, on_token)
    }

    pub fn generate_stream_with_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Qwen4ImageSpan],
        params: &crate::sampling::SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        let mut no_yield = |_: usize| false;
        let qres = self.generate_stream_quantum_with_images(
            prompt_tokens,
            images,
            params,
            QuantumOptions::no_yield(&mut no_yield),
            on_token,
        )?;
        Ok(qres.result)
    }

    pub fn generate_stream_quantum_with_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Qwen4ImageSpan],
        params: &crate::sampling::SamplingParams,
        opts: QuantumOptions<'_>,
        mut on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_tokens.is_empty() {
            bail!("empty prompt");
        }
        if prompt_tokens.len() >= self.max_ctx {
            bail!(
                "prompt length {} exceeds max_ctx {}",
                prompt_tokens.len(),
                self.max_ctx
            );
        }

        let use_mtp = self.mtp_k > 0 && self.weights.mtp.is_some();
        let can_reuse = images.is_empty()
            && !self.committed_tokens.is_empty()
            && self.committed_tokens.len() == self.state.pos
            && prompt_tokens.len() > self.committed_tokens.len()
            && prompt_tokens.starts_with(&self.committed_tokens)
            && (!self.state.mrope_active || opts.resume_active)
            && (!use_mtp
                || self
                    .state
                    .mtp
                    .as_ref()
                    .is_none_or(|ms| ms.kv_pos + 1 >= self.state.pos));
        let lcp = if can_reuse {
            self.committed_tokens.len()
        } else {
            0
        };

        if lcp == 0 {
            self.reset()?;
            self.weights.on_new_prompt();
            self.setup_imrope_for_prompt(prompt_tokens.len(), images)?;
        } else {
            if !opts.resume_active {
                self.prefix_reuse_tokens_total += lcp as u64;
                self.weights.on_new_prompt();
                if self.state.mrope_active {
                    self.state.mrope_pos += prompt_tokens.len() - lcp;
                } else {
                    self.state.mrope_pos = prompt_tokens.len();
                }
            }
        }

        let eval_slice = &prompt_tokens[lcp..];
        let greedy = params.temperature <= 0.0;
        let resuming_decode =
            opts.resume_active && lcp > 0 && eval_slice.len() == 1 && images.is_empty();

        let mut prof_pp = if self.profile_enabled {
            Some(Qwen4StepProfile::default())
        } else {
            None
        };

        let mut rng = params.seed;
        let (first_tok, prompt_tokens_evaluated, pp_secs, prefill_snapshot) = if resuming_decode {
            (eval_slice[0], 0usize, 0.0f64, None)
        } else {
            // 1. Prompt (or suffix) prefill (pipelined on 2 GPUs when multi-chunk + MTP KV direct on g1.r_buf)
            let t_pp0 = Instant::now();
            if lcp > 0 && use_mtp {
                let tbl = self.weights.token_embd_bytes()?;
                let hg = &self.state.gpus[self.state.gpus.len() - 1];
                let mrope_ptr = if self.state.mrope_active {
                    hg.ptr::<i32>(hg.off_mrope_pos3_dev).cast_const()
                } else {
                    std::ptr::null()
                };
                if let (Some(ref mw), Some(ref mut ms)) = (&self.weights.mtp, &mut self.state.mtp) {
                    mtp::mtp_prefill_kv_parts(
                        &self.cfg,
                        mw,
                        (tbl, self.weights.token_embd_row_bytes, self.weights.token_embd_type),
                        ms,
                        &hg.stream,
                        Some((&hg.side_stream, &hg.side_fork_ev, &hg.side_join_ev)),
                        mrope_ptr,
                        &self.pool,
                        &eval_slice[..1],
                        self.state.r_last.ptr.cast::<f32>(),
                        lcp - 1,
                        false,
                        false,
                    )?;
                }
            }

            let plen = eval_slice.len();
            let hits_before = self.state.cache_stats.hits;
            let misses_before = self.state.cache_stats.misses;
            let promos_before = self.state.cache_stats.promotions;
            let split_cut = if opts.capture_prefill_snap
                && images.is_empty()
                && !self.state.mrope_active
                && opts.prefill_snap_trim > 0
                && plen > opts.prefill_snap_trim
            {
                Some(plen - opts.prefill_snap_trim)
            } else {
                None
            };

            let (first_tok_opt, prefill_snap_early) = if let Some(head_len) = split_cut {
                let _ = forward_prefill_pipelined(
                    &mut self.weights,
                    &mut self.state,
                    self.ple.as_ref(),
                    &eval_slice[..head_len],
                    lcp,
                    &[],
                    use_mtp,
                    false,
                    prof_pp.as_mut(),
                )?;
                let cut_pos = lcp + head_len;
                self.state.pos = cut_pos;
                self.state.mrope_pos = cut_pos;
                self.committed_tokens = prompt_tokens[..cut_pos].to_vec();
                let early_snap = Some(self.capture_snapshot(false)?);
                if use_mtp {
                    let tbl = self.weights.token_embd_bytes()?;
                    let hg = &self.state.gpus[self.state.gpus.len() - 1];
                    if let (Some(ref mw), Some(ref mut ms)) =
                        (&self.weights.mtp, &mut self.state.mtp)
                    {
                        mtp::mtp_prefill_kv_parts(
                            &self.cfg,
                            mw,
                            (
                                tbl,
                                self.weights.token_embd_row_bytes,
                                self.weights.token_embd_type,
                            ),
                            ms,
                            &hg.stream,
                            Some((&hg.side_stream, &hg.side_fork_ev, &hg.side_join_ev)),
                            std::ptr::null(),
                            &self.pool,
                            &eval_slice[head_len..head_len + 1],
                            self.state.r_last.ptr.cast::<f32>(),
                            cut_pos - 1,
                            false,
                            false,
                        )?;
                    }
                }
                let tok_opt = forward_prefill_pipelined(
                    &mut self.weights,
                    &mut self.state,
                    self.ple.as_ref(),
                    &eval_slice[head_len..],
                    cut_pos,
                    &[],
                    use_mtp,
                    greedy,
                    prof_pp.as_mut(),
                )?;
                self.state.pos = prompt_tokens.len();
                self.state.mrope_pos = prompt_tokens.len();
                self.committed_tokens = prompt_tokens.to_vec();
                (tok_opt, early_snap)
            } else {
                let tok_opt = forward_prefill_pipelined(
                    &mut self.weights,
                    &mut self.state,
                    self.ple.as_ref(),
                    eval_slice,
                    lcp,
                    images,
                    use_mtp,
                    greedy,
                    prof_pp.as_mut(),
                )?;
                self.state.pos = prompt_tokens.len();
                self.committed_tokens = prompt_tokens.to_vec();
                (tok_opt, None)
            };

            let pp_hits = self.state.cache_stats.hits.saturating_sub(hits_before);
            let pp_misses = self.state.cache_stats.misses.saturating_sub(misses_before);
            let pp_tot = pp_hits + pp_misses;
            let pp_hit_rate = if pp_tot > 0 {
                (pp_hits as f64) / (pp_tot as f64)
            } else {
                1.0
            };

            let mut warmup_ms = 0.0f64;
            let mut warmup_swaps = 0usize;
            let max_swaps = if let Ok(v) = std::env::var("AMPLLM_QWEN4_WARMUP_MAX") {
                v.parse::<usize>().unwrap_or(0)
            } else if !self.weights.has_profile
                && plen >= 512
                && pp_hit_rate < 0.72
                && self.state.cache_stats.promotions == promos_before
                && matches!(self.state.adapt_state, AsyncAdaptState::Idle)
            {
                32
            } else {
                0
            };
            if max_swaps > 0 {
                let t_w = Instant::now();
                warmup_swaps = self.run_warmup(max_swaps, 0.0)?;
                warmup_ms = t_w.elapsed().as_secs_f64() * 1000.0;
            }

            let pp_secs = t_pp0.elapsed().as_secs_f64();
            if let Some(ref p) = prof_pp {
                p.print_summary("PREFILL");
                if max_swaps > 0 {
                    eprintln!(
                        "[QWEN4_PROFILE PREFILL warmup] swaps={warmup_swaps} warmup_ms={warmup_ms:.2}"
                    );
                }
            }

            if !self.weights.has_profile {
                for es in &mut self.weights.experts {
                    for e in 0..self.cfg.n_expert {
                        let th = es.prompt_tail_hits[e];
                        if th > 0 {
                            es.decode_usage[e] += (th as f32) * 0.5;
                        }
                    }
                }
            }

            let first_tok = match first_tok_opt {
                Some(tok) => tok,
                None => {
                    if greedy {
                        self.argmax_on_head()?
                    } else {
                        let logits = self.read_logits_rows(1)?;
                        crate::sampling::sample_next_token(logits, params, &mut rng)
                    }
                }
            };

            let prefill_snapshot = if prefill_snap_early.is_some() {
                prefill_snap_early
            } else if opts.capture_prefill_snap && images.is_empty() {
                Some(self.capture_snapshot(false)?)
            } else {
                None
            };

            (first_tok, plen, pp_secs, prefill_snapshot)
        };

        let adapt_swaps = std::env::var("AMPLLM_QWEN4_ADAPT_SWAPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if self.weights.has_profile { 32 } else { 96 });
        let decode_min_usage = if self.weights.has_profile { 3.5 } else { 1.5 };
        let adapt_every: usize = std::env::var("AMPLLM_QWEN4_ADAPT_EVERY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if self.weights.has_profile { 4 } else { 1 })
            .max(1);
        if !resuming_decode && !self.weights.has_profile && params.max_tokens > 1 {
            let _ = self.start_async_adapt(adapt_swaps, decode_min_usage)?;
        }

        let mut generated = Vec::with_capacity(params.max_tokens.min(4096));
        let eos_id = self.cfg.eos_token_id;
        let is_stop =
            |tok: u32| -> bool { eos_id == Some(tok) || params.stop_token_ids.contains(&tok) };

        let decode_promote = std::env::var("AMPLLM_QWEN4_DECODE_PROMOTE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let mut prof_tg = if self.profile_enabled {
            Some(Qwen4StepProfile::default())
        } else {
            None
        };
        let t_tg0 = Instant::now();
        let mut round_draft_n = 0usize;
        let mut round_draft_acc = 0usize;
        let mut verify_rounds = 0usize;
        let mut acc_ewma = 0.75f64;
        let mut yielded = false;

        let enter_decode = if resuming_decode {
            params.max_tokens > 0
        } else if params.max_tokens > 0 {
            generated.push(first_tok);
            let keep_going = on_token(first_tok);
            keep_going && !is_stop(first_tok)
        } else {
            false
        };

        if enter_decode {
            let mut decode_steps = 0usize;
            let mut cur_tok = first_tok;
            while generated.len() < params.max_tokens && self.state.pos + 1 < self.max_ctx {
                if (opts.should_yield)(generated.len()) {
                    yielded = true;
                    break;
                }
                self.poll_async_adapt()?;
                let rem_tokens = params.max_tokens - generated.len();
                let rem_ctx = self.max_ctx - 1 - self.state.pos;
                let k = if use_mtp {
                    self.mtp_k.min(rem_tokens - 1).min(rem_ctx)
                } else {
                    0
                };

                if k == 0 {
                    let p_cur = self.state.pos;
                    let nxt = if greedy {
                        let toks = forward_decode_verify_step(
                            &mut self.weights,
                            &mut self.state,
                            self.ple.as_ref(),
                            &[cur_tok],
                            p_cur,
                            false,
                            true,
                            prof_tg.as_mut(),
                        )?;
                        toks[0]
                    } else {
                        forward_decode_verify_step(
                            &mut self.weights,
                            &mut self.state,
                            self.ple.as_ref(),
                            &[cur_tok],
                            p_cur,
                            false,
                            false,
                            prof_tg.as_mut(),
                        )?;
                        let logits = self.read_logits_rows(1)?;
                        crate::sampling::sample_next_token(logits, params, &mut rng)
                    };
                    self.state.pos += 1;
                    self.state.mrope_pos += 1;
                    self.committed_tokens.push(cur_tok);
                    decode_steps += 1;
                    if decode_steps % adapt_every == 0
                        && self.start_async_adapt(adapt_swaps, decode_min_usage)? > 0
                    {
                        for es in &mut self.weights.experts {
                            for u in &mut es.decode_usage {
                                *u *= 0.7;
                            }
                        }
                    }
                    generated.push(nxt);
                    let cont = on_token(nxt);
                    if !cont || is_stop(nxt) {
                        break;
                    }
                    cur_tok = nxt;
                } else {
                    let pos_before = self.state.pos;
                    let mpos_before = self.state.mrope_pos;
                    let n_vocab = self.cfg.n_vocab;

                    let mut drafts = Vec::with_capacity(k);
                    let mut draft_probs_list: Vec<Vec<(u32, f32)>> = if greedy {
                        Vec::new()
                    } else {
                        Vec::with_capacity(k)
                    };

                    let mut d_in_tok = cur_tok;
                    for j in 0..k {
                        let r_in_ptr = if j == 0 {
                            self.state.r_last.ptr.cast::<f32>()
                        } else {
                            self.state.mtp.as_ref().unwrap().r_mtp.ptr.cast::<f32>()
                        };
                        let cell_pos = pos_before - 1 + j;
                        let (d_tok, d_prob) = if greedy {
                            mtp::mtp_draft_step(
                                &self.weights,
                                &mut self.state,
                                &self.pool,
                                d_in_tok,
                                r_in_ptr,
                                cell_pos,
                                true,
                            )?
                        } else {
                            let _ = mtp::mtp_draft_step(
                                &self.weights,
                                &mut self.state,
                                &self.pool,
                                d_in_tok,
                                r_in_ptr,
                                cell_pos,
                                false,
                            )?;
                            let ms = self.state.mtp.as_ref().unwrap();
                            let hg = self.state.gpus.last().unwrap();
                            set_device(hg.device)?;
                            let mut d_logits = vec![0.0f32; n_vocab];
                            unsafe {
                                copy_d2h_raw_async(
                                    bytemuck::cast_slice_mut(&mut d_logits),
                                    ms.logits.ptr,
                                    &hg.stream,
                                )?;
                            }
                            hg.stream.sync()?;
                            let probs = crate::sampling::build_top_p_probs(&d_logits, params);
                            let sampled = crate::sampling::sample_from_probs(&probs, &mut rng);
                            draft_probs_list.push(probs);
                            (sampled, 1.0f32)
                        };
                        drafts.push(d_tok);
                        d_in_tok = d_tok;
                        if greedy && j + 1 < k && d_prob < self.spec_min_p {
                            break;
                        }
                    }

                    let k_act = drafts.len();
                    let mut verify_toks = Vec::with_capacity(1 + k_act);
                    verify_toks.push(cur_tok);
                    verify_toks.extend_from_slice(&drafts);

                    let mut accept_count = 0usize;
                    let mut final_tok = 0u32;

                    self.poll_async_adapt()?;

                    if greedy {
                        let argmax_toks = forward_decode_verify_step(
                            &mut self.weights,
                            &mut self.state,
                            self.ple.as_ref(),
                            &verify_toks,
                            pos_before,
                            true,
                            true,
                            prof_tg.as_mut(),
                        )?;
                        for j in 0..k_act {
                            if argmax_toks[j] == drafts[j] {
                                accept_count += 1;
                            } else {
                                final_tok = argmax_toks[j];
                                break;
                            }
                        }
                        if accept_count == k_act {
                            final_tok = argmax_toks[k_act];
                        }
                    } else {
                        forward_decode_verify_step(
                            &mut self.weights,
                            &mut self.state,
                            self.ple.as_ref(),
                            &verify_toks,
                            pos_before,
                            true,
                            false,
                            prof_tg.as_mut(),
                        )?;
                        let t_logits = self.read_logits_rows(verify_toks.len())?;
                        for j in 0..k_act {
                            let row = &t_logits[j * n_vocab..(j + 1) * n_vocab];
                            let (res_tok, acc) = crate::sampling::leviathan_verify_sample(
                                row,
                                drafts[j],
                                &draft_probs_list[j],
                                params,
                                &mut rng,
                            );
                            if acc {
                                accept_count += 1;
                            } else {
                                final_tok = res_tok;
                                break;
                            }
                        }
                        if accept_count == k_act {
                            let last_row = &t_logits[k_act * n_vocab..(k_act + 1) * n_vocab];
                            final_tok =
                                crate::sampling::sample_next_token(last_row, params, &mut rng);
                        }
                    }

                    round_draft_n += k_act;
                    round_draft_acc += accept_count;
                    self.draft_n_total += k_act;
                    self.draft_accepted_total += accept_count;

                    if accept_count < k_act {
                        rollback_verify(
                            &self.weights,
                            &mut self.state,
                            &verify_toks,
                            accept_count,
                            pos_before,
                        )?;
                    }
                    self.state.pos = pos_before + accept_count + 1;
                    self.state.mrope_pos = mpos_before + accept_count + 1;
                    self.committed_tokens
                        .extend_from_slice(&verify_toks[..accept_count + 1]);

                    // Refresh r_last[0] to accepted row and sync MTP KV over newly accepted prefix.
                    if accept_count > 0 {
                        let r_last_ptr = self.state.r_last.ptr.cast::<f32>();
                        mtp::mtp_prefill_kv(
                            &self.weights,
                            &mut self.state,
                            &self.pool,
                            &drafts[..accept_count],
                            r_last_ptr,
                            pos_before,
                        )?;
                    }

                    verify_rounds += 1;
                    if self.spec_min_p <= 0.0 && self.mtp_k_adaptive && k_act == self.mtp_k {
                        acc_ewma = 0.7 * acc_ewma + 0.3 * (accept_count as f64 / k_act as f64);
                        if verify_rounds >= 2 {
                            if acc_ewma > 0.85 && self.mtp_k < 5 {
                                self.mtp_k += 1;
                            } else if acc_ewma < 0.50 && self.mtp_k > 2 {
                                self.mtp_k -= 1;
                            }
                        }
                    }

                    if decode_promote > 0 && verify_rounds > 0 && verify_rounds % 16 == 0 {
                        self.run_warmup(decode_promote, 2.0)?;
                    }

                    if verify_rounds % adapt_every == 0
                        && self.start_async_adapt(adapt_swaps, decode_min_usage)? > 0
                    {
                        for es in &mut self.weights.experts {
                            for u in &mut es.decode_usage {
                                *u *= 0.7;
                            }
                        }
                    }

                    let mut stopped = false;
                    for &dt in &drafts[..accept_count] {
                        generated.push(dt);
                        let cont = on_token(dt);
                        if !cont || is_stop(dt) || generated.len() >= params.max_tokens {
                            stopped = true;
                            break;
                        }
                    }
                    if !stopped && generated.len() < params.max_tokens {
                        generated.push(final_tok);
                        let cont = on_token(final_tok);
                        if !cont || is_stop(final_tok) {
                            stopped = true;
                        }
                        cur_tok = final_tok;
                    }
                    if stopped {
                        break;
                    }
                }
            }
        }

        let tg_secs = t_tg0.elapsed().as_secs_f64();
        if let Some(ref p) = prof_tg {
            p.print_summary("DECODE");
        }

        let tg_gen_after_first = if resuming_decode {
            generated.len()
        } else {
            generated.len().saturating_sub(1)
        };
        let prefill_tps = if pp_secs > 0.0 {
            (prompt_tokens_evaluated as f64) / pp_secs.max(1e-9)
        } else {
            0.0
        };
        let decode_tps = if tg_gen_after_first > 0 {
            (tg_gen_after_first as f64) / tg_secs.max(1e-9)
        } else if pp_secs > 0.0 {
            1.0 / pp_secs.max(1e-9)
        } else {
            0.0
        };

        Ok(QuantumGenerateResult {
            result: StreamGenerateResult {
                prompt_tokens_evaluated,
                prefix_reused_tokens: lcp,
                generated_tokens: generated,
                prefill_tps,
                decode_tps,
                draft_n: round_draft_n,
                draft_accepted: round_draft_acc,
            },
            yielded,
            prefill_snapshot,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        n_predict: usize,
    ) -> Result<(Vec<u32>, f64, f64, Vec<(u32, f32)>)> {
        let params = crate::sampling::SamplingParams {
            temperature: 0.0,
            max_tokens: n_predict,
            ..Default::default()
        };
        let res = self.generate_stream(prompt_tokens, &params, |_| true)?;
        Ok((
            res.generated_tokens,
            res.prefill_tps,
            res.decode_tps,
            Vec::new(),
        ))
    }
}
