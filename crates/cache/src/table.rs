use crate::adapt::EwmaTracker;
use crate::arena::{align_up_256, HostExpertArena};
use crate::profile::CacheProfile;
use ampllm_cuda_kernels::cuda_rt::copy_h2d_raw_async;
use ampllm_cuda_kernels::{
    mem_info, set_device, stream_priority_range, CudaEvent, CudaStream, DeviceBuffer,
    PinnedMappedBuffer,
};
use ampllm_gguf::{GgufFile, Inkling2Config};
use anyhow::Result;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBudget {
    Auto,
    Mb(usize),
}

pub fn layer_expert_sizes(gguf: &GgufFile, cfg: &Inkling2Config) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::with_capacity(cfg.n_layer_phys);
    for i in 0..cfg.n_layer_phys {
        let gu_info = gguf.tensor_info(&format!("blk.{i}.ffn_gate_up_exps.weight"))?;
        let dn_info = gguf.tensor_info(&format!("blk.{i}.ffn_down_exps.weight"))?;
        let gu_bytes = gu_info.ggml_type.row_bytes(cfg.n_embd) * (2 * cfg.n_ff_exp);
        let dn_bytes = dn_info.ggml_type.row_bytes(cfg.n_ff_exp) * cfg.n_embd;
        out.push((gu_bytes, dn_bytes));
    }
    Ok(out)
}

/// Plan per-physical-layer expert slot capacities within `budget_bytes`, giving Loop Core layers (`loop_start..loop_end`) 3x priority.
pub fn plan_layer_capacities(
    cfg: &Inkling2Config,
    layer_expert_bytes: &[(usize, usize)],
    budget_bytes: usize,
    max_per_layer: Option<&[usize]>,
) -> Vec<usize> {
    let n_layers = cfg.n_layer_phys;
    let max_caps: Vec<usize> = (0..n_layers)
        .map(|i| max_per_layer.map(|m| m[i]).unwrap_or(cfg.n_expert).min(cfg.n_expert))
        .collect();

    let per_exp_stride: Vec<usize> = layer_expert_bytes
        .iter()
        .map(|&(gu, dn)| align_up_256(gu) + align_up_256(dn))
        .collect();

    let total_needed: usize = (0..n_layers)
        .map(|i| max_caps[i] * per_exp_stride[i])
        .sum();
    if total_needed <= budget_bytes {
        return max_caps;
    }

    let weights: Vec<f64> = (0..n_layers)
        .map(|i| {
            if cfg.is_looped && i >= cfg.loop_start && i < cfg.loop_end {
                3.0
            } else {
                1.0
            }
        })
        .collect();

    // Scale `alpha` so initial continuous weighted slots fit inside budget_bytes.
    let denom: f64 = (0..n_layers)
        .filter(|&i| max_caps[i] > 0)
        .map(|i| weights[i] * (per_exp_stride[i] as f64))
        .sum();

    let mut caps = vec![0usize; n_layers];
    let mut used = 0usize;
    if denom > 0.0 {
        let alpha = (budget_bytes as f64) / denom;
        for i in 0..n_layers {
            let c = ((alpha * weights[i]).floor() as usize).min(max_caps[i]);
            let cost = c * per_exp_stride[i];
            if used + cost <= budget_bytes {
                caps[i] = c;
                used += cost;
            }
        }
    }

    // Greedy fill of remaining budget prioritizing Loop Core highest marginal utility.
    let mut remaining = budget_bytes.saturating_sub(used);
    loop {
        let mut best: Option<(usize, f64)> = None;
        for i in 0..n_layers {
            if caps[i] < max_caps[i] && per_exp_stride[i] <= remaining {
                let score = weights[i] / (caps[i] as f64 + 1.0);
                let better = match best {
                    None => true,
                    Some((_, bs)) => score > bs,
                };
                if better {
                    best = Some((i, score));
                }
            }
        }
        match best {
            Some((i, _)) => {
                caps[i] += 1;
                remaining -= per_exp_stride[i];
            }
            None => break,
        }
    }

    caps
}

/// Resolve remaining free VRAM (after non-expert weights + KV/scratch are allocated) into an expert slab budget.
/// Always leaves at least `safety_reserve_mb` (default 1024 MiB, safely >= 768 MiB for display on GPU 1).
pub fn resolve_auto_budget_bytes(device: i32, safety_reserve_mb: usize) -> Result<usize> {
    let (free, _total) = mem_info(device)?;
    let reserve = safety_reserve_mb * 1024 * 1024;
    Ok(free.saturating_sub(reserve))
}

struct PendingPromotion {
    layer: usize,
    exp: usize,
    slot: usize,
    gu_ptr: u64,
    dn_ptr: u64,
    ready_event: CudaEvent,
}

pub struct LayerVramSlab {
    pub cap: usize,
    pub gu_stride: usize,
    pub dn_stride: usize,
    pub gu_bytes: usize,
    pub dn_bytes: usize,
    pub gate_up_slab: DeviceBuffer,
    pub down_slab: DeviceBuffer,
    pub gate_up_ptrs_host: PinnedMappedBuffer<u64>,
    pub down_ptrs_host: PinnedMappedBuffer<u64>,
    pub gate_up_ptrs_dev: DeviceBuffer,
    pub down_ptrs_dev: DeviceBuffer,
    pub resident_slot: Vec<Option<usize>>,
    pub slot_owner: Vec<Option<usize>>,
    pub in_flight_slot: Vec<bool>,
    pub in_flight_exp: Vec<bool>,
    pub dirty: bool,
}

/// Dynamic VRAM expert slab table with Loop-Core priority, disjoint partitioning, and lagged async DMA promotion.
pub struct VramExpertCache {
    pub device: i32,
    pub n_expert: usize,
    pub all_resident: bool,
    pub cache_epoch: u64,
    pub layers: Vec<LayerVramSlab>,
    pub copy_stream: CudaStream,
    pending: Vec<PendingPromotion>,
}

impl VramExpertCache {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        device: i32,
        layer_caps: &[usize],
        layer_expert_bytes: &[(usize, usize)],
        profile: Option<&CacheProfile>,
        exclude_mask: Option<&[Vec<bool>]>,
        arena: Option<&HostExpertArena>,
        init_stream: &CudaStream,
    ) -> Result<Self> {
        set_device(device)?;
        let (least_prio, _) = stream_priority_range(device)?;
        let copy_stream = CudaStream::with_priority(device, least_prio)?;

        let all_resident = layer_caps.iter().all(|&c| c >= cfg.n_expert);
        let mut layers = Vec::with_capacity(cfg.n_layer_phys);

        for i in 0..cfg.n_layer_phys {
            let cap = layer_caps[i].min(cfg.n_expert);
            let (gu_bytes, dn_bytes) = layer_expert_bytes[i];
            let gu_stride = align_up_256(gu_bytes);
            let dn_stride = align_up_256(dn_bytes);

            let gate_up_slab = if cap > 0 {
                DeviceBuffer::alloc(device, cap * gu_stride)?
            } else {
                DeviceBuffer::alloc(device, 0)?
            };
            let down_slab = if cap > 0 {
                DeviceBuffer::alloc(device, cap * dn_stride)?
            } else {
                DeviceBuffer::alloc(device, 0)?
            };

            let mut gate_up_ptrs_host = PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?;
            let mut down_ptrs_host = PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?;
            let gate_up_ptrs_dev = DeviceBuffer::alloc(device, cfg.n_expert * 8)?;
            let down_ptrs_dev = DeviceBuffer::alloc(device, cfg.n_expert * 8)?;

            let mut resident_slot = vec![None; cfg.n_expert];
            let mut slot_owner = vec![None; cap];

            // Build initial fill order from profile (if any), skipping excluded experts.
            let mut order: Vec<usize> = if let Some(prof) = profile {
                if i < prof.ranked_experts.len() {
                    let mut list: Vec<usize> = prof.ranked_experts[i]
                        .iter()
                        .copied()
                        .filter(|&e| e < cfg.n_expert)
                        .collect();
                    for e in 0..cfg.n_expert {
                        if !list.contains(&e) {
                            list.push(e);
                        }
                    }
                    list
                } else {
                    (0..cfg.n_expert).collect()
                }
            } else {
                (0..cfg.n_expert).collect()
            };
            if let Some(excl) = exclude_mask {
                order.retain(|&e| !excl[i][e]);
            }

            let gguf_gu = if arena.as_ref().is_some_and(|a| a.has_layer(i)) {
                None
            } else {
                Some(gguf.tensor_bytes(&format!("blk.{i}.ffn_gate_up_exps.weight"))?)
            };
            let gguf_dn = if arena.as_ref().is_some_and(|a| a.has_layer(i)) {
                None
            } else {
                Some(gguf.tensor_bytes(&format!("blk.{i}.ffn_down_exps.weight"))?)
            };

            let gu_host_ptrs = gate_up_ptrs_host.as_mut_slice();
            let dn_host_ptrs = down_ptrs_host.as_mut_slice();

            for (slot, &exp) in order.iter().take(cap).enumerate() {
                let gu_src = if let (Some(a), Some(_)) = (arena, arena.filter(|a| a.has_layer(i))) {
                    a.expert_gate_up(i, exp)
                } else {
                    let raw = gguf_gu.unwrap();
                    &raw[exp * gu_bytes..(exp + 1) * gu_bytes]
                };
                let dn_src = if let (Some(a), Some(_)) = (arena, arena.filter(|a| a.has_layer(i))) {
                    a.expert_down(i, exp)
                } else {
                    let raw = gguf_dn.unwrap();
                    &raw[exp * dn_bytes..(exp + 1) * dn_bytes]
                };

                let gu_dst = gate_up_slab.as_u64() + (slot * gu_stride) as u64;
                let dn_dst = down_slab.as_u64() + (slot * dn_stride) as u64;
                unsafe {
                    copy_h2d_raw_async(gu_dst as *mut c_void, gu_src, init_stream)?;
                    copy_h2d_raw_async(dn_dst as *mut c_void, dn_src, init_stream)?;
                }

                gu_host_ptrs[exp] = gu_dst;
                dn_host_ptrs[exp] = dn_dst;
                resident_slot[exp] = Some(slot);
                slot_owner[slot] = Some(exp);
            }

            gate_up_ptrs_dev.copy_from_host_async(
                bytemuck::cast_slice(gate_up_ptrs_host.as_slice()),
                init_stream,
            )?;
            down_ptrs_dev.copy_from_host_async(
                bytemuck::cast_slice(down_ptrs_host.as_slice()),
                init_stream,
            )?;

            layers.push(LayerVramSlab {
                cap,
                gu_stride,
                dn_stride,
                gu_bytes,
                dn_bytes,
                gate_up_slab,
                down_slab,
                gate_up_ptrs_host,
                down_ptrs_host,
                gate_up_ptrs_dev,
                down_ptrs_dev,
                resident_slot,
                slot_owner,
                in_flight_slot: vec![false; cap],
                in_flight_exp: vec![false; cfg.n_expert],
                dirty: false,
            });
        }
        init_stream.sync()?;

        Ok(Self {
            device,
            n_expert: cfg.n_expert,
            all_resident,
            cache_epoch: 1,
            layers,
            copy_stream,
            pending: Vec::new(),
        })
    }

    #[inline]
    pub fn is_resident(&self, layer: usize, exp: usize) -> bool {
        self.layers[layer].resident_slot[exp].is_some()
    }

    #[inline]
    pub fn layer_ptrs_dev(&self, layer: usize) -> (u64, u64) {
        let l = &self.layers[layer];
        (l.gate_up_ptrs_dev.as_u64(), l.down_ptrs_dev.as_u64())
    }

    pub fn resident_mask(&self) -> Vec<Vec<bool>> {
        self.layers
            .iter()
            .map(|l| l.resident_slot.iter().map(|s| s.is_some()).collect())
            .collect()
    }

    pub fn total_vram_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.gate_up_slab.bytes
                    + l.down_slab.bytes
                    + l.gate_up_ptrs_dev.bytes
                    + l.down_ptrs_dev.bytes
            })
            .sum()
    }

    pub fn total_slots(&self) -> usize {
        self.layers.iter().map(|l| l.cap).sum()
    }

    /// Poll async H2D promotions; install ready slots and mark dirty layers.
    pub fn poll_promotions(&mut self) -> Result<bool> {
        if self.pending.is_empty() {
            return Ok(false);
        }
        let mut changed = false;
        let mut remaining = Vec::with_capacity(self.pending.len());
        for p in self.pending.drain(..) {
            if p.ready_event.is_ready()? {
                let l = &mut self.layers[p.layer];
                l.in_flight_slot[p.slot] = false;
                l.in_flight_exp[p.exp] = false;
                l.resident_slot[p.exp] = Some(p.slot);
                l.slot_owner[p.slot] = Some(p.exp);
                l.gate_up_ptrs_host.as_mut_slice()[p.exp] = p.gu_ptr;
                l.down_ptrs_host.as_mut_slice()[p.exp] = p.dn_ptr;
                l.dirty = true;
                changed = true;
            } else {
                remaining.push(p);
            }
        }
        self.pending = remaining;
        if changed {
            self.cache_epoch += 1;
        }
        Ok(changed)
    }

    /// Schedule up to `max_promotions` asynchronous promotions on `copy_stream`, waiting for `compute_done`
    /// so we never overwrite a victim slot still being read by compute kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn schedule_promotions(
        &mut self,
        arena: &HostExpertArena,
        tracker: &EwmaTracker,
        compute_done: &CudaEvent,
        recent_active: &[Vec<usize>],
        max_promotions: usize,
        max_in_flight: usize,
        exclude_mask: Option<&[Vec<bool>]>,
    ) -> Result<usize> {
        if self.all_resident || max_promotions == 0 || self.pending.len() >= max_in_flight {
            return Ok(0);
        }
        set_device(self.device)?;

        let mut candidates: Vec<(f32, usize, usize, usize)> = Vec::new();
        for l_idx in 0..self.layers.len() {
            let l = &self.layers[l_idx];
            if l.cap == 0 || l.cap >= self.n_expert || !arena.has_layer(l_idx) {
                continue;
            }
            let active_l = recent_active.get(l_idx).map(|v| v.as_slice()).unwrap_or(&[]);

            // Find lowest-EWMA resident slot not currently in flight or active on this step.
            let mut min_victim: Option<(usize, usize, f32)> = None;
            let mut free_slot: Option<usize> = None;
            for s in 0..l.cap {
                if l.in_flight_slot[s] {
                    continue;
                }
                match l.slot_owner[s] {
                    None => {
                        free_slot = Some(s);
                        break;
                    }
                    Some(owner) => {
                        if active_l.contains(&owner) {
                            continue;
                        }
                        let sc = tracker.scores[l_idx][owner];
                        if min_victim.is_none_or(|(_, _, ms)| sc < ms) {
                            min_victim = Some((s, owner, sc));
                        }
                    }
                }
            }

            let (victim_slot, victim_score) = if let Some(fs) = free_slot {
                (fs, -1.0f32)
            } else if let Some((vs, _, vsc)) = min_victim {
                (vs, vsc)
            } else {
                continue;
            };

            let mut best_miss: Option<(usize, f32)> = None;
            for &e in active_l {
                if l.resident_slot[e].is_some() || l.in_flight_exp[e] {
                    continue;
                }
                if exclude_mask.is_some_and(|ex| ex[l_idx][e]) {
                    continue;
                }
                let sc = tracker.scores[l_idx][e];
                if sc > victim_score * 1.15 && best_miss.is_none_or(|(_, bs)| sc > bs) {
                    best_miss = Some((e, sc));
                }
            }

            if let Some((miss_e, miss_sc)) = best_miss {
                let gain = miss_sc - victim_score.max(0.0);
                candidates.push((gain, l_idx, miss_e, victim_slot));
            }
        }

        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut scheduled = 0usize;
        for (_, l_idx, miss_e, slot) in candidates {
            if scheduled >= max_promotions || self.pending.len() >= max_in_flight {
                break;
            }
            if scheduled == 0 {
                self.copy_stream.wait_event(compute_done)?;
            }

            let l = &mut self.layers[l_idx];
            if let Some(old_exp) = l.slot_owner[slot].take() {
                l.resident_slot[old_exp] = None;
                l.gate_up_ptrs_host.as_mut_slice()[old_exp] = 0;
                l.down_ptrs_host.as_mut_slice()[old_exp] = 0;
                l.dirty = true;
            }
            l.in_flight_slot[slot] = true;
            l.in_flight_exp[miss_e] = true;

            let gu_ptr = l.gate_up_slab.as_u64() + (slot * l.gu_stride) as u64;
            let dn_ptr = l.down_slab.as_u64() + (slot * l.dn_stride) as u64;
            let gu_src = arena.expert_gate_up(l_idx, miss_e);
            let dn_src = arena.expert_down(l_idx, miss_e);

            unsafe {
                copy_h2d_raw_async(gu_ptr as *mut c_void, gu_src, &self.copy_stream)?;
                copy_h2d_raw_async(dn_ptr as *mut c_void, dn_src, &self.copy_stream)?;
            }

            let ready_event = CudaEvent::new(self.device)?;
            ready_event.record(&self.copy_stream)?;
            self.pending.push(PendingPromotion {
                layer: l_idx,
                exp: miss_e,
                slot,
                gu_ptr,
                dn_ptr,
                ready_event,
            });
            scheduled += 1;
        }

        Ok(scheduled)
    }

    /// Push any modified host pointer tables to GPU memory on `compute_stream` before the next forward pass.
    pub fn sync_dirty_ptr_tables(&mut self, compute_stream: &CudaStream) -> Result<()> {
        for l in &mut self.layers {
            if l.dirty {
                l.gate_up_ptrs_dev.copy_from_host_async(
                    bytemuck::cast_slice(l.gate_up_ptrs_host.as_slice()),
                    compute_stream,
                )?;
                l.down_ptrs_dev.copy_from_host_async(
                    bytemuck::cast_slice(l.down_ptrs_host.as_slice()),
                    compute_stream,
                )?;
                l.dirty = false;
            }
        }
        Ok(())
    }

    pub fn sync_all_pending(&mut self, compute_stream: &CudaStream) -> Result<()> {
        self.copy_stream.sync()?;
        self.poll_promotions()?;
        self.sync_dirty_ptr_tables(compute_stream)?;
        Ok(())
    }
}
