use crate::glm5::weights::{CpuColdArena, Glm5ModelWeights};
use ampllm_cache::CacheProfile;
use ampllm_cuda_kernels::{
    copy_d2h_raw_async, copy_h2d_raw_async, glm5::Glm5Doorbell, set_device, CudaEvent,
    CudaStream, DeviceBuffer, PinnedMappedBuffer,
};
use anyhow::Result;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::atomic::{fence, Ordering};

const WARMUP_WAVE_SLOTS: usize = 8;
const MAX_CONCURRENT_D2H: usize = 16;
const TABLE_U64_SLOTS: usize = 1024;
const TABLE_FULL_SLOTS: usize = 48;

struct PendingD2h {
    layer: usize,
    spare_idx: usize,
    vic_slot: usize,
    vic_e: usize,
    vic_ptr: u64,
    backup_slot: usize,
    ev: CudaEvent,
}

pub struct LayerCacheMeta {
    pub device: i32,
    pub is_moe: bool,
    pub cap_slots: usize,
    pub k_spares: usize,
    pub total_pool_slots: usize,
    pub exp_pool_ptr: u64,
    pub g_eb: usize,
    pub d_eb: usize,
    pub u_off: usize,
    pub d_off: usize,
    pub blob_stride: usize,
    pub slot_of_expert: Vec<i16>,  // [n_expert]
    pub expert_of_slot: Vec<i16>,  // [total_pool_slots], -1 when spare/empty
    pub last_used: Vec<u64>,       // [total_pool_slots]
    pub heat: Vec<f32>,            // [n_expert]
    pub spare_ptrs: Vec<u64>,      // [k_spares], 0 when claimed on GPU
    pub vram_ptrs_host: Vec<u64>,  // [n_expert]
    pub res_mask_host: Vec<u8>,    // [n_expert]
}

struct RetiringRamBatch {
    pairs: Vec<(usize, usize)>,
    ev: CudaEvent,
}

pub struct DevAsyncMover {
    pub copy_stream: CudaStream,
    draining: VecDeque<PendingD2h>,
    event_pool: Vec<CudaEvent>,
    retiring_ram: VecDeque<RetiringRamBatch>,
    retire_event_pool: Vec<CudaEvent>,
    u64_pin: PinnedMappedBuffer<u64>,
    u64_cursor: usize,
    table_pin: PinnedMappedBuffer<u64>,
    table_stride_u64: usize,
    table_cursor: usize,
}

impl DevAsyncMover {
    fn new(dev: i32, n_expert: usize) -> Result<Self> {
        let table_stride_u64 = (n_expert + 31) & !31;
        let copy_stream = CudaStream::new(dev)?;
        let mut event_pool = Vec::with_capacity(MAX_CONCURRENT_D2H);
        let mut retire_event_pool = Vec::with_capacity(MAX_CONCURRENT_D2H);
        for _ in 0..MAX_CONCURRENT_D2H {
            event_pool.push(CudaEvent::new(dev)?);
            retire_event_pool.push(CudaEvent::new(dev)?);
        }
        let u64_pin = PinnedMappedBuffer::<u64>::alloc(TABLE_U64_SLOTS)?;
        let table_pin = PinnedMappedBuffer::<u64>::alloc(TABLE_FULL_SLOTS * table_stride_u64)?;
        Ok(Self {
            copy_stream,
            draining: VecDeque::with_capacity(MAX_CONCURRENT_D2H),
            event_pool,
            retiring_ram: VecDeque::with_capacity(MAX_CONCURRENT_D2H),
            retire_event_pool,
            u64_pin,
            u64_cursor: 0,
            table_pin,
            table_stride_u64,
            table_cursor: 0,
        })
    }

    #[inline]
    fn queue_u64_h2d(&mut self, dst_dev_ptr: u64, val: u64, stream: &CudaStream) -> Result<()> {
        let idx = self.u64_cursor % TABLE_U64_SLOTS;
        self.u64_cursor = self.u64_cursor.wrapping_add(1);
        unsafe {
            let slot_ptr = self.u64_pin.host_ptr().add(idx);
            std::ptr::write(slot_ptr, val);
            let bytes = std::slice::from_raw_parts(slot_ptr as *const u8, 8);
            copy_h2d_raw_async(dst_dev_ptr as *mut c_void, bytes, stream)?;
        }
        Ok(())
    }

    #[inline]
    fn next_table_slot(&mut self) -> usize {
        let s = self.table_cursor % TABLE_FULL_SLOTS;
        self.table_cursor = self.table_cursor.wrapping_add(1);
        s
    }
}

pub struct DynamicExpertCache {
    pub layers: Vec<LayerCacheMeta>,
    movers: [DevAsyncMover; 2],
    pub tick: u64,
    pub promotions_total: u64,
    pub evictions_total: u64,
    pub d2h_evictions_total: u64,
    pub freeze: bool,
    pub keep_cold_ram: bool,
    pub has_profile: bool,
    pub evict_policy: u8,
    pub heat_decay: f32,
    pub evict_guard: u64,
}

impl DynamicExpertCache {
    pub fn from_weights(weights: &Glm5ModelWeights) -> Result<Self> {
        let mut layers = Vec::with_capacity(weights.layers.len());
        let n_expert = weights.cfg.n_expert;

        for lw in &weights.layers {
            let cap_slots = lw.initial_slot_experts.len();
            let k_spares = lw.k_spares;
            let total_pool_slots = cap_slots + k_spares;
            let mut slot_of_expert = vec![-1i16; n_expert];
            let mut expert_of_slot = vec![-1i16; total_pool_slots];
            let mut last_used = vec![0u64; total_pool_slots];
            for (slot, &eid) in lw.initial_slot_experts.iter().enumerate() {
                slot_of_expert[eid] = slot as i16;
                expert_of_slot[slot] = eid as i16;
                let score = lw.initial_heat.get(eid).copied().unwrap_or(0.0).clamp(0.0, 3.0);
                last_used[slot] = 900 + ((score / 3.0) * 100.0) as u64;
            }
            layers.push(LayerCacheMeta {
                device: lw.device,
                is_moe: lw.is_moe,
                cap_slots,
                k_spares,
                total_pool_slots,
                exp_pool_ptr: lw.exp_pool_ptr,
                g_eb: lw.g_eb,
                d_eb: lw.d_eb,
                u_off: lw.u_off,
                d_off: lw.d_off,
                blob_stride: lw.blob_stride,
                slot_of_expert,
                expert_of_slot,
                last_used,
                heat: if lw.initial_heat.len() == n_expert {
                    lw.initial_heat.clone()
                } else {
                    vec![0.0f32; n_expert]
                },
                spare_ptrs: lw.initial_spare_ptrs.clone(),
                vram_ptrs_host: lw.vram_ptrs_host.clone(),
                res_mask_host: lw.res_mask_host.clone(),
            });
        }

        let mover0 = DevAsyncMover::new(0, n_expert)?;
        let mover1 = DevAsyncMover::new(1, n_expert)?;
        let freeze = std::env::var_os("AMPLLM_FREEZE_CACHE").is_some();
        let keep_cold_ram = std::env::var("AMPLLM_KEEP_COLD_RAM")
            .map(|v| v != "0")
            .unwrap_or(false);
        let evict_policy = match std::env::var("AMPLLM_EVICT_POLICY")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("heat") => 1u8,
            Some("hybrid") => 2u8,
            _ => 0u8,
        };
        let heat_decay = std::env::var("AMPLLM_HEAT_DECAY")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.992)
            .clamp(0.5, 0.9999);
        let evict_guard = std::env::var("AMPLLM_EVICT_GUARD")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1);

        Ok(Self {
            layers,
            movers: [mover0, mover1],
            tick: 1000,
            promotions_total: 0,
            evictions_total: 0,
            d2h_evictions_total: 0,
            freeze,
            keep_cold_ram,
            has_profile: weights.has_profile,
            evict_policy,
            heat_decay,
            evict_guard,
        })
    }

    fn upload_vram_ptrs_table(
        lc: &LayerCacheMeta,
        weights: &Glm5ModelWeights,
        l: usize,
        mover: &DevAsyncMover,
        slot_idx: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let lw = &weights.layers[l];
        let n_exp = lc.vram_ptrs_host.len();
        if n_exp == 0 {
            return Ok(());
        }
        let ptr_bytes = n_exp * 8;
        unsafe {
            let dst = mover.table_pin.host_ptr().add(slot_idx * mover.table_stride_u64);
            std::ptr::copy_nonoverlapping(lc.vram_ptrs_host.as_ptr(), dst, n_exp);
            if let Some(ref b) = lw.vram_ptrs_dev {
                b.copy_from_host_async(
                    std::slice::from_raw_parts(dst as *const u8, ptr_bytes),
                    stream,
                )?;
            }
        }
        Ok(())
    }

    fn release_promoted_ram_sync(
        &mut self,
        dev: i32,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        stream: &CudaStream,
        pairs: &[(usize, usize)],
        mut dev_set: bool,
    ) -> Result<bool> {
        if self.keep_cold_ram {
            return Ok(dev_set);
        }
        let d = dev as usize;
        for &(l, e) in pairs {
            if self.layers[l].res_mask_host[e] != 0 && arena.release_ram_slot(l, e) {
                if !dev_set {
                    set_device(dev)?;
                    dev_set = true;
                }
                if let Some(ref rp) = weights.layers[l].ram_ptrs_dev {
                    self.movers[d].queue_u64_h2d(
                        rp.as_u64() + (e * 8) as u64,
                        0,
                        stream,
                    )?;
                }
            }
        }
        Ok(dev_set)
    }

    fn finish_pending_d2h(
        &mut self,
        dev: i32,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        compute_stream: &CudaStream,
        db: &mut Glm5Doorbell,
        force_sync: bool,
    ) -> Result<()> {
        let d = dev as usize;
        if self.movers[d].retiring_ram.is_empty() && self.movers[d].draining.is_empty() {
            return Ok(());
        }
        let mut dev_set = false;
        if !self.movers[d].retiring_ram.is_empty() {
            if force_sync {
                set_device(dev)?;
                dev_set = true;
                compute_stream.sync()?;
            }
            while !self.movers[d].retiring_ram.is_empty() {
                if !force_sync && !self.movers[d].retiring_ram.front().unwrap().ev.is_ready()? {
                    break;
                }
                let batch = self.movers[d].retiring_ram.pop_front().unwrap();
                self.movers[d].retire_event_pool.push(batch.ev);
                dev_set = self.release_promoted_ram_sync(
                    dev,
                    weights,
                    arena,
                    compute_stream,
                    &batch.pairs,
                    dev_set,
                )?;
            }
        }
        if self.movers[d].draining.is_empty() {
            return Ok(());
        }
        if force_sync {
            if !dev_set {
                set_device(dev)?;
                dev_set = true;
            }
            self.movers[d].copy_stream.sync()?;
        }

        while !self.movers[d].draining.is_empty() {
            if !force_sync && !self.movers[d].draining.front().unwrap().ev.is_ready()? {
                break;
            }
            let p = self.movers[d].draining.pop_front().unwrap();
            self.movers[d].event_pool.push(p.ev);

            let [d0, d1] = arena.commit_backup_slot(p.layer, p.vic_e, p.backup_slot);
            let ram_dev_ptr = if dev == 0 { d0 } else { d1 };

            if p.layer < 48 && p.spare_idx < 8 {
                unsafe {
                    std::ptr::write_volatile(&mut db.promo_vram_ptr[p.layer][p.spare_idx], 0);
                    std::ptr::write_volatile(&mut db.promo_eid[p.layer][p.spare_idx], -1);
                }
                fence(Ordering::Release);
            }

            let lw = &weights.layers[p.layer];
            if !dev_set {
                set_device(dev)?;
                dev_set = true;
            }
            if let Some(ref rp) = lw.ram_ptrs_dev {
                self.movers[d].queue_u64_h2d(
                    rp.as_u64() + (p.vic_e * 8) as u64,
                    ram_dev_ptr,
                    compute_stream,
                )?;
            }
            if let Some(ref vp) = lw.vram_ptrs_dev {
                self.movers[d].queue_u64_h2d(
                    vp.as_u64() + (p.vic_e * 8) as u64,
                    0,
                    compute_stream,
                )?;
            }
            if let Some(ref sp) = lw.spares_dev {
                self.movers[d].queue_u64_h2d(
                    sp.as_u64() + (p.spare_idx * 8) as u64,
                    p.vic_ptr,
                    compute_stream,
                )?;
            }

            let lc = &mut self.layers[p.layer];
            lc.slot_of_expert[p.vic_e] = -1;
            lc.expert_of_slot[p.vic_slot] = -1;
            lc.vram_ptrs_host[p.vic_e] = 0;
            lc.res_mask_host[p.vic_e] = 0;
            if p.spare_idx < lc.spare_ptrs.len() {
                lc.spare_ptrs[p.spare_idx] = p.vic_ptr;
            }
            self.evictions_total += 1;
            self.d2h_evictions_total += 1;
        }
        Ok(())
    }

    fn ingest_promos(&mut self, db: &mut Glm5Doorbell, layer_range: Range<usize>) -> Vec<(usize, usize)> {
        let mut promoted = Vec::new();
        for l in layer_range {
            if l >= self.layers.len() || l >= 48 {
                continue;
            }
            let lc = &mut self.layers[l];
            if !lc.is_moe || lc.k_spares == 0 {
                continue;
            }
            for s in 0..lc.k_spares.min(8) {
                let e_raw = unsafe { std::ptr::read_volatile(&db.promo_eid[l][s]) };
                if e_raw >= 0 {
                    fence(Ordering::Acquire);
                    let pptr = unsafe { std::ptr::read_volatile(&db.promo_vram_ptr[l][s]) };
                    unsafe {
                        std::ptr::write_volatile(&mut db.promo_vram_ptr[l][s], 0);
                        std::ptr::write_volatile(&mut db.promo_eid[l][s], -1);
                    }
                    fence(Ordering::Release);

                    let e = e_raw as usize;
                    if e < lc.res_mask_host.len() && pptr >= lc.exp_pool_ptr && lc.blob_stride > 0 {
                        let slot = ((pptr - lc.exp_pool_ptr) as usize) / lc.blob_stride;
                        if slot < lc.total_pool_slots {
                            lc.slot_of_expert[e] = slot as i16;
                            lc.expert_of_slot[slot] = e as i16;
                            lc.last_used[slot] = self.tick;
                            lc.vram_ptrs_host[e] = pptr;
                            lc.res_mask_host[e] = 1;
                            lc.heat[e] = lc.heat[e].max(1.75);
                            promoted.push((l, e));
                        }
                    }
                    if s < lc.spare_ptrs.len() {
                        lc.spare_ptrs[s] = 0;
                    }
                    self.promotions_total += 1;
                }
            }
        }
        promoted
    }

    /// Reconcile completed async D2H backups, absorb GPU spare-slot promotions,
    /// and refill empty spare slots by evicting the coldest resident experts.
    /// Returns `(promos_seen, routed_pairs, vram_hits)`.
    pub fn sync_promos_and_refill_dev(
        &mut self,
        dev: i32,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        compute_stream: &CudaStream,
        db: &mut Glm5Doorbell,
        layer_range: Range<usize>,
        note_routes: bool,
        route_slot: usize,
    ) -> Result<(u64, u64, u64)> {
        self.finish_pending_d2h(dev, weights, arena, compute_stream, db, false)?;

        let mut routed_pairs = 0u64;
        let mut vram_hits = 0u64;
        if note_routes {
            let topk = weights.cfg.n_expert_used.min(8);
            let rbase = (route_slot & 1) * 8;
            for l in layer_range.clone() {
                if l >= self.layers.len() || l >= 48 {
                    continue;
                }
                let lc = &mut self.layers[l];
                if !lc.is_moe {
                    continue;
                }
                let n_exp = lc.heat.len();
                for k in 0..topk {
                    let raw_id = unsafe { std::ptr::read_volatile(&db.routed_ids[l][rbase + k]) };
                    if raw_id >= 0 && (raw_id as usize) < n_exp {
                        unsafe { std::ptr::write_volatile(&mut db.routed_ids[l][rbase + k], -1) };
                        let e = raw_id as usize;
                        routed_pairs += 1;
                        if lc.res_mask_host[e] != 0 {
                            vram_hits += 1;
                        }
                        lc.heat[e] += 1.0;
                        let s = lc.slot_of_expert[e];
                        if s >= 0 {
                            lc.last_used[s as usize] = self.tick;
                        }
                    }
                }
            }
        }

        let promoted = self.ingest_promos(db, layer_range.clone());
        let promos_seen = promoted.len() as u64;
        let d = dev as usize;
        let mut dev_set = false;
        if !self.keep_cold_ram && !promoted.is_empty() {
            set_device(dev)?;
            dev_set = true;
            let ev = self.movers[d]
                .retire_event_pool
                .pop()
                .map(Ok)
                .unwrap_or_else(|| CudaEvent::new(dev))?;
            ev.record(compute_stream)?;
            self.movers[d].retiring_ram.push_back(RetiringRamBatch {
                pairs: promoted,
                ev,
            });
        }
        if self.freeze {
            return Ok((promos_seen, routed_pairs, vram_hits));
        }

        let n_expert = weights.cfg.n_expert;

        for l in layer_range {
            if l >= self.layers.len() {
                continue;
            }
            if !self.layers[l].is_moe || self.layers[l].k_spares == 0 || self.layers[l].cap_slots >= n_expert {
                continue;
            }
            let k_spares = self.layers[l].k_spares;
            for s in 0..k_spares {
                if self.layers[l].spare_ptrs[s] != 0 {
                    continue;
                }
                if self.movers[d]
                    .draining
                    .iter()
                    .any(|p| p.layer == l && p.spare_idx == s)
                {
                    continue;
                }

                let can_d2h = self.movers[d].draining.len() < MAX_CONCURRENT_D2H
                    && !self.movers[d].event_pool.is_empty()
                    && arena.backup_free_count_for_layer(l) > 0;
                let min_recent_cut = self.tick.saturating_sub(self.evict_guard);

                let vic_opt = {
                    let lc = &self.layers[l];
                    let draining = &self.movers[d].draining;
                    let mut best: Option<(u64, f32, f32, usize)> = None;
                    for slot in 0..lc.total_pool_slots {
                        let oe_raw = lc.expert_of_slot[slot];
                        if oe_raw < 0 {
                            continue;
                        }
                        if draining.iter().any(|p| p.layer == l && p.vic_slot == slot) {
                            continue;
                        }
                        let oe = oe_raw as usize;
                        let needs_backup = !arena.is_in_ram(l, oe);
                        if needs_backup && !can_d2h {
                            continue;
                        }
                        let lu = lc.last_used[slot];
                        if lu >= min_recent_cut {
                            continue;
                        }
                        let h = lc.heat[oe];
                        let score = if self.evict_policy == 2 {
                            h / (1.0 + (self.tick.saturating_sub(lu) as f32) * 0.015625)
                        } else {
                            0.0
                        };
                        let replace = match best {
                            None => true,
                            Some((blu, bh, bs, _)) => match self.evict_policy {
                                1 => h < bh || (h == bh && lu < blu),
                                2 => score < bs || (score == bs && lu < blu),
                                _ => lu < blu || (lu == blu && h < bh),
                            },
                        };
                        if replace {
                            best = Some((lu, h, score, slot));
                        }
                    }
                    best.map(|(_, _, _, slot)| slot)
                };

                let Some(vic_slot) = vic_opt else {
                    continue;
                };
                let vic_e = self.layers[l].expert_of_slot[vic_slot] as usize;
                let vic_ptr =
                    self.layers[l].exp_pool_ptr + (vic_slot * self.layers[l].blob_stride) as u64;
                let lw = &weights.layers[l];

                if arena.is_in_ram(l, vic_e) {
                    if !dev_set {
                        set_device(dev)?;
                        dev_set = true;
                    }
                    if let Some(ref vp) = lw.vram_ptrs_dev {
                        self.movers[d].queue_u64_h2d(
                            vp.as_u64() + (vic_e * 8) as u64,
                            0,
                            compute_stream,
                        )?;
                    }
                    if let Some(ref sp) = lw.spares_dev {
                        self.movers[d].queue_u64_h2d(
                            sp.as_u64() + (s * 8) as u64,
                            vic_ptr,
                            compute_stream,
                        )?;
                    }
                    let lc = &mut self.layers[l];
                    lc.slot_of_expert[vic_e] = -1;
                    lc.expert_of_slot[vic_slot] = -1;
                    lc.vram_ptrs_host[vic_e] = 0;
                    lc.res_mask_host[vic_e] = 0;
                    lc.spare_ptrs[s] = vic_ptr;
                    self.evictions_total += 1;
                } else if let Some((backup_slot, backup_hp, _)) = arena.reserve_backup_slot(l, vic_e)
                {
                    let Some(ev) = self.movers[d].event_pool.pop() else {
                        arena.cancel_backup_slot(backup_slot);
                        continue;
                    };
                    if !dev_set {
                        set_device(dev)?;
                        dev_set = true;
                    }
                    let blob_stride = self.layers[l].blob_stride;
                    let dst_slice =
                        unsafe { std::slice::from_raw_parts_mut(backup_hp, blob_stride) };
                    let res = unsafe {
                        copy_d2h_raw_async(
                            dst_slice,
                            vic_ptr as *const c_void,
                            &self.movers[d].copy_stream,
                        )
                    };
                    if let Err(err) = res {
                        arena.cancel_backup_slot(backup_slot);
                        self.movers[d].event_pool.push(ev);
                        return Err(err);
                    }
                    if let Err(err) = ev.record(&self.movers[d].copy_stream) {
                        arena.cancel_backup_slot(backup_slot);
                        self.movers[d].event_pool.push(ev);
                        return Err(err);
                    }
                    self.movers[d].draining.push_back(PendingD2h {
                        layer: l,
                        spare_idx: s,
                        vic_slot,
                        vic_e,
                        vic_ptr,
                        backup_slot,
                        ev,
                    });
                }
            }
        }

        Ok((promos_seen, routed_pairs, vram_hits))
    }

    pub fn drain_all(
        &mut self,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        s0: &CudaStream,
        s1: &CudaStream,
        db0: &mut Glm5Doorbell,
        db1: &mut Glm5Doorbell,
    ) -> Result<()> {
        set_device(0)?;
        s0.sync()?;
        set_device(1)?;
        s1.sync()?;
        self.finish_pending_d2h(0, weights, arena, s0, db0, true)?;
        self.finish_pending_d2h(1, weights, arena, s1, db1, true)?;
        let n_tot = self.layers.len();
        let sp = weights.split_layer;
        let p0 = self.ingest_promos(db0, 0..sp);
        if !p0.is_empty() {
            self.release_promoted_ram_sync(0, weights, arena, s0, &p0, false)?;
        }
        let p1 = self.ingest_promos(db1, sp..n_tot);
        if !p1.is_empty() {
            self.release_promoted_ram_sync(1, weights, arena, s1, &p1, false)?;
        }
        set_device(0)?;
        s0.sync()?;
        set_device(1)?;
        s1.sync()?;
        Ok(())
    }

    /// Decay stale heat across prompt boundaries while preserving LRU recency.
    pub fn on_new_prompt(&mut self) {
        for lc in &mut self.layers {
            if !lc.is_moe {
                continue;
            }
            for h in &mut lc.heat {
                *h *= 0.5;
            }
        }
    }

    pub fn decay_all(&mut self, decay: f32) {
        self.tick += 1;
        for lc in &mut self.layers {
            if !lc.is_moe {
                continue;
            }
            for h in &mut lc.heat {
                *h *= decay;
            }
        }
    }
}

impl LayerCacheMeta {
    pub fn note_route(&mut self, tick: u64, ids: &[i32]) {
        if !self.is_moe {
            return;
        }
        let n_exp = self.heat.len();
        for &raw_id in ids {
            if raw_id >= 0 && (raw_id as usize) < n_exp {
                let e = raw_id as usize;
                self.heat[e] += 1.0;
                let s = self.slot_of_expert[e];
                if s >= 0 {
                    self.last_used[s as usize] = tick;
                }
            }
        }
    }
}

pub struct LentPoolTail {
    pub dev: i32,
    pub base_ptr: u64,
    pub bytes: usize,
    pub saved_experts: Vec<(usize, usize, usize)>, // (layer, slot, expert_id)
    pub saved_spares: Vec<(usize, usize, u64)>,    // (layer, spare_idx, slot_ptr)
    pub temp_backups: Vec<(usize, usize, usize)>,  // (layer, expert_id, backup_slot)
}

struct PendingLendTail {
    dev: i32,
    lend_start: u64,
    lend_bytes: usize,
    saved_experts: Vec<(usize, usize, usize)>,
    saved_spares: Vec<(usize, usize, u64)>,
    touched_layers: Vec<usize>,
    pending_backups: Vec<(usize, usize, usize)>,
}

impl DynamicExpertCache {
    pub fn note_layer_route(&mut self, l: usize, ids: &[i32]) {
        self.layers[l].note_route(self.tick, ids);
    }

    fn prepare_lend_tail_async(
        &mut self,
        dev: i32,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        stream: &CudaStream,
        db: &mut Glm5Doorbell,
        need_bytes: usize,
    ) -> Result<PendingLendTail> {
        let d = dev as usize;
        set_device(dev)?;
        stream.sync()?;
        self.finish_pending_d2h(dev, weights, arena, stream, db, true)?;
        let sp = weights.split_layer;
        let range = if dev == 0 { 0..sp } else { sp..self.layers.len() };
        let promoted = self.ingest_promos(db, range);
        if !promoted.is_empty() {
            self.release_promoted_ram_sync(dev, weights, arena, stream, &promoted, true)?;
        }

        let pool_buf = &weights.exp_pool_dev[d];
        let pool_base = pool_buf.as_u64();
        let pool_bytes = pool_buf.bytes;
        let pool_end = pool_base + pool_bytes as u64;
        let lend_bytes = (need_bytes + 255) & !255;
        if lend_bytes > pool_bytes {
            anyhow::bail!(
                "lend_pool_tail(dev={dev}): need_bytes={lend_bytes} exceeds exp_pool_dev={pool_bytes}"
            );
        }
        let lend_start = pool_end - lend_bytes as u64;

        if dev == 1 {
            if let Some(mtp_l) = weights.cfg.mtp_layer {
                let mlc = &self.layers[mtp_l];
                if mlc.is_moe && mlc.total_pool_slots > 0 {
                    let mtp_end =
                        mlc.exp_pool_ptr + (mlc.total_pool_slots * mlc.blob_stride) as u64;
                    if lend_start < mtp_end {
                        anyhow::bail!(
                            "lend_pool_tail(dev=1): lend_start=0x{lend_start:x} overlaps blk.{mtp_l} MTP end=0x{mtp_end:x}"
                        );
                    }
                }
            }
        }

        let mut saved_experts = Vec::new();
        let mut saved_spares = Vec::new();
        let mut touched_layers = Vec::new();
        let mut pending_backups: Vec<(usize, usize, usize)> = Vec::new();

        for l in 0..self.layers.len() {
            let lc = &mut self.layers[l];
            if lc.device != dev || !lc.is_moe || lc.total_pool_slots == 0 || lc.blob_stride == 0 {
                continue;
            }
            let layer_end = lc.exp_pool_ptr + (lc.total_pool_slots * lc.blob_stride) as u64;
            if layer_end <= lend_start {
                continue;
            }
            let mut touched = false;
            for s in 0..lc.total_pool_slots {
                let slot_ptr = lc.exp_pool_ptr + (s * lc.blob_stride) as u64;
                if slot_ptr + lc.blob_stride as u64 <= lend_start {
                    continue;
                }
                touched = true;
                if lc.expert_of_slot[s] >= 0 {
                    let e = lc.expert_of_slot[s] as usize;
                    if !arena.is_in_ram(l, e) {
                        let Some((bs, bp, _)) = arena.reserve_lend_backup_slot(l, e) else {
                            let _ = self.movers[d].copy_stream.sync();
                            for (_, _, pbs) in pending_backups {
                                arena.cancel_backup_slot(pbs);
                            }
                            anyhow::bail!(
                                "lend_pool_tail(dev={dev}): backup pool exhausted while backing up l={l} e={e}"
                            );
                        };
                        let dst_slice =
                            unsafe { std::slice::from_raw_parts_mut(bp, lc.blob_stride) };
                        let res = unsafe {
                            copy_d2h_raw_async(
                                dst_slice,
                                slot_ptr as *const c_void,
                                &self.movers[d].copy_stream,
                            )
                        };
                        if let Err(err) = res {
                            let _ = self.movers[d].copy_stream.sync();
                            arena.cancel_backup_slot(bs);
                            for (_, _, pbs) in pending_backups {
                                arena.cancel_backup_slot(pbs);
                            }
                            return Err(err);
                        }
                        pending_backups.push((l, e, bs));
                    }
                    saved_experts.push((l, s, e));
                    lc.res_mask_host[e] = 0;
                    lc.vram_ptrs_host[e] = 0;
                    lc.slot_of_expert[e] = -1;
                    lc.expert_of_slot[s] = -1;
                }
                for si in 0..lc.spare_ptrs.len() {
                    if lc.spare_ptrs[si] == slot_ptr {
                        saved_spares.push((l, si, slot_ptr));
                        lc.spare_ptrs[si] = 0;
                    }
                }
            }
            if touched {
                touched_layers.push(l);
            }
        }

        Ok(PendingLendTail {
            dev,
            lend_start,
            lend_bytes,
            saved_experts,
            saved_spares,
            touched_layers,
            pending_backups,
        })
    }

    fn finish_lend_tail(
        &mut self,
        pending: PendingLendTail,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        stream: &CudaStream,
    ) -> Result<LentPoolTail> {
        let dev = pending.dev;
        let d = dev as usize;
        set_device(dev)?;
        if !pending.pending_backups.is_empty() {
            self.movers[d].copy_stream.sync()?;
            for &(l, e, bs) in &pending.pending_backups {
                let [d0, d1] = arena.commit_backup_slot(l, e, bs);
                let ram_dev_ptr = if dev == 0 { d0 } else { d1 };
                if let Some(ref rp) = weights.layers[l].ram_ptrs_dev {
                    self.movers[d].queue_u64_h2d(
                        rp.as_u64() + (e * 8) as u64,
                        ram_dev_ptr,
                        stream,
                    )?;
                }
            }
        }

        for &l in &pending.touched_layers {
            let ts = self.movers[d].next_table_slot();
            Self::upload_vram_ptrs_table(
                &self.layers[l],
                weights,
                l,
                &self.movers[d],
                ts,
                stream,
            )?;
            if let Some(ref sp_dev) = weights.layers[l].spares_dev {
                for (si, &val) in self.layers[l].spare_ptrs.iter().enumerate() {
                    self.movers[d].queue_u64_h2d(
                        sp_dev.as_u64() + (si * 8) as u64,
                        val,
                        stream,
                    )?;
                }
            }
        }
        stream.sync()?;

        Ok(LentPoolTail {
            dev,
            base_ptr: pending.lend_start,
            bytes: pending.lend_bytes,
            saved_experts: pending.saved_experts,
            saved_spares: pending.saved_spares,
            temp_backups: pending.pending_backups,
        })
    }

    pub fn lend_pool_tail(
        &mut self,
        dev: i32,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        stream: &CudaStream,
        db: &mut Glm5Doorbell,
        need_bytes: usize,
    ) -> Result<LentPoolTail> {
        let pending = self.prepare_lend_tail_async(dev, weights, arena, stream, db, need_bytes)?;
        self.finish_lend_tail(pending, weights, arena, stream)
    }

    pub fn lend_both_tails_parallel(
        &mut self,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        s0: &CudaStream,
        s1: &CudaStream,
        db0: &mut Glm5Doorbell,
        db1: &mut Glm5Doorbell,
        need_bytes0: usize,
        need_bytes1: usize,
    ) -> Result<(LentPoolTail, LentPoolTail)> {
        let p0 = self.prepare_lend_tail_async(0, weights, arena, s0, db0, need_bytes0)?;
        let p1 = match self.prepare_lend_tail_async(1, weights, arena, s1, db1, need_bytes1) {
            Ok(p) => p,
            Err(e) => {
                if let Ok(t0) = self.finish_lend_tail(p0, weights, arena, s0) {
                    let _ = self.return_both_tails_parallel(Some(t0), None, weights, arena, s0, s1);
                }
                return Err(e);
            }
        };
        let tail0 = self.finish_lend_tail(p0, weights, arena, s0)?;
        let tail1 = match self.finish_lend_tail(p1, weights, arena, s1) {
            Ok(t) => t,
            Err(e) => {
                let _ = self.return_both_tails_parallel(Some(tail0), None, weights, arena, s0, s1);
                return Err(e);
            }
        };
        Ok((tail0, tail1))
    }

    pub fn return_both_tails_parallel(
        &mut self,
        tail0: Option<LentPoolTail>,
        tail1: Option<LentPoolTail>,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        s0: &CudaStream,
        s1: &CudaStream,
    ) -> Result<()> {
        let mut touched_dev: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let tails = [tail0, tail1];
        for (d, tail_opt) in tails.iter().enumerate() {
            let Some(ref tail) = tail_opt else {
                continue;
            };
            let stream = if d == 0 { s0 } else { s1 };
            set_device(d as i32)?;
            stream.sync()?;
            for &(l, s, e) in &tail.saved_experts {
                let lc = &mut self.layers[l];
                let slot_ptr = lc.exp_pool_ptr + (s * lc.blob_stride) as u64;
                let Some((hp, _)) = arena.get_ram_blob(l, e, d as i32) else {
                    anyhow::bail!(
                        "return_pool_tail(dev={d}): missing RAM blob for l={l} e={e}"
                    );
                };
                let src_slice = unsafe { std::slice::from_raw_parts(hp, lc.blob_stride) };
                unsafe {
                    copy_h2d_raw_async(
                        slot_ptr as *mut c_void,
                        src_slice,
                        &self.movers[d].copy_stream,
                    )?;
                }
                lc.slot_of_expert[e] = s as i16;
                lc.expert_of_slot[s] = e as i16;
                lc.vram_ptrs_host[e] = slot_ptr;
                lc.res_mask_host[e] = 1;
                if !touched_dev[d].contains(&l) {
                    touched_dev[d].push(l);
                }
            }
            for &(l, si, slot_ptr) in &tail.saved_spares {
                let lc = &mut self.layers[l];
                if si < lc.spare_ptrs.len() {
                    lc.spare_ptrs[si] = slot_ptr;
                }
                if !touched_dev[d].contains(&l) {
                    touched_dev[d].push(l);
                }
            }
        }

        for (d, tail_opt) in tails.iter().enumerate() {
            let Some(ref tail) = tail_opt else {
                continue;
            };
            let stream = if d == 0 { s0 } else { s1 };
            set_device(d as i32)?;
            self.movers[d].copy_stream.sync()?;
            for &(l, e, bs) in &tail.temp_backups {
                arena.release_temp_backup_slot(l, e, bs);
                if let Some(ref rp) = weights.layers[l].ram_ptrs_dev {
                    self.movers[d].queue_u64_h2d(rp.as_u64() + (e * 8) as u64, 0, stream)?;
                }
            }
            for &l in &touched_dev[d] {
                let ts = self.movers[d].next_table_slot();
                Self::upload_vram_ptrs_table(
                    &self.layers[l],
                    weights,
                    l,
                    &self.movers[d],
                    ts,
                    stream,
                )?;
                if let Some(ref sp_dev) = weights.layers[l].spares_dev {
                    for (si, &val) in self.layers[l].spare_ptrs.iter().enumerate() {
                        self.movers[d].queue_u64_h2d(
                            sp_dev.as_u64() + (si * 8) as u64,
                            val,
                            stream,
                        )?;
                    }
                }
            }
            stream.sync()?;
        }
        Ok(())
    }

    pub fn return_pool_tail(
        &mut self,
        tail: LentPoolTail,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        stream: &CudaStream,
    ) -> Result<()> {
        if tail.dev == 0 {
            self.return_both_tails_parallel(Some(tail), None, weights, arena, stream, stream)
        } else {
            self.return_both_tails_parallel(None, Some(tail), weights, arena, stream, stream)
        }
    }

    /// One-shot end-of-prefill working-set warmup: promotes up to `max_per_layer` hottest cold experts
    /// per layer onto GPU 0 and GPU 1 directly from RegisteredMappedHostBuffer, saving evicted
    /// VRAM-only victims directly into BackupSlotPool with zero host bounce memcpy.
    pub fn warmup_from_prefill_heat(
        &mut self,
        weights: &Glm5ModelWeights,
        arena: &CpuColdArena,
        s0: &CudaStream,
        s1: &CudaStream,
        db0: &mut Glm5Doorbell,
        db1: &mut Glm5Doorbell,
        max_per_layer: usize,
    ) -> Result<usize> {
        if self.freeze || max_per_layer == 0 {
            return Ok(0);
        }
        self.drain_all(weights, arena, s0, s1, db0, db1)?;

        let min_cold_h = 2.5f32;
        let n_expert = weights.cfg.n_expert;
        let mut swaps: Vec<(usize, usize, usize, usize)> = Vec::new(); // (layer, slot, old_e, new_e)
        let mut slots_lru: Vec<(u64, usize, usize)> = Vec::new();

        for (l, lc) in self.layers.iter().enumerate() {
            if !lc.is_moe || lc.cap_slots == 0 || lc.cap_slots >= n_expert {
                continue;
            }
            let mut cold_list: Vec<(f32, usize)> = (0..n_expert)
                .filter(|&e| {
                    lc.slot_of_expert[e] < 0 && lc.heat[e] >= min_cold_h && arena.is_in_ram(l, e)
                })
                .map(|e| (lc.heat[e], e))
                .collect();
            if cold_list.is_empty() {
                continue;
            }
            cold_list.sort_by(|a, b| b.0.total_cmp(&a.0));

            slots_lru.clear();
            for s in 0..lc.total_pool_slots {
                let oe = lc.expert_of_slot[s];
                if oe >= 0 {
                    slots_lru.push((lc.last_used[s], s, oe as usize));
                }
            }
            let w = slots_lru.len().min(64);
            if w == 0 {
                continue;
            }
            if w < slots_lru.len() {
                slots_lru.select_nth_unstable_by_key(w - 1, |&x| x.0);
            }
            let mut vic_cands: Vec<(bool, f32, u64, usize, usize)> = slots_lru[..w]
                .iter()
                .filter_map(|&(lu, s, oe)| {
                    let in_ram = arena.is_in_ram(l, oe);
                    if !in_ram && arena.backup_free_count_for_layer(l) == 0 {
                        return None;
                    }
                    Some((in_ram, lc.heat[oe], lu, s, oe))
                })
                .collect();
            vic_cands.sort_by(|a, b| {
                a.1.total_cmp(&b.1)
                    .then_with(|| a.2.cmp(&b.2))
            });

            let n_pick = max_per_layer.min(cold_list.len()).min(vic_cands.len());
            for i in 0..n_pick {
                let (cold_h, new_e) = cold_list[i];
                let (_, vic_h, _, slot, old_e) = vic_cands[i];
                if cold_h >= vic_h * 1.35 + 1.0 {
                    swaps.push((l, slot, old_e, new_e));
                }
            }
        }

        if swaps.is_empty() {
            return Ok(0);
        }

        let mut swaps_dev: [Vec<(usize, usize, usize, usize)>; 2] = [Vec::new(), Vec::new()];
        for sw in swaps {
            let d = self.layers[sw.0].device as usize;
            swaps_dev[d].push(sw);
        }

        let max_len = swaps_dev[0].len().max(swaps_dev[1].len());
        let mut dirty_dev: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let mut ram_updates: [Vec<(usize, usize, u64)>; 2] = [Vec::new(), Vec::new()];
        let mut total_promoted = 0usize;

        let mut off = 0usize;
        while off < max_len {
            // Wave step 1: async direct D2H for VRAM-only victims into pre-allocated Backup Pool
            let mut d2h_slots: [Vec<Option<usize>>; 2] = [Vec::new(), Vec::new()];
            let mut d2h_issued = [false; 2];
            for d in 0..2 {
                let end = (off + WARMUP_WAVE_SLOTS).min(swaps_dev[d].len());
                if off >= end {
                    continue;
                }
                set_device(d as i32)?;
                for &(l, slot, old_e, _) in &swaps_dev[d][off..end] {
                    if arena.is_in_ram(l, old_e) {
                        d2h_slots[d].push(None);
                        continue;
                    }
                    let Some((bs, bp, _)) = arena.reserve_backup_slot(l, old_e) else {
                        d2h_slots[d].push(None);
                        continue;
                    };
                    let lc = &self.layers[l];
                    let vic_ptr = lc.exp_pool_ptr + (slot * lc.blob_stride) as u64;
                    let dst_slice = unsafe { std::slice::from_raw_parts_mut(bp, lc.blob_stride) };
                    let res = unsafe {
                        copy_d2h_raw_async(
                            dst_slice,
                            vic_ptr as *const c_void,
                            &self.movers[d].copy_stream,
                        )
                    };
                    if let Err(err) = res {
                        arena.cancel_backup_slot(bs);
                        return Err(err);
                    }
                    d2h_slots[d].push(Some(bs));
                    d2h_issued[d] = true;
                }
            }
            for d in 0..2 {
                if !d2h_issued[d] {
                    continue;
                }
                let end = (off + WARMUP_WAVE_SLOTS).min(swaps_dev[d].len());
                set_device(d as i32)?;
                self.movers[d].copy_stream.sync()?;
                for (bi, &(l, _slot, old_e, _)) in swaps_dev[d][off..end].iter().enumerate() {
                    if let Some(bs) = d2h_slots[d][bi] {
                        let [d0, d1] = arena.commit_backup_slot(l, old_e, bs);
                        ram_updates[d].push((l, old_e, if d == 0 { d0 } else { d1 }));
                    }
                }
            }

            // Wave step 2: zero-bounce direct H2D from RegisteredMappedHostBuffer into packed VRAM slot
            let mut promoted_wave: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
            for d in 0..2 {
                let end = (off + WARMUP_WAVE_SLOTS).min(swaps_dev[d].len());
                if off >= end {
                    continue;
                }
                set_device(d as i32)?;
                for &(l, slot, old_e, new_e) in &swaps_dev[d][off..end] {
                    if !arena.is_in_ram(l, old_e) {
                        continue;
                    }
                    let Some((hp, _)) = arena.get_ram_blob(l, new_e, d as i32) else {
                        continue;
                    };
                    let lc = &mut self.layers[l];
                    let slot_ptr = lc.exp_pool_ptr + (slot * lc.blob_stride) as u64;
                    let host_slice = unsafe { std::slice::from_raw_parts(hp, lc.blob_stride) };
                    unsafe {
                        copy_h2d_raw_async(
                            slot_ptr as *mut c_void,
                            host_slice,
                            &self.movers[d].copy_stream,
                        )?;
                    }
                    lc.slot_of_expert[old_e] = -1;
                    lc.vram_ptrs_host[old_e] = 0;
                    lc.res_mask_host[old_e] = 0;

                    lc.slot_of_expert[new_e] = slot as i16;
                    lc.expert_of_slot[slot] = new_e as i16;
                    lc.last_used[slot] = self.tick;
                    lc.vram_ptrs_host[new_e] = slot_ptr;
                    lc.res_mask_host[new_e] = 1;
                    lc.heat[new_e] = lc.heat[new_e].max(3.5);

                    if !dirty_dev[d].contains(&l) {
                        dirty_dev[d].push(l);
                    }
                    promoted_wave[d].push((l, new_e));
                    self.evictions_total += 1;
                    self.promotions_total += 1;
                    total_promoted += 1;
                }
            }
            for d in 0..2 {
                let end = (off + WARMUP_WAVE_SLOTS).min(swaps_dev[d].len());
                if off < end {
                    set_device(d as i32)?;
                    self.movers[d].copy_stream.sync()?;
                    for &(l, new_e) in &promoted_wave[d] {
                        if !self.keep_cold_ram && arena.release_ram_slot(l, new_e) {
                            ram_updates[d].push((l, new_e, 0u64));
                        }
                    }
                }
            }
            off += WARMUP_WAVE_SLOTS;
        }

        for (d, stream) in [(0i32, s0), (1i32, s1)] {
            let dd = &dirty_dev[d as usize];
            let ru = &ram_updates[d as usize];
            if dd.is_empty() && ru.is_empty() {
                continue;
            }
            set_device(d)?;
            for &(l, old_e, ram_dev_ptr) in ru {
                if let Some(ref rp) = weights.layers[l].ram_ptrs_dev {
                    self.movers[d as usize].queue_u64_h2d(
                        rp.as_u64() + (old_e * 8) as u64,
                        ram_dev_ptr,
                        stream,
                    )?;
                }
            }
            for &l in dd {
                let ts = self.movers[d as usize].next_table_slot();
                Self::upload_vram_ptrs_table(
                    &self.layers[l],
                    weights,
                    l,
                    &self.movers[d as usize],
                    ts,
                    stream,
                )?;
            }
            stream.sync()?;
        }

        Ok(total_promoted)
    }

    pub fn export_profile(&self, arch: &str, n_layers: usize, n_expert: usize) -> CacheProfile {
        let mut ranked_experts = vec![Vec::with_capacity(n_expert); n_layers];
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (l, lc) in self.layers.iter().enumerate() {
            if l >= n_layers {
                break;
            }
            if !lc.is_moe {
                ranked_experts[l] = (0..n_expert).collect();
                continue;
            }
            let mut order: Vec<usize> = (0..n_expert).collect();
            order.sort_by(|&a, &b| {
                let ra = lc.res_mask_host[a];
                let rb = lc.res_mask_host[b];
                rb.cmp(&ra).then_with(|| {
                    lc.heat[b]
                        .partial_cmp(&lc.heat[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });
            for &e in &order {
                let sc = lc.heat[e] + if lc.res_mask_host[e] != 0 { 1e6 } else { 0.0 };
                pairs.push((sc, l, e));
            }
            ranked_experts[l] = order;
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let global_ranked = pairs.into_iter().map(|(_, l, e)| (l, e)).collect();
        CacheProfile {
            arch: arch.to_owned(),
            n_layer_phys: n_layers,
            n_expert,
            ranked_experts,
            global_ranked,
        }
    }
}

/// Alias view into a backing `DeviceBuffer` so mutually exclusive scratch phases share VRAM cleanly.
pub struct GpuRegion {
    pub device: i32,
    pub ptr: *mut c_void,
    pub bytes: usize,
}
unsafe impl Send for GpuRegion {}
unsafe impl Sync for GpuRegion {}

impl GpuRegion {
    pub fn sub(buf: &DeviceBuffer, off: usize, bytes: usize) -> Self {
        assert!(off + bytes <= buf.bytes, "GpuRegion out of bounds: {off}+{bytes} > {}", buf.bytes);
        Self {
            device: buf.device,
            ptr: unsafe { (buf.ptr as *mut u8).add(off).cast::<c_void>() },
            bytes,
        }
    }

    pub fn zero_async(&self, stream: &CudaStream) -> Result<()> {
        unsafe { ampllm_cuda_kernels::zero_raw_async(self.ptr, self.bytes, stream) }
    }

    pub fn copy_from_host_async(&self, src: &[u8], stream: &CudaStream) -> Result<()> {
        if src.len() > self.bytes {
            anyhow::bail!("GpuRegion copy_from_host_async overflow: {} > {}", src.len(), self.bytes);
        }
        unsafe { copy_h2d_raw_async(self.ptr, src, stream) }
    }

    pub fn copy_to_host_async(&self, dst: &mut [u8], stream: &CudaStream) -> Result<()> {
        if dst.len() > self.bytes {
            anyhow::bail!("GpuRegion copy_to_host_async overflow: {} > {}", dst.len(), self.bytes);
        }
        unsafe { ampllm_cuda_kernels::copy_d2h_raw_async(dst, self.ptr, stream) }
    }
}
