use crate::config::EngineConfig;
use crate::kv_cache::KvConvCache;
use crate::rank_state::{GpuRankState, StaticMixPtrTables};
use crate::scratch::ActivationScratch;
use crate::weights::ModelWeights;
use ampllm_cache::{
    layer_expert_sizes, plan_layer_capacities, resolve_auto_budget_bytes, CacheBudget,
    CacheProfile, CacheStats, EwmaTracker, HostExpertArena, VramExpertCache,
};
use ampllm_cpu_kernels::{
    run_cpu_moe_batch, run_cpu_moe_m1, CpuBatchSlot, CpuExpertJob, CpuMoeScratch, CpuThreadPool,
};
use ampllm_cuda_kernels::cuda_rt::{copy_d2h_raw_async, copy_h2d_raw_async};
use ampllm_cuda_kernels::ffi::{
    launch_add_f32, launch_moe_grouped_f32, launch_p2p_push, launch_p2p_wait_reduce,
};
use ampllm_cuda_kernels::{
    mem_info, set_device, CudaEvent, CudaStream, DeviceBuffer, PinnedMappedBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Inkling2Config};
use anyhow::{bail, Result};
use std::ffi::c_void;

struct HelperGpu1Scratch {
    act_q8_1: DeviceBuffer,
    expert_ids: DeviceBuffer,
    expert_weights: DeviceBuffer,
    moe_inter_f32: DeviceBuffer,
    moe_inter_q8_1: DeviceBuffer,
    moe_slot_out: DeviceBuffer,
    moe_out: DeviceBuffer,
}

impl HelperGpu1Scratch {
    fn alloc(cfg: &Inkling2Config, device: i32, max_batch: usize) -> Result<Self> {
        let q_dim = cfg.n_head * cfg.head_dim;
        Ok(Self {
            act_q8_1: DeviceBuffer::alloc(device, max_batch * (q_dim.max(2 * cfg.n_embd) / 32) * 36)?,
            expert_ids: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * 4)?,
            expert_weights: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * 4)?,
            moe_inter_f32: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * cfg.n_ff_exp * 4)?,
            moe_inter_q8_1: DeviceBuffer::alloc(
                device,
                max_batch * cfg.n_expert_used * (cfg.n_ff_exp / 32) * 36,
            )?,
            moe_slot_out: DeviceBuffer::alloc(device, max_batch * cfg.n_expert_used * cfg.n_embd * 4)?,
            moe_out: DeviceBuffer::alloc(device, max_batch * cfg.n_embd * 4)?,
        })
    }
}

/// Dual-RTX-3090 Helper Mode: GPU 0 owns dense + primary cache; GPU 1 owns disjoint expert cache;
/// CPU thread pool computes any remaining cold misses, overlapping GPU 0 || GPU 1 || CPU concurrently.
pub struct HelperGpuEngine {
    pub engine_cfg: EngineConfig,
    pub rank0: GpuRankState,
    pub cache0: VramExpertCache,
    pub cache1: VramExpertCache,
    scales_gpu1: Vec<(Option<DeviceBuffer>, Option<DeviceBuffer>)>,
    stream1: CudaStream,
    scratch1: HelperGpu1Scratch,
    pub tracker: EwmaTracker,
    arena: Option<HostExpertArena>,
    cpu_pool: CpuThreadPool,
    cpu_scratch: CpuMoeScratch,

    // Shared pinned host buffers & device pointers mapped on both GPU 0 and GPU 1
    act_q8_pin: PinnedMappedBuffer<u8>,
    router_ids_pin: PinnedMappedBuffer<i32>,
    router_w_pin: PinnedMappedBuffer<f32>,
    hidden_pin: PinnedMappedBuffer<f32>,
    cpu_out_pin: PinnedMappedBuffer<f32>,
    _helper_out_pin: PinnedMappedBuffer<f32>,
    helper_out_dev0: *mut c_void,
    helper_out_dev1: *mut c_void,
    _helper_flag_pin: PinnedMappedBuffer<u32>,
    helper_flag_dev0: *mut c_void,
    helper_flag_dev1: *mut c_void,
    router_ready_event0: CudaEvent,
    helper_flush_event1: CudaEvent,
    compute_done_event0: CudaEvent,
    helper_seq: u32,
    pub cur_pos: usize,
}

unsafe impl Send for HelperGpuEngine {}

impl HelperGpuEngine {
    pub fn load(gguf: &GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let dev0 = 0;
        let dev1 = 1;
        set_device(dev0)?;
        let stream0 = CudaStream::new(dev0)?;
        let weights = ModelWeights::load_non_experts(gguf, &cfg, dev0, &stream0)?;
        let kv_cache = KvConvCache::alloc(&cfg, dev0, engine_cfg.max_ctx, engine_cfg.kv_mode)?;
        let scratch = ActivationScratch::alloc(&cfg, dev0, engine_cfg.max_batch)?;
        let mix_tables = StaticMixPtrTables::build(&cfg, &scratch, dev0)?;
        stream0.sync()?;

        set_device(dev1)?;
        let stream1 = CudaStream::new(dev1)?;
        let scratch1 = HelperGpu1Scratch::alloc(&cfg, dev1, engine_cfg.max_batch)?;
        let mut scales_gpu1 = Vec::with_capacity(cfg.n_layer_phys);
        for i in 0..cfg.n_layer_phys {
            let lw = weights.layer(i);
            let g_sc = if lw.gate_up_scale.is_some() {
                let b = DeviceBuffer::alloc(dev1, cfg.n_expert * 4)?;
                b.copy_from_host(bytemuck::cast_slice(&lw.gate_up_scale_host))?;
                Some(b)
            } else {
                None
            };
            let d_sc = if lw.down_scale.is_some() {
                let b = DeviceBuffer::alloc(dev1, cfg.n_expert * 4)?;
                b.copy_from_host(bytemuck::cast_slice(&lw.down_scale_host))?;
                Some(b)
            } else {
                None
            };
            scales_gpu1.push((g_sc, d_sc));
        }
        stream1.sync()?;

        let exp_sizes = layer_expert_sizes(gguf, &cfg)?;
        let max_half = vec![cfg.n_expert / 2; cfg.n_layer_phys];

        let (budget0, max_caps0) = match engine_cfg.expert_cache {
            CacheBudget::Auto => (resolve_auto_budget_bytes(dev0, 1024)?, Some(max_half)),
            CacheBudget::Mb(mb) => (mb * 1024 * 1024, None),
        };
        let caps0 = plan_layer_capacities(&cfg, &exp_sizes, budget0, max_caps0.as_deref());

        let profile = match &engine_cfg.load_profile {
            Some(path) => Some(CacheProfile::load(path)?),
            None => None,
        };
        let mut tracker = EwmaTracker::new(cfg.n_layer_phys, cfg.n_expert, 0.98);
        if let Some(ref prof) = profile {
            tracker.seed_from_profile(prof);
        }

        let cache0 = VramExpertCache::init(
            gguf,
            &cfg,
            dev0,
            &caps0,
            &exp_sizes,
            profile.as_ref(),
            None,
            None,
            &stream0,
        )?;
        let mask0 = cache0.resident_mask();
        let rem_max1: Vec<usize> = (0..cfg.n_layer_phys)
            .map(|i| cfg.n_expert.saturating_sub(caps0[i]))
            .collect();

        let auto1_cap = resolve_auto_budget_bytes(dev1, 1024)?;
        let budget1 = match engine_cfg.expert_cache {
            CacheBudget::Auto => auto1_cap,
            CacheBudget::Mb(mb) => (mb * 1024 * 1024).min(auto1_cap),
        };
        let caps1 = plan_layer_capacities(&cfg, &exp_sizes, budget1, Some(&rem_max1));
        let cache1 = VramExpertCache::init(
            gguf,
            &cfg,
            dev1,
            &caps1,
            &exp_sizes,
            profile.as_ref(),
            Some(&mask0),
            None,
            &stream1,
        )?;
        let mask1 = cache1.resident_mask();

        let (free1, _) = mem_info(dev1)?;
        if free1 < 768 * 1024 * 1024 {
            bail!(
                "GPU 1 free VRAM ({:.1} MiB) dropped below 768 MiB display safety reserve",
                free1 as f64 / (1024.0 * 1024.0)
            );
        }

        // Host arena ONLY needed for layers where caps0[i] + caps1[i] < n_expert
        let layer_needed: Vec<bool> = (0..cfg.n_layer_phys)
            .map(|i| caps0[i] + caps1[i] < cfg.n_expert)
            .collect();
        let arena = if layer_needed.iter().any(|&b| b) {
            Some(HostExpertArena::from_gguf(gguf, &cfg, &layer_needed, &exp_sizes)?)
        } else {
            None
        };

        let cpu_pool = CpuThreadPool::new(engine_cfg.cpu_threads);
        let cpu_scratch = CpuMoeScratch::new(cfg.n_embd, cfg.n_ff_exp, cfg.n_expert_used);

        let max_b = engine_cfg.max_batch;
        let act_q8_pin = PinnedMappedBuffer::<u8>::alloc(
            (max_b * (cfg.n_embd / 32) * 36).max(2 * cfg.n_embd * 2),
        )?;
        let router_ids_pin = PinnedMappedBuffer::<i32>::alloc(max_b * cfg.n_expert_used)?;
        let router_w_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_expert_used)?;
        let hidden_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_embd)?;
        let cpu_out_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_embd)?;
        let helper_out_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_embd)?;
        let helper_flag_pin = PinnedMappedBuffer::<u32>::alloc(1)?;

        let helper_out_dev0 = helper_out_pin.dev_ptr_for_device(dev0)?;
        let helper_out_dev1 = helper_out_pin.dev_ptr_for_device(dev1)?;
        let helper_flag_dev0 = helper_flag_pin.dev_ptr_for_device(dev0)?;
        let helper_flag_dev1 = helper_flag_pin.dev_ptr_for_device(dev1)?;

        let router_ready_event0 = CudaEvent::new(dev0)?;
        let compute_done_event0 = CudaEvent::new(dev0)?;
        let helper_flush_event1 = CudaEvent::new(dev1)?;
        let _ = mask1;

        set_device(dev0)?;
        Ok(Self {
            engine_cfg,
            rank0: GpuRankState {
                device: dev0,
                stream: stream0,
                cfg,
                weights,
                kv_cache,
                scratch,
                mix_tables,
            },
            cache0,
            cache1,
            scales_gpu1,
            stream1,
            scratch1,
            tracker,
            arena,
            cpu_pool,
            cpu_scratch,
            act_q8_pin,
            router_ids_pin,
            router_w_pin,
            hidden_pin,
            cpu_out_pin,
            _helper_out_pin: helper_out_pin,
            helper_out_dev0,
            helper_out_dev1,
            _helper_flag_pin: helper_flag_pin,
            helper_flag_dev0,
            helper_flag_dev1,
            router_ready_event0,
            helper_flush_event1,
            compute_done_event0,
            helper_seq: 0,
            cur_pos: 0,
        })
    }

    pub fn cache_stats(&self) -> &CacheStats {
        &self.tracker.stats
    }

    pub fn cache_summary(&self) -> String {
        let total = self.rank0.cfg.n_layer_phys * self.rank0.cfg.n_expert;
        let s0 = self.cache0.total_slots();
        let s1 = self.cache1.total_slots();
        let m0 = self.cache0.total_vram_bytes() as f64 / (1024.0 * 1024.0);
        let m1 = self.cache1.total_vram_bytes() as f64 / (1024.0 * 1024.0);
        let free1_mb = mem_info(1).map(|(f, _)| f as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
        format!(
            "Helper(GPU0+GPU1+CPU) slots=({s0}+{s1})/{total} vram=({m0:.1}+{m1:.1}) MiB gpu1_free={free1_mb:.1} MiB kv={:?}",
            self.rank0.kv_cache.kv_mode
        )
    }

    pub fn save_profile(&self) -> Result<()> {
        if let Some(ref path) = self.engine_cfg.save_profile {
            self.tracker.export_profile(&self.rank0.cfg.arch).save(path)?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        set_device(0)?;
        self.rank0.kv_cache.reset()?;
        self.cur_pos = 0;
        Ok(())
    }

    pub fn forward_tokens(
        &mut self,
        tokens: &[u32],
        capture_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>)> {
        if tokens.is_empty() {
            bail!("forward_tokens called with empty token slice");
        }
        let max_batch = self.rank0.scratch.max_batch;
        if tokens.len() > max_batch {
            let mut off = 0;
            while off + max_batch < tokens.len() {
                self.forward_chunk(&tokens[off..off + max_batch], false)?;
                off += max_batch;
            }
            return self.forward_chunk(&tokens[off..], capture_logits);
        }
        self.forward_chunk(tokens, capture_logits)
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        capture_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>)> {
        let m_usize = tokens.len();
        let m = m_usize as i32;
        let n_embd = self.rank0.cfg.n_embd;
        let topk = self.rank0.cfg.n_expert_used;
        let max_kv_len = (self.cur_pos + m_usize) as i32;

        let tok_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i32: Vec<i32> = (self.cur_pos..self.cur_pos + m_usize)
            .map(|p| p as i32)
            .collect();

        set_device(0)?;
        self.rank0.enqueue_inputs(&tok_i32, &pos_i32)?;

        let mut recent_active = vec![Vec::<usize>::new(); self.rank0.cfg.n_layer_phys];
        let exec_order = self.rank0.cfg.execution_order();
        let q8_bytes = m_usize * (n_embd / 32) * 36;
        let id_bytes = m_usize * topk * 4;
        let w_bytes = m_usize * topk * 4;

        for &slot in &exec_order {
            let ip = self.rank0.cfg.logical_layers[slot].phys_layer;
            set_device(0)?;
            self.rank0.enqueue_attn_through_wo(slot, m, max_kv_len, true)?;
            self.rank0.enqueue_post_wo_to_router(slot, m)?;

            let layer_w_host = self.rank0.weights.layer(ip);
            let use_mma_limbs = m_usize == 1
                && layer_w_host.gate_up_type == GgmlType::Nvfp4
                && layer_w_host.down_type == GgmlType::Nvfp4
                && n_embd % 64 == 0
                && self.rank0.cfg.n_ff_exp % 64 == 0;
            let act_xfer_bytes = if use_mma_limbs {
                2 * n_embd * 2
            } else {
                q8_bytes
            };
            let act_src_ptr0 = if use_mma_limbs {
                self.rank0.scratch.moe_inter_f32.ptr
            } else {
                self.rank0.scratch.act_q8_1.ptr
            };

            // Export router outputs + Q8_1 or FP16 limbs (and F32 hidden only when CPU fallback exists)
            let need_cpu_buf = self.arena.as_ref().is_some_and(|a| a.has_layer(ip));
            unsafe {
                copy_d2h_raw_async(
                    &mut self.act_q8_pin.as_mut_slice()[..act_xfer_bytes],
                    act_src_ptr0,
                    &self.rank0.stream,
                )?;
                copy_d2h_raw_async(
                    bytemuck::cast_slice_mut(&mut self.router_ids_pin.as_mut_slice()[..m_usize * topk]),
                    self.rank0.scratch.expert_ids.ptr,
                    &self.rank0.stream,
                )?;
                copy_d2h_raw_async(
                    bytemuck::cast_slice_mut(&mut self.router_w_pin.as_mut_slice()[..m_usize * topk]),
                    self.rank0.scratch.expert_weights.ptr,
                    &self.rank0.stream,
                )?;
                if need_cpu_buf {
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut self.hidden_pin.as_mut_slice()[..m_usize * n_embd]),
                        self.rank0.scratch.tmp_hidden.ptr,
                        &self.rank0.stream,
                    )?;
                }
            }
            self.router_ready_event0.record(&self.rank0.stream)?;

            // GPU 0 runs its resident experts asynchronously
            let (gu0, dn0) = self.cache0.layer_ptrs_dev(ip);
            self.rank0.enqueue_moe_local(ip, gu0, dn0, m);

            self.router_ready_event0.sync()?;

            self.tracker.decay_layer(ip);
            let ids = &self.router_ids_pin.as_slice()[..m_usize * topk];
            let ws = &self.router_w_pin.as_slice()[..m_usize * topk];

            let mut any_gpu1 = false;
            let mut any_cpu = false;
            let mut cpu_jobs: Vec<CpuExpertJob<'_>> = Vec::new();
            let mut cpu_slots: Vec<CpuBatchSlot<'_>> = Vec::new();
            let layer_w_host = self.rank0.weights.layer(ip);

            for tok_idx in 0..m_usize {
                for ks in 0..topk {
                    let s_idx = tok_idx * topk + ks;
                    let e = ids[s_idx] as usize;
                    self.tracker.record_activation(ip, e);
                    if !recent_active[ip].contains(&e) {
                        recent_active[ip].push(e);
                    }
                    if self.cache0.is_resident(ip, e) {
                        self.tracker.stats.hits += 1;
                    } else if self.cache1.is_resident(ip, e) {
                        self.tracker.stats.helper_hits += 1;
                        any_gpu1 = true;
                    } else {
                        self.tracker.stats.misses += 1;
                        any_cpu = true;
                        let arena = self.arena.as_ref().expect("host arena required for cold expert");
                        if m_usize == 1 {
                            cpu_jobs.push(CpuExpertJob {
                                gate_up: arena.expert_gate_up(ip, e),
                                down: arena.expert_down(ip, e),
                                gate_up_scale: layer_w_host.gate_up_scale_host[e],
                                weight: ws[s_idx] * layer_w_host.down_scale_host[e],
                            });
                        } else {
                            cpu_slots.push(CpuBatchSlot {
                                token_idx: tok_idx,
                                gate_up: arena.expert_gate_up(ip, e),
                                down: arena.expert_down(ip, e),
                                gate_up_scale: layer_w_host.gate_up_scale_host[e],
                                weight: ws[s_idx] * layer_w_host.down_scale_host[e],
                            });
                        }
                    }
                }
            }

            let mut active_seq: Option<u32> = None;
            if any_gpu1 {
                self.helper_seq += 1;
                let seq32 = self.helper_seq;
                active_seq = Some(seq32);

                set_device(1)?;
                unsafe {
                    let act_dst_ptr1 = if use_mma_limbs {
                        self.scratch1.moe_inter_f32.ptr
                    } else {
                        self.scratch1.act_q8_1.ptr
                    };
                    copy_h2d_raw_async(
                        act_dst_ptr1,
                        &self.act_q8_pin.as_slice()[..act_xfer_bytes],
                        &self.stream1,
                    )?;
                    copy_h2d_raw_async(
                        self.scratch1.expert_ids.ptr,
                        &bytemuck::cast_slice::<i32, u8>(self.router_ids_pin.as_slice())[..id_bytes],
                        &self.stream1,
                    )?;
                    copy_h2d_raw_async(
                        self.scratch1.expert_weights.ptr,
                        &bytemuck::cast_slice::<f32, u8>(self.router_w_pin.as_slice())[..w_bytes],
                        &self.stream1,
                    )?;
                    let layer_w = self.rank0.weights.layer(ip);
                    let (gu1, dn1) = self.cache1.layer_ptrs_dev(ip);
                    let gu_sc_ptr = self.scales_gpu1[ip]
                        .0
                        .as_ref()
                        .map(|b| b.ptr as *const f32)
                        .unwrap_or(std::ptr::null());
                    let dn_sc_ptr = self.scales_gpu1[ip]
                        .1
                        .as_ref()
                        .map(|b| b.ptr as *const f32)
                        .unwrap_or(std::ptr::null());
                    launch_moe_grouped_f32(
                        gu1 as *const u64,
                        dn1 as *const u64,
                        gu_sc_ptr,
                        dn_sc_ptr,
                        layer_w.gate_up_row_bytes,
                        layer_w.down_row_bytes,
                        layer_w.gate_up_type as i32,
                        layer_w.down_type as i32,
                        self.scratch1.act_q8_1.ptr,
                        if use_mma_limbs { 1usize as *const f32 } else { std::ptr::null() },
                        self.scratch1.expert_ids.ptr as *const i32,
                        self.scratch1.expert_weights.ptr as *const f32,
                        self.scratch1.moe_inter_f32.ptr as *mut f32,
                        self.scratch1.moe_inter_q8_1.ptr,
                        self.scratch1.moe_slot_out.ptr as *mut f32,
                        self.scratch1.moe_out.ptr as *mut f32,
                        n_embd as i32,
                        self.rank0.cfg.n_ff_exp as i32,
                        topk as i32,
                        m,
                        self.rank0.cfg.situ_beta,
                        self.rank0.cfg.situ_linear_beta,
                        self.stream1.raw,
                    );
                    launch_p2p_push(
                        self.scratch1.moe_out.ptr as *const f32,
                        self.helper_out_dev1 as *mut f32,
                        self.helper_flag_dev1 as *mut u32,
                        seq32,
                        (m_usize * n_embd) as i32,
                        self.stream1.raw,
                    );
                }
                self.helper_flush_event1.record(&self.stream1)?;
                let _ = self.helper_flush_event1.is_ready()?;
            }

            if any_cpu {
                let layer_w = self.rank0.weights.layer(ip);
                if m_usize == 1 {
                    run_cpu_moe_m1(
                        &self.cpu_pool,
                        &mut self.cpu_scratch,
                        layer_w.gate_up_type,
                        layer_w.down_type,
                        n_embd,
                        self.rank0.cfg.n_ff_exp,
                        self.rank0.cfg.situ_beta,
                        self.rank0.cfg.situ_linear_beta,
                        &self.hidden_pin.as_slice()[..n_embd],
                        &cpu_jobs,
                        &mut self.cpu_out_pin.as_mut_slice()[..n_embd],
                    );
                } else {
                    run_cpu_moe_batch(
                        &self.cpu_pool,
                        &mut self.cpu_scratch,
                        layer_w.gate_up_type,
                        layer_w.down_type,
                        n_embd,
                        self.rank0.cfg.n_ff_exp,
                        self.rank0.cfg.situ_beta,
                        self.rank0.cfg.situ_linear_beta,
                        m_usize,
                        &self.hidden_pin.as_slice()[..m_usize * n_embd],
                        &cpu_slots,
                        &mut self.cpu_out_pin.as_mut_slice()[..m_usize * n_embd],
                    );
                }
            }

            set_device(0)?;
            if let Some(seq32) = active_seq {
                unsafe {
                    launch_p2p_wait_reduce(
                        self.rank0.scratch.moe_out.ptr as *mut f32,
                        self.helper_out_dev0 as *const f32,
                        self.helper_flag_dev0 as *const u32,
                        seq32,
                        (m_usize * n_embd) as i32,
                        self.rank0.stream.raw,
                    );
                }
            }
            if any_cpu {
                unsafe {
                    copy_h2d_raw_async(
                        self.rank0.scratch.ffn_out.ptr,
                        bytemuck::cast_slice(&self.cpu_out_pin.as_slice()[..m_usize * n_embd]),
                        &self.rank0.stream,
                    )?;
                    launch_add_f32(
                        self.rank0.scratch.moe_out.ptr as *mut f32,
                        self.rank0.scratch.ffn_out.ptr as *const f32,
                        (m_usize * n_embd) as i32,
                        self.rank0.stream.raw,
                    );
                }
            }

            self.rank0.enqueue_post_moe(slot, m);
        }

        set_device(0)?;
        self.rank0
            .enqueue_output_head(m, self.rank0.cfg.n_vocab_unpadded as i32, false);

        if let Some(ref arena) = self.arena {
            self.compute_done_event0.record(&self.rank0.stream)?;
            self.cache0.poll_promotions()?;
            let mask1 = self.cache1.resident_mask();
            let n_prom = self.cache0.schedule_promotions(
                arena,
                &self.tracker,
                &self.compute_done_event0,
                &recent_active,
                2,
                4,
                Some(&mask1),
            )?;
            self.tracker.stats.promotions += n_prom as u64;
            self.cache0.sync_dirty_ptr_tables(&self.rank0.stream)?;
        }

        self.rank0.stream.sync()?;
        self.cur_pos += m_usize;

        let mut next_id = [0i32; 1];
        self.rank0
            .scratch
            .argmax_id
            .copy_to_host(bytemuck::cast_slice_mut(&mut next_id))?;

        let logits_out = if capture_logits {
            let mut v = vec![0.0f32; self.rank0.cfg.n_vocab_unpadded];
            self.rank0
                .scratch
                .logits
                .copy_to_host(bytemuck::cast_slice_mut(&mut v))?;
            Some(v)
        } else {
            None
        };

        Ok((next_id[0] as u32, logits_out))
    }
}
