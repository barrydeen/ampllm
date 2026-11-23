use crate::config::EngineConfig;
use crate::inkling2_draft::{Inkling2Drafter, DSPARK_BLOCK, DSPARK_MAX_PROPOSALS};
use crate::kv_cache::KvConvCache;
use crate::rank_state::{GpuRankState, StaticMixPtrTables};
use crate::scratch::ActivationScratch;
use crate::weights::ModelWeights;
use ampllm_cache::{
    align_up_256, layer_expert_sizes, plan_layer_capacities, resolve_auto_budget_bytes,
    CacheBudget, CacheProfile, CacheStats, EwmaTracker, HostExpertArena, VramExpertCache,
};
use ampllm_cpu_kernels::{
    run_cpu_moe_batch, run_cpu_moe_m1, CpuBatchSlot, CpuExpertJob, CpuMoeScratch, CpuThreadPool,
};
use ampllm_cuda_kernels::cuda_rt::{copy_d2h_raw_async, copy_h2d_raw_async, mem_info};
use ampllm_cuda_kernels::ffi::{launch_add_f32, launch_doorbell_signal, launch_doorbell_wait_and_add};
use ampllm_cuda_kernels::{
    set_device, CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, PinnedMappedBuffer,
};
use ampllm_gguf::{GgufFile, Inkling2Config};
use anyhow::{bail, Result};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

struct HybridOffloadState {
    arena: HostExpertArena,
    cpu_pool: CpuThreadPool,
    cpu_scratch: CpuMoeScratch,
    req_seq: PinnedMappedBuffer<u64>,
    done_seq: PinnedMappedBuffer<u64>,
    doorbell_seq: u64,
    router_ids_pin: PinnedMappedBuffer<i32>,
    router_w_pin: PinnedMappedBuffer<f32>,
    hidden_pin: PinnedMappedBuffer<f32>,
    cpu_out_pin: PinnedMappedBuffer<f32>,
    router_ready_event: CudaEvent,
    compute_done_event: CudaEvent,

    staging_gu: [DeviceBuffer; 2],
    staging_dn: [DeviceBuffer; 2],
    staging_gu_ptrs_host: [PinnedMappedBuffer<u64>; 2],
    staging_dn_ptrs_host: [PinnedMappedBuffer<u64>; 2],
    staging_gu_ptrs_dev: [DeviceBuffer; 2],
    staging_dn_ptrs_dev: [DeviceBuffer; 2],
    dma_stage_ready: [CudaEvent; 2],
    compute_stage_done: [CudaEvent; 2],
    staging_armed: [bool; 2],
}

pub struct SingleGpuEngine {
    pub engine_cfg: EngineConfig,
    pub rank: GpuRankState,
    pub vram_cache: VramExpertCache,
    pub tracker: EwmaTracker,
    pub drafter: Option<Inkling2Drafter>,
    hybrid: Option<HybridOffloadState>,
    graph_short_kv: Option<(u64, CudaGraphExec)>,
    graph_split_kv: Option<(u64, CudaGraphExec)>,
    window_graphs: Vec<Option<(u64, CudaGraphExec)>>,
    host_toks_pin: PinnedMappedBuffer<i32>,
    host_pos_pin: PinnedMappedBuffer<i32>,
    host_argmax_pin: PinnedMappedBuffer<i32>,
    pub cur_pos: usize,
}

impl SingleGpuEngine {
    pub fn load(gguf: &GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let device = engine_cfg.device;
        set_device(device)?;
        let stream = CudaStream::new(device)?;
        let weights = ModelWeights::load_non_experts(gguf, &cfg, device, &stream)?;
        let kv_cache = KvConvCache::alloc(&cfg, device, engine_cfg.max_ctx, engine_cfg.kv_mode)?;
        let scratch = ActivationScratch::alloc(&cfg, device, engine_cfg.max_batch)?;
        let mix_tables = StaticMixPtrTables::build(&cfg, &scratch, device)?;
        stream.sync()?;

        let exp_sizes = layer_expert_sizes(gguf, &cfg)?;
        let total_exp_bytes: usize = exp_sizes
            .iter()
            .map(|&(gu, dn)| align_up_256(gu) + align_up_256(dn))
            .sum::<usize>()
            * cfg.n_expert;

        let resolved_draft = if !engine_cfg.use_spec {
            None
        } else if let Some(ref p) = engine_cfg.draft_path {
            Some(p.clone())
        } else {
            gguf.shard_paths
                .first()
                .and_then(|p| p.parent())
                .and_then(|dir| {
                    ["dspark-Inkling-2-Mini-Q8_0.gguf", "dspark.gguf"]
                        .iter()
                        .map(|f| dir.join(f))
                        .find(|cand| cand.is_file())
                })
        };

        let drafter = if let Some(ref dpath) = resolved_draft {
            let (free_now, _) = mem_info(device)?;
            let est_ctx_bytes = (engine_cfg.max_ctx * cfg.n_head_kv * cfg.head_dim * 2) + 450 * 1024 * 1024;
            if engine_cfg.draft_path.is_some() || free_now >= total_exp_bytes + est_ctx_bytes + 512 * 1024 * 1024 {
                if let (Some(tok_e), Some(tok_n), Some(out_w)) = (
                    weights.tok_embd.as_ref(),
                    weights.tok_embd_norm.as_ref(),
                    weights.output.as_ref(),
                ) {
                    match Inkling2Drafter::load(
                        dpath,
                        &cfg,
                        tok_e,
                        tok_n,
                        out_w,
                        engine_cfg.max_ctx,
                        device,
                        &stream,
                    ) {
                        Ok(d) => Some(d),
                        Err(err) => {
                            eprintln!("warning: failed to load DSpark drafter ({}): {err}", dpath.display());
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let budget_bytes = match engine_cfg.expert_cache {
            CacheBudget::Auto => {
                let reserve_mb = if drafter.is_some() { 512 } else { 1024 };
                resolve_auto_budget_bytes(device, reserve_mb)?
            }
            CacheBudget::Mb(mb) => mb * 1024 * 1024,
        };
        let layer_caps = plan_layer_capacities(&cfg, &exp_sizes, budget_bytes, None);
        let layer_needed: Vec<bool> = layer_caps.iter().map(|&c| c < cfg.n_expert).collect();
        let any_offload = layer_needed.iter().any(|&b| b);

        let profile = match &engine_cfg.load_profile {
            Some(path) => Some(CacheProfile::load(path)?),
            None => None,
        };
        let mut tracker = EwmaTracker::new(cfg.n_layer_phys, cfg.n_expert, 0.98);
        if let Some(ref prof) = profile {
            tracker.seed_from_profile(prof);
        }

        let arena = if any_offload {
            Some(HostExpertArena::from_gguf(gguf, &cfg, &layer_needed, &exp_sizes)?)
        } else {
            None
        };

        let vram_cache = VramExpertCache::init(
            gguf,
            &cfg,
            device,
            &layer_caps,
            &exp_sizes,
            profile.as_ref(),
            None,
            arena.as_ref(),
            &stream,
        )?;

        let hybrid = if let Some(arena) = arena {
            let max_gu_stride = exp_sizes.iter().map(|&(g, _)| align_up_256(g)).max().unwrap_or(0);
            let max_dn_stride = exp_sizes.iter().map(|&(_, d)| align_up_256(d)).max().unwrap_or(0);
            let max_b = engine_cfg.max_batch;

            let cpu_pool = CpuThreadPool::new(engine_cfg.cpu_threads);
            let cpu_scratch = CpuMoeScratch::new(cfg.n_embd, cfg.n_ff_exp, cfg.n_expert_used);

            let req_seq = PinnedMappedBuffer::<u64>::alloc(1)?;
            let done_seq = PinnedMappedBuffer::<u64>::alloc(1)?;
            let router_ids_pin = PinnedMappedBuffer::<i32>::alloc(max_b * cfg.n_expert_used)?;
            let router_w_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_expert_used)?;
            let hidden_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_embd)?;
            let cpu_out_pin = PinnedMappedBuffer::<f32>::alloc(max_b * cfg.n_embd)?;
            let router_ready_event = CudaEvent::new(device)?;
            let compute_done_event = CudaEvent::new(device)?;

            let staging_gu = [
                DeviceBuffer::alloc(device, 2 * max_gu_stride)?,
                DeviceBuffer::alloc(device, 2 * max_gu_stride)?,
            ];
            let staging_dn = [
                DeviceBuffer::alloc(device, 2 * max_dn_stride)?,
                DeviceBuffer::alloc(device, 2 * max_dn_stride)?,
            ];
            let staging_gu_ptrs_host = [
                PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?,
                PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?,
            ];
            let staging_dn_ptrs_host = [
                PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?,
                PinnedMappedBuffer::<u64>::alloc(cfg.n_expert)?,
            ];
            let staging_gu_ptrs_dev = [
                DeviceBuffer::alloc(device, cfg.n_expert * 8)?,
                DeviceBuffer::alloc(device, cfg.n_expert * 8)?,
            ];
            let staging_dn_ptrs_dev = [
                DeviceBuffer::alloc(device, cfg.n_expert * 8)?,
                DeviceBuffer::alloc(device, cfg.n_expert * 8)?,
            ];
            let dma_stage_ready = [CudaEvent::new(device)?, CudaEvent::new(device)?];
            let compute_stage_done = [CudaEvent::new(device)?, CudaEvent::new(device)?];

            Some(HybridOffloadState {
                arena,
                cpu_pool,
                cpu_scratch,
                req_seq,
                done_seq,
                doorbell_seq: 0,
                router_ids_pin,
                router_w_pin,
                hidden_pin,
                cpu_out_pin,
                router_ready_event,
                compute_done_event,
                staging_gu,
                staging_dn,
                staging_gu_ptrs_host,
                staging_dn_ptrs_host,
                staging_gu_ptrs_dev,
                staging_dn_ptrs_dev,
                dma_stage_ready,
                compute_stage_done,
                staging_armed: [false, false],
            })
        } else {
            None
        };

        let mut window_graphs = Vec::with_capacity(DSPARK_BLOCK + 1);
        for _ in 0..=DSPARK_BLOCK {
            window_graphs.push(None);
        }
        let host_toks_pin = PinnedMappedBuffer::<i32>::alloc(DSPARK_BLOCK)?;
        let host_pos_pin = PinnedMappedBuffer::<i32>::alloc(DSPARK_BLOCK)?;
        let host_argmax_pin = PinnedMappedBuffer::<i32>::alloc(DSPARK_BLOCK)?;

        let mut this = Self {
            engine_cfg,
            rank: GpuRankState {
                device,
                stream,
                cfg,
                weights,
                kv_cache,
                scratch,
                mix_tables,
            },
            vram_cache,
            tracker,
            drafter,
            hybrid,
            graph_short_kv: None,
            graph_split_kv: None,
            window_graphs,
            host_toks_pin,
            host_pos_pin,
            host_argmax_pin,
            cur_pos: 0,
        };
        this.rank.init_conv_hist_ptrs()?;
        this.warmup_decode_graphs()?;
        Ok(this)
    }

    pub fn cache_stats(&self) -> &CacheStats {
        &self.tracker.stats
    }

    pub fn cache_summary(&self) -> String {
        let total_possible = self.rank.cfg.n_layer_phys * self.rank.cfg.n_expert;
        let resident = self.vram_cache.total_slots();
        let vram_mb = self.vram_cache.total_vram_bytes() as f64 / (1024.0 * 1024.0);
        let arena_mb = self
            .hybrid
            .as_ref()
            .map(|h| h.arena.total_bytes() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let kv_mb = self.rank.kv_cache.total_bytes() as f64 / (1024.0 * 1024.0);
        let draft_mb = self
            .drafter
            .as_ref()
            .map(|d| d.vram_bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let threads = self.hybrid.as_ref().map(|h| h.cpu_pool.num_threads()).unwrap_or(0);
        format!(
            "Single(GPU {}) slots={resident}/{total_possible} (all_resident={}), expert_vram={vram_mb:.1} MiB, kv_vram={kv_mb:.1} MiB, draft_vram={draft_mb:.1} MiB, host_arena={arena_mb:.1} MiB, cpu_threads={threads}, kv={:?}, cuda_graph={}",
            self.rank.device,
            self.vram_cache.all_resident,
            self.rank.kv_cache.kv_mode,
            self.engine_cfg.use_cuda_graph
        )
    }

    pub fn save_profile(&self) -> Result<()> {
        if let Some(ref path) = self.engine_cfg.save_profile {
            let prof = self.tracker.export_profile(&self.rank.cfg.arch);
            prof.save(path)?;
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        set_device(self.rank.device)?;
        self.rank.kv_cache.reset()?;
        self.cur_pos = 0;
        Ok(())
    }

    fn enqueue_forward_m1_body(&self, use_split_kv: bool) -> Result<()> {
        let n_embd = self.rank.cfg.n_embd as i32;
        let max_kv_len = if use_split_kv { 257 } else { 1 };
        let tok_embd = self.rank.weights.tok_embd.as_ref().unwrap();
        let tok_norm = self.rank.weights.tok_embd_norm.as_ref().unwrap();

        unsafe {
            ampllm_cuda_kernels::ffi::launch_compute_tau_f32(
                self.rank.scratch.pos_ids.ptr as *const i32,
                self.rank.scratch.tau.ptr as *mut f32,
                1,
                self.rank.cfg.log_scale_floor,
                self.rank.cfg.log_scale_alpha,
                self.rank.stream.raw,
            );
            ampllm_cuda_kernels::ffi::launch_embed_lookup_rmsnorm(
                tok_embd.buf.ptr,
                tok_embd.ggml_type as i32,
                self.rank.scratch.token_ids.ptr as *const i32,
                tok_norm.as_f32(),
                self.rank.scratch.cur.ptr as *mut f32,
                n_embd,
                1,
                self.rank.cfg.rms_norm_eps,
                self.rank.stream.raw,
            );
        }

        let exec_order = self.rank.cfg.execution_order();
        for &slot in &exec_order {
            let ip = self.rank.cfg.logical_layers[slot].phys_layer;
            self.rank.enqueue_attn_through_wo(slot, 1, max_kv_len, true)?;
            self.rank.enqueue_post_wo_to_router(slot, 1)?;
            let (gu_ptrs, dn_ptrs) = self.vram_cache.layer_ptrs_dev(ip);
            self.rank.enqueue_moe_and_post_m1_resident(slot, gu_ptrs, dn_ptrs);
        }

        self.rank
            .enqueue_output_head(1, self.rank.cfg.n_vocab_unpadded as i32, false);
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
        set_device(self.rank.device)?;
        let max_batch = self.rank.scratch.max_batch;
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
        let tok_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i32: Vec<i32> = (self.cur_pos..self.cur_pos + m_usize)
            .map(|p| p as i32)
            .collect();

        let max_kv_len = (self.cur_pos + m_usize) as i32;
        let trace = std::env::var_os("AMPLLM_TRACE").is_some() && self.cur_pos == 0 && m_usize == 5;

        if m_usize == 1
            && !trace
            && !capture_logits
            && self.vram_cache.all_resident
            && self.engine_cfg.use_cuda_graph
            && self.drafter.is_none()
            && self
                .rank
                .weights
                .output
                .as_ref()
                .map_or(false, |o| o.q8_planar_q.is_some())
        {
            let argmax = self.forward_window_spec(tokens)?;
            self.cur_pos += 1;
            return Ok((argmax[0], None));
        }

        if m_usize == 1
            && !trace
            && self.vram_cache.all_resident
            && self.engine_cfg.use_cuda_graph
            && self.drafter.is_none()
        {
            self.rank.scratch.token_ids.copy_from_host_async(
                bytemuck::cast_slice(&tok_i32),
                &self.rank.stream,
            )?;
            self.rank.scratch.pos_ids.copy_from_host_async(
                bytemuck::cast_slice(&pos_i32),
                &self.rank.stream,
            )?;
            let use_split_kv = max_kv_len > 256;
            let epoch = self.vram_cache.cache_epoch;
            let need_capture = {
                let slot = if use_split_kv {
                    &self.graph_split_kv
                } else {
                    &self.graph_short_kv
                };
                match slot {
                    Some((ge, _)) => *ge != epoch,
                    None => true,
                }
            };
            if need_capture {
                let g = CudaGraphExec::capture(&self.rank.stream, || {
                    self.enqueue_forward_m1_body(use_split_kv)
                })?;
                if use_split_kv {
                    self.graph_split_kv = Some((epoch, g));
                } else {
                    self.graph_short_kv = Some((epoch, g));
                }
            }
            let slot = if use_split_kv {
                self.graph_split_kv.as_ref()
            } else {
                self.graph_short_kv.as_ref()
            };
            if let Some((_, g)) = slot {
                g.launch(&self.rank.stream)?;
            }
        } else {
            self.rank.enqueue_inputs(&tok_i32, &pos_i32)?;
            self.forward_chunk_eager(m, max_kv_len, trace)?;
            if let Some(ref d) = self.drafter {
                self.rank.record_final_tap(m)?;
                d.write_context(
                    self.rank.scratch.taps.ptr as *const f32,
                    self.rank.scratch.max_batch,
                    0,
                    m_usize,
                    self.cur_pos,
                    &self.rank.stream,
                );
                d.set_f_rnn(
                    self.rank.scratch.taps.ptr as *const f32,
                    self.rank.scratch.max_batch,
                    m_usize - 1,
                    &self.rank.stream,
                );
            }
        }

        self.rank.stream.sync()?;
        if let Some(ref mut h) = self.hybrid {
            h.staging_armed = [false, false];
        }
        self.cur_pos += m_usize;

        let mut next_id = [0i32; 1];
        self.rank
            .scratch
            .argmax_id
            .copy_to_host(bytemuck::cast_slice_mut(&mut next_id))?;

        let logits_out = if capture_logits {
            let mut v = vec![0.0f32; self.rank.cfg.n_vocab_unpadded];
            self.rank
                .scratch
                .logits
                .copy_to_host(bytemuck::cast_slice_mut(&mut v))?;
            Some(v)
        } else {
            None
        };

        Ok((next_id[0] as u32, logits_out))
    }

    pub fn warmup_decode_graphs(&mut self) -> Result<()> {
        if !self.vram_cache.all_resident || !self.engine_cfg.use_cuda_graph {
            return Ok(());
        }
        set_device(self.rank.device)?;
        let epoch = self.vram_cache.cache_epoch;
        let can_fast_window = self
            .rank
            .weights
            .output
            .as_ref()
            .map_or(false, |o| o.q8_planar_q.is_some());
        if !can_fast_window {
            if self.graph_short_kv.as_ref().map(|(e, _)| *e != epoch).unwrap_or(true) {
                let g = CudaGraphExec::capture(&self.rank.stream, || {
                    self.enqueue_forward_m1_body(false)
                })?;
                self.graph_short_kv = Some((epoch, g));
            }
            if self.graph_split_kv.as_ref().map(|(e, _)| *e != epoch).unwrap_or(true) {
                let g = CudaGraphExec::capture(&self.rank.stream, || {
                    self.enqueue_forward_m1_body(true)
                })?;
                self.graph_split_kv = Some((epoch, g));
            }
        } else {
            let has_drafter = self.drafter.is_some();
            let max_w = if self.engine_cfg.use_spec || has_drafter {
                DSPARK_BLOCK
            } else {
                1
            };
            for i in 0..max_w {
                self.host_toks_pin.as_mut_slice()[i] = 0;
                self.host_pos_pin.as_mut_slice()[i] = i as i32;
            }
            self.rank.scratch.token_ids.copy_from_host_async(
                bytemuck::cast_slice(&self.host_toks_pin.as_slice()[..max_w]),
                &self.rank.stream,
            )?;
            self.rank.scratch.pos_ids.copy_from_host_async(
                bytemuck::cast_slice(&self.host_pos_pin.as_slice()[..max_w]),
                &self.rank.stream,
            )?;
            for t in 1..=max_w {
                if self.window_graphs[t].as_ref().map(|(e, _)| *e != epoch).unwrap_or(true) {
                    let g = CudaGraphExec::capture(&self.rank.stream, || {
                        self.rank.enqueue_forward_window(
                            t as i32,
                            true,
                            false,
                            has_drafter,
                            t > 1,
                            |ip| self.vram_cache.layer_ptrs_dev(ip),
                        )
                    })?;
                    self.window_graphs[t] = Some((epoch, g));
                }
                self.window_graphs[t].as_ref().unwrap().1.launch(&self.rank.stream)?;
            }
            if let Some(ref mut d) = self.drafter {
                d.warmup_graphs(&self.rank.stream)?;
            }
            self.rank.stream.sync()?;
            self.rank.kv_cache.reset()?;
        }
        Ok(())
    }

    pub fn can_spec_decode(&self) -> bool {
        self.engine_cfg.use_spec
            && self.vram_cache.all_resident
            && self.engine_cfg.use_cuda_graph
            && self
                .rank
                .weights
                .output
                .as_ref()
                .map_or(false, |o| o.q8_planar_q.is_some())
    }

    pub fn has_neural_drafter(&self) -> bool {
        self.drafter.is_some()
    }

    pub fn propose_draft(&mut self, anchor: u32, proposals: usize) -> Result<[u32; DSPARK_MAX_PROPOSALS]> {
        let d = self.drafter.as_mut().expect("drafter required");
        d.propose(anchor, self.cur_pos, proposals, &self.rank.stream)
    }

    pub fn forward_window_spec(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        let t = tokens.len();
        if t == 0 || t > DSPARK_BLOCK {
            bail!("forward_window_spec invalid width {t}");
        }
        set_device(self.rank.device)?;
        for (i, &tok) in tokens.iter().enumerate() {
            self.host_toks_pin.as_mut_slice()[i] = tok as i32;
            self.host_pos_pin.as_mut_slice()[i] = (self.cur_pos + i) as i32;
        }
        self.rank.scratch.token_ids.copy_from_host_async(
            bytemuck::cast_slice(&self.host_toks_pin.as_slice()[..t]),
            &self.rank.stream,
        )?;
        self.rank.scratch.pos_ids.copy_from_host_async(
            bytemuck::cast_slice(&self.host_pos_pin.as_slice()[..t]),
            &self.rank.stream,
        )?;

        let stage_prof = {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::OnceLock;
            static ENABLED: OnceLock<bool> = OnceLock::new();
            static DONE_M1: AtomicBool = AtomicBool::new(false);
            static DONE_MN: AtomicBool = AtomicBool::new(false);
            if *ENABLED.get_or_init(|| std::env::var_os("AMPLLM_STAGE_PROF").is_some()) {
                (t == 1 && self.cur_pos >= 10 && !DONE_M1.swap(true, Ordering::Relaxed))
                    || (t >= 5 && !DONE_MN.swap(true, Ordering::Relaxed))
            } else {
                false
            }
        };

        if stage_prof {
            let has_drafter = self.drafter.is_some();
            self.rank.enqueue_forward_window_profiled(
                t as i32,
                true,
                false,
                has_drafter,
                t > 1,
                |ip| self.vram_cache.layer_ptrs_dev(ip),
            )?;
        } else {
            let epoch = self.vram_cache.cache_epoch;
            if self.window_graphs[t].as_ref().map(|(e, _)| *e != epoch).unwrap_or(true) {
                let has_drafter = self.drafter.is_some();
                let g = CudaGraphExec::capture(&self.rank.stream, || {
                    self.rank.enqueue_forward_window(
                        t as i32,
                        true,
                        false,
                        has_drafter,
                        t > 1,
                        |ip| self.vram_cache.layer_ptrs_dev(ip),
                    )
                })?;
                self.window_graphs[t] = Some((epoch, g));
            }
            self.window_graphs[t].as_ref().unwrap().1.launch(&self.rank.stream)?;
        }

        unsafe {
            copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut self.host_argmax_pin.as_mut_slice()[..t]),
                self.rank.scratch.argmax_ids.ptr,
                &self.rank.stream,
            )?;
        }
        self.rank.stream.sync()?;

        Ok(self.host_argmax_pin.as_slice()[..t]
            .iter()
            .map(|&v| v as u32)
            .collect())
    }

    pub fn commit_window_spec(&mut self, accept_total: usize, total_window: usize) -> Result<()> {
        if accept_total < total_window {
            self.rank.enqueue_conv_rollback(accept_total as i32);
        }
        if let Some(ref d) = self.drafter {
            d.write_context(
                self.rank.scratch.taps.ptr as *const f32,
                self.rank.scratch.max_batch,
                0,
                accept_total,
                self.cur_pos,
                &self.rank.stream,
            );
            d.set_f_rnn(
                self.rank.scratch.taps.ptr as *const f32,
                self.rank.scratch.max_batch,
                accept_total - 1,
                &self.rank.stream,
            );
        }
        self.cur_pos += accept_total;
        Ok(())
    }

    fn forward_chunk_eager(&mut self, m: i32, max_kv_len: i32, trace: bool) -> Result<()> {
        let m_usize = m as usize;
        let n_embd = self.rank.cfg.n_embd;
        let topk = self.rank.cfg.n_expert_used;
        let record_taps = self.drafter.is_some();
        let mut recent_active = vec![Vec::<usize>::new(); self.rank.cfg.n_layer_phys];
        let exec_order = self.rank.cfg.execution_order();

        for (vis, &slot) in exec_order.iter().enumerate() {
            if record_taps {
                self.rank.maybe_record_visit_tap(vis, m)?;
            }
            let ip = self.rank.cfg.logical_layers[slot].phys_layer;
            self.rank.enqueue_attn_through_wo(slot, m, max_kv_len, true)?;
            self.rank.enqueue_post_wo_to_router(slot, m)?;

            let (gu_ptrs, dn_ptrs) = self.vram_cache.layer_ptrs_dev(ip);
            let all_res_layer = self.vram_cache.layers[ip].cap >= self.rank.cfg.n_expert;

            if all_res_layer || self.hybrid.is_none() {
                self.rank.enqueue_moe_local(ip, gu_ptrs, dn_ptrs, m);
                if self.vram_cache.layers[ip].cap >= self.rank.cfg.n_expert
                    && !self.vram_cache.all_resident
                {
                    self.tracker.stats.hits += (m_usize * topk) as u64;
                }
            } else if m_usize == 1 {
                let h = self.hybrid.as_mut().unwrap();
                h.doorbell_seq += 1;
                let seq = h.doorbell_seq;

                unsafe {
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut h.router_ids_pin.as_mut_slice()[..topk]),
                        self.rank.scratch.expert_ids.ptr,
                        &self.rank.stream,
                    )?;
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut h.router_w_pin.as_mut_slice()[..topk]),
                        self.rank.scratch.expert_weights.ptr,
                        &self.rank.stream,
                    )?;
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut h.hidden_pin.as_mut_slice()[..n_embd]),
                        self.rank.scratch.tmp_hidden.ptr,
                        &self.rank.stream,
                    )?;
                    launch_doorbell_signal(
                        h.req_seq.dev_ptr() as *mut u64,
                        seq,
                        self.rank.stream.raw,
                    );
                }
                h.router_ready_event.record(&self.rank.stream)?;

                self.rank.enqueue_moe_local(ip, gu_ptrs, dn_ptrs, 1);
                unsafe {
                    launch_doorbell_wait_and_add(
                        h.done_seq.dev_ptr() as *const u64,
                        seq,
                        h.cpu_out_pin.dev_ptr() as *const f32,
                        std::ptr::null(),
                        self.rank.scratch.moe_out.ptr as *mut f32,
                        n_embd as i32,
                        self.rank.stream.raw,
                    );
                }
                let _ = h.router_ready_event.is_ready()?;

                let req_atomic = unsafe { &*(h.req_seq.host_ptr() as *const AtomicU64) };
                while req_atomic.load(Ordering::Acquire) < seq {
                    std::hint::spin_loop();
                }

                self.tracker.decay_layer(ip);
                let ids = &h.router_ids_pin.as_slice()[..topk];
                let ws = &h.router_w_pin.as_slice()[..topk];
                let mut cpu_jobs: Vec<CpuExpertJob<'_>> = Vec::with_capacity(topk);

                let layer_w = self.rank.weights.layer(ip);
                for (&e_i32, &w) in ids.iter().zip(ws.iter()) {
                    let e = e_i32 as usize;
                    self.tracker.record_activation(ip, e);
                    if !recent_active[ip].contains(&e) {
                        recent_active[ip].push(e);
                    }
                    if self.vram_cache.is_resident(ip, e) {
                        self.tracker.stats.hits += 1;
                    } else {
                        self.tracker.stats.misses += 1;
                        cpu_jobs.push(CpuExpertJob {
                            gate_up: h.arena.expert_gate_up(ip, e),
                            down: h.arena.expert_down(ip, e),
                            gate_up_scale: layer_w.gate_up_scale_host[e],
                            weight: w * layer_w.down_scale_host[e],
                        });
                    }
                }

                if cpu_jobs.is_empty() {
                    h.cpu_out_pin.as_mut_slice()[..n_embd].fill(0.0);
                } else {
                    run_cpu_moe_m1(
                        &h.cpu_pool,
                        &mut h.cpu_scratch,
                        layer_w.gate_up_type,
                        layer_w.down_type,
                        n_embd,
                        self.rank.cfg.n_ff_exp,
                        self.rank.cfg.situ_beta,
                        self.rank.cfg.situ_linear_beta,
                        &h.hidden_pin.as_slice()[..n_embd],
                        &cpu_jobs,
                        &mut h.cpu_out_pin.as_mut_slice()[..n_embd],
                    );
                }

                let done_atomic = unsafe { &*(h.done_seq.host_ptr() as *const AtomicU64) };
                done_atomic.store(seq, Ordering::Release);
            } else {
                let h = self.hybrid.as_mut().unwrap();
                let n_slots = m_usize * topk;
                unsafe {
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut h.router_ids_pin.as_mut_slice()[..n_slots]),
                        self.rank.scratch.expert_ids.ptr,
                        &self.rank.stream,
                    )?;
                    copy_d2h_raw_async(
                        bytemuck::cast_slice_mut(&mut h.router_w_pin.as_mut_slice()[..n_slots]),
                        self.rank.scratch.expert_weights.ptr,
                        &self.rank.stream,
                    )?;
                    if m_usize < self.engine_cfg.dma_prefill_threshold {
                        copy_d2h_raw_async(
                            bytemuck::cast_slice_mut(
                                &mut h.hidden_pin.as_mut_slice()[..m_usize * n_embd],
                            ),
                            self.rank.scratch.tmp_hidden.ptr,
                            &self.rank.stream,
                        )?;
                    }
                }
                self.rank.stream.sync()?;

                self.tracker.decay_layer(ip);
                let mut cold_unique = BTreeSet::<usize>::new();
                let mut cpu_slots: Vec<CpuBatchSlot<'_>> = Vec::new();
                let ids = &h.router_ids_pin.as_slice()[..n_slots];
                let ws = &h.router_w_pin.as_slice()[..n_slots];

                let layer_w = self.rank.weights.layer(ip);
                for tok_idx in 0..m_usize {
                    for k_s in 0..topk {
                        let s_idx = tok_idx * topk + k_s;
                        let e = ids[s_idx] as usize;
                        self.tracker.record_activation(ip, e);
                        if !recent_active[ip].contains(&e) {
                            recent_active[ip].push(e);
                        }
                        if self.vram_cache.is_resident(ip, e) {
                            self.tracker.stats.hits += 1;
                        } else {
                            self.tracker.stats.misses += 1;
                            cold_unique.insert(e);
                            if m_usize < self.engine_cfg.dma_prefill_threshold {
                                cpu_slots.push(CpuBatchSlot {
                                    token_idx: tok_idx,
                                    gate_up: h.arena.expert_gate_up(ip, e),
                                    down: h.arena.expert_down(ip, e),
                                    gate_up_scale: layer_w.gate_up_scale_host[e],
                                    weight: ws[s_idx] * layer_w.down_scale_host[e],
                                });
                            }
                        }
                    }
                }

                self.rank.enqueue_moe_local(ip, gu_ptrs, dn_ptrs, m);

                if !cold_unique.is_empty() {
                    if m_usize < self.engine_cfg.dma_prefill_threshold {
                        run_cpu_moe_batch(
                            &h.cpu_pool,
                            &mut h.cpu_scratch,
                            layer_w.gate_up_type,
                            layer_w.down_type,
                            n_embd,
                            self.rank.cfg.n_ff_exp,
                            self.rank.cfg.situ_beta,
                            self.rank.cfg.situ_linear_beta,
                            m_usize,
                            &h.hidden_pin.as_slice()[..m_usize * n_embd],
                            &cpu_slots,
                            &mut h.cpu_out_pin.as_mut_slice()[..m_usize * n_embd],
                        );
                        unsafe {
                            copy_h2d_raw_async(
                                self.rank.scratch.ffn_out.ptr,
                                bytemuck::cast_slice(
                                    &h.cpu_out_pin.as_slice()[..m_usize * n_embd],
                                ),
                                &self.rank.stream,
                            )?;
                            launch_add_f32(
                                self.rank.scratch.moe_out.ptr as *mut f32,
                                self.rank.scratch.ffn_out.ptr as *const f32,
                                (m_usize * n_embd) as i32,
                                self.rank.stream.raw,
                            );
                        }
                    } else {
                        let cold_list: Vec<usize> = cold_unique.into_iter().collect();
                        let gu_stride = align_up_256(layer_w.gate_up_expert_bytes);
                        let dn_stride = align_up_256(layer_w.down_expert_bytes);
                        let copy_stream = &self.vram_cache.copy_stream;

                        for (chunk_idx, chunk) in cold_list.chunks(2).enumerate() {
                            let p = chunk_idx % 2;
                            if h.staging_armed[p] {
                                h.compute_stage_done[p].sync()?;
                            }

                            h.staging_gu_ptrs_host[p].as_mut_slice().fill(0);
                            h.staging_dn_ptrs_host[p].as_mut_slice().fill(0);

                            for (s, &exp) in chunk.iter().enumerate() {
                                let gu_dst = h.staging_gu[p].as_u64() + (s * gu_stride) as u64;
                                let dn_dst = h.staging_dn[p].as_u64() + (s * dn_stride) as u64;
                                unsafe {
                                    copy_h2d_raw_async(
                                        gu_dst as *mut c_void,
                                        h.arena.expert_gate_up(ip, exp),
                                        copy_stream,
                                    )?;
                                    copy_h2d_raw_async(
                                        dn_dst as *mut c_void,
                                        h.arena.expert_down(ip, exp),
                                        copy_stream,
                                    )?;
                                }
                                h.staging_gu_ptrs_host[p].as_mut_slice()[exp] = gu_dst;
                                h.staging_dn_ptrs_host[p].as_mut_slice()[exp] = dn_dst;
                            }

                            h.staging_gu_ptrs_dev[p].copy_from_host_async(
                                bytemuck::cast_slice(h.staging_gu_ptrs_host[p].as_slice()),
                                copy_stream,
                            )?;
                            h.staging_dn_ptrs_dev[p].copy_from_host_async(
                                bytemuck::cast_slice(h.staging_dn_ptrs_host[p].as_slice()),
                                copy_stream,
                            )?;
                            h.dma_stage_ready[p].record(copy_stream)?;

                            self.rank.stream.wait_event(&h.dma_stage_ready[p])?;
                            let gu_sc_ptr = layer_w
                                .gate_up_scale
                                .as_ref()
                                .map(|t| t.as_f32())
                                .unwrap_or(std::ptr::null());
                            let dn_sc_ptr = layer_w
                                .down_scale
                                .as_ref()
                                .map(|t| t.as_f32())
                                .unwrap_or(std::ptr::null());
                            unsafe {
                                ampllm_cuda_kernels::ffi::launch_moe_grouped_f32(
                                    h.staging_gu_ptrs_dev[p].ptr as *const u64,
                                    h.staging_dn_ptrs_dev[p].ptr as *const u64,
                                    gu_sc_ptr,
                                    dn_sc_ptr,
                                    layer_w.gate_up_row_bytes,
                                    layer_w.down_row_bytes,
                                    layer_w.gate_up_type as i32,
                                    layer_w.down_type as i32,
                                    self.rank.scratch.act_q8_1.ptr,
                                    std::ptr::null(),
                                    self.rank.scratch.expert_ids.ptr as *const i32,
                                    self.rank.scratch.expert_weights.ptr as *const f32,
                                    self.rank.scratch.moe_inter_f32.ptr as *mut f32,
                                    self.rank.scratch.moe_inter_q8_1.ptr,
                                    self.rank.scratch.moe_slot_out.ptr as *mut f32,
                                    self.rank.scratch.ffn_out.ptr as *mut f32,
                                    n_embd as i32,
                                    self.rank.cfg.n_ff_exp as i32,
                                    topk as i32,
                                    m,
                                    self.rank.cfg.situ_beta,
                                    self.rank.cfg.situ_linear_beta,
                                    self.rank.stream.raw,
                                );
                                launch_add_f32(
                                    self.rank.scratch.moe_out.ptr as *mut f32,
                                    self.rank.scratch.ffn_out.ptr as *const f32,
                                    (m_usize * n_embd) as i32,
                                    self.rank.stream.raw,
                                );
                            }
                            h.compute_stage_done[p].record(&self.rank.stream)?;
                            h.staging_armed[p] = true;
                        }
                    }
                }
            }

            self.rank.enqueue_post_moe(slot, m);

            if trace {
                self.rank.stream.sync()?;
                let mut host = vec![0.0f32; m_usize * n_embd];
                self.rank
                    .scratch
                    .cur
                    .copy_to_host(bytemuck::cast_slice_mut(&mut host))?;
                let l2: f64 = host.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                let last = &host[(m_usize - 1) * n_embd..];
                let last_l2: f64 = last.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                eprintln!(
                    "TRACE l_out-{slot}: l2={l2:.6} last_l2={last_l2:.6} last_head={:?}",
                    &last[..6.min(last.len())]
                );
            }
        }

        self.rank
            .enqueue_output_head(m, self.rank.cfg.n_vocab_unpadded as i32, false);

        if let Some(ref mut h) = self.hybrid {
            h.compute_done_event.record(&self.rank.stream)?;
            self.vram_cache.poll_promotions()?;
            let n_prom = self.vram_cache.schedule_promotions(
                &h.arena,
                &self.tracker,
                &h.compute_done_event,
                &recent_active,
                2,
                4,
                None,
            )?;
            self.tracker.stats.promotions += n_prom as u64;
            self.vram_cache.sync_dirty_ptr_tables(&self.rank.stream)?;
        }

        Ok(())
    }
}
