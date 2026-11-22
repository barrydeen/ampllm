use crate::config::EngineConfig;
use crate::kv_cache::KvConvCache;
use crate::rank_state::{GpuRankState, StaticMixPtrTables};
use crate::scratch::ActivationScratch;
use crate::weights::ModelWeights;
use ampllm_cache::{layer_expert_sizes, plan_layer_capacities, VramExpertCache};
use ampllm_cuda_kernels::{mem_info, set_device, CudaStream, DeviceBuffer, PinnedMappedBuffer};
use ampllm_gguf::{GgufFile, Inkling2Config};
use anyhow::{bail, Result};

/// Dual-RTX-3090 Pipeline Mode:
/// - GPU 0 owns physical layers 0..split (`0..12`), embedding, and prelude/pass-entry buffers
/// - GPU 1 owns physical layers split..24 (`12..24`), `output` head, and coda
/// Both halves fit 100% of their experts in VRAM (`canAccessPeer = 0` uses mapped pinned RAM).
pub struct PipelineGpuEngine {
    pub engine_cfg: EngineConfig,
    pub split: usize,
    pub rank0: GpuRankState,
    pub rank1: GpuRankState,
    cache0: VramExpertCache,
    cache1: VramExpertCache,
    xfer_cur: PinnedMappedBuffer<f32>,
    xfer_blk: PinnedMappedBuffer<f32>,
    pub cur_pos: usize,
}

impl PipelineGpuEngine {
    pub fn load(gguf: &GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let split = cfg.n_layer_phys / 2; // 12
        let dev0 = 0;
        let dev1 = 1;

        set_device(dev0)?;
        let stream0 = CudaStream::new(dev0)?;
        let weights0 =
            ModelWeights::load_non_experts_range(gguf, &cfg, 0..split, true, false, dev0, &stream0)?;
        let kv0 = KvConvCache::alloc_custom(&cfg, dev0, engine_cfg.max_ctx, engine_cfg.kv_mode, None, Some(0..split))?;
        let scratch0 = ActivationScratch::alloc(&cfg, dev0, engine_cfg.max_batch)?;
        let mix0 = StaticMixPtrTables::build(&cfg, &scratch0, dev0)?;
        stream0.sync()?;

        set_device(dev1)?;
        let stream1 = CudaStream::new(dev1)?;
        let weights1 = ModelWeights::load_non_experts_range(
            gguf,
            &cfg,
            split..cfg.n_layer_phys,
            false,
            true,
            dev1,
            &stream1,
        )?;
        let kv1 = KvConvCache::alloc_custom(
            &cfg,
            dev1,
            engine_cfg.max_ctx,
            engine_cfg.kv_mode,
            None,
            Some(split..cfg.n_layer_phys),
        )?;
        let scratch1 = ActivationScratch::alloc(&cfg, dev1, engine_cfg.max_batch)?;
        let mix1 = StaticMixPtrTables::build(&cfg, &scratch1, dev1)?;
        stream1.sync()?;

        let exp_sizes = layer_expert_sizes(gguf, &cfg)?;
        let max0: Vec<usize> = (0..cfg.n_layer_phys)
            .map(|i| if i < split { cfg.n_expert } else { 0 })
            .collect();
        let max1: Vec<usize> = (0..cfg.n_layer_phys)
            .map(|i| if i >= split { cfg.n_expert } else { 0 })
            .collect();
        let caps0 = plan_layer_capacities(&cfg, &exp_sizes, usize::MAX / 2, Some(&max0));
        let caps1 = plan_layer_capacities(&cfg, &exp_sizes, usize::MAX / 2, Some(&max1));

        let cache0 = VramExpertCache::init(gguf, &cfg, dev0, &caps0, &exp_sizes, None, None, None, &stream0)?;
        let cache1 = VramExpertCache::init(gguf, &cfg, dev1, &caps1, &exp_sizes, None, None, None, &stream1)?;

        let (free1, _) = mem_info(dev1)?;
        if free1 < 768 * 1024 * 1024 {
            bail!(
                "Pipeline GPU 1 free VRAM ({:.1} MiB) below 768 MiB display safety margin",
                free1 as f64 / (1024.0 * 1024.0)
            );
        }

        let xfer_cur = PinnedMappedBuffer::<f32>::alloc(engine_cfg.max_batch * cfg.n_embd)?;
        let xfer_blk = PinnedMappedBuffer::<f32>::alloc(engine_cfg.max_batch * cfg.n_embd)?;

        set_device(dev0)?;
        Ok(Self {
            engine_cfg,
            split,
            rank0: GpuRankState {
                device: dev0,
                stream: stream0,
                cfg: cfg.clone(),
                weights: weights0,
                kv_cache: kv0,
                scratch: scratch0,
                mix_tables: mix0,
            },
            rank1: GpuRankState {
                device: dev1,
                stream: stream1,
                cfg,
                weights: weights1,
                kv_cache: kv1,
                scratch: scratch1,
                mix_tables: mix1,
            },
            cache0,
            cache1,
            xfer_cur,
            xfer_blk,
            cur_pos: 0,
        })
    }

    pub fn cache_summary(&self) -> String {
        let total = self.rank0.cfg.n_layer_phys * self.rank0.cfg.n_expert;
        let s0 = self.cache0.total_slots();
        let s1 = self.cache1.total_slots();
        let m0 = self.cache0.total_vram_bytes() as f64 / (1024.0 * 1024.0);
        let m1 = self.cache1.total_vram_bytes() as f64 / (1024.0 * 1024.0);
        let free1_mb = mem_info(1).map(|(f, _)| f as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
        let kv_mode = self.rank0.kv_cache.kv_mode;
        let split = self.split;
        format!(
            "Pipeline(0..{split} GPU0 | {split}..24 GPU1) slots=({s0}+{s1})/{total} vram=({m0:.1}+{m1:.1}) MiB gpu1_free={free1_mb:.1} MiB kv={kv_mode:?}"
        )
    }

    pub fn reset(&mut self) -> Result<()> {
        set_device(0)?;
        self.rank0.kv_cache.reset()?;
        set_device(1)?;
        self.rank1.kv_cache.reset()?;
        set_device(0)?;
        self.cur_pos = 0;
        Ok(())
    }

    fn copy_between_ranks(
        src_dev: i32,
        src_buf: &DeviceBuffer,
        src_stream: &CudaStream,
        dst_dev: i32,
        dst_buf: &DeviceBuffer,
        dst_stream: &CudaStream,
        stage: &mut PinnedMappedBuffer<f32>,
        floats: usize,
    ) -> Result<()> {
        let bytes = floats * 4;
        set_device(src_dev)?;
        src_buf.copy_to_host_async(&mut bytemuck::cast_slice_mut(stage.as_mut_slice())[..bytes], src_stream)?;
        src_stream.sync()?;
        set_device(dst_dev)?;
        dst_buf.copy_from_host_async(&bytemuck::cast_slice(stage.as_slice())[..bytes], dst_stream)?;
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

    fn run_layer_on_rank(
        rank: &GpuRankState,
        cache: &VramExpertCache,
        slot: usize,
        m: i32,
        max_kv_len: i32,
    ) -> Result<()> {
        let ip = rank.cfg.logical_layers[slot].phys_layer;
        rank.enqueue_attn_through_wo(slot, m, max_kv_len, false)?;
        rank.enqueue_post_wo_to_router(slot, m)?;
        let (gu, dn) = cache.layer_ptrs_dev(ip);
        if m == 1 {
            rank.enqueue_moe_and_post_m1_resident(slot, gu, dn);
        } else {
            rank.enqueue_moe_local(ip, gu, dn, m);
            rank.enqueue_post_moe(slot, m);
        }
        Ok(())
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        capture_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>)> {
        let m_usize = tokens.len();
        let m = m_usize as i32;
        let floats = m_usize * self.rank0.cfg.n_embd;
        let hid_bytes = floats * 4;
        let max_kv_len = (self.cur_pos + m_usize) as i32;
        let tok_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i32: Vec<i32> = (self.cur_pos..self.cur_pos + m_usize)
            .map(|p| p as i32)
            .collect();

        set_device(0)?;
        self.rank0.enqueue_inputs(&tok_i32, &pos_i32)?;
        set_device(1)?;
        self.rank1.enqueue_compute_tau_only(&pos_i32)?;

        // 1. Prelude (ip = 0..6) on GPU 0
        set_device(0)?;
        for slot in 0..6 {
            Self::run_layer_on_rank(&self.rank0, &self.cache0, slot, m, max_kv_len)?;
        }
        // Save prelude output on GPU 0 and send b0 (blocks[0]) to GPU 1
        self.rank0.scratch.prelude.copy_from_device_async(
            &self.rank0.scratch.cur,
            hid_bytes,
            &self.rank0.stream,
        )?;
        Self::copy_between_ranks(
            0,
            &self.rank0.scratch.blocks[0],
            &self.rank0.stream,
            1,
            &self.rank1.scratch.blocks[0],
            &self.rank1.stream,
            &mut self.xfer_blk,
            floats,
        )?;

        // 2. 3 Looped passes on core (6..12 on GPU 0 -> 12..18 on GPU 1)
        let n_loop_passes = self.rank0.cfg.loop_count;
        for pass in 0..n_loop_passes {
            set_device(0)?;
            if pass > 0 {
                self.rank0.scratch.cur.copy_from_device_async(
                    &self.rank0.scratch.prelude,
                    hid_bytes,
                    &self.rank0.stream,
                )?;
            }
            for off in 0..6 {
                let slot = if pass == 0 {
                    6 + off
                } else {
                    24 + (pass - 1) * 12 + off
                };
                Self::run_layer_on_rank(&self.rank0, &self.cache0, slot, m, max_kv_len)?;
            }

            // Transfer (cur, blocks[1]) from GPU 0 -> GPU 1
            Self::copy_between_ranks(
                0,
                &self.rank0.scratch.cur,
                &self.rank0.stream,
                1,
                &self.rank1.scratch.cur,
                &self.rank1.stream,
                &mut self.xfer_cur,
                floats,
            )?;
            Self::copy_between_ranks(
                0,
                &self.rank0.scratch.blocks[1],
                &self.rank0.stream,
                1,
                &self.rank1.scratch.blocks[1],
                &self.rank1.stream,
                &mut self.xfer_blk,
                floats,
            )?;

            set_device(1)?;
            for off in 6..12 {
                let slot = if pass == 0 {
                    6 + off
                } else {
                    24 + (pass - 1) * 12 + off
                };
                Self::run_layer_on_rank(&self.rank1, &self.cache1, slot, m, max_kv_len)?;
            }

            if pass + 1 < n_loop_passes {
                // Save blocks[2] -> blocks[4] and cur -> bank_out on GPU 1, AND send both to GPU 0 for next pass
                self.rank1.scratch.blocks[4].copy_from_device_async(
                    &self.rank1.scratch.blocks[2],
                    hid_bytes,
                    &self.rank1.stream,
                )?;
                self.rank1.scratch.bank_out.copy_from_device_async(
                    &self.rank1.scratch.cur,
                    hid_bytes,
                    &self.rank1.stream,
                )?;
                Self::copy_between_ranks(
                    1,
                    &self.rank1.scratch.blocks[2],
                    &self.rank1.stream,
                    0,
                    &self.rank0.scratch.blocks[4],
                    &self.rank0.stream,
                    &mut self.xfer_blk,
                    floats,
                )?;
                Self::copy_between_ranks(
                    1,
                    &self.rank1.scratch.cur,
                    &self.rank1.stream,
                    0,
                    &self.rank0.scratch.bank_out,
                    &self.rank0.stream,
                    &mut self.xfer_cur,
                    floats,
                )?;
            }
        }

        // 3. Coda (18..24) + Output head on GPU 1
        set_device(1)?;
        for slot in 18..24 {
            Self::run_layer_on_rank(&self.rank1, &self.cache1, slot, m, max_kv_len)?;
        }
        self.rank1
            .enqueue_output_head(m, self.rank1.cfg.n_vocab_unpadded as i32, false);
        self.rank1.stream.sync()?;

        self.cur_pos += m_usize;
        let mut next_id = [0i32; 1];
        self.rank1
            .scratch
            .argmax_id
            .copy_to_host(bytemuck::cast_slice_mut(&mut next_id))?;

        let logits_out = if capture_logits {
            let mut v = vec![0.0f32; self.rank1.cfg.n_vocab_unpadded];
            self.rank1
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
