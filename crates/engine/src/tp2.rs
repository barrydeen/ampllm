use crate::config::EngineConfig;
use crate::kv_cache::KvConvCache;
use crate::rank_state::{GpuRankState, StaticMixPtrTables};
use crate::scratch::ActivationScratch;
use crate::weights::ModelWeights;
use ampllm_cuda_kernels::cuda_rt::copy_h2d_raw_async;
use ampllm_cuda_kernels::ffi::{launch_p2p_push, launch_p2p_wait_reduce};
use ampllm_cuda_kernels::{mem_info, set_device, CudaEvent, CudaStream, DeviceBuffer, PinnedMappedBuffer};
use ampllm_gguf::{GgufFile, Inkling2Config};
use anyhow::{bail, Result};
use std::ffi::c_void;

struct Tp2LayerExperts {
    gate_up_buf: DeviceBuffer,
    down_buf: DeviceBuffer,
    gate_up_ptrs_dev: DeviceBuffer,
    down_ptrs_dev: DeviceBuffer,
}

impl Tp2LayerExperts {
    fn expert_bytes_total(list: &[Self]) -> usize {
        list.iter()
            .map(|e| e.gate_up_buf.bytes + e.down_buf.bytes + e.gate_up_ptrs_dev.bytes + e.down_ptrs_dev.bytes)
            .sum()
    }
}

/// 2-Way Tensor Parallel engine across GPU 0 and GPU 1 using ping-pong mapped pinned-RAM all-reduce.
pub struct Tp2GpuEngine {
    pub engine_cfg: EngineConfig,
    pub rank0: GpuRankState,
    pub rank1: GpuRankState,
    exps0: Vec<Tp2LayerExperts>,
    exps1: Vec<Tp2LayerExperts>,
    vocab_half0: usize,
    vocab_half1: usize,

    // 2-slot ping-pong buffers & flags (indexed by seq & 1) make cross-GPU all-reduce mathematically overwrite-free
    _buf_0_to_1: [PinnedMappedBuffer<f32>; 2],
    _buf_1_to_0: [PinnedMappedBuffer<f32>; 2],
    _flag_0_to_1: [PinnedMappedBuffer<u32>; 2],
    _flag_1_to_0: [PinnedMappedBuffer<u32>; 2],
    ptr_0_to_1_dev0: [*mut c_void; 2],
    ptr_0_to_1_dev1: [*mut c_void; 2],
    ptr_1_to_0_dev0: [*mut c_void; 2],
    ptr_1_to_0_dev1: [*mut c_void; 2],
    fptr_0_to_1_dev0: [*mut c_void; 2],
    fptr_0_to_1_dev1: [*mut c_void; 2],
    fptr_1_to_0_dev0: [*mut c_void; 2],
    fptr_1_to_0_dev1: [*mut c_void; 2],
    flush_ev0: CudaEvent,
    flush_ev1: CudaEvent,
    comm_seq: u32,
    pub cur_pos: usize,
}

unsafe impl Send for Tp2GpuEngine {}

impl Tp2GpuEngine {
    pub fn load(gguf: &GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let n_head_l = cfg.n_head / 2;
        let n_head_kv_l = cfg.n_head_kv / 2;
        let n_ff_l = cfg.n_ff_exp / 2;
        let vocab_half0 = cfg.n_vocab_unpadded / 2;
        let vocab_half1 = cfg.n_vocab_unpadded - vocab_half0;

        set_device(0)?;
        let stream0 = CudaStream::new(0)?;
        let weights0 = ModelWeights::load_tp2_rank(gguf, &cfg, 0, 0, &stream0)?;
        let kv0 = KvConvCache::alloc_custom(&cfg, 0, engine_cfg.max_ctx, engine_cfg.kv_mode, Some(n_head_kv_l), None)?;
        let scratch0 = ActivationScratch::alloc_custom(
            &cfg,
            0,
            engine_cfg.max_batch,
            Some(n_head_l),
            Some(n_head_kv_l),
            Some(n_ff_l),
            Some(vocab_half0),
        )?;
        let mix0 = StaticMixPtrTables::build(&cfg, &scratch0, 0)?;

        let exps0 = Self::load_exps_vec(gguf, &cfg, &weights0, 0, 0)?;

        set_device(1)?;
        let stream1 = CudaStream::new(1)?;
        let weights1 = ModelWeights::load_tp2_rank(gguf, &cfg, 1, 1, &stream1)?;
        let kv1 = KvConvCache::alloc_custom(&cfg, 1, engine_cfg.max_ctx, engine_cfg.kv_mode, Some(n_head_kv_l), None)?;
        let scratch1 = ActivationScratch::alloc_custom(
            &cfg,
            1,
            engine_cfg.max_batch,
            Some(n_head_l),
            Some(n_head_kv_l),
            Some(n_ff_l),
            Some(vocab_half1),
        )?;
        let mix1 = StaticMixPtrTables::build(&cfg, &scratch1, 1)?;
        let exps1 = Self::load_exps_vec(gguf, &cfg, &weights1, 1, 1)?;

        let (free1, _) = mem_info(1)?;
        if free1 < 768 * 1024 * 1024 {
            bail!(
                "TP2 GPU 1 free VRAM ({:.1} MiB) below 768 MiB display safety margin",
                free1 as f64 / (1024.0 * 1024.0)
            );
        }

        let buf_len = engine_cfg.max_batch * cfg.n_embd;
        let buf_0_to_1 = [
            PinnedMappedBuffer::<f32>::alloc(buf_len)?,
            PinnedMappedBuffer::<f32>::alloc(buf_len)?,
        ];
        let buf_1_to_0 = [
            PinnedMappedBuffer::<f32>::alloc(buf_len)?,
            PinnedMappedBuffer::<f32>::alloc(buf_len)?,
        ];
        let flag_0_to_1 = [
            PinnedMappedBuffer::<u32>::alloc(1)?,
            PinnedMappedBuffer::<u32>::alloc(1)?,
        ];
        let flag_1_to_0 = [
            PinnedMappedBuffer::<u32>::alloc(1)?,
            PinnedMappedBuffer::<u32>::alloc(1)?,
        ];

        let ptr_0_to_1_dev0 = [
            buf_0_to_1[0].dev_ptr_for_device(0)?,
            buf_0_to_1[1].dev_ptr_for_device(0)?,
        ];
        let ptr_0_to_1_dev1 = [
            buf_0_to_1[0].dev_ptr_for_device(1)?,
            buf_0_to_1[1].dev_ptr_for_device(1)?,
        ];
        let ptr_1_to_0_dev0 = [
            buf_1_to_0[0].dev_ptr_for_device(0)?,
            buf_1_to_0[1].dev_ptr_for_device(0)?,
        ];
        let ptr_1_to_0_dev1 = [
            buf_1_to_0[0].dev_ptr_for_device(1)?,
            buf_1_to_0[1].dev_ptr_for_device(1)?,
        ];
        let fptr_0_to_1_dev0 = [
            flag_0_to_1[0].dev_ptr_for_device(0)?,
            flag_0_to_1[1].dev_ptr_for_device(0)?,
        ];
        let fptr_0_to_1_dev1 = [
            flag_0_to_1[0].dev_ptr_for_device(1)?,
            flag_0_to_1[1].dev_ptr_for_device(1)?,
        ];
        let fptr_1_to_0_dev0 = [
            flag_1_to_0[0].dev_ptr_for_device(0)?,
            flag_1_to_0[1].dev_ptr_for_device(0)?,
        ];
        let fptr_1_to_0_dev1 = [
            flag_1_to_0[0].dev_ptr_for_device(1)?,
            flag_1_to_0[1].dev_ptr_for_device(1)?,
        ];

        let flush_ev0 = CudaEvent::new(0)?;
        let flush_ev1 = CudaEvent::new(1)?;

        set_device(0)?;
        Ok(Self {
            engine_cfg,
            rank0: GpuRankState {
                device: 0,
                stream: stream0,
                cfg: cfg.clone(),
                weights: weights0,
                kv_cache: kv0,
                scratch: scratch0,
                mix_tables: mix0,
            },
            rank1: GpuRankState {
                device: 1,
                stream: stream1,
                cfg,
                weights: weights1,
                kv_cache: kv1,
                scratch: scratch1,
                mix_tables: mix1,
            },
            exps0,
            exps1,
            vocab_half0,
            vocab_half1,
            _buf_0_to_1: buf_0_to_1,
            _buf_1_to_0: buf_1_to_0,
            _flag_0_to_1: flag_0_to_1,
            _flag_1_to_0: flag_1_to_0,
            ptr_0_to_1_dev0,
            ptr_0_to_1_dev1,
            ptr_1_to_0_dev0,
            ptr_1_to_0_dev1,
            fptr_0_to_1_dev0,
            fptr_0_to_1_dev1,
            fptr_1_to_0_dev0,
            fptr_1_to_0_dev1,
            flush_ev0,
            flush_ev1,
            comm_seq: 0,
            cur_pos: 0,
        })
    }

    fn load_exps_vec(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        weights: &ModelWeights,
        rank: usize,
        device: i32,
    ) -> Result<Vec<Tp2LayerExperts>> {
        //Note: per-layer row_bytes can differ (IQ4_NL vs Q8_0)! Pass LayerWeights's row_bytes per layer!
        let _ = weights;
        set_device(device)?;
        let n_ff_full = cfg.n_ff_exp;
        let n_ff_l = n_ff_full / 2;
        let n_embd = cfg.n_embd;
        let mut out = Vec::with_capacity(cfg.n_layer_phys);
        for i in 0..cfg.n_layer_phys {
            let lw = weights.layer(i);
            let gu_bytes = gguf.tensor_bytes(&format!("blk.{i}.ffn_gate_up_exps.weight"))?;
            let dn_bytes = gguf.tensor_bytes(&format!("blk.{i}.ffn_down_exps.weight"))?;

            let gu_exp_full = lw.gate_up_row_bytes * (2 * n_ff_full);
            let gu_exp_local = lw.gate_up_row_bytes * (2 * n_ff_l);
            let mut gu_staged = Vec::with_capacity(cfg.n_expert * gu_exp_local);
            for e in 0..cfg.n_expert {
                let e_base = e * gu_exp_full;
                let gate_s = e_base + rank * n_ff_l * lw.gate_up_row_bytes;
                let up_s = e_base + (n_ff_full + rank * n_ff_l) * lw.gate_up_row_bytes;
                gu_staged.extend_from_slice(&gu_bytes[gate_s..gate_s + n_ff_l * lw.gate_up_row_bytes]);
                gu_staged.extend_from_slice(&gu_bytes[up_s..up_s + n_ff_l * lw.gate_up_row_bytes]);
            }
            let gu_buf = DeviceBuffer::alloc(device, gu_staged.len())?;
            gu_buf.copy_from_host(&gu_staged)?;

            let dn_full_rb = lw.down_type.row_bytes(n_ff_full);
            let dn_col_off = lw.down_type.row_bytes(rank * n_ff_l);
            let dn_exp_full = dn_full_rb * n_embd;
            let dn_exp_local = lw.down_row_bytes * n_embd;
            let mut dn_staged = Vec::with_capacity(cfg.n_expert * dn_exp_local);
            for e in 0..cfg.n_expert {
                let e_base = e * dn_exp_full;
                for r in 0..n_embd {
                    let s = e_base + r * dn_full_rb + dn_col_off;
                    dn_staged.extend_from_slice(&dn_bytes[s..s + lw.down_row_bytes]);
                }
            }
            let dn_buf = DeviceBuffer::alloc(device, dn_staged.len())?;
            dn_buf.copy_from_host(&dn_staged)?;

            let mut gu_ptrs = Vec::with_capacity(cfg.n_expert);
            let mut dn_ptrs = Vec::with_capacity(cfg.n_expert);
            for e in 0..cfg.n_expert {
                gu_ptrs.push(gu_buf.as_u64() + (e * gu_exp_local) as u64);
                dn_ptrs.push(dn_buf.as_u64() + (e * dn_exp_local) as u64);
            }
            let gu_ptrs_dev = DeviceBuffer::alloc(device, gu_ptrs.len() * 8)?;
            gu_ptrs_dev.copy_from_host(bytemuck::cast_slice(&gu_ptrs))?;
            let dn_ptrs_dev = DeviceBuffer::alloc(device, dn_ptrs.len() * 8)?;
            dn_ptrs_dev.copy_from_host(bytemuck::cast_slice(&dn_ptrs))?;

            out.push(Tp2LayerExperts {
                gate_up_buf: gu_buf,
                down_buf: dn_buf,
                gate_up_ptrs_dev: gu_ptrs_dev,
                down_ptrs_dev: dn_ptrs_dev,
            });
        }
        let _ = copy_h2d_raw_async;
        Ok(out)
    }

    pub fn cache_summary(&self) -> String {
        let m0 = Tp2LayerExperts::expert_bytes_total(&self.exps0) as f64 / (1024.0 * 1024.0);
        let m1 = Tp2LayerExperts::expert_bytes_total(&self.exps1) as f64 / (1024.0 * 1024.0);
        let free1_mb = mem_info(1).map(|(f, _)| f as f64 / (1024.0 * 1024.0)).unwrap_or(0.0);
        format!(
            "TP2(GPU0+GPU1, heads=12+12, ff=576+576) expert_vram=({m0:.1}+{m1:.1}) MiB gpu1_free={free1_mb:.1} MiB kv={:?}",
            self.rank0.kv_cache.kv_mode
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
        let n_elems = (m_usize * self.rank0.cfg.n_embd) as i32;
        let max_kv_len = (self.cur_pos + m_usize) as i32;
        let tok_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i32: Vec<i32> = (self.cur_pos..self.cur_pos + m_usize)
            .map(|p| p as i32)
            .collect();

        set_device(0)?;
        self.rank0.enqueue_inputs(&tok_i32, &pos_i32)?;
        set_device(1)?;
        self.rank1.enqueue_inputs(&tok_i32, &pos_i32)?;

        let exec_order = self.rank0.cfg.execution_order();
        for &slot in &exec_order {
            let ip = self.rank0.cfg.logical_layers[slot].phys_layer;

            // Phase A: Both GPUs run Attn -> wo -> push partial attn_proj
            self.comm_seq += 1;
            let seq_a = self.comm_seq;
            let slot_a = (seq_a & 1) as usize;

            set_device(0)?;
            self.rank0.enqueue_attn_through_wo(slot, m, max_kv_len, true)?;
            unsafe {
                launch_p2p_push(
                    self.rank0.scratch.attn_proj.ptr as *const f32,
                    self.ptr_0_to_1_dev0[slot_a] as *mut f32,
                    self.fptr_0_to_1_dev0[slot_a] as *mut u32,
                    seq_a,
                    n_elems,
                    self.rank0.stream.raw,
                );
            }
            self.flush_ev0.record(&self.rank0.stream)?;
            let _ = self.flush_ev0.is_ready()?;

            set_device(1)?;
            self.rank1.enqueue_attn_through_wo(slot, m, max_kv_len, true)?;
            unsafe {
                launch_p2p_push(
                    self.rank1.scratch.attn_proj.ptr as *const f32,
                    self.ptr_1_to_0_dev1[slot_a] as *mut f32,
                    self.fptr_1_to_0_dev1[slot_a] as *mut u32,
                    seq_a,
                    n_elems,
                    self.rank1.stream.raw,
                );
            }
            self.flush_ev1.record(&self.rank1.stream)?;
            let _ = self.flush_ev1.is_ready()?;

            // Phase B: Both GPUs wait-reduce attn_proj -> post_wo -> local MoE -> push partial moe_out
            self.comm_seq += 1;
            let seq_b = self.comm_seq;
            let slot_b = (seq_b & 1) as usize;

            set_device(0)?;
            unsafe {
                launch_p2p_wait_reduce(
                    self.rank0.scratch.attn_proj.ptr as *mut f32,
                    self.ptr_1_to_0_dev0[slot_a] as *const f32,
                    self.fptr_1_to_0_dev0[slot_a] as *const u32,
                    seq_a,
                    n_elems,
                    self.rank0.stream.raw,
                );
            }
            self.rank0.enqueue_post_wo_to_router(slot, m)?;
            self.rank0.enqueue_moe_local(
                ip,
                self.exps0[ip].gate_up_ptrs_dev.as_u64(),
                self.exps0[ip].down_ptrs_dev.as_u64(),
                m,
            );
            unsafe {
                launch_p2p_push(
                    self.rank0.scratch.moe_out.ptr as *const f32,
                    self.ptr_0_to_1_dev0[slot_b] as *mut f32,
                    self.fptr_0_to_1_dev0[slot_b] as *mut u32,
                    seq_b,
                    n_elems,
                    self.rank0.stream.raw,
                );
            }
            self.flush_ev0.record(&self.rank0.stream)?;
            let _ = self.flush_ev0.is_ready()?;

            set_device(1)?;
            unsafe {
                launch_p2p_wait_reduce(
                    self.rank1.scratch.attn_proj.ptr as *mut f32,
                    self.ptr_0_to_1_dev1[slot_a] as *const f32,
                    self.fptr_0_to_1_dev1[slot_a] as *const u32,
                    seq_a,
                    n_elems,
                    self.rank1.stream.raw,
                );
            }
            self.rank1.enqueue_post_wo_to_router(slot, m)?;
            self.rank1.enqueue_moe_local(
                ip,
                self.exps1[ip].gate_up_ptrs_dev.as_u64(),
                self.exps1[ip].down_ptrs_dev.as_u64(),
                m,
            );
            unsafe {
                launch_p2p_push(
                    self.rank1.scratch.moe_out.ptr as *const f32,
                    self.ptr_1_to_0_dev1[slot_b] as *mut f32,
                    self.fptr_1_to_0_dev1[slot_b] as *mut u32,
                    seq_b,
                    n_elems,
                    self.rank1.stream.raw,
                );
            }
            self.flush_ev1.record(&self.rank1.stream)?;
            let _ = self.flush_ev1.is_ready()?;

            // Phase C: Both GPUs wait-reduce moe_out -> ffn_post_conv -> residual
            set_device(0)?;
            unsafe {
                launch_p2p_wait_reduce(
                    self.rank0.scratch.moe_out.ptr as *mut f32,
                    self.ptr_1_to_0_dev0[slot_b] as *const f32,
                    self.fptr_1_to_0_dev0[slot_b] as *const u32,
                    seq_b,
                    n_elems,
                    self.rank0.stream.raw,
                );
            }
            self.rank0.enqueue_post_moe(slot, m);

            set_device(1)?;
            unsafe {
                launch_p2p_wait_reduce(
                    self.rank1.scratch.moe_out.ptr as *mut f32,
                    self.ptr_0_to_1_dev1[slot_b] as *const f32,
                    self.fptr_0_to_1_dev1[slot_b] as *const u32,
                    seq_b,
                    n_elems,
                    self.rank1.stream.raw,
                );
            }
            self.rank1.enqueue_post_moe(slot, m);
        }

        // Vocab-sharded lm_head + softcap + local (val, id) argmax
        set_device(0)?;
        self.rank0.enqueue_output_head(m, self.vocab_half0 as i32, true);
        set_device(1)?;
        self.rank1.enqueue_output_head(m, self.vocab_half1 as i32, true);

        set_device(0)?;
        self.rank0.stream.sync()?;
        set_device(1)?;
        self.rank1.stream.sync()?;

        self.cur_pos += m_usize;

        let (mut v0, mut id0) = ([0.0f32; 1], [0i32; 1]);
        let (mut v1, mut id1) = ([0.0f32; 1], [0i32; 1]);
        set_device(0)?;
        self.rank0.scratch.argmax_val.copy_to_host(bytemuck::cast_slice_mut(&mut v0))?;
        self.rank0.scratch.argmax_id.copy_to_host(bytemuck::cast_slice_mut(&mut id0))?;
        set_device(1)?;
        self.rank1.scratch.argmax_val.copy_to_host(bytemuck::cast_slice_mut(&mut v1))?;
        self.rank1.scratch.argmax_id.copy_to_host(bytemuck::cast_slice_mut(&mut id1))?;

        let global_id = if v1[0] > v0[0] {
            (id1[0] as usize + self.vocab_half0) as u32
        } else {
            id0[0] as u32
        };

        let logits_out = if capture_logits {
            let mut full = vec![0.0f32; self.rank0.cfg.n_vocab_unpadded];
            set_device(0)?;
            self.rank0
                .scratch
                .logits
                .copy_to_host(bytemuck::cast_slice_mut(&mut full[..self.vocab_half0]))?;
            set_device(1)?;
            self.rank1
                .scratch
                .logits
                .copy_to_host(bytemuck::cast_slice_mut(&mut full[self.vocab_half0..]))?;
            Some(full)
        } else {
            None
        };

        Ok((global_id, logits_out))
    }
}
