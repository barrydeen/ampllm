use crate::kv_cache::KvConvCache;
use crate::scratch::ActivationScratch;
use crate::weights::{DeviceTensor, LoopCtrlWeights, ModelWeights};
use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{CudaEvent, CudaStream, CudaStreamHandle, DeviceBuffer};
use ampllm_gguf::{GgmlType, Inkling2Config};
use anyhow::Result;
use std::ffi::c_void;

/// Pre-uploaded static device pointer lists for every AttnRes `mix()` call in the forward graph.
pub struct StaticMixPtrTables {
    pub attn_mix: Vec<Option<(DeviceBuffer, usize)>>,
    pub ffn_mix: Vec<(DeviceBuffer, usize)>,
    pub ffn_mix_pingpong: Vec<(DeviceBuffer, usize)>,
    pub loop_msg_mix: Vec<Option<(DeviceBuffer, usize)>>,
    pub output_mix: (DeviceBuffer, usize),
}

impl StaticMixPtrTables {
    fn alloc_ptrs(device: i32, ptrs: &[u64]) -> Result<(DeviceBuffer, usize)> {
        let buf = DeviceBuffer::alloc(device, ptrs.len() * 8)?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8) };
        buf.copy_from_host(bytes)?;
        Ok((buf, ptrs.len()))
    }

    pub fn build(cfg: &Inkling2Config, scratch: &ActivationScratch, device: i32) -> Result<Self> {
        let mut attn_mix = Vec::with_capacity(cfg.n_layer_logical);
        let mut ffn_mix = Vec::with_capacity(cfg.n_layer_logical);
        let mut ffn_mix_pingpong = Vec::with_capacity(cfg.n_layer_logical);
        let mut loop_msg_mix = Vec::with_capacity(cfg.n_layer_logical);

        let cur = scratch.cur.as_u64();
        let ffn_inp = scratch.ffn_inp.as_u64();
        let b0 = scratch.blocks[0].as_u64();
        let b1 = scratch.blocks[1].as_u64();
        let b2 = scratch.blocks[2].as_u64();
        let b3 = scratch.blocks[3].as_u64();
        let bank0 = scratch.blocks[4].as_u64();
        let bank_out = scratch.bank_out.as_u64();

        for slot in 0..cfg.n_layer_logical {
            let info = &cfg.logical_layers[slot];
            let ip = info.phys_layer;

            let mut blocks_before_attn: Vec<u64> = Vec::new();
            if ip > 0 {
                blocks_before_attn.push(b0);
            }
            if ip > 6 {
                blocks_before_attn.push(b1);
            }
            if ip > 12 {
                blocks_before_attn.push(b2);
            }
            if ip > 18 {
                blocks_before_attn.push(b3);
            }

            if ip > 0 {
                let mut states = blocks_before_attn.clone();
                states.push(cur);
                attn_mix.push(Some(Self::alloc_ptrs(device, &states)?));
            } else {
                attn_mix.push(None);
            }

            let mut blocks_before_ffn = blocks_before_attn;
            if ip % cfg.attn_res_block_size == 0 {
                let b_idx = ip / cfg.attn_res_block_size;
                blocks_before_ffn.push(scratch.blocks[b_idx].as_u64());
            }
            let mut ffn_states_pp = blocks_before_ffn.clone();
            let mut ffn_states = blocks_before_ffn;
            ffn_states.push(cur);
            ffn_mix.push(Self::alloc_ptrs(device, &ffn_states)?);
            ffn_states_pp.push(ffn_inp);
            ffn_mix_pingpong.push(Self::alloc_ptrs(device, &ffn_states_pp)?);

            if info.pass > 0 {
                loop_msg_mix.push(Some(Self::alloc_ptrs(device, &[bank0, bank_out])?));
            } else {
                loop_msg_mix.push(None);
            }
        }

        let output_mix = Self::alloc_ptrs(device, &[b0, b1, b2, b3, cur])?;
        Ok(Self {
            attn_mix,
            ffn_mix,
            ffn_mix_pingpong,
            loop_msg_mix,
            output_mix,
        })
    }
}

/// Shared single-GPU execution primitives usable across Single, Helper, Pipeline, and TP2 modes.
pub struct GpuRankState {
    pub device: i32,
    pub stream: CudaStream,
    pub cfg: Inkling2Config,
    pub weights: ModelWeights,
    pub kv_cache: KvConvCache,
    pub scratch: ActivationScratch,
    pub mix_tables: StaticMixPtrTables,
}

impl GpuRankState {
    #[inline]
    pub fn linear_on_stream(
        &self,
        w: &DeviceTensor,
        x_f32: *const f32,
        x_q8_reusable: bool,
        y: *mut f32,
        n: i32,
        k: i32,
        m: i32,
    ) {
        if let (Some(q), Some(d)) = (&w.q8_planar_q, &w.q8_planar_d) {
            if m == 1 {
                unsafe {
                    launch_mv_q8_planar_m1(
                        q.ptr, d.ptr, n, x_f32, y,
                        std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                        std::ptr::null(), 0,
                        std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                        std::ptr::null(), 0,
                        std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                        std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        0,
                        k,
                        self.stream.raw,
                    );
                }
                return;
            }
            let max_chunk_rows = (self.scratch.x16_0.bytes / (2 * (k as usize).max(1))).max(1) as i32;
            let mut off = 0i32;
            while off < m {
                let cm = (m - off).min(max_chunk_rows);
                unsafe {
                    let x_sub = x_f32.add((off as usize) * (k as usize));
                    let y_sub = y.add((off as usize) * (n as usize));
                    launch_f32_to_f16_fast(
                        x_sub,
                        self.scratch.x16_0.ptr as *mut u16,
                        (cm as usize) * (k as usize),
                        self.stream.raw,
                    );
                    launch_q8_mma_tile(
                        q.ptr as *const i8,
                        d.ptr as *const u16,
                        n,
                        k,
                        self.scratch.x16_0.ptr as *const u16,
                        k,
                        cm,
                        y_sub,
                        n,
                        self.stream.raw,
                    );
                }
                off += cm;
            }
            return;
        }
        let is_dense = matches!(w.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16);
        unsafe {
            if !is_dense && !x_q8_reusable {
                launch_quantize_q8_1(x_f32, self.scratch.act_q8_1.ptr, m, k, self.stream.raw);
            }
            launch_linear_q8_1(
                w.ggml_type as i32,
                w.buf.ptr,
                self.scratch.act_q8_1.ptr,
                x_f32,
                y,
                n,
                k,
                m,
                self.stream.raw,
            );
        }
    }

    pub fn run_loop_control(
        &self,
        cur: *mut f32,
        msg: *const f32,
        ctrl: &LoopCtrlWeights,
        ctrl_idx: usize,
        m: i32,
    ) {
        let n_embd = self.cfg.n_embd as i32;
        let rank = self.cfg.loop_gate_rank as i32;
        unsafe {
            launch_concat_two_rmsnorm_f32(
                cur,
                msg,
                self.scratch.ctrl_concat.ptr as *mut f32,
                n_embd,
                m,
                self.cfg.rms_norm_eps,
                self.stream.raw,
            );
            self.linear_on_stream(
                &ctrl.down[ctrl_idx],
                self.scratch.ctrl_concat.ptr as *const f32,
                false,
                self.scratch.ctrl_gate.ptr as *mut f32,
                rank,
                2 * n_embd,
                m,
            );
            self.linear_on_stream(
                &ctrl.gain[ctrl_idx],
                self.scratch.ctrl_gate.ptr as *const f32,
                false,
                self.scratch.ctrl_gain.ptr as *mut f32,
                n_embd,
                rank,
                m,
            );
            self.linear_on_stream(
                &ctrl.forget[ctrl_idx],
                self.scratch.ctrl_gate.ptr as *const f32,
                false,
                self.scratch.ctrl_forget.ptr as *mut f32,
                n_embd,
                rank,
                m,
            );
            launch_apply_loop_ctrl_f32(
                cur,
                msg,
                self.scratch.ctrl_forget.ptr as *const f32,
                self.scratch.ctrl_gain.ptr as *const f32,
                m * n_embd,
                self.stream.raw,
            );
        }
    }

    pub fn enqueue_inputs(&self, tok_i32: &[i32], pos_i32: &[i32]) -> Result<()> {
        let m = tok_i32.len() as i32;
        let n_embd = self.cfg.n_embd as i32;
        self.scratch
            .token_ids
            .copy_from_host_async(bytemuck::cast_slice(tok_i32), &self.stream)?;
        self.scratch
            .pos_ids
            .copy_from_host_async(bytemuck::cast_slice(pos_i32), &self.stream)?;

        let tok_embd = self.weights.tok_embd.as_ref().expect("tok_embd required");
        let tok_norm = self
            .weights
            .tok_embd_norm
            .as_ref()
            .expect("tok_embd_norm required");
        unsafe {
            launch_compute_tau_f32(
                self.scratch.pos_ids.ptr as *const i32,
                self.scratch.tau.ptr as *mut f32,
                m,
                self.cfg.log_scale_floor,
                self.cfg.log_scale_alpha,
                self.stream.raw,
            );
            launch_embed_lookup_rmsnorm(
                tok_embd.buf.ptr,
                tok_embd.ggml_type as i32,
                self.scratch.token_ids.ptr as *const i32,
                tok_norm.as_f32(),
                self.scratch.cur.ptr as *mut f32,
                n_embd,
                m,
                self.cfg.rms_norm_eps,
                self.stream.raw,
            );
        }
        Ok(())
    }

    pub fn enqueue_compute_tau_only(&self, pos_i32: &[i32]) -> Result<()> {
        let m = pos_i32.len() as i32;
        self.scratch
            .pos_ids
            .copy_from_host_async(bytemuck::cast_slice(pos_i32), &self.stream)?;
        unsafe {
            launch_compute_tau_f32(
                self.scratch.pos_ids.ptr as *const i32,
                self.scratch.tau.ptr as *mut f32,
                m,
                self.cfg.log_scale_floor,
                self.cfg.log_scale_alpha,
                self.stream.raw,
            );
        }
        Ok(())
    }

    /// Loop-core boundary state update + attention up through `wo` into `scratch.attn_proj`.
    pub fn enqueue_attn_through_wo(
        &self,
        slot: usize,
        m: i32,
        max_kv_len: i32,
        handle_loop_boundary: bool,
    ) -> Result<()> {
        let info = &self.cfg.logical_layers[slot];
        let ip = info.phys_layer;
        let pass = info.pass;
        let layer = self.weights.layer(ip);
        let slot_kv = self.kv_cache.slot(slot);

        let n_embd = self.cfg.n_embd as i32;
        let n_head = self.scratch.n_head_local as i32;
        let n_head_kv = self.scratch.n_head_kv_local as i32;
        let head_dim = self.cfg.head_dim as i32;
        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let rel_dim = n_head * (self.cfg.d_rel as i32);
        let dyn_dim = n_head_kv * (self.cfg.shortconv_l_cache as i32);
        let eps = self.cfg.rms_norm_eps;
        let hid_bytes = (m as usize) * self.cfg.n_embd * 4;
        let kv_is_q8 = match self.kv_cache.kv_mode {
            crate::kv_cache::KvQuantMode::F16 => 0,
            crate::kv_cache::KvQuantMode::Q8_0 => 1,
            crate::kv_cache::KvQuantMode::Fp8 => 2,
        };

        if handle_loop_boundary && self.cfg.is_looped && ip == self.cfg.loop_start && pass == 0 {
            self.scratch
                .prelude
                .copy_from_device_async(&self.scratch.cur, hid_bytes, &self.stream)?;
        }
        if handle_loop_boundary && self.cfg.is_looped && ip == self.cfg.loop_start && pass > 0 {
            self.scratch
                .blocks[4]
                .copy_from_device_async(&self.scratch.blocks[2], hid_bytes, &self.stream)?;
            self.scratch
                .bank_out
                .copy_from_device_async(&self.scratch.cur, hid_bytes, &self.stream)?;
            self.scratch
                .cur
                .copy_from_device_async(&self.scratch.prelude, hid_bytes, &self.stream)?;
            if let Some(ref all_mq) = self.weights.loop_msg_queries_all {
                let (ptrs_buf, n_states) = self.mix_tables.loop_msg_mix[slot].as_ref().unwrap();
                let n_core = (self.cfg.loop_end - self.cfg.loop_start) as i32;
                unsafe {
                    launch_attn_res_mix_jobs_f32(
                        ptrs_buf.ptr as *const *const f32,
                        *n_states as i32,
                        all_mq.ptr as *const f32,
                        self.scratch.loop_messages.ptr as *mut f32,
                        n_embd,
                        4,
                        n_core,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            }
        }

        let batched_loop_msg = handle_loop_boundary && self.weights.loop_msg_queries_all.is_some();
        if pass > 0 && !batched_loop_msg {
            let lc = layer.loop_ctrl.as_ref().unwrap();
            let (ptrs_buf, n_states) = self.mix_tables.loop_msg_mix[slot].as_ref().unwrap();
            unsafe {
                launch_attn_res_mix_f32(
                    ptrs_buf.ptr as *const *const f32,
                    *n_states as i32,
                    lc.msg_query.as_f32(),
                    self.scratch.loop_messages.ptr as *mut f32,
                    n_embd,
                    4,
                    m,
                    eps,
                    self.stream.raw,
                );
            }
        }

        self.scratch
            .inp_sa
            .copy_from_device_async(&self.scratch.cur, hid_bytes, &self.stream)?;

        let share_kv_inp = layer.is_swa && pass == 0;
        if pass > 0 {
            if let (Some(res_q), Some((ptrs_buf, n_states))) =
                (&layer.attn_res_query, &self.mix_tables.attn_mix[slot])
            {
                let n_q = if layer.is_swa { 1 } else { 3 };
                unsafe {
                    launch_attn_res_mix_f32(
                        ptrs_buf.ptr as *const *const f32,
                        *n_states as i32,
                        res_q.as_f32(),
                        self.scratch.mix_out.ptr as *mut f32,
                        n_embd,
                        n_q,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            }
            let lc = layer.loop_ctrl.as_ref().unwrap();
            let base_x = if layer.attn_res_query.is_some() {
                self.scratch.mix_out.ptr as *const f32
            } else {
                self.scratch.cur.ptr as *const f32
            };
            let x0 = base_x;
            let x1 = if !layer.is_swa && layer.attn_res_query.is_some() {
                unsafe { (base_x as *const u8).add(hid_bytes) as *const f32 }
            } else {
                base_x
            };
            let x2 = if !layer.is_swa && layer.attn_res_query.is_some() {
                unsafe { (base_x as *const u8).add(2 * hid_bytes) as *const f32 }
            } else {
                base_x
            };
            let msg_job_off = if batched_loop_msg {
                (ip - self.cfg.loop_start) * 4 * hid_bytes
            } else {
                0
            };
            let m0 = unsafe {
                (self.scratch.loop_messages.ptr as *const u8).add(msg_job_off) as *const f32
            };
            let m1 = unsafe { (m0 as *const u8).add(hid_bytes) as *const f32 };
            let m2 = unsafe { (m0 as *const u8).add(2 * hid_bytes) as *const f32 };
            let all_f32 = matches!(lc.down[0].ggml_type, GgmlType::F32)
                && matches!(lc.gain[0].ggml_type, GgmlType::F32)
                && matches!(lc.forget[0].ggml_type, GgmlType::F32);
            if all_f32 {
                unsafe {
                    launch_loop_gates_fused_f32(
                        x0, x1, x2,
                        m0, m1, m2,
                        lc.down[0].as_f32(), lc.down[1].as_f32(), lc.down[2].as_f32(),
                        lc.forget[0].as_f32(), lc.forget[1].as_f32(), lc.forget[2].as_f32(),
                        lc.gain[0].as_f32(), lc.gain[1].as_f32(), lc.gain[2].as_f32(),
                        self.scratch.inputs_qkv[0].ptr as *mut f32,
                        self.scratch.inputs_qkv[1].ptr as *mut f32,
                        self.scratch.inputs_qkv[2].ptr as *mut f32,
                        self.scratch.ctrl_gate.ptr as *mut f32,
                        3,
                        m,
                        n_embd,
                        self.cfg.loop_gate_rank as i32,
                        eps,
                        self.stream.raw,
                    );
                }
            } else {
                let xs = [x0, x1, x2];
                for j in 0..3 {
                    unsafe {
                        cuda_check_copy_d2d(
                            self.scratch.inputs_qkv[j].ptr,
                            xs[j] as *const c_void,
                            hid_bytes,
                            self.stream.raw,
                        )?;
                        let msg_ptr = (self.scratch.loop_messages.ptr as *const u8)
                            .add(msg_job_off + j * hid_bytes)
                            as *const f32;
                        self.run_loop_control(
                            self.scratch.inputs_qkv[j].ptr as *mut f32,
                            msg_ptr,
                            lc,
                            j,
                            m,
                        );
                    }
                }
            }
            let (nw0, nw1, nw2) = if layer.is_swa {
                let w = layer.attn_norm.as_ref().unwrap().as_f32();
                (w, w, w)
            } else {
                (
                    layer.attn_q_inp_norm.as_ref().unwrap().as_f32(),
                    layer.attn_k_inp_norm.as_ref().unwrap().as_f32(),
                    layer.attn_v_inp_norm.as_ref().unwrap().as_f32(),
                )
            };
            unsafe {
                launch_rmsnorm_multi_f32(
                    self.scratch.inputs_qkv[0].ptr as *const f32,
                    self.scratch.inputs_qkv[1].ptr as *const f32,
                    self.scratch.inputs_qkv[2].ptr as *const f32,
                    nw0, nw1, nw2,
                    self.scratch.inputs_norm[0].ptr as *mut f32,
                    self.scratch.inputs_norm[1].ptr as *mut f32,
                    self.scratch.inputs_norm[2].ptr as *mut f32,
                    3,
                    n_embd,
                    m,
                    eps,
                    self.stream.raw,
                );
            }
        } else if let (Some(res_q), Some((ptrs_buf, n_states))) =
            (&layer.attn_res_query, &self.mix_tables.attn_mix[slot])
        {
            if layer.is_swa {
                let nw = layer.attn_norm.as_ref().unwrap().as_f32();
                unsafe {
                    launch_attn_res_mix_rms_f32(
                        ptrs_buf.ptr as *const *const f32,
                        *n_states as i32,
                        res_q.as_f32(),
                        nw,
                        std::ptr::null(),
                        std::ptr::null(),
                        self.scratch.inputs_norm[0].ptr as *mut f32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        n_embd,
                        1,
                        m,
                        eps,
                        eps,
                        self.stream.raw,
                    );
                }
            } else {
                unsafe {
                    launch_attn_res_mix_rms_f32(
                        ptrs_buf.ptr as *const *const f32,
                        *n_states as i32,
                        res_q.as_f32(),
                        layer.attn_q_inp_norm.as_ref().unwrap().as_f32(),
                        layer.attn_k_inp_norm.as_ref().unwrap().as_f32(),
                        layer.attn_v_inp_norm.as_ref().unwrap().as_f32(),
                        self.scratch.inputs_norm[0].ptr as *mut f32,
                        self.scratch.inputs_norm[1].ptr as *mut f32,
                        self.scratch.inputs_norm[2].ptr as *mut f32,
                        n_embd,
                        3,
                        m,
                        eps,
                        eps,
                        self.stream.raw,
                    );
                }
            }
        } else if layer.is_swa {
            let norm_w = layer.attn_norm.as_ref().unwrap().as_f32();
            unsafe {
                launch_rmsnorm_f32(
                    self.scratch.cur.ptr as *const f32,
                    norm_w,
                    self.scratch.inputs_norm[0].ptr as *mut f32,
                    n_embd,
                    m,
                    eps,
                    self.stream.raw,
                );
            }
        } else {
            unsafe {
                launch_rmsnorm_multi_f32(
                    self.scratch.cur.ptr as *const f32,
                    self.scratch.cur.ptr as *const f32,
                    self.scratch.cur.ptr as *const f32,
                    layer.attn_q_inp_norm.as_ref().unwrap().as_f32(),
                    layer.attn_k_inp_norm.as_ref().unwrap().as_f32(),
                    layer.attn_v_inp_norm.as_ref().unwrap().as_f32(),
                    self.scratch.inputs_norm[0].ptr as *mut f32,
                    self.scratch.inputs_norm[1].ptr as *mut f32,
                    self.scratch.inputs_norm[2].ptr as *mut f32,
                    3,
                    n_embd,
                    m,
                    eps,
                    self.stream.raw,
                );
            }
        }

        let in0 = self.scratch.inputs_norm[0].ptr as *const f32;
        let in1 = if share_kv_inp {
            in0
        } else {
            self.scratch.inputs_norm[1].ptr as *const f32
        };
        let in2 = if share_kv_inp {
            in0
        } else {
            self.scratch.inputs_norm[2].ptr as *const f32
        };

        let same_qkvrg_quant = layer.wq.ggml_type == layer.wk.ggml_type
            && layer.wq.ggml_type == layer.wv.ggml_type
            && layer.wq.ggml_type == layer.attn_rel.ggml_type
            && layer.wq.ggml_type == layer.attn_gate.ggml_type
            && !matches!(
                layer.wq.ggml_type,
                GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16
            );

        let all_planar_qkvrg = layer.wq.q8_planar_q.is_some()
            && layer.wk.q8_planar_q.is_some()
            && layer.wv.q8_planar_q.is_some()
            && layer.attn_rel.q8_planar_q.is_some()
            && layer.attn_gate.q8_planar_q.is_some();

        let sink_fused = matches!(layer.attn_sink_proj.ggml_type, GgmlType::F32);
        let dyn_conv_fused = m == 1
            && !layer.is_swa
            && layer
                .attn_k_conv_proj
                .as_ref()
                .map(|t| matches!(t.ggml_type, GgmlType::F32))
                .unwrap_or(false)
            && layer
                .attn_v_conv_proj
                .as_ref()
                .map(|t| matches!(t.ggml_type, GgmlType::F32))
                .unwrap_or(false);

        if m == 1 && all_planar_qkvrg {
            let (wq_q, wq_d) = (layer.wq.q8_planar_q.as_ref().unwrap().ptr, layer.wq.q8_planar_d.as_ref().unwrap().ptr);
            let (wk_q, wk_d) = (layer.wk.q8_planar_q.as_ref().unwrap().ptr, layer.wk.q8_planar_d.as_ref().unwrap().ptr);
            let (wv_q, wv_d) = (layer.wv.q8_planar_q.as_ref().unwrap().ptr, layer.wv.q8_planar_d.as_ref().unwrap().ptr);
            let (wr_q, wr_d) = (layer.attn_rel.q8_planar_q.as_ref().unwrap().ptr, layer.attn_rel.q8_planar_d.as_ref().unwrap().ptr);
            let (wg_q, wg_d) = (layer.attn_gate.q8_planar_q.as_ref().unwrap().ptr, layer.attn_gate.q8_planar_d.as_ref().unwrap().ptr);
            unsafe {
                if layer.is_swa {
                    launch_mv_q8_planar_m1(
                        wq_q, wq_d, q_dim, in0, self.scratch.q_cur.ptr as *mut f32,
                        wk_q, wk_d, kv_dim, in1, self.scratch.k_cur.ptr as *mut f32,
                        layer.attn_k_conv.as_f32(), 0,
                        wv_q, wv_d, kv_dim, in2, self.scratch.v_cur.ptr as *mut f32,
                        layer.attn_v_conv.as_f32(), kv_dim,
                        wr_q, wr_d, rel_dim, in0, self.scratch.r_cur.ptr as *mut f32,
                        wg_q, wg_d, q_dim, in0, self.scratch.attn_gate_buf.ptr as *mut f32,
                        std::ptr::null(),
                        slot_kv.conv_history.ptr as *mut f32,
                        self.cfg.shortconv_l_cache as i32,
                        n_embd,
                        self.stream.raw,
                    );
                } else if dyn_conv_fused {
                    launch_mv_q8_planar_m1(
                        wq_q, wq_d, q_dim, in0, self.scratch.q_cur.ptr as *mut f32,
                        wk_q, wk_d, kv_dim, in1, self.scratch.k_cur.ptr as *mut f32,
                        std::ptr::null(), 0,
                        wv_q, wv_d, kv_dim, in2, self.scratch.v_cur.ptr as *mut f32,
                        std::ptr::null(), 0,
                        wr_q, wr_d, rel_dim, in0, self.scratch.r_cur.ptr as *mut f32,
                        wg_q, wg_d, q_dim, in0, self.scratch.attn_gate_buf.ptr as *mut f32,
                        std::ptr::null(),
                        slot_kv.conv_history.ptr as *mut f32,
                        self.cfg.shortconv_l_cache as i32,
                        n_embd,
                        self.stream.raw,
                    );
                } else {
                    launch_mv_q8_planar_m1(
                        wq_q, wq_d, q_dim, in0, self.scratch.q_cur.ptr as *mut f32,
                        wk_q, wk_d, kv_dim, in1, self.scratch.k_raw.ptr as *mut f32,
                        std::ptr::null(), 0,
                        wv_q, wv_d, kv_dim, in2, self.scratch.v_raw.ptr as *mut f32,
                        std::ptr::null(), 0,
                        wr_q, wr_d, rel_dim, in0, self.scratch.r_cur.ptr as *mut f32,
                        wg_q, wg_d, q_dim, in0, self.scratch.attn_gate_buf.ptr as *mut f32,
                        std::ptr::null(),
                        slot_kv.conv_history.ptr as *mut f32,
                        self.cfg.shortconv_l_cache as i32,
                        n_embd,
                        self.stream.raw,
                    );
                    let k_dyn_ptr = if let Some(kp) = &layer.attn_k_conv_proj {
                        self.linear_on_stream(kp, in1, false, self.scratch.dyn_proj.ptr as *mut f32, dyn_dim, n_embd, 1);
                        self.scratch.dyn_proj.ptr as *const f32
                    } else {
                        std::ptr::null()
                    };
                    launch_short_conv_f32(
                        self.scratch.k_raw.ptr as *const f32,
                        layer.attn_k_conv.as_f32(),
                        k_dyn_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.k_cur.ptr as *mut f32,
                        kv_dim,
                        n_head_kv,
                        1,
                        0,
                        self.cfg.shortconv_l_cache as i32,
                        self.stream.raw,
                    );
                    let v_dyn_ptr = if let Some(vp) = &layer.attn_v_conv_proj {
                        self.linear_on_stream(vp, in2, false, self.scratch.dyn_proj.ptr as *mut f32, dyn_dim, n_embd, 1);
                        self.scratch.dyn_proj.ptr as *const f32
                    } else {
                        std::ptr::null()
                    };
                    launch_short_conv_f32(
                        self.scratch.v_raw.ptr as *const f32,
                        layer.attn_v_conv.as_f32(),
                        v_dyn_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.v_cur.ptr as *mut f32,
                        kv_dim,
                        n_head_kv,
                        1,
                        kv_dim,
                        self.cfg.shortconv_l_cache as i32,
                        self.stream.raw,
                    );
                }
            }
            if !sink_fused {
                self.linear_on_stream(&layer.attn_sink_proj, in0, false, self.scratch.sinks.ptr as *mut f32, n_head, n_embd, 1);
            }
        } else if m == 1 && same_qkvrg_quant {
            let row_bytes = layer.wq.ggml_type.row_bytes(self.cfg.n_embd);
            let x0_q8 = self.scratch.act_q8_1.ptr;
            let (x1_q8, x2_q8) = if share_kv_inp {
                (x0_q8, x0_q8)
            } else {
                (self.scratch.act_q8_1_k.ptr, self.scratch.act_q8_1_v.ptr)
            };
            unsafe {
                launch_quantize_q8_1(in0, x0_q8, 1, n_embd, self.stream.raw);
                if !share_kv_inp {
                    launch_quantize_q8_1(in1, x1_q8, 1, n_embd, self.stream.raw);
                    launch_quantize_q8_1(in2, x2_q8, 1, n_embd, self.stream.raw);
                }
                if layer.is_swa {
                    launch_fused_qkvrg_m1(
                        layer.wq.ggml_type as i32,
                        layer.wq.buf.ptr, q_dim, x0_q8, self.scratch.q_cur.ptr as *mut f32,
                        layer.wk.buf.ptr, kv_dim, x1_q8, self.scratch.k_cur.ptr as *mut f32,
                        layer.attn_k_conv.as_f32(), 0,
                        layer.wv.buf.ptr, kv_dim, x2_q8, self.scratch.v_cur.ptr as *mut f32,
                        layer.attn_v_conv.as_f32(), kv_dim,
                        layer.attn_rel.buf.ptr, rel_dim, x0_q8, self.scratch.r_cur.ptr as *mut f32,
                        layer.attn_gate.buf.ptr, q_dim, x0_q8, self.scratch.attn_gate_buf.ptr as *mut f32,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.cfg.shortconv_l_cache as i32,
                        n_embd,
                        row_bytes,
                        self.stream.raw,
                    );
                } else if dyn_conv_fused {
                    launch_fused_qkvrg_m1(
                        layer.wq.ggml_type as i32,
                        layer.wq.buf.ptr, q_dim, x0_q8, self.scratch.q_cur.ptr as *mut f32,
                        layer.wk.buf.ptr, kv_dim, x1_q8, self.scratch.k_cur.ptr as *mut f32,
                        std::ptr::null(), 0,
                        layer.wv.buf.ptr, kv_dim, x2_q8, self.scratch.v_cur.ptr as *mut f32,
                        std::ptr::null(), 0,
                        layer.attn_rel.buf.ptr, rel_dim, x0_q8, self.scratch.r_cur.ptr as *mut f32,
                        layer.attn_gate.buf.ptr, q_dim, x0_q8, self.scratch.attn_gate_buf.ptr as *mut f32,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.cfg.shortconv_l_cache as i32,
                        n_embd,
                        row_bytes,
                        self.stream.raw,
                    );
                } else {
                    launch_fused_qkvrg_m1(
                        layer.wq.ggml_type as i32,
                        layer.wq.buf.ptr, q_dim, x0_q8, self.scratch.q_cur.ptr as *mut f32,
                        layer.wk.buf.ptr, kv_dim, x1_q8, self.scratch.k_raw.ptr as *mut f32,
                        std::ptr::null(), 0,
                        layer.wv.buf.ptr, kv_dim, x2_q8, self.scratch.v_raw.ptr as *mut f32,
                        std::ptr::null(), 0,
                        layer.attn_rel.buf.ptr, rel_dim, x0_q8, self.scratch.r_cur.ptr as *mut f32,
                        layer.attn_gate.buf.ptr, q_dim, x0_q8, self.scratch.attn_gate_buf.ptr as *mut f32,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.cfg.shortconv_l_cache as i32,
                        n_embd,
                        row_bytes,
                        self.stream.raw,
                    );
                    let k_dyn_ptr = if let Some(kp) = &layer.attn_k_conv_proj {
                        if matches!(kp.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16) {
                            launch_linear_q8_1(
                                kp.ggml_type as i32,
                                kp.buf.ptr,
                                x1_q8,
                                in1,
                                self.scratch.dyn_proj.ptr as *mut f32,
                                dyn_dim,
                                n_embd,
                                1,
                                self.stream.raw,
                            );
                        } else {
                            launch_linear_q8_1(
                                kp.ggml_type as i32,
                                kp.buf.ptr,
                                x1_q8,
                                in1,
                                self.scratch.dyn_proj.ptr as *mut f32,
                                dyn_dim,
                                n_embd,
                                1,
                                self.stream.raw,
                            );
                        }
                        self.scratch.dyn_proj.ptr as *const f32
                    } else {
                        std::ptr::null()
                    };
                    launch_short_conv_f32(
                        self.scratch.k_raw.ptr as *const f32,
                        layer.attn_k_conv.as_f32(),
                        k_dyn_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.k_cur.ptr as *mut f32,
                        kv_dim,
                        n_head_kv,
                        1,
                        0,
                        self.cfg.shortconv_l_cache as i32,
                        self.stream.raw,
                    );
                    let v_dyn_ptr = if let Some(vp) = &layer.attn_v_conv_proj {
                        launch_linear_q8_1(
                            vp.ggml_type as i32,
                            vp.buf.ptr,
                            x2_q8,
                            in2,
                            self.scratch.dyn_proj.ptr as *mut f32,
                            dyn_dim,
                            n_embd,
                            1,
                            self.stream.raw,
                        );
                        self.scratch.dyn_proj.ptr as *const f32
                    } else {
                        std::ptr::null()
                    };
                    launch_short_conv_f32(
                        self.scratch.v_raw.ptr as *const f32,
                        layer.attn_v_conv.as_f32(),
                        v_dyn_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.v_cur.ptr as *mut f32,
                        kv_dim,
                        n_head_kv,
                        1,
                        kv_dim,
                        self.cfg.shortconv_l_cache as i32,
                        self.stream.raw,
                    );
                }
            }
            if !sink_fused {
                self.linear_on_stream(&layer.attn_sink_proj, in0, true, self.scratch.sinks.ptr as *mut f32, n_head, n_embd, 1);
            }
        } else {
            unsafe {
                launch_quantize_q8_1(in0, self.scratch.act_q8_1.ptr, m, n_embd, self.stream.raw);
            }
            self.linear_on_stream(&layer.wq, in0, true, self.scratch.q_cur.ptr as *mut f32, q_dim, n_embd, m);
            self.linear_on_stream(&layer.attn_rel, in0, true, self.scratch.r_cur.ptr as *mut f32, rel_dim, n_embd, m);
            if !sink_fused {
                self.linear_on_stream(&layer.attn_sink_proj, in0, true, self.scratch.sinks.ptr as *mut f32, n_head, n_embd, m);
            }
            self.linear_on_stream(&layer.attn_gate, in0, true, self.scratch.attn_gate_buf.ptr as *mut f32, q_dim, n_embd, m);

            self.linear_on_stream(&layer.wk, in1, share_kv_inp, self.scratch.k_raw.ptr as *mut f32, kv_dim, n_embd, m);
            let k_dyn_ptr = if let Some(kp) = &layer.attn_k_conv_proj {
                self.linear_on_stream(kp, in1, true, self.scratch.dyn_proj.ptr as *mut f32, dyn_dim, n_embd, m);
                self.scratch.dyn_proj.ptr as *const f32
            } else {
                std::ptr::null()
            };
            unsafe {
                launch_short_conv_f32(
                    self.scratch.k_raw.ptr as *const f32,
                    layer.attn_k_conv.as_f32(),
                    k_dyn_ptr,
                    slot_kv.conv_history.ptr as *mut f32,
                    self.scratch.k_cur.ptr as *mut f32,
                    kv_dim,
                    n_head_kv,
                    m,
                    0,
                    self.cfg.shortconv_l_cache as i32,
                    self.stream.raw,
                );
            }

            self.linear_on_stream(
                &layer.wv,
                in2,
                share_kv_inp && layer.attn_k_conv_proj.is_none(),
                self.scratch.v_raw.ptr as *mut f32,
                kv_dim,
                n_embd,
                m,
            );
            let v_dyn_ptr = if let Some(vp) = &layer.attn_v_conv_proj {
                self.linear_on_stream(vp, in2, true, self.scratch.dyn_proj.ptr as *mut f32, dyn_dim, n_embd, m);
                self.scratch.dyn_proj.ptr as *const f32
            } else {
                std::ptr::null()
            };
            unsafe {
                launch_short_conv_f32(
                    self.scratch.v_raw.ptr as *const f32,
                    layer.attn_v_conv.as_f32(),
                    v_dyn_ptr,
                    slot_kv.conv_history.ptr as *mut f32,
                    self.scratch.v_cur.ptr as *mut f32,
                    kv_dim,
                    n_head_kv,
                    m,
                    kv_dim,
                    self.cfg.shortconv_l_cache as i32,
                    self.stream.raw,
                );
            }
        }

        let tau_ptr = if layer.is_swa {
            std::ptr::null()
        } else {
            self.scratch.tau.ptr as *const f32
        };
        let sink_w_ptr = if sink_fused {
            layer.attn_sink_proj.as_f32()
        } else {
            std::ptr::null()
        };
        let (kp_w_ptr, kp_x_ptr, k_conv_w_ptr, vp_w_ptr, vp_x_ptr, v_conv_w_ptr) = if dyn_conv_fused {
            (
                layer.attn_k_conv_proj.as_ref().unwrap().as_f32(),
                in1,
                layer.attn_k_conv.as_f32(),
                layer.attn_v_conv_proj.as_ref().unwrap().as_f32(),
                in2,
                layer.attn_v_conv.as_f32(),
            )
        } else {
            (
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        unsafe {
            launch_prep_qkv_r_and_kv_store(
                self.scratch.q_cur.ptr as *mut f32,
                layer.attn_q_clip.as_f32(),
                self.scratch.k_cur.ptr as *mut f32,
                self.scratch.v_cur.ptr as *mut f32,
                slot_kv.k_cache.ptr,
                slot_kv.v_cache.ptr,
                self.scratch.r_cur.ptr as *mut f32,
                layer.attn_rel_clip.as_f32(),
                tau_ptr,
                self.scratch.pos_ids.ptr as *const i32,
                sink_w_ptr,
                in0,
                self.scratch.sinks.ptr as *mut f32,
                kp_w_ptr,
                kp_x_ptr,
                k_conv_w_ptr,
                vp_w_ptr,
                vp_x_ptr,
                v_conv_w_ptr,
                slot_kv.conv_history.ptr as *mut f32,
                self.cfg.shortconv_l_cache as i32,
                n_embd,
                n_head,
                n_head_kv,
                head_dim,
                self.cfg.d_rel as i32,
                m,
                slot_kv.ring_cap as i32,
                kv_is_q8,
                self.cfg.qk_clip,
                self.cfg.rel_clip,
                eps,
                self.stream.raw,
            );

            if m == 1 {
                launch_flash_attn_rel_decode_m1(
                    self.scratch.q_cur.ptr as *const f32,
                    self.scratch.r_cur.ptr as *const f32,
                    layer.attn_rel_bias.as_f32(),
                    self.scratch.sinks.ptr as *const f32,
                    self.scratch.attn_gate_buf.ptr as *const f32,
                    slot_kv.k_cache.ptr,
                    slot_kv.v_cache.ptr,
                    self.scratch.pos_ids.ptr as *const i32,
                    self.scratch.split_out.ptr as *mut f32,
                    self.scratch.split_ml.ptr as *mut f32,
                    self.scratch.attn_out.ptr as *mut f32,
                    n_head,
                    n_head_kv,
                    head_dim,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    slot_kv.swa_window as i32,
                    slot_kv.ring_cap as i32,
                    kv_is_q8,
                    max_kv_len,
                    self.stream.raw,
                );
            } else {
                launch_flash_attn_rel(
                    self.scratch.q_cur.ptr as *const f32,
                    self.scratch.r_cur.ptr as *const f32,
                    layer.attn_rel_bias.as_f32(),
                    self.scratch.sinks.ptr as *const f32,
                    slot_kv.k_cache.ptr,
                    slot_kv.v_cache.ptr,
                    self.scratch.pos_ids.ptr as *const i32,
                    self.scratch.rel_table.ptr as *mut f32,
                    self.scratch.split_out.ptr as *mut f32,
                    self.scratch.split_ml.ptr as *mut f32,
                    self.scratch.attn_out.ptr as *mut f32,
                    n_head,
                    n_head_kv,
                    head_dim,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    slot_kv.swa_window as i32,
                    slot_kv.ring_cap as i32,
                    kv_is_q8,
                    max_kv_len,
                    m,
                    self.stream.raw,
                );

                launch_sigmoid_mul_f32(
                    self.scratch.attn_out.ptr as *mut f32,
                    self.scratch.attn_gate_buf.ptr as *const f32,
                    m * q_dim,
                    self.stream.raw,
                );
            }
        }

        let fuse_wo_planar = m == 1
            && layer.wo.q8_planar_q.is_some()
            && self.scratch.n_head_local == self.cfg.n_head;
        if fuse_wo_planar {
            let (wo_q, wo_d) = (
                layer.wo.q8_planar_q.as_ref().unwrap().ptr,
                layer.wo.q8_planar_d.as_ref().unwrap().ptr,
            );
            let add_res_ptr = if ip % self.cfg.attn_res_block_size == 0 {
                std::ptr::null()
            } else {
                self.scratch.inp_sa.ptr as *const f32
            };
            unsafe {
                launch_mv_q8_planar_m1(
                    wo_q, wo_d, n_embd, self.scratch.attn_out.ptr as *const f32, self.scratch.cur.ptr as *mut f32,
                    std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                    layer.attn_post_conv.as_f32(), 2 * kv_dim,
                    std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                    std::ptr::null(), 0,
                    std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                    std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut(),
                    add_res_ptr,
                    slot_kv.conv_history.ptr as *mut f32,
                    self.cfg.shortconv_l_cache as i32,
                    q_dim,
                    self.stream.raw,
                );
            }
        } else {
            self.linear_on_stream(
                &layer.wo,
                self.scratch.attn_out.ptr as *const f32,
                false,
                self.scratch.attn_proj.ptr as *mut f32,
                n_embd,
                q_dim,
                m,
            );
        }
        Ok(())
    }

    /// Post-`wo` (runs after TP2 all-reduce on `attn_proj`): `attn_post_conv` -> block push/residual -> FFN mix/norm/router/q8_1.
    pub fn enqueue_post_wo_to_router(&self, slot: usize, m: i32) -> Result<()> {
        let info = &self.cfg.logical_layers[slot];
        let ip = info.phys_layer;
        let pass = info.pass;
        let layer = self.weights.layer(ip);
        let slot_kv = self.kv_cache.slot(slot);
        let n_embd = self.cfg.n_embd as i32;
        let kv_dim = (self.scratch.n_head_kv_local * self.cfg.head_dim) as i32;
        let eps = self.cfg.rms_norm_eps;
        let hid_bytes = (m as usize) * self.cfg.n_embd * 4;

        let fuse_wo_planar = m == 1
            && layer.wo.q8_planar_q.is_some()
            && self.scratch.n_head_local == self.cfg.n_head;

        if ip % self.cfg.attn_res_block_size == 0 {
            if !fuse_wo_planar {
                unsafe {
                    launch_short_conv_f32(
                        self.scratch.attn_proj.ptr as *const f32,
                        layer.attn_post_conv.as_f32(),
                        std::ptr::null(),
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.cur.ptr as *mut f32,
                        n_embd,
                        0,
                        m,
                        2 * kv_dim,
                        self.cfg.shortconv_l_cache as i32,
                        self.stream.raw,
                    );
                }
            }
            let b_idx = ip / self.cfg.attn_res_block_size;
            self.scratch.blocks[b_idx].copy_from_device_async(
                &self.scratch.inp_sa,
                hid_bytes,
                &self.stream,
            )?;
        } else if !fuse_wo_planar {
            unsafe {
                launch_short_conv_add_f32(
                    self.scratch.attn_proj.ptr as *const f32,
                    layer.attn_post_conv.as_f32(),
                    std::ptr::null(),
                    self.scratch.inp_sa.ptr as *const f32,
                    slot_kv.conv_history.ptr as *mut f32,
                    self.scratch.cur.ptr as *mut f32,
                    n_embd,
                    0,
                    m,
                    2 * kv_dim,
                    self.cfg.shortconv_l_cache as i32,
                    self.stream.raw,
                );
            }
        }

        self.scratch
            .ffn_inp
            .copy_from_device_async(&self.scratch.cur, hid_bytes, &self.stream)?;

        let (ffn_ptrs_buf, ffn_n_states) = &self.mix_tables.ffn_mix[slot];
        if pass > 0 {
            unsafe {
                launch_attn_res_mix_f32(
                    ffn_ptrs_buf.ptr as *const *const f32,
                    *ffn_n_states as i32,
                    layer.ffn_res_query.as_f32(),
                    self.scratch.cur.ptr as *mut f32,
                    n_embd,
                    1,
                    m,
                    eps,
                    self.stream.raw,
                );
            }
            let lc = layer.loop_ctrl.as_ref().unwrap();
            let msg_job_off = if self.weights.loop_msg_queries_all.is_some() {
                (ip - self.cfg.loop_start) * 4 * hid_bytes
            } else {
                0
            };
            let msg_ptr = unsafe {
                (self.scratch.loop_messages.ptr as *const u8).add(msg_job_off + 3 * hid_bytes)
                    as *const f32
            };
            let all_f32 = matches!(lc.down[3].ggml_type, GgmlType::F32)
                && matches!(lc.gain[3].ggml_type, GgmlType::F32)
                && matches!(lc.forget[3].ggml_type, GgmlType::F32);
            if all_f32 {
                unsafe {
                    launch_loop_gates_fused_f32(
                        self.scratch.cur.ptr as *const f32, std::ptr::null(), std::ptr::null(),
                        msg_ptr, std::ptr::null(), std::ptr::null(),
                        lc.down[3].as_f32(), std::ptr::null(), std::ptr::null(),
                        lc.forget[3].as_f32(), std::ptr::null(), std::ptr::null(),
                        lc.gain[3].as_f32(), std::ptr::null(), std::ptr::null(),
                        self.scratch.cur.ptr as *mut f32, std::ptr::null_mut(), std::ptr::null_mut(),
                        self.scratch.ctrl_gate.ptr as *mut f32,
                        1,
                        m,
                        n_embd,
                        self.cfg.loop_gate_rank as i32,
                        eps,
                        self.stream.raw,
                    );
                }
            } else {
                self.run_loop_control(self.scratch.cur.ptr as *mut f32, msg_ptr, lc, 3, m);
            }
            unsafe {
                launch_rmsnorm_f32(
                    self.scratch.cur.ptr as *const f32,
                    layer.ffn_norm.as_f32(),
                    self.scratch.tmp_hidden.ptr as *mut f32,
                    n_embd,
                    m,
                    eps,
                    self.stream.raw,
                );
            }
        } else {
            unsafe {
                launch_attn_res_mix_rms_f32(
                    ffn_ptrs_buf.ptr as *const *const f32,
                    *ffn_n_states as i32,
                    layer.ffn_res_query.as_f32(),
                    layer.ffn_norm.as_f32(),
                    std::ptr::null(),
                    std::ptr::null(),
                    self.scratch.tmp_hidden.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    n_embd,
                    1,
                    m,
                    eps,
                    eps,
                    self.stream.raw,
                );
            }
        }

        self.linear_on_stream(
            &layer.ffn_gate_inp,
            self.scratch.tmp_hidden.ptr as *const f32,
            false,
            self.scratch.router_logits.ptr as *mut f32,
            self.cfg.n_expert as i32,
            n_embd,
            m,
        );
        let bias_ptr = layer
            .exp_probs_b
            .as_ref()
            .map(|t| t.as_f32())
            .unwrap_or(std::ptr::null());
        unsafe {
            let use_mma_limbs = m == 1
                && layer.gate_up_type == GgmlType::Nvfp4
                && layer.down_type == GgmlType::Nvfp4
                && self.cfg.n_embd % 64 == 0
                && self.scratch.n_ff_exp_local % 64 == 0;
            if use_mma_limbs && self.cfg.n_expert <= 128 && self.cfg.n_expert_used <= 16 {
                launch_router_topk_and_f16_limbs_m1(
                    self.scratch.router_logits.ptr as *const f32,
                    bias_ptr,
                    self.scratch.expert_ids.ptr as *mut i32,
                    self.scratch.expert_weights.ptr as *mut f32,
                    self.scratch.tmp_hidden.ptr as *const f32,
                    self.scratch.moe_inter_f32.ptr,
                    self.cfg.n_expert as i32,
                    self.cfg.n_expert_used as i32,
                    n_embd,
                    self.cfg.expert_weights_scale,
                    self.stream.raw,
                );
            } else {
                launch_router_topk_sigmoid_f32(
                    self.scratch.router_logits.ptr as *const f32,
                    bias_ptr,
                    self.scratch.expert_ids.ptr as *mut i32,
                    self.scratch.expert_weights.ptr as *mut f32,
                    self.cfg.n_expert as i32,
                    self.cfg.n_expert_used as i32,
                    m,
                    self.cfg.expert_weights_scale,
                    self.stream.raw,
                );
                launch_quantize_q8_1(
                    self.scratch.tmp_hidden.ptr as *const f32,
                    self.scratch.act_q8_1.ptr,
                    m,
                    n_embd,
                    self.stream.raw,
                );
            }
        }

        Ok(())
    }

    pub fn enqueue_moe_local(&self, ip: usize, gu_ptrs: u64, dn_ptrs: u64, m: i32) {
        let layer = self.weights.layer(ip);
        let gu_sc_ptr = layer
            .gate_up_scale
            .as_ref()
            .map(|t| t.as_f32())
            .unwrap_or(std::ptr::null());
        let dn_sc_ptr = layer
            .down_scale
            .as_ref()
            .map(|t| t.as_f32())
            .unwrap_or(std::ptr::null());
        unsafe {
            launch_moe_grouped_f32(
                gu_ptrs as *const u64,
                dn_ptrs as *const u64,
                gu_sc_ptr,
                dn_sc_ptr,
                layer.gate_up_row_bytes,
                layer.down_row_bytes,
                layer.gate_up_type as i32,
                layer.down_type as i32,
                self.scratch.act_q8_1.ptr,
                self.scratch.tmp_hidden.ptr as *const f32,
                self.scratch.expert_ids.ptr as *const i32,
                self.scratch.expert_weights.ptr as *const f32,
                self.scratch.moe_inter_f32.ptr as *mut f32,
                self.scratch.moe_inter_q8_1.ptr,
                self.scratch.moe_slot_out.ptr as *mut f32,
                self.scratch.moe_out.ptr as *mut f32,
                self.cfg.n_embd as i32,
                self.scratch.n_ff_exp_local as i32,
                self.cfg.n_expert_used as i32,
                m,
                self.cfg.situ_beta,
                self.cfg.situ_linear_beta,
                self.stream.raw,
            );
        }
    }

    pub fn enqueue_post_moe(&self, slot: usize, m: i32) {
        let info = &self.cfg.logical_layers[slot];
        let ip = info.phys_layer;
        let layer = self.weights.layer(ip);
        let slot_kv = self.kv_cache.slot(slot);
        let n_embd = self.cfg.n_embd as i32;
        let kv_dim = (self.scratch.n_head_kv_local * self.cfg.head_dim) as i32;

        unsafe {
            launch_short_conv_add_f32(
                self.scratch.moe_out.ptr as *const f32,
                layer.ffn_post_conv.as_f32(),
                std::ptr::null(),
                self.scratch.ffn_inp.ptr as *const f32,
                slot_kv.conv_history.ptr as *mut f32,
                self.scratch.cur.ptr as *mut f32,
                n_embd,
                0,
                m,
                2 * kv_dim + n_embd,
                self.cfg.shortconv_l_cache as i32,
                self.stream.raw,
            );
        }
    }

    pub fn enqueue_moe_and_post_m1_resident(&self, slot: usize, gu_ptrs: u64, dn_ptrs: u64) {
        let info = &self.cfg.logical_layers[slot];
        let ip = info.phys_layer;
        let layer = self.weights.layer(ip);
        let slot_kv = self.kv_cache.slot(slot);
        let n_embd = self.cfg.n_embd as i32;
        let kv_dim = (self.scratch.n_head_kv_local * self.cfg.head_dim) as i32;
        let gu_sc_ptr = layer
            .gate_up_scale
            .as_ref()
            .map(|t| t.as_f32())
            .unwrap_or(std::ptr::null());
        let dn_sc_ptr = layer
            .down_scale
            .as_ref()
            .map(|t| t.as_f32())
            .unwrap_or(std::ptr::null());
        unsafe {
            launch_moe_grouped_f32(
                gu_ptrs as *const u64,
                dn_ptrs as *const u64,
                gu_sc_ptr,
                dn_sc_ptr,
                layer.gate_up_row_bytes,
                layer.down_row_bytes,
                layer.gate_up_type as i32,
                layer.down_type as i32,
                self.scratch.act_q8_1.ptr,
                self.scratch.tmp_hidden.ptr as *const f32,
                self.scratch.expert_ids.ptr as *const i32,
                self.scratch.expert_weights.ptr as *const f32,
                self.scratch.moe_inter_f32.ptr as *mut f32,
                self.scratch.moe_inter_q8_1.ptr,
                self.scratch.moe_slot_out.ptr as *mut f32,
                std::ptr::null_mut(),
                n_embd,
                self.scratch.n_ff_exp_local as i32,
                self.cfg.n_expert_used as i32,
                1,
                self.cfg.situ_beta,
                self.cfg.situ_linear_beta,
                self.stream.raw,
            );
            launch_moe_reduce_conv_add_m1_f32(
                self.scratch.moe_slot_out.ptr as *const f32,
                layer.ffn_post_conv.as_f32(),
                slot_kv.conv_history.ptr as *mut f32,
                self.scratch.ffn_inp.ptr as *const f32,
                self.scratch.cur.ptr as *mut f32,
                n_embd,
                self.cfg.n_expert_used as i32,
                2 * kv_dim + n_embd,
                self.cfg.shortconv_l_cache as i32,
                self.stream.raw,
            );
        }
    }

    pub fn enqueue_output_head(&self, m: i32, n_vocab_out: i32, with_val_id: bool) {
        let m_usize = m as usize;
        let n_embd = self.cfg.n_embd as i32;
        let eps = self.cfg.rms_norm_eps;
        let (out_ptrs, out_n_states) = &self.mix_tables.output_mix;
        let out_w = self.weights.output.as_ref().expect("output required");
        let out_norm = self
            .weights
            .output_norm
            .as_ref()
            .expect("output_norm required");
        let out_res_q = self
            .weights
            .output_res_query
            .as_ref()
            .expect("output_res_query required");

        unsafe {
            launch_attn_res_mix_rms_f32(
                out_ptrs.ptr as *const *const f32,
                *out_n_states as i32,
                out_res_q.as_f32(),
                out_norm.as_f32(),
                std::ptr::null(),
                std::ptr::null(),
                self.scratch.tmp_hidden.ptr as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                n_embd,
                1,
                m,
                eps,
                eps,
                self.stream.raw,
            );
            let last_row =
                (self.scratch.tmp_hidden.ptr as *const f32).add((m_usize - 1) * self.cfg.n_embd);
            self.linear_on_stream(
                out_w,
                last_row,
                false,
                self.scratch.logits.ptr as *mut f32,
                n_vocab_out,
                n_embd,
                1,
            );
            if self.cfg.final_logit_softcapping > 0.0 {
                launch_softcap_tanh_f32(
                    self.scratch.logits.ptr as *mut f32,
                    n_vocab_out,
                    self.cfg.final_logit_softcapping,
                    self.stream.raw,
                );
            }
            if with_val_id {
                launch_argmax_val_id_f32(
                    self.scratch.logits.ptr as *const f32,
                    n_vocab_out,
                    self.scratch.argmax_val.ptr as *mut f32,
                    self.scratch.argmax_id.ptr as *mut i32,
                    self.stream.raw,
                );
            } else {
                launch_argmax_f32(
                    self.scratch.logits.ptr as *const f32,
                    n_vocab_out,
                    self.scratch.argmax_id.ptr as *mut i32,
                    self.stream.raw,
                );
            }
        }
    }

    pub fn init_conv_hist_ptrs(&self) -> Result<()> {
        let ptrs: Vec<u64> = (0..self.cfg.n_layer_logical)
            .map(|s| self.kv_cache.slot(s).conv_history.as_u64())
            .collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, ptrs.len() * 8)
        };
        self.scratch.conv_hist_ptrs_dev.copy_from_host(bytes)
    }

    pub fn enqueue_conv_rollback(&self, accept_n: i32) {
        let n_slots = self.cfg.n_layer_logical as i32;
        let kv_dim = (self.scratch.n_head_kv_local * self.cfg.head_dim) as i32;
        let n_sconv_channels = 2 * kv_dim + 2 * (self.cfg.n_embd as i32);
        unsafe {
            launch_conv_rollback_all(
                self.scratch.conv_hist_ptrs_dev.ptr as *const *mut f32,
                self.scratch.rec_old_hist.ptr as *const f32,
                self.scratch.rec_x.ptr as *const f32,
                n_slots,
                n_sconv_channels,
                accept_n,
                self.stream.raw,
            );
        }
    }

    #[inline]
    pub fn tap_stride_elems(&self) -> usize {
        self.scratch.max_batch * self.cfg.n_embd
    }

    /// Record final output-norm hidden state into `taps[final_tap]` for `m` rows (used in prefill context ingestion).
    pub fn record_final_tap(&self, m: i32) -> Result<()> {
        let hid_bytes = (m as usize) * self.cfg.n_embd * 4;
        let final_tap = if self.cfg.is_looped { 3usize } else { 7usize };
        let tap_off = final_tap * self.tap_stride_elems() * 4;
        unsafe {
            cuda_check_copy_d2d(
                (self.scratch.taps.ptr as *mut u8).add(tap_off) as *mut c_void,
                self.scratch.tmp_hidden.ptr,
                hid_bytes,
                self.stream.raw,
            )
        }
    }

    /// Record hidden state `cur` at the start of a visit if it corresponds to a drafter tap.
    pub fn maybe_record_visit_tap(&self, vis: usize, m: i32) -> Result<()> {
        let tap_opt = if self.cfg.is_looped {
            match vis {
                5 => Some(0usize),
                23 => Some(1usize),
                41 => Some(2usize),
                _ => None,
            }
        } else {
            match vis {
                5 => Some(0),
                11 => Some(1),
                17 => Some(2),
                23 => Some(3),
                29 => Some(4),
                35 => Some(5),
                41 => Some(6),
                _ => None,
            }
        };
        if let Some(ti) = tap_opt {
            let hid_bytes = (m as usize) * self.cfg.n_embd * 4;
            let tap_off = ti * self.tap_stride_elems() * 4;
            unsafe {
                cuda_check_copy_d2d(
                    (self.scratch.taps.ptr as *mut u8).add(tap_off) as *mut c_void,
                    self.scratch.cur.ptr,
                    hid_bytes,
                    self.stream.raw,
                )?;
            }
        }
        Ok(())
    }

    /// Run an entire forward step/verify window of `t` tokens (`1 <= t <= 8`) using fused sm_86 Tensor Core kernels.
    pub fn enqueue_forward_window<F>(
        &self,
        t: i32,
        all_resident: bool,
        capture_logits: bool,
        record_taps: bool,
        record_conv: bool,
        gu_ptrs_for_layer: F,
    ) -> Result<()>
    where
        F: FnMut(usize) -> (u64, u64),
    {
        self.enqueue_forward_window_impl(t, all_resident, capture_logits, record_taps, record_conv, false, gu_ptrs_for_layer)
    }

    pub fn enqueue_forward_window_profiled<F>(
        &self,
        t: i32,
        all_resident: bool,
        capture_logits: bool,
        record_taps: bool,
        record_conv: bool,
        gu_ptrs_for_layer: F,
    ) -> Result<()>
    where
        F: FnMut(usize) -> (u64, u64),
    {
        self.enqueue_forward_window_impl(t, all_resident, capture_logits, record_taps, record_conv, true, gu_ptrs_for_layer)
    }

    fn enqueue_forward_window_impl<F>(
        &self,
        t: i32,
        all_resident: bool,
        capture_logits: bool,
        record_taps: bool,
        record_conv: bool,
        stage_prof: bool,
        mut gu_ptrs_for_layer: F,
    ) -> Result<()>
    where
        F: FnMut(usize) -> (u64, u64),
    {
        let prof_events: Option<(CudaEvent, CudaEvent, Vec<[CudaEvent; 8]>, CudaEvent)> = if stage_prof {
            let e0 = CudaEvent::new_timed(self.device)?;
            let e_emb = CudaEvent::new_timed(self.device)?;
            let n_vis = self.cfg.execution_order().len();
            let mut lvs = Vec::with_capacity(n_vis);
            for _ in 0..n_vis {
                lvs.push([
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                    CudaEvent::new_timed(self.device)?,
                ]);
            }
            let e_head = CudaEvent::new_timed(self.device)?;
            e0.record(&self.stream)?;
            Some((e0, e_emb, lvs, e_head))
        } else {
            None
        };
        let m = t;
        let m_usize = m as usize;
        let n_embd = self.cfg.n_embd as i32;
        let n_head = self.scratch.n_head_local as i32;
        let n_head_kv = self.scratch.n_head_kv_local as i32;
        let head_dim = self.cfg.head_dim as i32;
        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let rel_dim = n_head * (self.cfg.d_rel as i32);
        let dyn_dim = n_head_kv * (self.cfg.shortconv_l_cache as i32);
        let n_sconv_channels = 2 * kv_dim + 2 * n_embd;
        let n_conv = self.cfg.shortconv_l_cache as i32;
        let eps = self.cfg.rms_norm_eps;
        let hid_bytes = m_usize * self.cfg.n_embd * 4;

        let kv_mode = match self.kv_cache.kv_mode {
            crate::kv_cache::KvQuantMode::F16 => 0,
            crate::kv_cache::KvQuantMode::Q8_0 => 1,
            crate::kv_cache::KvQuantMode::Fp8 => 2,
        };

        let rec_old_ptr = if record_conv {
            self.scratch.rec_old_hist.ptr as *mut f32
        } else {
            std::ptr::null_mut()
        };
        let rec_x_ptr = if record_conv {
            self.scratch.rec_x.ptr as *mut f32
        } else {
            std::ptr::null_mut()
        };

        let tok_embd = self.weights.tok_embd.as_ref().unwrap();
        let tok_norm = self.weights.tok_embd_norm.as_ref().unwrap();
        unsafe {
            launch_compute_tau_f32(
                self.scratch.pos_ids.ptr as *const i32,
                self.scratch.tau.ptr as *mut f32,
                m,
                self.cfg.log_scale_floor,
                self.cfg.log_scale_alpha,
                self.stream.raw,
            );
            launch_embed_lookup_rmsnorm(
                tok_embd.buf.ptr,
                tok_embd.ggml_type as i32,
                self.scratch.token_ids.ptr as *const i32,
                tok_norm.as_f32(),
                self.scratch.cur.ptr as *mut f32,
                n_embd,
                m,
                eps,
                self.stream.raw,
            );
        }

        if let Some((_, ref e_emb, _, _)) = prof_events {
            e_emb.record(&self.stream)?;
        }

        let exec_order = self.cfg.execution_order();
        for (vis, &slot) in exec_order.iter().enumerate() {
            let info = &self.cfg.logical_layers[slot];
            let ip = info.phys_layer;
            let pass = info.pass;
            let layer = self.weights.layer(ip);
            let slot_kv = self.kv_cache.slot(slot);

            if self.cfg.is_looped && ip == self.cfg.loop_start && pass == 0 {
                self.scratch
                    .prelude
                    .copy_from_device_async(&self.scratch.cur, hid_bytes, &self.stream)?;
            }
            if self.cfg.is_looped && ip == self.cfg.loop_start && pass > 0 {
                self.scratch
                    .blocks[4]
                    .copy_from_device_async(&self.scratch.blocks[2], hid_bytes, &self.stream)?;
                self.scratch
                    .bank_out
                    .copy_from_device_async(&self.scratch.cur, hid_bytes, &self.stream)?;
                self.scratch
                    .cur
                    .copy_from_device_async(&self.scratch.prelude, hid_bytes, &self.stream)?;
                if let Some(ref all_mq) = self.weights.loop_msg_queries_all {
                    let (ptrs_buf, n_states) = self.mix_tables.loop_msg_mix[slot].as_ref().unwrap();
                    let n_core = (self.cfg.loop_end - self.cfg.loop_start) as i32;
                    unsafe {
                        launch_attn_res_mix_jobs_f32(
                            ptrs_buf.ptr as *const *const f32,
                            *n_states as i32,
                            all_mq.ptr as *const f32,
                            self.scratch.loop_messages.ptr as *mut f32,
                            n_embd,
                            4,
                            n_core,
                            m,
                            eps,
                            self.stream.raw,
                        );
                    }
                }
            }
            let batched_loop_msg = self.weights.loop_msg_queries_all.is_some();
            if pass > 0 && !batched_loop_msg {
                let lc = layer.loop_ctrl.as_ref().unwrap();
                let (ptrs_buf, n_states) = self.mix_tables.loop_msg_mix[slot].as_ref().unwrap();
                unsafe {
                    launch_attn_res_mix_f32(
                        ptrs_buf.ptr as *const *const f32,
                        *n_states as i32,
                        lc.msg_query.as_f32(),
                        self.scratch.loop_messages.ptr as *mut f32,
                        n_embd,
                        4,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            }

            if record_taps {
                self.maybe_record_visit_tap(vis, m)?;
            }

            let all_planar_qkvrg = layer.wq.q8_planar_q.is_some()
                && layer.wk.q8_planar_q.is_some()
                && layer.wv.q8_planar_q.is_some()
                && layer.attn_rel.q8_planar_q.is_some()
                && layer.attn_gate.q8_planar_q.is_some();

            let share_kv_inp = layer.is_swa && pass == 0;
            if pass > 0 {
                if let (Some(res_q), Some((ptrs_buf, n_states))) =
                    (&layer.attn_res_query, &self.mix_tables.attn_mix[slot])
                {
                    let n_q = if layer.is_swa { 1 } else { 3 };
                    unsafe {
                        launch_attn_res_mix_f32(
                            ptrs_buf.ptr as *const *const f32,
                            *n_states as i32,
                            res_q.as_f32(),
                            self.scratch.mix_out.ptr as *mut f32,
                            n_embd,
                            n_q,
                            m,
                            eps,
                            self.stream.raw,
                        );
                    }
                }
                let lc = layer.loop_ctrl.as_ref().unwrap();
                let base_x = if layer.attn_res_query.is_some() {
                    self.scratch.mix_out.ptr as *const f32
                } else {
                    self.scratch.cur.ptr as *const f32
                };
                let x0 = base_x;
                let x1 = if !layer.is_swa && layer.attn_res_query.is_some() {
                    unsafe { (base_x as *const u8).add(hid_bytes) as *const f32 }
                } else {
                    base_x
                };
                let x2 = if !layer.is_swa && layer.attn_res_query.is_some() {
                    unsafe { (base_x as *const u8).add(2 * hid_bytes) as *const f32 }
                } else {
                    base_x
                };
                let msg_job_off = if batched_loop_msg {
                    (ip - self.cfg.loop_start) * 4 * hid_bytes
                } else {
                    0
                };
                let m0 = unsafe {
                    (self.scratch.loop_messages.ptr as *const u8).add(msg_job_off) as *const f32
                };
                let m1 = unsafe { (m0 as *const u8).add(hid_bytes) as *const f32 };
                let m2 = unsafe { (m0 as *const u8).add(2 * hid_bytes) as *const f32 };
                let all_f32 = matches!(lc.down[0].ggml_type, GgmlType::F32)
                    && matches!(lc.gain[0].ggml_type, GgmlType::F32)
                    && matches!(lc.forget[0].ggml_type, GgmlType::F32);
                if all_f32 {
                    unsafe {
                        launch_loop_gates_fast_f32(
                            x0, x1, x2,
                            m0, m1, m2,
                            lc.down[0].as_f32(), lc.down[1].as_f32(), lc.down[2].as_f32(),
                            lc.forget[0].as_f32(), lc.forget[1].as_f32(), lc.forget[2].as_f32(),
                            lc.gain[0].as_f32(), lc.gain[1].as_f32(), lc.gain[2].as_f32(),
                            self.scratch.inputs_qkv[0].ptr as *mut f32,
                            self.scratch.inputs_qkv[1].ptr as *mut f32,
                            self.scratch.inputs_qkv[2].ptr as *mut f32,
                            self.scratch.ctrl_gate.ptr as *mut f32,
                            3,
                            m,
                            n_embd,
                            self.cfg.loop_gate_rank as i32,
                            eps,
                            self.stream.raw,
                        );
                    }
                } else {
                    let xs = [x0, x1, x2];
                    for j in 0..3 {
                        unsafe {
                            cuda_check_copy_d2d(
                                self.scratch.inputs_qkv[j].ptr,
                                xs[j] as *const c_void,
                                hid_bytes,
                                self.stream.raw,
                            )?;
                            let msg_ptr = (self.scratch.loop_messages.ptr as *const u8)
                                .add(msg_job_off + j * hid_bytes)
                                as *const f32;
                            self.run_loop_control(
                                self.scratch.inputs_qkv[j].ptr as *mut f32,
                                msg_ptr,
                                lc,
                                j,
                                m,
                            );
                        }
                    }
                }
                let (nw0, nw1, nw2) = if layer.is_swa {
                    let w = layer.attn_norm.as_ref().unwrap().as_f32();
                    (w, w, w)
                } else {
                    (
                        layer.attn_q_inp_norm.as_ref().unwrap().as_f32(),
                        layer.attn_k_inp_norm.as_ref().unwrap().as_f32(),
                        layer.attn_v_inp_norm.as_ref().unwrap().as_f32(),
                    )
                };
                unsafe {
                    launch_rmsnorm_f16_multi(
                        self.scratch.inputs_qkv[0].ptr as *const f32,
                        self.scratch.inputs_qkv[1].ptr as *const f32,
                        self.scratch.inputs_qkv[2].ptr as *const f32,
                        nw0, nw1, nw2,
                        self.scratch.inputs_norm[0].ptr as *mut f32,
                        self.scratch.inputs_norm[1].ptr as *mut f32,
                        self.scratch.inputs_norm[2].ptr as *mut f32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        3,
                        n_embd,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            } else if let (Some(res_q), Some((ptrs_buf, n_states))) =
                (&layer.attn_res_query, &self.mix_tables.attn_mix[slot])
            {
                if layer.is_swa {
                    let nw = layer.attn_norm.as_ref().unwrap().as_f32();
                    unsafe {
                        launch_attn_res_mix_rms_f16(
                            ptrs_buf.ptr as *const *const f32,
                            *n_states as i32,
                            res_q.as_f32(),
                            nw,
                            std::ptr::null(),
                            std::ptr::null(),
                            self.scratch.inputs_norm[0].ptr as *mut f32,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            n_embd,
                            1,
                            m,
                            eps,
                            eps,
                            self.stream.raw,
                        );
                    }
                } else {
                    unsafe {
                        launch_attn_res_mix_rms_f16(
                            ptrs_buf.ptr as *const *const f32,
                            *n_states as i32,
                            res_q.as_f32(),
                            layer.attn_q_inp_norm.as_ref().unwrap().as_f32(),
                            layer.attn_k_inp_norm.as_ref().unwrap().as_f32(),
                            layer.attn_v_inp_norm.as_ref().unwrap().as_f32(),
                            self.scratch.inputs_norm[0].ptr as *mut f32,
                            self.scratch.inputs_norm[1].ptr as *mut f32,
                            self.scratch.inputs_norm[2].ptr as *mut f32,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            n_embd,
                            3,
                            m,
                            eps,
                            eps,
                            self.stream.raw,
                        );
                    }
                }
            } else if layer.is_swa {
                let norm_w = layer.attn_norm.as_ref().unwrap().as_f32();
                unsafe {
                    launch_rmsnorm_f16_multi(
                        self.scratch.cur.ptr as *const f32,
                        std::ptr::null(),
                        std::ptr::null(),
                        norm_w,
                        std::ptr::null(),
                        std::ptr::null(),
                        self.scratch.inputs_norm[0].ptr as *mut f32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        1,
                        n_embd,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            } else {
                unsafe {
                    launch_rmsnorm_f16_multi(
                        self.scratch.cur.ptr as *const f32,
                        self.scratch.cur.ptr as *const f32,
                        self.scratch.cur.ptr as *const f32,
                        layer.attn_q_inp_norm.as_ref().unwrap().as_f32(),
                        layer.attn_k_inp_norm.as_ref().unwrap().as_f32(),
                        layer.attn_v_inp_norm.as_ref().unwrap().as_f32(),
                        self.scratch.inputs_norm[0].ptr as *mut f32,
                        self.scratch.inputs_norm[1].ptr as *mut f32,
                        self.scratch.inputs_norm[2].ptr as *mut f32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        3,
                        n_embd,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            }

            let in0_32 = self.scratch.inputs_norm[0].ptr as *const f32;
            let in1_32 = if share_kv_inp {
                in0_32
            } else {
                self.scratch.inputs_norm[1].ptr as *const f32
            };
            let in2_32 = if share_kv_inp {
                in0_32
            } else {
                self.scratch.inputs_norm[2].ptr as *const f32
            };

            let sink_w_f16 = layer
                .attn_sink_proj
                .f16_buf
                .as_ref()
                .map(|b| b.ptr as *const u16)
                .unwrap_or(std::ptr::null());
            let sink_w_f32 = if sink_w_f16.is_null() && matches!(layer.attn_sink_proj.ggml_type, GgmlType::F32) {
                layer.attn_sink_proj.as_f32()
            } else {
                std::ptr::null()
            };
            let sink_fused = !sink_w_f16.is_null() || !sink_w_f32.is_null();

            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][0].record(&self.stream)?;
            }

            let mut kv_conv_fused = false;
            if all_planar_qkvrg && m == 1 && !record_conv && layer.is_swa {
                unsafe {
                    launch_mv_q8_planar_m1(
                        layer.wq.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wq.q8_planar_d.as_ref().unwrap().ptr,
                        q_dim,
                        in0_32,
                        self.scratch.q_cur.ptr as *mut f32,
                        layer.wk.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wk.q8_planar_d.as_ref().unwrap().ptr,
                        kv_dim,
                        in1_32,
                        self.scratch.k_cur.ptr as *mut f32,
                        layer.attn_k_conv.as_f32(),
                        0,
                        layer.wv.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wv.q8_planar_d.as_ref().unwrap().ptr,
                        kv_dim,
                        in2_32,
                        self.scratch.v_cur.ptr as *mut f32,
                        layer.attn_v_conv.as_f32(),
                        kv_dim,
                        layer.attn_rel.q8_planar_q.as_ref().unwrap().ptr,
                        layer.attn_rel.q8_planar_d.as_ref().unwrap().ptr,
                        rel_dim,
                        in0_32,
                        self.scratch.r_cur.ptr as *mut f32,
                        layer.attn_gate.q8_planar_q.as_ref().unwrap().ptr,
                        layer.attn_gate.q8_planar_d.as_ref().unwrap().ptr,
                        q_dim,
                        in0_32,
                        self.scratch.attn_gate_buf.ptr as *mut f32,
                        std::ptr::null(),
                        slot_kv.conv_history.ptr as *mut f32,
                        n_conv,
                        n_embd,
                        self.stream.raw,
                    );
                }
                kv_conv_fused = true;
            } else if all_planar_qkvrg && m == 1 {
                unsafe {
                    launch_mv_q8_planar_m1(
                        layer.wq.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wq.q8_planar_d.as_ref().unwrap().ptr,
                        q_dim,
                        in0_32,
                        self.scratch.q_cur.ptr as *mut f32,
                        layer.wk.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wk.q8_planar_d.as_ref().unwrap().ptr,
                        kv_dim,
                        in1_32,
                        self.scratch.k_raw.ptr as *mut f32,
                        std::ptr::null(),
                        0,
                        layer.wv.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wv.q8_planar_d.as_ref().unwrap().ptr,
                        kv_dim,
                        in2_32,
                        self.scratch.v_raw.ptr as *mut f32,
                        std::ptr::null(),
                        0,
                        layer.attn_rel.q8_planar_q.as_ref().unwrap().ptr,
                        layer.attn_rel.q8_planar_d.as_ref().unwrap().ptr,
                        rel_dim,
                        in0_32,
                        self.scratch.r_cur.ptr as *mut f32,
                        layer.attn_gate.q8_planar_q.as_ref().unwrap().ptr,
                        layer.attn_gate.q8_planar_d.as_ref().unwrap().ptr,
                        q_dim,
                        in0_32,
                        self.scratch.attn_gate_buf.ptr as *mut f32,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        0,
                        n_embd,
                        self.stream.raw,
                    );
                }
            } else if all_planar_qkvrg {
                unsafe {
                    launch_mv_q8p_colsx_qkvrg(
                        layer.wq.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                        layer.wq.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                        q_dim,
                        in0_32,
                        self.scratch.q_cur.ptr as *mut f32,
                        layer.wk.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                        layer.wk.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                        kv_dim,
                        in1_32,
                        self.scratch.k_raw.ptr as *mut f32,
                        layer.wv.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                        layer.wv.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                        kv_dim,
                        in2_32,
                        self.scratch.v_raw.ptr as *mut f32,
                        layer.attn_rel.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                        layer.attn_rel.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                        rel_dim,
                        in0_32,
                        self.scratch.r_cur.ptr as *mut f32,
                        layer.attn_gate.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                        layer.attn_gate.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                        q_dim,
                        in0_32,
                        self.scratch.attn_gate_buf.ptr as *mut f32,
                        n_embd,
                        m,
                        self.stream.raw,
                    );
                }
            } else {
                self.linear_on_stream(&layer.wq, in0_32, false, self.scratch.q_cur.ptr as *mut f32, q_dim, n_embd, m);
                self.linear_on_stream(&layer.attn_rel, in0_32, false, self.scratch.r_cur.ptr as *mut f32, rel_dim, n_embd, m);
                self.linear_on_stream(&layer.attn_gate, in0_32, false, self.scratch.attn_gate_buf.ptr as *mut f32, q_dim, n_embd, m);
                self.linear_on_stream(&layer.wk, in1_32, false, self.scratch.k_raw.ptr as *mut f32, kv_dim, n_embd, m);
                self.linear_on_stream(&layer.wv, in2_32, false, self.scratch.v_raw.ptr as *mut f32, kv_dim, n_embd, m);
            }

            if !sink_fused {
                self.linear_on_stream(
                    &layer.attn_sink_proj,
                    in0_32,
                    false,
                    self.scratch.sinks.ptr as *mut f32,
                    n_head,
                    n_embd,
                    m,
                );
            }

            if !kv_conv_fused {
                let (k_dyn_ptr, v_dyn_ptr) = match (&layer.attn_k_conv_proj, &layer.attn_v_conv_proj) {
                    (Some(kp), Some(vp)) => {
                        let d0 = self.scratch.dyn_proj.ptr as *mut f32;
                        let d1 = unsafe { d0.add(m_usize * (dyn_dim as usize)) };
                        match (&kp.f16_buf, &vp.f16_buf) {
                            (Some(kf16), Some(vf16)) => unsafe {
                                launch_mv_f16_pair_batch(
                                    kf16.ptr as *const u16,
                                    in1_32,
                                    d0,
                                    vf16.ptr as *const u16,
                                    in2_32,
                                    d1,
                                    dyn_dim,
                                    n_embd,
                                    m,
                                    self.stream.raw,
                                );
                            },
                            _ => {
                                self.linear_on_stream(kp, in1_32, false, d0, dyn_dim, n_embd, m);
                                self.linear_on_stream(vp, in2_32, false, d1, dyn_dim, n_embd, m);
                            }
                        }
                        (d0 as *const f32, d1 as *const f32)
                    }
                    (Some(kp), None) => {
                        let d0 = self.scratch.dyn_proj.ptr as *mut f32;
                        self.linear_on_stream(kp, in1_32, false, d0, dyn_dim, n_embd, m);
                        (d0 as *const f32, std::ptr::null())
                    }
                    (None, Some(vp)) => {
                        let d1 = self.scratch.dyn_proj.ptr as *mut f32;
                        self.linear_on_stream(vp, in2_32, false, d1, dyn_dim, n_embd, m);
                        (std::ptr::null(), d1 as *const f32)
                    }
                    (None, None) => (std::ptr::null(), std::ptr::null()),
                };
                unsafe {
                    launch_short_conv_kv_pair_rec_f32(
                        self.scratch.k_raw.ptr as *const f32,
                        layer.attn_k_conv.as_f32(),
                        k_dyn_ptr,
                        self.scratch.k_cur.ptr as *mut f32,
                        self.scratch.v_raw.ptr as *const f32,
                        layer.attn_v_conv.as_f32(),
                        v_dyn_ptr,
                        self.scratch.v_cur.ptr as *mut f32,
                        slot_kv.conv_history.ptr as *mut f32,
                        rec_old_ptr,
                        rec_x_ptr,
                        n_sconv_channels,
                        slot as i32,
                        kv_dim,
                        n_head_kv,
                        m,
                        n_conv,
                        self.stream.raw,
                    );
                }
            }

            let tau_ptr = if layer.is_swa {
                std::ptr::null()
            } else {
                self.scratch.tau.ptr as *const f32
            };
            let wo_planar = layer.wo.q8_planar_q.is_some();
            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][1].record(&self.stream)?;
            }
            unsafe {
                launch_prep_qkv_r_block(
                    self.scratch.q_cur.ptr as *mut f32,
                    self.scratch.q16.ptr as *mut u16,
                    layer.attn_q_clip.as_f32(),
                    self.scratch.k_cur.ptr as *mut f32,
                    slot_kv.k_cache.ptr,
                    self.scratch.v_cur.ptr as *mut f32,
                    slot_kv.v_cache.ptr,
                    self.scratch.r_cur.ptr as *mut f32,
                    layer.attn_rel_clip.as_f32(),
                    tau_ptr,
                    self.scratch.pos_ids.ptr as *const i32,
                    sink_w_f32,
                    sink_w_f16,
                    in0_32,
                    self.scratch.sinks.ptr as *mut f32,
                    n_embd,
                    n_head,
                    n_head_kv,
                    head_dim,
                    self.cfg.d_rel as i32,
                    m,
                    slot_kv.ring_cap as i32,
                    kv_mode,
                    self.cfg.qk_clip,
                    self.cfg.rel_clip,
                    eps,
                    self.stream.raw,
                );
            }
            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][2].record(&self.stream)?;
            }
            unsafe {
                launch_attn_block_mma(
                    self.scratch.q16.ptr as *const u16,
                    self.scratch.r_cur.ptr as *const f32,
                    layer.attn_rel_bias.as_f32(),
                    self.scratch.sinks.ptr as *const f32,
                    self.scratch.attn_gate_buf.ptr as *const f32,
                    slot_kv.k_cache.ptr,
                    slot_kv.v_cache.ptr,
                    self.scratch.pos_ids.ptr as *const i32,
                    self.scratch.split_out.ptr as *mut f32,
                    self.scratch.split_ml.ptr as *mut f32,
                    self.scratch.attn_out.ptr as *mut f32,
                    std::ptr::null_mut(),
                    n_head,
                    n_head_kv,
                    self.cfg.d_rel as i32,
                    self.cfg.rel_extent as i32,
                    slot_kv.swa_window as i32,
                    slot_kv.ring_cap as i32,
                    kv_mode,
                    m,
                    self.stream.raw,
                );
            }
            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][3].record(&self.stream)?;
            }

            if ip % self.cfg.attn_res_block_size == 0 {
                let b_idx = ip / self.cfg.attn_res_block_size;
                self.scratch.blocks[b_idx].copy_from_device_async(
                    &self.scratch.cur,
                    hid_bytes,
                    &self.stream,
                )?;
            }

            let add_res_ptr = if ip % self.cfg.attn_res_block_size == 0 {
                std::ptr::null()
            } else {
                self.scratch.cur.ptr as *const f32
            };
            if wo_planar && m == 1 && !record_conv {
                unsafe {
                    launch_mv_q8_planar_m1(
                        layer.wo.q8_planar_q.as_ref().unwrap().ptr,
                        layer.wo.q8_planar_d.as_ref().unwrap().ptr,
                        n_embd,
                        self.scratch.attn_out.ptr as *const f32,
                        self.scratch.ffn_inp.ptr as *mut f32,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        layer.attn_post_conv.as_f32(),
                        2 * kv_dim,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        add_res_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        n_conv,
                        q_dim,
                        self.stream.raw,
                    );
                }
            } else {
                if wo_planar {
                    unsafe {
                        launch_mv_q8p_colsx(
                            layer.wo.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                            layer.wo.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                            n_embd,
                            q_dim,
                            self.scratch.attn_out.ptr as *const f32,
                            q_dim,
                            m,
                            self.scratch.attn_proj.ptr as *mut f32,
                            n_embd,
                            self.stream.raw,
                        );
                    }
                } else {
                    self.linear_on_stream(
                        &layer.wo,
                        self.scratch.attn_out.ptr as *const f32,
                        false,
                        self.scratch.attn_proj.ptr as *mut f32,
                        n_embd,
                        q_dim,
                        m,
                    );
                }
                unsafe {
                    launch_short_conv_rec_f32(
                        self.scratch.attn_proj.ptr as *const f32,
                        layer.attn_post_conv.as_f32(),
                        std::ptr::null(),
                        add_res_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.ffn_inp.ptr as *mut f32,
                        rec_old_ptr,
                        rec_x_ptr,
                        n_sconv_channels,
                        slot as i32,
                        n_embd,
                        0,
                        m,
                        2 * kv_dim,
                        n_conv,
                        self.stream.raw,
                    );
                }
            }

            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][4].record(&self.stream)?;
            }

            let (ffn_ptrs_buf, ffn_n_states) = &self.mix_tables.ffn_mix_pingpong[slot];
            if pass > 0 {
                unsafe {
                    launch_attn_res_mix_f32(
                        ffn_ptrs_buf.ptr as *const *const f32,
                        *ffn_n_states as i32,
                        layer.ffn_res_query.as_f32(),
                        self.scratch.mix_out.ptr as *mut f32,
                        n_embd,
                        1,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
                let lc = layer.loop_ctrl.as_ref().unwrap();
                let msg_job_off = if batched_loop_msg {
                    (ip - self.cfg.loop_start) * 4 * hid_bytes
                } else {
                    0
                };
                let msg_ptr = unsafe {
                    (self.scratch.loop_messages.ptr as *const u8).add(msg_job_off + 3 * hid_bytes)
                        as *const f32
                };
                let all_f32 = matches!(lc.down[3].ggml_type, GgmlType::F32)
                    && matches!(lc.gain[3].ggml_type, GgmlType::F32)
                    && matches!(lc.forget[3].ggml_type, GgmlType::F32);
                if all_f32 {
                    unsafe {
                        launch_loop_gates_fast_f32(
                            self.scratch.mix_out.ptr as *const f32, std::ptr::null(), std::ptr::null(),
                            msg_ptr, std::ptr::null(), std::ptr::null(),
                            lc.down[3].as_f32(), std::ptr::null(), std::ptr::null(),
                            lc.forget[3].as_f32(), std::ptr::null(), std::ptr::null(),
                            lc.gain[3].as_f32(), std::ptr::null(), std::ptr::null(),
                            self.scratch.mix_out.ptr as *mut f32, std::ptr::null_mut(), std::ptr::null_mut(),
                            self.scratch.ctrl_gate.ptr as *mut f32,
                            1,
                            m,
                            n_embd,
                            self.cfg.loop_gate_rank as i32,
                            eps,
                            self.stream.raw,
                        );
                    }
                } else {
                    self.run_loop_control(self.scratch.mix_out.ptr as *mut f32, msg_ptr, lc, 3, m);
                }
                unsafe {
                    launch_rmsnorm_f32(
                        self.scratch.mix_out.ptr as *const f32,
                        layer.ffn_norm.as_f32(),
                        self.scratch.tmp_hidden.ptr as *mut f32,
                        n_embd,
                        m,
                        eps,
                        self.stream.raw,
                    );
                }
            } else {
                unsafe {
                    launch_attn_res_mix_rms_f32(
                        ffn_ptrs_buf.ptr as *const *const f32,
                        *ffn_n_states as i32,
                        layer.ffn_res_query.as_f32(),
                        layer.ffn_norm.as_f32(),
                        std::ptr::null(),
                        std::ptr::null(),
                        self.scratch.tmp_hidden.ptr as *mut f32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        n_embd,
                        1,
                        m,
                        eps,
                        eps,
                        self.stream.raw,
                    );
                }
            }

            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][5].record(&self.stream)?;
            }

            if let Some(ref f16_w) = layer.ffn_gate_inp.f16_buf {
                unsafe {
                    launch_mv_f16_batch(
                        f16_w.ptr as *const u16,
                        self.scratch.tmp_hidden.ptr as *const f32,
                        self.scratch.router_logits.ptr as *mut f32,
                        self.cfg.n_expert as i32,
                        n_embd,
                        m,
                        self.stream.raw,
                    );
                }
            } else if matches!(layer.ffn_gate_inp.ggml_type, GgmlType::F32) {
                unsafe {
                    launch_router_f32_batch(
                        layer.ffn_gate_inp.as_f32(),
                        self.scratch.tmp_hidden.ptr as *const f32,
                        self.scratch.router_logits.ptr as *mut f32,
                        self.cfg.n_expert as i32,
                        n_embd,
                        m,
                        self.stream.raw,
                    );
                }
            } else {
                self.linear_on_stream(
                    &layer.ffn_gate_inp,
                    self.scratch.tmp_hidden.ptr as *const f32,
                    false,
                    self.scratch.router_logits.ptr as *mut f32,
                    self.cfg.n_expert as i32,
                    n_embd,
                    m,
                );
            }
            let bias_ptr = layer
                .exp_probs_b
                .as_ref()
                .map(|t| t.as_f32())
                .unwrap_or(std::ptr::null());

            let (gu_ptrs, dn_ptrs) = gu_ptrs_for_layer(ip);
            let ffn_inp_ptr = self.scratch.ffn_inp.ptr as *const f32;
            let use_fast_moe = all_resident
                && matches!(layer.gate_up_type, GgmlType::Nvfp4 | GgmlType::Q8_0)
                && layer.gate_up_type == layer.down_type
                && self.cfg.n_embd % 64 == 0
                && self.scratch.n_ff_exp_local % 64 == 0;
            let use_fused_route_moe = use_fast_moe
                && self.cfg.n_expert <= 128
                && self.cfg.n_expert_used <= 8
                && m <= 8;

            if !use_fused_route_moe {
                unsafe {
                    launch_router_topk_sigmoid_f32(
                        self.scratch.router_logits.ptr as *const f32,
                        bias_ptr,
                        self.scratch.expert_ids.ptr as *mut i32,
                        self.scratch.expert_weights.ptr as *mut f32,
                        self.cfg.n_expert as i32,
                        self.cfg.n_expert_used as i32,
                        m,
                        self.cfg.expert_weights_scale,
                        self.stream.raw,
                    );
                }
            }

            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][6].record(&self.stream)?;
            }

            if use_fused_route_moe {
                let gu_sc_ptr = layer
                    .gate_up_scale
                    .as_ref()
                    .map(|s| s.as_f32())
                    .unwrap_or(std::ptr::null());
                let dn_sc_ptr = layer
                    .down_scale
                    .as_ref()
                    .map(|s| s.as_f32())
                    .unwrap_or(std::ptr::null());
                unsafe {
                    launch_route_and_moe_block_fast(
                        self.scratch.router_logits.ptr as *const f32,
                        bias_ptr,
                        self.cfg.n_expert as i32,
                        self.cfg.expert_weights_scale,
                        gu_ptrs as *const u64,
                        dn_ptrs as *const u64,
                        gu_sc_ptr,
                        dn_sc_ptr,
                        layer.gate_up_type as i32,
                        self.scratch.tmp_hidden.ptr as *const f32,
                        self.scratch.expert_ids.ptr as *mut i32,
                        self.scratch.expert_weights.ptr as *mut f32,
                        self.scratch.x_limbs.ptr as *mut u16,
                        self.scratch.moe_inter_f16.ptr as *mut u16,
                        self.scratch.moe_slot_out.ptr as *mut f32,
                        self.scratch.grp_gu_ptrs.ptr as *mut u64,
                        self.scratch.grp_dn_ptrs.ptr as *mut u64,
                        self.scratch.grp_n.ptr as *mut i32,
                        self.scratch.grp_tok.ptr as *mut i32,
                        self.scratch.grp_row.ptr as *mut i32,
                        self.scratch.slot_sg.ptr as *mut f32,
                        self.scratch.slot_su.ptr as *mut f32,
                        self.scratch.slot_wt.ptr as *mut f32,
                        layer.ffn_post_conv.as_f32(),
                        slot_kv.conv_history.ptr as *mut f32,
                        ffn_inp_ptr,
                        self.scratch.cur.ptr as *mut f32,
                        rec_old_ptr,
                        rec_x_ptr,
                        n_sconv_channels,
                        slot as i32,
                        2 * kv_dim + n_embd,
                        n_conv,
                        n_embd,
                        self.scratch.n_ff_exp_local as i32,
                        self.cfg.n_expert_used as i32,
                        m,
                        self.cfg.situ_beta,
                        self.cfg.situ_linear_beta,
                        self.stream.raw,
                    );
                }
            } else if use_fast_moe {
                let gu_sc_ptr = layer
                    .gate_up_scale
                    .as_ref()
                    .map(|s| s.as_f32())
                    .unwrap_or(std::ptr::null());
                let dn_sc_ptr = layer
                    .down_scale
                    .as_ref()
                    .map(|s| s.as_f32())
                    .unwrap_or(std::ptr::null());
                unsafe {
                    launch_moe_block_fast(
                        gu_ptrs as *const u64,
                        dn_ptrs as *const u64,
                        gu_sc_ptr,
                        dn_sc_ptr,
                        layer.gate_up_type as i32,
                        self.scratch.tmp_hidden.ptr as *const f32,
                        self.scratch.expert_ids.ptr as *const i32,
                        self.scratch.expert_weights.ptr as *const f32,
                        self.scratch.x_limbs.ptr as *mut u16,
                        self.scratch.moe_inter_f16.ptr as *mut u16,
                        self.scratch.moe_slot_out.ptr as *mut f32,
                        self.scratch.grp_gu_ptrs.ptr as *mut u64,
                        self.scratch.grp_dn_ptrs.ptr as *mut u64,
                        self.scratch.grp_n.ptr as *mut i32,
                        self.scratch.grp_tok.ptr as *mut i32,
                        self.scratch.grp_row.ptr as *mut i32,
                        self.scratch.slot_sg.ptr as *mut f32,
                        self.scratch.slot_su.ptr as *mut f32,
                        self.scratch.slot_wt.ptr as *mut f32,
                        layer.ffn_post_conv.as_f32(),
                        slot_kv.conv_history.ptr as *mut f32,
                        ffn_inp_ptr,
                        self.scratch.cur.ptr as *mut f32,
                        rec_old_ptr,
                        rec_x_ptr,
                        n_sconv_channels,
                        slot as i32,
                        2 * kv_dim + n_embd,
                        n_conv,
                        n_embd,
                        self.scratch.n_ff_exp_local as i32,
                        self.cfg.n_expert_used as i32,
                        m,
                        self.cfg.situ_beta,
                        self.cfg.situ_linear_beta,
                        self.stream.raw,
                    );
                }
            } else {
                unsafe {
                    launch_quantize_q8_1(
                        self.scratch.tmp_hidden.ptr as *const f32,
                        self.scratch.act_q8_1.ptr,
                        m,
                        n_embd,
                        self.stream.raw,
                    );
                }
                self.enqueue_moe_local(ip, gu_ptrs, dn_ptrs, m);
                unsafe {
                    launch_short_conv_rec_f32(
                        self.scratch.moe_out.ptr as *const f32,
                        layer.ffn_post_conv.as_f32(),
                        std::ptr::null(),
                        ffn_inp_ptr,
                        slot_kv.conv_history.ptr as *mut f32,
                        self.scratch.cur.ptr as *mut f32,
                        rec_old_ptr,
                        rec_x_ptr,
                        n_sconv_channels,
                        slot as i32,
                        n_embd,
                        0,
                        m,
                        2 * kv_dim + n_embd,
                        n_conv,
                        self.stream.raw,
                    );
                }
            }
            if let Some((_, _, ref lvs, _)) = prof_events {
                lvs[vis][7].record(&self.stream)?;
            }
        }

        let (out_ptrs, out_n_states) = &self.mix_tables.output_mix;
        let out_w = self.weights.output.as_ref().expect("output required");
        let out_norm = self
            .weights
            .output_norm
            .as_ref()
            .expect("output_norm required");
        let out_res_q = self
            .weights
            .output_res_query
            .as_ref()
            .expect("output_res_query required");
        let n_vocab_out = self.cfg.n_vocab_unpadded as i32;

        unsafe {
            launch_attn_res_mix_rms_f16(
                out_ptrs.ptr as *const *const f32,
                *out_n_states as i32,
                out_res_q.as_f32(),
                out_norm.as_f32(),
                std::ptr::null(),
                std::ptr::null(),
                self.scratch.tmp_hidden.ptr as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                n_embd,
                1,
                m,
                eps,
                eps,
                self.stream.raw,
            );
        }
        if record_taps {
            self.record_final_tap(m)?;
        }

        if !capture_logits && out_w.q8_planar_q.is_some() {
            unsafe {
                launch_mv_q8p_colsx_argmax(
                    out_w.q8_planar_q.as_ref().unwrap().ptr as *const i8,
                    out_w.q8_planar_d.as_ref().unwrap().ptr as *const u16,
                    n_vocab_out,
                    n_embd,
                    self.scratch.tmp_hidden.ptr as *const f32,
                    n_embd,
                    m,
                    self.scratch.argmax_part_vals.ptr as *mut f32,
                    self.scratch.argmax_part_ids.ptr as *mut i32,
                    self.scratch.argmax_ids.ptr as *mut i32,
                    self.stream.raw,
                );
            }
        } else {
            let last_row = unsafe {
                (self.scratch.tmp_hidden.ptr as *const f32).add((m_usize - 1) * self.cfg.n_embd)
            };
            self.linear_on_stream(
                out_w,
                last_row,
                false,
                self.scratch.logits.ptr as *mut f32,
                n_vocab_out,
                n_embd,
                1,
            );
            if self.cfg.final_logit_softcapping > 0.0 {
                unsafe {
                    launch_softcap_tanh_f32(
                        self.scratch.logits.ptr as *mut f32,
                        n_vocab_out,
                        self.cfg.final_logit_softcapping,
                        self.stream.raw,
                    );
                }
            }
            unsafe {
                launch_argmax_f32(
                    self.scratch.logits.ptr as *const f32,
                    n_vocab_out,
                    self.scratch.argmax_id.ptr as *mut i32,
                    self.stream.raw,
                );
            }
        }

        if let Some((ref e0, ref e_emb, ref lvs, ref e_head)) = prof_events {
            e_head.record(&self.stream)?;
            self.stream.sync()?;
            let embed_ms = e0.elapsed_ms(e_emb)?;
            let mut s_attn_pre = 0.0f32;
            let mut s_qkvrg = 0.0f32;
            let mut s_prep = 0.0f32;
            let mut s_attn = 0.0f32;
            let mut s_wo = 0.0f32;
            let mut s_ffn_pre = 0.0f32;
            let mut s_router = 0.0f32;
            let mut s_moe = 0.0f32;
            let mut prev = e_emb;
            for lv in lvs.iter() {
                s_attn_pre += prev.elapsed_ms(&lv[0])?;
                s_qkvrg += lv[0].elapsed_ms(&lv[1])?;
                s_prep += lv[1].elapsed_ms(&lv[2])?;
                s_attn += lv[2].elapsed_ms(&lv[3])?;
                s_wo += lv[3].elapsed_ms(&lv[4])?;
                s_ffn_pre += lv[4].elapsed_ms(&lv[5])?;
                s_router += lv[5].elapsed_ms(&lv[6])?;
                s_moe += lv[6].elapsed_ms(&lv[7])?;
                prev = &lv[7];
            }
            let head_ms = prev.elapsed_ms(e_head)?;
            let total_ms = e0.elapsed_ms(e_head)?;
            eprintln!(
                "stage_prof(T={m}): total={total_ms:.3}ms | embed={embed_ms:.3} attn_pre={s_attn_pre:.3} qkvrg={s_qkvrg:.3} prep={s_prep:.3} attn={s_attn:.3} wo={s_wo:.3} ffn_pre={s_ffn_pre:.3} router={s_router:.3} moe={s_moe:.3} head={head_ms:.3}"
            );
        }

        Ok(())
    }
}

unsafe fn cuda_check_copy_d2d(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: CudaStreamHandle,
) -> Result<()> {
    extern "C" {
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: std::os::raw::c_int,
            stream: *mut c_void,
        ) -> std::os::raw::c_int;
    }
    ampllm_cuda_kernels::cuda_rt::cuda_check(
        cudaMemcpyAsync(
            dst,
            src,
            bytes,
            ampllm_cuda_kernels::cuda_rt::CUDA_MEMCPY_DEVICE_TO_DEVICE,
            stream.0,
        ),
        "cudaMemcpyAsync D2D",
    )
}
