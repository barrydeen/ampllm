use ampllm_cuda_kernels::{set_device, CudaStream, DeviceBuffer};
use ampllm_gguf::{GgmlType, GgufFile, Inkling2Config};
use anyhow::{bail, Result};
use std::ops::Range;

pub struct DeviceTensor {
    pub buf: DeviceBuffer,
    pub q8_planar_q: Option<DeviceBuffer>,
    pub q8_planar_d: Option<DeviceBuffer>,
    pub f16_buf: Option<DeviceBuffer>,
    pub ggml_type: GgmlType,
    pub dims: Vec<u64>,
}

fn want_f16_copy(name: &str) -> bool {
    name.ends_with("ffn_gate_inp.weight")
        || name.ends_with("attn_sink_proj.weight")
        || name.ends_with("attn_k_conv_proj.weight")
        || name.ends_with("attn_v_conv_proj.weight")
}

fn build_f16_copy(
    bytes: &[u8],
    ggml_type: GgmlType,
    device: i32,
    stream: &CudaStream,
) -> Result<Option<DeviceBuffer>> {
    let f16_host: Vec<u16> = match ggml_type {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                half::f16::from_f32(v).to_bits()
            })
            .collect(),
        GgmlType::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let b = u16::from_le_bytes([c[0], c[1]]);
                let v = f32::from_bits((b as u32) << 16);
                half::f16::from_f32(v).to_bits()
            })
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
        _ => return Ok(None),
    };
    let buf = DeviceBuffer::alloc(device, f16_host.len() * 2)?;
    buf.copy_from_host_async(bytemuck::cast_slice(&f16_host), stream)?;
    Ok(Some(buf))
}

fn build_q8_0_planar(
    bytes: &[u8],
    k: usize,
    n_rows: usize,
    device: i32,
    stream: &CudaStream,
) -> Result<(Option<DeviceBuffer>, Option<DeviceBuffer>)> {
    if k % 128 != 0 {
        return Ok((None, None));
    }
    let nb = k / 32;
    if bytes.len() != n_rows * nb * 34 {
        return Ok((None, None));
    }
    let mut q_host = vec![0u8; n_rows * k];
    let mut d_host = vec![0u8; n_rows * nb * 2];
    for r in 0..n_rows {
        let row_src = &bytes[r * nb * 34..(r + 1) * nb * 34];
        let q_dst = &mut q_host[r * k..(r + 1) * k];
        let d_dst = &mut d_host[r * nb * 2..(r + 1) * nb * 2];
        for b in 0..nb {
            let blk = &row_src[b * 34..(b + 1) * 34];
            d_dst[b * 2..b * 2 + 2].copy_from_slice(&blk[0..2]);
            q_dst[b * 32..(b + 1) * 32].copy_from_slice(&blk[2..34]);
        }
    }
    let q_buf = DeviceBuffer::alloc(device, q_host.len())?;
    let d_buf = DeviceBuffer::alloc(device, d_host.len())?;
    q_buf.copy_from_host_async(&q_host, stream)?;
    d_buf.copy_from_host_async(&d_host, stream)?;
    Ok((Some(q_buf), Some(d_buf)))
}

impl DeviceTensor {
    pub fn from_gguf(gguf: &GgufFile, name: &str, device: i32, stream: &CudaStream) -> Result<Self> {
        let info = gguf.tensor_info(name)?;
        let bytes = gguf.tensor_bytes(name)?;
        let (q8_planar_q, q8_planar_d) =
            if info.ggml_type == GgmlType::Q8_0 && info.dims.len() == 2 && name != "token_embd.weight" {
                build_q8_0_planar(bytes, info.dims[0] as usize, info.dims[1] as usize, device, stream)?
            } else {
                (None, None)
            };
        let f16_buf = if want_f16_copy(name) {
            build_f16_copy(bytes, info.ggml_type, device, stream)?
        } else {
            None
        };
        let buf = if q8_planar_q.is_some() {
            DeviceBuffer::alloc(device, 0)?
        } else {
            let b = DeviceBuffer::alloc(device, bytes.len())?;
            b.copy_from_host_async(bytes, stream)?;
            b
        };
        Ok(Self {
            buf,
            q8_planar_q,
            q8_planar_d,
            f16_buf,
            ggml_type: info.ggml_type,
            dims: info.dims.clone(),
        })
    }

    /// Slice contiguous rows along outer dim `[row_start .. row_start + row_count]` of a 2D tensor `[ne0, ne1]`
    /// (or elements for 1D dense tensor `[ne0]`).
    pub fn from_gguf_row_shard(
        gguf: &GgufFile,
        name: &str,
        row_start: usize,
        row_count: usize,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        let info = gguf.tensor_info(name)?;
        let bytes = gguf.tensor_bytes(name)?;
        if info.dims.len() == 1 {
            let elem_bytes = info.ggml_type.row_bytes(1);
            let s = row_start * elem_bytes;
            let e = (row_start + row_count) * elem_bytes;
            let buf = DeviceBuffer::alloc(device, e - s)?;
            buf.copy_from_host(&bytes[s..e])?;
            return Ok(Self {
                buf,
                q8_planar_q: None,
                q8_planar_d: None,
                f16_buf: None,
                ggml_type: info.ggml_type,
                dims: vec![row_count as u64],
            });
        }
        if info.dims.len() != 2 {
            bail!("from_gguf_row_shard expected 1D or 2D tensor for {name}, got {:?}", info.dims);
        }
        let rb = info.ggml_type.row_bytes(info.dims[0] as usize);
        let s = row_start * rb;
        let e = (row_start + row_count) * rb;
        if e > bytes.len() {
            bail!("row shard out of bounds for {name}: {e} > {}", bytes.len());
        }
        let (q8_planar_q, q8_planar_d) =
            if info.ggml_type == GgmlType::Q8_0 && name != "token_embd.weight" {
                build_q8_0_planar(&bytes[s..e], info.dims[0] as usize, row_count, device, stream)?
            } else {
                (None, None)
            };
        let f16_buf = if want_f16_copy(name) {
            build_f16_copy(&bytes[s..e], info.ggml_type, device, stream)?
        } else {
            None
        };
        let buf = if q8_planar_q.is_some() {
            DeviceBuffer::alloc(device, 0)?
        } else {
            let b = DeviceBuffer::alloc(device, e - s)?;
            b.copy_from_host(&bytes[s..e])?;
            b
        };
        Ok(Self {
            buf,
            q8_planar_q,
            q8_planar_d,
            f16_buf,
            ggml_type: info.ggml_type,
            dims: vec![info.dims[0], row_count as u64],
        })
    }

    /// Slice quantized/dense columns `[col_start .. col_start + col_count]` along innermost dim 0 of a 2D tensor `[total_cols, n_rows]`.
    pub fn from_gguf_col_shard(
        gguf: &GgufFile,
        name: &str,
        col_start: usize,
        col_count: usize,
        total_cols: usize,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        let info = gguf.tensor_info(name)?;
        let bytes = gguf.tensor_bytes(name)?;
        if info.dims.len() != 2 || info.dims[0] as usize != total_cols {
            bail!(
                "from_gguf_col_shard mismatch for {name}: dims={:?}, total_cols={total_cols}",
                info.dims
            );
        }
        let n_rows = info.dims[1] as usize;
        let full_rb = info.ggml_type.row_bytes(total_cols);
        let off_b = info.ggml_type.row_bytes(col_start);
        let len_b = info.ggml_type.row_bytes(col_count);
        let mut staged = Vec::with_capacity(n_rows * len_b);
        for r in 0..n_rows {
            let s = r * full_rb + off_b;
            staged.extend_from_slice(&bytes[s..s + len_b]);
        }
        let (q8_planar_q, q8_planar_d) = if info.ggml_type == GgmlType::Q8_0 {
            build_q8_0_planar(&staged, col_count, n_rows, device, stream)?
        } else {
            (None, None)
        };
        let f16_buf = if want_f16_copy(name) {
            build_f16_copy(&staged, info.ggml_type, device, stream)?
        } else {
            None
        };
        let buf = if q8_planar_q.is_some() {
            DeviceBuffer::alloc(device, 0)?
        } else {
            let b = DeviceBuffer::alloc(device, staged.len())?;
            b.copy_from_host(&staged)?;
            b
        };
        Ok(Self {
            buf,
            q8_planar_q,
            q8_planar_d,
            f16_buf,
            ggml_type: info.ggml_type,
            dims: vec![col_count as u64, n_rows as u64],
        })
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.buf.ptr as *const T
    }

    pub fn as_f32(&self) -> *const f32 {
        self.buf.ptr as *const f32
    }
}

pub struct LoopCtrlWeights {
    pub msg_query: DeviceTensor,
    pub down: [DeviceTensor; 4],
    pub gain: [DeviceTensor; 4],
    pub forget: [DeviceTensor; 4],
}

pub struct LayerWeights {
    pub is_swa: bool,
    pub wq: DeviceTensor,
    pub wk: DeviceTensor,
    pub wv: DeviceTensor,
    pub wo: DeviceTensor,
    pub attn_gate: DeviceTensor,
    pub attn_q_clip: DeviceTensor,
    pub attn_sink_proj: DeviceTensor,
    pub attn_rel: DeviceTensor,
    pub attn_rel_bias: DeviceTensor,
    pub attn_rel_clip: DeviceTensor,
    pub attn_k_conv: DeviceTensor,
    pub attn_v_conv: DeviceTensor,
    pub attn_post_conv: DeviceTensor,
    pub ffn_post_conv: DeviceTensor,

    pub attn_norm: Option<DeviceTensor>,
    pub attn_q_inp_norm: Option<DeviceTensor>,
    pub attn_k_inp_norm: Option<DeviceTensor>,
    pub attn_v_inp_norm: Option<DeviceTensor>,
    pub attn_k_conv_proj: Option<DeviceTensor>,
    pub attn_v_conv_proj: Option<DeviceTensor>,

    pub attn_res_query: Option<DeviceTensor>,
    pub ffn_res_query: DeviceTensor,

    pub ffn_norm: DeviceTensor,
    pub ffn_gate_inp: DeviceTensor,
    pub exp_probs_b: Option<DeviceTensor>,
    pub gate_up_scale: Option<DeviceTensor>,
    pub down_scale: Option<DeviceTensor>,
    pub gate_up_scale_host: Vec<f32>,
    pub down_scale_host: Vec<f32>,
    pub gate_up_type: GgmlType,
    pub down_type: GgmlType,
    pub gate_up_row_bytes: usize,
    pub down_row_bytes: usize,
    pub gate_up_expert_bytes: usize,
    pub down_expert_bytes: usize,

    pub loop_ctrl: Option<LoopCtrlWeights>,
}

pub struct ModelWeights {
    pub device: i32,
    pub tok_embd: Option<DeviceTensor>,
    pub tok_embd_norm: Option<DeviceTensor>,
    pub output: Option<DeviceTensor>,
    pub output_norm: Option<DeviceTensor>,
    pub output_res_query: Option<DeviceTensor>,
    pub loop_msg_queries_all: Option<DeviceBuffer>,
    pub layers: Vec<Option<LayerWeights>>,
}

impl ModelWeights {
    #[inline]
    pub fn layer(&self, ip: usize) -> &LayerWeights {
        self.layers[ip].as_ref().expect("layer not loaded on this GPU")
    }

    pub fn load_non_experts(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        Self::load_non_experts_range(
            gguf,
            cfg,
            0..cfg.n_layer_phys,
            true,
            true,
            device,
            stream,
        )
    }

    pub fn load_non_experts_range(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        phys_range: Range<usize>,
        load_embd: bool,
        load_output: bool,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        set_device(device)?;
        let tok_embd = if load_embd {
            Some(DeviceTensor::from_gguf(gguf, "token_embd.weight", device, stream)?)
        } else {
            None
        };
        let tok_embd_norm = if load_embd {
            Some(DeviceTensor::from_gguf(gguf, "token_embd_norm.weight", device, stream)?)
        } else {
            None
        };
        let output = if load_output {
            Some(DeviceTensor::from_gguf(gguf, "output.weight", device, stream)?)
        } else {
            None
        };
        let output_norm = if load_output {
            Some(DeviceTensor::from_gguf(gguf, "output_norm.weight", device, stream)?)
        } else {
            None
        };
        let output_res_query = if load_output {
            Some(DeviceTensor::from_gguf(gguf, "output_res_query.weight", device, stream)?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layer_phys);
        for i in 0..cfg.n_layer_phys {
            if !phys_range.contains(&i) {
                layers.push(None);
                continue;
            }
            layers.push(Some(Self::load_layer_full(gguf, cfg, i, device, stream)?));
        }
        let loop_msg_queries_all = Self::build_loop_msg_queries_all(cfg, &layers, device, stream)?;
        stream.sync()?;

        Ok(Self {
            device,
            tok_embd,
            tok_embd_norm,
            output,
            output_norm,
            output_res_query,
            loop_msg_queries_all,
            layers,
        })
    }

    fn build_loop_msg_queries_all(
        cfg: &Inkling2Config,
        layers: &[Option<LayerWeights>],
        device: i32,
        stream: &CudaStream,
    ) -> Result<Option<DeviceBuffer>> {
        if !cfg.is_looped || cfg.loop_end <= cfg.loop_start {
            return Ok(None);
        }
        let n_core = cfg.loop_end - cfg.loop_start;
        for ip in cfg.loop_start..cfg.loop_end {
            if layers.get(ip).and_then(|x| x.as_ref()).and_then(|l| l.loop_ctrl.as_ref()).is_none() {
                return Ok(None);
            }
        }
        let per_layer_bytes = 4 * cfg.n_embd * 4;
        let buf = DeviceBuffer::alloc(device, n_core * per_layer_bytes)?;
        for (ci, ip) in (cfg.loop_start..cfg.loop_end).enumerate() {
            let mq = &layers[ip].as_ref().unwrap().loop_ctrl.as_ref().unwrap().msg_query.buf;
            unsafe {
                extern "C" {
                    fn cudaMemcpyAsync(
                        dst: *mut std::ffi::c_void,
                        src: *const std::ffi::c_void,
                        count: usize,
                        kind: std::os::raw::c_int,
                        stream: *mut std::ffi::c_void,
                    ) -> std::os::raw::c_int;
                }
                ampllm_cuda_kernels::cuda_rt::cuda_check(
                    cudaMemcpyAsync(
                        (buf.ptr as *mut u8).add(ci * per_layer_bytes) as *mut std::ffi::c_void,
                        mq.ptr,
                        per_layer_bytes,
                        ampllm_cuda_kernels::cuda_rt::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                        stream.raw.0,
                    ),
                    "cudaMemcpyAsync loop_msg_queries_all",
                )?;
            }
        }
        Ok(Some(buf))
    }

    fn load_layer_common(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        i: usize,
        device: i32,
        stream: &CudaStream,
    ) -> Result<(
        bool,
        Option<DeviceTensor>,
        Option<DeviceTensor>,
        Option<DeviceTensor>,
        Option<DeviceTensor>,
        Option<DeviceTensor>,
        Option<DeviceTensor>,
        Option<LoopCtrlWeights>,
    )> {
        let p = |name: &str| format!("blk.{i}.{name}");
        let is_swa = cfg.is_swa_phys[i];
        let attn_norm = if is_swa {
            Some(DeviceTensor::from_gguf(gguf, &p("attn_norm.weight"), device, stream)?)
        } else {
            None
        };
        let (q_inp, k_inp, v_inp) = if !is_swa {
            (
                Some(DeviceTensor::from_gguf(gguf, &p("attn_q_inp_norm.weight"), device, stream)?),
                Some(DeviceTensor::from_gguf(gguf, &p("attn_k_inp_norm.weight"), device, stream)?),
                Some(DeviceTensor::from_gguf(gguf, &p("attn_v_inp_norm.weight"), device, stream)?),
            )
        } else {
            (None, None, None)
        };
        let attn_res_query = if i > 0 {
            Some(DeviceTensor::from_gguf(gguf, &p("attn_res_query.weight"), device, stream)?)
        } else {
            None
        };
        let exp_probs_b = if gguf.has_tensor(&p("exp_probs_b.bias")) {
            Some(DeviceTensor::from_gguf(gguf, &p("exp_probs_b.bias"), device, stream)?)
        } else {
            None
        };
        let loop_ctrl = if cfg.is_looped && i >= cfg.loop_start && i < cfg.loop_end {
            if gguf.has_tensor(&p("loop_msg_query.weight")) {
                let names = ["q", "k", "v", "ffn"];
                let down = [
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_down.weight", names[0])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_down.weight", names[1])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_down.weight", names[2])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_down.weight", names[3])), device, stream)?,
                ];
                let gain = [
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_gain.weight", names[0])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_gain.weight", names[1])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_gain.weight", names[2])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_gain.weight", names[3])), device, stream)?,
                ];
                let forget = [
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_forget.weight", names[0])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_forget.weight", names[1])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_forget.weight", names[2])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_ctrl_forget.weight", names[3])), device, stream)?,
                ];
                Some(LoopCtrlWeights {
                    msg_query: DeviceTensor::from_gguf(gguf, &p("loop_msg_query.weight"), device, stream)?,
                    down,
                    gain,
                    forget,
                })
            } else {
                // Unsloth `inkling2` layout: `loop_res_query.weight` [C, 2*n_core] + `loop_kv_res_query.weight` [C, 2*n_core]
                // and `blk.{i}.loop_{attn,k,v,mlp}_{down,gain,forget}.weight`.
                let names = ["attn", "k", "v", "mlp"];
                let down = [
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_down.weight", names[0])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_down.weight", names[1])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_down.weight", names[2])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_down.weight", names[3])), device, stream)?,
                ];
                let gain = [
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_gain.weight", names[0])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_gain.weight", names[1])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_gain.weight", names[2])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_gain.weight", names[3])), device, stream)?,
                ];
                let forget = [
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_forget.weight", names[0])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_forget.weight", names[1])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_forget.weight", names[2])), device, stream)?,
                    DeviceTensor::from_gguf(gguf, &p(&format!("loop_{}_forget.weight", names[3])), device, stream)?,
                ];
                let ci = i - cfg.loop_start;
                let c = cfg.n_embd;
                let rq = gguf.tensor_f32_slice("loop_res_query.weight")?;
                let kvq = gguf.tensor_f32_slice("loop_kv_res_query.weight")?;
                let mut mq_host = Vec::with_capacity(4 * c);
                mq_host.extend_from_slice(&rq[(2 * ci) * c..(2 * ci + 1) * c]);
                mq_host.extend_from_slice(&kvq[(2 * ci) * c..(2 * ci + 1) * c]);
                mq_host.extend_from_slice(&kvq[(2 * ci + 1) * c..(2 * ci + 2) * c]);
                mq_host.extend_from_slice(&rq[(2 * ci + 1) * c..(2 * ci + 2) * c]);
                let mq_buf = DeviceBuffer::alloc(device, 4 * c * 4)?;
                mq_buf.copy_from_host_async(bytemuck::cast_slice(&mq_host), stream)?;
                let msg_query = DeviceTensor {
                    buf: mq_buf,
                    q8_planar_q: None,
                    q8_planar_d: None,
                    f16_buf: None,
                    ggml_type: GgmlType::F32,
                    dims: vec![c as u64, 4],
                };
                Some(LoopCtrlWeights {
                    msg_query,
                    down,
                    gain,
                    forget,
                })
            }
        } else {
            None
        };
        Ok((is_swa, attn_norm, q_inp, k_inp, v_inp, attn_res_query, exp_probs_b, loop_ctrl))
    }

    fn load_layer_full(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        i: usize,
        device: i32,
        stream: &CudaStream,
    ) -> Result<LayerWeights> {
        let p = |name: &str| format!("blk.{i}.{name}");
        let (is_swa, attn_norm, q_inp, k_inp, v_inp, attn_res_query, exp_probs_b, loop_ctrl) =
            Self::load_layer_common(gguf, cfg, i, device, stream)?;

        let gu_info = gguf.tensor_info(&p("ffn_gate_up_exps.weight"))?;
        let dn_info = gguf.tensor_info(&p("ffn_down_exps.weight"))?;
        let gate_up_row_bytes = gu_info.ggml_type.row_bytes(cfg.n_embd);
        let down_row_bytes = dn_info.ggml_type.row_bytes(cfg.n_ff_exp);
        let gate_up_expert_bytes = gate_up_row_bytes * (2 * cfg.n_ff_exp);
        let down_expert_bytes = down_row_bytes * cfg.n_embd;

        let (attn_k_conv_proj, attn_v_conv_proj) = if !is_swa {
            (
                Some(DeviceTensor::from_gguf(gguf, &p("attn_k_conv_proj.weight"), device, stream)?),
                Some(DeviceTensor::from_gguf(gguf, &p("attn_v_conv_proj.weight"), device, stream)?),
            )
        } else {
            (None, None)
        };

        let (gate_up_scale, gate_up_scale_host) = if gguf.has_tensor(&p("ffn_gate_up_exps.scale")) {
            (
                Some(DeviceTensor::from_gguf(gguf, &p("ffn_gate_up_exps.scale"), device, stream)?),
                gguf.tensor_f32_slice(&p("ffn_gate_up_exps.scale"))?.to_vec(),
            )
        } else {
            (None, vec![1.0f32; cfg.n_expert])
        };
        let (down_scale, down_scale_host) = if gguf.has_tensor(&p("ffn_down_exps.scale")) {
            (
                Some(DeviceTensor::from_gguf(gguf, &p("ffn_down_exps.scale"), device, stream)?),
                gguf.tensor_f32_slice(&p("ffn_down_exps.scale"))?.to_vec(),
            )
        } else {
            (None, vec![1.0f32; cfg.n_expert])
        };

        Ok(LayerWeights {
            is_swa,
            wq: DeviceTensor::from_gguf(gguf, &p("attn_q.weight"), device, stream)?,
            wk: DeviceTensor::from_gguf(gguf, &p("attn_k.weight"), device, stream)?,
            wv: DeviceTensor::from_gguf(gguf, &p("attn_v.weight"), device, stream)?,
            wo: DeviceTensor::from_gguf(gguf, &p("attn_output.weight"), device, stream)?,
            attn_gate: DeviceTensor::from_gguf(gguf, &p("attn_gate.weight"), device, stream)?,
            attn_q_clip: DeviceTensor::from_gguf(gguf, &p("attn_q_clip.weight"), device, stream)?,
            attn_sink_proj: DeviceTensor::from_gguf(gguf, &p("attn_sink_proj.weight"), device, stream)?,
            attn_rel: DeviceTensor::from_gguf(gguf, &p("attn_rel.weight"), device, stream)?,
            attn_rel_bias: DeviceTensor::from_gguf(gguf, &p("attn_rel_bias.weight"), device, stream)?,
            attn_rel_clip: DeviceTensor::from_gguf(gguf, &p("attn_rel_clip.weight"), device, stream)?,
            attn_k_conv: DeviceTensor::from_gguf(gguf, &p("attn_k_conv.weight"), device, stream)?,
            attn_v_conv: DeviceTensor::from_gguf(gguf, &p("attn_v_conv.weight"), device, stream)?,
            attn_post_conv: DeviceTensor::from_gguf(gguf, &p("attn_post_conv.weight"), device, stream)?,
            ffn_post_conv: DeviceTensor::from_gguf(gguf, &p("ffn_post_conv.weight"), device, stream)?,
            attn_norm,
            attn_q_inp_norm: q_inp,
            attn_k_inp_norm: k_inp,
            attn_v_inp_norm: v_inp,
            attn_k_conv_proj,
            attn_v_conv_proj,
            attn_res_query,
            ffn_res_query: DeviceTensor::from_gguf(gguf, &p("ffn_res_query.weight"), device, stream)?,
            ffn_norm: DeviceTensor::from_gguf(gguf, &p("ffn_norm.weight"), device, stream)?,
            ffn_gate_inp: DeviceTensor::from_gguf(gguf, &p("ffn_gate_inp.weight"), device, stream)?,
            exp_probs_b,
            gate_up_scale,
            down_scale,
            gate_up_scale_host,
            down_scale_host,
            gate_up_type: gu_info.ggml_type,
            down_type: dn_info.ggml_type,
            gate_up_row_bytes,
            down_row_bytes,
            gate_up_expert_bytes,
            down_expert_bytes,
            loop_ctrl,
        })
    }

    /// Load 2-way Tensor Parallel weights (`rank` in 0..2) onto `device`.
    pub fn load_tp2_rank(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        rank: usize,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        set_device(device)?;
        assert!(rank < 2);
        let tok_embd = Some(DeviceTensor::from_gguf(gguf, "token_embd.weight", device, stream)?);
        let tok_embd_norm = Some(DeviceTensor::from_gguf(gguf, "token_embd_norm.weight", device, stream)?);
        let output_norm = Some(DeviceTensor::from_gguf(gguf, "output_norm.weight", device, stream)?);
        let output_res_query = Some(DeviceTensor::from_gguf(gguf, "output_res_query.weight", device, stream)?);

        let v_half0 = cfg.n_vocab_unpadded / 2;
        let (v_start, v_count) = if rank == 0 {
            (0, v_half0)
        } else {
            (v_half0, cfg.n_vocab_unpadded - v_half0)
        };
        let output = Some(DeviceTensor::from_gguf_row_shard(
            gguf,
            "output.weight",
            v_start,
            v_count,
            device,
            stream,
        )?);

        let n_head_l = cfg.n_head / 2;
        let n_head_kv_l = cfg.n_head_kv / 2;
        let q_dim_l = n_head_l * cfg.head_dim;
        let kv_dim_l = n_head_kv_l * cfg.head_dim;
        let rel_dim_l = n_head_l * cfg.d_rel;
        let dyn_dim_l = n_head_kv_l * cfg.shortconv_l_cache;
        let n_ff_l = cfg.n_ff_exp / 2;

        let mut layers = Vec::with_capacity(cfg.n_layer_phys);
        for i in 0..cfg.n_layer_phys {
            let p = |name: &str| format!("blk.{i}.{name}");
            let (is_swa, attn_norm, q_inp, k_inp, v_inp, attn_res_query, exp_probs_b, loop_ctrl) =
                Self::load_layer_common(gguf, cfg, i, device, stream)?;

            let gu_info = gguf.tensor_info(&p("ffn_gate_up_exps.weight"))?;
            let dn_info = gguf.tensor_info(&p("ffn_down_exps.weight"))?;
            let gate_up_row_bytes = gu_info.ggml_type.row_bytes(cfg.n_embd);
            let down_row_bytes = dn_info.ggml_type.row_bytes(n_ff_l);
            let gate_up_expert_bytes = gate_up_row_bytes * (2 * n_ff_l);
            let down_expert_bytes = down_row_bytes * cfg.n_embd;

            let (attn_k_conv_proj, attn_v_conv_proj) = if !is_swa {
                (
                    Some(DeviceTensor::from_gguf_row_shard(
                        gguf,
                        &p("attn_k_conv_proj.weight"),
                        rank * dyn_dim_l,
                        dyn_dim_l,
                        device,
                        stream,
                    )?),
                    Some(DeviceTensor::from_gguf_row_shard(
                        gguf,
                        &p("attn_v_conv_proj.weight"),
                        rank * dyn_dim_l,
                        dyn_dim_l,
                        device,
                        stream,
                    )?),
                )
            } else {
                (None, None)
            };

            let (gate_up_scale, gate_up_scale_host) = if gguf.has_tensor(&p("ffn_gate_up_exps.scale")) {
                (
                    Some(DeviceTensor::from_gguf(gguf, &p("ffn_gate_up_exps.scale"), device, stream)?),
                    gguf.tensor_f32_slice(&p("ffn_gate_up_exps.scale"))?.to_vec(),
                )
            } else {
                (None, vec![1.0f32; cfg.n_expert])
            };
            let (down_scale, down_scale_host) = if gguf.has_tensor(&p("ffn_down_exps.scale")) {
                (
                    Some(DeviceTensor::from_gguf(gguf, &p("ffn_down_exps.scale"), device, stream)?),
                    gguf.tensor_f32_slice(&p("ffn_down_exps.scale"))?.to_vec(),
                )
            } else {
                (None, vec![1.0f32; cfg.n_expert])
            };

            layers.push(Some(LayerWeights {
                is_swa,
                wq: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_q.weight"), rank * q_dim_l, q_dim_l, device, stream)?,
                wk: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_k.weight"), rank * kv_dim_l, kv_dim_l, device, stream)?,
                wv: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_v.weight"), rank * kv_dim_l, kv_dim_l, device, stream)?,
                wo: DeviceTensor::from_gguf_col_shard(
                    gguf,
                    &p("attn_output.weight"),
                    rank * q_dim_l,
                    q_dim_l,
                    cfg.n_head * cfg.head_dim,
                    device,
                    stream,
                )?,
                attn_gate: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_gate.weight"), rank * q_dim_l, q_dim_l, device, stream)?,
                attn_q_clip: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_q_clip.weight"), rank * n_head_l, n_head_l, device, stream)?,
                attn_sink_proj: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_sink_proj.weight"), rank * n_head_l, n_head_l, device, stream)?,
                attn_rel: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_rel.weight"), rank * rel_dim_l, rel_dim_l, device, stream)?,
                attn_rel_bias: DeviceTensor::from_gguf(gguf, &p("attn_rel_bias.weight"), device, stream)?,
                attn_rel_clip: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_rel_clip.weight"), rank * n_head_l, n_head_l, device, stream)?,
                attn_k_conv: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_k_conv.weight"), rank * kv_dim_l, kv_dim_l, device, stream)?,
                attn_v_conv: DeviceTensor::from_gguf_row_shard(gguf, &p("attn_v_conv.weight"), rank * kv_dim_l, kv_dim_l, device, stream)?,
                attn_post_conv: DeviceTensor::from_gguf(gguf, &p("attn_post_conv.weight"), device, stream)?,
                ffn_post_conv: DeviceTensor::from_gguf(gguf, &p("ffn_post_conv.weight"), device, stream)?,
                attn_norm,
                attn_q_inp_norm: q_inp,
                attn_k_inp_norm: k_inp,
                attn_v_inp_norm: v_inp,
                attn_k_conv_proj,
                attn_v_conv_proj,
                attn_res_query,
                ffn_res_query: DeviceTensor::from_gguf(gguf, &p("ffn_res_query.weight"), device, stream)?,
                ffn_norm: DeviceTensor::from_gguf(gguf, &p("ffn_norm.weight"), device, stream)?,
                ffn_gate_inp: DeviceTensor::from_gguf(gguf, &p("ffn_gate_inp.weight"), device, stream)?,
                exp_probs_b,
                gate_up_scale,
                down_scale,
                gate_up_scale_host,
                down_scale_host,
                gate_up_type: gu_info.ggml_type,
                down_type: dn_info.ggml_type,
                gate_up_row_bytes,
                down_row_bytes,
                gate_up_expert_bytes,
                down_expert_bytes,
                loop_ctrl,
            }));
        }
        let loop_msg_queries_all = Self::build_loop_msg_queries_all(cfg, &layers, device, stream)?;
        stream.sync()?;

        Ok(Self {
            device,
            tok_embd,
            tok_embd_norm,
            output,
            output_norm,
            output_res_query,
            loop_msg_queries_all,
            layers,
        })
    }
}
