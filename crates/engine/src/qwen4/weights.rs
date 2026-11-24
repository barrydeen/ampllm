use crate::config::{EngineConfig, MultiGpuMode};
use crate::qwen4::vision::Qwen4Vision;
use ampllm_cache::{align_up_256, CacheBudget, CacheProfile};
use ampllm_cpu_kernels::{bf16_to_q8_0_cpu, CpuThreadPool};
use ampllm_cuda_kernels::{
    copy_d2h_raw_async, copy_h2d_raw_async, device_count, enable_peer_access, mem_info,
    set_device, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer, RegisteredMappedHostBuffer,
};
use ampllm_gguf::{
    discover_qwen4_aux_files, GgmlType, GgufFile, Qwen4Config, Qwen4VisionConfig,
};
use anyhow::{bail, Context, Result};
use half::bf16;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct QuantMatrixSpec {
    pub ptr: u64,
    pub ggml_type: GgmlType,
    pub n_out: usize,
    pub n_in: usize,
    pub row_bytes: usize,
}

pub struct QuantMatrix {
    pub buf: DeviceBuffer,
    pub ggml_type: GgmlType,
    pub n_out: usize,
    pub n_in: usize,
    pub row_bytes: usize,
}

impl QuantMatrix {
    #[inline]
    pub fn as_spec(&self) -> QuantMatrixSpec {
        QuantMatrixSpec {
            ptr: self.buf.as_u64(),
            ggml_type: self.ggml_type,
            n_out: self.n_out,
            n_in: self.n_in,
            row_bytes: self.row_bytes,
        }
    }
}

fn validate_qwen4_moe_type(role: &str, il: usize, ty: GgmlType) -> Result<()> {
    let supported = match role {
        "ffn_gate/up_exps.weight" => matches!(
            ty,
            GgmlType::Q2_0
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8_0
                | GgmlType::Iq4Nl
                | GgmlType::Iq4Xs
                | GgmlType::Iq3S
                | GgmlType::Iq3Xxs
                | GgmlType::Iq2S
                | GgmlType::Iq2Xs
                | GgmlType::Iq2Xxs
                | GgmlType::Iq1S
        ),
        _ => matches!(
            ty,
            GgmlType::Q2_0
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8_0
                | GgmlType::Iq4Nl
                | GgmlType::Iq4Xs
                | GgmlType::Iq3S
                | GgmlType::Iq3Xxs
                | GgmlType::Iq2S
        ),
    };
    if !supported {
        anyhow::bail!(
            "blk.{il}.{role}: unsupported GgmlType {ty:?} ({}) for Qwen4 MoE kernels",
            ty as i32
        );
    }
    Ok(())
}

pub struct HcWeights {
    pub norm: DeviceBuffer,
    pub down: QuantMatrix,
    pub up: QuantMatrix,
    pub inject: Option<DeviceBuffer>,
    pub inject_type: GgmlType,
}

pub struct PleLayerWeights {
    pub key: QuantMatrix,
    pub value: QuantMatrix,
    pub norm_key: DeviceBuffer,
    pub norm_query: DeviceBuffer,
    pub norm_conv: DeviceBuffer,
    pub conv1d: DeviceBuffer,
}

pub struct GdnLayerWeights {
    pub attn_qkv: QuantMatrix,
    pub attn_gate: QuantMatrix,
    pub ssm_out: QuantMatrix,
    pub ssm_alpha: DeviceBuffer,
    pub ssm_beta: DeviceBuffer,
    pub ssm_dt_bias: DeviceBuffer,
    pub ssm_a: DeviceBuffer,
    pub ssm_conv1d: DeviceBuffer,
    pub ssm_norm: DeviceBuffer,
}

pub struct QsaLayerWeights {
    pub attn_q: QuantMatrix,
    pub attn_k: QuantMatrix,
    pub attn_v: QuantMatrix,
    pub attn_output: QuantMatrix,
    pub attn_q_norm: DeviceBuffer,
    pub attn_k_norm: DeviceBuffer,
    pub idx_q_proj_bf16: DeviceBuffer,
    pub idx_k_proj_bf16: DeviceBuffer,
    pub idx_q_norm: DeviceBuffer,
    pub idx_k_norm: DeviceBuffer,
}

pub enum AttnLayerWeights {
    Gdn(GdnLayerWeights),
    Qsa(QsaLayerWeights),
}

pub struct LayerMoeDenseWeights {
    pub gate_inp: DeviceBuffer,
    pub gate_inp_shexp: DeviceBuffer,
    pub sh_gate: QuantMatrix,
    pub sh_up: QuantMatrix,
    pub sh_down: QuantMatrix,
    pub gu_type: GgmlType,
    pub d_type: GgmlType,
    pub gu_row_bytes: usize,
    pub d_row_bytes: usize,
    pub gate_bytes: usize,
    pub up_bytes: usize,
    pub down_bytes: usize,
    pub u_off: u32,
    pub d_off: u32,
    pub blob_bytes: usize,
}

pub struct LayerExpertsStorage {
    pub n_resident: usize,
    pub slot_of: Vec<i32>,
    pub expert_of_slot: Vec<usize>,
    pub is_resident: Vec<bool>,
    pub vram_slab: Option<DeviceBuffer>,
    pub vram_ptrs_host: Vec<u64>,
    pub ram_host_ptrs: Vec<usize>,
    pub ram_dev_ptrs: Vec<u64>,
    pub dev_vram_ptrs: DeviceBuffer,
    pub dev_ram_ptrs: DeviceBuffer,
    pub heat: Vec<f32>,
    pub last_used: Vec<u64>,
    pub all_pinned_cold: bool,
    pub decode_usage: Vec<f32>,
    pub prompt_hits: Vec<u32>,
    pub prompt_tail_hits: Vec<u32>,
    pub profile_rank: Vec<u16>,
    pub free_ram_slots: Vec<(usize, u64)>,
}

pub struct Qwen4LayerWeights {
    pub device: i32,
    pub hc_attn: HcWeights,
    pub hc_ffn: HcWeights,
    pub ple: Option<PleLayerWeights>,
    pub attn: AttnLayerWeights,
    pub moe: LayerMoeDenseWeights,
}

pub struct MtpWeights {
    pub device: i32,
    pub pre_fc_norm_embedding: DeviceBuffer,
    pub pre_fc_norm_hidden: DeviceBuffer,
    pub fc_embedding: QuantMatrix,
    pub fc_hidden: QuantMatrix,
    pub hc_attn: HcWeights,
    pub hc_mlp: HcWeights,
    pub hc_mixer: HcWeights,
    pub q_proj: QuantMatrix,
    pub k_proj: QuantMatrix,
    pub v_proj: QuantMatrix,
    pub o_proj: QuantMatrix,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub router_gate_f32: DeviceBuffer,
    pub sh_gate_scalar_f32: DeviceBuffer,
    pub sh_gate: QuantMatrix,
    pub sh_up: QuantMatrix,
    pub sh_down: QuantMatrix,
    pub gu_type: GgmlType,
    pub d_type: GgmlType,
    pub gu_row_bytes: usize,
    pub d_row_bytes: usize,
    pub u_off: u32,
    pub d_off: u32,
    pub blob_bytes: usize,
    pub vram_slab: DeviceBuffer,
    pub dev_gate_ptrs: DeviceBuffer,
    pub dev_up_ptrs: DeviceBuffer,
    pub dev_down_ptrs: DeviceBuffer,
    pub draft_vocab: Option<DeviceBuffer>,
    pub draft_vocab_len: usize,
    pub draft_vocab_ids: Vec<i32>,
    pub draft_head_q4: Option<QuantMatrix>,
}

pub enum ColdHostBacking {
    Pinned(RegisteredMappedHostBuffer),
    Pageable(Vec<u8>),
    Empty,
}

pub struct Qwen4Weights {
    pub gguf: GgufFile,
    pub cfg: Qwen4Config,
    pub gpu_ids: Vec<i32>,
    pub split_layer: usize,
    pub emb_shard_idx: usize,
    pub emb_offset: usize,
    pub emb_byte_len: usize,
    pub token_embd_type: GgmlType,
    pub token_embd_row_bytes: usize,
    pub ple_shard_idx: Option<usize>,
    pub ple_offset: usize,
    pub ple_byte_len: usize,
    pub layers: Vec<Qwen4LayerWeights>,
    pub experts: Vec<LayerExpertsStorage>,
    pub cold_backings: Vec<ColdHostBacking>,
    pub layers_host_atomic: Arc<Vec<Vec<AtomicUsize>>>,
    pub layer_u_off: Arc<Vec<usize>>,
    pub layer_d_off: Arc<Vec<usize>>,
    pub max_blob_bytes: usize,
    pub output_hc: HcWeights,
    pub output_head: QuantMatrix,
    pub mtp: Option<MtpWeights>,
    pub vision: Option<Qwen4Vision>,
    pub has_profile: bool,
}

fn upload_quant_matrix(
    dev: i32,
    gguf: &GgufFile,
    name: &str,
    exp_in: usize,
    exp_out: usize,
) -> Result<QuantMatrix> {
    let info = gguf.tensor_info(name)?;
    let ne0 = info.dims.first().copied().unwrap_or(0) as usize;
    let ne1 = info.dims.get(1).copied().unwrap_or(1) as usize;
    if ne0 != exp_in || ne1 != exp_out {
        bail!(
            "tensor {name} dims [{ne0}, {ne1}] != expected [{exp_in}, {exp_out}]"
        );
    }
    let rb = info.ggml_type.row_bytes(ne0);
    let raw = gguf.tensor_bytes(name)?;
    let buf = DeviceBuffer::alloc(dev, raw.len())?;
    buf.copy_from_host(raw)?;
    Ok(QuantMatrix {
        buf,
        ggml_type: info.ggml_type,
        n_out: ne1,
        n_in: ne0,
        row_bytes: rb,
    })
}

fn upload_f32_tensor(
    dev: i32,
    gguf: &GgufFile,
    name: &str,
    add_one: bool,
) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    let n = info.n_elements();
    let vals: Vec<f32> = match info.ggml_type {
        GgmlType::F32 => {
            let s: &[f32] = bytemuck::cast_slice(raw);
            if add_one {
                s.iter().map(|&v| v + 1.0f32).collect()
            } else {
                let buf = DeviceBuffer::alloc(dev, raw.len())?;
                buf.copy_from_host(raw)?;
                return Ok(buf);
            }
        }
        GgmlType::Bf16 => {
            let s: &[u16] = bytemuck::cast_slice(raw);
            let add = if add_one { 1.0f32 } else { 0.0f32 };
            s.iter().map(|&b| bf16::from_bits(b).to_f32() + add).collect()
        }
        _ => {
            let mut out = vec![0.0f32; n];
            ampllm_gguf::dequantize_row_to_f32(info.ggml_type, raw, &mut out);
            if add_one {
                for v in &mut out {
                    *v += 1.0f32;
                }
            }
            out
        }
    };
    let buf = DeviceBuffer::alloc(dev, vals.len() * 4)?;
    buf.copy_from_host(bytemuck::cast_slice(&vals))?;
    Ok(buf)
}

fn upload_bf16_tensor(dev: i32, gguf: &GgufFile, name: &str) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    if info.ggml_type == GgmlType::Bf16 {
        let buf = DeviceBuffer::alloc(dev, raw.len())?;
        buf.copy_from_host(raw)?;
        Ok(buf)
    } else {
        let vals = if info.ggml_type == GgmlType::F32 {
            bytemuck::cast_slice::<u8, f32>(raw).to_vec()
        } else {
            let mut out = vec![0.0f32; info.n_elements()];
            ampllm_gguf::dequantize_row_to_f32(info.ggml_type, raw, &mut out);
            out
        };
        let b16: Vec<u16> = vals.into_iter().map(|v| bf16::from_f32(v).to_bits()).collect();
        let buf = DeviceBuffer::alloc(dev, b16.len() * 2)?;
        buf.copy_from_host(bytemuck::cast_slice(&b16))?;
        Ok(buf)
    }
}

fn upload_bf16_as_q8_0(
    dev: i32,
    gguf: &GgufFile,
    name: &str,
    exp_in: usize,
    exp_out: usize,
) -> Result<QuantMatrix> {
    let info = gguf.tensor_info(name)?;
    let ne0 = info.dims.first().copied().unwrap_or(0) as usize;
    let ne1 = info.dims.get(1).copied().unwrap_or(1) as usize;
    if ne0 != exp_in || ne1 != exp_out {
        bail!(
            "tensor {name} dims [{ne0}, {ne1}] != expected [{exp_in}, {exp_out}]"
        );
    }
    let rb = GgmlType::Q8_0.row_bytes(ne0);
    let raw = gguf.tensor_bytes(name)?;
    let q8_bytes = match info.ggml_type {
        GgmlType::Q8_0 => {
            let buf = DeviceBuffer::alloc(dev, raw.len())?;
            buf.copy_from_host(raw)?;
            return Ok(QuantMatrix {
                buf,
                ggml_type: GgmlType::Q8_0,
                n_out: ne1,
                n_in: ne0,
                row_bytes: rb,
            });
        }
        GgmlType::Bf16 => {
            let b16: &[u16] = bytemuck::cast_slice(raw);
            let mut out = vec![0u8; ne1 * rb];
            bf16_to_q8_0_cpu(b16, ne1, ne0, &mut out);
            out
        }
        GgmlType::F32 => {
            let f: &[f32] = bytemuck::cast_slice(raw);
            let b16: Vec<u16> = f.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
            let mut out = vec![0u8; ne1 * rb];
            bf16_to_q8_0_cpu(&b16, ne1, ne0, &mut out);
            out
        }
        other => bail!("unsupported MTP tensor type {other:?} for {name}"),
    };
    let buf = DeviceBuffer::alloc(dev, q8_bytes.len())?;
    buf.copy_from_host(&q8_bytes)?;
    Ok(QuantMatrix {
        buf,
        ggml_type: GgmlType::Q8_0,
        n_out: ne1,
        n_in: ne0,
        row_bytes: rb,
    })
}

fn load_hc_main(
    dev: i32,
    gguf: &GgufFile,
    prefix: &str,
    hc: usize,
    n_embd: usize,
    hc_lr: usize,
    with_inject: bool,
) -> Result<HcWeights> {
    let hc_dim = hc * n_embd;
    let norm = upload_f32_tensor(dev, gguf, &format!("{prefix}_norm.weight"), false)?;
    let down = upload_quant_matrix(dev, gguf, &format!("{prefix}_down.weight"), hc_dim, hc_lr)?;
    let up = upload_quant_matrix(dev, gguf, &format!("{prefix}_up.weight"), hc_lr, hc_dim)?;
    let inject = if with_inject {
        Some(upload_f32_tensor(
            dev,
            gguf,
            &format!("{prefix}_inject.weight"),
            false,
        )?)
    } else {
        None
    };
    Ok(HcWeights {
        norm,
        down,
        up,
        inject,
        inject_type: GgmlType::F32,
    })
}

fn load_hc_mtp(
    dev: i32,
    gguf: &GgufFile,
    prefix: &str,
    hc: usize,
    n_embd: usize,
    hc_lr: usize,
    with_inject: bool,
) -> Result<HcWeights> {
    let hc_dim = hc * n_embd;
    // MTP GGUF F32 norm weights lack +1.0 (Gemma-style RMSNorm offset).
    let norm = upload_f32_tensor(dev, gguf, &format!("{prefix}.hc_norm.weight"), true)?;
    let down = upload_bf16_as_q8_0(
        dev,
        gguf,
        &format!("{prefix}.input_mix_weight_down.weight"),
        hc_dim,
        hc_lr,
    )?;
    let up = upload_bf16_as_q8_0(
        dev,
        gguf,
        &format!("{prefix}.input_mix_weight_up.weight"),
        hc_lr,
        hc_dim,
    )?;
    let inject = if with_inject {
        Some(upload_f32_tensor(
            dev,
            gguf,
            &format!("{prefix}.block_inject_weight.weight"),
            false,
        )?)
    } else {
        None
    };
    Ok(HcWeights {
        norm,
        down,
        up,
        inject,
        inject_type: GgmlType::F32,
    })
}

fn read_mem_available_bytes() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb = rest.split_whitespace().next()?.parse::<usize>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn load_mtp(mtp_gguf: &GgufFile, cfg: &Qwen4Config, dev: i32) -> Result<MtpWeights> {
    set_device(dev)?;
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let hc_dim = hc * n_embd;
    let hc_lr = cfg.hc_low_rank;
    let kv_dim = cfg.n_head_kv * cfg.head_dim;
    let q_full_dim = cfg.n_head * cfg.head_dim * 2;
    let q_dim = cfg.n_head * cfg.head_dim;
    let n_ff = cfg.n_ff_exp;
    let n_ff_sh = cfg.n_ff_shared;
    let n_exp = cfg.n_expert;

    let pre_fc_norm_embedding =
        upload_f32_tensor(dev, mtp_gguf, "mtp.pre_fc_norm_embedding.weight", true)?;
    let pre_fc_norm_hidden =
        upload_f32_tensor(dev, mtp_gguf, "mtp.pre_fc_norm_hidden.weight", true)?;
    let fc_embedding = upload_bf16_as_q8_0(dev, mtp_gguf, "mtp.fc_embedding.weight", n_embd, n_embd)?;
    let fc_hidden = upload_bf16_as_q8_0(dev, mtp_gguf, "mtp.fc_hidden.weight", n_embd, n_embd)?;

    let hc_attn = load_hc_mtp(
        dev,
        mtp_gguf,
        "mtp.layers.0.attn_hyper_connection",
        hc,
        n_embd,
        hc_lr,
        true,
    )?;
    let hc_mlp = load_hc_mtp(
        dev,
        mtp_gguf,
        "mtp.layers.0.mlp_hyper_connection",
        hc,
        n_embd,
        hc_lr,
        true,
    )?;
    let hc_mixer = load_hc_mtp(dev, mtp_gguf, "mtp.hyper_connection_mixer", hc, n_embd, hc_lr, false)?;

    let q_proj = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.self_attn.q_proj.weight",
        n_embd,
        q_full_dim,
    )?;
    let k_proj = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.self_attn.k_proj.weight",
        n_embd,
        kv_dim,
    )?;
    let v_proj = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.self_attn.v_proj.weight",
        n_embd,
        kv_dim,
    )?;
    let o_proj = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.self_attn.o_proj.weight",
        q_dim,
        n_embd,
    )?;
    let q_norm = upload_f32_tensor(dev, mtp_gguf, "mtp.layers.0.self_attn.q_norm.weight", true)?;
    let k_norm = upload_f32_tensor(dev, mtp_gguf, "mtp.layers.0.self_attn.k_norm.weight", true)?;

    let router_gate_f32 = upload_f32_tensor(dev, mtp_gguf, "mtp.layers.0.mlp.gate.weight", false)?;
    let sh_gate_scalar_f32 =
        upload_f32_tensor(dev, mtp_gguf, "mtp.layers.0.mlp.shared_expert_gate.weight", false)?;
    let sh_gate = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        n_embd,
        n_ff_sh,
    )?;
    let sh_up = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        n_embd,
        n_ff_sh,
    )?;
    let sh_down = upload_bf16_as_q8_0(
        dev,
        mtp_gguf,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        n_ff_sh,
        n_embd,
    )?;

    // Pack all 512 MTP Q2_0 routed experts into one resident VRAM slab on GPU `dev`.
    let gu_info = mtp_gguf.tensor_info("mtp.layers.0.mlp.experts.gate_up_proj")?;
    let dn_info = mtp_gguf.tensor_info("mtp.layers.0.mlp.experts.down_proj")?;
    let gu_type = gu_info.ggml_type;
    let d_type = dn_info.ggml_type;
    let gu_rb = gu_type.row_bytes(n_embd);
    let d_rb = d_type.row_bytes(n_ff);
    let gate_bytes = n_ff * gu_rb;
    let up_bytes = n_ff * gu_rb;
    let down_bytes = n_embd * d_rb;
    let u_off = align_up_256(gate_bytes);
    let d_off = u_off + align_up_256(up_bytes);
    let blob_bytes = d_off + align_up_256(down_bytes);

    let gu_raw = mtp_gguf.tensor_bytes("mtp.layers.0.mlp.experts.gate_up_proj")?;
    let dn_raw = mtp_gguf.tensor_bytes("mtp.layers.0.mlp.experts.down_proj")?;
    let gu_stride = 2 * n_ff * gu_rb;
    let mut host_slab = vec![0u8; n_exp * blob_bytes];
    for e in 0..n_exp {
        let src_gu = &gu_raw[e * gu_stride..(e + 1) * gu_stride];
        let src_dn = &dn_raw[e * down_bytes..(e + 1) * down_bytes];
        let dst = &mut host_slab[e * blob_bytes..(e + 1) * blob_bytes];
        dst[0..gate_bytes].copy_from_slice(&src_gu[0..gate_bytes]);
        dst[u_off..u_off + up_bytes].copy_from_slice(&src_gu[gate_bytes..2 * gate_bytes]);
        dst[d_off..d_off + down_bytes].copy_from_slice(src_dn);
    }
    let vram_slab = DeviceBuffer::alloc(dev, host_slab.len())?;
    vram_slab.copy_from_host(&host_slab)?;

    let base = vram_slab.as_u64();
    let gate_ptrs_h: Vec<u64> = (0..n_exp).map(|e| base + (e * blob_bytes) as u64).collect();
    let up_ptrs_h: Vec<u64> = gate_ptrs_h.iter().map(|&p| p + u_off as u64).collect();
    let down_ptrs_h: Vec<u64> = gate_ptrs_h.iter().map(|&p| p + d_off as u64).collect();

    let dev_gate_ptrs = DeviceBuffer::alloc(dev, gate_ptrs_h.len() * 8)?;
    let dev_up_ptrs = DeviceBuffer::alloc(dev, up_ptrs_h.len() * 8)?;
    let dev_down_ptrs = DeviceBuffer::alloc(dev, down_ptrs_h.len() * 8)?;
    dev_gate_ptrs.copy_from_host(bytemuck::cast_slice(&gate_ptrs_h))?;
    dev_up_ptrs.copy_from_host(bytemuck::cast_slice(&up_ptrs_h))?;
    dev_down_ptrs.copy_from_host(bytemuck::cast_slice(&down_ptrs_h))?;

    let _ = hc_dim;
    Ok(MtpWeights {
        device: dev,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc_embedding,
        fc_hidden,
        hc_attn,
        hc_mlp,
        hc_mixer,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        router_gate_f32,
        sh_gate_scalar_f32,
        sh_gate,
        sh_up,
        sh_down,
        gu_type,
        d_type,
        gu_row_bytes: gu_rb,
        d_row_bytes: d_rb,
        u_off: u_off as u32,
        d_off: d_off as u32,
        blob_bytes,
        vram_slab,
        dev_gate_ptrs,
        dev_up_ptrs,
        dev_down_ptrs,
        draft_vocab: None,
        draft_vocab_len: 0,
        draft_vocab_ids: Vec::new(),
        draft_head_q4: None,
    })
}

fn is_non_latin_draft_exclude(c: char) -> bool {
    let cp = u32::from(c);
    matches!(
        cp,
        0x0400..=0x1CFF
            | 0x2E80..=0x33FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA000..=0xD7FF
            | 0xF900..=0xFAFF
            | 0xFB50..=0xFDFF
            | 0xFE30..=0xFE4F
            | 0xFE70..=0xFFEF
            | 0x20000..=0x2FA1F
    )
}

fn build_draft_vocab_subset(
    gguf: &GgufFile,
    n_vocab: usize,
    dev: i32,
) -> Result<Option<(DeviceBuffer, Vec<i32>)>> {
    if let Ok(val) = std::env::var("AMPLLM_QWEN4_DRAFT_VOCAB") {
        let trimmed = val.trim();
        if trimmed == "full" || trimmed == "0" {
            return Ok(None);
        }
        let path = std::path::Path::new(trimmed);
        if !trimmed.is_empty() && path.exists() {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading AMPLLM_QWEN4_DRAFT_VOCAB {}", path.display()))?;
            let mut ids = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let id = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if id >= 0 && (id as usize) < n_vocab {
                    ids.push(id);
                }
            }
            if ids.len() >= 1024 && ids.len() < n_vocab {
                set_device(dev)?;
                let buf = DeviceBuffer::alloc(dev, ids.len() * 4)?;
                buf.copy_from_host(bytemuck::cast_slice(&ids))?;
                eprintln!(
                    "qwen4exp mtp: loaded draft vocab = {}/{} from {}",
                    ids.len(),
                    n_vocab,
                    path.display()
                );
                return Ok(Some((buf, ids)));
            }
            return Ok(None);
        }
    }

    let Some(tokens_arr) = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
    else {
        return Ok(None);
    };
    let types_arr = gguf
        .metadata
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_array());

    let (_, u2b) = ampllm_tokenizer::build_gpt2_byte_tables();
    let count = n_vocab.min(tokens_arr.len());
    let mut ids = Vec::with_capacity(count);
    let mut raw = Vec::with_capacity(64);

    for id in 0..count {
        if id < 256 {
            ids.push(id as i32);
            continue;
        }
        let ttype = types_arr
            .and_then(|a| a.get(id))
            .and_then(|v| v.as_i64())
            .unwrap_or(1);
        let tok_str = tokens_arr[id].as_str().unwrap_or("");
        if ttype != 1 || (tok_str.starts_with("<|") && tok_str.ends_with("|>")) {
            ids.push(id as i32);
            continue;
        }
        if id >= 100_000 {
            continue;
        }
        raw.clear();
        for ch in tok_str.chars() {
            if let Some(&b) = u2b.get(&ch) {
                raw.push(b);
            } else {
                let mut utf8 = [0u8; 4];
                raw.extend_from_slice(ch.encode_utf8(&mut utf8).as_bytes());
            }
        }
        match std::str::from_utf8(&raw) {
            Ok(decoded) => {
                if decoded.chars().any(is_non_latin_draft_exclude) {
                    continue;
                }
            }
            Err(_) => continue,
        }
        ids.push(id as i32);
    }

    if ids.len() >= 1024 && ids.len() < n_vocab {
        set_device(dev)?;
        let buf = DeviceBuffer::alloc(dev, ids.len() * 4)?;
        buf.copy_from_host(bytemuck::cast_slice(&ids))?;
        eprintln!("qwen4exp mtp: auto draft vocab = {}/{}", ids.len(), n_vocab);
        Ok(Some((buf, ids)))
    } else {
        Ok(None)
    }
}

impl Qwen4Weights {
    pub fn load_dense(
        gguf: GgufFile,
        cfg: Qwen4Config,
        engine_cfg: &EngineConfig,
    ) -> Result<Self> {
        let n_dev = device_count()?.max(1);
        let use_pipeline = matches!(engine_cfg.multi_gpu, MultiGpuMode::Pipeline) && n_dev >= 2;
        let gpu_ids = if use_pipeline {
            let _ = enable_peer_access(0, 1);
            let _ = enable_peer_access(1, 0);
            vec![0, 1]
        } else {
            vec![engine_cfg.device.clamp(0, n_dev - 1)]
        };

        let split_layer = if gpu_ids.len() >= 2 {
            let default_split = if cfg.n_layer >= 48 { 25 } else { cfg.n_layer / 2 };
            std::env::var("AMPLLM_QWEN4_SPLIT")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(default_split)
                .clamp(1, cfg.n_layer - 1)
        } else {
            cfg.n_layer
        };

        let tok_info = gguf.tensor_info("token_embd.weight")?;
        let emb_shard_idx = tok_info.shard_idx;
        let emb_offset = tok_info.data_offset;
        let emb_byte_len = tok_info.byte_len;
        let token_embd_type = tok_info.ggml_type;
        let token_embd_row_bytes = token_embd_type.row_bytes(cfg.n_embd);

        let (ple_shard_idx, ple_offset, ple_byte_len) = if gguf.has_tensor("per_layer_token_embd.weight")
        {
            let pi = gguf.tensor_info("per_layer_token_embd.weight")?;
            gguf.advise_shard_random(pi.shard_idx);
            (Some(pi.shard_idx), pi.data_offset, pi.byte_len)
        } else {
            (None, 0, 0)
        };

        let n_embd = cfg.n_embd;
        let hc = cfg.hc;
        let hc_dim = hc * n_embd;
        let hc_lr = cfg.hc_low_rank;
        let gdn_qkv = cfg.gdn_qkv_dim();
        let gdn_z = cfg.ssm_inner_size;
        let gdn_hv = cfg.ssm_time_step_rank;
        let q_full_dim = cfg.n_head * cfg.head_dim * 2;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let n_ff = cfg.n_ff_exp;
        let n_ff_sh = cfg.n_ff_shared;
        let n_exp = cfg.n_expert;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let dev = if il < split_layer { gpu_ids[0] } else { *gpu_ids.last().unwrap() };
            set_device(dev)?;

            let hc_attn = load_hc_main(
                dev,
                &gguf,
                &format!("blk.{il}.hc_attn"),
                hc,
                n_embd,
                hc_lr,
                true,
            )?;
            let hc_ffn = load_hc_main(
                dev,
                &gguf,
                &format!("blk.{il}.hc_ffn"),
                hc,
                n_embd,
                hc_lr,
                true,
            )?;

            let ple = if cfg.ple.as_ref().is_some_and(|p| p.layers.contains(&il)) {
                Some(PleLayerWeights {
                    key: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.ple_key.weight"), n_embd, hc_dim)?,
                    value: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.ple_value.weight"), n_embd, n_embd)?,
                    norm_key: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ple_norm_key.weight"), false)?,
                    norm_query: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ple_norm_query.weight"), false)?,
                    norm_conv: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ple_norm_conv.weight"), false)?,
                    conv1d: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ple_conv1d.weight"), false)?,
                })
            } else {
                None
            };

            let attn = if cfg.is_qsa(il) {
                AttnLayerWeights::Qsa(QsaLayerWeights {
                    attn_q: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.attn_q.weight"), n_embd, q_full_dim)?,
                    attn_k: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.attn_k.weight"), n_embd, kv_dim)?,
                    attn_v: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.attn_v.weight"), n_embd, kv_dim)?,
                    attn_output: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.attn_output.weight"), q_dim, n_embd)?,
                    attn_q_norm: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.attn_q_norm.weight"), false)?,
                    attn_k_norm: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.attn_k_norm.weight"), false)?,
                    idx_q_proj_bf16: upload_bf16_tensor(dev, &gguf, &format!("blk.{il}.indexer.q_proj.weight"))?,
                    idx_k_proj_bf16: upload_bf16_tensor(dev, &gguf, &format!("blk.{il}.indexer.k_proj.weight"))?,
                    idx_q_norm: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.indexer.q_norm.weight"), false)?,
                    idx_k_norm: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.indexer.k_norm.weight"), false)?,
                })
            } else {
                AttnLayerWeights::Gdn(GdnLayerWeights {
                    attn_qkv: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.attn_qkv.weight"), n_embd, gdn_qkv)?,
                    attn_gate: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.attn_gate.weight"), n_embd, gdn_z)?,
                    ssm_out: upload_quant_matrix(dev, &gguf, &format!("blk.{il}.ssm_out.weight"), gdn_z, n_embd)?,
                    ssm_alpha: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ssm_alpha.weight"), false)?,
                    ssm_beta: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ssm_beta.weight"), false)?,
                    ssm_dt_bias: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ssm_dt.bias"), false)?,
                    ssm_a: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ssm_a"), false)?,
                    ssm_conv1d: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ssm_conv1d.weight"), false)?,
                    ssm_norm: upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ssm_norm.weight"), false)?,
                })
            };

            let gate_inp = upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ffn_gate_inp.weight"), false)?;
            let gate_inp_shexp =
                upload_f32_tensor(dev, &gguf, &format!("blk.{il}.ffn_gate_inp_shexp.weight"), false)?;
            let sh_gate =
                upload_quant_matrix(dev, &gguf, &format!("blk.{il}.ffn_gate_shexp.weight"), n_embd, n_ff_sh)?;
            let sh_up =
                upload_quant_matrix(dev, &gguf, &format!("blk.{il}.ffn_up_shexp.weight"), n_embd, n_ff_sh)?;
            let sh_down =
                upload_quant_matrix(dev, &gguf, &format!("blk.{il}.ffn_down_shexp.weight"), n_ff_sh, n_embd)?;

            let g_info = gguf.tensor_info(&format!("blk.{il}.ffn_gate_exps.weight"))?;
            let u_info = gguf.tensor_info(&format!("blk.{il}.ffn_up_exps.weight"))?;
            let d_info = gguf.tensor_info(&format!("blk.{il}.ffn_down_exps.weight"))?;
            if g_info.ggml_type != u_info.ggml_type {
                bail!(
                    "blk.{il}: gate/up expert type mismatch {:?} vs {:?}",
                    g_info.ggml_type,
                    u_info.ggml_type
                );
            }
            let gu_type = g_info.ggml_type;
            let d_type = d_info.ggml_type;
            validate_qwen4_moe_type("ffn_gate/up_exps.weight", il, gu_type)?;
            validate_qwen4_moe_type("ffn_down_exps.weight", il, d_type)?;
            let gu_rb = gu_type.row_bytes(n_embd);
            let d_rb = d_type.row_bytes(n_ff);
            let gate_bytes = n_ff * gu_rb;
            let up_bytes = n_ff * gu_rb;
            let down_bytes = n_embd * d_rb;
            let u_off = align_up_256(gate_bytes);
            let d_off = u_off + align_up_256(up_bytes);
            let blob_bytes = d_off + align_up_256(down_bytes);

            let _ = gdn_hv;
            layers.push(Qwen4LayerWeights {
                device: dev,
                hc_attn,
                hc_ffn,
                ple,
                attn,
                moe: LayerMoeDenseWeights {
                    gate_inp,
                    gate_inp_shexp,
                    sh_gate,
                    sh_up,
                    sh_down,
                    gu_type,
                    d_type,
                    gu_row_bytes: gu_rb,
                    d_row_bytes: d_rb,
                    gate_bytes,
                    up_bytes,
                    down_bytes,
                    u_off: u_off as u32,
                    d_off: d_off as u32,
                    blob_bytes,
                },
            });
        }

        let head_dev = *gpu_ids.last().unwrap();
        set_device(head_dev)?;
        let output_hc = load_hc_main(head_dev, &gguf, "output_hc", hc, n_embd, hc_lr, false)?;
        let output_head = upload_quant_matrix(head_dev, &gguf, "output.weight", n_embd, cfg.n_vocab)?;

        // Auto-discover optional MTP and mmproj auxiliary GGUF files.
        let model_path = gguf
            .shard_paths
            .first()
            .cloned()
            .unwrap_or_default();
        let explicit_mtp = std::env::var("AMPLLM_QWEN4_MTP")
            .ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from);
        let explicit_mmproj = std::env::var("AMPLLM_QWEN4_MMPROJ")
            .ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from);
        let (mtp_path, mmproj_path) = if engine_cfg.use_spec {
            discover_qwen4_aux_files(&model_path, explicit_mtp.as_deref(), explicit_mmproj.as_deref())?
        } else {
            (
                None,
                discover_qwen4_aux_files(&model_path, None, explicit_mmproj.as_deref())?.1,
            )
        };

        let mut mtp = if engine_cfg.use_spec {
            if let Some(p) = mtp_path {
                let mtp_file = GgufFile::open(&p)
                    .with_context(|| format!("opening Qwen4 MTP GGUF {}", p.display()))?;
                Some(load_mtp(&mtp_file, &cfg, head_dev)?)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(ref mut mw) = mtp {
            if let Some((dv, ids)) = build_draft_vocab_subset(&gguf, cfg.n_vocab, head_dev)? {
                mw.draft_vocab_len = ids.len();
                mw.draft_vocab = Some(dv);
                mw.draft_vocab_ids = ids;
            }
        }

        let vision = if let Some(vp) = mmproj_path {
            let mm_file = GgufFile::open(&vp)
                .with_context(|| format!("opening Qwen4 mmproj GGUF {}", vp.display()))?;
            let vocab_tokens: Vec<String> = gguf
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|x| x.as_str().unwrap_or("").to_owned())
                        .collect()
                })
                .unwrap_or_default();
            let vcfg = Qwen4VisionConfig::from_mmproj(&mm_file, &vocab_tokens, n_embd)?;
            Some(Qwen4Vision::load(&mm_file, vcfg, gpu_ids[0])?)
        } else {
            None
        };

        // Drop non-PLE/non-embedding dense shard pages only after dense loading? Keep all shards for expert packing first.
        let max_blob_bytes = layers
            .iter()
            .map(|l| l.moe.blob_bytes)
            .max()
            .unwrap_or(0);
        let layer_u_off = Arc::new(
            layers
                .iter()
                .map(|l| l.moe.u_off as usize)
                .collect::<Vec<_>>(),
        );
        let layer_d_off = Arc::new(
            layers
                .iter()
                .map(|l| l.moe.d_off as usize)
                .collect::<Vec<_>>(),
        );
        let layers_host_atomic = Arc::new(
            (0..cfg.n_layer)
                .map(|_| (0..n_exp).map(|_| AtomicUsize::new(0)).collect())
                .collect(),
        );
        Ok(Self {
            gguf,
            cfg,
            gpu_ids,
            split_layer,
            emb_shard_idx,
            emb_offset,
            emb_byte_len,
            token_embd_type,
            token_embd_row_bytes,
            ple_shard_idx,
            ple_offset,
            ple_byte_len,
            layers,
            experts: Vec::new(),
            cold_backings: Vec::new(),
            layers_host_atomic,
            layer_u_off,
            layer_d_off,
            max_blob_bytes,
            output_hc,
            output_head,
            mtp,
            vision,
            has_profile: false,
        })
    }

    /// Allocate routed expert VRAM slabs and cold host slabs AFTER GPU state/KV/scratch have been allocated.
    pub fn allocate_experts(
        &mut self,
        engine_cfg: &EngineConfig,
        pool: &Arc<CpuThreadPool>,
    ) -> Result<()> {
        let n_layer = self.cfg.n_layer;
        let n_exp = self.cfg.n_expert;

        let profile: Option<CacheProfile> = if let Some(ref p) = engine_cfg.load_profile {
            if p.exists() {
                Some(CacheProfile::load(p)?)
            } else {
                None
            }
        } else {
            None
        };
        self.has_profile = profile.is_some();

        let mut orders = vec![Vec::new(); n_layer];
        for il in 0..n_layer {
            let mut order = Vec::with_capacity(n_exp);
            let mut seen = vec![false; n_exp];
            if let Some(ref prof) = profile {
                if let Some(ranked) = prof.ranked_experts.get(il) {
                    for &e in ranked {
                        if e < n_exp && !seen[e] {
                            seen[e] = true;
                            order.push(e);
                        }
                    }
                }
            }
            for e in 0..n_exp {
                if !seen[e] {
                    order.push(e);
                }
            }
            orders[il] = order;
        }

        // Compute VRAM expert budget per GPU after all dense + state allocations.
        let mut n_resident = vec![0usize; n_layer];
        let mut total_cold_per_gpu = vec![0usize; self.gpu_ids.len()];
        let gpu_count = self.gpu_ids.len();
        let has_global_ranked = profile
            .as_ref()
            .is_some_and(|p| !p.global_ranked.is_empty());

        const ADAPT_SLOTS_PER_LAYER: usize = 16;

        let build_mtp_q4_head =
            std::env::var("AMPLLM_QWEN4_MTP_Q4_HEAD").map_or(true, |v| v != "0");
        if build_mtp_q4_head {
            let gguf = &self.gguf;
            let n_embd = self.cfg.n_embd;
            if let Some(ref mut mw) = self.mtp {
                if !mw.draft_vocab_ids.is_empty() && mw.draft_head_q4.is_none() {
                    let ids = &mw.draft_vocab_ids;
                    let n_sub = ids.len();
                    let out_info = gguf.tensor_info("output.weight")?;
                    let out_ty = out_info.ggml_type;
                    let out_rb = out_ty.row_bytes(n_embd);
                    let out_raw = gguf.tensor_bytes("output.weight")?;
                    let q4_rb = GgmlType::Q4_0.row_bytes(n_embd);
                    let nb32 = n_embd / 32;
                    let mut q4_host = vec![0u8; n_sub * q4_rb];
                    let out_ptr_u = out_raw.as_ptr() as usize;
                    let ids_ptr_u = ids.as_ptr() as usize;
                    let dst_ptr_u = q4_host.as_mut_ptr() as usize;
                    pool.parallel_for(n_sub, |r0, r1| {
                        let mut frow = vec![0.0f32; n_embd];
                        let out_ptr = out_ptr_u as *const u8;
                        let ids_s = unsafe { std::slice::from_raw_parts(ids_ptr_u as *const i32, n_sub) };
                        let dst_ptr = dst_ptr_u as *mut u8;
                        for r in r0..r1 {
                            let tid = ids_s[r] as usize;
                            let src = unsafe {
                                std::slice::from_raw_parts(out_ptr.add(tid * out_rb), out_rb)
                            };
                            ampllm_gguf::dequantize_row_to_f32(out_ty, src, &mut frow);
                            let dst_row = unsafe {
                                std::slice::from_raw_parts_mut(dst_ptr.add(r * q4_rb), q4_rb)
                            };
                            for b in 0..nb32 {
                                let chunk = &frow[b * 32..(b + 1) * 32];
                                let mut amax = 0.0f32;
                                let mut max_v = 0.0f32;
                                for &v in chunk {
                                    let av = v.abs();
                                    if av > amax {
                                        amax = av;
                                        max_v = v;
                                    }
                                }
                                let d = max_v / -8.0f32;
                                let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
                                let d16 = half::f16::from_f32(d).to_le_bytes();
                                let blk = &mut dst_row[b * 18..(b + 1) * 18];
                                blk[0] = d16[0];
                                blk[1] = d16[1];
                                for j in 0..16 {
                                    let q0 = ((chunk[j] * id + 8.5f32).floor() as i32).clamp(0, 15) as u8;
                                    let q1 = ((chunk[j + 16] * id + 8.5f32).floor() as i32).clamp(0, 15) as u8;
                                    blk[2 + j] = q0 | (q1 << 4);
                                }
                            }
                        }
                    });
                    set_device(mw.device)?;
                    let buf = DeviceBuffer::alloc(mw.device, q4_host.len())?;
                    buf.copy_from_host(&q4_host)?;
                    eprintln!(
                        "qwen4exp mtp: Q4_0 draft head = {} rows ({:.1} MiB)",
                        n_sub,
                        (q4_host.len() as f64) / 1048576.0
                    );
                    mw.draft_head_q4 = Some(QuantMatrix {
                        buf,
                        ggml_type: GgmlType::Q4_0,
                        n_out: n_sub,
                        n_in: n_embd,
                        row_bytes: q4_rb,
                    });
                }
            }
        }

        for (_gidx, &dev) in self.gpu_ids.iter().enumerate() {
            let gidx = _gidx;
            let (free, _) = mem_info(dev)?;
            // Leave headroom for vision lazy scratch, CUDA workspace, and graphs (state is already allocated).
            let safety = if gidx == 0 && self.vision.is_some() {
                672usize * 1024 * 1024
            } else {
                448usize * 1024 * 1024
            };
            let max_usable = free.saturating_sub(safety);
            let budget = match engine_cfg.expert_cache {
                CacheBudget::Auto => max_usable,
                CacheBudget::Mb(mb) => mb
                    .saturating_mul(1024 * 1024)
                    .min(free.saturating_sub(384usize * 1024 * 1024)),
            };
            let dev_layers: Vec<usize> = (0..n_layer)
                .filter(|&il| self.layers[il].device == dev)
                .collect();
            let denom: usize = dev_layers
                .iter()
                .map(|&il| self.layers[il].moe.blob_bytes)
                .sum();
            if denom > 0 {
                if has_global_ranked {
                    let prof = profile.as_ref().unwrap();
                    let mut used_flag = vec![vec![false; n_exp]; n_layer];
                    let mut selected = vec![Vec::new(); n_layer];
                    let mut rem = budget;
                    for &(il, e) in &prof.global_ranked {
                        if il < n_layer
                            && e < n_exp
                            && self.layers[il].device == dev
                            && !used_flag[il][e]
                        {
                            let bb = self.layers[il].moe.blob_bytes;
                            if bb <= rem {
                                rem -= bb;
                                used_flag[il][e] = true;
                                selected[il].push(e);
                            }
                        }
                    }
                    let mut pos = vec![0usize; n_layer];
                    loop {
                        let mut added = false;
                        for &il in &dev_layers {
                            let bb = self.layers[il].moe.blob_bytes;
                            if bb > rem {
                                continue;
                            }
                            while pos[il] < n_exp && used_flag[il][orders[il][pos[il]]] {
                                pos[il] += 1;
                            }
                            if pos[il] < n_exp {
                                let e = orders[il][pos[il]];
                                pos[il] += 1;
                                rem -= bb;
                                used_flag[il][e] = true;
                                selected[il].push(e);
                                added = true;
                            }
                        }
                        if !added {
                            break;
                        }
                    }
                    for &il in &dev_layers {
                        n_resident[il] = selected[il].len();
                        let mut new_ord = Vec::with_capacity(n_exp);
                        new_ord.extend_from_slice(&selected[il]);
                        for &e in &orders[il] {
                            if !used_flag[il][e] {
                                new_ord.push(e);
                            }
                        }
                        orders[il] = new_ord;
                    }
                } else {
                    let mut used = 0usize;
                    let per_layer_base = budget / denom;
                    for &il in &dev_layers {
                        let cap = per_layer_base.min(n_exp);
                        n_resident[il] = cap;
                        used += cap * self.layers[il].moe.blob_bytes;
                    }
                    let mut rem = budget.saturating_sub(used);
                    loop {
                        let mut added = false;
                        for &il in &dev_layers {
                            let bb = self.layers[il].moe.blob_bytes;
                            if n_resident[il] < n_exp && bb <= rem {
                                n_resident[il] += 1;
                                rem -= bb;
                                added = true;
                            }
                        }
                        if !added {
                            break;
                        }
                    }
                }
            }
            for &il in &dev_layers {
                let cold_cnt = n_exp - n_resident[il];
                let extra = if n_resident[il] > 0 && cold_cnt > 0 {
                    ADAPT_SLOTS_PER_LAYER
                } else {
                    0usize
                };
                total_cold_per_gpu[gidx] += (cold_cnt + extra) * self.layers[il].moe.blob_bytes;
            }
        }

        // First allocate VRAM slabs & copy resident experts (so resident VRAM pointers are ready).
        let mut vram_slabs: Vec<Option<DeviceBuffer>> = (0..n_layer).map(|_| None).collect();
        let mut slot_of_all = vec![vec![-1i32; n_exp]; n_layer];
        let mut expert_of_slot_all = vec![Vec::new(); n_layer];
        let mut is_resident_all = vec![vec![false; n_exp]; n_layer];
        let mut vram_ptrs_host_all = vec![vec![0u64; n_exp]; n_layer];
        let mut ram_host_ptrs_all = vec![vec![0usize; n_exp]; n_layer];
        let mut ram_dev_ptrs_all = vec![vec![0u64; n_exp]; n_layer];
        let mut heat_all = vec![vec![0.0f32; n_exp]; n_layer];
        if let Some(ref prof) = profile {
            for il in 0..n_layer {
                if let Some(ranked) = prof.ranked_experts.get(il) {
                    for (rank, &e) in ranked.iter().enumerate() {
                        if e < n_exp {
                            heat_all[il][e] = ((n_exp - rank) as f32) * 0.1;
                        }
                    }
                }
            }
        }

        for il in 0..n_layer {
            let dev = self.layers[il].device;
            set_device(dev)?;
            let n_res = n_resident[il];
            if n_res == 0 {
                continue;
            }
            let mw = &self.layers[il].moe;
            let bb = mw.blob_bytes;
            let g_raw = self.gguf.tensor_bytes(&format!("blk.{il}.ffn_gate_exps.weight"))?;
            let u_raw = self.gguf.tensor_bytes(&format!("blk.{il}.ffn_up_exps.weight"))?;
            let d_raw = self.gguf.tensor_bytes(&format!("blk.{il}.ffn_down_exps.weight"))?;
            let mut stage = vec![0u8; n_res * bb];
            let g_b = mw.gate_bytes;
            let u_b = mw.up_bytes;
            let d_b = mw.down_bytes;
            let u_off = mw.u_off as usize;
            let d_off = mw.d_off as usize;
            let stage_addr = stage.as_mut_ptr() as usize;
            let ord = &orders[il];
            pool.parallel_for(n_res, |start, end| {
                for slot in start..end {
                    let e = ord[slot];
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut((stage_addr as *mut u8).add(slot * bb), bb)
                    };
                    dst[0..g_b].copy_from_slice(&g_raw[e * g_b..(e + 1) * g_b]);
                    dst[u_off..u_off + u_b].copy_from_slice(&u_raw[e * u_b..(e + 1) * u_b]);
                    dst[d_off..d_off + d_b].copy_from_slice(&d_raw[e * d_b..(e + 1) * d_b]);
                }
            });
            let vbuf = DeviceBuffer::alloc(dev, stage.len())?;
            vbuf.copy_from_host(&stage)?;
            let vbase = vbuf.as_u64();
            expert_of_slot_all[il].reserve(n_res);
            for slot in 0..n_res {
                let e = ord[slot];
                slot_of_all[il][e] = slot as i32;
                expert_of_slot_all[il].push(e);
                is_resident_all[il][e] = true;
                vram_ptrs_host_all[il][e] = vbase + (slot * bb) as u64;
            }
            vram_slabs[il] = Some(vbuf);
        }

        // Drop non-PLE/non-embedding shards? No, we still need to read cold experts from them!
        // Pack cold experts per GPU into pinned mapped host RAM when MemAvailable leaves headroom for PLE + OS.
        let total_cold_bytes: usize = total_cold_per_gpu.iter().sum();
        let mem_avail = read_mem_available_bytes().unwrap_or(0);
        let ple_reserve = (self.ple_byte_len / 2)
            .min(8usize * 1024 * 1024 * 1024)
            .saturating_add(4usize * 1024 * 1024 * 1024);
        let pin_cold_env = std::env::var("AMPLLM_QWEN4_PIN_COLD")
            .ok()
            .map(|v| v.trim().to_owned());
        let pin_cold = if total_cold_bytes == 0 {
            false
        } else {
            match pin_cold_env.as_deref() {
                Some("0") => false,
                Some("1") => true,
                _ => mem_avail >= total_cold_bytes.saturating_add(ple_reserve),
            }
        };

        let mut free_ram_slots_all: Vec<Vec<(usize, u64)>> = vec![Vec::new(); n_layer];

        self.cold_backings.clear();
        let mut all_pinned_cold = true;

        for (gidx, &dev) in self.gpu_ids.iter().enumerate() {
            let cold_sz = total_cold_per_gpu[gidx];
            if cold_sz == 0 {
                self.cold_backings.push(ColdHostBacking::Empty);
                continue;
            }
            let (mut backing, mapped_dev_base, host_base): (ColdHostBacking, u64, *mut u8) =
                if pin_cold {
                    match RegisteredMappedHostBuffer::alloc(cold_sz) {
                        Ok(buf) => {
                            let dptr = buf.dev_ptr_for_device(dev)?;
                            let hptr = buf.as_ptr() as *mut u8;
                            (ColdHostBacking::Pinned(buf), dptr, hptr)
                        }
                        Err(_) => {
                            all_pinned_cold = false;
                            let mut v = vec![0u8; cold_sz];
                            let hptr = v.as_mut_ptr();
                            (ColdHostBacking::Pageable(v), 0u64, hptr)
                        }
                    }
                } else {
                    all_pinned_cold = false;
                    let mut v = vec![0u8; cold_sz];
                    let hptr = v.as_mut_ptr();
                    (ColdHostBacking::Pageable(v), 0u64, hptr)
                };

            let mut cur_off = 0usize;
            for il in 0..n_layer {
                if self.layers[il].device != dev {
                    continue;
                }
                let n_res = n_resident[il];
                let cold_cnt = n_exp - n_res;
                if cold_cnt == 0 {
                    continue;
                }
                let mw = &self.layers[il].moe;
                let bb = mw.blob_bytes;
                let g_raw = self.gguf.tensor_bytes(&format!("blk.{il}.ffn_gate_exps.weight"))?;
                let u_raw = self.gguf.tensor_bytes(&format!("blk.{il}.ffn_up_exps.weight"))?;
                let d_raw = self.gguf.tensor_bytes(&format!("blk.{il}.ffn_down_exps.weight"))?;
                let g_b = mw.gate_bytes;
                let u_b = mw.up_bytes;
                let d_b = mw.down_bytes;
                let u_off = mw.u_off as usize;
                let d_off = mw.d_off as usize;
                let dst_layer_addr = unsafe { host_base.add(cur_off) } as usize;
                let ord = &orders[il];

                pool.parallel_for(cold_cnt, |start, end| {
                    for ci in start..end {
                        let e = ord[n_res + ci];
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                (dst_layer_addr as *mut u8).add(ci * bb),
                                bb,
                            )
                        };
                        dst[0..g_b].copy_from_slice(&g_raw[e * g_b..(e + 1) * g_b]);
                        dst[u_off..u_off + u_b].copy_from_slice(&u_raw[e * u_b..(e + 1) * u_b]);
                        dst[d_off..d_off + d_b].copy_from_slice(&d_raw[e * d_b..(e + 1) * d_b]);
                    }
                });

                for ci in 0..cold_cnt {
                    let e = ord[n_res + ci];
                    let off = cur_off + ci * bb;
                    let hptr = (host_base as usize) + off;
                    ram_host_ptrs_all[il][e] = hptr;
                    ram_dev_ptrs_all[il][e] = if mapped_dev_base != 0 {
                        mapped_dev_base + (off as u64)
                    } else {
                        0
                    };
                    self.layers_host_atomic[il][e].store(hptr, Ordering::Relaxed);
                }
                cur_off += cold_cnt * bb;
                if n_res > 0 {
                    if mapped_dev_base != 0 {
                        for s in 0..ADAPT_SLOTS_PER_LAYER {
                            let off = cur_off + s * bb;
                            free_ram_slots_all[il].push((
                                (host_base as usize) + off,
                                mapped_dev_base + (off as u64),
                            ));
                        }
                    }
                    cur_off += ADAPT_SLOTS_PER_LAYER * bb;
                }
            }
            let _ = &mut backing;
            self.cold_backings.push(backing);
        }

        // Drop dead page-cache ranges across ALL shards (preserving only token_embd and/or PLE ranges)
        // so the OS keeps the 28.8 GiB PLE table + 0.67 GiB token_embd warm in RAM without thrashing.
        for s_idx in 0..self.gguf.shard_paths.len() {
            let mut keep_start: Option<usize> = None;
            let mut keep_end: usize = 0;
            let mut add_range = |off: usize, len: usize| {
                if len > 0 {
                    keep_start = Some(keep_start.map_or(off, |cur| cur.min(off)));
                    keep_end = keep_end.max(off + len);
                }
            };
            if s_idx == self.emb_shard_idx {
                add_range(self.emb_offset, self.emb_byte_len);
            }
            if Some(s_idx) == self.ple_shard_idx {
                add_range(self.ple_offset, self.ple_byte_len);
            }
            let (ks, kl) = match keep_start {
                Some(st) => (st, keep_end.saturating_sub(st)),
                None => (0, 0),
            };
            self.gguf.advise_shard_dontneed_except(s_idx, ks, kl);
        }

        let touch_slice = |slice: &[u8]| {
            let addr = slice.as_ptr() as usize;
            let len = slice.len();
            if len == 0 {
                return;
            }
            let n_pages = (len + 4095) / 4096;
            let fold = AtomicUsize::new(0);
            pool.parallel_for(n_pages, |start, end| {
                let base = addr as *const u8;
                let mut acc = 0usize;
                for p in start..end {
                    let off = (p * 4096).min(len - 1);
                    let b = unsafe { std::ptr::read_volatile(base.add(off)) };
                    acc ^= b as usize;
                }
                fold.fetch_xor(acc, Ordering::Relaxed);
            });
            let _ = unsafe { std::ptr::read_volatile((addr as *const u8).add(len - 1)) };
            let _ = fold.load(Ordering::Relaxed);
        };

        let t_pf0 = std::time::Instant::now();
        self.gguf.advise_shard_normal(self.emb_shard_idx);
        self.gguf
            .advise_range_willneed(self.emb_shard_idx, self.emb_offset, self.emb_byte_len);
        if let Ok(emb_slice) = self.token_embd_bytes() {
            touch_slice(emb_slice);
        }

        // Pre-fault PLE table into page cache when headroom >= 2 GiB (eliminates cold page faults).
        let mem_avail_after = read_mem_available_bytes().unwrap_or(0);
        let mut ple_prefaulted = false;
        if std::env::var("AMPLLM_QWEN4_PREFAULT_PLE").map_or(true, |v| v != "0") {
            if let Some(ple_s) = self.ple_shard_idx {
                if self.ple_byte_len > 0
                    && mem_avail_after >= self.ple_byte_len.saturating_add(2usize * 1024 * 1024 * 1024)
                {
                    if ple_s != self.emb_shard_idx {
                        self.gguf.advise_shard_normal(ple_s);
                    }
                    const CHUNK: usize = 256 * 1024 * 1024;
                    let mut off = 0usize;
                    while off < self.ple_byte_len {
                        let clen = (self.ple_byte_len - off).min(CHUNK);
                        self.gguf
                            .advise_range_willneed(ple_s, self.ple_offset + off, clen);
                        off += clen;
                    }
                    touch_slice(self.ple_table_bytes());
                    ple_prefaulted = true;
                }
            }
        }
        self.gguf.advise_shard_random(self.emb_shard_idx);
        if let Some(ple_s) = self.ple_shard_idx {
            if ple_s != self.emb_shard_idx {
                self.gguf.advise_shard_random(ple_s);
            }
        }
        let pf_ms = t_pf0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "qwen4exp host: cpu_isa={} mem_avail={:.1}GiB pin_cold={} ple_prefault={} ({:.1}ms)",
            ampllm_cpu_kernels::active_cpu_isa(),
            (mem_avail_after as f64) / 1073741824.0,
            all_pinned_cold,
            ple_prefaulted,
            pf_ms,
        );

        // Upload per-layer VRAM and zero-copy RAM pointer tables on each owning GPU.
        self.experts.clear();
        for il in 0..n_layer {
            let dev = self.layers[il].device;
            set_device(dev)?;
            let vp = std::mem::take(&mut vram_ptrs_host_all[il]);
            let rdp = std::mem::take(&mut ram_dev_ptrs_all[il]);
            let dev_vram_ptrs = DeviceBuffer::alloc(dev, vp.len() * 8)?;
            let dev_ram_ptrs = DeviceBuffer::alloc(dev, rdp.len() * 8)?;
            dev_vram_ptrs.copy_from_host(bytemuck::cast_slice(&vp))?;
            dev_ram_ptrs.copy_from_host(bytemuck::cast_slice(&rdp))?;

            let mut profile_rank = vec![n_exp as u16; n_exp];
            for (rank, &e) in orders[il].iter().enumerate() {
                if e < n_exp {
                    profile_rank[e] = rank as u16;
                }
            }

            self.experts.push(LayerExpertsStorage {
                n_resident: n_resident[il],
                slot_of: std::mem::take(&mut slot_of_all[il]),
                expert_of_slot: std::mem::take(&mut expert_of_slot_all[il]),
                is_resident: std::mem::take(&mut is_resident_all[il]),
                vram_slab: vram_slabs[il].take(),
                vram_ptrs_host: vp,
                ram_host_ptrs: std::mem::take(&mut ram_host_ptrs_all[il]),
                ram_dev_ptrs: rdp,
                dev_vram_ptrs,
                dev_ram_ptrs,
                heat: std::mem::take(&mut heat_all[il]),
                last_used: vec![0u64; n_exp],
                all_pinned_cold,
                decode_usage: vec![0.0f32; n_exp],
                prompt_hits: vec![0u32; n_exp],
                prompt_tail_hits: vec![0u32; n_exp],
                profile_rank,
                free_ram_slots: if all_pinned_cold {
                    std::mem::take(&mut free_ram_slots_all[il])
                } else {
                    Vec::new()
                },
            });
        }

        let _ = gpu_count;
        Ok(())
    }

    pub fn on_new_prompt(&mut self) {
        for es in &mut self.experts {
            for h in &mut es.heat {
                *h *= 0.85;
            }
            for u in &mut es.decode_usage {
                *u *= 0.5;
            }
            es.prompt_hits.fill(0);
            es.prompt_tail_hits.fill(0);
        }
    }

    pub fn export_profile(&self) -> CacheProfile {
        let n_layer = self.experts.len();
        let n_exp = self.cfg.n_expert;
        let mut ranked_experts = Vec::with_capacity(n_layer);
        for es in &self.experts {
            let mut ord: Vec<usize> = (0..n_exp).collect();
            ord.sort_by(|&a, &b| {
                es.heat[b]
                    .total_cmp(&es.heat[a])
                    .then_with(|| es.is_resident[b].cmp(&es.is_resident[a]))
                    .then_with(|| es.last_used[b].cmp(&es.last_used[a]))
                    .then_with(|| a.cmp(&b))
            });
            ranked_experts.push(ord);
        }
        CacheProfile {
            arch: "qwen4exp".to_owned(),
            n_layer_phys: n_layer,
            n_expert: n_exp,
            ranked_experts,
            global_ranked: Vec::new(),
        }
    }

    /// 1-for-1 in-place expert swaps (VRAM slab <-> cold host backing) with full-duplex D2H/H2D overlap.
    pub fn warmup_from_heat(
        &mut self,
        streams: &[(&CudaStream, &CudaStream, &[CudaEvent; 3], &[CudaEvent; 3])],
        bounces: &mut [&mut PinnedHostBuffer],
        max_swaps_per_layer: usize,
        min_heat: f32,
    ) -> Result<usize> {
        if max_swaps_per_layer == 0 || self.experts.is_empty() {
            return Ok(0);
        }
        let n_exp = self.cfg.n_expert;
        let layer_bbs: Vec<usize> = self.layers.iter().map(|l| l.moe.blob_bytes).collect();
        let split = self.split_layer.min(self.experts.len());

        fn finish_pinned_wave(
            il: usize,
            bb: usize,
            wave_cap: usize,
            wi: usize,
            wave: &[(usize, usize)],
            es: &mut LayerExpertsStorage,
            layers_host_atomic: &[Vec<AtomicUsize>],
            bounce_ptr: *mut u8,
            h2d_ev: &CudaEvent,
        ) -> Result<()> {
            h2d_ev.sync()?;
            let half_off = (wi & 1) * wave_cap * bb;
            for (k, &(ev_e, in_e)) in wave.iter().enumerate() {
                let slot = es.slot_of[ev_e] as usize;
                let v_ptr = es.vram_ptrs_host[ev_e];
                let h_ptr = es.ram_host_ptrs[in_e];
                let h_dev = es.ram_dev_ptrs[in_e];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bounce_ptr.add(half_off + k * bb),
                        h_ptr as *mut u8,
                        bb,
                    );
                }
                es.slot_of[ev_e] = -1;
                es.slot_of[in_e] = slot as i32;
                es.expert_of_slot[slot] = in_e;
                es.is_resident[ev_e] = false;
                es.is_resident[in_e] = true;
                es.vram_ptrs_host[ev_e] = 0;
                es.vram_ptrs_host[in_e] = v_ptr;
                es.ram_host_ptrs[ev_e] = h_ptr;
                es.ram_host_ptrs[in_e] = 0;
                es.ram_dev_ptrs[ev_e] = h_dev;
                es.ram_dev_ptrs[in_e] = 0;
                layers_host_atomic[il][ev_e].store(h_ptr, Ordering::Release);
                layers_host_atomic[il][in_e].store(0, Ordering::Release);
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn swap_gpu_range(
            start_layer: usize,
            layer_bbs: &[usize],
            experts: &mut [LayerExpertsStorage],
            _cold_backing: &mut ColdHostBacking,
            layers_host_atomic: &[Vec<AtomicUsize>],
            dev: i32,
            copy_stream: &CudaStream,
            d2h_stream: &CudaStream,
            dma_events: &[CudaEvent; 3],
            compute_events: &[CudaEvent; 3],
            bounce: &mut PinnedHostBuffer,
            n_exp: usize,
            max_swaps_per_layer: usize,
            min_heat: f32,
        ) -> Result<usize> {
            set_device(dev)?;
            let mut swaps_done = 0usize;
            let bounce_ptr = bounce.as_mut_ptr();
            let bounce_cap = bounce.bytes();

            for (loc_il, es) in experts.iter_mut().enumerate() {
                let il = start_layer + loc_il;
                let bb = layer_bbs[loc_il];
                let n_res = es.n_resident;
                if n_res == 0 || n_res >= n_exp || bb == 0 || bounce_cap < 2 * bb {
                    continue;
                }
                let mut cold: Vec<usize> = (0..n_exp)
                    .filter(|&e| !es.is_resident[e] && es.ram_host_ptrs[e] != 0 && es.heat[e] >= min_heat)
                    .collect();
                if cold.is_empty() {
                    continue;
                }
                cold.sort_by(|&a, &b| {
                    es.heat[b]
                        .total_cmp(&es.heat[a])
                        .then_with(|| es.last_used[b].cmp(&es.last_used[a]))
                });

                let mut res: Vec<usize> = (0..n_exp)
                    .filter(|&e| es.is_resident[e] && es.slot_of[e] >= 0)
                    .collect();
                res.sort_by(|&a, &b| {
                    es.heat[a]
                        .total_cmp(&es.heat[b])
                        .then_with(|| es.last_used[a].cmp(&es.last_used[b]))
                });

                let cap = cold.len().min(res.len()).min(max_swaps_per_layer);
                let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(cap);
                for k in 0..cap {
                    let in_e = cold[k];
                    let ev_e = res[k];
                    if es.heat[in_e] <= es.heat[ev_e] * 1.35 + 2.0 {
                        break;
                    }
                    pairs.push((ev_e, in_e));
                }
                if pairs.is_empty() {
                    continue;
                }

                let wave_cap = (bounce_cap / (2 * bb)).min(8).max(1);
                if es.all_pinned_cold {
                    let waves: Vec<&[(usize, usize)]> = pairs.chunks(wave_cap).collect();
                    for (wi, wave) in waves.iter().enumerate() {
                        let half = wi & 1;
                        if wi >= 2 {
                            finish_pinned_wave(
                                il,
                                bb,
                                wave_cap,
                                wi - 2,
                                waves[wi - 2],
                                es,
                                layers_host_atomic,
                                bounce_ptr,
                                &compute_events[half],
                            )?;
                        }
                        let half_off = half * wave_cap * bb;
                        for (k, &(ev_e, _in_e)) in wave.iter().enumerate() {
                            let v_ptr = es.vram_ptrs_host[ev_e] as *const std::ffi::c_void;
                            unsafe {
                                let dst_ev = std::slice::from_raw_parts_mut(
                                    bounce_ptr.add(half_off + k * bb),
                                    bb,
                                );
                                copy_d2h_raw_async(dst_ev, v_ptr, d2h_stream)?;
                            }
                        }
                        dma_events[half].record(d2h_stream)?;
                        copy_stream.wait_event(&dma_events[half])?;
                        for &(ev_e, in_e) in wave.iter() {
                            let v_ptr = es.vram_ptrs_host[ev_e] as *mut std::ffi::c_void;
                            let h_ptr = es.ram_host_ptrs[in_e] as *const u8;
                            unsafe {
                                let src_in = std::slice::from_raw_parts(h_ptr, bb);
                                copy_h2d_raw_async(v_ptr, src_in, copy_stream)?;
                            }
                        }
                        compute_events[half].record(copy_stream)?;
                    }
                    let tail0 = waves.len().saturating_sub(2);
                    for wi in tail0..waves.len() {
                        let half = wi & 1;
                        finish_pinned_wave(
                            il,
                            bb,
                            wave_cap,
                            wi,
                            waves[wi],
                            es,
                            layers_host_atomic,
                            bounce_ptr,
                            &compute_events[half],
                        )?;
                    }
                } else {
                    for chunk in pairs.chunks(wave_cap) {
                        for (k, &(ev_e, in_e)) in chunk.iter().enumerate() {
                            let v_ptr = es.vram_ptrs_host[ev_e] as *const std::ffi::c_void;
                            let h_ptr = es.ram_host_ptrs[in_e] as *const u8;
                            unsafe {
                                let dst_ev = std::slice::from_raw_parts_mut(bounce_ptr.add(k * bb), bb);
                                copy_d2h_raw_async(dst_ev, v_ptr, copy_stream)?;
                                std::ptr::copy_nonoverlapping(
                                    h_ptr,
                                    bounce_ptr.add((wave_cap + k) * bb),
                                    bb,
                                );
                            }
                        }
                        copy_stream.sync()?;

                        for (k, &(ev_e, in_e)) in chunk.iter().enumerate() {
                            let slot = es.slot_of[ev_e] as usize;
                            let v_ptr = es.vram_ptrs_host[ev_e];
                            let h_ptr = es.ram_host_ptrs[in_e];
                            let h_dev = es.ram_dev_ptrs[in_e];
                            unsafe {
                                let src_in = std::slice::from_raw_parts(
                                    bounce_ptr.add((wave_cap + k) * bb),
                                    bb,
                                );
                                copy_h2d_raw_async(v_ptr as *mut std::ffi::c_void, src_in, copy_stream)?;
                                std::ptr::copy_nonoverlapping(
                                    bounce_ptr.add(k * bb),
                                    h_ptr as *mut u8,
                                    bb,
                                );
                            }
                            es.slot_of[ev_e] = -1;
                            es.slot_of[in_e] = slot as i32;
                            es.expert_of_slot[slot] = in_e;
                            es.is_resident[ev_e] = false;
                            es.is_resident[in_e] = true;
                            es.vram_ptrs_host[ev_e] = 0;
                            es.vram_ptrs_host[in_e] = v_ptr;
                            es.ram_host_ptrs[ev_e] = h_ptr;
                            es.ram_host_ptrs[in_e] = 0;
                            es.ram_dev_ptrs[ev_e] = h_dev;
                            es.ram_dev_ptrs[in_e] = 0;
                            layers_host_atomic[il][ev_e].store(h_ptr, Ordering::Release);
                            layers_host_atomic[il][in_e].store(0, Ordering::Release);
                        }
                        copy_stream.sync()?;
                    }
                }

                es.dev_vram_ptrs
                    .copy_from_host_async(bytemuck::cast_slice(&es.vram_ptrs_host), copy_stream)?;
                es.dev_ram_ptrs
                    .copy_from_host_async(bytemuck::cast_slice(&es.ram_dev_ptrs), copy_stream)?;
                swaps_done += pairs.len();
            }
            copy_stream.sync()?;
            Ok(swaps_done)
        }

        if self.gpu_ids.len() >= 2 && self.cold_backings.len() >= 2 && streams.len() >= 2 && bounces.len() >= 2 {
            let (bb0, bb1) = layer_bbs.split_at(split);
            let (exp0, exp1) = self.experts.split_at_mut(split);
            let (cold0_s, cold1_s) = self.cold_backings.split_at_mut(1);
            let (b0_s, b1_s) = bounces.split_at_mut(1);
            let dev0 = self.gpu_ids[0];
            let dev1 = *self.gpu_ids.last().unwrap();
            let (cs0, ds0, de0, ce0) = streams[0];
            let (cs1, ds1, de1, ce1) = streams[1];
            let host_atomic = &self.layers_host_atomic;
            let (r0, r1) = std::thread::scope(|s| {
                let h0 = s.spawn(|| {
                    swap_gpu_range(
                        0,
                        bb0,
                        exp0,
                        &mut cold0_s[0],
                        host_atomic,
                        dev0,
                        cs0,
                        ds0,
                        de0,
                        ce0,
                        b0_s[0],
                        n_exp,
                        max_swaps_per_layer,
                        min_heat,
                    )
                });
                let res1 = swap_gpu_range(
                    split,
                    bb1,
                    exp1,
                    &mut cold1_s[0],
                    host_atomic,
                    dev1,
                    cs1,
                    ds1,
                    de1,
                    ce1,
                    b1_s[0],
                    n_exp,
                    max_swaps_per_layer,
                    min_heat,
                );
                let res0 = h0.join().expect("gpu0 warmup thread panicked");
                (res0, res1)
            });
            Ok(r0? + r1?)
        } else {
            let dev0 = self.gpu_ids[0];
            let (cs0, ds0, de0, ce0) = streams[0];
            swap_gpu_range(
                0,
                &layer_bbs,
                &mut self.experts,
                &mut self.cold_backings[0],
                &self.layers_host_atomic,
                dev0,
                cs0,
                ds0,
                de0,
                ce0,
                bounces[0],
                n_exp,
                max_swaps_per_layer,
                min_heat,
            )
        }
    }

    #[inline]
    pub fn ple_table_bytes(&self) -> &[u8] {
        match self.ple_shard_idx {
            Some(s) => &self.gguf.shard_bytes(s)[self.ple_offset..self.ple_offset + self.ple_byte_len],
            None => &[],
        }
    }

    #[inline]
    pub fn token_embd_bytes(&self) -> Result<&[u8]> {
        Ok(self.gguf.tensor_bytes("token_embd.weight")?)
    }
}
