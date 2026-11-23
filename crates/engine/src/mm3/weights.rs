use crate::config::{EngineConfig, MultiGpuMode};
use crate::kv_cache::KvQuantMode;
use crate::mm3::dspark::Mm3Dspark;
use crate::mm3::vision::Mm3Vision;
use ampllm_cache::{align_up_256, CacheBudget, CacheProfile};
use ampllm_cpu_kernels::mm3_cpu::q6_k_row_to_f32_cpu;
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::{
    device_count, enable_peer_access, mem_info, set_device, DeviceBuffer,
    RegisteredMappedHostBuffer,
};
use ampllm_gguf::{
    discover_mm3_aux_files, GgmlType, GgufFile, Mm3AuxFiles, Mm3Config, Mm3DsparkConfig,
    Mm3VisionConfig,
};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mm3KvFormat {
    Bf16 = 0,
    Int8HadV = 1,
}

impl Mm3KvFormat {
    pub fn from_kv_mode(mode: KvQuantMode) -> Self {
        match mode {
            KvQuantMode::F16 => Self::Bf16,
            KvQuantMode::Q8_0 | KvQuantMode::Fp8 => Self::Int8HadV,
        }
    }
}

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

pub struct Mm3DenseMlpWeights {
    pub gate: QuantMatrix,
    pub up: QuantMatrix,
    pub down: QuantMatrix,
}

pub struct Mm3MoeMeta {
    pub gate_inp: DeviceBuffer,
    pub exp_probs_bias: DeviceBuffer,
    pub sh_gate: QuantMatrix,
    pub sh_up: QuantMatrix,
    pub sh_down: QuantMatrix,
    pub gate_type: GgmlType,
    pub up_type: GgmlType,
    pub down_type: GgmlType,
    pub gate_row_bytes: usize,
    pub up_row_bytes: usize,
    pub down_row_bytes: usize,
    pub gate_bytes: usize,
    pub up_bytes: usize,
    pub down_bytes: usize,
    pub u_off: usize,
    pub d_off: usize,
    pub blob_bytes: usize,
}

pub enum Mm3FfnWeights {
    Dense(Mm3DenseMlpWeights),
    Moe(Mm3MoeMeta),
}

pub struct Mm3LayerWeights {
    pub device: i32,
    pub gpu_idx: usize,
    pub attn_norm: DeviceBuffer,
    pub ffn_norm: DeviceBuffer,
    pub attn_q_norm: DeviceBuffer,
    pub attn_k_norm: DeviceBuffer,
    pub attn_q: QuantMatrix,
    pub attn_k: QuantMatrix,
    pub attn_v: QuantMatrix,
    pub attn_o: QuantMatrix,
    pub index_q_norm: DeviceBuffer,
    pub index_k_norm: DeviceBuffer,
    pub index_q: QuantMatrix,
    pub index_k: QuantMatrix,
    pub ffn: Mm3FfnWeights,
}

pub struct Mm3LayerExperts {
    pub is_static_resident: Vec<bool>,
    pub is_resident: Vec<bool>,
    pub promo_slot_of: Vec<i16>,
    pub vram_slab: Option<DeviceBuffer>,
    pub gate_ptrs_host: Vec<u64>,
    pub up_ptrs_host: Vec<u64>,
    pub down_ptrs_host: Vec<u64>,
    pub ram_host_ptrs: Vec<usize>,
    pub ram_dev_ptrs: Vec<u64>,
    pub dev_gate_ptrs: DeviceBuffer,
    pub dev_up_ptrs: DeviceBuffer,
    pub dev_down_ptrs: DeviceBuffer,
}

pub struct Mm3Weights {
    pub gguf: GgufFile,
    pub indexers_gguf: Option<GgufFile>,
    pub aux: Mm3AuxFiles,
    pub cfg: Mm3Config,
    pub kv_fmt: Mm3KvFormat,
    pub gpu_ids: Vec<i32>,
    pub p2p_enabled: bool,
    pub layer_split: usize,
    pub layers: Vec<Mm3LayerWeights>,
    pub experts: Vec<Option<Mm3LayerExperts>>,
    pub cold_slabs: Vec<RegisteredMappedHostBuffer>,
    pub promo_slabs: Vec<Option<DeviceBuffer>>,
    pub promo_slot_counts: Vec<usize>,
    pub promo_slot_strides: Vec<usize>,
    pub output_norm: DeviceBuffer,
    pub lm_head: QuantMatrix,
    pub rope_inv_freq: Vec<DeviceBuffer>,
    pub token_embd_bytes: usize,
    token_embd_row_bytes: usize,
    pub vision: Option<Mm3Vision>,
    pub dspark: Option<Mm3Dspark>,
    pub total_experts: usize,
    pub total_resident_experts: usize,
}

fn upload_f32_tensor(dev: i32, gguf: &GgufFile, name: &str) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    if info.ggml_type == GgmlType::F32 {
        let buf = DeviceBuffer::alloc(dev, raw.len())?;
        buf.copy_from_host(raw)?;
        Ok(buf)
    } else {
        let mut f32_vals = vec![0.0f32; info.n_elements()];
        ampllm_gguf::dequantize_row_to_f32(info.ggml_type, raw, &mut f32_vals);
        let buf = DeviceBuffer::alloc(dev, f32_vals.len() * 4)?;
        buf.copy_from_host(bytemuck::cast_slice(&f32_vals))?;
        Ok(buf)
    }
}

fn upload_quant_matrix(dev: i32, gguf: &GgufFile, name: &str) -> Result<QuantMatrix> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    if info.dims.len() != 2 {
        bail!("tensor {name} expected 2D dims, got {:?}", info.dims);
    }
    let n_in = info.dims[0] as usize;
    let n_out = info.dims[1] as usize;
    let row_bytes = info.ggml_type.row_bytes(n_in);
    if raw.len() != n_out * row_bytes {
        bail!(
            "tensor {name} byte_len {} != {}x{} ({})",
            raw.len(),
            n_out,
            row_bytes,
            n_out * row_bytes
        );
    }
    let buf = DeviceBuffer::alloc(dev, raw.len())?;
    buf.copy_from_host(raw)?;
    Ok(QuantMatrix {
        buf,
        ggml_type: info.ggml_type,
        n_out,
        n_in,
        row_bytes,
    })
}

fn lookup_tensor_source<'a>(
    primary: &'a GgufFile,
    sidecar: Option<&'a GgufFile>,
    name: &str,
) -> Result<&'a GgufFile> {
    if primary.has_tensor(name) {
        return Ok(primary);
    }
    if let Some(sc) = sidecar {
        if sc.has_tensor(name) {
            return Ok(sc);
        }
    }
    bail!("MM3 missing required tensor `{name}` (checked main GGUF and sidecar indexers file)");
}

fn kv_bytes_per_layer(cfg: &Mm3Config, max_ctx: usize, kv_fmt: Mm3KvFormat) -> usize {
    let ic = max_ctx * cfg.ix_head_dim * 2;
    let kv_elems = max_ctx * cfg.n_kv_heads * cfg.head_dim;
    match kv_fmt {
        Mm3KvFormat::Bf16 => ic + 2 * kv_elems * 2,
        Mm3KvFormat::Int8HadV => ic + 2 * kv_elems + 2 * max_ctx * cfg.n_kv_heads * 4,
    }
}

fn layer_fixed_vram_bytes(gguf: &GgufFile, sidecar: Option<&GgufFile>, cfg: &Mm3Config, il: usize) -> usize {
    let sum_names = |names: &[String]| -> usize {
        names
            .iter()
            .filter_map(|n| {
                lookup_tensor_source(gguf, sidecar, n)
                    .ok()
                    .and_then(|g| g.tensor_info(n).ok())
                    .map(|info| info.byte_len)
            })
            .sum()
    };
    let mut keys = vec![
        format!("blk.{il}.attn_norm.weight"),
        format!("blk.{il}.ffn_norm.weight"),
        format!("blk.{il}.attn_q_norm.weight"),
        format!("blk.{il}.attn_k_norm.weight"),
        format!("blk.{il}.attn_q.weight"),
        format!("blk.{il}.attn_k.weight"),
        format!("blk.{il}.attn_v.weight"),
        format!("blk.{il}.attn_output.weight"),
        format!("blk.{il}.indexer.q_norm.weight"),
        format!("blk.{il}.indexer.k_norm.weight"),
        format!("blk.{il}.indexer.q_proj.weight"),
        format!("blk.{il}.indexer.k_proj.weight"),
    ];
    if il < cfg.n_dense_layers {
        keys.push(format!("blk.{il}.ffn_gate.weight"));
        keys.push(format!("blk.{il}.ffn_up.weight"));
        keys.push(format!("blk.{il}.ffn_down.weight"));
    } else {
        keys.push(format!("blk.{il}.ffn_gate_inp.weight"));
        keys.push(format!("blk.{il}.exp_probs_b.bias"));
        keys.push(format!("blk.{il}.ffn_gate_shexp.weight"));
        keys.push(format!("blk.{il}.ffn_up_shexp.weight"));
        keys.push(format!("blk.{il}.ffn_down_shexp.weight"));
    }
    sum_names(&keys)
}

impl Mm3Weights {
    pub fn load_non_expert(gguf: GgufFile, engine_cfg: &EngineConfig) -> Result<Self> {
        let cfg = Mm3Config::from_gguf(&gguf)?;
        let model_path = gguf
            .shard_paths
            .first()
            .context("GGufFile has no shard paths")?
            .clone();
        let aux = discover_mm3_aux_files(&model_path, None, None)?;

        let indexers_gguf = if !gguf.has_tensor("blk.0.indexer.q_proj.weight") {
            let Some(ref ib) = aux.indexers_bin else {
                bail!(
                    "MM3 dense layers 0..{} lack indexer tensors and no *-maya-indexers.bin sidecar was found beside {}",
                    cfg.n_dense_layers,
                    model_path.display()
                );
            };
            Some(GgufFile::open(ib).with_context(|| {
                format!("failed to open MM3 indexers sidecar {}", ib.display())
            })?)
        } else {
            aux.indexers_bin
                .as_ref()
                .map(|p| GgufFile::open(p))
                .transpose()?
        };

        let kv_fmt = Mm3KvFormat::from_kv_mode(engine_cfg.kv_mode);
        let n_dev = device_count()? as usize;
        let gpu_ids = match engine_cfg.multi_gpu {
            MultiGpuMode::Single => vec![engine_cfg.device],
            _ if n_dev >= 2 => vec![0, 1],
            _ => vec![engine_cfg.device],
        };
        let p2p_enabled = if gpu_ids.len() >= 2 && std::env::var("AMPLLM_MM3_NO_P2P").as_deref() != Ok("1") {
            let p01 = enable_peer_access(gpu_ids[0], gpu_ids[1]).unwrap_or(false);
            let p10 = enable_peer_access(gpu_ids[1], gpu_ids[0]).unwrap_or(false);
            p01 && p10
        } else {
            false
        };

        // Token embedding kept in mmap'd host GGUF for fast AVX-512 Q6_K CPU row lookup.
        let tok_info = gguf.tensor_info("token_embd.weight")?;
        if tok_info.ggml_type != GgmlType::Q6K {
            bail!(
                "MM3 expects token_embd.weight to be Q6_K, got {:?}",
                tok_info.ggml_type
            );
        }
        let token_embd_row_bytes = tok_info.ggml_type.row_bytes(cfg.n_embd);
        let token_embd_bytes = tok_info.byte_len;
        let tok_raw_bytes = gguf.tensor_bytes("token_embd.weight")?;

        // Load optional Vision and DSpark modules cleanly.
        let vision = if let (Some(vst), Some(vc)) = (&aux.vision_safetensors, &aux.vision_config_json) {
            let vcfg = Mm3VisionConfig::from_companion_jsons(vc, aux.preprocessor_json.as_deref())?;
            Some(Mm3Vision::load(vst, vcfg)?)
        } else {
            None
        };

        let head_dev = *gpu_ids.last().unwrap();
        let dspark = if engine_cfg.use_spec {
            if let (Some(dst), Some(dc)) = (&aux.dspark_safetensors, &aux.dspark_config_json) {
                let dcfg = Mm3DsparkConfig::from_json(dc)?;
                Some(Mm3Dspark::load(
                    dst,
                    dcfg,
                    tok_raw_bytes,
                    token_embd_row_bytes,
                    head_dev,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        // Plan pipeline split AFTER DSpark is already on head_dev so mem_info() reflects real remaining VRAM!
        let layer_split = if gpu_ids.len() >= 2 {
            let kv_per_layer = kv_bytes_per_layer(&cfg, engine_cfg.max_ctx, kv_fmt);
            let per_layer_fixed: Vec<usize> = (0..cfg.n_layers)
                .map(|il| {
                    layer_fixed_vram_bytes(&gguf, indexers_gguf.as_ref(), &cfg, il) + kv_per_layer
                })
                .collect();
            let lm_head_bytes = gguf.tensor_info("output.weight")?.byte_len;
            let (free0, _) = mem_info(gpu_ids[0])?;
            let (free1, _) = mem_info(gpu_ids[1])?;
            let vis_reserve = if vision.is_some() { 384usize << 20 } else { 0 };
            let free0_adj = free0.saturating_sub(vis_reserve);
            let free1_adj = free1.saturating_sub(lm_head_bytes + (64usize << 20));
            let promo_reserve = 1152usize << 20;
            let mut best_split = cfg.n_layers / 2;
            let mut best_diff = f64::MAX;
            for split in (cfg.n_dense_layers + 1)..cfg.n_layers {
                let moe0 = split - cfg.n_dense_layers;
                let moe1 = cfg.n_layers - split;
                let c0: usize = per_layer_fixed[..split].iter().sum();
                let c1: usize = per_layer_fixed[split..].iter().sum();
                if c0 < free0_adj && c1 < free1_adj {
                    let rem0 = free0_adj - c0;
                    let rem1 = free1_adj - c1;
                    let (r0, r1) = if rem0 > promo_reserve && rem1 > promo_reserve {
                        (rem0 - promo_reserve, rem1 - promo_reserve)
                    } else {
                        (rem0, rem1)
                    };
                    let per_moe0 = (r0 as f64) / (moe0 as f64);
                    let per_moe1 = (r1 as f64) / (moe1 as f64);
                    let diff = (per_moe0 - per_moe1).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_split = split;
                    }
                }
            }
            best_split
        } else {
            cfg.n_layers
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let gpu_idx = if gpu_ids.len() >= 2 && il >= layer_split {
                1
            } else {
                0
            };
            let dev = gpu_ids[gpu_idx];
            set_device(dev)?;

            let p = format!("blk.{il}.");
            let iq_src = lookup_tensor_source(
                &gguf,
                indexers_gguf.as_ref(),
                &format!("{p}indexer.q_proj.weight"),
            )?;

            let ffn = if il < cfg.n_dense_layers {
                Mm3FfnWeights::Dense(Mm3DenseMlpWeights {
                    gate: upload_quant_matrix(dev, &gguf, &format!("{p}ffn_gate.weight"))?,
                    up: upload_quant_matrix(dev, &gguf, &format!("{p}ffn_up.weight"))?,
                    down: upload_quant_matrix(dev, &gguf, &format!("{p}ffn_down.weight"))?,
                })
            } else {
                let mt = cfg.moe_types[il].with_context(|| {
                    format!("layer {il} >= n_dense_layers ({}) missing moe_types", cfg.n_dense_layers)
                })?;
                let gate_rb = mt.gate.row_bytes(cfg.n_embd);
                let up_rb = mt.up.row_bytes(cfg.n_embd);
                let down_rb = mt.down.row_bytes(cfg.expert_ff);
                let gate_bytes = cfg.expert_ff * gate_rb;
                let up_bytes = cfg.expert_ff * up_rb;
                let down_bytes = cfg.n_embd * down_rb;
                let u_off = align_up_256(gate_bytes);
                let d_off = align_up_256(u_off + up_bytes);
                let blob_bytes = align_up_256(d_off + down_bytes);
                Mm3FfnWeights::Moe(Mm3MoeMeta {
                    gate_inp: upload_f32_tensor(dev, &gguf, &format!("{p}ffn_gate_inp.weight"))?,
                    exp_probs_bias: upload_f32_tensor(
                        dev,
                        &gguf,
                        &format!("{p}exp_probs_b.bias"),
                    )?,
                    sh_gate: upload_quant_matrix(dev, &gguf, &format!("{p}ffn_gate_shexp.weight"))?,
                    sh_up: upload_quant_matrix(dev, &gguf, &format!("{p}ffn_up_shexp.weight"))?,
                    sh_down: upload_quant_matrix(dev, &gguf, &format!("{p}ffn_down_shexp.weight"))?,
                    gate_type: mt.gate,
                    up_type: mt.up,
                    down_type: mt.down,
                    gate_row_bytes: gate_rb,
                    up_row_bytes: up_rb,
                    down_row_bytes: down_rb,
                    gate_bytes,
                    up_bytes,
                    down_bytes,
                    u_off,
                    d_off,
                    blob_bytes,
                })
            };

            layers.push(Mm3LayerWeights {
                device: dev,
                gpu_idx,
                attn_norm: upload_f32_tensor(dev, &gguf, &format!("{p}attn_norm.weight"))?,
                ffn_norm: upload_f32_tensor(dev, &gguf, &format!("{p}ffn_norm.weight"))?,
                attn_q_norm: upload_f32_tensor(dev, &gguf, &format!("{p}attn_q_norm.weight"))?,
                attn_k_norm: upload_f32_tensor(dev, &gguf, &format!("{p}attn_k_norm.weight"))?,
                attn_q: upload_quant_matrix(dev, &gguf, &format!("{p}attn_q.weight"))?,
                attn_k: upload_quant_matrix(dev, &gguf, &format!("{p}attn_k.weight"))?,
                attn_v: upload_quant_matrix(dev, &gguf, &format!("{p}attn_v.weight"))?,
                attn_o: upload_quant_matrix(dev, &gguf, &format!("{p}attn_output.weight"))?,
                index_q_norm: upload_f32_tensor(dev, iq_src, &format!("{p}indexer.q_norm.weight"))?,
                index_k_norm: upload_f32_tensor(dev, iq_src, &format!("{p}indexer.k_norm.weight"))?,
                index_q: upload_quant_matrix(dev, iq_src, &format!("{p}indexer.q_proj.weight"))?,
                index_k: upload_quant_matrix(dev, iq_src, &format!("{p}indexer.k_proj.weight"))?,
                ffn,
            });
        }

        set_device(head_dev)?;
        let output_norm = upload_f32_tensor(head_dev, &gguf, "output_norm.weight")?;
        let lm_head = upload_quant_matrix(head_dev, &gguf, "output.weight")?;

        let inv_h: Vec<f32> = (0..cfg.rot_dim / 2)
            .map(|i| 1.0f32 / cfg.rope_theta.powf((2 * i) as f32 / (cfg.rot_dim as f32)))
            .collect();
        let mut rope_inv_freq = Vec::with_capacity(gpu_ids.len());
        for &dev in &gpu_ids {
            set_device(dev)?;
            let b = DeviceBuffer::alloc(dev, inv_h.len() * 4)?;
            b.copy_from_host(bytemuck::cast_slice(&inv_h))?;
            rope_inv_freq.push(b);
        }

        let experts = (0..cfg.n_layers).map(|_| None).collect();
        let total_experts = (cfg.n_layers - cfg.n_dense_layers) * cfg.n_experts;
        let promo_slabs = (0..gpu_ids.len()).map(|_| None).collect();
        let promo_slot_counts = vec![0usize; gpu_ids.len()];
        let promo_slot_strides = vec![0usize; gpu_ids.len()];

        Ok(Self {
            gguf,
            indexers_gguf,
            aux,
            cfg,
            kv_fmt,
            gpu_ids,
            p2p_enabled,
            layer_split,
            layers,
            experts,
            cold_slabs: Vec::new(),
            promo_slabs,
            promo_slot_counts,
            promo_slot_strides,
            output_norm,
            lm_head,
            rope_inv_freq,
            token_embd_bytes,
            token_embd_row_bytes,
            vision,
            dspark,
            total_experts,
            total_resident_experts: 0,
        })
    }

    #[inline]
    pub fn embed_row_f32(&self, tok_id: u32, dst: &mut [f32]) -> Result<()> {
        let tid = (tok_id as usize).min(self.cfg.vocab_size.saturating_sub(1));
        let raw = self.gguf.tensor_bytes("token_embd.weight")?;
        let off = tid * self.token_embd_row_bytes;
        q6_k_row_to_f32_cpu(&raw[off..off + self.token_embd_row_bytes], dst);
        Ok(())
    }

    /// Pack routed experts into resident VRAM slabs and pinned mapped cold RAM slabs
    /// after KV cache and GPU scratch buffers are allocated.
    pub fn allocate_experts(
        &mut self,
        engine_cfg: &EngineConfig,
        pool: &Arc<CpuThreadPool>,
    ) -> Result<()> {
        let n_layer = self.cfg.n_layers;
        let n_exp = self.cfg.n_experts;
        let profile = if let Some(ref p) = engine_cfg.load_profile {
            if p.exists() {
                Some(CacheProfile::load_for_arch(
                    p,
                    "minimax-m3",
                    n_layer,
                    n_exp,
                )?)
            } else {
                None
            }
        } else {
            None
        };

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

        let has_global_ranked = profile
            .as_ref()
            .is_some_and(|p| !p.global_ranked.is_empty());

        let mut resident_flags = vec![vec![false; n_exp]; n_layer];
        let mut total_cold_bytes_per_gpu = vec![0usize; self.gpu_ids.len()];
        let mut promo_slabs: Vec<Option<DeviceBuffer>> = (0..self.gpu_ids.len()).map(|_| None).collect();
        let mut promo_slot_counts = vec![0usize; self.gpu_ids.len()];
        let mut promo_slot_strides = vec![0usize; self.gpu_ids.len()];
        let mut total_resident = 0usize;

        let req_promo_slots: usize = std::env::var("AMPLLM_MM3_PROMO_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64);

        for (gidx, &dev) in self.gpu_ids.iter().enumerate() {
            set_device(dev)?;
            let (free, _) = mem_info(dev)?;
            let safety = if gidx == 0 && self.vision.is_some() {
                672usize << 20
            } else {
                448usize << 20
            };
            let max_usable = free.saturating_sub(safety);
            let total_budget = match engine_cfg.expert_cache {
                CacheBudget::Auto => max_usable,
                CacheBudget::Mb(mb) => mb
                    .saturating_mul(1024 * 1024)
                    .min(free.saturating_sub(384usize << 20)),
            };

            let moe_layers: Vec<usize> = (self.cfg.n_dense_layers..n_layer)
                .filter(|&il| self.layers[il].device == dev)
                .collect();
            let denom: usize = moe_layers
                .iter()
                .map(|&il| match self.layers[il].ffn {
                    Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                    _ => 0,
                })
                .sum();
            let max_blob = moe_layers
                .iter()
                .map(|&il| match self.layers[il].ffn {
                    Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                    _ => 0,
                })
                .max()
                .unwrap_or(0);
            let slot_stride = align_up_256(max_blob.max(4096));
            let need_all_static = n_exp.saturating_mul(denom);
            let n_promo = if denom > 0 && need_all_static > total_budget && slot_stride > 0 {
                req_promo_slots.min((total_budget * 15 / 100) / slot_stride)
            } else {
                0
            };
            let budget = total_budget.saturating_sub(n_promo * slot_stride);
            promo_slot_strides[gidx] = slot_stride;

            let mut gpu_res_count = 0usize;
            if denom > 0 {
                let mut rem = budget;
                if has_global_ranked {
                    let prof = profile.as_ref().unwrap();
                    for &(il, e) in &prof.global_ranked {
                        if il >= self.cfg.n_dense_layers
                            && il < n_layer
                            && e < n_exp
                            && self.layers[il].device == dev
                            && !resident_flags[il][e]
                        {
                            let bsz = match self.layers[il].ffn {
                                Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                                _ => 0,
                            };
                            if rem >= bsz {
                                resident_flags[il][e] = true;
                                rem -= bsz;
                                total_resident += 1;
                                gpu_res_count += 1;
                            }
                        }
                    }
                    // Fill any leftover VRAM uniformly in per-layer order
                    'outer: loop {
                        let mut placed_any = false;
                        for &il in &moe_layers {
                            let bsz = match self.layers[il].ffn {
                                Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                                _ => 0,
                            };
                            if let Some(&e) = orders[il].iter().find(|&&e| !resident_flags[il][e]) {
                                if rem >= bsz {
                                    resident_flags[il][e] = true;
                                    rem -= bsz;
                                    total_resident += 1;
                                    gpu_res_count += 1;
                                    placed_any = true;
                                } else {
                                    break 'outer;
                                }
                            }
                        }
                        if !placed_any {
                            break;
                        }
                    }
                } else {
                    let round_budget = budget / denom;
                    let cap_per = round_budget.min(n_exp);
                    for &il in &moe_layers {
                        let bsz = match self.layers[il].ffn {
                            Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                            _ => 0,
                        };
                        for &e in orders[il].iter().take(cap_per) {
                            if rem >= bsz {
                                resident_flags[il][e] = true;
                                rem -= bsz;
                                total_resident += 1;
                                gpu_res_count += 1;
                            }
                        }
                    }
                    // Assign leftover odd budget to additional hot experts
                    'fill: for &il in &moe_layers {
                        let bsz = match self.layers[il].ffn {
                            Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                            _ => 0,
                        };
                        for &e in &orders[il] {
                            if !resident_flags[il][e] {
                                if rem >= bsz {
                                    resident_flags[il][e] = true;
                                    rem -= bsz;
                                    total_resident += 1;
                                    gpu_res_count += 1;
                                } else {
                                    break 'fill;
                                }
                            }
                        }
                    }
                }
            }

            for il in self.cfg.n_dense_layers..n_layer {
                if self.layers[il].device == dev {
                    let bsz = match self.layers[il].ffn {
                        Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                        _ => 0,
                    };
                    let n_cold = resident_flags[il].iter().filter(|&&r| !r).count();
                    total_cold_bytes_per_gpu[gidx] += n_cold * bsz;
                }
            }
            if n_promo > 0 && total_cold_bytes_per_gpu[gidx] > 0 {
                promo_slabs[gidx] = Some(DeviceBuffer::alloc(dev, n_promo * slot_stride)?);
                promo_slot_counts[gidx] = n_promo;
            }
            eprintln!(
                "mm3 gpu{dev}: free={} MiB budget={} MiB static_resident={gpu_res_count} promo_slots={}",
                free >> 20,
                total_budget >> 20,
                promo_slot_counts[gidx]
            );
        }

        self.total_resident_experts = total_resident;
        self.promo_slabs = promo_slabs;
        self.promo_slot_counts = promo_slot_counts;
        self.promo_slot_strides = promo_slot_strides;

        // Allocate single pinned mapped cold slab per GPU + pack in parallel across CPU threads.
        let mut cold_slabs = Vec::new();
        let mut cold_slab_gpu_idx = Vec::new();
        let mut cold_offsets = vec![vec![0usize; n_exp]; n_layer];
        let mut cold_cursor = vec![0usize; self.gpu_ids.len()];

        for (gidx, &cold_bytes) in total_cold_bytes_per_gpu.iter().enumerate() {
            if cold_bytes > 0 {
                let slab = RegisteredMappedHostBuffer::alloc(cold_bytes)?;
                cold_slabs.push(slab);
                cold_slab_gpu_idx.push(gidx);
            }
        }

        let mut cold_tasks: Vec<(usize, usize, usize, usize)> = Vec::new();
        for il in self.cfg.n_dense_layers..n_layer {
            let gidx = self.layers[il].gpu_idx;
            let bsz = match self.layers[il].ffn {
                Mm3FfnWeights::Moe(ref m) => m.blob_bytes,
                _ => 0,
            };
            let slab_idx_opt = cold_slab_gpu_idx.iter().position(|&gi| gi == gidx);
            for e in 0..n_exp {
                if !resident_flags[il][e] {
                    let off = cold_cursor[gidx];
                    cold_offsets[il][e] = off;
                    cold_cursor[gidx] += bsz;
                    if let Some(si) = slab_idx_opt {
                        cold_tasks.push((il, e, si, off));
                    }
                }
            }
        }

        // Parallel copy of cold experts into mapped pinned slabs + prepare per-layer VRAM slabs
        let slab_host_ptrs: Vec<usize> = cold_slabs.iter().map(|s| s.as_ptr() as usize).collect();
        let slab_dev_ptrs: Vec<u64> = cold_slabs
            .iter()
            .zip(cold_slab_gpu_idx.iter())
            .map(|(s, &gi)| s.dev_ptr_for_device(self.gpu_ids[gi]))
            .collect::<Result<Vec<_>>>()?;

        #[derive(Clone, Copy)]
        struct SrcSlice {
            gate: usize,
            up: usize,
            down: usize,
            gb: usize,
            ub: usize,
            db: usize,
            u_off: usize,
            d_off: usize,
        }
        let mut layer_srcs: Vec<Option<SrcSlice>> = vec![None; n_layer];
        for il in self.cfg.n_dense_layers..n_layer {
            if let Mm3FfnWeights::Moe(ref m) = self.layers[il].ffn {
                let g_raw = self
                    .gguf
                    .tensor_bytes(&format!("blk.{il}.ffn_gate_exps.weight"))?;
                let u_raw = self
                    .gguf
                    .tensor_bytes(&format!("blk.{il}.ffn_up_exps.weight"))?;
                let d_raw = self
                    .gguf
                    .tensor_bytes(&format!("blk.{il}.ffn_down_exps.weight"))?;
                layer_srcs[il] = Some(SrcSlice {
                    gate: g_raw.as_ptr() as usize,
                    up: u_raw.as_ptr() as usize,
                    down: d_raw.as_ptr() as usize,
                    gb: m.gate_bytes,
                    ub: m.up_bytes,
                    db: m.down_bytes,
                    u_off: m.u_off,
                    d_off: m.d_off,
                });
            }
        }

        let tasks_ptr = cold_tasks.as_ptr() as usize;
        let tasks_len = cold_tasks.len();
        let srcs_ptr = layer_srcs.as_ptr() as usize;
        let slab_ptrs_u = slab_host_ptrs.as_ptr() as usize;
        pool.parallel_for(tasks_len, |i0, i1| {
            let tasks = unsafe {
                std::slice::from_raw_parts(tasks_ptr as *const (usize, usize, usize, usize), tasks_len)
            };
            let srcs = unsafe {
                std::slice::from_raw_parts(srcs_ptr as *const Option<SrcSlice>, n_layer)
            };
            let slabs = unsafe { std::slice::from_raw_parts(slab_ptrs_u as *const usize, cold_slabs.len()) };
            for &(il, e, si, off) in &tasks[i0..i1] {
                let src = srcs[il].unwrap();
                let dst = (slabs[si] + off) as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping((src.gate + e * src.gb) as *const u8, dst, src.gb);
                    std::ptr::copy_nonoverlapping(
                        (src.up + e * src.ub) as *const u8,
                        dst.add(src.u_off),
                        src.ub,
                    );
                    std::ptr::copy_nonoverlapping(
                        (src.down + e * src.db) as *const u8,
                        dst.add(src.d_off),
                        src.db,
                    );
                }
            }
        });

        // Upload resident experts per MoE layer cleanly using host staging buffer -> single H2D copy.
        for il in self.cfg.n_dense_layers..n_layer {
            let dev = self.layers[il].device;
            let gidx = self.layers[il].gpu_idx;
            set_device(dev)?;
            let Mm3FfnWeights::Moe(ref m) = self.layers[il].ffn else {
                unreachable!();
            };
            let src = layer_srcs[il].unwrap();
            let bsz = m.blob_bytes;
            let res_list: Vec<usize> = (0..n_exp).filter(|&e| resident_flags[il][e]).collect();

            let mut gate_ptrs_host = vec![0u64; n_exp];
            let mut up_ptrs_host = vec![0u64; n_exp];
            let mut down_ptrs_host = vec![0u64; n_exp];
            let mut ram_host_ptrs = vec![0usize; n_exp];
            let mut ram_dev_ptrs = vec![0u64; n_exp];

            let vram_slab = if !res_list.is_empty() {
                let mut host_stage = vec![0u8; res_list.len() * bsz];
                for (slot, &e) in res_list.iter().enumerate() {
                    let dst = &mut host_stage[slot * bsz..(slot + 1) * bsz];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (src.gate + e * src.gb) as *const u8,
                            dst.as_mut_ptr(),
                            src.gb,
                        );
                        std::ptr::copy_nonoverlapping(
                            (src.up + e * src.ub) as *const u8,
                            dst.as_mut_ptr().add(src.u_off),
                            src.ub,
                        );
                        std::ptr::copy_nonoverlapping(
                            (src.down + e * src.db) as *const u8,
                            dst.as_mut_ptr().add(src.d_off),
                            src.db,
                        );
                    }
                }
                let slab = DeviceBuffer::alloc(dev, host_stage.len())?;
                slab.copy_from_host(&host_stage)?;
                let base = slab.as_u64();
                for (slot, &e) in res_list.iter().enumerate() {
                    let bptr = base + (slot * bsz) as u64;
                    gate_ptrs_host[e] = bptr;
                    up_ptrs_host[e] = bptr + src.u_off as u64;
                    down_ptrs_host[e] = bptr + src.d_off as u64;
                }
                Some(slab)
            } else {
                None
            };

            if let Some(si) = cold_slab_gpu_idx.iter().position(|&gi| gi == gidx) {
                let h_base = slab_host_ptrs[si];
                let d_base = slab_dev_ptrs[si];
                for e in 0..n_exp {
                    if !resident_flags[il][e] {
                        let off = cold_offsets[il][e];
                        ram_host_ptrs[e] = h_base + off;
                        ram_dev_ptrs[e] = d_base + off as u64;
                    }
                }
            }

            let dev_gate_ptrs = DeviceBuffer::alloc(dev, n_exp * 8)?;
            dev_gate_ptrs.copy_from_host(bytemuck::cast_slice(&gate_ptrs_host))?;
            let dev_up_ptrs = DeviceBuffer::alloc(dev, n_exp * 8)?;
            dev_up_ptrs.copy_from_host(bytemuck::cast_slice(&up_ptrs_host))?;
            let dev_down_ptrs = DeviceBuffer::alloc(dev, n_exp * 8)?;
            dev_down_ptrs.copy_from_host(bytemuck::cast_slice(&down_ptrs_host))?;

            self.experts[il] = Some(Mm3LayerExperts {
                is_static_resident: resident_flags[il].clone(),
                is_resident: resident_flags[il].clone(),
                promo_slot_of: vec![-1i16; n_exp],
                vram_slab,
                gate_ptrs_host,
                up_ptrs_host,
                down_ptrs_host,
                ram_host_ptrs,
                ram_dev_ptrs,
                dev_gate_ptrs,
                dev_up_ptrs,
                dev_down_ptrs,
            });
        }

        self.cold_slabs = cold_slabs;
        Ok(())
    }
}
