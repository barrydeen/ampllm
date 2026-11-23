use crate::format::{GgmlType, GgufError, GgufFile};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Supported tensor element types in SafeTensors companion files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorsDtype {
    F32,
    BF16,
    F16,
    F8E4M3,
    U8,
    I32,
    I64,
    U16,
    U32,
}

impl SafeTensorsDtype {
    pub fn from_str(s: &str) -> Result<Self, GgufError> {
        match s {
            "F32" => Ok(Self::F32),
            "BF16" => Ok(Self::BF16),
            "F16" => Ok(Self::F16),
            "F8_E4M3" => Ok(Self::F8E4M3),
            "U8" => Ok(Self::U8),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "U16" => Ok(Self::U16),
            "U32" => Ok(Self::U32),
            _ => Err(GgufError::MetadataTypeMismatch {
                key: format!("safetensors.dtype:{s}"),
                expected: "F32, BF16, F16, F8_E4M3, U8, I32, I64, U16, or U32",
            }),
        }
    }

    pub fn elem_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 | Self::F16 | Self::U16 => 2,
            Self::F8E4M3 | Self::U8 => 1,
        }
    }
}

/// Tensor metadata inside a single or sharded SafeTensors map.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: SafeTensorsDtype,
    pub shape: Vec<usize>,
    pub shard_idx: usize,
    pub data_offset: usize,
    pub byte_len: usize,
}

impl SafeTensorInfo {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Read-only memory-mapped single or sharded SafeTensors archive.
pub struct SafeTensorsFile {
    mmaps: Vec<Mmap>,
    pub shard_paths: Vec<PathBuf>,
    pub tensors: Vec<SafeTensorInfo>,
    tensor_index: HashMap<String, usize>,
}

impl SafeTensorsFile {
    /// Open a single `.safetensors` file via `mmap`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Self::open_shards(&[path.as_ref().to_path_buf()])
    }

    /// Open either a `model.safetensors.index.json` or an explicit list of `.safetensors` shards.
    pub fn open_from_path(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let p = path.as_ref();
        if p.extension().and_then(|s| s.to_str()) == Some("json") {
            let txt = std::fs::read_to_string(p)?;
            let root: Value = serde_json::from_str(&txt).map_err(|_| {
                GgufError::MetadataTypeMismatch {
                    key: p.display().to_string(),
                    expected: "valid model.safetensors.index.json",
                }
            })?;
            let wm = root
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| GgufError::MissingMetadata("weight_map".to_owned()))?;
            let parent = p.parent().unwrap_or_else(|| Path::new("."));
            let mut uniq: Vec<PathBuf> = wm
                .values()
                .filter_map(|v| v.as_str())
                .map(|rel| parent.join(rel))
                .collect();
            uniq.sort();
            uniq.dedup();
            return Self::open_shards(&uniq);
        }
        Self::open(p)
    }

    /// Open a list of `.safetensors` files together in a single lookup table.
    pub fn open_shards(paths: &[PathBuf]) -> Result<Self, GgufError> {
        if paths.is_empty() {
            return Err(GgufError::MissingMetadata("safetensors.shards".to_owned()));
        }
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut shard_paths = Vec::with_capacity(paths.len());
        let mut tensors = Vec::new();
        let mut tensor_index = HashMap::new();

        for (shard_idx, path) in paths.iter().enumerate() {
            let file = File::open(path)?;
            // SAFETY: Immutable read-only memory map of model weights.
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            if mmap.len() < 8 {
                return Err(GgufError::UnexpectedEof(mmap.len()));
            }
            let hdr_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let data_start = 8usize.saturating_add(hdr_len);
            if data_start > mmap.len() {
                return Err(GgufError::UnexpectedEof(data_start));
            }
            let hdr_str = std::str::from_utf8(&mmap[8..data_start])
                .map_err(|_| GgufError::InvalidUtf8)?;
            let hdr: serde_json::Map<String, Value> =
                serde_json::from_str(hdr_str).map_err(|_| GgufError::MetadataTypeMismatch {
                    key: path.display().to_string(),
                    expected: "valid SafeTensors JSON header",
                })?;

            for (name, val) in hdr {
                if name == "__metadata__" {
                    continue;
                }
                let obj = val.as_object().ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: name.clone(),
                    expected: "SafeTensors tensor entry object",
                })?;
                let dtype_str = obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| GgufError::MissingMetadata(format!("{name}.dtype")))?;
                let dtype = SafeTensorsDtype::from_str(dtype_str)?;
                let shape_arr = obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| GgufError::MissingMetadata(format!("{name}.shape")))?;
                let mut shape = Vec::with_capacity(shape_arr.len());
                for d in shape_arr {
                    let dim = d.as_u64().ok_or_else(|| GgufError::MetadataTypeMismatch {
                        key: format!("{name}.shape"),
                        expected: "unsigned integer dimension",
                    })?;
                    shape.push(dim as usize);
                }
                let offsets = obj
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| GgufError::MissingMetadata(format!("{name}.data_offsets")))?;
                if offsets.len() != 2 {
                    return Err(GgufError::MetadataTypeMismatch {
                        key: format!("{name}.data_offsets"),
                        expected: "2-element [begin, end] array",
                    });
                }
                let begin = offsets[0].as_u64().ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: format!("{name}.data_offsets[0]"),
                    expected: "u64",
                })? as usize;
                let end = offsets[1].as_u64().ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: format!("{name}.data_offsets[1]"),
                    expected: "u64",
                })? as usize;
                let abs_begin = data_start.saturating_add(begin);
                let byte_len = end.checked_sub(begin).ok_or_else(|| GgufError::TensorOutOfBounds(name.clone()))?;
                if abs_begin.saturating_add(byte_len) > mmap.len() {
                    return Err(GgufError::TensorOutOfBounds(name));
                }
                let info = SafeTensorInfo {
                    name: name.clone(),
                    dtype,
                    shape,
                    shard_idx,
                    data_offset: abs_begin,
                    byte_len,
                };
                if let Some(&prev) = tensor_index.get(&name) {
                    tensors[prev] = info;
                } else {
                    let idx = tensors.len();
                    tensor_index.insert(name, idx);
                    tensors.push(info);
                }
            }
            mmaps.push(mmap);
            shard_paths.push(path.to_path_buf());
        }

        Ok(Self {
            mmaps,
            shard_paths,
            tensors,
            tensor_index,
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    pub fn tensor_info(&self, name: &str) -> Result<&SafeTensorInfo, GgufError> {
        self.tensor_index
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| GgufError::MissingTensor(name.to_owned()))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self.tensor_info(name)?;
        Ok(&self.mmaps[info.shard_idx][info.data_offset..info.data_offset + info.byte_len])
    }

    /// Convert a `F32`, `BF16`, or `F16` SafeTensors tensor into a dense `Vec<f32>`.
    pub fn tensor_to_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self.tensor_info(name)?;
        let bytes = self.tensor_bytes(name)?;
        match info.dtype {
            SafeTensorsDtype::F32 => {
                let mut out = Vec::with_capacity(bytes.len() / 4);
                for ch in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes(ch.try_into().unwrap()));
                }
                Ok(out)
            }
            SafeTensorsDtype::BF16 => {
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for ch in bytes.chunks_exact(2) {
                    out.push(bf16::from_le_bytes([ch[0], ch[1]]).to_f32());
                }
                Ok(out)
            }
            SafeTensorsDtype::F16 => {
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for ch in bytes.chunks_exact(2) {
                    out.push(f16::from_le_bytes([ch[0], ch[1]]).to_f32());
                }
                Ok(out)
            }
            _ => Err(GgufError::MetadataTypeMismatch {
                key: name.to_owned(),
                expected: "F32, BF16, or F16 tensor",
            }),
        }
    }

    /// Reinterpret a `BF16` SafeTensors tensor as `&[u16]` raw bf16 bit patterns (zero-copy when aligned).
    pub fn tensor_bf16_u16(&self, name: &str) -> Result<Vec<u16>, GgufError> {
        let info = self.tensor_info(name)?;
        if info.dtype != SafeTensorsDtype::BF16 {
            return Err(GgufError::MetadataTypeMismatch {
                key: name.to_owned(),
                expected: "BF16 tensor",
            });
        }
        let bytes = self.tensor_bytes(name)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
}

/// Per-layer routed MoE quantization types for MiniMax-M3.1-Flash (since Maya-S uses
/// independent per-layer `gate` / `up` / `down` types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mm3LayerMoeTypes {
    pub gate: GgmlType,
    pub up: GgmlType,
    pub down: GgmlType,
}

/// Parsed hyperparameters for MiniMax-M3.1-Flash text backbone.
#[derive(Debug, Clone)]
pub struct Mm3Config {
    pub n_layers: usize,
    pub n_dense_layers: usize,
    pub max_context_length: usize,
    pub n_embd: usize,
    pub n_dense_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    pub rot_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub n_shared_experts: usize,
    pub expert_ff: usize,
    pub expert_gating_func: u32,
    pub routed_scaling_factor: f32,
    pub expert_weights_norm: bool,
    pub ix_heads: usize,
    pub ix_head_dim: usize,
    pub ix_top_k: usize,
    pub ix_block_size: usize,
    pub ix_local_blocks: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub moe_types: Vec<Option<Mm3LayerMoeTypes>>,
}

impl Mm3Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let arch = gguf.get_str("general.architecture")?;
        let k = |suffix: &str| format!("{arch}.{suffix}");

        let n_layers = gguf.get_u32(&k("block_count"))? as usize;
        let n_dense_layers = gguf
            .metadata
            .get(&k("leading_dense_block_count"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(3);
        let max_context_length = gguf.get_u32(&k("context_length"))? as usize;
        let n_embd = gguf.get_u32(&k("embedding_length"))? as usize;
        let n_dense_ff = gguf.get_u32(&k("feed_forward_length"))? as usize;
        let n_heads = gguf.get_u32(&k("attention.head_count"))? as usize;
        let n_kv_heads = gguf.get_u32(&k("attention.head_count_kv"))? as usize;
        let head_dim = gguf
            .metadata
            .get(&k("attention.key_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);
        let v_head_dim = gguf
            .metadata
            .get(&k("attention.value_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let rot_dim = gguf
            .metadata
            .get(&k("rope.dimension_count"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(64);
        let rope_theta = gguf.get_f32(&k("rope.freq_base"))?;
        let rms_norm_eps = gguf.get_f32(&k("attention.layer_norm_rms_epsilon"))?;

        let vocab_size = if let Ok(info) = gguf.tensor_info("token_embd.weight") {
            *info.dims.last().unwrap_or(&0) as usize
        } else {
            gguf.metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .ok_or_else(|| GgufError::MissingMetadata("vocab_size".to_owned()))?
        };

        let n_experts = gguf.get_u32(&k("expert_count"))? as usize;
        let n_experts_used = gguf.get_u32(&k("expert_used_count"))? as usize;
        let n_shared_experts = gguf
            .metadata
            .get(&k("expert_shared_count"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);
        let expert_ff = gguf.get_u32(&k("expert_feed_forward_length"))? as usize;
        let expert_gating_func = gguf
            .metadata
            .get(&k("expert_gating_func"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(2);
        let routed_scaling_factor = gguf
            .metadata
            .get(&k("expert_weights_scale"))
            .and_then(|v| v.as_f32())
            .unwrap_or(2.0);
        let expert_weights_norm = gguf
            .metadata
            .get(&k("expert_weights_norm"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let ix_heads = gguf
            .metadata
            .get(&k("attention.indexer.head_count"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);
        let ix_head_dim = gguf
            .metadata
            .get(&k("attention.indexer.key_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);
        let ix_top_k = gguf
            .metadata
            .get(&k("attention.indexer.top_k"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);
        let ix_block_size = gguf
            .metadata
            .get(&k("attention.indexer.block_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(128);
        let ix_local_blocks = gguf
            .metadata
            .get(&k("attention.indexer.local_blocks"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let mut moe_types = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let gk = format!("blk.{il}.ffn_gate_exps.weight");
            if let Ok(g_info) = gguf.tensor_info(&gk) {
                let u_info = gguf.tensor_info(&format!("blk.{il}.ffn_up_exps.weight"))?;
                let d_info = gguf.tensor_info(&format!("blk.{il}.ffn_down_exps.weight"))?;
                moe_types.push(Some(Mm3LayerMoeTypes {
                    gate: g_info.ggml_type,
                    up: u_info.ggml_type,
                    down: d_info.ggml_type,
                }));
            } else {
                moe_types.push(None);
            }
        }

        Ok(Self {
            n_layers,
            n_dense_layers,
            max_context_length,
            n_embd,
            n_dense_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
            rot_dim,
            rope_theta,
            rms_norm_eps,
            vocab_size,
            n_experts,
            n_experts_used,
            n_shared_experts,
            expert_ff,
            expert_gating_func,
            routed_scaling_factor,
            expert_weights_norm,
            ix_heads,
            ix_head_dim,
            ix_top_k,
            ix_block_size,
            ix_local_blocks,
            swiglu_alpha: 1.702,
            swiglu_limit: 7.0,
            moe_types,
        })
    }
}

/// Parsed hyperparameters for MiniMax-M3.1-Flash vision tower (`config.json` + `preprocessor_config.json`).
#[derive(Debug, Clone)]
pub struct Mm3VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub rope_theta: f32,
    pub layer_norm_eps: f32,
    pub image_size: usize,
    pub max_image_resolution: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub image_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub video_token_id: u32,
}

impl Mm3VisionConfig {
    pub fn from_companion_jsons(
        config_path: &Path,
        preproc_path: Option<&Path>,
    ) -> Result<Self, GgufError> {
        let cfg_txt = std::fs::read_to_string(config_path)?;
        let cfg: Value = serde_json::from_str(&cfg_txt).map_err(|_| {
            GgufError::MetadataTypeMismatch {
                key: config_path.display().to_string(),
                expected: "valid MiniMax-M3 companions config.json",
            }
        })?;
        let vc = cfg
            .get("vision_config")
            .ok_or_else(|| GgufError::MissingMetadata("vision_config".to_owned()))?;
        let comp = vc.get("img_token_compression_config");

        let hidden_size = vc.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1280) as usize;
        let num_heads = vc
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let head_dim = hidden_size / num_heads.max(1);
        let num_layers = vc
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let intermediate_size = vc
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(5120) as usize;
        let projection_dim = vc
            .get("projection_dim")
            .and_then(|v| v.as_u64())
            .or_else(|| cfg.get("projector_hidden_size").and_then(|v| v.as_u64()))
            .unwrap_or(6144) as usize;
        let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as usize;
        let spatial_merge_size = comp
            .and_then(|c| c.get("spatial_merge_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let temporal_patch_size = comp
            .and_then(|c| c.get("temporal_patch_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let rope_theta = vc.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
        let layer_norm_eps = vc
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;
        let image_size = vc.get("image_size").and_then(|v| v.as_u64()).unwrap_or(672) as usize;

        let image_token_id = cfg
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(200025) as u32;
        let video_token_id = cfg
            .get("video_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(200026) as u32;

        let mut max_image_resolution = image_size;
        let mut image_mean = [0.48145466f32, 0.4578275, 0.40821073];
        let mut image_std = [0.26862954f32, 0.26130258, 0.27577711];

        if let Some(pp) = preproc_path {
            if pp.exists() {
                let pt = std::fs::read_to_string(pp)?;
                if let Ok(pv) = serde_json::from_str::<Value>(&pt) {
                    if let Some(arr) = pv.get("image_mean").and_then(|v| v.as_array()) {
                        if arr.len() == 3 {
                            for i in 0..3 {
                                if let Some(x) = arr[i].as_f64() {
                                    image_mean[i] = x as f32;
                                }
                            }
                        }
                    }
                    if let Some(arr) = pv.get("image_std").and_then(|v| v.as_array()) {
                        if arr.len() == 3 {
                            for i in 0..3 {
                                if let Some(x) = arr[i].as_f64() {
                                    image_std[i] = x as f32;
                                }
                            }
                        }
                    }
                    if let Some(mr) = pv
                        .get("img_token_compression_config")
                        .and_then(|c| c.get("max_image_resolution"))
                        .and_then(|v| v.as_u64())
                    {
                        max_image_resolution = mr as usize;
                    }
                }
            }
        }

        let unit = patch_size * spatial_merge_size;
        let min_pixels = (2 * unit) * (2 * unit); // 3136 when unit=28
        let max_pixels = image_size * image_size; // 451584 when image_size=672

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            num_layers,
            intermediate_size,
            projection_dim,
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            rope_theta,
            layer_norm_eps,
            image_size,
            max_image_resolution,
            min_pixels,
            max_pixels,
            image_mean,
            image_std,
            image_token_id,
            image_start_token_id: 200029,
            image_end_token_id: 200030,
            video_token_id,
        })
    }
}

/// Parsed hyperparameters for MiniMax-M3.1-Flash DSpark block-diffusion drafter.
#[derive(Debug, Clone)]
pub struct Mm3DsparkConfig {
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub mask_token_id: u32,
    pub sliding_window: usize,
    pub target_layer_ids: Vec<usize>,
    pub enable_confidence_head: bool,
    pub confidence_head_with_markov: bool,
}

impl Mm3DsparkConfig {
    pub fn from_json(path: &Path) -> Result<Self, GgufError> {
        let txt = std::fs::read_to_string(path)?;
        let root: Value = serde_json::from_str(&txt).map_err(|_| {
            GgufError::MetadataTypeMismatch {
                key: path.display().to_string(),
                expected: "valid dspark/config.json",
            }
        })?;
        let tc = root.get("text_config");
        let df = root.get("dflash_config");

        let get_usize = |key: &str, def: usize| -> usize {
            root.get(key)
                .and_then(|v| v.as_u64())
                .or_else(|| tc.and_then(|t| t.get(key)).and_then(|v| v.as_u64()))
                .or_else(|| df.and_then(|d| d.get(key)).and_then(|v| v.as_u64()))
                .map(|v| v as usize)
                .unwrap_or(def)
        };

        let hidden_size = get_usize("hidden_size", 6144);
        let ffn_hidden_size = root
            .get("dspark_ffn_hidden_size")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                tc.and_then(|t| t.get("dense_intermediate_size"))
                    .and_then(|v| v.as_u64())
            })
            .map(|v| v as usize)
            .unwrap_or(12288);
        let num_layers = get_usize("num_hidden_layers", 5);
        let num_heads = tc
            .and_then(|t| t.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(64);
        let num_kv_heads = tc
            .and_then(|t| t.get("num_key_value_heads"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);
        let head_dim = get_usize("head_dim", 128);
        let rot_dim = get_usize("rotary_dim", 64);
        let rope_theta = tc
            .and_then(|t| t.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e7) as f32;
        let rms_norm_eps = tc
            .and_then(|t| t.get("rms_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let vocab_size = get_usize("vocab_size", 200064);
        let block_size = root
            .get("dspark_block_size")
            .and_then(|v| v.as_u64())
            .or_else(|| df.and_then(|d| d.get("block_size")).and_then(|v| v.as_u64()))
            .map(|v| v as usize)
            .unwrap_or(7);
        let markov_rank = root
            .get("dspark_markov_rank")
            .and_then(|v| v.as_u64())
            .or_else(|| root.get("markov_rank").and_then(|v| v.as_u64()))
            .map(|v| v as usize)
            .unwrap_or(256);
        let mask_token_id = root
            .get("dspark_noise_token_id")
            .and_then(|v| v.as_u64())
            .or_else(|| df.and_then(|d| d.get("mask_token_id")).and_then(|v| v.as_u64()))
            .unwrap_or(200058) as u32;
        let sliding_window = df
            .and_then(|d| d.get("sliding_window"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096);

        let target_layer_ids = root
            .get("dspark_target_layer_ids")
            .or_else(|| df.and_then(|d| d.get("target_layer_ids")))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![3, 17, 31, 45, 59]);

        let enable_confidence_head = root
            .get("enable_confidence_head")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let confidence_head_with_markov = root
            .get("confidence_head_with_markov")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Self {
            hidden_size,
            ffn_hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rot_dim,
            rope_theta,
            rms_norm_eps,
            vocab_size,
            block_size,
            markov_rank,
            mask_token_id,
            sliding_window,
            target_layer_ids,
            enable_confidence_head,
            confidence_head_with_markov,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_safetensors(path: &Path) {
        let bf0 = bf16::from_f32(1.5).to_le_bytes();
        let bf1 = bf16::from_f32(-2.25).to_le_bytes();
        let u8s = [127u8, 128u8];
        let payload = [bf0[0], bf0[1], bf1[0], bf1[1], u8s[0], u8s[1]];
        let hdr = serde_json::json!({
            "w_bf16": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] },
            "w_u8": { "dtype": "U8", "shape": [2], "data_offsets": [4, 6] }
        })
        .to_string();
        let hdr_bytes = hdr.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(hdr_bytes);
        buf.extend_from_slice(&payload);
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn test_safetensors_synthetic_roundtrip() {
        let tmp = std::env::temp_dir().join("ampllm_test_mm3_st.safetensors");
        write_test_safetensors(&tmp);
        let st = SafeTensorsFile::open(&tmp).unwrap();
        assert!(st.has_tensor("w_bf16"));
        assert_eq!(st.tensor_info("w_u8").unwrap().dtype, SafeTensorsDtype::U8);
        assert_eq!(st.tensor_to_f32("w_bf16").unwrap(), vec![1.5f32, -2.25f32]);
        assert_eq!(st.tensor_bytes("w_u8").unwrap(), &[127u8, 128u8]);
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn test_mm3_real_files_if_present() {
        let Ok(gguf_str) = std::env::var("AMPLLM_MM3_MODEL") else {
            return;
        };
        let gguf_path = Path::new(&gguf_str);
        if !gguf_path.exists() {
            return;
        }
        let aux = crate::format::discover_mm3_aux_files(gguf_path, None, None).unwrap();
        let mut gguf = GgufFile::open(gguf_path).unwrap();
        assert_eq!(gguf.architecture().unwrap(), crate::format::ModelArch::MinimaxM3);
        let idx_bin = aux.indexers_bin.expect("missing indexers bin");
        gguf.merge_aux_shard(&idx_bin).unwrap();
        assert!(gguf.has_tensor("blk.0.indexer.q_proj.weight"));
        assert!(gguf.has_tensor("blk.2.indexer.k_norm.weight"));

        let cfg = Mm3Config::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.n_layers, 60);
        assert_eq!(cfg.n_dense_layers, 3);
        assert_eq!(cfg.n_embd, 6144);
        assert_eq!(cfg.n_experts, 128);
        assert_eq!(cfg.n_experts_used, 4);
        assert_eq!(cfg.moe_types[22].unwrap().gate, GgmlType::Iq2Xs);
        assert_eq!(cfg.moe_types[22].unwrap().up, GgmlType::Iq2Xxs);
        assert_eq!(cfg.moe_types[22].unwrap().down, GgmlType::Iq3Xxs);

        let vcfg = Mm3VisionConfig::from_companion_jsons(
            aux.vision_config_json.as_ref().unwrap(),
            aux.preprocessor_json.as_deref(),
        )
        .unwrap();
        assert_eq!(vcfg.hidden_size, 1280);
        assert_eq!(vcfg.num_layers, 32);

        let dcfg = Mm3DsparkConfig::from_json(aux.dspark_config_json.as_ref().unwrap()).unwrap();
        assert_eq!(dcfg.num_layers, 5);
        assert_eq!(dcfg.block_size, 7);
        assert_eq!(dcfg.mask_token_id, 200058);

        let vis_st = SafeTensorsFile::open(aux.vision_safetensors.as_ref().unwrap()).unwrap();
        assert!(vis_st.has_tensor("patch_merge_mlp.linear_1.weight"));
        let dspark_st = SafeTensorsFile::open_from_path(aux.dspark_safetensors.as_ref().unwrap()).unwrap();
        assert!(dspark_st.has_tensor("language_model.model.dspark.fc.weight"));
    }
}
