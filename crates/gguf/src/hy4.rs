use crate::format::{GgmlType, GgufError, GgufFile, GgufValue};
use std::path::PathBuf;

/// Engram n-gram lookup configuration extracted from `hy_v4_flash.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Hy4EngramConfig {
    pub layers: Vec<usize>,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub head_dim: usize,
    pub pad_mapped_id: i32,
    pub table_rows: usize,
    pub token_map: Vec<i32>,
    pub multipliers: Vec<u64>,
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub engram_eps: f32,
    pub scales_path: Option<PathBuf>,
}

impl Hy4EngramConfig {
    #[inline]
    pub fn cols(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    #[inline]
    pub fn module_of(&self, il: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == il)
    }

    pub fn from_gguf(gguf: &GgufFile, vocab: usize) -> Result<Option<Self>, GgufError> {
        let arch = "hy_v4_flash";
        let k = |suffix: &str| format!("{arch}.{suffix}");

        let layers_key = k("ple.layers");
        let layers: Vec<usize> = match gguf.metadata.get(&layers_key) {
            Some(v) => v
                .as_array()
                .ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: layers_key.clone(),
                    expected: "array<u32>",
                })?
                .iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect(),
            None => return Ok(None),
        };
        if layers.is_empty() {
            return Ok(None);
        }

        let ngram_size = gguf.get_u32(&k("ple.ngram_size"))? as usize;
        let heads_per_ngram = gguf.get_u32(&k("ple.heads_per_ngram"))? as usize;
        let head_dim = gguf.get_u32(&k("embedding_length_per_layer_input"))? as usize;
        let cols = (ngram_size.saturating_sub(1)) * heads_per_ngram;
        let nm = layers.len();

        let multipliers = gguf.get_u64_array(&k("ple.layer_multipliers"))?;
        let offsets = gguf.get_u64_array(&k("ple.head_offsets"))?;
        let primes = gguf.get_u64_array(&k("ple.head_vocab_sizes"))?;
        let raw_tm = gguf.get_i32_array(&k("engram.token_map"))?;
        let pad_id = gguf.get_u32(&k("engram.pad_token_id"))? as usize;

        if multipliers.len() != nm * ngram_size
            || offsets.len() != nm * cols
            || primes.len() != nm * cols
        {
            return Err(GgufError::MetadataTypeMismatch {
                key: k("ple.layer_multipliers"),
                expected: "consistent Engram table arrays",
            });
        }
        if pad_id >= raw_tm.len() {
            return Err(GgufError::MetadataTypeMismatch {
                key: k("engram.pad_token_id"),
                expected: "pad_token_id inside engram.token_map",
            });
        }

        let info = gguf.tensor_info("per_layer_token_embd.weight")?;
        if info.ggml_type != GgmlType::F8E4m3 || info.dims.len() < 2 || info.dims[0] as usize != head_dim {
            return Err(GgufError::MetadataTypeMismatch {
                key: "per_layer_token_embd.weight".to_owned(),
                expected: "F8_E4M3 [head_dim, table_rows]",
            });
        }
        let table_rows = info.dims[1] as usize;
        for c in 0..(nm * cols) {
            if primes[c] < 2 || (offsets[c] + primes[c]) as usize > table_rows {
                return Err(GgufError::MetadataTypeMismatch {
                    key: k("ple.head_vocab_sizes"),
                    expected: "offsets + primes <= table_rows",
                });
            }
        }

        let mut token_map = Vec::with_capacity(vocab);
        token_map.extend_from_slice(&raw_tm[..raw_tm.len().min(vocab)]);
        if token_map.len() < vocab {
            token_map.resize(vocab, -1);
        }
        let pad_mapped_id = token_map[pad_id];

        // Auto-detect optional external row-scales file `*.engram-scales.bf16`.
        let mut scales_path = None;
        if let Some(dir) = gguf.shard_paths.first().and_then(|p| p.parent()) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut found: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|d| d.path()))
                    .filter(|p| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .is_some_and(|n| n.ends_with(".engram-scales.bf16"))
                    })
                    .collect();
                found.sort();
                if let Some(p) = found.into_iter().next() {
                    if std::fs::metadata(&p).map(|m| m.len() as usize).unwrap_or(0) == table_rows * 2 {
                        scales_path = Some(p);
                    }
                }
            }
        }

        // Default FP32 machine epsilon as in maya-hy4 (`1.1920929e-7f`).
        let engram_eps = std::env::var("AMPLLM_HY4_ENGRAM_EPS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.192_092_9e-7_f32);

        Ok(Some(Self {
            layers,
            ngram_size,
            heads_per_ngram,
            head_dim,
            pad_mapped_id,
            table_rows,
            token_map,
            multipliers,
            primes,
            offsets,
            engram_eps,
            scales_path,
        }))
    }
}

/// SigLIP vision tower + spatial-merge projector configuration from Hy4 `mmproj` GGUF.
#[derive(Debug, Clone)]
pub struct Hy4VisionConfig {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub spatial_merge: usize,
    pub grid: usize,
    pub out_dim: usize,
    pub eps: f32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub image_token_id: u32,
    pub image_start_id: u32,
    pub image_end_id: u32,
}

impl Hy4VisionConfig {
    #[inline]
    pub fn factor(&self) -> usize {
        self.patch_size * self.spatial_merge
    }

    #[inline]
    pub fn image_tokens(&self, width: usize, height: usize) -> usize {
        let f = self.factor();
        if width == 0 || height == 0 || width % f != 0 || height % f != 0 || width * height > self.max_pixels {
            return 0;
        }
        (height / f) * (width / f + 1)
    }

    pub fn from_mmproj(mmproj: &GgufFile, vocab_tokens: &[String], n_embd: usize) -> Result<Self, GgufError> {
        let clip_val = |k: &str| -> Option<&GgufValue> {
            mmproj
                .metadata
                .get(&format!("clip.vision.{k}"))
                .or_else(|| mmproj.metadata.get(&format!("clip.{k}")))
        };
        let get_u = |k: &str, d: Option<usize>| -> Result<usize, GgufError> {
            match clip_val(k).and_then(|v| v.as_u64()) {
                Some(n) => Ok(n as usize),
                None => d.ok_or_else(|| GgufError::MissingMetadata(format!("clip.vision.{k}"))),
            }
        };
        let get_f = |k: &str, d: f32| -> f32 {
            clip_val(k).and_then(|v| v.as_f32()).unwrap_or(d)
        };
        let get_f3 = |k: &str, d: [f32; 3]| -> [f32; 3] {
            if let Some(arr) = clip_val(k).and_then(|v| v.as_array()) {
                if arr.len() >= 3 {
                    return [
                        arr[0].as_f32().unwrap_or(d[0]),
                        arr[1].as_f32().unwrap_or(d[1]),
                        arr[2].as_f32().unwrap_or(d[2]),
                    ];
                }
            }
            d
        };

        let dim = get_u("embedding_length", None)?;
        let heads = get_u("attention.head_count", None)?;
        let ff = get_u("feed_forward_length", None)?;
        let layers = get_u("block_count", None)?;
        let patch_size = get_u("patch_size", None)?;
        let spatial_merge = get_u("spatial_merge_size", Some(2))?;
        let out_dim = get_u("projection_dim", None)?;
        let eps = get_f("attention.layer_norm_epsilon", 1e-5);
        let min_pixels = get_u("image_min_pixels", Some(4096))?;
        let max_pixels = get_u("image_max_pixels", Some(4_194_304))?;
        let image_mean = get_f3("image_mean", [0.5, 0.5, 0.5]);
        let image_std = get_f3("image_std", [0.5, 0.5, 0.5]);

        let pos_info = mmproj.tensor_info("v.position_embd.weight")?;
        let _ = mmproj.tensor_info("v.patch_embd.weight")?;
        let _ = mmproj.tensor_info("mm.0.weight")?;
        let _ = mmproj.tensor_info("v.image_newline")?;

        let pos_rows = pos_info.dims.get(1).copied().unwrap_or(0) as usize;
        let grid = (pos_rows as f64).sqrt().round() as usize;
        if grid * grid != pos_rows || out_dim != n_embd || spatial_merge != 2 || dim % heads != 0 {
            return Err(GgufError::MetadataTypeMismatch {
                key: "clip.vision".to_owned(),
                expected: "matching hy_v4_flash vision geometry",
            });
        }

        let find_tok = |prefix: &str| -> Result<u32, GgufError> {
            let mut hits = vocab_tokens
                .iter()
                .enumerate()
                .filter(|(_, s)| s.starts_with(prefix))
                .map(|(i, _)| i as u32);
            let first = hits.next().ok_or_else(|| GgufError::MissingMetadata(prefix.to_owned()))?;
            if hits.next().is_some() {
                return Err(GgufError::MetadataTypeMismatch {
                    key: prefix.to_owned(),
                    expected: "unique vision token in vocab",
                });
            }
            Ok(first)
        };

        Ok(Self {
            dim,
            heads,
            head_dim: dim / heads,
            ff,
            layers,
            patch_size,
            spatial_merge,
            grid,
            out_dim,
            eps,
            min_pixels,
            max_pixels,
            image_mean,
            image_std,
            image_token_id: find_tok("<｜hy_image▁pad")?,
            image_start_id: find_tok("<｜hy_image▁start")?,
            image_end_id: find_tok("<｜hy_image▁end")?,
        })
    }
}

/// Architecture configuration for Tencent Hy4-Flash (`hy_v4_flash`: iHC-4 + SWA/DSA MLA + LatentMoE + Engram + MTP).
#[derive(Debug, Clone)]
pub struct Hy4Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_mtp: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub context_length: usize,
    pub n_ff: usize,
    pub n_dense_lead: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_expert_shared: usize,
    pub n_expert_latent: usize,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub swiglu_limit: Vec<f32>,
    pub rms_norm_eps: f32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub k_mla: usize,
    pub v_mla: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub n_swa: usize,
    pub heads: Vec<usize>,
    pub swa: Vec<bool>,
    pub indexer: Vec<bool>,
    pub ix_heads: usize,
    pub ix_dim: usize,
    pub ix_top_k: usize,
    pub ix_kpool: usize,
    pub ix_select_tail: bool,
    pub hc: usize,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub engram: Option<Hy4EngramConfig>,
}

impl Hy4Config {
    #[inline]
    pub fn nope(&self) -> usize {
        self.k_mla - self.rope_dim
    }

    #[inline]
    pub fn kv_row(&self) -> usize {
        self.kv_lora_rank + self.rope_dim
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let arch = gguf.get_str("general.architecture")?.to_owned();
        if arch != "hy_v4_flash" {
            return Err(GgufError::MetadataTypeMismatch {
                key: "general.architecture".to_owned(),
                expected: "hy_v4_flash",
            });
        }
        let k = |suffix: &str| format!("{arch}.{suffix}");

        let n_all = gguf.get_u32(&k("block_count"))? as usize;
        let n_mtp = gguf.get_u32(&k("nextn_predict_layers")).unwrap_or(0) as usize;
        let n_layer = n_all.saturating_sub(n_mtp);

        let n_embd = gguf.get_u32(&k("embedding_length"))? as usize;
        let n_vocab = gguf
            .get_u32(&k("vocab_size"))
            .map(|v| v as usize)
            .unwrap_or_else(|_| {
                gguf.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0)
            });
        let context_length = gguf.get_u32(&k("context_length")).unwrap_or(131072) as usize;
        let n_ff = gguf.get_u32(&k("feed_forward_length"))? as usize;
        let n_dense_lead = gguf.get_u32(&k("leading_dense_block_count")).unwrap_or(0) as usize;
        let n_expert = gguf.get_u32(&k("expert_count"))? as usize;
        let n_expert_used = gguf.get_u32(&k("expert_used_count"))? as usize;
        let n_ff_exp = gguf.get_u32(&k("expert_feed_forward_length"))? as usize;
        let n_expert_shared = gguf.get_u32(&k("expert_shared_count")).unwrap_or(1) as usize;
        let n_expert_latent = gguf
            .get_u32(&k("expert_latent_length"))
            .map(|v| if v == 0 { n_embd } else { v as usize })
            .unwrap_or(n_embd);

        let expert_weights_scale = gguf.get_f32(&k("expert_weights_scale")).unwrap_or(1.0);
        let expert_weights_norm = gguf.get_bool(&k("expert_weights_norm")).unwrap_or(false);
        let gating_func = gguf.get_u32(&k("expert_gating_func")).unwrap_or(0);
        if gating_func != 2 {
            return Err(GgufError::MetadataTypeMismatch {
                key: k("expert_gating_func"),
                expected: "2 (sigmoid router)",
            });
        }

        let swiglu_limit = gguf.get_f32_array_or_broadcast(&k("swiglu_clamp_exp"), n_all, 0.0);
        let rms_norm_eps = gguf.get_f32(&k("attention.layer_norm_rms_epsilon"))?;
        let q_lora_rank = gguf.get_u32(&k("attention.q_lora_rank"))? as usize;
        let kv_lora_rank = gguf.get_u32(&k("attention.kv_lora_rank"))? as usize;
        let k_mla = gguf.get_u32(&k("attention.key_length_mla"))? as usize;
        let v_mla = gguf.get_u32(&k("attention.value_length_mla"))? as usize;
        let rope_dim = gguf.get_u32(&k("rope.dimension_count"))? as usize;
        let rope_base = gguf.get_f32(&k("rope.freq_base"))?;
        let rope_base_swa = gguf.get_f32(&k("rope.freq_base_swa"))?;
        let n_swa = gguf.get_u32(&k("attention.sliding_window"))? as usize;

        let heads_u32 = gguf.get_u32_array_or_broadcast(&k("attention.head_count"), n_all, 64);
        let heads: Vec<usize> = heads_u32.into_iter().map(|h| h as usize).collect();

        let kv_heads_arr = gguf
            .get_u32_array(&k("attention.head_count_kv"))
            .unwrap_or_else(|_| vec![1; n_all]);
        if kv_heads_arr.iter().any(|&h| h != 1) {
            return Err(GgufError::MetadataTypeMismatch {
                key: k("attention.head_count_kv"),
                expected: "1 for all MLA layers",
            });
        }

        let mut swa = match gguf.metadata.get(&k("attention.sliding_window_pattern")) {
            Some(v) => v
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| x.as_bool().unwrap_or_else(|| x.as_u64().unwrap_or(0) != 0))
                        .collect::<Vec<bool>>()
                })
                .unwrap_or_default(),
            None => Vec::new(),
        };
        if swa.len() < n_all {
            swa.resize(n_all, false);
        }

        let mut indexer = match gguf.metadata.get(&k("attention.indexer.types")) {
            Some(v) => v
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| x.as_bool().unwrap_or_else(|| x.as_u64().unwrap_or(0) != 0))
                        .collect::<Vec<bool>>()
                })
                .unwrap_or_default(),
            None => Vec::new(),
        };
        // MTP blocks always own an indexer in hy_v4_flash.
        if indexer.len() < n_all {
            indexer.resize(n_all, true);
        }

        for il in 0..n_layer {
            if swa[il] && indexer[il] {
                return Err(GgufError::MetadataTypeMismatch {
                    key: k("attention.indexer.types"),
                    expected: "SWA layer must not have indexer enabled",
                });
            }
        }

        let ix_heads = gguf.get_u32(&k("attention.indexer.head_count")).unwrap_or(0) as usize;
        let ix_dim = gguf.get_u32(&k("attention.indexer.key_length")).unwrap_or(0) as usize;
        let ix_top_k = gguf.get_u32(&k("attention.indexer.top_k")).unwrap_or(0) as usize;
        let ix_kpool = gguf.get_u32(&k("attention.indexer.kpool")).unwrap_or(0) as usize;
        let ix_select_tail = gguf
            .metadata
            .get(&k("attention.indexer.kpool_select_tail"))
            .and_then(|v| v.as_bool().or_else(|| v.as_u64().map(|n| n != 0)))
            .unwrap_or(false);

        let hc = gguf.get_u32(&k("hyper_connection.count"))? as usize;
        let hc_eps = gguf.get_f32(&k("hyper_connection.epsilon"))?;
        let hc_magnitude = gguf.get_f32(&k("hyper_connection.magnitude")).unwrap_or(1.0);

        if n_layer < 1
            || heads.len() != n_all
            || swiglu_limit.len() != n_all
            || hc < 1
            || rope_dim < 2
            || rope_dim % 2 != 0
            || k_mla <= rope_dim
            || kv_lora_rank % 32 != 0
            || n_expert_latent % 256 != 0
            || n_ff_exp % 256 != 0
            || n_embd % 256 != 0
            || n_expert_used > 16
            || ix_kpool < 2
            || !ix_select_tail
            || ix_dim <= rope_dim
            || ix_top_k % ix_kpool != 0
        {
            return Err(GgufError::MetadataTypeMismatch {
                key: "hy_v4_flash".to_owned(),
                expected: "valid hy_v4_flash geometry invariants",
            });
        }

        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id").ok();
        let engram = Hy4EngramConfig::from_gguf(gguf, n_vocab)?;

        Ok(Self {
            arch,
            n_layer,
            n_mtp,
            n_embd,
            n_vocab,
            context_length,
            n_ff,
            n_dense_lead,
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_expert_shared,
            n_expert_latent,
            expert_weights_scale,
            expert_weights_norm,
            swiglu_limit,
            rms_norm_eps,
            q_lora_rank,
            kv_lora_rank,
            k_mla,
            v_mla,
            rope_dim,
            rope_base,
            rope_base_swa,
            n_swa,
            heads,
            swa,
            indexer,
            ix_heads,
            ix_dim,
            ix_top_k,
            ix_kpool,
            ix_select_tail,
            hc,
            hc_eps,
            hc_magnitude,
            bos_token_id,
            eos_token_id,
            engram,
        })
    }
}
