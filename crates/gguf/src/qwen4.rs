use crate::format::{GgmlType, GgufError, GgufFile, GgufValue};

/// Per-Layer N-gram Embedding (PLE) configuration extracted from `qwen4exp.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Qwen4PleConfig {
    pub layers: Vec<usize>,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub head_dim: usize,
    pub eos_token_id: u32,
    pub image_token_id: u32,
    pub table_rows: usize,
    pub table_type: GgmlType,
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
}

impl Qwen4PleConfig {
    #[inline]
    pub fn n_heads(&self) -> usize {
        self.ngram_size.saturating_sub(1) * self.heads_per_ngram
    }

    #[inline]
    pub fn dilation(&self) -> usize {
        self.ngram_size
    }

    #[inline]
    pub fn hist_len(&self) -> usize {
        self.conv_kernel.saturating_sub(1) * self.ngram_size
    }

    #[inline]
    pub fn total_dim(&self) -> usize {
        self.n_heads() * self.head_dim
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>, GgufError> {
        let arch = gguf.get_str("general.architecture")?;
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
        let conv_kernel = gguf.get_u32(&k("ple.conv_kernel")).unwrap_or(4) as usize;
        let head_dim = gguf.get_u32(&k("embedding_length_per_layer_input"))? as usize;
        let eos_token_id = gguf
            .get_u32(&k("ple.eos_token_id"))
            .or_else(|_| gguf.get_u32("tokenizer.ggml.bos_token_id"))
            .unwrap_or(248044);
        let image_token_id = gguf.get_u32(&k("ple.image_token_id")).unwrap_or(248056);

        let multipliers = gguf.get_u64_array(&k("ple.layer_multipliers"))?;
        let head_offsets = gguf.get_u64_array(&k("ple.head_offsets"))?;
        let head_vocab_sizes = gguf.get_u64_array(&k("ple.head_vocab_sizes"))?;

        let n_heads = ngram_size.saturating_sub(1) * heads_per_ngram;
        let expect_len = layers.len() * n_heads;
        if multipliers.len() != layers.len() * ngram_size
            || head_offsets.len() != expect_len
            || head_vocab_sizes.len() != expect_len
        {
            return Err(GgufError::MetadataTypeMismatch {
                key: k("ple.head_offsets"),
                expected: "consistent Qwen4 PLE arrays matching layers * n_heads",
            });
        }

        let info = gguf.tensor_info("per_layer_token_embd.weight")?;
        if info.dims.len() < 2 || info.dims[0] as usize != head_dim {
            return Err(GgufError::MetadataTypeMismatch {
                key: "per_layer_token_embd.weight".to_owned(),
                expected: "[head_dim, table_rows] PLE table",
            });
        }
        let table_rows = info.dims[1] as usize;
        let table_type = info.ggml_type;
        for i in 0..expect_len {
            if head_vocab_sizes[i] < 2 || (head_offsets[i] + head_vocab_sizes[i]) as usize > table_rows {
                return Err(GgufError::MetadataTypeMismatch {
                    key: k("ple.head_vocab_sizes"),
                    expected: "head_offsets + head_vocab_sizes <= table_rows",
                });
            }
        }

        Ok(Some(Self {
            layers,
            ngram_size,
            heads_per_ngram,
            conv_kernel,
            head_dim,
            eos_token_id,
            image_token_id,
            table_rows,
            table_type,
            multipliers,
            head_offsets,
            head_vocab_sizes,
        }))
    }
}

/// Qwen3-VL SigLIP Vision + spatial-merge projector configuration from `mmproj` GGUF.
#[derive(Debug, Clone)]
pub struct Qwen4VisionConfig {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub spatial_merge: usize,
    pub image_size: usize,
    pub grid: usize,
    pub out_dim: usize,
    pub eps: f32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub image_token_id: u32,
    pub vision_start_id: u32,
    pub vision_end_id: u32,
}

impl Qwen4VisionConfig {
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
        (height / f) * (width / f)
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
        let image_size = get_u("image_size", Some(768))?;
        let out_dim = get_u("projection_dim", None)?;
        let eps = get_f("attention.layer_norm_epsilon", 1e-6);
        let min_pixels = get_u("image_min_pixels", Some(4 * 32 * 32))?;
        let max_pixels = get_u("image_max_pixels", Some(2048 * 2048))?;
        let image_mean = get_f3("image_mean", [0.5, 0.5, 0.5]);
        let image_std = get_f3("image_std", [0.5, 0.5, 0.5]);

        let pos_info = mmproj.tensor_info("v.position_embd.weight")?;
        let _ = mmproj.tensor_info("v.patch_embd.weight")?;
        let _ = mmproj.tensor_info("mm.0.weight")?;
        let _ = mmproj.tensor_info("mm.2.weight")?;
        let _ = mmproj.tensor_info("v.post_ln.weight")?;

        let pos_rows = pos_info.dims.get(1).copied().unwrap_or(0) as usize;
        let grid = (pos_rows as f64).sqrt().round() as usize;
        if grid * grid != pos_rows || out_dim != n_embd || spatial_merge != 2 || dim % heads != 0 {
            return Err(GgufError::MetadataTypeMismatch {
                key: "clip.vision".to_owned(),
                expected: "matching qwen4exp vision geometry",
            });
        }

        let find_tok = |exact: &str, default_id: u32| -> u32 {
            vocab_tokens
                .iter()
                .position(|s| s == exact)
                .map(|i| i as u32)
                .unwrap_or(default_id)
        };

        Ok(Self {
            dim,
            heads,
            head_dim: dim / heads,
            ff,
            layers,
            patch_size,
            spatial_merge,
            image_size,
            grid,
            out_dim,
            eps,
            min_pixels,
            max_pixels,
            image_mean,
            image_std,
            image_token_id: find_tok("<|image_pad|>", 248056),
            vision_start_id: find_tok("<|vision_start|>", 248053),
            vision_end_id: find_tok("<|vision_end|>", 248054),
        })
    }
}

/// Architecture configuration for Qwen3.8-Flash-Next (`qwen4exp`: HC-4 + GDN/QSA + Indexer + PLE + MoE).
#[derive(Debug, Clone)]
pub struct Qwen4Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub context_length: usize,
    pub rms_norm_eps: f32,
    pub hc: usize,
    pub hc_low_rank: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub full_attention_interval: usize,
    pub compress_ratios: Vec<i32>,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub rope_sections: [usize; 4],
    pub idx_heads: usize,
    pub idx_key_dim: usize,
    pub idx_top_k: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shared: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub ple: Option<Qwen4PleConfig>,
}

impl Qwen4Config {
    #[inline]
    pub fn is_qsa(&self, il: usize) -> bool {
        self.compress_ratios.get(il).copied().unwrap_or(0) > 0
    }

    #[inline]
    pub fn compress_ratio(&self, il: usize) -> usize {
        self.compress_ratios.get(il).copied().unwrap_or(0).max(0) as usize
    }

    #[inline]
    pub fn gdn_qkv_dim(&self) -> usize {
        2 * self.ssm_group_count * self.ssm_state_size + self.ssm_inner_size
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let arch = gguf.get_str("general.architecture")?.to_owned();
        if arch != "qwen4exp" && arch != "qwen4_exp" {
            return Err(GgufError::MetadataTypeMismatch {
                key: "general.architecture".to_owned(),
                expected: "qwen4exp",
            });
        }
        let k = |suffix: &str| format!("{arch}.{suffix}");

        let n_layer = gguf.get_u32(&k("block_count"))? as usize;
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
        let context_length = gguf.get_u32(&k("context_length")).unwrap_or(262144) as usize;
        let rms_norm_eps = gguf.get_f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

        let hc = gguf.get_u32(&k("hyper_connection.count")).unwrap_or(4) as usize;
        let hc_low_rank = gguf.get_u32(&k("hyper_connection.low_rank")).unwrap_or(320) as usize;

        let n_head = gguf.get_u32(&k("attention.head_count"))? as usize;
        let n_head_kv = gguf.get_u32(&k("attention.head_count_kv"))? as usize;
        let head_dim = gguf.get_u32(&k("attention.key_length"))? as usize;
        let full_attention_interval = gguf.get_u32(&k("full_attention_interval")).unwrap_or(4) as usize;
        let compress_ratios = gguf
            .get_i32_array(&k("attention.compress_ratios"))
            .unwrap_or_else(|_| {
                (0..n_layer)
                    .map(|l| if (l + 1) % full_attention_interval == 0 { 4 } else { 0 })
                    .collect()
            });
        if compress_ratios.len() != n_layer {
            return Err(GgufError::MetadataTypeMismatch {
                key: k("attention.compress_ratios"),
                expected: "array length == block_count",
            });
        }

        let rope_dim = gguf.get_u32(&k("rope.dimension_count")).unwrap_or(64) as usize;
        let rope_base = gguf.get_f32(&k("rope.freq_base")).unwrap_or(10_000_000.0);
        let mut rope_sections = [11usize, 11, 10, 0];
        if let Some(arr) = gguf.metadata.get(&k("rope.dimension_sections")).and_then(|v| v.as_array()) {
            for i in 0..arr.len().min(4) {
                if let Some(v) = arr[i].as_u64() {
                    rope_sections[i] = v as usize;
                }
            }
        }

        let idx_heads = gguf.get_u32(&k("attention.indexer.head_count")).unwrap_or(4) as usize;
        let idx_key_dim = gguf.get_u32(&k("attention.indexer.key_length")).unwrap_or(128) as usize;
        let idx_top_k = gguf.get_u32(&k("attention.indexer.top_k")).unwrap_or(2048) as usize;

        let ssm_conv_kernel = gguf.get_u32(&k("ssm.conv_kernel")).unwrap_or(4) as usize;
        let ssm_state_size = gguf.get_u32(&k("ssm.state_size")).unwrap_or(128) as usize;
        let ssm_group_count = gguf.get_u32(&k("ssm.group_count")).unwrap_or(16) as usize;
        let ssm_time_step_rank = gguf.get_u32(&k("ssm.time_step_rank")).unwrap_or(48) as usize;
        let ssm_inner_size = gguf.get_u32(&k("ssm.inner_size")).unwrap_or(6144) as usize;

        let n_expert = gguf.get_u32(&k("expert_count"))? as usize;
        let n_expert_used = gguf.get_u32(&k("expert_used_count"))? as usize;
        let n_ff_exp = gguf.get_u32(&k("expert_feed_forward_length"))? as usize;
        let n_ff_shared = gguf
            .get_u32(&k("expert_shared_feed_forward_length"))
            .unwrap_or(n_ff_exp as u32) as usize;

        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id").ok();
        let pad_token_id = gguf.get_u32("tokenizer.ggml.padding_token_id").ok();
        let ple = Qwen4PleConfig::from_gguf(gguf)?;

        Ok(Self {
            arch,
            n_layer,
            n_embd,
            n_vocab,
            context_length,
            rms_norm_eps,
            hc,
            hc_low_rank,
            n_head,
            n_head_kv,
            head_dim,
            full_attention_interval,
            compress_ratios,
            rope_dim,
            rope_base,
            rope_sections,
            idx_heads,
            idx_key_dim,
            idx_top_k,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_ff_shared,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            ple,
        })
    }
}
