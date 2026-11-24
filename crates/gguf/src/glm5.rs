use crate::format::{GgufError, GgufFile, GgufValue};

/// Architecture configuration for GLM-5.3-Flash (`glm5-next`: HC-4 + Hybrid KDA/DSA + MoE + NextN MTP).
#[derive(Debug, Clone)]
pub struct Glm5Config {
    pub arch: String,
    pub n_layer_total: usize,
    pub n_layers: usize,
    pub nextn_predict_layers: usize,
    pub has_mtp: bool,
    pub mtp_layer: Option<usize>,
    pub n_embd: usize,
    pub n_ff_dense: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub head_count_kv: Vec<u32>,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub context_length: usize,
    // Hyper-Connections
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    // KDA (Key-Driven Linear Attention)
    pub kda_head_dim: usize,
    pub d_conv: usize,
    pub kda_gate_lb: f32,
    // DSA (NoPE MLA + Lightning Indexer)
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_dim: usize,
    pub idx_heads: usize,
    pub idx_key_dim: usize,
    pub idx_top_k: usize,
    pub idx_kpool: usize,
    pub idx_select_tail: bool,
    // MoE / FFN
    pub leading_dense_block_count: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub n_shared_expert: usize,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub swiglu_clamp_exp: f32,
    pub swiglu_clamp_shexp: f32,
}

impl Glm5Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let arch = gguf.get_str("general.architecture")?.to_owned();
        if !arch.starts_with("glm5") {
            return Err(GgufError::MetadataTypeMismatch {
                key: "general.architecture".to_owned(),
                expected: "glm5*",
            });
        }
        let k = |suffix: &str| format!("{arch}.{suffix}");

        let raw_block_count = gguf.get_u32(&k("block_count"))? as usize;
        let nextn_predict_layers = gguf.get_u32(&k("nextn_predict_layers")).unwrap_or(0) as usize;
        let n_layers = raw_block_count.saturating_sub(nextn_predict_layers);
        let mtp_name = format!("blk.{n_layers}.nextn.eh_proj.weight");
        let has_mtp = nextn_predict_layers > 0 && gguf.has_tensor(&mtp_name);
        let mtp_layer = if has_mtp { Some(n_layers) } else { None };

        let n_embd = gguf.get_u32(&k("embedding_length"))? as usize;
        let n_ff_dense = gguf.get_u32(&k("feed_forward_length"))? as usize;
        let n_vocab = gguf.get_u32(&k("vocab_size"))? as usize;
        let n_head = gguf.get_u32_or_first(&k("attention.head_count"))? as usize;
        let head_count_kv = gguf
            .get_u32_array(&k("attention.head_count_kv"))
            .unwrap_or_else(|_| {
                let mut v = Vec::with_capacity(raw_block_count);
                for l in 0..raw_block_count {
                    v.push(if l % 4 == 3 || l == n_layers { 1 } else { 0 });
                }
                v
            });

        let rms_norm_eps = gguf.get_f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let layer_norm_eps = gguf.get_f32(&k("attention.layer_norm_epsilon")).unwrap_or(1e-6);
        let context_length = gguf.get_u32(&k("context_length")).unwrap_or(131072) as usize;

        let hc = gguf.get_u32(&k("hyper_connection.count")).unwrap_or(4) as usize;
        let sinkhorn_iters = gguf
            .get_u32(&k("hyper_connection.sinkhorn_iterations"))
            .unwrap_or(20) as usize;
        let hc_eps = gguf.get_f32(&k("hyper_connection.epsilon")).unwrap_or(1e-6);

        let kda_head_dim = gguf.get_u32(&k("kda.head_dim")).unwrap_or(128) as usize;
        let d_conv = gguf.get_u32(&k("ssm.conv_kernel")).unwrap_or(4) as usize;
        let kda_gate_lb = gguf.get_f32(&k("kda.gate_lower_bound")).unwrap_or(-5.0);

        let q_lora_rank = gguf.get_u32(&k("attention.q_lora_rank")).unwrap_or(1536) as usize;
        let kv_lora_rank = gguf.get_u32(&k("attention.kv_lora_rank")).unwrap_or(512) as usize;
        let qk_nope_head_dim = gguf.get_u32(&k("attention.key_length_mla")).unwrap_or(256) as usize;
        let v_head_dim = gguf.get_u32(&k("attention.value_length_mla")).unwrap_or(256) as usize;
        let rope_dim = gguf.get_u32(&k("rope.dimension_count")).unwrap_or(0) as usize;

        let idx_heads = gguf.get_u32(&k("attention.indexer.head_count")).unwrap_or(32) as usize;
        let idx_key_dim = gguf.get_u32(&k("attention.indexer.key_length")).unwrap_or(128) as usize;
        let idx_top_k = gguf.get_u32(&k("attention.indexer.top_k")).unwrap_or(2048) as usize;
        let idx_kpool = gguf.get_u32(&k("attention.indexer.kpool")).unwrap_or(4) as usize;

        let leading_dense_block_count = gguf
            .get_u32(&k("leading_dense_block_count"))
            .unwrap_or(3) as usize;
        let n_expert = gguf.get_u32(&k("expert_count")).unwrap_or(288) as usize;
        let n_expert_used = gguf.get_u32(&k("expert_used_count")).unwrap_or(8) as usize;
        let n_ff_exp = gguf.get_u32_or_first(&k("expert_feed_forward_length")).unwrap_or(2048) as usize;
        let n_ff_shexp = gguf
            .get_u32_or_first(&k("expert_shared_feed_forward_length"))
            .unwrap_or(2048) as usize;
        let n_shared_expert = gguf.get_u32(&k("expert_shared_count")).unwrap_or(1) as usize;
        let expert_weights_scale = gguf.get_f32(&k("expert_weights_scale")).unwrap_or(2.5);
        let expert_weights_norm = gguf.get_bool(&k("expert_weights_norm")).unwrap_or(true);
        let swiglu_clamp_exp = gguf.get_f32_or_first(&k("swiglu_clamp_exp")).unwrap_or(10.0);
        let swiglu_clamp_shexp = gguf.get_f32_or_first(&k("swiglu_clamp_shexp")).unwrap_or(10.0);

        Ok(Self {
            arch,
            n_layer_total: raw_block_count,
            n_layers,
            nextn_predict_layers,
            has_mtp,
            mtp_layer,
            n_embd,
            n_ff_dense,
            n_vocab,
            n_head,
            head_count_kv,
            rms_norm_eps,
            layer_norm_eps,
            context_length,
            hc,
            sinkhorn_iters,
            hc_eps,
            kda_head_dim,
            d_conv,
            kda_gate_lb,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            v_head_dim,
            rope_dim,
            idx_heads,
            idx_key_dim,
            idx_top_k,
            idx_kpool,
            idx_select_tail: true,
            leading_dense_block_count,
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_ff_shexp,
            n_shared_expert,
            expert_weights_scale,
            expert_weights_norm,
            swiglu_clamp_exp,
            swiglu_clamp_shexp,
        })
    }

    #[inline]
    pub fn is_kda(&self, layer: usize) -> bool {
        self.head_count_kv.get(layer).copied().unwrap_or(0) == 0
    }

    #[inline]
    pub fn is_dsa(&self, layer: usize) -> bool {
        !self.is_kda(layer)
    }

    #[inline]
    pub fn is_moe(&self, layer: usize) -> bool {
        layer >= self.leading_dense_block_count
    }

    #[inline]
    pub fn d_inner(&self) -> usize {
        self.n_head * self.kda_head_dim
    }
}

/// GLM-5.3-Flash (`glm5next`) Vision Tower + SwiGLU merger configuration from `mmproj` GGUF.
#[derive(Debug, Clone)]
pub struct Glm5VisionConfig {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub spatial_merge: usize,
    pub image_size: usize,
    pub out_dim: usize,
    pub merger_ff: usize,
    pub eps: f32,
    pub ffn_clamp: f32,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub image_token_id: u32,
    pub vision_start_id: u32,
    pub vision_end_id: u32,
}

impl Glm5VisionConfig {
    #[inline]
    pub fn factor(&self) -> usize {
        self.patch_size * self.spatial_merge
    }

    #[inline]
    pub fn image_tokens(&self, width: usize, height: usize) -> usize {
        let f = self.factor();
        if width == 0 || height == 0 || width % f != 0 || height % f != 0 {
            return 0;
        }
        let n_tok = (height / f) * (width / f);
        if n_tok > self.max_tokens {
            return 0;
        }
        n_tok
    }

    pub fn from_mmproj(
        mmproj: &GgufFile,
        vocab_tokens: &[String],
        n_embd: usize,
    ) -> Result<Self, GgufError> {
        let proj_ty = mmproj.metadata.get("clip.projector_type").and_then(|v| v.as_str());
        if let Some(p) = proj_ty {
            if p != "glm5next" {
                return Err(GgufError::MetadataTypeMismatch {
                    key: "clip.projector_type".to_owned(),
                    expected: "glm5next",
                });
            }
        }
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
        let image_size = get_u("image_size", Some(448))?;
        let out_dim = get_u("projection_dim", None)?;
        let eps = get_f("attention.layer_norm_epsilon", 1e-5);
        let ffn_clamp = get_f("ffn_clamp", 10.0);
        let min_tokens = get_u("image_min_tokens", Some(16))?;
        let meta_max_tokens = get_u("image_max_tokens", Some(8000))?;
        let max_tokens = std::env::var("AMPLLM_VISION_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| meta_max_tokens.min(2560))
            .clamp(min_tokens, meta_max_tokens);
        let image_mean = get_f3("image_mean", [0.48145467, 0.4578275, 0.40821072]);
        let image_std = get_f3("image_std", [0.26862955, 0.2613026, 0.2757771]);

        for req in [
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.patch_embd.bias",
            "v.post_ln.weight",
            "mm.patch_merger.weight",
            "mm.patch_merger.bias",
            "mm.model.fc.weight",
            "mm.post_norm.weight",
            "mm.post_norm.bias",
            "mm.gate.weight",
            "mm.up.weight",
            "mm.down.weight",
        ] {
            let _ = mmproj.tensor_info(req)?;
        }
        let merger_ff = mmproj
            .tensor_info("mm.gate.weight")?
            .dims
            .get(1)
            .copied()
            .unwrap_or(0) as usize;
        if out_dim != n_embd || spatial_merge != 2 || dim % heads != 0 || merger_ff == 0 {
            return Err(GgufError::MetadataTypeMismatch {
                key: "clip.vision".to_owned(),
                expected: "matching glm5next vision geometry",
            });
        }

        let factor = patch_size * spatial_merge;
        let factor_sq = factor * factor;
        let min_pixels = min_tokens * factor_sq;
        let max_pixels = max_tokens * factor_sq;

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
            out_dim,
            merger_ff,
            eps,
            ffn_clamp,
            min_tokens,
            max_tokens,
            min_pixels,
            max_pixels,
            image_mean,
            image_std,
            image_token_id: find_tok("<|image|>", 154854),
            vision_start_id: find_tok("<|begin_of_image|>", 154830),
            vision_end_id: find_tok("<|end_of_image|>", 154831),
        })
    }
}
