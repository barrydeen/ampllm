use crate::format::{GgufError, GgufFile};

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
