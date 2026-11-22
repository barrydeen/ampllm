use crate::format::{GgufError, GgufFile};

#[derive(Debug, Clone)]
pub struct LogicalLayerInfo {
    pub slot: usize,
    pub phys_layer: usize,
    pub pass: usize,
    pub is_swa: bool,
}

#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub arch: String,
    pub n_layer_phys: usize,
    pub n_layer_logical: usize,
    pub n_embd: usize,
    pub n_vocab_padded: usize,
    pub n_vocab_unpadded: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub context_length: usize,
    pub sliding_window: usize,
    pub is_swa_phys: Vec<bool>,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_floor: f32,
    pub log_scale_alpha: f32,
    pub shortconv_l_cache: usize,
    pub attn_res_block_size: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub expert_weights_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub final_logit_softcapping: f32,
    pub is_looped: bool,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    pub loop_gate_rank: usize,
    pub logical_layers: Vec<LogicalLayerInfo>,
}

impl Inkling2Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let arch = gguf.get_str("general.architecture")?.to_owned();
        if arch != "inkling2_loop" && arch != "inkling2" {
            return Err(GgufError::MetadataTypeMismatch {
                key: "general.architecture".to_owned(),
                expected: "inkling2_loop or inkling2",
            });
        }
        let k = |suffix: &str| format!("{arch}.{suffix}");

        let raw_block_count = gguf.get_u32(&k("block_count"))? as usize;
        let n_embd = gguf.get_u32(&k("embedding_length"))? as usize;
        let n_vocab_padded = gguf.get_u32(&k("vocab_size"))? as usize;
        let n_vocab_unpadded = gguf.get_u32(&k("vocab_size_unpadded"))? as usize;
        let n_head = gguf.get_u32_or_first(&k("attention.head_count"))? as usize;
        let n_head_kv = gguf.get_u32_or_first(&k("attention.head_count_kv"))? as usize;
        let head_dim = gguf.get_u32(&k("attention.key_length"))? as usize;
        let rms_norm_eps = gguf.get_f32(&k("attention.layer_norm_rms_epsilon"))?;
        let context_length = gguf.get_u32(&k("context_length"))? as usize;
        let sliding_window = gguf.get_u32(&k("attention.sliding_window"))? as usize;
        let swa_pattern = gguf.get_bool_array(&k("attention.sliding_window_pattern"))?;
        let d_rel = gguf.get_u32(&k("attention.relative_length"))? as usize;
        let rel_extent = gguf.get_u32(&k("attention.relative_extent"))? as usize;
        let qk_clip = gguf.get_f32(&k("attention.qk_clip"))?;
        let rel_clip = gguf.get_f32(&k("attention.relative_clip"))?;
        let log_scale_floor = gguf.get_u32(&k("attention.log_scale_floor"))? as f32;
        let log_scale_alpha = gguf.get_f32(&k("attention.log_scale_alpha"))?;
        let shortconv_l_cache = gguf.get_u32(&k("shortconv.l_cache"))? as usize;
        let attn_res_block_size = gguf.get_u32(&k("attn_res.block_size"))? as usize;
        let n_expert = gguf.get_u32(&k("expert_count"))? as usize;
        let n_expert_used = gguf.get_u32(&k("expert_used_count"))? as usize;
        let n_ff_exp = gguf.get_u32_or_first(&k("expert_feed_forward_length"))? as usize;
        let expert_weights_scale = gguf.get_f32(&k("expert_weights_scale"))?;
        let situ_beta = gguf.get_f32(&k("activation.situ_beta"))?;
        let situ_linear_beta = gguf.get_f32(&k("activation.situ_linear_beta"))?;
        let final_logit_softcapping = gguf.get_f32(&k("final_logit_softcapping")).unwrap_or(0.0);

        let (is_looped, n_layer_phys, n_layer_logical, loop_start, loop_end, loop_count, loop_gate_rank, is_swa_phys) =
            if arch == "inkling2_loop" {
                let ls = gguf.get_u32(&k("loop.start"))? as usize;
                let le = gguf.get_u32(&k("loop.end"))? as usize;
                let lc = gguf.get_u32(&k("loop.count"))? as usize;
                let lgr = gguf.get_u32(&k("loop.gate_rank"))? as usize;
                let n_phys = raw_block_count;
                let n_log = n_phys + (lc - 1) * (le - ls);
                (true, n_phys, n_log, ls, le, lc, lgr, swa_pattern)
            } else if let Ok(lc_u32) = gguf.get_u32(&k("num_loops")) {
                let lc = lc_u32 as usize;
                if lc > 1 {
                    let ls = gguf.get_u32(&k("loop.core_start"))? as usize;
                    let le = gguf.get_u32(&k("loop.core_end"))? as usize;
                    let lgr = gguf.get_u32(&k("loop.gate_rank"))? as usize;
                    let n_core = le - ls;
                    // In unsloth's `inkling2` layout, `block_count` is the total visit count (48),
                    // while physical layers are `48 - (3 - 1) * 12 = 24`.
                    let n_log = raw_block_count;
                    let n_phys = n_log - (lc - 1) * n_core;
                    let mut swa_phys = Vec::with_capacity(n_phys);
                    for l in 0..n_phys {
                        let visit_idx = if l < le { l } else { l + (lc - 1) * n_core };
                        swa_phys.push(swa_pattern[visit_idx]);
                    }
                    (true, n_phys, n_log, ls, le, lc, lgr, swa_phys)
                } else {
                    (false, raw_block_count, raw_block_count, 0, 0, 1, 0, swa_pattern)
                }
            } else {
                (false, raw_block_count, raw_block_count, 0, 0, 1, 0, swa_pattern)
            };

        let mut logical_layers = Vec::with_capacity(n_layer_logical);
        for slot in 0..n_layer_logical {
            let (phys_layer, pass) = if slot < n_layer_phys {
                (slot, 0)
            } else {
                let core_len = loop_end - loop_start;
                let off = slot - n_layer_phys;
                (loop_start + (off % core_len), 1 + (off / core_len))
            };
            logical_layers.push(LogicalLayerInfo {
                slot,
                phys_layer,
                pass,
                is_swa: is_swa_phys[phys_layer],
            });
        }

        Ok(Self {
            arch,
            n_layer_phys,
            n_layer_logical,
            n_embd,
            n_vocab_padded,
            n_vocab_unpadded,
            n_head,
            n_head_kv,
            head_dim,
            rms_norm_eps,
            context_length,
            sliding_window,
            is_swa_phys,
            d_rel,
            rel_extent,
            qk_clip,
            rel_clip,
            log_scale_floor,
            log_scale_alpha,
            shortconv_l_cache,
            attn_res_block_size,
            n_expert,
            n_expert_used,
            n_ff_exp,
            expert_weights_scale,
            situ_beta,
            situ_linear_beta,
            final_logit_softcapping,
            is_looped,
            loop_start,
            loop_end,
            loop_count,
            loop_gate_rank,
            logical_layers,
        })
    }

    /// Returns the execution sequence of logical slots in forward-pass order.
    pub fn execution_order(&self) -> Vec<usize> {
        if !self.is_looped {
            return (0..self.n_layer_phys).collect();
        }
        let mut order = Vec::with_capacity(self.n_layer_logical);
        for il in 0..self.loop_start {
            order.push(il);
        }
        for pass in 0..self.loop_count {
            for ip in self.loop_start..self.loop_end {
                let slot = if pass == 0 {
                    ip
                } else {
                    self.n_layer_phys + (pass - 1) * (self.loop_end - self.loop_start) + (ip - self.loop_start)
                };
                order.push(slot);
            }
        }
        for il in self.loop_end..self.n_layer_phys {
            order.push(il);
        }
        order
    }
}
