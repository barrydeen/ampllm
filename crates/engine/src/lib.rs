pub mod config;
pub mod glm5;
pub mod helper;
pub mod hy4;
pub mod inkling2;
pub mod inkling2_draft;
pub mod kv_cache;
pub mod mm3;
pub mod pipeline;
pub mod qwen4;
pub mod rank_state;
pub mod sampling;
pub mod scratch;
pub mod single;
pub mod snapshot;
pub mod tp2;
pub mod weights;

pub use ampllm_cache::CacheStats;
pub use config::{EngineConfig, MultiGpuMode};
pub use glm5::{Glm5Engine, Glm5ImageSpan};
pub use hy4::{Hy4Engine, Hy4ImageSpan};
pub use inkling2::{Inkling2Engine, StreamGenerateResult};
pub use kv_cache::{KvConvCache, KvQuantMode};
pub use mm3::{Mm3Engine, Mm3ImageSpan};
pub use qwen4::{Qwen4Engine, Qwen4ImageSpan};
pub use sampling::SamplingParams;
pub use snapshot::{
    match_prefix_policy, QuantumGenerateResult, QuantumOptions, SequenceSnapshot,
};

use anyhow::Result;

pub enum AnyEngine {
    Inkling2(Box<Inkling2Engine>),
    Glm5(Box<Glm5Engine>),
    Hy4(Box<Hy4Engine>),
    Qwen4(Box<Qwen4Engine>),
    Mm3(Box<Mm3Engine>),
}

impl AnyEngine {
    pub fn is_glm5(&self) -> bool {
        matches!(self, AnyEngine::Glm5(_))
    }

    pub fn is_hy4(&self) -> bool {
        matches!(self, AnyEngine::Hy4(_))
    }

    pub fn is_qwen4(&self) -> bool {
        matches!(self, AnyEngine::Qwen4(_))
    }

    pub fn is_mm3(&self) -> bool {
        matches!(self, AnyEngine::Mm3(_))
    }

    pub fn has_vision(&self) -> bool {
        match self {
            AnyEngine::Glm5(e) => e.has_vision(),
            AnyEngine::Hy4(e) => e.has_vision(),
            AnyEngine::Qwen4(e) => e.has_vision(),
            AnyEngine::Mm3(e) => e.has_vision(),
            _ => false,
        }
    }

    pub fn glm5_vision_config(&self) -> Option<&ampllm_gguf::Glm5VisionConfig> {
        match self {
            AnyEngine::Glm5(e) => e.vision_config(),
            _ => None,
        }
    }

    pub fn hy4_vision_config(&self) -> Option<&ampllm_gguf::Hy4VisionConfig> {
        match self {
            AnyEngine::Hy4(e) => e.vision_config(),
            _ => None,
        }
    }

    pub fn qwen4_vision_config(&self) -> Option<&ampllm_gguf::Qwen4VisionConfig> {
        match self {
            AnyEngine::Qwen4(e) => e.vision_config(),
            _ => None,
        }
    }

    pub fn mm3_vision_config(&self) -> Option<&ampllm_gguf::Mm3VisionConfig> {
        match self {
            AnyEngine::Mm3(e) => e.vision_config(),
            _ => None,
        }
    }

    pub fn encode_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        match self {
            AnyEngine::Glm5(e) => Ok(e.encode_image(rgb, width, height)?.2),
            AnyEngine::Hy4(e) => e.encode_image(rgb, width, height),
            AnyEngine::Qwen4(e) => Ok(e.encode_image(rgb, width, height)?.2),
            AnyEngine::Mm3(e) => e.encode_image(rgb, width, height),
            _ => anyhow::bail!("model does not support vision"),
        }
    }

    pub fn encode_glm5_image(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<(usize, usize, Vec<f32>)> {
        match self {
            AnyEngine::Glm5(e) => e.encode_image(rgb, width, height),
            _ => anyhow::bail!("model is not GLM-5 vision"),
        }
    }

    pub fn encode_qwen4_image(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<(usize, usize, Vec<f32>)> {
        match self {
            AnyEngine::Qwen4(e) => e.encode_image(rgb, width, height),
            _ => anyhow::bail!("model is not Qwen4 vision"),
        }
    }

    pub fn max_ctx(&self) -> usize {
        match self {
            AnyEngine::Inkling2(e) => e.max_ctx,
            AnyEngine::Glm5(e) => e.max_ctx,
            AnyEngine::Hy4(e) => e.max_ctx,
            AnyEngine::Qwen4(e) => e.max_ctx,
            AnyEngine::Mm3(e) => e.max_ctx,
        }
    }

    pub fn committed_tokens(&self) -> &[u32] {
        match self {
            AnyEngine::Inkling2(e) => &e.committed_tokens,
            AnyEngine::Glm5(e) => &e.committed_tokens,
            AnyEngine::Hy4(e) => &e.committed_tokens,
            AnyEngine::Qwen4(e) => &e.committed_tokens,
            AnyEngine::Mm3(e) => e.committed_tokens(),
        }
    }

    pub fn prefix_reuse_tokens_total(&self) -> u64 {
        match self {
            AnyEngine::Inkling2(e) => e.prefix_reuse_tokens_total,
            AnyEngine::Glm5(e) => e.prefix_reuse_tokens_total,
            AnyEngine::Hy4(e) => e.prefix_reuse_tokens_total,
            AnyEngine::Qwen4(e) => e.prefix_reuse_tokens_total,
            AnyEngine::Mm3(e) => e.prefix_reuse_tokens_total,
        }
    }

    pub fn cache_summary(&self) -> String {
        match self {
            AnyEngine::Inkling2(e) => e.cache_summary(),
            AnyEngine::Glm5(e) => e.cache_summary(),
            AnyEngine::Hy4(e) => e.cache_summary(),
            AnyEngine::Qwen4(e) => e.cache_summary(),
            AnyEngine::Mm3(e) => e.cache_summary(),
        }
    }

    pub fn cache_stats(&self) -> CacheStats {
        match self {
            AnyEngine::Inkling2(e) => e.cache_stats(),
            AnyEngine::Glm5(e) => e.cache_stats(),
            AnyEngine::Hy4(e) => e.cache_stats(),
            AnyEngine::Qwen4(e) => e.cache_stats(),
            AnyEngine::Mm3(e) => e.cache_stats().clone(),
        }
    }

    pub fn save_profile(&self) -> Result<()> {
        match self {
            AnyEngine::Inkling2(e) => e.save_profile(),
            AnyEngine::Glm5(e) => e.save_profile(),
            AnyEngine::Hy4(e) => e.save_profile(),
            AnyEngine::Qwen4(e) => e.save_profile(),
            AnyEngine::Mm3(e) => e.save_profile(),
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        match self {
            AnyEngine::Inkling2(e) => e.reset(),
            AnyEngine::Glm5(e) => e.reset(),
            AnyEngine::Hy4(e) => e.reset(),
            AnyEngine::Qwen4(e) => e.reset(),
            AnyEngine::Mm3(e) => e.reset(),
        }
    }

    pub fn supports_lcp_truncate(&self) -> bool {
        matches!(self, AnyEngine::Hy4(_) | AnyEngine::Mm3(_))
    }

    pub fn rollback_policy(&self) -> (bool, Option<usize>, usize) {
        match self {
            AnyEngine::Inkling2(_) | AnyEngine::Glm5(_) | AnyEngine::Qwen4(_) => (false, None, 1),
            AnyEngine::Hy4(e) => e.rollback_policy(),
            AnyEngine::Mm3(e) => e.rollback_policy(),
        }
    }

    pub fn capture_snapshot(&mut self, has_images: bool) -> Result<SequenceSnapshot> {
        match self {
            AnyEngine::Inkling2(e) => e.capture_sequence_snapshot(has_images),
            AnyEngine::Glm5(e) => e.capture_snapshot(has_images),
            AnyEngine::Hy4(e) => e.capture_snapshot(has_images),
            AnyEngine::Qwen4(e) => e.capture_snapshot(has_images),
            AnyEngine::Mm3(e) => e.capture_snapshot(has_images),
        }
    }

    pub fn restore_snapshot(
        &mut self,
        snap: &SequenceSnapshot,
        truncate_to: Option<usize>,
    ) -> Result<()> {
        match self {
            AnyEngine::Inkling2(e) => e.restore_sequence_snapshot(snap, truncate_to),
            AnyEngine::Glm5(e) => e.restore_snapshot(snap, truncate_to),
            AnyEngine::Hy4(e) => e.restore_snapshot(snap, truncate_to),
            AnyEngine::Qwen4(e) => e.restore_snapshot(snap, truncate_to),
            AnyEngine::Mm3(e) => e.restore_snapshot(snap, truncate_to),
        }
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Inkling2(e) => e.generate_stream(prompt_tokens, params, on_token),
            AnyEngine::Glm5(e) => e.generate_stream(prompt_tokens, params, on_token),
            AnyEngine::Hy4(e) => e.generate_stream(prompt_tokens, params, on_token),
            AnyEngine::Qwen4(e) => e.generate_stream(prompt_tokens, params, on_token),
            AnyEngine::Mm3(e) => e.generate_stream(prompt_tokens, params, on_token),
        }
    }

    pub fn generate_stream_quantum<F>(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Inkling2(e) => e.generate_stream_quantum(prompt_tokens, params, opts, on_token),
            AnyEngine::Glm5(e) => e.generate_stream_quantum(prompt_tokens, params, opts, on_token),
            AnyEngine::Hy4(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, &[], params, opts, on_token)
            }
            AnyEngine::Qwen4(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, &[], params, opts, on_token)
            }
            AnyEngine::Mm3(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, &[], params, opts, on_token)
            }
        }
    }

    pub fn generate_stream_with_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Hy4ImageSpan],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Hy4(e) => e.generate_stream_with_images(prompt_tokens, images, params, on_token),
            _ => {
                if images.is_empty() {
                    self.generate_stream(prompt_tokens, params, on_token)
                } else {
                    anyhow::bail!("model does not support multimodal image inputs")
                }
            }
        }
    }

    pub fn generate_stream_quantum_with_hy4_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Hy4ImageSpan],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Hy4(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, images, params, opts, on_token)
            }
            _ => {
                if images.is_empty() {
                    self.generate_stream_quantum(prompt_tokens, params, opts, on_token)
                } else {
                    anyhow::bail!("model does not support multimodal image inputs")
                }
            }
        }
    }

    pub fn generate_stream_with_qwen4_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Qwen4ImageSpan],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Qwen4(e) => e.generate_stream_with_images(prompt_tokens, images, params, on_token),
            _ => {
                if images.is_empty() {
                    self.generate_stream(prompt_tokens, params, on_token)
                } else {
                    anyhow::bail!("model is not Qwen4 vision")
                }
            }
        }
    }

    pub fn generate_stream_quantum_with_qwen4_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Qwen4ImageSpan],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Qwen4(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, images, params, opts, on_token)
            }
            _ => {
                if images.is_empty() {
                    self.generate_stream_quantum(prompt_tokens, params, opts, on_token)
                } else {
                    anyhow::bail!("model is not Qwen4 vision")
                }
            }
        }
    }

    pub fn generate_stream_with_mm3_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Mm3ImageSpan],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Mm3(e) => e.generate_stream_with_images(prompt_tokens, images, params, on_token),
            _ => {
                if images.is_empty() {
                    self.generate_stream(prompt_tokens, params, on_token)
                } else {
                    anyhow::bail!("model is not MM3 vision")
                }
            }
        }
    }

    pub fn generate_stream_quantum_with_mm3_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Mm3ImageSpan],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Mm3(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, images, params, opts, on_token)
            }
            _ => {
                if images.is_empty() {
                    self.generate_stream_quantum(prompt_tokens, params, opts, on_token)
                } else {
                    anyhow::bail!("model is not MM3 vision")
                }
            }
        }
    }

    pub fn generate_stream_with_glm5_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Glm5ImageSpan],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Glm5(e) => e.generate_stream_with_images(prompt_tokens, images, params, on_token),
            _ => {
                if images.is_empty() {
                    self.generate_stream(prompt_tokens, params, on_token)
                } else {
                    anyhow::bail!("model is not GLM-5 vision")
                }
            }
        }
    }

    pub fn generate_stream_quantum_with_glm5_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Glm5ImageSpan],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        match self {
            AnyEngine::Glm5(e) => {
                e.generate_stream_quantum_with_images(prompt_tokens, images, params, opts, on_token)
            }
            _ => {
                if images.is_empty() {
                    self.generate_stream_quantum(prompt_tokens, params, opts, on_token)
                } else {
                    anyhow::bail!("model is not GLM-5 vision")
                }
            }
        }
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        n_predict: usize,
    ) -> Result<(Vec<u32>, f64, f64, Vec<(u32, f32)>)> {
        match self {
            AnyEngine::Inkling2(e) => e.generate_greedy(prompt_tokens, n_predict),
            AnyEngine::Glm5(e) => e.generate_greedy(prompt_tokens, n_predict),
            AnyEngine::Hy4(e) => e.generate_greedy(prompt_tokens, n_predict),
            AnyEngine::Qwen4(e) => e.generate_greedy(prompt_tokens, n_predict),
            AnyEngine::Mm3(e) => {
                let res = e.generate_stream(
                    prompt_tokens,
                    &SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        seed: 42,
                        max_tokens: n_predict,
                        stop_token_ids: Vec::new(),
                    },
                    |_| true,
                )?;
                Ok((
                    res.generated_tokens,
                    res.prefill_tps,
                    res.decode_tps,
                    Vec::new(),
                ))
            }
        }
    }
}
