pub mod config;
pub mod glm5;
pub mod helper;
pub mod inkling2;
pub mod kv_cache;
pub mod pipeline;
pub mod rank_state;
pub mod sampling;
pub mod scratch;
pub mod single;
pub mod tp2;
pub mod weights;

pub use config::{EngineConfig, MultiGpuMode};
pub use glm5::Glm5Engine;
pub use inkling2::{Inkling2Engine, StreamGenerateResult};
pub use kv_cache::{KvConvCache, KvQuantMode};
pub use sampling::SamplingParams;

use ampllm_cache::CacheStats;
use anyhow::Result;

pub enum AnyEngine {
    Inkling2(Box<Inkling2Engine>),
    Glm5(Box<Glm5Engine>),
}

impl AnyEngine {
    pub fn is_glm5(&self) -> bool {
        matches!(self, AnyEngine::Glm5(_))
    }

    pub fn max_ctx(&self) -> usize {
        match self {
            AnyEngine::Inkling2(e) => e.cfg.sliding_window.max(32768),
            AnyEngine::Glm5(e) => e.max_ctx,
        }
    }

    pub fn prefix_reuse_tokens_total(&self) -> u64 {
        match self {
            AnyEngine::Inkling2(e) => e.prefix_reuse_tokens_total,
            AnyEngine::Glm5(e) => e.prefix_reuse_tokens_total,
        }
    }

    pub fn cache_summary(&self) -> String {
        match self {
            AnyEngine::Inkling2(e) => e.cache_summary(),
            AnyEngine::Glm5(e) => e.cache_summary(),
        }
    }

    pub fn cache_stats(&self) -> CacheStats {
        match self {
            AnyEngine::Inkling2(e) => e.cache_stats(),
            AnyEngine::Glm5(e) => e.cache_stats(),
        }
    }

    pub fn save_profile(&self) -> Result<()> {
        match self {
            AnyEngine::Inkling2(e) => e.save_profile(),
            AnyEngine::Glm5(e) => e.save_profile(),
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        match self {
            AnyEngine::Inkling2(e) => e.reset(),
            AnyEngine::Glm5(e) => e.reset(),
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
        }
    }
}
