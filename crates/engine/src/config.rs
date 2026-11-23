use ampllm_cache::CacheBudget;
use crate::kv_cache::KvQuantMode;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MultiGpuMode {
    #[default]
    Single,
    Helper,
    Pipeline,
    Tp2,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub device: i32,
    pub max_ctx: usize,
    pub max_batch: usize,
    pub kv_mode: KvQuantMode,
    pub expert_cache: CacheBudget,
    pub multi_gpu: MultiGpuMode,
    pub cpu_threads: Option<usize>,
    pub use_cuda_graph: bool,
    pub load_profile: Option<PathBuf>,
    pub save_profile: Option<PathBuf>,
    pub dma_prefill_threshold: usize,
    pub use_spec: bool,
    pub draft_path: Option<PathBuf>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device: 0,
            max_ctx: 4096,
            max_batch: 512,
            kv_mode: KvQuantMode::F16,
            expert_cache: CacheBudget::Auto,
            multi_gpu: MultiGpuMode::Single,
            cpu_threads: None,
            use_cuda_graph: true,
            load_profile: None,
            save_profile: None,
            dma_prefill_threshold: 8,
            use_spec: true,
            draft_path: None,
        }
    }
}
