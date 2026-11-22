pub mod glm5_cpu;
pub mod moe_cpu;
pub mod pool;
pub mod quant_x86;

pub use glm5_cpu::{
    quantize_f32_to_q8_k_cpu, quantize_glm5_cpu_x_q8k, run_glm5_cpu_moe_grouped,
    run_glm5_cpu_moe_grouped_prequant, run_glm5_cpu_moe_m1, CpuBlockQ8K, Glm5CpuExpertJob,
    Glm5CpuGroupedExpert, Glm5CpuScratch,
};
pub use moe_cpu::{run_cpu_moe_batch, run_cpu_moe_m1, CpuBatchSlot, CpuExpertJob, CpuMoeScratch};
pub use pool::{
    discover_physical_cores, discover_physical_cores_with_options, CpuThreadPool,
};
pub use quant_x86::{
    active_simd_tier, quantize_f32_to_q8_1_cpu, vec_dot_q_q8_1, vec_dot_scalar, CpuBlockQ8_1,
    SimdTier,
};
