use ampllm_cache::CacheStats;
use ampllm_cuda_kernels::{device_count, mem_info};
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct ServerMetrics {
    pub requests_total: AtomicU64,
    pub tokens_prefilled_total: AtomicU64,
    pub tokens_generated_total: AtomicU64,
    pub prefix_reuse_tokens_total: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_helper_hits: AtomicU64,
    pub cache_promotions: AtomicU64,
}

impl ServerMetrics {
    pub fn update_from_engine(&self, stats: &CacheStats, prefix_reused_total: u64) {
        self.cache_hits.store(stats.hits, Ordering::Relaxed);
        self.cache_misses.store(stats.misses, Ordering::Relaxed);
        self.cache_helper_hits
            .store(stats.helper_hits, Ordering::Relaxed);
        self.cache_promotions.store(stats.promotions, Ordering::Relaxed);
        self.prefix_reuse_tokens_total
            .store(prefix_reused_total, Ordering::Relaxed);
    }

    pub fn format_prometheus(&self) -> String {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        let prefill = self.tokens_prefilled_total.load(Ordering::Relaxed);
        let gen = self.tokens_generated_total.load(Ordering::Relaxed);
        let reuse = self.prefix_reuse_tokens_total.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let h_hits = self.cache_helper_hits.load(Ordering::Relaxed);
        let prom = self.cache_promotions.load(Ordering::Relaxed);
        let tot = hits + h_hits + misses;
        let hit_ratio = if tot == 0 {
            1.0
        } else {
            (hits + h_hits) as f64 / tot as f64
        };

        let mut out = String::with_capacity(1024);
        let _ = writeln!(out, "# TYPE ampllm_requests_total counter");
        let _ = writeln!(out, "ampllm_requests_total {reqs}");
        let _ = writeln!(out, "# TYPE ampllm_tokens_prefilled_total counter");
        let _ = writeln!(out, "ampllm_tokens_prefilled_total {prefill}");
        let _ = writeln!(out, "# TYPE ampllm_tokens_generated_total counter");
        let _ = writeln!(out, "ampllm_tokens_generated_total {gen}");
        let _ = writeln!(out, "# TYPE ampllm_prefix_reuse_tokens_total counter");
        let _ = writeln!(out, "ampllm_prefix_reuse_tokens_total {reuse}");
        let _ = writeln!(out, "# TYPE ampllm_cache_hits_total counter");
        let _ = writeln!(out, "ampllm_cache_hits_total {hits}");
        let _ = writeln!(out, "# TYPE ampllm_cache_helper_hits_total counter");
        let _ = writeln!(out, "ampllm_cache_helper_hits_total {h_hits}");
        let _ = writeln!(out, "# TYPE ampllm_cache_misses_total counter");
        let _ = writeln!(out, "ampllm_cache_misses_total {misses}");
        let _ = writeln!(out, "# TYPE ampllm_cache_hit_ratio gauge");
        let _ = writeln!(out, "ampllm_cache_hit_ratio {hit_ratio:.6}");
        let _ = writeln!(out, "# TYPE ampllm_cache_promotions_total counter");
        let _ = writeln!(out, "ampllm_cache_promotions_total {prom}");

        if let Ok(n) = device_count() {
            let _ = writeln!(out, "# TYPE ampllm_vram_free_bytes gauge");
            let _ = writeln!(out, "# TYPE ampllm_vram_total_bytes gauge");
            for d in 0..n {
                if let Ok((free, total)) = mem_info(d) {
                    let _ = writeln!(out, "ampllm_vram_free_bytes{{gpu=\"{d}\"}} {free}");
                    let _ = writeln!(out, "ampllm_vram_total_bytes{{gpu=\"{d}\"}} {total}");
                }
            }
        }
        out
    }
}
