use crate::profile::CacheProfile;

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub helper_hits: u64,
    pub promotions: u64,
}

impl CacheStats {
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses + self.helper_hits
    }

    pub fn vram_hit_rate(&self) -> f64 {
        let tot = self.total_lookups();
        if tot == 0 {
            1.0
        } else {
            (self.hits + self.helper_hits) as f64 / tot as f64
        }
    }
}

/// Per-layer EWMA frequency tracker for online expert cache promotion decisions.
pub struct EwmaTracker {
    pub decay: f32,
    pub scores: Vec<Vec<f32>>,
    pub stats: CacheStats,
}

impl EwmaTracker {
    pub fn new(n_layer_phys: usize, n_expert: usize, decay: f32) -> Self {
        Self {
            decay,
            scores: vec![vec![0.0f32; n_expert]; n_layer_phys],
            stats: CacheStats::default(),
        }
    }

    pub fn seed_from_profile(&mut self, profile: &CacheProfile) {
        let n_layers = self.scores.len().min(profile.ranked_experts.len());
        for l in 0..n_layers {
            let n_exp = self.scores[l].len();
            for (rank, &e) in profile.ranked_experts[l].iter().enumerate() {
                if e < n_exp {
                    self.scores[l][e] = ((n_exp - rank) as f32) * 0.1;
                }
            }
        }
    }

    pub fn decay_layer(&mut self, layer: usize) {
        for s in &mut self.scores[layer] {
            *s *= self.decay;
        }
    }

    #[inline]
    pub fn record_activation(&mut self, layer: usize, exp: usize) {
        if let Some(slot) = self.scores[layer].get_mut(exp) {
            *slot += 1.0f32;
        }
    }

    pub fn ranked_layer(&self, layer: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.scores[layer].len()).collect();
        order.sort_by(|&a, &b| {
            self.scores[layer][b]
                .total_cmp(&self.scores[layer][a])
                .then_with(|| a.cmp(&b))
        });
        order
    }

    pub fn export_profile(&self, arch: &str) -> CacheProfile {
        let n_layer_phys = self.scores.len();
        let n_expert = self.scores.first().map(|v| v.len()).unwrap_or(0);
        let ranked_experts = (0..n_layer_phys).map(|l| self.ranked_layer(l)).collect();
        CacheProfile {
            arch: arch.to_owned(),
            n_layer_phys,
            n_expert,
            ranked_experts,
            global_ranked: Vec::new(),
        }
    }
}
