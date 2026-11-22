use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheProfile {
    pub arch: String,
    pub n_layer_phys: usize,
    pub n_expert: usize,
    pub ranked_experts: Vec<Vec<usize>>,
    /// Optional global `(layer, expert)` ordering sorted by descending hotness (from `STRP` profiles).
    #[serde(default)]
    pub global_ranked: Vec<(usize, usize)>,
}

impl CacheProfile {
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.starts_with(b"STRP") {
            return Self::from_strp_bytes(&bytes);
        }
        let prof: Self = serde_json::from_slice(&bytes)?;
        if prof.ranked_experts.len() != prof.n_layer_phys {
            bail!(
                "invalid CacheProfile: ranked_experts len {} != n_layer_phys {}",
                prof.ranked_experts.len(),
                prof.n_layer_phys
            );
        }
        Ok(prof)
    }

    pub fn from_strp_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 24 || &bytes[0..4] != b"STRP" {
            bail!("invalid STRP profile header");
        }
        let rd_u32 = |off: usize| -> u32 {
            u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
        };
        let version = rd_u32(4);
        if version != 1 {
            bail!("unsupported STRP version {version}");
        }
        let n_layers = rd_u32(8) as usize;
        let n_expert = rd_u32(12) as usize;
        let _slots = rd_u32(16) as usize;
        let n_ranked = rd_u32(20) as usize;
        let pairs_end = 24 + n_ranked * 4;
        if bytes.len() < pairs_end {
            bail!("truncated STRP profile: expected at least {pairs_end} bytes, got {}", bytes.len());
        }
        let mut global_ranked = Vec::with_capacity(n_ranked);
        let mut ranked_experts = vec![Vec::new(); n_layers];
        let mut seen = vec![vec![false; n_expert]; n_layers];
        for i in 0..n_ranked {
            let off = 24 + i * 4;
            let layer = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) as usize;
            let expert = u16::from_le_bytes(bytes[off + 2..off + 4].try_into().unwrap()) as usize;
            if layer < n_layers && expert < n_expert && !seen[layer][expert] {
                seen[layer][expert] = true;
                global_ranked.push((layer, expert));
                ranked_experts[layer].push(expert);
            }
        }
        // Fill any experts absent from global ranking at the tail in expert ID order.
        for l in 0..n_layers {
            for e in 0..n_expert {
                if !seen[l][e] {
                    ranked_experts[l].push(e);
                }
            }
        }
        Ok(Self {
            arch: "glm5-next".to_owned(),
            n_layer_phys: n_layers,
            n_expert,
            ranked_experts,
            global_ranked,
        })
    }
}
