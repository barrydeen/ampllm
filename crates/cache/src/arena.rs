use ampllm_cuda_kernels::PinnedMappedBuffer;
use ampllm_gguf::{GgufFile, Inkling2Config};
use anyhow::Result;

#[inline]
pub fn align_up_256(n: usize) -> usize {
    (n + 255) & !255
}

pub struct HostLayerExperts {
    pub gate_up_pinned: PinnedMappedBuffer<u8>,
    pub down_pinned: PinnedMappedBuffer<u8>,
    pub gate_up_stride: usize,
    pub down_stride: usize,
    pub gate_up_expert_bytes: usize,
    pub down_expert_bytes: usize,
}

/// Pinned 256B-aligned host storage for experts that can be offloaded to CPU or streamed via DMA.
pub struct HostExpertArena {
    pub layers: Vec<Option<HostLayerExperts>>,
}

impl HostExpertArena {
    pub fn from_gguf(
        gguf: &GgufFile,
        cfg: &Inkling2Config,
        layer_needed: &[bool],
        layer_expert_bytes: &[(usize, usize)],
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layer_phys);
        for i in 0..cfg.n_layer_phys {
            if !layer_needed[i] {
                layers.push(None);
                continue;
            }
            let (gu_exp_bytes, dn_exp_bytes) = layer_expert_bytes[i];
            let gu_stride = align_up_256(gu_exp_bytes);
            let dn_stride = align_up_256(dn_exp_bytes);

            let gu_bytes = gguf.tensor_bytes(&format!("blk.{i}.ffn_gate_up_exps.weight"))?;
            let dn_bytes = gguf.tensor_bytes(&format!("blk.{i}.ffn_down_exps.weight"))?;

            let mut gate_up_pinned = PinnedMappedBuffer::<u8>::alloc(cfg.n_expert * gu_stride)?;
            let mut down_pinned = PinnedMappedBuffer::<u8>::alloc(cfg.n_expert * dn_stride)?;

            let gu_dst = gate_up_pinned.as_mut_slice();
            let dn_dst = down_pinned.as_mut_slice();
            for e in 0..cfg.n_expert {
                let s_gu = e * gu_exp_bytes;
                let d_gu = e * gu_stride;
                gu_dst[d_gu..d_gu + gu_exp_bytes].copy_from_slice(&gu_bytes[s_gu..s_gu + gu_exp_bytes]);

                let s_dn = e * dn_exp_bytes;
                let d_dn = e * dn_stride;
                dn_dst[d_dn..d_dn + dn_exp_bytes].copy_from_slice(&dn_bytes[s_dn..s_dn + dn_exp_bytes]);
            }

            layers.push(Some(HostLayerExperts {
                gate_up_pinned,
                down_pinned,
                gate_up_stride: gu_stride,
                down_stride: dn_stride,
                gate_up_expert_bytes: gu_exp_bytes,
                down_expert_bytes: dn_exp_bytes,
            }));
        }
        Ok(Self { layers })
    }

    pub fn has_layer(&self, layer: usize) -> bool {
        self.layers.get(layer).and_then(|o| o.as_ref()).is_some()
    }

    #[inline]
    pub fn expert_gate_up(&self, layer: usize, exp: usize) -> &[u8] {
        let l = self.layers[layer]
            .as_ref()
            .expect("host arena missing layer");
        let off = exp * l.gate_up_stride;
        &l.gate_up_pinned.as_slice()[off..off + l.gate_up_expert_bytes]
    }

    #[inline]
    pub fn expert_down(&self, layer: usize, exp: usize) -> &[u8] {
        let l = self.layers[layer]
            .as_ref()
            .expect("host arena missing layer");
        let off = exp * l.down_stride;
        &l.down_pinned.as_slice()[off..off + l.down_expert_bytes]
    }

    pub fn total_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.gate_up_pinned.as_slice().len() + l.down_pinned.as_slice().len())
            .sum()
    }
}
