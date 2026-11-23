use crate::snapshot::{d2h_full, d2h_prefix, h2d_bytes, InklingKvSnap};
use ampllm_cuda_kernels::DeviceBuffer;
use ampllm_gguf::Inkling2Config;
use anyhow::Result;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    F16,
    Q8_0,
}

pub struct SlotKvConv {
    pub k_cache: DeviceBuffer,
    pub v_cache: DeviceBuffer,
    pub conv_history: DeviceBuffer, // [n_sconv_channels * (l_cache - 1)] F32
    pub ring_cap: usize,
    pub swa_window: usize, // 0 if global, else sliding_window
}

pub struct KvConvCache {
    pub kv_mode: KvQuantMode,
    pub max_ctx: usize,
    pub n_sconv_channels: usize,
    pub slots: Vec<Option<SlotKvConv>>,
}

impl KvConvCache {
    pub fn alloc(
        cfg: &Inkling2Config,
        device: i32,
        max_ctx: usize,
        kv_mode: KvQuantMode,
    ) -> Result<Self> {
        Self::alloc_custom(cfg, device, max_ctx, kv_mode, None, None)
    }

    pub fn alloc_custom(
        cfg: &Inkling2Config,
        device: i32,
        max_ctx: usize,
        kv_mode: KvQuantMode,
        n_head_kv_override: Option<usize>,
        phys_range: Option<Range<usize>>,
    ) -> Result<Self> {
        let n_head_kv = n_head_kv_override.unwrap_or(cfg.n_head_kv);
        let gqa_dim = n_head_kv * cfg.head_dim;
        let n_sconv_channels = 2 * gqa_dim + 2 * cfg.n_embd;
        let conv_hist_floats = n_sconv_channels * (cfg.shortconv_l_cache - 1);
        let conv_bytes = conv_hist_floats * 4;

        let token_kv_bytes = match kv_mode {
            KvQuantMode::F16 => gqa_dim * 2,
            KvQuantMode::Q8_0 => (gqa_dim / 32) * 34,
        };

        let mut slots = Vec::with_capacity(cfg.n_layer_logical);
        for info in &cfg.logical_layers {
            if let Some(ref r) = phys_range {
                if !r.contains(&info.phys_layer) {
                    slots.push(None);
                    continue;
                }
            }
            let ring_cap = if info.is_swa {
                cfg.sliding_window.min(max_ctx)
            } else {
                max_ctx
            };
            let swa_window = if info.is_swa { cfg.sliding_window } else { 0 };
            let kv_bytes = ring_cap * token_kv_bytes;
            slots.push(Some(SlotKvConv {
                k_cache: DeviceBuffer::alloc_zeroed(device, kv_bytes)?,
                v_cache: DeviceBuffer::alloc_zeroed(device, kv_bytes)?,
                conv_history: DeviceBuffer::alloc_zeroed(device, conv_bytes)?,
                ring_cap,
                swa_window,
            }));
        }

        Ok(Self {
            kv_mode,
            max_ctx,
            n_sconv_channels,
            slots,
        })
    }

    #[inline]
    pub fn slot(&self, idx: usize) -> &SlotKvConv {
        self.slots[idx].as_ref().expect("kv slot not allocated on this GPU")
    }

    pub fn reset(&self) -> Result<()> {
        for s in self.slots.iter().flatten() {
            s.conv_history.zero()?;
        }
        Ok(())
    }

    pub fn capture_conv_snapshot(&self, device: i32) -> Result<Vec<Vec<u8>>> {
        ampllm_cuda_kernels::set_device(device)?;
        let mut out = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            match s {
                Some(slot) => {
                    let mut buf = vec![0u8; slot.conv_history.bytes];
                    slot.conv_history.copy_to_host(&mut buf)?;
                    out.push(buf);
                }
                None => out.push(Vec::new()),
            }
        }
        Ok(out)
    }

    pub fn restore_conv_snapshot(&self, device: i32, data: &[Vec<u8>]) -> Result<()> {
        ampllm_cuda_kernels::set_device(device)?;
        for (slot_opt, bytes) in self.slots.iter().zip(data.iter()) {
            if let Some(slot) = slot_opt {
                if !bytes.is_empty() {
                    slot.conv_history.copy_from_host(bytes)?;
                }
            }
        }
        Ok(())
    }

    pub fn total_bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|s| s.k_cache.bytes + s.v_cache.bytes + s.conv_history.bytes)
            .sum()
    }

    pub(crate) fn capture_full_snapshot(
        &self,
        device: i32,
        cur_pos: usize,
    ) -> Result<Vec<Option<InklingKvSnap>>> {
        ampllm_cuda_kernels::set_device(device)?;
        let mut out = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            match s {
                Some(slot) => {
                    let kv_bytes = if slot.ring_cap == 0 {
                        0
                    } else {
                        let row_bytes = slot.k_cache.bytes / slot.ring_cap;
                        let rows = if slot.swa_window > 0 && cur_pos >= slot.ring_cap {
                            slot.ring_cap
                        } else {
                            cur_pos.min(slot.ring_cap)
                        };
                        rows * row_bytes
                    };
                    out.push(Some(InklingKvSnap {
                        k_cache: d2h_prefix(&slot.k_cache, kv_bytes)?,
                        v_cache: d2h_prefix(&slot.v_cache, kv_bytes)?,
                        conv_history: d2h_full(&slot.conv_history)?,
                    }));
                }
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub(crate) fn restore_full_snapshot(
        &self,
        device: i32,
        snaps: &[Option<InklingKvSnap>],
    ) -> Result<()> {
        ampllm_cuda_kernels::set_device(device)?;
        for (slot_opt, snap_opt) in self.slots.iter().zip(snaps.iter()) {
            if let (Some(slot), Some(snap)) = (slot_opt, snap_opt) {
                h2d_bytes(&slot.k_cache, &snap.k_cache)?;
                h2d_bytes(&slot.v_cache, &snap.v_cache)?;
                h2d_bytes(&slot.conv_history, &snap.conv_history)?;
            }
        }
        Ok(())
    }
}
