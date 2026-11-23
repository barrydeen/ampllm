use crate::inkling2::StreamGenerateResult;
use crate::qwen4::ple::PleCursor;
use ampllm_cuda_kernels::{set_device, DeviceBuffer};
use anyhow::Result;

#[inline]
pub(crate) fn d2h_prefix(buf: &DeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
    let n = bytes.min(buf.bytes);
    let mut out = vec![0u8; n];
    if n > 0 {
        set_device(buf.device)?;
        buf.copy_to_host(&mut out)?;
    }
    Ok(out)
}

#[inline]
pub(crate) fn d2h_full(buf: &DeviceBuffer) -> Result<Vec<u8>> {
    d2h_prefix(buf, buf.bytes)
}

#[inline]
pub(crate) fn h2d_bytes(buf: &DeviceBuffer, bytes: &[u8]) -> Result<()> {
    if !bytes.is_empty() {
        set_device(buf.device)?;
        buf.copy_from_host(bytes)?;
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct InklingKvSnap {
    pub k_cache: Vec<u8>,
    pub v_cache: Vec<u8>,
    pub conv_history: Vec<u8>,
}

impl InklingKvSnap {
    pub(crate) fn byte_len(&self) -> usize {
        self.k_cache.len() + self.v_cache.len() + self.conv_history.len()
    }
}

#[derive(Clone)]
pub(crate) struct InklingSnapshot {
    pub rank0: Vec<Option<InklingKvSnap>>,
    pub rank1: Option<Vec<Option<InklingKvSnap>>>,
}

#[derive(Clone)]
pub(crate) struct Glm5KdaSnap {
    pub s_mat: Vec<u8>,
    pub conv_q: Vec<u8>,
    pub conv_k: Vec<u8>,
    pub conv_v: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct Glm5DsaSnap {
    pub vram_lat8: Vec<u8>,
    pub ram_lat8_tail: Vec<u8>,
    pub ik_ring: Vec<u8>,
    pub ig_ring: Vec<u8>,
    pub pooled_f16: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct Glm5Snapshot {
    pub head_x_pos: Option<usize>,
    pub mtp_cached_pos: usize,
    pub head_x_row0: Option<Vec<u8>>,
    pub kda: Vec<Option<Glm5KdaSnap>>,
    pub dsa: Vec<Option<Glm5DsaSnap>>,
}

#[derive(Clone)]
pub(crate) struct Hy4LayerSnap {
    pub kv: Vec<u8>,
    pub ivg: Option<Vec<u8>>,
    pub pooled: Option<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct Hy4Snapshot {
    pub end_pos: usize,
    pub mtp_done: usize,
    pub caches: Vec<Hy4LayerSnap>,
    pub mtp_cache: Option<Hy4LayerSnap>,
    pub hring: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct Qwen4GdnSnap {
    pub state: Vec<u8>,
    pub conv_hist: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct Qwen4QsaSnap {
    pub k_q: Vec<u8>,
    pub v_q: Vec<u8>,
    pub k_sc: Vec<u8>,
    pub v_sc: Vec<u8>,
    pub tail: Vec<u8>,
    pub dead: Vec<u8>,
    pub pooled: Vec<u8>,
}

#[derive(Clone)]
pub(crate) enum Qwen4LayerSnap {
    Gdn(Qwen4GdnSnap),
    Qsa(Qwen4QsaSnap),
}

#[derive(Clone)]
pub(crate) struct Qwen4MtpSnap {
    pub kv_pos: usize,
    pub k_q: Vec<u8>,
    pub v_q: Vec<u8>,
    pub k_sc: Vec<u8>,
    pub v_sc: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct Qwen4Snapshot {
    pub layers: Vec<Qwen4LayerSnap>,
    pub ple_history: Option<Vec<u8>>,
    pub ple_cursor: PleCursor,
    pub r_last_row0: Vec<u8>,
    pub mtp: Option<Qwen4MtpSnap>,
    pub mrope_active: bool,
    pub mrope_pos: usize,
    pub mrope_table_host: Vec<i32>,
}

#[derive(Clone)]
pub(crate) struct Mm3LayerSnap {
    pub ic_bf16: Vec<u8>,
    pub kc: Vec<u8>,
    pub vc: Vec<u8>,
    pub ks: Option<Vec<u8>>,
    pub vs: Option<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct Mm3Snapshot {
    pub layers: Vec<Mm3LayerSnap>,
    pub dspark_layers: Option<Vec<(Vec<u8>, Vec<u8>)>>,
}

#[derive(Clone)]
pub(crate) enum EngineSnapshotData {
    Inkling2(InklingSnapshot),
    Glm5(Glm5Snapshot),
    Hy4(Hy4Snapshot),
    Qwen4(Qwen4Snapshot),
    Mm3(Mm3Snapshot),
}

/// Host-RAM snapshot of an engine's active sequence state (`0..tokens.len()`).
#[derive(Clone)]
pub struct SequenceSnapshot {
    pub tokens: Vec<u32>,
    pub has_images: bool,
    pub allow_lcp_truncate: bool,
    pub max_safe_rollback_delta: Option<usize>,
    pub rollback_align: usize,
    pub byte_len: usize,
    pub(crate) inner: EngineSnapshotData,
}

/// Shared prefix-matching policy used by both host-RAM `SequenceSnapshot`s and resident VRAM slots.
pub fn match_prefix_policy(
    cached_tokens: &[u32],
    cached_has_images: bool,
    allow_lcp_truncate: bool,
    max_safe_rollback_delta: Option<usize>,
    rollback_align: usize,
    prompt: &[u32],
    incoming_has_images: bool,
) -> usize {
    if cached_has_images || incoming_has_images || cached_tokens.is_empty() || prompt.len() <= 1 {
        return 0;
    }
    let max_reuse = cached_tokens.len().min(prompt.len() - 1);
    let mut lcp = 0usize;
    while lcp < max_reuse && cached_tokens[lcp] == prompt[lcp] {
        lcp += 1;
    }
    if lcp == 0 {
        return 0;
    }
    if lcp == cached_tokens.len() {
        return lcp;
    }
    if !allow_lcp_truncate {
        return 0;
    }
    if let Some(max_delta) = max_safe_rollback_delta {
        if cached_tokens.len() - lcp > max_delta {
            return 0;
        }
    }
    let align = rollback_align.max(1);
    (lcp / align) * align
}

impl SequenceSnapshot {
    /// Compute the maximum safe prefix length reusable from this snapshot for `prompt`.
    /// Leaves at least 1 final token in `prompt` un-reused so logits are always produced.
    pub fn match_prefix(&self, prompt: &[u32], incoming_has_images: bool) -> usize {
        match_prefix_policy(
            &self.tokens,
            self.has_images,
            self.allow_lcp_truncate,
            self.max_safe_rollback_delta,
            self.rollback_align,
            prompt,
            incoming_has_images,
        )
    }
}

/// Output of one quantum-bounded generation pass.
pub struct QuantumGenerateResult {
    pub result: StreamGenerateResult,
    /// `true` only when execution paused at a clean step boundary because `should_yield` returned `true`
    /// while more tokens could still be generated.
    pub yielded: bool,
    /// Optional post-prefill checkpoint for recurrent/SSM models where assistant `<think>` output
    /// may be stripped in subsequent turns.
    pub prefill_snapshot: Option<SequenceSnapshot>,
}

/// Options controlling quantum yielding, post-prefill snapshotting, and same-request resumption.
pub struct QuantumOptions<'a> {
    pub should_yield: &'a mut dyn FnMut(usize) -> bool,
    pub capture_prefill_snap: bool,
    pub prefill_snap_trim: usize,
    pub resume_active: bool,
}

impl<'a> QuantumOptions<'a> {
    pub fn no_yield(should_yield: &'a mut dyn FnMut(usize) -> bool) -> Self {
        Self {
            should_yield,
            capture_prefill_snap: false,
            prefill_snap_trim: 0,
            resume_active: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_snap(
        tokens: Vec<u32>,
        has_images: bool,
        allow_lcp_truncate: bool,
        max_safe_rollback_delta: Option<usize>,
        rollback_align: usize,
    ) -> SequenceSnapshot {
        SequenceSnapshot {
            tokens,
            has_images,
            allow_lcp_truncate,
            max_safe_rollback_delta,
            rollback_align,
            byte_len: 0,
            inner: EngineSnapshotData::Mm3(Mm3Snapshot {
                layers: Vec::new(),
                dspark_layers: None,
            }),
        }
    }

    #[test]
    fn test_match_prefix_policies() {
        let pure_kv = dummy_snap(vec![1, 2, 3, 4, 5, 6], false, true, None, 1);
        assert_eq!(pure_kv.match_prefix(&[1, 2, 3, 9, 10], false), 3);
        assert_eq!(pure_kv.match_prefix(&[1, 2, 3, 4, 5, 6], false), 5);
        assert_eq!(pure_kv.match_prefix(&[1, 2, 3, 4, 5, 6, 7], false), 6);
        assert_eq!(pure_kv.match_prefix(&[1, 2, 3, 4, 5, 6, 7], true), 0);

        let img_snap = dummy_snap(vec![1, 2, 3], true, true, None, 1);
        assert_eq!(img_snap.match_prefix(&[1, 2, 3, 4], false), 0);

        let dspark = dummy_snap((0..100).collect(), false, true, Some(64), 1);
        assert_eq!(dspark.match_prefix(&(0..40).collect::<Vec<_>>(), false), 39);
        assert_eq!(dspark.match_prefix(&(0..10).collect::<Vec<_>>(), false), 0);

        let hy4_aligned = dummy_snap((0..10).collect(), false, true, None, 4);
        assert_eq!(hy4_aligned.match_prefix(&[0, 1, 2, 3, 4, 5, 99], false), 4);
        assert_eq!(hy4_aligned.match_prefix(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], false), 10);

        let recurrent = dummy_snap(vec![10, 20, 30, 40], false, false, None, 1);
        assert_eq!(recurrent.match_prefix(&[10, 20, 30, 99], false), 0);
        assert_eq!(recurrent.match_prefix(&[10, 20, 30, 40], false), 0);
        assert_eq!(recurrent.match_prefix(&[10, 20, 30, 40, 50], false), 4);
    }
}
