use crate::metrics::ServerMetrics;
use crate::parser::{ChatOutputMode, OutputParser};
use crate::routes::{EngineJob, ServerModelKind, StreamEvent};
use crate::{
    collect_stop_ids, decode_for_mode, forward_parsed_event, hy4_stop_ids, mm3_stop_ids,
    qwen4_stop_ids, vision,
};
use ampllm_engine::{
    match_prefix_policy, AnyEngine, CacheStats, Hy4ImageSpan, Mm3ImageSpan, QuantumOptions,
    Qwen4ImageSpan, SamplingParams, SequenceSnapshot, StreamGenerateResult,
};
use ampllm_tokenizer::BpeTokenizer;
use anyhow::Result;
use serde::Serialize;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

/// Configuration for the multi-slot concurrent inference scheduler.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub parallel_slots: usize,
    pub slot_quantum: usize,
    pub max_slot_cache_mb: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let parallel_slots = std::env::var("AMPLLM_PARALLEL_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 16);
        let slot_quantum = std::env::var("AMPLLM_SLOT_QUANTUM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32)
            .max(1);
        let max_slot_cache_mb = std::env::var("AMPLLM_SLOT_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8192);
        Self {
            parallel_slots,
            slot_quantum,
            max_slot_cache_mb,
        }
    }
}

impl ServerConfig {
    /// Construct a `ServerConfig` using CLI `--parallel` and `--slot-quantum` overrides.
    pub fn from_cli(parallel_slots: usize, slot_quantum: usize) -> Self {
        let mut cfg = Self::default();
        cfg.parallel_slots = parallel_slots.clamp(1, 16);
        cfg.slot_quantum = slot_quantum.max(1);
        cfg
    }
}

/// Public snapshot of a single scheduler slot exposed on `GET /slots`.
#[derive(Clone, Debug, Serialize)]
pub struct SlotPublicStatus {
    pub id: usize,
    pub state: &'static str,
    pub cached_tokens: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub has_images: bool,
    pub snapshot_bytes: usize,
    pub last_used_seq: u64,
}

/// Opaque per-sequence KV/state snapshot interface used by `SlotScheduler`.
pub trait BackendSnapshot: Send + Sync + 'static {
    fn tokens(&self) -> &[u32];
    fn has_images(&self) -> bool;
    fn byte_len(&self) -> usize;
    fn match_prefix(&self, prompt: &[u32], incoming_has_images: bool) -> usize;
}

impl BackendSnapshot for SequenceSnapshot {
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn has_images(&self) -> bool {
        self.has_images
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn match_prefix(&self, prompt: &[u32], incoming_has_images: bool) -> usize {
        SequenceSnapshot::match_prefix(self, prompt, incoming_has_images)
    }
}

/// Opaque prepared multimodal image payload passed during initial prefill only.
pub trait BackendImages: Send + 'static {
    fn is_empty(&self) -> bool;
}

pub enum RealImages {
    None,
    Hy4(Vec<Hy4ImageSpan>),
    Qwen4(Vec<Qwen4ImageSpan>),
    Mm3(Vec<Mm3ImageSpan>),
}

impl BackendImages for RealImages {
    fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Output of one quantum-bounded backend execution step.
pub struct BackendQuantumResult<S> {
    pub result: StreamGenerateResult,
    pub yielded: bool,
    pub prefill_snapshot: Option<S>,
}

/// Abstraction over `AnyEngine` + `BpeTokenizer` allowing full deterministic mock tests.
pub trait InferenceBackend: 'static {
    type Snapshot: BackendSnapshot;
    type Images: BackendImages;

    fn max_ctx(&self) -> usize;
    fn supports_lcp_truncate(&self) -> bool;
    fn rollback_policy(&self) -> (bool, Option<usize>, usize);
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>>;
    fn make_decoder(&self, mode: &ChatOutputMode) -> Box<dyn Fn(&[u32]) -> String>;
    fn mode_stop_ids(&self, mode: &ChatOutputMode) -> Vec<u32>;
    fn prepare_images(
        &mut self,
        prompt_ids: &[u32],
        image_sources: &[String],
    ) -> Result<(Vec<u32>, Self::Images)>;
    fn empty_images(&self) -> Self::Images;
    fn reset(&mut self) -> Result<()>;
    fn committed_tokens(&self) -> &[u32];
    fn capture_snapshot(&mut self, has_images: bool) -> Result<Self::Snapshot>;
    fn restore_snapshot(
        &mut self,
        snap: &Self::Snapshot,
        truncate_to: Option<usize>,
    ) -> Result<()>;
    fn generate_quantum(
        &mut self,
        prompt_tokens: &[u32],
        images: &Self::Images,
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<BackendQuantumResult<Self::Snapshot>>;
    fn cache_stats_and_reuse(&self) -> (CacheStats, u64);
}

/// Production backend wrapping `AnyEngine` and `BpeTokenizer`.
pub struct RealBackend {
    engine: AnyEngine,
    tokenizer: Rc<BpeTokenizer>,
    model_kind: ServerModelKind,
    glm5_stops: Vec<u32>,
    hy4_stops: Vec<u32>,
    qwen4_stops: Vec<u32>,
    mm3_stops: Vec<u32>,
    inkling_text_stops: Vec<u32>,
    inkling_structured_stops: Vec<u32>,
    plain_stops: Vec<u32>,
}

impl RealBackend {
    pub fn new(
        engine: AnyEngine,
        tokenizer: BpeTokenizer,
        model_kind: ServerModelKind,
        hy4_tag_suffix: &str,
    ) -> Self {
        let glm5_stops = collect_stop_ids(
            &tokenizer,
            &["<|endoftext|>", "<|user|>", "<|observation|>", "<|system|>"],
        );
        let hy4_stops = hy4_stop_ids(&tokenizer, hy4_tag_suffix);
        let qwen4_stops = qwen4_stop_ids(&tokenizer);
        let mm3_stops = mm3_stop_ids(&tokenizer);
        let inkling_text_stops = collect_stop_ids(
            &tokenizer,
            &[
                "<|endoftext|>",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
                "<|message_user|>",
                "<|message_system|>",
                "<|message_tool|>",
            ],
        );
        let inkling_structured_stops = collect_stop_ids(
            &tokenizer,
            &[
                "<|endoftext|>",
                "<|content_model_end_sampling|>",
                "<|message_user|>",
                "<|message_system|>",
                "<|message_tool|>",
            ],
        );
        let plain_stops = collect_stop_ids(&tokenizer, &["<|endoftext|>"]);
        Self {
            engine,
            tokenizer: Rc::new(tokenizer),
            model_kind,
            glm5_stops,
            hy4_stops,
            qwen4_stops,
            mm3_stops,
            inkling_text_stops,
            inkling_structured_stops,
            plain_stops,
        }
    }
}

impl InferenceBackend for RealBackend {
    type Snapshot = SequenceSnapshot;
    type Images = RealImages;

    fn max_ctx(&self) -> usize {
        self.engine.max_ctx()
    }

    fn supports_lcp_truncate(&self) -> bool {
        self.engine.supports_lcp_truncate()
    }

    fn rollback_policy(&self) -> (bool, Option<usize>, usize) {
        self.engine.rollback_policy()
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(prompt, true)
    }

    fn make_decoder(&self, mode: &ChatOutputMode) -> Box<dyn Fn(&[u32]) -> String> {
        let tok = Rc::clone(&self.tokenizer);
        let mode_cloned = mode.clone();
        Box::new(move |ids: &[u32]| decode_for_mode(&tok, &mode_cloned, ids))
    }

    fn mode_stop_ids(&self, mode: &ChatOutputMode) -> Vec<u32> {
        match mode {
            ChatOutputMode::Glm5 { .. } => self.glm5_stops.clone(),
            ChatOutputMode::Hy4 { .. } => self.hy4_stops.clone(),
            ChatOutputMode::Qwen4 { .. } => self.qwen4_stops.clone(),
            ChatOutputMode::Mm3 { .. } => self.mm3_stops.clone(),
            ChatOutputMode::Inkling2 {
                start_in_text,
                parse_tools,
            } => {
                if *start_in_text && !*parse_tools {
                    self.inkling_text_stops.clone()
                } else {
                    self.inkling_structured_stops.clone()
                }
            }
            ChatOutputMode::Plain => self.plain_stops.clone(),
        }
    }

    fn prepare_images(
        &mut self,
        prompt_ids: &[u32],
        image_sources: &[String],
    ) -> Result<(Vec<u32>, Self::Images)> {
        if image_sources.is_empty() {
            return Ok((prompt_ids.to_vec(), RealImages::None));
        }
        match self.model_kind {
            ServerModelKind::Mm3 => {
                let (ids, spans) =
                    vision::prepare_and_encode_mm3_images(&mut self.engine, prompt_ids, image_sources)?;
                Ok((ids, RealImages::Mm3(spans)))
            }
            ServerModelKind::Qwen4 => {
                let (ids, spans) = vision::prepare_and_encode_qwen4_images(
                    &mut self.engine,
                    prompt_ids,
                    image_sources,
                )?;
                Ok((ids, RealImages::Qwen4(spans)))
            }
            _ => {
                let (ids, spans) =
                    vision::prepare_and_encode_hy4_images(&mut self.engine, prompt_ids, image_sources)?;
                Ok((ids, RealImages::Hy4(spans)))
            }
        }
    }

    fn empty_images(&self) -> Self::Images {
        RealImages::None
    }

    fn reset(&mut self) -> Result<()> {
        self.engine.reset()
    }

    fn committed_tokens(&self) -> &[u32] {
        self.engine.committed_tokens()
    }

    fn capture_snapshot(&mut self, has_images: bool) -> Result<Self::Snapshot> {
        self.engine.capture_snapshot(has_images)
    }

    fn restore_snapshot(
        &mut self,
        snap: &Self::Snapshot,
        truncate_to: Option<usize>,
    ) -> Result<()> {
        self.engine.restore_snapshot(snap, truncate_to)
    }

    fn generate_quantum(
        &mut self,
        prompt_tokens: &[u32],
        images: &Self::Images,
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<BackendQuantumResult<Self::Snapshot>> {
        let qres = match images {
            RealImages::None => {
                self.engine
                    .generate_stream_quantum(prompt_tokens, params, opts, |t| on_token(t))?
            }
            RealImages::Hy4(spans) => self.engine.generate_stream_quantum_with_hy4_images(
                prompt_tokens,
                spans,
                params,
                opts,
                |t| on_token(t),
            )?,
            RealImages::Qwen4(spans) => self.engine.generate_stream_quantum_with_qwen4_images(
                prompt_tokens,
                spans,
                params,
                opts,
                |t| on_token(t),
            )?,
            RealImages::Mm3(spans) => self.engine.generate_stream_quantum_with_mm3_images(
                prompt_tokens,
                spans,
                params,
                opts,
                |t| on_token(t),
            )?,
        };
        Ok(BackendQuantumResult {
            result: qres.result,
            yielded: qres.yielded,
            prefill_snapshot: qres.prefill_snapshot,
        })
    }

    fn cache_stats_and_reuse(&self) -> (CacheStats, u64) {
        (
            self.engine.cache_stats(),
            self.engine.prefix_reuse_tokens_total(),
        )
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

enum InitFrom<S> {
    Empty,
    Resident { truncate_to: usize },
    IdleSnapshot { snap: Arc<S>, truncate_to: usize },
    PausedSnapshot { snap: Arc<S> },
}

struct ActiveRequest<B: InferenceBackend> {
    job: EngineJob,
    tokens: Vec<u32>,
    prompt_len: usize,
    generated: Vec<u32>,
    pending_images: Option<B::Images>,
    has_images: bool,
    init_from: InitFrom<B::Snapshot>,
    prefill_snap: Option<Arc<B::Snapshot>>,
    has_started: bool,
    quanta_run: u64,
    emitted_bytes: usize,
    hit_stop_str: bool,
    parser: OutputParser,
    max_stop_len: usize,
    prompt_tokens_evaluated: usize,
    prefix_reused_tokens: usize,
    prefill_tps: f64,
    decode_s: f64,
    draft_n: usize,
    draft_accepted: usize,
}

enum IdleBacking<S> {
    ResidentInVram {
        tokens: Vec<u32>,
        has_images: bool,
        allow_lcp_truncate: bool,
        max_safe_rollback_delta: Option<usize>,
        rollback_align: usize,
        prefill_cached: Option<Arc<S>>,
    },
    Snapshot {
        primary: Arc<S>,
        prefill_cached: Option<Arc<S>>,
    },
}

enum SlotState<B: InferenceBackend> {
    Empty,
    Active(ActiveRequest<B>),
    Paused {
        req: ActiveRequest<B>,
        snap: Arc<B::Snapshot>,
    },
    Idle {
        backing: IdleBacking<B::Snapshot>,
        last_used_seq: u64,
    },
}

struct Slot<B: InferenceBackend> {
    id: usize,
    state: SlotState<B>,
}

enum IdleMatch<S> {
    None,
    Resident { lcp: usize, from_primary: bool },
    Snapshot { snap: Arc<S>, lcp: usize, from_primary: bool },
}

impl<S: BackendSnapshot> IdleBacking<S> {
    fn byte_len(&self) -> usize {
        match self {
            IdleBacking::ResidentInVram { prefill_cached, .. } => {
                prefill_cached.as_ref().map_or(0, |s| s.byte_len())
            }
            IdleBacking::Snapshot {
                primary,
                prefill_cached,
            } => primary.byte_len() + prefill_cached.as_ref().map_or(0, |s| s.byte_len()),
        }
    }

    fn cached_tokens(&self) -> usize {
        match self {
            IdleBacking::ResidentInVram { tokens, .. } => tokens.len(),
            IdleBacking::Snapshot { primary, .. } => primary.tokens().len(),
        }
    }

    fn has_images(&self) -> bool {
        match self {
            IdleBacking::ResidentInVram { has_images, .. } => *has_images,
            IdleBacking::Snapshot { primary, .. } => primary.has_images(),
        }
    }

    fn take_prefill_cached(&mut self) -> Option<Arc<S>> {
        match self {
            IdleBacking::ResidentInVram { prefill_cached, .. } => prefill_cached.take(),
            IdleBacking::Snapshot { prefill_cached, .. } => prefill_cached.take(),
        }
    }

    fn match_best(&self, prompt: &[u32], incoming_has_images: bool) -> IdleMatch<S> {
        match self {
            IdleBacking::ResidentInVram {
                tokens,
                has_images,
                allow_lcp_truncate,
                max_safe_rollback_delta,
                rollback_align,
                prefill_cached,
            } => {
                let p_lcp = match_prefix_policy(
                    tokens,
                    *has_images,
                    *allow_lcp_truncate,
                    *max_safe_rollback_delta,
                    *rollback_align,
                    prompt,
                    incoming_has_images,
                );
                let s_lcp = prefill_cached
                    .as_ref()
                    .map_or(0, |s| s.match_prefix(prompt, incoming_has_images));
                if p_lcp >= s_lcp && p_lcp > 0 {
                    IdleMatch::Resident {
                        lcp: p_lcp,
                        from_primary: true,
                    }
                } else if s_lcp > 0 {
                    IdleMatch::Snapshot {
                        snap: Arc::clone(prefill_cached.as_ref().unwrap()),
                        lcp: s_lcp,
                        from_primary: false,
                    }
                } else {
                    IdleMatch::None
                }
            }
            IdleBacking::Snapshot {
                primary,
                prefill_cached,
            } => {
                let p_lcp = primary.match_prefix(prompt, incoming_has_images);
                let s_lcp = prefill_cached
                    .as_ref()
                    .map_or(0, |s| s.match_prefix(prompt, incoming_has_images));
                if p_lcp >= s_lcp && p_lcp > 0 {
                    IdleMatch::Snapshot {
                        snap: Arc::clone(primary),
                        lcp: p_lcp,
                        from_primary: true,
                    }
                } else if s_lcp > 0 {
                    IdleMatch::Snapshot {
                        snap: Arc::clone(prefill_cached.as_ref().unwrap()),
                        lcp: s_lcp,
                        from_primary: false,
                    }
                } else {
                    IdleMatch::None
                }
            }
        }
    }
}

/// Coordinates up to `cfg.parallel_slots` concurrent requests over a single CUDA engine
/// using host-RAM KV/state snapshots and fair round-robin quantum scheduling.
pub struct SlotScheduler<B: InferenceBackend> {
    backend: B,
    cfg: ServerConfig,
    slots: Vec<Slot<B>>,
    wait_queue: VecDeque<EngineJob>,
    resident_slot: Option<usize>,
    next_rr_cursor: usize,
    lru_clock: u64,
    metrics: Arc<ServerMetrics>,
    pending_jobs: Arc<AtomicUsize>,
    slot_statuses: Arc<RwLock<Vec<SlotPublicStatus>>>,
}

impl<B: InferenceBackend> SlotScheduler<B> {
    pub fn new(
        backend: B,
        cfg: ServerConfig,
        metrics: Arc<ServerMetrics>,
        pending_jobs: Arc<AtomicUsize>,
        slot_statuses: Arc<RwLock<Vec<SlotPublicStatus>>>,
    ) -> Self {
        let n_slots = cfg.parallel_slots.clamp(1, 16);
        let slots = (0..n_slots)
            .map(|id| Slot {
                id,
                state: SlotState::Empty,
            })
            .collect();
        metrics.slots_total.store(n_slots as u64, Ordering::Relaxed);
        let sched = Self {
            backend,
            cfg,
            slots,
            wait_queue: VecDeque::new(),
            resident_slot: None,
            next_rr_cursor: 0,
            lru_clock: 0,
            metrics,
            pending_jobs,
            slot_statuses,
        };
        sched.publish_status(None);
        sched
    }

    pub fn run(mut self, mut rx: mpsc::UnboundedReceiver<EngineJob>) {
        loop {
            while let Ok(job) = rx.try_recv() {
                self.wait_queue.push_back(job);
            }

            self.prune_cancelled();
            self.admit_queued_jobs();

            if let Some(slot_id) = self.pick_next_runnable_slot() {
                self.publish_status(Some(slot_id));
                self.run_one_quantum(slot_id);
                self.publish_status(None);
                continue;
            }

            self.publish_status(None);
            match rx.blocking_recv() {
                Some(job) => self.wait_queue.push_back(job),
                None => break,
            }
        }
    }

    fn next_lru_seq(&mut self) -> u64 {
        self.lru_clock += 1;
        self.lru_clock
    }

    fn prune_cancelled(&mut self) {
        let mut kept = VecDeque::with_capacity(self.wait_queue.len());
        while let Some(job) = self.wait_queue.pop_front() {
            if job.resp_tx.is_closed() {
                self.pending_jobs.fetch_sub(1, Ordering::Relaxed);
                self.metrics
                    .requests_cancelled_total
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                kept.push_back(job);
            }
        }
        self.wait_queue = kept;

        for slot in &mut self.slots {
            let cancelled = match &slot.state {
                SlotState::Active(req) | SlotState::Paused { req, .. } => {
                    req.job.resp_tx.is_closed()
                }
                SlotState::Empty | SlotState::Idle { .. } => false,
            };
            if cancelled {
                slot.state = SlotState::Empty;
                if self.resident_slot == Some(slot.id) {
                    self.resident_slot = None;
                }
                self.metrics
                    .requests_cancelled_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn has_admittable_slot(&self) -> bool {
        self.slots
            .iter()
            .any(|s| matches!(s.state, SlotState::Empty | SlotState::Idle { .. }))
    }

    fn admit_queued_jobs(&mut self) {
        while self.has_admittable_slot() {
            let Some(mut job) = self.wait_queue.pop_front() else {
                break;
            };
            self.pending_jobs.fetch_sub(1, Ordering::Relaxed);

            if job.resp_tx.is_closed() {
                self.metrics
                    .requests_cancelled_total
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            for s in self.backend.mode_stop_ids(&job.output_mode) {
                if !job.params.stop_token_ids.contains(&s) {
                    job.params.stop_token_ids.push(s);
                }
            }

            let prompt_ids = match self.backend.encode_prompt(&job.prompt) {
                Ok(ids) => ids,
                Err(e) => {
                    let _ = job.resp_tx.send(StreamEvent::Error(e.to_string()));
                    continue;
                }
            };
            let has_images = !job.image_sources.is_empty();
            let (eval_ids, pending_images) = if has_images {
                match self
                    .backend
                    .prepare_images(&prompt_ids, &job.image_sources)
                {
                    Ok((ids, imgs)) => (ids, Some(imgs)),
                    Err(e) => {
                        let _ = job.resp_tx.send(StreamEvent::Error(format!("{e:#}")));
                        continue;
                    }
                }
            } else {
                (prompt_ids, None)
            };

            let max_ctx = self.backend.max_ctx();
            if eval_ids.len() >= max_ctx {
                let _ = job.resp_tx.send(StreamEvent::Error(format!(
                    "prompt length {} >= max_ctx {max_ctx}",
                    eval_ids.len()
                )));
                continue;
            }
            let avail_tokens = max_ctx.saturating_sub(eval_ids.len()).max(1);
            job.params.max_tokens = job.params.max_tokens.min(avail_tokens);

            let (slot_id, init_from) = self.choose_slot_for_prompt(&eval_ids, has_images);
            let max_stop_len = job.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);
            let parser = OutputParser::new(job.output_mode.clone(), job.call_id_prefix.clone());
            let prompt_len = eval_ids.len();

            self.slots[slot_id].state = SlotState::Active(ActiveRequest {
                job,
                tokens: eval_ids,
                prompt_len,
                generated: Vec::new(),
                pending_images,
                has_images,
                init_from,
                prefill_snap: None,
                has_started: false,
                quanta_run: 0,
                emitted_bytes: 0,
                hit_stop_str: false,
                parser,
                max_stop_len,
                prompt_tokens_evaluated: 0,
                prefix_reused_tokens: 0,
                prefill_tps: 0.0,
                decode_s: 0.0,
                draft_n: 0,
                draft_accepted: 0,
            });
        }
    }

    fn choose_slot_for_prompt(
        &self,
        eval_ids: &[u32],
        incoming_has_images: bool,
    ) -> (usize, InitFrom<B::Snapshot>) {
        // 1. Best longest-prefix match among Idle slots
        let mut best_match: Option<(usize, usize, bool, u64, InitFrom<B::Snapshot>)> = None;
        for slot in &self.slots {
            if let SlotState::Idle {
                backing,
                last_used_seq,
            } = &slot.state
            {
                let (lcp, from_primary, init) = match backing.match_best(eval_ids, incoming_has_images)
                {
                    IdleMatch::Resident { lcp, from_primary } => (
                        lcp,
                        from_primary,
                        InitFrom::Resident { truncate_to: lcp },
                    ),
                    IdleMatch::Snapshot {
                        snap,
                        lcp,
                        from_primary,
                    } => (
                        lcp,
                        from_primary,
                        InitFrom::IdleSnapshot {
                            snap,
                            truncate_to: lcp,
                        },
                    ),
                    IdleMatch::None => continue,
                };
                let replace = match &best_match {
                    None => true,
                    Some((_, b_lcp, b_prim, b_seq, _)) => {
                        (lcp, from_primary, *last_used_seq) > (*b_lcp, *b_prim, *b_seq)
                    }
                };
                if replace {
                    best_match = Some((slot.id, lcp, from_primary, *last_used_seq, init));
                }
            }
        }
        if let Some((id, _, _, _, init)) = best_match {
            return (id, init);
        }

        // 2. Otherwise prefer Empty slot (never evicts warm Idle prefix cache prematurely)
        for slot in &self.slots {
            if matches!(slot.state, SlotState::Empty) {
                return (slot.id, InitFrom::Empty);
            }
        }

        // 3. Otherwise evict LRU Idle slot
        let lru_id = self
            .slots
            .iter()
            .filter_map(|s| match &s.state {
                SlotState::Idle { last_used_seq, .. } => Some((s.id, *last_used_seq)),
                _ => None,
            })
            .min_by_key(|&(_, seq)| seq)
            .map(|(id, _)| id)
            .expect("has_admittable_slot guaranteed at least one Empty or Idle slot");
        (lru_id, InitFrom::Empty)
    }

    fn pick_next_runnable_slot(&mut self) -> Option<usize> {
        let n = self.slots.len();
        for offset in 0..n {
            let idx = (self.next_rr_cursor + offset) % n;
            if matches!(
                self.slots[idx].state,
                SlotState::Active(_) | SlotState::Paused { .. }
            ) {
                self.next_rr_cursor = (idx + 1) % n;
                return Some(idx);
            }
        }
        None
    }

    fn offload_resident_idle_if_needed(&mut self, incoming_slot_id: usize) {
        let Some(res_id) = self.resident_slot else {
            return;
        };
        if res_id == incoming_slot_id {
            return;
        }
        match &mut self.slots[res_id].state {
            SlotState::Idle { backing, .. } => {
                let should_capture = matches!(
                    backing,
                    IdleBacking::ResidentInVram { tokens, .. }
                        if !tokens.is_empty() && self.cfg.max_slot_cache_mb > 0
                );
                if should_capture {
                    let (has_images, prefill_cached) = match backing {
                        IdleBacking::ResidentInVram {
                            has_images,
                            prefill_cached,
                            ..
                        } => (*has_images, prefill_cached.take()),
                        _ => unreachable!(),
                    };
                    match self.backend.capture_snapshot(has_images) {
                        Ok(snap) => {
                            *backing = IdleBacking::Snapshot {
                                primary: Arc::new(snap),
                                prefill_cached,
                            };
                        }
                        Err(e) => {
                            tracing::warn!("failed to offload idle slot {res_id} snapshot: {e:#}");
                            self.slots[res_id].state = SlotState::Empty;
                        }
                    }
                } else if matches!(backing, IdleBacking::ResidentInVram { .. }) {
                    self.slots[res_id].state = SlotState::Empty;
                }
            }
            SlotState::Active(req) => {
                if !req.has_started {
                    if let InitFrom::Resident { truncate_to } = req.init_from {
                        let has_committed = !self.backend.committed_tokens().is_empty();
                        if has_committed && self.cfg.max_slot_cache_mb > 0 {
                            match self.backend.capture_snapshot(false) {
                                Ok(snap) => {
                                    req.init_from = InitFrom::IdleSnapshot {
                                        snap: Arc::new(snap),
                                        truncate_to,
                                    };
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "failed to offload pending resident slot {res_id}: {e:#}"
                                    );
                                    req.init_from = InitFrom::Empty;
                                }
                            }
                        } else {
                            req.init_from = InitFrom::Empty;
                        }
                    }
                }
            }
            SlotState::Empty | SlotState::Paused { .. } => {}
        }
        self.resident_slot = None;
        self.evict_snapshots_over_budget();
    }

    fn ensure_resident_for_slot(
        &mut self,
        slot_id: usize,
        init_from: &InitFrom<B::Snapshot>,
    ) -> Result<()> {
        match init_from {
            InitFrom::Resident { .. } => {
                if self.resident_slot != Some(slot_id) {
                    if self.resident_slot.is_some() {
                        self.metrics.slot_swaps_total.fetch_add(1, Ordering::Relaxed);
                        self.offload_resident_idle_if_needed(slot_id);
                    }
                    if !self.backend.committed_tokens().is_empty() {
                        self.backend.reset()?;
                    }
                    self.resident_slot = Some(slot_id);
                }
                Ok(())
            }
            InitFrom::PausedSnapshot { snap } => {
                if self.resident_slot == Some(slot_id) {
                    return Ok(());
                }
                if self.resident_slot.is_some() {
                    self.metrics.slot_swaps_total.fetch_add(1, Ordering::Relaxed);
                }
                self.offload_resident_idle_if_needed(slot_id);
                self.backend.restore_snapshot(snap, None)?;
                self.resident_slot = Some(slot_id);
                Ok(())
            }
            InitFrom::IdleSnapshot { snap, truncate_to } => {
                let swapped_from_other = self.resident_slot.is_some_and(|r| r != slot_id);
                if swapped_from_other {
                    self.metrics.slot_swaps_total.fetch_add(1, Ordering::Relaxed);
                }
                self.offload_resident_idle_if_needed(slot_id);
                let trunc = if *truncate_to < snap.tokens().len() {
                    Some(*truncate_to)
                } else {
                    None
                };
                self.backend.restore_snapshot(snap, trunc)?;
                self.resident_slot = Some(slot_id);
                Ok(())
            }
            InitFrom::Empty => {
                if self.resident_slot.is_some_and(|r| r != slot_id) {
                    self.metrics.slot_swaps_total.fetch_add(1, Ordering::Relaxed);
                    self.offload_resident_idle_if_needed(slot_id);
                }
                if !self.backend.committed_tokens().is_empty() || self.resident_slot.is_some() {
                    self.backend.reset()?;
                }
                self.resident_slot = Some(slot_id);
                Ok(())
            }
        }
    }

    fn run_one_quantum(&mut self, slot_id: usize) {
        let taken = std::mem::replace(&mut self.slots[slot_id].state, SlotState::Empty);
        let mut req = match taken {
            SlotState::Active(r) => r,
            SlotState::Paused { mut req, snap } => {
                req.init_from = InitFrom::PausedSnapshot { snap };
                req
            }
            other => {
                self.slots[slot_id].state = other;
                return;
            }
        };

        if req.job.resp_tx.is_closed() {
            if self.resident_slot == Some(slot_id) {
                self.resident_slot = None;
            }
            self.metrics
                .requests_cancelled_total
                .fetch_add(1, Ordering::Relaxed);
            return;
        }

        if let Err(e) = self.ensure_resident_for_slot(slot_id, &req.init_from) {
            tracing::error!("slot {slot_id} state restore failed: {e:#}");
            let _ = req.job.resp_tx.send(StreamEvent::Error(format!("{e:#}")));
            self.resident_slot = None;
            let _ = self.backend.reset();
            return;
        }

        let other_slots_pending = self
            .slots
            .iter()
            .enumerate()
            .any(|(i, s)| i != slot_id && matches!(s.state, SlotState::Active(_) | SlotState::Paused { .. }));
        let has_waiters_now = !self.wait_queue.is_empty();
        let quantum_limit = self.cfg.slot_quantum;
        let capture_prefill_snap = !req.has_started
            && !self.backend.supports_lcp_truncate()
            && !req.has_images
            && self.cfg.max_slot_cache_mb > 0;
        let resume_active = req.has_started;
        let was_started = req.has_started;

        let remaining = req.job.params.max_tokens.saturating_sub(req.generated.len());
        if remaining == 0 {
            self.finish_request(slot_id, req);
            return;
        }

        let mut q_params = req.job.params.clone();
        q_params.max_tokens = remaining;
        if req.quanta_run > 0 && q_params.temperature > 0.0 {
            q_params.seed = splitmix64(req.job.params.seed.wrapping_add(req.quanta_run));
        }

        let images = if !was_started {
            req.pending_images
                .take()
                .unwrap_or_else(|| self.backend.empty_images())
        } else {
            self.backend.empty_images()
        };

        let decoder = self.backend.make_decoder(&req.job.output_mode);
        let prompt_slice = &req.tokens;
        let generated_ref = &mut req.generated;
        let emitted_ref = &mut req.emitted_bytes;
        let hit_stop_ref = &mut req.hit_stop_str;
        let parser_ref = &mut req.parser;
        let stop_strings = &req.job.stop_strings;
        let max_stop_len = req.max_stop_len;
        let resp_tx = &req.job.resp_tx;
        let pending_ref = &self.pending_jobs;
        let mut yielded_by_cancel = false;

        let mut on_tok_cb = |tok_id: u32| -> bool {
            generated_ref.push(tok_id);
            let decoded = decoder(generated_ref);
            if !stop_strings.is_empty() {
                for stop_s in stop_strings {
                    if let Some(pos) = decoded.find(stop_s.as_str()) {
                        if pos > *emitted_ref && decoded.is_char_boundary(pos) {
                            parser_ref.push(&decoded[*emitted_ref..pos], &mut |ev| {
                                forward_parsed_event(ev, resp_tx);
                            });
                            *emitted_ref = pos;
                        }
                        *hit_stop_ref = true;
                        return false;
                    }
                }
            }
            let mut send_until = decoded.len();
            if decoded.ends_with('\u{FFFD}') {
                send_until = decoded.len().saturating_sub('\u{FFFD}'.len_utf8());
            } else if max_stop_len > 1 {
                let keep_tail = max_stop_len - 1;
                let target = decoded.len().saturating_sub(keep_tail);
                let mut p = target;
                while p > *emitted_ref && !decoded.is_char_boundary(p) {
                    p -= 1;
                }
                send_until = p;
            }
            if send_until > *emitted_ref {
                let piece = &decoded[*emitted_ref..send_until];
                *emitted_ref = send_until;
                parser_ref.push(piece, &mut |ev| forward_parsed_event(ev, resp_tx));
            }
            true
        };

        let multi_slot = self.slots.len() > 1;
        let mut should_yield_cb = |n_gen_in_quantum: usize| -> bool {
            if resp_tx.is_closed() {
                yielded_by_cancel = true;
                return true;
            }
            if !multi_slot || n_gen_in_quantum < quantum_limit {
                return false;
            }
            other_slots_pending || has_waiters_now || pending_ref.load(Ordering::Relaxed) > 0
        };

        let prefill_snap_trim = if capture_prefill_snap
            && matches!(
                req.job.output_mode,
                ChatOutputMode::Qwen4 {
                    expect_think_close: true,
                    ..
                }
            )
        {
            1
        } else {
            0
        };

        let opts = QuantumOptions {
            should_yield: &mut should_yield_cb,
            capture_prefill_snap,
            prefill_snap_trim,
            resume_active,
        };

        let res = self.backend.generate_quantum(
            prompt_slice,
            &images,
            &q_params,
            opts,
            &mut on_tok_cb,
        );

        match res {
            Ok(qres) => {
                if !was_started {
                    req.prompt_tokens_evaluated = qres.result.prompt_tokens_evaluated;
                    req.prefix_reused_tokens = qres.result.prefix_reused_tokens;
                    req.prefill_tps = qres.result.prefill_tps;
                    if let Some(ps) = qres.prefill_snapshot {
                        req.prefill_snap = Some(Arc::new(ps));
                    }
                    self.metrics
                        .tokens_prefilled_total
                        .fetch_add(qres.result.prompt_tokens_evaluated as u64, Ordering::Relaxed);
                }

                let gen_len = qres.result.generated_tokens.len();
                self.metrics
                    .tokens_generated_total
                    .fetch_add(gen_len as u64, Ordering::Relaxed);
                let (cstats, reuse_tot) = self.backend.cache_stats_and_reuse();
                self.metrics.update_from_engine(&cstats, reuse_tot);

                if qres.result.decode_tps > 0.0 && gen_len > 0 {
                    req.decode_s += (gen_len as f64) / qres.result.decode_tps;
                }
                req.draft_n += qres.result.draft_n;
                req.draft_accepted += qres.result.draft_accepted;
                req.tokens.extend_from_slice(&qres.result.generated_tokens);
                req.has_started = true;
                req.quanta_run += 1;

                if yielded_by_cancel || req.job.resp_tx.is_closed() {
                    self.resident_slot = None;
                    self.metrics
                        .requests_cancelled_total
                        .fetch_add(1, Ordering::Relaxed);
                    return;
                }

                let reached_limit = req.generated.len() >= req.job.params.max_tokens;
                if qres.yielded && !req.hit_stop_str && !reached_limit {
                    match self.backend.capture_snapshot(req.has_images) {
                        Ok(snap) => {
                            req.init_from = InitFrom::Resident { truncate_to: 0 };
                            self.resident_slot = Some(slot_id);
                            self.slots[slot_id].state = SlotState::Paused {
                                req,
                                snap: Arc::new(snap),
                            };
                            self.evict_snapshots_over_budget();
                        }
                        Err(e) => {
                            tracing::error!("slot {slot_id} capture_snapshot failed: {e:#}");
                            let _ = req.job.resp_tx.send(StreamEvent::Error(format!("{e:#}")));
                            self.resident_slot = None;
                            let _ = self.backend.reset();
                        }
                    }
                } else {
                    self.finish_request(slot_id, req);
                }
            }
            Err(e) => {
                tracing::error!("infer worker error in slot {slot_id}: {e:#}");
                let _ = req.job.resp_tx.send(StreamEvent::Error(format!("{e:#}")));
                self.resident_slot = None;
                let _ = self.backend.reset();
            }
        }
    }

    fn finish_request(&mut self, slot_id: usize, mut req: ActiveRequest<B>) {
        if !req.hit_stop_str {
            let decoder = self.backend.make_decoder(&req.job.output_mode);
            let decoded = decoder(&req.generated);
            if decoded.len() > req.emitted_bytes {
                let tail = &decoded[req.emitted_bytes..];
                req.parser.push(tail, &mut |ev| {
                    forward_parsed_event(ev, &req.job.resp_tx);
                });
            }
        }
        req.parser
            .finish(&mut |ev| forward_parsed_event(ev, &req.job.resp_tx));

        let finish_reason = if req.parser.has_tool_calls() {
            "tool_calls".to_owned()
        } else if !req.hit_stop_str && req.generated.len() >= req.job.params.max_tokens {
            "length".to_owned()
        } else {
            "stop".to_owned()
        };
        let decode_tps = if req.decode_s > 0.0 {
            (req.generated.len() as f64) / req.decode_s
        } else {
            0.0
        };
        let _ = req.job.resp_tx.send(StreamEvent::Done {
            prompt_tokens: req.prompt_tokens_evaluated,
            completion_tokens: req.generated.len(),
            prefix_reused_tokens: req.prefix_reused_tokens,
            finish_reason,
            prefill_tps: req.prefill_tps,
            decode_tps,
            draft_n: req.draft_n,
            draft_accepted: req.draft_accepted,
        });

        let (allow_lcp_truncate, max_safe_rollback_delta, rollback_align) =
            self.backend.rollback_policy();
        let seq = self.next_lru_seq();
        let committed = self.backend.committed_tokens().to_vec();
        self.resident_slot = Some(slot_id);
        self.slots[slot_id].state = SlotState::Idle {
            backing: IdleBacking::ResidentInVram {
                tokens: committed,
                has_images: req.has_images,
                allow_lcp_truncate,
                max_safe_rollback_delta,
                rollback_align,
                prefill_cached: req.prefill_snap.take(),
            },
            last_used_seq: seq,
        };
        self.evict_snapshots_over_budget();
    }

    fn total_snapshot_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| match &s.state {
                SlotState::Empty => 0,
                SlotState::Active(req) => req.prefill_snap.as_ref().map_or(0, |p| p.byte_len()),
                SlotState::Paused { req, snap } => {
                    snap.byte_len() + req.prefill_snap.as_ref().map_or(0, |p| p.byte_len())
                }
                SlotState::Idle { backing, .. } => backing.byte_len(),
            })
            .sum()
    }

    fn evict_snapshots_over_budget(&mut self) {
        let budget = self.cfg.max_slot_cache_mb.saturating_mul(1024 * 1024);
        while self.total_snapshot_bytes() > budget {
            let victim = self
                .slots
                .iter()
                .filter_map(|s| match &s.state {
                    SlotState::Idle {
                        backing,
                        last_used_seq,
                    } if backing.byte_len() > 0 => Some((s.id, *last_used_seq)),
                    _ => None,
                })
                .min_by_key(|&(_, seq)| seq)
                .map(|(id, _)| id);
            let Some(vid) = victim else {
                break;
            };
            let drop_slot = match &mut self.slots[vid].state {
                SlotState::Idle { backing, .. } => {
                    if backing.take_prefill_cached().is_some() {
                        matches!(
                            backing,
                            IdleBacking::Snapshot { primary, .. } if primary.byte_len() == 0
                        )
                    } else {
                        true
                    }
                }
                _ => false,
            };
            if drop_slot {
                self.slots[vid].state = SlotState::Empty;
            }
        }
    }

    fn publish_status(&self, running_slot: Option<usize>) {
        let mut active_count = 0u64;
        let mut out = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            let st = match &s.state {
                SlotState::Empty => {
                    if running_slot == Some(s.id) {
                        active_count += 1;
                        SlotPublicStatus {
                            id: s.id,
                            state: "active",
                            cached_tokens: 0,
                            prompt_tokens: 0,
                            generated_tokens: 0,
                            has_images: false,
                            snapshot_bytes: 0,
                            last_used_seq: 0,
                        }
                    } else {
                        SlotPublicStatus {
                            id: s.id,
                            state: "empty",
                            cached_tokens: 0,
                            prompt_tokens: 0,
                            generated_tokens: 0,
                            has_images: false,
                            snapshot_bytes: 0,
                            last_used_seq: 0,
                        }
                    }
                }
                SlotState::Active(req) => {
                    active_count += 1;
                    SlotPublicStatus {
                        id: s.id,
                        state: "active",
                        cached_tokens: req.tokens.len(),
                        prompt_tokens: req.prompt_len,
                        generated_tokens: req.generated.len(),
                        has_images: req.has_images,
                        snapshot_bytes: req.prefill_snap.as_ref().map_or(0, |p| p.byte_len()),
                        last_used_seq: 0,
                    }
                }
                SlotState::Paused { req, snap } => {
                    active_count += 1;
                    SlotPublicStatus {
                        id: s.id,
                        state: "paused",
                        cached_tokens: snap.tokens().len(),
                        prompt_tokens: req.prompt_len,
                        generated_tokens: req.generated.len(),
                        has_images: req.has_images,
                        snapshot_bytes: snap.byte_len()
                            + req.prefill_snap.as_ref().map_or(0, |p| p.byte_len()),
                        last_used_seq: 0,
                    }
                }
                SlotState::Idle {
                    backing,
                    last_used_seq,
                } => SlotPublicStatus {
                    id: s.id,
                    state: "idle",
                    cached_tokens: backing.cached_tokens(),
                    prompt_tokens: 0,
                    generated_tokens: 0,
                    has_images: backing.has_images(),
                    snapshot_bytes: backing.byte_len(),
                    last_used_seq: *last_used_seq,
                },
            };
            out.push(st);
        }

        self.metrics
            .slots_active
            .store(active_count, Ordering::Relaxed);
        let qdepth = self.pending_jobs.load(Ordering::Relaxed) as u64;
        self.metrics.queue_depth.store(qdepth, Ordering::Relaxed);
        if let Ok(mut guard) = self.slot_statuses.write() {
            *guard = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[derive(Clone, Debug)]
    struct MockSnap {
        tokens: Vec<u32>,
        has_images: bool,
        allow_lcp_truncate: bool,
        max_safe_rollback_delta: Option<usize>,
        rollback_align: usize,
        byte_len: usize,
        step_offset: usize,
    }

    impl BackendSnapshot for MockSnap {
        fn tokens(&self) -> &[u32] {
            &self.tokens
        }
        fn has_images(&self) -> bool {
            self.has_images
        }
        fn byte_len(&self) -> usize {
            self.byte_len
        }
        fn match_prefix(&self, prompt: &[u32], incoming_has_images: bool) -> usize {
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

    struct MockEmptyImages;
    impl BackendImages for MockEmptyImages {
        fn is_empty(&self) -> bool {
            true
        }
    }

    #[derive(Default)]
    struct MockStats {
        capture_calls: AtomicUsize,
        restore_calls: AtomicUsize,
        reset_calls: AtomicUsize,
        decode_steps: AtomicUsize,
        max_active_slots_observed: AtomicUsize,
        event_log: Mutex<Vec<String>>,
    }

    struct MockBackend {
        max_ctx: usize,
        allow_lcp_truncate: bool,
        vram_tokens: Vec<u32>,
        step_offset: usize,
        stats: Arc<MockStats>,
        slot_statuses: Option<Arc<RwLock<Vec<SlotPublicStatus>>>>,
        per_token_delay_ms: u64,
    }

    impl MockBackend {
        fn new(allow_lcp_truncate: bool, stats: Arc<MockStats>) -> Self {
            Self {
                max_ctx: 4096,
                allow_lcp_truncate,
                vram_tokens: Vec::new(),
                step_offset: 0,
                stats,
                slot_statuses: None,
                per_token_delay_ms: 0,
            }
        }
    }

    impl InferenceBackend for MockBackend {
        type Snapshot = MockSnap;
        type Images = MockEmptyImages;

        fn max_ctx(&self) -> usize {
            self.max_ctx
        }

        fn supports_lcp_truncate(&self) -> bool {
            self.allow_lcp_truncate
        }

        fn rollback_policy(&self) -> (bool, Option<usize>, usize) {
            (self.allow_lcp_truncate, None, 1)
        }

        fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
            // Comma-separated integers if all parts parse as u32, else char codepoints + 100
            let trimmed = prompt.trim();
            if trimmed
                .split(',')
                .all(|p| !p.is_empty() && p.trim().parse::<u32>().is_ok())
            {
                Ok(trimmed
                    .split(',')
                    .map(|p| p.trim().parse::<u32>().unwrap())
                    .collect())
            } else {
                Ok(prompt.bytes().map(|b| 100 + b as u32).collect())
            }
        }

        fn make_decoder(&self, _mode: &ChatOutputMode) -> Box<dyn Fn(&[u32]) -> String> {
            Box::new(|ids: &[u32]| {
                ids.iter()
                    .map(|t| format!("[{t}]"))
                    .collect::<Vec<_>>()
                    .join("")
            })
        }

        fn mode_stop_ids(&self, _mode: &ChatOutputMode) -> Vec<u32> {
            vec![999_999]
        }

        fn prepare_images(
            &mut self,
            prompt_ids: &[u32],
            _image_sources: &[String],
        ) -> Result<(Vec<u32>, Self::Images)> {
            Ok((prompt_ids.to_vec(), MockEmptyImages))
        }

        fn empty_images(&self) -> Self::Images {
            MockEmptyImages
        }

        fn reset(&mut self) -> Result<()> {
            self.stats.reset_calls.fetch_add(1, Ordering::Relaxed);
            self.vram_tokens.clear();
            self.step_offset = 0;
            Ok(())
        }

        fn committed_tokens(&self) -> &[u32] {
            &self.vram_tokens
        }

        fn capture_snapshot(&mut self, has_images: bool) -> Result<Self::Snapshot> {
            self.stats.capture_calls.fetch_add(1, Ordering::Relaxed);
            Ok(MockSnap {
                tokens: self.vram_tokens.clone(),
                has_images,
                allow_lcp_truncate: self.allow_lcp_truncate,
                max_safe_rollback_delta: None,
                rollback_align: 1,
                byte_len: self.vram_tokens.len().max(1) * 1024,
                step_offset: self.step_offset,
            })
        }

        fn restore_snapshot(
            &mut self,
            snap: &Self::Snapshot,
            truncate_to: Option<usize>,
        ) -> Result<()> {
            self.stats.restore_calls.fetch_add(1, Ordering::Relaxed);
            let n = truncate_to.unwrap_or(snap.tokens.len()).min(snap.tokens.len());
            self.vram_tokens = snap.tokens[..n].to_vec();
            self.step_offset = snap.step_offset.min(n);
            Ok(())
        }

        fn generate_quantum(
            &mut self,
            prompt_tokens: &[u32],
            _images: &Self::Images,
            params: &SamplingParams,
            opts: QuantumOptions<'_>,
            on_token: &mut dyn FnMut(u32) -> bool,
        ) -> Result<BackendQuantumResult<Self::Snapshot>> {
            if let Some(ref st) = self.slot_statuses {
                if let Ok(g) = st.read() {
                    let act = g
                        .iter()
                        .filter(|s| s.state == "active" || s.state == "paused")
                        .count();
                    self.stats
                        .max_active_slots_observed
                        .fetch_max(act, Ordering::Relaxed);
                }
            }

            let lcp = match_prefix_policy(
                &self.vram_tokens,
                false,
                self.allow_lcp_truncate,
                None,
                1,
                prompt_tokens,
                false,
            );
            if lcp == 0 {
                self.vram_tokens.clear();
            } else {
                self.vram_tokens.truncate(lcp);
            }
            let suffix = &prompt_tokens[lcp..];
            let base = prompt_tokens[0];

            let prefill_snap = if opts.capture_prefill_snap && !opts.resume_active {
                let keep = prompt_tokens.len().saturating_sub(opts.prefill_snap_trim);
                self.vram_tokens = prompt_tokens[..keep].to_vec();
                Some(self.capture_snapshot(false)?)
            } else {
                None
            };

            let resuming = opts.resume_active && lcp > 0 && suffix.len() == 1;
            let mut produced = Vec::new();
            let mut yielded = false;
            let mut next_tok = if resuming {
                // Commit the single pending emitted token, produce next token
                self.vram_tokens.push(suffix[0]);
                base * 1000 + (self.vram_tokens.len() as u32)
            } else {
                self.vram_tokens = prompt_tokens.to_vec();
                base * 1000 + (self.vram_tokens.len() as u32)
            };

            for step in 0..params.max_tokens {
                if self.per_token_delay_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(self.per_token_delay_ms));
                }
                self.stats.decode_steps.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .event_log
                    .lock()
                    .unwrap()
                    .push(format!("{base}:{next_tok}"));
                produced.push(next_tok);
                if !on_token(next_tok) || step + 1 >= params.max_tokens {
                    break;
                }
                if (opts.should_yield)(produced.len()) {
                    yielded = true;
                    break;
                }
                self.vram_tokens.push(next_tok);
                next_tok = base * 1000 + (self.vram_tokens.len() as u32);
            }

            Ok(BackendQuantumResult {
                result: StreamGenerateResult {
                    prompt_tokens_evaluated: if resuming { 0 } else { suffix.len() },
                    prefix_reused_tokens: if opts.resume_active { 0 } else { lcp },
                    generated_tokens: produced,
                    prefill_tps: 100.0,
                    decode_tps: 100.0,
                    draft_n: 0,
                    draft_accepted: 0,
                },
                yielded,
                prefill_snapshot: prefill_snap,
            })
        }

        fn cache_stats_and_reuse(&self) -> (CacheStats, u64) {
            (CacheStats::default(), 0)
        }
    }

    fn make_job(prompt: &str, max_tokens: usize) -> (EngineJob, mpsc::UnboundedReceiver<StreamEvent>) {
        let (resp_tx, resp_rx) = mpsc::unbounded_channel();
        let job = EngineJob {
            prompt: prompt.to_owned(),
            image_sources: Vec::new(),
            params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                seed: 42,
                max_tokens,
                stop_token_ids: Vec::new(),
            },
            stop_strings: Vec::new(),
            output_mode: ChatOutputMode::Plain,
            call_id_prefix: "call_0".to_owned(),
            resp_tx,
        };
        (job, resp_rx)
    }

    fn collect_text_and_done(
        mut rx: mpsc::UnboundedReceiver<StreamEvent>,
    ) -> (String, usize, usize) {
        let mut txt = String::new();
        while let Some(ev) = rx.blocking_recv() {
            match ev {
                StreamEvent::Token(t) => txt.push_str(&t),
                StreamEvent::Done {
                    completion_tokens,
                    prefix_reused_tokens,
                    ..
                } => return (txt, completion_tokens, prefix_reused_tokens),
                StreamEvent::Error(e) => panic!("unexpected StreamEvent::Error: {e}"),
                _ => {}
            }
        }
        panic!("channel closed without Done");
    }

    #[test]
    fn test_four_parallel_interleaved_requests() {
        let stats = Arc::new(MockStats::default());
        let backend = MockBackend::new(true, Arc::clone(&stats));
        let metrics = Arc::new(ServerMetrics::default());
        let pending = Arc::new(AtomicUsize::new(0));
        let statuses = Arc::new(RwLock::new(Vec::new()));
        let cfg = ServerConfig {
            parallel_slots: 4,
            slot_quantum: 3,
            max_slot_cache_mb: 64,
        };
        let sched = SlotScheduler::new(
            backend,
            cfg,
            Arc::clone(&metrics),
            Arc::clone(&pending),
            Arc::clone(&statuses),
        );
        let (tx, rx) = mpsc::unbounded_channel();

        let mut receivers = Vec::new();
        for base in [1u32, 2, 3, 4] {
            let (job, r) = make_job(&format!("{base}"), 10);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job).unwrap();
            receivers.push((base, r));
        }
        let handle = std::thread::spawn(move || sched.run(rx));

        for (base, r) in receivers {
            let (txt, comp, _) = collect_text_and_done(r);
            assert_eq!(comp, 10);
            let expected: String = (1..=10)
                .map(|i| format!("[{}]", base * 1000 + i))
                .collect();
            assert_eq!(txt, expected);
        }
        drop(tx);
        handle.join().unwrap();

        assert!(metrics.slot_swaps_total.load(Ordering::Relaxed) >= 3);
        // Verify log shows interleaved quanta (base 1 appears both before and after base 2)
        let log = stats.event_log.lock().unwrap();
        let first_b2 = log.iter().position(|e| e.starts_with("2:")).unwrap();
        let last_b1 = log.iter().rposition(|e| e.starts_with("1:")).unwrap();
        assert!(last_b1 > first_b2, "expected interleaved tokens: {log:?}");
    }

    #[test]
    fn test_queue_overflow_six_requests_four_slots() {
        let stats = Arc::new(MockStats::default());
        let mut backend = MockBackend::new(true, Arc::clone(&stats));
        let metrics = Arc::new(ServerMetrics::default());
        let pending = Arc::new(AtomicUsize::new(0));
        let statuses = Arc::new(RwLock::new(Vec::new()));
        backend.slot_statuses = Some(Arc::clone(&statuses));
        let cfg = ServerConfig {
            parallel_slots: 4,
            slot_quantum: 2,
            max_slot_cache_mb: 64,
        };
        let sched = SlotScheduler::new(
            backend,
            cfg,
            Arc::clone(&metrics),
            Arc::clone(&pending),
            Arc::clone(&statuses),
        );
        let (tx, rx) = mpsc::unbounded_channel();

        let mut receivers = Vec::new();
        for base in 1..=6u32 {
            let (job, r) = make_job(&format!("{base}"), 6);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job).unwrap();
            receivers.push((base, r));
        }
        assert_eq!(pending.load(Ordering::Relaxed), 6);
        let handle = std::thread::spawn(move || sched.run(rx));

        for (base, r) in receivers {
            let (txt, comp, _) = collect_text_and_done(r);
            assert_eq!(comp, 6);
            let expected: String = (1..=6)
                .map(|i| format!("[{}]", base * 1000 + i))
                .collect();
            assert_eq!(txt, expected);
        }
        drop(tx);
        handle.join().unwrap();

        assert_eq!(metrics.slots_total.load(Ordering::Relaxed), 4);
        assert!(stats.max_active_slots_observed.load(Ordering::Relaxed) <= 4);
        assert_eq!(pending.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_prefix_cache_affinity_and_think_strip_checkpoint() {
        // Part A: Pure-KV prefix reuse across swapped slots
        {
            let stats = Arc::new(MockStats::default());
            let backend = MockBackend::new(true, Arc::clone(&stats));
            let metrics = Arc::new(ServerMetrics::default());
            let pending = Arc::new(AtomicUsize::new(0));
            let statuses = Arc::new(RwLock::new(Vec::new()));
            let cfg = ServerConfig {
                parallel_slots: 4,
                slot_quantum: 32,
                max_slot_cache_mb: 64,
            };
            let sched = SlotScheduler::new(
                backend,
                cfg,
                Arc::clone(&metrics),
                Arc::clone(&pending),
                Arc::clone(&statuses),
            );
            let (tx, rx) = mpsc::unbounded_channel();
            let handle = std::thread::spawn(move || sched.run(rx));

            // Req A: prompt [10,20,30] + 2 generated => commits [10,20,30,10003] (4 committed; last gen uncommitted)
            let (job_a, rx_a) = make_job("10,20,30", 2);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_a).unwrap();
            let (_, comp_a, reuse_a) = collect_text_and_done(rx_a);
            assert_eq!(comp_a, 2);
            assert_eq!(reuse_a, 0);

            // Req B forces Req A out of VRAM into host-RAM IdleSnapshot
            let (job_b, rx_b) = make_job("90,91,92", 1);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_b).unwrap();
            let _ = collect_text_and_done(rx_b);

            // Turn 2 of A matches [10,20,30,10003] + new turn token [55]
            let (job_a2, rx_a2) = make_job("10,20,30,10003,55", 1);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_a2).unwrap();
            let (_, _, reuse_a2) = collect_text_and_done(rx_a2);
            assert_eq!(reuse_a2, 4);
            assert!(stats.restore_calls.load(Ordering::Relaxed) >= 1);

            drop(tx);
            handle.join().unwrap();
        }

        // Part B: Recurrent/SSM think-strip secondary prefill checkpoint reuse
        {
            let stats = Arc::new(MockStats::default());
            let backend = MockBackend::new(false, Arc::clone(&stats));
            let metrics = Arc::new(ServerMetrics::default());
            let pending = Arc::new(AtomicUsize::new(0));
            let statuses = Arc::new(RwLock::new(Vec::new()));
            let cfg = ServerConfig {
                parallel_slots: 4,
                slot_quantum: 32,
                max_slot_cache_mb: 64,
            };
            let sched = SlotScheduler::new(
                backend,
                cfg,
                Arc::clone(&metrics),
                Arc::clone(&pending),
                Arc::clone(&statuses),
            );
            let (tx, rx) = mpsc::unbounded_channel();
            let handle = std::thread::spawn(move || sched.run(rx));

            // Turn 1: prompt [10,20,30,40] -> captures prefill_snapshot of len 4, generates tokens
            let (job_1, rx_1) = make_job("10,20,30,40", 3);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_1).unwrap();
            let _ = collect_text_and_done(rx_1);

            // Unrelated request forces eviction from VRAM
            let (job_other, rx_other) = make_job("77,78", 1);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_other).unwrap();
            let _ = collect_text_and_done(rx_other);

            // Turn 2 with stripped <think>: keeps prompt [10,20,30,40] then replaces assistant msg [999]
            // Primary snapshot has [10,20,30,40,10004,10005] which doesn't prefix-match because !allow_lcp_truncate;
            // prefill_cached HAS [10,20,30,40] and matches all 4 tokens!
            let (job_2, rx_2) = make_job("10,20,30,40,999", 1);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_2).unwrap();
            let (_, _, reuse_2) = collect_text_and_done(rx_2);
            assert_eq!(reuse_2, 4);

            drop(tx);
            handle.join().unwrap();
        }

        // Part C: Simultaneous arrival when resident slot matches Turn 2 while another slot runs first
        {
            let stats = Arc::new(MockStats::default());
            let backend = MockBackend::new(true, Arc::clone(&stats));
            let metrics = Arc::new(ServerMetrics::default());
            let pending = Arc::new(AtomicUsize::new(0));
            let statuses = Arc::new(RwLock::new(Vec::new()));
            let cfg = ServerConfig {
                parallel_slots: 4,
                slot_quantum: 32,
                max_slot_cache_mb: 64,
            };
            let sched = SlotScheduler::new(
                backend,
                cfg,
                Arc::clone(&metrics),
                Arc::clone(&pending),
                Arc::clone(&statuses),
            );
            let (tx, rx) = mpsc::unbounded_channel();
            let handle = std::thread::spawn(move || sched.run(rx));

            // Seed slot 0 -> ends in Idle(ResidentInVram) with tokens [10,20,30,10003]
            let (job_seed, rx_seed) = make_job("10,20,30", 2);
            pending.fetch_add(1, Ordering::Relaxed);
            tx.send(job_seed).unwrap();
            let _ = collect_text_and_done(rx_seed);

            // Simultaneously queue: Turn 2 matching slot 0 (InitFrom::Resident) AND an unrelated job on slot 1!
            // Scheduler RR cursor is at slot 1, so slot 1 runs first and steals VRAM before slot 0 starts.
            let (job_match, rx_match) = make_job("10,20,30,10003,88", 1);
            let (job_other, rx_other) = make_job("50,51", 1);
            pending.fetch_add(2, Ordering::Relaxed);
            tx.send(job_match).unwrap();
            tx.send(job_other).unwrap();

            let (_, _, reuse_match) = collect_text_and_done(rx_match);
            let _ = collect_text_and_done(rx_other);
            assert_eq!(reuse_match, 4);

            drop(tx);
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_client_disconnect_cancellation() {
        let stats = Arc::new(MockStats::default());
        let mut backend = MockBackend::new(true, Arc::clone(&stats));
        backend.per_token_delay_ms = 5;
        let metrics = Arc::new(ServerMetrics::default());
        let pending = Arc::new(AtomicUsize::new(0));
        let statuses = Arc::new(RwLock::new(Vec::new()));
        let cfg = ServerConfig {
            parallel_slots: 4,
            slot_quantum: 16,
            max_slot_cache_mb: 64,
        };
        let sched = SlotScheduler::new(
            backend,
            cfg,
            Arc::clone(&metrics),
            Arc::clone(&pending),
            Arc::clone(&statuses),
        );
        let (tx, rx) = mpsc::unbounded_channel();
        let handle = std::thread::spawn(move || sched.run(rx));

        let (job, mut rx_client) = make_job("5", 100);
        pending.fetch_add(1, Ordering::Relaxed);
        tx.send(job).unwrap();

        // Read 2 streamed tokens then drop client receiver to simulate disconnect
        assert!(matches!(rx_client.blocking_recv(), Some(StreamEvent::Token(_))));
        assert!(matches!(rx_client.blocking_recv(), Some(StreamEvent::Token(_))));
        drop(rx_client);

        drop(tx);
        handle.join().unwrap();

        assert_eq!(metrics.requests_cancelled_total.load(Ordering::Relaxed), 1);
        assert!(
            stats.decode_steps.load(Ordering::Relaxed) < 20,
            "generation continued too long after disconnect: {}",
            stats.decode_steps.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn test_single_request_zero_swap_fast_path() {
        let stats = Arc::new(MockStats::default());
        let backend = MockBackend::new(true, Arc::clone(&stats));
        let metrics = Arc::new(ServerMetrics::default());
        let pending = Arc::new(AtomicUsize::new(0));
        let statuses = Arc::new(RwLock::new(Vec::new()));
        let cfg = ServerConfig {
            parallel_slots: 4,
            slot_quantum: 3,
            max_slot_cache_mb: 64,
        };
        let sched = SlotScheduler::new(
            backend,
            cfg,
            Arc::clone(&metrics),
            Arc::clone(&pending),
            Arc::clone(&statuses),
        );
        let (tx, rx) = mpsc::unbounded_channel();
        let handle = std::thread::spawn(move || sched.run(rx));

        // 10 tokens > slot_quantum (3), yet alone => zero yields, zero snapshots, zero swaps
        let (job1, rx1) = make_job("7,8,9", 10);
        pending.fetch_add(1, Ordering::Relaxed);
        tx.send(job1).unwrap();
        let (_, comp1, reuse1) = collect_text_and_done(rx1);
        assert_eq!(comp1, 10);
        assert_eq!(reuse1, 0);
        assert_eq!(stats.capture_calls.load(Ordering::Relaxed), 0);
        assert_eq!(stats.restore_calls.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.slot_swaps_total.load(Ordering::Relaxed), 0);

        // Turn 2 continues directly from resident slot 0 with zero swaps/captures/restores
        let (job2, rx2) = make_job("7,8,9,7003,7004,99", 2);
        pending.fetch_add(1, Ordering::Relaxed);
        tx.send(job2).unwrap();
        let (_, comp2, reuse2) = collect_text_and_done(rx2);
        assert_eq!(comp2, 2);
        assert_eq!(reuse2, 5);
        assert_eq!(stats.capture_calls.load(Ordering::Relaxed), 0);
        assert_eq!(stats.restore_calls.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.slot_swaps_total.load(Ordering::Relaxed), 0);

        drop(tx);
        handle.join().unwrap();
    }
}
