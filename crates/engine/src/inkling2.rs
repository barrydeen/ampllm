use crate::config::{EngineConfig, MultiGpuMode};
use crate::helper::HelperGpuEngine;
use crate::kv_cache::KvQuantMode;
use crate::pipeline::PipelineGpuEngine;
use crate::sampling::{sample_next_token, SamplingParams};
use crate::single::SingleGpuEngine;
use crate::snapshot::{
    EngineSnapshotData, InklingSnapshot, QuantumGenerateResult, QuantumOptions, SequenceSnapshot,
};
use crate::tp2::Tp2GpuEngine;
use ampllm_cache::{CacheBudget, CacheStats};
use ampllm_gguf::{GgufFile, Inkling2Config};
use anyhow::Result;
use std::collections::VecDeque;
use std::time::Instant;

enum EngineBackend {
    Single(Box<SingleGpuEngine>),
    Helper(Box<HelperGpuEngine>),
    Pipeline(Box<PipelineGpuEngine>),
    Tp2(Box<Tp2GpuEngine>),
}

struct PrefixSnapshot {
    tokens: Vec<u32>,
    conv_rank0: Vec<Vec<u8>>,
    conv_rank1: Option<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone)]
pub struct StreamGenerateResult {
    pub prompt_tokens_evaluated: usize,
    pub prefix_reused_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub draft_n: usize,
    pub draft_accepted: usize,
}

pub struct Inkling2Engine {
    pub cfg: Inkling2Config,
    backend: EngineBackend,
    snapshots: VecDeque<PrefixSnapshot>,
    pub committed_tokens: Vec<u32>,
    pub prefix_reuse_tokens_total: u64,
}

impl Inkling2Engine {
    pub fn load_single_gpu_resident(
        gguf: &GgufFile,
        device: i32,
        max_ctx: usize,
        max_batch: usize,
        kv_mode: KvQuantMode,
    ) -> Result<Self> {
        let engine_cfg = EngineConfig {
            device,
            max_ctx,
            max_batch,
            kv_mode,
            expert_cache: CacheBudget::Auto,
            ..Default::default()
        };
        Self::load(gguf, engine_cfg)
    }

    pub fn load(gguf: &GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let backend = match engine_cfg.multi_gpu {
            MultiGpuMode::Single => EngineBackend::Single(Box::new(SingleGpuEngine::load(gguf, engine_cfg)?)),
            MultiGpuMode::Helper => EngineBackend::Helper(Box::new(HelperGpuEngine::load(gguf, engine_cfg)?)),
            MultiGpuMode::Pipeline => EngineBackend::Pipeline(Box::new(PipelineGpuEngine::load(gguf, engine_cfg)?)),
            MultiGpuMode::Tp2 => EngineBackend::Tp2(Box::new(Tp2GpuEngine::load(gguf, engine_cfg)?)),
        };
        Ok(Self {
            cfg,
            backend,
            snapshots: VecDeque::with_capacity(4),
            committed_tokens: Vec::new(),
            prefix_reuse_tokens_total: 0,
        })
    }

    pub fn cur_pos(&self) -> usize {
        match &self.backend {
            EngineBackend::Single(e) => e.cur_pos,
            EngineBackend::Helper(e) => e.cur_pos,
            EngineBackend::Pipeline(e) => e.cur_pos,
            EngineBackend::Tp2(e) => e.cur_pos,
        }
    }

    fn set_cur_pos(&mut self, pos: usize) {
        match &mut self.backend {
            EngineBackend::Single(e) => e.cur_pos = pos,
            EngineBackend::Helper(e) => e.cur_pos = pos,
            EngineBackend::Pipeline(e) => e.cur_pos = pos,
            EngineBackend::Tp2(e) => e.cur_pos = pos,
        }
    }

    fn capture_snapshot(&self, tokens: &[u32]) -> Result<PrefixSnapshot> {
        let (conv_rank0, conv_rank1) = match &self.backend {
            EngineBackend::Single(e) => (e.rank.kv_cache.capture_conv_snapshot(e.rank.device)?, None),
            EngineBackend::Helper(e) => (e.rank0.kv_cache.capture_conv_snapshot(0)?, None),
            EngineBackend::Pipeline(e) => (
                e.rank0.kv_cache.capture_conv_snapshot(0)?,
                Some(e.rank1.kv_cache.capture_conv_snapshot(1)?),
            ),
            EngineBackend::Tp2(e) => (
                e.rank0.kv_cache.capture_conv_snapshot(0)?,
                Some(e.rank1.kv_cache.capture_conv_snapshot(1)?),
            ),
        };
        Ok(PrefixSnapshot {
            tokens: tokens.to_vec(),
            conv_rank0,
            conv_rank1,
        })
    }

    fn restore_snapshot(&mut self, snap: &PrefixSnapshot) -> Result<()> {
        match &self.backend {
            EngineBackend::Single(e) => e.rank.kv_cache.restore_conv_snapshot(e.rank.device, &snap.conv_rank0)?,
            EngineBackend::Helper(e) => e.rank0.kv_cache.restore_conv_snapshot(0, &snap.conv_rank0)?,
            EngineBackend::Pipeline(e) => {
                e.rank0.kv_cache.restore_conv_snapshot(0, &snap.conv_rank0)?;
                if let Some(ref c1) = snap.conv_rank1 {
                    e.rank1.kv_cache.restore_conv_snapshot(1, c1)?;
                }
            }
            EngineBackend::Tp2(e) => {
                e.rank0.kv_cache.restore_conv_snapshot(0, &snap.conv_rank0)?;
                if let Some(ref c1) = snap.conv_rank1 {
                    e.rank1.kv_cache.restore_conv_snapshot(1, c1)?;
                }
            }
        }
        self.set_cur_pos(snap.tokens.len());
        Ok(())
    }

    fn push_snapshot(&mut self, snap: PrefixSnapshot) {
        if self.snapshots.len() >= 4 {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snap);
    }

    pub fn cache_summary(&self) -> String {
        match &self.backend {
            EngineBackend::Single(e) => e.cache_summary(),
            EngineBackend::Helper(e) => e.cache_summary(),
            EngineBackend::Pipeline(e) => e.cache_summary(),
            EngineBackend::Tp2(e) => e.cache_summary(),
        }
    }

    pub fn cache_stats(&self) -> CacheStats {
        match &self.backend {
            EngineBackend::Single(e) => e.cache_stats().clone(),
            EngineBackend::Helper(e) => e.cache_stats().clone(),
            EngineBackend::Pipeline(_) | EngineBackend::Tp2(_) => CacheStats::default(),
        }
    }

    pub fn save_profile(&self) -> Result<()> {
        match &self.backend {
            EngineBackend::Single(e) => e.save_profile(),
            EngineBackend::Helper(e) => e.save_profile(),
            EngineBackend::Pipeline(_) | EngineBackend::Tp2(_) => Ok(()),
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.committed_tokens.clear();
        self.snapshots.clear();
        match &mut self.backend {
            EngineBackend::Single(e) => e.reset(),
            EngineBackend::Helper(e) => e.reset(),
            EngineBackend::Pipeline(e) => e.reset(),
            EngineBackend::Tp2(e) => e.reset(),
        }
    }

    pub fn forward_tokens(
        &mut self,
        tokens: &[u32],
        capture_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>)> {
        let res = match &mut self.backend {
            EngineBackend::Single(e) => e.forward_tokens(tokens, capture_logits),
            EngineBackend::Helper(e) => e.forward_tokens(tokens, capture_logits),
            EngineBackend::Pipeline(e) => e.forward_tokens(tokens, capture_logits),
            EngineBackend::Tp2(e) => e.forward_tokens(tokens, capture_logits),
        }?;
        self.committed_tokens.extend_from_slice(tokens);
        Ok(res)
    }

    pub fn capture_sequence_snapshot(&self, has_images: bool) -> Result<SequenceSnapshot> {
        let pos = self.cur_pos();
        let (rank0, rank1) = match &self.backend {
            EngineBackend::Single(e) => (e.rank.kv_cache.capture_full_snapshot(e.rank.device, pos)?, None),
            EngineBackend::Helper(e) => (e.rank0.kv_cache.capture_full_snapshot(0, pos)?, None),
            EngineBackend::Pipeline(e) => (
                e.rank0.kv_cache.capture_full_snapshot(0, pos)?,
                Some(e.rank1.kv_cache.capture_full_snapshot(1, pos)?),
            ),
            EngineBackend::Tp2(e) => (
                e.rank0.kv_cache.capture_full_snapshot(0, pos)?,
                Some(e.rank1.kv_cache.capture_full_snapshot(1, pos)?),
            ),
        };
        let byte_len: usize = rank0.iter().flatten().map(|s| s.byte_len()).sum::<usize>()
            + rank1
                .as_ref()
                .map_or(0, |v| v.iter().flatten().map(|s| s.byte_len()).sum());
        Ok(SequenceSnapshot {
            tokens: self.committed_tokens.clone(),
            has_images,
            allow_lcp_truncate: false,
            max_safe_rollback_delta: None,
            rollback_align: 1,
            byte_len,
            inner: EngineSnapshotData::Inkling2(InklingSnapshot { rank0, rank1 }),
        })
    }

    pub fn restore_sequence_snapshot(
        &mut self,
        snap: &SequenceSnapshot,
        _truncate_to: Option<usize>,
    ) -> Result<()> {
        let EngineSnapshotData::Inkling2(ref is) = snap.inner else {
            anyhow::bail!("expected Inkling2 snapshot");
        };
        match &mut self.backend {
            EngineBackend::Single(e) => e.rank.kv_cache.restore_full_snapshot(e.rank.device, &is.rank0)?,
            EngineBackend::Helper(e) => e.rank0.kv_cache.restore_full_snapshot(0, &is.rank0)?,
            EngineBackend::Pipeline(e) => {
                e.rank0.kv_cache.restore_full_snapshot(0, &is.rank0)?;
                if let Some(ref r1) = is.rank1 {
                    e.rank1.kv_cache.restore_full_snapshot(1, r1)?;
                }
            }
            EngineBackend::Tp2(e) => {
                e.rank0.kv_cache.restore_full_snapshot(0, &is.rank0)?;
                if let Some(ref r1) = is.rank1 {
                    e.rank1.kv_cache.restore_full_snapshot(1, r1)?;
                }
            }
        }
        self.set_cur_pos(snap.tokens.len());
        self.committed_tokens = snap.tokens.clone();
        self.snapshots.clear();
        if let Ok(conv_snap) = self.capture_snapshot(&snap.tokens) {
            self.snapshots.push_back(conv_snap);
        }
        Ok(())
    }

    /// Generate tokens with exact prefix KV+conv reuse and streaming callback.
    /// Callback returns `true` to continue generating, or `false` to stop early (e.g. text stop string).
    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        let mut no_yield = |_: usize| false;
        let qres = self.generate_stream_quantum(
            prompt_tokens,
            params,
            QuantumOptions::no_yield(&mut no_yield),
            on_token,
        )?;
        Ok(qres.result)
    }

    pub fn generate_stream_quantum<F>(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        mut on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        let swa = self.cfg.sliding_window;
        let cur = self.cur_pos();

        let exact_resident_reuse = !self.committed_tokens.is_empty()
            && self.committed_tokens.len() == cur
            && prompt_tokens.len() > cur
            && prompt_tokens.starts_with(&self.committed_tokens);

        let reused = if exact_resident_reuse {
            if !opts.resume_active {
                self.prefix_reuse_tokens_total += cur as u64;
            }
            cur
        } else {
            // Find longest snapshotted prefix whose SWA KV slots are still intact.
            let matched_idx = self
                .snapshots
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    let slen = s.tokens.len();
                    slen > 0
                        && prompt_tokens.len() > slen
                        && prompt_tokens.starts_with(&s.tokens)
                        && (cur == slen || (cur <= swa && slen <= cur))
                })
                .max_by_key(|(_, s)| s.tokens.len())
                .map(|(idx, _)| idx);

            if let Some(idx) = matched_idx {
                let snap_tokens = self.snapshots[idx].tokens.clone();
                let conv0 = self.snapshots[idx].conv_rank0.clone();
                let conv1 = self.snapshots[idx].conv_rank1.clone();
                self.restore_snapshot(&PrefixSnapshot {
                    tokens: snap_tokens.clone(),
                    conv_rank0: conv0,
                    conv_rank1: conv1,
                })?;
                self.committed_tokens = snap_tokens;
                let rlen = self.cur_pos();
                if !opts.resume_active {
                    self.prefix_reuse_tokens_total += rlen as u64;
                }
                rlen
            } else {
                self.reset()?;
                0
            }
        };

        let suffix = &prompt_tokens[reused..];
        let need_logits = params.temperature > 0.0;
        let mut rng = params.seed.max(1);

        let t0 = Instant::now();
        let (greedy_first, logits_opt) = self.forward_tokens(suffix, need_logits)?;
        let prefill_s = t0.elapsed().as_secs_f64();

        // Snapshot immediately after prompt prefill so future parallel branches can reuse system/user prefix
        if let Ok(snap) = self.capture_snapshot(prompt_tokens) {
            self.push_snapshot(snap);
        }
        let prefill_snapshot = if opts.capture_prefill_snap {
            Some(self.capture_sequence_snapshot(false)?)
        } else {
            None
        };

        let mut next_tok = if let Some(ref logits) = logits_opt {
            sample_next_token(logits, params, &mut rng)
        } else {
            greedy_first
        };

        let mut generated = Vec::new();
        let mut full_seq = prompt_tokens.to_vec();
        let mut yielded = false;
        let t_dec = Instant::now();

        for step in 0..params.max_tokens {
            if params.stop_token_ids.contains(&next_tok) {
                break;
            }
            generated.push(next_tok);
            full_seq.push(next_tok);
            let keep_going = on_token(next_tok);
            if !keep_going || step + 1 >= params.max_tokens {
                break;
            }
            if (opts.should_yield)(generated.len()) {
                yielded = true;
                break;
            }
            let (g_nxt, l_opt) = self.forward_tokens(&[next_tok], need_logits)?;
            next_tok = if let Some(ref logits) = l_opt {
                sample_next_token(logits, params, &mut rng)
            } else {
                g_nxt
            };
        }
        let decode_s = t_dec.elapsed().as_secs_f64();

        if let Ok(snap) = self.capture_snapshot(&full_seq) {
            self.push_snapshot(snap);
        }

        let prompt_tokens_evaluated = if opts.resume_active { 0 } else { suffix.len() };
        let prefill_tps = if opts.resume_active {
            0.0
        } else {
            suffix.len() as f64 / prefill_s.max(1e-6)
        };
        let decode_tps = (generated.len().saturating_sub(if opts.resume_active { 0 } else { 1 }).max(1)) as f64
            / decode_s.max(1e-6);

        Ok(QuantumGenerateResult {
            result: StreamGenerateResult {
                prompt_tokens_evaluated,
                prefix_reused_tokens: reused,
                generated_tokens: generated,
                prefill_tps,
                decode_tps,
                draft_n: 0,
                draft_accepted: 0,
            },
            yielded,
            prefill_snapshot,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        n_predict: usize,
    ) -> Result<(Vec<u32>, f64, f64, Vec<(u32, f32)>)> {
        self.reset()?;
        self.snapshots.clear();
        let t0 = Instant::now();
        let (mut next_tok, step0_logits) = self.forward_tokens(prompt_tokens, true)?;
        let prefill_s = t0.elapsed().as_secs_f64();

        let mut top5 = Vec::new();
        if let Some(logits) = step0_logits {
            let mut pairs: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            pairs.truncate(5);
            top5 = pairs;
        }

        let mut generated = Vec::with_capacity(n_predict);
        let t_dec = Instant::now();
        for step in 0..n_predict {
            generated.push(next_tok);
            if step + 1 < n_predict {
                let (nxt, _) = self.forward_tokens(&[next_tok], false)?;
                next_tok = nxt;
            }
        }
        let decode_s = t_dec.elapsed().as_secs_f64();
        let prefill_tps = prompt_tokens.len() as f64 / prefill_s.max(1e-6);
        let decode_tps = (n_predict.saturating_sub(1).max(1)) as f64 / decode_s.max(1e-6);

        Ok((generated, prefill_tps, decode_tps, top5))
    }
}
