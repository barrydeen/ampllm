use crate::config::{EngineConfig, MultiGpuMode};
use crate::helper::HelperGpuEngine;
use crate::inkling2_draft::{suffix_propose, SpecController, DSPARK_BLOCK};
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
    pub max_ctx: usize,
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
        let max_ctx = engine_cfg.max_ctx;
        let backend = match engine_cfg.multi_gpu {
            MultiGpuMode::Single => EngineBackend::Single(Box::new(SingleGpuEngine::load(gguf, engine_cfg)?)),
            MultiGpuMode::Helper => EngineBackend::Helper(Box::new(HelperGpuEngine::load(gguf, engine_cfg)?)),
            MultiGpuMode::Pipeline => EngineBackend::Pipeline(Box::new(PipelineGpuEngine::load(gguf, engine_cfg)?)),
            MultiGpuMode::Tp2 => EngineBackend::Tp2(Box::new(Tp2GpuEngine::load(gguf, engine_cfg)?)),
        };
        Ok(Self {
            cfg,
            max_ctx,
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
        let mut draft_n = 0usize;
        let mut draft_accepted = 0usize;
        let t_dec = Instant::now();

        let use_spec = params.temperature == 0.0
            && match &self.backend {
                EngineBackend::Single(e) => e.can_spec_decode(),
                _ => false,
            };

        if use_spec {
            let mut ctl = SpecController::new();
            let EngineBackend::Single(ref mut s_eng) = self.backend else {
                unreachable!();
            };
            let has_drafter = s_eng.has_neural_drafter();
            let max_ctx = self.max_ctx;
            let spec_prof = std::env::var_os("AMPLLM_SPEC_PROF").is_some();
            let mut prof_win_steps = 0usize;
            let mut prof_plain_steps = 0usize;
            let mut prof_draft_ms = 0.0f64;
            let mut prof_verify_ms = 0.0f64;
            let mut prof_commit_ms = 0.0f64;
            let mut prof_plain_ms = 0.0f64;
            let mut prof_sum_w = 0usize;

            while generated.len() < params.max_tokens {
                if params.stop_token_ids.contains(&next_tok) {
                    break;
                }
                generated.push(next_tok);
                full_seq.push(next_tok);
                let keep_going = on_token(next_tok);
                if !keep_going || generated.len() >= params.max_tokens {
                    break;
                }
                if (opts.should_yield)(generated.len()) {
                    yielded = true;
                    break;
                }
                let room_ctx = max_ctx.saturating_sub(s_eng.cur_pos + 1);
                let room_toks = params.max_tokens - generated.len();
                let max_w = DSPARK_BLOCK.min(room_ctx + 1).min(room_toks);

                if max_w < 2 {
                    let t_step = Instant::now();
                    let argmax = s_eng.forward_window_spec(&[next_tok])?;
                    s_eng.commit_window_spec(1, 1)?;
                    self.committed_tokens.push(next_tok);
                    let ms = t_step.elapsed().as_secs_f64() * 1e3;
                    ctl.record_plain(ms);
                    prof_plain_steps += 1;
                    prof_plain_ms += ms;
                    next_tok = argmax[0];
                    continue;
                }

                let suffix_opt = suffix_propose(&full_seq, max_w - 1);
                let suf_len = suffix_opt.as_ref().map(|v| v.len()).unwrap_or(0);
                let (w, is_suffix) = ctl.pick(max_w, suf_len, has_drafter);

                if w < 2 {
                    let t_step = Instant::now();
                    let argmax = s_eng.forward_window_spec(&[next_tok])?;
                    s_eng.commit_window_spec(1, 1)?;
                    self.committed_tokens.push(next_tok);
                    let ms = t_step.elapsed().as_secs_f64() * 1e3;
                    ctl.record_plain(ms);
                    prof_plain_steps += 1;
                    prof_plain_ms += ms;
                    next_tok = argmax[0];
                    continue;
                }

                let t_step = Instant::now();
                let proposals = w - 1;
                let mut window = Vec::with_capacity(w);
                window.push(next_tok);
                if is_suffix {
                    let suf = suffix_opt.as_ref().unwrap();
                    let take = proposals.min(suf.len());
                    window.extend_from_slice(&suf[..take]);
                } else {
                    let dr = s_eng.propose_draft(next_tok, proposals)?;
                    window.extend_from_slice(&dr[..proposals]);
                }
                let t_after_draft = Instant::now();
                let actual_w = window.len();
                let actual_p = actual_w - 1;
                let target_out = s_eng.forward_window_spec(&window)?;
                let t_after_verify = Instant::now();

                let mut acc = 0usize;
                while acc < actual_p && window[1 + acc] == target_out[acc] {
                    acc += 1;
                }
                draft_n += actual_p;

                let mut stopped = false;
                let mut kept_drafts = 0usize;
                for j in 0..acc {
                    let dt = window[1 + j];
                    if params.stop_token_ids.contains(&dt) {
                        stopped = true;
                        break;
                    }
                    generated.push(dt);
                    full_seq.push(dt);
                    kept_drafts += 1;
                    let keep = on_token(dt);
                    if !keep || generated.len() >= params.max_tokens {
                        stopped = true;
                        break;
                    }
                    if (opts.should_yield)(generated.len()) {
                        yielded = true;
                        stopped = true;
                        break;
                    }
                }
                draft_accepted += kept_drafts;
                let keep_in_kv = 1 + kept_drafts;
                s_eng.commit_window_spec(keep_in_kv, actual_w)?;
                let t_after_commit = Instant::now();
                self.committed_tokens.extend_from_slice(&window[..keep_in_kv]);
                let step_ms = t_after_commit.duration_since(t_step).as_secs_f64() * 1e3;
                ctl.record_window(actual_w, acc, step_ms, is_suffix);
                if spec_prof {
                    prof_win_steps += 1;
                    prof_sum_w += actual_w;
                    prof_draft_ms += t_after_draft.duration_since(t_step).as_secs_f64() * 1e3;
                    prof_verify_ms += t_after_verify.duration_since(t_after_draft).as_secs_f64() * 1e3;
                    prof_commit_ms += t_after_commit.duration_since(t_after_verify).as_secs_f64() * 1e3;
                }

                if stopped {
                    break;
                }
                next_tok = target_out[acc];
            }

            if spec_prof {
                let ws = prof_win_steps.max(1) as f64;
                let ps = prof_plain_steps.max(1) as f64;
                eprintln!(
                    "spec_prof: win_steps={} plain_steps={} avg_draft_ms={:.3} avg_verify_ms={:.3} avg_commit_ms={:.3} avg_plain_ms={:.3} avg_w={:.2} avg_acc={:.2}",
                    prof_win_steps,
                    prof_plain_steps,
                    prof_draft_ms / ws,
                    prof_verify_ms / ws,
                    prof_commit_ms / ws,
                    prof_plain_ms / ps,
                    prof_sum_w as f64 / ws,
                    draft_accepted as f64 / ws,
                );
            }
        } else {
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
                draft_n,
                draft_accepted,
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
