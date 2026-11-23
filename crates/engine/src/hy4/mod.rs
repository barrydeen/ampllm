pub mod engram;
pub mod forward;
pub mod mtp;
pub mod state;
pub mod vision;
pub mod weights;

pub use forward::Hy4ImageSpan;

use crate::config::EngineConfig;
use crate::hy4::forward::{forward_trunk_window, hy4_dump_enabled};
use crate::hy4::mtp::{mtp_catch_up, mtp_draft_tokens};
use crate::hy4::state::{hy4_profile_enabled, Hy4State, HY4_W_TOKEN};
use crate::hy4::weights::Hy4ModelWeights;
use crate::inkling2::StreamGenerateResult;
use crate::sampling::{sample_next_token, SamplingParams};
use crate::snapshot::{
    d2h_full, d2h_prefix, h2d_bytes, EngineSnapshotData, Hy4LayerSnap, Hy4Snapshot,
    QuantumGenerateResult, QuantumOptions, SequenceSnapshot,
};
use ampllm_cache::CacheStats;
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_gguf::GgufFile;
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

/// Tencent Hy4-Flash (`hy_v4_flash`) inference engine.
/// Field order enforces safe Drop sequence:
/// 1. `state` (DeviceBuffers, streams, events, pinned host staging),
/// 2. `weights` (weight DeviceBuffers, RegisteredMappedHostBuffers, mmproj GGUF),
/// 3. `gguf` (main/MTP GGUF shard mmaps dropped last).
pub struct Hy4Engine {
    state: Hy4State,
    weights: Hy4ModelWeights,
    _gguf: GgufFile,
    pub max_ctx: usize,
    pub use_spec: bool,
    pub max_draft: usize,
    pub draft_floor: f32,
    pub committed_tokens: Vec<u32>,
    pub prefix_reuse_tokens_total: u64,
}

impl Hy4Engine {
    pub fn load(gguf: GgufFile, cfg: EngineConfig) -> Result<Self> {
        let pool = Arc::new(CpuThreadPool::new(cfg.cpu_threads));
        let (weights, gguf_out) = Hy4ModelWeights::load(gguf, &cfg, &pool, cfg.max_ctx, None)?;
        let state = Hy4State::alloc(&weights, pool)?;
        let max_ctx = weights.max_ctx;
        let use_spec = cfg.use_spec && weights.mtp_layer.is_some();
        let max_draft = std::env::var("AMPLLM_HY4_DRAFT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3)
            .clamp(0, HY4_W_TOKEN - 1);
        let draft_floor = std::env::var("AMPLLM_HY4_DRAFT_FLOOR")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.55);

        Ok(Self {
            state,
            weights,
            _gguf: gguf_out,
            max_ctx,
            use_spec,
            max_draft,
            draft_floor,
            committed_tokens: Vec::new(),
            prefix_reuse_tokens_total: 0,
        })
    }

    pub fn has_vision(&self) -> bool {
        self.weights.vision.is_some()
    }

    pub fn vision_config(&self) -> Option<&ampllm_gguf::Hy4VisionConfig> {
        self.weights.vision.as_ref().map(|v| &v.cfg)
    }

    /// Encode an RGB image (`[height, width, 3]`) into `[n_tok * n_embd]` F32 embeddings,
    /// borrowing GPU 0's idle prefill scratch arena so Vision uses zero permanent VRAM.
    pub fn encode_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        let Some(ref vis) = self.weights.vision else {
            anyhow::bail!("Hy4 model was loaded without mmproj vision weights");
        };
        let sc0 = &self.state.sc[0];
        let rows = vis.encode_image(rgb, width, height, Some(&sc0.arena), sc0.device, &sc0.stream)?;
        // Re-zero the scratch arena so transient vision activations cannot leak into persistent phases
        sc0.arena.zero()?;
        Ok(rows)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.committed_tokens.clear();
        self.state.reset()
    }

    pub fn rollback_policy(&self) -> (bool, Option<usize>, usize) {
        let min_ring = self
            .weights
            .swa_ring_cap
            .min(self.weights.ivg_ring_cap)
            .min(self.weights.hring_cap);
        let max_safe_rollback_delta = if self.state.end_pos <= min_ring {
            None
        } else {
            Some(0)
        };
        (
            true,
            max_safe_rollback_delta,
            self.weights.cfg.ix_kpool.max(1),
        )
    }

    pub fn capture_snapshot(&mut self, has_images: bool) -> Result<SequenceSnapshot> {
        for sc in &self.state.sc {
            ampllm_cuda_kernels::set_device(sc.device)?;
            sc.stream.sync()?;
            sc.copy_stream.sync()?;
            sc.d2h_stream.sync()?;
        }
        let end_pos = self.state.end_pos;
        let mtp_done = self.state.mtp_done;
        let kv_rb = self.weights.kv_rb;
        let swa_cap = self.weights.swa_ring_cap;
        let ix_dim = self.weights.cfg.ix_dim;
        let ix_kpool = self.weights.cfg.ix_kpool;
        let pe = if self.weights.kv_is_q8 { 2 } else { 4 };
        let mut byte_len = 0usize;

        let capture_layer = |lw: &weights::Hy4LayerWeights,
                             c: &state::Hy4LayerCache,
                             pos: usize|
         -> Result<Hy4LayerSnap> {
            let kv_rows = if lw.swa && pos >= swa_cap {
                swa_cap
            } else {
                pos.min(if lw.swa { swa_cap } else { self.weights.max_ctx })
            };
            let kv = d2h_prefix(&c.kv, kv_rows * kv_rb)?;
            let ivg = match c.ivg {
                Some(ref b) => Some(d2h_full(b)?),
                None => None,
            };
            let pooled = match c.pooled {
                Some(ref b) => {
                    let blocks = (pos / ix_kpool) + 1;
                    Some(d2h_prefix(b, blocks * ix_dim * pe)?)
                }
                None => None,
            };
            Ok(Hy4LayerSnap { kv, ivg, pooled })
        };

        let mut caches = Vec::with_capacity(self.state.caches.len());
        for (lw, c) in self.weights.layers.iter().zip(self.state.caches.iter()) {
            let ls = capture_layer(lw, c, end_pos)?;
            byte_len += ls.kv.len()
                + ls.ivg.as_ref().map_or(0, Vec::len)
                + ls.pooled.as_ref().map_or(0, Vec::len);
            caches.push(ls);
        }
        let mtp_cache = match (&self.weights.mtp_layer, &self.state.mtp_cache) {
            (Some(ref ml), Some(ref mc)) => {
                let ls = capture_layer(ml, mc, mtp_done)?;
                byte_len += ls.kv.len()
                    + ls.ivg.as_ref().map_or(0, Vec::len)
                    + ls.pooled.as_ref().map_or(0, Vec::len);
                Some(ls)
            }
            _ => None,
        };
        let hring = d2h_full(&self.state.hring)?;
        byte_len += hring.len();

        let min_ring = swa_cap
            .min(self.weights.ivg_ring_cap)
            .min(self.weights.hring_cap);
        let max_safe_rollback_delta = if end_pos <= min_ring { None } else { Some(0) };

        Ok(SequenceSnapshot {
            tokens: self.committed_tokens.clone(),
            has_images,
            allow_lcp_truncate: true,
            max_safe_rollback_delta,
            rollback_align: ix_kpool.max(1),
            byte_len,
            inner: EngineSnapshotData::Hy4(Hy4Snapshot {
                end_pos,
                mtp_done,
                caches,
                mtp_cache,
                hring,
            }),
        })
    }

    pub fn restore_snapshot(
        &mut self,
        snap: &SequenceSnapshot,
        truncate_to: Option<usize>,
    ) -> Result<()> {
        let EngineSnapshotData::Hy4(ref hs) = snap.inner else {
            anyhow::bail!("expected Hy4 snapshot");
        };
        for sc in &self.state.sc {
            ampllm_cuda_kernels::set_device(sc.device)?;
            sc.stream.sync()?;
            sc.copy_stream.sync()?;
            sc.d2h_stream.sync()?;
        }
        if hs.caches.len() != self.state.caches.len() {
            anyhow::bail!("Hy4 snapshot layer count mismatch");
        }
        let new_pos = truncate_to.unwrap_or(hs.end_pos).min(hs.end_pos);
        let new_mtp = hs.mtp_done.min(new_pos);
        let kv_rb = self.weights.kv_rb;
        let swa_cap = self.weights.swa_ring_cap;
        let ix_dim = self.weights.cfg.ix_dim;
        let ix_kpool = self.weights.cfg.ix_kpool;
        let pe = if self.weights.kv_is_q8 { 2 } else { 4 };

        let restore_layer = |lw: &weights::Hy4LayerWeights,
                             c: &state::Hy4LayerCache,
                             ls: &Hy4LayerSnap,
                             pos: usize|
         -> Result<()> {
            let kv_bytes = if lw.swa && pos >= swa_cap {
                ls.kv.len()
            } else {
                (pos * kv_rb).min(ls.kv.len())
            };
            h2d_bytes(&c.kv, &ls.kv[..kv_bytes])?;
            if let (Some(ref b), Some(ref src)) = (&c.ivg, &ls.ivg) {
                h2d_bytes(b, src)?;
            }
            if let (Some(ref b), Some(ref src)) = (&c.pooled, &ls.pooled) {
                let p_bytes = (((pos / ix_kpool) + 1) * ix_dim * pe).min(src.len());
                h2d_bytes(b, &src[..p_bytes])?;
            }
            Ok(())
        };

        for ((lw, c), ls) in self
            .weights
            .layers
            .iter()
            .zip(self.state.caches.iter())
            .zip(hs.caches.iter())
        {
            restore_layer(lw, c, ls, new_pos)?;
        }
        if let (Some(ref ml), Some(ref mc), Some(ref ls)) = (
            &self.weights.mtp_layer,
            &self.state.mtp_cache,
            &hs.mtp_cache,
        ) {
            restore_layer(ml, mc, ls, new_mtp)?;
        }
        h2d_bytes(&self.state.hring, &hs.hring)?;

        self.state.end_pos = new_pos;
        self.state.mtp_done = new_mtp;
        self.state.history.clear();
        self.state.history.extend_from_slice(&snap.tokens[..new_pos]);
        self.committed_tokens.clear();
        self.committed_tokens.extend_from_slice(&snap.tokens[..new_pos]);
        Ok(())
    }

    pub fn reset_profile(&mut self) {
        self.state.profile = Default::default();
    }

    pub fn print_profile(&self, label: &str) {
        self.state.profile.print(label);
    }

    pub fn cold_pinned_layers(&self) -> (usize, usize) {
        let mut pinned = 0usize;
        let mut total = 0usize;
        for lw in &self.weights.layers {
            if let Some(ref m) = lw.moe {
                total += 1;
                if m.cold_is_pinned {
                    pinned += 1;
                }
            }
        }
        (pinned, total)
    }

    pub fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }

    pub fn save_profile(&self) -> Result<()> {
        Ok(())
    }

    pub fn cache_summary(&self) -> String {
        let pct = 100.0 * self.weights.vram_resident_experts as f64
            / self.weights.total_trunk_routed_experts.max(1) as f64;
        format!(
            "Hy4-Flash split={} | VRAM experts {}/{} ({:.1}%) | KV={} | GPUs={} ({:?} MiB) | mtp={} | vision={}",
            self.weights.split_layer,
            self.weights.vram_resident_experts,
            self.weights.total_trunk_routed_experts,
            pct,
            if self.weights.kv_is_q8 { "Q8_0" } else { "F16" },
            self.weights.gpus.len(),
            self.weights.vram_used_mb,
            self.weights.mtp_layer.is_some(),
            self.weights.vision.is_some(),
        )
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        self.generate_stream_with_images(prompt_tokens, &[], params, on_token)
    }

    pub fn generate_stream_with_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Hy4ImageSpan],
        params: &SamplingParams,
        on_token: F,
    ) -> Result<StreamGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        let mut no_yield = |_: usize| false;
        let qres = self.generate_stream_quantum_with_images(
            prompt_tokens,
            images,
            params,
            QuantumOptions::no_yield(&mut no_yield),
            on_token,
        )?;
        Ok(qres.result)
    }

    pub fn generate_stream_quantum_with_images<F>(
        &mut self,
        prompt_tokens: &[u32],
        images: &[Hy4ImageSpan],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        mut on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must be non-empty");
        let n_prompt = prompt_tokens.len();
        anyhow::ensure!(
            n_prompt < self.max_ctx,
            "prompt length {n_prompt} >= max_ctx {}",
            self.max_ctx
        );

        // Reset KV/history for a clean prompt or reuse exact common prefix if text-only matches!
        let mut prefix_reused = 0usize;
        if images.is_empty() && !self.committed_tokens.is_empty() {
            let max_l = n_prompt.saturating_sub(1).min(self.committed_tokens.len());
            let mut lcp = 0usize;
            while lcp < max_l && self.committed_tokens[lcp] == prompt_tokens[lcp] {
                lcp += 1;
            }
            let min_ring = self
                .weights
                .swa_ring_cap
                .min(self.weights.ivg_ring_cap)
                .min(self.weights.hring_cap);
            let safe_lcp = if lcp == self.state.end_pos {
                lcp
            } else if opts.resume_active || self.state.end_pos > min_ring {
                0
            } else {
                let kpool = self.weights.cfg.ix_kpool.max(1);
                (lcp / kpool) * kpool
            };
            if safe_lcp > 0 && safe_lcp <= self.state.end_pos {
                prefix_reused = safe_lcp;
                self.state.end_pos = safe_lcp;
                self.state.mtp_done = self.state.mtp_done.min(safe_lcp);
                self.state.history.truncate(safe_lcp);
                self.committed_tokens.truncate(safe_lcp);
                if !opts.resume_active {
                    self.prefix_reuse_tokens_total += prefix_reused as u64;
                }
            } else {
                self.reset()?;
            }
        } else {
            self.reset()?;
        }

        let tc = self.weights.tc;
        let eval_slice = &prompt_tokens[prefix_reused..];
        let eval_len = eval_slice.len();
        let n_vocab = self.weights.cfg.n_vocab;
        let resuming_decode = opts.resume_active
            && prefix_reused > 0
            && prefix_reused == self.state.end_pos
            && eval_len == 1
            && images.is_empty();

        let mut rng = params.seed;
        let mut generated = Vec::with_capacity(params.max_tokens);
        let stop_check = |tok: u32| -> bool {
            tok == self.weights.cfg.eos_token_id.unwrap_or(u32::MAX)
                || params.stop_token_ids.contains(&tok)
        };

        let (mut cur_tok, prompt_tokens_evaluated, prefill_tps, t_decode0, mut stopped) =
            if resuming_decode {
                (eval_slice[0], 0usize, 0.0f64, Instant::now(), false)
            } else {
                let t_prefill0 = Instant::now();
                let mut p0 = prefix_reused;
                let mut idx = 0usize;
                while idx < eval_len {
                    let chunk = (eval_len - idx).min(tc);
                    let is_last = idx + chunk == eval_len;
                    let want = usize::from(is_last);
                    unsafe {
                        forward_trunk_window(
                            &self.weights,
                            &mut self.state,
                            &eval_slice[idx..idx + chunk],
                            p0,
                            want,
                            images,
                        )?;
                    }
                    if !is_last && self.use_spec {
                        let t_md0 = hy4_profile_enabled().then(Instant::now);
                        unsafe {
                            mtp_catch_up(&self.weights, &mut self.state, p0 + chunk, images)?;
                        }
                        if let Some(t0) = t_md0 {
                            self.state.profile.mtp_draft_ms += t0.elapsed().as_secs_f64() * 1e3;
                        }
                    }
                    p0 += chunk;
                    idx += chunk;
                }
                self.state.end_pos = n_prompt;
                self.committed_tokens.clear();
                self.committed_tokens.extend_from_slice(prompt_tokens);
                let prefill_s = t_prefill0.elapsed().as_secs_f64();
                let prefill_tps = eval_len as f64 / prefill_s.max(1e-9);

                let logits_slice = unsafe {
                    std::slice::from_raw_parts(self.state.h_logits.as_mut_ptr().cast::<f32>(), n_vocab)
                };
                if params.max_tokens == 0 {
                    return Ok(QuantumGenerateResult {
                        result: StreamGenerateResult {
                            prompt_tokens_evaluated: eval_len,
                            prefix_reused_tokens: prefix_reused,
                            generated_tokens: generated,
                            prefill_tps,
                            decode_tps: 0.0,
                            draft_n: 0,
                            draft_accepted: 0,
                        },
                        yielded: false,
                        prefill_snapshot: None,
                    });
                }

                let t_decode0 = Instant::now();
                let first_tok = sample_next_token(logits_slice, params, &mut rng);
                generated.push(first_tok);
                let stopped = stop_check(first_tok) || !on_token(first_tok);
                if stopped {
                    let dt = t_decode0.elapsed().as_secs_f64();
                    let n_gen = generated.len();
                    return Ok(QuantumGenerateResult {
                        result: StreamGenerateResult {
                            prompt_tokens_evaluated: eval_len,
                            prefix_reused_tokens: prefix_reused,
                            generated_tokens: generated,
                            prefill_tps,
                            decode_tps: n_gen as f64 / dt.max(1e-9),
                            draft_n: 0,
                            draft_accepted: 0,
                        },
                        yielded: false,
                        prefill_snapshot: None,
                    });
                }
                (first_tok, eval_len, prefill_tps, t_decode0, false)
            };

        let mut pos = if resuming_decode {
            self.state.end_pos
        } else {
            n_prompt
        };
        let mut draft_n_tot = 0usize;
        let mut draft_acc_tot = 0usize;
        let mut drafts: Vec<u32> = Vec::with_capacity(self.max_draft.max(1));
        let mut draft_conf: Vec<f32> = Vec::with_capacity(self.max_draft.max(1));
        let mut verify_ids: Vec<u32> = Vec::with_capacity(HY4_W_TOKEN);
        let dump = hy4_dump_enabled();
        let mut yielded = false;

        while generated.len() < params.max_tokens && pos + 1 < self.max_ctx && !stopped {
            if (opts.should_yield)(generated.len()) {
                yielded = true;
                break;
            }
            if self.use_spec && self.max_draft > 0 && !dump {
                let remain = params.max_tokens - generated.len();
                let allow_draft = self
                    .max_draft
                    .min(remain.saturating_sub(1))
                    .min(self.max_ctx - pos - 1);
                if allow_draft > 0 {
                    let t_md0 = hy4_profile_enabled().then(Instant::now);
                    unsafe {
                        mtp_draft_tokens(
                            &self.weights,
                            &mut self.state,
                            cur_tok,
                            pos,
                            allow_draft,
                            self.draft_floor,
                            images,
                            &mut drafts,
                            &mut draft_conf,
                        )?;
                    }
                    if let Some(t0) = t_md0 {
                        self.state.profile.mtp_draft_ms += t0.elapsed().as_secs_f64() * 1e3;
                    }
                } else {
                    drafts.clear();
                }
            } else {
                drafts.clear();
            }

            verify_ids.clear();
            verify_ids.push(cur_tok);
            verify_ids.extend_from_slice(&drafts);
            let m = verify_ids.len();
            draft_n_tot += drafts.len();

            unsafe {
                forward_trunk_window(
                    &self.weights,
                    &mut self.state,
                    &verify_ids,
                    pos,
                    m,
                    images,
                )?;
            }

            let all_logits = unsafe {
                std::slice::from_raw_parts(self.state.h_logits.as_mut_ptr().cast::<f32>(), m * n_vocab)
            };

            let mut accepted = 0usize;
            for j in 0..m {
                let row = &all_logits[j * n_vocab..(j + 1) * n_vocab];
                let target_tok = sample_next_token(row, params, &mut rng);
                if j < drafts.len() && target_tok == drafts[j] {
                    accepted += 1;
                    draft_acc_tot += 1;
                    generated.push(target_tok);
                    self.committed_tokens.push(verify_ids[j]);
                    if stop_check(target_tok) || !on_token(target_tok) || generated.len() >= params.max_tokens {
                        cur_tok = target_tok;
                        pos += j + 1;
                        stopped = true;
                        break;
                    }
                } else {
                    self.committed_tokens.push(verify_ids[j]);
                    generated.push(target_tok);
                    cur_tok = target_tok;
                    pos += j + 1;
                    if stop_check(target_tok) || !on_token(target_tok) || generated.len() >= params.max_tokens {
                        stopped = true;
                    }
                    break;
                }
            }
            if !stopped && accepted == drafts.len() && drafts.is_empty() {
                // Case m == 1 handled inside loop above
            }
            self.state.history.truncate(pos);
            self.committed_tokens.truncate(pos);
            self.state.end_pos = pos;
            self.state.mtp_done = self.state.mtp_done.min(pos);
        }

        let decode_s = t_decode0.elapsed().as_secs_f64();
        let decode_tps = generated.len() as f64 / decode_s.max(1e-9);
        Ok(QuantumGenerateResult {
            result: StreamGenerateResult {
                prompt_tokens_evaluated,
                prefix_reused_tokens: prefix_reused,
                generated_tokens: generated,
                prefill_tps,
                decode_tps,
                draft_n: draft_n_tot,
                draft_accepted: draft_acc_tot,
            },
            yielded,
            prefill_snapshot: None,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        n_predict: usize,
    ) -> Result<(Vec<u32>, f64, f64, Vec<(u32, f32)>)> {
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            max_tokens: n_predict,
            stop_token_ids: Vec::new(),
        };
        let res = self.generate_stream(prompt_tokens, &params, |_| true)?;
        Ok((res.generated_tokens, res.prefill_tps, res.decode_tps, Vec::new()))
    }
}
