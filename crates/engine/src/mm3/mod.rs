pub mod dspark;
pub mod forward;
pub mod state;
pub mod vision;
pub mod weights;

pub use forward::Mm3LayerTrace;
pub use vision::Mm3ImageSpan;

use crate::config::EngineConfig;
use crate::inkling2::StreamGenerateResult;
use crate::sampling::{
    build_top_p_probs, leviathan_verify_sample, sample_from_probs, SamplingParams,
};
use crate::snapshot::{
    d2h_prefix, h2d_bytes, EngineSnapshotData, Mm3LayerSnap, Mm3Snapshot, QuantumGenerateResult,
    QuantumOptions, SequenceSnapshot,
};
use ampllm_cache::{CacheProfile, CacheStats};
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_gguf::{GgufFile, Mm3Config, Mm3VisionConfig};
use anyhow::{bail, Result};
use forward::{
    commit_all_promotions, forward_decode_verify, forward_decode_verify_traced, forward_prefill,
    forward_prefill_traced,
};
use state::{Mm3State, MM3_MAX_VERIFY};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use weights::Mm3Weights;

fn argmax_slice(logits: &[f32]) -> u32 {
    let mut best_id = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i;
        }
    }
    best_id as u32
}

pub struct Mm3Engine {
    pub cfg: Mm3Config,
    pub max_ctx: usize,
    pub prefix_reuse_tokens_total: u64,
    weights: Mm3Weights,
    state: Mm3State,
    use_dspark: bool,
    dspark_conf: f32,
    save_profile_path: Option<PathBuf>,
    pub draft_n_total: usize,
    pub draft_accepted_total: usize,
}

impl Drop for Mm3Engine {
    fn drop(&mut self) {
        if self.save_profile_path.is_some() {
            let _ = self.save_profile();
        }
    }
}

impl Mm3Engine {
    pub fn load(gguf: GgufFile, engine_cfg: EngineConfig) -> Result<Self> {
        let pool = Arc::new(CpuThreadPool::new_all_physical(engine_cfg.cpu_threads));
        let mut weights = Mm3Weights::load_non_expert(gguf, &engine_cfg)?;
        let cfg = weights.cfg.clone();
        let mut state = Mm3State::alloc(&weights, &engine_cfg, Arc::clone(&pool))?;
        weights.allocate_experts(&engine_cfg, &pool)?;
        state.init_promo_from_weights(&weights)?;

        let dspark_conf = std::env::var("AMPLLM_MM3_DSPARK_CONF")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.7)
            .clamp(0.0, 1.0);

        let pct = if weights.total_experts > 0 {
            100.0 * (weights.total_resident_experts as f64) / (weights.total_experts as f64)
        } else {
            100.0
        };
        let total_promo: usize = weights.promo_slot_counts.iter().sum();
        eprintln!(
            "mm3 loaded: gpus={:?} p2p={} split={} ctx={} kv={:?} dspark={} vision={} experts={}/{} ({pct:.1}%) +promo={} threads={}",
            weights.gpu_ids,
            weights.p2p_enabled,
            weights.layer_split,
            state.max_ctx,
            weights.kv_fmt,
            weights.dspark.is_some(),
            weights.vision.is_some(),
            weights.total_resident_experts,
            weights.total_experts,
            total_promo,
            pool.num_threads()
        );

        let use_dspark = weights.dspark.is_some();
        Ok(Self {
            cfg,
            max_ctx: state.max_ctx,
            prefix_reuse_tokens_total: 0,
            weights,
            state,
            use_dspark,
            dspark_conf,
            save_profile_path: engine_cfg.save_profile.clone(),
            draft_n_total: 0,
            draft_accepted_total: 0,
        })
    }

    pub fn set_use_dspark(&mut self, enabled: bool) {
        self.use_dspark = enabled && self.weights.dspark.is_some();
    }

    pub fn has_vision(&self) -> bool {
        self.weights.vision.is_some()
    }

    pub fn vision_config(&self) -> Option<&Mm3VisionConfig> {
        self.weights.vision.as_ref().map(|v| &v.cfg)
    }

    pub fn encode_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        let Some(ref vis) = self.weights.vision else {
            bail!("MM3 vision requested but no vision safetensors module was loaded");
        };
        let dev = self.weights.gpu_ids[0];
        vis.encode_image(rgb, width, height, dev)
    }

    pub fn committed_tokens(&self) -> &[u32] {
        &self.state.history
    }

    pub fn reset(&mut self) -> Result<()> {
        commit_all_promotions(&mut self.weights, &mut self.state)?;
        self.state.reset()
    }

    pub fn rollback_policy(&self) -> (bool, Option<usize>, usize) {
        let max_safe_rollback_delta = if self.use_dspark && self.weights.dspark.is_some() {
            Some(64)
        } else {
            None
        };
        (true, max_safe_rollback_delta, 1)
    }

    pub fn capture_snapshot(&mut self, has_images: bool) -> Result<SequenceSnapshot> {
        commit_all_promotions(&mut self.weights, &mut self.state)?;
        self.state.sync_all()?;
        let pos = self.state.pos;
        let cfg = &self.cfg;
        let ix_bytes = pos * cfg.ix_head_dim * 2;
        let kv_elems = pos * cfg.n_kv_heads * cfg.head_dim;
        let scale_bytes = pos * cfg.n_kv_heads * 4;
        let mut layers = Vec::with_capacity(self.state.kv.len());
        let mut byte_len = 0usize;
        for kv in &self.state.kv {
            let ic_bf16 = d2h_prefix(&kv.ic_bf16, ix_bytes)?;
            let (kc, vc, ks, vs) = match self.weights.kv_fmt {
                weights::Mm3KvFormat::Bf16 => {
                    let k = d2h_prefix(kv.kc_bf16.as_ref().unwrap(), kv_elems * 2)?;
                    let v = d2h_prefix(kv.vc_bf16.as_ref().unwrap(), kv_elems * 2)?;
                    (k, v, None, None)
                }
                weights::Mm3KvFormat::Int8HadV => {
                    let k = d2h_prefix(kv.kc_i8.as_ref().unwrap(), kv_elems)?;
                    let v = d2h_prefix(kv.vc_i8.as_ref().unwrap(), kv_elems)?;
                    let ks = d2h_prefix(kv.ks_i8.as_ref().unwrap(), scale_bytes)?;
                    let vs = d2h_prefix(kv.vs_i8.as_ref().unwrap(), scale_bytes)?;
                    (k, v, Some(ks), Some(vs))
                }
            };
            byte_len += ic_bf16.len()
                + kc.len()
                + vc.len()
                + ks.as_ref().map_or(0, Vec::len)
                + vs.as_ref().map_or(0, Vec::len);
            layers.push(Mm3LayerSnap {
                ic_bf16,
                kc,
                vc,
                ks,
                vs,
            });
        }
        let dspark_layers = match self.weights.dspark {
            Some(ref ds) => {
                let ring = ds.snapshot_ring()?;
                for (k, v) in &ring {
                    byte_len += k.len() + v.len();
                }
                Some(ring)
            }
            None => None,
        };
        Ok(SequenceSnapshot {
            tokens: self.state.history.clone(),
            has_images,
            allow_lcp_truncate: true,
            max_safe_rollback_delta: if self.use_dspark && self.weights.dspark.is_some() {
                Some(64)
            } else {
                None
            },
            rollback_align: 1,
            byte_len,
            inner: EngineSnapshotData::Mm3(Mm3Snapshot {
                layers,
                dspark_layers,
            }),
        })
    }

    pub fn restore_snapshot(
        &mut self,
        snap: &SequenceSnapshot,
        truncate_to: Option<usize>,
    ) -> Result<()> {
        let EngineSnapshotData::Mm3(ref ms) = snap.inner else {
            bail!("expected Mm3 snapshot");
        };
        commit_all_promotions(&mut self.weights, &mut self.state)?;
        self.state.sync_all()?;
        if ms.layers.len() != self.state.kv.len() {
            bail!("MM3 snapshot layer count mismatch");
        }
        let pos = truncate_to
            .unwrap_or(snap.tokens.len())
            .min(snap.tokens.len());
        let cfg = &self.cfg;
        let ix_bytes = pos * cfg.ix_head_dim * 2;
        let kv_elems = pos * cfg.n_kv_heads * cfg.head_dim;
        let scale_bytes = pos * cfg.n_kv_heads * 4;

        for (kv, ls) in self.state.kv.iter().zip(ms.layers.iter()) {
            h2d_bytes(&kv.ic_bf16, &ls.ic_bf16[..ix_bytes.min(ls.ic_bf16.len())])?;
            match self.weights.kv_fmt {
                weights::Mm3KvFormat::Bf16 => {
                    let nb = (kv_elems * 2).min(ls.kc.len());
                    h2d_bytes(kv.kc_bf16.as_ref().unwrap(), &ls.kc[..nb])?;
                    h2d_bytes(kv.vc_bf16.as_ref().unwrap(), &ls.vc[..nb])?;
                }
                weights::Mm3KvFormat::Int8HadV => {
                    let nb = kv_elems.min(ls.kc.len());
                    h2d_bytes(kv.kc_i8.as_ref().unwrap(), &ls.kc[..nb])?;
                    h2d_bytes(kv.vc_i8.as_ref().unwrap(), &ls.vc[..nb])?;
                    if let (Some(ref ks), Some(ref vs)) = (&ls.ks, &ls.vs) {
                        let sb = scale_bytes.min(ks.len());
                        h2d_bytes(kv.ks_i8.as_ref().unwrap(), &ks[..sb])?;
                        h2d_bytes(kv.vs_i8.as_ref().unwrap(), &vs[..sb])?;
                    }
                }
            }
        }
        if let (Some(ref ds), Some(ref dsl)) = (&self.weights.dspark, &ms.dspark_layers) {
            ds.restore_ring(dsl)?;
        }
        self.state.pos = pos;
        self.state.history.clear();
        self.state.history.extend_from_slice(&snap.tokens[..pos]);
        self.state.prof.reset();
        Ok(())
    }

    pub fn cache_stats(&self) -> &CacheStats {
        &self.state.cache_stats
    }

    pub fn cache_summary(&self) -> String {
        let s = &self.state.cache_stats;
        format!(
            "vram={}/{} ({:.1}% hit_rate={:.1}%)",
            self.weights.total_resident_experts,
            self.weights.total_experts,
            100.0 * (self.weights.total_resident_experts as f64)
                / (self.weights.total_experts.max(1) as f64),
            100.0 * s.vram_hit_rate()
        )
    }

    pub fn save_profile(&self) -> Result<()> {
        let Some(ref path) = self.save_profile_path else {
            return Ok(());
        };
        let n_layers = self.cfg.n_layers;
        let n_exp = self.cfg.n_experts;
        let mut ranked_experts = Vec::with_capacity(n_layers);
        let mut pairs: Vec<(u32, usize)> = Vec::new();
        for l in 0..n_layers {
            let mut layer_pairs: Vec<(usize, u32)> = (0..n_exp)
                .map(|e| (e, self.state.profile_counts[l][e]))
                .collect();
            layer_pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for &(e, c) in &layer_pairs {
                if c > 0 {
                    pairs.push((c, l * n_exp + e));
                }
            }
            ranked_experts.push(layer_pairs.into_iter().map(|(e, _)| e).collect());
        }
        pairs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let prof = CacheProfile {
            arch: "minimax-m3".to_owned(),
            n_layer_phys: n_layers,
            n_expert: n_exp,
            ranked_experts,
            global_ranked: pairs
                .into_iter()
                .map(|(_, le)| (le / n_exp, le % n_exp))
                .collect(),
        };
        prof.save(path)
    }

    /// Evaluate `tokens` from clean state and return final-token `[vocab_size]` FP32 logits (for parity tests).
    pub fn eval_prompt_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.reset()?;
        forward_prefill(
            &mut self.weights,
            &mut self.state,
            tokens,
            0,
            &[],
            false,
            true,
            false,
        )?;
        Ok(self.state.logits_host[..self.cfg.vocab_size].to_vec())
    }

    /// Evaluate `tokens` one token at a time via `forward_decode_verify` (`t = 1`) and return final-token `[vocab_size]` FP32 logits.
    pub fn eval_prompt_logits_stepwise(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        self.reset()?;
        for (i, &tok) in tokens.iter().enumerate() {
            let is_last = i + 1 == tokens.len();
            forward_decode_verify(
                &mut self.weights,
                &mut self.state,
                &[tok],
                i,
                false,
                is_last,
                false,
            )?;
        }
        Ok(self.state.logits_host[..self.cfg.vocab_size].to_vec())
    }

    pub fn layer_quant_tag(&self, il: usize) -> String {
        match self.weights.layers[il].ffn {
            weights::Mm3FfnWeights::Dense(ref d) => {
                format!(
                    "Dense({:?}/{:?}/{:?})",
                    d.gate.ggml_type, d.up.ggml_type, d.down.ggml_type
                )
            }
            weights::Mm3FfnWeights::Moe(ref m) => {
                format!("MoE({:?}/{:?}/{:?})", m.gate_type, m.up_type, m.down_type)
            }
        }
    }

    /// Evaluate `tokens` with per-layer residual + router tracing (`stepwise = false` batched prefill vs `true` t=1 decode).
    pub fn eval_prompt_traced(
        &mut self,
        tokens: &[u32],
        stepwise: bool,
    ) -> Result<(Vec<f32>, Mm3LayerTrace, String)> {
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        self.reset()?;
        let mut tr = Mm3LayerTrace::new(self.cfg.n_layers);
        if !stepwise {
            forward_prefill_traced(
                &mut self.weights,
                &mut self.state,
                tokens,
                0,
                &[],
                false,
                true,
                false,
                Some(&mut tr),
            )?;
        } else {
            for (i, &tok) in tokens.iter().enumerate() {
                let is_last = i + 1 == tokens.len();
                forward_decode_verify_traced(
                    &mut self.weights,
                    &mut self.state,
                    &[tok],
                    i,
                    false,
                    is_last,
                    false,
                    if is_last { Some(&mut tr) } else { None },
                )?;
            }
        }
        Ok((
            self.state.logits_host[..self.cfg.vocab_size].to_vec(),
            tr,
            self.cache_summary(),
        ))
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
        images: &[Mm3ImageSpan],
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
        images: &[Mm3ImageSpan],
        params: &SamplingParams,
        opts: QuantumOptions<'_>,
        mut on_token: F,
    ) -> Result<QuantumGenerateResult>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_tokens.is_empty() {
            bail!("empty prompt");
        }
        if prompt_tokens.len() >= self.max_ctx {
            bail!(
                "prompt length {} exceeds max_ctx {}",
                prompt_tokens.len(),
                self.max_ctx
            );
        }
        if !images.is_empty() && self.weights.vision.is_none() {
            bail!("MM3 vision spans passed but no vision safetensors module was loaded");
        }

        // Reuse KV + DSpark context prefix on text-only prompts (or exact active resume).
        let mut lcp = 0usize;
        if images.is_empty() && self.state.pos > 0 && !self.state.history.is_empty() {
            if opts.resume_active
                && prompt_tokens.len() == self.state.history.len() + 1
                && prompt_tokens.starts_with(&self.state.history)
            {
                lcp = self.state.history.len();
            } else if !opts.resume_active {
                let lim = self
                    .state
                    .history
                    .len()
                    .min(prompt_tokens.len().saturating_sub(1));
                while lcp < lim && self.state.history[lcp] == prompt_tokens[lcp] {
                    lcp += 1;
                }
                if let Some(ref ds) = self.weights.dspark {
                    // Guard against rolling back past the valid DSpark ring window.
                    if lcp > 0 && self.state.pos > lcp && self.state.pos.saturating_sub(ds.ring) > lcp {
                        lcp = 0;
                    }
                }
            }
        }

        if lcp == 0 {
            self.reset()?;
        } else {
            self.state.history.truncate(lcp);
            self.state.pos = lcp;
            if !opts.resume_active {
                self.prefix_reuse_tokens_total += lcp as u64;
            }
        }

        let eval_slice = &prompt_tokens[lcp..];
        let use_dspark = self.use_dspark && self.weights.dspark.is_some();
        let greedy = params.temperature <= 0.0;
        let mut rng = params.seed;
        let vocab = self.cfg.vocab_size;

        let resuming_decode = opts.resume_active
            && lcp > 0
            && lcp == self.state.pos
            && eval_slice.len() == 1
            && images.is_empty();

        let (mut next_tok, prompt_tokens_evaluated, t_pp, mut generated, mut stop_hit) =
            if resuming_decode {
                (eval_slice[0], 0usize, 0.0f64, Vec::new(), false)
            } else {
                let t_pp0 = Instant::now();
                let first_greedy = forward_prefill(
                    &mut self.weights,
                    &mut self.state,
                    eval_slice,
                    lcp,
                    images,
                    use_dspark,
                    true,
                    greedy,
                )?;
                let t_pp = t_pp0.elapsed().as_secs_f64();
                self.state.history.extend_from_slice(eval_slice);
                let first_tok = if greedy {
                    first_greedy.unwrap_or_else(|| argmax_slice(&self.state.logits_host[..vocab]))
                } else {
                    let probs = build_top_p_probs(&self.state.logits_host[..vocab], params);
                    sample_from_probs(&probs, &mut rng)
                };
                (first_tok, eval_slice.len(), t_pp, Vec::new(), false)
            };

        let mut draft_n = 0usize;
        let mut draft_accepted = 0usize;
        let max_new = params.max_tokens.min(self.max_ctx - self.state.pos);
        if !resuming_decode && max_new > 0 {
            generated.push(next_tok);
            let is_stop = params.stop_token_ids.contains(&next_tok);
            let keep = on_token(next_tok);
            if is_stop || !keep || generated.len() >= max_new {
                stop_hit = true;
            }
        }

        self.state.prof.reset();
        let t_dec0 = Instant::now();

        let d = self.cfg.n_embd;
        let mut emb_row = vec![0.0f32; d];
        let mut pos = self.state.pos;
        let mut yielded = false;

        while !stop_hit && generated.len() < max_new && pos < self.max_ctx {
            if (opts.should_yield)(generated.len()) {
                yielded = true;
                break;
            }
            let rem_tokens = max_new - generated.len();
            let rem_ctx = self.max_ctx - pos;
            let can_draft =
                use_dspark && rem_tokens > 1 && rem_ctx > 1 && self.dspark_conf < 1.0;

            if can_draft {
                let t_dp0 = self.state.prof_enabled.then(Instant::now);
                self.weights.embed_row_f32(next_tok, &mut emb_row)?;
                let (drafts, confs) = {
                    let ds = self.weights.dspark.as_mut().unwrap();
                    let block = ds.cfg.block_size.min(rem_tokens - 1).min(rem_ctx - 1);
                    ds.propose_greedy(
                        &emb_row,
                        next_tok,
                        pos,
                        block,
                        &self.weights.lm_head.buf,
                    )?
                };
                if let Some(t0) = t_dp0 {
                    self.state.prof.dspark_prop_us += t0.elapsed().as_secs_f64() * 1e6;
                }
                let mut k = 0usize;
                let max_k = (MM3_MAX_VERIFY - 1).min(drafts.len()).min(rem_ctx - 1);
                while k < max_k && confs[k] >= self.dspark_conf {
                    k += 1;
                }

                let mut win = Vec::with_capacity(1 + k);
                win.push(next_tok);
                win.extend_from_slice(&drafts[..k]);

                let argmax_ids = forward_decode_verify(
                    &mut self.weights,
                    &mut self.state,
                    &win,
                    pos,
                    true,
                    true,
                    greedy,
                )?;

                let mut j = 0usize;
                let mut bonus_tok = 0u32;
                let mut rejected_early = false;
                if greedy {
                    while j < k {
                        let target_tok = argmax_ids[j];
                        if drafts[j] != target_tok {
                            bonus_tok = target_tok;
                            rejected_early = true;
                            break;
                        }
                        let tok = drafts[j];
                        generated.push(tok);
                        let is_stop = params.stop_token_ids.contains(&tok);
                        let keep = on_token(tok);
                        j += 1;
                        if is_stop || !keep || generated.len() >= max_new {
                            stop_hit = true;
                            break;
                        }
                    }
                    if !stop_hit && !rejected_early {
                        bonus_tok = argmax_ids[j];
                    }
                } else {
                    while j < k {
                        let row = &self.state.logits_host[j * vocab..(j + 1) * vocab];
                        let (chosen, accepted) = leviathan_verify_sample(
                            row,
                            drafts[j],
                            &[(drafts[j], 1.0f32)],
                            params,
                            &mut rng,
                        );
                        if !accepted {
                            bonus_tok = chosen;
                            rejected_early = true;
                            break;
                        }
                        let tok = chosen;
                        generated.push(tok);
                        let is_stop = params.stop_token_ids.contains(&tok);
                        let keep = on_token(tok);
                        j += 1;
                        if is_stop || !keep || generated.len() >= max_new {
                            stop_hit = true;
                            break;
                        }
                    }
                    if !stop_hit && !rejected_early {
                        let row = &self.state.logits_host[j * vocab..(j + 1) * vocab];
                        let probs = build_top_p_probs(row, params);
                        bonus_tok = sample_from_probs(&probs, &mut rng);
                    }
                }

                draft_n += k;
                draft_accepted += j;
                self.state.history.extend_from_slice(&win[..j + 1]);
                pos += j + 1;
                self.state.pos = pos;

                if stop_hit || generated.len() >= max_new {
                    break;
                }

                next_tok = bonus_tok;
                generated.push(next_tok);
                let is_stop = params.stop_token_ids.contains(&next_tok);
                let keep = on_token(next_tok);
                if is_stop || !keep || generated.len() >= max_new {
                    stop_hit = true;
                }
            } else {
                let argmax_ids = forward_decode_verify(
                    &mut self.weights,
                    &mut self.state,
                    &[next_tok],
                    pos,
                    use_dspark,
                    true,
                    greedy,
                )?;
                self.state.history.push(next_tok);
                pos += 1;
                self.state.pos = pos;

                next_tok = if greedy {
                    argmax_ids[0]
                } else {
                    let probs = build_top_p_probs(&self.state.logits_host[..vocab], params);
                    sample_from_probs(&probs, &mut rng)
                };
                generated.push(next_tok);
                let is_stop = params.stop_token_ids.contains(&next_tok);
                let keep = on_token(next_tok);
                if is_stop || !keep || generated.len() >= max_new {
                    stop_hit = true;
                }
            }
        }

        let t_dec = t_dec0.elapsed().as_secs_f64();
        if self.state.prof_enabled && self.state.prof.steps > 0 {
            let p = &self.state.prof;
            eprintln!(
                "[MM3 PROF] steps={} toks={} total={:.1}ms | attn={:.1}ms d2h={:.1}ms cpu_moe={:.1}ms handoff={:.1}ms lm_head={:.1}ms dspark_prop={:.1}ms promo_sync={:.1}ms | static_hit={} promo_hit={} staged={} cpu_exp={} promos={}",
                p.steps,
                p.tokens,
                t_dec * 1e3,
                p.attn_us / 1e3,
                p.moe_d2h_us / 1e3,
                p.moe_cpu_us / 1e3,
                p.handoff_us / 1e3,
                p.lm_head_us / 1e3,
                p.dspark_prop_us / 1e3,
                p.promo_sync_us / 1e3,
                p.static_hits,
                p.promo_hits,
                p.staged_exps,
                p.cpu_exps,
                p.promos_issued
            );
        }
        self.draft_n_total += draft_n;
        self.draft_accepted_total += draft_accepted;

        let prefill_tps = if t_pp > 0.0 {
            prompt_tokens_evaluated as f64 / t_pp
        } else {
            0.0
        };
        let decode_tps = if t_dec > 0.0 {
            generated.len() as f64 / t_dec
        } else {
            0.0
        };

        Ok(QuantumGenerateResult {
            result: StreamGenerateResult {
                generated_tokens: generated,
                prompt_tokens_evaluated,
                prefix_reused_tokens: lcp,
                prefill_tps,
                decode_tps,
                draft_n,
                draft_accepted,
            },
            yielded,
            prefill_snapshot: None,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[u32],
        max_new: usize,
        stop_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            max_tokens: max_new,
            stop_token_ids: stop_ids.to_vec(),
        };
        Ok(self.generate_stream(prompt_tokens, &params, |_| true)?.generated_tokens)
    }
}
