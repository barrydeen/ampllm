pub mod cache;
pub mod decode;
pub mod prefill;
pub mod state;
pub mod weights;

use crate::config::EngineConfig;
use crate::glm5::decode::{
    copy_head_x_row_to_base, decode_token_m1, enqueue_gpu1_half_and_head_m1,
    enqueue_mtp_draft_chained_on_gpu1, enqueue_mtp_draft_on_gpu1, finish_gpu1_head_sync_m1,
    mtp_cache_entry_m1, restore_gpu0_kda, rollback_kda_from_trace, run_gpu0_half_m1,
    snapshot_gpu0_kda, verify_batch_m,
};
use crate::glm5::prefill::{prefill_chunk, prefill_slice_pipelined, small_moe_max_nt, PrefillViews};
use crate::glm5::state::{DoorbellActiveGuard, Glm5State};
use crate::glm5::weights::Glm5ModelWeights;
use crate::inkling2::StreamGenerateResult;
use crate::sampling::{
    build_top_p_probs, build_top_p_probs_mapped, leviathan_verify_sample, sample_from_probs,
    sample_next_token, SamplingParams,
};
use crate::snapshot::{
    d2h_full, d2h_prefix, h2d_bytes, EngineSnapshotData, Glm5DsaSnap, Glm5KdaSnap, Glm5Snapshot,
    QuantumGenerateResult, QuantumOptions, SequenceSnapshot,
};
use ampllm_cache::{CacheProfile, CacheStats};
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::set_device;
use ampllm_gguf::{GgufFile, Glm5Config};
use ampllm_tokenizer::BpeTokenizer;
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[inline]
fn build_mtp_draft_probs(
    weights: &Glm5ModelWeights,
    state: &Glm5State,
    params: &SamplingParams,
) -> Vec<(u32, f32)> {
    let dv = &weights.draft_vocab;
    if state.last_mtp_used_dv && dv.n_lead < weights.cfg.n_vocab {
        let len = dv.n_lead + dv.extra_ids.len();
        build_top_p_probs_mapped(&state.mtp_logits_pin.as_slice()[..len], dv.n_lead, &dv.extra_ids, params)
    } else {
        build_top_p_probs(&state.mtp_logits_pin.as_slice()[..weights.cfg.n_vocab], params)
    }
}

fn ngram_predict(history: &[u32], prefix: &[u32]) -> Option<u32> {
    let h_len = history.len();
    let tot = h_len + prefix.len();
    let get = |i: usize| -> u32 {
        if i < h_len {
            history[i]
        } else {
            prefix[i - h_len]
        }
    };
    for &n in &[5usize, 4, 3] {
        if tot <= n {
            continue;
        }
        let pat_start = tot - n;
        let min_start = pat_start.saturating_sub(4096);
        let mut pos = pat_start - 1;
        loop {
            let mut ok = true;
            for k in 0..n {
                if get(pos + k) != get(pat_start + k) {
                    ok = false;
                    break;
                }
            }
            if ok && pos + n < tot {
                return Some(get(pos + n));
            }
            if pos == min_start {
                break;
            }
            pos -= 1;
        }
    }
    None
}

/// GLM-5.3-Flash engine owning state, weights, and mmap'd GGUF.
/// Field order enforces safe Drop sequence:
/// 1. `state` (joins `DoorbellWorker` thread first),
/// 2. `weights` (releases `DeviceBuffer`s & `CpuColdArena` second),
/// 3. `gguf` (unmaps GGUF shard mmaps last).
pub struct Glm5Engine {
    state: Glm5State,
    weights: Glm5ModelWeights,
    gguf: GgufFile,
    pub max_ctx: usize,
    pub use_cuda_graph: bool,
    pub use_spec: bool,
    pub use_draft_vocab: bool,
    pub chunk_size: usize,
    pub save_profile_path: Option<PathBuf>,
    pub committed_tokens: Vec<u32>,
    pub prefix_reuse_tokens_total: u64,
}

impl Glm5Engine {
    pub fn load(gguf: GgufFile, tok: &BpeTokenizer, cfg: EngineConfig) -> Result<Self> {
        let gcfg = Glm5Config::from_gguf(&gguf)?;
        let max_ctx = cfg.max_ctx.max(2048);
        let profile = match &cfg.load_profile {
            Some(p) => Some(CacheProfile::load(p)?),
            None => None,
        };
        let cpu_pool = Arc::new(CpuThreadPool::new(cfg.cpu_threads));
        let weights = Glm5ModelWeights::load(&gguf, tok, &gcfg, max_ctx, profile.as_ref(), None)?;
        let state = Glm5State::alloc(&weights, cpu_pool, max_ctx)?;
        tracing::info!("Glm5State ready: p2p_ok={}", state.p2p_ok);
        let chunk_size = state.max_prefill_chunk;

        Ok(Self {
            state,
            weights,
            gguf,
            max_ctx,
            use_cuda_graph: cfg.use_cuda_graph,
            use_spec: cfg.use_spec && gcfg.mtp_layer.is_some(),
            use_draft_vocab: true,
            chunk_size,
            save_profile_path: cfg.save_profile,
            committed_tokens: Vec::new(),
            prefix_reuse_tokens_total: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.committed_tokens.clear();
        let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
        let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
        self.state.expert_cache.drain_all(
            &self.weights,
            &self.weights.cold_arena,
            &self.state.sc0.stream,
            &self.state.sc1.stream,
            db0,
            db1,
        )?;
        self.state.expert_cache.on_new_prompt();
        self.state.reset()
    }

    pub fn capture_snapshot(&mut self, has_images: bool) -> Result<SequenceSnapshot> {
        {
            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
            self.state.expert_cache.drain_all(
                &self.weights,
                &self.weights.cold_arena,
                &self.state.sc0.stream,
                &self.state.sc1.stream,
                db0,
                db1,
            )?;
        }
        set_device(0)?;
        self.state.sc0.stream.sync()?;
        self.state.sc0.pf_stream.sync()?;
        self.state.sc0.d2h_stream.sync()?;
        set_device(1)?;
        self.state.sc1.stream.sync()?;
        self.state.sc1.pf_stream.sync()?;
        self.state.sc1.d2h_stream.sync()?;

        let pos = self.state.cur_pos;
        anyhow::ensure!(
            self.committed_tokens.len() == pos,
            "Glm5 snapshot invariant violated: committed={} != pos={pos}",
            self.committed_tokens.len()
        );
        let cfg = &self.weights.cfg;
        let lat_stride = cfg.kv_lora_rank + cfg.kv_lora_rank / 16;
        let res_rows = pos.min(self.weights.dsa_res_cap);
        let dsa_blocks = (pos / 4) + 1;
        let mut byte_len = 0usize;

        let head_x_row0 = if self.state.head_x_pos.is_some() {
            let b = d2h_prefix(&self.state.head_x, cfg.n_embd * 4)?;
            byte_len += b.len();
            Some(b)
        } else {
            None
        };

        let mut kda = Vec::with_capacity(self.state.kda.len());
        for k_opt in &self.state.kda {
            match k_opt {
                Some(k) => {
                    let s_mat = d2h_full(&k.s_mat)?;
                    let conv_q = d2h_full(&k.conv_q)?;
                    let conv_k = d2h_full(&k.conv_k)?;
                    let conv_v = d2h_full(&k.conv_v)?;
                    byte_len += s_mat.len() + conv_q.len() + conv_k.len() + conv_v.len();
                    kda.push(Some(Glm5KdaSnap {
                        s_mat,
                        conv_q,
                        conv_k,
                        conv_v,
                    }));
                }
                None => kda.push(None),
            }
        }

        let mut dsa = Vec::with_capacity(self.state.dsa.len());
        for d_opt in &self.state.dsa {
            match d_opt {
                Some(d) => {
                    let vram_lat8 = d2h_prefix(&d.vram_lat8, res_rows * lat_stride)?;
                    let ram_lat8_tail = if let Some(ref rl) = d.ram_lat8 {
                        if pos > self.weights.dsa_res_cap {
                            let s0 = self.weights.dsa_res_cap * lat_stride;
                            let s1 = pos * lat_stride;
                            rl.as_slice()[s0..s1].to_vec()
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    };
                    let ik_ring = d2h_full(&d.ik_ring)?;
                    let ig_ring = d2h_full(&d.ig_ring)?;
                    let pooled_f16 = d2h_prefix(&d.pooled_f16, dsa_blocks * cfg.idx_key_dim * 2)?;
                    byte_len += vram_lat8.len()
                        + ram_lat8_tail.len()
                        + ik_ring.len()
                        + ig_ring.len()
                        + pooled_f16.len();
                    dsa.push(Some(Glm5DsaSnap {
                        vram_lat8,
                        ram_lat8_tail,
                        ik_ring,
                        ig_ring,
                        pooled_f16,
                    }));
                }
                None => dsa.push(None),
            }
        }

        Ok(SequenceSnapshot {
            tokens: self.committed_tokens.clone(),
            has_images,
            allow_lcp_truncate: false,
            max_safe_rollback_delta: None,
            rollback_align: 1,
            byte_len,
            inner: EngineSnapshotData::Glm5(Glm5Snapshot {
                head_x_pos: self.state.head_x_pos,
                mtp_cached_pos: self.state.mtp_cached_pos,
                head_x_row0,
                kda,
                dsa,
            }),
        })
    }

    pub fn restore_snapshot(
        &mut self,
        snap: &SequenceSnapshot,
        _truncate_to: Option<usize>,
    ) -> Result<()> {
        let EngineSnapshotData::Glm5(ref gs) = snap.inner else {
            anyhow::bail!("expected Glm5 snapshot");
        };
        {
            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
            self.state.expert_cache.drain_all(
                &self.weights,
                &self.weights.cold_arena,
                &self.state.sc0.stream,
                &self.state.sc1.stream,
                db0,
                db1,
            )?;
        }
        set_device(0)?;
        self.state.sc0.stream.sync()?;
        self.state.sc0.pf_stream.sync()?;
        self.state.sc0.d2h_stream.sync()?;
        self.state.sc0.skip_flag_dev.zero()?;
        set_device(1)?;
        self.state.sc1.stream.sync()?;
        self.state.sc1.pf_stream.sync()?;
        self.state.sc1.d2h_stream.sync()?;
        self.state.sc1.skip_flag_dev.zero()?;

        {
            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
            db0.cancel_flag = 0;
        }
        {
            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
            db1.cancel_flag = 0;
        }

        if let Some(ref hx) = gs.head_x_row0 {
            h2d_bytes(&self.state.head_x, hx)?;
        }
        for (dst_opt, src_opt) in self.state.kda.iter().zip(gs.kda.iter()) {
            if let (Some(dst), Some(src)) = (dst_opt, src_opt) {
                h2d_bytes(&dst.s_mat, &src.s_mat)?;
                h2d_bytes(&dst.conv_q, &src.conv_q)?;
                h2d_bytes(&dst.conv_k, &src.conv_k)?;
                h2d_bytes(&dst.conv_v, &src.conv_v)?;
            }
        }
        let cfg = &self.weights.cfg;
        let lat_stride = cfg.kv_lora_rank + cfg.kv_lora_rank / 16;
        for (dst_opt, src_opt) in self.state.dsa.iter_mut().zip(gs.dsa.iter()) {
            if let (Some(dst), Some(src)) = (dst_opt, src_opt) {
                h2d_bytes(&dst.vram_lat8, &src.vram_lat8)?;
                if let Some(ref mut rl) = dst.ram_lat8 {
                    if !src.ram_lat8_tail.is_empty() {
                        let s0 = self.weights.dsa_res_cap * lat_stride;
                        let s1 = s0 + src.ram_lat8_tail.len();
                        rl.as_mut_slice()[s0..s1].copy_from_slice(&src.ram_lat8_tail);
                    }
                }
                h2d_bytes(&dst.ik_ring, &src.ik_ring)?;
                h2d_bytes(&dst.ig_ring, &src.ig_ring)?;
                h2d_bytes(&dst.pooled_f16, &src.pooled_f16)?;
            }
        }

        self.state.cur_pos = snap.tokens.len();
        self.state.head_x_pos = gs.head_x_pos;
        self.state.mtp_cached_pos = gs.mtp_cached_pos;
        self.committed_tokens.clear();
        self.committed_tokens.extend_from_slice(&snap.tokens);
        Ok(())
    }

    pub fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }

    pub fn save_profile(&self) -> Result<()> {
        if let Some(ref path) = self.save_profile_path {
            let prof = self.state.expert_cache.export_profile(
                &self.weights.cfg.arch,
                self.weights.cfg.n_layers,
                self.weights.cfg.n_expert,
            );
            prof.save(path)?;
        }
        Ok(())
    }

    pub fn cache_summary(&self) -> String {
        format!(
            "GLM-5.3 split_layer={} | VRAM experts {}/{} ({:.1}%) | GPU0={} MiB, GPU1={} MiB | dsa_res_cap={} | spec={} | graph={}",
            self.weights.split_layer,
            self.weights.vram_resident_experts,
            self.weights.total_routed_experts,
            100.0 * self.weights.vram_resident_experts as f64
                / self.weights.total_routed_experts.max(1) as f64,
            self.weights.vram_used_gpu0_mb,
            self.weights.vram_used_gpu1_mb,
            self.weights.dsa_res_cap,
            self.use_spec,
            self.use_cuda_graph,
        )
    }

    fn prefill_slice(
        &mut self,
        tokens: &[u32],
        need_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>)> {
        let total = tokens.len();
        if total == 0 {
            anyhow::bail!("empty prompt suffix in Glm5Engine::prefill_slice");
        }
        let out = if total <= self.state.max_prefill_chunk {
            let pos0 = self.state.cur_pos;
            let out = prefill_chunk(
                &self.gguf,
                &self.weights,
                &mut self.state,
                tokens,
                pos0,
                None,
                true,
                need_logits,
                None,
            )?
            .ok_or_else(|| anyhow::anyhow!("final prefill_chunk produced no logits"))?;
            self.committed_tokens.extend_from_slice(tokens);
            out
        } else {
            {
                let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                self.state.expert_cache.drain_all(
                    &self.weights,
                    &self.weights.cold_arena,
                    &self.state.sc0.stream,
                    &self.state.sc1.stream,
                    db0,
                    db1,
                )?;
            }
            let use_pipelined = total > self.state.max_lend_chunk;
            let chunk_size = if use_pipelined {
                let n_chunks = total.div_ceil(self.state.max_lend_chunk);
                total.div_ceil(n_chunks)
            } else {
                total
            };
            let small_max_nt = small_moe_max_nt();
            let gbuf_slots = 24usize;
            let gbuf_stride = self.weights.max_trunk_blob_stride.max(4096);
            let grp_window_rows = if chunk_size >= 1024 {
                4096usize
            } else {
                chunk_size.clamp(512, 2048)
            };
            let need_bytes = PrefillViews::required_lend_bytes(
                &self.weights.cfg,
                chunk_size,
                gbuf_stride,
                gbuf_slots,
                small_max_nt,
                grp_window_rows,
            );
            let (tail0, tail1) = {
                let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                self.state.expert_cache.lend_both_tails_parallel(
                    &self.weights,
                    &self.weights.cold_arena,
                    &self.state.sc0.stream,
                    &self.state.sc1.stream,
                    db0,
                    db1,
                    need_bytes,
                    need_bytes,
                )?
            };
            let cfg = &self.weights.cfg;
            let views_res = (|| -> Result<(PrefillViews, PrefillViews)> {
                set_device(0)?;
                let v0 = PrefillViews::from_lent_tail(
                    &tail0,
                    cfg,
                    chunk_size,
                    gbuf_stride,
                    gbuf_slots,
                    small_max_nt,
                    grp_window_rows,
                    &self.state.sc0.stream,
                )?;
                set_device(1)?;
                let v1 = PrefillViews::from_lent_tail(
                    &tail1,
                    cfg,
                    chunk_size,
                    gbuf_stride,
                    gbuf_slots,
                    small_max_nt,
                    grp_window_rows,
                    &self.state.sc1.stream,
                )?;
                Ok((v0, v1))
            })();
            let (views0, views1) = match views_res {
                Ok(v) => v,
                Err(e) => {
                    let _ = self.state.expert_cache.return_both_tails_parallel(
                        Some(tail0),
                        Some(tail1),
                        &self.weights,
                        &self.weights.cold_arena,
                        &self.state.sc0.stream,
                        &self.state.sc1.stream,
                    );
                    return Err(e);
                }
            };
            let exec_res = if use_pipelined {
                prefill_slice_pipelined(
                    &self.gguf,
                    &self.weights,
                    &mut self.state,
                    tokens,
                    chunk_size,
                    need_logits,
                    Some((&views0, &views1)),
                )
            } else {
                let pos0 = self.state.cur_pos;
                prefill_chunk(
                    &self.gguf,
                    &self.weights,
                    &mut self.state,
                    tokens,
                    pos0,
                    None,
                    true,
                    need_logits,
                    Some((&views0, &views1)),
                )?
                .ok_or_else(|| anyhow::anyhow!("final prefill_chunk produced no logits"))
            };
            set_device(0)?;
            self.state.sc0.pf_stream.sync()?;
            self.state.sc0.d2h_stream.sync()?;
            set_device(1)?;
            self.state.sc1.pf_stream.sync()?;
            self.state.sc1.d2h_stream.sync()?;
            let ret_res = self.state.expert_cache.return_both_tails_parallel(
                Some(tail0),
                Some(tail1),
                &self.weights,
                &self.weights.cold_arena,
                &self.state.sc0.stream,
                &self.state.sc1.stream,
            );
            let out = exec_res?;
            ret_res?;
            self.committed_tokens.extend_from_slice(tokens);
            out
        };
        if total >= 256 {
            let max_pl = if total >= 2048 {
                12
            } else if total >= 512 {
                6
            } else {
                2
            };
            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
            let _ = self.state.expert_cache.warmup_from_prefill_heat(
                &self.weights,
                &self.weights.cold_arena,
                &self.state.sc0.stream,
                &self.state.sc1.stream,
                db0,
                db1,
                max_pl,
            )?;
        }
        Ok(out)
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
        if prompt_tokens.is_empty() {
            anyhow::bail!("empty prompt tokens");
        }

        let reused = if !self.committed_tokens.is_empty()
            && self.committed_tokens.len() == self.state.cur_pos
            && prompt_tokens.len() > self.committed_tokens.len()
            && prompt_tokens.starts_with(&self.committed_tokens)
        {
            self.committed_tokens.len()
        } else {
            if !self.committed_tokens.is_empty() || self.state.cur_pos > 0 {
                self.reset()?;
            }
            0
        };
        if reused > 0 && !opts.resume_active {
            self.prefix_reuse_tokens_total += reused as u64;
        }

        let suffix = &prompt_tokens[reused..];
        let need_logits = params.temperature > 0.0;
        let mut rng = params.seed.max(1);
        let resuming_decode = opts.resume_active && reused > 0 && suffix.len() == 1;

        let (first_tok, prompt_tokens_evaluated, prefill_s, prefill_snapshot) = if resuming_decode {
            (suffix[0], 0usize, 0.0f64, None)
        } else {
            let t0 = Instant::now();
            let (greedy_first, logits_opt) = self.prefill_slice(suffix, need_logits)?;
            let prefill_s = t0.elapsed().as_secs_f64();
            let first_tok = if let Some(ref logits) = logits_opt {
                sample_next_token(logits, params, &mut rng)
            } else {
                greedy_first
            };
            let prefill_snapshot = if opts.capture_prefill_snap {
                Some(self.capture_snapshot(false)?)
            } else {
                None
            };
            (first_tok, suffix.len(), prefill_s, prefill_snapshot)
        };

        // Activate doorbell busy-spinners only after prefill (and any prefill snapshot) complete so CPU cores stay free.
        let _active_guard = DoorbellActiveGuard::new(&self.state.doorbell.active);

        let mut generated = Vec::new();
        let mut draft_n = 0usize;
        let mut draft_accepted = 0usize;
        let mut yielded = false;
        let t_dec = Instant::now();
        let prof = std::env::var_os("AMPLLM_PROFILE").is_some();
        let mut g0_ns = 0u128;
        let mut g1_ns = 0u128;
        let mut mtp_ns = 0u128;
        let mut host_ns = 0u128;
        let mut acc_ns = 0u128;
        let mut acc_n = 0usize;
        let mut rej_ns = 0u128;
        let mut rej_n = 0usize;
        let mut mtp_runs = 0usize;
        let mut prof_steps = 0usize;
        let mut gpu_promos_prof = 0u64;
        let mut routed_pairs_prof = 0u64;
        let mut vram_hits_prof = 0u64;
        let miss_exp0 = self.state.doorbell.miss_experts_total.load(Ordering::Relaxed);
        let uniq_exp0 = self
            .state
            .doorbell
            .unique_cold_experts_total
            .load(Ordering::Relaxed);
        let cpu_ns0 = self.state.doorbell.cpu_busy_ns.load(Ordering::Relaxed);
        let cpu0_ns0 = self.state.doorbell.cpu_busy_ns_g0.load(Ordering::Relaxed);
        let cpu1_ns0 = self.state.doorbell.cpu_busy_ns_g1.load(Ordering::Relaxed);
        let pci0_0 = self.state.doorbell.db0.as_slice()[0].pci_fetch_total;
        let pci1_0 = self.state.doorbell.db1.as_slice()[0].pci_fetch_total;
        if let Some(ref mut pp) = self.state.phase_prof {
            pp.reset();
        }
        let lead_dense = self.weights.cfg.leading_dense_block_count;
        let split_l = self.weights.split_layer;
        let n_layers = self.weights.cfg.n_layers;
        let mtp_l_opt = self.weights.cfg.mtp_layer;

        let verify_m = if !self.use_spec || mtp_l_opt.is_none() {
            1usize
        } else {
            std::env::var("AMPLLM_VERIFY_M")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 4)
        };
        let enable_ngram = std::env::var("AMPLLM_NGRAM")
            .map(|v| v == "1")
            .unwrap_or(false);
        let enable_chain = std::env::var("AMPLLM_DRAFT_CHAIN")
            .map(|v| v != "0")
            .unwrap_or(verify_m >= 3);

        let enter_decode = if resuming_decode {
            params.max_tokens > 0
        } else if params.max_tokens > 0 && !params.stop_token_ids.contains(&first_tok) {
            generated.push(first_tok);
            let keep = on_token(first_tok);
            keep && generated.len() < params.max_tokens
        } else {
            false
        };

        if enter_decode {
            {
                let use_graph = self.use_cuda_graph && self.state.phase_prof.is_none();
                if !self.use_spec || mtp_l_opt.is_none() {
                    let mut cur_tok = first_tok;
                    while generated.len() < params.max_tokens {
                        if (opts.should_yield)(generated.len()) {
                            yielded = true;
                            break;
                        }
                        let ts = Instant::now();
                        let pos = self.state.cur_pos;
                        run_gpu0_half_m1(
                            &self.gguf,
                            &self.weights,
                            &mut self.state,
                            cur_tok,
                            pos,
                            use_graph,
                        )?;
                        if prof {
                            set_device(0)?;
                            self.state.sc0.stream.sync()?;
                            g0_ns += ts.elapsed().as_nanos();
                        }
                        let ts1 = Instant::now();
                        enqueue_gpu1_half_and_head_m1(
                            &self.weights,
                            &mut self.state,
                            pos,
                            need_logits,
                            use_graph,
                        )?;
                        let g_nxt = finish_gpu1_head_sync_m1(&mut self.state, pos)?;
                        if let Some(ref mut pp) = self.state.phase_prof {
                            pp.accumulate_step(&self.weights, pos & 1)?;
                        }
                        {
                            let decay = self.state.expert_cache.heat_decay;
                            self.state.expert_cache.decay_all(decay);
                        }
                        {
                            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                            let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                0,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc0.stream,
                                db0,
                                lead_dense..split_l,
                                true,
                                pos & 1,
                            )?;
                            gpu_promos_prof += p;
                            routed_pairs_prof += rp;
                            vram_hits_prof += vh;
                        }
                        {
                            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                            let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                1,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc1.stream,
                                db1,
                                split_l..n_layers,
                                true,
                                pos & 1,
                            )?;
                            gpu_promos_prof += p;
                            routed_pairs_prof += rp;
                            vram_hits_prof += vh;
                        }
                        self.state.cur_pos = pos + 1;
                        if prof {
                            g1_ns += ts1.elapsed().as_nanos();
                            prof_steps += 1;
                        }
                        self.committed_tokens.push(cur_tok);
                        let next_tok = if need_logits {
                            sample_next_token(self.state.target_logits_pin.as_slice(), params, &mut rng)
                        } else {
                            g_nxt
                        };
                        if params.stop_token_ids.contains(&next_tok) {
                            break;
                        }
                        generated.push(next_tok);
                        if !on_token(next_tok) || generated.len() >= params.max_tokens {
                            break;
                        }
                        cur_tok = next_tok;
                    }
                } else if verify_m >= 2 {
                    let mut cur_tok = first_tok;
                    let mut pos0 = self.state.cur_pos;
                    let mut step_use_dv = self.use_draft_vocab
                        && self
                            .weights
                            .draft_vocab
                            .in_draft
                            .get(cur_tok as usize)
                            .copied()
                            .unwrap_or(false);
                    let mut batch_toks: Vec<u32> = Vec::with_capacity(verify_m);
                    let mut draft_probs: Vec<Vec<(u32, f32)>> = Vec::with_capacity(verify_m - 1);

                    while generated.len() < params.max_tokens {
                        if (opts.should_yield)(generated.len()) {
                            yielded = true;
                            break;
                        }
                        let rem = params.max_tokens - generated.len();
                        if rem == 1 {
                            let ts_1 = Instant::now();
                            let (g_nxt, logits_opt) = decode_token_m1(
                                &self.gguf,
                                &self.weights,
                                &mut self.state,
                                cur_tok,
                                need_logits,
                            )?;
                            if prof {
                                g1_ns += ts_1.elapsed().as_nanos();
                                prof_steps += 1;
                            }
                            self.committed_tokens.push(cur_tok);
                            let final_tok = if let Some(ref lg) = logits_opt {
                                sample_next_token(lg, params, &mut rng)
                            } else {
                                g_nxt
                            };
                            if !params.stop_token_ids.contains(&final_tok) {
                                generated.push(final_tok);
                                let _ = on_token(final_tok);
                            }
                            break;
                        }

                        let max_k = (verify_m - 1).min(rem - 1);
                        batch_toks.clear();
                        batch_toks.push(cur_tok);
                        draft_probs.clear();

                        let ts_mtp = Instant::now();
                        let mut mtp_drafted = false;
                        if max_k >= 1 {
                            if let Some(hx_p) = self.state.head_x_pos {
                                if hx_p + 1 == pos0 && self.state.mtp_cached_pos <= hx_p {
                                    let hx_ptr = self.state.head_x.ptr.cast();
                                    enqueue_mtp_draft_on_gpu1(
                                        &self.gguf,
                                        &self.weights,
                                        &mut self.state,
                                        hx_p,
                                        hx_ptr,
                                        cur_tok,
                                        step_use_dv,
                                        need_logits,
                                        use_graph,
                                    )?;
                                    set_device(1)?;
                                    self.state.sc1.stream.sync()?;
                                    let d1 = if need_logits {
                                        let pr = build_mtp_draft_probs(&self.weights, &self.state, params);
                                        let t = sample_from_probs(&pr, &mut rng);
                                        draft_probs.push(pr);
                                        t
                                    } else {
                                        self.state.mtp_tok_host.as_slice()[0] as u32
                                    };
                                    if let Some(mtp_l) = mtp_l_opt {
                                        let db1 = &self.state.doorbell.db1.as_slice()[0];
                                        self.state
                                            .expert_cache
                                            .note_layer_route(mtp_l, &db1.routed_ids[mtp_l][..8]);
                                    }
                                    if !params.stop_token_ids.contains(&d1) {
                                        batch_toks.push(d1);
                                        mtp_drafted = true;
                                    } else if need_logits {
                                        draft_probs.pop();
                                    }
                                }
                            }
                        }

                        let mut cur_mtp_pos = pos0;
                        for j in 1..max_k {
                            if !mtp_drafted || batch_toks.len() <= j {
                                break;
                            }
                            let mut next_d: Option<(u32, Option<Vec<(u32, f32)>>)> = None;
                            if enable_ngram {
                                if let Some(ng) = ngram_predict(&self.committed_tokens, &batch_toks) {
                                    if !params.stop_token_ids.contains(&ng) {
                                        next_d = Some((
                                            ng,
                                            if need_logits {
                                                Some(vec![(ng, 1.0f32)])
                                            } else {
                                                None
                                            },
                                        ));
                                    }
                                }
                            }
                            if next_d.is_none() && enable_chain {
                                let prev_d = *batch_toks.last().unwrap();
                                let chain_dv = step_use_dv
                                    && self
                                        .weights
                                        .draft_vocab
                                        .in_draft
                                        .get(prev_d as usize)
                                        .copied()
                                        .unwrap_or(false);
                                enqueue_mtp_draft_chained_on_gpu1(
                                    &self.gguf,
                                    &self.weights,
                                    &mut self.state,
                                    cur_mtp_pos,
                                    prev_d,
                                    chain_dv,
                                    need_logits,
                                    use_graph,
                                )?;
                                set_device(1)?;
                                self.state.sc1.stream.sync()?;
                                cur_mtp_pos += 1;
                                if let Some(mtp_l) = mtp_l_opt {
                                    let db1 = &self.state.doorbell.db1.as_slice()[0];
                                    self.state
                                        .expert_cache
                                        .note_layer_route(mtp_l, &db1.routed_ids[mtp_l][..8]);
                                }
                                let cd = if need_logits {
                                    let pr = build_mtp_draft_probs(&self.weights, &self.state, params);
                                    let t = sample_from_probs(&pr, &mut rng);
                                    Some((t, Some(pr)))
                                } else {
                                    Some((self.state.mtp_tok_host.as_slice()[0] as u32, None))
                                };
                                if let Some((tok, pr_opt)) = cd {
                                    if !params.stop_token_ids.contains(&tok) {
                                        next_d = Some((tok, pr_opt));
                                    }
                                }
                            }
                            match next_d {
                                Some((tok, pr_opt)) => {
                                    batch_toks.push(tok);
                                    if let Some(pr) = pr_opt {
                                        draft_probs.push(pr);
                                    }
                                }
                                None => break,
                            }
                        }
                        if prof {
                            mtp_ns += ts_mtp.elapsed().as_nanos();
                        }

                        let m_cur = batch_toks.len();
                        let k_cur = m_cur - 1;

                        let ts_ver = Instant::now();
                        let target_toks = verify_batch_m(
                            &self.gguf,
                            &self.weights,
                            &mut self.state,
                            &batch_toks,
                            pos0,
                            need_logits,
                        )?;
                        if prof {
                            g1_ns += ts_ver.elapsed().as_nanos();
                            prof_steps += 1;
                        }

                        {
                            let decay = self.state.expert_cache.heat_decay;
                            self.state.expert_cache.decay_all(decay);
                        }
                        {
                            let n_r = m_cur * self.weights.cfg.n_expert_used.min(8);
                            let n_exp = self.weights.cfg.n_expert;
                            let db0 = &self.state.doorbell.db0.as_slice()[0];
                            for l in lead_dense..split_l {
                                let ids = &db0.routed_ids[l][..n_r];
                                let lc = &self.state.expert_cache.layers[l];
                                for &raw_id in ids {
                                    if raw_id >= 0 && (raw_id as usize) < n_exp {
                                        routed_pairs_prof += 1;
                                        if lc.res_mask_host[raw_id as usize] != 0 {
                                            vram_hits_prof += 1;
                                        }
                                    }
                                }
                                self.state.expert_cache.note_layer_route(l, ids);
                            }
                            let db1 = &self.state.doorbell.db1.as_slice()[0];
                            for l in split_l..n_layers {
                                let ids = &db1.routed_ids[l][..n_r];
                                let lc = &self.state.expert_cache.layers[l];
                                for &raw_id in ids {
                                    if raw_id >= 0 && (raw_id as usize) < n_exp {
                                        routed_pairs_prof += 1;
                                        if lc.res_mask_host[raw_id as usize] != 0 {
                                            vram_hits_prof += 1;
                                        }
                                    }
                                }
                                self.state.expert_cache.note_layer_route(l, ids);
                            }
                        }
                        {
                            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                            let (p, _, _) = self.state.expert_cache.sync_promos_and_refill_dev(
                                0,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc0.stream,
                                db0,
                                lead_dense..split_l,
                                false,
                                0,
                            )?;
                            gpu_promos_prof += p;
                        }
                        {
                            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                            let (p, _, _) = self.state.expert_cache.sync_promos_and_refill_dev(
                                1,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc1.stream,
                                db1,
                                split_l..self.weights.layers.len(),
                                false,
                                0,
                            )?;
                            gpu_promos_prof += p;
                        }

                        let mut acc = 0usize;
                        let mut bonus_tok = 0u32;
                        let mut stop_emitted = false;
                        let mut client_cancelled = false;
                        if !need_logits {
                            while acc < k_cur {
                                let d_i = batch_toks[acc + 1];
                                let t_i = target_toks[acc];
                                if t_i == d_i {
                                    acc += 1;
                                    generated.push(d_i);
                                    if !on_token(d_i) {
                                        client_cancelled = true;
                                        break;
                                    }
                                } else {
                                    bonus_tok = t_i;
                                    break;
                                }
                            }
                            if acc == k_cur && !client_cancelled {
                                bonus_tok = target_toks[k_cur];
                            }
                        } else {
                            let nv = self.weights.cfg.n_vocab;
                            let t_logits = self.state.target_logits_pin.as_slice();
                            while acc < k_cur {
                                let d_i = batch_toks[acc + 1];
                                let row = &t_logits[acc * nv..(acc + 1) * nv];
                                let (tok, accepted) = leviathan_verify_sample(
                                    row,
                                    d_i,
                                    &draft_probs[acc],
                                    params,
                                    &mut rng,
                                );
                                if accepted {
                                    acc += 1;
                                    generated.push(d_i);
                                    if !on_token(d_i) {
                                        client_cancelled = true;
                                        break;
                                    }
                                } else {
                                    bonus_tok = tok;
                                    break;
                                }
                            }
                            if acc == k_cur && !client_cancelled {
                                let row = &t_logits[k_cur * nv..(k_cur + 1) * nv];
                                bonus_tok = sample_next_token(row, params, &mut rng);
                            }
                        }

                        draft_n += k_cur;
                        draft_accepted += acc;

                        let n_commit = acc + 1;
                        if n_commit < m_cur {
                            rollback_kda_from_trace(&self.weights, &mut self.state, n_commit)?;
                        }
                        self.committed_tokens.extend_from_slice(&batch_toks[..n_commit]);

                        if !client_cancelled && generated.len() < params.max_tokens {
                            if params.stop_token_ids.contains(&bonus_tok) {
                                stop_emitted = true;
                            } else {
                                generated.push(bonus_tok);
                                if !on_token(bonus_tok) {
                                    client_cancelled = true;
                                }
                            }
                        }

                        // Repair blk.45 DSA cache using true (head_x[j], batch_toks[j+1]) for j < acc
                        self.state.mtp_cached_pos = pos0;
                        let head_x_base = self.state.head_x.ptr as usize;
                        for j in 0..acc {
                            let hx_j = (head_x_base + j * self.weights.cfg.n_embd * 4) as *const f32;
                            mtp_cache_entry_m1(
                                &self.gguf,
                                &self.weights,
                                &mut self.state,
                                pos0 + j,
                                hx_j,
                                batch_toks[j + 1],
                            )?;
                        }
                        if acc > 0 {
                            copy_head_x_row_to_base(&mut self.state, acc, self.weights.cfg.n_embd)?;
                        }
                        let new_pos0 = pos0 + n_commit;
                        self.state.cur_pos = new_pos0;
                        self.state.head_x_pos = Some(new_pos0 - 1);
                        self.state.mtp_cached_pos = new_pos0 - 1;

                        if stop_emitted || client_cancelled || generated.len() >= params.max_tokens {
                            break;
                        }

                        cur_tok = bonus_tok;
                        if step_use_dv
                            && !self
                                .weights
                                .draft_vocab
                                .in_draft
                                .get(cur_tok as usize)
                                .copied()
                                .unwrap_or(false)
                        {
                            step_use_dv = false;
                        }
                        pos0 = new_pos0;
                    }
                } else {
                    let mut cur_tok = first_tok;
                    let mut pos = self.state.cur_pos;
                    let mut step_use_dv = self.use_draft_vocab
                        && self
                            .weights
                            .draft_vocab
                            .in_draft
                            .get(cur_tok as usize)
                            .copied()
                            .unwrap_or(false);

                    let db0_addr = self.state.doorbell.db0.host_ptr() as usize;
                    let write_cancel = |val: i32| unsafe {
                        let db = &mut *(db0_addr as *mut ampllm_cuda_kernels::glm5::Glm5Doorbell);
                        std::ptr::write_volatile(&mut db.cancel_flag, val);
                        std::sync::atomic::fence(Ordering::SeqCst);
                    };
                    write_cancel(0);

                    // Prime: run GPU 0 on committed cur_tok at pos WHILE GPU 1 blk.45 drafts + queues pos
                    run_gpu0_half_m1(
                        &self.gguf,
                        &self.weights,
                        &mut self.state,
                        cur_tok,
                        pos,
                        use_graph,
                    )?;
                    let mut draft_tok = 0u32;
                    let mut draft_probs: Vec<(u32, f32)> = Vec::new();
                    let mut has_draft = false;
                    let mut ran_mtp_prime = false;
                    if let Some(hx_p) = self.state.head_x_pos {
                        if hx_p + 1 == pos && self.state.mtp_cached_pos <= hx_p {
                            let hx_ptr = self.state.head_x.ptr.cast();
                            enqueue_mtp_draft_on_gpu1(
                                &self.gguf,
                                &self.weights,
                                &mut self.state,
                                hx_p,
                                hx_ptr,
                                cur_tok,
                                step_use_dv,
                                need_logits,
                                use_graph,
                            )?;
                            ran_mtp_prime = true;
                            mtp_runs += 1;
                        }
                    }

                    let mut g0_dev_ms = 0.0f64;
                    let mut g1_dev_ms = 0.0f64;
                    let mut mtp_dev_ms = 0.0f64;
                    let mut mtp_dev_n = 0usize;
                    let mut rp0_prof = 0u64;
                    let mut vh0_prof = 0u64;
                    let mut rp1_prof = 0u64;
                    let mut vh1_prof = 0u64;

                    let mut step_t0 = Instant::now();
                    enqueue_gpu1_half_and_head_m1(
                        &self.weights,
                        &mut self.state,
                        pos,
                        need_logits,
                        use_graph,
                    )?;

                    if ran_mtp_prime {
                        let ts_m = Instant::now();
                        set_device(1)?;
                        self.state.ev_mtp_done.sync()?;
                        if prof {
                            mtp_ns += ts_m.elapsed().as_nanos();
                            mtp_dev_ms += self.state.prof_mtp_start.elapsed_ms(&self.state.prof_mtp_end)? as f64;
                            mtp_dev_n += 1;
                        }
                        if need_logits {
                            draft_probs = build_mtp_draft_probs(&self.weights, &self.state, params);
                            draft_tok = sample_from_probs(&draft_probs, &mut rng);
                        } else {
                            draft_tok = self.state.mtp_tok_host.as_slice()[0] as u32;
                            if enable_ngram {
                                if let Some(ng) = ngram_predict(&self.committed_tokens, &[cur_tok]) {
                                    if !params.stop_token_ids.contains(&ng) {
                                        draft_tok = ng;
                                    }
                                }
                            }
                        }
                        has_draft = true;
                    }

                    let mut will_spec = has_draft
                        && generated.len() + 1 < params.max_tokens
                        && !params.stop_token_ids.contains(&draft_tok);
                    if will_spec {
                        if self.state.p2p_ok {
                            set_device(0)?;
                            self.state.sc0.stream.wait_event(&self.state.ev_g1_done)?;
                        }
                        snapshot_gpu0_kda(&self.weights, &mut self.state, use_graph)?;
                        write_cancel(0);
                        run_gpu0_half_m1(
                            &self.gguf,
                            &self.weights,
                            &mut self.state,
                            draft_tok,
                            pos + 1,
                            use_graph,
                        )?;
                    }

                    let mut iter_t0 = Instant::now();
                    let mut prev_was_acc: Option<bool> = None;
                    let mut mtp_unsynced = ran_mtp_prime;

                    loop {
                        let g_nxt = finish_gpu1_head_sync_m1(&mut self.state, pos)?;
                        self.state.cur_pos = pos + 1;
                        if let Some(ref mut pp) = self.state.phase_prof {
                            pp.accumulate_step(&self.weights, pos & 1)?;
                        }
                        if prof {
                            g1_ns += step_t0.elapsed().as_nanos();
                            g0_dev_ms += self.state.prof_g0_start[pos & 1]
                                .elapsed_ms(&self.state.prof_g0_end[pos & 1])? as f64;
                            g1_dev_ms += self.state.prof_g1_start
                                .elapsed_ms(&self.state.prof_g1_end)? as f64;
                            prof_steps += 1;
                            if let Some(was_acc) = prev_was_acc {
                                let dt = iter_t0.elapsed().as_nanos();
                                if was_acc {
                                    acc_ns += dt;
                                    acc_n += 1;
                                } else {
                                    rej_ns += dt;
                                    rej_n += 1;
                                }
                            }
                            iter_t0 = Instant::now();
                        }
                        let th0 = Instant::now();
                        self.committed_tokens.push(cur_tok);

                        let sampled_tok = if need_logits {
                            let nv = self.weights.cfg.n_vocab;
                            let t_logits = &self.state.target_logits_pin.as_slice()[..nv];
                            if will_spec {
                                let (tok, _) = leviathan_verify_sample(
                                    t_logits,
                                    draft_tok,
                                    &draft_probs,
                                    params,
                                    &mut rng,
                                );
                                tok
                            } else {
                                sample_next_token(t_logits, params, &mut rng)
                            }
                        } else {
                            g_nxt
                        };

                        let mut exit_now = false;
                        if params.stop_token_ids.contains(&sampled_tok) {
                            exit_now = true;
                        } else {
                            generated.push(sampled_tok);
                            let keep = on_token(sampled_tok);
                            if !keep || generated.len() >= params.max_tokens {
                                exit_now = true;
                            } else if (opts.should_yield)(generated.len()) {
                                yielded = true;
                                exit_now = true;
                            }
                        }
                        if exit_now {
                            if will_spec {
                                write_cancel(1);
                                set_device(0)?;
                                self.state.sc0.stream.sync()?;
                                self.state.sc0.pf_stream.sync()?;
                                self.state.sc0.promo_stream.sync()?;
                                write_cancel(0);
                            }
                            {
                            let decay = self.state.expert_cache.heat_decay;
                            self.state.expert_cache.decay_all(decay);
                        }
                            {
                                let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                                let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                    0,
                                    &self.weights,
                                    &self.weights.cold_arena,
                                    &self.state.sc0.stream,
                                    db0,
                                    lead_dense..split_l,
                                    true,
                                    pos & 1,
                                )?;
                                gpu_promos_prof += p;
                                routed_pairs_prof += rp;
                                vram_hits_prof += vh;
                                rp0_prof += rp;
                                vh0_prof += vh;
                            }
                            {
                                let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                                let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                    1,
                                    &self.weights,
                                    &self.weights.cold_arena,
                                    &self.state.sc1.stream,
                                    db1,
                                    split_l..n_layers,
                                    true,
                                    pos & 1,
                                )?;
                                gpu_promos_prof += p;
                                routed_pairs_prof += rp;
                                vram_hits_prof += vh;
                                rp1_prof += rp;
                                vh1_prof += vh;
                                if mtp_unsynced && self.weights.layers.len() > n_layers {
                                    let (p_m, rp_m, vh_m) =
                                        self.state.expert_cache.sync_promos_and_refill_dev(
                                            1,
                                            &self.weights,
                                            &self.weights.cold_arena,
                                            &self.state.sc1.stream,
                                            db1,
                                            n_layers..self.weights.layers.len(),
                                            true,
                                            0,
                                        )?;
                                    gpu_promos_prof += p_m;
                                    routed_pairs_prof += rp_m;
                                    vram_hits_prof += vh_m;
                                    rp1_prof += rp_m;
                                    vh1_prof += vh_m;
                                }
                            }
                            if will_spec {
                                restore_gpu0_kda(&self.weights, &mut self.state, use_graph)?;
                            }
                            break;
                        }

                        if step_use_dv
                            && !self
                                .weights
                                .draft_vocab
                                .in_draft
                                .get(sampled_tok as usize)
                                .copied()
                                .unwrap_or(false)
                        {
                            step_use_dv = false;
                        }

                        if prof {
                            host_ns += th0.elapsed().as_nanos();
                        }

                        let mut did_g0_cache = false;
                        if will_spec {
                            draft_n += 1;
                            if sampled_tok == draft_tok {
                                draft_accepted += 1;
                                prev_was_acc = Some(true);
                            } else {
                                prev_was_acc = Some(false);
                                let ts_redo = Instant::now();
                                write_cancel(1);
                                set_device(0)?;
                                self.state.sc0.stream.sync()?;
                                self.state.sc0.pf_stream.sync()?;
                                self.state.sc0.promo_stream.sync()?;
                                write_cancel(0);
                                {
                            let decay = self.state.expert_cache.heat_decay;
                            self.state.expert_cache.decay_all(decay);
                        }
                                {
                                    let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                                    let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                        0,
                                        &self.weights,
                                        &self.weights.cold_arena,
                                        &self.state.sc0.stream,
                                        db0,
                                        lead_dense..split_l,
                                        true,
                                        pos & 1,
                                    )?;
                                    gpu_promos_prof += p;
                                    routed_pairs_prof += rp;
                                    vram_hits_prof += vh;
                                    rp0_prof += rp;
                                    vh0_prof += vh;
                                }
                                did_g0_cache = true;
                                restore_gpu0_kda(&self.weights, &mut self.state, use_graph)?;
                                run_gpu0_half_m1(
                                    &self.gguf,
                                    &self.weights,
                                    &mut self.state,
                                    sampled_tok,
                                    pos + 1,
                                    use_graph,
                                )?;
                                if prof {
                                    g0_ns += ts_redo.elapsed().as_nanos();
                                }
                            }
                        } else {
                            prev_was_acc = Some(false);
                            let ts_g0 = Instant::now();
                            run_gpu0_half_m1(
                                &self.gguf,
                                &self.weights,
                                &mut self.state,
                                sampled_tok,
                                pos + 1,
                                use_graph,
                            )?;
                            if prof {
                                g0_ns += ts_g0.elapsed().as_nanos();
                            }
                        }

                        // Queue blk.45 MTP for pos+2 AND immediately queue GPU 1 for pos+1 behind it!
                        let hx_ptr = self.state.head_x.ptr.cast();
                        enqueue_mtp_draft_on_gpu1(
                            &self.gguf,
                            &self.weights,
                            &mut self.state,
                            pos,
                            hx_ptr,
                            sampled_tok,
                            step_use_dv,
                            need_logits,
                            use_graph,
                        )?;
                        mtp_runs += 1;
                        mtp_unsynced = true;
                        step_t0 = Instant::now();
                        enqueue_gpu1_half_and_head_m1(
                            &self.weights,
                            &mut self.state,
                            pos + 1,
                            need_logits,
                            use_graph,
                        )?;

                        // Hide host cache bookkeeping for GPU 0 behind GPU 1 MTP(pos) / GPU 0(pos+1) execution.
                        if !did_g0_cache {
                            {
                            let decay = self.state.expert_cache.heat_decay;
                            self.state.expert_cache.decay_all(decay);
                        }
                            let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                            let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                0,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc0.stream,
                                db0,
                                lead_dense..split_l,
                                true,
                                pos & 1,
                            )?;
                            gpu_promos_prof += p;
                            routed_pairs_prof += rp;
                            vram_hits_prof += vh;
                            rp0_prof += rp;
                            vh0_prof += vh;
                        }

                        let ngram_hit = if !need_logits && enable_ngram {
                            ngram_predict(&self.committed_tokens, &[sampled_tok])
                                .filter(|&ng| !params.stop_token_ids.contains(&ng))
                        } else {
                            None
                        };
                        if let Some(ng) = ngram_hit {
                            draft_tok = ng;
                            has_draft = true;
                        } else {
                            let ts_m = Instant::now();
                            set_device(1)?;
                            self.state.ev_mtp_done.sync()?;
                            if prof {
                                mtp_ns += ts_m.elapsed().as_nanos();
                                mtp_dev_ms += self.state.prof_mtp_start.elapsed_ms(&self.state.prof_mtp_end)? as f64;
                                mtp_dev_n += 1;
                            }

                            if need_logits {
                                draft_probs = build_mtp_draft_probs(&self.weights, &self.state, params);
                                draft_tok = sample_from_probs(&draft_probs, &mut rng);
                            } else {
                                draft_tok = self.state.mtp_tok_host.as_slice()[0] as u32;
                            }
                            has_draft = true;
                        }

                        will_spec = has_draft
                            && generated.len() + 1 < params.max_tokens
                            && !params.stop_token_ids.contains(&draft_tok);
                        if will_spec {
                            if self.state.p2p_ok {
                                set_device(0)?;
                                self.state.sc0.stream.wait_event(&self.state.ev_g1_done)?;
                            }
                            snapshot_gpu0_kda(&self.weights, &mut self.state, use_graph)?;
                            write_cancel(0);
                            run_gpu0_half_m1(
                                &self.gguf,
                                &self.weights,
                                &mut self.state,
                                draft_tok,
                                pos + 2,
                                use_graph,
                            )?;
                        }
                        {
                            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                            let (p, rp, vh) = self.state.expert_cache.sync_promos_and_refill_dev(
                                1,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc1.stream,
                                db1,
                                split_l..n_layers,
                                true,
                                pos & 1,
                            )?;
                            gpu_promos_prof += p;
                            routed_pairs_prof += rp;
                            vram_hits_prof += vh;
                            rp1_prof += rp;
                            vh1_prof += vh;
                        }
                        if self.weights.layers.len() > n_layers && ngram_hit.is_none() {
                            let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                            let (p_m, rp_m, vh_m) = self.state.expert_cache.sync_promos_and_refill_dev(
                                1,
                                &self.weights,
                                &self.weights.cold_arena,
                                &self.state.sc1.stream,
                                db1,
                                n_layers..self.weights.layers.len(),
                                has_draft,
                                0,
                            )?;
                            gpu_promos_prof += p_m;
                            routed_pairs_prof += rp_m;
                            vram_hits_prof += vh_m;
                            rp1_prof += rp_m;
                            vh1_prof += vh_m;
                            mtp_unsynced = false;
                        }

                        cur_tok = sampled_tok;
                        pos += 1;
                    }
                    if prof && prof_steps > 0 {
                        let pci0 = self.state.doorbell.db0.as_slice()[0]
                            .pci_fetch_total
                            .saturating_sub(pci0_0);
                        let pci1 = self.state.doorbell.db1.as_slice()[0]
                            .pci_fetch_total
                            .saturating_sub(pci1_0);
                        let hit0 = if rp0_prof > 0 {
                            100.0 * (vh0_prof as f64) / (rp0_prof as f64)
                        } else {
                            0.0
                        };
                        let hit1 = if rp1_prof > 0 {
                            100.0 * (vh1_prof as f64) / (rp1_prof as f64)
                        } else {
                            0.0
                        };
                        eprintln!(
                            "AMPLLM_DEV_DETAIL steps={prof_steps} g0_dev={:.2}ms g1_dev={:.2}ms mtp_dev={:.2}ms | vram_hit[g0={hit0:.1}% g1={hit1:.1}%] | pci_fetch={:.2}/step[g0={:.2} g1={:.2}]",
                            g0_dev_ms / prof_steps as f64,
                            g1_dev_ms / prof_steps as f64,
                            if mtp_dev_n > 0 { mtp_dev_ms / mtp_dev_n as f64 } else { 0.0 },
                            (pci0 + pci1) as f64 / prof_steps as f64,
                            pci0 as f64 / prof_steps as f64,
                            pci1 as f64 / prof_steps as f64,
                        );
                    }
                }
            }
        }

        let decode_s = t_dec.elapsed().as_secs_f64();
        let pci0_delta = self.state.doorbell.db0.as_slice()[0]
            .pci_fetch_total
            .saturating_sub(pci0_0);
        let pci1_delta = self.state.doorbell.db1.as_slice()[0]
            .pci_fetch_total
            .saturating_sub(pci1_0);
        if prof && prof_steps > 0 {
            let miss_exp = self
                .state
                .doorbell
                .miss_experts_total
                .load(Ordering::Relaxed)
                .saturating_sub(miss_exp0);
            let uniq_exp = self
                .state
                .doorbell
                .unique_cold_experts_total
                .load(Ordering::Relaxed)
                .saturating_sub(uniq_exp0);
            let cpu_ns = self
                .state
                .doorbell
                .cpu_busy_ns
                .load(Ordering::Relaxed)
                .saturating_sub(cpu_ns0);
            let cpu0_ns = self
                .state
                .doorbell
                .cpu_busy_ns_g0
                .load(Ordering::Relaxed)
                .saturating_sub(cpu0_ns0);
            let cpu1_ns = self
                .state
                .doorbell
                .cpu_busy_ns_g1
                .load(Ordering::Relaxed)
                .saturating_sub(cpu1_ns0);
            let trunk_moe_layers = n_layers.saturating_sub(lead_dense);
            let total_layers_routed = prof_steps * trunk_moe_layers + mtp_runs;
            let tot_routed = (total_layers_routed as f64) * (self.weights.cfg.n_expert_used as f64);
            let hit_pct = if routed_pairs_prof > 0 {
                100.0 * (vram_hits_prof as f64) / (routed_pairs_prof as f64)
            } else {
                100.0 * (1.0 - (miss_exp as f64) / tot_routed.max(1.0))
            };
            let pci_exp = pci0_delta + pci1_delta;
            let ms_per_uexp = if uniq_exp > 0 {
                (cpu_ns as f64) / 1e6 / (uniq_exp as f64)
            } else {
                0.0
            };
            let acc_ms = if acc_n > 0 { (acc_ns as f64) / 1e6 / (acc_n as f64) } else { 0.0 };
            let rej_ms = if rej_n > 0 { (rej_ns as f64) / 1e6 / (rej_n as f64) } else { 0.0 };
            let toks_emitted = generated.len().saturating_sub(1);
            eprintln!(
                "AMPLLM_PROFILE batch_m={verify_m} steps={prof_steps} toks={toks_emitted} ({:.2}tok/step) draft={draft_accepted}/{draft_n}: avg_step={:.2}ms (gpu+ver={:.2}ms acc={acc_ms:.2}ms[{acc_n}] rej={rej_ms:.2}ms[{rej_n}] gpu0={:.2}ms mtp={:.2}ms host={:.2}ms cpu={:.2}ms[cpu0={:.2} cpu1={:.2}] ({:.3}ms/uexp)) total={:.2}ms/tok | vram_hit={hit_pct:.1}% cpu_pairs={:.1}/step pci_fetch={:.1}/step cpu_uniq={:.1}/step | promo={gpu_promos_prof}/{} evict={} d2h_evict={}",
                toks_emitted as f64 / prof_steps as f64,
                decode_s * 1000.0 / prof_steps as f64,
                (g1_ns as f64) / 1e6 / prof_steps as f64,
                (g0_ns as f64) / 1e6 / prof_steps as f64,
                (mtp_ns as f64) / 1e6 / prof_steps as f64,
                (host_ns as f64) / 1e6 / prof_steps as f64,
                (cpu_ns as f64) / 1e6 / prof_steps as f64,
                (cpu0_ns as f64) / 1e6 / prof_steps as f64,
                (cpu1_ns as f64) / 1e6 / prof_steps as f64,
                ms_per_uexp,
                decode_s * 1000.0 / toks_emitted.max(1) as f64,
                (miss_exp as f64) / prof_steps as f64,
                (pci_exp as f64) / prof_steps as f64,
                (uniq_exp as f64) / prof_steps as f64,
                self.state.expert_cache.promotions_total,
                self.state.expert_cache.evictions_total,
                self.state.expert_cache.d2h_evictions_total,
            );
        }
        if let Some(ref mut pp) = self.state.phase_prof {
            pp.print_and_reset();
        }
        let prefill_tps = if prefill_s > 0.0 {
            prompt_tokens_evaluated as f64 / prefill_s.max(1e-6)
        } else {
            0.0
        };
        let decode_denom = generated
            .len()
            .saturating_sub(if resuming_decode { 0 } else { 1 })
            .max(1) as f64;
        let decode_tps = decode_denom / decode_s.max(1e-6);

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
        let t0 = Instant::now();
        let (first_tok, step0_logits) = self.prefill_slice(prompt_tokens, true)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let _active_guard = DoorbellActiveGuard::new(&self.state.doorbell.active);

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
        let mut next_tok = first_tok;
        let t_dec = Instant::now();
        let lead_dense = self.weights.cfg.leading_dense_block_count;
        let split_l = self.weights.split_layer;
        let n_layers = self.weights.cfg.n_layers;
        for step in 0..n_predict {
            generated.push(next_tok);
            if step + 1 < n_predict {
                let (nxt, _) = decode_token_m1(
                    &self.gguf,
                    &self.weights,
                    &mut self.state,
                    next_tok,
                    false,
                )?;
                {
                            let decay = self.state.expert_cache.heat_decay;
                            self.state.expert_cache.decay_all(decay);
                        }
                {
                    let db0 = &mut self.state.doorbell.db0.as_mut_slice()[0];
                    let _ = self.state.expert_cache.sync_promos_and_refill_dev(
                        0,
                        &self.weights,
                        &self.weights.cold_arena,
                        &self.state.sc0.stream,
                        db0,
                        lead_dense..split_l,
                        true,
                        0,
                    )?;
                }
                {
                    let db1 = &mut self.state.doorbell.db1.as_mut_slice()[0];
                    let _ = self.state.expert_cache.sync_promos_and_refill_dev(
                        1,
                        &self.weights,
                        &self.weights.cold_arena,
                        &self.state.sc1.stream,
                        db1,
                        split_l..n_layers,
                        true,
                        0,
                    )?;
                }
                self.committed_tokens.push(next_tok);
                next_tok = nxt;
            }
        }
        let decode_s = t_dec.elapsed().as_secs_f64();
        let prefill_tps = prompt_tokens.len() as f64 / prefill_s.max(1e-6);
        let decode_tps = (n_predict.saturating_sub(1).max(1)) as f64 / decode_s.max(1e-6);

        Ok((generated, prefill_tps, decode_tps, top5))
    }
}
