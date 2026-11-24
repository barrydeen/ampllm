#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            max_tokens: 256,
            stop_token_ids: Vec::new(),
        }
    }
}

#[inline]
fn next_rand_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

fn argmax_id(logits: &[f32]) -> u32 {
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

fn build_top_p_probs_inner(
    logits: &[f32],
    id_map: Option<(usize, &[i32])>,
    params: &SamplingParams,
) -> Vec<(u32, f32)> {
    let map_id = |i: usize| -> u32 {
        match id_map {
            Some((n_lead, extra_ids)) if i >= n_lead => extra_ids[i - n_lead] as u32,
            _ => i as u32,
        }
    };
    if params.temperature <= 0.0 {
        return vec![(map_id(argmax_id(logits) as usize), 1.0)];
    }

    let temp = params.temperature.max(1e-5);
    let inv_t = 1.0f32 / temp;
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_l {
            max_l = v;
        }
    }
    let cut_sum = max_l - 18.0 * temp;
    let cut_keep = max_l - 12.0 * temp;

    let mut probs: Vec<(u32, f32)> = Vec::with_capacity(logits.len().min(2048));
    let mut sum = 0.0f64;
    let mut kept_mass = 0.0f64;
    for (i, &v) in logits.iter().enumerate() {
        if v > cut_sum {
            let p = ((v - max_l) * inv_t).exp();
            sum += p as f64;
            if v > cut_keep {
                kept_mass += p as f64;
                probs.push((map_id(i), p));
            }
        }
    }
    if sum <= 0.0 {
        return vec![(map_id(0), 1.0)];
    }
    let need_full = params.top_p <= 0.0
        || params.top_p >= 1.0
        || kept_mass < (params.top_p as f64) * sum
        || probs.is_empty();
    if need_full {
        probs.clear();
        for (i, &v) in logits.iter().enumerate() {
            if v > cut_sum {
                probs.push((map_id(i), ((v - max_l) * inv_t).exp()));
            }
        }
    }
    if probs.is_empty() {
        return vec![(map_id(0), 1.0)];
    }

    probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let cutoff = if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut cum = 0.0f64;
        let mut keep = 0usize;
        for (idx, &(_, p)) in probs.iter().enumerate() {
            cum += (p as f64) / sum;
            keep = idx + 1;
            if cum >= params.top_p as f64 {
                break;
            }
        }
        keep.max(1)
    } else {
        probs.len()
    };

    probs.truncate(cutoff);
    let kept_sum: f64 = probs.iter().map(|&(_, p)| p as f64).sum();
    if kept_sum > 0.0 {
        let inv_kept = (1.0 / kept_sum) as f32;
        for (_, p) in &mut probs {
            *p *= inv_kept;
        }
    }
    probs
}

/// Build normalized sparse top-p probability distribution `[(token_id, prob)]` sorted descending by prob.
pub fn build_top_p_probs(logits: &[f32], params: &SamplingParams) -> Vec<(u32, f32)> {
    build_top_p_probs_inner(logits, None, params)
}

/// Build normalized sparse top-p probability distribution for compacted draft-vocab logits
/// (`[0..n_lead)` followed by `extra_ids`).
pub fn build_top_p_probs_mapped(
    logits: &[f32],
    n_lead: usize,
    extra_ids: &[i32],
    params: &SamplingParams,
) -> Vec<(u32, f32)> {
    debug_assert_eq!(logits.len(), n_lead + extra_ids.len());
    build_top_p_probs_inner(logits, Some((n_lead, extra_ids)), params)
}

/// Sample one token from a normalized probability distribution `[(token_id, prob)]`.
pub fn sample_from_probs(probs: &[(u32, f32)], rng: &mut u64) -> u32 {
    let total: f64 = probs.iter().map(|&(_, p)| p as f64).sum();
    let target = next_rand_f64(rng) * total;
    let mut acc = 0.0f64;
    for &(id, p) in probs {
        acc += p as f64;
        if acc >= target {
            return id;
        }
    }
    probs.last().map(|&(id, _)| id).unwrap_or(0)
}

/// Exact Leviathan et al. (2023) speculative rejection sampling:
/// accepts `draft_tok` with probability `min(1, p_target(draft_tok) / q_draft(draft_tok))`,
/// otherwise resamples from normalized residual `max(0, p_target - q_draft)`, preserving the
/// target distribution identically while achieving ~75-85% acceptance under temperature/top-p.
pub fn leviathan_verify_sample(
    target_logits: &[f32],
    draft_tok: u32,
    draft_probs: &[(u32, f32)],
    params: &SamplingParams,
    rng: &mut u64,
) -> (u32, bool) {
    let target_probs = build_top_p_probs(target_logits, params);
    let p_d = target_probs
        .iter()
        .find(|&&(id, _)| id == draft_tok)
        .map_or(0.0, |&(_, p)| p);
    let q_d = draft_probs
        .iter()
        .find(|&&(id, _)| id == draft_tok)
        .map_or(0.0, |&(_, q)| q);

    if q_d > 0.0 && next_rand_f64(rng) * (q_d as f64) <= (p_d as f64) {
        return (draft_tok, true);
    }

    // Residual r(v) = max(0, p(v) - q(v)) over sorted token IDs in O(N log N)
    let mut t_id = target_probs.clone();
    let mut d_id = draft_probs.to_vec();
    t_id.sort_unstable_by_key(|&(id, _)| id);
    d_id.sort_unstable_by_key(|&(id, _)| id);

    let mut residual: Vec<(u32, f32)> = Vec::with_capacity(t_id.len());
    let mut res_sum = 0.0f64;
    let (mut ti, mut di) = (0usize, 0usize);
    while ti < t_id.len() {
        let (tid, tp) = t_id[ti];
        while di < d_id.len() && d_id[di].0 < tid {
            di += 1;
        }
        let dq = if di < d_id.len() && d_id[di].0 == tid {
            d_id[di].1
        } else {
            0.0
        };
        let r = tp - dq;
        if r > 0.0 {
            res_sum += r as f64;
            residual.push((tid, r));
        }
        ti += 1;
    }

    if res_sum <= 1e-12 || residual.is_empty() {
        return (sample_from_probs(&target_probs, rng), false);
    }
    let target = next_rand_f64(rng) * res_sum;
    let mut acc = 0.0f64;
    for &(id, r) in &residual {
        acc += r as f64;
        if acc >= target {
            return (id, false);
        }
    }
    (residual.last().map(|&(id, _)| id).unwrap_or(0), false)
}

pub fn sample_next_token(logits: &[f32], params: &SamplingParams, rng: &mut u64) -> u32 {
    if params.temperature <= 0.0 {
        return argmax_id(logits);
    }
    let probs = build_top_p_probs(logits, params);
    sample_from_probs(&probs, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leviathan_preserves_target_distribution() {
        let target_logits = vec![2.2f32, 1.4, 0.7, -0.3, 1.9, 0.1, -1.5, 0.4];
        let draft_logits = vec![1.9f32, 1.5, 0.5, 0.2, 2.0, -0.2, -0.8, 0.0];
        let params = SamplingParams {
            temperature: 0.9,
            top_p: 0.95,
            ..Default::default()
        };
        let p_true = build_top_p_probs(&target_logits, &params);
        let q_draft = build_top_p_probs(&draft_logits, &params);

        let n_trials = 60_000usize;
        let mut counts = vec![0usize; target_logits.len()];
        let mut rng = 1234567u64;
        let mut accepted = 0usize;
        for _ in 0..n_trials {
            let d = sample_from_probs(&q_draft, &mut rng);
            let (tok, acc) = leviathan_verify_sample(&target_logits, d, &q_draft, &params, &mut rng);
            if acc {
                accepted += 1;
            }
            counts[tok as usize] += 1;
        }
        let mut p_vec = vec![0.0f64; target_logits.len()];
        for &(id, p) in &p_true {
            p_vec[id as usize] += p as f64;
        }
        let tv: f64 = (0..target_logits.len())
            .map(|i| ((counts[i] as f64 / n_trials as f64) - p_vec[i]).abs())
            .sum::<f64>()
            * 0.5;
        assert!(tv < 0.015, "Leviathan TV distance too large: {tv} (accepted {accepted}/{n_trials})");
        assert!(accepted > n_trials * 7 / 10, "unexpected low accept: {accepted}/{n_trials}");
    }

    #[test]
    fn test_build_top_p_probs_mapped_matches_full() {
        let full = vec![1.5f32, -100.0, 2.3, 0.4, -100.0, 2.1, -100.0, 1.9];
        let compacted = vec![1.5f32, -100.0, 2.3, 2.1, 1.9];
        let extra = [5i32, 7];
        let params = SamplingParams {
            temperature: 0.6,
            top_p: 0.95,
            ..Default::default()
        };
        let p_full = build_top_p_probs(&full, &params);
        let p_map = build_top_p_probs_mapped(&compacted, 3, &extra, &params);
        assert_eq!(p_full.len(), p_map.len());
        for (a, b) in p_full.iter().zip(p_map.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }
}
