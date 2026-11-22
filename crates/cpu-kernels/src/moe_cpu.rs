use crate::pool::CpuThreadPool;
use crate::quant_x86::{quantize_f32_to_q8_1_cpu, vec_dot_q_q8_1, CpuBlockQ8_1};
use ampllm_gguf::GgmlType;

/// One cold expert evaluation task for a single token.
#[derive(Clone, Copy)]
pub struct CpuExpertJob<'a> {
    pub gate_up: &'a [u8],
    pub down: &'a [u8],
    pub gate_up_scale: f32,
    pub weight: f32,
}

unsafe impl<'a> Send for CpuExpertJob<'a> {}
unsafe impl<'a> Sync for CpuExpertJob<'a> {}

/// Reusable scratch buffers for zero-allocation hybrid decode/prefill CPU MoE execution.
pub struct CpuMoeScratch {
    pub x_q8: Vec<CpuBlockQ8_1>,
    pub inter_f32: Vec<f32>,
    pub inter_q8: Vec<CpuBlockQ8_1>,
    pub acc_out: Vec<f32>,
}

impl CpuMoeScratch {
    pub fn new(n_embd: usize, n_ff: usize, max_jobs: usize) -> Self {
        Self {
            x_q8: vec![CpuBlockQ8_1::default(); n_embd / 32],
            inter_f32: vec![0.0f32; max_jobs * n_ff],
            inter_q8: vec![CpuBlockQ8_1::default(); max_jobs * (n_ff / 32)],
            acc_out: vec![0.0f32; max_jobs * n_embd],
        }
    }

    pub fn ensure_capacity(&mut self, n_embd: usize, n_ff: usize, max_jobs: usize) {
        let nb_x = n_embd / 32;
        if self.x_q8.len() < nb_x {
            self.x_q8.resize(nb_x, CpuBlockQ8_1::default());
        }
        let inter_len = max_jobs * n_ff;
        if self.inter_f32.len() < inter_len {
            self.inter_f32.resize(inter_len, 0.0);
        }
        let inter_q_len = max_jobs * (n_ff / 32);
        if self.inter_q8.len() < inter_q_len {
            self.inter_q8.resize(inter_q_len, CpuBlockQ8_1::default());
        }
        let acc_len = max_jobs * n_embd;
        if self.acc_out.len() < acc_len {
            self.acc_out.resize(acc_len, 0.0);
        }
    }
}

#[inline(always)]
fn situ_activate(g: f32, u: f32, beta: f32, linear_beta: f32) -> f32 {
    let g_cap = beta * (g / beta).tanh();
    let sig = 1.0f32 / (1.0f32 + (-g).exp());
    let u_cap = if linear_beta > 0.0 {
        linear_beta * (u / linear_beta).tanh()
    } else {
        u
    };
    g_cap * sig * u_cap
}

/// Execute `jobs` (0..=topk cold experts for a single decode token `M=1`), writing the
/// summed weighted expert contribution into `out[0..n_embd]`.
#[allow(clippy::too_many_arguments)]
pub fn run_cpu_moe_m1(
    pool: &CpuThreadPool,
    scratch: &mut CpuMoeScratch,
    gate_up_type: GgmlType,
    down_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    situ_beta: f32,
    situ_linear_beta: f32,
    x_f32: &[f32],
    jobs: &[CpuExpertJob<'_>],
    out: &mut [f32],
) {
    debug_assert_eq!(x_f32.len(), n_embd);
    debug_assert_eq!(out.len(), n_embd);
    if jobs.is_empty() {
        out.fill(0.0);
        return;
    }

    let n_jobs = jobs.len();
    scratch.ensure_capacity(n_embd, n_ff, n_jobs);

    let nb_x = n_embd / 32;
    let nb_ff = n_ff / 32;
    let gu_row_bytes = gate_up_type.row_bytes(n_embd);
    let dn_row_bytes = down_type.row_bytes(n_ff);

    let x_q8_slice = &mut scratch.x_q8[..nb_x];
    quantize_f32_to_q8_1_cpu(x_f32, x_q8_slice);

    let x_q8_ref: &[CpuBlockQ8_1] = &scratch.x_q8[..nb_x];
    let inter_ptr = scratch.inter_f32.as_mut_ptr() as usize;

    // Step 1: Gate + Up + SiTU parallelized over (job * n_ff) rows
    let total_gu = n_jobs * n_ff;
    pool.parallel_for(total_gu, |start, end| {
        let inter_base = inter_ptr as *mut f32;
        for idx in start..end {
            let j = idx / n_ff;
            let r = idx % n_ff;
            let gu = jobs[j].gate_up;
            let g_off = r * gu_row_bytes;
            let u_off = (n_ff + r) * gu_row_bytes;
            let g = vec_dot_q_q8_1(gate_up_type, &gu[g_off..g_off + gu_row_bytes], x_q8_ref, Some(x_f32)) * jobs[j].gate_up_scale;
            let u = vec_dot_q_q8_1(gate_up_type, &gu[u_off..u_off + gu_row_bytes], x_q8_ref, Some(x_f32)) * jobs[j].gate_up_scale;
            unsafe {
                *inter_base.add(idx) = situ_activate(g, u, situ_beta, situ_linear_beta);
            }
        }
    });

    // Step 2: Quantize intermediates for each cold job
    for j in 0..n_jobs {
        let inter_row = &scratch.inter_f32[j * n_ff..(j + 1) * n_ff];
        let inter_q = &mut scratch.inter_q8[j * nb_ff..(j + 1) * nb_ff];
        quantize_f32_to_q8_1_cpu(inter_row, inter_q);
    }

    // Step 3: Down projection + weighted reduction over n_embd rows
    let inter_f32_ref = &scratch.inter_f32[..n_jobs * n_ff];
    let inter_q8_ref = &scratch.inter_q8[..n_jobs * nb_ff];
    let out_ptr = out.as_mut_ptr() as usize;
    pool.parallel_for(n_embd, |start, end| {
        let dst_base = out_ptr as *mut f32;
        for r in start..end {
            let row_off = r * dn_row_bytes;
            let mut sum = 0.0f32;
            for j in 0..n_jobs {
                let dn = &jobs[j].down[row_off..row_off + dn_row_bytes];
                let iq = &inter_q8_ref[j * nb_ff..(j + 1) * nb_ff];
                let ifr = &inter_f32_ref[j * n_ff..(j + 1) * n_ff];
                sum += jobs[j].weight * vec_dot_q_q8_1(down_type, dn, iq, Some(ifr));
            }
            unsafe {
                *dst_base.add(r) = sum;
            }
        }
    });
}

/// One routed cold token slot in a batched prefill step.
#[derive(Clone, Copy)]
pub struct CpuBatchSlot<'a> {
    pub token_idx: usize,
    pub gate_up: &'a [u8],
    pub down: &'a [u8],
    pub gate_up_scale: f32,
    pub weight: f32,
}

unsafe impl<'a> Send for CpuBatchSlot<'a> {}
unsafe impl<'a> Sync for CpuBatchSlot<'a> {}

/// Execute batched cold expert slots for `2 <= M < T_dma` tokens, accumulating into `out[M * n_embd]`.
#[allow(clippy::too_many_arguments)]
pub fn run_cpu_moe_batch(
    pool: &CpuThreadPool,
    scratch: &mut CpuMoeScratch,
    gate_up_type: GgmlType,
    down_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    situ_beta: f32,
    situ_linear_beta: f32,
    m_tokens: usize,
    x_f32: &[f32],
    slots: &[CpuBatchSlot<'_>],
    out: &mut [f32],
) {
    out[..m_tokens * n_embd].fill(0.0);
    if slots.is_empty() {
        return;
    }

    let n_slots = slots.len();
    let nb_x = n_embd / 32;
    let nb_ff = n_ff / 32;

    scratch.ensure_capacity(n_embd, n_ff, n_slots.max(m_tokens));
    // Re-purpose x_q8 sized to m_tokens * nb_x cleanly
    if scratch.x_q8.len() < m_tokens * nb_x {
        scratch.x_q8.resize(m_tokens * nb_x, CpuBlockQ8_1::default());
    }
    for m in 0..m_tokens {
        quantize_f32_to_q8_1_cpu(
            &x_f32[m * n_embd..(m + 1) * n_embd],
            &mut scratch.x_q8[m * nb_x..(m + 1) * nb_x],
        );
    }

    let gu_row_bytes = gate_up_type.row_bytes(n_embd);
    let dn_row_bytes = down_type.row_bytes(n_ff);
    let x_q8_ref = &scratch.x_q8[..m_tokens * nb_x];
    let inter_ptr = scratch.inter_f32.as_mut_ptr() as usize;

    pool.parallel_for(n_slots * n_ff, |start, end| {
        let inter_base = inter_ptr as *mut f32;
        for idx in start..end {
            let s = idx / n_ff;
            let r = idx % n_ff;
            let slot = &slots[s];
            let xm_q8 = &x_q8_ref[slot.token_idx * nb_x..(slot.token_idx + 1) * nb_x];
            let xm_f32 = &x_f32[slot.token_idx * n_embd..(slot.token_idx + 1) * n_embd];
            let g_off = r * gu_row_bytes;
            let u_off = (n_ff + r) * gu_row_bytes;
            let g = vec_dot_q_q8_1(gate_up_type, &slot.gate_up[g_off..g_off + gu_row_bytes], xm_q8, Some(xm_f32)) * slot.gate_up_scale;
            let u = vec_dot_q_q8_1(gate_up_type, &slot.gate_up[u_off..u_off + gu_row_bytes], xm_q8, Some(xm_f32)) * slot.gate_up_scale;
            unsafe {
                *inter_base.add(idx) = situ_activate(g, u, situ_beta, situ_linear_beta);
            }
        }
    });

    for s in 0..n_slots {
        quantize_f32_to_q8_1_cpu(
            &scratch.inter_f32[s * n_ff..(s + 1) * n_ff],
            &mut scratch.inter_q8[s * nb_ff..(s + 1) * nb_ff],
        );
    }

    let inter_f32_ref = &scratch.inter_f32[..n_slots * n_ff];
    let inter_q8_ref = &scratch.inter_q8[..n_slots * nb_ff];
    let acc_ptr = scratch.acc_out.as_mut_ptr() as usize;

    pool.parallel_for(n_slots * n_embd, |start, end| {
        let acc_base = acc_ptr as *mut f32;
        for idx in start..end {
            let s = idx / n_embd;
            let r = idx % n_embd;
            let row_off = r * dn_row_bytes;
            let dn = &slots[s].down[row_off..row_off + dn_row_bytes];
            let iq = &inter_q8_ref[s * nb_ff..(s + 1) * nb_ff];
            let ifr = &inter_f32_ref[s * n_ff..(s + 1) * n_ff];
            let v = slots[s].weight * vec_dot_q_q8_1(down_type, dn, iq, Some(ifr));
            unsafe {
                *acc_base.add(idx) = v;
            }
        }
    });

    for (s, slot) in slots.iter().enumerate() {
        let src = &scratch.acc_out[s * n_embd..(s + 1) * n_embd];
        let dst = &mut out[slot.token_idx * n_embd..(slot.token_idx + 1) * n_embd];
        for r in 0..n_embd {
            dst[r] += src[r];
        }
    }
}
