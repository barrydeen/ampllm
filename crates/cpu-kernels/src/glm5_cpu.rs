use crate::pool::CpuThreadPool;
use ampllm_gguf::GgmlType;
use std::ffi::c_void;

pub fn has_glm5_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vpopcntdq")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Size in bytes of one 256-element GGML `block_q8_K`.
pub const Q8K_BLOCK_BYTES: usize = 292;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct CpuBlockQ8K {
    pub raw: [u8; Q8K_BLOCK_BYTES],
}

impl Default for CpuBlockQ8K {
    fn default() -> Self {
        Self {
            raw: [0u8; Q8K_BLOCK_BYTES],
        }
    }
}

extern "C" {
    pub fn ampllm_cpu_quantize_q8_k(x: *const f32, out_q8k: *mut c_void, n: i32);

    pub fn ampllm_cpu_iq_fused_gu_rows(
        ggml_type: i32,
        gate_w: *const u8,
        up_w: *const u8,
        row_bytes: usize,
        n_in: i32,
        x_q8k: *const *const c_void,
        nt: i32,
        ff_out: *const *mut f32,
        r0: i32,
        r1: i32,
        swiglu_limit: f32,
    );

    pub fn ampllm_cpu_iq_down_rows(
        ggml_type: i32,
        down_w: *const u8,
        row_bytes: usize,
        n_in: i32,
        ff_q8k: *const *const c_void,
        nt: i32,
        out: *const *mut f32,
        r0: i32,
        r1: i32,
        weight_scale: f32,
        accumulate: i32,
    );
}

#[inline]
pub fn quantize_f32_to_q8_k_cpu(x: &[f32], out: &mut [CpuBlockQ8K]) {
    debug_assert_eq!(x.len() % 256, 0);
    debug_assert_eq!(out.len(), x.len() / 256);
    unsafe {
        ampllm_cpu_quantize_q8_k(x.as_ptr(), out.as_mut_ptr() as *mut c_void, x.len() as i32);
    }
}

/// One GLM-5.3 cold expert job (`[gate | up | down]` in host RAM).
#[derive(Clone, Copy)]
pub struct Glm5CpuExpertJob {
    pub gate_ptr: *const u8,
    pub up_ptr: *const u8,
    pub down_ptr: *const u8,
    pub weight: f32,
}

unsafe impl Send for Glm5CpuExpertJob {}
unsafe impl Sync for Glm5CpuExpertJob {}

/// Reusable scratch buffers for zero-allocation GLM-5.3 CPU cold MoE execution.
pub struct Glm5CpuScratch {
    pub x_q8k: Vec<CpuBlockQ8K>,
    pub ff_f32: Vec<f32>,
    pub ff_q8k: Vec<CpuBlockQ8K>,
    pub down_tmp: Vec<f32>,
}

impl Glm5CpuScratch {
    pub fn new(n_embd: usize, n_ff: usize, max_jobs: usize) -> Self {
        Self {
            x_q8k: vec![CpuBlockQ8K::default(); n_embd / 256],
            ff_f32: vec![0.0; max_jobs * n_ff],
            ff_q8k: vec![CpuBlockQ8K::default(); max_jobs * (n_ff / 256)],
            down_tmp: vec![0.0; max_jobs * n_embd],
        }
    }

    pub fn ensure_capacity(&mut self, n_embd: usize, n_ff: usize, n_tok_x: usize, n_slots: usize) {
        let nb_x = n_tok_x * (n_embd / 256);
        if self.x_q8k.len() < nb_x {
            self.x_q8k.resize(nb_x, CpuBlockQ8K::default());
        }
        let ff_len = n_slots * n_ff;
        if self.ff_f32.len() < ff_len {
            self.ff_f32.resize(ff_len, 0.0);
        }
        let fq_len = n_slots * (n_ff / 256);
        if self.ff_q8k.len() < fq_len {
            self.ff_q8k.resize(fq_len, CpuBlockQ8K::default());
        }
        let dt_len = n_slots * n_embd;
        if self.down_tmp.len() < dt_len {
            self.down_tmp.resize(dt_len, 0.0);
        }
    }
}

/// Evaluate 1..8 cold GLM-5.3 experts for a single decode token (`M=1`) across `pool`
/// using fused AVX-512 VNNI + VPOPCNTDQ kernels, writing the weighted sum into `out[0..n_embd]`.
#[allow(clippy::too_many_arguments)]
pub fn run_glm5_cpu_moe_m1(
    pool: &CpuThreadPool,
    scratch: &mut Glm5CpuScratch,
    gu_type: GgmlType,
    dn_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    swiglu_limit: f32,
    x_f32: &[f32],
    x_q8k_pre: Option<&[u8]>,
    jobs: &[Glm5CpuExpertJob],
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), n_embd);
    let n_jobs = jobs.len();
    if n_jobs == 0 {
        out.fill(0.0);
        return;
    }

    scratch.ensure_capacity(n_embd, n_ff, 1, n_jobs);
    let nb_x = n_embd / 256;
    let nb_ff = n_ff / 256;
    let gu_rb = gu_type.row_bytes(n_embd);
    let dn_rb = dn_type.row_bytes(n_ff);
    let gu_ty_i = gu_type as i32;
    let dn_ty_i = dn_type as i32;

    if let Some(pre) = x_q8k_pre {
        debug_assert!(pre.len() >= nb_x * Q8K_BLOCK_BYTES);
        unsafe {
            std::ptr::copy_nonoverlapping(
                pre.as_ptr(),
                scratch.x_q8k.as_mut_ptr() as *mut u8,
                nb_x * Q8K_BLOCK_BYTES,
            );
        }
    } else {
        debug_assert_eq!(x_f32.len(), n_embd);
        // Stage x_f32 (which may live in uncached/WC cudaHostAllocMapped memory) into cacheable RAM first
        scratch.down_tmp[..n_embd].copy_from_slice(x_f32);
        quantize_f32_to_q8_k_cpu(&scratch.down_tmp[..n_embd], &mut scratch.x_q8k[..nb_x]);
    }

    let x_q8k_ptr = scratch.x_q8k.as_ptr() as usize;
    let ff_f32_ptr = scratch.ff_f32.as_mut_ptr() as usize;
    let ff_q8k_bytes_ptr = scratch.ff_q8k.as_mut_ptr() as *mut u8 as usize;
    let jobs_ptr = jobs.as_ptr() as usize;

    // 1) Fused Gate + Up + Clamped SwiGLU + immediate L1 Q8_K quantization (1 barrier total!)
    if n_jobs >= 2 {
        let total_tiles = n_jobs * nb_ff;
        let next_tile = std::sync::atomic::AtomicUsize::new(0);
        let next_tile_u = &next_tile as *const std::sync::atomic::AtomicUsize as usize;
        pool.parallel_for(pool.num_threads(), |_, _| {
            let next = unsafe { &*(next_tile_u as *const std::sync::atomic::AtomicUsize) };
            let jobs_s = unsafe { std::slice::from_raw_parts(jobs_ptr as *const Glm5CpuExpertJob, n_jobs) };
            let act_ptr = x_q8k_ptr as *const c_void;
            loop {
                let tile = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if tile >= total_tiles {
                    break;
                }
                let j = tile / nb_ff;
                let b = tile % nb_ff;
                let r0 = b * 256;
                let r1 = r0 + 256;
                let mut ff_ptr = unsafe { (ff_f32_ptr as *mut f32).add(j * n_ff) };
                unsafe {
                    ampllm_cpu_iq_fused_gu_rows(
                        gu_ty_i,
                        jobs_s[j].gate_ptr,
                        jobs_s[j].up_ptr,
                        gu_rb,
                        n_embd as i32,
                        &act_ptr,
                        1,
                        &mut ff_ptr,
                        r0 as i32,
                        r1 as i32,
                        swiglu_limit,
                    );
                    ampllm_cpu_quantize_q8_k(
                        ff_ptr.add(r0),
                        (ff_q8k_bytes_ptr as *mut u8).add((j * nb_ff + b) * Q8K_BLOCK_BYTES) as *mut c_void,
                        256,
                    );
                }
            }
        });
    } else if nb_ff <= 64 {
        let half_tiles = nb_ff * 2;
        let next_tile = std::sync::atomic::AtomicUsize::new(0);
        let next_tile_u = &next_tile as *const std::sync::atomic::AtomicUsize as usize;
        let blk_done: [std::sync::atomic::AtomicU8; 64] = std::array::from_fn(|_| std::sync::atomic::AtomicU8::new(0));
        let blk_done_u = blk_done.as_ptr() as usize;
        pool.parallel_for(pool.num_threads(), |_, _| {
            let next = unsafe { &*(next_tile_u as *const std::sync::atomic::AtomicUsize) };
            let done_arr = unsafe { std::slice::from_raw_parts(blk_done_u as *const std::sync::atomic::AtomicU8, nb_ff) };
            let jobs_s = unsafe { std::slice::from_raw_parts(jobs_ptr as *const Glm5CpuExpertJob, n_jobs) };
            let act_ptr = x_q8k_ptr as *const c_void;
            loop {
                let tile = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if tile >= half_tiles {
                    break;
                }
                let b = tile >> 1;
                let h = tile & 1;
                let r0 = b * 256 + h * 128;
                let r1 = r0 + 128;
                let mut ff_ptr = ff_f32_ptr as *mut f32;
                unsafe {
                    ampllm_cpu_iq_fused_gu_rows(
                        gu_ty_i,
                        jobs_s[0].gate_ptr,
                        jobs_s[0].up_ptr,
                        gu_rb,
                        n_embd as i32,
                        &act_ptr,
                        1,
                        &mut ff_ptr,
                        r0 as i32,
                        r1 as i32,
                        swiglu_limit,
                    );
                }
                if done_arr[b].fetch_add(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                    unsafe {
                        ampllm_cpu_quantize_q8_k(
                            (ff_f32_ptr as *const f32).add(b * 256),
                            (ff_q8k_bytes_ptr as *mut u8).add(b * Q8K_BLOCK_BYTES) as *mut c_void,
                            256,
                        );
                    }
                }
            }
        });
    } else {
        pool.parallel_for(n_ff, |r0, r1| {
            let jobs_s = unsafe { std::slice::from_raw_parts(jobs_ptr as *const Glm5CpuExpertJob, n_jobs) };
            let act_ptr = x_q8k_ptr as *const c_void;
            let mut ff_ptr = ff_f32_ptr as *mut f32;
            unsafe {
                ampllm_cpu_iq_fused_gu_rows(
                    gu_ty_i,
                    jobs_s[0].gate_ptr,
                    jobs_s[0].up_ptr,
                    gu_rb,
                    n_embd as i32,
                    &act_ptr,
                    1,
                    &mut ff_ptr,
                    r0 as i32,
                    r1 as i32,
                    swiglu_limit,
                );
            }
        });
        unsafe {
            ampllm_cpu_quantize_q8_k(
                ff_f32_ptr as *const f32,
                ff_q8k_bytes_ptr as *mut c_void,
                n_ff as i32,
            );
        }
    }

    // 2) Down projection via dynamic work-stealing 64-row tiles into cacheable scratch.down_tmp,
    // followed by a single sequential copy to mapped WC `out`.
    let tmp_ptr = scratch.down_tmp.as_mut_ptr() as usize;
    let n_dn_tiles = (n_embd + 63) / 64;
    let dn_next = std::sync::atomic::AtomicUsize::new(0);
    let dn_next_u = &dn_next as *const std::sync::atomic::AtomicUsize as usize;
    pool.parallel_for(pool.num_threads(), |_, _| {
        let next = unsafe { &*(dn_next_u as *const std::sync::atomic::AtomicUsize) };
        let jobs_s = unsafe { std::slice::from_raw_parts(jobs_ptr as *const Glm5CpuExpertJob, n_jobs) };
        let mut out_p = tmp_ptr as *mut f32;
        loop {
            let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t >= n_dn_tiles {
                break;
            }
            let r0 = t * 64;
            let r1 = (r0 + 64).min(n_embd);
            for (j, job) in jobs_s.iter().enumerate() {
                let ff_p = unsafe { (ff_q8k_bytes_ptr as *const u8).add(j * nb_ff * Q8K_BLOCK_BYTES) as *const c_void };
                unsafe {
                    ampllm_cpu_iq_down_rows(
                        dn_ty_i,
                        job.down_ptr,
                        dn_rb,
                        n_ff as i32,
                        &ff_p,
                        1,
                        &mut out_p,
                        r0 as i32,
                        r1 as i32,
                        job.weight,
                        if j == 0 { 0 } else { 1 },
                    );
                }
            }
        }
    });
    out[..n_embd].copy_from_slice(&scratch.down_tmp[..n_embd]);
}

/// One grouped cold expert with multiple tokens (`nt` tokens sharing `(gate, up, down)`).
pub struct Glm5CpuGroupedExpert<'a> {
    pub gate_ptr: *const u8,
    pub up_ptr: *const u8,
    pub down_ptr: *const u8,
    pub token_indices: &'a [usize],
    pub token_weights: &'a [f32],
}

unsafe impl<'a> Send for Glm5CpuGroupedExpert<'a> {}
unsafe impl<'a> Sync for Glm5CpuGroupedExpert<'a> {}

/// Quantize all `m_tokens` rows of `x_f32` (`[m_tokens, n_embd]`) into `scratch.x_q8k` ONCE per layer.
pub fn quantize_glm5_cpu_x_q8k(
    pool: &CpuThreadPool,
    scratch: &mut Glm5CpuScratch,
    n_embd: usize,
    m_tokens: usize,
    x_f32: &[f32],
) {
    if m_tokens == 0 {
        return;
    }
    let nb_x = n_embd / 256;
    let need_x = m_tokens * nb_x;
    if scratch.x_q8k.len() < need_x {
        scratch.x_q8k.resize(need_x, CpuBlockQ8K::default());
    }
    if m_tokens >= 2 {
        let x_q8k_base = scratch.x_q8k.as_mut_ptr() as usize;
        let x_f32_base = x_f32.as_ptr() as usize;
        pool.parallel_for(m_tokens, |t0, t1| {
            for t in t0..t1 {
                unsafe {
                    ampllm_cpu_quantize_q8_k(
                        (x_f32_base as *const f32).add(t * n_embd),
                        (x_q8k_base as *mut CpuBlockQ8K).add(t * nb_x) as *mut c_void,
                        n_embd as i32,
                    );
                }
            }
        });
    } else {
        quantize_f32_to_q8_k_cpu(&x_f32[..n_embd], &mut scratch.x_q8k[..nb_x]);
    }
}

/// Evaluate grouped cold GLM-5.3 experts using pre-quantized `scratch.x_q8k[..m_tokens * nb_x]`.
/// Accumulates weighted results into `out[..m_tokens * n_embd]` in parallel across output columns.
#[allow(clippy::too_many_arguments)]
pub fn run_glm5_cpu_moe_grouped_prequant(
    pool: &CpuThreadPool,
    scratch: &mut Glm5CpuScratch,
    gu_type: GgmlType,
    dn_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    swiglu_limit: f32,
    m_tokens: usize,
    groups: &[Glm5CpuGroupedExpert<'_>],
    out: &mut [f32],
    zero_out: bool,
) {
    if zero_out {
        out[..m_tokens * n_embd].fill(0.0);
    }
    if groups.is_empty() {
        return;
    }

    let nb_x = n_embd / 256;
    let nb_ff = n_ff / 256;
    let gu_rb = gu_type.row_bytes(n_embd);
    let dn_rb = dn_type.row_bytes(n_ff);
    let gu_ty_i = gu_type as i32;
    let dn_ty_i = dn_type as i32;

    let total_slots: usize = groups.iter().map(|g| g.token_indices.len()).sum();
    if total_slots == 0 {
        return;
    }
    debug_assert!(scratch.x_q8k.len() >= m_tokens * nb_x);
    // Only grow intermediate/down buffers for this batch's total_slots (do not over-allocate x_q8k)
    scratch.ensure_capacity(n_embd, n_ff, m_tokens, total_slots);

    let mut group_slot_off = Vec::with_capacity(groups.len());
    let mut act_ptrs: Vec<*const c_void> = Vec::with_capacity(total_slots);
    let mut cur = 0usize;
    for g in groups {
        group_slot_off.push(cur);
        for &tok in g.token_indices {
            act_ptrs.push(unsafe { scratch.x_q8k.as_ptr().add(tok * nb_x) as *const c_void });
        }
        cur += g.token_indices.len();
    }

    // Build flat task list across groups in 64-row tiles for dynamic work-stealing balance across all cores!
    let tile_rows = 64usize;
    let mut gu_tasks: Vec<(usize, usize, usize)> = Vec::new();
    let mut dn_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        if g.token_indices.is_empty() {
            continue;
        }
        for r0 in (0..n_ff).step_by(tile_rows) {
            gu_tasks.push((gi, r0, (r0 + tile_rows).min(n_ff)));
        }
        for r0 in (0..n_embd).step_by(tile_rows) {
            dn_tasks.push((gi, r0, (r0 + tile_rows).min(n_embd)));
        }
    }

    let mut ff_ptrs: Vec<*mut f32> = Vec::with_capacity(total_slots);
    let mut fq_ptrs: Vec<*const c_void> = Vec::with_capacity(total_slots);
    let mut dn_ptrs: Vec<*mut f32> = Vec::with_capacity(total_slots);
    for s in 0..total_slots {
        ff_ptrs.push(unsafe { scratch.ff_f32.as_mut_ptr().add(s * n_ff) });
        fq_ptrs.push(unsafe { scratch.ff_q8k.as_ptr().add(s * nb_ff) as *const c_void });
        dn_ptrs.push(unsafe { scratch.down_tmp.as_mut_ptr().add(s * n_embd) });
    }

    let groups_ptr = groups.as_ptr() as usize;
    let g_cnt = groups.len();
    let g_off_ptr = group_slot_off.as_ptr() as usize;
    let act_ptrs_u = act_ptrs.as_ptr() as usize;
    let ff_ptrs_u = ff_ptrs.as_ptr() as usize;
    let gu_tasks_u = gu_tasks.as_ptr() as usize;
    let gu_next = std::sync::atomic::AtomicUsize::new(0);
    let gu_next_u = &gu_next as *const std::sync::atomic::AtomicUsize as usize;
    let n_gu_tasks = gu_tasks.len();

    pool.parallel_for(pool.num_threads(), |_w0, _w1| {
        let next = unsafe { &*(gu_next_u as *const std::sync::atomic::AtomicUsize) };
        let tasks = unsafe { std::slice::from_raw_parts(gu_tasks_u as *const (usize, usize, usize), n_gu_tasks) };
        let grp = unsafe { std::slice::from_raw_parts(groups_ptr as *const Glm5CpuGroupedExpert<'_>, g_cnt) };
        let offs = unsafe { std::slice::from_raw_parts(g_off_ptr as *const usize, g_cnt) };
        let ap = act_ptrs_u as *const *const c_void;
        let fp = ff_ptrs_u as *const *mut f32;
        loop {
            let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t >= n_gu_tasks {
                break;
            }
            let (gi, r0, r1) = tasks[t];
            let nt = grp[gi].token_indices.len();
            let off = offs[gi];
            unsafe {
                ampllm_cpu_iq_fused_gu_rows(
                    gu_ty_i,
                    grp[gi].gate_ptr,
                    grp[gi].up_ptr,
                    gu_rb,
                    n_embd as i32,
                    ap.add(off),
                    nt as i32,
                    fp.add(off),
                    r0 as i32,
                    r1 as i32,
                    swiglu_limit,
                );
            }
        }
    });

    // Quantize all intermediate slots in parallel
    let ff_f32_u = scratch.ff_f32.as_ptr() as usize;
    let ff_q8k_u = scratch.ff_q8k.as_mut_ptr() as usize;
    pool.parallel_for(total_slots, |s0, s1| {
        for s in s0..s1 {
            unsafe {
                ampllm_cpu_quantize_q8_k(
                    (ff_f32_u as *const f32).add(s * n_ff),
                    (ff_q8k_u as *mut CpuBlockQ8K).add(s * nb_ff) as *mut c_void,
                    n_ff as i32,
                );
            }
        }
    });

    let fq_ptrs_u = fq_ptrs.as_ptr() as usize;
    let dn_ptrs_u = dn_ptrs.as_ptr() as usize;
    let dn_tasks_u = dn_tasks.as_ptr() as usize;
    let dn_next = std::sync::atomic::AtomicUsize::new(0);
    let dn_next_u = &dn_next as *const std::sync::atomic::AtomicUsize as usize;
    let n_dn_tasks = dn_tasks.len();

    pool.parallel_for(pool.num_threads(), |_w0, _w1| {
        let next = unsafe { &*(dn_next_u as *const std::sync::atomic::AtomicUsize) };
        let tasks = unsafe { std::slice::from_raw_parts(dn_tasks_u as *const (usize, usize, usize), n_dn_tasks) };
        let grp = unsafe { std::slice::from_raw_parts(groups_ptr as *const Glm5CpuGroupedExpert<'_>, g_cnt) };
        let offs = unsafe { std::slice::from_raw_parts(g_off_ptr as *const usize, g_cnt) };
        let fp = fq_ptrs_u as *const *const c_void;
        let dp = dn_ptrs_u as *const *mut f32;
        loop {
            let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t >= n_dn_tasks {
                break;
            }
            let (gi, r0, r1) = tasks[t];
            let nt = grp[gi].token_indices.len();
            let off = offs[gi];
            unsafe {
                ampllm_cpu_iq_down_rows(
                    dn_ty_i,
                    grp[gi].down_ptr,
                    dn_rb,
                    n_ff as i32,
                    fp.add(off),
                    nt as i32,
                    dp.add(off),
                    r0 as i32,
                    r1 as i32,
                    1.0,
                    0,
                );
            }
        }
    });

    // Parallel weighted scatter-add into `out[tok * n_embd + r]` across disjoint output columns `r in r0..r1`
    let down_tmp_u = scratch.down_tmp.as_ptr() as usize;
    let out_u = out.as_mut_ptr() as usize;
    pool.parallel_for(n_embd, |r0, r1| {
        let grp = unsafe { std::slice::from_raw_parts(groups_ptr as *const Glm5CpuGroupedExpert<'_>, g_cnt) };
        let offs = unsafe { std::slice::from_raw_parts(g_off_ptr as *const usize, g_cnt) };
        let dt_p = down_tmp_u as *const f32;
        let out_p = out_u as *mut f32;
        for (gi, g) in grp.iter().enumerate() {
            let off = offs[gi];
            for (k, (&tok, &w)) in g.token_indices.iter().zip(g.token_weights.iter()).enumerate() {
                unsafe {
                    let src = dt_p.add((off + k) * n_embd);
                    let dst = out_p.add(tok * n_embd);
                    for r in r0..r1 {
                        *dst.add(r) += w * *src.add(r);
                    }
                }
            }
        }
    });
}

/// Evaluate grouped cold GLM-5.3 experts for batched prefill (`M >= 1` tokens),
/// loading each expert's weights ONCE for all tokens that routed to it.
/// Accumulates weighted results into `out[M * n_embd]` (`out` is zeroed or added depending on `zero_out`).
#[allow(clippy::too_many_arguments)]
pub fn run_glm5_cpu_moe_grouped(
    pool: &CpuThreadPool,
    scratch: &mut Glm5CpuScratch,
    gu_type: GgmlType,
    dn_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    swiglu_limit: f32,
    m_tokens: usize,
    x_f32: &[f32],
    groups: &[Glm5CpuGroupedExpert<'_>],
    out: &mut [f32],
    zero_out: bool,
) {
    if zero_out {
        out[..m_tokens * n_embd].fill(0.0);
    }
    if groups.is_empty() {
        return;
    }
    quantize_glm5_cpu_x_q8k(pool, scratch, n_embd, m_tokens, x_f32);
    run_glm5_cpu_moe_grouped_prequant(
        pool,
        scratch,
        gu_type,
        dn_type,
        n_embd,
        n_ff,
        swiglu_limit,
        m_tokens,
        groups,
        out,
        false,
    );
}
