use crate::glm5_cpu::{CpuBlockQ8K, Q8K_BLOCK_BYTES};
use crate::pool::CpuThreadPool;
use ampllm_gguf::GgmlType;
use std::ffi::c_void;

extern "C" {
    pub fn ampllm_cpu_mm3_quantize_q8_k(x: *const f32, out_q8k: *mut c_void, n: i32);

    pub fn ampllm_cpu_mm3_fused_gu_rows(
        gate_type: i32,
        up_type: i32,
        gate_w: *const u8,
        up_w: *const u8,
        gate_row_bytes: usize,
        up_row_bytes: usize,
        n_in: i32,
        x_q8k: *const *const c_void,
        nt: i32,
        ff_out: *const *mut f32,
        r0: i32,
        r1: i32,
        swiglu_alpha: f32,
        swiglu_limit: f32,
    );

    pub fn ampllm_cpu_mm3_down_rows(
        down_type: i32,
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

    pub fn ampllm_cpu_mm3_q6_k_row_to_f32(src: *const u8, k: i32, dst_f32: *mut f32);
}

/// Dequantize a single `Q6_K` embedding / weight row on CPU into `dst_f32`.
#[inline]
pub fn q6_k_row_to_f32_cpu(src: &[u8], dst: &mut [f32]) {
    debug_assert!(src.len() >= GgmlType::Q6K.row_bytes(dst.len()));
    unsafe {
        ampllm_cpu_mm3_q6_k_row_to_f32(src.as_ptr(), dst.len() as i32, dst.as_mut_ptr());
    }
}

/// One MiniMax-M3.1 cold expert job for `M=1` decode.
#[derive(Clone, Copy)]
pub struct Mm3CpuExpertJob {
    pub gate_ptr: *const u8,
    pub up_ptr: *const u8,
    pub down_ptr: *const u8,
    pub weight: f32,
}

unsafe impl Send for Mm3CpuExpertJob {}
unsafe impl Sync for Mm3CpuExpertJob {}

/// One grouped MiniMax-M3.1 cold expert with multiple tokens sharing `(gate, up, down)`.
pub struct Mm3CpuGroupedExpert<'a> {
    pub gate_ptr: *const u8,
    pub up_ptr: *const u8,
    pub down_ptr: *const u8,
    pub token_indices: &'a [usize],
    pub token_weights: &'a [f32],
}

unsafe impl<'a> Send for Mm3CpuGroupedExpert<'a> {}
unsafe impl<'a> Sync for Mm3CpuGroupedExpert<'a> {}

/// Reusable scratch buffers for zero-allocation MiniMax-M3.1 CPU cold MoE execution.
pub struct Mm3CpuScratch {
    pub x_q8k: Vec<CpuBlockQ8K>,
    pub ff_f32: Vec<f32>,
    pub ff_q8k: Vec<CpuBlockQ8K>,
    pub down_tmp: Vec<f32>,
}

impl Mm3CpuScratch {
    pub fn new(n_embd: usize, n_ff: usize, max_slots: usize) -> Self {
        let nb_x = n_embd.div_ceil(256);
        let nb_ff = n_ff.div_ceil(256);
        Self {
            x_q8k: vec![CpuBlockQ8K::default(); max_slots * nb_x],
            ff_f32: vec![0.0; max_slots * n_ff],
            ff_q8k: vec![CpuBlockQ8K::default(); max_slots * nb_ff],
            down_tmp: vec![0.0; max_slots * n_embd],
        }
    }

    pub fn ensure_capacity(&mut self, n_embd: usize, n_ff: usize, n_tok_x: usize, n_slots: usize) {
        let nb_x = n_tok_x * n_embd.div_ceil(256);
        if self.x_q8k.len() < nb_x {
            self.x_q8k.resize(nb_x, CpuBlockQ8K::default());
        }
        let ff_len = n_slots * n_ff;
        if self.ff_f32.len() < ff_len {
            self.ff_f32.resize(ff_len, 0.0);
        }
        let fq_len = n_slots * n_ff.div_ceil(256);
        if self.ff_q8k.len() < fq_len {
            self.ff_q8k.resize(fq_len, CpuBlockQ8K::default());
        }
        let dt_len = n_slots * n_embd;
        if self.down_tmp.len() < dt_len {
            self.down_tmp.resize(dt_len, 0.0);
        }
    }
}

/// Quantize `m_tokens` rows of `x_f32[m_tokens, n_embd]` into `scratch.x_q8k`.
pub fn quantize_mm3_cpu_x_q8k(
    pool: &CpuThreadPool,
    scratch: &mut Mm3CpuScratch,
    n_embd: usize,
    m_tokens: usize,
    x_f32: &[f32],
) {
    if m_tokens == 0 {
        return;
    }
    let nb_x = n_embd.div_ceil(256);
    let need = m_tokens * nb_x;
    if scratch.x_q8k.len() < need {
        scratch.x_q8k.resize(need, CpuBlockQ8K::default());
    }
    if m_tokens >= 2 {
        let x_q8k_base = scratch.x_q8k.as_mut_ptr() as usize;
        let x_f32_base = x_f32.as_ptr() as usize;
        pool.parallel_for(m_tokens, |t0, t1| {
            for t in t0..t1 {
                unsafe {
                    ampllm_cpu_mm3_quantize_q8_k(
                        (x_f32_base as *const f32).add(t * n_embd),
                        (x_q8k_base as *mut CpuBlockQ8K).add(t * nb_x) as *mut c_void,
                        n_embd as i32,
                    );
                }
            }
        });
    } else {
        unsafe {
            ampllm_cpu_mm3_quantize_q8_k(
                x_f32.as_ptr(),
                scratch.x_q8k.as_mut_ptr() as *mut c_void,
                n_embd as i32,
            );
        }
    }
}

/// Evaluate 1..8 cold MiniMax-M3.1 experts for a single token (`M=1`) with independent
/// `(gate_type, up_type, down_type)` and SwiGLU-OAI `(alpha=1.702, limit=7.0)`.
#[allow(clippy::too_many_arguments)]
pub fn run_mm3_cpu_moe_m1(
    pool: &CpuThreadPool,
    scratch: &mut Mm3CpuScratch,
    gate_type: GgmlType,
    up_type: GgmlType,
    down_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    x_f32: &[f32],
    x_q8k_pre: Option<&[u8]>,
    jobs: &[Mm3CpuExpertJob],
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), n_embd);
    let n_jobs = jobs.len();
    if n_jobs == 0 {
        out.fill(0.0);
        return;
    }

    scratch.ensure_capacity(n_embd, n_ff, 1, n_jobs);
    let nb_x = n_embd.div_ceil(256);
    let nb_ff = n_ff.div_ceil(256);
    let g_rb = gate_type.row_bytes(n_embd);
    let u_rb = up_type.row_bytes(n_embd);
    let d_rb = down_type.row_bytes(n_ff);
    let g_ty = gate_type as i32;
    let u_ty = up_type as i32;
    let d_ty = down_type as i32;

    let x_q8k_ptr: *const c_void = match x_q8k_pre {
        Some(pre) => {
            debug_assert!(pre.len() >= nb_x * Q8K_BLOCK_BYTES);
            pre.as_ptr() as *const c_void
        }
        None => {
            unsafe {
                ampllm_cpu_mm3_quantize_q8_k(
                    x_f32.as_ptr(),
                    scratch.x_q8k.as_mut_ptr() as *mut c_void,
                    n_embd as i32,
                );
            }
            scratch.x_q8k.as_ptr() as *const c_void
        }
    };

    let jobs_ptr = jobs.as_ptr() as usize;
    let ff_f32_ptr = scratch.ff_f32.as_mut_ptr() as usize;
    let ff_q8k_ptr = scratch.ff_q8k.as_mut_ptr() as usize;
    let x_ptr_u = x_q8k_ptr as usize;

    let use_split_128 = n_jobs == 1 && pool.num_threads() > nb_ff && (n_ff % 256 == 0);
    let gu_tile_rows = if use_split_128 { 128usize } else { 256usize };
    let mut gu_tasks = [(0usize, 0usize, 0usize); 256];
    let mut n_gu_tasks = 0usize;
    'outer_tasks: for ji in 0..n_jobs {
        for r0 in (0..n_ff).step_by(gu_tile_rows) {
            if n_gu_tasks < gu_tasks.len() {
                gu_tasks[n_gu_tasks] = (ji, r0, (r0 + gu_tile_rows).min(n_ff));
                n_gu_tasks += 1;
            } else {
                break 'outer_tasks;
            }
        }
    }
    let gu_tasks_u = gu_tasks.as_ptr() as usize;
    let gu_next = std::sync::atomic::AtomicUsize::new(0);
    let gu_next_u = &gu_next as *const std::sync::atomic::AtomicUsize as usize;

    pool.parallel_for(pool.num_threads(), |_w0, _w1| {
        let next = unsafe { &*(gu_next_u as *const std::sync::atomic::AtomicUsize) };
        let tasks = unsafe {
            std::slice::from_raw_parts(gu_tasks_u as *const (usize, usize, usize), n_gu_tasks)
        };
        let jobs_s = unsafe { std::slice::from_raw_parts(jobs_ptr as *const Mm3CpuExpertJob, n_jobs) };
        let xp = x_ptr_u as *const c_void;
        loop {
            let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t >= n_gu_tasks {
                break;
            }
            let (ji, r0, r1) = tasks[t];
            let job = &jobs_s[ji];
            unsafe {
                let mut ffp = (ff_f32_ptr as *mut f32).add(ji * n_ff);
                ampllm_cpu_mm3_fused_gu_rows(
                    g_ty,
                    u_ty,
                    job.gate_ptr,
                    job.up_ptr,
                    g_rb,
                    u_rb,
                    n_embd as i32,
                    &xp,
                    1,
                    &mut ffp,
                    r0 as i32,
                    r1 as i32,
                    swiglu_alpha,
                    swiglu_limit,
                );
                if !use_split_128 {
                    let b = r0 / 256;
                    ampllm_cpu_mm3_quantize_q8_k(
                        ffp.add(r0),
                        (ff_q8k_ptr as *mut u8).add((ji * nb_ff + b) * Q8K_BLOCK_BYTES)
                            as *mut c_void,
                        (r1 - r0) as i32,
                    );
                }
            }
        }
    });

    if use_split_128 {
        pool.parallel_for(nb_ff, |b0, b1| {
            for b in b0..b1 {
                unsafe {
                    let ffp = (ff_f32_ptr as *const f32).add(b * 256);
                    let q8p =
                        (ff_q8k_ptr as *mut u8).add(b * Q8K_BLOCK_BYTES) as *mut c_void;
                    ampllm_cpu_mm3_quantize_q8_k(ffp, q8p, 256);
                }
            }
        });
    }

    let n_dn_tiles = n_embd.div_ceil(64);
    let dn_next = std::sync::atomic::AtomicUsize::new(0);
    let dn_next_u = &dn_next as *const std::sync::atomic::AtomicUsize as usize;
    let out_ptr = out.as_mut_ptr() as usize;

    pool.parallel_for(pool.num_threads(), |_w0, _w1| {
        let next = unsafe { &*(dn_next_u as *const std::sync::atomic::AtomicUsize) };
        let jobs_s = unsafe { std::slice::from_raw_parts(jobs_ptr as *const Mm3CpuExpertJob, n_jobs) };
        let mut out_p = out_ptr as *mut f32;
        loop {
            let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t >= n_dn_tiles {
                break;
            }
            let r0 = t * 64;
            let r1 = (r0 + 64).min(n_embd);
            for (j, job) in jobs_s.iter().enumerate() {
                let ff_p =
                    unsafe { (ff_q8k_ptr as *const u8).add(j * nb_ff * Q8K_BLOCK_BYTES) as *const c_void };
                unsafe {
                    ampllm_cpu_mm3_down_rows(
                        d_ty,
                        job.down_ptr,
                        d_rb,
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
}

/// Evaluate grouped cold MiniMax-M3.1 experts using pre-quantized `scratch.x_q8k[..m_tokens * nb_x]`.
#[allow(clippy::too_many_arguments)]
pub fn run_mm3_cpu_moe_grouped_prequant(
    pool: &CpuThreadPool,
    scratch: &mut Mm3CpuScratch,
    gate_type: GgmlType,
    up_type: GgmlType,
    down_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    m_tokens: usize,
    groups: &[Mm3CpuGroupedExpert<'_>],
    out: &mut [f32],
    zero_out: bool,
) {
    if m_tokens == 0 {
        return;
    }
    if groups.is_empty() {
        if zero_out {
            out[..m_tokens * n_embd].fill(0.0);
        }
        return;
    }

    let nb_x = n_embd.div_ceil(256);
    let nb_ff = n_ff.div_ceil(256);
    let g_rb = gate_type.row_bytes(n_embd);
    let u_rb = up_type.row_bytes(n_embd);
    let d_rb = down_type.row_bytes(n_ff);
    let g_ty = gate_type as i32;
    let u_ty = up_type as i32;
    let d_ty = down_type as i32;

    let total_slots: usize = groups.iter().map(|g| g.token_indices.len()).sum();
    if total_slots == 0 {
        if zero_out {
            out[..m_tokens * n_embd].fill(0.0);
        }
        return;
    }
    debug_assert!(scratch.x_q8k.len() >= m_tokens * nb_x);
    scratch.ensure_capacity(n_embd, n_ff, m_tokens, total_slots);

    let max_nb_ff = (n_ff + 255) / 256;
    if m_tokens <= 8 && groups.len() <= 32 && total_slots <= 32 && groups.len() * max_nb_ff <= 768
    {
        let mut group_slot_off = [0usize; 32];
        let mut act_ptrs = [std::ptr::null::<c_void>(); 32];
        let mut ff_ptrs = [std::ptr::null_mut::<f32>(); 32];
        let mut fq_ptrs = [std::ptr::null::<c_void>(); 32];
        let mut dn_ptrs = [std::ptr::null_mut::<f32>(); 32];
        let fq_base_ptr = scratch.ff_q8k.as_mut_ptr() as *mut u8;
        let mut cur = 0usize;
        for (gi, g) in groups.iter().enumerate() {
            group_slot_off[gi] = cur;
            for &tok in g.token_indices {
                act_ptrs[cur] =
                    unsafe { scratch.x_q8k.as_ptr().add(tok * nb_x) as *const c_void };
                ff_ptrs[cur] = unsafe { scratch.ff_f32.as_mut_ptr().add(cur * n_ff) };
                fq_ptrs[cur] = unsafe {
                    (fq_base_ptr as *const u8).add(cur * nb_ff * Q8K_BLOCK_BYTES) as *const c_void
                };
                dn_ptrs[cur] = unsafe { scratch.down_tmp.as_mut_ptr().add(cur * n_embd) };
                cur += 1;
            }
        }

        let can_split_128 = groups.len() * nb_ff < pool.num_threads()
            && (n_ff % 256 == 0)
            && (groups.len() * (n_ff / 128) <= 768);
        let gu_tile_rows = if can_split_128 { 128usize } else { 256usize };
        let mut gu_tasks = [(0usize, 0usize, 0usize); 768];
        let mut n_gu_tasks = 0usize;
        for (gi, g) in groups.iter().enumerate() {
            if g.token_indices.is_empty() {
                continue;
            }
            for r0 in (0..n_ff).step_by(gu_tile_rows) {
                gu_tasks[n_gu_tasks] = (gi, r0, (r0 + gu_tile_rows).min(n_ff));
                n_gu_tasks += 1;
            }
        }

        let groups_ptr = groups.as_ptr() as usize;
        let g_cnt = groups.len();
        let g_off_ptr = group_slot_off.as_ptr() as usize;
        let act_ptrs_u = act_ptrs.as_ptr() as usize;
        let ff_ptrs_u = ff_ptrs.as_ptr() as usize;
        let fq_bytes_u = fq_base_ptr as usize;
        let gu_tasks_u = gu_tasks.as_ptr() as usize;
        let gu_next = std::sync::atomic::AtomicUsize::new(0);
        let gu_next_u = &gu_next as *const std::sync::atomic::AtomicUsize as usize;

        pool.parallel_for(pool.num_threads(), |_w0, _w1| {
            let next = unsafe { &*(gu_next_u as *const std::sync::atomic::AtomicUsize) };
            let tasks = unsafe {
                std::slice::from_raw_parts(gu_tasks_u as *const (usize, usize, usize), n_gu_tasks)
            };
            let grp = unsafe {
                std::slice::from_raw_parts(groups_ptr as *const Mm3CpuGroupedExpert<'_>, g_cnt)
            };
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
                let b = r0 / 256;
                let nr = (r1 - r0) as i32;
                unsafe {
                    ampllm_cpu_mm3_fused_gu_rows(
                        g_ty,
                        u_ty,
                        grp[gi].gate_ptr,
                        grp[gi].up_ptr,
                        g_rb,
                        u_rb,
                        n_embd as i32,
                        ap.add(off),
                        nt as i32,
                        fp.add(off),
                        r0 as i32,
                        r1 as i32,
                        swiglu_alpha,
                        swiglu_limit,
                    );
                    if !can_split_128 {
                        for s in 0..nt {
                            let ff_s = *fp.add(off + s);
                            ampllm_cpu_mm3_quantize_q8_k(
                                ff_s.add(r0),
                                (fq_bytes_u as *mut u8)
                                    .add(((off + s) * nb_ff + b) * Q8K_BLOCK_BYTES)
                                    as *mut c_void,
                                nr,
                            );
                        }
                    }
                }
            }
        });

        if can_split_128 {
            let tot_q_blocks = total_slots * nb_ff;
            pool.parallel_for(tot_q_blocks, |q0, q1| {
                let fp = ff_ptrs_u as *const *mut f32;
                for q in q0..q1 {
                    let slot = q / nb_ff;
                    let b = q - slot * nb_ff;
                    unsafe {
                        let ff_s = *fp.add(slot);
                        ampllm_cpu_mm3_quantize_q8_k(
                            ff_s.add(b * 256),
                            (fq_bytes_u as *mut u8).add(q * Q8K_BLOCK_BYTES) as *mut c_void,
                            256,
                        );
                    }
                }
            });
        }

        // Fused Down + parallel zero_out + weighted accumulate on disjoint 64-row tiles.
        let n_row_tiles = (n_embd + 63) / 64;
        let fq_ptrs_u = fq_ptrs.as_ptr() as usize;
        let dn_ptrs_u = dn_ptrs.as_ptr() as usize;
        let down_tmp_u = scratch.down_tmp.as_ptr() as usize;
        let out_u = out.as_mut_ptr() as usize;
        let dn_next = std::sync::atomic::AtomicUsize::new(0);
        let dn_next_u = &dn_next as *const std::sync::atomic::AtomicUsize as usize;
        let n_workers = pool.num_threads().min(n_row_tiles);

        pool.parallel_for(n_workers, |_w0, _w1| {
            let next = unsafe { &*(dn_next_u as *const std::sync::atomic::AtomicUsize) };
            let grp = unsafe {
                std::slice::from_raw_parts(groups_ptr as *const Mm3CpuGroupedExpert<'_>, g_cnt)
            };
            let offs = unsafe { std::slice::from_raw_parts(g_off_ptr as *const usize, g_cnt) };
            let fp = fq_ptrs_u as *const *const c_void;
            let dp = dn_ptrs_u as *const *mut f32;
            let dt_p = down_tmp_u as *const f32;
            let out_p = out_u as *mut f32;
            loop {
                let ti = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if ti >= n_row_tiles {
                    break;
                }
                let r0 = ti * 64;
                let r1 = (r0 + 64).min(n_embd);
                if zero_out {
                    for tok in 0..m_tokens {
                        unsafe {
                            let dst = out_p.add(tok * n_embd + r0);
                            std::ptr::write_bytes(dst, 0, r1 - r0);
                        }
                    }
                }
                for (gi, g) in grp.iter().enumerate() {
                    let nt = g.token_indices.len();
                    if nt == 0 {
                        continue;
                    }
                    let off = offs[gi];
                    if nt == 1 {
                        let tok = g.token_indices[0];
                        let w = g.token_weights[0];
                        unsafe {
                            let mut out_p1 = out_p.add(tok * n_embd);
                            ampllm_cpu_mm3_down_rows(
                                d_ty,
                                g.down_ptr,
                                d_rb,
                                n_ff as i32,
                                fp.add(off),
                                1,
                                &mut out_p1,
                                r0 as i32,
                                r1 as i32,
                                w,
                                1,
                            );
                        }
                    } else {
                        unsafe {
                            ampllm_cpu_mm3_down_rows(
                                d_ty,
                                g.down_ptr,
                                d_rb,
                                n_ff as i32,
                                fp.add(off),
                                nt as i32,
                                dp.add(off),
                                r0 as i32,
                                r1 as i32,
                                1.0,
                                0,
                            );
                            for k in 0..nt {
                                let tok = g.token_indices[k];
                                let w = g.token_weights[k];
                                let src = dt_p.add((off + k) * n_embd);
                                let dst = out_p.add(tok * n_embd);
                                for r in r0..r1 {
                                    *dst.add(r) += w * *src.add(r);
                                }
                            }
                        }
                    }
                }
            }
        });
        return;
    }

    if zero_out {
        out[..m_tokens * n_embd].fill(0.0);
    }

    let mut group_slot_off = Vec::with_capacity(groups.len());
    let mut act_ptrs: Vec<*const c_void> = Vec::with_capacity(total_slots);
    let mut ff_ptrs: Vec<*mut f32> = Vec::with_capacity(total_slots);
    let mut fq_ptrs: Vec<*const c_void> = Vec::with_capacity(total_slots);
    let mut dn_ptrs: Vec<*mut f32> = Vec::with_capacity(total_slots);
    let fq_base_ptr = scratch.ff_q8k.as_mut_ptr() as *mut u8;
    let mut cur = 0usize;
    for g in groups {
        group_slot_off.push(cur);
        for &tok in g.token_indices {
            act_ptrs.push(unsafe { scratch.x_q8k.as_ptr().add(tok * nb_x) as *const c_void });
            ff_ptrs.push(unsafe { scratch.ff_f32.as_mut_ptr().add(cur * n_ff) });
            fq_ptrs.push(unsafe {
                (fq_base_ptr as *const u8).add(cur * nb_ff * Q8K_BLOCK_BYTES) as *const c_void
            });
            dn_ptrs.push(unsafe { scratch.down_tmp.as_mut_ptr().add(cur * n_embd) });
            cur += 1;
        }
    }

    let use_split_128 = groups.len() * nb_ff < pool.num_threads() && (n_ff % 256 == 0);
    let gu_tile_rows = if use_split_128 { 128usize } else { 256usize };
    let mut gu_tasks: Vec<(usize, usize, usize)> = Vec::new();
    let mut dn_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        if g.token_indices.is_empty() {
            continue;
        }
        for r0 in (0..n_ff).step_by(gu_tile_rows) {
            gu_tasks.push((gi, r0, (r0 + gu_tile_rows).min(n_ff)));
        }
        for r0 in (0..n_embd).step_by(128) {
            dn_tasks.push((gi, r0, (r0 + 128).min(n_embd)));
        }
    }

    let groups_ptr = groups.as_ptr() as usize;
    let g_cnt = groups.len();
    let g_off_ptr = group_slot_off.as_ptr() as usize;
    let act_ptrs_u = act_ptrs.as_ptr() as usize;
    let ff_ptrs_u = ff_ptrs.as_ptr() as usize;
    let fq_bytes_u = fq_base_ptr as usize;
    let gu_tasks_u = gu_tasks.as_ptr() as usize;
    let n_gu_tasks = gu_tasks.len();
    let gu_next = std::sync::atomic::AtomicUsize::new(0);
    let gu_next_u = &gu_next as *const std::sync::atomic::AtomicUsize as usize;

    pool.parallel_for(pool.num_threads(), |_w0, _w1| {
        let next = unsafe { &*(gu_next_u as *const std::sync::atomic::AtomicUsize) };
        let tasks = unsafe {
            std::slice::from_raw_parts(gu_tasks_u as *const (usize, usize, usize), n_gu_tasks)
        };
        let grp = unsafe {
            std::slice::from_raw_parts(groups_ptr as *const Mm3CpuGroupedExpert<'_>, g_cnt)
        };
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
            let b = r0 / 256;
            let nr = (r1 - r0) as i32;
            unsafe {
                ampllm_cpu_mm3_fused_gu_rows(
                    g_ty,
                    u_ty,
                    grp[gi].gate_ptr,
                    grp[gi].up_ptr,
                    g_rb,
                    u_rb,
                    n_embd as i32,
                    ap.add(off),
                    nt as i32,
                    fp.add(off),
                    r0 as i32,
                    r1 as i32,
                    swiglu_alpha,
                    swiglu_limit,
                );
                if !use_split_128 {
                    for s in 0..nt {
                        let ff_s = *fp.add(off + s);
                        ampllm_cpu_mm3_quantize_q8_k(
                            ff_s.add(r0),
                            (fq_bytes_u as *mut u8).add(((off + s) * nb_ff + b) * Q8K_BLOCK_BYTES)
                                as *mut c_void,
                            nr,
                        );
                    }
                }
            }
        }
    });

    if use_split_128 {
        let tot_q_blocks = total_slots * nb_ff;
        pool.parallel_for(tot_q_blocks, |q0, q1| {
            let fp = ff_ptrs_u as *const *mut f32;
            for q in q0..q1 {
                let slot = q / nb_ff;
                let b = q - slot * nb_ff;
                unsafe {
                    let ff_s = *fp.add(slot);
                    ampllm_cpu_mm3_quantize_q8_k(
                        ff_s.add(b * 256),
                        (fq_bytes_u as *mut u8).add(q * Q8K_BLOCK_BYTES) as *mut c_void,
                        256,
                    );
                }
            }
        });
    }

    let fq_ptrs_u = fq_ptrs.as_ptr() as usize;
    let dn_ptrs_u = dn_ptrs.as_ptr() as usize;
    let dn_tasks_u = dn_tasks.as_ptr() as usize;
    let n_dn_tasks = dn_tasks.len();
    let dn_next = std::sync::atomic::AtomicUsize::new(0);
    let dn_next_u = &dn_next as *const std::sync::atomic::AtomicUsize as usize;

    pool.parallel_for(pool.num_threads(), |_w0, _w1| {
        let next = unsafe { &*(dn_next_u as *const std::sync::atomic::AtomicUsize) };
        let tasks = unsafe {
            std::slice::from_raw_parts(dn_tasks_u as *const (usize, usize, usize), n_dn_tasks)
        };
        let grp = unsafe {
            std::slice::from_raw_parts(groups_ptr as *const Mm3CpuGroupedExpert<'_>, g_cnt)
        };
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
                ampllm_cpu_mm3_down_rows(
                    d_ty,
                    grp[gi].down_ptr,
                    d_rb,
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

    let down_tmp_u = scratch.down_tmp.as_ptr() as usize;
    let out_u = out.as_mut_ptr() as usize;
    pool.parallel_for(n_embd, |r0, r1| {
        let grp = unsafe {
            std::slice::from_raw_parts(groups_ptr as *const Mm3CpuGroupedExpert<'_>, g_cnt)
        };
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

/// Evaluate grouped cold MiniMax-M3.1 experts for batched prefill / verify (`M >= 1`).
#[allow(clippy::too_many_arguments)]
pub fn run_mm3_cpu_moe_grouped(
    pool: &CpuThreadPool,
    scratch: &mut Mm3CpuScratch,
    gate_type: GgmlType,
    up_type: GgmlType,
    down_type: GgmlType,
    n_embd: usize,
    n_ff: usize,
    swiglu_alpha: f32,
    swiglu_limit: f32,
    m_tokens: usize,
    x_f32: &[f32],
    groups: &[Mm3CpuGroupedExpert<'_>],
    out: &mut [f32],
    zero_out: bool,
) {
    if zero_out && m_tokens > 0 {
        out[..m_tokens * n_embd].fill(0.0);
    }
    if groups.is_empty() || m_tokens == 0 {
        return;
    }
    quantize_mm3_cpu_x_q8k(pool, scratch, n_embd, m_tokens, x_f32);
    run_mm3_cpu_moe_grouped_prequant(
        pool,
        scratch,
        gate_type,
        up_type,
        down_type,
        n_embd,
        n_ff,
        swiglu_alpha,
        swiglu_limit,
        m_tokens,
        groups,
        out,
        false,
    );
}
