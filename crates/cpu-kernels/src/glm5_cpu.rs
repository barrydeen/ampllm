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
/// Size in bytes of 8x 32-element `Q8_1` sub-blocks (`block_q8_1_g8`).
pub const Q8_1_G8_BLOCK_BYTES: usize = 352;

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

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct CpuBlockQ8_1G8 {
    pub d: [f32; 8],
    pub s: [f32; 8],
    pub bsum: [i32; 8],
    pub qs: [i8; 256],
}

const _: () = assert!(std::mem::size_of::<CpuBlockQ8_1G8>() == Q8_1_G8_BLOCK_BYTES);

impl Default for CpuBlockQ8_1G8 {
    fn default() -> Self {
        Self {
            d: [0.0; 8],
            s: [0.0; 8],
            bsum: [0; 8],
            qs: [0; 256],
        }
    }
}

extern "C" {
    pub fn ampllm_cpu_quantize_q8_k(x: *const f32, out_q8k: *mut c_void, n: i32);
    pub fn ampllm_cpu_quantize_q8_1_g8(x: *const f32, out_g8: *mut c_void, n: i32);

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

    pub fn ampllm_cpu_hy4_fused_gu_rows(
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

    pub fn ampllm_cpu_hy4_fused_gu_rows_q8_1(
        ggml_type: i32,
        gate_w: *const u8,
        up_w: *const u8,
        row_bytes: usize,
        n_in: i32,
        x_q8_1: *const *const c_void,
        nt: i32,
        ff_out: *const *mut f32,
        r0: i32,
        r1: i32,
        swiglu_limit: f32,
    );

    pub fn ampllm_cpu_hy4_down_rows(
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

    pub fn ampllm_cpu_hy4_down_rows_q8_1(
        ggml_type: i32,
        down_w: *const u8,
        row_bytes: usize,
        n_in: i32,
        ff_q8_1: *const *const c_void,
        nt: i32,
        out: *const *mut f32,
        r0: i32,
        r1: i32,
        weight_scale: f32,
        accumulate: i32,
    );

    pub fn ampllm_cpu_bf16_to_q8_0(
        src_bf16: *const u16,
        src_f32: *const f32,
        rows: usize,
        k: i32,
        out_q8_0: *mut u8,
    );

    pub fn ampllm_cpu_tok_row_to_f32(
        ggml_type: i32,
        src: *const u8,
        k: i32,
        dst_f32: *mut f32,
    );
}

#[inline(always)]
unsafe fn dispatch_fused_gu_rows(
    use_q8_1: bool,
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
) {
    if use_q8_1 && matches!(ggml_type, 12 | 13) {
        ampllm_cpu_hy4_fused_gu_rows_q8_1(
            ggml_type,
            gate_w,
            up_w,
            row_bytes,
            n_in,
            x_q8k,
            nt,
            ff_out,
            r0,
            r1,
            swiglu_limit,
        );
    } else if matches!(ggml_type, 7 | 8 | 12 | 13 | 14 | 23) {
        ampllm_cpu_hy4_fused_gu_rows(
            ggml_type,
            gate_w,
            up_w,
            row_bytes,
            n_in,
            x_q8k,
            nt,
            ff_out,
            r0,
            r1,
            swiglu_limit,
        );
    } else {
        ampllm_cpu_iq_fused_gu_rows(
            ggml_type,
            gate_w,
            up_w,
            row_bytes,
            n_in,
            x_q8k,
            nt,
            ff_out,
            r0,
            r1,
            swiglu_limit,
        );
    }
}

#[inline(always)]
unsafe fn quantize_ff_blk(use_q8_1: bool, src: *const f32, dst: *mut c_void, n: i32) {
    if use_q8_1 {
        ampllm_cpu_quantize_q8_1_g8(src, dst, n);
    } else {
        ampllm_cpu_quantize_q8_k(src, dst, n);
    }
}

#[inline(always)]
unsafe fn dispatch_down_rows(
    ggml_type: i32,
    down_w: *const u8,
    row_bytes: usize,
    n_in: i32,
    ff_q8: *const *const c_void,
    nt: i32,
    out: *const *mut f32,
    r0: i32,
    r1: i32,
    weight_scale: f32,
    accumulate: i32,
) {
    if matches!(ggml_type, 7 | 8) {
        ampllm_cpu_hy4_down_rows_q8_1(
            ggml_type,
            down_w,
            row_bytes,
            n_in,
            ff_q8,
            nt,
            out,
            r0,
            r1,
            weight_scale,
            accumulate,
        );
    } else if matches!(ggml_type, 12 | 13 | 14 | 23) {
        ampllm_cpu_hy4_down_rows(
            ggml_type,
            down_w,
            row_bytes,
            n_in,
            ff_q8,
            nt,
            out,
            r0,
            r1,
            weight_scale,
            accumulate,
        );
    } else {
        ampllm_cpu_iq_down_rows(
            ggml_type,
            down_w,
            row_bytes,
            n_in,
            ff_q8,
            nt,
            out,
            r0,
            r1,
            weight_scale,
            accumulate,
        );
    }
}

#[inline]
pub fn bf16_to_q8_0_cpu(src_bf16: &[u16], rows: usize, k: usize, out: &mut [u8]) {
    debug_assert_eq!(k % 32, 0);
    debug_assert!(src_bf16.len() >= rows * k);
    debug_assert!(out.len() >= rows * (k / 32) * 34);
    unsafe {
        ampllm_cpu_bf16_to_q8_0(
            src_bf16.as_ptr(),
            std::ptr::null(),
            rows,
            k as i32,
            out.as_mut_ptr(),
        );
    }
}

#[inline]
pub fn f32_to_q8_0_cpu(src_f32: &[f32], rows: usize, k: usize, out: &mut [u8]) {
    debug_assert_eq!(k % 32, 0);
    debug_assert!(src_f32.len() >= rows * k);
    debug_assert!(out.len() >= rows * (k / 32) * 34);
    unsafe {
        ampllm_cpu_bf16_to_q8_0(
            std::ptr::null(),
            src_f32.as_ptr(),
            rows,
            k as i32,
            out.as_mut_ptr(),
        );
    }
}

#[inline]
pub fn tok_row_to_f32_cpu(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    debug_assert!(src.len() >= ty.row_bytes(dst.len()));
    unsafe {
        ampllm_cpu_tok_row_to_f32(ty as i32, src.as_ptr(), dst.len() as i32, dst.as_mut_ptr());
    }
}

#[inline]
pub fn quantize_f32_to_q8_k_cpu(x: &[f32], out: &mut [CpuBlockQ8K]) {
    debug_assert_eq!(x.len() % 32, 0);
    debug_assert_eq!(out.len(), x.len().div_ceil(256));
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
    pub x_q8_1: Vec<CpuBlockQ8_1G8>,
    pub x_prequant_is_q8_1: bool,
    pub ff_f32: Vec<f32>,
    pub ff_q8k: Vec<CpuBlockQ8K>,
    pub ff_q8_1: Vec<CpuBlockQ8_1G8>,
    pub down_tmp: Vec<f32>,
}

impl Glm5CpuScratch {
    pub fn new(n_embd: usize, n_ff: usize, max_jobs: usize) -> Self {
        let nb_ff = n_ff.div_ceil(256);
        Self {
            x_q8k: vec![CpuBlockQ8K::default(); n_embd.div_ceil(256)],
            x_q8_1: vec![CpuBlockQ8_1G8::default(); n_embd.div_ceil(256)],
            x_prequant_is_q8_1: false,
            ff_f32: vec![0.0; max_jobs * n_ff],
            ff_q8k: vec![CpuBlockQ8K::default(); max_jobs * nb_ff],
            ff_q8_1: vec![CpuBlockQ8_1G8::default(); max_jobs * nb_ff],
            down_tmp: vec![0.0; max_jobs * n_embd],
        }
    }

    pub fn ensure_capacity(&mut self, n_embd: usize, n_ff: usize, n_tok_x: usize, n_slots: usize) {
        let nb_x = n_tok_x * n_embd.div_ceil(256);
        if self.x_q8k.len() < nb_x {
            self.x_q8k.resize(nb_x, CpuBlockQ8K::default());
        }
        if self.x_q8_1.len() < nb_x {
            self.x_q8_1.resize(nb_x, CpuBlockQ8_1G8::default());
        }
        let ff_len = n_slots * n_ff;
        if self.ff_f32.len() < ff_len {
            self.ff_f32.resize(ff_len, 0.0);
        }
        let fq_len = n_slots * n_ff.div_ceil(256);
        if self.ff_q8k.len() < fq_len {
            self.ff_q8k.resize(fq_len, CpuBlockQ8K::default());
        }
        if self.ff_q8_1.len() < fq_len {
            self.ff_q8_1.resize(fq_len, CpuBlockQ8_1G8::default());
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
    let nb_x = n_embd.div_ceil(256);
    let nb_ff = n_ff.div_ceil(256);
    let gu_rb = gu_type.row_bytes(n_embd);
    let dn_rb = dn_type.row_bytes(n_ff);
    let gu_ty_i = gu_type as i32;
    let dn_ty_i = dn_type as i32;
    let use_q8_1 = matches!(dn_type, GgmlType::Q5_1 | GgmlType::Q8_0);
    let fq_blk_bytes = if use_q8_1 {
        Q8_1_G8_BLOCK_BYTES
    } else {
        Q8K_BLOCK_BYTES
    };
    let use_x_q8_1 = x_q8k_pre.is_none() && matches!(gu_type, GgmlType::Q4K | GgmlType::Q5K);

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
        if use_x_q8_1 {
            unsafe {
                ampllm_cpu_quantize_q8_1_g8(
                    scratch.down_tmp.as_ptr(),
                    scratch.x_q8_1.as_mut_ptr() as *mut c_void,
                    n_embd as i32,
                );
            }
        } else {
            quantize_f32_to_q8_k_cpu(&scratch.down_tmp[..n_embd], &mut scratch.x_q8k[..nb_x]);
        }
    }

    let x_q8k_ptr = if use_x_q8_1 {
        scratch.x_q8_1.as_ptr() as usize
    } else {
        scratch.x_q8k.as_ptr() as usize
    };
    let ff_f32_ptr = scratch.ff_f32.as_mut_ptr() as usize;
    let ff_q8k_bytes_ptr = if use_q8_1 {
        scratch.ff_q8_1.as_mut_ptr() as *mut u8 as usize
    } else {
        scratch.ff_q8k.as_mut_ptr() as *mut u8 as usize
    };
    let jobs_ptr = jobs.as_ptr() as usize;

    // 1) Fused Gate + Up + Clamped SwiGLU + immediate L1 quantization (1 barrier total!)
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
                let r1 = (r0 + 256).min(n_ff);
                let blk_n = (r1 - r0) as i32;
                let mut ff_ptr = unsafe { (ff_f32_ptr as *mut f32).add(j * n_ff) };
                unsafe {
                    dispatch_fused_gu_rows(
                        use_x_q8_1,
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
                    quantize_ff_blk(
                        use_q8_1,
                        ff_ptr.add(r0),
                        (ff_q8k_bytes_ptr as *mut u8).add((j * nb_ff + b) * fq_blk_bytes) as *mut c_void,
                        blk_n,
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
                let r1 = (r0 + 128).min(n_ff);
                if r0 < r1 {
                    let mut ff_ptr = ff_f32_ptr as *mut f32;
                    unsafe {
                        dispatch_fused_gu_rows(
                            use_x_q8_1,
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
                }
                if done_arr[b].fetch_add(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                    let b0 = b * 256;
                    let blk_n = (n_ff - b0).min(256) as i32;
                    unsafe {
                        quantize_ff_blk(
                            use_q8_1,
                            (ff_f32_ptr as *const f32).add(b0),
                            (ff_q8k_bytes_ptr as *mut u8).add(b * fq_blk_bytes) as *mut c_void,
                            blk_n,
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
                dispatch_fused_gu_rows(
                    use_x_q8_1,
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
            quantize_ff_blk(
                use_q8_1,
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
                let ff_p = unsafe { (ff_q8k_bytes_ptr as *const u8).add(j * nb_ff * fq_blk_bytes) as *const c_void };
                unsafe {
                    dispatch_down_rows(
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
    scratch.x_prequant_is_q8_1 = false;
    if m_tokens == 0 {
        return;
    }
    let nb_x = n_embd.div_ceil(256);
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

/// Quantize all `m_tokens` rows of `x_f32` (`[m_tokens, n_embd]`) into `scratch.x_q8_1` ONCE per layer.
pub fn quantize_glm5_cpu_x_q8_1(
    pool: &CpuThreadPool,
    scratch: &mut Glm5CpuScratch,
    n_embd: usize,
    m_tokens: usize,
    x_f32: &[f32],
) {
    scratch.x_prequant_is_q8_1 = true;
    if m_tokens == 0 {
        return;
    }
    let nb_x = n_embd.div_ceil(256);
    let need_x = m_tokens * nb_x;
    if scratch.x_q8_1.len() < need_x {
        scratch.x_q8_1.resize(need_x, CpuBlockQ8_1G8::default());
    }
    if m_tokens >= 2 {
        let x_q8_base = scratch.x_q8_1.as_mut_ptr() as usize;
        let x_f32_base = x_f32.as_ptr() as usize;
        pool.parallel_for(m_tokens, |t0, t1| {
            for t in t0..t1 {
                unsafe {
                    ampllm_cpu_quantize_q8_1_g8(
                        (x_f32_base as *const f32).add(t * n_embd),
                        (x_q8_base as *mut CpuBlockQ8_1G8).add(t * nb_x) as *mut c_void,
                        n_embd as i32,
                    );
                }
            }
        });
    } else {
        unsafe {
            ampllm_cpu_quantize_q8_1_g8(
                x_f32.as_ptr(),
                scratch.x_q8_1.as_mut_ptr() as *mut c_void,
                n_embd as i32,
            );
        }
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

    let nb_x = n_embd.div_ceil(256);
    let nb_ff = n_ff.div_ceil(256);
    let gu_rb = gu_type.row_bytes(n_embd);
    let dn_rb = dn_type.row_bytes(n_ff);
    let gu_ty_i = gu_type as i32;
    let dn_ty_i = dn_type as i32;
    let use_q8_1 = matches!(dn_type, GgmlType::Q5_1 | GgmlType::Q8_0);
    let fq_blk_bytes = if use_q8_1 {
        Q8_1_G8_BLOCK_BYTES
    } else {
        Q8K_BLOCK_BYTES
    };

    let use_x_q8_1 = scratch.x_prequant_is_q8_1 && matches!(gu_type, GgmlType::Q4K | GgmlType::Q5K);
    let total_slots: usize = groups.iter().map(|g| g.token_indices.len()).sum();
    if total_slots == 0 {
        return;
    }
    if use_x_q8_1 {
        debug_assert!(scratch.x_q8_1.len() >= m_tokens * nb_x);
    } else {
        debug_assert!(scratch.x_q8k.len() >= m_tokens * nb_x);
    }
    // Only grow intermediate/down buffers for this batch's total_slots (do not over-allocate x_q8k)
    scratch.ensure_capacity(n_embd, n_ff, m_tokens, total_slots);

    let mut group_slot_off = Vec::with_capacity(groups.len());
    let mut act_ptrs: Vec<*const c_void> = Vec::with_capacity(total_slots);
    let mut cur = 0usize;
    for g in groups {
        group_slot_off.push(cur);
        for &tok in g.token_indices {
            if use_x_q8_1 {
                act_ptrs.push(unsafe { scratch.x_q8_1.as_ptr().add(tok * nb_x) as *const c_void });
            } else {
                act_ptrs.push(unsafe { scratch.x_q8k.as_ptr().add(tok * nb_x) as *const c_void });
            }
        }
        cur += g.token_indices.len();
    }

    // Build flat task list across groups:
    // - Gate+Up uses 256-row tiles (1 full Q8_K/Q8_1 block) so each worker immediately quantizes in L1 cache!
    // - Down uses 128-row tiles to halve activation-prep overhead while keeping >=12 tasks per expert.
    let gu_tile_rows = 256usize;
    let dn_tile_rows = 128usize;
    let can_fuse_gu_q8k = true;
    let mut gu_tasks: Vec<(usize, usize, usize)> = Vec::new();
    let mut dn_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        if g.token_indices.is_empty() {
            continue;
        }
        for r0 in (0..n_ff).step_by(gu_tile_rows) {
            gu_tasks.push((gi, r0, (r0 + gu_tile_rows).min(n_ff)));
        }
        for r0 in (0..n_embd).step_by(dn_tile_rows) {
            dn_tasks.push((gi, r0, (r0 + dn_tile_rows).min(n_embd)));
        }
    }

    let fq_base_ptr = if use_q8_1 {
        scratch.ff_q8_1.as_mut_ptr() as *mut u8
    } else {
        scratch.ff_q8k.as_mut_ptr() as *mut u8
    };
    let mut ff_ptrs: Vec<*mut f32> = Vec::with_capacity(total_slots);
    let mut fq_ptrs: Vec<*const c_void> = Vec::with_capacity(total_slots);
    let mut dn_ptrs: Vec<*mut f32> = Vec::with_capacity(total_slots);
    for s in 0..total_slots {
        ff_ptrs.push(unsafe { scratch.ff_f32.as_mut_ptr().add(s * n_ff) });
        fq_ptrs.push(unsafe {
            (fq_base_ptr as *const u8).add(s * nb_ff * fq_blk_bytes) as *const c_void
        });
        dn_ptrs.push(unsafe { scratch.down_tmp.as_mut_ptr().add(s * n_embd) });
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
                dispatch_fused_gu_rows(
                    use_x_q8_1,
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
                if can_fuse_gu_q8k {
                    let b = r0 / 256;
                    let nr = (r1 - r0) as i32;
                    for s in 0..nt {
                        let ff_s = *fp.add(off + s);
                        quantize_ff_blk(
                            use_q8_1,
                            ff_s.add(r0),
                            (fq_bytes_u as *mut u8).add(((off + s) * nb_ff + b) * fq_blk_bytes)
                                as *mut c_void,
                            nr,
                        );
                    }
                }
            }
        }
    });

    if !can_fuse_gu_q8k {
        let ff_f32_u = scratch.ff_f32.as_ptr() as usize;
        pool.parallel_for(total_slots, |s0, s1| {
            for s in s0..s1 {
                unsafe {
                    quantize_ff_blk(
                        use_q8_1,
                        (ff_f32_u as *const f32).add(s * n_ff),
                        (fq_bytes_u as *mut u8).add(s * nb_ff * fq_blk_bytes) as *mut c_void,
                        n_ff as i32,
                    );
                }
            }
        });
    }

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
                dispatch_down_rows(
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
    if matches!(gu_type, GgmlType::Q4K | GgmlType::Q5K) {
        quantize_glm5_cpu_x_q8_1(pool, scratch, n_embd, m_tokens, x_f32);
    } else {
        quantize_glm5_cpu_x_q8k(pool, scratch, n_embd, m_tokens, x_f32);
    }
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

/// Zero-allocation 2-barrier grouped cold MoE path optimized for small verify batches (`m_tokens <= 8`).
/// Replaces 4 cross-CCD barriers + per-layer Vec allocs + `down_tmp` RAM round-trip with 2 barriers
/// and L1-local fused down+accumulate.
#[allow(clippy::too_many_arguments)]
pub fn run_glm5_cpu_moe_grouped_small(
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
    if m_tokens == 0 || groups.is_empty() {
        if zero_out && m_tokens > 0 {
            out[..m_tokens * n_embd].fill(0.0);
        }
        return;
    }
    let q8_1_dn = matches!(dn_type, GgmlType::Q5_1 | GgmlType::Q8_0);
    if m_tokens > 8
        || groups.len() > 80
        || n_ff == 0
        || n_ff > 1024
        || (!q8_1_dn && (n_ff % 256) != 0)
        || (q8_1_dn && (n_ff % 32) != 0)
    {
        run_glm5_cpu_moe_grouped(
            pool,
            scratch,
            gu_type,
            dn_type,
            n_embd,
            n_ff,
            swiglu_limit,
            m_tokens,
            x_f32,
            groups,
            out,
            zero_out,
        );
        return;
    }

    let mut total_slots = 0usize;
    for g in groups {
        let nt = g.token_indices.len();
        if nt > 8 {
            run_glm5_cpu_moe_grouped(
                pool,
                scratch,
                gu_type,
                dn_type,
                n_embd,
                n_ff,
                swiglu_limit,
                m_tokens,
                x_f32,
                groups,
                out,
                zero_out,
            );
            return;
        }
        total_slots += nt;
    }
    if total_slots == 0 {
        if zero_out {
            out[..m_tokens * n_embd].fill(0.0);
        }
        return;
    }
    if total_slots > 80 {
        run_glm5_cpu_moe_grouped(
            pool,
            scratch,
            gu_type,
            dn_type,
            n_embd,
            n_ff,
            swiglu_limit,
            m_tokens,
            x_f32,
            groups,
            out,
            zero_out,
        );
        return;
    }

    let nb_x = n_embd.div_ceil(256);
    let nb_ff = n_ff.div_ceil(256);
    if groups.len() * nb_ff > 512 {
        run_glm5_cpu_moe_grouped(
            pool,
            scratch,
            gu_type,
            dn_type,
            n_embd,
            n_ff,
            swiglu_limit,
            m_tokens,
            x_f32,
            groups,
            out,
            zero_out,
        );
        return;
    }

    let n_threads = pool.num_threads().max(1);
    let min_slots = (n_threads * 8).max(total_slots);
    scratch.ensure_capacity(n_embd, n_ff, m_tokens, min_slots);

    let gu_rb = gu_type.row_bytes(n_embd);
    let dn_rb = dn_type.row_bytes(n_ff);
    let gu_ty_i = gu_type as i32;
    let dn_ty_i = dn_type as i32;
    let use_q8_1 = matches!(dn_type, GgmlType::Q5_1 | GgmlType::Q8_0);
    let use_x_q8_1 = matches!(gu_type, GgmlType::Q4K | GgmlType::Q5K);
    let fq_blk_bytes = if use_q8_1 {
        Q8_1_G8_BLOCK_BYTES
    } else {
        Q8K_BLOCK_BYTES
    };

    // Step 0: Quantize only active tokens (0 barriers)
    let mut tok_mask = 0u8;
    for g in groups {
        for &tok in g.token_indices {
            if tok < m_tokens {
                tok_mask |= 1u8 << tok;
            }
        }
    }
    for tok in 0..m_tokens {
        if (tok_mask & (1u8 << tok)) != 0 {
            if use_x_q8_1 {
                unsafe {
                    ampllm_cpu_quantize_q8_1_g8(
                        x_f32[tok * n_embd..(tok + 1) * n_embd].as_ptr(),
                        scratch.x_q8_1[tok * nb_x..].as_mut_ptr() as *mut c_void,
                        n_embd as i32,
                    );
                }
            } else {
                quantize_f32_to_q8_k_cpu(
                    &x_f32[tok * n_embd..(tok + 1) * n_embd],
                    &mut scratch.x_q8k[tok * nb_x..(tok + 1) * nb_x],
                );
            }
        }
    }

    let mut group_slot_off = [0usize; 80];
    let mut act_ptrs = [std::ptr::null::<c_void>(); 80];
    let mut ff_ptrs = [std::ptr::null_mut::<f32>(); 80];
    let mut fq_ptrs = [std::ptr::null::<c_void>(); 80];
    let mut cur = 0usize;
    let x_q8k_ptr = scratch.x_q8k.as_ptr() as *const u8;
    let x_q8_1_ptr = scratch.x_q8_1.as_ptr() as *const u8;
    let ff_f32_ptr = scratch.ff_f32.as_mut_ptr();
    let fq_base_ptr = if use_q8_1 {
        scratch.ff_q8_1.as_mut_ptr() as *mut u8
    } else {
        scratch.ff_q8k.as_mut_ptr() as *mut u8
    };
    let fq_bytes_ptr = fq_base_ptr as *const u8;
    for (gi, g) in groups.iter().enumerate() {
        group_slot_off[gi] = cur;
        for &tok in g.token_indices {
            act_ptrs[cur] = if use_x_q8_1 {
                unsafe { x_q8_1_ptr.add(tok * nb_x * Q8_1_G8_BLOCK_BYTES) as *const c_void }
            } else {
                unsafe { x_q8k_ptr.add(tok * nb_x * Q8K_BLOCK_BYTES) as *const c_void }
            };
            ff_ptrs[cur] = unsafe { ff_f32_ptr.add(cur * n_ff) };
            fq_ptrs[cur] = unsafe { fq_bytes_ptr.add(cur * nb_ff * fq_blk_bytes) as *const c_void };
            cur += 1;
        }
    }

    // Step 1 (Barrier #1): Gate+Up + SwiGLU + inline Q8_K/Q8_1 quantize (128-row half-tiles when work < threads)
    let split_128 = groups.len() * nb_ff < n_threads;
    let gu_tile_rows = if split_128 { 128usize } else { 256usize };
    let mut gu_tasks = [(0usize, 0usize, 0usize); 512];
    let mut n_gu_tasks = 0usize;
    for (gi, g) in groups.iter().enumerate() {
        if g.token_indices.is_empty() {
            continue;
        }
        let mut r0 = 0usize;
        while r0 < n_ff {
            let r1 = (r0 + gu_tile_rows).min(n_ff);
            gu_tasks[n_gu_tasks] = (gi, r0, r1);
            n_gu_tasks += 1;
            r0 = r1;
        }
    }

    let blk_done: [std::sync::atomic::AtomicU8; 512] =
        std::array::from_fn(|_| std::sync::atomic::AtomicU8::new(0));
    let groups_ptr = groups.as_ptr() as usize;
    let g_cnt = groups.len();
    let g_off_ptr = group_slot_off.as_ptr() as usize;
    let act_ptrs_u = act_ptrs.as_ptr() as usize;
    let ff_ptrs_u = ff_ptrs.as_ptr() as usize;
    let fq_ptrs_u = fq_ptrs.as_ptr() as usize;
    let fq_bytes_u = fq_base_ptr as usize;
    let blk_done_u = blk_done.as_ptr() as usize;
    let gu_tasks_u = gu_tasks.as_ptr() as usize;
    let gu_next = std::sync::atomic::AtomicUsize::new(0);
    let gu_next_u = &gu_next as *const std::sync::atomic::AtomicUsize as usize;

    pool.parallel_for(n_threads, |_w0, _w1| {
        let next = unsafe { &*(gu_next_u as *const std::sync::atomic::AtomicUsize) };
        let tasks = unsafe { std::slice::from_raw_parts(gu_tasks_u as *const (usize, usize, usize), n_gu_tasks) };
        let grp = unsafe { std::slice::from_raw_parts(groups_ptr as *const Glm5CpuGroupedExpert<'_>, g_cnt) };
        let offs = unsafe { std::slice::from_raw_parts(g_off_ptr as *const usize, g_cnt) };
        let bd = unsafe { std::slice::from_raw_parts(blk_done_u as *const std::sync::atomic::AtomicU8, 512) };
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
            unsafe {
                dispatch_fused_gu_rows(
                    use_x_q8_1,
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
                let need_parts = if split_128 && (n_ff - b * 256) > 128 { 2 } else { 1 };
                let ready = need_parts == 1
                    || bd[gi * nb_ff + b].fetch_add(1, std::sync::atomic::Ordering::AcqRel) == 1;
                if ready {
                    let b0 = b * 256;
                    let blk_n = (n_ff - b0).min(256) as i32;
                    for s in 0..nt {
                        let ff_s = *fp.add(off + s);
                        quantize_ff_blk(
                            use_q8_1,
                            ff_s.add(b0),
                            (fq_bytes_u as *mut u8).add(((off + s) * nb_ff + b) * fq_blk_bytes)
                                as *mut c_void,
                            blk_n,
                        );
                    }
                }
            }
        }
    });

    // Step 2 (Barrier #2): Fused Down + Weighted Accumulate in L1 across disjoint 64-row slices
    let n_slices = n_embd.div_ceil(64);
    let dn_next = std::sync::atomic::AtomicUsize::new(0);
    let dn_next_u = &dn_next as *const std::sync::atomic::AtomicUsize as usize;
    let down_tmp_u = scratch.down_tmp.as_mut_ptr() as usize;
    let out_u = out.as_mut_ptr() as usize;

    pool.parallel_for(n_threads, |w_idx, _w_end| {
        let next = unsafe { &*(dn_next_u as *const std::sync::atomic::AtomicUsize) };
        let grp = unsafe { std::slice::from_raw_parts(groups_ptr as *const Glm5CpuGroupedExpert<'_>, g_cnt) };
        let offs = unsafe { std::slice::from_raw_parts(g_off_ptr as *const usize, g_cnt) };
        let fp = fq_ptrs_u as *const *const c_void;
        let out_p = out_u as *mut f32;
        let dt_p = down_tmp_u as *mut f32;
        let mut w_tmp_ptrs = [std::ptr::null_mut::<f32>(); 8];
        for k in 0..8 {
            w_tmp_ptrs[k] = unsafe { dt_p.add((w_idx * 8 + k) * n_embd) };
        }

        loop {
            let s = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if s >= n_slices {
                break;
            }
            let r0 = s * 64;
            let r1 = (r0 + 64).min(n_embd);
            let nr = r1 - r0;
            let mut acc = [[0.0f32; 64]; 8];

            for gi in 0..g_cnt {
                let g = &grp[gi];
                let nt = g.token_indices.len();
                if nt == 0 {
                    continue;
                }
                let off = offs[gi];
                unsafe {
                    dispatch_down_rows(
                        dn_ty_i,
                        g.down_ptr,
                        dn_rb,
                        n_ff as i32,
                        fp.add(off),
                        nt as i32,
                        w_tmp_ptrs.as_ptr(),
                        r0 as i32,
                        r1 as i32,
                        1.0,
                        0,
                    );
                    for k in 0..nt {
                        let tok = g.token_indices[k];
                        let w = g.token_weights[k];
                        let src = w_tmp_ptrs[k].add(r0);
                        let acc_t = &mut acc[tok];
                        for i in 0..nr {
                            acc_t[i] += w * *src.add(i);
                        }
                    }
                }
            }

            unsafe {
                for tok in 0..m_tokens {
                    let dst = out_p.add(tok * n_embd + r0);
                    let src = &acc[tok];
                    if zero_out {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, nr);
                    } else {
                        for i in 0..nr {
                            *dst.add(i) += src[i];
                        }
                    }
                }
            }
        }
    });
}
