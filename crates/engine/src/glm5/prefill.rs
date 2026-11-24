use crate::glm5::cache::{LayerCacheMeta, LentPoolTail};
use crate::glm5::decode::{
    decode_token_m1, dequant_embed_to_host, enqueue_layer_m1, hop_gpu0_to_gpu1_r0,
    mtp_cache_entry_m1_with_images,
};
use crate::glm5::state::{
    DoorbellActiveGuard, DsaLayerState, Glm5State, GpuScratch, KdaLayerState,
};
use crate::glm5::vision::Glm5ImageSpan;
use crate::glm5::weights::{DevWeight, Glm5LayerWeights, Glm5ModelWeights};
use ampllm_cpu_kernels::glm5_cpu::{
    quantize_glm5_cpu_x_q8k, run_glm5_cpu_moe_grouped_prequant, Glm5CpuGroupedExpert,
    Glm5CpuScratch,
};
use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_cuda_kernels::ffi::{launch_linear_q8_1, launch_quantize_q8_1};
use ampllm_cuda_kernels::{
    copy_d2d_raw_async, copy_d2h_raw_async, copy_h2d_raw_async, copy_peer_raw_async,
    device_synchronize, glm5, set_device, CudaStream,
};
use ampllm_gguf::{GgmlType, GgufFile, Glm5Config};
use anyhow::Result;
use std::ffi::c_void;
use std::sync::{mpsc, OnceLock};
use std::time::Instant;

#[inline]
fn prefill_prof_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("AMPLLM_PREFILL_PROFILE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[derive(Default, Clone, Copy)]
struct PrefillProf {
    kda_ms: f64,
    dsa_ms: f64,
    dense_ms: f64,
    moe_router_sh_ms: f64,
    moe_vram_ms: f64,
    moe_cold_gpu_ms: f64,
    moe_cold_cpu_ms: f64,
    hop_ms: f64,
    head_mtp_ms: f64,
    gpu_cold_exps: usize,
    gpu_cold_rows: usize,
    cpu_exps: usize,
    cpu_rows: usize,
}

impl PrefillProf {
    fn add_from(&mut self, o: &Self) {
        self.kda_ms += o.kda_ms;
        self.dsa_ms += o.dsa_ms;
        self.dense_ms += o.dense_ms;
        self.moe_router_sh_ms += o.moe_router_sh_ms;
        self.moe_vram_ms += o.moe_vram_ms;
        self.moe_cold_gpu_ms += o.moe_cold_gpu_ms;
        self.moe_cold_cpu_ms += o.moe_cold_cpu_ms;
        self.hop_ms += o.hop_ms;
        self.head_mtp_ms += o.head_mtp_ms;
        self.gpu_cold_exps += o.gpu_cold_exps;
        self.gpu_cold_rows += o.gpu_cold_rows;
        self.cpu_exps += o.cpu_exps;
        self.cpu_rows += o.cpu_rows;
    }

    fn log(&self, tag: &str, m: usize) {
        eprintln!(
            "[AMPLLM_PREFILL_PROFILE {tag}] m={m} kda={:.2}ms dsa={:.2}ms dense={:.2}ms \
moe_rsh={:.2}ms moe_vram={:.2}ms moe_cold_gpu={:.2}ms ({}/{}r) moe_cold_cpu={:.2}ms ({}/{}r) \
hop={:.2}ms head_mtp={:.2}ms",
            self.kda_ms,
            self.dsa_ms,
            self.dense_ms,
            self.moe_router_sh_ms,
            self.moe_vram_ms,
            self.moe_cold_gpu_ms,
            self.gpu_cold_exps,
            self.gpu_cold_rows,
            self.moe_cold_cpu_ms,
            self.cpu_exps,
            self.cpu_rows,
            self.hop_ms,
            self.head_mtp_ms
        );
    }
}

pub fn small_moe_max_nt() -> usize {
    std::env::var("AMPLLM_SMALL_MOE_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 64)
}

#[derive(Clone, Copy)]
pub struct PrefillViews {
    pub r_buf0: *mut f32,
    pub post: *mut f32,
    pub comb: *mut f32,
    pub x_f32: *mut f32,
    pub x_q8: *mut c_void,
    pub gpu_moe_out: *mut f32,
    pub pos_arr_dev: *mut i32,
    pub moe_logits: *mut f32,
    pub moe_ids: *mut i32,
    pub moe_rw: *mut f32,
    pub moe_rows_dev: *mut i32,
    pub moe_wts_dev: *mut f32,
    pub moe_inter_q8: *mut c_void,
    pub gbuf_base: u64,
    pub gbuf_ptrs_dev: u64,
    pub gbuf_stride: usize,
    pub gbuf_slots: usize,
    pub moe_gu_f32: *mut f32,
    pub moe_grp_q8: *mut c_void,
    pub grp_window_rows: usize,
    pub small_max_nt: usize,
    pub small_cap_rows: usize,
    pub attn_union_base: u64,
    pub attn_union_bytes: usize,
}

unsafe impl Send for PrefillViews {}
unsafe impl Sync for PrefillViews {}

impl PrefillViews {
    fn layout_sizes(
        cfg: &Glm5Config,
        m: usize,
        gbuf_stride: usize,
        gbuf_slots: usize,
        small_max_nt: usize,
        grp_window_rows: usize,
    ) -> [usize; 17] {
        let align256 = |x: usize| (x + 255) & !255;
        let n_embd = cfg.n_embd;
        let hc = cfg.hc;
        let q8_x = (n_embd / 32) * 36;
        let q8_ff = (cfg.n_ff_exp / 32) * 36;
        let small_cap_rows = (m * cfg.n_expert_used).min(cfg.n_expert * small_max_nt);
        let aligned_stride = align256(gbuf_stride);
        [
            align256(m * hc * n_embd * 4),               // 0: r_buf0
            align256(m * hc * 4),                        // 1: post
            align256(m * hc * hc * 4),                   // 2: comb
            align256(m * n_embd * 4),                    // 3: x_f32
            align256(m * q8_x),                          // 4: x_q8
            align256(m * 4),                             // 5: pos_arr_dev
            align256(gbuf_slots * 8),                    // 6: gbuf_ptrs_dev
            align256(m * n_embd * 4),                    // 7: gpu_moe_out (attn_union start)
            align256(m * cfg.n_expert * 4),              // 8: moe_logits
            align256(m * cfg.n_expert_used * 4),         // 9: moe_ids
            align256(m * cfg.n_expert_used * 4),         // 10: moe_rw
            align256(m * cfg.n_expert_used * 4),         // 11: moe_rows_dev
            align256(m * cfg.n_expert_used * 4),         // 12: moe_wts_dev
            align256(small_cap_rows * q8_ff),            // 13: moe_inter_q8
            align256(gbuf_slots * aligned_stride),       // 14: gbuf
            align256(grp_window_rows * 2 * cfg.n_ff_exp * 4), // 15: moe_gu_f32 (also attn_union!)
            align256(grp_window_rows * q8_ff),           // 16: moe_grp_q8 (attn_union end!)
        ]
    }

    pub fn from_scratch(sc: &GpuScratch, cfg: &Glm5Config) -> Self {
        let gbuf_slots = 8usize;
        Self {
            r_buf0: sc.r_buf[0].ptr.cast(),
            post: sc.post.ptr.cast(),
            comb: sc.comb.ptr.cast(),
            x_f32: sc.x_f32.ptr.cast(),
            x_q8: sc.x_q8.ptr,
            gpu_moe_out: sc.gpu_moe_out.ptr.cast(),
            pos_arr_dev: sc.pos_arr_dev.ptr.cast(),
            moe_logits: sc.moe_logits.ptr.cast(),
            moe_ids: sc.moe_ids.ptr.cast(),
            moe_rw: sc.moe_rw.ptr.cast(),
            moe_rows_dev: sc.moe_rows_dev.ptr.cast(),
            moe_wts_dev: sc.moe_wts_dev.ptr.cast(),
            moe_inter_q8: sc.moe_inter_q8.ptr,
            gbuf_base: sc.moe_scratch_pool.as_u64(),
            gbuf_ptrs_dev: sc.moe_scratch_ptrs_dev.as_u64(),
            gbuf_stride: sc.moe_scratch_pool.bytes / gbuf_slots,
            gbuf_slots,
            moe_gu_f32: std::ptr::null_mut(),
            moe_grp_q8: std::ptr::null_mut(),
            grp_window_rows: 0,
            small_max_nt: small_moe_max_nt(),
            small_cap_rows: sc.max_prefill_chunk * cfg.n_expert_used,
            attn_union_base: 0,
            attn_union_bytes: 0,
        }
    }

    pub fn required_lend_bytes(
        cfg: &Glm5Config,
        m: usize,
        gbuf_stride: usize,
        gbuf_slots: usize,
        small_max_nt: usize,
        grp_window_rows: usize,
    ) -> usize {
        Self::layout_sizes(cfg, m, gbuf_stride, gbuf_slots, small_max_nt, grp_window_rows)
            .iter()
            .sum()
    }

    pub fn from_lent_tail(
        tail: &LentPoolTail,
        cfg: &Glm5Config,
        m: usize,
        gbuf_stride: usize,
        gbuf_slots: usize,
        small_max_nt: usize,
        grp_window_rows: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        let align256 = |x: usize| (x + 255) & !255;
        let aligned_stride = align256(gbuf_stride);
        let sizes = Self::layout_sizes(cfg, m, gbuf_stride, gbuf_slots, small_max_nt, grp_window_rows);
        let total: usize = sizes.iter().sum();
        if total > tail.bytes {
            anyhow::bail!(
                "PrefillViews::from_lent_tail: need {} > tail.bytes {}",
                total,
                tail.bytes
            );
        }
        let mut cur = 0usize;
        let mut take = |sz: usize| -> u64 {
            let p = tail.base_ptr + cur as u64;
            cur += sz;
            p
        };
        let r_buf0 = take(sizes[0]) as *mut f32;
        let post = take(sizes[1]) as *mut f32;
        let comb = take(sizes[2]) as *mut f32;
        let x_f32 = take(sizes[3]) as *mut f32;
        let x_q8 = take(sizes[4]) as *mut c_void;
        let pos_arr_dev = take(sizes[5]) as *mut i32;
        let gbuf_ptrs_dev = take(sizes[6]);
        let gpu_moe_out = take(sizes[7]) as *mut f32;
        let attn_union_base = gpu_moe_out as u64;
        let attn_union_bytes: usize = sizes[7..17].iter().sum();
        let moe_logits = take(sizes[8]) as *mut f32;
        let moe_ids = take(sizes[9]) as *mut i32;
        let moe_rw = take(sizes[10]) as *mut f32;
        let moe_rows_dev = take(sizes[11]) as *mut i32;
        let moe_wts_dev = take(sizes[12]) as *mut f32;
        let moe_inter_q8 = take(sizes[13]) as *mut c_void;
        let gbuf_base = take(sizes[14]);
        let moe_gu_f32 = take(sizes[15]) as *mut f32;
        let moe_grp_q8 = take(sizes[16]) as *mut c_void;
        debug_assert_eq!(cur, total);

        let gbuf_ptrs: Vec<u64> = (0..gbuf_slots)
            .map(|i| gbuf_base + (i * aligned_stride) as u64)
            .collect();
        unsafe {
            copy_h2d_raw_async(
                gbuf_ptrs_dev as *mut c_void,
                bytemuck::cast_slice(&gbuf_ptrs),
                stream,
            )?;
        }
        stream.sync()?;

        Ok(Self {
            r_buf0,
            post,
            comb,
            x_f32,
            x_q8,
            gpu_moe_out,
            pos_arr_dev,
            moe_logits,
            moe_ids,
            moe_rw,
            moe_rows_dev,
            moe_wts_dev,
            moe_inter_q8,
            gbuf_base,
            gbuf_ptrs_dev,
            gbuf_stride: aligned_stride,
            gbuf_slots,
            moe_gu_f32,
            moe_grp_q8,
            grp_window_rows,
            small_max_nt,
            small_cap_rows: (m * cfg.n_expert_used).min(cfg.n_expert * small_max_nt),
            attn_union_base,
            attn_union_bytes,
        })
    }
}

struct HalfPrefillCtx<'a> {
    dev: i32,
    weights: &'a Glm5ModelWeights,
    sc: &'a GpuScratch,
    views: PrefillViews,
    kda: &'a mut [Option<KdaLayerState>],
    dsa: &'a mut [Option<DsaLayerState>],
    cache: &'a mut [LayerCacheMeta],
    cache_tick: u64,
    lb: usize,
    pool: &'a CpuThreadPool,
    cpu_scratch: &'a mut Glm5CpuScratch,
    meta_cursor: usize,
    prof: PrefillProf,
}

struct ColdJob {
    e: usize,
    row_off: usize,
    nt: usize,
    tok_indices: Vec<usize>,
    tok_wts: Vec<f32>,
}

#[inline]
fn mmq_ok(ty: GgmlType, k_in: usize) -> bool {
    matches!(
        ty,
        GgmlType::Iq2S
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S
            | GgmlType::Q6K
            | GgmlType::Q8_0
            | GgmlType::Q4K
    ) && k_in % 32 == 0
}

unsafe fn linear_batch(
    sc: &GpuScratch,
    w: &DevWeight,
    x_f32: *const f32,
    x_q8_opt: *const c_void,
    y_f32: *mut f32,
    n_out: usize,
    k_in: usize,
    m: usize,
    alpha: f32,
    stream: &CudaStream,
) {
    if m < 16 {
        let xq = if !x_q8_opt.is_null() {
            x_q8_opt
        } else {
            launch_quantize_q8_1(x_f32, sc.mma_a_f16.ptr, m as i32, k_in as i32, stream.raw);
            sc.mma_a_f16.ptr as *const c_void
        };
        launch_linear_q8_1(
            w.ggml_type as i32,
            w.ptr as *const c_void,
            xq,
            x_f32,
            y_f32,
            n_out as i32,
            k_in as i32,
            m as i32,
            stream.raw,
        );
        if (alpha - 1.0).abs() > 1e-7 {
            glm5::scale_f32(y_f32, alpha, (m * n_out) as i32, stream);
        }
    } else if mmq_ok(w.ggml_type, k_in)
        && (!x_q8_opt.is_null() || m * (k_in / 32) * 36 <= sc.mma_a_f16.bytes)
    {
        let xq = if !x_q8_opt.is_null() {
            x_q8_opt
        } else {
            launch_quantize_q8_1(x_f32, sc.mma_a_f16.ptr, m as i32, k_in as i32, stream.raw);
            sc.mma_a_f16.ptr as *const c_void
        };
        glm5::gemm_mmq_mma(
            w.ggml_type as i32,
            xq,
            w.ptr as *const c_void,
            w.ggml_type.row_bytes(k_in),
            y_f32,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            m as i32,
            n_out as i32,
            k_in as i32,
            alpha,
            0.0,
            stream,
        );
    } else {
        glm5::f32_to_f16(x_f32, sc.mma_a_f16.ptr, (m * k_in) as i32, stream);
        glm5::dequant_to_f16(
            w.ggml_type as i32,
            w.ptr as *const c_void,
            sc.mma_b_f16.ptr,
            (n_out * k_in) as i64,
            stream,
        );
        glm5::gemm_f16_mma_nt(
            sc.mma_a_f16.ptr,
            sc.mma_b_f16.ptr,
            y_f32,
            m as i32,
            n_out as i32,
            k_in as i32,
            alpha,
            0.0,
            stream,
        );
    }
}

unsafe fn linear_tiled_f16(
    sc: &GpuScratch,
    w: &DevWeight,
    x_f32: *const f32,
    y_f32: *mut f32,
    n_out: usize,
    k_in: usize,
    m: usize,
    alpha: f32,
    stream: &CudaStream,
) {
    debug_assert!(n_out * k_in * 2 <= sc.mma_b_f16.bytes);
    glm5::dequant_to_f16(
        w.ggml_type as i32,
        w.ptr as *const c_void,
        sc.mma_b_f16.ptr,
        (n_out * k_in) as i64,
        stream,
    );
    let a_rows = (sc.mma_a_f16.bytes / (k_in * 2)).max(1);
    for off in (0..m).step_by(a_rows) {
        let ms = (m - off).min(a_rows);
        glm5::f32_to_f16(x_f32.add(off * k_in), sc.mma_a_f16.ptr, (ms * k_in) as i32, stream);
        glm5::gemm_f16_mma_nt(
            sc.mma_a_f16.ptr,
            sc.mma_b_f16.ptr,
            y_f32.add(off * n_out),
            ms as i32,
            n_out as i32,
            k_in as i32,
            alpha,
            0.0,
            stream,
        );
    }
}

unsafe fn linear_from_a_q8(
    sc: &GpuScratch,
    w: &DevWeight,
    x_f32: *const f32,
    a_q8: *const c_void,
    a_f16: *const c_void,
    y_f32: *mut f32,
    n_out: usize,
    k_in: usize,
    m: usize,
    alpha: f32,
    stream: &CudaStream,
) {
    if mmq_ok(w.ggml_type, k_in) {
        glm5::gemm_mmq_mma(
            w.ggml_type as i32,
            a_q8,
            w.ptr as *const c_void,
            w.ggml_type.row_bytes(k_in),
            y_f32,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            m as i32,
            n_out as i32,
            k_in as i32,
            alpha,
            0.0,
            stream,
        );
    } else if !a_f16.is_null() {
        linear_from_a_f16(sc, w, a_f16, y_f32, n_out, k_in, m, alpha, stream);
    } else {
        linear_tiled_f16(sc, w, x_f32, y_f32, n_out, k_in, m, alpha, stream);
    }
}

unsafe fn linear_from_a_f16(
    sc: &GpuScratch,
    w: &DevWeight,
    a_f16: *const c_void,
    y_f32: *mut f32,
    n_out: usize,
    k_in: usize,
    m: usize,
    alpha: f32,
    stream: &CudaStream,
) {
    debug_assert!(n_out * k_in * 2 <= sc.mma_b_f16.bytes);
    glm5::dequant_to_f16(
        w.ggml_type as i32,
        w.ptr as *const c_void,
        sc.mma_b_f16.ptr,
        (n_out * k_in) as i64,
        stream,
    );
    glm5::gemm_f16_mma_nt(
        a_f16,
        sc.mma_b_f16.ptr,
        y_f32,
        m as i32,
        n_out as i32,
        k_in as i32,
        alpha,
        0.0,
        stream,
    );
}

unsafe fn run_moe_expert_mma(
    sc: &GpuScratch,
    views: &PrefillViews,
    lw: &Glm5LayerWeights,
    cfg: &Glm5Config,
    blob_ptr: u64,
    row_off: usize,
    nt: usize,
    stream: &CudaStream,
) {
    let n_embd = cfg.n_embd;
    let n_ff_exp = cfg.n_ff_exp;
    let clamp_lim = cfg.swiglu_clamp_exp;
    let fused_gu = lw.u_off == lw.g_eb;
    let gu_elems = (n_ff_exp * n_embd) as i64;
    let d_elems = (n_embd * n_ff_exp) as i64;
    let hoist_ok = views.attn_union_bytes > 0
        && fused_gu
        && ((2 * gu_elems + d_elems) as usize) * 2 <= sc.mma_b_f16.bytes;
    if fused_gu && mmq_ok(lw.gu_type, n_embd) && mmq_ok(lw.d_type, n_ff_exp) {
        let gu_rb = lw.gu_type.row_bytes(n_embd);
        let d_rb = lw.d_type.row_bytes(n_ff_exp);
        let q8_ff_rb = (n_ff_exp / 32) * 36;
        let nt_tile = (sc.mma_a_f16.bytes / q8_ff_rb)
            .min(sc.gate.bytes / (2 * n_ff_exp * 4))
            .min(384)
            .max(1);
        for t0 in (0..nt).step_by(nt_tile) {
            let nt_s = (nt - t0).min(nt_tile);
            let off_s = row_off + t0;
            let rows_ptr = views.moe_rows_dev.add(off_s);
            let wts_ptr = views.moe_wts_dev.add(off_s);

            glm5::gemm_mmq_mma(
                lw.gu_type as i32,
                views.x_q8,
                blob_ptr as *const c_void,
                gu_rb,
                sc.gate.ptr.cast(),
                rows_ptr,
                std::ptr::null(),
                std::ptr::null(),
                nt_s as i32,
                (2 * n_ff_exp) as i32,
                n_embd as i32,
                1.0,
                0.0,
                stream,
            );
            glm5::swiglu_clamp_interleaved_q8(
                sc.gate.ptr.cast(),
                clamp_lim,
                nt_s as i32,
                n_ff_exp as i32,
                sc.mma_a_f16.ptr,
                stream,
            );
            glm5::gemm_mmq_mma(
                lw.d_type as i32,
                sc.mma_a_f16.ptr,
                (blob_ptr + lw.d_off as u64) as *const c_void,
                d_rb,
                views.gpu_moe_out,
                std::ptr::null(),
                rows_ptr,
                wts_ptr,
                nt_s as i32,
                n_embd as i32,
                n_ff_exp as i32,
                1.0,
                0.0,
                stream,
            );
        }
        return;
    }

    let gu_slots = if fused_gu { 2 } else { 1 };
    let (gu_f16, dn_f16, max_nt_cap) = if hoist_ok {
        let gu_ptr = sc.mma_b_f16.ptr;
        let dn_ptr = gu_ptr.cast::<u8>().add((2 * gu_elems as usize) * 2).cast::<c_void>();
        glm5::dequant_to_f16(lw.gu_type as i32, blob_ptr as *const c_void, gu_ptr, 2 * gu_elems, stream);
        glm5::dequant_to_f16(lw.d_type as i32, (blob_ptr + lw.d_off as u64) as *const c_void, dn_ptr, d_elems, stream);
        (gu_ptr, dn_ptr, 384usize)
    } else {
        (sc.mma_b_f16.ptr, sc.mma_b_f16.ptr, if fused_gu { 256 } else { sc.max_batch })
    };
    let nt_tile = (sc.mma_a_f16.bytes / (n_embd.max(n_ff_exp) * 2))
        .min(sc.gate.bytes / (gu_slots * n_ff_exp * 4))
        .min(sc.moe_tok_dn.bytes / (n_embd * 4))
        .min(max_nt_cap)
        .max(1);

    for t0 in (0..nt).step_by(nt_tile) {
        let nt_s = (nt - t0).min(nt_tile);
        let off_s = row_off + t0;
        let rows_ptr = views.moe_rows_dev.add(off_s);
        let wts_ptr = views.moe_wts_dev.add(off_s);

        glm5::gather_rows_f16(
            views.x_f32,
            rows_ptr,
            sc.mma_a_f16.ptr,
            nt_s as i32,
            n_embd as i32,
            stream,
        );
        if fused_gu {
            if !hoist_ok {
                glm5::dequant_to_f16(
                    lw.gu_type as i32,
                    blob_ptr as *const c_void,
                    gu_f16,
                    2 * gu_elems,
                    stream,
                );
            }
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr,
                gu_f16,
                sc.gate.ptr.cast(),
                nt_s as i32,
                (2 * n_ff_exp) as i32,
                n_embd as i32,
                1.0,
                0.0,
                stream,
            );
            glm5::swiglu_clamp_interleaved_to_f16(
                sc.gate.ptr.cast(),
                clamp_lim,
                nt_s as i32,
                n_ff_exp as i32,
                sc.mma_a_f16.ptr,
                stream,
            );
        } else {
            glm5::dequant_to_f16(
                lw.gu_type as i32,
                blob_ptr as *const c_void,
                sc.mma_b_f16.ptr,
                gu_elems,
                stream,
            );
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr,
                sc.mma_b_f16.ptr,
                sc.gate.ptr.cast(),
                nt_s as i32,
                n_ff_exp as i32,
                n_embd as i32,
                1.0,
                0.0,
                stream,
            );
            glm5::dequant_to_f16(
                lw.gu_type as i32,
                (blob_ptr + lw.u_off as u64) as *const c_void,
                sc.mma_b_f16.ptr,
                gu_elems,
                stream,
            );
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr,
                sc.mma_b_f16.ptr,
                sc.up.ptr.cast(),
                nt_s as i32,
                n_ff_exp as i32,
                n_embd as i32,
                1.0,
                0.0,
                stream,
            );
            glm5::swiglu_clamp_to_f16(
                sc.gate.ptr.cast(),
                sc.up.ptr.cast(),
                clamp_lim,
                (nt_s * n_ff_exp) as i32,
                sc.mma_a_f16.ptr,
                stream,
            );
        }
        if !hoist_ok {
            glm5::dequant_to_f16(
                lw.d_type as i32,
                (blob_ptr + lw.d_off as u64) as *const c_void,
                dn_f16,
                d_elems,
                stream,
            );
        }
        glm5::gemm_f16_mma_nt(
            sc.mma_a_f16.ptr,
            dn_f16,
            sc.moe_tok_dn.ptr.cast(),
            nt_s as i32,
            n_embd as i32,
            n_ff_exp as i32,
            1.0,
            0.0,
            stream,
        );
        glm5::scatter_weighted_add_f32(
            sc.moe_tok_dn.ptr.cast(),
            rows_ptr,
            wts_ptr,
            views.gpu_moe_out,
            nt_s as i32,
            n_embd as i32,
            stream,
        );
    }
}

unsafe fn run_moe_group_mmq_class(
    sc: &GpuScratch,
    views: &PrefillViews,
    lw: &Glm5LayerWeights,
    cfg: &Glm5Config,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)],
    tile_mode: i32,
    bm: usize,
    stream: &CudaStream,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let n_embd = cfg.n_embd;
    let n_ff_exp = cfg.n_ff_exp;
    let clamp_lim = cfg.swiglu_clamp_exp;
    let gu_rb = lw.gu_type.row_bytes(n_embd);
    let d_rb = lw.d_type.row_bytes(n_ff_exp);
    let max_win_rows = views.grp_window_rows;
    let max_exps = sc.moe_grp_stride_exps;
    let max_tiles = sc.moe_tile_stride;
    let n_meta_slots = sc.ev_meta_done.len();

    let dispatch_one_window =
        |win: &[(u64, usize, usize)], meta_cursor: &mut usize| -> Result<()> {
            if win.is_empty() {
                return Ok(());
            }
            let slot = *meta_cursor % n_meta_slots;
            if *meta_cursor >= n_meta_slots {
                sc.ev_meta_done[slot].sync()?;
            }
            *meta_cursor += 1;

            let base_row_off = win[0].1;
            let n_exp = win.len();
            let g_base = slot * max_exps;
            let t_base = slot * max_tiles;

            let ptrs_pin = std::slice::from_raw_parts_mut(
                sc.prefill_grp_ptrs_pin.host_ptr().add(g_base),
                n_exp,
            );
            let off_pin = std::slice::from_raw_parts_mut(
                sc.prefill_grp_off_pin.host_ptr().add(g_base),
                n_exp,
            );
            let nt_pin = std::slice::from_raw_parts_mut(
                sc.prefill_grp_nt_pin.host_ptr().add(g_base),
                n_exp,
            );
            let te_pin = std::slice::from_raw_parts_mut(
                sc.prefill_tile_exp_pin.host_ptr().add(t_base),
                max_tiles,
            );
            let tm_pin = std::slice::from_raw_parts_mut(
                sc.prefill_tile_moff_pin.host_ptr().add(t_base),
                max_tiles,
            );

            let mut w_rows = 0usize;
            let mut n_tiles = 0usize;
            for (ei, &(ptr, row_off, nt)) in win.iter().enumerate() {
                debug_assert_eq!(row_off - base_row_off, w_rows);
                ptrs_pin[ei] = ptr;
                off_pin[ei] = w_rows as i32;
                nt_pin[ei] = nt as i32;
                w_rows += nt;
                let mut m_off = 0usize;
                while m_off < nt {
                    te_pin[n_tiles] = ei as i32;
                    tm_pin[n_tiles] = m_off as i32;
                    n_tiles += 1;
                    m_off += bm;
                }
            }

            let d_ptrs = (sc.moe_grp_ptrs_dev.ptr as u64 + (g_base * 8) as u64) as *mut c_void;
            let d_off = (sc.moe_grp_off_dev.ptr as u64 + (g_base * 4) as u64) as *mut c_void;
            let d_nt = (sc.moe_grp_nt_dev.ptr as u64 + (g_base * 4) as u64) as *mut c_void;
            let d_te = (sc.moe_tile_exp_dev.ptr as u64 + (t_base * 4) as u64) as *mut c_void;
            let d_tm = (sc.moe_tile_moff_dev.ptr as u64 + (t_base * 4) as u64) as *mut c_void;

            copy_h2d_raw_async(
                d_ptrs,
                std::slice::from_raw_parts(ptrs_pin.as_ptr() as *const u8, n_exp * 8),
                stream,
            )?;
            copy_h2d_raw_async(
                d_off,
                std::slice::from_raw_parts(off_pin.as_ptr() as *const u8, n_exp * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                d_nt,
                std::slice::from_raw_parts(nt_pin.as_ptr() as *const u8, n_exp * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                d_te,
                std::slice::from_raw_parts(te_pin.as_ptr() as *const u8, n_tiles * 4),
                stream,
            )?;
            copy_h2d_raw_async(
                d_tm,
                std::slice::from_raw_parts(tm_pin.as_ptr() as *const u8, n_tiles * 4),
                stream,
            )?;

            let rows_dev = views.moe_rows_dev.add(base_row_off);
            let wts_dev = views.moe_wts_dev.add(base_row_off);
            glm5::gemm_mmq_mma_grouped(
                lw.gu_type as i32,
                views.x_q8,
                d_ptrs as *const u64,
                0,
                gu_rb,
                views.moe_gu_f32,
                rows_dev,
                std::ptr::null(),
                std::ptr::null(),
                d_off as *const i32,
                d_nt as *const i32,
                d_te as *const i32,
                d_tm as *const i32,
                n_tiles as i32,
                (2 * n_ff_exp) as i32,
                n_embd as i32,
                1.0,
                false,
                tile_mode,
                stream,
            );
            glm5::swiglu_clamp_interleaved_q8(
                views.moe_gu_f32,
                clamp_lim,
                w_rows as i32,
                n_ff_exp as i32,
                views.moe_grp_q8,
                stream,
            );
            glm5::gemm_mmq_mma_grouped(
                lw.d_type as i32,
                views.moe_grp_q8,
                d_ptrs as *const u64,
                lw.d_off,
                d_rb,
                views.gpu_moe_out,
                std::ptr::null(),
                rows_dev,
                wts_dev,
                d_off as *const i32,
                d_nt as *const i32,
                d_te as *const i32,
                d_tm as *const i32,
                n_tiles as i32,
                n_embd as i32,
                n_ff_exp as i32,
                1.0,
                true,
                tile_mode,
                stream,
            );
            sc.ev_meta_done[slot].record(stream)?;
            Ok(())
        };

    let mut idx = 0usize;
    while idx < jobs.len() {
        let (ptr0, roff0, nt0) = jobs[idx];
        if nt0 > max_win_rows {
            for t0 in (0..nt0).step_by(max_win_rows) {
                let slice_job = [(ptr0, roff0 + t0, (nt0 - t0).min(max_win_rows))];
                dispatch_one_window(&slice_job, meta_cursor)?;
            }
            idx += 1;
            continue;
        }
        let mut end = idx;
        let mut sum_rows = 0usize;
        let mut sum_tiles = 0usize;
        while end < jobs.len() && (end - idx) < max_exps {
            let nt = jobs[end].2;
            let t_exp = nt.div_ceil(bm);
            if end > idx && (sum_rows + nt > max_win_rows || sum_tiles + t_exp > max_tiles) {
                break;
            }
            sum_rows += nt;
            sum_tiles += t_exp;
            end += 1;
        }
        dispatch_one_window(&jobs[idx..end], meta_cursor)?;
        idx = end;
    }
    Ok(())
}

unsafe fn run_moe_group_mmq(
    sc: &GpuScratch,
    views: &PrefillViews,
    lw: &Glm5LayerWeights,
    cfg: &Glm5Config,
    meta_cursor: &mut usize,
    jobs: &[(u64, usize, usize)], // (blob_ptr, row_off, nt) sorted desc by nt and contiguous in rows_pin
    stream: &CudaStream,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let fused_gu = lw.u_off == lw.g_eb;
    let use_grp = fused_gu
        && mmq_ok(lw.gu_type, cfg.n_embd)
        && mmq_ok(lw.d_type, cfg.n_ff_exp)
        && !views.moe_gu_f32.is_null()
        && !views.moe_grp_q8.is_null()
        && views.grp_window_rows > 0;
    if !use_grp {
        for &(ptr, row_off, nt) in jobs {
            run_moe_expert_mma(sc, views, lw, cfg, ptr, row_off, nt, stream);
        }
        return Ok(());
    }
    let s1 = jobs.iter().take_while(|j| j.2 > 96).count();
    let s2 = s1 + jobs[s1..].iter().take_while(|j| j.2 > 32).count();
    run_moe_group_mmq_class(sc, views, lw, cfg, meta_cursor, &jobs[..s1], 1, 128, stream)?;
    run_moe_group_mmq_class(sc, views, lw, cfg, meta_cursor, &jobs[s1..s2], 2, 64, stream)?;
    run_moe_group_mmq_class(sc, views, lw, cfg, meta_cursor, &jobs[s2..], 3, 32, stream)?;
    Ok(())
}

unsafe fn run_moe_hybrid_split(
    ctx: &mut HalfPrefillCtx<'_>,
    l_rel: usize,
    m: usize,
) -> Result<()> {
    let l = ctx.lb + l_rel;
    let weights = ctx.weights;
    let cfg = &weights.cfg;
    let lw = &weights.layers[l];
    let sc = ctx.sc;
    let views = &ctx.views;
    let stream = &sc.stream;
    let d2h_stream = &sc.d2h_stream;
    let pf_stream = &sc.pf_stream;
    let n_embd = cfg.n_embd;
    let n_ff_exp = cfg.n_ff_exp;
    let topk = cfg.n_expert_used;
    let clamp_lim = cfg.swiglu_clamp_exp;
    let q8_row_bytes = (n_embd / 32) * 36;
    let prof_on = prefill_prof_enabled();
    let t_rsh = prof_on.then(Instant::now);

    // 1. Overlapped D2H of x_f32 on dedicated d2h_stream while router + shared expert run
    sc.ev_stream_comp.record(stream)?;
    d2h_stream.wait_event(&sc.ev_stream_comp)?;
    {
        let x_pin_slice =
            std::slice::from_raw_parts_mut(sc.prefill_io_pin.host_ptr().cast::<u8>(), m * n_embd * 4);
        copy_d2h_raw_async(x_pin_slice, views.x_f32.cast(), d2h_stream)?;
    }
    sc.ev_stream_copy.record(d2h_stream)?;

    // 2. Router logits -> sigmoid top-k -> async pinned D2H of moe_ids/moe_rw
    let ginp = lw.ffn_gate_inp.as_ref().unwrap();
    if m < 16 {
        linear_batch(
            sc,
            ginp,
            views.x_f32,
            views.x_q8,
            views.moe_logits,
            cfg.n_expert,
            n_embd,
            m,
            1.0,
            stream,
        );
    } else {
        glm5::dequant_to_f16(
            ginp.ggml_type as i32,
            ginp.ptr as *const c_void,
            sc.mma_b_f16.ptr,
            (cfg.n_expert * n_embd) as i64,
            stream,
        );
        let router_tile = (sc.mma_a_f16.bytes / (n_embd * 2)).min(512).max(1);
        for off in (0..m).step_by(router_tile) {
            let m_sub = (m - off).min(router_tile);
            let xf_sub = views.x_f32.add(off * n_embd);
            let logits_sub = views.moe_logits.add(off * cfg.n_expert);
            glm5::f32_to_f16(xf_sub, sc.mma_a_f16.ptr, (m_sub * n_embd) as i32, stream);
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr,
                sc.mma_b_f16.ptr,
                logits_sub,
                m_sub as i32,
                cfg.n_expert as i32,
                n_embd as i32,
                1.0,
                0.0,
                stream,
            );
        }
    }
    glm5::router_sigmoid_probe(
        views.moe_logits,
        lw.exp_probs_bias.unwrap() as *const f32,
        cfg.n_expert as i32,
        topk as i32,
        cfg.expert_weights_scale,
        cfg.expert_weights_norm,
        std::ptr::null(),
        std::ptr::null(),
        n_embd as i32,
        n_ff_exp as i32,
        l as i32,
        lw.gu_type as i32,
        lw.d_type as i32,
        clamp_lim,
        views.moe_ids,
        views.moe_rw,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        m as i32,
        stream,
    );
    {
        let ids_pin = std::slice::from_raw_parts_mut(
            sc.prefill_ids_pin.host_ptr().cast::<u8>(),
            m * topk * 4,
        );
        let rw_pin = std::slice::from_raw_parts_mut(
            sc.prefill_rw_pin.host_ptr().cast::<u8>(),
            m * topk * 4,
        );
        copy_d2h_raw_async(ids_pin, views.moe_ids.cast(), stream)?;
        copy_d2h_raw_async(rw_pin, views.moe_rw.cast(), stream)?;
    }
    sc.ev_pf_done[0].record(stream)?;

    // 3. Shared Expert writes directly into gpu_moe_out
    let gs = lw.ffn_gate_sh.as_ref().unwrap();
    let us = lw.ffn_up_sh.as_ref().unwrap();
    let ds = lw.ffn_down_sh.as_ref().unwrap();
    let n_ff_sh = cfg.n_ff_shexp * cfg.n_shared_expert;
    let sh_f16_bytes = (2 * n_ff_sh * n_embd + n_embd * n_ff_sh) * 2;
    let sh_mmq_ok = m >= 16
        && mmq_ok(gs.ggml_type, n_embd)
        && mmq_ok(us.ggml_type, n_embd)
        && mmq_ok(ds.ggml_type, n_ff_sh);
    if sh_mmq_ok {
        let gs_rb = gs.ggml_type.row_bytes(n_embd);
        let us_rb = us.ggml_type.row_bytes(n_embd);
        let ds_rb = ds.ggml_type.row_bytes(n_ff_sh);
        let q8_sh_rb = (n_ff_sh / 32) * 36;
        let (gate_buf, up_buf, h_q8_buf, sh_tile) =
            if !views.moe_gu_f32.is_null()
                && !views.moe_grp_q8.is_null()
                && views.grp_window_rows >= 1024
                && n_ff_sh <= cfg.n_ff_exp
            {
                let st = views.grp_window_rows.min(2048);
                (
                    views.moe_gu_f32,
                    views.moe_gu_f32.add(st * n_ff_sh),
                    views.moe_grp_q8,
                    st,
                )
            } else {
                let st = (sc.mma_a_f16.bytes / q8_sh_rb)
                    .min(sc.gate.bytes / (n_ff_sh * 4))
                    .min(sc.up.bytes / (n_ff_sh * 4))
                    .min(512)
                    .max(1);
                (sc.gate.ptr.cast(), sc.up.ptr.cast(), sc.mma_a_f16.ptr, st)
            };
        for off in (0..m).step_by(sh_tile) {
            let m_sub = (m - off).min(sh_tile);
            let xq_sub = (views.x_q8 as *const u8).add(off * q8_row_bytes).cast::<c_void>();
            let sh_sub = views.gpu_moe_out.add(off * n_embd);
            glm5::gemm_mmq_mma(
                gs.ggml_type as i32, xq_sub, gs.ptr as *const c_void, gs_rb, gate_buf,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                m_sub as i32, n_ff_sh as i32, n_embd as i32, 1.0, 0.0, stream,
            );
            glm5::gemm_mmq_mma(
                us.ggml_type as i32, xq_sub, us.ptr as *const c_void, us_rb, up_buf,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                m_sub as i32, n_ff_sh as i32, n_embd as i32, 1.0, 0.0, stream,
            );
            glm5::swiglu_clamp_q8(
                gate_buf, up_buf, cfg.swiglu_clamp_shexp,
                n_ff_sh as i32, std::ptr::null_mut(), h_q8_buf, m_sub as i32, stream,
            );
            glm5::gemm_mmq_mma(
                ds.ggml_type as i32, h_q8_buf, ds.ptr as *const c_void, ds_rb, sh_sub,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                m_sub as i32, n_embd as i32, n_ff_sh as i32, 1.0, 0.0, stream,
            );
        }
    } else if m >= 16 && views.attn_union_bytes > 0 && sh_f16_bytes <= sc.mma_b_f16.bytes {
        let gs_f16 = sc.mma_b_f16.ptr;
        let us_f16 = gs_f16.cast::<u8>().add(n_ff_sh * n_embd * 2).cast::<c_void>();
        let ds_f16 = us_f16.cast::<u8>().add(n_ff_sh * n_embd * 2).cast::<c_void>();
        glm5::dequant_to_f16(gs.ggml_type as i32, gs.ptr as *const c_void, gs_f16, (n_ff_sh * n_embd) as i64, stream);
        glm5::dequant_to_f16(us.ggml_type as i32, us.ptr as *const c_void, us_f16, (n_ff_sh * n_embd) as i64, stream);
        glm5::dequant_to_f16(ds.ggml_type as i32, ds.ptr as *const c_void, ds_f16, (n_embd * n_ff_sh) as i64, stream);
        let sh_tile = (sc.mma_a_f16.bytes / (n_embd.max(n_ff_sh) * 2))
            .min(sc.gate.bytes / (n_ff_sh * 4))
            .min(sc.up.bytes / (n_ff_sh * 4))
            .min(512)
            .max(1);
        for off in (0..m).step_by(sh_tile) {
            let m_sub = (m - off).min(sh_tile);
            let xf_sub = views.x_f32.add(off * n_embd);
            let sh_sub = views.gpu_moe_out.add(off * n_embd);
            glm5::f32_to_f16(xf_sub, sc.mma_a_f16.ptr, (m_sub * n_embd) as i32, stream);
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr, gs_f16, sc.gate.ptr.cast(),
                m_sub as i32, n_ff_sh as i32, n_embd as i32, 1.0, 0.0, stream,
            );
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr, us_f16, sc.up.ptr.cast(),
                m_sub as i32, n_ff_sh as i32, n_embd as i32, 1.0, 0.0, stream,
            );
            glm5::swiglu_clamp_to_f16(
                sc.gate.ptr.cast(), sc.up.ptr.cast(), cfg.swiglu_clamp_shexp,
                (m_sub * n_ff_sh) as i32, sc.mma_a_f16.ptr, stream,
            );
            glm5::gemm_f16_mma_nt(
                sc.mma_a_f16.ptr, ds_f16, sh_sub,
                m_sub as i32, n_embd as i32, n_ff_sh as i32, 1.0, 0.0, stream,
            );
        }
    } else {
        let router_tile = (sc.mma_a_f16.bytes / (n_embd * 2)).min(512).max(1);
        for off in (0..m).step_by(router_tile) {
            let m_sub = (m - off).min(router_tile);
            let xf_sub = views.x_f32.add(off * n_embd);
            let xq_sub = (views.x_q8 as *const u8).add(off * q8_row_bytes).cast::<c_void>();
            let sh_sub = views.gpu_moe_out.add(off * n_embd);
            linear_batch(
                sc, gs, xf_sub, xq_sub, sc.gate.ptr.cast(), n_ff_sh, n_embd, m_sub, 1.0, stream,
            );
            linear_batch(
                sc, us, xf_sub, xq_sub, sc.up.ptr.cast(), n_ff_sh, n_embd, m_sub, 1.0, stream,
            );
            if m_sub < 16 {
                glm5::swiglu_clamp_q8(
                    sc.gate.ptr.cast(),
                    sc.up.ptr.cast(),
                    cfg.swiglu_clamp_shexp,
                    n_ff_sh as i32,
                    std::ptr::null_mut(),
                    sc.h_q8.ptr,
                    m_sub as i32,
                    stream,
                );
                linear_batch(
                    sc,
                    ds,
                    sc.h_f32.ptr.cast(),
                    sc.h_q8.ptr,
                    sh_sub,
                    n_embd,
                    n_ff_sh,
                    m_sub,
                    1.0,
                    stream,
                );
            } else {
                glm5::swiglu_clamp_to_f16(
                    sc.gate.ptr.cast(),
                    sc.up.ptr.cast(),
                    cfg.swiglu_clamp_shexp,
                    (m_sub * n_ff_sh) as i32,
                    sc.mma_a_f16.ptr,
                    stream,
                );
                glm5::dequant_to_f16(
                    ds.ggml_type as i32,
                    ds.ptr as *const c_void,
                    sc.mma_b_f16.ptr,
                    (n_embd * n_ff_sh) as i64,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    sc.mma_a_f16.ptr,
                    sc.mma_b_f16.ptr,
                    sh_sub,
                    m_sub as i32,
                    n_embd as i32,
                    n_ff_sh as i32,
                    1.0,
                    0.0,
                    stream,
                );
            }
        }
    }

    // 4. Host wait for moe_ids/moe_rw + counting-sort bucketing across cfg.n_expert
    sc.ev_pf_done[0].sync()?;
    if let Some(t0) = t_rsh {
        stream.sync()?;
        ctx.prof.moe_router_sh_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    let total_routes = m * topk;
    let host_ids = std::slice::from_raw_parts(sc.prefill_ids_pin.host_ptr(), total_routes);
    let host_rw = std::slice::from_raw_parts(sc.prefill_rw_pin.host_ptr(), total_routes);

    let lc = &mut ctx.cache[l_rel];
    if m > 1 {
        let tail_toks = m.min(512);
        let tail_start = (m - tail_toks) * topk;
        lc.note_route(ctx.cache_tick, &host_ids[tail_start..]);
    }

    let n_expert = cfg.n_expert;
    let mut counts = vec![0usize; n_expert];
    for &raw_eid in host_ids {
        let e = raw_eid as usize;
        if e < n_expert {
            counts[e] += 1;
        }
    }
    let mut offsets = vec![0usize; n_expert];
    let mut run_off = 0usize;
    for e in 0..n_expert {
        offsets[e] = run_off;
        run_off += counts[e];
    }
    let mut cursors = offsets.clone();
    let mut bucket_rows = vec![0i32; run_off];
    let mut bucket_wts = vec![0.0f32; run_off];
    for t in 0..m {
        let base = t * topk;
        for slot in 0..topk {
            let e = host_ids[base + slot] as usize;
            if e < n_expert {
                let dst = cursors[e];
                cursors[e] = dst + 1;
                bucket_rows[dst] = t as i32;
                bucket_wts[dst] = host_rw[base + slot];
            }
        }
    }

    let exp_ptrs_pin = std::slice::from_raw_parts_mut(sc.prefill_exp_ptrs_pin.host_ptr(), n_expert);
    let slot_off_pin = std::slice::from_raw_parts_mut(sc.prefill_slot_off_pin.host_ptr(), n_expert);
    let slot_nt_pin = std::slice::from_raw_parts_mut(sc.prefill_slot_nt_pin.host_ptr(), n_expert);
    let rows_pin = std::slice::from_raw_parts_mut(sc.prefill_rows_pin.host_ptr(), total_routes);
    let wts_pin = std::slice::from_raw_parts_mut(sc.prefill_wts_pin.host_ptr(), total_routes);

    // Pack vram_small + cold_small at offsets 0..small_rows <= views.small_cap_rows,
    // then vram_large (sorted desc by nt) + cold_large (sorted desc by nt).
    let mut n_vram_small = 0usize;
    let mut vram_small_rows = 0usize;
    let mut vram_large_raw: Vec<(u64, usize, usize)> = Vec::new();
    let mut cold_raw: Vec<(usize, usize, usize)> = Vec::new(); // (e, bucket_off, nt)
    let mut cur_off = 0usize;

    for e in 0..n_expert {
        let nt = counts[e];
        if nt == 0 {
            continue;
        }
        let b_off = offsets[e];
        if lc.res_mask_host[e] != 0 {
            if nt <= views.small_max_nt {
                let ptr = lc.vram_ptrs_host[e];
                rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
                wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
                exp_ptrs_pin[n_vram_small] = ptr;
                slot_off_pin[n_vram_small] = cur_off as i32;
                slot_nt_pin[n_vram_small] = nt as i32;
                n_vram_small += 1;
                vram_small_rows += nt;
                cur_off += nt;
            } else {
                vram_large_raw.push((lc.vram_ptrs_host[e], b_off, nt));
            }
        } else {
            cold_raw.push((e, b_off, nt));
        }
    }

    cold_raw.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    let k_cold = cold_raw.len();
    let k_cold_large = cold_raw.iter().take_while(|x| x.2 > views.small_max_nt).count();
    let mut cold_row_offs = vec![0usize; k_cold];
    for (c_idx, &(_, b_off, nt)) in cold_raw.iter().enumerate() {
        if nt <= views.small_max_nt {
            rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
            wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
            cold_row_offs[c_idx] = cur_off;
            cur_off += nt;
        }
    }
    assert!(
        cur_off <= views.small_cap_rows,
        "small MoE inter rows {cur_off} exceeded small_cap_rows {}",
        views.small_cap_rows
    );

    vram_large_raw.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    let mut vram_large: Vec<(u64, usize, usize)> = Vec::with_capacity(vram_large_raw.len());
    for &(ptr, b_off, nt) in &vram_large_raw {
        rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
        wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
        vram_large.push((ptr, cur_off, nt));
        cur_off += nt;
    }

    for (c_idx, &(_, b_off, nt)) in cold_raw[..k_cold_large].iter().enumerate() {
        rows_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_rows[b_off..b_off + nt]);
        wts_pin[cur_off..cur_off + nt].copy_from_slice(&bucket_wts[b_off..b_off + nt]);
        cold_row_offs[c_idx] = cur_off;
        cur_off += nt;
    }

    let mut cold_list: Vec<ColdJob> = Vec::with_capacity(k_cold);
    for (c_idx, &(e, b_off, nt)) in cold_raw.iter().enumerate() {
        let rs = &bucket_rows[b_off..b_off + nt];
        let ws = &bucket_wts[b_off..b_off + nt];
        let row_off = cold_row_offs[c_idx];
        if c_idx >= k_cold_large {
            let sm_idx = n_vram_small + (c_idx - k_cold_large);
            slot_off_pin[sm_idx] = row_off as i32;
            slot_nt_pin[sm_idx] = nt as i32;
        }
        let tok_indices = rs.iter().map(|&r| r as usize).collect();
        let tok_wts = ws.to_vec();
        cold_list.push(ColdJob {
            e,
            row_off,
            nt,
            tok_indices,
            tok_wts,
        });
    }
    let n_slot_meta = n_vram_small + (k_cold - k_cold_large);

    // Calibrated cost-model split for 3-tier grouped MMQ + NG=3xGE=8 ring streaming (200W sustained)
    let mut n_gpu_cold = k_cold;
    if k_cold > 0 {
        let cpu_max_nt = std::env::var("AMPLLM_CPU_MAX_NT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or((m / 16).clamp(48, 320));
        let cpu_scale = 8.0 / (ctx.pool.num_threads().max(1) as f64);
        let blob_us = (lw.blob_stride as f64) / 10_800.0;
        let comp_cost = |nt: usize| -> f64 {
            if nt <= views.small_max_nt {
                18.0 + 1.55 * (nt as f64)
            } else {
                22.0 + 1.35 * (nt as f64)
            }
        };
        let vram_large_us: f64 = vram_large.iter().map(|&(_, _, nt)| comp_cost(nt)).sum();
        let vram_us = if n_vram_small > 0 || !vram_large.is_empty() {
            150.0
                + (n_vram_small as f64) * 18.0
                + (vram_small_rows as f64) * 1.55
                + vram_large_us
        } else {
            0.0
        };
        let cpu_quant_us = 60.0 + 0.7 * (m as f64);
        let out_h2d_us = ((m * n_embd * 4) as f64) / 11_500.0;

        let mut pref_gpu_comp_us = vec![0.0f64; k_cold + 1];
        for i in 0..k_cold {
            pref_gpu_comp_us[i + 1] = pref_gpu_comp_us[i] + comp_cost(cold_list[i].nt);
        }
        let mut suff_cpu_us = vec![0.0f64; k_cold + 1];
        let mut suff_ok = vec![true; k_cold + 1];
        for i in (0..k_cold).rev() {
            let nt = cold_list[i].nt;
            suff_ok[i] = suff_ok[i + 1] && nt <= cpu_max_nt;
            suff_cpu_us[i] = suff_cpu_us[i + 1] + (250.0 + 50.0 * (nt as f64)) * cpu_scale;
        }

        let mut split = k_cold;
        for cand in (0..k_cold).rev() {
            if !suff_ok[cand] {
                break;
            }
            let cpu_est = cpu_quant_us + suff_cpu_us[cand] + out_h2d_us;
            let gpu_comp_est = vram_us + pref_gpu_comp_us[cand];
            let gpu_pcie_est = (cand as f64) * blob_us;
            let gpu_est = gpu_comp_est.max(gpu_pcie_est);
            if cpu_est <= 0.96 * gpu_est {
                split = cand;
            } else {
                break;
            }
        }
        n_gpu_cold = split;
    }

    let t_moe_gpu = prof_on.then(Instant::now);

    if cur_off > 0 {
        let r_bytes = std::slice::from_raw_parts(rows_pin.as_ptr() as *const u8, cur_off * 4);
        let w_bytes = std::slice::from_raw_parts(wts_pin.as_ptr() as *const u8, cur_off * 4);
        copy_h2d_raw_async(views.moe_rows_dev.cast(), r_bytes, stream)?;
        copy_h2d_raw_async(views.moe_wts_dev.cast(), w_bytes, stream)?;
    }
    if n_vram_small > 0 {
        let ep_bytes =
            std::slice::from_raw_parts(exp_ptrs_pin.as_ptr() as *const u8, n_vram_small * 8);
        sc.moe_exp_ptrs_dev.copy_from_host_async(ep_bytes, stream)?;
    }
    if n_slot_meta > 0 {
        let so_bytes =
            std::slice::from_raw_parts(slot_off_pin.as_ptr() as *const u8, n_slot_meta * 4);
        let sn_bytes =
            std::slice::from_raw_parts(slot_nt_pin.as_ptr() as *const u8, n_slot_meta * 4);
        sc.moe_slot_off_dev.copy_from_host_async(so_bytes, stream)?;
        sc.moe_slot_nt_dev.copy_from_host_async(sn_bytes, stream)?;
    }

    let gbuf_slots = views.gbuf_slots;
    let slot_ptr = |s: usize| views.gbuf_base + (s * views.gbuf_stride) as u64;
    let (ring_groups, group_cap) = if gbuf_slots >= 24 {
        (3usize, 8usize)
    } else {
        (gbuf_slots.clamp(1, 16), 1usize)
    };
    let mut cold_groups: Vec<(usize, usize)> = Vec::with_capacity(n_gpu_cold.div_ceil(group_cap));
    let mut gs_idx = 0usize;
    while gs_idx < n_gpu_cold {
        let ge_idx = (gs_idx + group_cap).min(n_gpu_cold);
        cold_groups.push((gs_idx, ge_idx));
        gs_idx = ge_idx;
    }

    let cold_arena = &weights.cold_arena;
    let kick_group = |g_idx: usize, r_slot: usize| -> Result<()> {
        let (g_start, g_end) = cold_groups[g_idx];
        let base_s = r_slot * group_cap;
        for (i, c_idx) in (g_start..g_end).enumerate() {
            let e = cold_list[c_idx].e;
            let (hp, _) = cold_arena
                .get_ram_blob(l, e, ctx.dev)
                .ok_or_else(|| anyhow::anyhow!("missing cold blob l={l} e={e}"))?;
            let host_slice = std::slice::from_raw_parts(hp, lw.blob_stride);
            copy_h2d_raw_async(slot_ptr(base_s + i) as *mut c_void, host_slice, pf_stream)?;
        }
        sc.ev_ring_ready[r_slot].record(pf_stream)?;
        Ok(())
    };

    // Prime initial NG=3 ring groups onto pf_stream so DMA runs while VRAM experts execute
    let init_g = ring_groups.min(cold_groups.len());
    for g_idx in 0..init_g {
        kick_group(g_idx, g_idx)?;
    }

    let cpu_jobs = &cold_list[n_gpu_cold..];
    let dev_id = ctx.dev;
    let cpu_pool = ctx.pool;
    let cpu_scratch = &mut *ctx.cpu_scratch;
    let meta_cursor = &mut ctx.meta_cursor;
    let io_pin_addr = sc.prefill_io_pin.host_ptr() as usize;
    let gu_rb = lw.gu_type.row_bytes(n_embd);
    let d_rb = lw.d_type.row_bytes(n_ff_exp);

    // 5. Execute CPU cold experts concurrently with GPU VRAM + ring-grouped cold streaming
    let (cpu_res, gpu_res) = std::thread::scope(|scope| {
        let h_cpu = (!cpu_jobs.is_empty()).then(|| {
            scope.spawn(move || -> Result<(bool, f64)> {
                set_device(dev_id)?;
                sc.ev_stream_copy.sync()?;
                let t_cold_cpu = prof_on.then(Instant::now);
                let mut did_any = false;
                let mut idx = 0usize;
                let nj = cpu_jobs.len();
                let io_ptr = io_pin_addr as *mut f32;
                while idx < nj {
                    let mut end = idx;
                    let mut batch_rows = 0usize;
                    let mut count = 0usize;
                    while end < nj && count < 16 {
                        let next_nt = cpu_jobs[end].nt;
                        if count > 0 && batch_rows + next_nt > 1024 {
                            break;
                        }
                        batch_rows += next_nt;
                        end += 1;
                        count += 1;
                    }
                    let mut groups = Vec::with_capacity(end - idx);
                    for job in &cpu_jobs[idx..end] {
                        if let Some((gp, up, dp)) = cold_arena.get_ram_ptrs(l, job.e) {
                            groups.push(Glm5CpuGroupedExpert {
                                gate_ptr: gp,
                                up_ptr: up,
                                down_ptr: dp,
                                token_indices: &job.tok_indices,
                                token_weights: &job.tok_wts,
                            });
                        }
                    }
                    if !groups.is_empty() {
                        if !did_any {
                            let host_x = std::slice::from_raw_parts(io_ptr as *const f32, m * n_embd);
                            quantize_glm5_cpu_x_q8k(cpu_pool, cpu_scratch, n_embd, m, host_x);
                        }
                        let cpu_out = std::slice::from_raw_parts_mut(io_ptr, m * n_embd);
                        run_glm5_cpu_moe_grouped_prequant(
                            cpu_pool,
                            cpu_scratch,
                            lw.gu_type,
                            lw.d_type,
                            n_embd,
                            n_ff_exp,
                            clamp_lim,
                            m,
                            &groups,
                            cpu_out,
                            !did_any,
                        );
                        did_any = true;
                    }
                    idx = end;
                }
                let ms = t_cold_cpu.map_or(0.0, |t0| t0.elapsed().as_secs_f64() * 1000.0);
                Ok((did_any, ms))
            })
        });

        let gpu_closure = (|| -> Result<f64> {
            if n_vram_small > 0 {
                glm5::moe_prefill_small_gu(
                    lw.gu_type as i32,
                    sc.moe_exp_ptrs_dev.ptr.cast(),
                    sc.moe_slot_off_dev.ptr.cast(),
                    sc.moe_slot_nt_dev.ptr.cast(),
                    views.moe_rows_dev.cast(),
                    lw.u_off as u32,
                    gu_rb,
                    views.x_q8,
                    views.moe_inter_q8,
                    n_embd as i32,
                    n_ff_exp as i32,
                    n_vram_small as i32,
                    clamp_lim,
                    stream,
                );
                glm5::moe_prefill_small_down(
                    lw.d_type as i32,
                    sc.moe_exp_ptrs_dev.ptr.cast(),
                    sc.moe_slot_off_dev.ptr.cast(),
                    sc.moe_slot_nt_dev.ptr.cast(),
                    views.moe_rows_dev.cast(),
                    views.moe_wts_dev.cast(),
                    lw.d_off as u32,
                    d_rb,
                    views.moe_inter_q8,
                    views.gpu_moe_out,
                    n_embd as i32,
                    n_ff_exp as i32,
                    n_vram_small as i32,
                    stream,
                );
            }
            if !vram_large.is_empty() {
                run_moe_group_mmq(sc, views, lw, cfg, meta_cursor, &vram_large, stream)?;
            }

            let mut grp_jobs: Vec<(u64, usize, usize)> = Vec::with_capacity(group_cap);
            for (g_idx, &(g_start, g_end)) in cold_groups.iter().enumerate() {
                let r_slot = g_idx % ring_groups;
                let base_s = r_slot * group_cap;
                stream.wait_event(&sc.ev_ring_ready[r_slot])?;

                let k_g_large = cold_list[g_start..g_end]
                    .iter()
                    .take_while(|j| j.nt > views.small_max_nt)
                    .count();
                let k_g_small = (g_end - g_start) - k_g_large;

                if k_g_large > 0 {
                    grp_jobs.clear();
                    for i in 0..k_g_large {
                        let j = &cold_list[g_start + i];
                        grp_jobs.push((slot_ptr(base_s + i), j.row_off, j.nt));
                    }
                    run_moe_group_mmq(sc, views, lw, cfg, meta_cursor, &grp_jobs, stream)?;
                }
                if k_g_small > 0 {
                    let exp_ptr_dev = views.gbuf_ptrs_dev + ((base_s + k_g_large) * 8) as u64;
                    let meta_idx = n_vram_small + (g_start + k_g_large - k_cold_large);
                    let off_dev = (sc.moe_slot_off_dev.ptr as u64) + (meta_idx * 4) as u64;
                    let nt_dev = (sc.moe_slot_nt_dev.ptr as u64) + (meta_idx * 4) as u64;
                    glm5::moe_prefill_small_gu(
                        lw.gu_type as i32,
                        exp_ptr_dev as *const u64,
                        off_dev as *const i32,
                        nt_dev as *const i32,
                        views.moe_rows_dev.cast(),
                        lw.u_off as u32,
                        gu_rb,
                        views.x_q8,
                        views.moe_inter_q8,
                        n_embd as i32,
                        n_ff_exp as i32,
                        k_g_small as i32,
                        clamp_lim,
                        stream,
                    );
                    glm5::moe_prefill_small_down(
                        lw.d_type as i32,
                        exp_ptr_dev as *const u64,
                        off_dev as *const i32,
                        nt_dev as *const i32,
                        views.moe_rows_dev.cast(),
                        views.moe_wts_dev.cast(),
                        lw.d_off as u32,
                        d_rb,
                        views.moe_inter_q8,
                        views.gpu_moe_out,
                        n_embd as i32,
                        n_ff_exp as i32,
                        k_g_small as i32,
                        stream,
                    );
                }

                sc.ev_ring_free[r_slot].record(stream)?;
                let next_g = g_idx + ring_groups;
                if next_g < cold_groups.len() {
                    pf_stream.wait_event(&sc.ev_ring_free[r_slot])?;
                    kick_group(next_g, r_slot)?;
                }
            }

            if let Some(tg) = t_moe_gpu {
                stream.sync()?;
                Ok(tg.elapsed().as_secs_f64() * 1000.0)
            } else {
                Ok(0.0)
            }
        })();

        let cpu_res = match h_cpu {
            Some(h) => h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("CPU MoE prefill thread panicked"))),
            None => Ok((false, 0.0)),
        };
        (cpu_res, gpu_closure)
    });

    let gpu_ms = gpu_res?;
    let (cpu_did_any, cpu_ms) = cpu_res?;
    if prof_on {
        ctx.prof.moe_cold_gpu_ms += gpu_ms;
        ctx.prof.moe_cold_cpu_ms += cpu_ms;
        if k_cold > 0 {
            ctx.prof.gpu_cold_exps += n_gpu_cold;
            ctx.prof.gpu_cold_rows += cold_list[..n_gpu_cold].iter().map(|j| j.nt).sum::<usize>();
            ctx.prof.cpu_exps += k_cold - n_gpu_cold;
            ctx.prof.cpu_rows += cold_list[n_gpu_cold..].iter().map(|j| j.nt).sum::<usize>();
        }
    }

    // 6. In-place HC post-add back into r_buf0
    if cpu_did_any {
        let out_bytes = std::slice::from_raw_parts(
            sc.prefill_io_pin.host_ptr().cast::<u8>(),
            m * n_embd * 4,
        );
        copy_h2d_raw_async(views.x_f32.cast(), out_bytes, stream)?;
        glm5::hc_post_add(
            views.r_buf0,
            views.post,
            views.comb,
            views.gpu_moe_out,
            views.x_f32,
            std::ptr::null(),
            views.r_buf0,
            n_embd as i32,
            cfg.hc as i32,
            m as i32,
            stream,
        );
    } else {
        glm5::hc_post_add(
            views.r_buf0,
            views.post,
            views.comb,
            views.gpu_moe_out,
            std::ptr::null(),
            std::ptr::null(),
            views.r_buf0,
            n_embd as i32,
            cfg.hc as i32,
            m as i32,
            stream,
        );
    }

    Ok(())
}

fn run_half_layers_split(ctx: &mut HalfPrefillCtx<'_>, pos0: usize, m: usize) -> Result<()> {
    let weights = ctx.weights;
    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let norm_eps = cfg.rms_norm_eps;
    let hc_eps = cfg.hc_eps;
    let q8_row_bytes = (n_embd / 32) * 36;
    let views = ctx.views;
    let dbg_sync = std::env::var_os("AMPLLM_DBG_SYNC").is_some();
    let sync_chk = |tag: &str, l: usize| -> Result<()> {
        if dbg_sync {
            device_synchronize().map_err(|e| {
                anyhow::anyhow!("CUDA fail pos0={pos0} m={m} l={l} [{tag}]: {e:#}")
            })?;
        }
        Ok(())
    };

    let prof_on = prefill_prof_enabled();
    let n_half_trunk = weights.cfg.n_layers.saturating_sub(ctx.lb).min(ctx.kda.len());
    for l_rel in 0..n_half_trunk {
        let l = ctx.lb + l_rel;
        let lw = &weights.layers[l];
        let sc = ctx.sc;
        let stream = &sc.stream;

        unsafe {
            // 1..3 Attention / Mixer: use large attn_union sub-batches when lent tail has space and m >= 16
            let align256 = |x: usize| (x + 255) & !255;
            let t_attn = prof_on.then(Instant::now);
            let mut ran_large_attn = false;
            if m >= 16 && views.attn_union_bytes > 0 {
                if lw.is_kda {
                    let d_inner = cfg.d_inner();
                    let kda_fit = [1024usize, 512, 256].into_iter().find(|&csub| {
                        let s8 = align256(csub * d_inner * 4);
                        let sfa = align256(csub * cfg.kda_head_dim * 4);
                        let sbt = align256(csub * cfg.n_head * 4);
                        let smx = align256(csub * n_embd * 4);
                        5 * s8 + 2 * sfa + sbt + smx <= views.attn_union_bytes
                    });
                    if let Some(kda_sub) = kda_fit {
                        ran_large_attn = true;
                        let hca = lw.hc_attn_fn.as_ref().unwrap();
                        glm5::hc_pre_norm_q8(
                            views.r_buf0, hca.ptr as *const c_void, hca.ggml_type as i32,
                            lw.hc_attn_scale.unwrap() as *const f32, lw.hc_attn_base.unwrap() as *const f32,
                            lw.attn_norm as *const f32, n_embd as i32, hc as i32, cfg.sinkhorn_iters as i32,
                            norm_eps, hc_eps, views.post, views.comb, views.x_f32, views.x_q8,
                            m as i32, stream,
                        );
                        let mut cur = views.attn_union_base;
                        let mut take = |sz: usize| -> *mut f32 {
                            let p = cur as *mut f32;
                            cur += align256(sz) as u64;
                            p
                        };
                        let tmp_8k = take(kda_sub * d_inner * 4);
                        let qc = take(kda_sub * d_inner * 4);
                        let kc = take(kda_sub * d_inner * 4);
                        let vc = take(kda_sub * d_inner * 4);
                        let g1 = take(kda_sub * d_inner * 4);
                        let fa = take(kda_sub * cfg.kda_head_dim * 4);
                        let ga = take(kda_sub * cfg.kda_head_dim * 4);
                        let beta = take(kda_sub * cfg.n_head * 4);
                        let mix = take(kda_sub * n_embd * 4);
                        let kst = ctx.kda[l_rel].as_ref().unwrap();
                        let fb = lw.ssm_f_b.as_ref().unwrap();
                        let gb = lw.ssm_g_b.as_ref().unwrap();
                        let aq = lw.attn_q.as_ref().unwrap();
                        let ak = lw.attn_k.as_ref().unwrap();
                        let av = lw.attn_v.as_ref().unwrap();
                        let fa_w = lw.ssm_f_a.as_ref().unwrap();
                        let ga_w = lw.ssm_g_a.as_ref().unwrap();
                        let sb_w = lw.ssm_beta.as_ref().unwrap();
                        let need_x_f16 = [aq, ak, av, fa_w, ga_w, sb_w]
                            .iter()
                            .any(|w| !mmq_ok(w.ggml_type, n_embd));

                        for off in (0..m).step_by(kda_sub) {
                            let m_sub = (m - off).min(kda_sub);
                            let r_sub = views.r_buf0.add(off * hc * n_embd);
                            let post_sub = views.post.add(off * hc);
                            let comb_sub = views.comb.add(off * hc * hc);
                            let xf_sub = views.x_f32.add(off * n_embd);
                            let xq_sub = (views.x_q8 as *const u8).add(off * q8_row_bytes).cast::<c_void>();
                            let x_f16 = g1 as *mut c_void;
                            if need_x_f16 {
                                glm5::f32_to_f16(xf_sub, x_f16, (m_sub * n_embd) as i32, stream);
                            }

                            linear_from_a_q8(sc, aq, xf_sub, xq_sub, if need_x_f16 { x_f16 } else { std::ptr::null() }, tmp_8k, d_inner, n_embd, m_sub, 1.0, stream);
                            glm5::kda_conv_silu_l2(
                                tmp_8k, lw.ssm_conv_q.unwrap() as *const f32, kst.conv_q.ptr.cast(),
                                d_inner as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                                true, m_sub as i32, qc, stream,
                            );
                            linear_from_a_q8(sc, ak, xf_sub, xq_sub, if need_x_f16 { x_f16 } else { std::ptr::null() }, tmp_8k, d_inner, n_embd, m_sub, 1.0, stream);
                            glm5::kda_conv_silu_l2(
                                tmp_8k, lw.ssm_conv_k.unwrap() as *const f32, kst.conv_k.ptr.cast(),
                                d_inner as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                                true, m_sub as i32, kc, stream,
                            );
                            linear_from_a_q8(sc, av, xf_sub, xq_sub, if need_x_f16 { x_f16 } else { std::ptr::null() }, tmp_8k, d_inner, n_embd, m_sub, 1.0, stream);
                            glm5::kda_conv_silu_l2(
                                tmp_8k, lw.ssm_conv_v.unwrap() as *const f32, kst.conv_v.ptr.cast(),
                                d_inner as i32, cfg.d_conv as i32, cfg.kda_head_dim as i32, cfg.n_head as i32,
                                false, m_sub as i32, vc, stream,
                            );
                            linear_from_a_q8(sc, fa_w, xf_sub, xq_sub, if need_x_f16 { x_f16 } else { std::ptr::null() }, fa, cfg.kda_head_dim, n_embd, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, ga_w, xf_sub, xq_sub, if need_x_f16 { x_f16 } else { std::ptr::null() }, ga, cfg.kda_head_dim, n_embd, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, sb_w, xf_sub, xq_sub, if need_x_f16 { x_f16 } else { std::ptr::null() }, beta, cfg.n_head, n_embd, m_sub, 1.0, stream);
                            if mmq_ok(fb.ggml_type, cfg.kda_head_dim)
                                && m_sub * (cfg.kda_head_dim / 32) * 36 <= sc.mma_a_f16.bytes
                            {
                                launch_quantize_q8_1(fa, sc.mma_a_f16.ptr, m_sub as i32, cfg.kda_head_dim as i32, stream.raw);
                                linear_from_a_q8(sc, fb, fa, sc.mma_a_f16.ptr, std::ptr::null(), tmp_8k, d_inner, cfg.kda_head_dim, m_sub, 1.0, stream);
                            } else {
                                linear_tiled_f16(sc, fb, fa, tmp_8k, d_inner, cfg.kda_head_dim, m_sub, 1.0, stream);
                            }
                            glm5::kda_gate_beta(
                                tmp_8k, lw.ssm_dt_bias.unwrap() as *const f32, lw.ssm_a.unwrap() as *const f32,
                                cfg.kda_gate_lb, cfg.kda_head_dim as i32, cfg.n_head as i32, m_sub as i32,
                                g1, beta, stream,
                            );
                            glm5::kda_recurrence(
                                qc, kc, vc, g1, beta, kst.s_mat.ptr.cast(),
                                cfg.n_head as i32, cfg.kda_head_dim as i32, m_sub as i32, tmp_8k, stream,
                            );
                            if mmq_ok(gb.ggml_type, cfg.kda_head_dim)
                                && m_sub * (cfg.kda_head_dim / 32) * 36 <= sc.mma_a_f16.bytes
                            {
                                launch_quantize_q8_1(ga, sc.mma_a_f16.ptr, m_sub as i32, cfg.kda_head_dim as i32, stream.raw);
                                linear_from_a_q8(sc, gb, ga, sc.mma_a_f16.ptr, std::ptr::null(), kc, d_inner, cfg.kda_head_dim, m_sub, 1.0, stream);
                            } else {
                                linear_tiled_f16(sc, gb, ga, kc, d_inner, cfg.kda_head_dim, m_sub, 1.0, stream);
                            }
                            if mmq_ok(lw.attn_output.ggml_type, d_inner) {
                                glm5::kda_out_gate_q8(
                                    tmp_8k, lw.ssm_norm.unwrap() as *const f32, kc,
                                    cfg.kda_head_dim as i32, cfg.n_head as i32, m_sub as i32, norm_eps,
                                    std::ptr::null_mut(), x_f16, stream,
                                );
                                linear_from_a_q8(sc, &lw.attn_output, std::ptr::null(), x_f16, std::ptr::null(), mix, n_embd, d_inner, m_sub, 1.0, stream);
                            } else {
                                glm5::kda_out_gate_q8(
                                    tmp_8k, lw.ssm_norm.unwrap() as *const f32, kc,
                                    cfg.kda_head_dim as i32, cfg.n_head as i32, m_sub as i32, norm_eps,
                                    qc, std::ptr::null_mut(), stream,
                                );
                                glm5::f32_to_f16(qc, x_f16, (m_sub * d_inner) as i32, stream);
                                linear_from_a_f16(sc, &lw.attn_output, x_f16, mix, n_embd, d_inner, m_sub, 1.0, stream);
                            }
                            sync_chk("kda", l)?;
                            glm5::hc_post_add(
                                r_sub, post_sub, comb_sub, mix, std::ptr::null(), std::ptr::null(),
                                r_sub, n_embd as i32, hc as i32, m_sub as i32, stream,
                            );
                        }
                    }
                } else {
                    let qk_out = cfg.n_head * cfg.qk_nope_head_dim;
                    let v_tot = cfg.n_head * cfg.v_head_dim;
                    let qfull_elems = qk_out.max(v_tot);
                    let n_sel = cfg.idx_top_k + cfg.idx_kpool - 1;
                    let dsa_fit = [512usize, 256].into_iter().find(|&csub| {
                        let need = align256(csub * cfg.q_lora_rank * 4)
                            + align256(csub * cfg.kv_lora_rank * 4)
                            + align256(csub * qfull_elems * 4)
                            + align256(csub * cfg.n_head * cfg.kv_lora_rank * 4)
                            + 2 * align256(csub * cfg.idx_key_dim * 4)
                            + align256(csub * cfg.idx_heads * cfg.idx_key_dim * 4)
                            + align256(csub * cfg.idx_heads * 4)
                            + align256(csub * weights.pool_cap * 4)
                            + align256(csub * n_sel * 4)
                            + align256(csub * n_embd * 4);
                        need <= views.attn_union_bytes
                    });
                    if let Some(dsa_sub) = dsa_fit {
                        ran_large_attn = true;
                        let hca = lw.hc_attn_fn.as_ref().unwrap();
                        glm5::hc_pre_norm_q8(
                            views.r_buf0, hca.ptr as *const c_void, hca.ggml_type as i32,
                            lw.hc_attn_scale.unwrap() as *const f32, lw.hc_attn_base.unwrap() as *const f32,
                            lw.attn_norm as *const f32, n_embd as i32, hc as i32, cfg.sinkhorn_iters as i32,
                            norm_eps, hc_eps, views.post, views.comb, views.x_f32, views.x_q8,
                            m as i32, stream,
                        );
                        let mut cur = views.attn_union_base;
                        let mut take = |sz: usize| -> u64 {
                            let p = cur;
                            cur += align256(sz) as u64;
                            p
                        };
                        let qr = take(dsa_sub * cfg.q_lora_rank * 4) as *mut f32;
                        let kv = take(dsa_sub * cfg.kv_lora_rank * 4) as *mut f32;
                        let q_full = take(dsa_sub * qfull_elems * 4) as *mut f32;
                        let q_abs = take(dsa_sub * cfg.n_head * cfg.kv_lora_rank * 4) as *mut f32;
                        let ik_tmp = take(dsa_sub * cfg.idx_key_dim * 4) as *mut f32;
                        let ig_tmp = take(dsa_sub * cfg.idx_key_dim * 4) as *mut f32;
                        let iq = take(dsa_sub * cfg.idx_heads * cfg.idx_key_dim * 4) as *mut f32;
                        let iw = take(dsa_sub * cfg.idx_heads * 4) as *mut f32;
                        let dsa_scores = take(dsa_sub * weights.pool_cap * 4) as *mut f32;
                        let dsa_cells = take(dsa_sub * n_sel * 4) as *mut i32;
                        let mix = take(dsa_sub * n_embd * 4) as *mut f32;
                        let dst = ctx.dsa[l_rel].as_ref().unwrap();
                        let prescale = 1.0f32 / ((cfg.idx_key_dim * cfg.idx_heads) as f32).sqrt();
                        let ram_lat_dev = dst
                            .ram_lat8
                            .as_ref()
                            .map_or(std::ptr::null_mut(), |b| b.dev_ptr_for_device(lw.device).unwrap_or(std::ptr::null_mut()));
                        let top_pools_cap = cfg.idx_top_k / cfg.idx_kpool;
                        let qa_w = lw.attn_q_a.as_ref().unwrap();
                        let kva_w = lw.attn_kv_a_mqa.as_ref().unwrap();
                        let ik_w = lw.idx_k.as_ref().unwrap();
                        let ig_w = lw.idx_gate.as_ref().unwrap();
                        let ip_w = lw.idx_proj.as_ref().unwrap();
                        let need_x_f16 = [qa_w, kva_w, ik_w, ig_w, ip_w]
                            .iter()
                            .any(|w| !mmq_ok(w.ggml_type, n_embd));
                        let qb = lw.attn_q_b.as_ref().unwrap();
                        let iqb = lw.idx_q_b.as_ref().unwrap();
                        let qb_mmq = mmq_ok(qb.ggml_type, cfg.q_lora_rank);
                        let iqb_mmq = mmq_ok(iqb.ggml_type, cfg.q_lora_rank);

                        for off in (0..m).step_by(dsa_sub) {
                            let m_sub = (m - off).min(dsa_sub);
                            let pos_sub = pos0 + off;
                            let r_sub = views.r_buf0.add(off * hc * n_embd);
                            let post_sub = views.post.add(off * hc);
                            let comb_sub = views.comb.add(off * hc * hc);
                            let xf_sub = views.x_f32.add(off * n_embd);
                            let xq_sub = (views.x_q8 as *const u8).add(off * q8_row_bytes).cast::<c_void>();
                            let pos_arr_sub = views.pos_arr_dev.add(off);
                            let q_abs_f16 = q_abs as *mut c_void;

                            if need_x_f16 {
                                glm5::f32_to_f16(xf_sub, q_abs_f16, (m_sub * n_embd) as i32, stream);
                            }
                            let xf16_opt = if need_x_f16 { q_abs_f16 as *const c_void } else { std::ptr::null() };
                            linear_from_a_q8(sc, qa_w, xf_sub, xq_sub, xf16_opt, qr, cfg.q_lora_rank, n_embd, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, kva_w, xf_sub, xq_sub, xf16_opt, kv, cfg.kv_lora_rank, n_embd, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, ik_w, xf_sub, xq_sub, xf16_opt, ik_tmp, cfg.idx_key_dim, n_embd, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, ig_w, xf_sub, xq_sub, xf16_opt, ig_tmp, cfg.idx_key_dim, n_embd, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, ip_w, xf_sub, xq_sub, xf16_opt, iw, cfg.idx_heads, n_embd, m_sub, prescale, stream);

                            let qb_f16 = !qb_mmq;
                            let iqb_f16 = !iqb_mmq;
                            let need_qr_q8 = !qb_f16 || !iqb_f16;
                            let need_qr_f16 = qb_f16 || iqb_f16;
                            let qr_q8_ptr = if need_qr_f16 {
                                (((q_abs_f16 as usize) + m_sub * cfg.q_lora_rank * 2 + 255) & !255) as *mut c_void
                            } else {
                                q_abs_f16
                            };

                            glm5::rms_norm_q8(
                                qr, std::ptr::null(), lw.attn_q_a_norm.unwrap() as *const f32,
                                norm_eps, cfg.q_lora_rank as i32, false, qr,
                                if need_qr_q8 { qr_q8_ptr } else { std::ptr::null_mut() },
                                m_sub as i32, stream,
                            );
                            glm5::dsa_prep_kv_ik(
                                kv, lw.attn_kv_a_norm.unwrap() as *const f32, dst.vram_lat8.ptr, ram_lat_dev,
                                weights.dsa_res_cap as i32, ik_tmp, lw.idx_k_norm_w.unwrap() as *const f32,
                                lw.idx_k_norm_b.unwrap() as *const f32, dst.ik_ring.ptr.cast(), ig_tmp,
                                dst.ig_ring.ptr.cast(), cfg.kv_lora_rank as i32, cfg.idx_key_dim as i32,
                                pos_sub as i32, 516, norm_eps, m_sub as i32, stream,
                            );
                            let pool0 = pos_sub / cfg.idx_kpool;
                            let pool_end = (pos_sub + m_sub) / cfg.idx_kpool;
                            let n_new_pools = pool_end.saturating_sub(pool0);
                            if n_new_pools > 0 {
                                glm5::dsa_pool(
                                    dst.ik_ring.ptr.cast(), dst.ig_ring.ptr.cast(),
                                    lw.idx_ape.unwrap() as *const f32, cfg.idx_key_dim as i32,
                                    cfg.idx_kpool as i32, pool0 as i32, n_new_pools as i32, 516,
                                    dst.pooled_f16.ptr, stream,
                                );
                            }
                            if need_qr_f16 {
                                glm5::f32_to_f16(qr, q_abs_f16, (m_sub * cfg.q_lora_rank) as i32, stream);
                            }
                            let qrf16_opt = if need_qr_f16 { q_abs_f16 as *const c_void } else { std::ptr::null() };
                            linear_from_a_q8(sc, qb, qr, qr_q8_ptr, qrf16_opt, q_full, qk_out, cfg.q_lora_rank, m_sub, 1.0, stream);
                            linear_from_a_q8(sc, iqb, qr, qr_q8_ptr, qrf16_opt, iq, cfg.idx_heads * cfg.idx_key_dim, cfg.q_lora_rank, m_sub, 1.0, stream);
                            glm5::dsa_q_absorb(
                                lw.attn_k_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32,
                                q_full, q_abs, cfg.qk_nope_head_dim as i32, cfg.kv_lora_rank as i32,
                                cfg.n_head as i32, m_sub as i32, stream,
                            );
                            glm5::dsa_score_select(
                                iq, dst.pooled_f16.ptr, iw, cfg.idx_key_dim as i32, cfg.idx_heads as i32,
                                cfg.idx_kpool as i32, top_pools_cap as i32, cfg.idx_select_tail,
                                n_sel as i32, pos_arr_sub, dsa_scores, dsa_cells, m_sub as i32,
                                pool_end as i32, stream,
                            );
                            sync_chk("dsa_score_select", l)?;
                            if m_sub > 1 {
                                glm5::dsa_mla_prefill_mma(
                                    q_abs, dst.vram_lat8.ptr, ram_lat_dev, weights.dsa_res_cap as i32,
                                    dsa_cells, lw.attn_v_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32,
                                    cfg.kv_lora_rank as i32, cfg.n_head as i32, cfg.v_head_dim as i32,
                                    cfg.qk_nope_head_dim as i32, n_sel as i32, q_abs, q_full,
                                    m_sub as i32, stream,
                                );
                            } else {
                                glm5::dsa_mla_attn(
                                    q_abs, dst.vram_lat8.ptr, ram_lat_dev, weights.dsa_res_cap as i32,
                                    dsa_cells, lw.attn_v_b_f16.unwrap() as *const c_void, GgmlType::F16 as i32,
                                    cfg.kv_lora_rank as i32, cfg.n_head as i32, cfg.v_head_dim as i32,
                                    cfg.qk_nope_head_dim as i32, n_sel as i32, q_full, m_sub as i32, stream,
                                );
                            }
                            sync_chk("dsa_mla_attn", l)?;
                            if mmq_ok(lw.attn_output.ggml_type, v_tot) {
                                launch_quantize_q8_1(q_full, q_abs_f16, m_sub as i32, v_tot as i32, stream.raw);
                                linear_from_a_q8(sc, &lw.attn_output, q_full, q_abs_f16, std::ptr::null(), mix, n_embd, v_tot, m_sub, 1.0, stream);
                            } else {
                                glm5::f32_to_f16(q_full, q_abs_f16, (m_sub * v_tot) as i32, stream);
                                linear_from_a_f16(sc, &lw.attn_output, q_abs_f16, mix, n_embd, v_tot, m_sub, 1.0, stream);
                            }
                            sync_chk("dsa_out", l)?;
                            glm5::hc_post_add(
                                r_sub, post_sub, comb_sub, mix, std::ptr::null(), std::ptr::null(),
                                r_sub, n_embd as i32, hc as i32, m_sub as i32, stream,
                            );
                        }
                    }
                }
            }
            if !ran_large_attn {
            for off in (0..m).step_by(sc.max_batch) {
                let m_sub = (m - off).min(sc.max_batch);
                let pos_sub = pos0 + off;
                let r_sub = views.r_buf0.add(off * hc * n_embd);
                let post_sub = views.post.add(off * hc);
                let comb_sub = views.comb.add(off * hc * hc);
                let xf_sub = views.x_f32.add(off * n_embd);
                let xq_sub = (views.x_q8 as *mut u8).add(off * q8_row_bytes).cast::<c_void>();
                let mix_sub = sc.mixer.ptr.cast::<f32>();
                let pos_arr_sub = views.pos_arr_dev.add(off);

                let hca = lw.hc_attn_fn.as_ref().unwrap();
                glm5::hc_pre_norm_q8(
                    r_sub,
                    hca.ptr as *const c_void,
                    hca.ggml_type as i32,
                    lw.hc_attn_scale.unwrap() as *const f32,
                    lw.hc_attn_base.unwrap() as *const f32,
                    lw.attn_norm as *const f32,
                    n_embd as i32,
                    hc as i32,
                    cfg.sinkhorn_iters as i32,
                    norm_eps,
                    hc_eps,
                    post_sub,
                    comb_sub,
                    xf_sub,
                    xq_sub,
                    m_sub as i32,
                    stream,
                );

                if lw.is_kda {
                    let kst = ctx.kda[l_rel].as_ref().unwrap();
                    let aq = lw.attn_q.as_ref().unwrap();
                    let ak = lw.attn_k.as_ref().unwrap();
                    let av = lw.attn_v.as_ref().unwrap();
                    let fa = lw.ssm_f_a.as_ref().unwrap();
                    let ga = lw.ssm_g_a.as_ref().unwrap();
                    let sb = lw.ssm_beta.as_ref().unwrap();

                    linear_batch(sc, aq, xf_sub, xq_sub, sc.q_p.ptr.cast(), cfg.d_inner(), n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, ak, xf_sub, xq_sub, sc.k_p.ptr.cast(), cfg.d_inner(), n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, av, xf_sub, xq_sub, sc.v_p.ptr.cast(), cfg.d_inner(), n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, fa, xf_sub, xq_sub, sc.f_a.ptr.cast(), cfg.kda_head_dim, n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, ga, xf_sub, xq_sub, sc.g_a.ptr.cast(), cfg.kda_head_dim, n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, sb, xf_sub, xq_sub, sc.beta.ptr.cast(), cfg.n_head, n_embd, m_sub, 1.0, stream);

                    glm5::kda_conv_silu_l2(
                        sc.q_p.ptr.cast(),
                        lw.ssm_conv_q.unwrap() as *const f32,
                        kst.conv_q.ptr.cast(),
                        cfg.d_inner() as i32,
                        cfg.d_conv as i32,
                        cfg.kda_head_dim as i32,
                        cfg.n_head as i32,
                        true,
                        m_sub as i32,
                        sc.qc.ptr.cast(),
                        stream,
                    );
                    glm5::kda_conv_silu_l2(
                        sc.k_p.ptr.cast(),
                        lw.ssm_conv_k.unwrap() as *const f32,
                        kst.conv_k.ptr.cast(),
                        cfg.d_inner() as i32,
                        cfg.d_conv as i32,
                        cfg.kda_head_dim as i32,
                        cfg.n_head as i32,
                        true,
                        m_sub as i32,
                        sc.kc.ptr.cast(),
                        stream,
                    );
                    glm5::kda_conv_silu_l2(
                        sc.v_p.ptr.cast(),
                        lw.ssm_conv_v.unwrap() as *const f32,
                        kst.conv_v.ptr.cast(),
                        cfg.d_inner() as i32,
                        cfg.d_conv as i32,
                        cfg.kda_head_dim as i32,
                        cfg.n_head as i32,
                        false,
                        m_sub as i32,
                        sc.vc.ptr.cast(),
                        stream,
                    );

                    linear_batch(
                        sc,
                        lw.ssm_f_b.as_ref().unwrap(),
                        sc.f_a.ptr.cast(),
                        std::ptr::null(),
                        sc.g_raw.ptr.cast(),
                        cfg.d_inner(),
                        cfg.kda_head_dim,
                        m_sub,
                        1.0,
                        stream,
                    );
                    linear_batch(
                        sc,
                        lw.ssm_g_b.as_ref().unwrap(),
                        sc.g_a.ptr.cast(),
                        std::ptr::null(),
                        sc.g2.ptr.cast(),
                        cfg.d_inner(),
                        cfg.kda_head_dim,
                        m_sub,
                        1.0,
                        stream,
                    );

                    glm5::kda_gate_beta(
                        sc.g_raw.ptr.cast(),
                        lw.ssm_dt_bias.unwrap() as *const f32,
                        lw.ssm_a.unwrap() as *const f32,
                        cfg.kda_gate_lb,
                        cfg.kda_head_dim as i32,
                        cfg.n_head as i32,
                        m_sub as i32,
                        sc.g1.ptr.cast(),
                        sc.beta.ptr.cast(),
                        stream,
                    );
                    glm5::kda_recurrence(
                        sc.qc.ptr.cast(),
                        sc.kc.ptr.cast(),
                        sc.vc.ptr.cast(),
                        sc.g1.ptr.cast(),
                        sc.beta.ptr.cast(),
                        kst.s_mat.ptr.cast(),
                        cfg.n_head as i32,
                        cfg.kda_head_dim as i32,
                        m_sub as i32,
                        sc.scan.ptr.cast(),
                        stream,
                    );
                    glm5::kda_out_gate_q8(
                        sc.scan.ptr.cast(),
                        lw.ssm_norm.unwrap() as *const f32,
                        sc.g2.ptr.cast(),
                        cfg.kda_head_dim as i32,
                        cfg.n_head as i32,
                        m_sub as i32,
                        norm_eps,
                        sc.gated.ptr.cast(),
                        if m_sub < 16 || mmq_ok(lw.attn_output.ggml_type, cfg.d_inner()) {
                            sc.gated_q8.ptr
                        } else {
                            std::ptr::null_mut()
                        },
                        stream,
                    );
                    linear_batch(
                        sc,
                        &lw.attn_output,
                        sc.gated.ptr.cast(),
                        sc.gated_q8.ptr,
                        mix_sub,
                        n_embd,
                        cfg.d_inner(),
                        m_sub,
                        1.0,
                        stream,
                    );
                    sync_chk("kda", l)?;
                } else {
                    let dst = ctx.dsa[l_rel].as_ref().unwrap();
                    let prescale = 1.0f32 / ((cfg.idx_key_dim * cfg.idx_heads) as f32).sqrt();
                    linear_batch(sc, lw.attn_q_a.as_ref().unwrap(), xf_sub, xq_sub, sc.qr.ptr.cast(), cfg.q_lora_rank, n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, lw.attn_kv_a_mqa.as_ref().unwrap(), xf_sub, xq_sub, sc.kv.ptr.cast(), cfg.kv_lora_rank, n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, lw.idx_k.as_ref().unwrap(), xf_sub, xq_sub, sc.ik_tmp.ptr.cast(), cfg.idx_key_dim, n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, lw.idx_gate.as_ref().unwrap(), xf_sub, xq_sub, sc.ig_tmp.ptr.cast(), cfg.idx_key_dim, n_embd, m_sub, 1.0, stream);
                    linear_batch(sc, lw.idx_proj.as_ref().unwrap(), xf_sub, xq_sub, sc.iw.ptr.cast(), cfg.idx_heads, n_embd, m_sub, prescale, stream);

                    let qb = lw.attn_q_b.as_ref().unwrap();
                    let iqb = lw.idx_q_b.as_ref().unwrap();
                    let want_qr_q8 = m_sub < 16
                        || mmq_ok(qb.ggml_type, cfg.q_lora_rank)
                        || mmq_ok(iqb.ggml_type, cfg.q_lora_rank);
                    glm5::rms_norm_q8(
                        sc.qr.ptr.cast(),
                        std::ptr::null(),
                        lw.attn_q_a_norm.unwrap() as *const f32,
                        norm_eps,
                        cfg.q_lora_rank as i32,
                        false,
                        sc.qr.ptr.cast(),
                        if want_qr_q8 { sc.qr_q8.ptr } else { std::ptr::null_mut() },
                        m_sub as i32,
                        stream,
                    );

                    let ram_lat_dev = dst
                        .ram_lat8
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |b| b.dev_ptr_for_device(lw.device).unwrap_or(std::ptr::null_mut()));
                    glm5::dsa_prep_kv_ik(
                        sc.kv.ptr.cast(),
                        lw.attn_kv_a_norm.unwrap() as *const f32,
                        dst.vram_lat8.ptr,
                        ram_lat_dev,
                        weights.dsa_res_cap as i32,
                        sc.ik_tmp.ptr.cast(),
                        lw.idx_k_norm_w.unwrap() as *const f32,
                        lw.idx_k_norm_b.unwrap() as *const f32,
                        dst.ik_ring.ptr.cast(),
                        sc.ig_tmp.ptr.cast(),
                        dst.ig_ring.ptr.cast(),
                        cfg.kv_lora_rank as i32,
                        cfg.idx_key_dim as i32,
                        pos_sub as i32,
                        516,
                        norm_eps,
                        m_sub as i32,
                        stream,
                    );

                    let pool0 = pos_sub / cfg.idx_kpool;
                    let pool_end = (pos_sub + m_sub) / cfg.idx_kpool;
                    let n_new_pools = pool_end.saturating_sub(pool0);
                    if n_new_pools > 0 {
                        glm5::dsa_pool(
                            dst.ik_ring.ptr.cast(),
                            dst.ig_ring.ptr.cast(),
                            lw.idx_ape.unwrap() as *const f32,
                            cfg.idx_key_dim as i32,
                            cfg.idx_kpool as i32,
                            pool0 as i32,
                            n_new_pools as i32,
                            516,
                            dst.pooled_f16.ptr,
                            stream,
                        );
                    }

                    linear_batch(
                        sc,
                        lw.attn_q_b.as_ref().unwrap(),
                        sc.qr.ptr.cast(),
                        sc.qr_q8.ptr,
                        sc.q_full.ptr.cast(),
                        cfg.n_head * cfg.qk_nope_head_dim,
                        cfg.q_lora_rank,
                        m_sub,
                        1.0,
                        stream,
                    );
                    linear_batch(
                        sc,
                        lw.idx_q_b.as_ref().unwrap(),
                        sc.qr.ptr.cast(),
                        sc.qr_q8.ptr,
                        sc.iq.ptr.cast(),
                        cfg.idx_heads * cfg.idx_key_dim,
                        cfg.q_lora_rank,
                        m_sub,
                        1.0,
                        stream,
                    );

                    glm5::dsa_q_absorb(
                        lw.attn_k_b_f16.unwrap() as *const c_void,
                        GgmlType::F16 as i32,
                        sc.q_full.ptr.cast(),
                        sc.q_abs.ptr.cast(),
                        cfg.qk_nope_head_dim as i32,
                        cfg.kv_lora_rank as i32,
                        cfg.n_head as i32,
                        m_sub as i32,
                        stream,
                    );

                    let top_pools_cap = cfg.idx_top_k / cfg.idx_kpool;
                    let n_sel = cfg.idx_top_k + cfg.idx_kpool - 1;
                    glm5::dsa_score_select(
                        sc.iq.ptr.cast(),
                        dst.pooled_f16.ptr,
                        sc.iw.ptr.cast(),
                        cfg.idx_key_dim as i32,
                        cfg.idx_heads as i32,
                        cfg.idx_kpool as i32,
                        top_pools_cap as i32,
                        cfg.idx_select_tail,
                        n_sel as i32,
                        pos_arr_sub,
                        sc.dsa_scores.ptr.cast(),
                        sc.dsa_cells.ptr.cast(),
                        m_sub as i32,
                        pool_end as i32,
                        stream,
                    );
                    sync_chk("dsa_score_select", l)?;

                    if m_sub > 1 {
                        glm5::dsa_mla_prefill_mma(
                            sc.q_abs.ptr.cast(),
                            dst.vram_lat8.ptr,
                            ram_lat_dev,
                            weights.dsa_res_cap as i32,
                            sc.dsa_cells.ptr.cast(),
                            lw.attn_v_b_f16.unwrap() as *const c_void,
                            GgmlType::F16 as i32,
                            cfg.kv_lora_rank as i32,
                            cfg.n_head as i32,
                            cfg.v_head_dim as i32,
                            cfg.qk_nope_head_dim as i32,
                            n_sel as i32,
                            sc.q_abs.ptr.cast(),
                            sc.dsa_attn.ptr.cast(),
                            m_sub as i32,
                            stream,
                        );
                    } else {
                        glm5::dsa_mla_attn(
                            sc.q_abs.ptr.cast(),
                            dst.vram_lat8.ptr,
                            ram_lat_dev,
                            weights.dsa_res_cap as i32,
                            sc.dsa_cells.ptr.cast(),
                            lw.attn_v_b_f16.unwrap() as *const c_void,
                            GgmlType::F16 as i32,
                            cfg.kv_lora_rank as i32,
                            cfg.n_head as i32,
                            cfg.v_head_dim as i32,
                            cfg.qk_nope_head_dim as i32,
                            n_sel as i32,
                            sc.dsa_attn.ptr.cast(),
                            m_sub as i32,
                            stream,
                        );
                    }
                    sync_chk("dsa_mla_attn", l)?;

                    let v_tot = cfg.n_head * cfg.v_head_dim;
                    if m_sub < 16 || mmq_ok(lw.attn_output.ggml_type, v_tot) {
                        launch_quantize_q8_1(
                            sc.dsa_attn.ptr.cast(),
                            sc.dsa_attn_q8.ptr,
                            m_sub as i32,
                            v_tot as i32,
                            stream.raw,
                        );
                    }
                    linear_batch(
                        sc,
                        &lw.attn_output,
                        sc.dsa_attn.ptr.cast(),
                        sc.dsa_attn_q8.ptr,
                        mix_sub,
                        n_embd,
                        v_tot,
                        m_sub,
                        1.0,
                        stream,
                    );
                    sync_chk("dsa_out", l)?;
                }

                // In-place HC post (attn half): r_buf[0] -> r_buf[0]
                glm5::hc_post_add(
                    r_sub,
                    post_sub,
                    comb_sub,
                    mix_sub,
                    std::ptr::null(),
                    std::ptr::null(),
                    r_sub,
                    n_embd as i32,
                    hc as i32,
                    m_sub as i32,
                    stream,
                );
            }
            }
            if let Some(t0) = t_attn {
                stream.sync()?;
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                if lw.is_kda {
                    ctx.prof.kda_ms += dt;
                } else {
                    ctx.prof.dsa_ms += dt;
                }
            }

            // 4..5 FFN half: Dense FFN sub-chunked by max_batch with in-place hc_post_add,
            // or full-chunk MoE Hybrid (always populates x_q8 for small_gu kernel).
            let hcf = lw.hc_ffn_fn.as_ref().unwrap();
            if !lw.is_moe {
                let t_dense = prof_on.then(Instant::now);
                let fg = lw.ffn_gate.as_ref().unwrap();
                let fu = lw.ffn_up.as_ref().unwrap();
                let fd = lw.ffn_down.as_ref().unwrap();
                let q8_ff_dense_rb = (cfg.n_ff_dense / 32) * 36;
                let dense_fit = if m >= 256
                    && views.attn_union_bytes > 0
                    && mmq_ok(fg.ggml_type, n_embd)
                    && mmq_ok(fu.ggml_type, n_embd)
                    && mmq_ok(fd.ggml_type, cfg.n_ff_dense)
                {
                    [1024usize, 512, 256, 128].into_iter().find(|&dt| {
                        let g_sz = align256(dt * cfg.n_ff_dense * 4);
                        let h_sz = align256(dt * q8_ff_dense_rb);
                        let o_sz = align256(dt * n_embd * 4);
                        2 * g_sz + h_sz + o_sz <= views.attn_union_bytes
                    })
                } else {
                    None
                };
                if let Some(d_tile) = dense_fit {
                    glm5::hc_pre_norm_q8(
                        views.r_buf0,
                        hcf.ptr as *const c_void,
                        hcf.ggml_type as i32,
                        lw.hc_ffn_scale.unwrap() as *const f32,
                        lw.hc_ffn_base.unwrap() as *const f32,
                        lw.ffn_norm as *const f32,
                        n_embd as i32,
                        hc as i32,
                        cfg.sinkhorn_iters as i32,
                        norm_eps,
                        hc_eps,
                        views.post,
                        views.comb,
                        views.x_f32,
                        views.x_q8,
                        m as i32,
                        stream,
                    );
                    let mut cur = views.attn_union_base;
                    let mut take = |sz: usize| -> *mut c_void {
                        let p = cur as *mut c_void;
                        cur += align256(sz) as u64;
                        p
                    };
                    let gate_buf = take(d_tile * cfg.n_ff_dense * 4) as *mut f32;
                    let up_buf = take(d_tile * cfg.n_ff_dense * 4) as *mut f32;
                    let hq8_buf = take(d_tile * q8_ff_dense_rb);
                    let out_buf = take(d_tile * n_embd * 4) as *mut f32;
                    let fg_rb = fg.ggml_type.row_bytes(n_embd);
                    let fu_rb = fu.ggml_type.row_bytes(n_embd);
                    let fd_rb = fd.ggml_type.row_bytes(cfg.n_ff_dense);
                    for off in (0..m).step_by(d_tile) {
                        let m_sub = (m - off).min(d_tile);
                        let r_sub = views.r_buf0.add(off * hc * n_embd);
                        let post_sub = views.post.add(off * hc);
                        let comb_sub = views.comb.add(off * hc * hc);
                        let xq_sub = (views.x_q8 as *const u8).add(off * q8_row_bytes).cast::<c_void>();
                        glm5::gemm_mmq_mma(
                            fg.ggml_type as i32,
                            xq_sub,
                            fg.ptr as *const c_void,
                            fg_rb,
                            gate_buf,
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            m_sub as i32,
                            cfg.n_ff_dense as i32,
                            n_embd as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                        glm5::gemm_mmq_mma(
                            fu.ggml_type as i32,
                            xq_sub,
                            fu.ptr as *const c_void,
                            fu_rb,
                            up_buf,
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            m_sub as i32,
                            cfg.n_ff_dense as i32,
                            n_embd as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                        glm5::swiglu_clamp_q8(
                            gate_buf,
                            up_buf,
                            cfg.swiglu_clamp_shexp,
                            cfg.n_ff_dense as i32,
                            std::ptr::null_mut(),
                            hq8_buf,
                            m_sub as i32,
                            stream,
                        );
                        glm5::gemm_mmq_mma(
                            fd.ggml_type as i32,
                            hq8_buf,
                            fd.ptr as *const c_void,
                            fd_rb,
                            out_buf,
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            m_sub as i32,
                            n_embd as i32,
                            cfg.n_ff_dense as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                        glm5::hc_post_add(
                            r_sub,
                            post_sub,
                            comb_sub,
                            out_buf,
                            std::ptr::null(),
                            std::ptr::null(),
                            r_sub,
                            n_embd as i32,
                            hc as i32,
                            m_sub as i32,
                            stream,
                        );
                    }
                } else {
                for off in (0..m).step_by(sc.max_batch) {
                    let m_sub = (m - off).min(sc.max_batch);
                    let r_sub = views.r_buf0.add(off * hc * n_embd);
                    let post_sub = views.post.add(off * hc);
                    let comb_sub = views.comb.add(off * hc * hc);
                    let xf_sub = views.x_f32.add(off * n_embd);
                    let xq_sub = (views.x_q8 as *mut u8).add(off * q8_row_bytes).cast::<c_void>();
                    let sh_sub = sc.sh_out.ptr.cast::<f32>();

                    glm5::hc_pre_norm_q8(
                        r_sub,
                        hcf.ptr as *const c_void,
                        hcf.ggml_type as i32,
                        lw.hc_ffn_scale.unwrap() as *const f32,
                        lw.hc_ffn_base.unwrap() as *const f32,
                        lw.ffn_norm as *const f32,
                        n_embd as i32,
                        hc as i32,
                        cfg.sinkhorn_iters as i32,
                        norm_eps,
                        hc_eps,
                        post_sub,
                        comb_sub,
                        xf_sub,
                        xq_sub,
                        m_sub as i32,
                        stream,
                    );
                    linear_batch(
                        sc,
                        fg,
                        xf_sub,
                        xq_sub,
                        sc.gate.ptr.cast(),
                        cfg.n_ff_dense,
                        n_embd,
                        m_sub,
                        1.0,
                        stream,
                    );
                    linear_batch(
                        sc,
                        fu,
                        xf_sub,
                        xq_sub,
                        sc.up.ptr.cast(),
                        cfg.n_ff_dense,
                        n_embd,
                        m_sub,
                        1.0,
                        stream,
                    );
                    if m_sub < 16 || mmq_ok(fd.ggml_type, cfg.n_ff_dense) {
                        glm5::swiglu_clamp_q8(
                            sc.gate.ptr.cast(),
                            sc.up.ptr.cast(),
                            cfg.swiglu_clamp_shexp,
                            cfg.n_ff_dense as i32,
                            std::ptr::null_mut(),
                            sc.h_q8.ptr,
                            m_sub as i32,
                            stream,
                        );
                        linear_batch(
                            sc,
                            fd,
                            sc.h_f32.ptr.cast(),
                            sc.h_q8.ptr,
                            sh_sub,
                            n_embd,
                            cfg.n_ff_dense,
                            m_sub,
                            1.0,
                            stream,
                        );
                    } else {
                        glm5::swiglu_clamp_to_f16(
                            sc.gate.ptr.cast(),
                            sc.up.ptr.cast(),
                            cfg.swiglu_clamp_shexp,
                            (m_sub * cfg.n_ff_dense) as i32,
                            sc.mma_a_f16.ptr,
                            stream,
                        );
                        glm5::dequant_to_f16(
                            fd.ggml_type as i32,
                            fd.ptr as *const c_void,
                            sc.mma_b_f16.ptr,
                            (n_embd * cfg.n_ff_dense) as i64,
                            stream,
                        );
                        glm5::gemm_f16_mma_nt(
                            sc.mma_a_f16.ptr,
                            sc.mma_b_f16.ptr,
                            sh_sub,
                            m_sub as i32,
                            n_embd as i32,
                            cfg.n_ff_dense as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                    }
                    glm5::hc_post_add(
                        r_sub,
                        post_sub,
                        comb_sub,
                        sh_sub,
                        std::ptr::null(),
                        std::ptr::null(),
                        r_sub,
                        n_embd as i32,
                        hc as i32,
                        m_sub as i32,
                        stream,
                    );
                }
                }
                if let Some(t0) = t_dense {
                    stream.sync()?;
                    ctx.prof.dense_ms += t0.elapsed().as_secs_f64() * 1000.0;
                }
            } else {
                glm5::hc_pre_norm_q8(
                    views.r_buf0,
                    hcf.ptr as *const c_void,
                    hcf.ggml_type as i32,
                    lw.hc_ffn_scale.unwrap() as *const f32,
                    lw.hc_ffn_base.unwrap() as *const f32,
                    lw.ffn_norm as *const f32,
                    n_embd as i32,
                    hc as i32,
                    cfg.sinkhorn_iters as i32,
                    norm_eps,
                    hc_eps,
                    views.post,
                    views.comb,
                    views.x_f32,
                    views.x_q8,
                    m as i32,
                    stream,
                );
                run_moe_hybrid_split(ctx, l_rel, m)?;
                sync_chk("moe_hybrid", l)?;
            }
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn image_span_covers(images: &[Glm5ImageSpan], n_embd: usize, pos: usize) -> bool {
    images
        .iter()
        .any(|span| pos >= span.start_pos && pos < span.start_pos + span.n_tokens(n_embd))
}

#[inline]
pub(crate) fn fill_token_embed_host(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    token: u32,
    pos: usize,
    images: &[Glm5ImageSpan],
    dst: &mut [f32],
) -> Result<()> {
    let n_embd = weights.cfg.n_embd;
    for span in images {
        let n_tok = span.n_tokens(n_embd);
        if pos >= span.start_pos && pos < span.start_pos + n_tok {
            let local = pos - span.start_pos;
            dst.copy_from_slice(&span.rows[local * n_embd..(local + 1) * n_embd]);
            return Ok(());
        }
    }
    dequant_embed_to_host(gguf, weights, token, dst)
}

unsafe fn mtp_cache_chunk_split(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    sc: &GpuScratch,
    dst: &DsaLayerState,
    head_x_ptr: *const c_void,
    hop_emb_pin1: &mut [f32],
    mtp_cached_pos: &mut usize,
    pos0: usize,
    next_toks: &[u32],
    images: &[Glm5ImageSpan],
) -> Result<()> {
    let Some(mtp_l) = weights.cfg.mtp_layer else {
        return Ok(());
    };
    let n_mtp = next_toks.len();
    if n_mtp == 0 {
        return Ok(());
    }
    let cfg = &weights.cfg;
    let lw = &weights.layers[mtp_l];
    let n_embd = cfg.n_embd;
    let eps = cfg.rms_norm_eps;
    let stream = &sc.stream;

    for (j, &nt) in next_toks.iter().enumerate() {
        fill_token_embed_host(
            gguf,
            weights,
            nt,
            pos0 + j + 1,
            images,
            &mut hop_emb_pin1[j * n_embd..(j + 1) * n_embd],
        )?;
    }
    let emb_bytes = std::slice::from_raw_parts(hop_emb_pin1.as_ptr().cast::<u8>(), n_mtp * n_embd * 4);
    sc.emb.copy_from_host_async(emb_bytes, stream)?;

    // Dequant eh_proj ONCE into mma_b_f16 and sub-chunk projection + DSA prep/pool by max_batch (128)
    let eh = lw.nextn_eh_proj.as_ref().unwrap();
    glm5::dequant_to_f16(
        eh.ggml_type as i32,
        eh.ptr as *const c_void,
        sc.mma_b_f16.ptr,
        (n_embd * 2 * n_embd) as i64,
        stream,
    );
    for sub_off in (0..n_mtp).step_by(sc.max_batch) {
        let m_sub = (n_mtp - sub_off).min(sc.max_batch);
        let emb_sub = sc.emb.ptr.cast::<f32>().add(sub_off * n_embd);
        let hx_sub = head_x_ptr.cast::<f32>().add(sub_off * n_embd);
        let mix_sub = sc.mixer.ptr.cast::<f32>().add(sub_off * n_embd);
        glm5::mtp_in_batch_f16(
            emb_sub,
            hx_sub,
            lw.nextn_enorm.unwrap() as *const f32,
            lw.nextn_hnorm.unwrap() as *const f32,
            eps,
            n_embd as i32,
            m_sub as i32,
            sc.mma_a_f16.ptr,
            stream,
        );
        glm5::gemm_f16_mma_nt(
            sc.mma_a_f16.ptr,
            sc.mma_b_f16.ptr,
            mix_sub,
            m_sub as i32,
            n_embd as i32,
            (2 * n_embd) as i32,
            1.0,
            0.0,
            stream,
        );
    }

    glm5::rms_norm_q8(
        sc.mixer.ptr.cast(),
        std::ptr::null(),
        lw.attn_norm as *const f32,
        eps,
        n_embd as i32,
        false,
        sc.x_f32.ptr.cast(),
        std::ptr::null_mut(),
        n_mtp as i32,
        stream,
    );

    let ram_lat_dev = dst
        .ram_lat8
        .as_ref()
        .map_or(std::ptr::null_mut(), |b| b.dev_ptr_for_device(1).unwrap_or(std::ptr::null_mut()));

    for sub_off in (0..n_mtp).step_by(sc.max_batch) {
        let m_sub = (n_mtp - sub_off).min(sc.max_batch);
        let p_sub = pos0 + sub_off;
        let xf_sub = sc.x_f32.ptr.cast::<f32>().add(sub_off * n_embd);

        linear_batch(sc, lw.attn_kv_a_mqa.as_ref().unwrap(), xf_sub, std::ptr::null(), sc.kv.ptr.cast(), cfg.kv_lora_rank, n_embd, m_sub, 1.0, stream);
        linear_batch(sc, lw.idx_k.as_ref().unwrap(), xf_sub, std::ptr::null(), sc.ik_tmp.ptr.cast(), cfg.idx_key_dim, n_embd, m_sub, 1.0, stream);
        linear_batch(sc, lw.idx_gate.as_ref().unwrap(), xf_sub, std::ptr::null(), sc.ig_tmp.ptr.cast(), cfg.idx_key_dim, n_embd, m_sub, 1.0, stream);

        glm5::dsa_prep_kv_ik(
            sc.kv.ptr.cast(),
            lw.attn_kv_a_norm.unwrap() as *const f32,
            dst.vram_lat8.ptr,
            ram_lat_dev,
            weights.dsa_res_cap as i32,
            sc.ik_tmp.ptr.cast(),
            lw.idx_k_norm_w.unwrap() as *const f32,
            lw.idx_k_norm_b.unwrap() as *const f32,
            dst.ik_ring.ptr.cast(),
            sc.ig_tmp.ptr.cast(),
            dst.ig_ring.ptr.cast(),
            cfg.kv_lora_rank as i32,
            cfg.idx_key_dim as i32,
            p_sub as i32,
            516,
            eps,
            m_sub as i32,
            stream,
        );

        let pool0 = p_sub / cfg.idx_kpool;
        let pool_end = (p_sub + m_sub) / cfg.idx_kpool;
        let n_new_pools = pool_end.saturating_sub(pool0);
        if n_new_pools > 0 {
            glm5::dsa_pool(
                dst.ik_ring.ptr.cast(),
                dst.ig_ring.ptr.cast(),
                lw.idx_ape.unwrap() as *const f32,
                cfg.idx_key_dim as i32,
                cfg.idx_kpool as i32,
                pool0 as i32,
                n_new_pools as i32,
                516,
                dst.pooled_f16.ptr,
                stream,
            );
        }
    }

    if *mtp_cached_pos < pos0 + n_mtp {
        *mtp_cached_pos = pos0 + n_mtp;
    }
    Ok(())
}

unsafe fn compute_final_head_and_mtp(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    sc1: &GpuScratch,
    dsa1: &[Option<DsaLayerState>],
    r_buf0_ptr: *const f32,
    head_x_ptr: *mut c_void,
    head_xq_ptr: *mut c_void,
    logits_dev: &ampllm_cuda_kernels::DeviceBuffer,
    argmax_pv_ptr: *mut c_void,
    argmax_pi_ptr: *mut c_void,
    tok_dev: &ampllm_cuda_kernels::DeviceBuffer,
    tok_host_ptr: *mut i32,
    hop_emb_pin1: &mut [f32],
    mtp_cached_pos: &mut usize,
    tokens: &[u32],
    images: &[Glm5ImageSpan],
    pos0: usize,
    next_tok_after_chunk: Option<u32>,
    is_final_chunk: bool,
    need_logits: bool,
) -> Result<()> {
    let cfg = &weights.cfg;
    let m = tokens.len();
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let split = weights.split_layer;
    let slice_cap = sc1.max_prefill_chunk;

    set_device(1)?;
    for s_off in (0..m).step_by(slice_cap) {
        let ms = (m - s_off).min(slice_cap);
        let is_last_slice = s_off + ms == m;

        glm5::mean_streams_norm_q8(
            r_buf0_ptr.add(s_off * hc * n_embd),
            weights.output_norm_dev.ptr.cast(),
            cfg.rms_norm_eps,
            n_embd as i32,
            hc as i32,
            head_x_ptr.cast(),
            std::ptr::null_mut(),
            ms as i32,
            &sc1.stream,
        );

        if is_final_chunk && is_last_slice {
            let last_hx = head_x_ptr.cast::<f32>().add((ms - 1) * n_embd);
            launch_quantize_q8_1(last_hx, head_xq_ptr, 1, n_embd as i32, sc1.stream.raw);
            launch_linear_q8_1(
                weights.output_weight_type as i32,
                weights.output_weight_dev.ptr,
                head_xq_ptr,
                last_hx,
                logits_dev.ptr.cast(),
                cfg.n_vocab as i32,
                n_embd as i32,
                1,
                sc1.stream.raw,
            );
            if !need_logits {
                glm5::argmax(
                    logits_dev.ptr.cast(),
                    cfg.n_vocab as i32,
                    cfg.n_vocab as i32,
                    std::ptr::null(),
                    argmax_pv_ptr.cast(),
                    argmax_pi_ptr.cast(),
                    tok_dev.ptr.cast(),
                    &sc1.stream,
                );
                let th_slice = std::slice::from_raw_parts_mut(tok_host_ptr.cast::<u8>(), 4);
                tok_dev.copy_to_host_async(th_slice, &sc1.stream)?;
            }
        }

        if let Some(mtp_l) = cfg.mtp_layer {
            let mut mtp_next_toks = Vec::with_capacity(ms);
            let mut first_pos = None;
            let mut first_j = None;
            for j in 0..ms {
                let gj = s_off + j;
                let nt_opt = if gj + 1 < m {
                    Some(tokens[gj + 1])
                } else {
                    next_tok_after_chunk
                };
                if let Some(nt) = nt_opt {
                    if *mtp_cached_pos <= pos0 + gj {
                        if first_pos.is_none() {
                            first_pos = Some(pos0 + gj);
                            first_j = Some(j);
                        }
                        mtp_next_toks.push(nt);
                    }
                }
            }
            if !mtp_next_toks.is_empty() {
                let dsa_mtp = dsa1[mtp_l - split].as_ref().unwrap();
                let hx_mtp = head_x_ptr.cast::<f32>().add(first_j.unwrap() * n_embd).cast();
                mtp_cache_chunk_split(
                    gguf,
                    weights,
                    sc1,
                    dsa_mtp,
                    hx_mtp,
                    &mut hop_emb_pin1[..ms * n_embd],
                    mtp_cached_pos,
                    first_pos.unwrap(),
                    &mtp_next_toks,
                    images,
                )?;
            }
        }

        if !is_last_slice {
            sc1.ev_stream_copy.record(&sc1.stream)?;
            sc1.ev_stream_copy.sync()?;
        } else if ms > 1 {
            let last_hx = head_x_ptr.cast::<f32>().add((ms - 1) * n_embd);
            copy_d2d_raw_async(head_x_ptr, last_hx.cast(), n_embd * 4, &sc1.stream)?;
        }
    }

    Ok(())
}

pub fn prefill_chunk(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    tokens: &[u32],
    images: &[Glm5ImageSpan],
    pos0: usize,
    next_tok_after_chunk: Option<u32>,
    is_final_chunk: bool,
    need_logits: bool,
    lent_views: Option<(&PrefillViews, &PrefillViews)>,
) -> Result<Option<(u32, Option<Vec<f32>>)>> {
    let m = tokens.len();
    if m == 0 {
        return Ok(None);
    }
    if lent_views.is_none() {
        let db0 = &mut state.doorbell.db0.as_mut_slice()[0];
        let db1 = &mut state.doorbell.db1.as_mut_slice()[0];
        state.expert_cache.drain_all(
            weights,
            &weights.cold_arena,
            &state.sc0.stream,
            &state.sc1.stream,
            db0,
            db1,
        )?;
    }

    let _active_guard = (m == 1).then(|| DoorbellActiveGuard::new(&state.doorbell.active));

    if let Some(prev_p) = state.head_x_pos {
        if prev_p + 1 == pos0 && state.mtp_cached_pos == prev_p {
            mtp_cache_entry_m1_with_images(
                gguf,
                weights,
                state,
                prev_p,
                state.head_x.ptr.cast(),
                tokens[0],
                images,
            )?;
        }
    }

    if m == 1
        && next_tok_after_chunk.is_none()
        && is_final_chunk
        && lent_views.is_none()
        && !image_span_covers(images, weights.cfg.n_embd, pos0)
    {
        let res = decode_token_m1(gguf, weights, state, tokens[0], need_logits)?;
        return Ok(Some(res));
    }

    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let split = weights.split_layer;
    let max_slice = state.max_prefill_chunk;
    let views0 = lent_views
        .map(|(v0, _)| *v0)
        .unwrap_or_else(|| PrefillViews::from_scratch(&state.sc0, cfg));
    let views1 = lent_views
        .map(|(_, v1)| *v1)
        .unwrap_or_else(|| PrefillViews::from_scratch(&state.sc1, cfg));

    {
        let pos0_dst = &mut state.pos_pin0.as_mut_slice()[..m];
        for (i, p) in pos0_dst.iter_mut().enumerate() {
            *p = (pos0 + i) as i32;
        }
        let pos1_dst = &mut state.pos_pin1.as_mut_slice()[..m];
        for (i, p) in pos1_dst.iter_mut().enumerate() {
            *p = (pos0 + i) as i32;
        }
    }

    // 1. Sliced host dequant embeddings [m, n_embd] -> sc0.emb -> views0.r_buf0
    set_device(0)?;
    unsafe {
        let pos0_bytes = std::slice::from_raw_parts(state.pos_pin0.host_ptr().cast::<u8>(), m * 4);
        copy_h2d_raw_async(views0.pos_arr_dev.cast(), pos0_bytes, &state.sc0.stream)?;
        for s_off in (0..m).step_by(max_slice) {
            let ms = (m - s_off).min(max_slice);
            let dst = &mut state.hop_emb_pin0.as_mut_slice()[..ms * n_embd];
            for (i, &tok) in tokens[s_off..s_off + ms].iter().enumerate() {
                fill_token_embed_host(
                    gguf,
                    weights,
                    tok,
                    pos0 + s_off + i,
                    images,
                    &mut dst[i * n_embd..(i + 1) * n_embd],
                )?;
            }
            let emb_bytes =
                std::slice::from_raw_parts(state.hop_emb_pin0.host_ptr().cast::<u8>(), ms * n_embd * 4);
            state.sc0.emb.copy_from_host_async(emb_bytes, &state.sc0.stream)?;
            glm5::embed_replicate(
                state.sc0.emb.ptr.cast(),
                views0.r_buf0.add(s_off * hc * n_embd),
                n_embd as i32,
                hc as i32,
                ms as i32,
                &state.sc0.stream,
            );
            if s_off + ms < m {
                state.sc0.ev_stream_copy.record(&state.sc0.stream)?;
                state.sc0.ev_stream_copy.sync()?;
            }
        }
    }

    let prof_on = prefill_prof_enabled();
    let mut prof_tot = PrefillProf::default();

    // 2. GPU 0 layers [0..split_layer)
    if m == 1 && lent_views.is_none() {
        for l in 0..split {
            enqueue_layer_m1(
                weights,
                state,
                l,
                pos0,
                std::ptr::null(),
                false,
                false,
                None,
                unsafe { &*(&raw const state.sc0.stream) },
            )?;
        }
    } else {
        state.expert_cache.tick += 1;
        let tick = state.expert_cache.tick;
        let (kda0, _) = state.kda.split_at_mut(split);
        let (dsa0, _) = state.dsa.split_at_mut(split);
        let (cache0, _) = state.expert_cache.layers.split_at_mut(split);
        let mut ctx0 = HalfPrefillCtx {
            dev: 0,
            weights,
            sc: &state.sc0,
            views: views0,
            kda: kda0,
            dsa: dsa0,
            cache: cache0,
            cache_tick: tick,
            lb: 0,
            pool: &state.doorbell.pool,
            cpu_scratch: &mut state.prefill_cpu_scratch0,
            meta_cursor: 0,
            prof: PrefillProf::default(),
        };
        run_half_layers_split(&mut ctx0, pos0, m)?;
        prof_tot.add_from(&ctx0.prof);
    }

    let t_hop = prof_on.then(Instant::now);
    if m <= max_slice && lent_views.is_none() {
        set_device(0)?;
        if state.p2p_ok {
            state.ev_g0_done[0].record(&state.sc0.stream)?;
        } else {
            let hop_bytes = m * hc * n_embd * 4;
            let h_slice = unsafe {
                std::slice::from_raw_parts_mut(state.hop_pin[0].host_ptr().cast::<u8>(), hop_bytes)
            };
            state.sc0.r_buf[0].copy_to_host_async(h_slice, &state.sc0.stream)?;
            state.ev_g0_done[0].record(&state.sc0.stream)?;
        }
        hop_gpu0_to_gpu1_r0(weights, state, m, 0)?;
    } else if state.p2p_ok {
        set_device(0)?;
        state.ev_g0_done[0].record(&state.sc0.stream)?;
        set_device(1)?;
        state.sc1.stream.wait_event(&state.ev_g0_done[0])?;
        unsafe {
            copy_peer_raw_async(
                views1.r_buf0.cast(),
                1,
                views0.r_buf0.cast(),
                0,
                m * hc * n_embd * 4,
                &state.sc1.stream,
            )?;
        }
        state.ev_g1_done.record(&state.sc1.stream)?;
    } else {
        for s_off in (0..m).step_by(max_slice) {
            let ms = (m - s_off).min(max_slice);
            let hop_bytes = ms * hc * n_embd * 4;
            set_device(0)?;
            unsafe {
                let h_slice = std::slice::from_raw_parts_mut(
                    state.hop_pin[0].host_ptr().cast::<u8>(),
                    hop_bytes,
                );
                copy_d2h_raw_async(
                    h_slice,
                    views0.r_buf0.add(s_off * hc * n_embd).cast(),
                    &state.sc0.stream,
                )?;
            }
            state.sc0.stream.sync()?;
            set_device(1)?;
            unsafe {
                let h_slice = std::slice::from_raw_parts(
                    state.hop_pin[0].host_ptr().cast::<u8>(),
                    hop_bytes,
                );
                copy_h2d_raw_async(
                    views1.r_buf0.add(s_off * hc * n_embd).cast(),
                    h_slice,
                    &state.sc1.stream,
                )?;
            }
            state.sc1.stream.sync()?;
        }
    }

    set_device(1)?;
    unsafe {
        let pos1_bytes = std::slice::from_raw_parts(state.pos_pin1.host_ptr().cast::<u8>(), m * 4);
        copy_h2d_raw_async(views1.pos_arr_dev.cast(), pos1_bytes, &state.sc1.stream)?;
    }
    if let Some(t0) = t_hop {
        state.sc1.stream.sync()?;
        prof_tot.hop_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    // 3. GPU 1 layers [split_layer..n_layers)
    if m == 1 && lent_views.is_none() {
        for l in split..cfg.n_layers {
            enqueue_layer_m1(
                weights,
                state,
                l,
                pos0,
                std::ptr::null(),
                false,
                false,
                None,
                unsafe { &*(&raw const state.sc1.stream) },
            )?;
        }
    } else {
        let tick = state.expert_cache.tick;
        let (_, kda1) = state.kda.split_at_mut(split);
        let (_, dsa1) = state.dsa.split_at_mut(split);
        let (_, cache1) = state.expert_cache.layers.split_at_mut(split);
        let mut ctx1 = HalfPrefillCtx {
            dev: 1,
            weights,
            sc: &state.sc1,
            views: views1,
            kda: kda1,
            dsa: dsa1,
            cache: cache1,
            cache_tick: tick,
            lb: split,
            pool: &state.doorbell.pool,
            cpu_scratch: &mut state.prefill_cpu_scratch1,
            meta_cursor: 0,
            prof: PrefillProf::default(),
        };
        run_half_layers_split(&mut ctx1, pos0, m)?;
        prof_tot.add_from(&ctx1.prof);
    }

    // 4..7 Head + Logits/Argmax + sliced MTP
    let t_hm = prof_on.then(Instant::now);
    unsafe {
        let hop_emb1 = state.hop_emb_pin1.as_mut_slice();
        let mut mtp_pos = state.mtp_cached_pos;
        let (_, dsa1) = state.dsa.split_at(split);
        compute_final_head_and_mtp(
            gguf,
            weights,
            &state.sc1,
            dsa1,
            views1.r_buf0,
            state.head_x.ptr,
            state.head_xq.ptr,
            &state.logits_dev,
            state.argmax_pv.ptr,
            state.argmax_pi.ptr,
            &state.tok_dev,
            state.tok_host.host_ptr(),
            hop_emb1,
            &mut mtp_pos,
            tokens,
            images,
            pos0,
            next_tok_after_chunk,
            is_final_chunk,
            need_logits,
        )?;
        state.mtp_cached_pos = mtp_pos;
    }

    state.sc1.stream.sync().map_err(|e| {
        anyhow::anyhow!("GPU1 head/mtp prefill fail pos0={pos0} m={m}: {e:#}")
    })?;
    if let Some(t0) = t_hm {
        prof_tot.head_mtp_ms += t0.elapsed().as_secs_f64() * 1000.0;
        if m > 1 || lent_views.is_some() {
            prof_tot.log("chunk", m);
        }
    }
    state.cur_pos = pos0 + m;
    state.head_x_pos = Some(pos0 + m - 1);

    if is_final_chunk {
        if need_logits {
            let mut logits = vec![0.0f32; cfg.n_vocab];
            state.logits_dev.copy_to_host(bytemuck::cast_slice_mut(&mut logits))?;
            let mut best_id = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_id = i;
                }
            }
            Ok(Some((best_id as u32, Some(logits))))
        } else {
            Ok(Some((state.tok_host.as_slice()[0] as u32, None)))
        }
    } else {
        Ok(None)
    }
}

/// Pipelined dual-GPU prefill for prompts longer than `chunk_size`:
/// overlaps chunk `c` on GPU 0 + CCD0 (`pool0`) with chunk `c - 1` on GPU 1 + CCD1 (`pool1`).
pub fn prefill_slice_pipelined(
    gguf: &GgufFile,
    weights: &Glm5ModelWeights,
    state: &mut Glm5State,
    tokens: &[u32],
    images: &[Glm5ImageSpan],
    chunk_size: usize,
    need_logits: bool,
    lent_views: Option<(&PrefillViews, &PrefillViews)>,
) -> Result<(u32, Option<Vec<f32>>)> {
    let total = tokens.len();
    if total == 0 {
        anyhow::bail!("empty prompt suffix in prefill_slice_pipelined");
    }

    if lent_views.is_none() {
        let db0 = &mut state.doorbell.db0.as_mut_slice()[0];
        let db1 = &mut state.doorbell.db1.as_mut_slice()[0];
        state.expert_cache.drain_all(
            weights,
            &weights.cold_arena,
            &state.sc0.stream,
            &state.sc1.stream,
            db0,
            db1,
        )?;
    }

    let base_pos0 = state.cur_pos;
    if let Some(prev_p) = state.head_x_pos {
        if prev_p + 1 == base_pos0 && state.mtp_cached_pos == prev_p {
            let _active_guard = DoorbellActiveGuard::new(&state.doorbell.active);
            mtp_cache_entry_m1_with_images(
                gguf,
                weights,
                state,
                prev_p,
                state.head_x.ptr.cast(),
                tokens[0],
                images,
            )?;
        }
    }

    let cfg = &weights.cfg;
    let n_embd = cfg.n_embd;
    let hc = cfg.hc;
    let split = weights.split_layer;
    let p2p_ok = state.p2p_ok;
    let max_slice = state.max_prefill_chunk;
    let views0 = lent_views
        .map(|(v0, _)| *v0)
        .unwrap_or_else(|| PrefillViews::from_scratch(&state.sc0, cfg));
    let views1 = lent_views
        .map(|(_, v1)| *v1)
        .unwrap_or_else(|| PrefillViews::from_scratch(&state.sc1, cfg));

    let mut chunks = Vec::new();
    let mut off = 0usize;
    while off < total {
        let end = (off + chunk_size).min(total);
        chunks.push(off..end);
        off = end;
    }
    let n_chunks = chunks.len();
    let base_tick = state.expert_cache.tick;

    let (kda0, kda1) = state.kda.split_at_mut(split);
    let (dsa0, dsa1) = state.dsa.split_at_mut(split);
    let (cache0, cache1) = state.expert_cache.layers.split_at_mut(split);

    struct ChunkMsg {
        c: usize,
        slot: usize,
        off: usize,
        end: usize,
    }

    let (free_tx, free_rx) = mpsc::channel::<usize>();
    free_tx.send(0)?;
    if !p2p_ok {
        free_tx.send(1)?;
    }
    let (work_tx, work_rx) = mpsc::channel::<ChunkMsg>();
    let (slice_tx, slice_rx) = mpsc::channel::<(usize, usize, usize, usize)>();

    let sc0_ref = &state.sc0;
    let sc1_ref = &state.sc1;
    let hop_pin_ref = &state.hop_pin;
    let ev_g0_done_ref = &state.ev_g0_done;
    let pool0_ref = &state.doorbell.pool0;
    let pool1_ref = &state.doorbell.pool1;
    let cpu_scratch0_ref = &mut state.prefill_cpu_scratch0;
    let cpu_scratch1_ref = &mut state.prefill_cpu_scratch1;
    let hop_emb_pin0_slice = state.hop_emb_pin0.as_mut_slice();
    let hop_emb_pin1_slice = state.hop_emb_pin1.as_mut_slice();
    let pos_pin0_slice = state.pos_pin0.as_mut_slice();
    let pos_pin1_slice = state.pos_pin1.as_mut_slice();
    let head_x_ref = &state.head_x;
    let head_xq_ref = &state.head_xq;
    let logits_dev_ref = &state.logits_dev;
    let argmax_pv_ref = &state.argmax_pv;
    let argmax_pi_ref = &state.argmax_pi;
    let tok_dev_ref = &state.tok_dev;
    let tok_host_ref = &state.tok_host;
    let mut mtp_cached_pos = state.mtp_cached_pos;
    let chunks_ref = &chunks;

    let prof_on = prefill_prof_enabled();

    let (r0, r1) = std::thread::scope(|scope| {
        let h0 = scope.spawn(move || -> Result<()> {
            set_device(0)?;
            let mut ctx0 = HalfPrefillCtx {
                dev: 0,
                weights,
                sc: sc0_ref,
                views: views0,
                kda: kda0,
                dsa: dsa0,
                cache: cache0,
                cache_tick: base_tick,
                lb: 0,
                pool: pool0_ref,
                cpu_scratch: cpu_scratch0_ref,
                meta_cursor: 0,
                prof: PrefillProf::default(),
            };
            for (c, range) in chunks_ref.iter().enumerate() {
                let m_c = range.len();
                let pos0_c = base_pos0 + range.start;
                ctx0.cache_tick = base_tick + 1 + c as u64;

                let p2p_slot = if p2p_ok { Some(free_rx.recv()?) } else { None };

                for (i, p) in pos_pin0_slice[..m_c].iter_mut().enumerate() {
                    *p = (pos0_c + i) as i32;
                }
                unsafe {
                    let pos_bytes =
                        std::slice::from_raw_parts(pos_pin0_slice.as_ptr().cast::<u8>(), m_c * 4);
                    copy_h2d_raw_async(views0.pos_arr_dev.cast(), pos_bytes, &sc0_ref.stream)?;
                    for s_off in (0..m_c).step_by(max_slice) {
                        let ms = (m_c - s_off).min(max_slice);
                        for (i, &tok) in tokens[range.start + s_off..range.start + s_off + ms]
                            .iter()
                            .enumerate()
                        {
                            fill_token_embed_host(
                                gguf,
                                weights,
                                tok,
                                pos0_c + s_off + i,
                                images,
                                &mut hop_emb_pin0_slice[i * n_embd..(i + 1) * n_embd],
                            )?;
                        }
                        let emb_bytes = std::slice::from_raw_parts(
                            hop_emb_pin0_slice.as_ptr().cast::<u8>(),
                            ms * n_embd * 4,
                        );
                        sc0_ref.emb.copy_from_host_async(emb_bytes, &sc0_ref.stream)?;
                        // Ensure pinned host staging finished copying before next slice/chunk overwrites
                        sc0_ref.ev_stream_copy.record(&sc0_ref.stream)?;
                        sc0_ref.ev_stream_copy.sync()?;
                        glm5::embed_replicate(
                            sc0_ref.emb.ptr.cast(),
                            views0.r_buf0.add(s_off * hc * n_embd),
                            n_embd as i32,
                            hc as i32,
                            ms as i32,
                            &sc0_ref.stream,
                        );
                    }
                }

                run_half_layers_split(&mut ctx0, pos0_c, m_c)?;

                if let Some(s) = p2p_slot {
                    ev_g0_done_ref[s].record(&sc0_ref.stream)?;
                    work_tx
                        .send(ChunkMsg {
                            c,
                            slot: s,
                            off: range.start,
                            end: range.end,
                        })
                        .map_err(|e| anyhow::anyhow!("GPU1 worker closed early: {e}"))?;
                } else {
                    work_tx
                        .send(ChunkMsg {
                            c,
                            slot: 0,
                            off: range.start,
                            end: range.end,
                        })
                        .map_err(|e| anyhow::anyhow!("GPU1 worker closed early: {e}"))?;
                    for s_off in (0..m_c).step_by(max_slice) {
                        let ms = (m_c - s_off).min(max_slice);
                        let s = free_rx.recv()?;
                        let hop_bytes = ms * hc * n_embd * 4;
                        unsafe {
                            let h_slice = std::slice::from_raw_parts_mut(
                                hop_pin_ref[s].host_ptr().cast::<u8>(),
                                hop_bytes,
                            );
                            copy_d2h_raw_async(
                                h_slice,
                                views0.r_buf0.add(s_off * hc * n_embd).cast(),
                                &sc0_ref.stream,
                            )?;
                        }
                        ev_g0_done_ref[s].record(&sc0_ref.stream)?;
                        slice_tx
                            .send((c, s_off, ms, s))
                            .map_err(|e| anyhow::anyhow!("GPU1 slice worker closed early: {e}"))?;
                    }
                }
            }
            sc0_ref.stream.sync()?;
            if prof_on {
                ctx0.prof.log(&format!("dev=0 chunks={n_chunks}"), total);
            }
            Ok(())
        });

        let h1 = scope.spawn(move || -> Result<(u32, Option<Vec<f32>>, usize)> {
            set_device(1)?;
            let views0_copy = views0;
            let mut ctx1 = HalfPrefillCtx {
                dev: 1,
                weights,
                sc: sc1_ref,
                views: views1,
                kda: kda1,
                dsa: dsa1,
                cache: cache1,
                cache_tick: base_tick,
                lb: split,
                pool: pool1_ref,
                cpu_scratch: cpu_scratch1_ref,
                meta_cursor: 0,
                prof: PrefillProf::default(),
            };
            while let Ok(msg) = work_rx.recv() {
                let m_c = msg.end - msg.off;
                let pos0_c = base_pos0 + msg.off;
                let is_final = msg.c + 1 == n_chunks;
                let next_after = if !is_final { Some(tokens[msg.end]) } else { None };
                ctx1.cache_tick = base_tick + 1 + msg.c as u64;

                for (i, p) in pos_pin1_slice[..m_c].iter_mut().enumerate() {
                    *p = (pos0_c + i) as i32;
                }

                let t_hop = prof_on.then(Instant::now);
                if p2p_ok {
                    sc1_ref.stream.wait_event(&ev_g0_done_ref[msg.slot])?;
                    let hop_bytes = m_c * hc * n_embd * 4;
                    unsafe {
                        copy_peer_raw_async(
                            views1.r_buf0.cast(),
                            1,
                            views0_copy.r_buf0.cast(),
                            0,
                            hop_bytes,
                            &sc1_ref.stream,
                        )?;
                        let pos1_bytes = std::slice::from_raw_parts(
                            pos_pin1_slice.as_ptr().cast::<u8>(),
                            m_c * 4,
                        );
                        copy_h2d_raw_async(
                            views1.pos_arr_dev.cast(),
                            pos1_bytes,
                            &sc1_ref.stream,
                        )?;
                    }
                    sc1_ref.ev_stream_copy.record(&sc1_ref.stream)?;
                    sc1_ref.ev_stream_copy.sync()?;
                    let _ = free_tx.send(msg.slot);
                } else {
                    for s_off in (0..m_c).step_by(max_slice) {
                        let (sc_idx, rx_off, ms, s_slot) = slice_rx.recv()?;
                        debug_assert_eq!(sc_idx, msg.c);
                        debug_assert_eq!(rx_off, s_off);
                        sc1_ref.stream.wait_event(&ev_g0_done_ref[s_slot])?;
                        let hop_bytes = ms * hc * n_embd * 4;
                        unsafe {
                            let h_slice = std::slice::from_raw_parts(
                                hop_pin_ref[s_slot].host_ptr().cast::<u8>(),
                                hop_bytes,
                            );
                            copy_h2d_raw_async(
                                views1.r_buf0.add(s_off * hc * n_embd).cast(),
                                h_slice,
                                &sc1_ref.stream,
                            )?;
                        }
                        sc1_ref.ev_stream_copy.record(&sc1_ref.stream)?;
                        sc1_ref.ev_stream_copy.sync()?;
                        let _ = free_tx.send(s_slot);
                    }
                    unsafe {
                        let pos1_bytes = std::slice::from_raw_parts(
                            pos_pin1_slice.as_ptr().cast::<u8>(),
                            m_c * 4,
                        );
                        copy_h2d_raw_async(
                            views1.pos_arr_dev.cast(),
                            pos1_bytes,
                            &sc1_ref.stream,
                        )?;
                    }
                    sc1_ref.ev_stream_copy.record(&sc1_ref.stream)?;
                    sc1_ref.ev_stream_copy.sync()?;
                }
                if let Some(t0) = t_hop {
                    ctx1.prof.hop_ms += t0.elapsed().as_secs_f64() * 1000.0;
                }

                run_half_layers_split(&mut ctx1, pos0_c, m_c)?;

                let t_hm = prof_on.then(Instant::now);
                unsafe {
                    compute_final_head_and_mtp(
                        gguf,
                        weights,
                        sc1_ref,
                        ctx1.dsa,
                        views1.r_buf0,
                        head_x_ref.ptr,
                        head_xq_ref.ptr,
                        logits_dev_ref,
                        argmax_pv_ref.ptr,
                        argmax_pi_ref.ptr,
                        tok_dev_ref,
                        tok_host_ref.host_ptr(),
                        hop_emb_pin1_slice,
                        &mut mtp_cached_pos,
                        &tokens[msg.off..msg.end],
                        images,
                        pos0_c,
                        next_after,
                        is_final,
                        if is_final { need_logits } else { false },
                    )?;
                }
                sc1_ref.stream.sync().map_err(|e| {
                    anyhow::anyhow!("GPU1 head/mtp prefill fail pos0={pos0_c} m={m_c}: {e:#}")
                })?;
                if let Some(t0) = t_hm {
                    ctx1.prof.head_mtp_ms += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }
            if prof_on {
                ctx1.prof.log(&format!("dev=1 chunks={n_chunks}"), total);
            }

            if need_logits {
                let mut logits = vec![0.0f32; cfg.n_vocab];
                logits_dev_ref.copy_to_host(bytemuck::cast_slice_mut(&mut logits))?;
                let mut best_id = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best_id = i;
                    }
                }
                Ok((best_id as u32, Some(logits), mtp_cached_pos))
            } else {
                let first_tok = tok_host_ref.as_slice()[0] as u32;
                Ok((first_tok, None, mtp_cached_pos))
            }
        });

        (
            h0.join().unwrap_or_else(|_| Err(anyhow::anyhow!("GPU0 prefill thread panicked"))),
            h1.join().unwrap_or_else(|_| Err(anyhow::anyhow!("GPU1 prefill thread panicked"))),
        )
    });

    r0?;
    let (first_tok, logits_opt, final_mtp_pos) = r1?;
    state.expert_cache.tick = base_tick + n_chunks as u64;
    state.cur_pos = base_pos0 + total;
    state.head_x_pos = Some(base_pos0 + total - 1);
    state.mtp_cached_pos = final_mtp_pos;
    Ok((first_tok, logits_opt))
}
