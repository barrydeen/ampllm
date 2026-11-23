use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_gguf::Hy4EngramConfig;
use anyhow::Result;
use half::bf16;
use memmap2::{Advice, Mmap};
use std::fs::File;

/// Host-resident Engram table view (`per_layer_token_embd.weight` F8_E4M3 + optional BF16 row scales).
pub struct Hy4EngramTable {
    pub table_ptr: *const u8,
    pub table_bytes: usize,
    pub scales_mmap: Option<Mmap>,
}

unsafe impl Send for Hy4EngramTable {}
unsafe impl Sync for Hy4EngramTable {}

impl Hy4EngramTable {
    pub fn open(cfg: &Hy4EngramConfig, table_ptr: *const u8, table_bytes: usize) -> Result<Self> {
        let scales_mmap = if let Some(ref p) = cfg.scales_path {
            let f = File::open(p)?;
            // SAFETY: Read-only mapping of immutable Engram row-scale weights.
            let m = unsafe { Mmap::map(&f)? };
            let _ = m.advise(Advice::Random);
            Some(m)
        } else {
            None
        };
        Ok(Self {
            table_ptr,
            table_bytes,
            scales_mmap,
        })
    }
}

#[inline(always)]
unsafe fn fill_toks(
    cfg: &Hy4EngramConfig,
    history: &[u32],
    p: usize,
    toks: &mut [u64; 8],
) -> bool {
    let n = cfg.ngram_size;
    debug_assert!(n <= 8);
    let cid = |tok: u32| -> i64 {
        let idx = tok as usize;
        if idx >= cfg.token_map.len() {
            -1
        } else {
            cfg.token_map[idx] as i64
        }
    };
    let c0 = cid(history[p]);
    let dead = c0 < 0;
    toks[0] = if dead {
        cfg.pad_mapped_id as u64
    } else {
        c0 as u64
    };
    for (sft, tok_slot) in toks.iter_mut().enumerate().take(n).skip(1) {
        let c = if p < sft {
            -1
        } else {
            cid(history[p - sft])
        };
        *tok_slot = if c < 0 {
            cfg.pad_mapped_id as u64
        } else {
            c as u64
        };
    }
    dead
}

#[inline(always)]
unsafe fn gather_one_col(
    cfg: &Hy4EngramConfig,
    table: &Hy4EngramTable,
    history: &[u32],
    e: usize,
    t: usize,
    col: usize,
    p0: usize,
    tc: usize,
    h_eg: *mut u8,
    h_egs: *mut f32,
) {
    let n = cfg.ngram_size;
    let cols = cfg.cols();
    let hd = cfg.head_dim;
    let hpn = cfg.heads_per_ngram;
    let mult = &cfg.multipliers[e * n..(e + 1) * n];
    let primes = &cfg.primes[e * cols..(e + 1) * cols];
    let offs = &cfg.offsets[e * cols..(e + 1) * cols];

    let mut toks = [0u64; 8];
    fill_toks(cfg, history, p0 + t, &mut toks);

    let oo_target = 1 + col / hpn;
    let mut rolling = toks[0].wrapping_mul(mult[0]);
    for (&tv, &mv) in toks[1..=oo_target].iter().zip(&mult[1..=oo_target]) {
        rolling ^= tv.wrapping_mul(mv);
    }
    let row = ((rolling % primes[col]) + offs[col]) as usize;
    let src = table.table_ptr.add(row * hd);
    let dst = h_eg.add(((e * tc + t) * cols + col) * hd);
    std::ptr::copy_nonoverlapping(src, dst, hd);
    if !h_egs.is_null() {
        if let Some(ref m) = table.scales_mmap {
            let sp = m.as_ptr();
            let bits = u16::from_le_bytes([*sp.add(row * 2), *sp.add(row * 2 + 1)]);
            *h_egs.add((e * tc + t) * cols + col) = bf16::from_bits(bits).to_f32();
        }
    }
}

#[inline(always)]
unsafe fn gather_one(
    cfg: &Hy4EngramConfig,
    table: &Hy4EngramTable,
    history: &[u32],
    e: usize,
    t: usize,
    p0: usize,
    tc: usize,
    h_eg: *mut u8,
    h_egs: *mut f32,
) {
    let n = cfg.ngram_size;
    let cols = cfg.cols();
    let hd = cfg.head_dim;
    let mult = &cfg.multipliers[e * n..(e + 1) * n];
    let primes = &cfg.primes[e * cols..(e + 1) * cols];
    let offs = &cfg.offsets[e * cols..(e + 1) * cols];

    let mut toks = [0u64; 8];
    fill_toks(cfg, history, p0 + t, &mut toks);

    let scales_ptr = table
        .scales_mmap
        .as_ref()
        .map_or(std::ptr::null::<u8>(), |m| m.as_ptr());

    let mut rolling = toks[0].wrapping_mul(mult[0]);
    for (oo, &tok_val) in toks.iter().enumerate().take(n).skip(1) {
        rolling ^= tok_val.wrapping_mul(mult[oo]);
        for h in 0..cfg.heads_per_ngram {
            let col = (oo - 1) * cfg.heads_per_ngram + h;
            let row = ((rolling % primes[col]) + offs[col]) as usize;
            let src = table.table_ptr.add(row * hd);
            let dst = h_eg.add(((e * tc + t) * cols + col) * hd);
            std::ptr::copy_nonoverlapping(src, dst, hd);
            if !scales_ptr.is_null() && !h_egs.is_null() {
                let bits = u16::from_le_bytes([*scales_ptr.add(row * 2), *scales_ptr.add(row * 2 + 1)]);
                *h_egs.add((e * tc + t) * cols + col) = bf16::from_bits(bits).to_f32();
            }
        }
    }
}

/// Gather all Engram rows for `t_len` tokens at positions `p0..p0 + t_len` into pinned staging.
/// Parallelizes across all `(e, t, col)` when `t_len <= 16` so even T=1..4 achieves full QD=16 page faults.
#[allow(clippy::too_many_arguments)]
pub fn gather_engram_window(
    cfg: &Hy4EngramConfig,
    table: &Hy4EngramTable,
    pool: &CpuThreadPool,
    history: &[u32],
    p0: usize,
    t_len: usize,
    tc: usize,
    h_eg: &mut [u8],
    h_valid: &mut [f32],
    mut h_egs: Option<&mut [f32]>,
) {
    let nm = cfg.layers.len();
    let cols = cfg.cols();
    if nm == 0 || t_len == 0 || cols == 0 {
        return;
    }

    // Compute valid mask once cleanly without data races
    for (t, v) in h_valid.iter_mut().take(t_len).enumerate() {
        let tok = history[p0 + t] as usize;
        let dead = tok >= cfg.token_map.len() || cfg.token_map[tok] < 0;
        *v = if dead { 0.0 } else { 1.0 };
    }

    let h_eg_u = h_eg.as_mut_ptr() as usize;
    let h_egs_u = h_egs
        .as_mut()
        .map_or(0usize, |s| s.as_mut_ptr() as usize);
    let table_ref = table as *const Hy4EngramTable as usize;
    let cfg_ref = cfg as *const Hy4EngramConfig as usize;
    let hist_ptr = history.as_ptr() as usize;
    let hist_len = history.len();

    if t_len <= 16 {
        let total_cols = nm * t_len * cols;
        pool.parallel_for(total_cols, |i0, i1| {
            let cfg_s = unsafe { &*(cfg_ref as *const Hy4EngramConfig) };
            let tbl_s = unsafe { &*(table_ref as *const Hy4EngramTable) };
            let hist_s = unsafe { std::slice::from_raw_parts(hist_ptr as *const u32, hist_len) };
            for i in i0..i1 {
                let col = i % cols;
                let rem = i / cols;
                let t = rem % t_len;
                let e = rem / t_len;
                unsafe {
                    gather_one_col(
                        cfg_s,
                        tbl_s,
                        hist_s,
                        e,
                        t,
                        col,
                        p0,
                        tc,
                        h_eg_u as *mut u8,
                        h_egs_u as *mut f32,
                    );
                }
            }
        });
    } else {
        let n_look = nm * t_len;
        pool.parallel_for(n_look, |i0, i1| {
            let cfg_s = unsafe { &*(cfg_ref as *const Hy4EngramConfig) };
            let tbl_s = unsafe { &*(table_ref as *const Hy4EngramTable) };
            let hist_s = unsafe { std::slice::from_raw_parts(hist_ptr as *const u32, hist_len) };
            for i in i0..i1 {
                let e = i / t_len;
                let t = i % t_len;
                unsafe {
                    gather_one(
                        cfg_s,
                        tbl_s,
                        hist_s,
                        e,
                        t,
                        p0,
                        tc,
                        h_eg_u as *mut u8,
                        h_egs_u as *mut f32,
                    );
                }
            }
        });
    }
}
