use ampllm_cpu_kernels::CpuThreadPool;
use ampllm_gguf::{GgmlType, Qwen4PleConfig};
use anyhow::{bail, Result};
use half::f16;
use std::sync::Arc;

/// IQ4_NL quantization codebook (`ggml` / `strata` standard non-linear 4-bit table).
const K_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Rolling 2-token predecessor window for Qwen3.8-Flash-Next PLE n-gram hashing (`ngram_size = 3`).
/// Stores predecessors oldest-first: `[tok_{t-2}, tok_{t-1}]`, initialized to `[-1, -1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleCursor {
    pub prev: [i32; 2],
}

impl Default for PleCursor {
    fn default() -> Self {
        Self { prev: [-1, -1] }
    }
}

impl PleCursor {
    pub fn reset(&mut self) {
        self.prev = [-1, -1];
    }
}

/// CPU PLE (`per_layer_token_embd.weight`) hash + IQ4_NL row lookup engine.
pub struct PleEngine {
    pub cfg: Qwen4PleConfig,
    pub row_bytes: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub total_dim: usize,
    pool: Arc<CpuThreadPool>,
}

impl PleEngine {
    pub fn new(cfg: Qwen4PleConfig, pool: Arc<CpuThreadPool>) -> Result<Self> {
        let n_heads = cfg.n_heads();
        let head_dim = cfg.head_dim;
        let total_dim = cfg.total_dim();
        let row_bytes = cfg.table_type.row_bytes(head_dim);
        if cfg.ngram_size != 3 || n_heads != 16 {
            bail!(
                "unsupported Qwen4 PLE geometry: ngram_size={}, n_heads={}",
                cfg.ngram_size,
                n_heads
            );
        }
        if cfg.multipliers.len() != cfg.ngram_size
            || cfg.head_offsets.len() != n_heads
            || cfg.head_vocab_sizes.len() != n_heads
        {
            bail!("invalid Qwen4 PLE multiplier / head table metadata lengths");
        }
        Ok(Self {
            cfg,
            row_bytes,
            n_heads,
            head_dim,
            total_dim,
            pool,
        })
    }

    #[inline]
    pub fn row_indices_for_token(&self, tok: i32, prev: &[i32; 2], out_rows: &mut [u32]) {
        debug_assert_eq!(out_rows.len(), self.n_heads);
        let eos = self.cfg.eos_token_id as i32;
        let mut ctx = [0i64; 3];
        ctx[0] = tok as i64;
        let mut cut = false;
        // prev is oldest-first: s=1 reads prev[1] (t-1), s=2 reads prev[0] (t-2).
        for s in 1..self.cfg.ngram_size {
            let t = if cut { -1 } else { prev[2 - s] };
            cut = cut || t < 0 || t == eos;
            ctx[s] = if cut { eos as i64 } else { t as i64 };
        }
        for n in 2..=self.cfg.ngram_size {
            let mut mixed = (ctx[0] as u64).wrapping_mul(self.cfg.multipliers[0]);
            for j in 1..n {
                mixed ^= (ctx[j] as u64).wrapping_mul(self.cfg.multipliers[j]);
            }
            let base = (n - 2) * self.cfg.heads_per_ngram;
            for g in 0..self.cfg.heads_per_ngram {
                let h = base + g;
                out_rows[h] =
                    (mixed % self.cfg.head_vocab_sizes[h] + self.cfg.head_offsets[h]) as u32;
            }
        }
    }

    #[inline]
    fn dequant_row(&self, row_bytes: &[u8], out: &mut [f32]) {
        if self.cfg.table_type == GgmlType::Iq4Nl {
            iq4_nl_dequant_row(row_bytes, out);
        } else {
            ampllm_gguf::dequantize_row_to_f32(self.cfg.table_type, row_bytes, out);
        }
    }

    /// Compute n-gram PLE embeddings `[tokens.len(), 2560]` F32 from the mmapped table bytes and advance `cursor`.
    pub fn lookup_batch(
        &self,
        table_bytes: &[u8],
        tokens: &[u32],
        cursor: &mut PleCursor,
        out_f32: &mut [f32],
    ) -> Result<()> {
        let nt = tokens.len();
        if nt == 0 {
            return Ok(());
        }
        if out_f32.len() < nt * self.total_dim {
            bail!("PLE out_f32 buffer too small");
        }
        let need_min_bytes = self.cfg.table_rows * self.row_bytes;
        if table_bytes.len() < need_min_bytes {
            bail!(
                "PLE table truncated: {} < {}",
                table_bytes.len(),
                need_min_bytes
            );
        }

        let mut rows = vec![0u32; nt * self.n_heads];
        for (i, &tok) in tokens.iter().enumerate() {
            let t_i32 = tok as i32;
            self.row_indices_for_token(
                t_i32,
                &cursor.prev,
                &mut rows[i * self.n_heads..(i + 1) * self.n_heads],
            );
            cursor.prev[0] = cursor.prev[1];
            cursor.prev[1] = t_i32;
        }

        let row_bytes = self.row_bytes;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let total_dim = self.total_dim;
        if nt < 24 || self.pool.num_threads() <= 1 {
            for (i, out_tok) in out_f32[..nt * total_dim].chunks_exact_mut(total_dim).enumerate() {
                for h in 0..n_heads {
                    let row_idx = rows[i * n_heads + h] as usize;
                    let off = row_idx * row_bytes;
                    self.dequant_row(
                        &table_bytes[off..off + row_bytes],
                        &mut out_tok[h * head_dim..(h + 1) * head_dim],
                    );
                }
            }
        } else {
            let out_addr = out_f32.as_mut_ptr() as usize;
            let tbl = table_bytes;
            let rows_ref = &rows;
            self.pool.parallel_for(nt, move |start, end| {
                for i in start..end {
                    let dst_tok = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_addr as *mut f32).add(i * total_dim),
                            total_dim,
                        )
                    };
                    for h in 0..n_heads {
                        let row_idx = rows_ref[i * n_heads + h] as usize;
                        let off = row_idx * row_bytes;
                        if self.cfg.table_type == GgmlType::Iq4Nl {
                            iq4_nl_dequant_row(
                                &tbl[off..off + row_bytes],
                                &mut dst_tok[h * head_dim..(h + 1) * head_dim],
                            );
                        } else {
                            ampllm_gguf::dequantize_row_to_f32(
                                self.cfg.table_type,
                                &tbl[off..off + row_bytes],
                                &mut dst_tok[h * head_dim..(h + 1) * head_dim],
                            );
                        }
                    }
                }
            });
        }

        Ok(())
    }
}

/// Dequantize a single IQ4_NL row using ggml's split-halves element layout (`qs[j]` holds `j` and `j+16`).
#[inline]
pub fn iq4_nl_dequant_row(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(dst.len() % 32, 0);
    let nb = dst.len() / 32;
    debug_assert!(src.len() >= nb * 18);
    for b in 0..nb {
        let blk = &src[b * 18..(b + 1) * 18];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..18];
        let out = &mut dst[b * 32..(b + 1) * 32];
        for j in 0..16 {
            let q = qs[j];
            out[j] = d * (K_IQ4_NL[(q & 0x0F) as usize] as f32);
            out[j + 16] = d * (K_IQ4_NL[(q >> 4) as usize] as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> Qwen4PleConfig {
        Qwen4PleConfig {
            layers: vec![1],
            ngram_size: 3,
            heads_per_ngram: 8,
            conv_kernel: 4,
            head_dim: 160,
            eos_token_id: 248044,
            image_token_id: 248056,
            table_rows: 320_001_536,
            table_type: GgmlType::Iq4Nl,
            multipliers: vec![23703573157769, 20109073645365, 8052911324071],
            head_offsets: vec![
                0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                300001275,
            ],
            head_vocab_sizes: vec![
                20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
            ],
        }
    }

    #[test]
    fn test_ple_ngram_rows_and_eos_cut() {
        let cfg = test_cfg();
        let pool = Arc::new(CpuThreadPool::new(Some(1)));
        let ple = PleEngine::new(cfg.clone(), pool).unwrap();

        let mut r_eos_seq = [0u32; 16];
        let mut r_cut_seq = [0u32; 16];
        // At start (prev=[-1, -1]), predecessor slots should be replaced by EOS=248044.
        ple.row_indices_for_token(846, &[-1, -1], &mut r_eos_seq);
        ple.row_indices_for_token(846, &[248044, 248044], &mut r_cut_seq);
        assert_eq!(r_eos_seq, r_cut_seq);

        // If newest predecessor is EOS (248044), older predecessor (123) MUST also be cut to EOS!
        let mut r_mid_cut = [0u32; 16];
        ple.row_indices_for_token(999, &[123, 248044], &mut r_mid_cut);
        let mut r_both_eos = [0u32; 16];
        ple.row_indices_for_token(999, &[248044, 248044], &mut r_both_eos);
        assert_eq!(r_mid_cut, r_both_eos);

        // If older predecessor is EOS and newest is valid (846), only older becomes EOS while newest stays 846.
        let mut r_tail_cut = [0u32; 16];
        ple.row_indices_for_token(999, &[-5, 846], &mut r_tail_cut);
        let mut r_exp = [0u32; 16];
        ple.row_indices_for_token(999, &[248044, 846], &mut r_exp);
        assert_eq!(r_tail_cut, r_exp);
        assert_ne!(r_tail_cut[8..16], r_both_eos[8..16]);

        // Verify exact Strata mix formula on [198, 846, 248045]
        let m0 = (198i64 as u64).wrapping_mul(cfg.multipliers[0])
            ^ (846i64 as u64).wrapping_mul(cfg.multipliers[1]);
        let m1 = m0 ^ (248045i64 as u64).wrapping_mul(cfg.multipliers[2]);
        let mut got = [0u32; 16];
        ple.row_indices_for_token(198, &[248045, 846], &mut got);
        for h in 0..8 {
            assert_eq!(
                got[h],
                (m0 % cfg.head_vocab_sizes[h] + cfg.head_offsets[h]) as u32
            );
            assert_eq!(
                got[8 + h],
                (m1 % cfg.head_vocab_sizes[8 + h] + cfg.head_offsets[8 + h]) as u32
            );
        }
    }

    #[test]
    fn test_iq4_nl_dequant_matches_gguf_ref() {
        let mut src = vec![0u8; 5 * 18];
        for (i, b) in src.iter_mut().enumerate() {
            *b = ((i * 73 + 19) & 0xFF) as u8;
        }
        // Valid FP16 scale ~0.05 in each block
        for blk in 0..5 {
            let d = f16::from_f32(0.03 + (blk as f32) * 0.01).to_le_bytes();
            src[blk * 18] = d[0];
            src[blk * 18 + 1] = d[1];
        }
        let mut got = vec![0.0f32; 160];
        let mut exp = vec![0.0f32; 160];
        iq4_nl_dequant_row(&src, &mut got);
        ampllm_gguf::dequantize_row_to_f32(GgmlType::Iq4Nl, &src, &mut exp);
        assert_eq!(got, exp);
    }
}
