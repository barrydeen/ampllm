use ampllm_cuda_kernels::{
    copy_d2h_raw_async, copy_h2d_raw_async, glm5, hy4, set_device, CudaStream, DeviceBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Hy4VisionConfig};
use anyhow::{bail, Result};
use half::f16;
use std::ffi::c_void;

#[inline]
fn align256(n: usize) -> usize {
    (n + 255) & !255
}

/// Zero-dedicated-VRAM SigLIP vision tower + spatial-merge projector (`mmproj`).
/// During `encode_image`, it temporarily borrows GPU 0's idle prefill scratch arena,
/// falling back to a transient allocation only when an oversized image exceeds arena capacity.
pub struct Hy4Vision {
    gguf: GgufFile,
    pub cfg: Hy4VisionConfig,
}

impl Hy4Vision {
    pub fn from_gguf(gguf: GgufFile, vocab_tokens: &[String], n_embd: usize) -> Result<Self> {
        let cfg = Hy4VisionConfig::from_mmproj(&gguf, vocab_tokens, n_embd)?;
        let d = cfg.dim;
        let f = cfg.ff;
        let p = 3 * cfg.patch_size * cfg.patch_size;
        let m4 = cfg.spatial_merge * cfg.spatial_merge * d;
        let o = cfg.out_dim;

        let chk = |name: &str, elems: usize| -> Result<()> {
            let info = gguf.tensor_info(name)?;
            if info.n_elements() != elems {
                bail!(
                    "vision tensor {name}: expected {elems} elements, got {}",
                    info.n_elements()
                );
            }
            Ok(())
        };
        chk("v.patch_embd.weight", d * p)?;
        chk("v.patch_embd.bias", d)?;
        chk("v.position_embd.weight", cfg.grid * cfg.grid * d)?;
        chk("mm.pre_norm.weight", d)?;
        chk("mm.0.weight", m4 * m4)?;
        chk("mm.0.bias", m4)?;
        chk("mm.2.weight", o * m4)?;
        chk("mm.2.bias", o)?;
        chk("v.image_newline", o)?;

        for i in 0..cfg.layers {
            let pre = format!("v.blk.{i}.");
            for suf in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"] {
                chk(&format!("{pre}{suf}"), d)?;
            }
            for suf in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_out.weight",
            ] {
                chk(&format!("{pre}{suf}"), d * d)?;
            }
            for suf in ["attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_out.bias"] {
                chk(&format!("{pre}{suf}"), d)?;
            }
            chk(&format!("{pre}ffn_up.weight"), f * d)?;
            chk(&format!("{pre}ffn_up.bias"), f)?;
            chk(&format!("{pre}ffn_down.weight"), d * f)?;
            chk(&format!("{pre}ffn_down.bias"), d)?;
        }

        Ok(Self { gguf, cfg })
    }

    pub fn mmproj_path(&self) -> Option<&std::path::Path> {
        self.gguf.shard_paths.first().map(|p| p.as_path())
    }

    /// Estimate workspace bytes for an image of `width x height` pixels.
    pub fn workspace_bytes(&self, width: usize, height: usize) -> usize {
        let g = &self.cfg;
        let ps = g.patch_size;
        let gh = height / ps;
        let gw = width / ps;
        let n = gh * gw;
        let d = g.dim;
        let f = g.ff;
        let p = 3 * ps * ps;
        let ow = gw / g.spatial_merge;
        let oh = gh / g.spatial_merge;
        let nb = ow * oh;
        let n_tok = (ow + 1) * oh;
        let m4 = g.spatial_merge * g.spatial_merge * d;
        let o = g.out_dim;
        let rc = n.min(2048);
        let max_w_f16_elems = (m4 * m4).max(3 * d * d).max(2 * f * d);
        let max_w_raw_bytes = (m4 * m4 * 4).max(f * d * 4);
        let pre_vec_floats = d + d + m4 + o + o;
        let layer_vec_floats = 10 * d + f;
        align256(n * p * 2)
            + align256(d * p * 2)
            + align256(pre_vec_floats * 4)
            + align256(layer_vec_floats * 4)
            + align256(n * d * 4)
            + align256((n * 3 * d).max(nb * m4) * 4)
            + align256(n.max(nb) * d.max(m4) * 4)
            + align256(rc * f * 4)
            + align256((n * d).max(rc * f).max(nb * m4) * 2)
            + align256(n_tok * o * 4)
            + align256(max_w_f16_elems * 2)
            + align256(max_w_raw_bytes)
    }

    /// Encode an RGB image (`[height, width, 3]`, uint8) into `[n_tok, out_dim]` F32 embeddings.
    pub fn encode_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        borrowed_arena: Option<&DeviceBuffer>,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Vec<f32>> {
        let g = &self.cfg;
        let n_tok = g.image_tokens(width, height);
        if n_tok == 0 {
            bail!(
                "vision: invalid dimensions {width}x{height} (must be multiples of {}, <= {} pixels)",
                g.factor(),
                g.max_pixels
            );
        }
        if rgb.len() != width * height * 3 {
            bail!(
                "vision: RGB buffer len {} != expected {}",
                rgb.len(),
                width * height * 3
            );
        }

        let ps = g.patch_size;
        let gh = height / ps;
        let gw = width / ps;
        let n = gh * gw;
        let d = g.dim;
        let f = g.ff;
        let p = 3 * ps * ps;
        let ow = gw / g.spatial_merge;
        let oh = gh / g.spatial_merge;
        let nb = ow * oh;
        let m4 = g.spatial_merge * g.spatial_merge * d;
        let o = g.out_dim;
        let hd = g.head_dim;

        // 1. Prepare normalized patches [N, 3, ps, ps] -> F16
        let mut hp_f16 = vec![0u16; n * p];
        for patch_idx in 0..n {
            let py = patch_idx / gw;
            let px = patch_idx % gw;
            let dst = &mut hp_f16[patch_idx * p..(patch_idx + 1) * p];
            for c in 0..3 {
                let inv_std = 1.0f32 / g.image_std[c];
                let mean = g.image_mean[c];
                for y in 0..ps {
                    for x in 0..ps {
                        let raw = rgb[((py * ps + y) * width + px * ps + x) * 3 + c];
                        let val = (raw as f32 * (1.0f32 / 255.0f32) - mean) * inv_std;
                        dst[(c * ps + y) * ps + x] = f16::from_f32(val).to_bits();
                    }
                }
            }
        }

        // 2. Bilinear resample of v.position_embd.weight -> hpos [N, D] F32
        let pos_tab = self.gguf.tensor_f32_slice("v.position_embd.weight")?;
        let mut hpos = vec![0.0f32; n * d];
        {
            let grid = g.grid as i32;
            let sx = (gw as f32 + 0.1f32) / grid as f32;
            let sy = (gh as f32 + 0.1f32) / grid as f32;
            let max_idx = grid - 1;
            for y in 0..gh {
                let fy = (y as f32 + 0.5f32) / sy - 0.5f32;
                let y0f = fy.floor();
                let y0 = (y0f as i32).clamp(0, max_idx) as usize;
                let y1 = ((y0f as i32) + 1).clamp(0, max_idx) as usize;
                let wy1 = (fy - y0f).clamp(0.0, 1.0);
                let wy0 = 1.0f32 - wy1;
                for x in 0..gw {
                    let fx = (x as f32 + 0.5f32) / sx - 0.5f32;
                    let x0f = fx.floor();
                    let x0 = (x0f as i32).clamp(0, max_idx) as usize;
                    let x1 = ((x0f as i32) + 1).clamp(0, max_idx) as usize;
                    let wx1 = (fx - x0f).clamp(0.0, 1.0);
                    let wx0 = 1.0f32 - wx1;

                    let w00 = wy0 * wx0;
                    let w01 = wy0 * wx1;
                    let w10 = wy1 * wx0;
                    let w11 = wy1 * wx1;
                    let s00 = &pos_tab[(y0 * g.grid + x0) * d..][..d];
                    let s01 = &pos_tab[(y0 * g.grid + x1) * d..][..d];
                    let s10 = &pos_tab[(y1 * g.grid + x0) * d..][..d];
                    let s11 = &pos_tab[(y1 * g.grid + x1) * d..][..d];
                    let dst = &mut hpos[(y * gw + x) * d..][..d];
                    for c in 0..d {
                        dst[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
                    }
                }
            }
        }

        // 3. Carve workspace from borrowed GPU arena (or temporary DeviceBuffer)
        let rc = n.min(2048);
        let max_w_f16_elems = (m4 * m4).max(3 * d * d).max(2 * f * d);
        let max_w_raw_bytes = (m4 * m4 * 4).max(f * d * 4);
        let pre_vec_floats = d + d + m4 + o + o;
        let layer_vec_floats = 10 * d + f;

        let s_px16 = align256(n * p * 2);
        let s_wp16 = align256(d * p * 2);
        let s_pre = align256(pre_vec_floats * 4);
        let s_lvec = align256(layer_vec_floats * 4);
        let s_h = align256(n * d * 4);
        let s_qkv = align256((n * 3 * d).max(nb * m4) * 4);
        let s_att = align256(n.max(nb) * d.max(m4) * 4);
        let s_big = align256(rc * f * 4);
        let s_x16 = align256((n * d).max(rc * f).max(nb * m4) * 2);
        let s_out = align256(n_tok * o * 4);
        let s_wf16 = align256(max_w_f16_elems * 2);
        let s_wraw = align256(max_w_raw_bytes);

        let total_bytes = s_px16
            + s_wp16
            + s_pre
            + s_lvec
            + s_h
            + s_qkv
            + s_att
            + s_big
            + s_x16
            + s_out
            + s_wf16
            + s_wraw;

        set_device(device)?;
        let _temp_buf: Option<DeviceBuffer>;
        let base_ptr: *mut u8 = if let Some(arena) = borrowed_arena {
            if total_bytes <= arena.bytes {
                _temp_buf = None;
                arena.ptr.cast()
            } else {
                let b = DeviceBuffer::alloc(device, total_bytes)?;
                let p_raw = b.ptr.cast();
                _temp_buf = Some(b);
                p_raw
            }
        } else {
            let b = DeviceBuffer::alloc(device, total_bytes)?;
            let p_raw = b.ptr.cast();
            _temp_buf = Some(b);
            p_raw
        };

        let mut off = 0usize;
        let mut slice_u8 = |sz: usize| -> *mut u8 {
            let ptr = unsafe { base_ptr.add(off) };
            off += sz;
            ptr
        };
        let d_px16 = slice_u8(s_px16).cast::<c_void>();
        let d_wp16 = slice_u8(s_wp16).cast::<c_void>();
        let d_pre = slice_u8(s_pre).cast::<f32>();
        let d_lvec = slice_u8(s_lvec).cast::<f32>();
        let d_h = slice_u8(s_h).cast::<f32>();
        let d_qkv = slice_u8(s_qkv).cast::<f32>();
        let d_att = slice_u8(s_att).cast::<f32>();
        let d_big = slice_u8(s_big).cast::<f32>();
        let d_x16 = slice_u8(s_x16).cast::<u16>();
        let d_out = slice_u8(s_out).cast::<f32>();
        let d_wf16 = slice_u8(s_wf16).cast::<u16>();
        let d_wraw = slice_u8(s_wraw).cast::<u8>();
        debug_assert_eq!(off, total_bytes);

        let d_bpatch = d_pre;
        let d_norm = unsafe { d_pre.add(d) };
        let d_b0 = unsafe { d_norm.add(d) };
        let d_b2 = unsafe { d_b0.add(m4) };
        let d_nl = unsafe { d_b2.add(o) };

        let d_ln1w = d_lvec;
        let d_ln1b = unsafe { d_ln1w.add(d) };
        let d_ln2w = unsafe { d_ln1b.add(d) };
        let d_ln2b = unsafe { d_ln2w.add(d) };
        let d_bqkv = unsafe { d_ln2b.add(d) };
        let d_bo = unsafe { d_bqkv.add(3 * d) };
        let d_b1 = unsafe { d_bo.add(d) };
        let d_b2v = unsafe { d_b1.add(f) };

        let mut h_pre = Vec::with_capacity(pre_vec_floats);
        h_pre.extend_from_slice(self.gguf.tensor_f32_slice("v.patch_embd.bias")?);
        h_pre.extend_from_slice(self.gguf.tensor_f32_slice("mm.pre_norm.weight")?);
        h_pre.extend_from_slice(self.gguf.tensor_f32_slice("mm.0.bias")?);
        h_pre.extend_from_slice(self.gguf.tensor_f32_slice("mm.2.bias")?);
        h_pre.extend_from_slice(self.gguf.tensor_f32_slice("v.image_newline")?);

        let wpatch_bytes = self.gguf.tensor_bytes("v.patch_embd.weight")?;
        let wpatch_ty = self.gguf.tensor_info("v.patch_embd.weight")?.ggml_type;

        unsafe {
            copy_h2d_raw_async(d_px16, bytemuck::cast_slice(&hp_f16), stream)?;
            copy_h2d_raw_async(d_h.cast(), bytemuck::cast_slice(&hpos), stream)?;
            copy_h2d_raw_async(d_pre.cast(), bytemuck::cast_slice(&h_pre), stream)?;
            copy_h2d_raw_async(d_wraw.cast(), wpatch_bytes, stream)?;
            glm5::dequant_to_f16(wpatch_ty as i32, d_wraw.cast(), d_wp16, (d * p) as i64, stream);
            glm5::gemm_f16_mma_nt(
                d_px16,
                d_wp16,
                d_h,
                n as i32,
                d as i32,
                p as i32,
                1.0,
                1.0,
                stream,
            );
            hy4::vis_add_bias(d_h, d_bpatch, n * d, d as i32, stream);
        }

        let upload_mat_f16 = |name: &str, elems: usize, dst_f16: *mut u16| -> Result<()> {
            let info = self.gguf.tensor_info(name)?;
            let raw = self.gguf.tensor_bytes(name)?;
            match info.ggml_type {
                GgmlType::F16 | GgmlType::Bf16 | GgmlType::Q8_0 | GgmlType::F32 => unsafe {
                    copy_h2d_raw_async(d_wraw.cast(), raw, stream)?;
                    glm5::dequant_to_f16(
                        info.ggml_type as i32,
                        d_wraw.cast(),
                        dst_f16.cast(),
                        elems as i64,
                        stream,
                    );
                },
                other => bail!("vision matrix {name} has unsupported type {other:?}"),
            }
            Ok(())
        };

        let attn_scale = 1.0f32 / (hd as f32).sqrt();
        let mut h_lvec = vec![0.0f32; layer_vec_floats];
        let ffn_chunk = rc.min(2048);

        for li in 0..g.layers {
            let pre = format!("v.blk.{li}.");
            let mut cursor = 0usize;
            for (suf, len) in [
                ("ln1.weight", d),
                ("ln1.bias", d),
                ("ln2.weight", d),
                ("ln2.bias", d),
                ("attn_q.bias", d),
                ("attn_k.bias", d),
                ("attn_v.bias", d),
                ("attn_out.bias", d),
                ("ffn_up.bias", f),
                ("ffn_down.bias", d),
            ] {
                let s = self.gguf.tensor_f32_slice(&format!("{pre}{suf}"))?;
                h_lvec[cursor..cursor + len].copy_from_slice(s);
                cursor += len;
            }
            debug_assert_eq!(cursor, layer_vec_floats);

            unsafe {
                copy_h2d_raw_async(d_lvec.cast(), bytemuck::cast_slice(&h_lvec), stream)?;
            }

            // Attention
            upload_mat_f16(&format!("{pre}attn_q.weight"), d * d, d_wf16)?;
            upload_mat_f16(&format!("{pre}attn_k.weight"), d * d, unsafe {
                d_wf16.add(d * d)
            })?;
            upload_mat_f16(&format!("{pre}attn_v.weight"), d * d, unsafe {
                d_wf16.add(2 * d * d)
            })?;

            unsafe {
                hy4::vis_layernorm_f16(d_h, d_ln1w, d_ln1b, n as i32, d as i32, g.eps, d_x16, stream);
                glm5::gemm_f16_mma_nt(
                    d_x16.cast(),
                    d_wf16.cast(),
                    d_qkv,
                    n as i32,
                    (3 * d) as i32,
                    d as i32,
                    1.0,
                    0.0,
                    stream,
                );
                hy4::vis_add_bias(d_qkv, d_bqkv, n * 3 * d, (3 * d) as i32, stream);
                hy4::vis_flash_attn(d_qkv, n as i32, g.heads as i32, hd as i32, attn_scale, d_att, stream);
                glm5::f32_to_f16(d_att, d_x16.cast(), (n * d) as i32, stream);
            }

            upload_mat_f16(&format!("{pre}attn_out.weight"), d * d, d_wf16)?;
            unsafe {
                glm5::gemm_f16_mma_nt(
                    d_x16.cast(),
                    d_wf16.cast(),
                    d_att,
                    n as i32,
                    d as i32,
                    d as i32,
                    1.0,
                    0.0,
                    stream,
                );
                hy4::vis_residual_bias(d_h, d_att, d_bo, n * d, d as i32, stream);
            }

            // MLP: both w1 [F, D] and w2 [D, F] fit simultaneously inside d_wf16!
            let d_w1_16 = d_wf16;
            let d_w2_16 = unsafe { d_wf16.add(f * d) };
            upload_mat_f16(&format!("{pre}ffn_up.weight"), f * d, d_w1_16)?;
            upload_mat_f16(&format!("{pre}ffn_down.weight"), d * f, d_w2_16)?;

            for r0 in (0..n).step_by(ffn_chunk) {
                let nr = (n - r0).min(ffn_chunk);
                let hr = unsafe { d_h.add(r0 * d) };
                unsafe {
                    hy4::vis_layernorm_f16(hr, d_ln2w, d_ln2b, nr as i32, d as i32, g.eps, d_x16, stream);
                    glm5::gemm_f16_mma_nt(
                        d_x16.cast(),
                        d_w1_16.cast(),
                        d_big,
                        nr as i32,
                        f as i32,
                        d as i32,
                        1.0,
                        0.0,
                        stream,
                    );
                    hy4::vis_bias_act_f16(d_big, d_b1, nr * f, f as i32, 1, d_x16, stream);
                    glm5::gemm_f16_mma_nt(
                        d_x16.cast(),
                        d_w2_16.cast(),
                        d_att,
                        nr as i32,
                        d as i32,
                        f as i32,
                        1.0,
                        0.0,
                        stream,
                    );
                    hy4::vis_residual_bias(hr, d_att, d_b2v, nr * d, d as i32, stream);
                }
            }
        }

        // Spatial-merge projector + image_newline placement
        unsafe {
            hy4::vis_rmsnorm_inplace(d_h, d_norm, n as i32, d as i32, g.eps, stream);
            hy4::vis_merge_gather_f16(d_h, gw as i32, d as i32, ow as i32, oh as i32, d_x16, stream);
        }
        upload_mat_f16("mm.0.weight", m4 * m4, d_wf16)?;
        unsafe {
            glm5::gemm_f16_mma_nt(
                d_x16.cast(),
                d_wf16.cast(),
                d_qkv,
                nb as i32,
                m4 as i32,
                m4 as i32,
                1.0,
                0.0,
                stream,
            );
            // Output rows fit in d_att ([nb, m4] F16 fits easily inside s_att)
            let d_g16 = d_att.cast::<u16>();
            hy4::vis_bias_act_f16(d_qkv, d_b0, nb * m4, m4 as i32, 2, d_g16, stream);
        }
        upload_mat_f16("mm.2.weight", o * m4, d_wf16)?;
        unsafe {
            let d_g16 = d_att.cast::<u16>();
            glm5::gemm_f16_mma_nt(
                d_g16.cast(),
                d_wf16.cast(),
                d_qkv,
                nb as i32,
                o as i32,
                m4 as i32,
                1.0,
                0.0,
                stream,
            );
            hy4::vis_place_rows(
                d_qkv,
                d_b2,
                d_nl,
                ow as i32,
                oh as i32,
                o as i32,
                d_out,
                stream,
            );
        }

        let mut rows = vec![0.0f32; n_tok * o];
        unsafe {
            copy_d2h_raw_async(bytemuck::cast_slice_mut(&mut rows), d_out.cast(), stream)?;
        }
        stream.sync()?;
        Ok(rows)
    }
}
