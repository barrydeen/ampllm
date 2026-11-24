use ampllm_cuda_kernels::{
    copy_h2d_raw_async, glm5, hy4, qwen4, set_device, CudaStream, DeviceBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Glm5VisionConfig};
use anyhow::{bail, Result};
use half::f16;
use std::ffi::c_void;

#[derive(Debug, Clone)]
pub struct Glm5ImageSpan {
    pub start_pos: usize,
    pub nx: usize,
    pub ny: usize,
    pub rows: Vec<f32>,
}

impl Glm5ImageSpan {
    #[inline]
    pub fn n_tokens(&self, n_embd: usize) -> usize {
        if n_embd == 0 {
            0
        } else {
            self.rows.len() / n_embd
        }
    }
}

struct Glm5VisionLayerWeights {
    ln1_w: DeviceBuffer,
    ln2_w: DeviceBuffer,
    attn_qkv_w: DeviceBuffer,
    attn_qkv_b: DeviceBuffer,
    attn_q_norm_w: DeviceBuffer,
    attn_k_norm_w: DeviceBuffer,
    attn_out_w: DeviceBuffer,
    attn_out_b: DeviceBuffer,
    ffn_gate_w: DeviceBuffer,
    ffn_gate_b: DeviceBuffer,
    ffn_up_w: DeviceBuffer,
    ffn_up_b: DeviceBuffer,
    ffn_down_w: DeviceBuffer,
    ffn_down_b: DeviceBuffer,
}

pub struct Glm5Vision {
    pub cfg: Glm5VisionConfig,
    pub device: i32,
    patch_w_f16: DeviceBuffer,
    patch_b_f32: DeviceBuffer,
    layers: Vec<Glm5VisionLayerWeights>,
    post_ln_w: DeviceBuffer,
    mm_patch_merger_w_f16: DeviceBuffer,
    mm_patch_merger_b_f32: DeviceBuffer,
    mm_fc_w_f16: DeviceBuffer,
    mm_post_norm_w: DeviceBuffer,
    mm_post_norm_b: DeviceBuffer,
    mm_gate_w_f16: DeviceBuffer,
    mm_up_w_f16: DeviceBuffer,
    mm_down_w_f16: DeviceBuffer,
}

fn load_tensor_f32_vec(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    if info.ggml_type == GgmlType::F32 {
        Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec())
    } else {
        let mut out = vec![0.0f32; info.n_elements()];
        ampllm_gguf::dequantize_row_to_f32(info.ggml_type, raw, &mut out);
        Ok(out)
    }
}

fn upload_f32(dev: i32, gguf: &GgufFile, name: &str) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    if info.ggml_type == GgmlType::F32 {
        let buf = DeviceBuffer::alloc(dev, raw.len())?;
        buf.copy_from_host(raw)?;
        Ok(buf)
    } else {
        let f32_vals = load_tensor_f32_vec(gguf, name)?;
        let buf = DeviceBuffer::alloc(dev, f32_vals.len() * 4)?;
        buf.copy_from_host(bytemuck::cast_slice(&f32_vals))?;
        Ok(buf)
    }
}

fn upload_as_f16(
    dev: i32,
    gguf: &GgufFile,
    name: &str,
    stream: &CudaStream,
) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    let n_elems = info.n_elements();
    let raw = gguf.tensor_bytes(name)?;
    let dst = DeviceBuffer::alloc(dev, n_elems * 2)?;
    if info.ggml_type == GgmlType::F16 {
        dst.copy_from_host(raw)?;
        return Ok(dst);
    }
    let stage = DeviceBuffer::alloc(dev, raw.len())?;
    stage.copy_from_host(raw)?;
    unsafe {
        glm5::dequant_to_f16(
            info.ggml_type as i32,
            stage.ptr,
            dst.ptr,
            n_elems as i64,
            stream,
        );
    }
    stream.sync()?;
    Ok(dst)
}

impl Glm5Vision {
    pub fn load(mmproj: &GgufFile, cfg: Glm5VisionConfig, device: i32) -> Result<Self> {
        set_device(device)?;
        let stream = CudaStream::new(device)?;

        // GLM-5.3 vision patch_embed sums the two temporal frames of conv3d:
        // v.patch_embd.weight + v.patch_embd.weight.1 -> F16 [dim, 3*p*p].
        let pw0 = load_tensor_f32_vec(mmproj, "v.patch_embd.weight")?;
        let pw1 = load_tensor_f32_vec(mmproj, "v.patch_embd.weight.1")?;
        if pw0.len() != pw1.len() {
            bail!(
                "v.patch_embd.weight.1 size mismatch: {} != {}",
                pw1.len(),
                pw0.len()
            );
        }
        let pw_sum_f16: Vec<u16> = pw0
            .iter()
            .zip(pw1.iter())
            .map(|(&a, &b)| f16::from_f32(a + b).to_bits())
            .collect();
        let patch_w_f16 = DeviceBuffer::alloc(device, pw_sum_f16.len() * 2)?;
        patch_w_f16.copy_from_host(bytemuck::cast_slice(&pw_sum_f16))?;
        let patch_b_f32 = upload_f32(device, mmproj, "v.patch_embd.bias")?;

        let mut layers = Vec::with_capacity(cfg.layers);
        for il in 0..cfg.layers {
            layers.push(Glm5VisionLayerWeights {
                ln1_w: upload_f32(device, mmproj, &format!("v.blk.{il}.ln1.weight"))?,
                ln2_w: upload_f32(device, mmproj, &format!("v.blk.{il}.ln2.weight"))?,
                attn_qkv_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.attn_qkv.weight"),
                    &stream,
                )?,
                attn_qkv_b: upload_f32(device, mmproj, &format!("v.blk.{il}.attn_qkv.bias"))?,
                attn_q_norm_w: upload_f32(device, mmproj, &format!("v.blk.{il}.attn_q_norm.weight"))?,
                attn_k_norm_w: upload_f32(device, mmproj, &format!("v.blk.{il}.attn_k_norm.weight"))?,
                attn_out_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.attn_out.weight"),
                    &stream,
                )?,
                attn_out_b: upload_f32(device, mmproj, &format!("v.blk.{il}.attn_out.bias"))?,
                ffn_gate_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.ffn_gate.weight"),
                    &stream,
                )?,
                ffn_gate_b: upload_f32(device, mmproj, &format!("v.blk.{il}.ffn_gate.bias"))?,
                ffn_up_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.ffn_up.weight"),
                    &stream,
                )?,
                ffn_up_b: upload_f32(device, mmproj, &format!("v.blk.{il}.ffn_up.bias"))?,
                ffn_down_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.ffn_down.weight"),
                    &stream,
                )?,
                ffn_down_b: upload_f32(device, mmproj, &format!("v.blk.{il}.ffn_down.bias"))?,
            });
        }

        let post_ln_w = upload_f32(device, mmproj, "v.post_ln.weight")?;

        // mm.patch_merger.weight has PyTorch/GGUF layout [out_dim, d, sm, sm].
        // Permute once at load time to [out_dim, sm, sm, d] so GEMM operates directly
        // on the 2x2 merge-window ordered ViT output [n_merged, sm*sm*d].
        let pm_raw = load_tensor_f32_vec(mmproj, "mm.patch_merger.weight")?;
        let d = cfg.dim;
        let sm = cfg.spatial_merge;
        let out_dim = cfg.out_dim;
        let expected_pm = out_dim * d * sm * sm;
        if pm_raw.len() != expected_pm {
            bail!(
                "mm.patch_merger.weight len {} != expected {}",
                pm_raw.len(),
                expected_pm
            );
        }
        let mut pm_perm_f16 = vec![0u16; expected_pm];
        for o in 0..out_dim {
            for c in 0..d {
                for ly in 0..sm {
                    for lx in 0..sm {
                        let src_idx = ((o * d + c) * sm + ly) * sm + lx;
                        let dst_idx = ((o * sm + ly) * sm + lx) * d + c;
                        pm_perm_f16[dst_idx] = f16::from_f32(pm_raw[src_idx]).to_bits();
                    }
                }
            }
        }
        let mm_patch_merger_w_f16 = DeviceBuffer::alloc(device, expected_pm * 2)?;
        mm_patch_merger_w_f16.copy_from_host(bytemuck::cast_slice(&pm_perm_f16))?;
        let mm_patch_merger_b_f32 = upload_f32(device, mmproj, "mm.patch_merger.bias")?;

        let mm_fc_w_f16 = upload_as_f16(device, mmproj, "mm.model.fc.weight", &stream)?;
        let mm_post_norm_w = upload_f32(device, mmproj, "mm.post_norm.weight")?;
        let mm_post_norm_b = upload_f32(device, mmproj, "mm.post_norm.bias")?;
        let mm_gate_w_f16 = upload_as_f16(device, mmproj, "mm.gate.weight", &stream)?;
        let mm_up_w_f16 = upload_as_f16(device, mmproj, "mm.up.weight", &stream)?;
        let mm_down_w_f16 = upload_as_f16(device, mmproj, "mm.down.weight", &stream)?;

        Ok(Self {
            cfg,
            device,
            patch_w_f16,
            patch_b_f32,
            layers,
            post_ln_w,
            mm_patch_merger_w_f16,
            mm_patch_merger_b_f32,
            mm_fc_w_f16,
            mm_post_norm_w,
            mm_post_norm_b,
            mm_gate_w_f16,
            mm_up_w_f16,
            mm_down_w_f16,
        })
    }

    pub fn encode_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        stream: &CudaStream,
    ) -> Result<(usize, usize, Vec<f32>)> {
        let p = self.cfg.patch_size;
        let sm = self.cfg.spatial_merge;
        let factor = p * sm;
        if width == 0 || height == 0 || width % factor != 0 || height % factor != 0 {
            bail!(
                "GLM-5 vision image dimensions ({width}x{height}) must be non-zero multiples of {factor}"
            );
        }
        if rgb.len() != width * height * 3 {
            bail!(
                "RGB buffer size {} != {} ({width}x{height}x3)",
                rgb.len(),
                width * height * 3
            );
        }

        let nx_p = width / p;
        let ny_p = height / p;
        let nx_m = nx_p / sm;
        let ny_m = ny_p / sm;
        let n_patches = nx_p * ny_p;
        let n_merged = nx_m * ny_m;
        if n_merged > self.cfg.max_tokens {
            bail!(
                "GLM-5 image {width}x{height} produces {n_merged} vision tokens exceeding max_tokens={}",
                self.cfg.max_tokens
            );
        }

        let d = self.cfg.dim;
        let ff = self.cfg.ff;
        let out_dim = self.cfg.out_dim;
        let merger_ff = self.cfg.merger_ff;
        let patch_in = 3 * p * p;

        let mut patches_f16_h = vec![0u16; n_patches * patch_in];
        let mut pos_yx_h = vec![0i32; n_patches * 2];

        let inv_255 = 1.0f32 / 255.0f32;
        let mean = self.cfg.image_mean;
        let inv_std = [
            1.0f32 / self.cfg.image_std[0],
            1.0f32 / self.cfg.image_std[1],
            1.0f32 / self.cfg.image_std[2],
        ];

        // Extract patches in 2x2 spatial-merge window order: (by, bx, ly, lx).
        let mut idx = 0usize;
        for by in 0..ny_m {
            for bx in 0..nx_m {
                for ly in 0..sm {
                    for lx in 0..sm {
                        let py = by * sm + ly;
                        let px = bx * sm + lx;
                        pos_yx_h[2 * idx] = py as i32;
                        pos_yx_h[2 * idx + 1] = px as i32;

                        let dst_patch = &mut patches_f16_h[idx * patch_in..(idx + 1) * patch_in];
                        for c in 0..3 {
                            let c_off = c * p * p;
                            let m = mean[c];
                            let s = inv_std[c];
                            for dy in 0..p {
                                let iy = py * p + dy;
                                let row_base = iy * width * 3;
                                for dx in 0..p {
                                    let ix = px * p + dx;
                                    let pix = rgb[row_base + ix * 3 + c] as f32;
                                    let norm = (pix * inv_255 - m) * s;
                                    dst_patch[c_off + dy * p + dx] = f16::from_f32(norm).to_bits();
                                }
                            }
                        }
                        idx += 1;
                    }
                }
            }
        }

        set_device(self.device)?;

        // Phase 1: ViT blocks (transient scratch released before merger allocation).
        let x_f16 = DeviceBuffer::alloc(self.device, n_patches * d * 2)?;
        {
            let x_f32 = DeviceBuffer::alloc(self.device, n_patches * d * 4)?;
            {
                let patches_f16 = DeviceBuffer::alloc(self.device, n_patches * patch_in * 2)?;
                unsafe {
                    copy_h2d_raw_async(patches_f16.ptr, bytemuck::cast_slice(&patches_f16_h), stream)?;
                    glm5::gemm_f16_mma_nt(
                        patches_f16.ptr,
                        self.patch_w_f16.ptr,
                        x_f32.ptr.cast::<f32>(),
                        n_patches as i32,
                        d as i32,
                        patch_in as i32,
                        1.0,
                        0.0,
                        stream,
                    );
                    hy4::vis_add_bias(
                        x_f32.ptr.cast::<f32>(),
                        self.patch_b_f32.ptr.cast::<f32>(),
                        n_patches * d,
                        d as i32,
                        stream,
                    );
                }
            }

            let pos_yx = DeviceBuffer::alloc(self.device, n_patches * 2 * 4)?;
            let qkv_f32 = DeviceBuffer::alloc(self.device, n_patches * 3 * d * 4)?;
            let attn_f32 = DeviceBuffer::alloc(self.device, n_patches * d * 4)?;
            let tmp_f32 = DeviceBuffer::alloc(self.device, n_patches * d * 4)?;
            let ffn_tile = n_patches.min(512);
            let ff_gate_f32 = DeviceBuffer::alloc(self.device, ffn_tile * ff * 4)?;
            let ff_up_f32 = DeviceBuffer::alloc(self.device, ffn_tile * ff * 4)?;
            let ff_inter_f16 = DeviceBuffer::alloc(self.device, ffn_tile * ff * 2)?;

            let attn_scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
            unsafe {
                copy_h2d_raw_async(pos_yx.ptr, bytemuck::cast_slice(&pos_yx_h), stream)?;
                for lw in &self.layers {
                    qwen4::rmsnorm_f32(
                        x_f32.ptr.cast::<f32>(),
                        lw.ln1_w.ptr.cast::<f32>(),
                        n_patches as i32,
                        d as i32,
                        self.cfg.eps,
                        tmp_f32.ptr.cast::<f32>(),
                        std::ptr::null_mut(),
                        stream,
                    );
                    glm5::f32_to_f16(
                        tmp_f32.ptr.cast::<f32>(),
                        x_f16.ptr,
                        (n_patches * d) as i32,
                        stream,
                    );
                    glm5::gemm_f16_mma_nt(
                        x_f16.ptr,
                        lw.attn_qkv_w.ptr,
                        qkv_f32.ptr.cast::<f32>(),
                        n_patches as i32,
                        (3 * d) as i32,
                        d as i32,
                        1.0,
                        0.0,
                        stream,
                    );
                    hy4::vis_add_bias(
                        qkv_f32.ptr.cast::<f32>(),
                        lw.attn_qkv_b.ptr.cast::<f32>(),
                        n_patches * 3 * d,
                        (3 * d) as i32,
                        stream,
                    );
                    glm5::vis_qk_rmsnorm_inplace(
                        qkv_f32.ptr.cast::<f32>(),
                        lw.attn_q_norm_w.ptr.cast::<f32>(),
                        lw.attn_k_norm_w.ptr.cast::<f32>(),
                        n_patches as i32,
                        self.cfg.heads as i32,
                        self.cfg.head_dim as i32,
                        self.cfg.eps,
                        stream,
                    );
                    qwen4::vis_rope2d(
                        qkv_f32.ptr.cast::<f32>(),
                        pos_yx.ptr.cast::<i32>(),
                        n_patches as i32,
                        self.cfg.heads as i32,
                        self.cfg.head_dim as i32,
                        10_000.0,
                        stream,
                    );
                    hy4::vis_flash_attn(
                        qkv_f32.ptr.cast::<f32>(),
                        n_patches as i32,
                        self.cfg.heads as i32,
                        self.cfg.head_dim as i32,
                        attn_scale,
                        attn_f32.ptr.cast::<f32>(),
                        stream,
                    );
                    glm5::f32_to_f16(
                        attn_f32.ptr.cast::<f32>(),
                        x_f16.ptr,
                        (n_patches * d) as i32,
                        stream,
                    );
                    glm5::gemm_f16_mma_nt(
                        x_f16.ptr,
                        lw.attn_out_w.ptr,
                        tmp_f32.ptr.cast::<f32>(),
                        n_patches as i32,
                        d as i32,
                        d as i32,
                        1.0,
                        0.0,
                        stream,
                    );
                    hy4::vis_residual_bias(
                        x_f32.ptr.cast::<f32>(),
                        tmp_f32.ptr.cast::<f32>(),
                        lw.attn_out_b.ptr.cast::<f32>(),
                        n_patches * d,
                        d as i32,
                        stream,
                    );

                    for row0 in (0..n_patches).step_by(ffn_tile) {
                        let nr = (n_patches - row0).min(ffn_tile);
                        let x_row = x_f32.ptr.cast::<f32>().add(row0 * d);
                        qwen4::rmsnorm_f32(
                            x_row,
                            lw.ln2_w.ptr.cast::<f32>(),
                            nr as i32,
                            d as i32,
                            self.cfg.eps,
                            tmp_f32.ptr.cast::<f32>(),
                            std::ptr::null_mut(),
                            stream,
                        );
                        glm5::f32_to_f16(
                            tmp_f32.ptr.cast::<f32>(),
                            x_f16.ptr,
                            (nr * d) as i32,
                            stream,
                        );
                        glm5::gemm_f16_mma_nt(
                            x_f16.ptr,
                            lw.ffn_gate_w.ptr,
                            ff_gate_f32.ptr.cast::<f32>(),
                            nr as i32,
                            ff as i32,
                            d as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                        hy4::vis_add_bias(
                            ff_gate_f32.ptr.cast::<f32>(),
                            lw.ffn_gate_b.ptr.cast::<f32>(),
                            nr * ff,
                            ff as i32,
                            stream,
                        );
                        glm5::gemm_f16_mma_nt(
                            x_f16.ptr,
                            lw.ffn_up_w.ptr,
                            ff_up_f32.ptr.cast::<f32>(),
                            nr as i32,
                            ff as i32,
                            d as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                        hy4::vis_add_bias(
                            ff_up_f32.ptr.cast::<f32>(),
                            lw.ffn_up_b.ptr.cast::<f32>(),
                            nr * ff,
                            ff as i32,
                            stream,
                        );
                        glm5::swiglu_clamp_to_f16(
                            ff_gate_f32.ptr.cast::<f32>(),
                            ff_up_f32.ptr.cast::<f32>(),
                            self.cfg.ffn_clamp,
                            (nr * ff) as i32,
                            ff_inter_f16.ptr,
                            stream,
                        );
                        glm5::gemm_f16_mma_nt(
                            ff_inter_f16.ptr,
                            lw.ffn_down_w.ptr,
                            tmp_f32.ptr.cast::<f32>(),
                            nr as i32,
                            d as i32,
                            ff as i32,
                            1.0,
                            0.0,
                            stream,
                        );
                        hy4::vis_residual_bias(
                            x_row,
                            tmp_f32.ptr.cast::<f32>(),
                            lw.ffn_down_b.ptr.cast::<f32>(),
                            nr * d,
                            d as i32,
                            stream,
                        );
                    }
                }

                hy4::vis_rmsnorm_inplace(
                    x_f32.ptr.cast::<f32>(),
                    self.post_ln_w.ptr.cast::<f32>(),
                    n_patches as i32,
                    d as i32,
                    self.cfg.eps,
                    stream,
                );
                glm5::f32_to_f16(
                    x_f32.ptr.cast::<f32>(),
                    x_f16.ptr,
                    (n_patches * d) as i32,
                    stream,
                );
            }
        }

        // Phase 2: 2x2 downsample + FC + LayerNorm/GELU-erf + clamped SwiGLU merger.
        let mm_tile = n_merged.min(512);
        let merge_dim = d * sm * sm;
        let mm_a_f32 = DeviceBuffer::alloc(self.device, mm_tile * out_dim * 4)?;
        let mm_a_f16 = DeviceBuffer::alloc(self.device, mm_tile * out_dim * 2)?;
        let mm_b_f32 = DeviceBuffer::alloc(self.device, mm_tile * out_dim * 4)?;
        let mm_gate_f32 = DeviceBuffer::alloc(self.device, mm_tile * merger_ff * 4)?;
        let mm_up_f32 = DeviceBuffer::alloc(self.device, mm_tile * merger_ff * 4)?;
        let mm_inter_f16 = DeviceBuffer::alloc(self.device, mm_tile * merger_ff * 2)?;

        let mut out_host = vec![0.0f32; n_merged * out_dim];
        for m0 in (0..n_merged).step_by(mm_tile) {
            let nm = (n_merged - m0).min(mm_tile);
            unsafe {
                let x_slice = (x_f16.ptr as *const u16).add(m0 * merge_dim) as *const c_void;
                glm5::gemm_f16_mma_nt(
                    x_slice,
                    self.mm_patch_merger_w_f16.ptr,
                    mm_a_f32.ptr.cast::<f32>(),
                    nm as i32,
                    out_dim as i32,
                    merge_dim as i32,
                    1.0,
                    0.0,
                    stream,
                );
                hy4::vis_add_bias(
                    mm_a_f32.ptr.cast::<f32>(),
                    self.mm_patch_merger_b_f32.ptr.cast::<f32>(),
                    nm * out_dim,
                    out_dim as i32,
                    stream,
                );
                glm5::f32_to_f16(
                    mm_a_f32.ptr.cast::<f32>(),
                    mm_a_f16.ptr,
                    (nm * out_dim) as i32,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    mm_a_f16.ptr,
                    self.mm_fc_w_f16.ptr,
                    mm_b_f32.ptr.cast::<f32>(),
                    nm as i32,
                    out_dim as i32,
                    out_dim as i32,
                    1.0,
                    0.0,
                    stream,
                );
                glm5::vis_layernorm_gelu_erf_f16(
                    mm_b_f32.ptr.cast::<f32>(),
                    self.mm_post_norm_w.ptr.cast::<f32>(),
                    self.mm_post_norm_b.ptr.cast::<f32>(),
                    nm as i32,
                    out_dim as i32,
                    1e-5,
                    mm_a_f16.ptr,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    mm_a_f16.ptr,
                    self.mm_gate_w_f16.ptr,
                    mm_gate_f32.ptr.cast::<f32>(),
                    nm as i32,
                    merger_ff as i32,
                    out_dim as i32,
                    1.0,
                    0.0,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    mm_a_f16.ptr,
                    self.mm_up_w_f16.ptr,
                    mm_up_f32.ptr.cast::<f32>(),
                    nm as i32,
                    merger_ff as i32,
                    out_dim as i32,
                    1.0,
                    0.0,
                    stream,
                );
                glm5::swiglu_clamp_to_f16(
                    mm_gate_f32.ptr.cast::<f32>(),
                    mm_up_f32.ptr.cast::<f32>(),
                    self.cfg.ffn_clamp,
                    (nm * merger_ff) as i32,
                    mm_inter_f16.ptr,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    mm_inter_f16.ptr,
                    self.mm_down_w_f16.ptr,
                    mm_a_f32.ptr.cast::<f32>(),
                    nm as i32,
                    out_dim as i32,
                    merger_ff as i32,
                    1.0,
                    0.0,
                    stream,
                );
                let out_slice = &mut out_host[m0 * out_dim..(m0 + nm) * out_dim];
                let bytes = std::slice::from_raw_parts_mut(
                    out_slice.as_mut_ptr().cast::<u8>(),
                    nm * out_dim * 4,
                );
                ampllm_cuda_kernels::copy_d2h_raw_async(bytes, mm_a_f32.ptr.cast(), stream)?;
            }
        }
        stream.sync()?;
        Ok((nx_m, ny_m, out_host))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_glm5_vision_parity_vs_sve() {
        let mmproj_path =
            std::env::var("GLM5_MMPROJ").expect("set GLM5_MMPROJ to mmproj-GLM-5.3-Flash-F16.gguf");
        let rgb_path =
            std::env::var("GLM5_VIS_RAW_RGB").expect("set GLM5_VIS_RAW_RGB to 448x448 RGB file");
        let sve_path =
            std::env::var("GLM5_VIS_REF_SVE").expect("set GLM5_VIS_REF_SVE to maya-vision .sve");

        let mm_gguf = GgufFile::open(&mmproj_path).expect("open mmproj gguf");
        let empty_vocab: Vec<String> = Vec::new();
        let vcfg = Glm5VisionConfig::from_mmproj(&mm_gguf, &empty_vocab, 4096)
            .expect("parse Glm5VisionConfig");
        let vis = Glm5Vision::load(&mm_gguf, vcfg, 0).expect("load Glm5Vision");
        let stream = CudaStream::new(0).expect("create CUDA stream");

        let rgb = std::fs::read(&rgb_path).expect("read raw RGB");
        let (nx, ny, got) = vis.encode_image(&rgb, 448, 448, &stream).expect("encode_image");

        let sve_bytes = std::fs::read(&sve_path).expect("read .sve");
        assert!(sve_bytes.len() >= 20);
        let hdr: [i32; 5] = *bytemuck::from_bytes(&sve_bytes[..20]);
        let (magic, ref_n, ref_nx, ref_ny, ref_embd) = (hdr[0], hdr[1] as usize, hdr[2] as usize, hdr[3] as usize, hdr[4] as usize);
        assert_eq!(magic, 0x31455653, "unexpected SVE magic");
        assert_eq!(nx * ny, ref_n, "token count mismatch");
        assert_eq!(ref_nx * ref_ny, ref_n, "ref grid product mismatch");
        assert_eq!(got.len(), ref_n * ref_embd, "embedding length mismatch");

        let ref_rows: &[f32] = bytemuck::cast_slice(&sve_bytes[20..]);
        assert_eq!(ref_rows.len(), got.len());

        let mut min_cos = f64::INFINITY;
        let mut worst_t = 0usize;
        let mut sum_cos = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        let mut below_99 = 0usize;
        for t in 0..ref_n {
            let g = &got[t * ref_embd..(t + 1) * ref_embd];
            let r = &ref_rows[t * ref_embd..(t + 1) * ref_embd];
            let mut dot = 0.0f64;
            let mut gn = 0.0f64;
            let mut rn = 0.0f64;
            for i in 0..ref_embd {
                let gf = g[i] as f64;
                let rf = r[i] as f64;
                dot += gf * rf;
                gn += gf * gf;
                rn += rf * rf;
                diff_sq += (gf - rf) * (gf - rf);
                ref_sq += rf * rf;
            }
            let cos = dot / (gn.sqrt() * rn.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
                worst_t = t;
            }
            if cos < 0.99 {
                below_99 += 1;
                eprintln!(
                    "  low cos token t={t} (y={}, x={}): cos={cos:.6} ref_norm={:.4} got_norm={:.4}",
                    t / nx,
                    t % nx,
                    rn.sqrt(),
                    gn.sqrt()
                );
            }
            sum_cos += cos;
        }
        let mean_cos = sum_cos / ref_n as f64;
        let rel_l2 = (diff_sq / (ref_sq + 1e-30)).sqrt();
        println!(
            "GLM5 vision parity vs maya-vision: {ref_n}x{ref_embd} ({nx}x{ny}) min_cos={min_cos:.6} (t={worst_t}) below_0.99={below_99} mean_cos={mean_cos:.6} rel_l2={rel_l2:.5}"
        );
        assert!(min_cos > 0.95, "min_cos too low: {min_cos}");
        assert!(mean_cos > 0.999, "mean_cos too low: {mean_cos}");
    }
}
