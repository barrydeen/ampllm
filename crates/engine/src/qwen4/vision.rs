use ampllm_cuda_kernels::{
    copy_h2d_raw_async, glm5, hy4, qwen4, set_device, CudaStream, DeviceBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Qwen4VisionConfig};
use anyhow::{bail, Result};
use half::f16;
use std::ffi::c_void;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct Qwen4ImageSpan {
    pub start_pos: usize,
    pub nx: usize,
    pub ny: usize,
    pub rows: Vec<f32>,
}

struct VisionLayerWeights {
    ln1_w: DeviceBuffer,
    ln1_b: DeviceBuffer,
    ln2_w: DeviceBuffer,
    ln2_b: DeviceBuffer,
    attn_qkv_w: DeviceBuffer,
    attn_qkv_b: DeviceBuffer,
    attn_out_w: DeviceBuffer,
    attn_out_b: DeviceBuffer,
    ffn_up_w: DeviceBuffer,
    ffn_up_b: DeviceBuffer,
    ffn_down_w: DeviceBuffer,
    ffn_down_b: DeviceBuffer,
}

#[derive(Default)]
struct VisionScratch {
    cap_patches: usize,
    cap_merged: usize,
    patches_f16: Option<DeviceBuffer>,
    pos_emb_f32: Option<DeviceBuffer>,
    pos_yx: Option<DeviceBuffer>,
    x_f32: Option<DeviceBuffer>,
    x_f16: Option<DeviceBuffer>,
    qkv_f32: Option<DeviceBuffer>,
    attn_f32: Option<DeviceBuffer>,
    attn_f16: Option<DeviceBuffer>,
    tmp_f32: Option<DeviceBuffer>,
    ff_f32: Option<DeviceBuffer>,
    ff_f16: Option<DeviceBuffer>,
    mm_mid_f32: Option<DeviceBuffer>,
    mm_mid_f16: Option<DeviceBuffer>,
    out_f32: Option<DeviceBuffer>,
}

impl VisionScratch {
    fn ensure(
        &mut self,
        dev: i32,
        n_patches: usize,
        n_merged: usize,
        cfg: &Qwen4VisionConfig,
    ) -> Result<()> {
        let d = cfg.dim;
        let ff = cfg.ff;
        let patch_in = 3 * cfg.patch_size * cfg.patch_size;
        let merge_dim = d * cfg.spatial_merge * cfg.spatial_merge;
        if n_patches > self.cap_patches {
            self.patches_f16 = Some(DeviceBuffer::alloc(dev, n_patches * patch_in * 2)?);
            self.pos_emb_f32 = Some(DeviceBuffer::alloc(dev, n_patches * d * 4)?);
            self.pos_yx = Some(DeviceBuffer::alloc(dev, n_patches * 2 * 4)?);
            self.x_f32 = Some(DeviceBuffer::alloc(dev, n_patches * d * 4)?);
            self.x_f16 = Some(DeviceBuffer::alloc(dev, n_patches * d * 2)?);
            self.qkv_f32 = Some(DeviceBuffer::alloc(dev, n_patches * 3 * d * 4)?);
            self.attn_f32 = Some(DeviceBuffer::alloc(dev, n_patches * d * 4)?);
            self.attn_f16 = Some(DeviceBuffer::alloc(dev, n_patches * d * 2)?);
            self.tmp_f32 = Some(DeviceBuffer::alloc(dev, n_patches * d * 4)?);
            self.ff_f32 = Some(DeviceBuffer::alloc(dev, n_patches * ff * 4)?);
            self.ff_f16 = Some(DeviceBuffer::alloc(dev, n_patches * ff * 2)?);
            self.cap_patches = n_patches;
        }
        if n_merged > self.cap_merged {
            self.mm_mid_f32 = Some(DeviceBuffer::alloc(dev, n_merged * merge_dim * 4)?);
            self.mm_mid_f16 = Some(DeviceBuffer::alloc(dev, n_merged * merge_dim * 2)?);
            self.out_f32 = Some(DeviceBuffer::alloc(dev, n_merged * cfg.out_dim * 4)?);
            self.cap_merged = n_merged;
        }
        Ok(())
    }
}

pub struct Qwen4Vision {
    pub cfg: Qwen4VisionConfig,
    pub device: i32,
    pub use_gelu_tanh: bool,
    patch_w_f16: DeviceBuffer,
    patch_b_f32: DeviceBuffer,
    pos_embd_host: Vec<f32>,
    layers: Vec<VisionLayerWeights>,
    post_ln_w: DeviceBuffer,
    post_ln_b: DeviceBuffer,
    mm0_w_f16: DeviceBuffer,
    mm0_b_f32: DeviceBuffer,
    mm2_w_f16: DeviceBuffer,
    mm2_b_f32: DeviceBuffer,
    scratch: Mutex<VisionScratch>,
}

fn upload_f32(dev: i32, gguf: &GgufFile, name: &str) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    let raw = gguf.tensor_bytes(name)?;
    if info.ggml_type == GgmlType::F32 {
        let buf = DeviceBuffer::alloc(dev, raw.len())?;
        buf.copy_from_host(raw)?;
        Ok(buf)
    } else {
        let mut f32_vals = vec![0.0f32; info.n_elements()];
        ampllm_gguf::dequantize_row_to_f32(info.ggml_type, raw, &mut f32_vals);
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

impl Qwen4Vision {
    pub fn load(mmproj: &GgufFile, cfg: Qwen4VisionConfig, device: i32) -> Result<Self> {
        set_device(device)?;
        let stream = CudaStream::new(device)?;
        let use_gelu_tanh = mmproj.get_bool("clip.use_gelu").unwrap_or(true);

        // Sum temporal pair conv weights (v.patch_embd.weight + v.patch_embd.weight.1) for still images.
        let pw0 = load_tensor_f32_vec(mmproj, "v.patch_embd.weight")?;
        let mut pw_sum_f16 = Vec::with_capacity(pw0.len());
        if mmproj.has_tensor("v.patch_embd.weight.1") {
            let pw1 = load_tensor_f32_vec(mmproj, "v.patch_embd.weight.1")?;
            if pw1.len() != pw0.len() {
                bail!("v.patch_embd.weight.1 size mismatch: {} != {}", pw1.len(), pw0.len());
            }
            for (&a, &b) in pw0.iter().zip(pw1.iter()) {
                pw_sum_f16.push(f16::from_f32(a + b).to_bits());
            }
        } else {
            for &a in &pw0 {
                pw_sum_f16.push(f16::from_f32(a).to_bits());
            }
        }
        let patch_w_f16 = DeviceBuffer::alloc(device, pw_sum_f16.len() * 2)?;
        patch_w_f16.copy_from_host(bytemuck::cast_slice(&pw_sum_f16))?;
        let patch_b_f32 = upload_f32(device, mmproj, "v.patch_embd.bias")?;
        let pos_embd_host = load_tensor_f32_vec(mmproj, "v.position_embd.weight")?;
        if pos_embd_host.len() != cfg.grid * cfg.grid * cfg.dim {
            bail!(
                "v.position_embd.weight len {} != {}",
                pos_embd_host.len(),
                cfg.grid * cfg.grid * cfg.dim
            );
        }

        let mut layers = Vec::with_capacity(cfg.layers);
        for il in 0..cfg.layers {
            layers.push(VisionLayerWeights {
                ln1_w: upload_f32(device, mmproj, &format!("v.blk.{il}.ln1.weight"))?,
                ln1_b: upload_f32(device, mmproj, &format!("v.blk.{il}.ln1.bias"))?,
                ln2_w: upload_f32(device, mmproj, &format!("v.blk.{il}.ln2.weight"))?,
                ln2_b: upload_f32(device, mmproj, &format!("v.blk.{il}.ln2.bias"))?,
                attn_qkv_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.attn_qkv.weight"),
                    &stream,
                )?,
                attn_qkv_b: upload_f32(device, mmproj, &format!("v.blk.{il}.attn_qkv.bias"))?,
                attn_out_w: upload_as_f16(
                    device,
                    mmproj,
                    &format!("v.blk.{il}.attn_out.weight"),
                    &stream,
                )?,
                attn_out_b: upload_f32(device, mmproj, &format!("v.blk.{il}.attn_out.bias"))?,
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
        let post_ln_b = upload_f32(device, mmproj, "v.post_ln.bias")?;
        let mm0_w_f16 = upload_as_f16(device, mmproj, "mm.0.weight", &stream)?;
        let mm0_b_f32 = upload_f32(device, mmproj, "mm.0.bias")?;
        let mm2_w_f16 = upload_as_f16(device, mmproj, "mm.2.weight", &stream)?;
        let mm2_b_f32 = upload_f32(device, mmproj, "mm.2.bias")?;

        Ok(Self {
            cfg,
            device,
            use_gelu_tanh,
            patch_w_f16,
            patch_b_f32,
            pos_embd_host,
            layers,
            post_ln_w,
            post_ln_b,
            mm0_w_f16,
            mm0_b_f32,
            mm2_w_f16,
            mm2_b_f32,
            scratch: Mutex::new(VisionScratch::default()),
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
                "Qwen4 vision image dimensions ({width}x{height}) must be non-zero multiples of {factor}"
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
        let d = self.cfg.dim;
        let ff = self.cfg.ff;
        let patch_in = 3 * p * p;

        let mut patches_f16_h = vec![0u16; n_patches * patch_in];
        let mut pos_emb_h = vec![0.0f32; n_patches * d];
        let mut pos_yx_h = vec![0i32; n_patches * 2];

        let inv_255 = 1.0f32 / 255.0f32;
        let mean = self.cfg.image_mean;
        let inv_std = [
            1.0f32 / self.cfg.image_std[0],
            1.0f32 / self.cfg.image_std[1],
            1.0f32 / self.cfg.image_std[2],
        ];
        let grid = self.cfg.grid;
        let max_g = (grid - 1) as i32;

        // Extract patches and bilinearly interpolated 2D pos embeddings in exact 2x2 merge order:
        // (by, bx, ly, lx) matching llama.cpp clip_graph_qwen3vl.
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

                        let dst_pos = &mut pos_emb_h[idx * d..(idx + 1) * d];
                        if ny_p == grid && nx_p == grid {
                            let src = &self.pos_embd_host[(py * grid + px) * d..(py * grid + px + 1) * d];
                            dst_pos.copy_from_slice(src);
                        } else {
                            let fy = if ny_p > 1 {
                                (py as f32) * (max_g as f32) / ((ny_p - 1) as f32)
                            } else {
                                0.0
                            };
                            let fx = if nx_p > 1 {
                                (px as f32) * (max_g as f32) / ((nx_p - 1) as f32)
                            } else {
                                0.0
                            };
                            let y0f = fy.floor();
                            let x0f = fx.floor();
                            let y0 = (y0f as i32).clamp(0, max_g) as usize;
                            let y1 = ((y0f as i32) + 1).clamp(0, max_g) as usize;
                            let x0 = (x0f as i32).clamp(0, max_g) as usize;
                            let x1 = ((x0f as i32) + 1).clamp(0, max_g) as usize;
                            let wy1 = (fy - y0f).clamp(0.0, 1.0);
                            let wx1 = (fx - x0f).clamp(0.0, 1.0);
                            let wy0 = 1.0 - wy1;
                            let wx0 = 1.0 - wx1;
                            let w00 = wy0 * wx0;
                            let w01 = wy0 * wx1;
                            let w10 = wy1 * wx0;
                            let w11 = wy1 * wx1;
                            let r00 = &self.pos_embd_host[(y0 * grid + x0) * d..(y0 * grid + x0 + 1) * d];
                            let r01 = &self.pos_embd_host[(y0 * grid + x1) * d..(y0 * grid + x1 + 1) * d];
                            let r10 = &self.pos_embd_host[(y1 * grid + x0) * d..(y1 * grid + x0 + 1) * d];
                            let r11 = &self.pos_embd_host[(y1 * grid + x1) * d..(y1 * grid + x1 + 1) * d];
                            for i in 0..d {
                                dst_pos[i] = w00 * r00[i] + w01 * r01[i] + w10 * r10[i] + w11 * r11[i];
                            }
                        }

                        idx += 1;
                    }
                }
            }
        }

        set_device(self.device)?;
        let mut sc = self.scratch.lock().map_err(|_| anyhow::anyhow!("vision scratch poisoned"))?;
        sc.ensure(self.device, n_patches, n_merged, &self.cfg)?;

        let patches_f16 = sc.patches_f16.as_ref().unwrap();
        let pos_emb_f32 = sc.pos_emb_f32.as_ref().unwrap();
        let pos_yx = sc.pos_yx.as_ref().unwrap();
        let x_f32 = sc.x_f32.as_ref().unwrap();
        let x_f16 = sc.x_f16.as_ref().unwrap();
        let qkv_f32 = sc.qkv_f32.as_ref().unwrap();
        let attn_f32 = sc.attn_f32.as_ref().unwrap();
        let attn_f16 = sc.attn_f16.as_ref().unwrap();
        let tmp_f32 = sc.tmp_f32.as_ref().unwrap();
        let ff_f32 = sc.ff_f32.as_ref().unwrap();
        let ff_f16 = sc.ff_f16.as_ref().unwrap();
        let mm_mid_f32 = sc.mm_mid_f32.as_ref().unwrap();
        let mm_mid_f16 = sc.mm_mid_f16.as_ref().unwrap();
        let out_f32 = sc.out_f32.as_ref().unwrap();

        unsafe {
            copy_h2d_raw_async(patches_f16.ptr, bytemuck::cast_slice(&patches_f16_h), stream)?;
            copy_h2d_raw_async(pos_emb_f32.ptr, bytemuck::cast_slice(&pos_emb_h), stream)?;
            copy_h2d_raw_async(pos_yx.ptr, bytemuck::cast_slice(&pos_yx_h), stream)?;

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
            hy4::add_f32(
                x_f32.ptr.cast::<f32>(),
                pos_emb_f32.ptr.cast::<f32>(),
                n_patches * d,
                stream,
            );

            let attn_scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
            let act_mode = if self.use_gelu_tanh { 1 } else { 2 };
            for lw in &self.layers {
                hy4::vis_layernorm_f16(
                    x_f32.ptr.cast::<f32>(),
                    lw.ln1_w.ptr.cast::<f32>(),
                    lw.ln1_b.ptr.cast::<f32>(),
                    n_patches as i32,
                    d as i32,
                    self.cfg.eps,
                    x_f16.ptr.cast::<u16>(),
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
                    attn_f16.ptr as *mut c_void,
                    (n_patches * d) as i32,
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    attn_f16.ptr,
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

                hy4::vis_layernorm_f16(
                    x_f32.ptr.cast::<f32>(),
                    lw.ln2_w.ptr.cast::<f32>(),
                    lw.ln2_b.ptr.cast::<f32>(),
                    n_patches as i32,
                    d as i32,
                    self.cfg.eps,
                    x_f16.ptr.cast::<u16>(),
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    x_f16.ptr,
                    lw.ffn_up_w.ptr,
                    ff_f32.ptr.cast::<f32>(),
                    n_patches as i32,
                    ff as i32,
                    d as i32,
                    1.0,
                    0.0,
                    stream,
                );
                hy4::vis_bias_act_f16(
                    ff_f32.ptr.cast::<f32>(),
                    lw.ffn_up_b.ptr.cast::<f32>(),
                    n_patches * ff,
                    ff as i32,
                    act_mode,
                    ff_f16.ptr.cast::<u16>(),
                    stream,
                );
                glm5::gemm_f16_mma_nt(
                    ff_f16.ptr,
                    lw.ffn_down_w.ptr,
                    tmp_f32.ptr.cast::<f32>(),
                    n_patches as i32,
                    d as i32,
                    ff as i32,
                    1.0,
                    0.0,
                    stream,
                );
                hy4::vis_residual_bias(
                    x_f32.ptr.cast::<f32>(),
                    tmp_f32.ptr.cast::<f32>(),
                    lw.ffn_down_b.ptr.cast::<f32>(),
                    n_patches * d,
                    d as i32,
                    stream,
                );
            }

            // Post-LN directly produces F16 [n_patches, d] already in (by, bx, ly, lx) order = [n_merged, 4*d].
            hy4::vis_layernorm_f16(
                x_f32.ptr.cast::<f32>(),
                self.post_ln_w.ptr.cast::<f32>(),
                self.post_ln_b.ptr.cast::<f32>(),
                n_patches as i32,
                d as i32,
                self.cfg.eps,
                x_f16.ptr.cast::<u16>(),
                stream,
            );
            let merge_dim = d * sm * sm;
            glm5::gemm_f16_mma_nt(
                x_f16.ptr,
                self.mm0_w_f16.ptr,
                mm_mid_f32.ptr.cast::<f32>(),
                n_merged as i32,
                merge_dim as i32,
                merge_dim as i32,
                1.0,
                0.0,
                stream,
            );
            hy4::vis_bias_act_f16(
                mm_mid_f32.ptr.cast::<f32>(),
                self.mm0_b_f32.ptr.cast::<f32>(),
                n_merged * merge_dim,
                merge_dim as i32,
                act_mode,
                mm_mid_f16.ptr.cast::<u16>(),
                stream,
            );
            glm5::gemm_f16_mma_nt(
                mm_mid_f16.ptr,
                self.mm2_w_f16.ptr,
                out_f32.ptr.cast::<f32>(),
                n_merged as i32,
                self.cfg.out_dim as i32,
                merge_dim as i32,
                1.0,
                0.0,
                stream,
            );
            hy4::vis_add_bias(
                out_f32.ptr.cast::<f32>(),
                self.mm2_b_f32.ptr.cast::<f32>(),
                n_merged * self.cfg.out_dim,
                self.cfg.out_dim as i32,
                stream,
            );
        }

        let out_elems = n_merged * self.cfg.out_dim;
        let mut out_host = vec![0.0f32; out_elems];
        out_f32.copy_to_host_async(bytemuck::cast_slice_mut(&mut out_host), stream)?;
        stream.sync()?;
        Ok((nx_m, ny_m, out_host))
    }
}
