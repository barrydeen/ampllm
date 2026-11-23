use ampllm_cuda_kernels::{
    copy_d2h_raw_async, copy_h2d_raw_async, mm3, set_device, CudaEvent, CudaStream, DeviceBuffer,
    PinnedHostBuffer,
};
use ampllm_gguf::{Mm3VisionConfig, SafeTensorsFile};
use anyhow::{bail, Result};
use half::bf16;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Mm3ImageSpan {
    pub start_pos: usize,
    pub n_tokens: usize,
    pub rows: Vec<f32>,
}

struct VisionLayerCpu<'a> {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    q_w: &'a [u8],
    k_w: &'a [u8],
    v_w: &'a [u8],
    q_b: Vec<f32>,
    k_b: Vec<f32>,
    v_b: Vec<f32>,
    o_w: &'a [u8],
    o_b: Vec<f32>,
    w1: &'a [u8],
    b1: Vec<f32>,
    w2: &'a [u8],
    b2: Vec<f32>,
}

pub struct Mm3Vision {
    pub cfg: Mm3VisionConfig,
    st: SafeTensorsFile,
    inv_freq: Vec<f32>,
    w_patch_f32: Vec<f32>,
    pre_ln_w: Vec<f32>,
    pre_ln_b: Vec<f32>,
    proj_l1_b: Vec<f32>,
    proj_l2_b: Vec<f32>,
    merge_l1_b: Vec<f32>,
    merge_l2_b: Vec<f32>,
}

impl Mm3Vision {
    pub fn load(safetensors_path: &Path, cfg: Mm3VisionConfig) -> Result<Self> {
        let st = SafeTensorsFile::open(safetensors_path)?;
        let prefix = "vision_tower.vision_model.";
        if !st.has_tensor(&format!("{prefix}pre_layrnorm.weight")) {
            bail!(
                "vision safetensors missing {}pre_layrnorm.weight",
                prefix
            );
        }
        let nf = (cfg.head_dim / 3) / 2;
        if cfg.head_dim < 6 || cfg.temporal_patch_size != 2 || cfg.spatial_merge_size != 2 {
            bail!(
                "unsupported MM3 vision geometry: hd={} tps={} merge={}",
                cfg.head_dim,
                cfg.temporal_patch_size,
                cfg.spatial_merge_size
            );
        }
        let inv_freq: Vec<f32> = (0..nf)
            .map(|j| 1.0f32 / cfg.rope_theta.powf((2 * j) as f32 / (2 * nf) as f32))
            .collect();

        let w_patch_f32 = st.tensor_to_f32(&format!("{prefix}embeddings.patch_embedding.weight"))?;
        let pre_ln_w = st.tensor_to_f32(&format!("{prefix}pre_layrnorm.weight"))?;
        let pre_ln_b = st.tensor_to_f32(&format!("{prefix}pre_layrnorm.bias"))?;
        let proj_l1_b = st.tensor_to_f32("multi_modal_projector.linear_1.bias")?;
        let proj_l2_b = st.tensor_to_f32("multi_modal_projector.linear_2.bias")?;
        let merge_l1_b = st.tensor_to_f32("patch_merge_mlp.linear_1.bias")?;
        let merge_l2_b = st.tensor_to_f32("patch_merge_mlp.linear_2.bias")?;

        Ok(Self {
            cfg,
            st,
            inv_freq,
            w_patch_f32,
            pre_ln_w,
            pre_ln_b,
            proj_l1_b,
            proj_l2_b,
            merge_l1_b,
            merge_l2_b,
        })
    }

    #[inline]
    pub fn tokens_for_size(&self, width: usize, height: usize) -> Option<usize> {
        let f = self.cfg.patch_size * self.cfg.spatial_merge_size;
        if width == 0
            || height == 0
            || width % f != 0
            || height % f != 0
            || width * height > self.cfg.max_pixels
            || width * height < self.cfg.min_pixels
        {
            return None;
        }
        Some((height / f) * (width / f))
    }

    pub fn encode_image(&self, rgb: &[u8], width: usize, height: usize, device: i32) -> Result<Vec<f32>> {
        let Some(nb) = self.tokens_for_size(width, height) else {
            let f = self.cfg.patch_size * self.cfg.spatial_merge_size;
            bail!(
                "MM3 vision invalid dimensions {width}x{height} (must be multiples of {f} with area in [{}..={}])",
                self.cfg.min_pixels,
                self.cfg.max_pixels
            );
        };
        let expected_rgb = width * height * 3;
        if rgb.len() != expected_rgb {
            bail!("MM3 vision expected {expected_rgb} RGB bytes, got {}", rgb.len());
        }

        let ps = self.cfg.patch_size;
        let tps = self.cfg.temporal_patch_size;
        let merge = self.cfg.spatial_merge_size;
        let d = self.cfg.hidden_size;
        let ff = self.cfg.intermediate_size;
        let heads = self.cfg.num_heads;
        let hd = self.cfg.head_dim;
        let nf = (hd / 3) / 2;
        let p_dim = 3 * tps * ps * ps;
        let gh = height / ps;
        let gw = width / ps;
        let n_patches = gh * gw;
        let proj = self.cfg.projection_dim;
        let out_dim = self.cfg.projection_dim;

        // Build patches & 3D coordinates in MiniMax 2x2 merge-block order.
        let mut h_px = vec![0.0f32; n_patches * p_dim];
        let mut h_pos = vec![0i32; n_patches * 3];
        for p in 0..n_patches {
            let b = p / (merge * merge);
            let inner = p % (merge * merge);
            let bh = b / (gw / merge);
            let bw = b % (gw / merge);
            let py = bh * merge + inner / merge;
            let px = bw * merge + inner % merge;
            h_pos[p * 3] = 0;
            h_pos[p * 3 + 1] = py as i32;
            h_pos[p * 3 + 2] = px as i32;
            let dst = &mut h_px[p * p_dim..(p + 1) * p_dim];
            for c in 0..3 {
                let m = self.cfg.image_mean[c];
                let s = self.cfg.image_std[c];
                for t in 0..tps {
                    for yy in 0..ps {
                        let y_abs = py * ps + yy;
                        for xx in 0..ps {
                            let x_abs = px * ps + xx;
                            let val = (rgb[(y_abs * width + x_abs) * 3 + c] as f32) / 255.0f32;
                            dst[((c * tps + t) * ps + yy) * ps + xx] = (val - m) / s;
                        }
                    }
                }
            }
        }

        // Gather layer weight views cleanly before launching async double-buffered stream.
        let prefix = "vision_tower.vision_model.encoder.layers.";
        let mut layers = Vec::with_capacity(self.cfg.num_layers);
        for li in 0..self.cfg.num_layers {
            let lp = format!("{prefix}{li}.");
            layers.push(VisionLayerCpu {
                ln1_w: self.st.tensor_to_f32(&format!("{lp}layer_norm1.weight"))?,
                ln1_b: self.st.tensor_to_f32(&format!("{lp}layer_norm1.bias"))?,
                ln2_w: self.st.tensor_to_f32(&format!("{lp}layer_norm2.weight"))?,
                ln2_b: self.st.tensor_to_f32(&format!("{lp}layer_norm2.bias"))?,
                q_w: self.st.tensor_bytes(&format!("{lp}self_attn.q_proj.weight"))?,
                k_w: self.st.tensor_bytes(&format!("{lp}self_attn.k_proj.weight"))?,
                v_w: self.st.tensor_bytes(&format!("{lp}self_attn.v_proj.weight"))?,
                q_b: self.st.tensor_to_f32(&format!("{lp}self_attn.q_proj.bias"))?,
                k_b: self.st.tensor_to_f32(&format!("{lp}self_attn.k_proj.bias"))?,
                v_b: self.st.tensor_to_f32(&format!("{lp}self_attn.v_proj.bias"))?,
                o_w: self.st.tensor_bytes(&format!("{lp}self_attn.out_proj.weight"))?,
                o_b: self.st.tensor_to_f32(&format!("{lp}self_attn.out_proj.bias"))?,
                w1: self.st.tensor_bytes(&format!("{lp}mlp.fc1.weight"))?,
                b1: self.st.tensor_to_f32(&format!("{lp}mlp.fc1.bias"))?,
                w2: self.st.tensor_bytes(&format!("{lp}mlp.fc2.weight"))?,
                b2: self.st.tensor_to_f32(&format!("{lp}mlp.fc2.bias"))?,
            });
        }

        let proj_mats: [(&[u8], usize, usize); 4] = [
            (
                self.st.tensor_bytes("multi_modal_projector.linear_1.weight")?,
                proj,
                d,
            ),
            (
                self.st.tensor_bytes("multi_modal_projector.linear_2.weight")?,
                out_dim,
                proj,
            ),
            (
                self.st.tensor_bytes("patch_merge_mlp.linear_1.weight")?,
                proj,
                out_dim * merge * merge,
            ),
            (
                self.st.tensor_bytes("patch_merge_mlp.linear_2.weight")?,
                out_dim,
                proj,
            ),
        ];

        // Compute transient per-layer / per-chunk staging slot size (~48 MiB).
        #[inline]
        const fn align256(n: usize) -> usize {
            (n + 255) & !255
        }
        let dd2 = d * d * 2;
        let layer_stage_bytes = align256(
            align256(d * 4) * 4
                + align256(3 * dd2)
                + align256(3 * d * 4)
                + align256(dd2)
                + align256(d * 4)
                + align256(ff * d * 2)
                + align256(ff * 4)
                + align256(d * ff * 2)
                + align256(d * 4),
        );
        let max_proj_chunk_bytes = 48usize * 1024 * 1024;
        let slot_bytes = layer_stage_bytes.max(align256(max_proj_chunk_bytes));

        set_device(device)?;
        let stream = CudaStream::new(device)?;
        let copy_stream = CudaStream::new(device)?;
        let ev_up = [CudaEvent::new(device)?, CudaEvent::new(device)?];
        let ev_free = [CudaEvent::new(device)?, CudaEvent::new(device)?];

        let d_slot = [
            DeviceBuffer::alloc(device, slot_bytes)?,
            DeviceBuffer::alloc(device, slot_bytes)?,
        ];
        let mut bounce = [
            PinnedHostBuffer::alloc(slot_bytes)?,
            PinnedHostBuffer::alloc(slot_bytes)?,
        ];

        // Upload small F32 vectors once.
        let alloc_f32 = |v: &[f32]| -> Result<DeviceBuffer> {
            let b = DeviceBuffer::alloc(device, v.len() * 4)?;
            b.copy_from_host(bytemuck::cast_slice(v))?;
            Ok(b)
        };
        let d_wpatch = alloc_f32(&self.w_patch_f32)?;
        let d_prew = alloc_f32(&self.pre_ln_w)?;
        let d_preb = alloc_f32(&self.pre_ln_b)?;
        let d_inv = alloc_f32(&self.inv_freq)?;
        let d_pb1 = alloc_f32(&self.proj_l1_b)?;
        let d_pb2 = alloc_f32(&self.proj_l2_b)?;
        let d_pmb1 = alloc_f32(&self.merge_l1_b)?;
        let d_pmb2 = alloc_f32(&self.merge_l2_b)?;

        let d_px = alloc_f32(&h_px)?;
        let d_pos = {
            let b = DeviceBuffer::alloc(device, h_pos.len() * 4)?;
            b.copy_from_host(bytemuck::cast_slice(&h_pos))?;
            b
        };

        let w_max = ff.max(proj).max(out_dim);
        let d_h = DeviceBuffer::alloc(device, n_patches * d * 4)?;
        let d_qkv = DeviceBuffer::alloc(device, (n_patches * 3 * d * 4).max(nb * proj * 2))?;
        let d_att = DeviceBuffer::alloc(device, n_patches * d * 4)?;
        let d_big = DeviceBuffer::alloc(device, n_patches * w_max * 4)?;
        let d_out = DeviceBuffer::alloc(device, nb * out_dim * 4)?;
        let d_x16 = DeviceBuffer::alloc(device, n_patches * w_max * 2)?;

        for b in 0..2 {
            ev_free[b].record(&stream)?;
            ev_up[b].record(&copy_stream)?;
        }

        // Helper to stage a single transformer layer into bounce slot b and kick off H2D copy.
        #[derive(Clone, Copy)]
        struct LOff {
            ln1w: usize,
            ln1b: usize,
            ln2w: usize,
            ln2b: usize,
            wqkv: usize,
            bqkv: usize,
            wo: usize,
            bo: usize,
            w1: usize,
            b1: usize,
            w2: usize,
            b2: usize,
        }

        enum StageTask<'a> {
            Layer(&'a VisionLayerCpu<'a>, LOff, usize),
            ProjSlice(&'a [u8], usize, usize, usize),
        }

        let mut stage_tasks: Vec<StageTask<'_>> = Vec::with_capacity(self.cfg.num_layers + 12);
        let mut layer_offs = Vec::with_capacity(self.cfg.num_layers);
        for lc in &layers {
            let mut off = 0usize;
            let mut take = |bytes: usize| -> usize {
                let cur = off;
                off += align256(bytes);
                cur
            };
            let lo = LOff {
                ln1w: take(d * 4),
                ln1b: take(d * 4),
                ln2w: take(d * 4),
                ln2b: take(d * 4),
                wqkv: take(3 * dd2),
                bqkv: take(3 * d * 4),
                wo: take(dd2),
                bo: take(d * 4),
                w1: take(ff * d * 2),
                b1: take(ff * 4),
                w2: take(d * ff * 2),
                b2: take(d * 4),
            };
            layer_offs.push(lo);
            stage_tasks.push(StageTask::Layer(lc, lo, off));
        }

        let mut proj_task_ranges = [(0usize, 0usize); 4];
        for (mi, &(src, n_tot, k_tot)) in proj_mats.iter().enumerate() {
            let start = stage_tasks.len();
            let row_bytes = k_tot * 2;
            let rc = (slot_bytes / row_bytes).max(1).min(n_tot);
            let mut n0 = 0usize;
            while n0 < n_tot {
                let nc = rc.min(n_tot - n0);
                stage_tasks.push(StageTask::ProjSlice(src, n0, nc, k_tot));
                n0 += nc;
            }
            proj_task_ranges[mi] = (start, stage_tasks.len());
        }

        let upload_stage = |idx: usize,
                            bounce: &mut [PinnedHostBuffer; 2],
                            ev_up: &[CudaEvent; 2],
                            ev_free: &[CudaEvent; 2]|
         -> Result<()> {
            if idx >= stage_tasks.len() {
                return Ok(());
            }
            let b = idx & 1;
            ev_up[b].sync()?;
            let dst = bounce[b].as_mut_slice();
            let copy_f32 = |buf: &mut [u8], o: usize, v: &[f32]| {
                let raw = bytemuck::cast_slice(v);
                buf[o..o + raw.len()].copy_from_slice(raw);
            };
            let total_bytes = match stage_tasks[idx] {
                StageTask::Layer(lc, lo, sz) => {
                    copy_f32(dst, lo.ln1w, &lc.ln1_w);
                    copy_f32(dst, lo.ln1b, &lc.ln1_b);
                    copy_f32(dst, lo.ln2w, &lc.ln2_w);
                    copy_f32(dst, lo.ln2b, &lc.ln2_b);
                    dst[lo.wqkv..lo.wqkv + dd2].copy_from_slice(lc.q_w);
                    dst[lo.wqkv + dd2..lo.wqkv + 2 * dd2].copy_from_slice(lc.k_w);
                    dst[lo.wqkv + 2 * dd2..lo.wqkv + 3 * dd2].copy_from_slice(lc.v_w);
                    copy_f32(dst, lo.bqkv, &lc.q_b);
                    copy_f32(dst, lo.bqkv + d * 4, &lc.k_b);
                    copy_f32(dst, lo.bqkv + 2 * d * 4, &lc.v_b);
                    dst[lo.wo..lo.wo + lc.o_w.len()].copy_from_slice(lc.o_w);
                    copy_f32(dst, lo.bo, &lc.o_b);
                    dst[lo.w1..lo.w1 + lc.w1.len()].copy_from_slice(lc.w1);
                    copy_f32(dst, lo.b1, &lc.b1);
                    dst[lo.w2..lo.w2 + lc.w2.len()].copy_from_slice(lc.w2);
                    copy_f32(dst, lo.b2, &lc.b2);
                    sz
                }
                StageTask::ProjSlice(src, n0, nc, k_tot) => {
                    let row_bytes = k_tot * 2;
                    let off_b = n0 * row_bytes;
                    let nb_bytes = nc * row_bytes;
                    dst[..nb_bytes].copy_from_slice(&src[off_b..off_b + nb_bytes]);
                    nb_bytes
                }
            };
            copy_stream.wait_event(&ev_free[b])?;
            unsafe {
                copy_h2d_raw_async(d_slot[b].ptr, &dst[..total_bytes], &copy_stream)?;
            }
            ev_up[b].record(&copy_stream)?;
            Ok(())
        };

        upload_stage(0, &mut bounce, &ev_up, &ev_free)?;
        upload_stage(1, &mut bounce, &ev_up, &ev_free)?;

        // Patch embedding (F32 GEMM) + pre-layernorm
        unsafe {
            mm3::vision_patch_embed_f32(
                d_px.ptr as *const f32,
                d_wpatch.ptr as *const f32,
                n_patches as i32,
                d as i32,
                p_dim as i32,
                d_h.ptr as *mut f32,
                &stream,
            );
            mm3::vision_layernorm(
                d_h.ptr as *const f32,
                d_prew.ptr as *const f32,
                d_preb.ptr as *const f32,
                n_patches as i32,
                d as i32,
                self.cfg.layer_norm_eps,
                d_h.ptr as *mut f32,
                std::ptr::null_mut(),
                &stream,
            );
        }

        let nd = n_patches * d;
        let attn_scale = 1.0f32 / (hd as f32).sqrt();

        // 32 Vision encoder layers using zero-cuBLAS custom BF16 MMA & attention kernels.
        for li in 0..self.cfg.num_layers {
            let b = li & 1;
            stream.wait_event(&ev_up[b])?;
            let sb = d_slot[b].ptr as usize;
            let lo = layer_offs[li];
            let fp = |off: usize| (sb + off) as *const f32;
            let u16p = |off: usize| (sb + off) as *const u16;
            unsafe {
                mm3::vision_layernorm(
                    d_h.ptr as *const f32,
                    fp(lo.ln1w),
                    fp(lo.ln1b),
                    n_patches as i32,
                    d as i32,
                    self.cfg.layer_norm_eps,
                    std::ptr::null_mut(),
                    d_x16.ptr as *mut u16,
                    &stream,
                );
                mm3::bf16_gemm_mma(
                    d_x16.ptr as *const u16,
                    d as i32,
                    u16p(lo.wqkv),
                    d as i32,
                    d_qkv.ptr as *mut f32,
                    (3 * d) as i32,
                    n_patches as i32,
                    (3 * d) as i32,
                    d as i32,
                    1.0,
                    0.0,
                    &stream,
                );
                mm3::vision_qkv_rope_3d(
                    d_qkv.ptr as *mut f32,
                    fp(lo.bqkv),
                    d_pos.ptr as *const i32,
                    d_inv.ptr as *const f32,
                    n_patches as i32,
                    d as i32,
                    heads as i32,
                    hd as i32,
                    nf as i32,
                    &stream,
                );
                mm3::vision_full_attn(
                    d_qkv.ptr as *const f32,
                    n_patches as i32,
                    d as i32,
                    heads as i32,
                    hd as i32,
                    attn_scale,
                    d_att.ptr as *mut f32,
                    &stream,
                );
                mm3::vision_bias_gelu(
                    d_att.ptr as *const f32,
                    std::ptr::null(),
                    nd,
                    d as i32,
                    false,
                    std::ptr::null_mut(),
                    d_x16.ptr as *mut u16,
                    &stream,
                );
                mm3::bf16_gemm_mma(
                    d_x16.ptr as *const u16,
                    d as i32,
                    u16p(lo.wo),
                    d as i32,
                    d_big.ptr as *mut f32,
                    d as i32,
                    n_patches as i32,
                    d as i32,
                    d as i32,
                    1.0,
                    0.0,
                    &stream,
                );
                mm3::vision_residual_bias(
                    d_h.ptr as *mut f32,
                    d_big.ptr as *const f32,
                    fp(lo.bo),
                    nd,
                    d as i32,
                    &stream,
                );
                mm3::vision_layernorm(
                    d_h.ptr as *const f32,
                    fp(lo.ln2w),
                    fp(lo.ln2b),
                    n_patches as i32,
                    d as i32,
                    self.cfg.layer_norm_eps,
                    std::ptr::null_mut(),
                    d_x16.ptr as *mut u16,
                    &stream,
                );
                mm3::bf16_gemm_mma(
                    d_x16.ptr as *const u16,
                    d as i32,
                    u16p(lo.w1),
                    d as i32,
                    d_big.ptr as *mut f32,
                    ff as i32,
                    n_patches as i32,
                    ff as i32,
                    d as i32,
                    1.0,
                    0.0,
                    &stream,
                );
                mm3::vision_bias_gelu(
                    d_big.ptr as *const f32,
                    fp(lo.b1),
                    n_patches * ff,
                    ff as i32,
                    true,
                    std::ptr::null_mut(),
                    d_x16.ptr as *mut u16,
                    &stream,
                );
                mm3::bf16_gemm_mma(
                    d_x16.ptr as *const u16,
                    ff as i32,
                    u16p(lo.w2),
                    ff as i32,
                    d_att.ptr as *mut f32,
                    d as i32,
                    n_patches as i32,
                    d as i32,
                    ff as i32,
                    1.0,
                    0.0,
                    &stream,
                );
                mm3::vision_residual_bias(
                    d_h.ptr as *mut f32,
                    d_att.ptr as *const f32,
                    fp(lo.b2),
                    nd,
                    d as i32,
                    &stream,
                );
            }
            ev_free[b].record(&stream)?;
            upload_stage(li + 2, &mut bounce, &ev_up, &ev_free)?;
        }

        // Run projector MLPs & 2x2 spatial merge MLP cleanly chunked through double-buffered slots.
        let mut run_proj_mat = |mi: usize, x_dev: *const u16, rows_in: usize, c_dev: *mut f32| -> Result<()> {
            let (s_idx, e_idx) = proj_task_ranges[mi];
            let n_tot = proj_mats[mi].1;
            let k_tot = proj_mats[mi].2;
            for tidx in s_idx..e_idx {
                let StageTask::ProjSlice(_, n0, nc, _) = stage_tasks[tidx] else {
                    unreachable!();
                };
                let b = tidx & 1;
                stream.wait_event(&ev_up[b])?;
                unsafe {
                    mm3::bf16_gemm_mma(
                        x_dev,
                        k_tot as i32,
                        d_slot[b].ptr as *const u16,
                        k_tot as i32,
                        c_dev.add(n0),
                        n_tot as i32,
                        rows_in as i32,
                        nc as i32,
                        k_tot as i32,
                        1.0,
                        0.0,
                        &stream,
                    );
                }
                ev_free[b].record(&stream)?;
                upload_stage(tidx + 2, &mut bounce, &ev_up, &ev_free)?;
            }
            Ok(())
        };

        unsafe {
            mm3::vision_bias_gelu(
                d_h.ptr as *const f32,
                std::ptr::null(),
                nd,
                d as i32,
                false,
                std::ptr::null_mut(),
                d_x16.ptr as *mut u16,
                &stream,
            );
        }
        run_proj_mat(0, d_x16.ptr as *const u16, n_patches, d_big.ptr as *mut f32)?;
        unsafe {
            mm3::vision_bias_gelu(
                d_big.ptr as *const f32,
                d_pb1.ptr as *const f32,
                n_patches * proj,
                proj as i32,
                true,
                std::ptr::null_mut(),
                d_x16.ptr as *mut u16,
                &stream,
            );
        }
        run_proj_mat(1, d_x16.ptr as *const u16, n_patches, d_big.ptr as *mut f32)?;
        unsafe {
            mm3::vision_bias_gelu(
                d_big.ptr as *const f32,
                d_pb2.ptr as *const f32,
                n_patches * out_dim,
                out_dim as i32,
                false,
                std::ptr::null_mut(),
                d_x16.ptr as *mut u16,
                &stream,
            );
        }
        // Reinterpret [n_patches, out_dim] BF16 as [nb, 4 * out_dim] BF16 for patch_merge_mlp
        run_proj_mat(2, d_x16.ptr as *const u16, nb, d_big.ptr as *mut f32)?;
        let m16 = d_qkv.ptr as *mut u16;
        unsafe {
            mm3::vision_bias_gelu(
                d_big.ptr as *const f32,
                d_pmb1.ptr as *const f32,
                nb * proj,
                proj as i32,
                true,
                std::ptr::null_mut(),
                m16,
                &stream,
            );
        }
        run_proj_mat(3, m16 as *const u16, nb, d_out.ptr as *mut f32)?;
        unsafe {
            mm3::vision_bias_gelu(
                d_out.ptr as *const f32,
                d_pmb2.ptr as *const f32,
                nb * out_dim,
                out_dim as i32,
                false,
                d_out.ptr as *mut f32,
                std::ptr::null_mut(),
                &stream,
            );
        }

        let mut rows = vec![0.0f32; nb * out_dim];
        unsafe {
            copy_d2h_raw_async(bytemuck::cast_slice_mut(&mut rows), d_out.ptr, &stream)?;
        }
        stream.sync()?;

        // Sanity verify BF16 unused helper isn't needed (explicitly silence if unused)
        let _ = std::mem::size_of::<bf16>();
        Ok(rows)
    }
}
