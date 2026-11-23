use crate::weights::DeviceTensor;
use ampllm_cuda_kernels::ffi::{
    launch_dspark_add_rms_f16, launch_dspark_attn_mma, launch_dspark_ctx_rope_kv_store,
    launch_dspark_fold_chunk, launch_dspark_fold_quant, launch_dspark_init_block,
    launch_dspark_mix_rms, launch_dspark_rms_neox_qk, launch_dspark_rnn_step_argmax,
    launch_dspark_silu_mul_f16, launch_embed_lookup_rmsnorm, launch_q8_mma_qkvrg,
    launch_q8_mma_tile,
};
use ampllm_cuda_kernels::{
    set_device, CudaGraphExec, CudaStream, DeviceBuffer, PinnedMappedBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Inkling2Config};
use anyhow::{bail, Context, Result};
use half::f16;
use std::ffi::c_void;
use std::path::Path;

pub const DSPARK_BLOCK: usize = 8;
pub const DSPARK_MAX_PROPOSALS: usize = DSPARK_BLOCK - 1;
const DSPARK_MIX_ROWS: usize = 256;
const DSPARK_HEAD_DIM: usize = 128;
const DSPARK_K_SCALE: f32 = 16.0;
const DSPARK_V_SCALE: f32 = 4.0;
const DSPARK_DRAFT_WARPS: usize = 4;
const DSPARK_MAX_SPLITS: usize = 32;

struct DMatQ8Planar {
    q: DeviceBuffer,
    qd: DeviceBuffer,
}

struct DraftLayer {
    local: bool,
    cap: usize,
    is_fp8: bool,
    in_norm: DeviceBuffer,
    post_norm: DeviceBuffer,
    dflare_norm: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    k_norm_ctx: DeviceBuffer,
    wq: DMatQ8Planar,
    wk: DMatQ8Planar,
    wv: DMatQ8Planar,
    wo: DMatQ8Planar,
    wkc: DMatQ8Planar,
    wvc: DMatQ8Planar,
    wgu: DMatQ8Planar,
    wd: DMatQ8Planar,
    kc: DeviceBuffer,
    vc: DeviceBuffer,
}

pub struct Inkling2Drafter {
    pub c: usize,
    pub h: usize,
    pub kv: usize,
    pub f: usize,
    pub r: usize,
    pub v: usize,
    pub taps: usize,
    pub ctx: usize,
    pub cap: usize,
    pub n_rows: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub theta: f32,
    pub vram_bytes: usize,

    layers: Vec<DraftLayer>,
    alpha: DeviceBuffer,
    norm: DeviceBuffer,
    mask: DeviceBuffer,
    mk_dflare: DeviceBuffer,
    w1_bf16: DeviceBuffer,
    rnn_w_bf16: DeviceBuffer,
    rnn_b: DeviceBuffer,
    fold_q: DeviceBuffer,
    fold_sc: DeviceBuffer,
    fold_b: DeviceBuffer,
    base: DeviceBuffer,

    // Cached pointers to target embedding & lm_head (target stays resident in GPU)
    tok_embd_ptr: *const c_void,
    tok_embd_type: i32,
    tok_norm_ptr: *const f32,
    head_q_ptr: *const i8,
    head_qd_ptr: *const u16,

    // Pre-allocated work buffers (zero allocations during decode/context ingest)
    anchor_h: DeviceBuffer,
    x: DeviceBuffer,
    res: DeviceBuffer,
    h_buf: DeviceBuffer,
    q_buf: DeviceBuffer,
    kk: DeviceBuffer,
    vv: DeviceBuffer,
    gu: DeviceBuffer,
    hb: DeviceBuffer,
    hb2: DeviceBuffer,
    attn_scratch: DeviceBuffer,
    f_rnn: DeviceBuffer,
    state: DeviceBuffer,
    proj: DeviceBuffer,
    fold_xs: DeviceBuffer,
    argmax_pv: DeviceBuffer,
    argmax_pi: DeviceBuffer,
    pos_buf: DeviceBuffer,
    toks: DeviceBuffer,
    par: DeviceBuffer,

    par_h: PinnedMappedBuffer<i32>,
    graphs: Vec<Option<CudaGraphExec>>,
}

unsafe impl Send for Inkling2Drafter {}
unsafe impl Sync for Inkling2Drafter {}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn read_tensor_f32(gguf: &GgufFile, name: &str, expected_len: usize) -> Result<Vec<f32>> {
    let info = gguf.tensor_info(name)?;
    let bytes = gguf.tensor_bytes(name)?;
    let out = match info.ggml_type {
        GgmlType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        GgmlType::Bf16 => bf16_bytes_to_f32(bytes),
        other => bail!("drafter tensor {name} has unsupported float type {other:?}"),
    };
    if out.len() != expected_len {
        bail!("drafter tensor {name} len {} != expected {expected_len}", out.len());
    }
    Ok(out)
}

fn upload_f32(device: i32, data: &[f32], vram: &mut usize) -> Result<DeviceBuffer> {
    let buf = DeviceBuffer::alloc(device, data.len() * 4)?;
    buf.copy_from_host(bytemuck::cast_slice(data))?;
    *vram += data.len() * 4;
    Ok(buf)
}

fn upload_bf16_raw(
    gguf: &GgufFile,
    name: &str,
    expected_elems: usize,
    device: i32,
    vram: &mut usize,
) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    if info.ggml_type != GgmlType::Bf16 {
        bail!("drafter tensor {name}: expected BF16, got {:?}", info.ggml_type);
    }
    let bytes = gguf.tensor_bytes(name)?;
    if bytes.len() != expected_elems * 2 {
        bail!(
            "drafter tensor {name} bytes {} != expected {}",
            bytes.len(),
            expected_elems * 2
        );
    }
    let buf = DeviceBuffer::alloc(device, bytes.len())?;
    buf.copy_from_host(bytes)?;
    *vram += bytes.len();
    Ok(buf)
}

fn load_q8_planar(
    gguf: &GgufFile,
    names: &[(&str, usize)],
    k: usize,
    device: i32,
    vram: &mut usize,
) -> Result<DMatQ8Planar> {
    let total_rows: usize = names.iter().map(|&(_, r)| r).sum();
    let n_elems = total_rows * k;
    let n_blocks = n_elems / 32;
    let mut hq = vec![0i8; n_elems];
    let mut hqd = vec![0u16; n_blocks];
    let mut blk_idx = 0usize;

    for &(name, rows) in names {
        let info = gguf.tensor_info(name)?;
        if info.ggml_type != GgmlType::Q8_0 {
            bail!("drafter weight {name}: expected Q8_0, got {:?}", info.ggml_type);
        }
        if info.dims.len() != 2 || info.dims[0] as usize != k || info.dims[1] as usize != rows {
            bail!(
                "drafter weight {name} shape {:?} != [{k}, {rows}]",
                info.dims
            );
        }
        let bytes = gguf.tensor_bytes(name)?;
        let expected_blocks = (rows * k) / 32;
        if bytes.len() != expected_blocks * 34 {
            bail!("drafter weight {name} byte len mismatch");
        }
        for b in 0..expected_blocks {
            let p = &bytes[b * 34..(b + 1) * 34];
            hqd[blk_idx] = u16::from_le_bytes([p[0], p[1]]);
            let q_dst = &mut hq[blk_idx * 32..(blk_idx + 1) * 32];
            for j in 0..32 {
                q_dst[j] = p[2 + j] as i8;
            }
            blk_idx += 1;
        }
    }

    let q = DeviceBuffer::alloc(device, hq.len())?;
    let qd = DeviceBuffer::alloc(device, hqd.len() * 2)?;
    q.copy_from_host(bytemuck::cast_slice(&hq))?;
    qd.copy_from_host(bytemuck::cast_slice(&hqd))?;
    *vram += hq.len() + hqd.len() * 2;
    Ok(DMatQ8Planar { q, qd })
}

impl Inkling2Drafter {
    pub fn load(
        path: &Path,
        target_cfg: &Inkling2Config,
        tok_embd: &DeviceTensor,
        tok_norm: &DeviceTensor,
        output_w: &DeviceTensor,
        max_ctx: usize,
        device: i32,
        stream: &CudaStream,
    ) -> Result<Self> {
        set_device(device)?;
        let gguf = GgufFile::open(path)
            .with_context(|| format!("opening DSpark drafter GGUF {}", path.display()))?;

        let n_layers = gguf.get_u32("dflash.block_count")? as usize;
        let c = gguf.get_u32("dflash.embedding_length")? as usize;
        let f = gguf.get_u32("dflash.feed_forward_length")? as usize;
        let h = gguf.get_u32("dflash.attention.head_count")? as usize;
        let kv = gguf.get_u32("dflash.attention.head_count_kv")? as usize;
        let key_len = gguf.get_u32("dflash.attention.key_length")? as usize;
        let r = gguf.get_u32("dflash.dspark.markov_rank")? as usize;
        let block_sz = gguf.get_u32("dflash.block_size")? as usize;
        let theta = gguf.get_f32("dflash.rope.freq_base")?;
        let eps = gguf.get_f32("dflash.attention.layer_norm_rms_epsilon")?;
        let target_layers = gguf.get_u32_array("dflash.target_layers")?;
        let swa_pattern = gguf.get_bool_array("dflash.attention.sliding_window_pattern")?;

        if c != target_cfg.n_embd || key_len != DSPARK_HEAD_DIM || block_sz != DSPARK_BLOCK {
            bail!(
                "DSpark GGUF mismatch: c={c} (target {}), head_dim={key_len}, block={block_sz}",
                target_cfg.n_embd
            );
        }
        let taps = target_layers.len();
        let expected_taps: Vec<u32> = if target_cfg.is_looped {
            vec![5, 23, 41, 48]
        } else {
            vec![5, 11, 17, 23, 29, 35, 41, 48]
        };
        if target_layers != expected_taps {
            bail!("DSpark target_layers {:?} != expected {:?}", target_layers, expected_taps);
        }
        if swa_pattern.len() != n_layers {
            bail!("DSpark sliding_window_pattern len mismatch");
        }

        let w1_info = gguf.tensor_info("markov_w1.weight")?;
        let v = w1_info.dims[1] as usize;
        let mut vram_bytes = 0usize;

        // Softmax over each row of dspark_fuse.weight [n_layers + 1, taps]
        let fw = read_tensor_f32(&gguf, "dspark_fuse.weight", (n_layers + 1) * taps)?;
        let mut alpha_host = vec![0.0f32; fw.len()];
        for row in 0..=n_layers {
            let src = &fw[row * taps..(row + 1) * taps];
            let mx = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..taps {
                let e = (src[t] - mx).exp();
                alpha_host[row * taps + t] = e;
                sum += e;
            }
            for t in 0..taps {
                alpha_host[row * taps + t] /= sum;
            }
        }
        let alpha = upload_f32(device, &alpha_host, &mut vram_bytes)?;

        let ctx = max_ctx;
        let cap = ctx.max(2048);
        let kd = kv * DSPARK_HEAD_DIM;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let local = swa_pattern[i];
            let p = format!("blk.{i}");
            let in_norm = upload_f32(device, &read_tensor_f32(&gguf, &format!("{p}.attn_norm.weight"), c)?, &mut vram_bytes)?;
            let post_norm = upload_f32(device, &read_tensor_f32(&gguf, &format!("{p}.ffn_norm.weight"), c)?, &mut vram_bytes)?;
            let dflare_norm = upload_f32(device, &read_tensor_f32(&gguf, &format!("{p}.dflare_norm.weight"), c)?, &mut vram_bytes)?;
            let q_norm = upload_f32(device, &read_tensor_f32(&gguf, &format!("{p}.attn_q_norm.weight"), DSPARK_HEAD_DIM)?, &mut vram_bytes)?;
            let k_norm = upload_f32(device, &read_tensor_f32(&gguf, &format!("{p}.attn_k_norm.weight"), DSPARK_HEAD_DIM)?, &mut vram_bytes)?;
            let k_norm_ctx = upload_f32(device, &read_tensor_f32(&gguf, &format!("{p}.attn_k_ctx_norm.weight"), DSPARK_HEAD_DIM)?, &mut vram_bytes)?;

            let wq = load_q8_planar(&gguf, &[(&format!("{p}.attn_q.weight"), h * DSPARK_HEAD_DIM)], c, device, &mut vram_bytes)?;
            let wk = load_q8_planar(&gguf, &[(&format!("{p}.attn_k.weight"), kd)], c, device, &mut vram_bytes)?;
            let wv = load_q8_planar(&gguf, &[(&format!("{p}.attn_v.weight"), kd)], c, device, &mut vram_bytes)?;
            let wo = load_q8_planar(&gguf, &[(&format!("{p}.attn_output.weight"), c)], h * DSPARK_HEAD_DIM, device, &mut vram_bytes)?;
            let wkc = load_q8_planar(&gguf, &[(&format!("{p}.attn_k_ctx.weight"), kd)], c, device, &mut vram_bytes)?;
            let wvc = load_q8_planar(&gguf, &[(&format!("{p}.attn_v_ctx.weight"), kd)], c, device, &mut vram_bytes)?;
            let gate_name = format!("{p}.ffn_gate.weight");
            let up_name = format!("{p}.ffn_up.weight");
            let wgu = load_q8_planar(&gguf, &[(&gate_name, f), (&up_name, f)], c, device, &mut vram_bytes)?;
            let wd = load_q8_planar(&gguf, &[(&format!("{p}.ffn_down.weight"), c)], f, device, &mut vram_bytes)?;

            let lcap = if local { ctx.min(2048) } else { cap };
            let is_fp8 = !local;
            let elem_bytes = if is_fp8 { 1usize } else { 2usize };
            let kv_bytes = lcap * kd * elem_bytes;
            let kc = DeviceBuffer::alloc(device, kv_bytes)?;
            let vc = DeviceBuffer::alloc(device, kv_bytes)?;
            vram_bytes += 2 * kv_bytes;

            layers.push(DraftLayer {
                local,
                cap: lcap,
                is_fp8,
                in_norm,
                post_norm,
                dflare_norm,
                q_norm,
                k_norm,
                k_norm_ctx,
                wq,
                wk,
                wv,
                wo,
                wkc,
                wvc,
                wgu,
                wd,
                kc,
                vc,
            });
        }

        let norm = upload_f32(device, &read_tensor_f32(&gguf, "output_norm.weight", c)?, &mut vram_bytes)?;
        let mask = upload_f32(device, &read_tensor_f32(&gguf, "mask_embd.weight", c)?, &mut vram_bytes)?;
        let mk_dflare = upload_f32(device, &read_tensor_f32(&gguf, "dspark_norm.weight", c)?, &mut vram_bytes)?;
        let w1_bf16 = upload_bf16_raw(&gguf, "markov_w1.weight", v * r, device, &mut vram_bytes)?;
        let rnn_w_bf16 = upload_bf16_raw(&gguf, "joint_proj.weight", 3 * r * (2 * r + 2 * c), device, &mut vram_bytes)?;
        let rnn_b = upload_f32(device, &read_tensor_f32(&gguf, "joint_proj.bias", 3 * r)?, &mut vram_bytes)?;

        // Offline GPU Markov fold: fold_q[R, n_rows] i8, fold_sc[R] f32, fold_b[n_rows] f16
        let n_rows = output_w.dims.get(1).copied().unwrap_or(target_cfg.n_vocab_padded as u64) as usize;
        let n_vocab = target_cfg.n_vocab_unpadded.min(n_rows);
        let (head_q_ptr, head_qd_ptr) = match (&output_w.q8_planar_q, &output_w.q8_planar_d) {
            (Some(q), Some(d)) => (q.ptr as *const i8, d.ptr as *const u16),
            _ => bail!("DSpark drafter requires target output.weight to be planar Q8_0"),
        };

        let hout_w_f32 = read_tensor_f32(&gguf, "dspark_hidden_out.weight", c * r)?;
        let hout_b_f32 = read_tensor_f32(&gguf, "dspark_hidden_out.bias", c)?;
        let mut ht_host = vec![0.0f32; r * c];
        for col in 0..c {
            for row in 0..r {
                ht_host[row * c + col] = hout_w_f32[col * r + row];
            }
        }

        let fold_q = DeviceBuffer::alloc(device, r * n_rows)?;
        let fold_sc = DeviceBuffer::alloc(device, r * 4)?;
        let fold_b = DeviceBuffer::alloc(device, n_rows * 2)?;
        let base = DeviceBuffer::alloc(device, DSPARK_MAX_PROPOSALS * n_rows * 4)?;
        vram_bytes += r * n_rows + r * 4 + n_rows * 2 + DSPARK_MAX_PROPOSALS * n_rows * 4;

        {
            let chunk = 4096usize.min(n_rows.max(1));
            let dht = DeviceBuffer::alloc(device, r * c * 4)?;
            let dhb = DeviceBuffer::alloc(device, c * 4)?;
            let dlm_tmp = DeviceBuffer::alloc(device, chunk * c * 4)?;
            let fold_w_f32 = DeviceBuffer::alloc(device, r * n_rows * 4)?;
            dht.copy_from_host(bytemuck::cast_slice(&ht_host))?;
            dhb.copy_from_host(bytemuck::cast_slice(&hout_b_f32))?;

            let mut r0 = 0usize;
            while r0 < n_rows {
                let m = (n_rows - r0).min(chunk);
                unsafe {
                    launch_dspark_fold_chunk(
                        head_q_ptr,
                        head_qd_ptr,
                        std::ptr::null(),
                        dht.ptr as *const f32,
                        dhb.ptr as *const f32,
                        dlm_tmp.ptr as *mut f32,
                        r0 as i32,
                        m as i32,
                        n_rows as i32,
                        r as i32,
                        c as i32,
                        fold_w_f32.ptr as *mut f32,
                        fold_b.ptr as *mut u16,
                        stream.raw,
                    );
                }
                r0 += m;
            }
            unsafe {
                launch_dspark_fold_quant(
                    fold_w_f32.ptr as *const f32,
                    r as i32,
                    n_rows as i32,
                    fold_q.ptr as *mut i8,
                    fold_sc.ptr as *mut f32,
                    stream.raw,
                );
            }
            stream.sync()?;
        }

        // Work buffers
        let t = DSPARK_BLOCK;
        let hd = h * DSPARK_HEAD_DIM;
        let hb_elems = (t * c.max(hd)).max(DSPARK_MIX_ROWS * c);
        let kk_elems = t.max(DSPARK_MIX_ROWS) * kd;
        let attn_scratch_floats = DSPARK_MAX_SPLITS * DSPARK_DRAFT_WARPS * t * h * (DSPARK_HEAD_DIM + 2);

        let anchor_h = DeviceBuffer::alloc(device, c * 4)?;
        let x = DeviceBuffer::alloc(device, t * c * 4)?;
        let res = DeviceBuffer::alloc(device, t * c * 4)?;
        let h_buf = DeviceBuffer::alloc(device, t * c * 4)?;
        let q_buf = DeviceBuffer::alloc(device, t * hd * 4)?;
        let kk = DeviceBuffer::alloc(device, kk_elems * 4)?;
        let vv = DeviceBuffer::alloc(device, kk_elems * 4)?;
        let gu = DeviceBuffer::alloc(device, t * 2 * f * 4)?;
        let hb = DeviceBuffer::alloc(device, hb_elems * 2)?;
        let hb2 = DeviceBuffer::alloc(device, t * f * 2)?;
        let attn_scratch = DeviceBuffer::alloc(device, attn_scratch_floats * 4)?;
        let f_rnn = DeviceBuffer::alloc(device, c * 4)?;
        let state = DeviceBuffer::alloc(device, r * 4)?;
        let proj = DeviceBuffer::alloc(device, 3 * r * 4)?;
        let fold_xs = DeviceBuffer::alloc(device, r * 4)?;
        let argmax_pv = DeviceBuffer::alloc(device, 512 * 4)?;
        let argmax_pi = DeviceBuffer::alloc(device, 512 * 4)?;
        let pos_buf = DeviceBuffer::alloc(device, t * 4)?;
        let toks = DeviceBuffer::alloc(device, t * 4)?;
        let par = DeviceBuffer::alloc(device, 8 * 4)?;

        vram_bytes += anchor_h.bytes
            + x.bytes
            + res.bytes
            + h_buf.bytes
            + q_buf.bytes
            + kk.bytes
            + vv.bytes
            + gu.bytes
            + hb.bytes
            + hb2.bytes
            + attn_scratch.bytes
            + f_rnn.bytes
            + state.bytes
            + proj.bytes
            + fold_xs.bytes
            + argmax_pv.bytes
            + argmax_pi.bytes
            + pos_buf.bytes
            + toks.bytes
            + par.bytes;

        let mut par_h = PinnedMappedBuffer::alloc(8)?;
        par_h.as_mut_slice().fill(0);

        let mut graphs = Vec::with_capacity(DSPARK_BLOCK);
        for _ in 0..DSPARK_BLOCK {
            graphs.push(None);
        }

        Ok(Self {
            c,
            h,
            kv,
            f,
            r,
            v,
            taps,
            ctx,
            cap,
            n_rows,
            n_vocab,
            eps,
            theta,
            vram_bytes,
            layers,
            alpha,
            norm,
            mask,
            mk_dflare,
            w1_bf16,
            rnn_w_bf16,
            rnn_b,
            fold_q,
            fold_sc,
            fold_b,
            base,
            tok_embd_ptr: tok_embd.buf.ptr,
            tok_embd_type: tok_embd.ggml_type as i32,
            tok_norm_ptr: tok_norm.as_f32(),
            head_q_ptr,
            head_qd_ptr,
            anchor_h,
            x,
            res,
            h_buf,
            q_buf,
            kk,
            vv,
            gu,
            hb,
            hb2,
            attn_scratch,
            f_rnn,
            state,
            proj,
            fold_xs,
            argmax_pv,
            argmax_pi,
            pos_buf,
            toks,
            par,
            par_h,
            graphs,
        })
    }

    pub fn write_context(
        &self,
        taps_ptr: *const f32,
        tap_stride_rows: usize,
        r0: usize,
        m: usize,
        pos0: usize,
        stream: &CudaStream,
    ) {
        if m == 0 {
            return;
        }
        let kd = self.kv * DSPARK_HEAD_DIM;
        let tap_stride_elems = tap_stride_rows * self.c;
        for (li, l) in self.layers.iter().enumerate() {
            let alpha_ptr = unsafe { (self.alpha.ptr as *const f32).add(li * self.taps) };
            let mut a = 0usize;
            while a < m {
                let at = (pos0 + a) % l.cap;
                let n = DSPARK_MIX_ROWS.min(m - a).min(l.cap - at);
                unsafe {
                    launch_dspark_mix_rms(
                        taps_ptr,
                        self.taps as i32,
                        tap_stride_elems,
                        (r0 + a) as i32,
                        n as i32,
                        self.c as i32,
                        alpha_ptr,
                        l.dflare_norm.ptr as *const f32,
                        self.eps,
                        std::ptr::null_mut(),
                        self.hb.ptr as *mut u16,
                        stream.raw,
                    );
                    launch_q8_mma_qkvrg(
                        l.wkc.q.ptr as *const i8,
                        l.wkc.qd.ptr as *const u16,
                        kd as i32,
                        self.hb.ptr as *const u16,
                        self.kk.ptr as *mut f32,
                        l.wvc.q.ptr as *const i8,
                        l.wvc.qd.ptr as *const u16,
                        kd as i32,
                        self.hb.ptr as *const u16,
                        self.vv.ptr as *mut f32,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        self.c as i32,
                        n as i32,
                        stream.raw,
                    );
                    launch_dspark_ctx_rope_kv_store(
                        self.kk.ptr as *const f32,
                        self.vv.ptr as *const f32,
                        l.k_norm_ctx.ptr as *const f32,
                        n as i32,
                        self.kv as i32,
                        (pos0 + a) as i32,
                        l.cap as i32,
                        self.theta,
                        self.eps,
                        if l.is_fp8 { 1 } else { 0 },
                        DSPARK_K_SCALE,
                        DSPARK_V_SCALE,
                        l.kc.ptr,
                        l.vc.ptr,
                        stream.raw,
                    );
                }
                a += n;
            }
        }
    }

    pub fn set_f_rnn(
        &self,
        taps_ptr: *const f32,
        tap_stride_rows: usize,
        row: usize,
        stream: &CudaStream,
    ) {
        let tap_stride_elems = tap_stride_rows * self.c;
        let alpha_ptr = unsafe { (self.alpha.ptr as *const f32).add(self.layers.len() * self.taps) };
        unsafe {
            launch_dspark_mix_rms(
                taps_ptr,
                self.taps as i32,
                tap_stride_elems,
                row as i32,
                1,
                self.c as i32,
                alpha_ptr,
                self.mk_dflare.ptr as *const f32,
                self.eps,
                self.f_rnn.ptr as *mut f32,
                std::ptr::null_mut(),
                stream.raw,
            );
        }
    }

    fn enqueue_propose_body(&self, proposals: usize, stream: &CudaStream) -> Result<()> {
        let t = DSPARK_BLOCK as i32;
        let c = self.c as i32;
        let h = self.h as i32;
        let kv = self.kv as i32;
        let f = self.f as i32;
        let r = self.r as i32;
        let hd = (self.h * DSPARK_HEAD_DIM) as i32;
        let kd = (self.kv * DSPARK_HEAD_DIM) as i32;

        unsafe {
            // Embed + RMSNorm the anchor token from par[1] directly inside the graph!
            launch_embed_lookup_rmsnorm(
                self.tok_embd_ptr,
                self.tok_embd_type,
                (self.par.ptr as *const i32).add(1),
                self.tok_norm_ptr,
                self.anchor_h.ptr as *mut f32,
                c,
                1,
                self.eps,
                stream.raw,
            );
            launch_dspark_init_block(
                self.anchor_h.ptr as *const f32,
                self.mask.ptr as *const f32,
                self.res.ptr as *mut f32,
                self.state.ptr as *mut f32,
                self.pos_buf.ptr as *mut i32,
                self.toks.ptr as *mut i32,
                self.par.ptr as *const i32,
                0,
                0,
                t,
                c,
                r,
                stream.raw,
            );

            for (li, l) in self.layers.iter().enumerate() {
                let src_x = if li == 0 {
                    std::ptr::null()
                } else {
                    self.x.ptr as *const f32
                };
                launch_dspark_add_rms_f16(
                    src_x,
                    self.res.ptr as *mut f32,
                    l.in_norm.ptr as *const f32,
                    self.eps,
                    t,
                    c,
                    self.h_buf.ptr as *mut f32,
                    self.hb.ptr as *mut u16,
                    stream.raw,
                );
                launch_q8_mma_qkvrg(
                    l.wq.q.ptr as *const i8,
                    l.wq.qd.ptr as *const u16,
                    hd,
                    self.hb.ptr as *const u16,
                    self.q_buf.ptr as *mut f32,
                    l.wk.q.ptr as *const i8,
                    l.wk.qd.ptr as *const u16,
                    kd,
                    self.hb.ptr as *const u16,
                    self.kk.ptr as *mut f32,
                    l.wv.q.ptr as *const i8,
                    l.wv.qd.ptr as *const u16,
                    kd,
                    self.hb.ptr as *const u16,
                    self.vv.ptr as *mut f32,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    c,
                    t,
                    stream.raw,
                );
                launch_dspark_rms_neox_qk(
                    self.q_buf.ptr as *mut f32,
                    self.kk.ptr as *mut f32,
                    l.q_norm.ptr as *const f32,
                    l.k_norm.ptr as *const f32,
                    self.pos_buf.ptr as *const i32,
                    t,
                    h,
                    kv,
                    self.theta,
                    self.eps,
                    stream.raw,
                );
                let span = if l.local { 1023 } else { l.cap as i32 };
                launch_dspark_attn_mma(
                    self.q_buf.ptr as *const f32,
                    l.kc.ptr,
                    l.vc.ptr,
                    self.kk.ptr as *const f32,
                    self.vv.ptr as *const f32,
                    0,
                    0,
                    self.par.ptr as *const i32,
                    span,
                    l.cap as i32,
                    if l.is_fp8 { 1 } else { 0 },
                    if l.is_fp8 { 1.0 / DSPARK_K_SCALE } else { 1.0 },
                    if l.is_fp8 { 1.0 / DSPARK_V_SCALE } else { 1.0 },
                    t,
                    h,
                    kv,
                    self.attn_scratch.ptr as *mut f32,
                    self.hb.ptr as *mut u16,
                    stream.raw,
                );
                launch_q8_mma_tile(
                    l.wo.q.ptr as *const i8,
                    l.wo.qd.ptr as *const u16,
                    c,
                    hd,
                    self.hb.ptr as *const u16,
                    hd,
                    t,
                    self.x.ptr as *mut f32,
                    c,
                    stream.raw,
                );
                launch_dspark_add_rms_f16(
                    self.x.ptr as *const f32,
                    self.res.ptr as *mut f32,
                    l.post_norm.ptr as *const f32,
                    self.eps,
                    t,
                    c,
                    self.h_buf.ptr as *mut f32,
                    self.hb.ptr as *mut u16,
                    stream.raw,
                );
                launch_q8_mma_tile(
                    l.wgu.q.ptr as *const i8,
                    l.wgu.qd.ptr as *const u16,
                    2 * f,
                    c,
                    self.hb.ptr as *const u16,
                    c,
                    t,
                    self.gu.ptr as *mut f32,
                    2 * f,
                    stream.raw,
                );
                launch_dspark_silu_mul_f16(
                    self.gu.ptr as *const f32,
                    t,
                    f,
                    self.hb2.ptr as *mut u16,
                    stream.raw,
                );
                launch_q8_mma_tile(
                    l.wd.q.ptr as *const i8,
                    l.wd.qd.ptr as *const u16,
                    c,
                    f,
                    self.hb2.ptr as *const u16,
                    f,
                    t,
                    self.x.ptr as *mut f32,
                    c,
                    stream.raw,
                );
            }

            launch_dspark_add_rms_f16(
                self.x.ptr as *const f32,
                self.res.ptr as *mut f32,
                self.norm.ptr as *const f32,
                self.eps,
                t,
                c,
                self.h_buf.ptr as *mut f32,
                self.hb.ptr as *mut u16,
                stream.raw,
            );

            // Base logits for rows 1..=proposals through the target head
            launch_q8_mma_tile(
                self.head_q_ptr,
                self.head_qd_ptr,
                self.n_rows as i32,
                c,
                (self.hb.ptr as *const u16).add(self.c),
                c,
                proposals as i32,
                self.base.ptr as *mut f32,
                self.n_rows as i32,
                stream.raw,
            );

            for step in 0..proposals {
                launch_dspark_rnn_step_argmax(
                    self.rnn_w_bf16.ptr as *const u16,
                    self.rnn_b.ptr as *const f32,
                    self.w1_bf16.ptr as *const u16,
                    self.fold_q.ptr as *const i8,
                    self.fold_sc.ptr as *const f32,
                    self.fold_b.ptr as *const u16,
                    (self.base.ptr as *const f32).add(step * self.n_rows),
                    (self.h_buf.ptr as *const f32).add((step + 1) * self.c),
                    self.f_rnn.ptr as *const f32,
                    r,
                    c,
                    self.n_rows as i32,
                    self.n_vocab as i32,
                    self.state.ptr as *mut f32,
                    self.proj.ptr as *mut f32,
                    self.fold_xs.ptr as *mut f32,
                    self.argmax_pv.ptr as *mut f32,
                    self.argmax_pi.ptr as *mut i32,
                    (self.toks.ptr as *const i32).add(step),
                    (self.toks.ptr as *mut i32).add(step + 1),
                    stream.raw,
                );
            }
        }
        Ok(())
    }

    pub fn warmup_graphs(&mut self, stream: &CudaStream) -> Result<()> {
        self.par_h.as_mut_slice()[0] = 1;
        self.par_h.as_mut_slice()[1] = 0;
        self.par
            .copy_from_host_async(bytemuck::cast_slice(&self.par_h.as_slice()[..2]), stream)?;
        for p in 1..=DSPARK_MAX_PROPOSALS {
            if self.graphs[p].is_none() {
                let g = CudaGraphExec::capture(stream, || self.enqueue_propose_body(p, stream))?;
                self.graphs[p] = Some(g);
            }
            self.graphs[p].as_ref().unwrap().launch(stream)?;
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        anchor: u32,
        pos: usize,
        proposals: usize,
        stream: &CudaStream,
    ) -> Result<[u32; DSPARK_MAX_PROPOSALS]> {
        if proposals == 0 || proposals > DSPARK_MAX_PROPOSALS {
            bail!("invalid proposal count {proposals}");
        }
        self.par_h.as_mut_slice()[0] = pos as i32;
        self.par_h.as_mut_slice()[1] = anchor as i32;
        self.par
            .copy_from_host_async(bytemuck::cast_slice(&self.par_h.as_slice()[..2]), stream)?;

        if self.graphs[proposals].is_none() {
            let g =
                CudaGraphExec::capture(stream, || self.enqueue_propose_body(proposals, stream))?;
            self.graphs[proposals] = Some(g);
        }
        self.graphs[proposals].as_ref().unwrap().launch(stream)?;

        let mut out_raw = [0i32; DSPARK_MAX_PROPOSALS];
        unsafe {
            ampllm_cuda_kernels::cuda_rt::copy_d2h_raw_async(
                bytemuck::cast_slice_mut(&mut out_raw[..proposals]),
                (self.toks.ptr as *const u8).add(4) as *const c_void,
                stream,
            )?;
        }
        stream.sync()?;

        let mut res = [0u32; DSPARK_MAX_PROPOSALS];
        for i in 0..proposals {
            res[i] = out_raw[i] as u32;
        }
        Ok(res)
    }
}

/// Exact zero-cost CPU history suffix matcher (looks back up to 8192 tokens for length >= 3 suffix).
pub fn suffix_propose(history: &[u32], max_proposals: usize) -> Option<Vec<u32>> {
    let n = history.len();
    if n < 4 || max_proposals == 0 {
        return None;
    }
    let min_match = 3usize;
    let max_match = 16usize.min(n - 1);
    let search_start = n.saturating_sub(8192);
    let mut best_len = 0usize;
    let mut best_pos = 0usize;

    // Scan backwards comparing suffixes of increasing length up to max_match (preferring most recent on ties).
    let anchor = history[n - 1];
    let mut idx = n.saturating_sub(2);
    while idx >= search_start + min_match - 1 {
        if history[idx] == anchor {
            let mut m = 1usize;
            while m < max_match && idx >= m && history[idx - m] == history[n - 1 - m] {
                m += 1;
            }
            if m >= min_match && m > best_len && idx + 1 < n {
                best_len = m;
                best_pos = idx + 1;
                if best_len >= 8 {
                    break;
                }
            }
        }
        if idx == 0 {
            break;
        }
        idx -= 1;
    }

    if best_len >= min_match {
        let avail = (n - best_pos).min(max_proposals);
        if avail > 0 {
            return Some(history[best_pos..best_pos + avail].to_vec());
        }
    }
    None
}

/// Adaptive width (W = 1..=8) controller maximizing expected accepted tokens per millisecond.
pub struct SpecController {
    pn: [f64; DSPARK_MAX_PROPOSALS],
    pa: [f64; DSPARK_MAX_PROPOSALS],
    c: [f64; DSPARK_BLOCK + 1],
    seen: [usize; DSPARK_BLOCK + 1],
    prior: [f64; DSPARK_BLOCK + 1],
    c_plain: f64,
    plain_run: usize,
    pos_decay: f64,
    sl: f64,
    fl: f64,
    lookup_decay: f64,
}

impl Default for SpecController {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecController {
    pub fn new() -> Self {
        let prior = [0.0, 9.3, 13.5, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0];
        Self {
            pn: [3.0; DSPARK_MAX_PROPOSALS],
            pa: [2.1, 2.25, 2.4, 2.4, 2.4, 2.55, 2.55],
            c: prior,
            seen: [0; DSPARK_BLOCK + 1],
            prior,
            c_plain: 9.3,
            plain_run: 0,
            pos_decay: 0.90,
            sl: 2.0,
            fl: 1.0,
            lookup_decay: 0.85,
        }
    }

    #[inline]
    fn expect(&self, w: usize) -> f64 {
        let mut e = 1.0f64;
        let mut reach = 1.0f64;
        for j in 0..(w - 1) {
            reach *= (self.pa[j] / self.pn[j]).min(0.98);
            e += reach;
        }
        e
    }

    /// Pick `(W in 1..=max_w, use_lookup)`. Competes prompt lookup vs neural drafter vs plain decoding.
    pub fn pick(&mut self, max_w: usize, lookup_len: usize, has_drafter: bool) -> (usize, bool) {
        if max_w < 2 {
            return (1, false);
        }
        let bmax = max_w.min(DSPARK_BLOCK);
        let mut best_w = 1usize;
        let mut best_use_lookup = false;
        let mut best_rate = 1.0 / self.c_plain.max(0.1);

        if lookup_len > 0 {
            let a = ((self.sl + 0.5) / (self.sl + self.fl + 1.0)).min(0.98);
            let lmax = bmax.min(lookup_len + 1);
            for w in 2..=lmax {
                let exp = (1.0 - a.powi(w as i32)) / (1.0 - a);
                let cost = (self.c[w] - 1.5).max(self.c_plain).max(0.1);
                let rate = exp / cost;
                if rate > best_rate {
                    best_rate = rate;
                    best_w = w;
                    best_use_lookup = true;
                }
            }
        }
        if has_drafter {
            for w in 2..=bmax {
                let rate = self.expect(w) / self.c[w].max(0.1);
                if rate > best_rate {
                    best_rate = rate;
                    best_w = w;
                    best_use_lookup = false;
                }
            }
        }
        if best_w == 1 && self.plain_run >= 4 && has_drafter {
            return (4usize.min(bmax), false);
        }
        (best_w, best_use_lookup)
    }

    pub fn record_plain(&mut self, elapsed_ms: f64) {
        let t = elapsed_ms.max(0.1);
        self.c_plain = 0.9 * self.c_plain + 0.1 * t;
        self.c[1] = self.c_plain;
        self.plain_run += 1;
    }

    pub fn record_window(&mut self, w: usize, accepted_drafts: usize, elapsed_ms: f64, is_suffix: bool) {
        self.plain_run = 0;
        if (2..=DSPARK_BLOCK).contains(&w) {
            let proposals = w - 1;
            if is_suffix {
                self.sl = self.lookup_decay * self.sl + accepted_drafts as f64;
                self.fl = self.lookup_decay * self.fl
                    + if accepted_drafts < proposals { 1.0 } else { 0.0 };
                return;
            }
            let t = elapsed_ms.max(0.1);
            let prev = self.seen[w];
            self.seen[w] = prev + 1;
            if prev >= 1 {
                self.c[w] = if prev == 1 { t } else { 0.8 * self.c[w] + 0.2 * t };
                let mut ratio_sum = 0.0;
                let mut cnt = 0usize;
                for b in 2..=DSPARK_BLOCK {
                    if self.seen[b] >= 2 {
                        ratio_sum += self.c[b] / self.prior[b];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    let avg_r = ratio_sum / (cnt as f64);
                    for b in 2..=DSPARK_BLOCK {
                        if self.seen[b] < 2 {
                            self.c[b] = self.prior[b] * avg_r;
                        }
                    }
                }
            }
            for j in 0..proposals {
                self.pn[j] = self.pos_decay * self.pn[j] + 1.0;
                if j < accepted_drafts {
                    self.pa[j] = self.pos_decay * self.pa[j] + 1.0;
                } else {
                    self.pa[j] *= self.pos_decay;
                    break;
                }
            }
            if accepted_drafts == proposals && proposals < DSPARK_MAX_PROPOSALS {
                let tail_rate = (self.pa[proposals - 1] / self.pn[proposals - 1]).min(0.98);
                for j in proposals..DSPARK_MAX_PROPOSALS {
                    self.pn[j] = 0.9 * self.pn[j] + 0.5;
                    self.pa[j] = 0.9 * self.pa[j] + 0.5 * tail_rate;
                }
            }
        }
    }
}
