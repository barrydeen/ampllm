use ampllm_cpu_kernels::mm3_cpu::q6_k_row_to_f32_cpu;
use ampllm_cuda_kernels::{
    copy_d2h_raw_async, mm3, set_device, CudaStream, DeviceBuffer, PinnedHostBuffer,
    PinnedMappedBuffer,
};
use ampllm_gguf::{Mm3DsparkConfig, SafeTensorsFile};
use anyhow::{bail, Result};
use half::bf16;
use std::path::Path;

const DSPARK_PREFIX: &str = "language_model.model.dspark.";
const DSPARK_MAX_CTX_CHUNK: usize = 256;
const DSPARK_ATTN_SPLITS: i32 = 16;

struct MxMatrix {
    w: DeviceBuffer,
    s: DeviceBuffer,
    n: usize,
    k: usize,
}

struct DsparkLayerWeights {
    in_norm: DeviceBuffer,
    post_norm: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    q: MxMatrix,
    k: MxMatrix,
    v: MxMatrix,
    o: MxMatrix,
    gate: MxMatrix,
    up: MxMatrix,
    down: MxMatrix,
    kc_bf16: DeviceBuffer,
    vc_bf16: DeviceBuffer,
}

pub struct Mm3Dspark {
    pub cfg: Mm3DsparkConfig,
    pub device: i32,
    pub ring: usize,
    pub stream: CudaStream,

    fc_bf16: DeviceBuffer,
    hidden_norm: DeviceBuffer,
    final_norm: DeviceBuffer,
    markov_w1_bf16: DeviceBuffer,
    markov_w2_bf16: DeviceBuffer,
    conf_w: DeviceBuffer,
    conf_b: DeviceBuffer,
    inv_freq: DeviceBuffer,
    mask_emb_f32: DeviceBuffer,

    layers: Vec<DsparkLayerWeights>,

    // Device scratch
    ctx: DeviceBuffer,
    xn: DeviceBuffer,
    h: DeviceBuffer,
    q: DeviceBuffer,
    kk: DeviceBuffer,
    vv: DeviceBuffer,
    att: DeviceBuffer,
    gt: DeviceBuffer,
    up: DeviceBuffer,
    hid: DeviceBuffer,
    hid_q8_1: DeviceBuffer,
    logits: DeviceBuffer,
    x16: DeviceBuffer,
    kv_w_bf16_scratch: DeviceBuffer,
    mx_part: DeviceBuffer,
    attn_part: DeviceBuffer,
    mk_part: DeviceBuffer,
    pos_dev: DeviceBuffer,
    lo_dev: DeviceBuffer,
    prev_dev: DeviceBuffer,
    out_dev: DeviceBuffer,
    conf_dev: DeviceBuffer,

    // Pinned host staging for zero-overhead H2D/D2H transfers
    host_anc_mapped: PinnedMappedBuffer<f32>,
    host_out_pin: PinnedHostBuffer,
    host_conf_pin: PinnedHostBuffer,
}

fn upload_bf16_norm(st: &SafeTensorsFile, dev: i32, name: &str, add_one: f32) -> Result<DeviceBuffer> {
    let raw_u16 = st.tensor_bf16_u16(name)?;
    let mut vals = Vec::with_capacity(raw_u16.len());
    for b in raw_u16 {
        vals.push(bf16::from_bits(b).to_f32() + add_one);
    }
    let buf = DeviceBuffer::alloc(dev, vals.len() * 4)?;
    buf.copy_from_host(bytemuck::cast_slice(&vals))?;
    Ok(buf)
}

fn upload_bf16_raw(st: &SafeTensorsFile, dev: i32, name: &str, expected_elems: usize) -> Result<DeviceBuffer> {
    let bytes = st.tensor_bytes(name)?;
    if bytes.len() != expected_elems * 2 {
        bail!(
            "dspark {name}: expected {} bytes ({} bf16), got {}",
            expected_elems * 2,
            expected_elems,
            bytes.len()
        );
    }
    let buf = DeviceBuffer::alloc(dev, bytes.len())?;
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn upload_mxfp8(
    st: &SafeTensorsFile,
    dev: i32,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<MxMatrix> {
    let w_name = format!("{prefix}.weight");
    let s_name = format!("{prefix}.weight_scale_inv");
    let w_bytes = st.tensor_bytes(&w_name)?;
    let s_bytes = st.tensor_bytes(&s_name)?;
    if w_bytes.len() != n * k || s_bytes.len() != n * (k / 32) {
        bail!(
            "dspark {prefix}: shape mismatch for [{n}x{k}] (w={}B, s={}B)",
            w_bytes.len(),
            s_bytes.len()
        );
    }
    let w = DeviceBuffer::alloc(dev, w_bytes.len())?;
    w.copy_from_host(w_bytes)?;
    let s = DeviceBuffer::alloc(dev, s_bytes.len())?;
    s.copy_from_host(s_bytes)?;
    Ok(MxMatrix { w, s, n, k })
}

impl Mm3Dspark {
    pub fn load(
        safetensors_index_or_file: &Path,
        cfg: Mm3DsparkConfig,
        token_embd_q6k: &[u8],
        token_embd_row_bytes: usize,
        device: i32,
    ) -> Result<Self> {
        let st = SafeTensorsFile::open_from_path(safetensors_index_or_file)?;
        if cfg.block_size == 0 || cfg.block_size > 16 || cfg.head_dim != 128 || cfg.markov_rank != 256 {
            bail!(
                "unsupported DSpark geometry: block={} hd={} rank={}",
                cfg.block_size,
                cfg.head_dim,
                cfg.markov_rank
            );
        }
        set_device(device)?;
        let stream = CudaStream::new(device)?;

        let d = cfg.hidden_size;
        let ff = cfg.ffn_hidden_size;
        let hq = cfg.num_heads;
        let hkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let q_dim = hq * hd;
        let kv_dim = hkv * hd;
        let vocab = cfg.vocab_size;
        let rank = cfg.markov_rank;
        let n_taps = cfg.target_layer_ids.len();
        let block = cfg.block_size;
        let ring = cfg.sliding_window + 64;

        let dp = |s: &str| format!("{DSPARK_PREFIX}{s}");
        let fc_bf16 = upload_bf16_raw(&st, device, &dp("fc.weight"), d * n_taps * d)?;
        let hidden_norm = upload_bf16_norm(&st, device, &dp("hidden_norm.weight"), 1.0)?;
        let final_norm = upload_bf16_norm(&st, device, &dp("final_norm.weight"), 1.0)?;
        let markov_w1_bf16 = upload_bf16_raw(&st, device, &dp("markov_head.markov_w1.weight"), vocab * rank)?;
        let markov_w2_bf16 = upload_bf16_raw(&st, device, &dp("markov_head.markov_w2.weight"), vocab * rank)?;
        let conf_w = upload_bf16_norm(&st, device, &dp("confidence_head.proj.weight"), 0.0)?;
        let conf_b = upload_bf16_norm(&st, device, &dp("confidence_head.proj.bias"), 0.0)?;

        let inv_h: Vec<f32> = (0..cfg.rot_dim / 2)
            .map(|i| 1.0f32 / cfg.rope_theta.powf((2 * i) as f32 / (cfg.rot_dim as f32)))
            .collect();
        let inv_freq = DeviceBuffer::alloc(device, inv_h.len() * 4)?;
        inv_freq.copy_from_host(bytemuck::cast_slice(&inv_h))?;

        let mask_id = cfg.mask_token_id as usize;
        let mask_off = mask_id
            .checked_mul(token_embd_row_bytes)
            .filter(|&o| o + token_embd_row_bytes <= token_embd_q6k.len())
            .ok_or_else(|| anyhow::anyhow!("DSpark mask_token_id {mask_id} out of range"))?;
        let mut mask_f32 = vec![0.0f32; d];
        q6_k_row_to_f32_cpu(
            &token_embd_q6k[mask_off..mask_off + token_embd_row_bytes],
            &mut mask_f32,
        );
        let mask_emb_f32 = DeviceBuffer::alloc(device, d * 4)?;
        mask_emb_f32.copy_from_host(bytemuck::cast_slice(&mask_f32))?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = dp(&format!("layers.{i}.decoder_layer."));
            layers.push(DsparkLayerWeights {
                in_norm: upload_bf16_norm(&st, device, &format!("{lp}input_layernorm.weight"), 1.0)?,
                post_norm: upload_bf16_norm(
                    &st,
                    device,
                    &format!("{lp}post_attention_layernorm.weight"),
                    1.0,
                )?,
                q_norm: upload_bf16_norm(&st, device, &format!("{lp}self_attn.q_norm.weight"), 1.0)?,
                k_norm: upload_bf16_norm(&st, device, &format!("{lp}self_attn.k_norm.weight"), 1.0)?,
                q: upload_mxfp8(&st, device, &format!("{lp}self_attn.q_proj"), q_dim, d)?,
                k: upload_mxfp8(&st, device, &format!("{lp}self_attn.k_proj"), kv_dim, d)?,
                v: upload_mxfp8(&st, device, &format!("{lp}self_attn.v_proj"), kv_dim, d)?,
                o: upload_mxfp8(&st, device, &format!("{lp}self_attn.o_proj"), d, q_dim)?,
                gate: upload_mxfp8(&st, device, &format!("{lp}mlp.gate_proj"), ff, d)?,
                up: upload_mxfp8(&st, device, &format!("{lp}mlp.up_proj"), ff, d)?,
                down: upload_mxfp8(&st, device, &format!("{lp}mlp.down_proj"), d, ff)?,
                kc_bf16: DeviceBuffer::alloc_zeroed(device, ring * kv_dim * 2)?,
                vc_bf16: DeviceBuffer::alloc_zeroed(device, ring * kv_dim * 2)?,
            });
        }

        let ctx_rows = DSPARK_MAX_CTX_CHUNK;
        let max_qkv_ff = d.max(q_dim).max(ff);
        let q8_1_row_bytes = (d / 32) * 36;
        let apart_floats = block * hkv * (DSPARK_ATTN_SPLITS as usize) * (hq / hkv) * mm3::ATTN_PART_STRIDE;

        let ctx = DeviceBuffer::alloc(device, ctx_rows * d * 4)?;
        let xn = DeviceBuffer::alloc(device, ctx_rows * d * 4)?;
        let h = DeviceBuffer::alloc(device, block * d * 4)?;
        let q = DeviceBuffer::alloc(device, block * q_dim * 4)?;
        let kk = DeviceBuffer::alloc(device, ctx_rows * kv_dim * 4)?;
        let vv = DeviceBuffer::alloc(device, ctx_rows * kv_dim * 4)?;
        let att = DeviceBuffer::alloc(device, block * q_dim * 4)?;
        let gt = DeviceBuffer::alloc(device, block * ff * 4)?;
        let up = DeviceBuffer::alloc(device, block * ff * 4)?;
        let hid = DeviceBuffer::alloc(device, block * d * 4)?;
        let hid_q8_1 = DeviceBuffer::alloc(device, block * q8_1_row_bytes)?;
        let logits = DeviceBuffer::alloc(device, block * vocab * 4)?;
        let x16 = DeviceBuffer::alloc(device, ctx_rows * d * 2)?;
        // Only KV MXFP8 projections [kv_dim x d] run with rows > 16 during context fill.
        let kv_w_bf16_scratch = DeviceBuffer::alloc(device, kv_dim * d * 2)?;
        let mx_part = DeviceBuffer::alloc(device, 16 * 16 * max_qkv_ff * 4)?;
        let attn_part = DeviceBuffer::alloc(device, apart_floats * 4)?;
        let mk_part = DeviceBuffer::alloc(device, mm3::MARKOV_PART_FLOATS * 4)?;
        let pos_dev = DeviceBuffer::alloc(device, ctx_rows.max(block) * 4)?;
        let lo_dev = DeviceBuffer::alloc(device, block * 4)?;
        let prev_dev = DeviceBuffer::alloc(device, 4)?;
        let out_dev = DeviceBuffer::alloc(device, block * 4)?;
        let conf_dev = DeviceBuffer::alloc(device, block * 4)?;

        Ok(Self {
            cfg,
            device,
            ring,
            stream,
            fc_bf16,
            hidden_norm,
            final_norm,
            markov_w1_bf16,
            markov_w2_bf16,
            conf_w,
            conf_b,
            inv_freq,
            mask_emb_f32,
            layers,
            ctx,
            xn,
            h,
            q,
            kk,
            vv,
            att,
            gt,
            up,
            hid,
            hid_q8_1,
            logits,
            x16,
            kv_w_bf16_scratch,
            mx_part,
            attn_part,
            mk_part,
            pos_dev,
            lo_dev,
            prev_dev,
            out_dev,
            conf_dev,
            host_anc_mapped: {
                set_device(device)?;
                PinnedMappedBuffer::alloc(d)?
            },
            host_out_pin: PinnedHostBuffer::alloc(block * 4)?,
            host_conf_pin: PinnedHostBuffer::alloc(block * 4)?,
        })
    }

    /// Project target tap hidden states (`[n_taps, stride_tokens, dim]` on `self.device`) into DSpark's BF16 KV cache.
    pub fn context_from_taps(
        &mut self,
        taps_dev: *const f32,
        n_tokens: usize,
        pos0: usize,
        stride_tokens: usize,
    ) -> Result<()> {
        if n_tokens == 0 {
            return Ok(());
        }
        set_device(self.device)?;
        let d = self.cfg.hidden_size;
        let kv_dim = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_taps = self.cfg.target_layer_ids.len();
        let eps = self.cfg.rms_norm_eps;
        let ring = self.ring as i32;

        let mut a = 0usize;
        while a < n_tokens {
            let n = (n_tokens - a).min(DSPARK_MAX_CTX_CHUNK);
            for j in 0..n_taps {
                unsafe {
                    let tap_src = taps_dev.add((j * stride_tokens + a) * d);
                    mm3::vision_bias_gelu(
                        tap_src,
                        std::ptr::null(),
                        n * d,
                        d as i32,
                        false,
                        std::ptr::null_mut(),
                        self.x16.ptr as *mut u16,
                        &self.stream,
                    );
                    mm3::bf16_gemm_mma(
                        self.x16.ptr as *const u16,
                        d as i32,
                        (self.fc_bf16.ptr as *const u16).add(j * d),
                        (n_taps * d) as i32,
                        self.ctx.ptr as *mut f32,
                        d as i32,
                        n as i32,
                        d as i32,
                        d as i32,
                        1.0,
                        if j == 0 { 0.0 } else { 1.0 },
                        &self.stream,
                    );
                }
            }
            unsafe {
                mm3::rms_norm_f32(
                    self.ctx.ptr as *const f32,
                    self.hidden_norm.ptr as *const f32,
                    n as i32,
                    d as i32,
                    eps,
                    false,
                    self.xn.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
            }
            unsafe {
                mm3::iota(
                    self.pos_dev.ptr as *mut i32,
                    n as i32,
                    (pos0 + a) as i32,
                    &self.stream,
                );
            }
            let x16_ptr = if n > 16 {
                self.x16.ptr as *mut u16
            } else {
                std::ptr::null_mut()
            };
            let w16_ptr = if n > 16 {
                self.kv_w_bf16_scratch.ptr as *mut u16
            } else {
                std::ptr::null_mut()
            };
            for l in &self.layers {
                unsafe {
                    mm3::mxfp8_linear(
                        l.k.w.ptr as *const u8,
                        l.k.s.ptr as *const u8,
                        l.k.n as i32,
                        l.k.k as i32,
                        self.xn.ptr as *const f32,
                        d as i32,
                        n as i32,
                        self.kk.ptr as *mut f32,
                        kv_dim as i32,
                        0.0,
                        self.mx_part.ptr as *mut f32,
                        w16_ptr,
                        x16_ptr,
                        &self.stream,
                    );
                    mm3::mxfp8_linear(
                        l.v.w.ptr as *const u8,
                        l.v.s.ptr as *const u8,
                        l.v.n as i32,
                        l.v.k as i32,
                        self.xn.ptr as *const f32,
                        d as i32,
                        n as i32,
                        self.vv.ptr as *mut f32,
                        kv_dim as i32,
                        0.0,
                        self.mx_part.ptr as *mut f32,
                        w16_ptr,
                        x16_ptr,
                        &self.stream,
                    );
                    mm3::qk_norm_rope_kv_store(
                        std::ptr::null_mut(),
                        self.kk.ptr as *const f32,
                        self.vv.ptr as *const f32,
                        std::ptr::null(),
                        l.k_norm.ptr as *const f32,
                        self.pos_dev.ptr as *const i32,
                        self.inv_freq.ptr as *const f32,
                        n as i32,
                        0,
                        self.cfg.num_kv_heads as i32,
                        self.cfg.head_dim as i32,
                        self.cfg.rot_dim as i32,
                        eps,
                        0,
                        0,
                        ring,
                        l.kc_bf16.ptr as *mut u16,
                        l.vc_bf16.ptr as *mut u16,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &self.stream,
                    );
                }
            }
            a += n;
        }
        Ok(())
    }

    pub fn snapshot_ring(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        set_device(self.device)?;
        self.stream.sync()?;
        let mut out = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let mut k = vec![0u8; l.kc_bf16.bytes];
            let mut v = vec![0u8; l.vc_bf16.bytes];
            l.kc_bf16.copy_to_host(&mut k)?;
            l.vc_bf16.copy_to_host(&mut v)?;
            out.push((k, v));
        }
        Ok(out)
    }

    pub fn restore_ring(&self, layers: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        if layers.len() != self.layers.len() {
            bail!(
                "dspark ring restore layer count mismatch: {} != {}",
                layers.len(),
                self.layers.len()
            );
        }
        set_device(self.device)?;
        self.stream.sync()?;
        for (l, (k, v)) in self.layers.iter().zip(layers.iter()) {
            l.kc_bf16.copy_from_host(k)?;
            l.vc_bf16.copy_from_host(v)?;
        }
        Ok(())
    }

    /// Generate up to `n_draft` (`<= block_size`) greedy draft tokens + acceptance confidences in one block forward.
    pub fn propose_greedy(
        &mut self,
        anchor_emb_f32: &[f32],
        anchor_id: u32,
        pos: usize,
        n_draft: usize,
        lm_head_q6k: &DeviceBuffer,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let d = self.cfg.hidden_size;
        let b = self.cfg.block_size;
        let n = n_draft.clamp(1, b);
        let q_dim = self.cfg.num_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.num_kv_heads * self.cfg.head_dim;
        let ff = self.cfg.ffn_hidden_size;
        let eps = self.cfg.rms_norm_eps;
        let scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
        let ring = self.ring as i32;
        let window = self.cfg.sliding_window;

        set_device(self.device)?;

        // Prepare anchor + mask embeddings + block positions [pos..pos+B]
        self.host_anc_mapped.as_mut_slice()[..d].copy_from_slice(&anchor_emb_f32[..d]);
        unsafe {
            mm3::dspark_draft_block(
                self.host_anc_mapped.dev_ptr() as *const f32,
                self.mask_emb_f32.ptr as *const f32,
                d as i32,
                b as i32,
                pos as i32,
                window as i32,
                anchor_id as i32,
                self.h.ptr as *mut f32,
                self.pos_dev.ptr as *mut i32,
                self.lo_dev.ptr as *mut i32,
                self.prev_dev.ptr as *mut i32,
                &self.stream,
            );
        }

        let hi = (pos + b - 1) as i32;
        for l in &self.layers {
            unsafe {
                mm3::rms_norm_f32(
                    self.h.ptr as *const f32,
                    l.in_norm.ptr as *const f32,
                    b as i32,
                    d as i32,
                    eps,
                    false,
                    self.xn.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.q.w.ptr as *const u8,
                    l.q.s.ptr as *const u8,
                    l.q.n as i32,
                    l.q.k as i32,
                    self.xn.ptr as *const f32,
                    d as i32,
                    b as i32,
                    self.q.ptr as *mut f32,
                    q_dim as i32,
                    0.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.k.w.ptr as *const u8,
                    l.k.s.ptr as *const u8,
                    l.k.n as i32,
                    l.k.k as i32,
                    self.xn.ptr as *const f32,
                    d as i32,
                    b as i32,
                    self.kk.ptr as *mut f32,
                    kv_dim as i32,
                    0.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.v.w.ptr as *const u8,
                    l.v.s.ptr as *const u8,
                    l.v.n as i32,
                    l.v.k as i32,
                    self.xn.ptr as *const f32,
                    d as i32,
                    b as i32,
                    self.vv.ptr as *mut f32,
                    kv_dim as i32,
                    0.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::qk_norm_rope_kv_store(
                    self.q.ptr as *mut f32,
                    self.kk.ptr as *const f32,
                    self.vv.ptr as *const f32,
                    l.q_norm.ptr as *const f32,
                    l.k_norm.ptr as *const f32,
                    self.pos_dev.ptr as *const i32,
                    self.inv_freq.ptr as *const f32,
                    b as i32,
                    self.cfg.num_heads as i32,
                    self.cfg.num_kv_heads as i32,
                    self.cfg.head_dim as i32,
                    self.cfg.rot_dim as i32,
                    eps,
                    0,
                    0,
                    ring,
                    l.kc_bf16.ptr as *mut u16,
                    l.vc_bf16.ptr as *mut u16,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::dspark_window_attn(
                    self.q.ptr as *const f32,
                    l.kc_bf16.ptr as *const u16,
                    l.vc_bf16.ptr as *const u16,
                    ring,
                    self.lo_dev.ptr as *const i32,
                    hi,
                    b as i32,
                    self.cfg.num_heads as i32,
                    self.cfg.num_kv_heads as i32,
                    scale,
                    self.attn_part.ptr as *mut f32,
                    DSPARK_ATTN_SPLITS,
                    self.att.ptr as *mut f32,
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.o.w.ptr as *const u8,
                    l.o.s.ptr as *const u8,
                    l.o.n as i32,
                    l.o.k as i32,
                    self.att.ptr as *const f32,
                    q_dim as i32,
                    b as i32,
                    self.h.ptr as *mut f32,
                    d as i32,
                    1.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::rms_norm_f32(
                    self.h.ptr as *const f32,
                    l.post_norm.ptr as *const f32,
                    b as i32,
                    d as i32,
                    eps,
                    false,
                    self.xn.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.gate.w.ptr as *const u8,
                    l.gate.s.ptr as *const u8,
                    l.gate.n as i32,
                    l.gate.k as i32,
                    self.xn.ptr as *const f32,
                    d as i32,
                    b as i32,
                    self.gt.ptr as *mut f32,
                    ff as i32,
                    0.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.up.w.ptr as *const u8,
                    l.up.s.ptr as *const u8,
                    l.up.n as i32,
                    l.up.k as i32,
                    self.xn.ptr as *const f32,
                    d as i32,
                    b as i32,
                    self.up.ptr as *mut f32,
                    ff as i32,
                    0.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::swigluoai(
                    self.gt.ptr as *const f32,
                    self.up.ptr as *const f32,
                    b as i32,
                    ff as i32,
                    1.702,
                    7.0,
                    self.gt.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
                mm3::mxfp8_linear(
                    l.down.w.ptr as *const u8,
                    l.down.s.ptr as *const u8,
                    l.down.n as i32,
                    l.down.k as i32,
                    self.gt.ptr as *const f32,
                    ff as i32,
                    b as i32,
                    self.h.ptr as *mut f32,
                    d as i32,
                    1.0,
                    self.mx_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &self.stream,
                );
            }
        }

        // Final norm -> Q6_K lm_head over first `n` block tokens -> rank-256 Markov greedy decode
        unsafe {
            mm3::rms_norm_f32(
                self.h.ptr as *const f32,
                self.final_norm.ptr as *const f32,
                b as i32,
                d as i32,
                eps,
                false,
                self.hid.ptr as *mut f32,
                self.hid_q8_1.ptr,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &self.stream,
            );
            mm3::gv_q6k_multi(
                1,
                lm_head_q6k.ptr,
                self.logits.ptr as *mut f32,
                self.cfg.vocab_size as i32,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                self.hid_q8_1.ptr,
                d as i32,
                n as i32,
                &self.stream,
            );
            let vocab = self.cfg.vocab_size;
            let rank = self.cfg.markov_rank;
            for i in 0..n {
                mm3::dspark_markov_greedy(
                    (self.logits.ptr as *const f32).add(i * vocab),
                    self.markov_w2_bf16.ptr as *const u16,
                    self.markov_w1_bf16.ptr as *const u16,
                    vocab as i32,
                    rank as i32,
                    self.prev_dev.ptr as *mut i32,
                    (self.out_dev.ptr as *mut i32).add(i),
                    self.conf_w.ptr as *const f32,
                    self.conf_b.ptr as *const f32,
                    (self.hid.ptr as *const f32).add(i * d),
                    d as i32,
                    (self.conf_dev.ptr as *mut f32).add(i),
                    self.mk_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    &self.stream,
                );
            }
            copy_d2h_raw_async(
                &mut self.host_out_pin.as_mut_slice()[..n * 4],
                self.out_dev.ptr,
                &self.stream,
            )?;
            copy_d2h_raw_async(
                &mut self.host_conf_pin.as_mut_slice()[..n * 4],
                self.conf_dev.ptr,
                &self.stream,
            )?;
        }
        self.stream.sync()?;

        let out_ids = bytemuck::cast_slice::<u8, i32>(&self.host_out_pin.as_slice()[..n * 4]);
        let conf_vals = bytemuck::cast_slice::<u8, f32>(&self.host_conf_pin.as_slice()[..n * 4]);
        let drafts = out_ids.iter().map(|&x| x.max(0) as u32).collect();
        let confs = conf_vals.to_vec();
        Ok((drafts, confs))
    }
}
