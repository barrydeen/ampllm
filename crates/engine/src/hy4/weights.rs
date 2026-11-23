use crate::config::{EngineConfig, MultiGpuMode};
use crate::hy4::engram::Hy4EngramTable;
use crate::hy4::vision::Hy4Vision;
use crate::kv_cache::KvQuantMode;
use ampllm_cache::CacheProfile;
use ampllm_cpu_kernels::{bf16_to_q8_0_cpu, CpuThreadPool};
use ampllm_cuda_kernels::{
    device_count, enable_peer_access, hy4, mem_info, set_device, DeviceBuffer,
    RegisteredMappedHostBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Hy4Config};
use anyhow::{bail, Context, Result};
use std::ffi::c_void;

#[inline]
pub fn align256(n: usize) -> usize {
    (n + 255) & !255
}

pub fn host_mem_available_bytes() -> usize {
    let Ok(txt) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            if let Some(kb_str) = rest.split_whitespace().next() {
                if let Ok(kb) = kb_str.parse::<usize>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

pub struct GpuMat {
    pub buf: DeviceBuffer,
    pub ggml_type: GgmlType,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
    pub row_bytes: usize,
}

impl GpuMat {
    #[inline]
    pub fn ptr(&self) -> *const c_void {
        self.buf.ptr
    }

    #[inline]
    pub fn u64_ptr(&self) -> u64 {
        self.buf.as_u64()
    }
}

pub enum ColdExpertHostStorage {
    /// Pinned WB mapped host slabs holding contiguous `[Gate | Up | Down]` blobs for all cold experts.
    PinnedSlabs(Vec<RegisteredMappedHostBuffer>),
    /// Fallback pageable host slabs if `cudaHostRegister` limit or RAM budget is tight.
    PageableSlabs(Vec<Vec<u8>>),
    /// Direct zero-copy pointers into GGUF mmap when host RAM is constrained.
    MmapDirect,
}

pub struct Hy4MoeWeights {
    pub tg: GgmlType,
    pub td: GgmlType,
    pub u_off: usize,
    pub d_off: usize,
    pub blob_bytes: usize,
    pub gu_row_bytes: usize,
    pub dn_row_bytes: usize,
    pub clamp_lim: f32,
    pub router: GpuMat,
    pub router_b: DeviceBuffer,
    pub rdown: GpuMat,
    pub rup: GpuMat,
    pub sh_gate: GpuMat,
    pub sh_up: GpuMat,
    pub sh_down: GpuMat,
    pub n_res: usize,
    pub is_partial: bool,
    pub slot_of: Vec<i32>,
    pub slot_d: DeviceBuffer,
    pub vram_experts: Option<DeviceBuffer>,
    pub cold_is_pinned: bool,
    pub cold_blob_ptrs: Vec<*const u8>,
    pub cold_gate_ptrs: Vec<*const u8>,
    pub cold_up_ptrs: Vec<*const u8>,
    pub cold_down_ptrs: Vec<*const u8>,
}

unsafe impl Send for Hy4MoeWeights {}
unsafe impl Sync for Hy4MoeWeights {}

pub struct Hy4DenseFfnWeights {
    pub gate: GpuMat,
    pub up: GpuMat,
    pub down: GpuMat,
}

pub struct Hy4IndexerWeights {
    pub q_b: GpuMat,
    pub k: GpuMat,
    /// Concatenated `[indexer_compressor_value ; indexer_compressor_gate]` -> output `[2 * ix_dim]`
    pub vg: GpuMat,
    pub proj: GpuMat,
    pub k_norm: DeviceBuffer,
    pub ape: DeviceBuffer,
    pub pnorm: DeviceBuffer,
}

pub struct Hy4EngramLayerWeights {
    pub module_idx: usize,
    pub kv: GpuMat,
    pub kv_b: DeviceBuffer,
    pub gate: DeviceBuffer,
    pub lookup_norm: DeviceBuffer,
    pub pre_gamma: DeviceBuffer,
    pub hc_scale: DeviceBuffer,
}

pub struct Hy4HcWeights {
    pub fn_w: DeviceBuffer, // Transposed to [cols, rows] for `fn_by_el = true`
    pub base: DeviceBuffer,
    pub scale: DeviceBuffer,
}

pub struct Hy4LayerWeights {
    pub il: usize,
    pub gpu_idx: usize, // index into weights.gpus (0 or 1)
    pub device: i32,
    pub heads: usize,
    pub swa: bool,
    pub ix: bool,
    pub is_mtp: bool,
    pub rope_base: f32,
    // Attention norms & mats
    pub attn_norm: DeviceBuffer,
    pub ffn_norm: DeviceBuffer,
    pub post_attn: DeviceBuffer,
    pub post_ffn: DeviceBuffer,
    pub q_a_norm: DeviceBuffer,
    pub kv_a_norm: DeviceBuffer,
    pub sinks: DeviceBuffer,
    pub q_a: GpuMat,
    pub q_b: GpuMat,
    pub kv_a: GpuMat,
    pub k_b: GpuMat,
    pub v_b: GpuMat,
    pub wo: GpuMat,
    pub gate: GpuMat,
    // Hyper-connections (None on MTP)
    pub hc_attn: Option<Hy4HcWeights>,
    pub hc_ffn: Option<Hy4HcWeights>,
    // MTP extras (Some on MTP)
    pub mtp_enorm: Option<DeviceBuffer>,
    pub mtp_hnorm: Option<DeviceBuffer>,
    pub mtp_head_norm: Option<DeviceBuffer>,
    pub mtp_eh_proj: Option<GpuMat>,
    // Indexer / Engram / FFN
    pub indexer: Option<Hy4IndexerWeights>,
    pub engram: Option<Hy4EngramLayerWeights>,
    pub dense_ffn: Option<Hy4DenseFfnWeights>,
    pub moe: Option<Hy4MoeWeights>,
}

pub struct Hy4HeadWeights {
    pub gpu_idx: usize,
    pub device: i32,
    pub out_w: GpuMat,
    pub out_norm: DeviceBuffer,
    pub hh: Hy4HcWeights,
}

pub struct Hy4ModelWeights {
    pub cfg: Hy4Config,
    pub max_ctx: usize,
    pub tc: usize,
    pub swa_ring_cap: usize,
    pub ivg_ring_cap: usize,
    pub hring_cap: usize,
    pub kv_is_q8: bool,
    pub kv_rb: usize,
    pub gpus: Vec<i32>,
    pub p2p_ok: bool,
    pub split_layer: usize,
    pub head_gpu_idx: usize,
    pub layers: Vec<Hy4LayerWeights>,
    pub mtp_layer: Option<Hy4LayerWeights>,
    pub head: Hy4HeadWeights,
    pub tok_embd_ptr: *const u8,
    pub tok_type: GgmlType,
    pub tok_row_bytes: usize,
    pub engram_table: Option<Hy4EngramTable>,
    pub vision: Option<Hy4Vision>,
    pub _cold_storage: ColdExpertHostStorage,
    pub max_blob_bytes: usize,
    pub vram_resident_experts: usize,
    pub total_trunk_routed_experts: usize,
    pub vram_used_mb: Vec<usize>,
}

unsafe impl Send for Hy4ModelWeights {}
unsafe impl Sync for Hy4ModelWeights {}

fn upload_f32(dev: i32, gguf: &GgufFile, name: &str, expected_elems: usize) -> Result<DeviceBuffer> {
    let info = gguf.tensor_info(name)?;
    if info.ggml_type != GgmlType::F32 || info.n_elements() != expected_elems {
        bail!(
            "tensor {name}: expected F32[{expected_elems}], got {:?}[{}]",
            info.ggml_type,
            info.n_elements()
        );
    }
    let bytes = gguf.tensor_bytes(name)?;
    let buf = DeviceBuffer::alloc(dev, bytes.len())?;
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn upload_hc_fn_by_col(
    dev: i32,
    gguf: &GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer> {
    let src = gguf.tensor_f32_slice(name)?;
    if src.len() != rows * cols {
        bail!(
            "tensor {name}: expected {} F32 elements, got {}",
            rows * cols,
            src.len()
        );
    }
    let mut col_major = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            col_major[c * rows + r] = src[r * cols + c];
        }
    }
    let buf = DeviceBuffer::alloc(dev, col_major.len() * 4)?;
    buf.copy_from_host(bytemuck::cast_slice(&col_major))?;
    Ok(buf)
}

fn bf16_bytes_to_q8_0_parallel(
    src_bytes: &[u8],
    rows: usize,
    k: usize,
    pool: &CpuThreadPool,
) -> Vec<u8> {
    let q8_rb = GgmlType::Q8_0.row_bytes(k);
    let mut dst = vec![0u8; rows * q8_rb];
    let src_u16: &[u16] = bytemuck::cast_slice(src_bytes);
    let src_ptr = src_u16.as_ptr() as usize;
    let dst_ptr = dst.as_mut_ptr() as usize;
    let chunk = (rows / pool.num_threads().max(1)).clamp(8, 256);
    let n_chunks = rows.div_ceil(chunk);
    pool.parallel_for(n_chunks, |c0, c1| {
        let r0 = c0 * chunk;
        let r1 = (c1 * chunk).min(rows);
        if r1 <= r0 {
            return;
        }
        unsafe {
            let s = std::slice::from_raw_parts((src_ptr as *const u16).add(r0 * k), (r1 - r0) * k);
            let d = std::slice::from_raw_parts_mut(
                (dst_ptr as *mut u8).add(r0 * q8_rb),
                (r1 - r0) * q8_rb,
            );
            bf16_to_q8_0_cpu(s, r1 - r0, k, d);
        }
    });
    dst
}

fn upload_mat(
    dev: i32,
    gguf: &GgufFile,
    name: &str,
    k: usize,
    n: usize,
    batch: usize,
    quantize_bf16_to_q8: bool,
    pool: &CpuThreadPool,
) -> Result<GpuMat> {
    let info = gguf.tensor_info(name)?;
    let bytes = gguf.tensor_bytes(name)?;
    let d0 = info.dims.first().copied().unwrap_or(1) as usize;
    let d1 = info.dims.get(1).copied().unwrap_or(1) as usize;
    let d2 = info.dims.get(2).copied().unwrap_or(1) as usize;
    if d0 != k || d1 != n || d2 != batch {
        bail!(
            "matrix {name}: expected [{k}, {n}, {batch}], got [{d0}, {d1}, {d2}]"
        );
    }

    if info.ggml_type == GgmlType::Bf16 && quantize_bf16_to_q8 && k % 32 == 0 {
        let rows = n * batch;
        let q = bf16_bytes_to_q8_0_parallel(bytes, rows, k, pool);
        let q_rb = GgmlType::Q8_0.row_bytes(k);
        let buf = DeviceBuffer::alloc(dev, q.len())?;
        buf.copy_from_host(&q)?;
        return Ok(GpuMat {
            buf,
            ggml_type: GgmlType::Q8_0,
            k,
            n,
            batch,
            row_bytes: q_rb,
        });
    }

    let rb = info.ggml_type.row_bytes(k);
    if rb * n * batch != bytes.len() {
        bail!(
            "matrix {name}: byte len mismatch {} != {}",
            bytes.len(),
            rb * n * batch
        );
    }
    let buf = DeviceBuffer::alloc(dev, bytes.len())?;
    buf.copy_from_host(bytes)?;
    Ok(GpuMat {
        buf,
        ggml_type: info.ggml_type,
        k,
        n,
        batch,
        row_bytes: rb,
    })
}

fn upload_concat_ix_vg(
    dev: i32,
    gguf: &GgufFile,
    val_name: &str,
    gate_name: &str,
    dim: usize,
    quantize_bf16_to_q8: bool,
    pool: &CpuThreadPool,
) -> Result<GpuMat> {
    let vi = gguf.tensor_info(val_name)?;
    let gi = gguf.tensor_info(gate_name)?;
    if vi.ggml_type != gi.ggml_type || vi.n_elements() != dim * dim || gi.n_elements() != dim * dim
    {
        bail!("indexer compressor value/gate shape or type mismatch for {val_name}");
    }
    let vb = gguf.tensor_bytes(val_name)?;
    let gb = gguf.tensor_bytes(gate_name)?;
    if vi.ggml_type == GgmlType::Bf16 && quantize_bf16_to_q8 && dim % 32 == 0 {
        let vq = bf16_bytes_to_q8_0_parallel(vb, dim, dim, pool);
        let gq = bf16_bytes_to_q8_0_parallel(gb, dim, dim, pool);
        let mut combined = Vec::with_capacity(vq.len() + gq.len());
        combined.extend_from_slice(&vq);
        combined.extend_from_slice(&gq);
        let buf = DeviceBuffer::alloc(dev, combined.len())?;
        buf.copy_from_host(&combined)?;
        return Ok(GpuMat {
            buf,
            ggml_type: GgmlType::Q8_0,
            k: dim,
            n: 2 * dim,
            batch: 1,
            row_bytes: GgmlType::Q8_0.row_bytes(dim),
        });
    }
    let mut combined = Vec::with_capacity(vb.len() + gb.len());
    combined.extend_from_slice(vb);
    combined.extend_from_slice(gb);
    let buf = DeviceBuffer::alloc(dev, combined.len())?;
    buf.copy_from_host(&combined)?;
    Ok(GpuMat {
        buf,
        ggml_type: vi.ggml_type,
        k: dim,
        n: 2 * dim,
        batch: 1,
        row_bytes: vi.ggml_type.row_bytes(dim),
    })
}

fn layer_cache_bytes(
    cfg: &Hy4Config,
    il: usize,
    max_ctx: usize,
    swa_ring_cap: usize,
    ivg_ring_cap: usize,
    kv_rb: usize,
    pool_elem_bytes: usize,
) -> usize {
    if cfg.swa[il] {
        swa_ring_cap * kv_rb
    } else {
        let n_pools = max_ctx / cfg.ix_kpool + 1;
        max_ctx * kv_rb
            + n_pools * cfg.ix_dim * pool_elem_bytes
            + ivg_ring_cap * 2 * cfg.ix_dim * pool_elem_bytes
    }
}

fn layer_dense_vram_bytes(
    gguf: &GgufFile,
    cfg: &Hy4Config,
    il: usize,
    max_ctx: usize,
    swa_ring_cap: usize,
    ivg_ring_cap: usize,
    kv_rb: usize,
    pool_elem_bytes: usize,
) -> Result<usize> {
    let is_mtp = il >= cfg.n_layer;
    let pre = format!("blk.{il}.");
    let mut b = 0usize;
    for t in &gguf.tensors {
        if t.name.starts_with(&pre) && !t.name.contains("_exps.") {
            // In MTP, BF16 matrices are quantized to Q8_0 (slightly larger than half BF16).
            let sz = if is_mtp && t.ggml_type == GgmlType::Bf16 {
                (t.byte_len / 2) * 34 / 32 + 256
            } else {
                t.byte_len + 256
            };
            b += sz;
        }
    }
    b += layer_cache_bytes(cfg, il, max_ctx, swa_ring_cap, ivg_ring_cap, kv_rb, pool_elem_bytes);
    Ok(b)
}

fn estimate_scratch_arena_bytes(cfg: &Hy4Config, tc: usize, max_ctx: usize, h_max: usize) -> usize {
    crate::hy4::state::Hy4ScratchLayout::compute(cfg, tc, max_ctx, h_max).total_arena
}

impl Hy4ModelWeights {
    pub fn load(
        mut gguf: GgufFile,
        engine_cfg: &EngineConfig,
        pool: &CpuThreadPool,
        max_ctx_req: usize,
        profile: Option<&CacheProfile>,
    ) -> Result<(Self, GgufFile)> {
        if gguf.shard_paths.is_empty() {
            bail!("GgufFile has no shard paths");
        }
        let main_path = gguf.shard_paths[0].clone();
        let (mtp_path_opt, mmproj_path_opt) =
            ampllm_gguf::discover_hy4_aux_files(&main_path, None, None)?;

        // Merge standalone MTP block shard if trunk didn't already include blk.<n_layer>.
        let pre_cfg = Hy4Config::from_gguf(&gguf)?;
        if pre_cfg.n_mtp == 0 || !gguf.has_tensor(&format!("blk.{}.attn_norm.weight", pre_cfg.n_layer)) {
            if let Some(ref mp) = mtp_path_opt {
                if !gguf.shard_paths.iter().any(|p| p == mp) {
                    tracing::info!("Hy4: attaching standalone MTP GGUF {}", mp.display());
                    gguf.merge_aux_shard(mp)?;
                }
            }
        }

        let mut cfg = Hy4Config::from_gguf(&gguf)?;
        let max_ctx = max_ctx_req.max(2048).min(cfg.context_length);
        let max_ctx = max_ctx.div_ceil(cfg.ix_kpool) * cfg.ix_kpool;

        // Extract vocab strings directly from GGUF metadata to configure SigLIP vision
        let vocab_tokens: Vec<String> = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or_default().to_owned())
                    .collect()
            })
            .unwrap_or_default();

        let vision = if let Some(ref vp) = mmproj_path_opt {
            match GgufFile::open(vp) {
                Ok(v_gguf) => {
                    let v = Hy4Vision::from_gguf(v_gguf, &vocab_tokens, cfg.n_embd)?;
                    // Mask image pad token in Engram token_map just like maya-hy4
                    if let Some(ref mut eg) = cfg.engram {
                        let idx = v.cfg.image_token_id as usize;
                        if idx < eg.token_map.len() {
                            eg.token_map[idx] = -1;
                        }
                    }
                    tracing::info!("Hy4: loaded SigLIP vision tower from {}", vp.display());
                    Some(v)
                }
                Err(e) => {
                    tracing::warn!("Hy4: failed to open mmproj {}: {e:#}", vp.display());
                    None
                }
            }
        } else {
            None
        };

        // Determine KV mode (Q8_0 by default for max_ctx > 16384 unless AMPLLM_HY4_KV_Q8=0)
        let kv_is_q8 = match std::env::var("AMPLLM_HY4_KV_Q8").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => engine_cfg.kv_mode == KvQuantMode::Q8_0 || max_ctx > 16384,
        };
        let kv_rb = if kv_is_q8 {
            hy4::kv_q8_row_bytes(cfg.kv_lora_rank, cfg.rope_dim)
        } else {
            cfg.kv_row() * 2
        };
        let pool_elem_bytes = if kv_is_q8 { 2 } else { 4 };

        let tc = std::env::var("AMPLLM_HY4_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| if max_ctx >= 4096 { 4096 } else { engine_cfg.max_batch })
            .clamp(256, max_ctx.min(4096));
        let swa_ring_cap = (cfg.n_swa + tc + 64).min(max_ctx).max(512);
        let ivg_ring_cap = (tc + 64).max(4096).min(max_ctx.div_ceil(cfg.ix_kpool) * cfg.ix_kpool);
        let ivg_ring_cap = ivg_ring_cap.div_ceil(cfg.ix_kpool) * cfg.ix_kpool;
        let hring_cap = swa_ring_cap.max(2048);

        // Select GPUs
        let ndev = device_count()?;
        if ndev < 1 {
            bail!("no CUDA GPUs found");
        }
        let gpus = if engine_cfg.multi_gpu != MultiGpuMode::Single && ndev >= 2 {
            vec![0, 1]
        } else {
            vec![engine_cfg.device]
        };
        let p2p_ok = if gpus.len() >= 2 {
            let a = enable_peer_access(gpus[0], gpus[1]).unwrap_or(false);
            let b = enable_peer_access(gpus[1], gpus[0]).unwrap_or(false);
            a && b
        } else {
            false
        };

        // Query free VRAM on each GPU before allocations
        let mut free_vram = Vec::with_capacity(gpus.len());
        for &d in &gpus {
            let (f, _) = mem_info(d)?;
            free_vram.push(f);
        }

        // Compute per-layer fixed bytes and single-expert blob sizes
        let n_layer = cfg.n_layer;
        let has_mtp = cfg.n_mtp > 0 && engine_cfg.use_spec;
        let n_tot = n_layer + usize::from(has_mtp);

        let mut layer_fixed = vec![0usize; n_tot];
        let mut layer_blob = vec![0usize; n_tot];
        let mut layer_u_off = vec![0usize; n_tot];
        let mut layer_d_off = vec![0usize; n_tot];
        let mut layer_tg = vec![GgmlType::F32; n_tot];
        let mut layer_td = vec![GgmlType::F32; n_tot];
        let mut layer_gu_rb = vec![0usize; n_tot];
        let mut layer_dn_rb = vec![0usize; n_tot];

        for il in 0..n_tot {
            layer_fixed[il] = layer_dense_vram_bytes(
                &gguf,
                &cfg,
                il,
                max_ctx,
                swa_ring_cap,
                ivg_ring_cap,
                kv_rb,
                pool_elem_bytes,
            )?;
            let is_mtp = il >= n_layer;
            let is_dense = !is_mtp && il < cfg.n_dense_lead;
            if !is_dense {
                let g_info = gguf.tensor_info(&format!("blk.{il}.ffn_gate_exps.weight"))?;
                let d_info = gguf.tensor_info(&format!("blk.{il}.ffn_down_exps.weight"))?;
                let mut tg = g_info.ggml_type;
                let mut td = d_info.ggml_type;
                if is_mtp && tg == GgmlType::Bf16 && td == GgmlType::Bf16 {
                    tg = GgmlType::Q8_0;
                    td = GgmlType::Q8_0;
                }
                let gu_rb = tg.row_bytes(cfg.n_expert_latent);
                let dn_rb = td.row_bytes(cfg.n_ff_exp);
                let g_bytes = gu_rb * cfg.n_ff_exp;
                let d_bytes = dn_rb * cfg.n_expert_latent;
                let u_off = align256(g_bytes);
                let d_off = 2 * u_off;
                let blob = align256(d_off + d_bytes);
                layer_tg[il] = tg;
                layer_td[il] = td;
                layer_gu_rb[il] = gu_rb;
                layer_dn_rb[il] = dn_rb;
                layer_u_off[il] = u_off;
                layer_d_off[il] = d_off;
                layer_blob[il] = blob;
            }
        }

        let max_blob_bytes = layer_blob[..n_layer].iter().copied().max().unwrap_or(0);
        let dma_stage_bytes = if max_blob_bytes > 0 {
            24 * max_blob_bytes
        } else {
            0
        };

        let h_max_all = cfg.heads.iter().copied().max().unwrap_or(64);
        let scratch_per_gpu = estimate_scratch_arena_bytes(&cfg, tc, max_ctx, h_max_all);
        let head_bytes = gguf.tensor_bytes("output.weight")?.len()
            + gguf.tensor_bytes("output_norm.weight")?.len()
            + gguf.tensor_bytes("output_hc_fn.weight")?.len()
            + hring_cap * cfg.n_embd * 4
            + if has_mtp {
                5 * crate::hy4::state::HY4_ATTN_SUB.min(tc) * cfg.n_embd * 4
            } else {
                0
            }
            + (16 << 20);
        let mtp_total_exp_bytes = if has_mtp {
            cfg.n_expert * layer_blob[n_layer]
        } else {
            0
        };

        // Plan split across 2 GPUs (or 1 GPU)
        let head_gpu_idx = gpus.len() - 1;
        let safety_margin = 512usize << 20;
        let split_layer = if gpus.len() == 1 {
            n_layer
        } else if let Some(sp) = std::env::var("AMPLLM_HY4_SPLIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&sp| sp >= 1 && sp < n_layer)
        {
            sp
        } else {
            let mut best_sp = n_layer / 2;
            let mut best_score = (-1.0f64, 0usize);
            for sp in 1..n_layer {
                let f0: usize = layer_fixed[..sp].iter().sum();
                let mut f1: usize = layer_fixed[sp..n_layer].iter().sum();
                if has_mtp {
                    f1 += layer_fixed[n_layer] + mtp_total_exp_bytes;
                }
                let need0 = f0 + scratch_per_gpu + dma_stage_bytes + safety_margin;
                let need1 = f1 + head_bytes + scratch_per_gpu + dma_stage_bytes + safety_margin;
                if need0 >= free_vram[0] || need1 >= free_vram[1] {
                    continue;
                }
                let avail0 = free_vram[0] - need0;
                let avail1 = free_vram[1] - need1;
                let tot0: usize = (0..sp).map(|il| cfg.n_expert * layer_blob[il]).sum();
                let tot1: usize = (sp..n_layer).map(|il| cfg.n_expert * layer_blob[il]).sum();
                let frac0 = if tot0 > 0 {
                    (avail0 as f64 / tot0 as f64).min(1.0)
                } else {
                    1.0
                };
                let frac1 = if tot1 > 0 {
                    (avail1 as f64 / tot1 as f64).min(1.0)
                } else {
                    1.0
                };
                let est_res0 = (avail0 as f64 / (max_blob_bytes.max(1) as f64))
                    .min(((sp.saturating_sub(cfg.n_dense_lead)) * cfg.n_expert) as f64)
                    as usize;
                let est_res1 = (avail1 as f64 / (max_blob_bytes.max(1) as f64))
                    .min(((n_layer - sp) * cfg.n_expert) as f64)
                    as usize;
                let score = (frac0.min(frac1), est_res0 + est_res1);
                if score > best_score {
                    best_score = score;
                    best_sp = sp;
                }
            }
            best_sp
        };

        // Plan exact VRAM expert count per trunk layer
        let mut n_res_per_layer = vec![0usize; n_tot];
        for (gi, range) in if gpus.len() == 1 {
            vec![0..n_layer]
        } else {
            vec![0..split_layer, split_layer..n_layer]
        }
        .into_iter()
        .enumerate()
        {
            let mut fixed_sum: usize = range.clone().map(|il| layer_fixed[il]).sum();
            fixed_sum += scratch_per_gpu + dma_stage_bytes + safety_margin;
            if gi == head_gpu_idx {
                fixed_sum += head_bytes;
                if has_mtp {
                    fixed_sum += layer_fixed[n_layer] + mtp_total_exp_bytes;
                }
            }
            if fixed_sum >= free_vram[gi] {
                bail!(
                    "GPU {} insufficient VRAM: need {} MiB fixed, have {} MiB free",
                    gpus[gi],
                    fixed_sum >> 20,
                    free_vram[gi] >> 20
                );
            }
            let budget = free_vram[gi] - fixed_sum;
            let total_exp_bytes: usize = range.clone().map(|il| cfg.n_expert * layer_blob[il]).sum();
            let frac = if total_exp_bytes > 0 {
                (budget as f64 / total_exp_bytes as f64).min(1.0)
            } else {
                0.0
            };
            let mut used_exp = 0usize;
            for il in range.clone() {
                if layer_blob[il] > 0 {
                    let k = ((frac * cfg.n_expert as f64).floor() as usize).min(cfg.n_expert);
                    n_res_per_layer[il] = k;
                    used_exp += k * layer_blob[il];
                }
            }
            // Top up remaining budget round-robin across MoE layers on this GPU
            let mut progress = true;
            while progress {
                progress = false;
                for il in range.clone() {
                    if layer_blob[il] > 0
                        && n_res_per_layer[il] < cfg.n_expert
                        && used_exp + layer_blob[il] <= budget
                    {
                        n_res_per_layer[il] += 1;
                        used_exp += layer_blob[il];
                        progress = true;
                    }
                }
            }
        }
        if has_mtp {
            n_res_per_layer[n_layer] = cfg.n_expert;
        }

        // Load layers per GPU in parallel
        let load_one_layer = |il: usize, gi: usize| -> Result<Hy4LayerWeights> {
            let dev = gpus[gi];
            set_device(dev)?;
            let is_mtp = il >= n_layer;
            let qbf16 = is_mtp;
            let pre = format!("blk.{il}.");
            let e = cfg.n_embd;
            let h = cfg.heads[il];
            let km = cfg.k_mla;
            let vm = cfg.v_mla;
            let kl = cfg.kv_lora_rank;
            let nope = cfg.nope();
            let kr = cfg.kv_row();
            let swa = cfg.swa[il];
            let ix = !swa && cfg.indexer[il] && cfg.ix_top_k > 0;
            let rope_base = if swa { cfg.rope_base_swa } else { cfg.rope_base };

            let attn_norm = upload_f32(dev, &gguf, &format!("{pre}attn_norm.weight"), e)?;
            let ffn_norm = upload_f32(dev, &gguf, &format!("{pre}ffn_norm.weight"), e)?;
            let post_attn = upload_f32(dev, &gguf, &format!("{pre}post_attention_norm.weight"), e)?;
            let post_ffn = upload_f32(dev, &gguf, &format!("{pre}post_ffw_norm.weight"), e)?;
            let q_a_norm = upload_f32(dev, &gguf, &format!("{pre}attn_q_a_norm.weight"), cfg.q_lora_rank)?;
            let kv_a_norm = upload_f32(dev, &gguf, &format!("{pre}attn_kv_a_norm.weight"), kl)?;
            let sinks = upload_f32(dev, &gguf, &format!("{pre}attn_sinks.weight"), h)?;

            let q_a = upload_mat(dev, &gguf, &format!("{pre}attn_q_a.weight"), e, cfg.q_lora_rank, 1, qbf16, pool)?;
            let q_b = upload_mat(dev, &gguf, &format!("{pre}attn_q_b.weight"), cfg.q_lora_rank, h * km, 1, qbf16, pool)?;
            let kv_a = upload_mat(dev, &gguf, &format!("{pre}attn_kv_a_mqa.weight"), e, kr, 1, qbf16, pool)?;
            let k_b = upload_mat(dev, &gguf, &format!("{pre}attn_k_b.weight"), nope, kl, h, qbf16, pool)?;
            let v_b = upload_mat(dev, &gguf, &format!("{pre}attn_v_b.weight"), kl, vm, h, qbf16, pool)?;
            let wo = upload_mat(dev, &gguf, &format!("{pre}attn_output.weight"), h * vm, e, 1, qbf16, pool)?;
            let gate = upload_mat(dev, &gguf, &format!("{pre}attn_gate.weight"), e, h * vm, 1, qbf16, pool)?;

            let (hc_attn, hc_ffn) = if !is_mtp {
                let a = Hy4HcWeights {
                    fn_w: upload_hc_fn_by_col(dev, &gguf, &format!("{pre}hc_attn_fn.weight"), 2 * cfg.hc, cfg.hc * e)?,
                    base: upload_f32(dev, &gguf, &format!("{pre}hc_attn_base.weight"), 2 * cfg.hc)?,
                    scale: upload_f32(dev, &gguf, &format!("{pre}hc_attn_scale.weight"), 2)?,
                };
                let f = Hy4HcWeights {
                    fn_w: upload_hc_fn_by_col(dev, &gguf, &format!("{pre}hc_ffn_fn.weight"), 2 * cfg.hc, cfg.hc * e)?,
                    base: upload_f32(dev, &gguf, &format!("{pre}hc_ffn_base.weight"), 2 * cfg.hc)?,
                    scale: upload_f32(dev, &gguf, &format!("{pre}hc_ffn_scale.weight"), 2)?,
                };
                (Some(a), Some(f))
            } else {
                (None, None)
            };

            let (mtp_enorm, mtp_hnorm, mtp_head_norm, mtp_eh_proj) = if is_mtp {
                (
                    Some(upload_f32(dev, &gguf, &format!("{pre}nextn.enorm.weight"), e)?),
                    Some(upload_f32(dev, &gguf, &format!("{pre}nextn.hnorm.weight"), e)?),
                    Some(upload_f32(dev, &gguf, &format!("{pre}nextn.shared_head_norm.weight"), e)?),
                    Some(upload_mat(dev, &gguf, &format!("{pre}nextn.eh_proj.weight"), 2 * e, e, 1, qbf16, pool)?),
                )
            } else {
                (None, None, None, None)
            };

            let indexer = if ix {
                Some(Hy4IndexerWeights {
                    q_b: upload_mat(dev, &gguf, &format!("{pre}indexer.attn_q_b.weight"), cfg.q_lora_rank, cfg.ix_heads * cfg.ix_dim, 1, qbf16, pool)?,
                    k: upload_mat(dev, &gguf, &format!("{pre}indexer.attn_k.weight"), e, cfg.ix_dim, 1, qbf16, pool)?,
                    vg: upload_concat_ix_vg(
                        dev,
                        &gguf,
                        &format!("{pre}indexer_compressor_value.weight"),
                        &format!("{pre}indexer_compressor_gate.weight"),
                        cfg.ix_dim,
                        qbf16,
                        pool,
                    )?,
                    proj: upload_mat(dev, &gguf, &format!("{pre}indexer.proj.weight"), e, cfg.ix_heads, 1, false, pool)?,
                    k_norm: upload_f32(dev, &gguf, &format!("{pre}indexer.k_norm.weight"), cfg.ix_dim)?,
                    ape: upload_f32(dev, &gguf, &format!("{pre}indexer_compressor_ape.weight"), cfg.ix_dim * cfg.ix_kpool)?,
                    pnorm: upload_f32(dev, &gguf, &format!("{pre}indexer_compressor_norm.weight"), cfg.ix_dim)?,
                })
            } else {
                None
            };

            let engram = if !is_mtp {
                if let Some(ref eg_cfg) = cfg.engram {
                    if let Some(mod_idx) = eg_cfg.module_of(il) {
                        let cols = eg_cfg.cols();
                        let hd = eg_cfg.head_dim;
                        let hc = cfg.hc;
                        Some(Hy4EngramLayerWeights {
                            module_idx: mod_idx,
                            kv: upload_mat(dev, &gguf, &format!("{pre}engram_kv.weight"), cols * hd, (hc + 1) * e, 1, false, pool)?,
                            kv_b: upload_f32(dev, &gguf, &format!("{pre}engram_kv_b.weight"), (hc + 1) * e)?,
                            gate: upload_f32(dev, &gguf, &format!("{pre}engram_gate.weight"), hc * e)?,
                            lookup_norm: upload_f32(dev, &gguf, &format!("{pre}engram_lookup_norm.weight"), cols * hd)?,
                            pre_gamma: upload_f32(dev, &gguf, &format!("{pre}engram_pre_gamma.weight"), e)?,
                            hc_scale: upload_f32(dev, &gguf, &format!("{pre}engram_hc_scale.weight"), hc * e)?,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let is_dense = !is_mtp && il < cfg.n_dense_lead;
            let (dense_ffn, moe) = if is_dense {
                let dff = Hy4DenseFfnWeights {
                    gate: upload_mat(dev, &gguf, &format!("{pre}ffn_gate.weight"), e, cfg.n_ff, 1, false, pool)?,
                    up: upload_mat(dev, &gguf, &format!("{pre}ffn_up.weight"), e, cfg.n_ff, 1, false, pool)?,
                    down: upload_mat(dev, &gguf, &format!("{pre}ffn_down.weight"), cfg.n_ff, e, 1, false, pool)?,
                };
                (Some(dff), None)
            } else {
                let lat = cfg.n_expert_latent;
                let ffe = cfg.n_ff_exp;
                let fs = ffe * cfg.n_expert_shared;
                let router = upload_mat(dev, &gguf, &format!("{pre}ffn_gate_inp.weight"), e, cfg.n_expert, 1, false, pool)?;
                let router_b = upload_f32(dev, &gguf, &format!("{pre}exp_probs_b.bias"), cfg.n_expert)?;
                let rdown = upload_mat(dev, &gguf, &format!("{pre}ffn_routed_down.weight"), e, lat, 1, qbf16, pool)?;
                let rup = upload_mat(dev, &gguf, &format!("{pre}ffn_routed_up.weight"), lat, e, 1, qbf16, pool)?;
                let sh_gate = upload_mat(dev, &gguf, &format!("{pre}ffn_gate_shexp.weight"), e, fs, 1, qbf16, pool)?;
                let sh_up = upload_mat(dev, &gguf, &format!("{pre}ffn_up_shexp.weight"), e, fs, 1, qbf16, pool)?;
                let sh_down = upload_mat(dev, &gguf, &format!("{pre}ffn_down_shexp.weight"), fs, e, 1, qbf16, pool)?;

                let n_res = n_res_per_layer[il];
                let mut order: Vec<usize> = (0..cfg.n_expert).collect();
                if !is_mtp {
                    if let Some(prof) = profile {
                        if let Some(ranked) = prof.ranked_experts.get(il) {
                            if ranked.len() == cfg.n_expert {
                                order.copy_from_slice(ranked);
                            }
                        }
                    }
                }
                let mut res_ids: Vec<usize> = order[..n_res].to_vec();
                res_ids.sort_unstable();
                let mut slot_of = vec![-1i32; cfg.n_expert];
                for (slot, &eid) in res_ids.iter().enumerate() {
                    slot_of[eid] = slot as i32;
                }
                let slot_d = DeviceBuffer::alloc(dev, cfg.n_expert * 4)?;
                slot_d.copy_from_host(bytemuck::cast_slice(&slot_of))?;

                let tg = layer_tg[il];
                let td = layer_td[il];
                let gu_rb = layer_gu_rb[il];
                let dn_rb = layer_dn_rb[il];
                let u_off = layer_u_off[il];
                let d_off = layer_d_off[il];
                let blob_bytes = layer_blob[il];

                let sg_raw = gguf.tensor_bytes(&format!("{pre}ffn_gate_exps.weight"))?;
                let su_raw = gguf.tensor_bytes(&format!("{pre}ffn_up_exps.weight"))?;
                let sd_raw = gguf.tensor_bytes(&format!("{pre}ffn_down_exps.weight"))?;

                let g_raw_exp_b = if is_mtp && sg_raw.len() == cfg.n_expert * ffe * lat * 2 {
                    ffe * lat * 2
                } else {
                    gu_rb * ffe
                };
                let d_raw_exp_b = if is_mtp && sd_raw.len() == cfg.n_expert * lat * ffe * 2 {
                    lat * ffe * 2
                } else {
                    dn_rb * lat
                };
                let round_bf16 = is_mtp && g_raw_exp_b == ffe * lat * 2;

                let vram_experts = if n_res > 0 {
                    let vbuf = DeviceBuffer::alloc(dev, n_res * blob_bytes)?;
                    let chunk_slots = 64usize;
                    let mut stage = vec![0u8; chunk_slots.min(n_res) * blob_bytes];
                    for s0 in (0..n_res).step_by(chunk_slots) {
                        let ns = (n_res - s0).min(chunk_slots);
                        for i in 0..ns {
                            let eid = res_ids[s0 + i];
                            let dst = &mut stage[i * blob_bytes..(i + 1) * blob_bytes];
                            if round_bf16 {
                                let g_slice: &[u16] = bytemuck::cast_slice(&sg_raw[eid * g_raw_exp_b..(eid + 1) * g_raw_exp_b]);
                                let u_slice: &[u16] = bytemuck::cast_slice(&su_raw[eid * g_raw_exp_b..(eid + 1) * g_raw_exp_b]);
                                let d_slice: &[u16] = bytemuck::cast_slice(&sd_raw[eid * d_raw_exp_b..(eid + 1) * d_raw_exp_b]);
                                bf16_to_q8_0_cpu(g_slice, ffe, lat, &mut dst[..gu_rb * ffe]);
                                bf16_to_q8_0_cpu(u_slice, ffe, lat, &mut dst[u_off..u_off + gu_rb * ffe]);
                                bf16_to_q8_0_cpu(d_slice, lat, ffe, &mut dst[d_off..d_off + dn_rb * lat]);
                            } else {
                                dst[..g_raw_exp_b].copy_from_slice(&sg_raw[eid * g_raw_exp_b..(eid + 1) * g_raw_exp_b]);
                                dst[u_off..u_off + g_raw_exp_b].copy_from_slice(&su_raw[eid * g_raw_exp_b..(eid + 1) * g_raw_exp_b]);
                                dst[d_off..d_off + d_raw_exp_b].copy_from_slice(&sd_raw[eid * d_raw_exp_b..(eid + 1) * d_raw_exp_b]);
                            }
                        }
                        let dst_dev_ptr = unsafe { vbuf.ptr.cast::<u8>().add(s0 * blob_bytes) };
                        let st_stream = ampllm_cuda_kernels::CudaStream::new(dev)?;
                        unsafe {
                            ampllm_cuda_kernels::copy_h2d_raw_async(
                                dst_dev_ptr.cast(),
                                &stage[..ns * blob_bytes],
                                &st_stream,
                            )?;
                        }
                        st_stream.sync()?;
                    }
                    Some(vbuf)
                } else {
                    None
                };

                // Initially populate cold mmap pointers; cold packing below will upgrade them to pinned blobs if RAM allows.
                let mut cold_gate_ptrs = vec![std::ptr::null(); cfg.n_expert];
                let mut cold_up_ptrs = vec![std::ptr::null(); cfg.n_expert];
                let mut cold_down_ptrs = vec![std::ptr::null(); cfg.n_expert];
                let cold_blob_ptrs = vec![std::ptr::null(); cfg.n_expert];
                for eid in 0..cfg.n_expert {
                    if slot_of[eid] < 0 {
                        unsafe {
                            cold_gate_ptrs[eid] = sg_raw.as_ptr().add(eid * g_raw_exp_b);
                            cold_up_ptrs[eid] = su_raw.as_ptr().add(eid * g_raw_exp_b);
                            cold_down_ptrs[eid] = sd_raw.as_ptr().add(eid * d_raw_exp_b);
                        }
                    }
                }

                let moe_w = Hy4MoeWeights {
                    tg,
                    td,
                    u_off,
                    d_off,
                    blob_bytes,
                    gu_row_bytes: gu_rb,
                    dn_row_bytes: dn_rb,
                    clamp_lim: cfg.swiglu_limit[il],
                    router,
                    router_b,
                    rdown,
                    rup,
                    sh_gate,
                    sh_up,
                    sh_down,
                    n_res,
                    is_partial: n_res < cfg.n_expert,
                    slot_of,
                    slot_d,
                    vram_experts,
                    cold_is_pinned: false,
                    cold_blob_ptrs,
                    cold_gate_ptrs,
                    cold_up_ptrs,
                    cold_down_ptrs,
                };
                (None, Some(moe_w))
            };

            Ok(Hy4LayerWeights {
                il,
                gpu_idx: gi,
                device: dev,
                heads: h,
                swa,
                ix,
                is_mtp,
                rope_base,
                attn_norm,
                ffn_norm,
                post_attn,
                post_ffn,
                q_a_norm,
                kv_a_norm,
                sinks,
                q_a,
                q_b,
                kv_a,
                k_b,
                v_b,
                wo,
                gate,
                hc_attn,
                hc_ffn,
                mtp_enorm,
                mtp_hnorm,
                mtp_head_norm,
                mtp_eh_proj,
                indexer,
                engram,
                dense_ffn,
                moe,
            })
        };

        // Load trunk layers in parallel per GPU, plus Head & MTP on head_gpu
        let mut layers: Vec<Option<Hy4LayerWeights>> = (0..n_layer).map(|_| None).collect();
        let mut mtp_layer: Option<Hy4LayerWeights> = None;
        let mut head_opt: Option<Hy4HeadWeights> = None;

        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::with_capacity(gpus.len());
            for (gi, _) in gpus.iter().enumerate() {
                let range = if gpus.len() == 1 {
                    0..n_layer
                } else if gi == 0 {
                    0..split_layer
                } else {
                    split_layer..n_layer
                };
                let is_head_gpu = gi == head_gpu_idx;
                let dev = gpus[gi];
                let gguf_ref = &gguf;
                let cfg_ref = &cfg;
                handles.push(scope.spawn(move || -> Result<(Vec<(usize, Hy4LayerWeights)>, Option<Hy4LayerWeights>, Option<Hy4HeadWeights>)> {
                    let mut out_l = Vec::with_capacity(range.len());
                    for il in range {
                        out_l.push((il, load_one_layer(il, gi)?));
                    }
                    let mtp_l = if is_head_gpu && has_mtp {
                        Some(load_one_layer(n_layer, gi)?)
                    } else {
                        None
                    };
                    let head_w = if is_head_gpu {
                        set_device(dev)?;
                        let e = cfg_ref.n_embd;
                        let out_w = upload_mat(dev, gguf_ref, "output.weight", e, cfg_ref.n_vocab, 1, false, pool)?;
                        let out_norm = upload_f32(dev, gguf_ref, "output_norm.weight", e)?;
                        let hh = Hy4HcWeights {
                            fn_w: upload_hc_fn_by_col(dev, gguf_ref, "output_hc_fn.weight", cfg_ref.hc, cfg_ref.hc * e)?,
                            base: upload_f32(dev, gguf_ref, "output_hc_base.weight", cfg_ref.hc)?,
                            scale: upload_f32(dev, gguf_ref, "output_hc_scale.weight", 1)?,
                        };
                        Some(Hy4HeadWeights {
                            gpu_idx: gi,
                            device: dev,
                            out_w,
                            out_norm,
                            hh,
                        })
                    } else {
                        None
                    };
                    Ok((out_l, mtp_l, head_w))
                }));
            }
            for h in handles {
                let (ls, ml, hw) = h.join().map_err(|_| anyhow::anyhow!("GPU loader thread panicked"))??;
                for (il, l) in ls {
                    layers[il] = Some(l);
                }
                if ml.is_some() {
                    mtp_layer = ml;
                }
                if hw.is_some() {
                    head_opt = hw;
                }
            }
            Ok(())
        })?;

        let mut layers: Vec<Hy4LayerWeights> = layers.into_iter().map(|x| x.unwrap()).collect();
        let head = head_opt.context("missing head weights")?;

        // Pack non-VRAM cold trunk experts into pinned RAM slabs (`[G | U | D]`) if MemAvailable permits,
        // eliminating SSD faults AND bounce-buffer memcpy on both prefill DMA and AVX-512 CPU lane!
        let mut total_cold_bytes = 0usize;
        let mut vram_resident_experts = 0usize;
        let mut total_trunk_routed_experts = 0usize;
        for lw in &layers {
            if let Some(ref m) = lw.moe {
                vram_resident_experts += m.n_res;
                total_trunk_routed_experts += cfg.n_expert;
                let n_cold = cfg.n_expert - m.n_res;
                total_cold_bytes += n_cold * m.blob_bytes;
            }
        }

        let mem_avail = host_mem_available_bytes();
        let want_pin = std::env::var("AMPLLM_HY4_PIN_COLD")
            .ok()
            .as_deref()
            != Some("0")
            && total_cold_bytes > 0
            && mem_avail >= total_cold_bytes.saturating_add(8usize << 30);

        let cold_storage = if want_pin {
            let mut slabs: Vec<RegisteredMappedHostBuffer> = Vec::with_capacity(n_layer);
            let mut failed_pin = false;
            for lw in &mut layers {
                let Some(ref mut m) = lw.moe else { continue };
                let cold_ids: Vec<usize> = (0..cfg.n_expert).filter(|&e| m.slot_of[e] < 0).collect();
                if cold_ids.is_empty() {
                    continue;
                }
                let slab_sz = cold_ids.len() * m.blob_bytes;
                let mut slab = match RegisteredMappedHostBuffer::alloc(slab_sz) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("Hy4: RegisteredMappedHostBuffer({} MiB) failed ({e}); keeping mmap pointers", slab_sz >> 20);
                        failed_pin = true;
                        break;
                    }
                };
                let slab_base = slab.as_mut_ptr() as usize;
                let g_raw_eb = m.gu_row_bytes * cfg.n_ff_exp;
                let d_raw_eb = m.dn_row_bytes * cfg.n_expert_latent;
                let sg_addr: Vec<usize> = m.cold_gate_ptrs.iter().map(|&p| p as usize).collect();
                let su_addr: Vec<usize> = m.cold_up_ptrs.iter().map(|&p| p as usize).collect();
                let sd_addr: Vec<usize> = m.cold_down_ptrs.iter().map(|&p| p as usize).collect();
                let blob_b = m.blob_bytes;
                let u_off = m.u_off;
                let d_off = m.d_off;
                pool.parallel_for(cold_ids.len(), |i0, i1| {
                    for idx in i0..i1 {
                        let eid = cold_ids[idx];
                        unsafe {
                            let dst = (slab_base as *mut u8).add(idx * blob_b);
                            std::ptr::copy_nonoverlapping(sg_addr[eid] as *const u8, dst, g_raw_eb);
                            std::ptr::copy_nonoverlapping(su_addr[eid] as *const u8, dst.add(u_off), g_raw_eb);
                            std::ptr::copy_nonoverlapping(sd_addr[eid] as *const u8, dst.add(d_off), d_raw_eb);
                        }
                    }
                });
                for (idx, &eid) in cold_ids.iter().enumerate() {
                    unsafe {
                        let base = slab.as_ptr().add(idx * m.blob_bytes);
                        m.cold_blob_ptrs[eid] = base;
                        m.cold_gate_ptrs[eid] = base;
                        m.cold_up_ptrs[eid] = base.add(m.u_off);
                        m.cold_down_ptrs[eid] = base.add(m.d_off);
                    }
                }
                m.cold_is_pinned = true;
                slabs.push(slab);
            }
            if failed_pin {
                for lw in &mut layers {
                    if let Some(ref mut m) = lw.moe {
                        m.cold_is_pinned = false;
                        m.cold_blob_ptrs.fill(std::ptr::null());
                    }
                }
                ColdExpertHostStorage::MmapDirect
            } else {
                tracing::info!(
                    "Hy4: packed {:.2} GB of non-VRAM cold experts into CUDA-registered pinned host slabs",
                    total_cold_bytes as f64 / 1e9
                );
                ColdExpertHostStorage::PinnedSlabs(slabs)
            }
        } else {
            ColdExpertHostStorage::MmapDirect
        };

        // Token embeddings & Engram table pointers in host mmap
        let te_info = gguf.tensor_info("token_embd.weight")?;
        let tok_type = te_info.ggml_type;
        let tok_row_bytes = tok_type.row_bytes(cfg.n_embd);
        let te_shard = te_info.shard_idx;
        let te_offset = te_info.data_offset;
        let te_len = te_info.byte_len;
        let tok_bytes = gguf.tensor_bytes("token_embd.weight")?;
        if tok_row_bytes * cfg.n_vocab > tok_bytes.len() {
            bail!("token_embd.weight byte length shorter than n_vocab * row_bytes");
        }
        let tok_embd_ptr = tok_bytes.as_ptr();

        let engram_table = if let Some(ref eg_cfg) = cfg.engram {
            let eg_info = gguf.tensor_info("per_layer_token_embd.weight")?;
            let eg_shard = eg_info.shard_idx;
            gguf.advise_shard_random(eg_shard);
            let eg_bytes = gguf.tensor_bytes("per_layer_token_embd.weight")?;
            Some(Hy4EngramTable::open(eg_cfg, eg_bytes.as_ptr(), eg_bytes.len())?)
        } else {
            None
        };

        // Once cold experts and MTP/GPU weights are pinned/uploaded, drop dead OS page cache
        // from the main weight shard (preserving token_embd) and the BF16 MTP shard so Engram stays warm in RAM.
        if matches!(cold_storage, ColdExpertHostStorage::PinnedSlabs(_)) {
            gguf.advise_shard_dontneed_except(te_shard, te_offset, te_len);
            if has_mtp {
                let mtp_shard = gguf
                    .tensor_info(&format!("blk.{n_layer}.attn_norm.weight"))
                    .map(|t| t.shard_idx)
                    .unwrap_or(te_shard);
                if mtp_shard != te_shard {
                    gguf.advise_shard_dontneed_except(mtp_shard, 0, 0);
                }
            }
        }

        let mut vram_used_mb = Vec::with_capacity(gpus.len());
        for (gi, &d) in gpus.iter().enumerate() {
            let (now_free, total_dev) = mem_info(d)?;
            let _ = total_dev;
            vram_used_mb.push(free_vram[gi].saturating_sub(now_free) >> 20);
        }

        Ok((
            Self {
                cfg,
                max_ctx,
                tc,
                swa_ring_cap,
                ivg_ring_cap,
                hring_cap,
                kv_is_q8,
                kv_rb,
                gpus,
                p2p_ok,
                split_layer,
                head_gpu_idx,
                layers,
                mtp_layer,
                head,
                tok_embd_ptr,
                tok_type,
                tok_row_bytes,
                engram_table,
                vision,
                _cold_storage: cold_storage,
                max_blob_bytes,
                vram_resident_experts,
                total_trunk_routed_experts,
                vram_used_mb,
            },
            gguf,
        ))
    }
}
