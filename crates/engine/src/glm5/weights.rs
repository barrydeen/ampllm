use ampllm_cache::CacheProfile;
use ampllm_cuda_kernels::{
    glm5, mem_info, set_device, CudaStream, DeviceBuffer, RegisteredMappedHostBuffer,
};
use ampllm_gguf::{GgmlType, GgufFile, Glm5Config};
use ampllm_tokenizer::BpeTokenizer;
use anyhow::{Context, Result};
use std::ffi::c_void;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub const K_SPARES: usize = 2;
const COLD_CHUNK_TARGET_BYTES: usize = 4usize << 30;
const BACKUP_CHUNK_SLOTS: usize = 512;
const TRUNK_BACKUP_SLOTS: usize = 640;
pub const LEND_RESERVED_TRUNK_SLOTS: usize = 240;
const MTP_BACKUP_SLOTS: usize = 32;
const BACKUP_MTP_TAG: usize = 1usize << 24;
const BACKUP_RECYCLED_TAG: usize = 1usize << 25;

fn advise_dontneed_range(base: usize, len: usize) {
    let page = 4096usize;
    let start = (base + page - 1) & !(page - 1);
    let end = (base + len) & !(page - 1);
    if end > start {
        unsafe {
            libc::madvise(start as *mut c_void, end - start, libc::MADV_DONTNEED);
        }
    }
}

fn fadvise_dontneed_file(
    shard_files: &[std::fs::File],
    shard_idx: usize,
    off: usize,
    len: usize,
) {
    if let Some(f) = shard_files.get(shard_idx) {
        let page = 4096usize;
        let s = (off + page - 1) & !(page - 1);
        let e = (off + len) & !(page - 1);
        if e > s {
            unsafe {
                libc::posix_fadvise(
                    f.as_raw_fd(),
                    s as libc::off_t,
                    (e - s) as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
    }
}

struct BackupSlotPool {
    mtp_layer: Option<usize>,
    trunk_ptrs: Vec<(usize, u64, u64)>, // (host_ptr, dev0_ptr, dev1_ptr)
    trunk_free: Vec<usize>,
    mtp_ptrs: Vec<(usize, u64, u64)>,
    mtp_free: Vec<usize>,
    recycled_entries: Vec<(usize, usize, u64, u64)>, // (layer, host_ptr, dev0_ptr, dev1_ptr)
    layer_free_recycled: Vec<Vec<usize>>,            // [n_tot] free indices into recycled_entries
    slot_of: Vec<Vec<i32>>,                          // [n_tot][n_expert]
}

fn return_slot_to_pool(pool: &mut BackupSlotPool, slot_id: usize) {
    let is_recycled = (slot_id & BACKUP_RECYCLED_TAG) != 0;
    let is_mtp = (slot_id & BACKUP_MTP_TAG) != 0;
    let idx = slot_id & !(BACKUP_MTP_TAG | BACKUP_RECYCLED_TAG);
    if is_recycled {
        let layer = pool.recycled_entries[idx].0;
        pool.layer_free_recycled[layer].push(idx);
    } else if is_mtp {
        pool.mtp_free.push(idx);
    } else {
        pool.trunk_free.push(idx);
    }
}

pub struct CpuColdArena {
    _chunks: Vec<RegisteredMappedHostBuffer>,
    pub layers_atomic: Arc<Vec<Vec<AtomicUsize>>>, // host base ptr to [gate | up | down]
    layers_dev: [Vec<Vec<AtomicU64>>; 2],          // GPU-mapped ptrs for dev 0 / dev 1
    pub layer_u_off: Vec<usize>,
    pub layer_d_off: Vec<usize>,
    pub layer_blob_stride: Vec<usize>,
    backup_pool: Mutex<BackupSlotPool>,
}
unsafe impl Send for CpuColdArena {}
unsafe impl Sync for CpuColdArena {}

impl CpuColdArena {
    #[inline]
    pub fn is_in_ram(&self, l: usize, e: usize) -> bool {
        self.layers_atomic[l][e].load(Ordering::Acquire) != 0
    }

    #[inline]
    pub fn get_ram_ptrs(&self, l: usize, e: usize) -> Option<(*const u8, *const u8, *const u8)> {
        let base = self.layers_atomic[l][e].load(Ordering::Acquire);
        if base == 0 {
            return None;
        }
        let u_off = self.layer_u_off[l];
        let d_off = self.layer_d_off[l];
        Some((
            base as *const u8,
            (base + u_off) as *const u8,
            (base + d_off) as *const u8,
        ))
    }

    #[inline]
    pub fn get_ram_blob(&self, l: usize, e: usize, dev: i32) -> Option<(*const u8, u64)> {
        let base = self.layers_atomic[l][e].load(Ordering::Acquire);
        if base == 0 {
            return None;
        }
        let dev_ptr = self.layers_dev[dev as usize][l][e].load(Ordering::Acquire);
        Some((base as *const u8, dev_ptr))
    }

    #[inline]
    pub fn get_dev_ptr(&self, l: usize, e: usize, dev: i32) -> u64 {
        self.layers_dev[dev as usize][l][e].load(Ordering::Acquire)
    }

    pub fn backup_free_count_for_layer(&self, l: usize) -> usize {
        let pool = self.backup_pool.lock().unwrap();
        let recycled = pool.layer_free_recycled.get(l).map_or(0, Vec::len);
        if pool.mtp_layer == Some(l) {
            recycled + pool.mtp_free.len()
        } else {
            recycled + pool.trunk_free.len().saturating_sub(LEND_RESERVED_TRUNK_SLOTS)
        }
    }

    /// Reserve a pre-allocated pinned-mapped backup slot for an evicted VRAM expert (zero runtime alloc).
    /// Checks per-layer recycled slots first, then falls back to shared MTP/trunk backup buffers.
    pub fn reserve_backup_slot(&self, l: usize, e: usize) -> Option<(usize, *mut u8, [u64; 2])> {
        let mut pool = self.backup_pool.lock().unwrap();
        if self.layers_atomic[l][e].load(Ordering::Acquire) != 0 || pool.slot_of[l][e] >= 0 {
            return None;
        }
        if let Some(idx) = pool.layer_free_recycled.get_mut(l).and_then(Vec::pop) {
            let (_, hp, d0, d1) = pool.recycled_entries[idx];
            return Some((idx | BACKUP_RECYCLED_TAG, hp as *mut u8, [d0, d1]));
        }
        if pool.mtp_layer == Some(l) {
            let idx = pool.mtp_free.pop()?;
            let (hp, d0, d1) = pool.mtp_ptrs[idx];
            Some((idx | BACKUP_MTP_TAG, hp as *mut u8, [d0, d1]))
        } else {
            if pool.trunk_free.len() <= LEND_RESERVED_TRUNK_SLOTS {
                return None;
            }
            let idx = pool.trunk_free.pop()?;
            let (hp, d0, d1) = pool.trunk_ptrs[idx];
            Some((idx, hp as *mut u8, [d0, d1]))
        }
    }

    /// Reserve a pre-allocated backup slot for temporary prefill pool lending (can draw from lend-reserved slots).
    pub fn reserve_lend_backup_slot(&self, l: usize, e: usize) -> Option<(usize, *mut u8, [u64; 2])> {
        let mut pool = self.backup_pool.lock().unwrap();
        if self.layers_atomic[l][e].load(Ordering::Acquire) != 0 || pool.slot_of[l][e] >= 0 {
            return None;
        }
        if pool.mtp_layer == Some(l) {
            let idx = pool.mtp_free.pop()?;
            let (hp, d0, d1) = pool.mtp_ptrs[idx];
            Some((idx | BACKUP_MTP_TAG, hp as *mut u8, [d0, d1]))
        } else {
            let idx = pool.trunk_free.pop()?;
            let (hp, d0, d1) = pool.trunk_ptrs[idx];
            Some((idx, hp as *mut u8, [d0, d1]))
        }
    }

    /// Publish a D2H-populated backup slot cleanly for both CPU workers and GPU zero-copy fetch.
    pub fn commit_backup_slot(&self, l: usize, e: usize, slot_id: usize) -> [u64; 2] {
        let mut pool = self.backup_pool.lock().unwrap();
        if pool.slot_of[l][e] >= 0 || self.layers_atomic[l][e].load(Ordering::Acquire) != 0 {
            return_slot_to_pool(&mut pool, slot_id);
            return [
                self.layers_dev[0][l][e].load(Ordering::Acquire),
                self.layers_dev[1][l][e].load(Ordering::Acquire),
            ];
        }
        pool.slot_of[l][e] = slot_id as i32;
        let is_recycled = (slot_id & BACKUP_RECYCLED_TAG) != 0;
        let is_mtp = (slot_id & BACKUP_MTP_TAG) != 0;
        let idx = slot_id & !(BACKUP_MTP_TAG | BACKUP_RECYCLED_TAG);
        let (hp, d0, d1) = if is_recycled {
            let (_, hp, d0, d1) = pool.recycled_entries[idx];
            (hp, d0, d1)
        } else if is_mtp {
            pool.mtp_ptrs[idx]
        } else {
            pool.trunk_ptrs[idx]
        };
        self.layers_dev[0][l][e].store(d0, Ordering::Release);
        self.layers_dev[1][l][e].store(d1, Ordering::Release);
        self.layers_atomic[l][e].store(hp, Ordering::Release);
        [d0, d1]
    }

    pub fn release_temp_backup_slot(&self, l: usize, e: usize, slot_id: usize) {
        let mut pool = self.backup_pool.lock().unwrap();
        if pool.slot_of[l][e] != slot_id as i32 {
            return;
        }
        pool.slot_of[l][e] = -1;
        self.layers_atomic[l][e].store(0, Ordering::Release);
        self.layers_dev[0][l][e].store(0, Ordering::Release);
        self.layers_dev[1][l][e].store(0, Ordering::Release);
        return_slot_to_pool(&mut pool, slot_id);
    }

    pub fn cancel_backup_slot(&self, slot_id: usize) {
        let mut pool = self.backup_pool.lock().unwrap();
        return_slot_to_pool(&mut pool, slot_id);
    }

    /// Release a RAM slot belonging to a promoted expert into the per-layer free pool or backup free list.
    pub fn release_ram_slot(&self, l: usize, e: usize) -> bool {
        let mut pool = self.backup_pool.lock().unwrap();
        let hp = self.layers_atomic[l][e].load(Ordering::Acquire);
        if hp == 0 {
            return false;
        }
        let d0 = self.layers_dev[0][l][e].load(Ordering::Acquire);
        let d1 = self.layers_dev[1][l][e].load(Ordering::Acquire);
        self.layers_atomic[l][e].store(0, Ordering::Release);
        self.layers_dev[0][l][e].store(0, Ordering::Release);
        self.layers_dev[1][l][e].store(0, Ordering::Release);
        let prev = pool.slot_of[l][e];
        pool.slot_of[l][e] = -1;
        if prev >= 0 {
            return_slot_to_pool(&mut pool, prev as usize);
        } else {
            let idx = pool.recycled_entries.len();
            pool.recycled_entries.push((l, hp, d0, d1));
            pool.layer_free_recycled[l].push(idx);
        }
        true
    }
}

#[derive(Clone, Copy)]
pub struct DevWeight {
    pub ptr: u64,
    pub ggml_type: GgmlType,
    pub n_bytes: usize,
    pub dims: [usize; 3],
}

pub struct Glm5LayerWeights {
    pub layer_idx: usize,
    pub device: i32,
    pub is_kda: bool,
    pub is_moe: bool,
    pub is_mtp: bool,
    // Keep ownership of device allocations for this layer
    pub _bufs: Vec<DeviceBuffer>,
    // HyperConnections (None for blk.45 MTP)
    pub hc_attn_fn: Option<DevWeight>,
    pub hc_attn_scale: Option<u64>,
    pub hc_attn_base: Option<u64>,
    pub hc_ffn_fn: Option<DevWeight>,
    pub hc_ffn_scale: Option<u64>,
    pub hc_ffn_base: Option<u64>,
    pub attn_norm: u64,
    pub ffn_norm: u64,
    // KDA dense weights
    pub attn_q: Option<DevWeight>,
    pub attn_k: Option<DevWeight>,
    pub attn_v: Option<DevWeight>,
    pub ssm_a: Option<u64>,
    pub ssm_dt_bias: Option<u64>,
    pub ssm_beta: Option<DevWeight>,
    pub ssm_conv_q: Option<u64>,
    pub ssm_conv_k: Option<u64>,
    pub ssm_conv_v: Option<u64>,
    pub ssm_f_a: Option<DevWeight>,
    pub ssm_f_b: Option<DevWeight>,
    pub ssm_g_a: Option<DevWeight>,
    pub ssm_g_b: Option<DevWeight>,
    pub ssm_norm: Option<u64>,
    // DSA dense weights
    pub attn_q_a: Option<DevWeight>,
    pub attn_q_a_norm: Option<u64>,
    pub attn_q_b: Option<DevWeight>,
    pub attn_kv_a_mqa: Option<DevWeight>,
    pub attn_kv_a_norm: Option<u64>,
    pub attn_k_b_f16: Option<u64>,
    pub attn_v_b_f16: Option<u64>,
    pub idx_k: Option<DevWeight>,
    pub idx_q_b: Option<DevWeight>,
    pub idx_k_norm_w: Option<u64>,
    pub idx_k_norm_b: Option<u64>,
    pub idx_proj: Option<DevWeight>,
    pub idx_ape: Option<u64>,
    pub idx_gate: Option<DevWeight>,
    // Shared attn_output
    pub attn_output: DevWeight,
    // Dense FFN (L0..2)
    pub ffn_gate: Option<DevWeight>,
    pub ffn_up: Option<DevWeight>,
    pub ffn_down: Option<DevWeight>,
    // MoE router + shared + contiguous routed blob pool
    pub ffn_gate_inp: Option<DevWeight>,
    pub exp_probs_bias: Option<u64>,
    pub ffn_gate_sh: Option<DevWeight>,
    pub ffn_up_sh: Option<DevWeight>,
    pub ffn_down_sh: Option<DevWeight>,
    pub gu_type: GgmlType,
    pub d_type: GgmlType,
    pub vram_ptrs_dev: Option<DeviceBuffer>,
    pub ram_ptrs_dev: Option<DeviceBuffer>,
    pub spares_dev: Option<DeviceBuffer>,
    pub dcnt_dev: Option<DeviceBuffer>,
    pub vram_ptrs_host: Vec<u64>,
    pub res_mask_host: Vec<u8>,
    pub exp_pool_ptr: u64,
    pub g_eb: usize,
    pub d_eb: usize,
    pub u_off: usize,
    pub d_off: usize,
    pub blob_stride: usize,
    pub k_spares: usize,
    pub initial_spare_ptrs: Vec<u64>,
    pub initial_slot_experts: Vec<usize>,
    pub initial_heat: Vec<f32>,
    // MTP extras (L45)
    pub nextn_eh_proj: Option<DevWeight>,
    pub nextn_enorm: Option<u64>,
    pub nextn_hnorm: Option<u64>,
    pub nextn_shnorm: Option<u64>,
}

pub struct DraftVocab {
    pub n_lead: usize,
    pub extra_ids: Vec<i32>,
    pub extra_ids_dev: Option<DeviceBuffer>,
    pub extra_w_dev: Option<DeviceBuffer>,
    pub in_draft: Vec<bool>,
}

pub struct Glm5ModelWeights {
    pub cfg: Glm5Config,
    pub split_layer: usize, // GPU 0 owns [0..split_layer), GPU 1 owns [split_layer..n_layers) + L45 MTP
    pub layers: Vec<Glm5LayerWeights>,
    pub exp_pool_dev: [DeviceBuffer; 2], // Contiguous per-GPU expert pools (MTP lowest on GPU 1)
    pub max_trunk_blob_stride: usize,
    pub dsa_res_cap: usize,
    pub pool_cap: usize,
    // Final output head on GPU 1
    pub output_norm_dev: DeviceBuffer,
    pub output_weight_dev: DeviceBuffer,
    pub output_weight_type: GgmlType,
    // Host-resident token_embd Q6_K slice in mmap'd GGUF (borrowed via GgufFile lifetime stored on engine)
    pub token_embd_type: GgmlType,
    pub token_embd_row_bytes: usize,
    // Cold experts on host RAM
    pub cold_arena: CpuColdArena,
    // Optional pruned draft vocabulary for fast blk.45 MTP drafting
    pub draft_vocab: DraftVocab,
    // Stats
    pub has_profile: bool,
    pub vram_resident_experts: usize,
    pub total_routed_experts: usize,
    pub vram_used_gpu0_mb: usize,
    pub vram_used_gpu1_mb: usize,
}

fn upload_tensor_f32_or_raw(
    gguf: &GgufFile,
    name: &str,
    dev: i32,
    stream: &CudaStream,
    bufs: &mut Vec<DeviceBuffer>,
) -> Result<DevWeight> {
    let info = gguf.tensor_info(name).with_context(|| format!("missing tensor {name}"))?;
    let bytes = gguf.tensor_bytes(name)?;
    let buf = DeviceBuffer::alloc(dev, bytes.len())?;
    buf.copy_from_host_async(bytes, stream)?;
    let ptr = buf.as_u64();
    let mut dims = [1usize; 3];
    for (i, &d) in info.dims.iter().take(3).enumerate() {
        dims[i] = d as usize;
    }
    let res = DevWeight {
        ptr,
        ggml_type: info.ggml_type,
        n_bytes: bytes.len(),
        dims,
    };
    bufs.push(buf);
    Ok(res)
}

#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((u >> 16) | 0x0040) as u16;
    }
    let bias = 0x7FFFu32.wrapping_add((u >> 16) & 1);
    (u.wrapping_add(bias) >> 16) as u16
}

fn upload_f32_to_bf16(
    gguf: &GgufFile,
    name: &str,
    dev: i32,
    stream: &CudaStream,
    bufs: &mut Vec<DeviceBuffer>,
) -> Result<DevWeight> {
    let info = gguf.tensor_info(name).with_context(|| format!("missing tensor {name}"))?;
    if info.ggml_type != GgmlType::F32 {
        return upload_tensor_f32_or_raw(gguf, name, dev, stream, bufs);
    }
    let raw = gguf.tensor_bytes(name)?;
    let floats: &[f32] = bytemuck::cast_slice(raw);
    let mut bf16_bits = Vec::with_capacity(floats.len());
    for &v in floats {
        bf16_bits.push(f32_to_bf16_bits(v));
    }
    let bytes: &[u8] = bytemuck::cast_slice(&bf16_bits);
    let buf = DeviceBuffer::alloc(dev, bytes.len())?;
    buf.copy_from_host_async(bytes, stream)?;
    let ptr = buf.as_u64();
    let mut dims = [1usize; 3];
    for (i, &d) in info.dims.iter().take(3).enumerate() {
        dims[i] = d as usize;
    }
    let res = DevWeight {
        ptr,
        ggml_type: GgmlType::Bf16,
        n_bytes: bytes.len(),
        dims,
    };
    bufs.push(buf);
    Ok(res)
}

fn upload_q6_to_f16(
    gguf: &GgufFile,
    name: &str,
    dev: i32,
    stream: &CudaStream,
    bufs: &mut Vec<DeviceBuffer>,
) -> Result<u64> {
    let info = gguf.tensor_info(name).with_context(|| format!("missing tensor {name}"))?;
    let bytes = gguf.tensor_bytes(name)?;
    let n_elems = info.n_elements();
    let tmp = DeviceBuffer::alloc(dev, bytes.len())?;
    tmp.copy_from_host_async(bytes, stream)?;
    let out = DeviceBuffer::alloc(dev, n_elems * 2)?;
    unsafe {
        glm5::dequant_to_f16(info.ggml_type as i32, tmp.ptr, out.ptr, n_elems as i64, stream);
    }
    stream.sync()?;
    let ptr = out.as_u64();
    bufs.push(out);
    Ok(ptr)
}

fn token_kind_glm(s: &str) -> (bool, bool) {
    if s.is_empty() {
        return (false, false);
    }
    let mut ascii = true;
    let mut digits = true;
    for ch in s.chars() {
        let cp = ch as u32;
        if !((0x21..=0x7E).contains(&cp) || (0x100..=0x121).contains(&cp)) {
            ascii = false;
        }
        if !ch.is_ascii_digit() {
            digits = false;
        }
    }
    (ascii, digits)
}

fn build_draft_vocab(
    gguf: &GgufFile,
    tok: &BpeTokenizer,
    cfg: &Glm5Config,
    output_weight_dev: &DeviceBuffer,
    dev: i32,
    stream: &CudaStream,
) -> Result<DraftVocab> {
    let vocab = tok.vocab();
    let ttypes = tok.token_types();
    let n_vocab = cfg.n_vocab;
    if vocab.len() < n_vocab || n_vocab < 4096 {
        tracing::info!("GLM-5.3 draft_vocab: disabled (vocab={}, n_vocab={n_vocab})", vocab.len());
        return Ok(DraftVocab {
            n_lead: n_vocab,
            extra_ids: Vec::new(),
            extra_ids_dev: None,
            extra_w_dev: None,
            in_draft: vec![true; n_vocab],
        });
    }

    let mut n_lead = 0usize;
    let mut chunk_start = 0usize;
    while chunk_start + 1024 <= n_vocab {
        let ascii_cnt = (chunk_start..chunk_start + 1024)
            .filter(|&i| token_kind_glm(&vocab[i]).0)
            .count();
        if ascii_cnt < 512 {
            n_lead = chunk_start;
            while n_lead < n_vocab && token_kind_glm(&vocab[n_lead]).0 {
                n_lead += 1;
            }
            break;
        }
        chunk_start += 1024;
    }
    if n_lead == 0 || n_lead * 8 < n_vocab {
        tracing::info!("GLM-5.3 draft_vocab: disabled (n_lead={n_lead}/{n_vocab})");
        return Ok(DraftVocab {
            n_lead: n_vocab,
            extra_ids: Vec::new(),
            extra_ids_dev: None,
            extra_w_dev: None,
            in_draft: vec![true; n_vocab],
        });
    }

    let mut extra_ids = Vec::new();
    for id in n_lead..n_vocab {
        let (_, digits) = token_kind_glm(&vocab[id]);
        let ty = ttypes.get(id).copied().unwrap_or(1);
        if digits || ty == 3 || ty == 4 {
            extra_ids.push(id as i32);
        }
    }
    let total_draft = n_lead + extra_ids.len();
    if total_draft * 10 > n_vocab * 9 {
        tracing::info!("GLM-5.3 draft_vocab: disabled (total={total_draft}/{n_vocab})");
        return Ok(DraftVocab {
            n_lead: n_vocab,
            extra_ids: Vec::new(),
            extra_ids_dev: None,
            extra_w_dev: None,
            in_draft: vec![true; n_vocab],
        });
    }

    let mut in_draft = vec![false; n_vocab];
    for i in 0..n_lead {
        in_draft[i] = true;
    }
    for &eid in &extra_ids {
        in_draft[eid as usize] = true;
    }

    let out_info = gguf.tensor_info("output.weight")?;
    let out_bytes = gguf.tensor_bytes("output.weight")?;
    let rb = out_info.ggml_type.row_bytes(cfg.n_embd);

    let (extra_ids_dev, extra_w_dev) = if !extra_ids.is_empty() {
        let mut host_rows = Vec::with_capacity(extra_ids.len() * rb);
        for &eid in &extra_ids {
            let off = (eid as usize) * rb;
            host_rows.extend_from_slice(&out_bytes[off..off + rb]);
        }
        let ids_d = DeviceBuffer::alloc(dev, extra_ids.len() * 4)?;
        ids_d.copy_from_host_async(bytemuck::cast_slice(&extra_ids), stream)?;
        let w_d = DeviceBuffer::alloc(dev, host_rows.len())?;
        w_d.copy_from_host_async(&host_rows, stream)?;
        (Some(ids_d), Some(w_d))
    } else {
        (None, None)
    };
    let _ = output_weight_dev;
    tracing::info!(
        "GLM-5.3 draft_vocab: n_lead={}, extra={}, total={}/{}",
        n_lead,
        extra_ids.len(),
        total_draft,
        n_vocab
    );

    Ok(DraftVocab {
        n_lead,
        extra_ids,
        extra_ids_dev,
        extra_w_dev,
        in_draft,
    })
}

impl Glm5ModelWeights {
    pub fn load(
        gguf: &GgufFile,
        tok: &BpeTokenizer,
        cfg: &Glm5Config,
        max_ctx: usize,
        profile: Option<&CacheProfile>,
        max_vram_experts_override: Option<usize>,
        extra_reserve0_bytes: usize,
    ) -> Result<Self> {
        let n_dev = ampllm_cuda_kernels::device_count()?;
        if n_dev < 2 {
            anyhow::bail!("GLM-5.3-Flash currently requires at least 2 CUDA GPUs (detected {n_dev})");
        }
        eprintln!(
            "glm5 CPU MoE offload ISA: {}",
            ampllm_cpu_kernels::active_cpu_isa()
        );
        let s0 = CudaStream::new(0)?;
        let s1 = CudaStream::new(1)?;

        let (free0, _) = mem_info(0)?;
        let (free1, _) = mem_info(1)?;

        // Determine per-layer fixed dense + state VRAM cost
        let n_tot = cfg.n_layer_total;
        let k_spares = std::env::var("AMPLLM_K_SPARES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|k| k.min(8))
            .unwrap_or(K_SPARES);
        let env_dsa_res_cap = std::env::var("AMPLLM_DSA_RES_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(24576)
            .clamp(4096usize.min(max_ctx), max_ctx);
        let dsa_res_cap = max_ctx.min(env_dsa_res_cap);
        let pool_cap = (max_ctx + cfg.idx_kpool - 1) / cfg.idx_kpool;

        let mut layer_fixed_bytes = vec![0usize; n_tot];
        let mut layer_snap_bytes = vec![0usize; n_tot];
        let mut layer_exp_bytes = vec![0usize; n_tot];
        let mut layer_u_off = vec![0usize; n_tot];
        let mut layer_d_off = vec![0usize; n_tot];
        let mut layer_blob_stride = vec![0usize; n_tot];
        let mut layer_gu_types = vec![GgmlType::Iq2S; n_tot];
        let mut layer_d_types = vec![GgmlType::Iq3Xxs; n_tot];

        for l in 0..n_tot {
            let is_mtp = cfg.mtp_layer == Some(l);
            let kda = !is_mtp && cfg.is_kda(l);
            let moe = is_mtp || cfg.is_moe(l);
            let mut fix = 0usize;
            if !is_mtp {
                for suf in [
                    "hc_attn_fn.weight",
                    "hc_attn_scale.weight",
                    "hc_attn_base.weight",
                    "hc_ffn_fn.weight",
                    "hc_ffn_scale.weight",
                    "hc_ffn_base.weight",
                ] {
                    let tname = format!("blk.{l}.{suf}");
                    let info = gguf.tensor_info(&tname)?;
                    if (suf == "hc_attn_fn.weight" || suf == "hc_ffn_fn.weight")
                        && info.ggml_type == GgmlType::F32
                    {
                        fix += info.n_elements() * 2;
                    } else {
                        fix += gguf.tensor_bytes(&tname)?.len();
                    }
                }
            }
            for suf in ["attn_norm.weight", "ffn_norm.weight", "attn_output.weight"] {
                fix += gguf.tensor_bytes(&format!("blk.{l}.{suf}"))?.len();
            }

            if kda {
                for suf in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "ssm_a",
                    "ssm_dt.bias",
                    "ssm_beta.weight",
                    "ssm_conv1d_q.weight",
                    "ssm_conv1d_k.weight",
                    "ssm_conv1d_v.weight",
                    "ssm_f_a.weight",
                    "ssm_f_b.weight",
                    "ssm_g_a.weight",
                    "ssm_g_b.weight",
                    "ssm_norm.weight",
                ] {
                    fix += gguf.tensor_bytes(&format!("blk.{l}.{suf}"))?.len();
                }
                let kda_state = cfg.n_head * cfg.kda_head_dim * cfg.kda_head_dim * 4
                    + 3 * cfg.d_inner() * (cfg.d_conv - 1) * 4;
                fix += kda_state;
                layer_snap_bytes[l] = kda_state;
            } else {
                // DSA
                for suf in [
                    "attn_q_a.weight",
                    "attn_q_a_norm.weight",
                    "attn_q_b.weight",
                    "attn_kv_a_mqa.weight",
                    "attn_kv_a_norm.weight",
                    "indexer.attn_k.weight",
                    "indexer.attn_q_b.weight",
                    "indexer.k_norm.weight",
                    "indexer.k_norm.bias",
                    "indexer.proj.weight",
                    "indexer_compressor_ape.weight",
                    "indexer_compressor_gate.weight",
                ] {
                    fix += gguf.tensor_bytes(&format!("blk.{l}.{suf}"))?.len();
                }
                // F16 wk_b + wv_b
                fix += 2 * cfg.n_head * cfg.kv_lora_rank * cfg.qk_nope_head_dim * 2;
                // pooled FP16 cache + resident lat8 window + ik/ig rings
                fix += pool_cap * cfg.idx_key_dim * 2;
                fix += dsa_res_cap * (cfg.kv_lora_rank + cfg.kv_lora_rank / 16);
                fix += 2 * 516 * cfg.idx_key_dim * 4;
            }
            if !moe {
                for suf in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    fix += gguf.tensor_bytes(&format!("blk.{l}.{suf}"))?.len();
                }
            } else {
                for suf in [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                ] {
                    fix += gguf.tensor_bytes(&format!("blk.{l}.{suf}"))?.len();
                }
                let g_t = gguf.tensor_info(&format!("blk.{l}.ffn_gate_exps.weight"))?.ggml_type;
                let d_t = gguf.tensor_info(&format!("blk.{l}.ffn_down_exps.weight"))?.ggml_type;
                layer_gu_types[l] = g_t;
                layer_d_types[l] = d_t;
                let g_b = g_t.row_bytes(cfg.n_embd) * cfg.n_ff_exp;
                let d_b = d_t.row_bytes(cfg.n_ff_exp) * cfg.n_embd;
                let u_off = (g_b + 15) & !15;
                let d_off = (u_off + g_b + 15) & !15;
                let blob_stride = (d_off + d_b + 15) & !15;
                layer_u_off[l] = u_off;
                layer_d_off[l] = d_off;
                layer_blob_stride[l] = blob_stride;
                layer_exp_bytes[l] = blob_stride;
                fix += k_spares * blob_stride;
            }
            if is_mtp {
                for suf in [
                    "nextn.eh_proj.weight",
                    "nextn.enorm.weight",
                    "nextn.hnorm.weight",
                    "nextn.shared_head_norm.weight",
                ] {
                    fix += gguf.tensor_bytes(&format!("blk.{l}.{suf}"))?.len();
                }
            }
            fix += 2 * 1024 * 1024; // alignment / ptr arrays cushion
            layer_fixed_bytes[l] = fix;
        }

        let head_bytes = gguf.tensor_bytes("output.weight")?.len()
            + gguf.tensor_bytes("output_norm.weight")?.len()
            + 128 * 1024 * 1024; // includes draft_vocab extra rows + logits_dev + mtp scratch
        // Reserve 850 MiB on GPU 0 (headroom for dynamic carveout/scratch),
        // 1384 MiB on GPU 1 (Hyprland/desktop + shrunk phase_arena + GpuScratch + >= 768 MiB free floor)
        let reserve0_mb = std::env::var("AMPLLM_RESERVE0_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(850);
        let reserve1_mb = std::env::var("AMPLLM_RESERVE1_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1384);
        let reserve0 = (reserve0_mb << 20).saturating_add(extra_reserve0_bytes);
        let reserve1 = reserve1_mb << 20;
        let avail0 = free0.saturating_sub(reserve0);
        let avail1 = free1.saturating_sub(reserve1 + head_bytes);

        let env_split = std::env::var("AMPLLM_SPLIT_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&sp| sp >= 10 && sp < cfg.n_layers);

        let mut best_split = cfg.n_layers / 2;
        if let Some(sp) = env_split {
            best_split = sp;
        } else {
            let mut best_bottleneck = f64::INFINITY;
            for sp in 18..=28 {
                let fix0: usize = layer_fixed_bytes[..sp].iter().sum::<usize>()
                    + layer_snap_bytes[..sp].iter().sum::<usize>();
                let mut fix1: usize = layer_fixed_bytes[sp..cfg.n_layers].iter().sum();
                if let Some(mtp) = cfg.mtp_layer {
                    fix1 += layer_fixed_bytes[mtp];
                }
                if fix0 >= avail0 || fix1 >= avail1 {
                    continue;
                }
                let r0 = avail0 - fix0;
                let r1 = avail1 - fix1;
                let n_moe0 = sp.saturating_sub(cfg.leading_dense_block_count);
                let n_moe1 = cfg.n_layers.saturating_sub(sp);
                let avg_exp0 = layer_exp_bytes[cfg.leading_dense_block_count..sp]
                    .iter()
                    .sum::<usize>()
                    / n_moe0.max(1);
                let avg_exp1 = layer_exp_bytes[sp..cfg.n_layers]
                    .iter()
                    .sum::<usize>()
                    / n_moe1.max(1);
                let n0 = (r0 / avg_exp0.max(1)).min(n_moe0 * cfg.n_expert);
                let n1 = (r1 / avg_exp1.max(1))
                    .min((n_moe1 + usize::from(cfg.has_mtp)) * cfg.n_expert);
                let frac0 = (n0 as f64) / (n_moe0.max(1) * cfg.n_expert).max(1) as f64;
                let frac1 = (n1 as f64)
                    / ((n_moe1 + usize::from(cfg.has_mtp)) * cfg.n_expert).max(1) as f64;
                let miss_est = |f: f64| (1.0 - (0.48 + 0.72 * f).min(0.95)).max(0.05);
                let n_dense0 = cfg.leading_dense_block_count.min(sp);
                let cost0 = 1.15 * ((n_dense0 as f64) * 0.50
                    + (n_moe0 as f64) * (0.30 + 1.85 * miss_est(frac0)));
                let cost1 = (n_moe1 as f64) * (0.30 + 1.85 * miss_est(frac1)) + 0.65;
                let bottleneck = cost0.max(cost1) + 0.03 * (cost0 - cost1).abs();
                if bottleneck < best_bottleneck {
                    best_bottleneck = bottleneck;
                    best_split = sp;
                }
            }
        }
        let split_layer = best_split;

        let fix0: usize = layer_fixed_bytes[..split_layer].iter().sum::<usize>()
            + layer_snap_bytes[..split_layer].iter().sum::<usize>();
        let mut fix1: usize = layer_fixed_bytes[split_layer..cfg.n_layers].iter().sum();
        if let Some(mtp) = cfg.mtp_layer {
            fix1 += layer_fixed_bytes[mtp];
        }
        let mut exp_budget0 = avail0.saturating_sub(fix0);
        let mut exp_budget1 = avail1.saturating_sub(fix1);
        let init_exp_budget0_mb = exp_budget0 >> 20;
        let init_exp_budget1_mb = exp_budget1 >> 20;
        tracing::info!(
            "GLM-5.3 plan: free0={} MiB avail0={} MiB fix0={} MiB exp_budget0={} MiB vis_res0={} MiB | free1={} MiB avail1={} MiB fix1={} MiB exp_budget1={} MiB",
            free0 >> 20, avail0 >> 20, fix0 >> 20, init_exp_budget0_mb, extra_reserve0_bytes >> 20,
            free1 >> 20, avail1 >> 20, fix1 >> 20, init_exp_budget1_mb
        );

        // Select which (layer, expert) pairs reside in GPU 0 vs GPU 1 VRAM
        let mut is_vram = vec![vec![false; cfg.n_expert]; n_tot];
        let mut order: Vec<(usize, usize)> = Vec::with_capacity(n_tot * cfg.n_expert);
        if let Some(prof) = profile {
            if !prof.global_ranked.is_empty() {
                for &(l, e) in &prof.global_ranked {
                    if l < n_tot && e < cfg.n_expert && layer_exp_bytes[l] > 0 {
                        order.push((l, e));
                    }
                }
            }
            // Append any unranked pairs (e.g. L45 if absent from profile)
            let mut seen = vec![vec![false; cfg.n_expert]; n_tot];
            for &(l, e) in &order {
                seen[l][e] = true;
            }
            for e in 0..cfg.n_expert {
                for l in 0..n_tot {
                    if layer_exp_bytes[l] > 0 && !seen[l][e] {
                        let ee = prof
                            .ranked_experts
                            .get(l)
                            .and_then(|v| v.get(e).copied())
                            .unwrap_or(e);
                        if !seen[l][ee] {
                            order.push((l, ee));
                            seen[l][ee] = true;
                        }
                    }
                }
            }
        } else {
            for e in 0..cfg.n_expert {
                for l in 0..n_tot {
                    if layer_exp_bytes[l] > 0 {
                        order.push((l, e));
                    }
                }
            }
        }

        let max_vram_limit = max_vram_experts_override.unwrap_or(usize::MAX);
        let mut vram_count = 0usize;
        let mut mtp_vram_count = 0usize;
        let mtp_vram_limit = std::env::var("AMPLLM_MTP_VRAM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(40)
            .min(cfg.n_expert);
        if let Some(mtp_l) = cfg.mtp_layer {
            let sz = layer_exp_bytes[mtp_l];
            if sz > 0 {
                let mtp_iter: Box<dyn Iterator<Item = usize>> = if let Some(ranked) =
                    profile.and_then(|p| p.ranked_experts.get(mtp_l))
                {
                    Box::new(ranked.iter().copied())
                } else {
                    Box::new(0..cfg.n_expert)
                };
                for e in mtp_iter {
                    if mtp_vram_count >= mtp_vram_limit || vram_count >= max_vram_limit || exp_budget1 < sz {
                        break;
                    }
                    if e < cfg.n_expert && !is_vram[mtp_l][e] {
                        exp_budget1 -= sz;
                        is_vram[mtp_l][e] = true;
                        vram_count += 1;
                        mtp_vram_count += 1;
                    }
                }
            }
        }
        for (l, e) in order {
            if vram_count >= max_vram_limit {
                break;
            }
            let sz = layer_exp_bytes[l];
            if sz == 0 || is_vram[l][e] {
                continue;
            }
            let on_gpu0 = l < split_layer;
            if on_gpu0 {
                if exp_budget0 >= sz {
                    exp_budget0 -= sz;
                    is_vram[l][e] = true;
                    vram_count += 1;
                }
            } else if exp_budget1 >= sz {
                exp_budget1 -= sz;
                is_vram[l][e] = true;
                vram_count += 1;
            }
        }

        tracing::info!(
            "GLM-5.3 dual-GPU split at L{split_layer} (GPU0=[0..{split_layer}), GPU1=[{split_layer}..44]+L45) | VRAM experts={vram_count} (L45 MTP={mtp_vram_count})"
        );

        // Precompute per-layer resident experts + spares and allocate ONE contiguous exp_pool per GPU.
        // GPU 1 places blk.45 (MTP) FIRST (lowest offset, never lent) so trunk layers sit at the lendable tail.
        let mut layer_res_ids: Vec<Vec<usize>> = vec![Vec::new(); n_tot];
        let mut layer_k_spares: Vec<usize> = vec![0usize; n_tot];
        for l in 0..n_tot {
            let is_mtp = cfg.mtp_layer == Some(l);
            let is_moe = is_mtp || cfg.is_moe(l);
            if !is_moe {
                continue;
            }
            for e in 0..cfg.n_expert {
                if is_vram[l][e] {
                    layer_res_ids[l].push(e);
                }
            }
            if layer_res_ids[l].len() < cfg.n_expert {
                layer_k_spares[l] = k_spares;
            }
        }

        let mut layer_pool_off = vec![0usize; n_tot];
        let mut pool_bytes = [0usize; 2];
        let align256 = |x: usize| (x + 255) & !255;
        for l in 0..split_layer.min(n_tot) {
            let total_slots = layer_res_ids[l].len() + layer_k_spares[l];
            if total_slots > 0 && layer_blob_stride[l] > 0 {
                layer_pool_off[l] = pool_bytes[0];
                pool_bytes[0] += align256(total_slots * layer_blob_stride[l]);
            }
        }
        if let Some(mtp_l) = cfg.mtp_layer {
            if mtp_l >= split_layer && mtp_l < n_tot {
                let total_slots = layer_res_ids[mtp_l].len() + layer_k_spares[mtp_l];
                if total_slots > 0 && layer_blob_stride[mtp_l] > 0 {
                    layer_pool_off[mtp_l] = pool_bytes[1];
                    pool_bytes[1] += align256(total_slots * layer_blob_stride[mtp_l]);
                }
            }
        }
        for l in split_layer..cfg.n_layers.min(n_tot) {
            let total_slots = layer_res_ids[l].len() + layer_k_spares[l];
            if total_slots > 0 && layer_blob_stride[l] > 0 {
                layer_pool_off[l] = pool_bytes[1];
                pool_bytes[1] += align256(total_slots * layer_blob_stride[l]);
            }
        }
        let exp_pool_dev = [
            DeviceBuffer::alloc(0, pool_bytes[0])?,
            DeviceBuffer::alloc(1, pool_bytes[1])?,
        ];
        let pool_base_addr = [exp_pool_dev[0].as_u64(), exp_pool_dev[1].as_u64()];
        let max_trunk_blob_stride = (0..cfg.n_layers.min(n_tot))
            .map(|l| layer_blob_stride[l])
            .max()
            .unwrap_or(0);

        // Load dense weights and upload VRAM experts per layer
        let mut layers = Vec::with_capacity(n_tot);
        let mut total_routed_experts = 0usize;
        let mut dev_dense_bytes = [0usize; 2];
        let mut dev_exp_bytes = [0usize; 2];

        for l in 0..n_tot {
            let dev = if l < split_layer { 0 } else { 1 };
            let stream = if dev == 0 { &s0 } else { &s1 };
            set_device(dev)?;
            let is_mtp = cfg.mtp_layer == Some(l);
            let is_kda = !is_mtp && cfg.is_kda(l);
            let is_moe = is_mtp || cfg.is_moe(l);
            let mut bufs = Vec::new();

            let (hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base) = if !is_mtp {
                let haf = upload_f32_to_bf16(gguf, &format!("blk.{l}.hc_attn_fn.weight"), dev, stream, &mut bufs)?;
                let has = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.hc_attn_scale.weight"), dev, stream, &mut bufs)?.ptr;
                let hab = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.hc_attn_base.weight"), dev, stream, &mut bufs)?.ptr;
                let hff = upload_f32_to_bf16(gguf, &format!("blk.{l}.hc_ffn_fn.weight"), dev, stream, &mut bufs)?;
                let hfs = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.hc_ffn_scale.weight"), dev, stream, &mut bufs)?.ptr;
                let hfb = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.hc_ffn_base.weight"), dev, stream, &mut bufs)?.ptr;
                (Some(haf), Some(has), Some(hab), Some(hff), Some(hfs), Some(hfb))
            } else {
                (None, None, None, None, None, None)
            };

            let attn_norm = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_norm.weight"), dev, stream, &mut bufs)?.ptr;
            let ffn_norm = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_norm.weight"), dev, stream, &mut bufs)?.ptr;
            let attn_output = upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_output.weight"), dev, stream, &mut bufs)?;

            let mut lw = Glm5LayerWeights {
                layer_idx: l,
                device: dev,
                is_kda,
                is_moe,
                is_mtp,
                _bufs: Vec::new(),
                hc_attn_fn,
                hc_attn_scale,
                hc_attn_base,
                hc_ffn_fn,
                hc_ffn_scale,
                hc_ffn_base,
                attn_norm,
                ffn_norm,
                attn_q: None,
                attn_k: None,
                attn_v: None,
                ssm_a: None,
                ssm_dt_bias: None,
                ssm_beta: None,
                ssm_conv_q: None,
                ssm_conv_k: None,
                ssm_conv_v: None,
                ssm_f_a: None,
                ssm_f_b: None,
                ssm_g_a: None,
                ssm_g_b: None,
                ssm_norm: None,
                attn_q_a: None,
                attn_q_a_norm: None,
                attn_q_b: None,
                attn_kv_a_mqa: None,
                attn_kv_a_norm: None,
                attn_k_b_f16: None,
                attn_v_b_f16: None,
                idx_k: None,
                idx_q_b: None,
                idx_k_norm_w: None,
                idx_k_norm_b: None,
                idx_proj: None,
                idx_ape: None,
                idx_gate: None,
                attn_output,
                ffn_gate: None,
                ffn_up: None,
                ffn_down: None,
                ffn_gate_inp: None,
                exp_probs_bias: None,
                ffn_gate_sh: None,
                ffn_up_sh: None,
                ffn_down_sh: None,
                gu_type: layer_gu_types[l],
                d_type: layer_d_types[l],
                vram_ptrs_dev: None,
                ram_ptrs_dev: None,
                spares_dev: None,
                dcnt_dev: None,
                vram_ptrs_host: Vec::new(),
                res_mask_host: vec![0u8; cfg.n_expert],
                exp_pool_ptr: 0,
                g_eb: 0,
                d_eb: 0,
                u_off: layer_u_off[l],
                d_off: layer_d_off[l],
                blob_stride: layer_blob_stride[l],
                k_spares: 0,
                initial_spare_ptrs: Vec::new(),
                initial_slot_experts: Vec::new(),
                initial_heat: vec![0.0f32; cfg.n_expert],
                nextn_eh_proj: None,
                nextn_enorm: None,
                nextn_hnorm: None,
                nextn_shnorm: None,
            };

            if is_kda {
                lw.attn_q = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_q.weight"), dev, stream, &mut bufs)?);
                lw.attn_k = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_k.weight"), dev, stream, &mut bufs)?);
                lw.attn_v = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_v.weight"), dev, stream, &mut bufs)?);
                lw.ssm_a = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_a"), dev, stream, &mut bufs)?.ptr);
                lw.ssm_dt_bias = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_dt.bias"), dev, stream, &mut bufs)?.ptr);
                lw.ssm_beta = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_beta.weight"), dev, stream, &mut bufs)?);
                lw.ssm_conv_q = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_conv1d_q.weight"), dev, stream, &mut bufs)?.ptr);
                lw.ssm_conv_k = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_conv1d_k.weight"), dev, stream, &mut bufs)?.ptr);
                lw.ssm_conv_v = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_conv1d_v.weight"), dev, stream, &mut bufs)?.ptr);
                lw.ssm_f_a = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_f_a.weight"), dev, stream, &mut bufs)?);
                lw.ssm_f_b = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_f_b.weight"), dev, stream, &mut bufs)?);
                lw.ssm_g_a = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_g_a.weight"), dev, stream, &mut bufs)?);
                lw.ssm_g_b = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_g_b.weight"), dev, stream, &mut bufs)?);
                lw.ssm_norm = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ssm_norm.weight"), dev, stream, &mut bufs)?.ptr);
            } else {
                lw.attn_q_a = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_q_a.weight"), dev, stream, &mut bufs)?);
                lw.attn_q_a_norm = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_q_a_norm.weight"), dev, stream, &mut bufs)?.ptr);
                lw.attn_q_b = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_q_b.weight"), dev, stream, &mut bufs)?);
                lw.attn_kv_a_mqa = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_kv_a_mqa.weight"), dev, stream, &mut bufs)?);
                lw.attn_kv_a_norm = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.attn_kv_a_norm.weight"), dev, stream, &mut bufs)?.ptr);
                lw.attn_k_b_f16 = Some(upload_q6_to_f16(gguf, &format!("blk.{l}.attn_k_b.weight"), dev, stream, &mut bufs)?);
                lw.attn_v_b_f16 = Some(upload_q6_to_f16(gguf, &format!("blk.{l}.attn_v_b.weight"), dev, stream, &mut bufs)?);
                lw.idx_k = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer.attn_k.weight"), dev, stream, &mut bufs)?);
                lw.idx_q_b = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer.attn_q_b.weight"), dev, stream, &mut bufs)?);
                lw.idx_k_norm_w = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer.k_norm.weight"), dev, stream, &mut bufs)?.ptr);
                lw.idx_k_norm_b = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer.k_norm.bias"), dev, stream, &mut bufs)?.ptr);
                lw.idx_proj = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer.proj.weight"), dev, stream, &mut bufs)?);
                lw.idx_ape = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer_compressor_ape.weight"), dev, stream, &mut bufs)?.ptr);
                lw.idx_gate = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.indexer_compressor_gate.weight"), dev, stream, &mut bufs)?);
            }

            if !is_moe {
                lw.ffn_gate = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_gate.weight"), dev, stream, &mut bufs)?);
                lw.ffn_up = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_up.weight"), dev, stream, &mut bufs)?);
                lw.ffn_down = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_down.weight"), dev, stream, &mut bufs)?);
            } else {
                total_routed_experts += cfg.n_expert;
                lw.ffn_gate_inp = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_gate_inp.weight"), dev, stream, &mut bufs)?);
                lw.exp_probs_bias = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.exp_probs_b.bias"), dev, stream, &mut bufs)?.ptr);
                lw.ffn_gate_sh = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_gate_shexp.weight"), dev, stream, &mut bufs)?);
                lw.ffn_up_sh = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_up_shexp.weight"), dev, stream, &mut bufs)?);
                lw.ffn_down_sh = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.ffn_down_shexp.weight"), dev, stream, &mut bufs)?);

                // Upload VRAM-resident routed experts + K_SPARES spare slots in 1 contiguous packed pool
                let g_name = format!("blk.{l}.ffn_gate_exps.weight");
                let u_name = format!("blk.{l}.ffn_up_exps.weight");
                let d_name = format!("blk.{l}.ffn_down_exps.weight");
                let g_raw = gguf.tensor_bytes(&g_name)?;
                let u_raw = gguf.tensor_bytes(&u_name)?;
                let d_raw = gguf.tensor_bytes(&d_name)?;
                let g_eb = lw.gu_type.row_bytes(cfg.n_embd) * cfg.n_ff_exp;
                let d_eb = lw.d_type.row_bytes(cfg.n_ff_exp) * cfg.n_embd;
                let u_off = lw.u_off;
                let d_off = lw.d_off;
                let blob_stride = lw.blob_stride;

                let res_ids = layer_res_ids[l].clone();
                for &e in &res_ids {
                    lw.res_mask_host[e] = 1;
                }
                let k_spares = layer_k_spares[l];
                let total_slots = res_ids.len() + k_spares;
                let mut vram_ptrs = vec![0u64; cfg.n_expert];
                let mut spare_ptrs = Vec::with_capacity(k_spares);
                if total_slots > 0 {
                    let pool_base = pool_base_addr[dev as usize] + layer_pool_off[l] as u64;
                    let mut h_pool = vec![0u8; total_slots * blob_stride];
                    for (slot, &e) in res_ids.iter().enumerate() {
                        let slot_base = slot * blob_stride;
                        h_pool[slot_base..slot_base + g_eb]
                            .copy_from_slice(&g_raw[e * g_eb..(e + 1) * g_eb]);
                        h_pool[slot_base + u_off..slot_base + u_off + g_eb]
                            .copy_from_slice(&u_raw[e * g_eb..(e + 1) * g_eb]);
                        h_pool[slot_base + d_off..slot_base + d_off + d_eb]
                            .copy_from_slice(&d_raw[e * d_eb..(e + 1) * d_eb]);
                        vram_ptrs[e] = pool_base + slot_base as u64;
                    }
                    for s_idx in 0..k_spares {
                        let slot = res_ids.len() + s_idx;
                        spare_ptrs.push(pool_base + (slot * blob_stride) as u64);
                    }
                    unsafe {
                        ampllm_cuda_kernels::copy_h2d_raw_async(
                            pool_base as *mut c_void,
                            &h_pool,
                            stream,
                        )?;
                    }
                    stream.sync()?;
                    lw.exp_pool_ptr = pool_base;
                    dev_exp_bytes[dev as usize] += total_slots * blob_stride;
                }
                lw.g_eb = g_eb;
                lw.d_eb = d_eb;
                lw.k_spares = k_spares;
                lw.initial_spare_ptrs = spare_ptrs.clone();
                if let Some(prof) = profile {
                    let mut rank_key = vec![usize::MAX; cfg.n_expert];
                    if let Some(rl) = prof.ranked_experts.get(l) {
                        for (r, &eid) in rl.iter().enumerate() {
                            if eid < cfg.n_expert && rank_key[eid] == usize::MAX {
                                rank_key[eid] = r;
                            }
                        }
                    }
                    let mut ranked_res = res_ids.clone();
                    ranked_res.sort_by_key(|&e| (rank_key[e], e));
                    let cap = ranked_res.len().max(1);
                    for (r, &e) in ranked_res.iter().enumerate() {
                        lw.initial_heat[e] = 3.0 - 2.5 * (r as f32 / cap as f32);
                    }
                } else {
                    for &e in &res_ids {
                        lw.initial_heat[e] = 0.25;
                    }
                }
                lw.initial_slot_experts = res_ids;
                let vp_d = DeviceBuffer::alloc(dev, cfg.n_expert * 8)?;
                vp_d.copy_from_host_async(bytemuck::cast_slice(&vram_ptrs), stream)?;
                let rp_d = DeviceBuffer::alloc_zeroed(dev, cfg.n_expert * 8)?;
                let dc_d = if let Some(prof) = profile {
                    let mut init_dcnt = vec![0u32; cfg.n_expert];
                    if let Some(rl) = prof.ranked_experts.get(l) {
                        for (r, &eid) in rl.iter().enumerate() {
                            if eid < cfg.n_expert && init_dcnt[eid] == 0 {
                                if r < 128 {
                                    init_dcnt[eid] = 2;
                                } else if r < 192 {
                                    init_dcnt[eid] = 1;
                                }
                            }
                        }
                    }
                    let b = DeviceBuffer::alloc(dev, cfg.n_expert * 4)?;
                    b.copy_from_host_async(bytemuck::cast_slice(&init_dcnt), stream)?;
                    b
                } else {
                    DeviceBuffer::alloc_zeroed(dev, cfg.n_expert * 4)?
                };
                let sp_d = if k_spares > 0 {
                    let b = DeviceBuffer::alloc(dev, k_spares * 8)?;
                    b.copy_from_host_async(bytemuck::cast_slice(&spare_ptrs), stream)?;
                    Some(b)
                } else {
                    None
                };
                lw.vram_ptrs_dev = Some(vp_d);
                lw.ram_ptrs_dev = Some(rp_d);
                lw.spares_dev = sp_d;
                lw.dcnt_dev = Some(dc_d);
                lw.vram_ptrs_host = vram_ptrs;
            }

            if is_mtp {
                lw.nextn_eh_proj = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.nextn.eh_proj.weight"), dev, stream, &mut bufs)?);
                lw.nextn_enorm = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.nextn.enorm.weight"), dev, stream, &mut bufs)?.ptr);
                lw.nextn_hnorm = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.nextn.hnorm.weight"), dev, stream, &mut bufs)?.ptr);
                lw.nextn_shnorm = Some(upload_tensor_f32_or_raw(gguf, &format!("blk.{l}.nextn.shared_head_norm.weight"), dev, stream, &mut bufs)?.ptr);
            }

            dev_dense_bytes[dev as usize] += bufs.iter().map(|b| b.bytes).sum::<usize>();
            lw._bufs = bufs;
            layers.push(lw);
        }
        dev_dense_bytes[0] = dev_dense_bytes[0].saturating_sub(dev_exp_bytes[0]);
        dev_dense_bytes[1] = dev_dense_bytes[1].saturating_sub(dev_exp_bytes[1]);
        tracing::info!(
            "GLM-5.3 actual alloc: GPU0 dense={} MiB exp={} MiB | GPU1 dense={} MiB exp={} MiB",
            dev_dense_bytes[0] >> 20, dev_exp_bytes[0] >> 20,
            dev_dense_bytes[1] >> 20, dev_exp_bytes[1] >> 20
        );

        // Output head on GPU 1
        set_device(1)?;
        let on_bytes = gguf.tensor_bytes("output_norm.weight")?;
        let output_norm_dev = DeviceBuffer::alloc(1, on_bytes.len())?;
        output_norm_dev.copy_from_host_async(on_bytes, &s1)?;
        let ow_info = gguf.tensor_info("output.weight")?;
        let ow_bytes = gguf.tensor_bytes("output.weight")?;
        let output_weight_dev = DeviceBuffer::alloc(1, ow_bytes.len())?;
        output_weight_dev.copy_from_host_async(ow_bytes, &s1)?;

        let draft_vocab = build_draft_vocab(gguf, tok, cfg, &output_weight_dev, 1, &s1)?;

        s0.sync()?;
        s1.sync()?;

        let emb_info = gguf.tensor_info("token_embd.weight")?;
        let token_embd_type = emb_info.ggml_type;
        let token_embd_row_bytes = token_embd_type.row_bytes(cfg.n_embd);

        // Pack initially-cold experts into 4 GiB RegisteredMappedHostBuffer chunks layer-by-layer
        // (12-thread copy + immediate GGUF madvise/fadvise DONTNEED per layer so host RAM stays <= 80 GiB).
        let shard_files: Vec<std::fs::File> = gguf
            .shard_paths
            .iter()
            .filter_map(|p| std::fs::File::open(p).ok())
            .collect();

        let layers_atomic_vec: Vec<Vec<AtomicUsize>> = (0..n_tot)
            .map(|_| (0..cfg.n_expert).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let layers_dev0_vec: Vec<Vec<AtomicU64>> = (0..n_tot)
            .map(|_| (0..cfg.n_expert).map(|_| AtomicU64::new(0)).collect())
            .collect();
        let layers_dev1_vec: Vec<Vec<AtomicU64>> = (0..n_tot)
            .map(|_| (0..cfg.n_expert).map(|_| AtomicU64::new(0)).collect())
            .collect();

        // Drop page cache for all uploaded dense tensors + output weights (already resident in VRAM).
        for t in &gguf.tensors {
            if t.name != "token_embd.weight" && !t.name.contains("_exps.weight") {
                if let Ok(raw) = gguf.tensor_bytes(&t.name) {
                    advise_dontneed_range(raw.as_ptr() as usize, raw.len());
                }
                fadvise_dontneed_file(&shard_files, t.shard_idx, t.data_offset, t.byte_len);
            }
        }
        // Drop page cache for VRAM-resident experts immediately before allocating host chunks.
        for l in 0..n_tot {
            if layer_blob_stride[l] == 0 {
                continue;
            }
            let g_eb = layer_gu_types[l].row_bytes(cfg.n_embd) * cfg.n_ff_exp;
            let d_eb = layer_d_types[l].row_bytes(cfg.n_ff_exp) * cfg.n_embd;
            let g_name = format!("blk.{l}.ffn_gate_exps.weight");
            let u_name = format!("blk.{l}.ffn_up_exps.weight");
            let d_name = format!("blk.{l}.ffn_down_exps.weight");
            let g_info = gguf.tensor_info(&g_name)?;
            let u_info = gguf.tensor_info(&u_name)?;
            let d_info = gguf.tensor_info(&d_name)?;
            let g_raw = gguf.tensor_bytes(&g_name)?;
            let u_raw = gguf.tensor_bytes(&u_name)?;
            let d_raw = gguf.tensor_bytes(&d_name)?;
            for e in 0..cfg.n_expert {
                if is_vram[l][e] {
                    advise_dontneed_range(unsafe { g_raw.as_ptr().add(e * g_eb) } as usize, g_eb);
                    advise_dontneed_range(unsafe { u_raw.as_ptr().add(e * g_eb) } as usize, g_eb);
                    advise_dontneed_range(unsafe { d_raw.as_ptr().add(e * d_eb) } as usize, d_eb);
                    fadvise_dontneed_file(&shard_files, g_info.shard_idx, g_info.data_offset + e * g_eb, g_eb);
                    fadvise_dontneed_file(&shard_files, u_info.shard_idx, u_info.data_offset + e * g_eb, g_eb);
                    fadvise_dontneed_file(&shard_files, d_info.shard_idx, d_info.data_offset + e * d_eb, d_eb);
                }
            }
        }

        // Plan sequential cold chunk offsets (grouped per layer so each layer finishes & drops cache immediately)
        let mut chunk_sizes: Vec<usize> = Vec::new();
        let mut cur_chunk_off = 0usize;
        let mut layer_jobs: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_tot];
        let mut total_cold_bytes = 0usize;
        for l in 0..n_tot {
            let stride = layer_blob_stride[l];
            if stride == 0 {
                continue;
            }
            for e in 0..cfg.n_expert {
                if !is_vram[l][e] {
                    if cur_chunk_off + stride > COLD_CHUNK_TARGET_BYTES && cur_chunk_off > 0 {
                        chunk_sizes.push(cur_chunk_off);
                        cur_chunk_off = 0;
                    }
                    let cidx = chunk_sizes.len();
                    layer_jobs[l].push((e, cidx, cur_chunk_off));
                    cur_chunk_off += stride;
                    total_cold_bytes += stride;
                }
            }
        }
        if cur_chunk_off > 0 {
            chunk_sizes.push(cur_chunk_off);
        }

        let t_warm = std::time::Instant::now();
        let mut cold_chunk_opts: Vec<Option<RegisteredMappedHostBuffer>> =
            (0..chunk_sizes.len()).map(|_| None).collect();
        let mut chunk_host_base = vec![0usize; chunk_sizes.len()];
        let mut chunk_dev0_base = vec![0u64; chunk_sizes.len()];
        let mut chunk_dev1_base = vec![0u64; chunk_sizes.len()];
        let n_copy_threads = 12usize;

        for l in 0..n_tot {
            let jobs = &layer_jobs[l];
            if jobs.is_empty() {
                continue;
            }
            for &(_, cidx, _) in jobs {
                if cold_chunk_opts[cidx].is_none() {
                    let buf = RegisteredMappedHostBuffer::alloc(chunk_sizes[cidx])?;
                    chunk_host_base[cidx] = buf.as_ptr() as usize;
                    chunk_dev0_base[cidx] = buf.dev_ptr_for_device(0)?;
                    chunk_dev1_base[cidx] = buf.dev_ptr_for_device(1)?;
                    cold_chunk_opts[cidx] = Some(buf);
                }
            }
            let g_eb = layer_gu_types[l].row_bytes(cfg.n_embd) * cfg.n_ff_exp;
            let d_eb = layer_d_types[l].row_bytes(cfg.n_ff_exp) * cfg.n_embd;
            let u_off = layer_u_off[l];
            let d_off = layer_d_off[l];
            let g_name = format!("blk.{l}.ffn_gate_exps.weight");
            let u_name = format!("blk.{l}.ffn_up_exps.weight");
            let d_name = format!("blk.{l}.ffn_down_exps.weight");
            let g_info = gguf.tensor_info(&g_name)?;
            let u_info = gguf.tensor_info(&u_name)?;
            let d_info = gguf.tensor_info(&d_name)?;
            let g_raw = gguf.tensor_bytes(&g_name)?;
            let u_raw = gguf.tensor_bytes(&u_name)?;
            let d_raw = gguf.tensor_bytes(&d_name)?;

            let g_addr = g_raw.as_ptr() as usize;
            let u_addr = u_raw.as_ptr() as usize;
            let d_addr = d_raw.as_ptr() as usize;
            let next_job = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..n_copy_threads {
                    s.spawn(|| loop {
                        let ji = next_job.fetch_add(1, Ordering::Relaxed);
                        if ji >= jobs.len() {
                            break;
                        }
                        let (e, cidx, off) = jobs[ji];
                        let dst = (chunk_host_base[cidx] + off) as *mut u8;
                        unsafe {
                            std::ptr::copy_nonoverlapping((g_addr + e * g_eb) as *const u8, dst, g_eb);
                            std::ptr::copy_nonoverlapping(
                                (u_addr + e * g_eb) as *const u8,
                                dst.add(u_off),
                                g_eb,
                            );
                            std::ptr::copy_nonoverlapping(
                                (d_addr + e * d_eb) as *const u8,
                                dst.add(d_off),
                                d_eb,
                            );
                        }
                    });
                }
            });

            for &(e, cidx, off) in jobs {
                let hp = chunk_host_base[cidx] + off;
                let d0 = chunk_dev0_base[cidx] + off as u64;
                let d1 = chunk_dev1_base[cidx] + off as u64;
                layers_atomic_vec[l][e].store(hp, Ordering::Relaxed);
                layers_dev0_vec[l][e].store(d0, Ordering::Relaxed);
                layers_dev1_vec[l][e].store(d1, Ordering::Relaxed);
            }

            // Immediately release page cache for all three expert weight tensors of this layer
            advise_dontneed_range(g_addr, g_raw.len());
            advise_dontneed_range(u_addr, u_raw.len());
            advise_dontneed_range(d_addr, d_raw.len());
            fadvise_dontneed_file(&shard_files, g_info.shard_idx, g_info.data_offset, g_info.byte_len);
            fadvise_dontneed_file(&shard_files, u_info.shard_idx, u_info.data_offset, u_info.byte_len);
            fadvise_dontneed_file(&shard_files, d_info.shard_idx, d_info.data_offset, d_info.byte_len);
        }

        let mut cold_chunks: Vec<RegisteredMappedHostBuffer> = cold_chunk_opts
            .into_iter()
            .map(|o| o.expect("all planned cold chunks allocated"))
            .collect();

        // Pre-allocate tiered pinned-mapped Backup Pool chunks (trunk vs larger MTP stride)
        let trunk_max_stride = (0..cfg.n_layers.min(n_tot))
            .map(|l| layer_blob_stride[l])
            .max()
            .unwrap_or(0);
        let mtp_max_stride = cfg
            .mtp_layer
            .and_then(|ml| layer_blob_stride.get(ml).copied())
            .unwrap_or(0);
        let mut alloc_backup_tier = |total_slots: usize, stride: usize| -> Result<Vec<(usize, u64, u64)>> {
            let mut ptrs = Vec::with_capacity(total_slots);
            if stride > 0 && total_slots > 0 {
                let mut rem = total_slots;
                while rem > 0 {
                    let n_slots = rem.min(BACKUP_CHUNK_SLOTS);
                    let buf = RegisteredMappedHostBuffer::alloc(n_slots * stride)?;
                    let hp = buf.as_ptr() as usize;
                    let d0 = buf.dev_ptr_for_device(0)?;
                    let d1 = buf.dev_ptr_for_device(1)?;
                    for s in 0..n_slots {
                        let o = s * stride;
                        ptrs.push((hp + o, d0 + o as u64, d1 + o as u64));
                    }
                    cold_chunks.push(buf);
                    rem -= n_slots;
                }
            }
            Ok(ptrs)
        };
        let trunk_backup_slots = std::env::var("AMPLLM_BACKUP_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(TRUNK_BACKUP_SLOTS)
            .clamp(LEND_RESERVED_TRUNK_SLOTS + 64, 4096);
        let trunk_ptrs = alloc_backup_tier(trunk_backup_slots, trunk_max_stride)?;
        let mtp_ptrs = alloc_backup_tier(MTP_BACKUP_SLOTS, mtp_max_stride)?;
        let trunk_free: Vec<usize> = (0..trunk_ptrs.len()).rev().collect();
        let mtp_free: Vec<usize> = (0..mtp_ptrs.len()).rev().collect();
        let backup_total_slots = trunk_ptrs.len() + mtp_ptrs.len();
        let backup_bytes = trunk_ptrs.len() * trunk_max_stride + mtp_ptrs.len() * mtp_max_stride;
        std::sync::atomic::fence(Ordering::Release);

        let cold_arena = CpuColdArena {
            _chunks: cold_chunks,
            layers_atomic: Arc::new(layers_atomic_vec),
            layers_dev: [layers_dev0_vec, layers_dev1_vec],
            layer_u_off,
            layer_d_off,
            layer_blob_stride,
            backup_pool: Mutex::new(BackupSlotPool {
                mtp_layer: cfg.mtp_layer,
                trunk_ptrs,
                trunk_free,
                mtp_ptrs,
                mtp_free,
                recycled_entries: Vec::new(),
                layer_free_recycled: vec![Vec::new(); n_tot],
                slot_of: vec![vec![-1i32; cfg.n_expert]; n_tot],
            }),
        };

        // Upload device-mapped RAM pointer tables for every MoE layer
        for l in 0..n_tot {
            if !layers[l].is_moe {
                continue;
            }
            let dev = layers[l].device;
            set_device(dev)?;
            let stream = if dev == 0 { &s0 } else { &s1 };
            let mut ram_ptrs = vec![0u64; cfg.n_expert];
            for e in 0..cfg.n_expert {
                ram_ptrs[e] = cold_arena.get_dev_ptr(l, e, dev);
            }
            if let Some(ref b) = layers[l].ram_ptrs_dev {
                b.copy_from_host_async(bytemuck::cast_slice(&ram_ptrs), stream)?;
            }
        }
        s0.sync()?;
        s1.sync()?;

        let cold_gib = total_cold_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let backup_gib = backup_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        tracing::info!(
            "GLM-5.3 zero-copy cold arena ({cold_gib:.2} GiB across {} chunks) + backup pool ({backup_gib:.2} GiB, {backup_total_slots} slots) ready in {:.2}s",
            chunk_sizes.len(),
            t_warm.elapsed().as_secs_f64()
        );

        let (free0_after, tot0) = mem_info(0)?;
        let (free1_after, tot1) = mem_info(1)?;
        let vram_used_gpu0_mb = (tot0 - free0_after) >> 20;
        let vram_used_gpu1_mb = (tot1 - free1_after) >> 20;

        tracing::info!(
            "GLM-5.3 weights ready: GPU0={} MiB used ({} MiB free), GPU1={} MiB used ({} MiB free), cold={cold_gib:.2} GiB, backup={backup_gib:.2} GiB, VRAM={}/{} experts ({:.1}%)",
            vram_used_gpu0_mb, free0_after >> 20,
            vram_used_gpu1_mb, free1_after >> 20,
            vram_count, total_routed_experts,
            100.0 * vram_count as f64 / total_routed_experts.max(1) as f64
        );

        Ok(Self {
            cfg: cfg.clone(),
            split_layer,
            layers,
            exp_pool_dev,
            max_trunk_blob_stride,
            dsa_res_cap,
            pool_cap,
            output_norm_dev,
            output_weight_dev,
            output_weight_type: ow_info.ggml_type,
            token_embd_type,
            token_embd_row_bytes,
            cold_arena,
            draft_vocab,
            has_profile: profile.is_some(),
            vram_resident_experts: vram_count,
            total_routed_experts,
            vram_used_gpu0_mb,
            vram_used_gpu1_mb,
        })
    }
}
