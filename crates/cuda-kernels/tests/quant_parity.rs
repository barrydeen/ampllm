use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{CudaStream, DeviceBuffer};
use ampllm_gguf::{vec_dot_q_f32_ref, GgmlType, KVALUES_IQ4NL};
use half::f16;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() % 2000) as f32 - 1000.0) / 500.0
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
}

fn fill_valid_ggml_row(ggml_type: GgmlType, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / ggml_type.block_size();
    let bsz = ggml_type.type_size();
    let mut bytes = vec![0u8; nb * bsz];
    for b in 0..nb {
        let blk = &mut bytes[b * bsz..(b + 1) * bsz];
        match ggml_type {
            GgmlType::F32 => {
                let f: &mut f32 = bytemuck::from_bytes_mut(blk);
                *f = rng.next_f32();
            }
            GgmlType::Q8_0 => {
                let d = f16::from_f32(0.015 + (b as f32) * 0.0003).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for q in &mut blk[2..34] {
                    *q = rng.next_u8();
                }
            }
            GgmlType::Q4_0 | GgmlType::Iq4Nl => {
                let d = f16::from_f32(0.02 + (b as f32) * 0.0005).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for q in &mut blk[2..18] {
                    *q = rng.next_u8();
                }
            }
            GgmlType::Iq4Xs => {
                let d = f16::from_f32(0.005).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for byte in &mut blk[2..136] {
                    *byte = rng.next_u8();
                }
            }
            GgmlType::Q4K => {
                let d = f16::from_f32(0.01).to_le_bytes();
                let dm = f16::from_f32(0.008).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&dm);
                for byte in &mut blk[4..144] {
                    *byte = rng.next_u8();
                }
            }
            GgmlType::Q5K => {
                let d = f16::from_f32(0.01).to_le_bytes();
                let dm = f16::from_f32(0.008).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&dm);
                for byte in &mut blk[4..176] {
                    *byte = rng.next_u8();
                }
            }
            GgmlType::Q6K => {
                let d = f16::from_f32(0.01).to_le_bytes();
                blk[208..210].copy_from_slice(&d);
                for byte in &mut blk[0..208] {
                    *byte = rng.next_u8();
                }
            }
            GgmlType::Nvfp4 => {
                // ue4m3 bytes around 0x38..0x42 (~0.06..0.25)
                for sb in 0..4 {
                    blk[sb] = 0x38 + (rng.next_u8() & 0x07);
                }
                for byte in &mut blk[4..36] {
                    *byte = rng.next_u8();
                }
            }
            _ => unreachable!(),
        }
    }
    bytes
}

#[test]
fn test_cuda_quant_gemv_parity() {
    let _ = KVALUES_IQ4NL;
    let stream = CudaStream::new(0).expect("CUDA GPU 0 available");
    let k = 2304usize;
    let n = 64usize;
    let m = 3usize;
    let mut rng = Rng(0x1234_5678_9abc_def0);

    let mut x = vec![0.0f32; m * k];
    for v in &mut x {
        *v = rng.next_f32();
    }
    let d_x = DeviceBuffer::alloc(0, m * k * 4).unwrap();
    d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();

    let nb_x = k / 32;
    let d_x_q8 = DeviceBuffer::alloc(0, m * nb_x * 36).unwrap();
    let d_y = DeviceBuffer::alloc(0, m * n * 4).unwrap();

    unsafe {
        launch_quantize_q8_1(
            d_x.ptr as *const f32,
            d_x_q8.ptr,
            m as i32,
            k as i32,
            stream.raw,
        );
    }

    let types = [
        GgmlType::Q8_0,
        GgmlType::Iq4Nl,
        GgmlType::Q4_0,
        GgmlType::F32,
        // K-quants need K multiple of 256 (2304 == 9 * 256!)
        GgmlType::Iq4Xs,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Nvfp4,
    ];

    for ty in types {
        let row_bytes = ty.row_bytes(k);
        let mut w = Vec::with_capacity(n * row_bytes);
        for _ in 0..n {
            w.extend_from_slice(&fill_valid_ggml_row(ty, k, &mut rng));
        }
        let d_w = DeviceBuffer::alloc(0, w.len()).unwrap();
        d_w.copy_from_host(&w).unwrap();

        unsafe {
            launch_linear_q8_1(
                ty as i32,
                d_w.ptr,
                d_x_q8.ptr,
                d_x.ptr as *const f32,
                d_y.ptr as *mut f32,
                n as i32,
                k as i32,
                m as i32,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut gpu_y = vec![0.0f32; m * n];
        d_y.copy_to_host(bytemuck::cast_slice_mut(&mut gpu_y)).unwrap();

        let mut max_abs = 0.0f32;
        let mut max_rel_norm = 0.0f32;
        let mut sample_pair = (0.0f32, 0.0f32);
        for mi in 0..m {
            let x_row = &x[mi * k..(mi + 1) * k];
            let x_norm = x_row.iter().map(|v| v * v).sum::<f32>().sqrt();
            for ni in 0..n {
                let mut w_f32 = vec![0.0f32; k];
                ampllm_gguf::dequantize_row_to_f32(ty, &w[ni * row_bytes..(ni + 1) * row_bytes], &mut w_f32);
                let w_norm = w_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                let ref_val = vec_dot_q_f32_ref(
                    ty,
                    &w[ni * row_bytes..(ni + 1) * row_bytes],
                    x_row,
                );
                let got = gpu_y[mi * n + ni];
                let diff = (got - ref_val).abs();
                let rel_norm = diff / (w_norm * x_norm).max(1e-6);
                if diff > max_abs {
                    max_abs = diff;
                    sample_pair = (ref_val, got);
                }
                if rel_norm > max_rel_norm {
                    max_rel_norm = rel_norm;
                }
            }
        }
        let tol = if ty == GgmlType::F32 { 1e-5 } else { 0.005 };
        println!("Type {ty:?}: max_abs={max_abs:.6} max_rel_norm={max_rel_norm:.6} worst=({:.4}, {:.4})", sample_pair.0, sample_pair.1);
        assert!(max_rel_norm < tol, "Type {ty:?} exceeded tol {tol}: {max_rel_norm}");
    }
}
