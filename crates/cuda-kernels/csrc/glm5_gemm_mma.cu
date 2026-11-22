#include "common/sm86_common.cuh"
#include <cstdint>
#include <cmath>

// -----------------------------------------------------------------------------
// Zero-cuBLAS Custom Ampere (sm_86) FP16 Dequantizer & Tensor-Core MMA GEMM
// C[M, N] (F32) = alpha * A[M, K] (F16) * B[N, K]^T (F16) + beta * C[M, N]
// -----------------------------------------------------------------------------

__global__ void f32_to_f16_kernel(const float* __restrict__ src, uint16_t* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = f32_to_fp16_bits(src[idx]);
}

__global__ void f32_to_f16_vec4_kernel(const float4* __restrict__ src4, uint2* __restrict__ dst2, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n4) return;
    float4 v = src4[idx];
    uint32_t h01 = (uint32_t)f32_to_fp16_bits(v.x) | ((uint32_t)f32_to_fp16_bits(v.y) << 16);
    uint32_t h23 = (uint32_t)f32_to_fp16_bits(v.z) | ((uint32_t)f32_to_fp16_bits(v.w) << 16);
    dst2[idx] = make_uint2(h01, h23);
}

extern "C" void launch_f32_to_f16(const float* src, void* dst_f16, int n, cudaStream_t stream) {
    if (n <= 0) return;
    if ((n & 3) == 0 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u && (reinterpret_cast<uintptr_t>(dst_f16) & 7u) == 0u) {
        int n4 = n >> 2;
        f32_to_f16_vec4_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const float4*>(src), reinterpret_cast<uint2*>(dst_f16), n4
        );
        return;
    }
    f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, reinterpret_cast<uint16_t*>(dst_f16), n);
}

// Parallel dequantizer: 16 threads per Q8_0 block (packed uint32_t writes)
__global__ void __launch_bounds__(256) dequant_q8_0_to_f16_kernel(
    const block_q8_0* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_idx = tid >> 4;
    int p = tid & 15;
    if (blk_idx >= n_blocks) return;
    const block_q8_0* b = src + blk_idx;
    float d = fp16_bits_to_f32(load_u16_align2(&b->d));
    uint32_t h0 = (uint32_t)f32_to_fp16_bits(d * (float)b->qs[2 * p + 0]);
    uint32_t h1 = (uint32_t)f32_to_fp16_bits(d * (float)b->qs[2 * p + 1]);
    if ((reinterpret_cast<uintptr_t>(dst) & 3u) == 0u) {
        reinterpret_cast<uint32_t*>(dst)[(size_t)blk_idx * 16 + p] = h0 | (h1 << 16);
    } else {
        size_t o = (size_t)blk_idx * 32 + 2 * p;
        dst[o] = (uint16_t)h0;
        dst[o + 1] = (uint16_t)h1;
    }
}

__global__ void __launch_bounds__(256) dequant_q6_K_to_f16_kernel(
    const block_q6_K* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_idx = tid >> 6; // 64 threads per 256-elem block (4 elems/thread)
    int t64 = tid & 63;
    if (blk_idx >= n_blocks) return;
    const block_q6_K* b = src + blk_idx;
    float d = fp16_bits_to_f32(load_u16_align2(&b->d));
    int half = t64 >> 5; // 0..1
    int l = t64 & 31;
    const uint8_t* ql = b->ql + half * 64;
    const uint8_t* qh = b->qh + half * 32;
    const int8_t* sc = b->scales + half * 8;
    int is = l >> 4;
    uint8_t qh_v = qh[l];
    uint8_t ql0 = ql[l];
    uint8_t ql1 = ql[l + 32];
    int q1 = (int)((ql0 & 0xF) | (((qh_v >> 0) & 3) << 4)) - 32;
    int q2 = (int)((ql1 & 0xF) | (((qh_v >> 2) & 3) << 4)) - 32;
    int q3 = (int)((ql0 >> 4)  | (((qh_v >> 4) & 3) << 4)) - 32;
    int q4 = (int)((ql1 >> 4)  | (((qh_v >> 6) & 3) << 4)) - 32;
    size_t base = (size_t)blk_idx * 256 + half * 128 + l;
    dst[base + 0]  = f32_to_fp16_bits(d * (float)sc[is + 0] * (float)q1);
    dst[base + 32] = f32_to_fp16_bits(d * (float)sc[is + 2] * (float)q2);
    dst[base + 64] = f32_to_fp16_bits(d * (float)sc[is + 4] * (float)q3);
    dst[base + 96] = f32_to_fp16_bits(d * (float)sc[is + 6] * (float)q4);
}

__global__ void __launch_bounds__(256) dequant_q4_K_to_f16_kernel(
    const block_q4_K* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_idx = tid >> 7; // 128 threads/block -> 2 elems/thread (pair & l)
    int t128 = tid & 127;
    if (blk_idx >= n_blocks) return;
    const block_q4_K* b = src + blk_idx;
    float2 dm = fp16x2_bits_to_f32(&b->d);
    int j = t128 >> 5; // 0..3 (pair of 32)
    int l = t128 & 31;
    uint8_t sc1, m1, sc2, m2;
    get_scale_min_k4_dev(j * 2 + 0, b->scales, sc1, m1);
    get_scale_min_k4_dev(j * 2 + 1, b->scales, sc2, m2);
    uint8_t qb = b->qs[j * 32 + l];
    size_t base = (size_t)blk_idx * 256 + j * 64 + l;
    dst[base + 0]  = f32_to_fp16_bits(dm.x * (float)sc1 * (float)(qb & 0xF) - dm.y * (float)m1);
    dst[base + 32] = f32_to_fp16_bits(dm.x * (float)sc2 * (float)(qb >> 4)  - dm.y * (float)m2);
}

__global__ void __launch_bounds__(256) dequant_q3_K_to_f16_kernel(
    const block_q3_K* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_idx = tid >> 3; // 8 threads per block (one sub=0..7 each)
    int sub = tid & 7;
    if (blk_idx >= n_blocks) return;
    const block_q3_K* b = src + blk_idx;
    float d = fp16_bits_to_f32(load_u16_align2(&b->d));
    int a_idx = (sub >> 1) & 1;
    uint32_t low4 = ((uint32_t)load_u16_align2(b->scales + 4 * a_idx + 2 * (sub & 1)) >> (sub & 4)) & 0x0F0Fu;
    uint32_t hi2 = ((uint32_t)load_u16_align2(b->scales + 8 + 2 * (sub & 1)) >> (2 * (sub >> 1))) & 0x0303u;
    uint32_t sc_pair = low4 | (hi2 << 4);
    float d0 = d * (float)((int)(sc_pair & 0xFFu) - 32);
    float d1 = d * (float)((int)((sc_pair >> 8) & 0xFFu) - 32);
    int q_shift = 2 * (sub & 3);
    const uint8_t* qp = b->qs + 32 * (sub >> 2);
    uint8_t m = (uint8_t)(1u << sub);
    size_t base = (size_t)blk_idx * 256 + sub * 32;
    for (int l = 0; l < 16; ++l) {
        int q0 = (int)((qp[l] >> q_shift) & 3) - ((b->hmask[l] & m) ? 0 : 4);
        int q1 = (int)((qp[l + 16] >> q_shift) & 3) - ((b->hmask[l + 16] & m) ? 0 : 4);
        dst[base + l]      = f32_to_fp16_bits(d0 * (float)q0);
        dst[base + l + 16] = f32_to_fp16_bits(d1 * (float)q1);
    }
}

__global__ void __launch_bounds__(256) dequant_iq2_s_to_f16_kernel(
    const block_iq2_s* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    __shared__ __align__(16) uint32_t s_grid_u32[2048];
    const uint32_t* g_u32 = reinterpret_cast<const uint32_t*>(d_iq2s_grid);
    for (int i = threadIdx.x; i < 2048; i += 256) {
        s_grid_u32[i] = g_u32[i];
    }
    __syncthreads();

    const uint64_t* s_grid = reinterpret_cast<const uint64_t*>(s_grid_u32);
    const int total_threads = n_blocks * 8;
    const int stride = gridDim.x * 256;
    for (int tid = blockIdx.x * 256 + threadIdx.x; tid < total_threads; tid += stride) {
        int blk_idx = tid >> 3;
        int sub = tid & 7;
        const block_iq2_s* b = src + blk_idx;
        float d = fp16_bits_to_f32(load_u16_align2(&b->d));
        const uint8_t* qs = b->qs + 4 * sub;
        const uint8_t* sb = b->qs + 32 + 4 * sub;
        uint32_t qh = (uint32_t)b->qh[sub];
        uint8_t sc = b->scales[sub];
        float db0 = d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
        float db1 = d * (0.5f + (float)(sc >> 4))   * 0.25f;
        size_t base = (size_t)blk_idx * 256 + sub * 32;
        for (int idx = 0; idx < 4; ++idx) {
            int l0 = idx * 2;
            uint64_t g64 = s_grid[(uint32_t)qs[idx] | ((qh << (8 - l0)) & 0x300u)];
            float db = (idx < 2) ? db0 : db1;
            uint8_t signs = sb[idx];
            uint32_t w[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float gv0 = (float)((g64 >> (16 * p)) & 0xFFu);
                float sv0 = ((signs >> (2 * p)) & 1) ? -1.0f : 1.0f;
                float gv1 = (float)((g64 >> (16 * p + 8)) & 0xFFu);
                float sv1 = ((signs >> (2 * p + 1)) & 1) ? -1.0f : 1.0f;
                uint32_t h0 = (uint32_t)f32_to_fp16_bits(db * gv0 * sv0);
                uint32_t h1 = (uint32_t)f32_to_fp16_bits(db * gv1 * sv1);
                w[p] = h0 | (h1 << 16);
            }
            *reinterpret_cast<uint4*>(dst + base + idx * 8) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

__global__ void __launch_bounds__(256) dequant_iq3_xxs_to_f16_kernel(
    const block_iq3_xxs* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    __shared__ uint32_t s_grid[256];
    __shared__ uint8_t s_signs[128];
    s_grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) s_signs[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();

    const int total_threads = n_blocks * 8;
    const int stride = gridDim.x * 256;
    for (int tid = blockIdx.x * 256 + threadIdx.x; tid < total_threads; tid += stride) {
        int blk_idx = tid >> 3;
        int sub = tid & 7;
        const block_iq3_xxs* b = src + blk_idx;
        float d = fp16_bits_to_f32(load_u16_align2(&b->d));
        const uint8_t* qs = b->qs + 8 * sub;
        uint32_t aux32 = (uint32_t)load_i32_align2(b->qs + 64 + 4 * sub);
        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        size_t base = (size_t)blk_idx * 256 + sub * 32;
        for (int idx = 0; idx < 4; ++idx) {
            uint32_t g0 = s_grid[qs[2 * idx + 0]];
            uint32_t g1 = s_grid[qs[2 * idx + 1]];
            uint8_t signs = s_signs[(aux32 >> (7 * idx)) & 0x7Fu];
            uint32_t w[4];
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                float v0a = (float)((g0 >> (16 * p)) & 0xFFu)     * (((signs >> (2 * p)) & 1) ? -1.0f : 1.0f);
                float v0b = (float)((g0 >> (16 * p + 8)) & 0xFFu) * (((signs >> (2 * p + 1)) & 1) ? -1.0f : 1.0f);
                float v1a = (float)((g1 >> (16 * p)) & 0xFFu)     * (((signs >> (2 * p + 4)) & 1) ? -1.0f : 1.0f);
                float v1b = (float)((g1 >> (16 * p + 8)) & 0xFFu) * (((signs >> (2 * p + 5)) & 1) ? -1.0f : 1.0f);
                w[p]     = (uint32_t)f32_to_fp16_bits(db * v0a) | ((uint32_t)f32_to_fp16_bits(db * v0b) << 16);
                w[p + 2] = (uint32_t)f32_to_fp16_bits(db * v1a) | ((uint32_t)f32_to_fp16_bits(db * v1b) << 16);
            }
            *reinterpret_cast<uint4*>(dst + base + idx * 8) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

__global__ void __launch_bounds__(256) dequant_iq3_s_to_f16_kernel(
    const block_iq3_s* __restrict__ src,
    uint16_t* __restrict__ dst,
    int n_blocks
) {
    __shared__ uint32_t s_grid[512];
    for (int i = threadIdx.x; i < 512; i += 256) {
        s_grid[i] = d_iq3s_grid[i];
    }
    __syncthreads();

    const int total_threads = n_blocks * 8;
    const int stride = gridDim.x * 256;
    for (int tid = blockIdx.x * 256 + threadIdx.x; tid < total_threads; tid += stride) {
        int blk_idx = tid >> 3;
        int sub = tid & 7;
        const block_iq3_s* b = src + blk_idx;
        float d = fp16_bits_to_f32(load_u16_align2(&b->d));
        const uint8_t* qs = b->qs + 8 * sub;
        uint32_t qh = (uint32_t)b->qh[sub];
        const uint8_t* sb = b->signs + 4 * sub;
        float db = d * (float)(1 + 2 * (int)((b->scales[sub >> 1] >> (4 * (sub & 1))) & 0x0Fu));
        size_t base = (size_t)blk_idx * 256 + sub * 32;
        for (int idx = 0; idx < 4; ++idx) {
            int l0 = idx * 2;
            uint32_t g0 = s_grid[qs[2 * idx + 0] | ((qh << (8 - l0)) & 0x100u)];
            uint32_t g1 = s_grid[qs[2 * idx + 1] | ((qh << (7 - l0)) & 0x100u)];
            uint8_t signs = sb[idx];
            uint32_t w[4];
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                float v0a = (float)((g0 >> (16 * p)) & 0xFFu)     * (((signs >> (2 * p)) & 1) ? -1.0f : 1.0f);
                float v0b = (float)((g0 >> (16 * p + 8)) & 0xFFu) * (((signs >> (2 * p + 1)) & 1) ? -1.0f : 1.0f);
                float v1a = (float)((g1 >> (16 * p)) & 0xFFu)     * (((signs >> (2 * p + 4)) & 1) ? -1.0f : 1.0f);
                float v1b = (float)((g1 >> (16 * p + 8)) & 0xFFu) * (((signs >> (2 * p + 5)) & 1) ? -1.0f : 1.0f);
                w[p]     = (uint32_t)f32_to_fp16_bits(db * v0a) | ((uint32_t)f32_to_fp16_bits(db * v0b) << 16);
                w[p + 2] = (uint32_t)f32_to_fp16_bits(db * v1a) | ((uint32_t)f32_to_fp16_bits(db * v1b) << 16);
            }
            *reinterpret_cast<uint4*>(dst + base + idx * 8) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

extern "C" void launch_dequant_to_f16(
    int32_t w_type,
    const void* src,
    void* dst_f16,
    int64_t n_elems,
    cudaStream_t stream
) {
    if (n_elems <= 0) return;
    uint16_t* d16 = reinterpret_cast<uint16_t*>(dst_f16);
    if (w_type == GGML_TYPE_F32) {
        launch_f32_to_f16(reinterpret_cast<const float*>(src), dst_f16, (int)n_elems, stream);
        return;
    }
    if (w_type == GGML_TYPE_F16) {
        cudaMemcpyAsync(dst_f16, src, (size_t)n_elems * 2, cudaMemcpyDeviceToDevice, stream);
        return;
    }
    if (w_type == GGML_TYPE_Q8_0) {
        int nb = (int)(n_elems / 32);
        dequant_q8_0_to_f16_kernel<<<(nb * 16 + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const block_q8_0*>(src), d16, nb
        );
        return;
    }
    int nb = (int)(n_elems / 256);
    int iq_grid = min((nb * 8 + 255) / 256, 328);
    switch (w_type) {
        case GGML_TYPE_Q6_K:
            dequant_q6_K_to_f16_kernel<<<(nb * 64 + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const block_q6_K*>(src), d16, nb
            );
            break;
        case GGML_TYPE_Q4_K:
            dequant_q4_K_to_f16_kernel<<<(nb * 128 + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const block_q4_K*>(src), d16, nb
            );
            break;
        case GGML_TYPE_Q3_K:
            dequant_q3_K_to_f16_kernel<<<(nb * 8 + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const block_q3_K*>(src), d16, nb
            );
            break;
        case GGML_TYPE_IQ2_S:
            if (iq_grid > 0) dequant_iq2_s_to_f16_kernel<<<iq_grid, 256, 0, stream>>>(
                reinterpret_cast<const block_iq2_s*>(src), d16, nb
            );
            break;
        case GGML_TYPE_IQ3_XXS:
            if (iq_grid > 0) dequant_iq3_xxs_to_f16_kernel<<<iq_grid, 256, 0, stream>>>(
                reinterpret_cast<const block_iq3_xxs*>(src), d16, nb
            );
            break;
        case GGML_TYPE_IQ3_S:
            if (iq_grid > 0) dequant_iq3_s_to_f16_kernel<<<iq_grid, 256, 0, stream>>>(
                reinterpret_cast<const block_iq3_s*>(src), d16, nb
            );
            break;
        default:
            break;
    }
}

// -----------------------------------------------------------------------------
// Ampere sm_86 Double-Buffered Tensor Core GEMM: mma.sync.aligned.m16n8k16
// C[M, N] (F32) = alpha * A[M, K] (F16) * B[N, K]^T (F16) + beta * C[M, N]
// -----------------------------------------------------------------------------

__device__ __forceinline__ void mma16816_f32_f16(
    float (&c)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* glob_ptr, bool valid) {
    uint32_t sp = smem_ptr_u32(smem_ptr);
    if (valid) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sp), "l"(glob_ptr));
    } else {
        reinterpret_cast<uint4*>(smem_ptr)[0] = make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ int swz_col32(int r, int col) {
    return col ^ (((r >> 1) & 3) << 3);
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr
) {
    uint32_t addr = smem_ptr_u32(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

template <int ROWS, int THREADS>
__device__ __forceinline__ void load_tile_swz32(
    uint16_t* __restrict__ s_dst,
    const uint16_t* __restrict__ g_src,
    int row0,
    int limit,
    int K,
    int k0,
    int tid,
    bool vec16_ok
) {
    constexpr int VECS = ROWS * 4;
    #pragma unroll
    for (int idx = tid; idx < VECS; idx += THREADS) {
        int r = idx >> 2;
        int c8 = (idx & 3) * 8;
        int gr = row0 + r;
        int gc = k0 + c8;
        uint16_t* sp = &s_dst[r * 32 + swz_col32(r, c8)];
        bool row_ok = (gr < limit);
        if (vec16_ok) {
            cp_async_ca_16(sp, g_src + (size_t)gr * K + gc, row_ok && (gc < K));
        } else if (row_ok && (gc < K)) {
            const uint16_t* gp = g_src + (size_t)gr * K + gc;
            uint16_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int n_rem = K - gc;
            int n_cpy = n_rem < 8 ? n_rem : 8;
            for (int i = 0; i < n_cpy; ++i) tmp[i] = gp[i];
            *reinterpret_cast<uint4*>(sp) = *reinterpret_cast<const uint4*>(tmp);
        } else {
            *reinterpret_cast<uint4*>(sp) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int MI, int NJ>
__device__ __forceinline__ void store_mma_epilogue(
    float (&acc)[MI][NJ][4],
    float* __restrict__ C,
    int M, int N, int bm, int bn, int wm, int wn,
    int gid, int col_pair, float alpha, float beta
) {
    const bool use_vec2 = (N & 1) == 0 && (reinterpret_cast<uintptr_t>(C) & 7u) == 0u;
    #pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        int r0 = bm + wm + mi * 16 + gid;
        int r1 = r0 + 8;
        #pragma unroll
        for (int nj = 0; nj < NJ; ++nj) {
            int c0 = bn + wn + nj * 8 + col_pair;
            int c1 = c0 + 1;
            if (r0 < M) {
                if (use_vec2 && c1 < N) {
                    float2 v;
                    if (beta == 0.0f) {
                        v = make_float2(alpha * acc[mi][nj][0], alpha * acc[mi][nj][1]);
                    } else {
                        float2 old = *reinterpret_cast<const float2*>(&C[(size_t)r0 * N + c0]);
                        v = make_float2(alpha * acc[mi][nj][0] + beta * old.x, alpha * acc[mi][nj][1] + beta * old.y);
                    }
                    *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c0]) = v;
                } else {
                    if (c0 < N) C[(size_t)r0 * N + c0] = (beta == 0.0f) ? (alpha * acc[mi][nj][0]) : (alpha * acc[mi][nj][0] + beta * C[(size_t)r0 * N + c0]);
                    if (c1 < N) C[(size_t)r0 * N + c1] = (beta == 0.0f) ? (alpha * acc[mi][nj][1]) : (alpha * acc[mi][nj][1] + beta * C[(size_t)r0 * N + c1]);
                }
            }
            if (r1 < M) {
                if (use_vec2 && c1 < N) {
                    float2 v;
                    if (beta == 0.0f) {
                        v = make_float2(alpha * acc[mi][nj][2], alpha * acc[mi][nj][3]);
                    } else {
                        float2 old = *reinterpret_cast<const float2*>(&C[(size_t)r1 * N + c0]);
                        v = make_float2(alpha * acc[mi][nj][2] + beta * old.x, alpha * acc[mi][nj][3] + beta * old.y);
                    }
                    *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c0]) = v;
                } else {
                    if (c0 < N) C[(size_t)r1 * N + c0] = (beta == 0.0f) ? (alpha * acc[mi][nj][2]) : (alpha * acc[mi][nj][2] + beta * C[(size_t)r1 * N + c0]);
                    if (c1 < N) C[(size_t)r1 * N + c1] = (beta == 0.0f) ? (alpha * acc[mi][nj][3]) : (alpha * acc[mi][nj][3] + beta * C[(size_t)r1 * N + c1]);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128, 3) gemm_f16_mma_nt_32x64_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha, float beta
) {
    constexpr int BM = 32, BN = 64, BK = 32;
    __shared__ __align__(16) uint16_t sA[3][BM * BK];
    __shared__ __align__(16) uint16_t sB[3][BN * BK];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = 0;
    const int wn = warp * 16;

    const bool a_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(A) & 15u) == 0u;
    const bool b_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;

    float acc[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto load_tile = [&](int stage, int k0) {
        load_tile_swz32<BM, 128>(sA[stage], A, bm, M, K, k0, tid, a_vec);
        load_tile_swz32<BN, 128>(sB[stage], B, bn, N, K, k0, tid, b_vec);
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int n_tiles = (K + BK - 1) / BK;
    if (n_tiles <= 0) return;
    load_tile(0, 0);
    if (n_tiles > 1) load_tile(1, BK);

    for (int kt = 0; kt < n_tiles; ++kt) {
        if (kt + 2 < n_tiles) {
            load_tile((kt + 2) % 3, (kt + 2) * BK);
            asm volatile("cp.async.wait_group 2;\n" ::);
        } else if (kt + 1 < n_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int stage = kt % 3;
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 16;
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int row_a = wm + mi * 16 + (lane & 15);
                int col_a = k_off + ((lane >> 4) * 8);
                ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    &sA[stage][row_a * BK + swz_col32(row_a, col_a)]
                );
            }
            int row_b = wn + ((lane >> 4) * 8) + (lane & 7);
            int col_b = k_off + ((lane >> 3) & 1) * 8;
            uint32_t b0_0, b0_1, b1_0, b1_1;
            ldmatrix_x4(
                b0_0, b0_1, b1_0, b1_1,
                &sB[stage][row_b * BK + swz_col32(row_b, col_b)]
            );
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                mma16816_f32_f16(
                    acc[mi][0],
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    b0_0, b0_1
                );
                mma16816_f32_f16(
                    acc[mi][1],
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    b1_0, b1_1
                );
            }
        }
        __syncthreads();
    }

    store_mma_epilogue<2, 2>(acc, C, M, N, bm, bn, wm, wn, lane >> 2, (lane & 3) * 2, alpha, beta);
}

__global__ void __launch_bounds__(128, 2) gemm_f16_mma_nt_64x64_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha, float beta
) {
    constexpr int BM = 64, BN = 64, BK = 32;
    __shared__ __align__(16) uint16_t sA[3][BM * BK];
    __shared__ __align__(16) uint16_t sB[3][BN * BK];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp >> 1) * 32;
    const int wn = (warp & 1) * 32;

    const bool a_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(A) & 15u) == 0u;
    const bool b_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;

    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto load_tile = [&](int stage, int k0) {
        load_tile_swz32<BM, 128>(sA[stage], A, bm, M, K, k0, tid, a_vec);
        load_tile_swz32<BN, 128>(sB[stage], B, bn, N, K, k0, tid, b_vec);
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int n_tiles = (K + BK - 1) / BK;
    if (n_tiles <= 0) return;
    load_tile(0, 0);
    if (n_tiles > 1) load_tile(1, BK);

    for (int kt = 0; kt < n_tiles; ++kt) {
        if (kt + 2 < n_tiles) {
            load_tile((kt + 2) % 3, (kt + 2) * BK);
            asm volatile("cp.async.wait_group 2;\n" ::);
        } else if (kt + 1 < n_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int stage = kt % 3;
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 16;
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int row_a = wm + mi * 16 + (lane & 15);
                int col_a = k_off + ((lane >> 4) * 8);
                ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    &sA[stage][row_a * BK + swz_col32(row_a, col_a)]
                );
            }
            uint32_t b_frag[4][2];
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                int row_b = wn + np * 16 + ((lane >> 4) * 8) + (lane & 7);
                int col_b = k_off + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(
                    b_frag[np * 2 + 0][0], b_frag[np * 2 + 0][1],
                    b_frag[np * 2 + 1][0], b_frag[np * 2 + 1][1],
                    &sB[stage][row_b * BK + swz_col32(row_b, col_b)]
                );
            }
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma16816_f32_f16(
                        acc[mi][nj],
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b_frag[nj][0], b_frag[nj][1]
                    );
                }
            }
        }
        __syncthreads();
    }

    store_mma_epilogue<2, 4>(acc, C, M, N, bm, bn, wm, wn, lane >> 2, (lane & 3) * 2, alpha, beta);
}

__global__ void __launch_bounds__(128, 2) gemm_f16_mma_nt_64x128_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha, float beta
) {
    constexpr int BM = 64, BN = 128, BK = 32;
    __shared__ __align__(16) uint16_t sA[3][BM * BK];
    __shared__ __align__(16) uint16_t sB[3][BN * BK];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp >> 1) * 32;
    const int wn = (warp & 1) * 64;

    const bool a_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(A) & 15u) == 0u;
    const bool b_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto load_tile = [&](int stage, int k0) {
        load_tile_swz32<BM, 128>(sA[stage], A, bm, M, K, k0, tid, a_vec);
        load_tile_swz32<BN, 128>(sB[stage], B, bn, N, K, k0, tid, b_vec);
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int n_tiles = (K + BK - 1) / BK;
    if (n_tiles <= 0) return;
    load_tile(0, 0);
    if (n_tiles > 1) load_tile(1, BK);

    for (int kt = 0; kt < n_tiles; ++kt) {
        if (kt + 2 < n_tiles) {
            load_tile((kt + 2) % 3, (kt + 2) * BK);
            asm volatile("cp.async.wait_group 2;\n" ::);
        } else if (kt + 1 < n_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int stage = kt % 3;
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 16;
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int row_a = wm + mi * 16 + (lane & 15);
                int col_a = k_off + ((lane >> 4) * 8);
                ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    &sA[stage][row_a * BK + swz_col32(row_a, col_a)]
                );
            }
            #pragma unroll
            for (int np = 0; np < 4; ++np) {
                int row_b = wn + np * 16 + ((lane >> 4) * 8) + (lane & 7);
                int col_b = k_off + ((lane >> 3) & 1) * 8;
                uint32_t b0_0, b0_1, b1_0, b1_1;
                ldmatrix_x4(
                    b0_0, b0_1, b1_0, b1_1,
                    &sB[stage][row_b * BK + swz_col32(row_b, col_b)]
                );
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma16816_f32_f16(
                        acc[mi][np * 2 + 0],
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b0_0, b0_1
                    );
                    mma16816_f32_f16(
                        acc[mi][np * 2 + 1],
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b1_0, b1_1
                    );
                }
            }
        }
        __syncthreads();
    }

    store_mma_epilogue<2, 8>(acc, C, M, N, bm, bn, wm, wn, lane >> 2, (lane & 3) * 2, alpha, beta);
}

__global__ void __launch_bounds__(256, 1) gemm_f16_mma_nt_128x128_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha, float beta
) {
    constexpr int BM = 128, BN = 128, BK = 32;
    __shared__ __align__(16) uint16_t sA[3][BM * BK];
    __shared__ __align__(16) uint16_t sB[3][BN * BK];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5; // 0..7 in 2x4 grid
    const int lane = tid & 31;
    const int wm = (warp >> 2) * 64; // 0 or 64 (4x M16)
    const int wn = (warp & 3) * 32;  // 0, 32, 64, 96 (4x N8 = 2x N16)

    const bool a_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(A) & 15u) == 0u;
    const bool b_vec = ((K & 7) == 0) && (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto load_tile = [&](int stage, int k0) {
        load_tile_swz32<BM, 256>(sA[stage], A, bm, M, K, k0, tid, a_vec);
        load_tile_swz32<BN, 256>(sB[stage], B, bn, N, K, k0, tid, b_vec);
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int n_tiles = (K + BK - 1) / BK;
    if (n_tiles <= 0) return;
    load_tile(0, 0);
    if (n_tiles > 1) load_tile(1, BK);

    for (int kt = 0; kt < n_tiles; ++kt) {
        if (kt + 2 < n_tiles) {
            load_tile((kt + 2) % 3, (kt + 2) * BK);
            asm volatile("cp.async.wait_group 2;\n" ::);
        } else if (kt + 1 < n_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int stage = kt % 3;
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 16;
            uint32_t a_frag[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row_a = wm + mi * 16 + (lane & 15);
                int col_a = k_off + ((lane >> 4) * 8);
                ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    &sA[stage][row_a * BK + swz_col32(row_a, col_a)]
                );
            }
            uint32_t b_frag[4][2];
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                int row_b = wn + np * 16 + ((lane >> 4) * 8) + (lane & 7);
                int col_b = k_off + ((lane >> 3) & 1) * 8;
                ldmatrix_x4(
                    b_frag[np * 2 + 0][0], b_frag[np * 2 + 0][1],
                    b_frag[np * 2 + 1][0], b_frag[np * 2 + 1][1],
                    &sB[stage][row_b * BK + swz_col32(row_b, col_b)]
                );
            }
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma16816_f32_f16(
                        acc[mi][nj],
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b_frag[nj][0], b_frag[nj][1]
                    );
                }
            }
        }
        __syncthreads();
    }

    store_mma_epilogue<4, 4>(acc, C, M, N, bm, bn, wm, wn, lane >> 2, (lane & 3) * 2, alpha, beta);
}

extern "C" void launch_gemm_f16_mma_nt(
    const void* A_f16,
    const void* B_f16,
    float* C_f32,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int g128 = ((N + 127) / 128) * ((M + 127) / 128);
    int g64x128 = ((N + 127) / 128) * ((M + 63) / 64);
    if (M <= 32 && (K & 15) == 0) {
        dim3 grid((N + 63) / 64, 1);
        dim3 block(128);
        gemm_f16_mma_nt_32x64_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(A_f16),
            reinterpret_cast<const uint16_t*>(B_f16),
            C_f32, M, N, K, alpha, beta
        );
    } else if (M >= 192 && N >= 128 && (K & 15) == 0 && g128 >= 80) {
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        dim3 block(256);
        gemm_f16_mma_nt_128x128_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(A_f16),
            reinterpret_cast<const uint16_t*>(B_f16),
            C_f32, M, N, K, alpha, beta
        );
    } else if (N >= 128 && (K & 15) == 0 && g64x128 >= 80) {
        dim3 grid((N + 127) / 128, (M + 63) / 64);
        dim3 block(128);
        gemm_f16_mma_nt_64x128_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(A_f16),
            reinterpret_cast<const uint16_t*>(B_f16),
            C_f32, M, N, K, alpha, beta
        );
    } else {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        dim3 block(128);
        gemm_f16_mma_nt_64x64_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(A_f16),
            reinterpret_cast<const uint16_t*>(B_f16),
            C_f32, M, N, K, alpha, beta
        );
    }
}

__global__ void glm5_gather_rows_f16_kernel(
    const float* __restrict__ src,     // [M_tot, K] F32
    const int32_t* __restrict__ rows,  // [nt]
    uint16_t* __restrict__ dst_f16,    // [nt, K] F16
    int nt,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * K) return;
    int t = idx / K;
    int c = idx % K;
    int r = rows[t];
    dst_f16[idx] = f32_to_fp16_bits(src[(size_t)r * K + c]);
}

__global__ void glm5_gather_rows_f16_vec4_kernel(
    const float* __restrict__ src,
    const int32_t* __restrict__ rows,
    uint16_t* __restrict__ dst_f16,
    int nt,
    int k4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * k4) return;
    int t = idx / k4;
    int c4 = idx - t * k4;
    int r = rows[t];
    float4 v = reinterpret_cast<const float4*>(src + (size_t)r * (k4 * 4))[c4];
    uint32_t h01 = (uint32_t)f32_to_fp16_bits(v.x) | ((uint32_t)f32_to_fp16_bits(v.y) << 16);
    uint32_t h23 = (uint32_t)f32_to_fp16_bits(v.z) | ((uint32_t)f32_to_fp16_bits(v.w) << 16);
    reinterpret_cast<uint2*>(dst_f16 + (size_t)idx * 4)[0] = make_uint2(h01, h23);
}

__global__ void glm5_swiglu_clamp_to_f16_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float lim,
    int n,
    uint16_t* __restrict__ out_f16
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float h = glm_swiglu_clamp_val(gate[idx], up[idx], lim);
    out_f16[idx] = f32_to_fp16_bits(h);
}

__global__ void glm5_scatter_weighted_add_f32_kernel(
    const float* __restrict__ src,     // [nt, K]
    const int32_t* __restrict__ rows,  // [nt]
    const float* __restrict__ weights, // [nt]
    float* __restrict__ dst,           // [M_tot, K]
    int nt,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * K) return;
    int t = idx / K;
    int c = idx % K;
    int r = rows[t];
    dst[(size_t)r * K + c] += weights[t] * src[idx];
}

__global__ void glm5_scatter_weighted_add_f32_vec4_kernel(
    const float* __restrict__ src,
    const int32_t* __restrict__ rows,
    const float* __restrict__ weights,
    float* __restrict__ dst,
    int nt,
    int k4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * k4) return;
    int t = idx / k4;
    int c4 = idx - t * k4;
    int r = rows[t];
    float w = weights[t];
    float4 sv = reinterpret_cast<const float4*>(src)[idx];
    float4* dp = reinterpret_cast<float4*>(dst + (size_t)r * (k4 * 4)) + c4;
    float4 dv = *dp;
    dv.x += w * sv.x;
    dv.y += w * sv.y;
    dv.z += w * sv.z;
    dv.w += w * sv.w;
    *dp = dv;
}

extern "C" void launch_glm5_gather_rows_f16(
    const float* src,
    const int32_t* rows,
    void* dst_f16,
    int nt,
    int K,
    cudaStream_t stream
) {
    int tot = nt * K;
    if (tot <= 0) return;
    if ((K & 3) == 0 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u && (reinterpret_cast<uintptr_t>(dst_f16) & 7u) == 0u) {
        int tot4 = nt * (K >> 2);
        glm5_gather_rows_f16_vec4_kernel<<<(tot4 + 255) / 256, 256, 0, stream>>>(
            src, rows, reinterpret_cast<uint16_t*>(dst_f16), nt, K >> 2
        );
        return;
    }
    glm5_gather_rows_f16_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(
        src, rows, reinterpret_cast<uint16_t*>(dst_f16), nt, K
    );
}

__global__ void glm5_swiglu_clamp_interleaved_to_f16_kernel(
    const float* __restrict__ gu,     // [nt, 2 * n_ff]
    float lim,
    int nt,
    int n_ff,
    uint16_t* __restrict__ out_f16    // [nt, n_ff]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * n_ff) return;
    int t = idx / n_ff;
    int c = idx - t * n_ff;
    const float* row = gu + (size_t)t * (2 * n_ff);
    float h = glm_swiglu_clamp_val(row[c], row[n_ff + c], lim);
    out_f16[idx] = f32_to_fp16_bits(h);
}

__global__ void glm5_swiglu_clamp_interleaved_to_f16_vec4_kernel(
    const float* __restrict__ gu,
    float lim,
    int nt,
    int n_ff4,
    uint16_t* __restrict__ out_f16
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nt * n_ff4) return;
    int t = idx / n_ff4;
    int c4 = idx - t * n_ff4;
    int n_ff = n_ff4 * 4;
    const float* row = gu + (size_t)t * (2 * n_ff);
    float4 g = reinterpret_cast<const float4*>(row)[c4];
    float4 u = reinterpret_cast<const float4*>(row + n_ff)[c4];
    float h0 = glm_swiglu_clamp_val(g.x, u.x, lim);
    float h1 = glm_swiglu_clamp_val(g.y, u.y, lim);
    float h2 = glm_swiglu_clamp_val(g.z, u.z, lim);
    float h3 = glm_swiglu_clamp_val(g.w, u.w, lim);
    uint32_t h01 = (uint32_t)f32_to_fp16_bits(h0) | ((uint32_t)f32_to_fp16_bits(h1) << 16);
    uint32_t h23 = (uint32_t)f32_to_fp16_bits(h2) | ((uint32_t)f32_to_fp16_bits(h3) << 16);
    reinterpret_cast<uint2*>(out_f16)[idx] = make_uint2(h01, h23);
}

extern "C" void launch_glm5_swiglu_clamp_to_f16(
    const float* gate,
    const float* up,
    float lim,
    int n,
    void* out_f16,
    cudaStream_t stream
) {
    if (n <= 0) return;
    glm5_swiglu_clamp_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        gate, up, lim, n, reinterpret_cast<uint16_t*>(out_f16)
    );
}

extern "C" void launch_glm5_swiglu_clamp_interleaved_to_f16(
    const float* gu,
    float lim,
    int nt,
    int n_ff,
    void* out_f16,
    cudaStream_t stream
) {
    int n = nt * n_ff;
    if (n <= 0) return;
    if ((n_ff & 3) == 0 && (reinterpret_cast<uintptr_t>(gu) & 15u) == 0u && (reinterpret_cast<uintptr_t>(out_f16) & 7u) == 0u) {
        int n4 = nt * (n_ff >> 2);
        glm5_swiglu_clamp_interleaved_to_f16_vec4_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
            gu, lim, nt, n_ff >> 2, reinterpret_cast<uint16_t*>(out_f16)
        );
        return;
    }
    glm5_swiglu_clamp_interleaved_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        gu, lim, nt, n_ff, reinterpret_cast<uint16_t*>(out_f16)
    );
}

extern "C" void launch_glm5_scatter_weighted_add_f32(
    const float* src,
    const int32_t* rows,
    const float* weights,
    float* dst,
    int nt,
    int K,
    cudaStream_t stream
) {
    int tot = nt * K;
    if (tot <= 0) return;
    if ((K & 3) == 0 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u && (reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) {
        int tot4 = nt * (K >> 2);
        glm5_scatter_weighted_add_f32_vec4_kernel<<<(tot4 + 255) / 256, 256, 0, stream>>>(
            src, rows, weights, dst, nt, K >> 2
        );
        return;
    }
    glm5_scatter_weighted_add_f32_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(
        src, rows, weights, dst, nt, K
    );
}

__global__ void glm5_scale_f32_kernel(float* __restrict__ x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= scale;
}

extern "C" void launch_glm5_scale_f32(float* x, float scale, int n, cudaStream_t stream) {
    if (n <= 0) return;
    glm5_scale_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, scale, n);
}

__global__ void __launch_bounds__(256) glm5_mtp_in_batch_f16_kernel(
    const float* __restrict__ emb,    // [M, n_embd]
    const float* __restrict__ h,      // [M, n_embd]
    const float* __restrict__ enorm,  // [n_embd]
    const float* __restrict__ hnorm,  // [n_embd]
    float eps,
    int n_embd,
    uint16_t* __restrict__ cat_f16    // [M, 2 * n_embd]
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_re[8], s_rh[8], s_ie, s_ih;

    const float* emb_t = emb + (size_t)t * n_embd;
    const float* h_t = h + (size_t)t * n_embd;
    uint16_t* dst_t = cat_f16 + (size_t)t * (2 * n_embd);

    float sse = 0.0f, ssh = 0.0f;
    for (int e = tid; e < n_embd; e += 256) {
        float ve = emb_t[e], vh = h_t[e];
        sse += ve * ve;
        ssh += vh * vh;
    }
    sse = warp_reduce_sum(sse);
    ssh = warp_reduce_sum(ssh);
    if (lane == 0) { s_re[warp] = sse; s_rh[warp] = ssh; }
    __syncthreads();
    if (tid == 0) {
        float te = 0.0f, th = 0.0f;
        for (int w = 0; w < 8; ++w) { te += s_re[w]; th += s_rh[w]; }
        s_ie = rsqrtf(te / (float)n_embd + eps);
        s_ih = rsqrtf(th / (float)n_embd + eps);
    }
    __syncthreads();

    float ie = s_ie, ih = s_ih;
    for (int e = tid; e < n_embd; e += 256) {
        dst_t[e] = f32_to_fp16_bits(emb_t[e] * ie * enorm[e]);
        dst_t[n_embd + e] = f32_to_fp16_bits(h_t[e] * ih * hnorm[e]);
    }
}

extern "C" void launch_glm5_mtp_in_batch_f16(
    const float* emb,
    const float* h,
    const float* enorm,
    const float* hnorm,
    float eps,
    int n_embd,
    int M,
    void* cat_f16,
    cudaStream_t stream
) {
    if (M <= 0 || n_embd <= 0) return;
    glm5_mtp_in_batch_f16_kernel<<<M, 256, 0, stream>>>(
        emb, h, enorm, hnorm, eps, n_embd, reinterpret_cast<uint16_t*>(cat_f16)
    );
}
