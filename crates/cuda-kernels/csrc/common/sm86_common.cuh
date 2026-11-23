#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include "iq_codebooks.cuh"

// GGML type IDs matching ampllm_gguf::GgmlType
enum AmpllmGgmlType : int32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_Q2_0    = 42,
    GGML_TYPE_F8_E4M3 = 43,
};

#define QK4_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32
#define QK4_NL 32
#define QK2_0 64
#define QK_NVFP4 64
#define QK_K 256
#define K_SCALE_SIZE 12

struct __attribute__((packed)) block_q4_0 {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
};

struct __attribute__((packed)) block_q5_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
};

struct __attribute__((packed)) block_q2_0 {
    uint16_t d;
    uint8_t qs[QK2_0 / 4];
};

struct __attribute__((packed)) block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

struct __attribute__((packed)) block_q8_1 {
    uint16_t d;
    uint16_t s;
    int8_t qs[QK8_1];
};

struct __attribute__((packed)) block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

struct __attribute__((packed)) block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};

struct __attribute__((packed)) block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};

struct __attribute__((packed)) block_iq4_nl {
    uint16_t d;
    uint8_t qs[QK4_NL / 2];
};

struct __attribute__((packed)) block_iq4_xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
};

struct __attribute__((packed)) block_nvfp4 {
    uint8_t d[4];
    uint8_t qs[QK_NVFP4 / 2];
};

static __device__ __constant__ int8_t KVALUES_IQ4NL_DEV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static __device__ __constant__ int8_t KVALUES_NVFP4_DEV[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

__device__ __forceinline__ float ue4m3_to_f32_dev(uint8_t x) {
    uint32_t e = (x >> 3) & 0xFu;
    uint32_t m = x & 7u;
    float norm_f = __uint_as_float(((e + 119u) << 23) | (m << 20));
    float sub_f = (float)m * 0x1.0p-10f;
    float v = (e == 0u) ? sub_f : norm_f;
    return (x == 0x7Fu) ? 0.0f : v;
}

__device__ __forceinline__ float e4m3fn_to_f32(uint8_t x) {
    const uint32_t ax = x & 0x7Fu;
    if (ax == 0x7Fu) return 0.0f;
    const uint32_t s = (uint32_t)(x & 0x80u) << 24;
    const uint32_t e = (ax >> 3) & 0xFu;
    const uint32_t m = ax & 7u;
    if (e == 0u) {
        const float sub = (float)m * 0x1.0p-9f;
        return s ? -sub : sub;
    }
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ uint8_t f32_to_e4m3fn(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#else
    const uint32_t u = __float_as_uint(v);
    const uint8_t sign = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t ax = u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return sign | 0x7Fu;
    if (ax >= 0x43E00000u) return sign | 0x7Eu;
    if (ax < 0x3C800000u) {
        const uint32_t sub = __float_as_uint(__fadd_rn(__uint_as_float(ax), 16384.0f)) - 0x46800000u;
        return sign | (uint8_t)sub;
    }
    const uint32_t lsb = (ax >> 20) & 1u;
    return sign | (uint8_t)(((ax + 0x7FFFFu + lsb) >> 20) - (120u << 3));
#endif
}

__device__ __forceinline__ void e4m3x4_h4(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t t = w & 0x78787878u;
    uint32_t z = (t - 0x01010101u) & ~t & 0x80808080u;
    z -= z >> 7;
    const uint32_t v = w & 0x07070707u, u = v | (v >> 4);
    const uint32_t dd = __byte_perm(0x070A1140u, 0x01020304u, __byte_perm(u, 0u, 0x4420u)) & z;
    const uint32_t x0 = __byte_perm(w, 0u, 0x1404u), x1 = __byte_perm(w, 0u, 0x3424u);
    const uint32_t y0 = __byte_perm(dd, 0u, 0x1404u), y1 = __byte_perm(dd, 0u, 0x3424u);
    lo = x0 - (((x0 & 0x7F007F00u) + y0) >> 1) + 0x20002000u;
    hi = x1 - (((x1 & 0x7F007F00u) + y1) >> 1) + 0x20002000u;
}

__device__ __forceinline__ float fp16_bits_to_f32(uint16_t bits) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = bits;
    return __half2float(h);
}

__device__ __forceinline__ uint16_t f32_to_fp16_bits(float v) {
    __half h = __float2half_rn(v);
    return *reinterpret_cast<uint16_t*>(&h);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    __nv_bfloat16 b;
    *reinterpret_cast<uint16_t*>(&b) = bits;
    return __bfloat162float(b);
}

__device__ __forceinline__ int dp4a_packed(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    return c;
#endif
}

__device__ __forceinline__ uint32_t h2_mul(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ void mma16816_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
    );
#else
    (void)c; (void)a; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ uint32_t ue4m3_h2(uint32_t x) {
    const uint32_t e = (x >> 3) & 15u, m = x & 7u;
    const uint32_t sub = __byte_perm(0x1a181400u, 0x1f1e1d1cu, m) << 8;
    uint32_t h = e ? (((e + 7u) << 10) | (m << 7)) : sub;
    h = (x == 0x7Fu) ? 0u : h;
    return h | (h << 16);
}

__device__ __forceinline__ uint32_t nv_pair_h2(uint32_t s, uint32_t scale) {
    const uint32_t mag = __byte_perm(0x42403c00u, 0x4a484644u, (s & 0x0707u) << 4);
    const uint32_t sign = ((s & 0x0808u) * 0x101000u) & 0x80008000u;
    return h2_mul(mag | sign, scale);
}

struct NvMmaStep {
    uint32_t sa, sb, wa0, wa1, wb0, wb1;
};

__device__ __forceinline__ NvMmaStep nv_mma_step_load(const uint8_t* a, const uint8_t* b, int k, int t) {
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(a + (size_t)k * 36);
    const uint32_t* bp = reinterpret_cast<const uint32_t*>(b + (size_t)k * 36);
    NvMmaStep r;
    r.sa = __ldg(ap);
    r.sb = __ldg(bp);
    r.wa0 = __ldg(ap + 1 + 2 * t);
    r.wa1 = __ldg(ap + 2 + 2 * t);
    r.wb0 = __ldg(bp + 1 + 2 * t);
    r.wb1 = __ldg(bp + 2 + 2 * t);
    return r;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ void get_scale_min_k4_dev(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
    }
}

__device__ __forceinline__ int pack_i8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int)(uint8_t)a) | (((int)(uint8_t)b) << 8) | (((int)(uint8_t)c) << 16) | (((int)(uint8_t)d) << 24);
}

__device__ __forceinline__ int load_i32_align4(const void* ptr) {
    return *reinterpret_cast<const int*>(ptr);
}

__device__ __forceinline__ int load_i32_align2(const void* ptr) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr);
    return (int)p[0] | ((int)p[1] << 16);
}

__device__ __forceinline__ uint16_t load_u16_align2(const void* ptr) {
    return *reinterpret_cast<const uint16_t*>(ptr);
}

__device__ __forceinline__ float2 fp16x2_bits_to_f32(const void* ptr) {
    return __half22float2(*reinterpret_cast<const __half2*>(ptr));
}

__device__ __forceinline__ int load_i32_unaligned(const void* ptr) {
    return load_i32_align2(ptr);
}

__device__ __forceinline__ uint16_t load_u16_unaligned(const void* ptr) {
    return load_u16_align2(ptr);
}

// Pure-register IQ4_NL nibble LUT using __byte_perm + LOP3 (zero constant-cache serialization).
__device__ __forceinline__ int lut_iq4nl_x4(uint32_t nibs) {
    uint32_t t = (nibs & 0x00070007u) | ((nibs >> 4) & 0x00700070u);
    uint32_t sel = t | (t >> 8);
    uint32_t lo8 = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel);
    uint32_t hi8 = __byte_perm(0x26190D01u, 0x71594535u, sel);
    uint32_t mask = ((nibs & 0x08080808u) >> 3) * 0xFFu;
    return (int)((hi8 & mask) | (lo8 & ~mask));
}

// Pure-register NVFP4 E2M1 nibble LUT using __byte_perm (maps 4 nibbles -> 4 int8 in {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}).
__device__ __forceinline__ int lut_nvfp4_x4(uint32_t nibs) {
    uint32_t t = (nibs & 0x00070007u) | ((nibs >> 4) & 0x00700070u);
    uint32_t sel = t | (t >> 8);
    uint32_t pos8 = __byte_perm(0x03020100u, 0x0C080604u, sel);
    uint32_t neg8 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel);
    uint32_t mask = ((nibs & 0x08080808u) >> 3) * 0xFFu;
    return (int)((neg8 & mask) | (pos8 & ~mask));
}

__device__ __forceinline__ float dot_nvfp4_q8_1(const block_nvfp4* w, const block_q8_1* x) {
    uint32_t d4 = (uint32_t)load_i32_align4(w->d);
    float acc0 = 0.0f;
    #pragma unroll
    for (int sb = 0; sb < 2; ++sb) {
        uint32_t w0 = (uint32_t)load_i32_align4(w->qs + sb * 8);
        uint32_t w1 = (uint32_t)load_i32_align4(w->qs + sb * 8 + 4);
        const int8_t* xq = x[0].qs + sb * 16;
        int sumi = 0;
        sumi = dp4a_packed(lut_nvfp4_x4(w0 & 0x0F0F0F0Fu),        load_i32_align4(xq + 0),  sumi);
        sumi = dp4a_packed(lut_nvfp4_x4(w1 & 0x0F0F0F0Fu),        load_i32_align4(xq + 4),  sumi);
        sumi = dp4a_packed(lut_nvfp4_x4((w0 >> 4) & 0x0F0F0F0Fu), load_i32_align4(xq + 8),  sumi);
        sumi = dp4a_packed(lut_nvfp4_x4((w1 >> 4) & 0x0F0F0F0Fu), load_i32_align4(xq + 12), sumi);
        acc0 += ue4m3_to_f32_dev((uint8_t)(d4 >> (sb * 8))) * (float)sumi;
    }
    float acc1 = 0.0f;
    #pragma unroll
    for (int sb = 2; sb < 4; ++sb) {
        uint32_t w0 = (uint32_t)load_i32_align4(w->qs + sb * 8);
        uint32_t w1 = (uint32_t)load_i32_align4(w->qs + sb * 8 + 4);
        const int8_t* xq = x[1].qs + (sb - 2) * 16;
        int sumi = 0;
        sumi = dp4a_packed(lut_nvfp4_x4(w0 & 0x0F0F0F0Fu),        load_i32_align4(xq + 0),  sumi);
        sumi = dp4a_packed(lut_nvfp4_x4(w1 & 0x0F0F0F0Fu),        load_i32_align4(xq + 4),  sumi);
        sumi = dp4a_packed(lut_nvfp4_x4((w0 >> 4) & 0x0F0F0F0Fu), load_i32_align4(xq + 8),  sumi);
        sumi = dp4a_packed(lut_nvfp4_x4((w1 >> 4) & 0x0F0F0F0Fu), load_i32_align4(xq + 12), sumi);
        acc1 += ue4m3_to_f32_dev((uint8_t)(d4 >> (sb * 8))) * (float)sumi;
    }
    return acc0 * fp16_bits_to_f32(load_u16_align2(&x[0].d))
         + acc1 * fp16_bits_to_f32(load_u16_align2(&x[1].d));
}

// Coalesced word-per-lane sub-block dot product helpers:
// Every lane in the 32-lane warp loads ONE contiguous 32-bit `qs` word (`4` bytes) per iteration!
// - NVFP4: 8 lanes per block_nvfp4 (`4` blocks = 256 elements per warp iteration)
// - Q8_0:  8 lanes per block_q8_0  (`4` blocks = 128 elements per warp iteration)
// - IQ4_NL: 4 lanes per block_iq4_nl (`8` blocks = 256 elements per warp iteration)
// - Q4_0:  4 lanes per block_q4_0  (`8` blocks = 256 elements per warp iteration)
__device__ __forceinline__ float dot_row_nvfp4_sub(
    const block_nvfp4* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K >> 6; // K / 64
    int sub = lane & 7; // 0..7 (8 uint32_t words per block_nvfp4)
    int blk0 = lane >> 3; // 0..3
    int sb = sub >> 1; // 0..3
    int h4 = sub & 1;  // 0..1
    int x_q8_off = sb >> 1; // 0 or 1
    int x_byte_lo = (sb & 1) * 16 + h4 * 4;
    int x_byte_hi = x_byte_lo + 8;
    int scale_shift = sb * 8;

    float acc = 0.0f;
    for (int b = blk0; b < nb; b += 4) {
        const block_nvfp4* wb = w_row + b;
        uint32_t d4 = (uint32_t)load_i32_align4(wb->d);
        uint32_t qw = (uint32_t)load_i32_align4(wb->qs + sub * 4);
        const block_q8_1* xb = x_q8 + b * 2 + x_q8_off;
        int sumi = 0;
        sumi = dp4a_packed(lut_nvfp4_x4(qw & 0x0F0F0F0Fu),        load_i32_align4(xb->qs + x_byte_lo), sumi);
        sumi = dp4a_packed(lut_nvfp4_x4((qw >> 4) & 0x0F0F0F0Fu), load_i32_align4(xb->qs + x_byte_hi), sumi);
        float sc = ue4m3_to_f32_dev((uint8_t)(d4 >> scale_shift)) * fp16_bits_to_f32(load_u16_align2(&xb->d));
        acc += sc * (float)sumi;
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_nvfp4_sub(
    const block_nvfp4* __restrict__ w_row0,
    const block_nvfp4* __restrict__ w_row1,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 6; // K / 64
    int sub = lane & 7; // 0..7
    int blk0 = lane >> 3; // 0..3
    int sb = sub >> 1;
    int h4 = sub & 1;
    int x_q8_off = sb >> 1;
    int x_byte_lo = (sb & 1) * 16 + h4 * 4;
    int x_byte_hi = x_byte_lo + 8;
    int scale_shift = sb * 8;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int b = blk0; b < nb; b += 4) {
        const block_q8_1* xb = x_q8 + b * 2 + x_q8_off;
        int xlo = load_i32_align4(xb->qs + x_byte_lo);
        int xhi = load_i32_align4(xb->qs + x_byte_hi);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));

        const block_nvfp4* wb0 = w_row0 + b;
        uint32_t d4_0 = (uint32_t)load_i32_align4(wb0->d);
        uint32_t qw0  = (uint32_t)load_i32_align4(wb0->qs + sub * 4);
        int s0 = 0;
        s0 = dp4a_packed(lut_nvfp4_x4(qw0 & 0x0F0F0F0Fu),        xlo, s0);
        s0 = dp4a_packed(lut_nvfp4_x4((qw0 >> 4) & 0x0F0F0F0Fu), xhi, s0);
        acc0 += (ue4m3_to_f32_dev((uint8_t)(d4_0 >> scale_shift)) * xd) * (float)s0;

        const block_nvfp4* wb1 = w_row1 + b;
        uint32_t d4_1 = (uint32_t)load_i32_align4(wb1->d);
        uint32_t qw1  = (uint32_t)load_i32_align4(wb1->qs + sub * 4);
        int s1 = 0;
        s1 = dp4a_packed(lut_nvfp4_x4(qw1 & 0x0F0F0F0Fu),        xlo, s1);
        s1 = dp4a_packed(lut_nvfp4_x4((qw1 >> 4) & 0x0F0F0F0Fu), xhi, s1);
        acc1 += (ue4m3_to_f32_dev((uint8_t)(d4_1 >> scale_shift)) * xd) * (float)s1;
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}


__device__ __forceinline__ float dot_row_q8_0_sub(
    const block_q8_0* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K >> 5; // K / 32
    int sub = lane & 7; // 0..7 (8 int32 words per block_q8_0)
    int blk0 = lane >> 3; // 0..3
    float acc = 0.0f;
    for (int b = blk0; b < nb; b += 4) {
        const block_q8_0* wb = w_row + b;
        const block_q8_1* xb = x_q8 + b;
        int wa = load_i32_align2(wb->qs + sub * 4);
        int xa = load_i32_align4(xb->qs + sub * 4);
        int sumi = dp4a_packed(wa, xa, 0);
        float sc = fp16_bits_to_f32(load_u16_align2(&wb->d)) * fp16_bits_to_f32(load_u16_align2(&xb->d));
        acc += sc * (float)sumi;
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_q8_0_sub(
    const block_q8_0* __restrict__ w_row0,
    const block_q8_0* __restrict__ w_row1,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 5;
    int sub = lane & 7;
    int blk0 = lane >> 3;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int b = blk0; b < nb; b += 4) {
        const block_q8_1* xb = x_q8 + b;
        int xa = load_i32_align4(xb->qs + sub * 4);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));

        const block_q8_0* wb0 = w_row0 + b;
        int wa0 = load_i32_align2(wb0->qs + sub * 4);
        int s0 = dp4a_packed(wa0, xa, 0);
        acc0 += (fp16_bits_to_f32(load_u16_align2(&wb0->d)) * xd) * (float)s0;

        const block_q8_0* wb1 = w_row1 + b;
        int wa1 = load_i32_align2(wb1->qs + sub * 4);
        int s1 = dp4a_packed(wa1, xa, 0);
        acc1 += (fp16_bits_to_f32(load_u16_align2(&wb1->d)) * xd) * (float)s1;
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}


__device__ __forceinline__ float dot_row_iq4_nl_sub(
    const block_iq4_nl* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K >> 5; // K / 32
    int sub = lane & 3; // 0..3 (4 uint32_t words per block_iq4_nl)
    int blk0 = lane >> 2; // 0..7
    float acc = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_iq4_nl* wb = w_row + b;
        const block_q8_1* xb = x_q8 + b;
        uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
        int lo = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
        int hi = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
        int xlo = load_i32_align4(xb->qs + sub * 4);
        int xhi = load_i32_align4(xb->qs + 16 + sub * 4);
        int sumi = dp4a_packed(lo, xlo, 0);
        sumi = dp4a_packed(hi, xhi, sumi);
        float sc = fp16_bits_to_f32(load_u16_align2(&wb->d)) * fp16_bits_to_f32(load_u16_align2(&xb->d));
        acc += sc * (float)sumi;
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float dot_row_q4_0_sub(
    const block_q4_0* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K >> 5;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    float acc = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q4_0* wb = w_row + b;
        const block_q8_1* xb = x_q8 + b;
        uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
        int lo = (int)__vsub4(q4 & 0x0F0F0F0Fu, 0x08080808u);
        int hi = (int)__vsub4((q4 >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        int xlo = load_i32_align4(xb->qs + sub * 4);
        int xhi = load_i32_align4(xb->qs + 16 + sub * 4);
        int sumi = dp4a_packed(lo, xlo, 0);
        sumi = dp4a_packed(hi, xhi, sumi);
        float sc = fp16_bits_to_f32(load_u16_align2(&wb->d)) * fp16_bits_to_f32(load_u16_align2(&xb->d));
        acc += sc * (float)sumi;
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_iq4_nl_sub(
    const block_iq4_nl* __restrict__ w_row0,
    const block_iq4_nl* __restrict__ w_row1,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 5;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b;
        int xlo = load_i32_align4(xb->qs + sub * 4);
        int xhi = load_i32_align4(xb->qs + 16 + sub * 4);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));

        const block_iq4_nl* wb0 = w_row0 + b;
        uint32_t q4_0 = (uint32_t)load_i32_align2(wb0->qs + sub * 4);
        int s0 = dp4a_packed(lut_iq4nl_x4(q4_0 & 0x0F0F0F0Fu), xlo, 0);
        s0 = dp4a_packed(lut_iq4nl_x4((q4_0 >> 4) & 0x0F0F0F0Fu), xhi, s0);
        acc0 += (fp16_bits_to_f32(load_u16_align2(&wb0->d)) * xd) * (float)s0;

        const block_iq4_nl* wb1 = w_row1 + b;
        uint32_t q4_1 = (uint32_t)load_i32_align2(wb1->qs + sub * 4);
        int s1 = dp4a_packed(lut_iq4nl_x4(q4_1 & 0x0F0F0F0Fu), xlo, 0);
        s1 = dp4a_packed(lut_iq4nl_x4((q4_1 >> 4) & 0x0F0F0F0Fu), xhi, s1);
        acc1 += (fp16_bits_to_f32(load_u16_align2(&wb1->d)) * xd) * (float)s1;
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}

__device__ __forceinline__ uint32_t expand_q5_1_hi4(uint32_t hb) {
    return (((hb & 0xFu) * 0x00204081u) & 0x01010101u) << 4;
}

__device__ __forceinline__ int unpack_q2_0_x4(uint32_t raw_byte) {
    uint32_t e = (raw_byte & 0x03u)
               | ((raw_byte & 0x0Cu) << 6)
               | ((raw_byte & 0x30u) << 12)
               | ((raw_byte & 0xC0u) << 18);
    return (int)__vsub4(e, 0x01010101u);
}

__device__ __forceinline__ float dot_sub_q5_1_blk(
    const block_q5_1* wb,
    int xlo,
    int xhi,
    float2 ds8,
    int sub
) {
    uint32_t qh = (uint32_t)load_i32_align2(wb->qh);
    uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
    uint32_t lo = (q4 & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (sub * 4));
    uint32_t hi = ((q4 >> 4) & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (16 + sub * 4));
    int sumi = dp4a_packed((int)lo, xlo, 0);
    sumi = dp4a_packed((int)hi, xhi, sumi);
    float dw = fp16_bits_to_f32(load_u16_align2(&wb->d));
    float res = dw * ds8.x * (float)sumi;
    if (sub == 0) {
        res += fp16_bits_to_f32(load_u16_align2(&wb->m)) * ds8.y;
    }
    return res;
}

__device__ __forceinline__ float dot_row_q5_1_sub(
    const block_q5_1* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K >> 5;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    float acc = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b;
        int xlo = load_i32_align4(xb->qs + sub * 4);
        int xhi = load_i32_align4(xb->qs + 16 + sub * 4);
        float2 ds8 = fp16x2_bits_to_f32(&xb->d);
        acc += dot_sub_q5_1_blk(w_row + b, xlo, xhi, ds8, sub);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_q5_1_sub(
    const block_q5_1* __restrict__ w_row0,
    const block_q5_1* __restrict__ w_row1,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 5;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b;
        int xlo = load_i32_align4(xb->qs + sub * 4);
        int xhi = load_i32_align4(xb->qs + 16 + sub * 4);
        float2 ds8 = fp16x2_bits_to_f32(&xb->d);
        acc0 += dot_sub_q5_1_blk(w_row0 + b, xlo, xhi, ds8, sub);
        acc1 += dot_sub_q5_1_blk(w_row1 + b, xlo, xhi, ds8, sub);
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}

__device__ __forceinline__ void dot_quad_q5_1_sub(
    const block_q5_1* __restrict__ w0,
    const block_q5_1* __restrict__ w1,
    const block_q5_1* __restrict__ w2,
    const block_q5_1* __restrict__ w3,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& o0,
    float& o1,
    float& o2,
    float& o3
) {
    int nb = K >> 5;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b;
        int xlo = load_i32_align4(xb->qs + sub * 4);
        int xhi = load_i32_align4(xb->qs + 16 + sub * 4);
        float2 ds8 = fp16x2_bits_to_f32(&xb->d);
        a0 += dot_sub_q5_1_blk(w0 + b, xlo, xhi, ds8, sub);
        a1 += dot_sub_q5_1_blk(w1 + b, xlo, xhi, ds8, sub);
        a2 += dot_sub_q5_1_blk(w2 + b, xlo, xhi, ds8, sub);
        a3 += dot_sub_q5_1_blk(w3 + b, xlo, xhi, ds8, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ float dot_sub_q2_0_blk(
    const block_q2_0* wb,
    const int (&x4)[4],
    float xd,
    int sub
) {
    uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        sumi = dp4a_packed(unpack_q2_0_x4((q4 >> (8 * j)) & 0xFFu), x4[j], sumi);
    }
    return fp16_bits_to_f32(load_u16_align2(&wb->d)) * xd * (float)sumi;
}

__device__ __forceinline__ float dot_row_q2_0_sub(
    const block_q2_0* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K >> 6;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    int xoff = (sub & 1) * 16;
    float acc = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b * 2 + (sub >> 1);
        int x4[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) x4[j] = load_i32_align4(xb->qs + xoff + j * 4);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));
        acc += dot_sub_q2_0_blk(w_row + b, x4, xd, sub);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_q2_0_sub(
    const block_q2_0* __restrict__ w_row0,
    const block_q2_0* __restrict__ w_row1,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 6;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    int xoff = (sub & 1) * 16;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b * 2 + (sub >> 1);
        int x4[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) x4[j] = load_i32_align4(xb->qs + xoff + j * 4);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));
        acc0 += dot_sub_q2_0_blk(w_row0 + b, x4, xd, sub);
        acc1 += dot_sub_q2_0_blk(w_row1 + b, x4, xd, sub);
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}

__device__ __forceinline__ void dot_quad_q2_0_sub(
    const block_q2_0* __restrict__ w0,
    const block_q2_0* __restrict__ w1,
    const block_q2_0* __restrict__ w2,
    const block_q2_0* __restrict__ w3,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& o0,
    float& o1,
    float& o2,
    float& o3
) {
    int nb = K >> 6;
    int sub = lane & 3;
    int blk0 = lane >> 2;
    int xoff = (sub & 1) * 16;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = blk0; b < nb; b += 8) {
        const block_q8_1* xb = x_q8 + b * 2 + (sub >> 1);
        int x4[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) x4[j] = load_i32_align4(xb->qs + xoff + j * 4);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));
        a0 += dot_sub_q2_0_blk(w0 + b, x4, xd, sub);
        a1 += dot_sub_q2_0_blk(w1 + b, x4, xd, sub);
        a2 += dot_sub_q2_0_blk(w2 + b, x4, xd, sub);
        a3 += dot_sub_q2_0_blk(w3 + b, x4, xd, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ float dot_q5_1_q8_1(const block_q5_1* w, const block_q8_1* x) {
    uint32_t qh = (uint32_t)load_i32_align2(w->qh);
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t q4 = (uint32_t)load_i32_align2(w->qs + j * 4);
        uint32_t lo = (q4 & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (4 * j));
        uint32_t hi = ((q4 >> 4) & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (16 + 4 * j));
        int xlo = load_i32_align4(x->qs + j * 4);
        int xhi = load_i32_align4(x->qs + 16 + j * 4);
        sumi = dp4a_packed((int)lo, xlo, sumi);
        sumi = dp4a_packed((int)hi, xhi, sumi);
    }
    float dw = fp16_bits_to_f32(load_u16_align2(&w->d));
    float mw = fp16_bits_to_f32(load_u16_align2(&w->m));
    float2 dsx = fp16x2_bits_to_f32(&x->d);
    return dw * dsx.x * (float)sumi + mw * dsx.y;
}

__device__ __forceinline__ float dot_q2_0_q8_1(const block_q2_0* w, const block_q8_1* x) {
    int sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        sum0 = dp4a_packed(unpack_q2_0_x4(w->qs[j]),     load_i32_align4(x[0].qs + j * 4), sum0);
        sum1 = dp4a_packed(unpack_q2_0_x4(w->qs[j + 8]), load_i32_align4(x[1].qs + j * 4), sum1);
    }
    float dw = fp16_bits_to_f32(load_u16_align2(&w->d));
    float d8_0 = fp16_bits_to_f32(load_u16_align2(&x[0].d));
    float d8_1 = fp16_bits_to_f32(load_u16_align2(&x[1].d));
    return dw * (d8_0 * (float)sum0 + d8_1 * (float)sum1);
}

// Dot product helpers between 1 weight block and matching block_q8_1 activations.
__device__ __forceinline__ float dot_q8_0_q8_1(const block_q8_0* w, const block_q8_1* x) {
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int wa = load_i32_align2(w->qs + i * 4);
        int xa = load_i32_align4(x->qs + i * 4);
        sumi = dp4a_packed(wa, xa, sumi);
    }
    return fp16_bits_to_f32(load_u16_align2(&w->d)) * fp16_bits_to_f32(load_u16_align2(&x->d)) * (float)sumi;
}

__device__ __forceinline__ float dot_q4_0_q8_1(const block_q4_0* w, const block_q8_1* x) {
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t q4 = (uint32_t)load_i32_align2(w->qs + j * 4);
        int lo = (int)(q4 & 0x0F0F0F0Fu);
        int hi = (int)((q4 >> 4) & 0x0F0F0F0Fu);
        int xlo = load_i32_align4(x->qs + j * 4);
        int xhi = load_i32_align4(x->qs + 16 + j * 4);
        sumi = dp4a_packed(lo, xlo, sumi);
        sumi = dp4a_packed(hi, xhi, sumi);
    }
    float dw = fp16_bits_to_f32(load_u16_align2(&w->d));
    float2 dsx = fp16x2_bits_to_f32(&x->d);
    return dw * (dsx.x * (float)sumi - 8.0f * dsx.y);
}

__device__ __forceinline__ float dot_iq4_nl_q8_1(const block_iq4_nl* w, const block_q8_1* x) {
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t q4 = (uint32_t)load_i32_align2(w->qs + j * 4);
        int lo = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
        int hi = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
        int xlo = load_i32_align4(x->qs + j * 4);
        int xhi = load_i32_align4(x->qs + 16 + j * 4);
        sumi = dp4a_packed(lo, xlo, sumi);
        sumi = dp4a_packed(hi, xhi, sumi);
    }
    float dw = fp16_bits_to_f32(load_u16_align2(&w->d));
    float dx = fp16_bits_to_f32(load_u16_align2(&x->d));
    return dw * dx * (float)sumi;
}

__device__ __forceinline__ float dot_iq4_xs_q8_1(const block_iq4_xs* w, const block_q8_1* x) {
    float2 ws = fp16x2_bits_to_f32(&w->d); // ws.x = d, dummy ws.y
    (void)ws;
    float dw = fp16_bits_to_f32(load_u16_align2(&w->d));
    uint16_t scales_h = load_u16_align2(&w->scales_h);
    float acc = 0.0f;
    for (int ib = 0; ib < 8; ++ib) {
        int ls = ((w->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        float dl = dw * (float)(ls - 32);
        const uint8_t* qs = w->qs + ib * 16;
        const block_q8_1* xb = x + ib;
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t q4 = (uint32_t)load_i32_align4(qs + j * 4);
            int lo = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
            int hi = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
            sumi = dp4a_packed(lo, load_i32_align4(xb->qs + j * 4), sumi);
            sumi = dp4a_packed(hi, load_i32_align4(xb->qs + 16 + j * 4), sumi);
        }
        acc += dl * fp16_bits_to_f32(load_u16_align2(&xb->d)) * (float)sumi;
    }
    return acc;
}

__device__ __forceinline__ float dot_q4_K_q8_1(const block_q4_K* w, const block_q8_1* x) {
    float2 dpair = fp16x2_bits_to_f32(&w->d);
    float dw = dpair.x;
    float dmin = dpair.y;
    float acc = 0.0f;
    for (int ib = 0; ib < 4; ++ib) {
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4_dev(ib * 2,     w->scales, sc1, m1);
        get_scale_min_k4_dev(ib * 2 + 1, w->scales, sc2, m2);
        const uint8_t* qs = w->qs + ib * 32;
        const block_q8_1* xb0 = x + ib * 2;
        const block_q8_1* xb1 = x + ib * 2 + 1;
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t q4 = (uint32_t)load_i32_align4(qs + j * 4);
            int lo = (int)(q4 & 0x0F0F0F0Fu);
            int hi = (int)((q4 >> 4) & 0x0F0F0F0Fu);
            sumi0 = dp4a_packed(lo, load_i32_align4(xb0->qs + j * 4), sumi0);
            sumi1 = dp4a_packed(hi, load_i32_align4(xb1->qs + j * 4), sumi1);
        }
        float2 ds0 = fp16x2_bits_to_f32(&xb0->d);
        float2 ds1 = fp16x2_bits_to_f32(&xb1->d);
        acc += dw * ((float)sc1 * ds0.x * (float)sumi0 + (float)sc2 * ds1.x * (float)sumi1)
             - dmin * ((float)m1 * ds0.y + (float)m2 * ds1.y);
    }
    return acc;
}

__device__ __forceinline__ float dot_q5_K_q8_1(const block_q5_K* w, const block_q8_1* x) {
    float2 dpair = fp16x2_bits_to_f32(&w->d);
    float dw = dpair.x;
    float dmin = dpair.y;
    float acc = 0.0f;
    uint8_t u1 = 1, u2 = 2;
    for (int ib = 0; ib < 4; ++ib) {
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4_dev(ib * 2,     w->scales, sc1, m1);
        get_scale_min_k4_dev(ib * 2 + 1, w->scales, sc2, m2);
        const uint8_t* qs = w->qs + ib * 32;
        const block_q8_1* xb0 = x + ib * 2;
        const block_q8_1* xb1 = x + ib * 2 + 1;
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t q4 = (uint32_t)load_i32_align4(qs + j * 4);
            int l0 = j * 4;
            int lo = pack_i8x4(
                (q4 & 0xF) + ((w->qh[l0 + 0] & u1) ? 16 : 0),
                ((q4 >> 8) & 0xF) + ((w->qh[l0 + 1] & u1) ? 16 : 0),
                ((q4 >> 16) & 0xF) + ((w->qh[l0 + 2] & u1) ? 16 : 0),
                ((q4 >> 24) & 0xF) + ((w->qh[l0 + 3] & u1) ? 16 : 0)
            );
            int hi = pack_i8x4(
                ((q4 >> 4) & 0xF) + ((w->qh[l0 + 0] & u2) ? 16 : 0),
                ((q4 >> 12) & 0xF) + ((w->qh[l0 + 1] & u2) ? 16 : 0),
                ((q4 >> 20) & 0xF) + ((w->qh[l0 + 2] & u2) ? 16 : 0),
                ((q4 >> 28) & 0xF) + ((w->qh[l0 + 3] & u2) ? 16 : 0)
            );
            sumi0 = dp4a_packed(lo, load_i32_align4(xb0->qs + l0), sumi0);
            sumi1 = dp4a_packed(hi, load_i32_align4(xb1->qs + l0), sumi1);
        }
        float2 ds0 = fp16x2_bits_to_f32(&xb0->d);
        float2 ds1 = fp16x2_bits_to_f32(&xb1->d);
        acc += dw * ((float)sc1 * ds0.x * (float)sumi0 + (float)sc2 * ds1.x * (float)sumi1)
             - dmin * ((float)m1 * ds0.y + (float)m2 * ds1.y);
        u1 <<= 2;
        u2 <<= 2;
    }
    return acc;
}

__device__ __forceinline__ float dot_q6_K_q8_1(const block_q6_K* w, const block_q8_1* x) {
    float dw = fp16_bits_to_f32(load_u16_align2(&w->d));
    float acc = 0.0f;
    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql = w->ql + half * 64;
        const uint8_t* qh = w->qh + half * 32;
        const int8_t*  sc = w->scales + half * 8;
        const block_q8_1* xb = x + half * 4;
        for (int is = 0; is < 2; ++is) {
            int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
            #pragma unroll
            for (int l = is * 16; l < is * 16 + 16; l += 4) {
                int q1 = pack_i8x4(
                    (int8_t)((int)((ql[l+0] & 0xF) | ((qh[l+0] & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+1] & 0xF) | ((qh[l+1] & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+2] & 0xF) | ((qh[l+2] & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+3] & 0xF) | ((qh[l+3] & 3) << 4)) - 32)
                );
                int q2 = pack_i8x4(
                    (int8_t)((int)((ql[l+32] & 0xF) | (((qh[l+0] >> 2) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+33] & 0xF) | (((qh[l+1] >> 2) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+34] & 0xF) | (((qh[l+2] >> 2) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+35] & 0xF) | (((qh[l+3] >> 2) & 3) << 4)) - 32)
                );
                int q3 = pack_i8x4(
                    (int8_t)((int)((ql[l+0] >> 4) | (((qh[l+0] >> 4) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+1] >> 4) | (((qh[l+1] >> 4) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+2] >> 4) | (((qh[l+2] >> 4) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+3] >> 4) | (((qh[l+3] >> 4) & 3) << 4)) - 32)
                );
                int q4 = pack_i8x4(
                    (int8_t)((int)((ql[l+32] >> 4) | (((qh[l+0] >> 6) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+33] >> 4) | (((qh[l+1] >> 6) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+34] >> 4) | (((qh[l+2] >> 6) & 3) << 4)) - 32),
                    (int8_t)((int)((ql[l+35] >> 4) | (((qh[l+3] >> 6) & 3) << 4)) - 32)
                );
                int l16 = l - is * 16;
                int x1 = load_i32_align4(xb[0].qs + is * 16 + l16);
                int x2 = load_i32_align4(xb[1].qs + is * 16 + l16);
                int x3 = load_i32_align4(xb[2].qs + is * 16 + l16);
                int x4 = load_i32_align4(xb[3].qs + is * 16 + l16);
                sum1 = dp4a_packed(q1, x1, sum1);
                sum2 = dp4a_packed(q2, x2, sum2);
                sum3 = dp4a_packed(q3, x3, sum3);
                sum4 = dp4a_packed(q4, x4, sum4);
            }
            float dx0 = fp16_bits_to_f32(load_u16_align2(&xb[0].d));
            float dx1 = fp16_bits_to_f32(load_u16_align2(&xb[1].d));
            float dx2 = fp16_bits_to_f32(load_u16_align2(&xb[2].d));
            float dx3 = fp16_bits_to_f32(load_u16_align2(&xb[3].d));
            acc += dw * (
                (float)sc[is + 0] * dx0 * (float)sum1 +
                (float)sc[is + 2] * dx1 * (float)sum2 +
                (float)sc[is + 4] * dx2 * (float)sum3 +
                (float)sc[is + 6] * dx3 * (float)sum4
            );
        }
    }
    return acc;
}

// -----------------------------------------------------------------------------
// GLM-5.3 extensions: Q3_K, IQ2_S, IQ3_XXS, IQ3_S, 8-lane sub-block dots, q8_1
// -----------------------------------------------------------------------------

struct __attribute__((packed)) block_q3_K {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    uint16_t d;
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K size");

struct __attribute__((packed)) block_iq3_xxs {
    uint16_t d;
    uint8_t qs[3 * (QK_K / 8)];
};
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs size");

struct __attribute__((packed)) block_iq3_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
};
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s size");

struct __attribute__((packed)) block_iq2_xxs {
    uint16_t d;
    uint8_t qs[QK_K / 4];
};
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs size");

struct __attribute__((packed)) block_iq2_xs {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs size");

struct __attribute__((packed)) block_iq1_s {
    uint16_t d;
    uint8_t qs[QK_K / 8];
    uint16_t qh[QK_K / 32];
};
static_assert(sizeof(block_iq1_s) == 50, "block_iq1_s size");

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size");

struct __attribute__((packed)) block_iq2_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_s) == 82, "block_iq2_s size");

__device__ __forceinline__ float glm_swiglu_clamp_val(float g, float u, float lim) {
    if (lim > 0.0f) {
        float gc = fminf(g, lim);
        float uc = fminf(fmaxf(u, -lim), lim);
        return (gc / (1.0f + expf(-gc))) * uc;
    }
    return (g / (1.0f + expf(-g))) * u;
}

__device__ __forceinline__ void warp_quantize_q8_1_store(float val, block_q8_1* dst, int lane) {
    float amax = warp_reduce_max(fabsf(val));
    float d = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_d);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst->qs[lane] = (int8_t)q;
    int q_sum = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        q_sum += __shfl_xor_sync(0xFFFFFFFFu, q_sum, off);
    }
    if (lane == 0) {
        uint16_t d16 = f32_to_fp16_bits(d);
        float d_act = fp16_bits_to_f32(d16);
        uint16_t s16 = f32_to_fp16_bits(d_act * (float)q_sum);
        dst->d = d16;
        dst->s = s16;
    }
}

__device__ __forceinline__ void load_q8_1_sub_words(
    const block_q8_1* xb,
    int (&u)[8],
    float2& ds8
) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        u[j] = load_i32_align4(xb->qs + j * 4);
    }
    ds8 = fp16x2_bits_to_f32(&xb->d);
}

__device__ __forceinline__ float dot_sub_q6_K(
    const block_q6_K* blk,
    const int (&u)[8],
    float d8,
    int sub
) {
    float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    int half = sub >> 2;
    int q = sub & 3;
    const uint8_t* ql = blk->ql + 64 * half + 32 * (q & 1);
    const uint8_t* qh = blk->qh + 32 * half;
    int ql_shift = 4 * (q >> 1);
    int qh_shift = 2 * q;
    int sc0 = (int)blk->scales[2 * sub + 0];
    int sc1 = (int)blk->scales[2 * sub + 1];
    int sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t lo4 = ((uint32_t)load_i32_align2(ql + 4 * j) >> ql_shift) & 0x0F0F0F0Fu;
        uint32_t hi2 = ((uint32_t)load_i32_align2(qh + 4 * j) >> qh_shift) & 0x03030303u;
        uint32_t raw6 = lo4 | (hi2 << 4);
        int w4 = (int)__vsub4(raw6, 0x20202020u);
        if (j < 4) sum0 = dp4a_packed(w4, u[j], sum0);
        else       sum1 = dp4a_packed(w4, u[j], sum1);
    }
    return dw * d8 * (float)(sc0 * sum0 + sc1 * sum1);
}

__device__ __forceinline__ float dot_sub_q4_K(
    const block_q4_K* blk,
    const int (&u)[8],
    float2 ds8,
    int sub
) {
    float2 dpair = fp16x2_bits_to_f32(&blk->d);
    uint8_t sc = 0, mn = 0;
    get_scale_min_k4_dev(sub, blk->scales, sc, mn);
    const uint8_t* qs = blk->qs + 32 * (sub >> 1);
    int shift = 4 * (sub & 1);
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int w4 = (int)(((uint32_t)load_i32_align4(qs + 4 * j) >> shift) & 0x0F0F0F0Fu);
        sumi = dp4a_packed(w4, u[j], sumi);
    }
    return dpair.x * (float)sc * ds8.x * (float)sumi - dpair.y * (float)mn * ds8.y;
}

__device__ __forceinline__ float dot_sub_q3_K(
    const block_q3_K* blk,
    const int (&u)[8],
    float d8,
    int sub
) {
    float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    int a_idx = (sub >> 1) & 1;
    uint32_t low4 = ((uint32_t)load_u16_align2(blk->scales + 4 * a_idx + 2 * (sub & 1)) >> (sub & 4)) & 0x0F0Fu;
    uint32_t hi2 = ((uint32_t)load_u16_align2(blk->scales + 8 + 2 * (sub & 1)) >> (2 * (sub >> 1))) & 0x0303u;
    uint32_t sc_pair = low4 | (hi2 << 4);
    int sc0 = (int)(sc_pair & 0xFFu) - 32;
    int sc1 = (int)((sc_pair >> 8) & 0xFFu) - 32;

    int q_shift = 2 * (sub & 3);
    const uint8_t* qp = blk->qs + 32 * (sub >> 2);
    int sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t q4 = ((uint32_t)load_i32_align2(qp + 4 * j) >> q_shift) & 0x03030303u;
        uint32_t hm = ((uint32_t)load_i32_align2(blk->hmask + 4 * j) >> sub) & 0x01010101u;
        unsigned sub4 = __vsub4(0x04040404u, hm << 2);
        int w4 = (int)__vsub4(q4, sub4);
        if (j < 4) sum0 = dp4a_packed(w4, u[j], sum0);
        else       sum1 = dp4a_packed(w4, u[j], sum1);
    }
    return dw * d8 * (float)(sc0 * sum0 + sc1 * sum1);
}

__device__ __forceinline__ float dot_sub_iq2_s(
    const block_iq2_s* blk,
    const int (&u)[8],
    float d8,
    int sub,
    const uint64_t* grid_lut
) {
    float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    uint32_t qs_packed = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
    uint32_t signs_packed = (uint32_t)load_i32_align2(blk->qs + 32 + 4 * sub);
    uint32_t qh = (uint32_t)blk->qh[sub];
    uint8_t sc_byte = blk->scales[sub];
    int ls0 = (int)(sc_byte & 0x0Fu);
    int ls1 = (int)(sc_byte >> 4);
    const uint64_t* lut = grid_lut ? grid_lut : d_iq2s_grid;

    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int sh = (l0 >> 1) * 8;
        uint32_t q_b = (qs_packed >> sh) & 0xFFu;
        uint32_t sbyte = (signs_packed >> sh) & 0xFFu;
        uint32_t grid_idx = q_b | ((qh << (8 - l0)) & 0x300u);
        uint64_t g64 = lut[grid_idx];
        uint32_t g_lo = (uint32_t)g64;
        uint32_t g_hi = (uint32_t)(g64 >> 32);
        unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
        unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
        int w0 = (int)__vsub4(g_lo ^ s0, s0);
        int w1 = (int)__vsub4(g_hi ^ s1, s1);
        if (l0 < 4) {
            sumi0 = dp4a_packed(w0, u[l0],     sumi0);
            sumi0 = dp4a_packed(w1, u[l0 + 1], sumi0);
        } else {
            sumi1 = dp4a_packed(w0, u[l0],     sumi1);
            sumi1 = dp4a_packed(w1, u[l0 + 1], sumi1);
        }
    }
    return dw * d8 * 0.125f * (float)(sumi0 * (2 * ls0 + 1) + sumi1 * (2 * ls1 + 1));
}

__device__ __forceinline__ float dot_sub_iq3_xxs(
    const block_iq3_xxs* blk,
    const int (&u)[8],
    float d8,
    int sub,
    const uint32_t* grid_lut
) {
    float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    uint32_t q3_words[2];
    q3_words[0] = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
    q3_words[1] = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
    const uint8_t* q3 = reinterpret_cast<const uint8_t*>(q3_words);
    uint32_t aux32 = (uint32_t)load_i32_align2(blk->qs + 64 + 4 * sub);
    const uint32_t* lut = grid_lut ? grid_lut : d_iq3xxs_grid;

    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t g0 = lut[q3[l0 + 0]];
        uint32_t g1 = lut[q3[l0 + 1]];
        uint32_t s7 = (aux32 >> (7 * (l0 >> 1))) & 0x7Fu;
        uint32_t ks = (s7 | ((uint32_t)(__popc(s7) & 1) << 7)) * 0x01010101u;
        unsigned s0 = __vcmpne4(ks & 0x08040201u, 0u);
        unsigned s1 = __vcmpne4(ks & 0x80402010u, 0u);
        int w0 = (int)__vsub4(g0 ^ s0, s0);
        int w1 = (int)__vsub4(g1 ^ s1, s1);
        sumi = dp4a_packed(w0, u[l0],     sumi);
        sumi = dp4a_packed(w1, u[l0 + 1], sumi);
    }
    int ls = (int)(aux32 >> 28);
    return dw * d8 * 0.25f * (float)(sumi * (2 * ls + 1));
}

__device__ __forceinline__ float dot_sub_iq3_s(
    const block_iq3_s* blk,
    const int (&u)[8],
    float d8,
    int sub,
    const uint32_t* grid_lut
) {
    float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    uint32_t q3_words[2];
    q3_words[0] = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
    q3_words[1] = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
    const uint8_t* qs = reinterpret_cast<const uint8_t*>(q3_words);
    uint32_t qh = (uint32_t)blk->qh[sub];
    uint32_t signs32 = (uint32_t)load_i32_align2(blk->signs + 4 * sub);
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(&signs32);
    const uint32_t* lut = grid_lut ? grid_lut : d_iq3s_grid;

    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t g0 = lut[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100u)];
        uint32_t g1 = lut[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100u)];
        uint32_t sbyte = (uint32_t)sb[l0 >> 1];
        unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
        unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
        int w0 = (int)__vsub4(g0 ^ s0, s0);
        int w1 = (int)__vsub4(g1 ^ s1, s1);
        sumi = dp4a_packed(w0, u[l0],     sumi);
        sumi = dp4a_packed(w1, u[l0 + 1], sumi);
    }
    int ls = 1 + 2 * (int)((blk->scales[sub >> 1] >> (4 * (sub & 1))) & 0x0Fu);
    return dw * d8 * (float)(sumi * ls);
}

// 1-superblock full wrappers (for mmvq_quant_kernel fallback on M > 1)
__device__ __forceinline__ float dot_q3_K_q8_1(const block_q3_K* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_q3_K(w, u, ds8.x, sub);
    }
    return acc;
}
__device__ __forceinline__ float dot_iq2_s_q8_1(const block_iq2_s* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_iq2_s(w, u, ds8.x, sub, nullptr);
    }
    return acc;
}
__device__ __forceinline__ float dot_iq3_xxs_q8_1(const block_iq3_xxs* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_iq3_xxs(w, u, ds8.x, sub, nullptr);
    }
    return acc;
}
__device__ __forceinline__ float dot_iq3_s_q8_1(const block_iq3_s* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_iq3_s(w, u, ds8.x, sub, nullptr);
    }
    return acc;
}

// 8-lane-per-superblock row & pair helpers (4 superblocks = 1024 values/iter/warp)
__device__ __forceinline__ float dot_row_q6_K_sub(const block_q6_K* w_row, const block_q8_1* x_q8, int K, int lane) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_q6_K(w_row + b, u, ds8.x, sub);
    }
    return warp_reduce_sum(acc);
}
__device__ __forceinline__ void dot_pair_q6_K_sub(const block_q6_K* w0, const block_q6_K* w1, const block_q8_1* x_q8, int K, int lane, float& out0, float& out1) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc0 += dot_sub_q6_K(w0 + b, u, ds8.x, sub);
        acc1 += dot_sub_q6_K(w1 + b, u, ds8.x, sub);
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}


__device__ __forceinline__ float dot_row_q4_K_sub(const block_q4_K* w_row, const block_q8_1* x_q8, int K, int lane) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_q4_K(w_row + b, u, ds8, sub);
    }
    return warp_reduce_sum(acc);
}
__device__ __forceinline__ void dot_pair_q4_K_sub(const block_q4_K* w0, const block_q4_K* w1, const block_q8_1* x_q8, int K, int lane, float& out0, float& out1) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc0 += dot_sub_q4_K(w0 + b, u, ds8, sub);
        acc1 += dot_sub_q4_K(w1 + b, u, ds8, sub);
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}


__device__ __forceinline__ float dot_row_q3_K_sub(const block_q3_K* w_row, const block_q8_1* x_q8, int K, int lane) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_q3_K(w_row + b, u, ds8.x, sub);
    }
    return warp_reduce_sum(acc);
}
__device__ __forceinline__ void dot_pair_q3_K_sub(const block_q3_K* w0, const block_q3_K* w1, const block_q8_1* x_q8, int K, int lane, float& out0, float& out1) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc0 += dot_sub_q3_K(w0 + b, u, ds8.x, sub);
        acc1 += dot_sub_q3_K(w1 + b, u, ds8.x, sub);
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}


__device__ __forceinline__ float dot_row_iq2_s_sub(const block_iq2_s* w_row, const block_q8_1* x_q8, int K, int lane, const uint64_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq2_s(w_row + b, u, ds8.x, sub, grid_lut);
    }
    return warp_reduce_sum(acc);
}
__device__ __forceinline__ void dot_pair_iq2_s_sub(const block_iq2_s* w0, const block_iq2_s* w1, const block_q8_1* x_q8, int K, int lane, float& out0, float& out1, const uint64_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc0 += dot_sub_iq2_s(w0 + b, u, ds8.x, sub, grid_lut);
        acc1 += dot_sub_iq2_s(w1 + b, u, ds8.x, sub, grid_lut);
    }
    out0 = warp_reduce_sum(acc0);
    out1 = warp_reduce_sum(acc1);
}


__device__ __forceinline__ float dot_row_iq3_xxs_sub(const block_iq3_xxs* w_row, const block_q8_1* x_q8, int K, int lane, const uint32_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq3_xxs(w_row + b, u, ds8.x, sub, grid_lut);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float dot_row_iq3_s_sub(const block_iq3_s* w_row, const block_q8_1* x_q8, int K, int lane, const uint32_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq3_s(w_row + b, u, ds8.x, sub, grid_lut);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_quad_q6_K_sub(
    const block_q6_K* w0, const block_q6_K* w1, const block_q6_K* w2, const block_q6_K* w3,
    const block_q8_1* x_q8, int K, int lane,
    float& o0, float& o1, float& o2, float& o3
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_q6_K(w0 + b, u, ds8.x, sub);
        a1 += dot_sub_q6_K(w1 + b, u, ds8.x, sub);
        a2 += dot_sub_q6_K(w2 + b, u, ds8.x, sub);
        a3 += dot_sub_q6_K(w3 + b, u, ds8.x, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ void dot_quad_q4_K_sub(
    const block_q4_K* w0, const block_q4_K* w1, const block_q4_K* w2, const block_q4_K* w3,
    const block_q8_1* x_q8, int K, int lane,
    float& o0, float& o1, float& o2, float& o3
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_q4_K(w0 + b, u, ds8, sub);
        a1 += dot_sub_q4_K(w1 + b, u, ds8, sub);
        a2 += dot_sub_q4_K(w2 + b, u, ds8, sub);
        a3 += dot_sub_q4_K(w3 + b, u, ds8, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ void dot_quad_q3_K_sub(
    const block_q3_K* w0, const block_q3_K* w1, const block_q3_K* w2, const block_q3_K* w3,
    const block_q8_1* x_q8, int K, int lane,
    float& o0, float& o1, float& o2, float& o3
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_q3_K(w0 + b, u, ds8.x, sub);
        a1 += dot_sub_q3_K(w1 + b, u, ds8.x, sub);
        a2 += dot_sub_q3_K(w2 + b, u, ds8.x, sub);
        a3 += dot_sub_q3_K(w3 + b, u, ds8.x, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ float dot_sub_iq2_xxs(
    const block_iq2_xxs* blk,
    const int (&u)[8],
    float d8,
    int sub,
    const uint64_t* grid_lut = nullptr
) {
    const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    const uint32_t aux0 = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
    const uint32_t aux1 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
    const int ls = (int)(aux1 >> 28);
    const uint64_t* lut = grid_lut ? grid_lut : d_iq2xxs_grid;
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t g_idx = (aux0 >> (8 * l)) & 0xFFu;
        const uint32_t s7 = (aux1 >> (7 * l)) & 0x7Fu;
        const uint32_t sbyte = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
        const uint64_t g64 = lut[g_idx];
        const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
        const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
        const int w0 = (int)__vsub4((uint32_t)g64 ^ s0, s0);
        const int w1 = (int)__vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        sumi = dp4a_packed(w0, u[2 * l],     sumi);
        sumi = dp4a_packed(w1, u[2 * l + 1], sumi);
    }
    return dw * d8 * 0.125f * (float)(sumi * (2 * ls + 1));
}

__device__ __forceinline__ float dot_sub_iq2_xs(
    const block_iq2_xs* blk,
    const int (&u)[8],
    float d8,
    int sub,
    const uint64_t* grid_lut = nullptr
) {
    const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    const uint32_t q01 = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
    const uint32_t q23 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
    const uint8_t sc_byte = blk->scales[sub];
    const int ls0 = (int)(sc_byte & 0x0Fu);
    const int ls1 = (int)(sc_byte >> 4);
    const uint64_t* lut = grid_lut ? grid_lut : d_iq2xs_grid;
    int sumi0 = 0, sumi1 = 0;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qw = (l < 2) ? q01 : q23;
        const uint32_t q16 = (l & 1) ? (qw >> 16) : (qw & 0xFFFFu);
        const uint32_t g_idx = q16 & 511u;
        const uint32_t s7 = q16 >> 9;
        const uint32_t sbyte = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
        const uint64_t g64 = lut[g_idx];
        const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
        const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
        const int w0 = (int)__vsub4((uint32_t)g64 ^ s0, s0);
        const int w1 = (int)__vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        if (l < 2) {
            sumi0 = dp4a_packed(w0, u[2 * l],     sumi0);
            sumi0 = dp4a_packed(w1, u[2 * l + 1], sumi0);
        } else {
            sumi1 = dp4a_packed(w0, u[2 * l],     sumi1);
            sumi1 = dp4a_packed(w1, u[2 * l + 1], sumi1);
        }
    }
    return dw * d8 * 0.125f * (float)(sumi0 * (2 * ls0 + 1) + sumi1 * (2 * ls1 + 1));
}

__device__ __forceinline__ float dot_sub_iq1_s(
    const block_iq1_s* blk,
    const int (&u)[8],
    float2 ds8,
    int sub,
    const uint64_t* grid_lut = nullptr
) {
    const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    const uint16_t qh = load_u16_align2(&blk->qh[sub]);
    const uint32_t qs4 = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
    const int sc = 2 * (int)((qh >> 12) & 7u) + 1;
    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    const uint64_t* lut = grid_lut ? grid_lut : d_iq1s_grid;
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t g_idx = ((qs4 >> (8 * l)) & 0xFFu) | ((((uint32_t)qh >> (3 * l)) & 7u) << 8);
        const uint64_t g64 = lut[g_idx];
        sumi = dp4a_packed((int)(uint32_t)g64, u[2 * l], sumi);
        sumi = dp4a_packed((int)(uint32_t)(g64 >> 32), u[2 * l + 1], sumi);
    }
    return dw * (float)sc * (ds8.x * (float)sumi + delta * ds8.y);
}

__device__ __forceinline__ float dot_iq2_xxs_q8_1(const block_iq2_xxs* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_iq2_xxs(w, u, ds8.x, sub);
    }
    return acc;
}
__device__ __forceinline__ float dot_iq2_xs_q8_1(const block_iq2_xs* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_iq2_xs(w, u, ds8.x, sub);
    }
    return acc;
}
__device__ __forceinline__ float dot_iq1_s_q8_1(const block_iq1_s* w, const block_q8_1* x) {
    float acc = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x + sub, u, ds8);
        acc += dot_sub_iq1_s(w, u, ds8, sub);
    }
    return acc;
}

__device__ __forceinline__ float dot_row_iq2_xxs_sub(const block_iq2_xxs* w_row, const block_q8_1* x_q8, int K, int lane, const uint64_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq2_xxs(w_row + b, u, ds8.x, sub, grid_lut);
    }
    return warp_reduce_sum(acc);
}
__device__ __forceinline__ float dot_row_iq2_xs_sub(const block_iq2_xs* w_row, const block_q8_1* x_q8, int K, int lane, const uint64_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq2_xs(w_row + b, u, ds8.x, sub, grid_lut);
    }
    return warp_reduce_sum(acc);
}
__device__ __forceinline__ float dot_row_iq1_s_sub(const block_iq1_s* w_row, const block_q8_1* x_q8, int K, int lane, const uint64_t* grid_lut = nullptr) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq1_s(w_row + b, u, ds8, sub, grid_lut);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_quad_iq2_s_sub(
    const block_iq2_s* w0, const block_iq2_s* w1, const block_iq2_s* w2, const block_iq2_s* w3,
    const block_q8_1* x_q8, int K, int lane,
    float& o0, float& o1, float& o2, float& o3,
    const uint64_t* grid_lut = nullptr
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        #pragma unroll 1
        for (int r = 0; r < 4; ++r) {
            const block_iq2_s* wr = (r == 0) ? w0 : (r == 1) ? w1 : (r == 2) ? w2 : w3;
            float d = dot_sub_iq2_s(wr + b, u, ds8.x, sub, grid_lut);
            if (r == 0) a0 += d;
            else if (r == 1) a1 += d;
            else if (r == 2) a2 += d;
            else a3 += d;
        }
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ void dot_quad_iq3_xxs_sub(
    const block_iq3_xxs* w0, const block_iq3_xxs* w1, const block_iq3_xxs* w2, const block_iq3_xxs* w3,
    const block_q8_1* x_q8, int K, int lane,
    float& o0, float& o1, float& o2, float& o3,
    const uint32_t* grid_lut = nullptr
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_iq3_xxs(w0 + b, u, ds8.x, sub, grid_lut);
        a1 += dot_sub_iq3_xxs(w1 + b, u, ds8.x, sub, grid_lut);
        a2 += dot_sub_iq3_xxs(w2 + b, u, ds8.x, sub, grid_lut);
        a3 += dot_sub_iq3_xxs(w3 + b, u, ds8.x, sub, grid_lut);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ void dot_quad_iq3_s_sub(
    const block_iq3_s* w0, const block_iq3_s* w1, const block_iq3_s* w2, const block_iq3_s* w3,
    const block_q8_1* x_q8, int K, int lane,
    float& o0, float& o1, float& o2, float& o3,
    const uint32_t* grid_lut = nullptr
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_iq3_s(w0 + b, u, ds8.x, sub, grid_lut);
        a1 += dot_sub_iq3_s(w1 + b, u, ds8.x, sub, grid_lut);
        a2 += dot_sub_iq3_s(w2 + b, u, ds8.x, sub, grid_lut);
        a3 += dot_sub_iq3_s(w3 + b, u, ds8.x, sub, grid_lut);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ float dot_sub_iq4_xs(
    const block_iq4_xs* blk,
    const int (&u)[8],
    float d8,
    int sub
) {
    float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    uint16_t scales_h = load_u16_align2(&blk->scales_h);
    int ls = ((blk->scales_l[sub >> 1] >> (4 * (sub & 1))) & 0xF) | (((scales_h >> (2 * sub)) & 3) << 4);
    float dl = dw * (float)(ls - 32);
    const uint8_t* qs = blk->qs + sub * 16;
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t q4 = (uint32_t)load_i32_align4(qs + j * 4);
        int lo = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
        int hi = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
        sumi = dp4a_packed(lo, u[j], sumi);
        sumi = dp4a_packed(hi, u[4 + j], sumi);
    }
    return dl * d8 * (float)sumi;
}

__device__ __forceinline__ float dot_sub_q5_K(
    const block_q5_K* blk,
    const int (&u)[8],
    float2 ds8,
    int sub
) {
    float2 dpair = fp16x2_bits_to_f32(&blk->d);
    uint8_t sc = 0, mn = 0;
    get_scale_min_k4_dev(sub, blk->scales, sc, mn);
    const uint8_t* qs = blk->qs + 32 * (sub >> 1);
    int q4_shift = 4 * (sub & 1);
    int sumi = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t q4 = ((uint32_t)load_i32_align4(qs + 4 * j) >> q4_shift) & 0x0F0F0F0Fu;
        uint32_t qh4 = (uint32_t)load_i32_align4(blk->qh + 4 * j);
        uint32_t qh_hi = ((sub <= 4) ? (qh4 << (4 - sub)) : (qh4 >> (sub - 4))) & 0x10101010u;
        int w5 = (int)(q4 | qh_hi);
        sumi = dp4a_packed(w5, u[j], sumi);
    }
    return dpair.x * (float)sc * ds8.x * (float)sumi - dpair.y * (float)mn * ds8.y;
}

__device__ __forceinline__ float dot_row_iq4_xs_sub(
    const block_iq4_xs* w_row,
    const block_q8_1* x_q8,
    int K,
    int lane
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_iq4_xs(w_row + b, u, ds8.x, sub);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_iq4_xs_sub(
    const block_iq4_xs* w0,
    const block_iq4_xs* w1,
    const block_q8_1* x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_iq4_xs(w0 + b, u, ds8.x, sub);
        a1 += dot_sub_iq4_xs(w1 + b, u, ds8.x, sub);
    }
    out0 = warp_reduce_sum(a0);
    out1 = warp_reduce_sum(a1);
}

__device__ __forceinline__ void dot_quad_iq4_xs_sub(
    const block_iq4_xs* w0,
    const block_iq4_xs* w1,
    const block_iq4_xs* w2,
    const block_iq4_xs* w3,
    const block_q8_1* x_q8,
    int K,
    int lane,
    float& o0,
    float& o1,
    float& o2,
    float& o3
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_iq4_xs(w0 + b, u, ds8.x, sub);
        a1 += dot_sub_iq4_xs(w1 + b, u, ds8.x, sub);
        a2 += dot_sub_iq4_xs(w2 + b, u, ds8.x, sub);
        a3 += dot_sub_iq4_xs(w3 + b, u, ds8.x, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ float dot_row_q5_K_sub(
    const block_q5_K* w_row,
    const block_q8_1* x_q8,
    int K,
    int lane
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float acc = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        acc += dot_sub_q5_K(w_row + b, u, ds8, sub);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ void dot_pair_q5_K_sub(
    const block_q5_K* w0,
    const block_q5_K* w1,
    const block_q8_1* x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f;
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_q5_K(w0 + b, u, ds8, sub);
        a1 += dot_sub_q5_K(w1 + b, u, ds8, sub);
    }
    out0 = warp_reduce_sum(a0);
    out1 = warp_reduce_sum(a1);
}

__device__ __forceinline__ void dot_quad_q5_K_sub(
    const block_q5_K* w0,
    const block_q5_K* w1,
    const block_q5_K* w2,
    const block_q5_K* w3,
    const block_q8_1* x_q8,
    int K,
    int lane,
    float& o0,
    float& o1,
    float& o2,
    float& o3
) {
    int nb = K >> 8, sub = lane & 7, b0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    #pragma unroll 1
    for (int b = b0; b < nb; b += 4) {
        int u[8]; float2 ds8;
        load_q8_1_sub_words(x_q8 + b * 8 + sub, u, ds8);
        a0 += dot_sub_q5_K(w0 + b, u, ds8, sub);
        a1 += dot_sub_q5_K(w1 + b, u, ds8, sub);
        a2 += dot_sub_q5_K(w2 + b, u, ds8, sub);
        a3 += dot_sub_q5_K(w3 + b, u, ds8, sub);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

__device__ __forceinline__ void dot_quad_q8_0_sub(
    const block_q8_0* __restrict__ w0,
    const block_q8_0* __restrict__ w1,
    const block_q8_0* __restrict__ w2,
    const block_q8_0* __restrict__ w3,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane,
    float& o0,
    float& o1,
    float& o2,
    float& o3
) {
    int nb = K >> 5;
    int sub = lane & 7;
    int blk0 = lane >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = blk0; b < nb; b += 4) {
        const block_q8_1* xb = x_q8 + b;
        int xa = load_i32_align4(xb->qs + sub * 4);
        float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));

        const block_q8_0* wb0 = w0 + b;
        a0 += (fp16_bits_to_f32(load_u16_align2(&wb0->d)) * xd) * (float)dp4a_packed(load_i32_align2(wb0->qs + sub * 4), xa, 0);

        const block_q8_0* wb1 = w1 + b;
        a1 += (fp16_bits_to_f32(load_u16_align2(&wb1->d)) * xd) * (float)dp4a_packed(load_i32_align2(wb1->qs + sub * 4), xa, 0);

        const block_q8_0* wb2 = w2 + b;
        a2 += (fp16_bits_to_f32(load_u16_align2(&wb2->d)) * xd) * (float)dp4a_packed(load_i32_align2(wb2->qs + sub * 4), xa, 0);

        const block_q8_0* wb3 = w3 + b;
        a3 += (fp16_bits_to_f32(load_u16_align2(&wb3->d)) * xd) * (float)dp4a_packed(load_i32_align2(wb3->qs + sub * 4), xa, 0);
    }
    o0 = warp_reduce_sum(a0);
    o1 = warp_reduce_sum(a1);
    o2 = warp_reduce_sum(a2);
    o3 = warp_reduce_sum(a3);
}

extern "C" void launch_glm5_mmvq_sub256_skip(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    float* Y,
    int N,
    int K,
    const int32_t* skip_dev,
    cudaStream_t stream
);

extern "C" void launch_glm5_mmvq_sub256(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    float* Y,
    int N,
    int K,
    cudaStream_t stream
);
