#include "common/sm86_common.cuh"

// System-scope push + -0.0f sentinel reduction between 2 GPUs (works via P2P or mapped pinned RAM).
__device__ __forceinline__ bool is_neg_zero(float v) {
    return __float_as_uint(v) == 0x80000000u;
}

__device__ __forceinline__ float sanitize_neg_zero(float v) {
    return is_neg_zero(v) ? 0.0f : v;
}

__global__ void p2p_push_kernel(
    const float* __restrict__ local_partial,
    volatile float* __restrict__ peer_recv,
    volatile uint32_t* __restrict__ peer_flag,
    uint32_t seq,
    int n_total
) {
    // Single-CTA push guarantees __syncthreads() finishes all stores before *peer_flag is published.
    for (int i = threadIdx.x; i < n_total; i += blockDim.x) {
        peer_recv[i] = sanitize_neg_zero(local_partial[i]);
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) {
        *peer_flag = seq;
        __threadfence_system();
    }
}

__global__ void p2p_wait_reduce_kernel(
    float* __restrict__ local_buf,
    const volatile float* __restrict__ recv_buf,
    const volatile uint32_t* __restrict__ flag_buf,
    uint32_t expected_seq,
    int n_total
) {
    // Single CTA (512 threads) handles up to full hidden vector cleanly with deterministic wait
    if (threadIdx.x == 0) {
        while (*flag_buf < expected_seq) {
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_total; i += blockDim.x) {
        local_buf[i] += recv_buf[i];
    }
}

extern "C" {

void launch_p2p_push(
    const float* local_partial,
    float* peer_recv,
    uint32_t* peer_flag,
    uint32_t seq,
    int n_total,
    cudaStream_t stream
) {
    p2p_push_kernel<<<1, 512, 0, stream>>>(local_partial, peer_recv, peer_flag, seq, n_total);
}

void launch_p2p_wait_reduce(
    float* local_buf,
    const float* recv_buf,
    const uint32_t* flag_buf,
    uint32_t expected_seq,
    int n_total,
    cudaStream_t stream
) {
    p2p_wait_reduce_kernel<<<1, 512, 0, stream>>>(local_buf, recv_buf, flag_buf, expected_seq, n_total);
}

} // extern "C"
