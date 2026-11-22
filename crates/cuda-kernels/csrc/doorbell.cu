#include "common/sm86_common.cuh"

// Wait on host-mapped completion sequence flag and add partial vector into GPU output.
__global__ void doorbell_wait_and_add_kernel(
    const volatile uint64_t* __restrict__ done_seq,
    uint64_t expected_seq,
    const float* __restrict__ partial_a, // nullable (e.g., CPU mapped output)
    const float* __restrict__ partial_b, // nullable (e.g., Helper GPU1 mapped output)
    float* __restrict__ out,
    int n_total
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (*done_seq < expected_seq) {
            // Spin-wait on mapped pinned host completion counter
        }
        __threadfence_system();
    }
    __syncthreads();
    // Note: Only block 0 synchronized on *done_seq; for multi-block safety we either launch 1 block
    // (256-1024 threads handles 2304 floats easily in 1 block!) or broadcast via global memory.
    int i = threadIdx.x;
    for (; i < n_total; i += blockDim.x) {
        float add = 0.0f;
        if (partial_a) add += partial_a[i];
        if (partial_b) add += partial_b[i];
        out[i] += add;
    }
}

__global__ void doorbell_signal_kernel(
    volatile uint64_t* __restrict__ req_seq,
    uint64_t seq_val
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *req_seq = seq_val;
        __threadfence_system();
    }
}

extern "C" {

void launch_doorbell_signal(
    uint64_t* req_seq_dev,
    uint64_t seq_val,
    cudaStream_t stream
) {
    doorbell_signal_kernel<<<1, 1, 0, stream>>>(req_seq_dev, seq_val);
}

void launch_doorbell_wait_and_add(
    const uint64_t* done_seq_dev,
    uint64_t expected_seq,
    const float* partial_a,
    const float* partial_b,
    float* out,
    int n_total,
    cudaStream_t stream
) {
    // Single block of 512 threads keeps barrier wait inside one CTA with zero global sync needed.
    doorbell_wait_and_add_kernel<<<1, 512, 0, stream>>>(
        done_seq_dev, expected_seq, partial_a, partial_b, out, n_total
    );
}

} // extern "C"
