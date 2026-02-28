/*
 * CUDA Kernel Template for FlashInfer Competition.
 *
 * Implement your kernel logic here. The entry point function name should match
 * the `entry_point` setting in config.toml.
 *
 * See the track definition for required function signature and semantics.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstddef>
#include <cmath>
#include <cstdio>

__device__ __forceinline__ float sigmoid(float x) {
    // Stable sigmoid
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

__global__ void gdn_v1(
    const void* q,
    const void* k,
    const void* v,
    const float* state,
    const float* A_log,
    const float* a,
    const float* dt_bias,
    const float* b,
    float* out,
    float* new_state,
    int B,
    int num_v_heads,
    int K,
    int V,
    float scale
) {
    // Read offsets for a, dt_bias, A_log, b
    float gate = 0.0f;
    float beta = 0.0f;

    if (threadIdx.x % 32 == 0) {
        int common_scalar_offset = B*num_v_heads*blockIdx.x + num_v_heads*blockIdx.y;
        float x = a[common_scalar_offset] + dt_bias[common_scalar_offset];
        gate = expf(expf(A_log[common_scalar_offset]) * log1pf(expf(x)));
        beta = sigmoid(b[common_scalar_offset]);
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        gate = __shfl_xor_sync(0xffffffff, gate, offset, 32);
        beta = __shfl_xor_sync(0xffffffff, beta, offset, 32);
    }

    __syncthreads();

    // Get the correct read offset for state matrix
    int state_read_in = B*num_v_heads*blockIdx.x + num_v_heads*blockIdx.y;

}

extern "C" cudaError_t launch_gdn_v1(
    const void* q,
    const void* k,
    const void* v,
    const float* state,
    const float* A_log,
    const float* a,
    const float* dt_bias,
    const float* b,
    float* out,
    float* new_state,
    int B,
    int num_v_heads,
    int K,
    int V,
    float scale,
    cudaStream_t stream
) {

    dim3 threads_per_block(K, 16);
    dim3 grid_size(B, num_v_heads);

    gdn_v1<<<grid_size, threads_per_block>>>(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        out,
        new_state,
        B,
        num_v_heads,
        K,
        V,
        scale
    );
    return cudaGetLastError();
}

#ifndef TORCH_EXTENSION_NAME
int main() {
    constexpr int B = 2;
    constexpr int T = 1;
    constexpr int num_q_heads = 16;
    constexpr int num_k_heads = 16;
    constexpr int num_v_heads = 32;
    constexpr int K = 128;
    constexpr int V = 128;

    const size_t q_elems = B * T * num_q_heads * K;
    const size_t k_elems = B * T * num_k_heads * K;
    const size_t v_elems = B * T * num_v_heads * V;
    const size_t state_elems = B * num_v_heads * V * K;
    const size_t gate_elems = B * num_v_heads;
    const size_t out_elems = B * num_v_heads * V;

    half* q = nullptr;
    half* k = nullptr;
    half* v = nullptr;
    float* state = nullptr;
    float* A_log = nullptr;
    float* a = nullptr;
    float* dt_bias = nullptr;
    float* b = nullptr;
    float* out = nullptr;
    float* new_state = nullptr;
    const float scale = 1.0f / std::sqrt(static_cast<float>(K));

    cudaMallocManaged(&q, q_elems * sizeof(half));
    cudaMallocManaged(&k, k_elems * sizeof(half));
    cudaMallocManaged(&v, v_elems * sizeof(half));
    cudaMallocManaged(&state, state_elems * sizeof(float));
    cudaMallocManaged(&A_log, gate_elems * sizeof(float));
    cudaMallocManaged(&a, gate_elems * sizeof(float));
    cudaMallocManaged(&dt_bias, gate_elems * sizeof(float));
    cudaMallocManaged(&b, gate_elems * sizeof(float));
    cudaMallocManaged(&out, out_elems * sizeof(float));
    cudaMallocManaged(&new_state, state_elems * sizeof(float));

    for (size_t i = 0; i < k_elems; ++i) k[i] = __float2half(0.02f * static_cast<float>(i % 13));
    for (size_t i = 0; i < v_elems; ++i) v[i] = __float2half(0.03f * static_cast<float>(i % 11));
    for (size_t i = 0; i < state_elems; ++i) {
        state[i] = 0.0f;
        new_state[i] = 0.0f;
    }
    for (size_t i = 0; i < gate_elems; ++i) {
        A_log[i] = -2.0f;
        a[i] = 0.0f;
        dt_bias[i] = 0.1f;
        b[i] = 0.0f;
    }
    for (size_t i = 0; i < out_elems; ++i) out[i] = 0.0f;

    const cudaError_t launch_err = launch_gdn_v1(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        out,
        new_state,
        B,
        num_k_heads,
        K,
        V,
        scale,
        nullptr
    );
    if (launch_err != cudaSuccess) {
        std::printf("launch_gdn_v1 failed: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(state);
    cudaFree(A_log);
    cudaFree(a);
    cudaFree(dt_bias);
    cudaFree(b);
    cudaFree(out);
    cudaFree(new_state);
    return 0;
}
#endif  // TORCH_EXTENSION_NAME

#ifdef TORCH_EXTENSION_NAME
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void launch_gdn_v1_torch(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor out,
    torch::Tensor new_state,
    double scale
) {
    const int B = static_cast<int>(q.size(0));
    const int T = static_cast<int>(q.size(1));
    const int num_q_heads = static_cast<int>(q.size(2));
    const int num_k_heads = static_cast<int>(k.size(2));
    const int num_v_heads = static_cast<int>(v.size(2));
    const int K = static_cast<int>(q.size(3));
    const int V = static_cast<int>(v.size(3));

    torch::Tensor A_log_f32 = A_log.scalar_type() == torch::kFloat32 ? A_log : A_log.to(torch::kFloat32);
    torch::Tensor a_f32 = a.scalar_type() == torch::kFloat32 ? a : a.to(torch::kFloat32);
    torch::Tensor dt_bias_f32 =
        dt_bias.scalar_type() == torch::kFloat32 ? dt_bias : dt_bias.to(torch::kFloat32);
    torch::Tensor b_f32 = b.scalar_type() == torch::kFloat32 ? b : b.to(torch::kFloat32);

    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    (void)launch_gdn_v1(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        state.data_ptr<float>(),
        A_log_f32.data_ptr<float>(),
        a_f32.data_ptr<float>(),
        dt_bias_f32.data_ptr<float>(),
        b_f32.data_ptr<float>(),
        out.data_ptr<float>(),
        new_state.data_ptr<float>(),
        B,
        num_k_heads,
        K,
        V,
        static_cast<float>(scale),
        stream.stream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gdn_v1", &launch_gdn_v1_torch, "Launch gdn_v1 CUDA kernel");
}
#endif  // TORCH_EXTENSION_NAME
