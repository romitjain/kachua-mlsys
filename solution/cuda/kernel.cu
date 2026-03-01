/*
 * CUDA Kernel Template for FlashInfer Competition.
 *
 * Implement your kernel logic here. The entry point function name should match
 * the `entry_point` setting in config.toml.
 *
 * See the track definition for required function signature and semantics.
 */

//  Launch command
// nvcc -std=c++17 -Wno-deprecated-gpu-targets -o kernel.o kernel.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cmath>

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
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    const float *__restrict__ state,
    const float *__restrict__ A_log,
    const float *__restrict__ a,
    const float *__restrict__ dt_bias,
    const float *__restrict__ b,
    float *__restrict__ out,
    float *__restrict__ new_state,
    int B,
    int num_v_heads,
    int num_k_heads,
    int K,
    int V,
    float scale,
    int split_v
) {
    __shared__ float gate_s;
    __shared__ float beta_s;
    int common_scalar_offset = num_v_heads*blockIdx.x + blockIdx.y;
    int qk_head_factor = num_v_heads/num_k_heads;

    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        float x = a[common_scalar_offset] + dt_bias[common_scalar_offset];
        gate_s = expf(-expf(A_log[common_scalar_offset]) * log1pf(expf(x)));
        beta_s = sigmoid(b[common_scalar_offset]);
    }
    __syncthreads();

    float gate = gate_s;
    float beta = beta_s;

    __shared__ float STATE[8][32];
    __shared__ float OLD_V[8]; // how to initialize to 0?
    __shared__ float NEW_V[8];

    int k_offset = (num_k_heads*blockIdx.x + blockIdx.y/qk_head_factor)*K;
    int v_offset = common_scalar_offset*V;
    // Get the read offset for state matrix (global)
    // State matrix is BxHxVxK, we want VxK for every batch,head combination
    int state_read_go = common_scalar_offset*V*K;
    // Read offset for state matrix (local)
    int local_v = blockIdx.z*split_v + threadIdx.y;
    int state_read_lo = local_v*K;

    float old_v_accum = 0.0f;

    // Vector matmul b/w state and k
    for (int k0=0;k0<K;k0+=32) {
        // HMEM <> SMEM for state tile
        int local_k = k0+threadIdx.x;
        if (local_v<V and local_k<K) {
            STATE[threadIdx.y][threadIdx.x] = state[state_read_go+state_read_lo+local_k];
            old_v_accum += gate*STATE[threadIdx.y][threadIdx.x] * k[k_offset+local_k];
        }
        else {
            STATE[threadIdx.y][threadIdx.x] = 0.0f;
        }
    }

    // Warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        old_v_accum += __shfl_down_sync(0xffffffff, old_v_accum, off);
    }

    // v_new computation
    if (threadIdx.x == 0) {
        OLD_V[threadIdx.y] = old_v_accum;
        if (local_v < V) {
            NEW_V[threadIdx.y] = beta * v[v_offset + local_v] + (1.0f - beta) * old_v_accum;
        } else {
            NEW_V[threadIdx.y] = 0.0f;
        }
    }

    __syncthreads();

    float out_acc = 0.0f;

    // Updated state computation and write back
    for (int k0=0; k0<K ;k0+=32) {
        int local_k = k0+threadIdx.x;
        if (local_v < V and local_k < K) {
            float s = state[state_read_go+state_read_lo+local_k];
            float kval = k[k_offset+local_k];
            float val = gate*s - OLD_V[threadIdx.y] * kval + NEW_V[threadIdx.y] * kval;

            new_state[state_read_go+state_read_lo+local_k] = val;

            out_acc += scale * val * q[k_offset+local_k];
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        out_acc += __shfl_down_sync(0xffffffff, out_acc, off);
    }

    // Write out
    if (threadIdx.x == 0) {
        if (local_v < V) {
            out[v_offset+local_v] = out_acc;
        }
    }
}

extern "C" cudaError_t launch_gdn_v1(
    const float* q,
    const float* k,
    const float* v,
    const float* state,
    const float* A_log,
    const float* a,
    const float* dt_bias,
    const float* b,
    float* out,
    float* new_state,
    int B,
    int num_v_heads,
    int num_k_heads,
    int K,
    int V,
    float scale,
    cudaStream_t stream
) {

    // V dim is split into these many blocks
    int split_v = 8;
    dim3 threads_per_block(32, split_v);
    dim3 grid_size(B, num_v_heads, (V+split_v-1)/split_v);

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
        num_k_heads,
        K,
        V,
        scale,
        split_v
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

    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* state = nullptr;
    float* A_log = nullptr;
    float* a = nullptr;
    float* dt_bias = nullptr;
    float* b = nullptr;
    float* out = nullptr;
    float* new_state = nullptr;
    const float scale = 1.0f / std::sqrt(static_cast<float>(K));

    cudaMallocManaged(&q, q_elems * sizeof(float));
    cudaMallocManaged(&k, k_elems * sizeof(float));
    cudaMallocManaged(&v, v_elems * sizeof(float));
    cudaMallocManaged(&state, state_elems * sizeof(float));
    cudaMallocManaged(&A_log, gate_elems * sizeof(float));
    cudaMallocManaged(&a, gate_elems * sizeof(float));
    cudaMallocManaged(&dt_bias, gate_elems * sizeof(float));
    cudaMallocManaged(&b, gate_elems * sizeof(float));
    cudaMallocManaged(&out, out_elems * sizeof(float));
    cudaMallocManaged(&new_state, state_elems * sizeof(float));

    for (size_t i = 0; i < k_elems; ++i) k[i] = (0.02f * static_cast<float>(i % 13));
    for (size_t i = 0; i < v_elems; ++i) v[i] = (0.03f * static_cast<float>(i % 11));
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
        num_v_heads,
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
        num_v_heads,
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
