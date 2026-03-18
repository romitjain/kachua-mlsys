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
#include <cuda_bf16.h>
#include <mma.h>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cmath>

constexpr int kSplitV = 4;

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
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const float *__restrict__ state,
    const float *__restrict__ A_log,
    const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
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
        float x = __bfloat162float(a[common_scalar_offset]) + dt_bias[common_scalar_offset];
        gate_s = expf(-expf(A_log[common_scalar_offset]) * log1pf(expf(x)));
        beta_s = sigmoid(__bfloat162float(b[common_scalar_offset]));
    }
    __syncthreads();

    float gate = gate_s;
    float beta = beta_s;

    __shared__ float STATE[16][32];
    __shared__ float OLD_V[16]; // how to initialize to 0?
    __shared__ float NEW_V[16];

    int k_offset = (num_k_heads*blockIdx.x + blockIdx.y/qk_head_factor)*K;
    int v_offset = common_scalar_offset*V;
    // Get the read offset for state matrix (global)
    // State matrix is BxHxVxK, we want VxK for every batch,head combination
    int state_read_go = common_scalar_offset*V*K;
    // Read offset for state matrix (local)
    int local_v = blockIdx.z*split_v + threadIdx.y;
    int state_read_lo = local_v*K;

    float old_v_accum = 0.0f;

# pragma unroll
    // Vector matmul b/w state and k
    for (int k0=0;k0<K;k0+=32) {
        // HMEM <> SMEM for state tile
        int local_k = k0+threadIdx.x;

        STATE[threadIdx.y][threadIdx.x] = state[state_read_go+state_read_lo+local_k];
        old_v_accum += gate*STATE[threadIdx.y][threadIdx.x] * __bfloat162float(k[k_offset+local_k]);
    }

# pragma unroll
    // Warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        old_v_accum += __shfl_down_sync(0xffffffff, old_v_accum, off);
    }

    // v_new computation
    if (threadIdx.x == 0) {
        OLD_V[threadIdx.y] = old_v_accum;
        NEW_V[threadIdx.y] = beta * __bfloat162float(v[v_offset + local_v]) + (1.0f - beta) * old_v_accum;
    }

    float out_acc = 0.0f;

# pragma unroll
    // Updated state computation and write back
    for (int k0=0; k0<K ;k0+=32) {
        int local_k = k0+threadIdx.x;
        float s = state[state_read_go+state_read_lo+local_k];
        float kval = __bfloat162float(k[k_offset+local_k]);
        float val = gate*s - OLD_V[threadIdx.y] * kval + NEW_V[threadIdx.y] * kval;

        new_state[state_read_go+state_read_lo+local_k] = val;
        out_acc += scale * val * __bfloat162float(q[k_offset+local_k]);
    }

# pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        out_acc += __shfl_down_sync(0xffffffff, out_acc, off);
    }

    // Write out
    if (threadIdx.x == 0) {
        // if (local_v < V) {
        out[v_offset+local_v] = __float2bfloat16_rn(out_acc);
        // }
    }
}

__global__ void gdn_v2(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const float *__restrict__ state,
    const float *__restrict__ A_log,
    const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ new_state,
    int B,
    int num_v_heads,
    int num_k_heads,
    int K,
    int V,
    float scale
) {
    int common_scalar_offset = num_v_heads*blockIdx.x + blockIdx.y;
    int qk_head_factor = num_v_heads/num_k_heads;

    float gate;
    float beta;

    if (threadIdx.x == 0) {
        float x = __bfloat162float(a[common_scalar_offset]) + dt_bias[common_scalar_offset];
        gate = expf(-expf(A_log[common_scalar_offset]) * log1pf(expf(x)));
        beta = sigmoid(__bfloat162float(b[common_scalar_offset]));
    }

    gate = __shfl_sync(0xffffffff, gate, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);

    int k_offset = (num_k_heads*blockIdx.x + blockIdx.y/qk_head_factor)*K;
    int v_offset = common_scalar_offset*V;
    int local_v = blockIdx.z*kSplitV + threadIdx.y;
    // Get the read offset for state matrix (global)
    // State matrix is BxHxVxK, we want VxK for every batch,head combination
    int state_read = common_scalar_offset*V*K + local_v*K;

    // Vector matmul b/w state and k
    int local_k = threadIdx.x*4;
    const float4 s4 = reinterpret_cast<const float4*> (&state[state_read+local_k])[0];

    // For bf16 loads, I have to do more things
    // Load 4 bf16 values, split in to float2 values
    // (why cant I save to a single float4) - because the nv_bfloat162 gives me 2 pair of bfloat16
    // hmm, if there was nv_bfloat164, then it would have been possible to convert to float4 directly
    const nv_bfloat162* k_cache4 = reinterpret_cast<const nv_bfloat162*> (&k[k_offset+local_k]);
    float2 k_cache01 = __bfloat1622float2(k_cache4[0]);
    float2 k_cache02 = __bfloat1622float2(k_cache4[1]);

    float old_v_accum = 0.0f;
    old_v_accum += gate*s4.x*k_cache01.x;
    old_v_accum += gate*s4.y*k_cache01.y;
    old_v_accum += gate*s4.z*k_cache02.x;
    old_v_accum += gate*s4.w*k_cache02.y;

# pragma unroll
    // Warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        old_v_accum += __shfl_down_sync(0xffffffff, old_v_accum, off);
    }

    float old_v = __shfl_sync(0xffffffff, old_v_accum, 0);
    float new_v = 0.0f;
    if (threadIdx.x == 0) {
        new_v = beta * __bfloat162float(v[v_offset + local_v]) + (1.0f - beta) * old_v;
    }
    new_v = __shfl_sync(0xffffffff, new_v, 0);

    float valx = gate*s4.x - old_v*k_cache01.x + new_v*k_cache01.x;
    float valy = gate*s4.y - old_v*k_cache01.y + new_v*k_cache01.y;
    float valz = gate*s4.z - old_v*k_cache02.x + new_v*k_cache02.x;
    float valw = gate*s4.w - old_v*k_cache02.y + new_v*k_cache02.y;

    reinterpret_cast<float4*>(&new_state[state_read+local_k])[0] = make_float4(valx, valy, valz, valw);

    const nv_bfloat162* q_cache4 = reinterpret_cast<const nv_bfloat162*> (&q[k_offset+local_k]);
    float2 q_cache01 = __bfloat1622float2(q_cache4[0]);
    float2 q_cache02 = __bfloat1622float2(q_cache4[1]);

    float out_acc = 0.0f;
    out_acc += scale * valx * q_cache01.x;
    out_acc += scale * valy * q_cache01.y;
    out_acc += scale * valz * q_cache02.x;
    out_acc += scale * valw * q_cache02.y;

# pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        out_acc += __shfl_down_sync(0xffffffff, out_acc, off);
    }

    // Write out
    if (threadIdx.x == 0) {
        out[v_offset+local_v] = __float2bfloat16_rn(out_acc);
    }
}

extern "C" cudaError_t launch_gdn(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* state,
    const float* A_log,
    const __nv_bfloat16* a,
    const float* dt_bias,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    float* new_state,
    int B,
    int num_v_heads,
    int num_k_heads,
    int K,
    int V,
    float scale,
    cudaStream_t stream
) {

    dim3 threads_per_block(32, kSplitV);
    dim3 grid_size(B, num_v_heads, (V+kSplitV-1)/kSplitV);

    gdn_v2<<<grid_size, threads_per_block, 0, stream>>>(
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

    __nv_bfloat16* q = nullptr;
    __nv_bfloat16* k = nullptr;
    __nv_bfloat16* v = nullptr;
    float* state = nullptr;
    float* A_log = nullptr;
    __nv_bfloat16* a = nullptr;
    float* dt_bias = nullptr;
    __nv_bfloat16* b = nullptr;
    __nv_bfloat16* out = nullptr;
    float* new_state = nullptr;
    const float scale = 1.0f / std::sqrt(static_cast<float>(K));

    cudaMallocManaged(&q, q_elems * sizeof(__nv_bfloat16));
    cudaMallocManaged(&k, k_elems * sizeof(__nv_bfloat16));
    cudaMallocManaged(&v, v_elems * sizeof(__nv_bfloat16));
    cudaMallocManaged(&state, state_elems * sizeof(float));
    cudaMallocManaged(&A_log, gate_elems * sizeof(float));
    cudaMallocManaged(&a, gate_elems * sizeof(__nv_bfloat16));
    cudaMallocManaged(&dt_bias, gate_elems * sizeof(float));
    cudaMallocManaged(&b, gate_elems * sizeof(__nv_bfloat16));
    cudaMallocManaged(&out, out_elems * sizeof(__nv_bfloat16));
    cudaMallocManaged(&new_state, state_elems * sizeof(float));

    for (size_t i = 0; i < q_elems; ++i) q[i] = __float2bfloat16_rn(0.01f * static_cast<float>(i % 17));
    for (size_t i = 0; i < k_elems; ++i) k[i] = __float2bfloat16_rn(0.02f * static_cast<float>(i % 13));
    for (size_t i = 0; i < v_elems; ++i) v[i] = __float2bfloat16_rn(0.03f * static_cast<float>(i % 11));
    for (size_t i = 0; i < state_elems; ++i) {
        state[i] = 0.0f;
        new_state[i] = 0.0f;
    }
    for (size_t i = 0; i < gate_elems; ++i) {
        A_log[i] = -2.0f;
        a[i] = __float2bfloat16_rn(0.0f);
        dt_bias[i] = 0.1f;
        b[i] = __float2bfloat16_rn(0.0f);
    }
    for (size_t i = 0; i < out_elems; ++i) out[i] = __float2bfloat16_rn(0.0f);

    const cudaError_t launch_err = launch_gdn(
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
        std::printf("launch_gdn failed: %s\n", cudaGetErrorString(launch_err));
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

void launch_gdn_torch(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    float scale,
    torch::Tensor out,
    torch::Tensor new_state
) {
    const int B = static_cast<int>(q.size(0));
    const int num_k_heads = static_cast<int>(k.size(2));
    const int num_v_heads = static_cast<int>(v.size(2));
    const int K = static_cast<int>(q.size(3));
    const int V = static_cast<int>(v.size(3));

    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    (void)launch_gdn(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        state.data_ptr<float>(),
        A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        new_state.data_ptr<float>(),
        B,
        num_v_heads,
        num_k_heads,
        K,
        V,
        scale,
        stream.stream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gdn", &launch_gdn_torch, "Launch gdn_v1 CUDA kernel");
}
#endif  // TORCH_EXTENSION_NAME
