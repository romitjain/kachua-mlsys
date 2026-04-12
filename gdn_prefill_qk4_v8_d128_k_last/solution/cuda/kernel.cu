#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kNumQHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHeadGroupRatio = kNumVHeads / kNumQHeads;
constexpr int kKVec = kHeadDim / kWarpSize;
constexpr int kBV = 8;
constexpr int kNumVTiles = kHeadDim / kBV;

__device__ __forceinline__ float softplus_fast(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float z = __expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = __expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ void warp_reduce_2(float& a, float& b) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

template <int BV>
__global__ __launch_bounds__(kWarpSize, 1) void gdn_prefill_v1(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    const int64_t* __restrict__ cu_seqlens,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ new_state,
    int num_seqs,
    float scale
) {
    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumVTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) {
        return;
    }

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumVTiles;
    const int v_tile = rem - v_head * kNumVTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int k_base = tid * kKVec;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    float h[BV][kKVec];
#pragma unroll
    for (int row = 0; row < BV; ++row) {
        const int v_idx = v_base + row;
        const float4 h_vec = __ldg(reinterpret_cast<const float4*>(
            state + state_base + v_idx * kHeadDim + k_base
        ));
        h[row][0] = h_vec.x;
        h[row][1] = h_vec.y;
        h[row][2] = h_vec.z;
        h[row][3] = h_vec.w;
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    if (seq_end <= seq_start) {
#pragma unroll
        for (int row = 0; row < BV; ++row) {
            const int v_idx = v_base + row;
            const float4 out_vec = {h[row][0], h[row][1], h[row][2], h[row][3]};
            reinterpret_cast<float4*>(
                new_state + state_base + v_idx * kHeadDim + k_base
            )[0] = out_vec;
        }
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];

    for (int token_idx = seq_start; token_idx < seq_end; ++token_idx) {
        float q_reg[kKVec];
        float k_reg[kKVec];
        {
            const int qk_offset = (token_idx * kNumQHeads + qk_head) * kHeadDim + k_base;
            const uint2 q_raw = *reinterpret_cast<const uint2*>(q + qk_offset);
            const uint2 k_raw = *reinterpret_cast<const uint2*>(k + qk_offset);
            const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
            const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
            const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
            const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
            q_reg[0] = __bfloat162float(q01.x) * scale;
            q_reg[1] = __bfloat162float(q01.y) * scale;
            q_reg[2] = __bfloat162float(q23.x) * scale;
            q_reg[3] = __bfloat162float(q23.y) * scale;
            k_reg[0] = __bfloat162float(k01.x);
            k_reg[1] = __bfloat162float(k01.y);
            k_reg[2] = __bfloat162float(k23.x);
            k_reg[3] = __bfloat162float(k23.y);
        }

        float v_reg[BV];
        {
            const __nv_bfloat16* v_ptr = v + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base;
            const uint4 v_raw = *reinterpret_cast<const uint4*>(v_ptr);
            const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.x);
            const __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.y);
            const __nv_bfloat162 v45 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.z);
            const __nv_bfloat162 v67 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.w);
            v_reg[0] = __bfloat162float(v01.x);
            v_reg[1] = __bfloat162float(v01.y);
            v_reg[2] = __bfloat162float(v23.x);
            v_reg[3] = __bfloat162float(v23.y);
            v_reg[4] = __bfloat162float(v45.x);
            v_reg[5] = __bfloat162float(v45.y);
            v_reg[6] = __bfloat162float(v67.x);
            v_reg[7] = __bfloat162float(v67.y);
        }

        const int gate_offset = token_idx * kNumVHeads + v_head;
        const float gate_x = __bfloat162float(a[gate_offset]) + dt_bias_val;
        const float g = __expf(-decay * softplus_fast(gate_x));
        const float beta = sigmoid_fast(__bfloat162float(b[gate_offset]));

        float kq = 0.0f;
#pragma unroll
        for (int i = 0; i < kKVec; ++i) {
            kq = fmaf(k_reg[i], q_reg[i], kq);
        }
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            kq += __shfl_down_sync(0xffffffff, kq, offset);
        }
        kq = __shfl_sync(0xffffffff, kq, 0);

        float out_vals[BV];
#pragma unroll
        for (int row = 0; row < BV; row += 2) {
            float old_v0 = 0.0f;
            float old_v1 = 0.0f;
            float old_o0 = 0.0f;
            float old_o1 = 0.0f;
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[row][i] *= g;
                h[row + 1][i] *= g;
                old_v0 = fmaf(k_reg[i], h[row][i], old_v0);
                old_v1 = fmaf(k_reg[i], h[row + 1][i], old_v1);
                old_o0 = fmaf(q_reg[i], h[row][i], old_o0);
                old_o1 = fmaf(q_reg[i], h[row + 1][i], old_o1);
            }

            warp_reduce_2(old_v0, old_v1);
            warp_reduce_2(old_o0, old_o1);

            float delta_v0 = 0.0f;
            float delta_v1 = 0.0f;
            if (tid == 0) {
                delta_v0 = beta * (v_reg[row] - old_v0);
                delta_v1 = beta * (v_reg[row + 1] - old_v1);
                out_vals[row] = old_o0 + delta_v0 * kq;
                out_vals[row + 1] = old_o1 + delta_v1 * kq;
            }
            delta_v0 = __shfl_sync(0xffffffff, delta_v0, 0);
            delta_v1 = __shfl_sync(0xffffffff, delta_v1, 0);

#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[row][i] = fmaf(delta_v0, k_reg[i], h[row][i]);
                h[row + 1][i] = fmaf(delta_v1, k_reg[i], h[row + 1][i]);
            }
        }

        if (tid == 0) {
            __nv_bfloat16* out_ptr = out + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base;
#pragma unroll
            for (int row = 0; row < BV; ++row) {
                out_ptr[row] = __float2bfloat16_rn(out_vals[row]);
            }
        }
    }

#pragma unroll
    for (int row = 0; row < BV; ++row) {
        const int v_idx = v_base + row;
        const float4 out_vec = {h[row][0], h[row][1], h[row][2], h[row][3]};
        reinterpret_cast<float4*>(
            new_state + state_base + v_idx * kHeadDim + k_base
        )[0] = out_vec;
    }
}

}  // namespace

void launch_gdn(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    float scale,
    torch::Tensor out,
    torch::Tensor new_state
) {
    const int num_seqs = static_cast<int>(cu_seqlens.size(0) - 1);
    const int total_tiles = num_seqs * kNumVHeads * kNumVTiles;
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    gdn_prefill_v1<kBV><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        state.data_ptr<float>(),
        A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        cu_seqlens.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        new_state.data_ptr<float>(),
        num_seqs,
        scale
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gdn", &launch_gdn, "Launch GDN prefill CUDA kernel");
}
