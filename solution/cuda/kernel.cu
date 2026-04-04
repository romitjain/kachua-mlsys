//  Launch command
// nvcc -std=c++17 -Wno-deprecated-gpu-targets -o kernel.o kernel.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cmath>

constexpr int kSplitV = 8;
constexpr int kSmallVTileRows = 2;
constexpr int kMediumVTileRows = 4;
constexpr int kLargeVTileRows = 8;
static constexpr int kWarpSize = 32;
static constexpr int kHeadSize = 128;
static constexpr int kNumQKHeads = 4;
static constexpr int kNumVHeads = 8;
static constexpr int kHeadGroupRatio = kNumVHeads / kNumQKHeads;
static constexpr int kKVec = kHeadSize / kWarpSize;
static constexpr int kVDim = 128;
static constexpr float kScaleConst = 0.0883883476f;

__device__ __forceinline__ float sigmoid(float x) {
    // Stable sigmoid
    if (x >= 0.0f) {
        float z = __expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = __expf(x);
        return z / (1.0f + z);
    }
}

__device__ __forceinline__ void warp_reduce_2(float& a, float& b) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float ta;
        float tb;
        asm volatile(
            "shfl.sync.bfly.b32 %0, %2, %4, 0x1f, 0xffffffff;\n\t"
            "shfl.sync.bfly.b32 %1, %3, %4, 0x1f, 0xffffffff;"
            : "=f"(ta), "=f"(tb)
            : "f"(a), "f"(b), "r"(off)
        );
        a += ta;
        b += tb;
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
        float x = __bfloat162float(a[common_scalar_offset]) + dt_bias[blockIdx.y];
        gate = __expf(-__expf(A_log[blockIdx.y]) * __logf(1.0 + __expf(x)));
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

template <int BV>
__global__ __launch_bounds__(kWarpSize, 1)
void gdn_v3(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float*         __restrict__ h0,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_gate,
    const float*         __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_gate,
    __nv_bfloat16* __restrict__ output,
    float*         __restrict__ ht,
    int B,
    int num_v_heads,
    int num_k_heads,
    int K,
    int V,
    float scale
) {
    const int i_v   = blockIdx.x;
    const int i_nh  = blockIdx.y;
    const int i_hv  = i_nh % kNumVHeads;
    const int i_h   = i_hv / kHeadGroupRatio;
    const int i_n   = i_nh / kNumVHeads;
    const int tid   = threadIdx.x;
    const int k_base = tid * kKVec;

    // ---- Load state tile [BV, BK] via float4 (__ldg read-only cache) ----
    // Issue ALL loads before extracting to maximize memory-level parallelism
    const int sho = i_nh * kVDim * kHeadSize;
    float4 h4[BV];
#pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = i_v * BV + bv;
        h4[bv] = __ldg(reinterpret_cast<const float4*>(
            h0 + sho + v_idx * kHeadSize + k_base));
    }
    float h[BV][kKVec];
#pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        h[bv][0] = h4[bv].x; h[bv][1] = h4[bv].y;
        h[bv][2] = h4[bv].z; h[bv][3] = h4[bv].w;
    }

    // ---- Load q, k [KVEC per thread] bf16→f32 via uint2 (64-bit vectorized) ----
    float q_reg[kKVec], k_reg[kKVec];
    {
        const int qko = (i_n * kNumQKHeads + i_h) * kHeadSize + k_base;
        uint2 q_raw = *reinterpret_cast<const uint2*>(q + qko);
        uint2 k_raw = *reinterpret_cast<const uint2*>(k + qko);
        __nv_bfloat162 q01 = *reinterpret_cast<__nv_bfloat162*>(&q_raw.x);
        __nv_bfloat162 q23 = *reinterpret_cast<__nv_bfloat162*>(&q_raw.y);
        __nv_bfloat162 k01 = *reinterpret_cast<__nv_bfloat162*>(&k_raw.x);
        __nv_bfloat162 k23 = *reinterpret_cast<__nv_bfloat162*>(&k_raw.y);
        q_reg[0] = __bfloat162float(q01.x) * kScaleConst;
        q_reg[1] = __bfloat162float(q01.y) * kScaleConst;
        q_reg[2] = __bfloat162float(q23.x) * kScaleConst;
        q_reg[3] = __bfloat162float(q23.y) * kScaleConst;
        k_reg[0] = __bfloat162float(k01.x);
        k_reg[1] = __bfloat162float(k01.y);
        k_reg[2] = __bfloat162float(k23.x);
        k_reg[3] = __bfloat162float(k23.y);
    }

    // ---- Load v [BV] bf16→f32 via uint4 (128-bit broadcast) ----
    float v_reg[BV];
    {
        const __nv_bfloat16* v_ptr = v + (i_n * kNumVHeads + i_hv) * kVDim + i_v * BV;
        uint4 v_raw = *reinterpret_cast<const uint4*>(v_ptr);
        __nv_bfloat162 v01 = *reinterpret_cast<__nv_bfloat162*>(&v_raw.x);
        __nv_bfloat162 v23 = *reinterpret_cast<__nv_bfloat162*>(&v_raw.y);
        __nv_bfloat162 v45 = *reinterpret_cast<__nv_bfloat162*>(&v_raw.z);
        __nv_bfloat162 v67 = *reinterpret_cast<__nv_bfloat162*>(&v_raw.w);
        v_reg[0] = __bfloat162float(v01.x); v_reg[1] = __bfloat162float(v01.y);
        v_reg[2] = __bfloat162float(v23.x); v_reg[3] = __bfloat162float(v23.y);
        v_reg[4] = __bfloat162float(v45.x); v_reg[5] = __bfloat162float(v45.y);
        v_reg[6] = __bfloat162float(v67.x); v_reg[7] = __bfloat162float(v67.y);
    }

    // ---- Gate computation (fast math intrinsics) ----
    const float x  = __bfloat162float(a_gate[i_n * kNumVHeads + i_hv]) + dt_bias[i_hv];
    const float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x));
    const float g  = __expf(-__expf(A_log[i_hv]) * sp);
    const float beta = __frcp_rn(1.0f + __expf(
        -__bfloat162float(b_gate[i_n * kNumVHeads + i_hv])));

    // ---- Fused decay + delta rule (2 rows at a time for paired reduction) ----
#pragma unroll
    for (int bv = 0; bv < BV; bv += 2) {
        float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
        for (int i = 0; i < kKVec; i++) {
            h[bv][i] *= g;
            h[bv+1][i] *= g;
            p0 = fmaf(k_reg[i], h[bv][i], p0);
            p1 = fmaf(k_reg[i], h[bv+1][i], p1);
        }
        warp_reduce_2(p0, p1);
        float dv0 = beta * (v_reg[bv] - p0);
        float dv1 = beta * (v_reg[bv+1] - p1);
#pragma unroll
        for (int i = 0; i < kKVec; i++) {
            h[bv][i] = fmaf(dv0, k_reg[i], h[bv][i]);
            h[bv+1][i] = fmaf(dv1, k_reg[i], h[bv+1][i]);
        }
    }

    // ---- Interleaved output + state store ----
    {
        __nv_bfloat16* o_ptr = output + (i_n * kNumVHeads + i_hv) * kVDim + i_v * BV;
        float out_vals[BV];
#pragma unroll
        for (int bv = 0; bv < BV; bv += 2) {
            float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
            for (int i = 0; i < kKVec; i++) {
                p0 = fmaf(q_reg[i], h[bv][i], p0);
                p1 = fmaf(q_reg[i], h[bv+1][i], p1);
            }
            warp_reduce_2(p0, p1);
            out_vals[bv] = p0;
            out_vals[bv+1] = p1;

            int vi0 = i_v * BV + bv;
            int vi1 = vi0 + 1;
            float4 sv0 = {h[bv][0], h[bv][1], h[bv][2], h[bv][3]};
            float4 sv1 = {h[bv+1][0], h[bv+1][1], h[bv+1][2], h[bv+1][3]};
            __stcs(reinterpret_cast<float4*>(ht + sho + vi0 * kHeadSize + k_base), sv0);
            __stcs(reinterpret_cast<float4*>(ht + sho + vi1 * kHeadSize + k_base), sv1);
        }
        if (tid < BV)
            o_ptr[tid] = __float2bfloat16(out_vals[tid]);
    }
}

template <int BV>
__global__ __launch_bounds__(kWarpSize, 1) void gdn_v4(
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
    const int i_v = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_hv = i_nh % kNumVHeads;
    const int i_h = i_hv / kHeadGroupRatio;
    const int i_n = i_nh / kNumVHeads;
    const int tid = threadIdx.x;
    const int k_base = tid * kKVec;
    const int state_base = i_nh * kVDim * kHeadSize;

    // Loading state - in 2D foramt (rows, 4)
    // We do float4 loads we get 4 individual values.
    // Done for the complete warp, we get 4*32 = 128 vals (head dim)
    float h[BV][kKVec];
#pragma unroll
    for (int row = 0; row < BV; ++row) {
        const int v_idx = i_v * BV + row;
        const float* state_ptr = state + state_base + v_idx * kHeadSize + k_base;
        const float4 h_vec = __ldg(reinterpret_cast<const float4*>(state_ptr));
        h[row][0] = h_vec.x;
        h[row][1] = h_vec.y;
        h[row][2] = h_vec.z;
        h[row][3] = h_vec.w;
    }

    // Loading q/k rows - same for multiple rows of V
    float q_reg[kKVec];
    float k_reg[kKVec];
    {
        const int qk_offset = (i_n * kNumQKHeads + i_h) * kHeadSize + k_base;
        const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
        const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
        __nv_bfloat162 q_pair01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
        __nv_bfloat162 q_pair23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
        __nv_bfloat162 k_pair01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
        __nv_bfloat162 k_pair23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
        q_reg[0] = __bfloat162float(q_pair01.x) * kScaleConst;
        q_reg[1] = __bfloat162float(q_pair01.y) * kScaleConst;
        q_reg[2] = __bfloat162float(q_pair23.x) * kScaleConst;
        q_reg[3] = __bfloat162float(q_pair23.y) * kScaleConst;
        k_reg[0] = __bfloat162float(k_pair01.x);
        k_reg[1] = __bfloat162float(k_pair01.y);
        k_reg[2] = __bfloat162float(k_pair23.x);
        k_reg[3] = __bfloat162float(k_pair23.y);
    }

    // Loading BV rows of V
    float v_reg[BV];
    {
        const __nv_bfloat16* v_ptr = v + i_nh * kVDim + i_v * BV;
        if constexpr (BV == 8) {
            const uint4 v_raw = *reinterpret_cast<const uint4*>(v_ptr);
            __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.x);
            __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.y);
            __nv_bfloat162 v45 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.z);
            __nv_bfloat162 v67 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.w);
            v_reg[0] = __bfloat162float(v01.x);
            v_reg[1] = __bfloat162float(v01.y);
            v_reg[2] = __bfloat162float(v23.x);
            v_reg[3] = __bfloat162float(v23.y);
            v_reg[4] = __bfloat162float(v45.x);
            v_reg[5] = __bfloat162float(v45.y);
            v_reg[6] = __bfloat162float(v67.x);
            v_reg[7] = __bfloat162float(v67.y);
        } else if constexpr (BV == 4) {
            const uint2 v_raw = *reinterpret_cast<const uint2*>(v_ptr);
            __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.x);
            __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.y);
            v_reg[0] = __bfloat162float(v01.x);
            v_reg[1] = __bfloat162float(v01.y);
            v_reg[2] = __bfloat162float(v23.x);
            v_reg[3] = __bfloat162float(v23.y);
        } else if constexpr (BV == 2) {
            const uint32_t v_raw = *reinterpret_cast<const uint32_t*>(v_ptr);
            __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw);
            v_reg[0] = __bfloat162float(v01.x);
            v_reg[1] = __bfloat162float(v01.y);
        }
    }

    // Computing scalars
    const float gate_x = __bfloat162float(a[i_nh]) + dt_bias[i_hv];
    const float gate_sp = __logf(1.0f + __expf(gate_x));
    const float gate = __expf(-__expf(A_log[i_hv]) * gate_sp);
    const float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b[i_nh])));

    // Fused state update + out computation + state store
    __nv_bfloat16* out_ptr = out + i_nh * kVDim + i_v * BV;
    float out_vals[BV];
#pragma unroll
    for (int row = 0; row < BV; row += 2) {
        float old_v0 = 0.0f;
        float old_v1 = 0.0f;
#pragma unroll
        for (int i = 0; i < kKVec; ++i) {
            h[row][i] *= gate;
            h[row + 1][i] *= gate;
            old_v0 = fmaf(k_reg[i], h[row][i], old_v0);
            old_v1 = fmaf(k_reg[i], h[row + 1][i], old_v1);
        }
        warp_reduce_2(old_v0, old_v1);
        const float dv0 = beta * (v_reg[row] - old_v0);
        const float dv1 = beta * (v_reg[row + 1] - old_v1);
        float out0 = 0.0f;
        float out1 = 0.0f;
#pragma unroll
        for (int i = 0; i < kKVec; ++i) {
            h[row][i] = fmaf(dv0, k_reg[i], h[row][i]);
            h[row + 1][i] = fmaf(dv1, k_reg[i], h[row + 1][i]);
            out0 = fmaf(q_reg[i], h[row][i], out0);
            out1 = fmaf(q_reg[i], h[row + 1][i], out1);
        }
        warp_reduce_2(out0, out1);
        out_vals[row] = out0;
        out_vals[row + 1] = out1;

        const int v_idx0 = i_v * BV + row;
        const int v_idx1 = v_idx0 + 1;
        const float4 new_state_vec0 = {h[row][0], h[row][1], h[row][2], h[row][3]};
        const float4 new_state_vec1 = {h[row + 1][0], h[row + 1][1], h[row + 1][2], h[row + 1][3]};
        __stcs(reinterpret_cast<float4*>(new_state + state_base + v_idx0 * kHeadSize + k_base), new_state_vec0);
        __stcs(reinterpret_cast<float4*>(new_state + state_base + v_idx1 * kHeadSize + k_base), new_state_vec1);
    }

    // Out store
    if (tid < BV) {
        out_ptr[tid] = __float2bfloat16_rn(out_vals[tid]);
    }
}

#ifndef TORCH_EXTENSION_NAME
int main() {
    constexpr int B = 1;
    constexpr int T = 1;
    // Official decode contest shape: q/k heads = 4, v heads = 8.
    constexpr int num_q_heads = 4;
    constexpr int num_k_heads = 4;
    constexpr int num_v_heads = 8;
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

    dim3 threads_per_block(kWarpSize);
    if (B == 1) {
        dim3 grid_size(kVDim / kSmallVTileRows, B * kNumVHeads);
        gdn_v4<kSmallVTileRows><<<grid_size, threads_per_block, 0, nullptr>>>(
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
    } else if (B <= 4) {
        dim3 grid_size(kVDim / kMediumVTileRows, B * kNumVHeads);
        gdn_v4<kMediumVTileRows><<<grid_size, threads_per_block, 0, nullptr>>>(
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
    } else {
        dim3 grid_size(kVDim / kMediumVTileRows, B * kNumVHeads);
        gdn_v4<kMediumVTileRows><<<grid_size, threads_per_block, 0, nullptr>>>(
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
    }
    const cudaError_t launch_err = cudaGetLastError();
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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_gdn(
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
    dim3 threads_per_block(kWarpSize);
    if (B == 1) {
        dim3 grid_size(kVDim / kSmallVTileRows, B * kNumVHeads);
        gdn_v4<kSmallVTileRows><<<grid_size, threads_per_block, 0, stream.stream()>>>(
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
            scale
        );
    } else if (B <= 4) {
        dim3 grid_size(kVDim / kMediumVTileRows, B * kNumVHeads);
        gdn_v4<kMediumVTileRows><<<grid_size, threads_per_block, 0, stream.stream()>>>(
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
            scale
        );
    } else {
        dim3 grid_size(kVDim / kMediumVTileRows, B * kNumVHeads);
        gdn_v4<kMediumVTileRows><<<grid_size, threads_per_block, 0, stream.stream()>>>(
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
            scale
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gdn", &launch_gdn, "Launch gdn CUDA kernel");
}
#endif