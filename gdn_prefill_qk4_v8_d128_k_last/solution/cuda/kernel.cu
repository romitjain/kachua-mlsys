#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kNumQHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHeadGroupRatio = kNumVHeads / kNumQHeads;
constexpr int kKVec = kHeadDim / kWarpSize;
constexpr int kBV = 8;
constexpr int kNumVTiles = kHeadDim / kBV;
constexpr int kLongBV = 4;
constexpr int kStarvedBV = 2;
constexpr int kBenchmarkKernelVersion = 8;

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
        float ta;
        float tb;
        asm volatile(
            "shfl.sync.bfly.b32 %0, %2, %4, 0x1f, 0xffffffff;\n\t"
            "shfl.sync.bfly.b32 %1, %3, %4, 0x1f, 0xffffffff;"
            : "=f"(ta), "=f"(tb)
            : "f"(a), "f"(b), "r"(offset)
        );
        a += ta;
        b += tb;
    }
}

template <int BV>
__device__ __forceinline__ void load_v_regs(
    const __nv_bfloat16* __restrict__ v_ptr,
    float (&v_reg)[BV]
) {
    if constexpr (BV == 8) {
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
    } else if constexpr (BV == 4) {
        const uint2 v_raw = *reinterpret_cast<const uint2*>(v_ptr);
        const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.x);
        const __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.y);
        v_reg[0] = __bfloat162float(v01.x);
        v_reg[1] = __bfloat162float(v01.y);
        v_reg[2] = __bfloat162float(v23.x);
        v_reg[3] = __bfloat162float(v23.y);
    } else {
        const uint32_t v_raw = *reinterpret_cast<const uint32_t*>(v_ptr);
        const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&v_raw);
        v_reg[0] = __bfloat162float(v01.x);
        v_reg[1] = __bfloat162float(v01.y);
    }
}

template <int BV>
__device__ __forceinline__ void store_state_tile(
    float* __restrict__ dst,
    int state_base,
    int v_base,
    int k_base,
    const float (&h)[BV][kKVec]
) {
#pragma unroll
    for (int row = 0; row < BV; ++row) {
        const int v_idx = v_base + row;
        const float4 out_vec = {h[row][0], h[row][1], h[row][2], h[row][3]};
        __stcs(reinterpret_cast<float4*>(dst + state_base + v_idx * kHeadDim + k_base), out_vec);
    }
}

template <int ROWS>
__device__ __forceinline__ void load_v_fragment(
    const __nv_bfloat16* __restrict__ v_ptr,
    float (&v_reg)[ROWS]
) {
#pragma unroll
    for (int row = 0; row < ROWS; ++row) {
        v_reg[row] = __bfloat162float(v_ptr[row]);
    }
}

template <int BV>
__global__ __launch_bounds__(kWarpSize, 1) void gdn_prefill_v2(
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
    constexpr int kNumTiles = kHeadDim / BV;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) {
        return;
    }

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int k_base = tid * kKVec;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    float4 h4[BV];
#pragma unroll
    for (int row = 0; row < BV; ++row) {
        const int v_idx = v_base + row;
        h4[row] = __ldg(reinterpret_cast<const float4*>(
            state + state_base + v_idx * kHeadDim + k_base
        ));
    }

    float h[BV][kKVec];
#pragma unroll
    for (int row = 0; row < BV; ++row) {
        h[row][0] = h4[row].x;
        h[row][1] = h4[row].y;
        h[row][2] = h4[row].z;
        h[row][3] = h4[row].w;
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    if (seq_end <= seq_start) {
        store_state_tile<BV>(new_state, state_base, v_base, k_base, h);
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
            const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
            const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
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
        load_v_regs<BV>(v + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base, v_reg);

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

    store_state_tile<BV>(new_state, state_base, v_base, k_base, h);
}

template <int BV, int CHUNK>
__global__ __launch_bounds__(kWarpSize, 1) void gdn_prefill_v4(
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
    constexpr int kNumTiles = kHeadDim / BV;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) {
        return;
    }

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
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
        store_state_tile<BV>(new_state, state_base, v_base, k_base, h);
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];

    __shared__ float s_g[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_gamma[CHUNK];
    __shared__ float s_v[CHUNK][BV];
    __shared__ float s_kproj[CHUNK][BV];
    __shared__ float s_qproj[CHUNK][BV];
    __shared__ float s_kk[CHUNK][CHUNK];
    __shared__ float s_qk[CHUNK][CHUNK];
    __shared__ float s_lower[CHUNK][CHUNK];
    __shared__ float s_qcoeff[CHUNK][CHUNK];
    __shared__ float s_self_q[CHUNK];
    __shared__ float s_state_coeff[CHUNK];
    __shared__ float s_delta[CHUNK][BV];
    __shared__ float s_out[CHUNK][BV];

    float q_chunk[CHUNK][kKVec];
    float k_chunk[CHUNK][kKVec];

    int token_idx = seq_start;
    for (; token_idx + CHUNK <= seq_end; token_idx += CHUNK) {
#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            const int curr_token = token_idx + m;
            const int qk_offset = (curr_token * kNumQHeads + qk_head) * kHeadDim + k_base;
            const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
            const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
            const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
            const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
            const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
            const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
            q_chunk[m][0] = __bfloat162float(q01.x) * scale;
            q_chunk[m][1] = __bfloat162float(q01.y) * scale;
            q_chunk[m][2] = __bfloat162float(q23.x) * scale;
            q_chunk[m][3] = __bfloat162float(q23.y) * scale;
            k_chunk[m][0] = __bfloat162float(k01.x);
            k_chunk[m][1] = __bfloat162float(k01.y);
            k_chunk[m][2] = __bfloat162float(k23.x);
            k_chunk[m][3] = __bfloat162float(k23.y);

            if (tid == 0) {
                float v_reg[BV];
                load_v_regs<BV>(v + (curr_token * kNumVHeads + v_head) * kHeadDim + v_base, v_reg);
#pragma unroll
                for (int row = 0; row < BV; ++row) {
                    s_v[m][row] = v_reg[row];
                }
                const int gate_offset = curr_token * kNumVHeads + v_head;
                const float gate_x = __bfloat162float(a[gate_offset]) + dt_bias_val;
                s_g[m] = __expf(-decay * softplus_fast(gate_x));
                s_beta[m] = sigmoid_fast(__bfloat162float(b[gate_offset]));
            }
        }

        __syncwarp();

        if (tid == 0) {
            float gamma = 1.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                gamma *= s_g[m];
                s_gamma[m] = gamma;
            }
        }

        __syncwarp();

#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
#pragma unroll
            for (int row = 0; row < BV; ++row) {
                float k_proj = 0.0f;
                float q_proj = 0.0f;
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    k_proj = fmaf(k_chunk[m][i], h[row][i], k_proj);
                    q_proj = fmaf(q_chunk[m][i], h[row][i], q_proj);
                }
                warp_reduce_2(k_proj, q_proj);
                if (tid == 0) {
                    s_kproj[m][row] = k_proj;
                    s_qproj[m][row] = q_proj;
                }
            }
        }

#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
#pragma unroll
            for (int j = 0; j <= m; ++j) {
                float qk_dot = 0.0f;
                float kk_dot = 0.0f;
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    qk_dot = fmaf(q_chunk[m][i], k_chunk[j][i], qk_dot);
                    if (j < m) {
                        kk_dot = fmaf(k_chunk[m][i], k_chunk[j][i], kk_dot);
                    }
                }
                warp_reduce_2(qk_dot, kk_dot);
                if (tid == 0) {
                    s_qk[m][j] = qk_dot;
                    if (j < m) {
                        s_kk[m][j] = kk_dot;
                    }
                }
            }
        }

        __syncwarp();

        if (tid == 0) {
            const float gamma_last = s_gamma[CHUNK - 1];
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                s_self_q[m] = s_qk[m][m] * s_beta[m];
                s_state_coeff[m] = (gamma_last / s_gamma[m]) * s_beta[m];
#pragma unroll
                for (int j = 0; j < m; ++j) {
                    const float ratio = s_gamma[m] / s_gamma[j];
                    s_lower[m][j] = ratio * s_kk[m][j] * s_beta[j];
                    s_qcoeff[m][j] = ratio * s_qk[m][j] * s_beta[j];
                }
            }

#pragma unroll
            for (int row = 0; row < BV; ++row) {
#pragma unroll
                for (int m = 0; m < CHUNK; ++m) {
                    float delta = s_v[m][row] - s_gamma[m] * s_kproj[m][row];
#pragma unroll
                    for (int j = 0; j < m; ++j) {
                        delta -= s_lower[m][j] * s_delta[j][row];
                    }
                    s_delta[m][row] = delta;

                    float out_val = s_gamma[m] * s_qproj[m][row];
#pragma unroll
                    for (int j = 0; j < m; ++j) {
                        out_val += s_qcoeff[m][j] * s_delta[j][row];
                    }
                    out_val += s_self_q[m] * delta;
                    s_out[m][row] = out_val;
                }
            }
        }

        __syncwarp();

        if (tid == 0) {
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                __nv_bfloat16* out_ptr = out + ((token_idx + m) * kNumVHeads + v_head) * kHeadDim + v_base;
#pragma unroll
                for (int row = 0; row < BV; ++row) {
                    out_ptr[row] = __float2bfloat16_rn(s_out[m][row]);
                }
            }
        }

        const float gamma_last = s_gamma[CHUNK - 1];
#pragma unroll
        for (int row = 0; row < BV; ++row) {
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[row][i] *= gamma_last;
            }
        }

#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
#pragma unroll
            for (int row = 0; row < BV; ++row) {
                float delta_scaled = 0.0f;
                if (tid == 0) {
                    delta_scaled = s_state_coeff[m] * s_delta[m][row];
                }
                delta_scaled = __shfl_sync(0xffffffff, delta_scaled, 0);
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    h[row][i] = fmaf(delta_scaled, k_chunk[m][i], h[row][i]);
                }
            }
        }
    }

    for (; token_idx < seq_end; ++token_idx) {
        float q_reg[kKVec];
        float k_reg[kKVec];
        {
            const int qk_offset = (token_idx * kNumQHeads + qk_head) * kHeadDim + k_base;
            const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
            const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
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
        load_v_regs<BV>(v + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base, v_reg);

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

    store_state_tile<BV>(new_state, state_base, v_base, k_base, h);
}

template <int BV, int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 1) void gdn_prefill_v6(
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
    constexpr int kGroupBV = BV * WARPS;
    constexpr int kNumTileGroups = kHeadDim / kGroupBV;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTileGroups;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) {
        return;
    }

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTileGroups;
    const int tile_group = rem - v_head * kNumTileGroups;
    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int k_base = lane_id * kKVec;
    const int qk_head = v_head / kHeadGroupRatio;
    const int v_base = tile_group * kGroupBV + warp_id * BV;
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
        store_state_tile<BV>(new_state, state_base, v_base, k_base, h);
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];

    __shared__ float s_q_chunk[CHUNK][kHeadDim];
    __shared__ float s_k_chunk[CHUNK][kHeadDim];
    __shared__ float s_g[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_gamma[CHUNK];
    __shared__ float s_v[WARPS][CHUNK][BV];
    __shared__ float s_kproj[WARPS][CHUNK][BV];
    __shared__ float s_qproj[WARPS][CHUNK][BV];
    __shared__ float s_delta[WARPS][CHUNK][BV];
    __shared__ float s_out[WARPS][CHUNK][BV];
    __shared__ float s_kk[CHUNK][CHUNK];
    __shared__ float s_qk[CHUNK][CHUNK];
    __shared__ float s_lower[CHUNK][CHUNK];
    __shared__ float s_qcoeff[CHUNK][CHUNK];
    __shared__ float s_self_q[CHUNK];
    __shared__ float s_state_coeff[CHUNK];

    int token_idx = seq_start;
    for (; token_idx + CHUNK <= seq_end; token_idx += CHUNK) {
#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            const int curr_token = token_idx + m;
            if (warp_id == 0) {
                const int qk_offset = (curr_token * kNumQHeads + qk_head) * kHeadDim + k_base;
                const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
                const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
                const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
                const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
                const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
                const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
                s_q_chunk[m][k_base + 0] = __bfloat162float(q01.x) * scale;
                s_q_chunk[m][k_base + 1] = __bfloat162float(q01.y) * scale;
                s_q_chunk[m][k_base + 2] = __bfloat162float(q23.x) * scale;
                s_q_chunk[m][k_base + 3] = __bfloat162float(q23.y) * scale;
                s_k_chunk[m][k_base + 0] = __bfloat162float(k01.x);
                s_k_chunk[m][k_base + 1] = __bfloat162float(k01.y);
                s_k_chunk[m][k_base + 2] = __bfloat162float(k23.x);
                s_k_chunk[m][k_base + 3] = __bfloat162float(k23.y);
            }

            if (lane_id == 0) {
                float v_reg[BV];
                load_v_regs<BV>(v + (curr_token * kNumVHeads + v_head) * kHeadDim + v_base, v_reg);
#pragma unroll
                for (int row = 0; row < BV; ++row) {
                    s_v[warp_id][m][row] = v_reg[row];
                }
            }

            if (threadIdx.x == 0) {
                const int gate_offset = curr_token * kNumVHeads + v_head;
                const float gate_x = __bfloat162float(a[gate_offset]) + dt_bias_val;
                s_g[m] = __expf(-decay * softplus_fast(gate_x));
                s_beta[m] = sigmoid_fast(__bfloat162float(b[gate_offset]));
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            float gamma = 1.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                gamma *= s_g[m];
                s_gamma[m] = gamma;
            }
        }

        __syncthreads();

#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            float q_vec[kKVec];
            float k_vec[kKVec];
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                q_vec[i] = s_q_chunk[m][k_base + i];
                k_vec[i] = s_k_chunk[m][k_base + i];
            }

#pragma unroll
            for (int row = 0; row < BV; ++row) {
                float k_proj = 0.0f;
                float q_proj = 0.0f;
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    k_proj = fmaf(k_vec[i], h[row][i], k_proj);
                    q_proj = fmaf(q_vec[i], h[row][i], q_proj);
                }
                warp_reduce_2(k_proj, q_proj);
                if (lane_id == 0) {
                    s_kproj[warp_id][m][row] = k_proj;
                    s_qproj[warp_id][m][row] = q_proj;
                }
            }
        }

        if (warp_id == 0) {
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                float q_m[kKVec];
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    q_m[i] = s_q_chunk[m][k_base + i];
                }

#pragma unroll
                for (int j = 0; j <= m; ++j) {
                    float qk_dot = 0.0f;
                    float kk_dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < kKVec; ++i) {
                        const float k_j = s_k_chunk[j][k_base + i];
                        qk_dot = fmaf(q_m[i], k_j, qk_dot);
                        if (j < m) {
                            kk_dot = fmaf(s_k_chunk[m][k_base + i], k_j, kk_dot);
                        }
                    }
                    warp_reduce_2(qk_dot, kk_dot);
                    if (lane_id == 0) {
                        s_qk[m][j] = qk_dot;
                        if (j < m) {
                            s_kk[m][j] = kk_dot;
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            const float gamma_last = s_gamma[CHUNK - 1];
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                s_self_q[m] = s_qk[m][m] * s_beta[m];
                s_state_coeff[m] = (gamma_last / s_gamma[m]) * s_beta[m];
#pragma unroll
                for (int j = 0; j < m; ++j) {
                    const float ratio = s_gamma[m] / s_gamma[j];
                    s_lower[m][j] = ratio * s_kk[m][j] * s_beta[j];
                    s_qcoeff[m][j] = ratio * s_qk[m][j] * s_beta[j];
                }
            }
        }

        __syncthreads();

        if (lane_id == 0) {
#pragma unroll
            for (int row = 0; row < BV; ++row) {
#pragma unroll
                for (int m = 0; m < CHUNK; ++m) {
                    float delta = s_v[warp_id][m][row] - s_gamma[m] * s_kproj[warp_id][m][row];
#pragma unroll
                    for (int j = 0; j < m; ++j) {
                        delta -= s_lower[m][j] * s_delta[warp_id][j][row];
                    }
                    s_delta[warp_id][m][row] = delta;

                    float out_val = s_gamma[m] * s_qproj[warp_id][m][row];
#pragma unroll
                    for (int j = 0; j < m; ++j) {
                        out_val += s_qcoeff[m][j] * s_delta[warp_id][j][row];
                    }
                    out_val += s_self_q[m] * delta;
                    s_out[warp_id][m][row] = out_val;
                }
            }
        }

        __syncthreads();

        if (lane_id == 0) {
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                __nv_bfloat16* out_ptr = out + ((token_idx + m) * kNumVHeads + v_head) * kHeadDim + v_base;
#pragma unroll
                for (int row = 0; row < BV; ++row) {
                    out_ptr[row] = __float2bfloat16_rn(s_out[warp_id][m][row]);
                }
            }
        }

        const float gamma_last = s_gamma[CHUNK - 1];
#pragma unroll
        for (int row = 0; row < BV; ++row) {
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[row][i] *= gamma_last;
            }
        }

#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            float k_vec[kKVec];
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                k_vec[i] = s_k_chunk[m][k_base + i];
            }

#pragma unroll
            for (int row = 0; row < BV; ++row) {
                float delta_scaled = 0.0f;
                if (lane_id == 0) {
                    delta_scaled = s_state_coeff[m] * s_delta[warp_id][m][row];
                }
                delta_scaled = __shfl_sync(0xffffffff, delta_scaled, 0);
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    h[row][i] = fmaf(delta_scaled, k_vec[i], h[row][i]);
                }
            }
        }

        __syncthreads();
    }

    for (; token_idx < seq_end; ++token_idx) {
        float q_reg[kKVec];
        float k_reg[kKVec];
        {
            const int qk_offset = (token_idx * kNumQHeads + qk_head) * kHeadDim + k_base;
            const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
            const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
            const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
            const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
            const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
            const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
            q_reg[0] = __bfloat162float(q01.x) * scale;
            q_reg[1] = __bfloat162float(q01.y) * scale;
            q_reg[2] = __bfloat162float(q23.x) * scale;
            q_reg[3] = __bfloat162float(q23.y);
            k_reg[0] = __bfloat162float(k01.x);
            k_reg[1] = __bfloat162float(k01.y);
            k_reg[2] = __bfloat162float(k23.x);
            k_reg[3] = __bfloat162float(k23.y);
        }

        float v_reg[BV];
        float g = 0.0f;
        float beta = 0.0f;
        if (lane_id == 0) {
            load_v_regs<BV>(v + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base, v_reg);
            const int gate_offset = token_idx * kNumVHeads + v_head;
            const float gate_x = __bfloat162float(a[gate_offset]) + dt_bias_val;
            g = __expf(-decay * softplus_fast(gate_x));
            beta = sigmoid_fast(__bfloat162float(b[gate_offset]));
        }
        g = __shfl_sync(0xffffffff, g, 0);

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
            if (lane_id == 0) {
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

        if (lane_id == 0) {
            __nv_bfloat16* out_ptr = out + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base;
#pragma unroll
            for (int row = 0; row < BV; ++row) {
                out_ptr[row] = __float2bfloat16_rn(out_vals[row]);
            }
        }
    }

    store_state_tile<BV>(new_state, state_base, v_base, k_base, h);
}

template <int BV, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 1) void gdn_prefill_v7(
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
    static_assert(BV % WARPS == 0, "BV must divide WARPS");
    constexpr int kRowsPerWarp = BV / WARPS;
    constexpr int kNumTiles = kHeadDim / BV;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) {
        return;
    }

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int qk_head = v_head / kHeadGroupRatio;
    const int k_base = lane_id * kKVec;
    const int v_base = v_tile * BV + warp_id * kRowsPerWarp;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    float h[kRowsPerWarp][kKVec];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
        const float4 h_vec = __ldg(reinterpret_cast<const float4*>(
            state + state_base + (v_base + row) * kHeadDim + k_base
        ));
        h[row][0] = h_vec.x;
        h[row][1] = h_vec.y;
        h[row][2] = h_vec.z;
        h[row][3] = h_vec.w;
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    if (seq_end <= seq_start) {
        store_state_tile<kRowsPerWarp>(new_state, state_base, v_base, k_base, h);
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
            const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
            const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
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

        float v_reg[kRowsPerWarp];
        float g = 0.0f;
        float beta = 0.0f;
        if (lane_id == 0) {
            load_v_fragment<kRowsPerWarp>(
                v + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base,
                v_reg
            );
            const int gate_offset = token_idx * kNumVHeads + v_head;
            const float gate_x = __bfloat162float(a[gate_offset]) + dt_bias_val;
            g = __expf(-decay * softplus_fast(gate_x));
            beta = sigmoid_fast(__bfloat162float(b[gate_offset]));
        }
        g = __shfl_sync(0xffffffff, g, 0);
        beta = __shfl_sync(0xffffffff, beta, 0);

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

        float out_vals[kRowsPerWarp];
        if constexpr (kRowsPerWarp == 1) {
            float old_v = 0.0f;
            float old_o = 0.0f;
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[0][i] *= g;
                old_v = fmaf(k_reg[i], h[0][i], old_v);
                old_o = fmaf(q_reg[i], h[0][i], old_o);
            }
            warp_reduce_2(old_v, old_o);

            float delta_v = 0.0f;
            if (lane_id == 0) {
                delta_v = beta * (v_reg[0] - old_v);
                out_vals[0] = old_o + delta_v * kq;
            }
            delta_v = __shfl_sync(0xffffffff, delta_v, 0);
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[0][i] = fmaf(delta_v, k_reg[i], h[0][i]);
            }
        } else {
            float old_v0 = 0.0f;
            float old_v1 = 0.0f;
            float old_o0 = 0.0f;
            float old_o1 = 0.0f;
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[0][i] *= g;
                h[1][i] *= g;
                old_v0 = fmaf(k_reg[i], h[0][i], old_v0);
                old_v1 = fmaf(k_reg[i], h[1][i], old_v1);
                old_o0 = fmaf(q_reg[i], h[0][i], old_o0);
                old_o1 = fmaf(q_reg[i], h[1][i], old_o1);
            }
            warp_reduce_2(old_v0, old_v1);
            warp_reduce_2(old_o0, old_o1);

            float delta_v0 = 0.0f;
            float delta_v1 = 0.0f;
            if (lane_id == 0) {
                delta_v0 = beta * (v_reg[0] - old_v0);
                delta_v1 = beta * (v_reg[1] - old_v1);
                out_vals[0] = old_o0 + delta_v0 * kq;
                out_vals[1] = old_o1 + delta_v1 * kq;
            }
            delta_v0 = __shfl_sync(0xffffffff, delta_v0, 0);
            delta_v1 = __shfl_sync(0xffffffff, delta_v1, 0);
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[0][i] = fmaf(delta_v0, k_reg[i], h[0][i]);
                h[1][i] = fmaf(delta_v1, k_reg[i], h[1][i]);
            }
        }

        if (lane_id == 0) {
            __nv_bfloat16* out_ptr = out + (token_idx * kNumVHeads + v_head) * kHeadDim + v_base;
#pragma unroll
            for (int row = 0; row < kRowsPerWarp; ++row) {
                out_ptr[row] = __float2bfloat16_rn(out_vals[row]);
            }
        }
    }

    store_state_tile<kRowsPerWarp>(new_state, state_base, v_base, k_base, h);
}

template <int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 1) void gdn_prefill_v8(
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
    static_assert(kHeadDim % WARPS == 0, "kHeadDim must be divisible by WARPS");
    static_assert((kHeadDim / WARPS) % 2 == 0, "kRowsPerWarp must be even");
    constexpr int kRowsPerWarp = kHeadDim / WARPS;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_idx = flat / kNumVHeads;
    if (seq_idx >= num_seqs) return;

    const int v_head = flat - seq_idx * kNumVHeads;
    const int qk_head = v_head / kHeadGroupRatio;
    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int k_base = lane_id * kKVec;
    const int v_base = warp_id * kRowsPerWarp;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    float h[kRowsPerWarp][kKVec];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
        const float4 h_vec = __ldg(reinterpret_cast<const float4*>(
            state + state_base + (v_base + row) * kHeadDim + k_base
        ));
        h[row][0] = h_vec.x;
        h[row][1] = h_vec.y;
        h[row][2] = h_vec.z;
        h[row][3] = h_vec.w;
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    if (seq_end <= seq_start) {
        store_state_tile<kRowsPerWarp>(new_state, state_base, v_base, k_base, h);
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];

    __shared__ float s_q_chunk[CHUNK][kHeadDim];
    __shared__ float s_k_chunk[CHUNK][kHeadDim];
    __shared__ float s_g[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_gamma[CHUNK];
    __shared__ float s_kk[CHUNK][CHUNK];
    __shared__ float s_qk[CHUNK][CHUNK];
    __shared__ float s_lower[CHUNK][CHUNK];
    __shared__ float s_qcoeff[CHUNK][CHUNK];
    __shared__ float s_self_q[CHUNK];
    __shared__ float s_state_coeff[CHUNK];

    float my_kproj[CHUNK];
    float my_qproj[CHUNK];
    float my_delta[CHUNK];
    float my_v[CHUNK];

    int token_idx = seq_start;
    for (; token_idx + CHUNK <= seq_end; token_idx += CHUNK) {
        // Load q/k (warp 0 → smem), v (per-row lane → register), gates (thread 0 → smem)
#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            const int curr_token = token_idx + m;
            if (warp_id == 0) {
                const int qk_offset = (curr_token * kNumQHeads + qk_head) * kHeadDim + k_base;
                const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
                const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
                const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
                const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
                const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
                const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
                s_q_chunk[m][k_base + 0] = __bfloat162float(q01.x) * scale;
                s_q_chunk[m][k_base + 1] = __bfloat162float(q01.y) * scale;
                s_q_chunk[m][k_base + 2] = __bfloat162float(q23.x) * scale;
                s_q_chunk[m][k_base + 3] = __bfloat162float(q23.y) * scale;
                s_k_chunk[m][k_base + 0] = __bfloat162float(k01.x);
                s_k_chunk[m][k_base + 1] = __bfloat162float(k01.y);
                s_k_chunk[m][k_base + 2] = __bfloat162float(k23.x);
                s_k_chunk[m][k_base + 3] = __bfloat162float(k23.y);
            }
            if (lane_id < kRowsPerWarp) {
                my_v[m] = __bfloat162float(
                    v[(curr_token * kNumVHeads + v_head) * kHeadDim + v_base + lane_id]
                );
            }
            if (threadIdx.x == 0) {
                const int gate_offset = curr_token * kNumVHeads + v_head;
                s_g[m] = __expf(-decay * softplus_fast(
                    __bfloat162float(a[gate_offset]) + dt_bias_val
                ));
                s_beta[m] = sigmoid_fast(__bfloat162float(b[gate_offset]));
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            float gamma = 1.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                gamma *= s_g[m];
                s_gamma[m] = gamma;
            }
        }

        __syncthreads();

        // kproj/qproj: each warp computes for its own rows
#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            float q_vec[kKVec];
            float k_vec[kKVec];
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                q_vec[i] = s_q_chunk[m][k_base + i];
                k_vec[i] = s_k_chunk[m][k_base + i];
            }
#pragma unroll
            for (int row = 0; row < kRowsPerWarp; ++row) {
                float k_proj = 0.0f;
                float q_proj = 0.0f;
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    k_proj = fmaf(k_vec[i], h[row][i], k_proj);
                    q_proj = fmaf(q_vec[i], h[row][i], q_proj);
                }
                warp_reduce_2(k_proj, q_proj);
                if (lane_id == row) {
                    my_kproj[m] = k_proj;
                    my_qproj[m] = q_proj;
                }
            }
        }

        // qk/kk: warp 0 computes once for the whole block
        if (warp_id == 0) {
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                float q_m[kKVec];
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    q_m[i] = s_q_chunk[m][k_base + i];
                }
#pragma unroll
                for (int j = 0; j <= m; ++j) {
                    float qk_dot = 0.0f;
                    float kk_dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < kKVec; ++i) {
                        qk_dot = fmaf(q_m[i], s_k_chunk[j][k_base + i], qk_dot);
                        if (j < m) {
                            kk_dot = fmaf(s_k_chunk[m][k_base + i], s_k_chunk[j][k_base + i], kk_dot);
                        }
                    }
                    warp_reduce_2(qk_dot, kk_dot);
                    if (lane_id == 0) {
                        s_qk[m][j] = qk_dot;
                        if (j < m) s_kk[m][j] = kk_dot;
                    }
                }
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            const float gamma_last = s_gamma[CHUNK - 1];
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                s_self_q[m] = s_qk[m][m] * s_beta[m];
                s_state_coeff[m] = (gamma_last / s_gamma[m]) * s_beta[m];
#pragma unroll
                for (int j = 0; j < m; ++j) {
                    const float ratio = s_gamma[m] / s_gamma[j];
                    s_lower[m][j] = ratio * s_kk[m][j] * s_beta[j];
                    s_qcoeff[m][j] = ratio * s_qk[m][j] * s_beta[j];
                }
            }
        }

        __syncthreads();

        // Delta solve + output: lanes 0..kRowsPerWarp-1 work in parallel per warp
        if (lane_id < kRowsPerWarp) {
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                float delta = my_v[m] - s_gamma[m] * my_kproj[m];
                for (int j = 0; j < m; ++j) {
                    delta -= s_lower[m][j] * my_delta[j];
                }
                my_delta[m] = delta;

                float out_val = s_gamma[m] * my_qproj[m];
                for (int j = 0; j < m; ++j) {
                    out_val += s_qcoeff[m][j] * my_delta[j];
                }
                out_val += s_self_q[m] * delta;
                out[((token_idx + m) * kNumVHeads + v_head) * kHeadDim + v_base + lane_id]
                    = __float2bfloat16_rn(out_val);
            }
        }

        // State update: decay h then accumulate all rows' deltas
        const float gamma_last = s_gamma[CHUNK - 1];
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[row][i] *= gamma_last;
            }
        }

#pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            const float state_coeff = s_state_coeff[m];
            float k_reg[kKVec];
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                k_reg[i] = s_k_chunk[m][k_base + i];
            }
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const float delta_scaled =
                    __shfl_sync(0xffffffff, my_delta[m], r) * state_coeff;
#pragma unroll
                for (int i = 0; i < kKVec; ++i) {
                    h[r][i] = fmaf(delta_scaled, k_reg[i], h[r][i]);
                }
            }
        }

        __syncthreads();
    }

    // Tail: per-token fallback for remaining tokens
    for (; token_idx < seq_end; ++token_idx) {
        if (warp_id == 0) {
            const int qk_offset = (token_idx * kNumQHeads + qk_head) * kHeadDim + k_base;
            const uint2 q_raw = __ldg(reinterpret_cast<const uint2*>(q + qk_offset));
            const uint2 k_raw = __ldg(reinterpret_cast<const uint2*>(k + qk_offset));
            const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.x);
            const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&q_raw.y);
            const __nv_bfloat162 k01 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
            const __nv_bfloat162 k23 = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
            s_q_chunk[0][k_base + 0] = __bfloat162float(q01.x) * scale;
            s_q_chunk[0][k_base + 1] = __bfloat162float(q01.y) * scale;
            s_q_chunk[0][k_base + 2] = __bfloat162float(q23.x) * scale;
            s_q_chunk[0][k_base + 3] = __bfloat162float(q23.y) * scale;
            s_k_chunk[0][k_base + 0] = __bfloat162float(k01.x);
            s_k_chunk[0][k_base + 1] = __bfloat162float(k01.y);
            s_k_chunk[0][k_base + 2] = __bfloat162float(k23.x);
            s_k_chunk[0][k_base + 3] = __bfloat162float(k23.y);
        }
        float v_val = 0.0f;
        if (lane_id < kRowsPerWarp) {
            v_val = __bfloat162float(
                v[(token_idx * kNumVHeads + v_head) * kHeadDim + v_base + lane_id]
            );
        }
        if (threadIdx.x == 0) {
            const int gate_offset = token_idx * kNumVHeads + v_head;
            s_g[0] = __expf(-decay * softplus_fast(
                __bfloat162float(a[gate_offset]) + dt_bias_val
            ));
            s_beta[0] = sigmoid_fast(__bfloat162float(b[gate_offset]));
        }

        __syncthreads();

        const float g = s_g[0];
        const float beta = s_beta[0];

        float q_reg[kKVec];
        float k_reg[kKVec];
#pragma unroll
        for (int i = 0; i < kKVec; ++i) {
            q_reg[i] = s_q_chunk[0][k_base + i];
            k_reg[i] = s_k_chunk[0][k_base + i];
        }

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

#pragma unroll
        for (int row = 0; row < kRowsPerWarp; row += 2) {
            float old_v0 = 0.0f, old_v1 = 0.0f;
            float old_o0 = 0.0f, old_o1 = 0.0f;
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

            float delta_v0 = 0.0f, delta_v1 = 0.0f;
            if (lane_id == row) {
                delta_v0 = beta * (v_val - old_v0);
                out[(token_idx * kNumVHeads + v_head) * kHeadDim + v_base + row]
                    = __float2bfloat16_rn(old_o0 + delta_v0 * kq);
            }
            if (lane_id == row + 1) {
                delta_v1 = beta * (v_val - old_v1);
                out[(token_idx * kNumVHeads + v_head) * kHeadDim + v_base + row + 1]
                    = __float2bfloat16_rn(old_o1 + delta_v1 * kq);
            }
            delta_v0 = __shfl_sync(0xffffffff, delta_v0, row);
            delta_v1 = __shfl_sync(0xffffffff, delta_v1, row + 1);
#pragma unroll
            for (int i = 0; i < kKVec; ++i) {
                h[row][i] = fmaf(delta_v0, k_reg[i], h[row][i]);
                h[row + 1][i] = fmaf(delta_v1, k_reg[i], h[row + 1][i]);
            }
        }

        __syncthreads();
    }

    store_state_tile<kRowsPerWarp>(new_state, state_base, v_base, k_base, h);
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
    const int total_tokens = static_cast<int>(q.size(0));
    const int avg_seq_len = (total_tokens + num_seqs - 1) / num_seqs;
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    const auto* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    const auto* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
    const auto* A_log_ptr = A_log.data_ptr<float>();
    const auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
    const auto* dt_bias_ptr = dt_bias.data_ptr<float>();
    const auto* b_ptr = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
    const auto* cu_seqlens_ptr = cu_seqlens.data_ptr<int64_t>();
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto* new_state_ptr = new_state.data_ptr<float>();

    auto launch_v2 = [&]() {
        if (num_seqs == 1 && avg_seq_len >= 512) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kStarvedBV);
            gdn_prefill_v2<kStarvedBV><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (num_seqs <= 2 && avg_seq_len >= 256) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kLongBV);
            gdn_prefill_v2<kLongBV><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else {
            const int total_tiles = num_seqs * kNumVHeads * kNumVTiles;
            gdn_prefill_v2<kBV><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        }
    };

    auto launch_v4 = [&]() {
        if (num_seqs == 1 && avg_seq_len >= 1024) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kStarvedBV);
            gdn_prefill_v4<kStarvedBV, 8><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (num_seqs <= 4 && avg_seq_len >= 512) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kLongBV);
            gdn_prefill_v4<kLongBV, 8><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (num_seqs == 1 && avg_seq_len >= 512) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kStarvedBV);
            gdn_prefill_v4<kStarvedBV, 4><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (num_seqs <= 4 && avg_seq_len >= 192) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kLongBV);
            gdn_prefill_v4<kLongBV, 4><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (avg_seq_len >= 128) {
            const int total_tiles = num_seqs * kNumVHeads * kNumVTiles;
            gdn_prefill_v4<kBV, 4><<<total_tiles, kWarpSize, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else {
            launch_v2();
        }
    };

    auto launch_v6 = [&]() {
        if (num_seqs > 4 && avg_seq_len >= 128) {
            const int total_groups = num_seqs * kNumVHeads * (kHeadDim / (kBV * 2));
            gdn_prefill_v6<kBV, 4, 2><<<total_groups, kWarpSize * 2, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else {
            launch_v2();
        }
    };

    auto launch_v7 = [&]() {
        if (num_seqs == 1 && avg_seq_len >= 512) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kStarvedBV);
            gdn_prefill_v7<kStarvedBV, 2><<<total_tiles, kWarpSize * 2, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (num_seqs <= 2 && avg_seq_len >= 256) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kLongBV);
            gdn_prefill_v7<kLongBV, 4><<<total_tiles, kWarpSize * 4, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else if (avg_seq_len >= 128) {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kLongBV);
            gdn_prefill_v7<kLongBV, 4><<<total_tiles, kWarpSize * 4, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        } else {
            launch_v2();
        }
    };

    auto launch_v8 = [&]() {
        constexpr int kV8Warps = 16;
        constexpr int kV8Chunk = 8;
        const int total_heads = num_seqs * kNumVHeads;
        gdn_prefill_v8<kV8Chunk, kV8Warps><<<total_heads, kWarpSize * kV8Warps, 0, stream.stream()>>>(
            q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
        );
    };

    switch (kBenchmarkKernelVersion) {
        case 2:
            launch_v2();
            break;
        case 4:
            launch_v4();
            break;
        case 6:
            launch_v6();
            break;
        case 7:
            launch_v7();
            break;
        case 8:
            launch_v8();
            break;
        default:
            launch_v2();
            break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gdn", &launch_gdn, "Launch GDN prefill CUDA kernel");
}
