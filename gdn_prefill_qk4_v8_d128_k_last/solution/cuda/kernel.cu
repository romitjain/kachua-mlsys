#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kNumQHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHeadGroupRatio = kNumVHeads / kNumQHeads;
constexpr int kKVec = kHeadDim / kWarpSize;
constexpr int kLongBV = 4;
constexpr int kStarvedBV = 2;
constexpr int kBenchmarkKernelVersion = 18;

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


// ============================================================================
// v12 — Phase B: TMA bulk loads for Q/K/V (SM_90+). Same per-chunk algorithm
// as v11, but the Q/K/V loads are issued as a single bulk tensor copy per
// stage per tensor, completed via mbarrier. The compute threads are free
// during the load, and the HW copy engine has higher bandwidth than cp.async.
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define KACHUA_HAS_TMA 1
#else
#define KACHUA_HAS_TMA 0
#endif

__device__ __forceinline__ void mbarrier_init_one(uint64_t* addr, uint32_t count) {
#if KACHUA_HAS_TMA
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(sa), "r"(count));
#endif
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* addr, uint32_t tx_count) {
#if KACHUA_HAS_TMA
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(sa), "r"(tx_count)
    );
#endif
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* addr, uint32_t phase) {
#if KACHUA_HAS_TMA
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "waitLoop_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra done_%=;\n"
        "bra waitLoop_%=;\n"
        "done_%=:\n"
        "}\n"
        :: "r"(sa), "r"(phase)
    );
#endif
}

__device__ __forceinline__ void fence_async_shared() {
#if KACHUA_HAS_TMA
    asm volatile("fence.proxy.async.shared::cta;\n" ::);
#endif
}

__device__ __forceinline__ void cp_tma_3d(
    void* smem_dst,
    const void* tma_desc,
    int c0, int c1, int c2,
    uint64_t* mbar
) {
#if KACHUA_HAS_TMA
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(sa), "l"(tma_desc), "r"(c0), "r"(c1), "r"(c2), "r"(ma)
    );
#endif
}

template <int BV, int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 2) void gdn_prefill_v12(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
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
    static_assert(CHUNK == 16 && WARPS == 4, "v12: CHUNK=16, WARPS=4");
    static_assert(BV == 16, "v12 Phase B: BV=16 only for first cut");
    namespace wmma = nvcuda::wmma;
    using BfTile = __nv_bfloat16;

    constexpr int BV_PAD = BV;
    constexpr int kNumNTiles = BV_PAD / 16;
    constexpr int kNumTiles = kHeadDim / BV;
    constexpr int kBlockThreads = kWarpSize * WARPS;
    constexpr int kNumKTiles = kHeadDim / 16;
    constexpr int kKTilesPerWarp = kNumKTiles / WARPS;
    constexpr int kWarpsPerNTile = WARPS / kNumNTiles;
    constexpr int kKTilesPerWarpEff = kNumKTiles / kWarpsPerNTile;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) return;

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    alignas(128) __shared__ BfTile s_Q[2][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_K[2][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_V[2][CHUNK][BV];
    alignas(16)  __shared__ BfTile s_state_bf[BV_PAD][kHeadDim];
    __shared__ BfTile s_nil[CHUNK][CHUNK];
    __shared__ BfTile s_x_bf[CHUNK][BV_PAD];
    __shared__ BfTile s_qk_tril[CHUNK][CHUNK];
    __shared__ float s_gram_kk[CHUNK][CHUNK];
    __shared__ float s_gram_qk[CHUNK][CHUNK];
    __shared__ float s_rhs[CHUNK][BV_PAD];
    __shared__ float s_state_q[CHUNK][BV_PAD];
    __shared__ float s_state_k[CHUNK][BV_PAD];
    __shared__ float s_qk_contrib[CHUNK][BV_PAD];
    __shared__ float s_log2_g[CHUNK];
    __shared__ float s_G[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_Gsafe[CHUNK];
    __shared__ float s_partial[WARPS][16][16];
    alignas(8) __shared__ uint64_t mbar_load[2];

    extern __shared__ char dyn_smem_raw[];
    float (&s_state_accum)[BV][kHeadDim] =
        *reinterpret_cast<float(*)[BV][kHeadDim]>(dyn_smem_raw);

    if (tid == 0) {
        mbarrier_init_one(&mbar_load[0], 1);
        mbarrier_init_one(&mbar_load[1], 1);
        fence_async_shared();
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        s_state_accum[r][c] = state[state_base + (v_base + r) * kHeadDim + c];
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    __syncthreads();

    if (seq_end <= seq_start) {
        for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            new_state[state_base + (v_base + r) * kHeadDim + c] = s_state_accum[r][c];
        }
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;

    constexpr uint32_t kQBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kKBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kVBytes = CHUNK * BV * sizeof(BfTile);
    constexpr uint32_t kTotalTMABytes = kQBytes + kKBytes + kVBytes;

    // Issue all 3 TMA loads for `chunk_start` into buffer `stage`. Elected by tid==0.
    auto issue_tma = [&](int stage, int chunk_start) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar_load[stage], kTotalTMABytes);
            cp_tma_3d(&s_Q[stage][0][0], &tma_q, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_K[stage][0][0], &tma_k, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_V[stage][0][0], &tma_v, v_base, v_head,  chunk_start, &mbar_load[stage]);
        }
    };

    uint32_t phase[2] = {0, 0};
    auto wait_stage = [&](int stage) {
        mbarrier_wait_parity(&mbar_load[stage], phase[stage]);
        phase[stage] ^= 1;
    };

    issue_tma(0, seq_start);
    wait_stage(0);
    __syncthreads();

    int stage = 0;
    int chunk_start = seq_start;
    while (chunk_start < seq_end) {
        const int next_chunk = chunk_start + CHUNK;
        const bool has_next = (next_chunk < seq_end);
        const int next_stage = 1 - stage;

        if (has_next) {
            issue_tma(next_stage, next_chunk);
        }

        for (int i = tid; i < BV_PAD * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            const float src = (r < BV) ? s_state_accum[r][c] : 0.0f;
            s_state_bf[r][c] = __float2bfloat16_rn(src);
        }

        if (tid < CHUNK) {
            const int m = tid;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const int gate_off = tok * kNumVHeads + v_head;
                const float a_val = __bfloat162float(a[gate_off]);
                const float b_val = __bfloat162float(b[gate_off]);
                const float x_g = a_val + dt_bias_val;
                const float sp = (x_g > 20.0f) ? x_g : (log2f(1.0f + exp2f(x_g * kLog2e)) * kLn2);
                s_log2_g[m] = -decay * sp * kLog2e;
                s_beta[m] = 1.0f / (1.0f + __expf(-b_val));
            } else {
                s_log2_g[m] = 0.0f;
                s_beta[m] = 0.0f;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float cum = 0.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                cum += s_log2_g[m];
                const float G_m = exp2f(cum);
                s_G[m] = G_m;
                s_Gsafe[m] = fmaxf(G_m, 1e-30f);
            }
        }
        __syncthreads();

        const float Gc = s_G[CHUNK - 1];

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_kk[m][j] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_qk[m][j] = s;
        }

        const int n_tile_id = warp_id / kWarpsPerNTile;
        const int warp_in_n = warp_id % kWarpsPerNTile;

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_k[m][n] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_q[m][n] = s;
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
            const int m = i / CHUNK, j = i % CHUNK;
            const float nv = (j < m) ? (s_beta[m] * s_gram_kk[m][j]) : 0.0f;
            s_nil[m][j] = __float2bfloat16_rn(nv);
            const float tv = (j <= m) ? s_gram_qk[m][j] : 0.0f;
            s_qk_tril[m][j] = __float2bfloat16_rn(tv);
        }
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i % BV_PAD;
            float sol = 0.0f;
            if (r < BV) {
                const float vv = __bfloat162float(s_V[stage][m][r]);
                sol = s_beta[m] * (vv / s_Gsafe[m] - s_state_k[m][r]);
            }
            s_rhs[m][r] = sol;
            s_x_bf[m][r] = __float2bfloat16_rn(sol);
        }
        __syncthreads();

        if (warp_id == 0 && lane < BV) {
            const int r = lane;
            float x_reg[CHUNK];
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                float sum = s_rhs[m][r];
#pragma unroll
                for (int j = 0; j < CHUNK; ++j) {
                    if (j < m) sum -= __bfloat162float(s_nil[m][j]) * x_reg[j];
                }
                x_reg[m] = sum;
                s_x_bf[m][r] = __float2bfloat16_rn(sum);
            }
        }
        __syncthreads();

        if (warp_id < kNumNTiles) {
            const int n_tile = warp_id;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            wmma::load_matrix_sync(a_f, &s_qk_tril[0][0], CHUNK);
            wmma::load_matrix_sync(b_f, &s_x_bf[0][n_tile * 16], BV_PAD);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
            wmma::store_matrix_sync(&s_qk_contrib[0][n_tile * 16], c_f, BV_PAD, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * BV; i += kBlockThreads) {
            const int m = i / BV, r = i % BV;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const float out_val = scale * s_G[m] * (s_state_q[m][r] + s_qk_contrib[m][r]);
                out[(tok * kNumVHeads + v_head) * kHeadDim + v_base + r] = __float2bfloat16_rn(out_val);
            }
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::col_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            const int nt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int m_tile = 0; m_tile < kNumNTiles; ++m_tile) {
                wmma::load_matrix_sync(a_f, &s_x_bf[0][m_tile * 16], BV_PAD);
#pragma unroll
                for (int i = 0; i < kKTilesPerWarp; ++i) {
                    const int nt = (nt_start + i) * 16;
                    wmma::load_matrix_sync(b_f, &s_K[stage][0][nt], kHeadDim);
                    wmma::fill_fragment(c_f, 0.0f);
                    wmma::mma_sync(c_f, a_f, b_f, c_f);
                    wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
                    for (int j = lane; j < 16 * 16; j += kWarpSize) {
                        const int rr = j / 16, cc = j % 16;
                        const int global_r = m_tile * 16 + rr;
                        if (global_r < BV) {
                            s_state_accum[global_r][nt + cc] = Gc *
                                (s_state_accum[global_r][nt + cc] + s_partial[warp_id][rr][cc]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (!has_next) break;
        wait_stage(next_stage);
        __syncthreads();
        stage = next_stage;
        chunk_start = next_chunk;
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        __stcs(&new_state[state_base + (v_base + r) * kHeadDim + c], s_state_accum[r][c]);
    }
}


// ============================================================================
// v13 — v12 with a 3-stage TMA buffer. Two TMA loads may be in flight at once
// (N+1 issued at start of N, N+2 issued at end of N's compute). Tests whether
// deeper load/compute overlap helps; WGS is a larger follow-up if it does.
// ============================================================================

template <int BV, int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 2) void gdn_prefill_v13(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
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
    static_assert(CHUNK == 16 && WARPS == 4, "v13: CHUNK=16, WARPS=4");
    static_assert(BV == 16, "v13: BV=16 only");
    namespace wmma = nvcuda::wmma;
    using BfTile = __nv_bfloat16;

    constexpr int STAGES = 3;
    constexpr int BV_PAD = BV;
    constexpr int kNumNTiles = BV_PAD / 16;
    constexpr int kNumTiles = kHeadDim / BV;
    constexpr int kBlockThreads = kWarpSize * WARPS;
    constexpr int kNumKTiles = kHeadDim / 16;
    constexpr int kKTilesPerWarp = kNumKTiles / WARPS;
    constexpr int kWarpsPerNTile = WARPS / kNumNTiles;
    constexpr int kKTilesPerWarpEff = kNumKTiles / kWarpsPerNTile;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) return;

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    alignas(128) __shared__ BfTile s_Q[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_K[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_V[STAGES][CHUNK][BV];
    alignas(16)  __shared__ BfTile s_state_bf[BV_PAD][kHeadDim];
    __shared__ BfTile s_nil[CHUNK][CHUNK];
    __shared__ BfTile s_x_bf[CHUNK][BV_PAD];
    __shared__ BfTile s_qk_tril[CHUNK][CHUNK];
    __shared__ float s_gram_kk[CHUNK][CHUNK];
    __shared__ float s_gram_qk[CHUNK][CHUNK];
    __shared__ float s_rhs[CHUNK][BV_PAD];
    __shared__ float s_state_q[CHUNK][BV_PAD];
    __shared__ float s_state_k[CHUNK][BV_PAD];
    __shared__ float s_qk_contrib[CHUNK][BV_PAD];
    __shared__ float s_log2_g[CHUNK];
    __shared__ float s_G[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_Gsafe[CHUNK];
    __shared__ float s_partial[WARPS][16][16];
    alignas(8) __shared__ uint64_t mbar_load[STAGES];

    extern __shared__ char dyn_smem_raw[];
    float (&s_state_accum)[BV][kHeadDim] =
        *reinterpret_cast<float(*)[BV][kHeadDim]>(dyn_smem_raw);

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) mbarrier_init_one(&mbar_load[s], 1);
        fence_async_shared();
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        s_state_accum[r][c] = state[state_base + (v_base + r) * kHeadDim + c];
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    __syncthreads();

    if (seq_end <= seq_start) {
        for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            new_state[state_base + (v_base + r) * kHeadDim + c] = s_state_accum[r][c];
        }
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;

    constexpr uint32_t kQBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kKBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kVBytes = CHUNK * BV * sizeof(BfTile);
    constexpr uint32_t kTotalTMABytes = kQBytes + kKBytes + kVBytes;

    auto issue_tma = [&](int stage, int chunk_start) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar_load[stage], kTotalTMABytes);
            cp_tma_3d(&s_Q[stage][0][0], &tma_q, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_K[stage][0][0], &tma_k, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_V[stage][0][0], &tma_v, v_base, v_head,  chunk_start, &mbar_load[stage]);
        }
    };

    uint32_t phase[STAGES] = {0, 0, 0};
    auto wait_stage = [&](int stage) {
        mbarrier_wait_parity(&mbar_load[stage], phase[stage]);
        phase[stage] ^= 1;
    };

    // Keep at most (STAGES - 1) loads in flight: one slot is always the one
    // being consumed. With STAGES=3, two chunks ahead are queued while the
    // current chunk is being read.
    const int total_chunks = (seq_end - seq_start + CHUNK - 1) / CHUNK;
    const int kLookahead = STAGES - 1;
    const int pre_issued = (total_chunks < kLookahead) ? total_chunks : kLookahead;
    for (int s = 0; s < pre_issued; ++s) {
        issue_tma(s, seq_start + s * CHUNK);
    }
    wait_stage(0);
    __syncthreads();

    int stage = 0;
    int chunk_start = seq_start;
    int next_issue = pre_issued;
    while (chunk_start < seq_end) {
        const int next_chunk = chunk_start + CHUNK;
        const bool has_next = (next_chunk < seq_end);
        const int next_stage = (stage + 1) % STAGES;

        // Issue the chunk at `next_issue` into its slot. Safe because slot
        // next_issue % STAGES differs from `stage` (we keep kLookahead < STAGES ahead)
        // and any earlier compute on that slot is complete (the ring wraps only
        // after the consumer has moved past it).
        if (next_issue < total_chunks) {
            const int issue_slot = next_issue % STAGES;
            issue_tma(issue_slot, seq_start + next_issue * CHUNK);
            next_issue++;
        }

        for (int i = tid; i < BV_PAD * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            const float src = (r < BV) ? s_state_accum[r][c] : 0.0f;
            s_state_bf[r][c] = __float2bfloat16_rn(src);
        }

        if (tid < CHUNK) {
            const int m = tid;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const int gate_off = tok * kNumVHeads + v_head;
                const float a_val = __bfloat162float(a[gate_off]);
                const float b_val = __bfloat162float(b[gate_off]);
                const float x_g = a_val + dt_bias_val;
                const float sp = (x_g > 20.0f) ? x_g : (log2f(1.0f + exp2f(x_g * kLog2e)) * kLn2);
                s_log2_g[m] = -decay * sp * kLog2e;
                s_beta[m] = 1.0f / (1.0f + __expf(-b_val));
            } else {
                s_log2_g[m] = 0.0f;
                s_beta[m] = 0.0f;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float cum = 0.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                cum += s_log2_g[m];
                const float G_m = exp2f(cum);
                s_G[m] = G_m;
                s_Gsafe[m] = fmaxf(G_m, 1e-30f);
            }
        }
        __syncthreads();

        const float Gc = s_G[CHUNK - 1];

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_kk[m][j] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_qk[m][j] = s;
        }

        const int n_tile_id = warp_id / kWarpsPerNTile;
        const int warp_in_n = warp_id % kWarpsPerNTile;

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_k[m][n] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_q[m][n] = s;
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
            const int m = i / CHUNK, j = i % CHUNK;
            const float nv = (j < m) ? (s_beta[m] * s_gram_kk[m][j]) : 0.0f;
            s_nil[m][j] = __float2bfloat16_rn(nv);
            const float tv = (j <= m) ? s_gram_qk[m][j] : 0.0f;
            s_qk_tril[m][j] = __float2bfloat16_rn(tv);
        }
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i % BV_PAD;
            float sol = 0.0f;
            if (r < BV) {
                const float vv = __bfloat162float(s_V[stage][m][r]);
                sol = s_beta[m] * (vv / s_Gsafe[m] - s_state_k[m][r]);
            }
            s_rhs[m][r] = sol;
            s_x_bf[m][r] = __float2bfloat16_rn(sol);
        }
        __syncthreads();

        if (warp_id == 0 && lane < BV) {
            const int r = lane;
            float x_reg[CHUNK];
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                float sum = s_rhs[m][r];
#pragma unroll
                for (int j = 0; j < CHUNK; ++j) {
                    if (j < m) sum -= __bfloat162float(s_nil[m][j]) * x_reg[j];
                }
                x_reg[m] = sum;
                s_x_bf[m][r] = __float2bfloat16_rn(sum);
            }
        }
        __syncthreads();

        if (warp_id < kNumNTiles) {
            const int n_tile = warp_id;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            wmma::load_matrix_sync(a_f, &s_qk_tril[0][0], CHUNK);
            wmma::load_matrix_sync(b_f, &s_x_bf[0][n_tile * 16], BV_PAD);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
            wmma::store_matrix_sync(&s_qk_contrib[0][n_tile * 16], c_f, BV_PAD, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * BV; i += kBlockThreads) {
            const int m = i / BV, r = i % BV;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const float out_val = scale * s_G[m] * (s_state_q[m][r] + s_qk_contrib[m][r]);
                out[(tok * kNumVHeads + v_head) * kHeadDim + v_base + r] = __float2bfloat16_rn(out_val);
            }
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::col_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            const int nt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int m_tile = 0; m_tile < kNumNTiles; ++m_tile) {
                wmma::load_matrix_sync(a_f, &s_x_bf[0][m_tile * 16], BV_PAD);
#pragma unroll
                for (int i = 0; i < kKTilesPerWarp; ++i) {
                    const int nt = (nt_start + i) * 16;
                    wmma::load_matrix_sync(b_f, &s_K[stage][0][nt], kHeadDim);
                    wmma::fill_fragment(c_f, 0.0f);
                    wmma::mma_sync(c_f, a_f, b_f, c_f);
                    wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
                    for (int j = lane; j < 16 * 16; j += kWarpSize) {
                        const int rr = j / 16, cc = j % 16;
                        const int global_r = m_tile * 16 + rr;
                        if (global_r < BV) {
                            s_state_accum[global_r][nt + cc] = Gc *
                                (s_state_accum[global_r][nt + cc] + s_partial[warp_id][rr][cc]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (!has_next) break;
        wait_stage(next_stage);
        __syncthreads();
        stage = next_stage;
        chunk_start = next_chunk;
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        __stcs(&new_state[state_base + (v_base + r) * kHeadDim + c], s_state_accum[r][c]);
    }
}



// ============================================================================
// v14 — v13 + Neumann-doubling fp32 solve (replaces serial forward-sub).
//
// v13's forward-sub was serial in the row index and ran on warp 0, lanes 0..BV-1
// only (16 of 128 threads active). v14 reformulates (I + N)^{-1} rhs as four
// doubling steps — each step is a parallel [16,16] × [16,16] mat-mul in smem
// on fp32 data, so all 128 threads stay live through the solve. Identity used:
//   (I + N)^{-1} = (I - N)(I + N^2)(I + N^4)(I + N^8), since N^16 = 0 at CHUNK=16.
// fp32 throughout matches v13's serial precision (no extra rounding vs. the
// __bfloat162float path the serial solve already used).
// ============================================================================

template <int BV, int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 2) void gdn_prefill_v14(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
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
    static_assert(CHUNK == 16 && WARPS == 4, "v13: CHUNK=16, WARPS=4");
    static_assert(BV == 16, "v13: BV=16 only");
    namespace wmma = nvcuda::wmma;
    using BfTile = __nv_bfloat16;

    constexpr int STAGES = 3;
    constexpr int BV_PAD = BV;
    constexpr int kNumNTiles = BV_PAD / 16;
    constexpr int kNumTiles = kHeadDim / BV;
    constexpr int kBlockThreads = kWarpSize * WARPS;
    constexpr int kNumKTiles = kHeadDim / 16;
    constexpr int kKTilesPerWarp = kNumKTiles / WARPS;
    constexpr int kWarpsPerNTile = WARPS / kNumNTiles;
    constexpr int kKTilesPerWarpEff = kNumKTiles / kWarpsPerNTile;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) return;

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    alignas(128) __shared__ BfTile s_Q[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_K[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_V[STAGES][CHUNK][BV];
    alignas(16)  __shared__ BfTile s_state_bf[BV_PAD][kHeadDim];
    __shared__ float s_power[CHUNK][CHUNK];   // v14: running power of N (N, N^2, N^4, N^8) for Neumann solve
    __shared__ float s_tmp_sol[CHUNK][BV];    // v14: scratch for power * sol
    __shared__ float s_tmp_pow[CHUNK][CHUNK]; // v14: scratch for power * power
    __shared__ BfTile s_x_bf[CHUNK][BV_PAD];
    __shared__ BfTile s_qk_tril[CHUNK][CHUNK];
    __shared__ float s_gram_kk[CHUNK][CHUNK];
    __shared__ float s_gram_qk[CHUNK][CHUNK];
    __shared__ float s_rhs[CHUNK][BV_PAD];
    __shared__ float s_state_q[CHUNK][BV_PAD];
    __shared__ float s_state_k[CHUNK][BV_PAD];
    __shared__ float s_qk_contrib[CHUNK][BV_PAD];
    __shared__ float s_log2_g[CHUNK];
    __shared__ float s_G[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_Gsafe[CHUNK];
    __shared__ float s_partial[WARPS][16][16];
    alignas(8) __shared__ uint64_t mbar_load[STAGES];

    extern __shared__ char dyn_smem_raw[];
    float (&s_state_accum)[BV][kHeadDim] =
        *reinterpret_cast<float(*)[BV][kHeadDim]>(dyn_smem_raw);

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) mbarrier_init_one(&mbar_load[s], 1);
        fence_async_shared();
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        s_state_accum[r][c] = state[state_base + (v_base + r) * kHeadDim + c];
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    __syncthreads();

    if (seq_end <= seq_start) {
        for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            new_state[state_base + (v_base + r) * kHeadDim + c] = s_state_accum[r][c];
        }
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;

    constexpr uint32_t kQBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kKBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kVBytes = CHUNK * BV * sizeof(BfTile);
    constexpr uint32_t kTotalTMABytes = kQBytes + kKBytes + kVBytes;

    auto issue_tma = [&](int stage, int chunk_start) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar_load[stage], kTotalTMABytes);
            cp_tma_3d(&s_Q[stage][0][0], &tma_q, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_K[stage][0][0], &tma_k, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_V[stage][0][0], &tma_v, v_base, v_head,  chunk_start, &mbar_load[stage]);
        }
    };

    uint32_t phase[STAGES] = {0, 0, 0};
    auto wait_stage = [&](int stage) {
        mbarrier_wait_parity(&mbar_load[stage], phase[stage]);
        phase[stage] ^= 1;
    };

    // Keep at most (STAGES - 1) loads in flight: one slot is always the one
    // being consumed. With STAGES=3, two chunks ahead are queued while the
    // current chunk is being read.
    const int total_chunks = (seq_end - seq_start + CHUNK - 1) / CHUNK;
    const int kLookahead = STAGES - 1;
    const int pre_issued = (total_chunks < kLookahead) ? total_chunks : kLookahead;
    for (int s = 0; s < pre_issued; ++s) {
        issue_tma(s, seq_start + s * CHUNK);
    }
    wait_stage(0);
    __syncthreads();

    int stage = 0;
    int chunk_start = seq_start;
    int next_issue = pre_issued;
    while (chunk_start < seq_end) {
        const int next_chunk = chunk_start + CHUNK;
        const bool has_next = (next_chunk < seq_end);
        const int next_stage = (stage + 1) % STAGES;

        // Issue the chunk at `next_issue` into its slot. Safe because slot
        // next_issue % STAGES differs from `stage` (we keep kLookahead < STAGES ahead)
        // and any earlier compute on that slot is complete (the ring wraps only
        // after the consumer has moved past it).
        if (next_issue < total_chunks) {
            const int issue_slot = next_issue % STAGES;
            issue_tma(issue_slot, seq_start + next_issue * CHUNK);
            next_issue++;
        }

        for (int i = tid; i < BV_PAD * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            const float src = (r < BV) ? s_state_accum[r][c] : 0.0f;
            s_state_bf[r][c] = __float2bfloat16_rn(src);
        }

        if (tid < CHUNK) {
            const int m = tid;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const int gate_off = tok * kNumVHeads + v_head;
                const float a_val = __bfloat162float(a[gate_off]);
                const float b_val = __bfloat162float(b[gate_off]);
                const float x_g = a_val + dt_bias_val;
                const float sp = (x_g > 20.0f) ? x_g : (log2f(1.0f + exp2f(x_g * kLog2e)) * kLn2);
                s_log2_g[m] = -decay * sp * kLog2e;
                s_beta[m] = 1.0f / (1.0f + __expf(-b_val));
            } else {
                s_log2_g[m] = 0.0f;
                s_beta[m] = 0.0f;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float cum = 0.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                cum += s_log2_g[m];
                const float G_m = exp2f(cum);
                s_G[m] = G_m;
                s_Gsafe[m] = fmaxf(G_m, 1e-30f);
            }
        }
        __syncthreads();

        const float Gc = s_G[CHUNK - 1];

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_kk[m][j] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_qk[m][j] = s;
        }

        const int n_tile_id = warp_id / kWarpsPerNTile;
        const int warp_in_n = warp_id % kWarpsPerNTile;

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_k[m][n] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_q[m][n] = s;
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
            const int m = i / CHUNK, j = i % CHUNK;
            const float nv = (j < m) ? (s_beta[m] * s_gram_kk[m][j]) : 0.0f;
            s_power[m][j] = nv;  // power <- N (fp32)
            const float tv = (j <= m) ? s_gram_qk[m][j] : 0.0f;
            s_qk_tril[m][j] = __float2bfloat16_rn(tv);
        }
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i % BV_PAD;
            float sol = 0.0f;
            if (r < BV) {
                const float vv = __bfloat162float(s_V[stage][m][r]);
                sol = s_beta[m] * (vv / s_Gsafe[m] - s_state_k[m][r]);
            }
            s_rhs[m][r] = sol;
        }
        __syncthreads();

        // v14: Neumann doubling for (I + N)^{-1} * rhs.
        // N is strict-lower, so nilpotent with N^16 = 0 at CHUNK=16 → 4 doublings suffice.
        // Identity: (I+N)^{-1} = (I - N)(I + N^2)(I + N^4)(I + N^8).
        // Each step fuses "sol update" and "power squaring" into two smem passes, so all 128
        // threads stay busy the whole solve (vs. warp-0/16-thread serial in v13).
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const float sgn = (step == 0) ? -1.0f : 1.0f;
            // Pass 1: tmp_sol = power * sol; tmp_pow = power * power (last step skips tmp_pow).
            const bool squares = (step < 3);
            for (int i = tid; i < CHUNK * BV + (squares ? CHUNK * CHUNK : 0); i += kBlockThreads) {
                if (i < CHUNK * BV) {
                    const int m = i / BV, r = i % BV;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < CHUNK; ++j) acc += s_power[m][j] * s_rhs[j][r];
                    s_tmp_sol[m][r] = acc;
                } else {
                    const int k = i - CHUNK * BV;
                    const int m = k / CHUNK, r = k % CHUNK;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < CHUNK; ++j) acc += s_power[m][j] * s_power[j][r];
                    s_tmp_pow[m][r] = acc;
                }
            }
            __syncthreads();
            // Pass 2: sol += sgn * tmp_sol; power <- tmp_pow.
            for (int i = tid; i < CHUNK * BV + (squares ? CHUNK * CHUNK : 0); i += kBlockThreads) {
                if (i < CHUNK * BV) {
                    const int m = i / BV, r = i % BV;
                    s_rhs[m][r] += sgn * s_tmp_sol[m][r];
                } else {
                    const int k = i - CHUNK * BV;
                    const int m = k / CHUNK, r = k % CHUNK;
                    s_power[m][r] = s_tmp_pow[m][r];
                }
            }
            __syncthreads();
        }

        // Cast sol (s_rhs) to bf16 for the qk_contrib wmma.
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i % BV_PAD;
            s_x_bf[m][r] = __float2bfloat16_rn(s_rhs[m][r]);
        }
        __syncthreads();

        if (warp_id < kNumNTiles) {
            const int n_tile = warp_id;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            wmma::load_matrix_sync(a_f, &s_qk_tril[0][0], CHUNK);
            wmma::load_matrix_sync(b_f, &s_x_bf[0][n_tile * 16], BV_PAD);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
            wmma::store_matrix_sync(&s_qk_contrib[0][n_tile * 16], c_f, BV_PAD, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * BV; i += kBlockThreads) {
            const int m = i / BV, r = i % BV;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const float out_val = scale * s_G[m] * (s_state_q[m][r] + s_qk_contrib[m][r]);
                out[(tok * kNumVHeads + v_head) * kHeadDim + v_base + r] = __float2bfloat16_rn(out_val);
            }
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::col_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            const int nt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int m_tile = 0; m_tile < kNumNTiles; ++m_tile) {
                wmma::load_matrix_sync(a_f, &s_x_bf[0][m_tile * 16], BV_PAD);
#pragma unroll
                for (int i = 0; i < kKTilesPerWarp; ++i) {
                    const int nt = (nt_start + i) * 16;
                    wmma::load_matrix_sync(b_f, &s_K[stage][0][nt], kHeadDim);
                    wmma::fill_fragment(c_f, 0.0f);
                    wmma::mma_sync(c_f, a_f, b_f, c_f);
                    wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
                    for (int j = lane; j < 16 * 16; j += kWarpSize) {
                        const int rr = j / 16, cc = j % 16;
                        const int global_r = m_tile * 16 + rr;
                        if (global_r < BV) {
                            s_state_accum[global_r][nt + cc] = Gc *
                                (s_state_accum[global_r][nt + cc] + s_partial[warp_id][rr][cc]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (!has_next) break;
        wait_stage(next_stage);
        __syncthreads();
        stage = next_stage;
        chunk_start = next_chunk;
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        __stcs(&new_state[state_base + (v_base + r) * kHeadDim + c], s_state_accum[r][c]);
    }
}


// ============================================================================
// v17 — v13 body with __launch_bounds__(128, 3). Min 3 blocks per SM to raise
// occupancy on SM-starved workloads (num_seqs=1 at BV=16). Trades register
// budget for more warps in flight.
// ============================================================================

template <int BV, int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 3) void gdn_prefill_v17(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
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
    static_assert(CHUNK == 16 && WARPS == 4, "v13: CHUNK=16, WARPS=4");
    static_assert(BV == 16, "v13: BV=16 only");
    namespace wmma = nvcuda::wmma;
    using BfTile = __nv_bfloat16;

    constexpr int STAGES = 3;
    constexpr int BV_PAD = BV;
    constexpr int kNumNTiles = BV_PAD / 16;
    constexpr int kNumTiles = kHeadDim / BV;
    constexpr int kBlockThreads = kWarpSize * WARPS;
    constexpr int kNumKTiles = kHeadDim / 16;
    constexpr int kKTilesPerWarp = kNumKTiles / WARPS;
    constexpr int kWarpsPerNTile = WARPS / kNumNTiles;
    constexpr int kKTilesPerWarpEff = kNumKTiles / kWarpsPerNTile;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) return;

    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;

    alignas(128) __shared__ BfTile s_Q[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_K[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_V[STAGES][CHUNK][BV];
    alignas(16)  __shared__ BfTile s_state_bf[BV_PAD][kHeadDim];
    __shared__ BfTile s_nil[CHUNK][CHUNK];
    __shared__ BfTile s_x_bf[CHUNK][BV_PAD];
    __shared__ BfTile s_qk_tril[CHUNK][CHUNK];
    __shared__ float s_gram_kk[CHUNK][CHUNK];
    __shared__ float s_gram_qk[CHUNK][CHUNK];
    __shared__ float s_rhs[CHUNK][BV_PAD];
    __shared__ float s_state_q[CHUNK][BV_PAD];
    __shared__ float s_state_k[CHUNK][BV_PAD];
    __shared__ float s_qk_contrib[CHUNK][BV_PAD];
    __shared__ float s_log2_g[CHUNK];
    __shared__ float s_G[CHUNK];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_Gsafe[CHUNK];
    __shared__ float s_partial[WARPS][16][16];
    alignas(8) __shared__ uint64_t mbar_load[STAGES];

    extern __shared__ char dyn_smem_raw[];
    float (&s_state_accum)[BV][kHeadDim] =
        *reinterpret_cast<float(*)[BV][kHeadDim]>(dyn_smem_raw);

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) mbarrier_init_one(&mbar_load[s], 1);
        fence_async_shared();
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        s_state_accum[r][c] = state[state_base + (v_base + r) * kHeadDim + c];
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    __syncthreads();

    if (seq_end <= seq_start) {
        for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            new_state[state_base + (v_base + r) * kHeadDim + c] = s_state_accum[r][c];
        }
        return;
    }

    const float A_log_val = A_log[v_head];
    const float decay = __expf(A_log_val);
    const float dt_bias_val = dt_bias[v_head];
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;

    constexpr uint32_t kQBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kKBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kVBytes = CHUNK * BV * sizeof(BfTile);
    constexpr uint32_t kTotalTMABytes = kQBytes + kKBytes + kVBytes;

    auto issue_tma = [&](int stage, int chunk_start) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar_load[stage], kTotalTMABytes);
            cp_tma_3d(&s_Q[stage][0][0], &tma_q, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_K[stage][0][0], &tma_k, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_V[stage][0][0], &tma_v, v_base, v_head,  chunk_start, &mbar_load[stage]);
        }
    };

    uint32_t phase[STAGES] = {0, 0, 0};
    auto wait_stage = [&](int stage) {
        mbarrier_wait_parity(&mbar_load[stage], phase[stage]);
        phase[stage] ^= 1;
    };

    // Keep at most (STAGES - 1) loads in flight: one slot is always the one
    // being consumed. With STAGES=3, two chunks ahead are queued while the
    // current chunk is being read.
    const int total_chunks = (seq_end - seq_start + CHUNK - 1) / CHUNK;
    const int kLookahead = STAGES - 1;
    const int pre_issued = (total_chunks < kLookahead) ? total_chunks : kLookahead;
    for (int s = 0; s < pre_issued; ++s) {
        issue_tma(s, seq_start + s * CHUNK);
    }
    wait_stage(0);
    __syncthreads();

    int stage = 0;
    int chunk_start = seq_start;
    int next_issue = pre_issued;
    while (chunk_start < seq_end) {
        const int next_chunk = chunk_start + CHUNK;
        const bool has_next = (next_chunk < seq_end);
        const int next_stage = (stage + 1) % STAGES;

        // Issue the chunk at `next_issue` into its slot. Safe because slot
        // next_issue % STAGES differs from `stage` (we keep kLookahead < STAGES ahead)
        // and any earlier compute on that slot is complete (the ring wraps only
        // after the consumer has moved past it).
        if (next_issue < total_chunks) {
            const int issue_slot = next_issue % STAGES;
            issue_tma(issue_slot, seq_start + next_issue * CHUNK);
            next_issue++;
        }

        for (int i = tid; i < BV_PAD * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i % kHeadDim;
            const float src = (r < BV) ? s_state_accum[r][c] : 0.0f;
            s_state_bf[r][c] = __float2bfloat16_rn(src);
        }

        if (tid < CHUNK) {
            const int m = tid;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const int gate_off = tok * kNumVHeads + v_head;
                const float a_val = __bfloat162float(a[gate_off]);
                const float b_val = __bfloat162float(b[gate_off]);
                const float x_g = a_val + dt_bias_val;
                const float sp = (x_g > 20.0f) ? x_g : (log2f(1.0f + exp2f(x_g * kLog2e)) * kLn2);
                s_log2_g[m] = -decay * sp * kLog2e;
                s_beta[m] = 1.0f / (1.0f + __expf(-b_val));
            } else {
                s_log2_g[m] = 0.0f;
                s_beta[m] = 0.0f;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float cum = 0.0f;
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                cum += s_log2_g[m];
                const float G_m = exp2f(cum);
                s_G[m] = G_m;
                s_Gsafe[m] = fmaxf(G_m, 1e-30f);
            }
        }
        __syncthreads();

        const float Gc = s_G[CHUNK - 1];

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_kk[m][j] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_K[stage][0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 16; i += kBlockThreads) {
            const int m = i / 16, j = i % 16;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
            s_gram_qk[m][j] = s;
        }

        const int n_tile_id = warp_id / kWarpsPerNTile;
        const int warp_in_n = warp_id % kWarpsPerNTile;

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_K[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_k[m][n] = s;
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_in_n * kKTilesPerWarpEff;
#pragma unroll
            for (int i = 0; i < kKTilesPerWarpEff; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[n_tile_id * 16][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            __syncthreads();
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i % BV_PAD;
            const int n_tile = n / 16;
            const int col = n - n_tile * 16;
            float s = 0.0f;
#pragma unroll
            for (int w_in = 0; w_in < kWarpsPerNTile; ++w_in) {
                s += s_partial[n_tile * kWarpsPerNTile + w_in][m][col];
            }
            s_state_q[m][n] = s;
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
            const int m = i / CHUNK, j = i % CHUNK;
            const float nv = (j < m) ? (s_beta[m] * s_gram_kk[m][j]) : 0.0f;
            s_nil[m][j] = __float2bfloat16_rn(nv);
            const float tv = (j <= m) ? s_gram_qk[m][j] : 0.0f;
            s_qk_tril[m][j] = __float2bfloat16_rn(tv);
        }
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i % BV_PAD;
            float sol = 0.0f;
            if (r < BV) {
                const float vv = __bfloat162float(s_V[stage][m][r]);
                sol = s_beta[m] * (vv / s_Gsafe[m] - s_state_k[m][r]);
            }
            s_rhs[m][r] = sol;
            s_x_bf[m][r] = __float2bfloat16_rn(sol);
        }
        __syncthreads();

        if (warp_id == 0 && lane < BV) {
            const int r = lane;
            float x_reg[CHUNK];
#pragma unroll
            for (int m = 0; m < CHUNK; ++m) {
                float sum = s_rhs[m][r];
#pragma unroll
                for (int j = 0; j < CHUNK; ++j) {
                    if (j < m) sum -= __bfloat162float(s_nil[m][j]) * x_reg[j];
                }
                x_reg[m] = sum;
                s_x_bf[m][r] = __float2bfloat16_rn(sum);
            }
        }
        __syncthreads();

        if (warp_id < kNumNTiles) {
            const int n_tile = warp_id;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            wmma::load_matrix_sync(a_f, &s_qk_tril[0][0], CHUNK);
            wmma::load_matrix_sync(b_f, &s_x_bf[0][n_tile * 16], BV_PAD);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
            wmma::store_matrix_sync(&s_qk_contrib[0][n_tile * 16], c_f, BV_PAD, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < CHUNK * BV; i += kBlockThreads) {
            const int m = i / BV, r = i % BV;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const float out_val = scale * s_G[m] * (s_state_q[m][r] + s_qk_contrib[m][r]);
                out[(tok * kNumVHeads + v_head) * kHeadDim + v_base + r] = __float2bfloat16_rn(out_val);
            }
        }

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::col_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            const int nt_start = warp_id * kKTilesPerWarp;
#pragma unroll
            for (int m_tile = 0; m_tile < kNumNTiles; ++m_tile) {
                wmma::load_matrix_sync(a_f, &s_x_bf[0][m_tile * 16], BV_PAD);
#pragma unroll
                for (int i = 0; i < kKTilesPerWarp; ++i) {
                    const int nt = (nt_start + i) * 16;
                    wmma::load_matrix_sync(b_f, &s_K[stage][0][nt], kHeadDim);
                    wmma::fill_fragment(c_f, 0.0f);
                    wmma::mma_sync(c_f, a_f, b_f, c_f);
                    wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
                    for (int j = lane; j < 16 * 16; j += kWarpSize) {
                        const int rr = j / 16, cc = j % 16;
                        const int global_r = m_tile * 16 + rr;
                        if (global_r < BV) {
                            s_state_accum[global_r][nt + cc] = Gc *
                                (s_state_accum[global_r][nt + cc] + s_partial[warp_id][rr][cc]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (!has_next) break;
        wait_stage(next_stage);
        __syncthreads();
        stage = next_stage;
        chunk_start = next_chunk;
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i % kHeadDim;
        __stcs(&new_state[state_base + (v_base + r) * kHeadDim + c], s_state_accum[r][c]);
    }
}
// ============================================================================
// v18 — two-kernel (prepass + consumer) strategy, mirroring Triton's split_wy.
//
// Motivation: v13/v15 recompute gram_kk, gram_qk, gate cumsum, and the solve
// data 8× per (seq, head) because the grid fans out over N_V_TILES = 128/16 = 8.
// In v18 the per-chunk/per-head shared data is precomputed once in a prepass
// and the consumer (grid still = N_V_TILES wide) just loads the intermediates
// from global. Triton's 60 µs vs our 344 µs largely comes from this split.
//
// Per-(seq, v_head, chunk) prepass outputs, laid out in global:
//   wy_a_inv : fp32 [N_S, 8, CMAX, 16, 16]      — (I + N)^{-1}
//   wy_w     : bf16 [N_S, 8, CMAX, 16, 128]     — a_inv @ (beta * K)
//   wy_qk    : bf16 [N_S, 8, CMAX, 16, 16]      — tril(Q @ K^T)
//   wy_g     : fp32 [N_S, 8, CMAX, 16]          — G (cumulative gate)
//   wy_gc    : fp32 [N_S, 8, CMAX]              — Gc (chunk-final gate)
//   wy_bg    : fp32 [N_S, 8, CMAX, 16]          — beta / max(G, eps)
//
// Consumer math per chunk (replacing v13's forward-sub block):
//   u_tile   = a_inv @ (bg[:,None] * V_tile)   // [16, BV] fp32
//   w_state  = w @ state^T                     // [16, BV] fp32
//   x_chunk  = u_tile - w_state
//   (rest — state_q, qk_contrib, output, state update — unchanged from v13)
// Identity check: x = a_inv @ beta * (V/G - K @ state^T) = u_tile - w @ state^T. ✓
// ============================================================================

constexpr int kV18Chunk = 16;

__global__ __launch_bounds__(128, 2) void gdn_prefill_v18_prepass(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    const int64_t* __restrict__ cu_seqlens,
    float* __restrict__ wy_a_inv,   // [N_S, 8, CMAX, 16, 16]
    __nv_bfloat16* __restrict__ wy_w,        // [N_S, 8, CMAX, 16, 128]
    __nv_bfloat16* __restrict__ wy_qk,       // [N_S, 8, CMAX, 16, 16]
    float* __restrict__ wy_g,       // [N_S, 8, CMAX, 16]
    float* __restrict__ wy_gc,      // [N_S, 8, CMAX]
    float* __restrict__ wy_bg,      // [N_S, 8, CMAX, 16]
    int num_seqs,
    int max_chunks
) {
    constexpr int CHUNK = 16;
    constexpr int WARPS = 4;
    constexpr int kBlockThreads = kWarpSize * WARPS;
    constexpr int kNumKTiles = kHeadDim / 16;
    constexpr int kKTilesPerWarp = kNumKTiles / WARPS;  // 2
    namespace wmma = nvcuda::wmma;
    using BfTile = __nv_bfloat16;

    const int sh = static_cast<int>(blockIdx.x);
    const int chunk_id = static_cast<int>(blockIdx.y);
    const int seq_idx = sh / kNumVHeads;
    const int v_head = sh - seq_idx * kNumVHeads;
    if (seq_idx >= num_seqs) return;
    const int qk_head = v_head / kHeadGroupRatio;

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int chunk_start = seq_start + chunk_id * CHUNK;
    if (chunk_start >= seq_end) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    alignas(128) __shared__ BfTile s_Q[CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_K[CHUNK][kHeadDim];
    alignas(16)  __shared__ BfTile s_a_inv_bf[CHUNK][CHUNK];
    alignas(128) __shared__ BfTile s_beta_k_bf[CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_w[CHUNK][kHeadDim];
    alignas(16) __shared__ float s_gram_kk[CHUNK][CHUNK];
    alignas(16) __shared__ float s_gram_qk[CHUNK][CHUNK];
    alignas(16) __shared__ float s_a_inv[CHUNK][CHUNK];
    alignas(16) __shared__ float s_power[CHUNK][CHUNK];
    alignas(16) __shared__ float s_tmp_pow[CHUNK][CHUNK];
    alignas(16) __shared__ float s_tmp_sol[CHUNK][CHUNK];
    __shared__ float  s_log2_g[CHUNK];
    __shared__ float  s_G[CHUNK];
    __shared__ float  s_Gsafe[CHUNK];
    __shared__ float  s_beta[CHUNK];
    __shared__ float  s_bg[CHUNK];
    alignas(16) __shared__ float s_partial[WARPS][16][16];

    // 1. Load Q and K into smem.
    for (int i = tid; i < CHUNK * kHeadDim; i += kBlockThreads) {
        const int m = i / kHeadDim;
        const int c = i - m * kHeadDim;
        const int tok = chunk_start + m;
        BfTile qv = __float2bfloat16(0.0f);
        BfTile kv = __float2bfloat16(0.0f);
        if (tok < seq_end) {
            qv = q[(tok * kNumQHeads + qk_head) * kHeadDim + c];
            kv = k[(tok * kNumQHeads + qk_head) * kHeadDim + c];
        }
        s_Q[m][c] = qv;
        s_K[m][c] = kv;
    }

    // 2. Gates: compute log2_g, G, Gc, beta, bg for this chunk.
    if (tid < CHUNK) {
        const int m = tid;
        const int tok = chunk_start + m;
        const float A_log_val = A_log[v_head];
        const float dt_bias_val = dt_bias[v_head];
        constexpr float kLog2e = 1.4426950408889634f;
        constexpr float kLn2 = 0.6931471805599453f;
        const float decay = __expf(A_log_val);
        float log2_g_v = 0.0f;
        float beta_v = 0.0f;
        if (tok < seq_end) {
            const int gate_off = tok * kNumVHeads + v_head;
            const float a_val = __bfloat162float(a[gate_off]);
            const float b_val = __bfloat162float(b[gate_off]);
            const float x_g = a_val + dt_bias_val;
            const float sp = (x_g > 20.0f) ? x_g : (log2f(1.0f + exp2f(x_g * kLog2e)) * kLn2);
            log2_g_v = -decay * sp * kLog2e;
            beta_v = 1.0f / (1.0f + __expf(-b_val));
        }
        s_log2_g[m] = log2_g_v;
        s_beta[m] = beta_v;
    }
    __syncthreads();

    // Cumsum for G / Gc / Gsafe / bg.
    if (tid == 0) {
        float cum = 0.0f;
        #pragma unroll
        for (int m = 0; m < CHUNK; ++m) {
            cum += s_log2_g[m];
            const float g = exp2f(cum);
            s_G[m] = g;
            s_Gsafe[m] = fmaxf(g, 1e-30f);
            s_bg[m] = s_beta[m] / fmaxf(g, 1e-30f);
        }
    }
    __syncthreads();
    const float Gc = s_G[CHUNK - 1];

    // 3. gram_kk = K @ K^T (split-K across warps, reduce in smem).
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
        wmma::fill_fragment(c_f, 0.0f);
        const int kt_start = warp_id * kKTilesPerWarp;
        #pragma unroll
        for (int i = 0; i < kKTilesPerWarp; ++i) {
            const int kt = (kt_start + i) * 16;
            wmma::load_matrix_sync(a_f, &s_K[0][kt], kHeadDim);
            wmma::load_matrix_sync(b_f, &s_K[0][kt], kHeadDim);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
        }
        wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < 16 * 16; i += kBlockThreads) {
        const int m = i / 16, j = i - m * 16;
        float s = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
        s_gram_kk[m][j] = s;
    }

    // 4. gram_qk = Q @ K^T (split-K, reduce).
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
        wmma::fill_fragment(c_f, 0.0f);
        const int kt_start = warp_id * kKTilesPerWarp;
        #pragma unroll
        for (int i = 0; i < kKTilesPerWarp; ++i) {
            const int kt = (kt_start + i) * 16;
            wmma::load_matrix_sync(a_f, &s_Q[0][kt], kHeadDim);
            wmma::load_matrix_sync(b_f, &s_K[0][kt], kHeadDim);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
        }
        __syncthreads();
        wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < 16 * 16; i += kBlockThreads) {
        const int m = i / 16, j = i - m * 16;
        float s = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][j];
        s_gram_qk[m][j] = s;
    }

    // 5. N = tril_strict(beta * gram_kk)  [fp32 in s_power]. Init a_inv = I in s_a_inv.
    for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
        const int m = i / CHUNK, j = i - m * CHUNK;
        const float nv = (j < m) ? (s_beta[m] * s_gram_kk[m][j]) : 0.0f;
        s_power[m][j] = nv;
        s_a_inv[m][j] = (m == j) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // 6. Neumann doubling: a_inv = (I - N)(I + N^2)(I + N^4)(I + N^8) * I.
    //    Equivalently, progressively fold power into a_inv with signed add.
    //    sol_0 = I, sol_{k+1} = sol_k + sgn_k * power_k @ sol_k.
    //    For (I+N)^-1 = (I-N)(I+N^2)(I+N^4)(I+N^8): sgn_0 = -1, rest = +1.
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const float sgn = (step == 0) ? -1.0f : 1.0f;
        const bool squares = (step < 3);
        // Pass 1: tmp_sol = power @ a_inv; tmp_pow = power @ power (last step skips).
        for (int i = tid; i < CHUNK * CHUNK + (squares ? CHUNK * CHUNK : 0); i += kBlockThreads) {
            if (i < CHUNK * CHUNK) {
                const int m = i / CHUNK, r = i - m * CHUNK;
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < CHUNK; ++j) acc += s_power[m][j] * s_a_inv[j][r];
                s_tmp_sol[m][r] = acc;
            } else {
                const int k = i - CHUNK * CHUNK;
                const int m = k / CHUNK, r = k - m * CHUNK;
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < CHUNK; ++j) acc += s_power[m][j] * s_power[j][r];
                s_tmp_pow[m][r] = acc;
            }
        }
        __syncthreads();
        // Pass 2: a_inv += sgn * tmp_sol; power <- tmp_pow.
        for (int i = tid; i < CHUNK * CHUNK + (squares ? CHUNK * CHUNK : 0); i += kBlockThreads) {
            if (i < CHUNK * CHUNK) {
                const int m = i / CHUNK, r = i - m * CHUNK;
                s_a_inv[m][r] += sgn * s_tmp_sol[m][r];
            } else {
                const int k = i - CHUNK * CHUNK;
                const int m = k / CHUNK, r = k - m * CHUNK;
                s_power[m][r] = s_tmp_pow[m][r];
            }
        }
        __syncthreads();
    }

    // 7. Cast a_inv to bf16 and build beta_k = beta * K (bf16) for wmma.
    for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
        const int m = i / CHUNK, j = i - m * CHUNK;
        s_a_inv_bf[m][j] = __float2bfloat16_rn(s_a_inv[m][j]);
    }
    for (int i = tid; i < CHUNK * kHeadDim; i += kBlockThreads) {
        const int m = i / kHeadDim, c = i - m * kHeadDim;
        s_beta_k_bf[m][c] = __float2bfloat16_rn(s_beta[m] * __bfloat162float(s_K[m][c]));
    }
    __syncthreads();

    // 8. w = a_inv @ (beta * K).   wmma shapes: [16,16] x [16,128] → [16,128]
    //    Split across warps on the K=128 output dim (4 warps × 32 cols each = 128).
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
        const int kt_start = warp_id * kKTilesPerWarp;
        wmma::load_matrix_sync(a_f, &s_a_inv_bf[0][0], CHUNK);
        #pragma unroll
        for (int i = 0; i < kKTilesPerWarp; ++i) {
            const int kt = (kt_start + i) * 16;
            wmma::fill_fragment(c_f, 0.0f);
            wmma::load_matrix_sync(b_f, &s_beta_k_bf[0][kt], kHeadDim);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
            for (int j = lane; j < 16 * 16; j += kWarpSize) {
                const int rr = j / 16, cc = j - rr * 16;
                s_w[rr][kt + cc] = __float2bfloat16_rn(s_partial[warp_id][rr][cc]);
            }
        }
    }
    __syncthreads();

    // 9. Store outputs to global.
    const int base_record = (seq_idx * kNumVHeads + v_head) * max_chunks + chunk_id;
    // a_inv: fp32 [16][16]
    {
        float* dst = wy_a_inv + base_record * CHUNK * CHUNK;
        for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
            const int m = i / CHUNK, j = i - m * CHUNK;
            dst[m * CHUNK + j] = s_a_inv[m][j];
        }
    }
    // w: bf16 [16][128]
    {
        __nv_bfloat16* dst = wy_w + base_record * CHUNK * kHeadDim;
        for (int i = tid; i < CHUNK * kHeadDim; i += kBlockThreads) {
            const int m = i / kHeadDim, c = i - m * kHeadDim;
            dst[m * kHeadDim + c] = s_w[m][c];
        }
    }
    // qk_lower: bf16 [16][16] — tril(gram_qk).
    {
        __nv_bfloat16* dst = wy_qk + base_record * CHUNK * CHUNK;
        for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
            const int m = i / CHUNK, j = i - m * CHUNK;
            const float v = (j <= m) ? s_gram_qk[m][j] : 0.0f;
            dst[m * CHUNK + j] = __float2bfloat16_rn(v);
        }
    }
    // G, Gc, bg: fp32 vectors.
    if (tid < CHUNK) {
        wy_g[base_record * CHUNK + tid] = s_G[tid];
        wy_bg[base_record * CHUNK + tid] = s_bg[tid];
    }
    if (tid == 0) {
        wy_gc[base_record] = Gc;
    }
}

// ---- v18 Consumer ---------------------------------------------------------
template <int BV, int CHUNK, int WARPS>
__global__ __launch_bounds__(kWarpSize * WARPS, 2) void gdn_prefill_v18_consumer(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    const float* __restrict__ state,
    const int64_t* __restrict__ cu_seqlens,
    const float* __restrict__ wy_a_inv,
    const __nv_bfloat16* __restrict__ wy_w,
    const __nv_bfloat16* __restrict__ wy_qk,
    const float* __restrict__ wy_g,
    const float* __restrict__ wy_gc,
    const float* __restrict__ wy_bg,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ new_state,
    int num_seqs,
    int max_chunks,
    float scale
) {
    static_assert(CHUNK == 16 && WARPS == 4 && BV == 16, "v18 consumer: fixed shape");
    namespace wmma = nvcuda::wmma;
    using BfTile = __nv_bfloat16;

    constexpr int STAGES = 3;
    constexpr int BV_PAD = BV;
    constexpr int kNumNTiles = BV_PAD / 16;
    constexpr int kNumTiles = kHeadDim / BV;
    constexpr int kBlockThreads = kWarpSize * WARPS;
    constexpr int kNumKTiles = kHeadDim / 16;
    constexpr int kKTilesPerWarp = kNumKTiles / WARPS;
    constexpr int kWarpsPerNTile = WARPS / kNumNTiles;
    constexpr int kKTilesPerWarpEff = kNumKTiles / kWarpsPerNTile;

    const int flat = static_cast<int>(blockIdx.x);
    const int seq_stride = kNumVHeads * kNumTiles;
    const int seq_idx = flat / seq_stride;
    if (seq_idx >= num_seqs) return;
    const int rem = flat - seq_idx * seq_stride;
    const int v_head = rem / kNumTiles;
    const int v_tile = rem - v_head * kNumTiles;
    const int qk_head = v_head / kHeadGroupRatio;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int v_base = v_tile * BV;
    const int state_base = (seq_idx * kNumVHeads + v_head) * kHeadDim * kHeadDim;
    const int record_base_seqhead = (seq_idx * kNumVHeads + v_head) * max_chunks;

    alignas(128) __shared__ BfTile s_Q[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_K[STAGES][CHUNK][kHeadDim];
    alignas(128) __shared__ BfTile s_V[STAGES][CHUNK][BV];
    alignas(16)  __shared__ BfTile s_state_bf[BV_PAD][kHeadDim];
    alignas(128) __shared__ BfTile s_w_bf[CHUNK][kHeadDim];
    alignas(16)  __shared__ BfTile s_x_bf[CHUNK][BV_PAD];
    alignas(16)  __shared__ BfTile s_qk_lower_bf[CHUNK][CHUNK];
    alignas(16) __shared__ float s_a_inv[CHUNK][CHUNK];
    __shared__ float  s_G[CHUNK];
    __shared__ float  s_bg[CHUNK];
    __shared__ float  s_Gc_v;
    alignas(16) __shared__ float s_rhs[CHUNK][BV_PAD];
    alignas(16) __shared__ float s_u_tile[CHUNK][BV_PAD];
    alignas(16) __shared__ float s_w_state[CHUNK][BV_PAD];
    alignas(16) __shared__ float s_state_q[CHUNK][BV_PAD];
    alignas(16) __shared__ float s_qk_contrib[CHUNK][BV_PAD];
    alignas(16) __shared__ float s_partial[WARPS][16][16];
    alignas(8) __shared__ uint64_t mbar_load[STAGES];

    extern __shared__ char dyn_smem_raw[];
    float (&s_state_accum)[BV][kHeadDim] =
        *reinterpret_cast<float(*)[BV][kHeadDim]>(dyn_smem_raw);

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) mbarrier_init_one(&mbar_load[s], 1);
        fence_async_shared();
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i - r * kHeadDim;
        s_state_accum[r][c] = state[state_base + (v_base + r) * kHeadDim + c];
    }

    const int seq_start = static_cast<int>(cu_seqlens[seq_idx]);
    const int seq_end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    __syncthreads();

    if (seq_end <= seq_start) {
        for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i - r * kHeadDim;
            new_state[state_base + (v_base + r) * kHeadDim + c] = s_state_accum[r][c];
        }
        return;
    }

    constexpr uint32_t kQBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kKBytes = CHUNK * kHeadDim * sizeof(BfTile);
    constexpr uint32_t kVBytes = CHUNK * BV * sizeof(BfTile);
    constexpr uint32_t kTotalTMABytes = kQBytes + kKBytes + kVBytes;

    auto issue_tma = [&](int stage, int chunk_start) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar_load[stage], kTotalTMABytes);
            cp_tma_3d(&s_Q[stage][0][0], &tma_q, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_K[stage][0][0], &tma_k, 0,      qk_head, chunk_start, &mbar_load[stage]);
            cp_tma_3d(&s_V[stage][0][0], &tma_v, v_base, v_head,  chunk_start, &mbar_load[stage]);
        }
    };
    uint32_t phase[STAGES] = {0, 0, 0};
    auto wait_stage = [&](int stage) {
        mbarrier_wait_parity(&mbar_load[stage], phase[stage]);
        phase[stage] ^= 1;
    };

    const int total_chunks = (seq_end - seq_start + CHUNK - 1) / CHUNK;
    const int kLookahead = STAGES - 1;
    const int pre_issued = (total_chunks < kLookahead) ? total_chunks : kLookahead;
    for (int s = 0; s < pre_issued; ++s) {
        issue_tma(s, seq_start + s * CHUNK);
    }
    wait_stage(0);
    __syncthreads();

    int stage = 0;
    int chunk_start = seq_start;
    int next_issue = pre_issued;
    int chunk_id = 0;
    while (chunk_start < seq_end) {
        const int next_chunk = chunk_start + CHUNK;
        const bool has_next = (next_chunk < seq_end);
        const int next_stage = (stage + 1) % STAGES;

        if (next_issue < total_chunks) {
            const int issue_slot = next_issue % STAGES;
            issue_tma(issue_slot, seq_start + next_issue * CHUNK);
            next_issue++;
        }

        // Materialize state bf16 shadow.
        for (int i = tid; i < BV_PAD * kHeadDim; i += kBlockThreads) {
            const int r = i / kHeadDim;
            const int c = i - r * kHeadDim;
            const float src = (r < BV) ? s_state_accum[r][c] : 0.0f;
            s_state_bf[r][c] = __float2bfloat16_rn(src);
        }

        // Load wy data from global (prepass outputs).
        const int record_base = record_base_seqhead + chunk_id;
        {
            const float* src_ainv = wy_a_inv + record_base * CHUNK * CHUNK;
            for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
                const int m = i / CHUNK, j = i - m * CHUNK;
                s_a_inv[m][j] = src_ainv[m * CHUNK + j];
            }
        }
        {
            const __nv_bfloat16* src_w = wy_w + record_base * CHUNK * kHeadDim;
            for (int i = tid; i < CHUNK * kHeadDim; i += kBlockThreads) {
                const int m = i / kHeadDim, c = i - m * kHeadDim;
                s_w_bf[m][c] = src_w[m * kHeadDim + c];
            }
        }
        {
            const __nv_bfloat16* src_qk = wy_qk + record_base * CHUNK * CHUNK;
            for (int i = tid; i < CHUNK * CHUNK; i += kBlockThreads) {
                const int m = i / CHUNK, j = i - m * CHUNK;
                s_qk_lower_bf[m][j] = src_qk[m * CHUNK + j];
            }
        }
        if (tid < CHUNK) {
            s_G[tid] = wy_g[record_base * CHUNK + tid];
            s_bg[tid] = wy_bg[record_base * CHUNK + tid];
        }
        if (tid == 0) {
            s_Gc_v = wy_gc[record_base];
        }
        __syncthreads();
        const float Gc = s_Gc_v;

        // Build rhs = bg[:,None] * V_tile  (fp32).
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i - m * BV_PAD;
            float vv = 0.0f;
            if (r < BV) vv = __bfloat162float(s_V[stage][m][r]);
            s_rhs[m][r] = s_bg[m] * vv;
        }
        __syncthreads();

        // u_tile = a_inv @ rhs  (fp32 scalar matvec, parallel across threads).
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i - m * BV_PAD;
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < CHUNK; ++j) acc += s_a_inv[m][j] * s_rhs[j][r];
            s_u_tile[m][r] = acc;
        }

        // w_state = w @ state^T  (bf16 wmma, split-K across warps, reduce).
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
            #pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_w_bf[0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i - m * BV_PAD;
            if (n < 16) {
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][n];
                s_w_state[m][n] = s;
            } else {
                s_w_state[m][n] = 0.0f;
            }
        }
        __syncthreads();

        // state_q = Q @ state^T  (bf16 wmma, split-K, reduce).
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::col_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            const int kt_start = warp_id * kKTilesPerWarp;
            #pragma unroll
            for (int i = 0; i < kKTilesPerWarp; ++i) {
                const int kt = (kt_start + i) * 16;
                wmma::load_matrix_sync(a_f, &s_Q[stage][0][kt], kHeadDim);
                wmma::load_matrix_sync(b_f, &s_state_bf[0][kt], kHeadDim);
                wmma::mma_sync(c_f, a_f, b_f, c_f);
            }
            wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < 16 * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, n = i - m * BV_PAD;
            if (n < 16) {
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) s += s_partial[w][m][n];
                s_state_q[m][n] = s;
            } else {
                s_state_q[m][n] = 0.0f;
            }
        }
        __syncthreads();

        // x_chunk = u_tile - w_state; zero masked rows; cast to bf16.
        for (int i = tid; i < CHUNK * BV_PAD; i += kBlockThreads) {
            const int m = i / BV_PAD, r = i - m * BV_PAD;
            const int tok = chunk_start + m;
            const bool active = (tok < seq_end) && (r < BV);
            const float x = active ? (s_u_tile[m][r] - s_w_state[m][r]) : 0.0f;
            s_x_bf[m][r] = __float2bfloat16_rn(x);
        }
        __syncthreads();

        // qk_contrib = qk_lower_bf @ x_bf  (wmma, warp-0 since shape is 16x16).
        if (warp_id < kNumNTiles) {
            const int n_tile = warp_id;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::row_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            wmma::fill_fragment(c_f, 0.0f);
            wmma::load_matrix_sync(a_f, &s_qk_lower_bf[0][0], CHUNK);
            wmma::load_matrix_sync(b_f, &s_x_bf[0][n_tile * 16], BV_PAD);
            wmma::mma_sync(c_f, a_f, b_f, c_f);
            wmma::store_matrix_sync(&s_qk_contrib[0][n_tile * 16], c_f, BV_PAD, wmma::mem_row_major);
        }
        __syncthreads();

        // Output.
        for (int i = tid; i < CHUNK * BV; i += kBlockThreads) {
            const int m = i / BV, r = i - m * BV;
            const int tok = chunk_start + m;
            if (tok < seq_end) {
                const float out_val = scale * s_G[m] * (s_state_q[m][r] + s_qk_contrib[m][r]);
                out[(tok * kNumVHeads + v_head) * kHeadDim + v_base + r] = __float2bfloat16_rn(out_val);
            }
        }

        // State update: state = Gc * (state + x^T @ K)  via wmma.
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, BfTile, wmma::col_major> a_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, BfTile, wmma::row_major> b_f;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;
            const int nt_start = warp_id * kKTilesPerWarp;
            #pragma unroll
            for (int m_tile = 0; m_tile < kNumNTiles; ++m_tile) {
                wmma::load_matrix_sync(a_f, &s_x_bf[0][m_tile * 16], BV_PAD);
                #pragma unroll
                for (int i = 0; i < kKTilesPerWarp; ++i) {
                    const int nt = (nt_start + i) * 16;
                    wmma::load_matrix_sync(b_f, &s_K[stage][0][nt], kHeadDim);
                    wmma::fill_fragment(c_f, 0.0f);
                    wmma::mma_sync(c_f, a_f, b_f, c_f);
                    wmma::store_matrix_sync(&s_partial[warp_id][0][0], c_f, 16, wmma::mem_row_major);
                    for (int j = lane; j < 16 * 16; j += kWarpSize) {
                        const int rr = j / 16, cc = j - rr * 16;
                        const int global_r = m_tile * 16 + rr;
                        if (global_r < BV) {
                            s_state_accum[global_r][nt + cc] = Gc *
                                (s_state_accum[global_r][nt + cc] + s_partial[warp_id][rr][cc]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (!has_next) break;
        wait_stage(next_stage);
        __syncthreads();
        stage = next_stage;
        chunk_start = next_chunk;
        chunk_id += 1;
    }

    for (int i = tid; i < BV * kHeadDim; i += kBlockThreads) {
        const int r = i / kHeadDim;
        const int c = i - r * kHeadDim;
        __stcs(&new_state[state_base + (v_base + r) * kHeadDim + c], s_state_accum[r][c]);
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
        } else {
            const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kLongBV);
            gdn_prefill_v7<kLongBV, 4><<<total_tiles, kWarpSize * 4, 0, stream.stream()>>>(
                q_ptr, k_ptr, v_ptr, state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
                cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
            );
        }
    };

    auto build_qk_tma = [&](const __nv_bfloat16* gptr) -> CUtensorMap {
        CUtensorMap desc = {};
        uint64_t globalDim[3] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(kNumQHeads),
            static_cast<uint64_t>(total_tokens)
        };
        uint64_t globalStrides[2] = {
            static_cast<uint64_t>(kHeadDim) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(kNumQHeads) * kHeadDim * sizeof(__nv_bfloat16)
        };
        uint32_t boxDim[3] = {static_cast<uint32_t>(kHeadDim), 1u, 16u};
        uint32_t elementStrides[3] = {1u, 1u, 1u};
        cuTensorMapEncodeTiled(
            &desc,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
            const_cast<void*>(static_cast<const void*>(gptr)),
            globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        return desc;
    };
    auto build_v_tma = [&](const __nv_bfloat16* gptr, uint32_t box_w) -> CUtensorMap {
        CUtensorMap desc = {};
        uint64_t globalDim[3] = {
            static_cast<uint64_t>(kHeadDim),
            static_cast<uint64_t>(kNumVHeads),
            static_cast<uint64_t>(total_tokens)
        };
        uint64_t globalStrides[2] = {
            static_cast<uint64_t>(kHeadDim) * sizeof(__nv_bfloat16),
            static_cast<uint64_t>(kNumVHeads) * kHeadDim * sizeof(__nv_bfloat16)
        };
        uint32_t boxDim[3] = {box_w, 1u, 16u};
        uint32_t elementStrides[3] = {1u, 1u, 1u};
        cuTensorMapEncodeTiled(
            &desc,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
            const_cast<void*>(static_cast<const void*>(gptr)),
            globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        return desc;
    };

    auto launch_v12 = [&]() {
        // v12: TMA-pipelined Q/K/V loads. Requires SM_90+; otherwise v7.
        cudaDeviceProp props;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&props, dev);
        if (props.major < 9) {
            launch_v7();
            return;
        }
        // num_seqs=1 long-seq: still under-fills the grid at BV=16.
        if (num_seqs == 1 && avg_seq_len >= 512) {
            launch_v7();
            return;
        }
        CUtensorMap tma_q = build_qk_tma(q_ptr);
        CUtensorMap tma_k = build_qk_tma(k_ptr);
        CUtensorMap tma_v = build_v_tma(v_ptr, 16u);

        constexpr int kV12BV = 16;
        constexpr int kV12Chunk = 16;
        constexpr int kV12Warps = 4;
        const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kV12BV);
        const size_t dyn_smem_bytes = sizeof(float) * kV12BV * kHeadDim;
        auto* kernel = gdn_prefill_v12<kV12BV, kV12Chunk, kV12Warps>;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            100 * 1024
        );
        kernel<<<total_tiles, kWarpSize * kV12Warps, dyn_smem_bytes, stream.stream()>>>(
            tma_q, tma_k, tma_v,
            state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
        );
    };

    auto launch_v13 = [&]() {
        // v13: v12 + 3-stage TMA buffer. Requires SM_90+; otherwise v7.
        cudaDeviceProp props;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&props, dev);
        if (props.major < 9) {
            launch_v7();
            return;
        }
        if (num_seqs == 1 && avg_seq_len >= 512) {
            launch_v7();
            return;
        }
        CUtensorMap tma_q = build_qk_tma(q_ptr);
        CUtensorMap tma_k = build_qk_tma(k_ptr);
        CUtensorMap tma_v = build_v_tma(v_ptr, 16u);

        constexpr int kV13BV = 16;
        constexpr int kV13Chunk = 16;
        constexpr int kV13Warps = 4;
        const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kV13BV);
        const size_t dyn_smem_bytes = sizeof(float) * kV13BV * kHeadDim;
        auto* kernel = gdn_prefill_v13<kV13BV, kV13Chunk, kV13Warps>;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            100 * 1024
        );
        kernel<<<total_tiles, kWarpSize * kV13Warps, dyn_smem_bytes, stream.stream()>>>(
            tma_q, tma_k, tma_v,
            state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
        );
    };

    auto launch_v14 = [&]() {
        // v14: v13 + Neumann-doubling fp32 solve. Requires SM_90+; otherwise v7.
        cudaDeviceProp props;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&props, dev);
        if (props.major < 9) {
            launch_v7();
            return;
        }
        if (num_seqs == 1 && avg_seq_len >= 512) {
            launch_v7();
            return;
        }
        CUtensorMap tma_q = build_qk_tma(q_ptr);
        CUtensorMap tma_k = build_qk_tma(k_ptr);
        CUtensorMap tma_v = build_v_tma(v_ptr, 16u);

        constexpr int kV14BV = 16;
        constexpr int kV14Chunk = 16;
        constexpr int kV14Warps = 4;
        const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kV14BV);
        const size_t dyn_smem_bytes = sizeof(float) * kV14BV * kHeadDim;
        auto* kernel = gdn_prefill_v14<kV14BV, kV14Chunk, kV14Warps>;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            100 * 1024
        );
        kernel<<<total_tiles, kWarpSize * kV14Warps, dyn_smem_bytes, stream.stream()>>>(
            tma_q, tma_k, tma_v,
            state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
        );
    };

    auto launch_v18 = [&]() {
        // v18: two-kernel prepass + consumer, mirroring Triton's split_wy. SM_90+.
        cudaDeviceProp props;
        int dev18 = 0;
        cudaGetDevice(&dev18);
        cudaGetDeviceProperties(&props, dev18);
        if (props.major < 9) { launch_v7(); return; }

        constexpr int kV18BV = 16;
        constexpr int kV18Chunk = 16;
        constexpr int kV18Warps = 4;
        const int max_chunks = (total_tokens + kV18Chunk - 1) / kV18Chunk;
        if (max_chunks <= 0) { launch_v7(); return; }

        // Allocate prepass scratch buffers on the device.
        auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
        auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
        const int64_t n_records = static_cast<int64_t>(num_seqs) * kNumVHeads * max_chunks;
        torch::Tensor wy_a_inv_t = torch::empty({n_records, kV18Chunk, kV18Chunk}, opt_f32);
        torch::Tensor wy_w_t     = torch::empty({n_records, kV18Chunk, kHeadDim}, opt_bf16);
        torch::Tensor wy_qk_t    = torch::empty({n_records, kV18Chunk, kV18Chunk}, opt_bf16);
        torch::Tensor wy_g_t     = torch::empty({n_records, kV18Chunk}, opt_f32);
        torch::Tensor wy_gc_t    = torch::empty({n_records}, opt_f32);
        torch::Tensor wy_bg_t    = torch::empty({n_records, kV18Chunk}, opt_f32);

        float* wy_a_inv_ptr = wy_a_inv_t.data_ptr<float>();
        __nv_bfloat16* wy_w_ptr  = reinterpret_cast<__nv_bfloat16*>(wy_w_t.data_ptr());
        __nv_bfloat16* wy_qk_ptr = reinterpret_cast<__nv_bfloat16*>(wy_qk_t.data_ptr());
        float* wy_g_ptr  = wy_g_t.data_ptr<float>();
        float* wy_gc_ptr = wy_gc_t.data_ptr<float>();
        float* wy_bg_ptr = wy_bg_t.data_ptr<float>();

        // Prepass launch: grid (num_seqs * 8_v_heads, max_chunks).
        dim3 pre_grid(num_seqs * kNumVHeads, max_chunks);
        gdn_prefill_v18_prepass<<<pre_grid, 128, 0, stream.stream()>>>(
            q_ptr, k_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr,
            wy_a_inv_ptr, wy_w_ptr, wy_qk_ptr, wy_g_ptr, wy_gc_ptr, wy_bg_ptr,
            num_seqs, max_chunks
        );

        // Consumer launch via TMA.
        CUtensorMap tma_q = build_qk_tma(q_ptr);
        CUtensorMap tma_k = build_qk_tma(k_ptr);
        CUtensorMap tma_v = build_v_tma(v_ptr, 16u);

        const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kV18BV);
        const size_t dyn_smem_bytes = sizeof(float) * kV18BV * kHeadDim;
        auto* kernel = gdn_prefill_v18_consumer<kV18BV, kV18Chunk, kV18Warps>;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            100 * 1024
        );
        kernel<<<total_tiles, kWarpSize * kV18Warps, dyn_smem_bytes, stream.stream()>>>(
            tma_q, tma_k, tma_v,
            state.data_ptr<float>(), cu_seqlens_ptr,
            wy_a_inv_ptr, wy_w_ptr, wy_qk_ptr, wy_g_ptr, wy_gc_ptr, wy_bg_ptr,
            out_ptr, new_state_ptr, num_seqs, max_chunks, scale
        );
    };

    auto launch_v17 = [&]() {
        // v17: v13 body with __launch_bounds__(128, 3) — 3 blocks/SM min for higher occupancy.
        cudaDeviceProp props;
        int dev17 = 0;
        cudaGetDevice(&dev17);
        cudaGetDeviceProperties(&props, dev17);
        if (props.major < 9) { launch_v7(); return; }
        CUtensorMap tma_q = build_qk_tma(q_ptr);
        CUtensorMap tma_k = build_qk_tma(k_ptr);
        CUtensorMap tma_v = build_v_tma(v_ptr, 16u);
        constexpr int kV17BV = 16;
        constexpr int kV17Chunk = 16;
        constexpr int kV17Warps = 4;
        const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kV17BV);
        const size_t dyn_smem_bytes = sizeof(float) * kV17BV * kHeadDim;
        auto* kernel = gdn_prefill_v17<kV17BV, kV17Chunk, kV17Warps>;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            100 * 1024
        );
        kernel<<<total_tiles, kWarpSize * kV17Warps, dyn_smem_bytes, stream.stream()>>>(
            tma_q, tma_k, tma_v,
            state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
        );
    };

    auto launch_v15 = [&]() {
        // v15: v13 TMA kernel reused, but *without* the num_seqs=1 long-seq fallback to v7.
        // Tests whether v13's TMA-pipelined BV=16 path beats v7 for long single-seq workloads.
        cudaDeviceProp props;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&props, dev);
        if (props.major < 9) {
            launch_v7();
            return;
        }
        CUtensorMap tma_q = build_qk_tma(q_ptr);
        CUtensorMap tma_k = build_qk_tma(k_ptr);
        CUtensorMap tma_v = build_v_tma(v_ptr, 16u);

        constexpr int kV15BV = 16;
        constexpr int kV15Chunk = 16;
        constexpr int kV15Warps = 4;
        const int total_tiles = num_seqs * kNumVHeads * (kHeadDim / kV15BV);
        const size_t dyn_smem_bytes = sizeof(float) * kV15BV * kHeadDim;
        auto* kernel = gdn_prefill_v13<kV15BV, kV15Chunk, kV15Warps>;
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            100 * 1024
        );
        kernel<<<total_tiles, kWarpSize * kV15Warps, dyn_smem_bytes, stream.stream()>>>(
            tma_q, tma_k, tma_v,
            state.data_ptr<float>(), A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
            cu_seqlens_ptr, out_ptr, new_state_ptr, num_seqs, scale
        );
    };

    switch (kBenchmarkKernelVersion) {
        case 7:
            launch_v7();
            break;
        case 12:
            launch_v12();
            break;
        case 13:
            launch_v13();
            break;
        case 14:
            launch_v14();
            break;
        case 15:
            launch_v15();
            break;
        case 17:
            launch_v17();
            break;
        case 18:
            launch_v18();
            break;
        default:
            launch_v18();
            break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gdn", &launch_gdn, "Launch GDN prefill CUDA kernel");
}
