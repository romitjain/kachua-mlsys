"""Triton v2 GDN prefill kernel — chunkwise matmul + Neumann inverse.

Each sequence is split into fixed-size chunks. Within a chunk:
    S_j = g_j S_{j-1} (I - beta_j k_j k_j^T) + beta_j v_j k_j^T
  =>  S_out = G_c * (S_in (I - U K^T) + V_hat R^T)
    where G_c = prod_j g_j and (I + N)^{-1} is computed via Neumann doubling
    since N is strictly lower-triangular and nilpotent (N^C = 0).

Key knobs layered on top of the algorithm:
- Tensor-core `tl.dot`: TF32 for the Neumann solve (Blackwell); bf16 for
  single-shot K=128 contractions.
- Adaptive BV (V tile) and CHUNK sized per (num_seqs, avg_seq_len).
- `maxnreg = 128` caps register pressure to keep two warps resident per SM.
- log2 / exp2 gate cumsum using the SFU path.
- `_safe_dot_bf16` inline-asm wrapper dodges a Blackwell
  TritonGPUHoistTMEMAlloc compiler bug.

State layout in memory is kept in external k-last form [V, K].
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from tvm_ffi import register_global_func as register_func

NUM_Q_HEADS: int = 4
NUM_K_HEADS: int = 4
NUM_V_HEADS: int = 8
HEAD_DIM: int = 128
GVA_RATIO: int = NUM_V_HEADS // NUM_Q_HEADS
CHUNK_SIZE: int = 16


def _bucket_seq_len(max_len: int) -> int:
    if max_len <= 64:
        return 0
    if max_len <= 256:
        return 1
    if max_len <= 1024:
        return 2
    return 3


def _fallback_bv(num_seqs: int, total_tokens: int, avg_seq_len: int) -> int:
    if num_seqs >= 16:
        return 32
    if num_seqs >= 4 or total_tokens <= 512:
        return 16
    if num_seqs == 1 and avg_seq_len >= 1024:
        return 4
    return 8


@register_func("flashinfer.kernel")
def launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    q = torch.from_dlpack(q)
    k = torch.from_dlpack(k)
    v = torch.from_dlpack(v)
    state_tensor = torch.from_dlpack(state) if state is not None else None
    A_log = torch.from_dlpack(A_log)
    a = torch.from_dlpack(a)
    dt_bias = torch.from_dlpack(dt_bias)
    b = torch.from_dlpack(b)
    cu_seqlens = torch.from_dlpack(cu_seqlens)

    num_seqs = int(cu_seqlens.numel() - 1)
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(HEAD_DIM)

    if output is None:
        output_tensor = torch.empty(
            (q.shape[0], NUM_V_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        output_tensor = torch.from_dlpack(output)

    if new_state is None:
        new_state_tensor = torch.empty(
            (num_seqs, NUM_V_HEADS, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        new_state_tensor = torch.from_dlpack(new_state)

    if state_tensor is None:
        state_tensor = torch.zeros_like(new_state_tensor)

    total_tokens = int(q.shape[0])
    avg_seq_len = max(1, (total_tokens + max(1, num_seqs) - 1) // max(1, num_seqs))
    # Use avg_seq_len as dispatch proxy to avoid a .item() GPU sync per launch
    # (workloads with mixed seq lengths are rare; avg is close enough).
    max_seq_len_proxy = avg_seq_len
    seq_bucket = _bucket_seq_len(max_seq_len_proxy)

    # Adaptive BV based on grid size (B200 has 148 SMs, want ≥1 wave):
    if num_seqs >= 4:
        bv = 32
    elif num_seqs >= 2:
        bv = 16
    else:
        bv = 8
    n_v_tiles = HEAD_DIM // bv

    # Adaptive CHUNK: longer sequences with few seqs benefit from CHUNK=32
    # (fewer sequential chunks amortizes per-chunk overhead). For multi-seq
    # with lots of parallelism, CHUNK=16 is better (smaller gram matrices).
    chunk = 32 if (num_seqs <= 3 and avg_seq_len >= 512) else 16

    grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
    gdn_prefill_kernel_v2[grid](
        q, k, v, state_tensor, A_log, a, dt_bias, b, cu_seqlens,
        output_tensor, new_state_tensor, scale, seq_bucket,
        K=HEAD_DIM, V_DIM=HEAD_DIM,
        NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS, GVA_RATIO=GVA_RATIO,
        CHUNK=chunk, BV=bv, N_V_TILES=n_v_tiles,
    )
    return output_tensor, new_state_tensor


# ---------------------------------------------------------------------------
# v2 chunkwise matmul-based kernel (adaptive C=16/32)
#
# Exact algebra for one chunk of C tokens, state kept in [V, K]:
#   G_j = prod_{i<=j} g_i, G_c = G_{C-1}
#   (I + N) X = beta * (V / G - K @ S_in^T)   where N[j,i] = beta_j*(k_j·k_i) for i<j
# Then:
#   O    = scale * G * (Q @ S_in^T + tril(Q @ K^T) @ X)
#   S_out = G_c * (S_in + X^T @ K)
#
# N is strictly-lower with zero diagonal and C-step nilpotent, so
#   (I + N)^{-1} = (I - N)(I + N^2)(I + N^4)... terms until power >= C.
# This is applied by `_apply_unit_lower_inverse`.
# ---------------------------------------------------------------------------


@triton.jit
def _dot_f32(a, b):
    # TF32 tensor cores: ~10-19 bits mantissa, fast on Blackwell. Tolerances
    # for the benchmark are 1e-2 atol/rtol, so TF32 is within budget.
    # Wrapped with inline-asm mov.f32 as a workaround for a Triton Blackwell
    # code-gen bug (TritonGPUHoistTMEMAlloc) that fuses dot with downstream
    # adds incorrectly — observed by FLA / Tomás Ruiz on B200.
    out = tl.dot(a, b, input_precision="tf32", out_dtype=tl.float32)
    return tl.inline_asm_elementwise(
        asm="mov.f32 $0, $1;",
        constraints="=r,r",
        args=[out],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _safe_dot_bf16(a, b):
    """bf16 tensor-core matmul with Blackwell safe-dot workaround."""
    out = tl.dot(a, b, out_dtype=tl.float32)
    return tl.inline_asm_elementwise(
        asm="mov.f32 $0, $1;",
        constraints="=r,r",
        args=[out],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _apply_unit_lower_inverse(nil, rhs, BV: tl.constexpr, CHUNK: tl.constexpr):
    """Apply (I + N)^{-1} via doubling: (I+N)^{-1}=(I-N)(I+N^2)(I+N^4)...

    Uses TF32 tensor cores for the [C,C]×[C,C] and [C,C]×[C,BV] matmuls.
    BF16 was previously used here but caused numerical drift on long
    sequences (>10 chunks) outside the 1e-2 benchmark tolerance. TF32
    keeps ~19 bits of mantissa, comfortably inside tolerance even after
    the log-depth Neumann chain. The big K=128 contractions in the main
    kernel body still use BF16 because they're single-shot matmuls whose
    rounding is bounded by the tolerance budget.
    """
    sol = rhs - _dot_f32(nil, rhs)
    power = _dot_f32(nil, nil)
    if CHUNK >= 4:
        sol = sol + _dot_f32(power, sol)
        power = _dot_f32(power, power)
    if CHUNK >= 8:
        sol = sol + _dot_f32(power, sol)
        power = _dot_f32(power, power)
    if CHUNK >= 16:
        sol = sol + _dot_f32(power, sol)
        power = _dot_f32(power, power)
    if CHUNK >= 32:
        sol = sol + _dot_f32(power, sol)
    return sol


@triton.autotune(
    configs=[
        # Mix capped-reg (high occupancy) and uncapped (fewer spills) configs
        # so autotune picks per (BV, seq_bucket): multi-seq favors capped,
        # single-seq long favors uncapped since the grid is already tiny.
        # num_stages=4 is Blackwell-friendly for deeper async pipelines.
        triton.Config({}, num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({}, num_warps=4, num_stages=3, maxnreg=128),
        triton.Config({}, num_warps=4, num_stages=4, maxnreg=128),
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=128),
        triton.Config({}, num_warps=8, num_stages=3, maxnreg=128),
        triton.Config({}, num_warps=8, num_stages=4, maxnreg=128),
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=96),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["BV", "seq_bucket"],
)
@triton.jit
def gdn_prefill_kernel_v2(
    q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr, out_ptr, new_state_ptr, scale, seq_bucket,
    K: tl.constexpr, V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr, GVA_RATIO: tl.constexpr,
    CHUNK: tl.constexpr, BV: tl.constexpr, N_V_TILES: tl.constexpr,
):
    """Chunkwise GDN prefill, tiled along V. One program per (sequence, V-head, V-tile)."""
    pid_seq = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES
    qk_head = pid_h // GVA_RATIO

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, BV)
    rows = tl.arange(0, CHUNK)
    cols = tl.arange(0, CHUNK)
    lower = rows[:, None] > cols[None, :]
    lower_eq = rows[:, None] >= cols[None, :]

    v_start = pid_v * BV

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]

    state_tile = tl.load(s_ptrs).to(tl.float32)  # [BV, K]

    if seq_end <= seq_start:
        tl.store(ns_ptrs, state_tile)
        return

    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    # g_t = exp(-exp(A_log) * softplus(a+dt_bias)) so log(g_t) = -exp(A_log) * softplus(...).
    # Keep everything in log space (scaled by 1/ln(2) → exp2) to avoid an exp→log round-trip
    # and use the single-cycle EX2 SFU op instead of 4-cycle EXP (FLA trick).
    neg_exp_A_log2e = -tl.exp(A_log_val) * 1.4426950408889634  # 1/ln(2)

    for chunk_start in tl.range(seq_start, seq_end, CHUNK):
        tok = chunk_start + rows
        active = tok < seq_end

        gate_idx = tok * NUM_V_HEADS + pid_h
        a_vals = tl.load(a_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)

        # Compute log2(g) directly (no redundant exp→log roundtrip).
        # log2-space softplus: 4-cycle EXP/LOG → 1-cycle EX2/LG2 SFU ops.
        #   softplus(x) = log(1 + exp(x)) = log2(1 + exp2(x*log2e)) * ln(2)
        x = a_vals + dt_bias_val
        x_log2e = x * 1.4426950408889634
        sp_x = tl.where(x > 20.0, x, tl.log2(1.0 + tl.exp2(x_log2e)) * 0.6931471805599453)
        log2_g = tl.where(active, neg_exp_A_log2e * sp_x, 0.0)
        beta_chunk = tl.where(active, tl.sigmoid(b_vals), 0.0)

        log2_G = tl.cumsum(log2_g, axis=0)                         # [C]
        G = tl.exp2(log2_G)                                         # [C]
        Gc = tl.sum(G * (rows == (CHUNK - 1)).to(tl.float32), axis=0)

        # Keep K/Q as bf16 for tensor-core matmul with fp32 accumulator.
        # On B200 bf16 tensor cores are ~4x faster than tf32 for the big K=128
        # contractions (K@K^T, Q@K^T, K@state^T, Q@state^T, x^T@K).
        qk_ptrs = tok[:, None] * (NUM_K_HEADS * K) + qk_head * K + o_k[None, :]
        K_tile_bf = tl.load(k_ptr + qk_ptrs, mask=active[:, None], other=0.0)

        v_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
        V_tile = tl.load(v_ptr + v_ptrs, mask=active[:, None], other=0.0).to(tl.float32)

        # bf16 × bf16 → fp32 tensor-core matmul for the K-dim contractions.
        K_tile_bf_T = tl.trans(K_tile_bf)
        gram_kk = _safe_dot_bf16(K_tile_bf, K_tile_bf_T)                        # [C, C]
        state_tile_bf = state_tile.to(tl.bfloat16)
        state_tile_bf_T = tl.trans(state_tile_bf)
        state_k = _safe_dot_bf16(K_tile_bf, state_tile_bf_T)                    # [C, BV]

        nil = tl.where(lower, gram_kk * beta_chunk[:, None], 0.0)
        # Clamp the divisor: G = exp2(cumsum(log2_g)) can underflow to 0 in fp32
        # for strong-decay workloads on CHUNK=32 (per-step g ~ exp(-5)+).
        # The downstream G multiplier at out_tile is left unclamped so that if
        # G is truly ~0, the gated contribution collapses to 0 rather than NaN.
        G_safe = tl.maximum(G, 1e-30)
        rhs = beta_chunk[:, None] * (V_tile / G_safe[:, None] - state_k)
        x_chunk = _apply_unit_lower_inverse(nil, rhs, BV=BV, CHUNK=CHUNK)      # [C, BV]
        x_chunk = tl.where(active[:, None], x_chunk, 0.0)

        Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)
        state_q = _safe_dot_bf16(Q_tile_bf, state_tile_bf_T)                   # [C, BV]
        gram_qk = _safe_dot_bf16(Q_tile_bf, K_tile_bf_T)                       # [C, C]
        qk_lower = tl.where(lower_eq, gram_qk, 0.0)

        # Cast x_chunk once for both remaining matmuls.
        x_chunk_bf = x_chunk.to(tl.bfloat16)
        qk_contrib = _safe_dot_bf16(qk_lower.to(tl.bfloat16), x_chunk_bf)
        out_tile = scale * G[:, None] * (state_q + qk_contrib)
        out_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
        tl.store(out_ptr + out_ptrs, out_tile.to(tl.bfloat16), mask=active[:, None])

        state_tile = Gc * (state_tile + _safe_dot_bf16(tl.trans(x_chunk_bf), K_tile_bf))

    tl.store(ns_ptrs, state_tile)


# ---------------------------------------------------------------------------
# v1 reference kernel (per-token scalar recurrence). Preserved inline — not
# dispatched via @register_func. Kept so the evolution from v1 -> v2 stays
# visible in one file, matching the decode kernel.cu convention of
# keeping historical variants alongside the production one.
# ---------------------------------------------------------------------------

BALANCED_BV: int = 8
BALANCED_NUM_WARPS: int = 8
BALANCED_NUM_STAGES: int = 4
MICRO_BV: int = 8
MICRO_NUM_WARPS: int = 4
MICRO_NUM_STAGES: int = 2
LONG_BV: int = 4
LONG_NUM_WARPS: int = 1
LONG_NUM_STAGES: int = 4
STARVED_BV: int = 2
STARVED_NUM_WARPS: int = 1
STARVED_NUM_STAGES: int = 2


def _launch_gdn_v1(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    """Launch the Triton prefill kernel for one packed batch of sequences."""
    q = torch.from_dlpack(q)
    k = torch.from_dlpack(k)
    v = torch.from_dlpack(v)
    state_tensor = torch.from_dlpack(state) if state is not None else None
    A_log = torch.from_dlpack(A_log)
    a = torch.from_dlpack(a)
    dt_bias = torch.from_dlpack(dt_bias)
    b = torch.from_dlpack(b)
    cu_seqlens = torch.from_dlpack(cu_seqlens)

    num_seqs = int(cu_seqlens.numel() - 1)
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(HEAD_DIM)
    avg_seq_len = max(1, (q.shape[0] + num_seqs - 1) // num_seqs)

    if num_seqs == 1 and avg_seq_len >= 512:
        kernel = gdn_prefill_kernel_v1_long
        bv, num_warps, num_stages = STARVED_BV, STARVED_NUM_WARPS, STARVED_NUM_STAGES
    elif num_seqs <= 2 and avg_seq_len >= 256:
        kernel = gdn_prefill_kernel_v1_long
        bv, num_warps, num_stages = LONG_BV, LONG_NUM_WARPS, LONG_NUM_STAGES
    else:
        kernel = gdn_prefill_kernel_v1
        bv, num_warps, num_stages = BALANCED_BV, BALANCED_NUM_WARPS, BALANCED_NUM_STAGES
    n_v_tiles = HEAD_DIM // bv

    if output is None:
        output_tensor = torch.empty(
            (q.shape[0], NUM_V_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        output_tensor = torch.from_dlpack(output)

    if new_state is None:
        new_state_tensor = torch.empty(
            (num_seqs, NUM_V_HEADS, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        new_state_tensor = torch.from_dlpack(new_state)

    if state_tensor is None:
        state_tensor = torch.zeros_like(new_state_tensor)

    grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
    kernel[grid](
        q,
        k,
        v,
        state_tensor,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        output_tensor,
        new_state_tensor,
        scale,
        K=HEAD_DIM,
        V_DIM=HEAD_DIM,
        NUM_V_HEADS=NUM_V_HEADS,
        NUM_K_HEADS=NUM_K_HEADS,
        GVA_RATIO=GVA_RATIO,
        BV=bv,
        N_V_TILES=n_v_tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output_tensor, new_state_tensor


@triton.jit
def gdn_prefill_kernel_v1(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    cu_seqlens_ptr,
    out_ptr,
    new_state_ptr,
    scale,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BV: tl.constexpr,
    N_V_TILES: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES
    qk_head = pid_h // GVA_RATIO

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, BV)
    v_start = pid_v * BV

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    state_tile = tl.load(s_ptrs).to(tl.float32)

    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    if seq_end <= seq_start:
        tl.store(ns_ptrs, state_tile)
        return
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)

    for token_idx in tl.range(seq_start, seq_end):
        gate_index = token_idx * NUM_V_HEADS + pid_h
        x = tl.load(a_ptr + gate_index).to(tl.float32) + dt_bias_val
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)
        beta = tl.sigmoid(tl.load(b_ptr + gate_index).to(tl.float32))

        qk_base = token_idx * NUM_K_HEADS * K + qk_head * K
        b_q = tl.load(q_ptr + qk_base + o_k).to(tl.float32)
        b_k = tl.load(k_ptr + qk_base + o_k).to(tl.float32)

        old_state = g * state_tile
        old_v = tl.sum(old_state * b_k[None, :], axis=1)

        v_base = token_idx * NUM_V_HEADS * V_DIM + pid_h * V_DIM + v_start
        b_v = tl.load(v_ptr + v_base + o_v).to(tl.float32)
        delta_v = beta * (b_v - old_v)

        old_o = tl.sum(old_state * b_q[None, :], axis=1)
        kq = tl.sum(b_k * b_q)
        b_o = scale * (old_o + delta_v * kq)

        out_base = out_ptr + token_idx * NUM_V_HEADS * V_DIM + pid_h * V_DIM + v_start
        tl.store(out_base + o_v, b_o.to(tl.bfloat16))

        state_tile = old_state + delta_v[:, None] * b_k[None, :]

    tl.store(ns_ptrs, state_tile)


@triton.jit
def _prefill_update_tile_pair_v1(
    state0, state1, b_q, b_k, b_v0, b_v1, g, beta, scale,
):
    """Fused update for two state half-tiles, computing kq once."""
    old_state0 = g * state0
    old_state1 = g * state1
    old_v0 = tl.sum(old_state0 * b_k[None, :], axis=1)
    old_v1 = tl.sum(old_state1 * b_k[None, :], axis=1)
    delta_v0 = beta * (b_v0 - old_v0)
    delta_v1 = beta * (b_v1 - old_v1)
    kq = tl.sum(b_k * b_q)
    old_o0 = tl.sum(old_state0 * b_q[None, :], axis=1)
    old_o1 = tl.sum(old_state1 * b_q[None, :], axis=1)
    out0 = scale * (old_o0 + delta_v0 * kq)
    out1 = scale * (old_o1 + delta_v1 * kq)
    new_state0 = old_state0 + delta_v0[:, None] * b_k[None, :]
    new_state1 = old_state1 + delta_v1[:, None] * b_k[None, :]
    return out0, out1, new_state0, new_state1


@triton.jit
def _load_long_token_v1(
    token_idx,
    seq_end,
    pid_h,
    qk_head,
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_val,
    dt_bias_val,
    o_k,
    pair_rows,
    v_start,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    PAIR_ROWS: tl.constexpr,
):
    active = token_idx < seq_end
    gate_index = token_idx * NUM_V_HEADS + pid_h
    a_val = tl.load(a_ptr + gate_index, mask=active, other=0.0).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.where(active, tl.exp(-tl.exp(A_log_val) * softplus_x), 0.0)
    beta = tl.sigmoid(tl.load(b_ptr + gate_index, mask=active, other=0.0).to(tl.float32))

    mask_k = active & (o_k < K)
    qk_base = token_idx * NUM_K_HEADS * K + qk_head * K
    b_q = tl.load(q_ptr + qk_base + o_k, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(k_ptr + qk_base + o_k, mask=mask_k, other=0.0).to(tl.float32)

    mask_pair = active & (pair_rows < PAIR_ROWS)
    v_base = token_idx * NUM_V_HEADS * V_DIM + pid_h * V_DIM + v_start
    b_v0 = tl.load(v_ptr + v_base + pair_rows, mask=mask_pair, other=0.0).to(tl.float32)
    b_v1 = tl.load(
        v_ptr + v_base + pair_rows + PAIR_ROWS,
        mask=mask_pair,
        other=0.0,
    ).to(tl.float32)
    return g, beta, b_q, b_k, b_v0, b_v1


@triton.jit
def gdn_prefill_kernel_v1_long(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    cu_seqlens_ptr,
    out_ptr,
    new_state_ptr,
    scale,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BV: tl.constexpr,
    N_V_TILES: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES
    qk_head = pid_h // GVA_RATIO
    pair_rows = tl.arange(0, BV // 2)

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)
    o_k = tl.arange(0, K)
    v_start = pid_v * BV

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    s_ptrs0 = s_base + pair_rows[:, None] * K + o_k[None, :]
    s_ptrs1 = s_base + (pair_rows + (BV // 2))[:, None] * K + o_k[None, :]
    state_tile0 = tl.load(s_ptrs0).to(tl.float32)
    state_tile1 = tl.load(s_ptrs1).to(tl.float32)

    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    ns_ptrs0 = ns_base + pair_rows[:, None] * K + o_k[None, :]
    ns_ptrs1 = ns_base + (pair_rows + (BV // 2))[:, None] * K + o_k[None, :]
    if seq_end <= seq_start:
        tl.store(ns_ptrs0, state_tile0)
        tl.store(ns_ptrs1, state_tile1)
        return
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)

    curr_g, curr_beta, curr_q, curr_k, curr_v0, curr_v1 = _load_long_token_v1(
        seq_start,
        seq_end,
        pid_h,
        qk_head,
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        A_log_val,
        dt_bias_val,
        o_k,
        pair_rows,
        v_start,
        K=K,
        V_DIM=V_DIM,
        NUM_V_HEADS=NUM_V_HEADS,
        NUM_K_HEADS=NUM_K_HEADS,
        PAIR_ROWS=BV // 2,
    )

    for token_idx in tl.range(seq_start, seq_end):
        next_token = token_idx + 1
        next_g, next_beta, next_q, next_k, next_v0, next_v1 = _load_long_token_v1(
            next_token,
            seq_end,
            pid_h,
            qk_head,
            q_ptr,
            k_ptr,
            v_ptr,
            a_ptr,
            b_ptr,
            A_log_val,
            dt_bias_val,
            o_k,
            pair_rows,
            v_start,
            K=K,
            V_DIM=V_DIM,
            NUM_V_HEADS=NUM_V_HEADS,
            NUM_K_HEADS=NUM_K_HEADS,
            PAIR_ROWS=BV // 2,
        )

        out0, out1, state_tile0, state_tile1 = _prefill_update_tile_pair_v1(
            state_tile0,
            state_tile1,
            curr_q,
            curr_k,
            curr_v0,
            curr_v1,
            curr_g,
            curr_beta,
            scale,
        )

        out_base = out_ptr + token_idx * NUM_V_HEADS * V_DIM + pid_h * V_DIM + v_start
        tl.store(out_base + pair_rows, out0.to(tl.bfloat16))
        tl.store(out_base + pair_rows + (BV // 2), out1.to(tl.bfloat16))

        curr_g = next_g
        curr_beta = next_beta
        curr_q = next_q
        curr_k = next_k
        curr_v0 = next_v0
        curr_v1 = next_v1

    tl.store(ns_ptrs0, state_tile0)
    tl.store(ns_ptrs1, state_tile1)
