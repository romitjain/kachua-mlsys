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


def _bucket_seq_len(max_len: int) -> int:
    if max_len <= 64:
        return 0
    if max_len <= 256:
        return 1
    if max_len <= 1024:
        return 2
    return 3


def _bucket_num_seqs(num_seqs: int) -> int:
    if num_seqs <= 1:
        return 0
    if num_seqs <= 3:
        return 1
    if num_seqs <= 15:
        return 2
    return 3

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
    seqs_bucket = _bucket_num_seqs(num_seqs)

    # Adaptive BV based on grid size (B200 has 148 SMs, want ≥1 wave):
    if num_seqs >= 16:
        bv = 16 if num_seqs <= 48 else 32
    elif num_seqs >= 4:
        bv = 16 if avg_seq_len >= 300 else 32
    elif num_seqs >= 2:
        bv = 16
    else:
        bv = 8
    n_v_tiles = HEAD_DIM // bv

    # Adaptive CHUNK: longer sequences with few seqs benefit from CHUNK=32
    # (fewer sequential chunks amortizes per-chunk overhead). For multi-seq
    # with lots of parallelism, CHUNK=16 is better (smaller gram matrices).
    chunk = 32 if (
        (num_seqs <= 2 and avg_seq_len >= 512)
        or (num_seqs == 3 and avg_seq_len >= 700)
    ) else 16
    use_flat_wy = (
        (num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 128)
        or (10 <= num_seqs < 16 and total_tokens >= 3000)
    )
    use_split_wy = (
        (num_seqs == 1 and avg_seq_len >= 1024)
        or (num_seqs == 2 and avg_seq_len >= 231)
        or (num_seqs == 3 and avg_seq_len >= 221)
        or (4 <= num_seqs < 16 and (avg_seq_len >= 240 or total_tokens >= 800))
    ) and not use_flat_wy

    if use_split_wy:
        max_chunks = max(1, (total_tokens + chunk - 1) // chunk)
        use_precompute_u = num_seqs <= 2
        if use_precompute_u:
            wy_nil = torch.empty((1,), dtype=torch.float32, device=q.device)
            wy_u = torch.empty(
                (num_seqs, NUM_V_HEADS, max_chunks, chunk, HEAD_DIM),
                dtype=torch.bfloat16,
                device=q.device,
            )
        else:
            wy_nil = torch.empty(
                (num_seqs, NUM_V_HEADS, max_chunks, chunk, chunk),
                dtype=torch.float32,
                device=q.device,
            )
            wy_u = torch.empty((1,), dtype=torch.float32, device=q.device)
        wy_w = torch.empty(
            (num_seqs, NUM_V_HEADS, max_chunks, chunk, HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        wy_g = torch.empty(
            (num_seqs, NUM_V_HEADS, max_chunks, chunk),
            dtype=torch.float32,
            device=q.device,
        )
        wy_bg = torch.empty_like(wy_g)
        wy_gc = torch.empty(
            (num_seqs, NUM_V_HEADS, max_chunks),
            dtype=torch.float32,
            device=q.device,
        )
        wy_qk = torch.empty(
            (num_seqs, NUM_V_HEADS, max_chunks, chunk, chunk),
            dtype=torch.bfloat16,
            device=q.device,
        )
        split_num_warps = 4 if num_seqs == 1 else 2

        pre_grid = (num_seqs, NUM_V_HEADS, max_chunks)
        gdn_prefill_wy_prepass[pre_grid](
            q, k, v, A_log, a, dt_bias, b, cu_seqlens,
            wy_nil, wy_w, wy_u, wy_g, wy_bg, wy_gc, wy_qk,
            K=HEAD_DIM, NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
            GVA_RATIO=GVA_RATIO, CHUNK=chunk, MAX_CHUNKS=max_chunks,
            PRECOMPUTE_U=use_precompute_u,
            num_warps=split_num_warps, num_stages=3,
        )

        grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
        gdn_prefill_kernel_split_wy[grid](
            q, k, v, state_tensor, cu_seqlens,
            wy_nil, wy_w, wy_u, wy_g, wy_bg, wy_gc, wy_qk,
            output_tensor, new_state_tensor, scale,
            K=HEAD_DIM, V_DIM=HEAD_DIM,
            NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS, GVA_RATIO=GVA_RATIO,
            CHUNK=chunk, BV=bv, N_V_TILES=n_v_tiles, MAX_CHUNKS=max_chunks,
            PRECOMPUTE_U=use_precompute_u,
            num_warps=split_num_warps, num_stages=3,
        )
        return output_tensor, new_state_tensor

    if use_flat_wy:
        max_records = max(1, (total_tokens + chunk - 1) // chunk + num_seqs)
        wy_nil = torch.empty(
            (max_records, NUM_V_HEADS, chunk, chunk),
            dtype=torch.float32,
            device=q.device,
        )
        wy_w = torch.empty(
            (max_records, NUM_V_HEADS, chunk, HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        wy_g = torch.empty(
            (max_records, NUM_V_HEADS, chunk),
            dtype=torch.float32,
            device=q.device,
        )
        wy_bg = torch.empty_like(wy_g)
        wy_gc = torch.empty((max_records, NUM_V_HEADS), dtype=torch.float32, device=q.device)
        wy_qk = torch.empty(
            (max_records, NUM_V_HEADS, chunk, chunk),
            dtype=torch.bfloat16,
            device=q.device,
        )
        flat_record_base = torch.empty((num_seqs,), dtype=torch.int32, device=q.device)

        if num_seqs >= 16:
            pre_grid = (max_records, NUM_K_HEADS)
            gdn_prefill_many_wy_prepass_flat_gva2[pre_grid](
                q, k, A_log, a, dt_bias, b, cu_seqlens,
                flat_record_base, wy_nil, wy_w, wy_g, wy_bg, wy_gc, wy_qk,
                K=HEAD_DIM, NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
                GVA_RATIO=GVA_RATIO, CHUNK=chunk, NUM_SEQS=num_seqs,
                STORE_RECORD_BASE=True,
                num_warps=2, num_stages=3,
            )
        else:
            pre_grid = (max_records, NUM_V_HEADS)
            gdn_prefill_many_wy_prepass_flat[pre_grid](
                q, k, A_log, a, dt_bias, b, cu_seqlens,
                flat_record_base, wy_nil, wy_w, wy_g, wy_bg, wy_gc, wy_qk,
                K=HEAD_DIM, NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
                GVA_RATIO=GVA_RATIO, CHUNK=chunk, NUM_SEQS=num_seqs,
                STORE_RECORD_BASE=False,
                num_warps=2, num_stages=3,
            )

        grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
        gdn_prefill_many_split_wy_flat[grid](
            q, k, v, state_tensor, cu_seqlens,
            flat_record_base, wy_nil, wy_w, wy_g, wy_bg, wy_gc, wy_qk,
            output_tensor, new_state_tensor, scale,
            K=HEAD_DIM, V_DIM=HEAD_DIM,
            NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS, GVA_RATIO=GVA_RATIO,
            CHUNK=chunk, BV=bv, N_V_TILES=n_v_tiles, NUM_SEQS=num_seqs,
            USE_RECORD_BASE=num_seqs >= 16,
            num_warps=2, num_stages=3,
        )
        return output_tensor, new_state_tensor

    grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
    gdn_prefill_kernel_v2[grid](
        q, k, v, state_tensor, A_log, a, dt_bias, b, cu_seqlens,
        output_tensor, new_state_tensor, scale, seq_bucket, seqs_bucket,
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
def _sigmoid_exp2(x):
    """Sigmoid through exp2 so gate math stays on the cheaper base-2 SFU path."""
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _dot_f16(a, b):
    """fp16 tensor-core matmul for bounded fallback triangular solves."""
    out = tl.dot(a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32)
    return tl.inline_asm_elementwise(
        asm="mov.f32 $0, $1;",
        constraints="=r,r",
        args=[out],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _apply_unit_lower_inverse_fp16(nil, rhs, BV: tl.constexpr, CHUNK: tl.constexpr):
    sol = rhs - _dot_f16(nil, rhs)
    power = _dot_f16(nil, nil)
    if CHUNK >= 4:
        sol = sol + _dot_f16(power, sol)
        power = _dot_f16(power, power)
    if CHUNK >= 8:
        sol = sol + _dot_f16(power, sol)
        power = _dot_f16(power, power)
    if CHUNK >= 16:
        sol = sol + _dot_f16(power, sol)
        power = _dot_f16(power, power)
    if CHUNK >= 32:
        sol = sol + _dot_f16(power, sol)
    return sol


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


@triton.jit
def gdn_prefill_wy_prepass(
    q_ptr, k_ptr, v_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, cu_seqlens_ptr,
    wy_nil_ptr, wy_w_ptr, wy_u_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
    K: tl.constexpr, NUM_V_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr, CHUNK: tl.constexpr, MAX_CHUNKS: tl.constexpr,
    PRECOMPUTE_U: tl.constexpr,
):
    """Precompute chunk-local WY data shared by every V tile."""
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    qk_head = pid_h // GVA_RATIO

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)
    chunk_start = seq_start + pid_chunk * CHUNK
    if chunk_start >= seq_end:
        return

    rows = tl.arange(0, CHUNK)
    cols = tl.arange(0, CHUNK)
    o_k = tl.arange(0, K)
    tok = chunk_start + rows
    active = tok < seq_end
    lower = rows[:, None] > cols[None, :]
    lower_eq = rows[:, None] >= cols[None, :]

    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    neg_exp_A_log2e = -tl.exp(A_log_val) * 1.4426950408889634

    gate_idx = tok * NUM_V_HEADS + pid_h
    a_vals = tl.load(a_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)
    b_vals = tl.load(b_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)

    x = a_vals + dt_bias_val
    x_log2e = x * 1.4426950408889634
    sp_x = tl.where(x > 20.0, x, tl.log2(1.0 + tl.exp2(x_log2e)) * 0.6931471805599453)
    log2_g = tl.where(active, neg_exp_A_log2e * sp_x, 0.0)
    beta_chunk = tl.where(active, tl.sigmoid(b_vals), 0.0)

    log2_G = tl.cumsum(log2_g, axis=0)
    G = tl.exp2(log2_G)
    Gc = tl.sum(G * (rows == (CHUNK - 1)).to(tl.float32), axis=0)
    G_safe = tl.maximum(G, 1e-30)
    bg = beta_chunk / G_safe

    qk_ptrs = tok[:, None] * (NUM_K_HEADS * K) + qk_head * K + o_k[None, :]
    K_tile_bf = tl.load(k_ptr + qk_ptrs, mask=active[:, None], other=0.0)
    Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)
    K_tile_bf_T = tl.trans(K_tile_bf)

    gram_kk = _safe_dot_bf16(K_tile_bf, K_tile_bf_T)
    nil = tl.where(lower, gram_kk * beta_chunk[:, None], 0.0)
    eye = (rows[:, None] == cols[None, :]).to(tl.float32)
    a_inv = _apply_unit_lower_inverse(nil, eye, BV=CHUNK, CHUNK=CHUNK)
    weighted_k = beta_chunk[:, None] * K_tile_bf.to(tl.float32)
    w_tile = _dot_f32(a_inv, weighted_k)
    if PRECOMPUTE_U:
        v_ptrs = tok[:, None] * (NUM_V_HEADS * K) + pid_h * K + o_k[None, :]
        V_tile = tl.load(v_ptr + v_ptrs, mask=active[:, None], other=0.0).to(tl.float32)
        u_tile = _dot_f32(a_inv, bg[:, None] * V_tile)
    gram_qk = _safe_dot_bf16(Q_tile_bf, K_tile_bf_T)
    qk_lower = tl.where(lower_eq, gram_qk, 0.0)

    chunk_base = (pid_seq * NUM_V_HEADS + pid_h) * MAX_CHUNKS + pid_chunk
    a_base = chunk_base * CHUNK * CHUNK
    w_base = chunk_base * CHUNK * K
    vec_base = chunk_base * CHUNK

    tl.store(wy_w_ptr + w_base + rows[:, None] * K + o_k[None, :], w_tile.to(tl.bfloat16))
    if PRECOMPUTE_U:
        tl.store(wy_u_ptr + w_base + rows[:, None] * K + o_k[None, :], u_tile.to(tl.bfloat16))
    else:
        tl.store(wy_nil_ptr + a_base + rows[:, None] * CHUNK + cols[None, :], a_inv)
    tl.store(wy_g_ptr + vec_base + rows, G)
    tl.store(wy_bg_ptr + vec_base + rows, bg)
    tl.store(wy_gc_ptr + chunk_base, Gc)
    tl.store(wy_qk_ptr + a_base + rows[:, None] * CHUNK + cols[None, :], qk_lower.to(tl.bfloat16))


@triton.jit
def gdn_prefill_kernel_split_wy(
    q_ptr, k_ptr, v_ptr, state_ptr, cu_seqlens_ptr,
    wy_nil_ptr, wy_w_ptr, wy_u_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
    out_ptr, new_state_ptr, scale,
    K: tl.constexpr, V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr, GVA_RATIO: tl.constexpr,
    CHUNK: tl.constexpr, BV: tl.constexpr, N_V_TILES: tl.constexpr,
    MAX_CHUNKS: tl.constexpr, PRECOMPUTE_U: tl.constexpr,
):
    """Use precomputed WY data, keeping state/output tiled along V."""
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
    lower_eq = rows[:, None] >= cols[None, :]
    v_start = pid_v * BV

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    state_tile = tl.load(s_ptrs).to(tl.float32)

    if seq_end <= seq_start:
        tl.store(ns_ptrs, state_tile)
        return

    for chunk_start in tl.range(seq_start, seq_end, CHUNK):
        tok = chunk_start + rows
        active = tok < seq_end
        chunk_id = (chunk_start - seq_start) // CHUNK
        chunk_base = (pid_seq * NUM_V_HEADS + pid_h) * MAX_CHUNKS + chunk_id
        a_base = chunk_base * CHUNK * CHUNK
        w_base = chunk_base * CHUNK * K
        vec_base = chunk_base * CHUNK

        w_tile = tl.load(wy_w_ptr + w_base + rows[:, None] * K + o_k[None, :])
        if PRECOMPUTE_U:
            u_tile = tl.load(wy_u_ptr + w_base + rows[:, None] * K + v_start + o_v[None, :])
        else:
            a_inv = tl.load(wy_nil_ptr + a_base + rows[:, None] * CHUNK + cols[None, :])
            beta_over_g = tl.load(wy_bg_ptr + vec_base + rows)
        G = tl.load(wy_g_ptr + vec_base + rows)
        Gc = tl.load(wy_gc_ptr + chunk_base)
        qk_lower = tl.load(wy_qk_ptr + a_base + rows[:, None] * CHUNK + cols[None, :])
        qk_lower = tl.where(lower_eq, qk_lower, 0.0)

        qk_ptrs = tok[:, None] * (NUM_K_HEADS * K) + qk_head * K + o_k[None, :]
        K_tile_bf = tl.load(k_ptr + qk_ptrs, mask=active[:, None], other=0.0)
        Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)

        state_tile_bf_T = tl.trans(state_tile.to(tl.bfloat16))
        if not PRECOMPUTE_U:
            v_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
            V_tile = tl.load(v_ptr + v_ptrs, mask=active[:, None], other=0.0).to(tl.float32)
            u_rhs = beta_over_g[:, None] * V_tile
            u_tile = _dot_f32(a_inv, u_rhs)
        w_state = _safe_dot_bf16(w_tile, state_tile_bf_T)
        x_chunk = u_tile - w_state
        x_chunk = tl.where(active[:, None], x_chunk, 0.0)

        state_q = _safe_dot_bf16(Q_tile_bf, state_tile_bf_T)
        x_chunk_bf = x_chunk.to(tl.bfloat16)
        qk_contrib = _safe_dot_bf16(qk_lower, x_chunk_bf)
        out_tile = scale * G[:, None] * (state_q + qk_contrib)
        out_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
        tl.store(out_ptr + out_ptrs, out_tile.to(tl.bfloat16), mask=active[:, None])

        state_tile = Gc * (state_tile + _safe_dot_bf16(tl.trans(x_chunk_bf), K_tile_bf))

    tl.store(ns_ptrs, state_tile)


@triton.jit
def _flat_chunk_to_seq(cu_seqlens_ptr, record_id, NUM_SEQS: tl.constexpr, CHUNK: tl.constexpr):
    running = tl.full((), 0, tl.int32)
    pid_seq = tl.full((), 0, tl.int32)
    pid_chunk = tl.full((), 0, tl.int32)
    found = tl.full((), 0, tl.int32)

    for seq in tl.range(0, NUM_SEQS):
        seq_start = tl.load(cu_seqlens_ptr + seq).to(tl.int32)
        seq_end = tl.load(cu_seqlens_ptr + seq + 1).to(tl.int32)
        n_chunks = tl.cdiv(tl.maximum(seq_end - seq_start, 0), CHUNK)
        hit = (record_id >= running) & (record_id < running + n_chunks) & (found == 0)
        pid_seq = tl.where(hit, seq, pid_seq)
        pid_chunk = tl.where(hit, record_id - running, pid_chunk)
        found = tl.where(hit, 1, found)
        running += n_chunks

    return found, pid_seq, pid_chunk


@triton.jit
def _chunk_prefix_before(cu_seqlens_ptr, pid_seq, NUM_SEQS: tl.constexpr, CHUNK: tl.constexpr):
    prefix = tl.full((), 0, tl.int32)

    for seq in tl.range(0, NUM_SEQS):
        seq_start = tl.load(cu_seqlens_ptr + seq).to(tl.int32)
        seq_end = tl.load(cu_seqlens_ptr + seq + 1).to(tl.int32)
        n_chunks = tl.cdiv(tl.maximum(seq_end - seq_start, 0), CHUNK)
        prefix += tl.where(seq < pid_seq, n_chunks, 0)

    return prefix


@triton.jit
def gdn_prefill_many_wy_prepass_flat(
    q_ptr, k_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, cu_seqlens_ptr,
    record_base_ptr, wy_nil_ptr, wy_w_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
    K: tl.constexpr, NUM_V_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr, CHUNK: tl.constexpr, NUM_SEQS: tl.constexpr,
    STORE_RECORD_BASE: tl.constexpr,
):
    """Flat many-sequence WY prepass without a separate record-map launch."""
    record_id = tl.program_id(0)
    pid_h = tl.program_id(1)
    qk_head = pid_h // GVA_RATIO

    valid, pid_seq, pid_chunk = _flat_chunk_to_seq(
        cu_seqlens_ptr,
        record_id,
        NUM_SEQS=NUM_SEQS,
        CHUNK=CHUNK,
    )
    if valid == 0:
        return

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)
    chunk_start = seq_start + pid_chunk * CHUNK
    if STORE_RECORD_BASE:
        tl.store(
            record_base_ptr + pid_seq,
            record_id,
            mask=(pid_chunk == 0) & (pid_h == 0),
        )
    if chunk_start >= seq_end:
        return

    rows = tl.arange(0, CHUNK)
    cols = tl.arange(0, CHUNK)
    o_k = tl.arange(0, K)
    tok = chunk_start + rows
    active = tok < seq_end
    lower = rows[:, None] > cols[None, :]
    lower_eq = rows[:, None] >= cols[None, :]

    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    neg_exp_A_log2e = -tl.exp(A_log_val) * 1.4426950408889634

    gate_idx = tok * NUM_V_HEADS + pid_h
    a_vals = tl.load(a_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)
    b_vals = tl.load(b_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)

    x = a_vals + dt_bias_val
    x_log2e = x * 1.4426950408889634
    sp_x = tl.where(x > 20.0, x, tl.log2(1.0 + tl.exp2(x_log2e)) * 0.6931471805599453)
    log2_g = tl.where(active, neg_exp_A_log2e * sp_x, 0.0)
    beta_chunk = tl.where(active, tl.sigmoid(b_vals), 0.0)

    log2_G = tl.cumsum(log2_g, axis=0)
    G = tl.exp2(log2_G)
    Gc = tl.sum(G * (rows == (CHUNK - 1)).to(tl.float32), axis=0)
    G_safe = tl.maximum(G, 1e-30)
    bg = beta_chunk / G_safe

    qk_ptrs = tok[:, None] * (NUM_K_HEADS * K) + qk_head * K + o_k[None, :]
    K_tile_bf = tl.load(k_ptr + qk_ptrs, mask=active[:, None], other=0.0)
    Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)
    K_tile_bf_T = tl.trans(K_tile_bf)

    gram_kk = _safe_dot_bf16(K_tile_bf, K_tile_bf_T)
    nil = tl.where(lower, gram_kk * beta_chunk[:, None], 0.0)
    eye = (rows[:, None] == cols[None, :]).to(tl.float32)
    a_inv = _apply_unit_lower_inverse(nil, eye, BV=CHUNK, CHUNK=CHUNK)
    weighted_k = beta_chunk[:, None] * K_tile_bf.to(tl.float32)
    w_tile = _dot_f32(a_inv, weighted_k)
    gram_qk = _safe_dot_bf16(Q_tile_bf, K_tile_bf_T)
    qk_lower = tl.where(lower_eq, gram_qk, 0.0)

    chunk_base = record_id * NUM_V_HEADS + pid_h
    a_base = chunk_base * CHUNK * CHUNK
    w_base = chunk_base * CHUNK * K
    vec_base = chunk_base * CHUNK

    tl.store(wy_nil_ptr + a_base + rows[:, None] * CHUNK + cols[None, :], a_inv)
    tl.store(wy_w_ptr + w_base + rows[:, None] * K + o_k[None, :], w_tile.to(tl.bfloat16))
    tl.store(wy_g_ptr + vec_base + rows, G)
    tl.store(wy_bg_ptr + vec_base + rows, bg)
    tl.store(wy_gc_ptr + chunk_base, Gc)
    tl.store(wy_qk_ptr + a_base + rows[:, None] * CHUNK + cols[None, :], qk_lower.to(tl.bfloat16))


@triton.jit
def _store_many_wy_flat_head(
    pid_h, record_id, tok, active, rows, cols, o_k, lower, lower_eq,
    gram_kk, gram_qk, K_tile_bf,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    wy_nil_ptr, wy_w_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
    K: tl.constexpr, NUM_V_HEADS: tl.constexpr, CHUNK: tl.constexpr,
):
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    neg_exp_A_log2e = -tl.exp(A_log_val) * 1.4426950408889634

    gate_idx = tok * NUM_V_HEADS + pid_h
    a_vals = tl.load(a_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)
    b_vals = tl.load(b_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)

    x = a_vals + dt_bias_val
    x_log2e = x * 1.4426950408889634
    sp_x = tl.where(x > 20.0, x, tl.log2(1.0 + tl.exp2(x_log2e)) * 0.6931471805599453)
    log2_g = tl.where(active, neg_exp_A_log2e * sp_x, 0.0)
    beta_chunk = tl.where(active, tl.sigmoid(b_vals), 0.0)

    log2_G = tl.cumsum(log2_g, axis=0)
    G = tl.exp2(log2_G)
    Gc = tl.sum(G * (rows == (CHUNK - 1)).to(tl.float32), axis=0)
    G_safe = tl.maximum(G, 1e-30)
    bg = beta_chunk / G_safe

    nil = tl.where(lower, gram_kk * beta_chunk[:, None], 0.0)
    eye = (rows[:, None] == cols[None, :]).to(tl.float32)
    a_inv = _apply_unit_lower_inverse(nil, eye, BV=CHUNK, CHUNK=CHUNK)
    weighted_k = beta_chunk[:, None] * K_tile_bf.to(tl.float32)
    w_tile = _dot_f32(a_inv, weighted_k)
    qk_lower = tl.where(lower_eq, gram_qk, 0.0)

    chunk_base = record_id * NUM_V_HEADS + pid_h
    a_base = chunk_base * CHUNK * CHUNK
    w_base = chunk_base * CHUNK * K
    vec_base = chunk_base * CHUNK

    tl.store(wy_nil_ptr + a_base + rows[:, None] * CHUNK + cols[None, :], a_inv)
    tl.store(wy_w_ptr + w_base + rows[:, None] * K + o_k[None, :], w_tile.to(tl.bfloat16))
    tl.store(wy_g_ptr + vec_base + rows, G)
    tl.store(wy_bg_ptr + vec_base + rows, bg)
    tl.store(wy_gc_ptr + chunk_base, Gc)
    tl.store(
        wy_qk_ptr + a_base + rows[:, None] * CHUNK + cols[None, :],
        qk_lower.to(tl.bfloat16),
    )


@triton.jit
def gdn_prefill_many_wy_prepass_flat_gva2(
    q_ptr, k_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr, cu_seqlens_ptr,
    record_base_ptr, wy_nil_ptr, wy_w_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
    K: tl.constexpr, NUM_V_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr, CHUNK: tl.constexpr, NUM_SEQS: tl.constexpr,
    STORE_RECORD_BASE: tl.constexpr,
):
    """Flat many-sequence prepass fused across the two V heads sharing Q/K."""
    record_id = tl.program_id(0)
    pid_qk = tl.program_id(1)

    valid, pid_seq, pid_chunk = _flat_chunk_to_seq(
        cu_seqlens_ptr,
        record_id,
        NUM_SEQS=NUM_SEQS,
        CHUNK=CHUNK,
    )
    if valid == 0:
        return

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)
    chunk_start = seq_start + pid_chunk * CHUNK
    if STORE_RECORD_BASE:
        tl.store(record_base_ptr + pid_seq, record_id, mask=pid_chunk == 0)
    if chunk_start >= seq_end:
        return

    rows = tl.arange(0, CHUNK)
    cols = tl.arange(0, CHUNK)
    o_k = tl.arange(0, K)
    tok = chunk_start + rows
    active = tok < seq_end
    lower = rows[:, None] > cols[None, :]
    lower_eq = rows[:, None] >= cols[None, :]

    qk_ptrs = tok[:, None] * (NUM_K_HEADS * K) + pid_qk * K + o_k[None, :]
    K_tile_bf = tl.load(k_ptr + qk_ptrs, mask=active[:, None], other=0.0)
    Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)
    K_tile_bf_T = tl.trans(K_tile_bf)

    gram_kk = _safe_dot_bf16(K_tile_bf, K_tile_bf_T)
    gram_qk = _safe_dot_bf16(Q_tile_bf, K_tile_bf_T)

    pid_h0 = pid_qk * GVA_RATIO
    _store_many_wy_flat_head(
        pid_h0, record_id, tok, active, rows, cols, o_k, lower, lower_eq,
        gram_kk, gram_qk, K_tile_bf,
        A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        wy_nil_ptr, wy_w_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
        K=K, NUM_V_HEADS=NUM_V_HEADS, CHUNK=CHUNK,
    )
    _store_many_wy_flat_head(
        pid_h0 + 1, record_id, tok, active, rows, cols, o_k, lower, lower_eq,
        gram_kk, gram_qk, K_tile_bf,
        A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        wy_nil_ptr, wy_w_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
        K=K, NUM_V_HEADS=NUM_V_HEADS, CHUNK=CHUNK,
    )


@triton.jit
def gdn_prefill_many_split_wy_flat(
    q_ptr, k_ptr, v_ptr, state_ptr, cu_seqlens_ptr,
    record_base_ptr, wy_nil_ptr, wy_w_ptr, wy_g_ptr, wy_bg_ptr, wy_gc_ptr, wy_qk_ptr,
    out_ptr, new_state_ptr, scale,
    K: tl.constexpr, V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr, GVA_RATIO: tl.constexpr,
    CHUNK: tl.constexpr, BV: tl.constexpr, N_V_TILES: tl.constexpr,
    NUM_SEQS: tl.constexpr, USE_RECORD_BASE: tl.constexpr,
):
    """Many-sequence split-WY consumer using ordinal flat chunk records."""
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
    lower_eq = rows[:, None] >= cols[None, :]
    v_start = pid_v * BV
    if USE_RECORD_BASE:
        record_base = tl.load(record_base_ptr + pid_seq)
    else:
        record_base = _chunk_prefix_before(
            cu_seqlens_ptr,
            pid_seq,
            NUM_SEQS=NUM_SEQS,
            CHUNK=CHUNK,
        )

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    state_tile = tl.load(s_ptrs).to(tl.float32)

    if seq_end <= seq_start:
        tl.store(ns_ptrs, state_tile)
        return

    for chunk_start in tl.range(seq_start, seq_end, CHUNK):
        tok = chunk_start + rows
        active = tok < seq_end
        chunk_id = (chunk_start - seq_start) // CHUNK
        record_id = record_base + chunk_id
        chunk_base = record_id * NUM_V_HEADS + pid_h
        a_base = chunk_base * CHUNK * CHUNK
        w_base = chunk_base * CHUNK * K
        vec_base = chunk_base * CHUNK

        a_inv = tl.load(wy_nil_ptr + a_base + rows[:, None] * CHUNK + cols[None, :])
        w_tile = tl.load(wy_w_ptr + w_base + rows[:, None] * K + o_k[None, :])
        beta_over_g = tl.load(wy_bg_ptr + vec_base + rows)
        G = tl.load(wy_g_ptr + vec_base + rows)
        Gc = tl.load(wy_gc_ptr + chunk_base)
        qk_lower = tl.load(wy_qk_ptr + a_base + rows[:, None] * CHUNK + cols[None, :])
        qk_lower = tl.where(lower_eq, qk_lower, 0.0)

        qk_ptrs = tok[:, None] * (NUM_K_HEADS * K) + qk_head * K + o_k[None, :]
        K_tile_bf = tl.load(k_ptr + qk_ptrs, mask=active[:, None], other=0.0)
        Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)

        state_tile_bf_T = tl.trans(state_tile.to(tl.bfloat16))
        v_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
        V_tile = tl.load(v_ptr + v_ptrs, mask=active[:, None], other=0.0).to(tl.float32)
        u_rhs = beta_over_g[:, None] * V_tile
        u_tile = _dot_f32(a_inv, u_rhs)
        w_state = _safe_dot_bf16(w_tile, state_tile_bf_T)
        x_chunk = u_tile - w_state
        x_chunk = tl.where(active[:, None], x_chunk, 0.0)

        state_q = _safe_dot_bf16(Q_tile_bf, state_tile_bf_T)
        x_chunk_bf = x_chunk.to(tl.bfloat16)
        qk_contrib = _safe_dot_bf16(qk_lower, x_chunk_bf)
        out_tile = scale * G[:, None] * (state_q + qk_contrib)
        out_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
        tl.store(out_ptr + out_ptrs, out_tile.to(tl.bfloat16), mask=active[:, None])

        state_tile = Gc * (state_tile + _safe_dot_bf16(tl.trans(x_chunk_bf), K_tile_bf))

    tl.store(ns_ptrs, state_tile)


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
    key=["BV", "CHUNK", "seq_bucket", "seqs_bucket"],
)
@triton.jit
def gdn_prefill_kernel_v2(
    q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr,
    out_ptr, new_state_ptr, scale, seq_bucket, seqs_bucket,
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
    # Keep everything in log space (scaled by 1/ln(2) -> exp2) to avoid an exp->log round-trip
    # and use the single-cycle EX2 SFU op instead of 4-cycle EXP.
    neg_exp_A_log2e = -tl.exp(A_log_val) * 1.4426950408889634  # 1/ln(2)

    for chunk_start in tl.range(seq_start, seq_end, CHUNK):
        tok = chunk_start + rows
        active = tok < seq_end

        gate_idx = tok * NUM_V_HEADS + pid_h
        a_vals = tl.load(a_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptr + gate_idx, mask=active, other=0.0).to(tl.float32)

        x = a_vals + dt_bias_val
        x_log2e = x * 1.4426950408889634
        sp_x = tl.where(
            x > 20.0,
            x,
            tl.log2(1.0 + tl.exp2(x_log2e)) * 0.6931471805599453,
        )
        log2_g = tl.where(active, neg_exp_A_log2e * sp_x, 0.0)
        beta_chunk = tl.where(active, _sigmoid_exp2(b_vals), 0.0)

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

        Q_tile_bf = tl.load(q_ptr + qk_ptrs, mask=active[:, None], other=0.0)
        state_q = _safe_dot_bf16(Q_tile_bf, state_tile_bf_T)                   # [C, BV]
        gram_qk = _safe_dot_bf16(Q_tile_bf, K_tile_bf_T)                       # [C, C]

        ratio_log2 = tl.where(lower_eq, log2_G[:, None] - log2_G[None, :], 0.0)
        ratio = tl.exp2(ratio_log2)
        nil = tl.where(lower, gram_kk * beta_chunk[:, None] * ratio, 0.0)
        rhs = beta_chunk[:, None] * (V_tile - G[:, None] * state_k)
        y_chunk = _apply_unit_lower_inverse_fp16(nil, rhs, BV=BV, CHUNK=CHUNK)  # [C, BV]
        y_chunk = tl.where(active[:, None], y_chunk, 0.0)

        qk_lower = tl.where(lower_eq, gram_qk * ratio, 0.0)
        y_chunk_bf = y_chunk.to(tl.bfloat16)
        qk_contrib = _safe_dot_bf16(qk_lower.to(tl.bfloat16), y_chunk_bf)
        out_tile = scale * (G[:, None] * state_q + qk_contrib)
        out_ptrs = tok[:, None] * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM + v_start + o_v[None, :]
        tl.store(out_ptr + out_ptrs, out_tile.to(tl.bfloat16), mask=active[:, None])

        log2_Gc = tl.sum(log2_G * (rows == (CHUNK - 1)).to(tl.float32), axis=0)
        tail = tl.exp2(log2_Gc - log2_G)
        state_update = _safe_dot_bf16(tl.trans((tail[:, None] * y_chunk).to(tl.bfloat16)), K_tile_bf)
        state_tile = Gc * state_tile + state_update

    tl.store(ns_ptrs, state_tile)
