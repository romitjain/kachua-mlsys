"""Triton v1 GDN prefill kernel."""

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

CHUNK_SIZE: int = 32
CHUNK_THRESHOLD: int = 128
CHUNK_BK: int = 64


@triton.jit
def gdn_prefill_chunked_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    cu_seqlens_ptr, out_ptr, new_state_ptr,
    scale,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BV: tl.constexpr,
    N_V_TILES: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    """Fused chunkwise GDN prefill kernel.

    Each CTA handles one (seq, v_head, v_tile) and loops over C-token chunks.
    Within each chunk: Neumann-series triangular solve + matmul-based output.
    """
    pid_seq = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES
    qk_head = pid_h // GVA_RATIO
    v_start = pid_v * BV

    seq_start = tl.load(cu_seqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1).to(tl.int32)

    o_c = tl.arange(0, C)
    o_bv = tl.arange(0, BV)
    o_bk = tl.arange(0, BK)

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    state0 = tl.load(s_base + o_bv[:, None] * K + o_bk[None, :]).to(tl.float32)
    state1 = tl.load(s_base + o_bv[:, None] * K + (o_bk + BK)[None, :]).to(tl.float32)

    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    if seq_end <= seq_start:
        tl.store(ns_base + o_bv[:, None] * K + o_bk[None, :], state0)
        tl.store(ns_base + o_bv[:, None] * K + (o_bk + BK)[None, :], state1)
        return

    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)

    eye_bool = o_c[:, None] == o_c[None, :]
    eye_cc = eye_bool.to(tl.float32)
    causal_bool = o_c[:, None] >= o_c[None, :]
    mask_cc = causal_bool.to(tl.float32)
    strict_lower = causal_bool & ~eye_bool

    for chunk_off in tl.range(seq_start, seq_end, C):
        c_end = tl.minimum(chunk_off + C, seq_end)
        c_mask = o_c < (c_end - chunk_off)

        # --- gates ---
        gate_ptrs = a_ptr + (chunk_off + o_c) * NUM_V_HEADS + pid_h
        a_vals = tl.load(gate_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptr + (chunk_off + o_c) * NUM_V_HEADS + pid_h, mask=c_mask, other=0.0).to(tl.float32)

        x = a_vals + dt_bias_val
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        alpha = tl.where(c_mask, tl.exp(-tl.exp(A_log_val) * sp), 1.0)
        beta = tl.where(c_mask, tl.sigmoid(b_vals), 0.0)

        log_alpha = tl.log(tl.maximum(alpha, 1e-30))
        log_gamma = tl.cumsum(log_alpha, axis=0)
        last_mask = (o_c == C - 1)
        log_gamma_C = tl.sum(tl.where(last_mask, log_gamma, 0.0))
        gamma = tl.exp(log_gamma)
        gamma_C = tl.exp(log_gamma_C)

        # --- L = tril(beta * K @ K^T, -1): accumulate over BK tiles ---
        L = tl.zeros((C, C), dtype=tl.float32)
        for bk_off in tl.static_range(0, K, BK):
            k_base = k_ptr + (chunk_off + o_c[:, None]) * NUM_K_HEADS * K + qk_head * K + (o_bk[None, :] + bk_off)
            k_tile = tl.load(k_base, mask=c_mask[:, None], other=0.0).to(tl.float32)
            L += tl.dot(k_tile, tl.trans(k_tile), input_precision="ieee")
        L = tl.where(strict_lower, beta[:, None] * L, 0.0)

        # --- Forward substitution: solve (I+L)@W = B*K and (I+L)@U_rhs = G^{-1}*B*V ---
        # Jacobi iteration: W = rhs - L@W, converges in C-1 steps for strictly lower L
        delta_bv = tl.zeros((C, BV), dtype=tl.float32)
        o_state_bv = tl.zeros((C, BV), dtype=tl.float32)
        QK = tl.zeros((C, C), dtype=tl.float32)

        # Solve for W (per BK tile) and accumulate Delta, O_state, QK
        for bk_off in tl.static_range(0, K, BK):
            k_base = k_ptr + (chunk_off + o_c[:, None]) * NUM_K_HEADS * K + qk_head * K + (o_bk[None, :] + bk_off)
            k_tile = tl.load(k_base, mask=c_mask[:, None], other=0.0).to(tl.float32)
            q_base = q_ptr + (chunk_off + o_c[:, None]) * NUM_K_HEADS * K + qk_head * K + (o_bk[None, :] + bk_off)
            q_tile = tl.load(q_base, mask=c_mask[:, None], other=0.0).to(tl.float32)
            s_tile = (state0 if bk_off == 0 else state1).to(tl.float32)

            rhs_w = beta[:, None] * k_tile
            W_bk = rhs_w + tl.zeros((C, BK), dtype=tl.float32)
            for _ in tl.static_range(C - 1):
                W_bk = rhs_w - tl.dot(L, W_bk, input_precision="ieee")

            W_bar_bk = gamma[:, None] * W_bk
            delta_bv -= tl.dot(W_bar_bk, tl.trans(s_tile), input_precision="ieee")

            Q_bar_bk = gamma[:, None] * q_tile
            o_state_bv += tl.dot(Q_bar_bk, tl.trans(s_tile), input_precision="ieee")

            QK += tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee")

        # Solve for U_tilde (BV-slice of v) using N^{-1} = G * M^{-1} * G^{-1}
        v_base = v_ptr + (chunk_off + o_c[:, None]) * NUM_V_HEADS * V_DIM + pid_h * V_DIM + (v_start + o_bv[None, :])
        v_tile = tl.load(v_base, mask=c_mask[:, None], other=0.0).to(tl.float32)
        beta_over_gamma = beta * tl.exp(-log_gamma)
        rhs_u = beta_over_gamma[:, None] * v_tile
        u_temp = rhs_u + tl.zeros((C, BV), dtype=tl.float32)
        for _ in tl.static_range(C - 1):
            u_temp = rhs_u - tl.dot(L, u_temp, input_precision="ieee")
        u_tilde_bv = gamma[:, None] * u_temp
        delta_bv += u_tilde_bv

        # --- Output ---
        log_gamma_ratio = log_gamma[:, None] - log_gamma[None, :]
        gamma_ratio = tl.exp(log_gamma_ratio)
        QK_masked = QK * mask_cc * gamma_ratio
        o_intra_bv = tl.dot(QK_masked, delta_bv, input_precision="ieee")
        o_chunk = scale * (o_state_bv + o_intra_bv)

        out_base = out_ptr + (chunk_off + o_c[:, None]) * NUM_V_HEADS * V_DIM + pid_h * V_DIM + (v_start + o_bv[None, :])
        tl.store(out_base, o_chunk.to(tl.bfloat16), mask=c_mask[:, None])

        # --- State update: S = gamma_C * S + Delta^T @ K_fwd ---
        delta_T = tl.trans(delta_bv)
        for bk_off in tl.static_range(0, K, BK):
            k_base = k_ptr + (chunk_off + o_c[:, None]) * NUM_K_HEADS * K + qk_head * K + (o_bk[None, :] + bk_off)
            k_tile = tl.load(k_base, mask=c_mask[:, None], other=0.0).to(tl.float32)
            kfwd_tile = tl.exp(log_gamma_C - log_gamma)[:, None] * k_tile
            update = tl.dot(delta_T, kfwd_tile, input_precision="ieee")
            if bk_off == 0:
                state0 = gamma_C * state0 + update
            else:
                state1 = gamma_C * state1 + update

    tl.store(ns_base + o_bv[:, None] * K + o_bk[None, :], state0)
    tl.store(ns_base + o_bv[:, None] * K + (o_bk + BK)[None, :], state1)


@register_func("flashinfer.kernel")
def launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
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

    avg_seq_len = max(1, (q.shape[0] + num_seqs - 1) // num_seqs)

    if avg_seq_len >= CHUNK_THRESHOLD:
        bv = BALANCED_BV
        n_v_tiles = HEAD_DIM // bv
        grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
        gdn_prefill_chunked_kernel[grid](
            q, k, v, state_tensor, A_log, a, dt_bias, b,
            cu_seqlens, output_tensor, new_state_tensor, scale,
            K=HEAD_DIM, V_DIM=HEAD_DIM,
            NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
            GVA_RATIO=GVA_RATIO, BV=bv, N_V_TILES=n_v_tiles,
            C=CHUNK_SIZE, BK=CHUNK_BK,
            num_warps=8, num_stages=1,
        )
        return output_tensor, new_state_tensor

    if num_seqs == 1 and avg_seq_len >= 512:
        kernel = gdn_prefill_kernel_long
        bv, num_warps, num_stages = STARVED_BV, STARVED_NUM_WARPS, STARVED_NUM_STAGES
    elif num_seqs <= 2 and avg_seq_len >= 256:
        kernel = gdn_prefill_kernel_long
        bv, num_warps, num_stages = LONG_BV, LONG_NUM_WARPS, LONG_NUM_STAGES
    else:
        kernel = gdn_prefill_kernel
        bv, num_warps, num_stages = BALANCED_BV, BALANCED_NUM_WARPS, BALANCED_NUM_STAGES
    n_v_tiles = HEAD_DIM // bv

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
def gdn_prefill_kernel(
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
def _prefill_update_tile_pair(
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
def _load_long_token(
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
def gdn_prefill_kernel_long(
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

    curr_g, curr_beta, curr_q, curr_k, curr_v0, curr_v1 = _load_long_token(
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
        next_g, next_beta, next_q, next_k, next_v0, next_v1 = _load_long_token(
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

        out0, out1, state_tile0, state_tile1 = _prefill_update_tile_pair(
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
