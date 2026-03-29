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
BV: int = 8
N_V_TILES: int = HEAD_DIM // BV


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

    grid = (num_seqs, NUM_V_HEADS * N_V_TILES)
    gdn_prefill_kernel[grid](
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
        BV=BV,
        N_V_TILES=N_V_TILES,
        num_warps=8,
        num_stages=4,
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

    ns_base = new_state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    if seq_end <= seq_start:
        tl.store(ns_ptrs, tl.zeros((BV, K), dtype=tl.float32))
        return

    s_base = state_ptr + (pid_seq * NUM_V_HEADS + pid_h) * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    state_tile = tl.load(s_ptrs).to(tl.float32)
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
