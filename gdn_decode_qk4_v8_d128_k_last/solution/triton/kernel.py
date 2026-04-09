"""Triton v3 GDN Decode Kernel.

Evolved from v2 through 24 experiments on B200. Key changes from v2:
1. BV=8 (128 blocks) for 86% SM coverage vs v2's BV=16 (64 blocks, 43%).
2. Plain pointer arithmetic instead of make_block_ptr — tighter PTX.
3. Live range optimization: output via algebraic identity before
   materializing state_out, improving IPC 25%.
4. Compact delta: beta*(v - old_v) eliminates intermediate new_v.
"""

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
def launch_gdn(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    q = torch.from_dlpack(q)
    k = torch.from_dlpack(k)
    v = torch.from_dlpack(v)
    state = torch.from_dlpack(state)
    A_log = torch.from_dlpack(A_log)
    a = torch.from_dlpack(a)
    dt_bias = torch.from_dlpack(dt_bias)
    b = torch.from_dlpack(b)
    output = torch.from_dlpack(output)
    new_state = torch.from_dlpack(new_state)

    B = q.shape[0]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(HEAD_DIM)

    grid = (B, NUM_V_HEADS * N_V_TILES)

    gdn_decode_kernel[grid](
        q, k, v, state, A_log, a, dt_bias, b,
        output, new_state,
        scale,
        K=HEAD_DIM, V_DIM=HEAD_DIM,
        NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
        GVA_RATIO=GVA_RATIO, BV=BV, N_V_TILES=N_V_TILES,
        num_warps=8, num_stages=4,
    )


@triton.jit
def gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    out_ptr, new_state_ptr,
    scale,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BV: tl.constexpr,
    N_V_TILES: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES

    qk_head = pid_h // GVA_RATIO
    i_nh = pid_b * NUM_V_HEADS + pid_h

    # Gate computation
    a_val = tl.load(a_ptr + i_nh).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    b_val = tl.load(b_ptr + i_nh).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    # Load q, k
    o_k = tl.arange(0, K)
    qk_base = pid_b * (NUM_K_HEADS * K) + qk_head * K
    b_q = tl.load(q_ptr + qk_base + o_k).to(tl.float32)
    b_k = tl.load(k_ptr + qk_base + o_k).to(tl.float32)

    # V-tile
    v_start = pid_v * BV
    o_v = tl.arange(0, BV)

    # Load state [BV, K] via plain pointers (row-major, K stride 1)
    s_base = state_ptr + i_nh * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(s_ptrs).to(tl.float32)

    # Decay
    old_state = g * b_h

    # old_v = k @ old_state per V-row
    old_v = tl.sum(old_state * b_k[None, :], axis=1)

    # Load value
    v_base = v_ptr + i_nh * V_DIM
    b_v = tl.load(v_base + v_start + o_v).to(tl.float32)

    # Compact delta (Q3: reduces peak live registers)
    delta_v = beta * (b_v - old_v)

    # Output via identity (Q7: avoids state_out live during reduction)
    # output = scale * (old_state@q + delta_v * dot(k,q))
    old_o = tl.sum(old_state * b_q[None, :], axis=1)
    kq = tl.sum(b_k * b_q)
    b_o = scale * (old_o + delta_v * kq)

    # Store output BEFORE building state_out (frees old_o registers)
    out_base = out_ptr + i_nh * V_DIM
    tl.store(out_base + v_start + o_v, b_o.to(tl.bfloat16))

    # State update (state_out only needed for store, not output)
    state_out = old_state + delta_v[:, None] * b_k[None, :]

    # Store state
    ns_base = new_state_ptr + i_nh * V_DIM * K + v_start * K
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    tl.store(ns_ptrs, state_out)
