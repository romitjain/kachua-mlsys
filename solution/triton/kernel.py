"""Triton v2 GDN Decode Kernel.

Evolved from v1 naive baseline through 30 experiments on B200. Key changes:
1. 3D grid (B, NUM_V_HEADS, V_DIM//BV) — parallelizes V-tile dimension.
2. Block pointers via tl.make_block_ptr — TMA-eligible on Blackwell (sm100).
3. Tuned: num_warps=8, num_stages=4, BLOCK_V=16.

Best result: ~13.7x speedup (median) on Modal B200.
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


@register_func("flashinfer.kernel")
def launch_gdn_v2(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """Launch the Triton v2 decode kernel.

    The framework calls this as kernel(*inputs, *outputs), passing
    pre-allocated output tensors. Inputs arrive as tvm_ffi.core.Tensor
    and must be converted to PyTorch tensors via DLPack.
    """
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

    grid = (B, NUM_V_HEADS, HEAD_DIM // 16)

    gdn_decode_kernel[grid](
        q, k, v, state, A_log, a, dt_bias, b,
        output, new_state,
        scale,
        K=HEAD_DIM, V_DIM=HEAD_DIM,
        NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
        GVA_RATIO=GVA_RATIO, BLOCK_V=16,
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
    BLOCK_V: tl.constexpr,
):
    """GDN decode: one program per (batch, v_head, v_tile).

    3D grid decomposes work as:
      pid_b  = batch index
      pid_h  = V-head index (0..7)
      pid_v  = V-tile index (0..V_DIM//BLOCK_V-1)

    State update math (same as reference):
      g         = exp(-exp(A_log) * softplus(a + dt_bias))
      beta      = sigmoid(b)
      old_state = g * state
      old_v     = k @ old_state           (dot over K dim)
      new_v     = beta * v + (1-beta) * old_v
      state_out = old_state + (new_v - old_v)[:, None] * k[None, :]
      output    = scale * q @ state_out   (dot over K dim)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)

    # GVA: map V-head -> shared Q/K-head (8 V-heads, 4 Q/K-heads, ratio 2:1)
    qk_head = pid_h // GVA_RATIO

    # Gate computation (scalars, f32)
    offs_h = pid_b * NUM_V_HEADS + pid_h
    a_val = tl.load(a_ptr + offs_h).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    b_val = tl.load(b_ptr + offs_h).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    # Load q and k vectors [K]
    k_offs = tl.arange(0, K)
    qk_base = pid_b * (NUM_K_HEADS * K) + qk_head * K
    q_h = tl.load(q_ptr + qk_base + k_offs).to(tl.float32)
    k_h = tl.load(k_ptr + qk_base + k_offs).to(tl.float32)

    # V-tile range for this program
    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)

    # Load state tile [BLOCK_V, K] via block pointer (TMA-eligible)
    state_base = state_ptr + (pid_b * NUM_V_HEADS + pid_h) * V_DIM * K
    state_bptr = tl.make_block_ptr(
        base=state_base,
        shape=(V_DIM, K),
        strides=(K, 1),
        offsets=(v_start, 0),
        block_shape=(BLOCK_V, K),
        order=(1, 0),
    )
    state_tile = tl.load(state_bptr).to(tl.float32)

    # Decay: apply exponential gate to state
    old_state = g * state_tile

    # old_v = k @ old_state per V-row -> [BLOCK_V]
    old_v = tl.sum(old_state * k_h[None, :], axis=1)

    # Load value tile [BLOCK_V]
    v_base = v_ptr + (pid_b * NUM_V_HEADS + pid_h) * V_DIM
    v_tile = tl.load(v_base + v_offs).to(tl.float32)

    # Blend: weighted mix of fresh value and recalled memory
    new_v = beta * v_tile + (1 - beta) * old_v

    # Delta rule: erase old association, write new
    delta_v = new_v - old_v
    state_out = old_state + delta_v[:, None] * k_h[None, :]

    # Output: q @ state_out per V-row -> [BLOCK_V]
    out_tile = scale * tl.sum(state_out * q_h[None, :], axis=1)

    # Store output [BLOCK_V] as bf16
    out_base = out_ptr + (pid_b * NUM_V_HEADS + pid_h) * V_DIM
    tl.store(out_base + v_offs, out_tile.to(tl.bfloat16))

    # Store updated state [BLOCK_V, K] via block pointer
    ns_base = new_state_ptr + (pid_b * NUM_V_HEADS + pid_h) * V_DIM * K
    ns_bptr = tl.make_block_ptr(
        base=ns_base,
        shape=(V_DIM, K),
        strides=(K, 1),
        offsets=(v_start, 0),
        block_shape=(BLOCK_V, K),
        order=(1, 0),
    )
    tl.store(ns_bptr, state_out)
