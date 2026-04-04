"""Triton GDN decode kernels.

The active entry point is ``launch_gdn``, which uses the current single-warp
variant. Historical decode versions remain inline in this file for easier
inspection:

* ``gdn_v1``: naive baseline, one program per (batch, v_head)
* ``gdn_v2``: tiled 3D-grid version using ``make_block_ptr``
* ``gdn_v3``: archived 8-warp v3 variant
* ``gdn_v4``: active single-warp variant
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


def _prepare_inputs(q, k, v, state, A_log, a, dt_bias, b, output, new_state):
    """Convert framework DLPack tensors to torch tensors."""
    return (
        torch.from_dlpack(q),
        torch.from_dlpack(k),
        torch.from_dlpack(v),
        torch.from_dlpack(state),
        torch.from_dlpack(A_log),
        torch.from_dlpack(a),
        torch.from_dlpack(dt_bias),
        torch.from_dlpack(b),
        torch.from_dlpack(output),
        torch.from_dlpack(new_state),
    )


def _resolve_scale(scale):
    """Use the standard decode scale when the caller omits it."""
    if scale is None or scale == 0.0:
        return 1.0 / math.sqrt(HEAD_DIM)
    return scale


@register_func("flashinfer.kernel")
def launch_gdn(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """Launch the active Triton decode kernel."""
    (
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        output,
        new_state,
    ) = _prepare_inputs(q, k, v, state, A_log, a, dt_bias, b, output, new_state)
    scale = _resolve_scale(scale)
    B = q.shape[0]
    grid = (B, NUM_V_HEADS * N_V_TILES)

    gdn_v4[grid](
        q, k, v, state, A_log, a, dt_bias, b,
        output, new_state,
        scale,
        K=HEAD_DIM, V_DIM=HEAD_DIM,
        NUM_V_HEADS=NUM_V_HEADS, NUM_K_HEADS=NUM_K_HEADS,
        GVA_RATIO=GVA_RATIO, BV=BV, N_V_TILES=N_V_TILES,
        num_warps=1, num_stages=4,
    )


@triton.jit
def gdn_v4(
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


@triton.jit
def gdn_v1(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    out_ptr, new_state_ptr,
    scale, B,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Archived Triton v1 decode kernel."""
    pid = tl.program_id(0)
    pid_b = pid // NUM_V_HEADS
    pid_h = pid % NUM_V_HEADS
    qk_head = pid_h // GVA_RATIO

    a_val = tl.load(a_ptr + pid_b * NUM_V_HEADS + pid_h).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    b_val = tl.load(b_ptr + pid_b * NUM_V_HEADS + pid_h).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    k_offs = tl.arange(0, K)
    qk_base = pid_b * (NUM_K_HEADS * K) + qk_head * K
    q_h = tl.load(q_ptr + qk_base + k_offs).to(tl.float32)
    k_h = tl.load(k_ptr + qk_base + k_offs).to(tl.float32)

    state_base = state_ptr + pid_b * (NUM_V_HEADS * V_DIM * K) + pid_h * (V_DIM * K)
    v_base = v_ptr + pid_b * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM
    out_base = out_ptr + pid_b * (NUM_V_HEADS * V_DIM) + pid_h * V_DIM
    ns_base = new_state_ptr + pid_b * (NUM_V_HEADS * V_DIM * K) + pid_h * (V_DIM * K)

    for v_start in range(0, V_DIM, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        tile_ptrs = state_base + v_offs[:, None] * K + k_offs[None, :]
        state_tile = tl.load(tile_ptrs)
        old_state = g * state_tile
        old_v = tl.sum(old_state * k_h[None, :], axis=1)
        v_tile = tl.load(v_base + v_offs).to(tl.float32)
        new_v = beta * v_tile + (1 - beta) * old_v
        delta_v = new_v - old_v
        state_out = old_state + delta_v[:, None] * k_h[None, :]
        out_tile = scale * tl.sum(state_out * q_h[None, :], axis=1)
        tl.store(out_base + v_offs, out_tile.to(tl.bfloat16))
        ns_ptrs = ns_base + v_offs[:, None] * K + k_offs[None, :]
        tl.store(ns_ptrs, state_out)


@triton.jit
def gdn_v2(
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
    """Archived Triton v2 decode kernel."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)
    qk_head = pid_h // GVA_RATIO
    offs_h = pid_b * NUM_V_HEADS + pid_h

    a_val = tl.load(a_ptr + offs_h).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    b_val = tl.load(b_ptr + offs_h).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    k_offs = tl.arange(0, K)
    qk_base = pid_b * (NUM_K_HEADS * K) + qk_head * K
    q_h = tl.load(q_ptr + qk_base + k_offs).to(tl.float32)
    k_h = tl.load(k_ptr + qk_base + k_offs).to(tl.float32)

    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)

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
    old_state = g * state_tile
    old_v = tl.sum(old_state * k_h[None, :], axis=1)

    v_base = v_ptr + (pid_b * NUM_V_HEADS + pid_h) * V_DIM
    v_tile = tl.load(v_base + v_offs).to(tl.float32)
    new_v = beta * v_tile + (1 - beta) * old_v
    delta_v = new_v - old_v
    state_out = old_state + delta_v[:, None] * k_h[None, :]
    out_tile = scale * tl.sum(state_out * q_h[None, :], axis=1)

    out_base = out_ptr + (pid_b * NUM_V_HEADS + pid_h) * V_DIM
    tl.store(out_base + v_offs, out_tile.to(tl.bfloat16))

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


@triton.jit
def gdn_v3(
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
    """Archived Triton v3 decode kernel."""
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES
    qk_head = pid_h // GVA_RATIO
    i_nh = pid_b * NUM_V_HEADS + pid_h

    a_val = tl.load(a_ptr + i_nh).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    b_val = tl.load(b_ptr + i_nh).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    o_k = tl.arange(0, K)
    qk_base = pid_b * (NUM_K_HEADS * K) + qk_head * K
    b_q = tl.load(q_ptr + qk_base + o_k).to(tl.float32)
    b_k = tl.load(k_ptr + qk_base + o_k).to(tl.float32)

    v_start = pid_v * BV
    o_v = tl.arange(0, BV)
    s_base = state_ptr + i_nh * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(s_ptrs).to(tl.float32)
    old_state = g * b_h
    old_v = tl.sum(old_state * b_k[None, :], axis=1)

    v_base = v_ptr + i_nh * V_DIM
    b_v = tl.load(v_base + v_start + o_v).to(tl.float32)
    delta_v = beta * (b_v - old_v)
    old_o = tl.sum(old_state * b_q[None, :], axis=1)
    kq = tl.sum(b_k * b_q)
    b_o = scale * (old_o + delta_v * kq)

    out_base = out_ptr + i_nh * V_DIM
    tl.store(out_base + v_start + o_v, b_o.to(tl.bfloat16))

    state_out = old_state + delta_v[:, None] * b_k[None, :]
    ns_base = new_state_ptr + i_nh * V_DIM * K + v_start * K
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    tl.store(ns_ptrs, state_out)
