"""Reference implementation for GDN prefill in k-last state layout."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run a float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Run the reference GDN prefill recurrence for variable-length sequences."""
    total_seq_len, num_q_heads, head_size = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size),
        dtype=torch.bfloat16,
        device=device,
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        if seq_end <= seq_start:
            continue

        if state is not None:
            state_hkv = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_hkv = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        for token_idx in range(seq_start, seq_end):
            q_h1k = q_exp[token_idx].unsqueeze(1).float()
            k_h1k = k_exp[token_idx].unsqueeze(1).float()
            v_h1v = v[token_idx].unsqueeze(1).float()
            g_h11 = g[token_idx].unsqueeze(1).unsqueeze(2)
            beta_h11 = beta[token_idx].unsqueeze(1).unsqueeze(2)

            old_state_hkv = g_h11 * state_hkv
            old_v_h1v = matmul(k_h1k, old_state_hkv)
            new_v_h1v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v_h1v
            state_remove = torch.einsum(
                "hkl,hlv->hkv",
                k_h1k.transpose(-1, -2),
                old_v_h1v,
            )
            state_update = torch.einsum(
                "hkl,hlv->hkv",
                k_h1k.transpose(-1, -2),
                new_v_h1v,
            )
            state_hkv = old_state_hkv - state_remove + state_update

            output[token_idx] = (scale * matmul(q_h1k, state_hkv)).squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_hkv.transpose(-1, -2)

    return output, new_state
