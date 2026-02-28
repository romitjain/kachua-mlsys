import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Gated Delta Net decode reference implementation (k-last layout).
    
    State layout: [B, H, V, K] (k-last, K dimension at the end)
    
    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)
    
    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device
    
    assert num_q_heads == 16
    assert num_k_heads == 16
    assert num_v_heads == 32
    assert K == 128 and V == 128
    assert T == 1
    
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    
    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]
    
    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()
    
    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)
    
    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    
    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)
    
    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_f32[b_idx, h_idx]
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [V,K] -> [K,V]
            g_val = g_f32[b_idx, h_idx]
            beta_val = beta_f32[b_idx, h_idx]
            
            old_state = g_val * h_state
            old_v = k_h @ old_state
            new_v = beta_val * v_h + (1 - beta_val) * old_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
            h_state = old_state - state_remove + state_update
            
            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K]
    
    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state