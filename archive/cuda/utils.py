import torch

def _device_dtype_check(x: torch.Tensor, dtype: torch.dtype):
    assert x.is_cuda, "tensor is not on CUDA"
    assert x.dtype == dtype, "tensor is not of expected dtype"

    return x.contiguous()


def checks(q, k, v, state, A_log, a, dt_bias, b, scale=None):
    q = _device_dtype_check(q, torch.float32)
    k = _device_dtype_check(k, torch.float32)
    v = _device_dtype_check(v, torch.float32)

    A_log = _device_dtype_check(A_log, torch.float32)
    a = _device_dtype_check(a, torch.float32)
    dt_bias = _device_dtype_check(dt_bias, torch.float32)
    b = _device_dtype_check(b, torch.float32)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, (
        "either of q, k, v is not 4D tensor"
    )

    B, T, num_q_heads, K = q.shape
    Bk, Tk, num_k_heads, Kk = k.shape
    Bv, Tv, num_v_heads, V = v.shape

    assert B == Bk == Bv, "q, k, and v must share batch dimensions"
    assert T == 1, f"Only T=1 is supported in this debug binding, got T={T}"
    assert num_q_heads == 16 and num_k_heads == 16 and num_v_heads == 32, (
        "Only head counts (q=16, k=16, v=32) are supported in this debug binding"
    )
    assert K == 128 and V == 128, "Unsupported outer (K) dimension"

    state = _device_dtype_check(state, torch.float32)
    assert state.shape == (B, num_v_heads, V, K)
