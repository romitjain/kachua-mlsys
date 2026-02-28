"""
TVM FFI Bindings Template for CUDA Kernels.

This file provides Python bindings for your CUDA kernel using TVM FFI.
The entry point function name should match the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

from torch import __name
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from tvm.ffi import register_func

_EXTENSION_MODULE = None

def _load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is None:
        source_path = Path(__file__).resolve().with_name("kernel.cu")
        _EXTENSION_MODULE = load(
            name="kachua_gdn_cuda_ext",
            sources=[str(source_path)],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=True,
        )
    return _EXTENSION_MODULE

def _device_dtype_check(x: torch.Tensor, dtype: torch.dtype):
    assert x.is_cuda(), "tensor is not on CUDA"
    assert x.dtype == dtype, "tensor is not of expected dtype"

    return x.contiguous()

@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale=None):
    """
    Python binding for your CUDA kernel.
    """
    q = _device_dtype_check(q, torch.float32)
    k = _device_dtype_check(k, torch.float32)
    v = _device_dtype_check(v, torch.float32)

    A_log = _device_dtype_check(A_log, torch.float32)
    a = _device_dtype_check(a, torch.float32)
    dt_bias = _device_dtype_check(dt_bias, torch.float32)
    b = _device_dtype_check(b, torch.float32)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "either of q, k, v is not 4D tensor"

    B, T, num_q_heads, K = q.shape
    Bk, Tk, num_k_heads, Kk = k.shape
    Bv, Tv, num_v_heads, V = v.shape

    assert B==Bk==Bv, "q, k, and v must share batch and sequence dimensions"
    assert T==1, f"Only T=1 is supported in this debug binding, got T={T}"
    assert num_q_heads == 16 and num_k_heads == 16 and num_v_heads ==32, "Only head counts (q=16, k=16, v=32) are supported in this debug binding"
    assert K==128 and V == 128, "Unsupported outer (K) dimension"

    if state is None:
        state = torch.zeros((B, num_v_heads, V, K), dtype=torch.float32, device=q.device)
    else:
        state = _device_dtype_check(state, torch.float32)
        assert state.shape == (B, num_v_heads, V, K)

    if scale is None or scale == 0:
        scale = 1.0 / math.sqrt(K)
    else:
        scale = float(scale)

    out_fp32 = torch.empty((B, num_v_heads, V), dtype=torch.float32, device=q.device)
    new_state = torch.empty_like(state)

    ext = _load_extension()
    ext.launch_gdn_v1(q, k, v, state, A_log, a, dt_bias, b, out_fp32, new_state, scale)

    out = out_fp32.unsqueeze(1).to(torch.float32)
    return out, new_state

if __name__ == "__main__":
    """
    Launch the kernel with some example data
    """
    pass
