"""
TVM FFI Bindings Template for CUDA Kernels.

This file provides Python bindings for your CUDA kernel using TVM FFI.
The entry point function name should match the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from tvm_ffi import register_global_func as register_func

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

@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale=None):
    """
    Python binding for your CUDA kernel.
    """
    B, T, num_q_heads, K = q.shape
    Bv, Tv, num_v_heads, V = v.shape

    if scale is None or scale == 0:
        scale = 1.0 / math.sqrt(K)
    else:
        scale = float(scale)

    out = torch.empty((B, num_v_heads, V), dtype=torch.float32, device=q.device)
    new_state = torch.empty_like(state)

    ext = _load_extension()
    ext.launch_gdn_v1(q, k, v, state, A_log, a, dt_bias, b, scale, out, new_state)  # type: ignore

    return out, new_state

if __name__ == "__main__":
    """
    Launch the kernel with some example data
    """
    import triton
    from ..reference_torch_impl import run as torch_impl
    from .utils import checks

    torch.manual_seed(7)
    device = "cuda"

    B, T = 2, 1
    HQ, HK, HV = 16, 16, 32
    K, V = 128, 128

    q = torch.randn(B, T, HQ, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, HK, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.float32)
    state = torch.randn(B, HV, V, K, device=device, dtype=torch.float32)

    A_log = torch.randn(B, 1, HV, device=device, dtype=torch.float32)
    a = torch.randn(B, 1, HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(B, 1, HV, device=device, dtype=torch.float32)
    b = torch.randn(B, 1, HV, device=device, dtype=torch.float32)
    scale = 1.0 / (K**0.5)

    checks(q, k, v, state, A_log, a, dt_bias, b, scale)

    out_cuda, state_cuda = kernel(q, k, v, state, A_log, a, dt_bias, b, scale)
    out_ref, state_ref = torch_impl(q, k, v, state, A_log, a, dt_bias, b, scale)

    out_diff = (out_cuda.float() - out_ref.float()).abs().max().item()
    state_diff = (state_cuda - state_ref).abs().max().item()

    atol, rtol = 1e-2, 1e-2
    ok_out = torch.allclose(out_cuda.float(), out_ref.float(), atol=atol, rtol=rtol)
    ok_state = torch.allclose(state_cuda, state_ref, atol=atol, rtol=rtol)
    ok = ok_out and ok_state

    print(f"out max abs diff: {out_diff:.2e}")
    print(f"state max abs diff: {state_diff:.2e}")
    print("PASS" if ok else "FAIL")

    if not ok:
        raise SystemExit(1)

    # Minimal speed check
    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: kernel(q, k, v, state, A_log, a, dt_bias, b, scale),
        quantiles=quantiles,
        warmup=20,
        rep=100,
    ) # type: ignore

    print(f"CUDA kernel: {ms:.2e} ms. min/max: {min_ms:.2e}, {max_ms:.2e}")

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: torch_impl(q, k, v, state, A_log, a, dt_bias, b, scale),
        quantiles=quantiles,
        warmup=20,
        rep=100,
    )  # type: ignore

    print(
        f"Reference torch implementation: {ms:.2e} ms. min/max: {min_ms:.2e}, {max_ms:.2e}"
    )
