import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from tvm_ffi import register_global_func as register_func

_EXT = None


def _load_extension():
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="kachua_gdn_prefill_cuda_v1",
            sources=[str(Path(__file__).resolve().with_name("kernel.cu"))],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=False,
        )
    return _EXT


def launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None, output=None, new_state=None):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    A_log = A_log.contiguous()
    a = a.contiguous()
    dt_bias = dt_bias.contiguous()
    b = b.contiguous()
    cu_seqlens = cu_seqlens.contiguous()

    if scale is None or scale == 0:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if state is None:
        state = torch.zeros(
            (cu_seqlens.numel() - 1, 8, 128, 128),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = state.contiguous()

    if output is None:
        output = torch.empty((q.shape[0], 8, 128), dtype=torch.bfloat16, device=q.device)
    else:
        output = output.contiguous()

    if new_state is None:
        new_state = torch.empty_like(state)
    else:
        new_state = new_state.contiguous()

    _load_extension().launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), output, new_state)
    return output, new_state


@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None, output=None, new_state=None):
    return launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
