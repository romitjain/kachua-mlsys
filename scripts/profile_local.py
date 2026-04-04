from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main(input_path) -> None:
    source_path = Path(__file__).resolve().parent.parent
    source_path = Path(source_path, input_path)

    ext = load(
        name="kachua_gdn_cuda_ext",
        sources=[str(source_path)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )

    torch.manual_seed(7)
    device = "cuda"
    b = 1
    k_dim = 128
    v_dim = 128
    num_q_heads = 4
    num_k_heads = 4
    num_v_heads = 8
    scale = 1.0 / (k_dim**0.5)

    q = torch.randn(b, 1, num_q_heads, k_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, 1, num_k_heads, k_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, 1, num_v_heads, v_dim, device=device, dtype=torch.bfloat16)
    state = torch.randn(b, num_v_heads, v_dim, k_dim, device=device, dtype=torch.float32)
    a_log = torch.randn(num_v_heads, device=device, dtype=torch.float32)
    a = torch.randn(b, 1, num_v_heads, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.float32)
    b_gate = torch.randn(b, 1, num_v_heads, device=device, dtype=torch.bfloat16)

    out = torch.empty((b, num_v_heads, v_dim), dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)

    ext.launch_gdn(q, k, v, state, a_log, a, dt_bias, b_gate, scale, out, new_state)
    torch.cuda.synchronize()

    for _ in range(5):
        ext.launch_gdn(q, k, v, state, a_log, a, dt_bias, b_gate, scale, out, new_state)
    torch.cuda.synchronize()

    ext.launch_gdn(q, k, v, state, a_log, a, dt_bias, b_gate, scale, out, new_state)
    torch.cuda.synchronize()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="solutions/cuda/kernel.cu")
    args = parser.parse_args()

    main(args.input)
