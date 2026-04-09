"""Local Nsight Compute profiling harness for the prefill Triton kernel.

Run under ncu (e.g. via Makefile):
    ncu --set detailed ... python scripts/profile_local.py
"""

import importlib.util
import math
from argparse import ArgumentParser
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_triton_kernel():
    """Load the active Triton kernel from solution/triton/kernel.py."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    entry_file, entry_function = config["build"]["entry_point"].split("::", maxsplit=1)
    kernel_path = PROJECT_ROOT / "solution" / "triton" / entry_file
    spec = importlib.util.spec_from_file_location("triton_kernel", str(kernel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry_function)


def build_cu_seqlens(total_seq_len: int, num_seqs: int, device: str) -> torch.Tensor:
    lengths = torch.full((num_seqs,), total_seq_len // num_seqs, dtype=torch.int64)
    lengths[: total_seq_len % num_seqs] += 1
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
    cu_seqlens[1:] = torch.cumsum(lengths.to(device), dim=0)
    return cu_seqlens


def main(total_seq_len: int = 128, num_seqs: int = 4, warmup: int = 5, seed: int = 7) -> None:
    torch.manual_seed(seed)
    device = "cuda"

    T = total_seq_len
    N = num_seqs
    K_DIM = 128
    V_DIM = 128
    NUM_Q_HEADS = 4
    NUM_K_HEADS = 4
    NUM_V_HEADS = 8

    q = torch.randn(T, NUM_Q_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(T, NUM_K_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(T, NUM_V_HEADS, V_DIM, device=device, dtype=torch.bfloat16)
    state = torch.randn(N, NUM_V_HEADS, V_DIM, K_DIM, device=device, dtype=torch.float32)
    A_log = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    a = torch.randn(T, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    b = torch.randn(T, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
    cu_seqlens = build_cu_seqlens(T, N, device)
    scale = 1.0 / math.sqrt(K_DIM)

    output = torch.empty((T, NUM_V_HEADS, V_DIM), dtype=torch.bfloat16, device=device)
    new_state = torch.empty((N, NUM_V_HEADS, V_DIM, K_DIM), dtype=torch.float32, device=device)

    launch_gdn = load_triton_kernel()

    launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
    torch.cuda.synchronize()

    for _ in range(warmup):
        launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
    torch.cuda.synchronize()

    launch_gdn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--total-seq-len", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(
        total_seq_len=args.total_seq_len,
        num_seqs=args.num_seqs,
        warmup=args.warmup,
        seed=args.seed,
    )
