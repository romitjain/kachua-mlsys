from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_runtime import KernelRunner, load_kernel_runtime
from scripts.profile_utils import build_profile_inputs, build_profile_shape
from scripts.workspace_utils import resolve_build_target, resolve_workspace


def _build_call_args(target, inputs: dict[str, torch.Tensor | float | None]) -> tuple:
    if target.problem_kind == "decode":
        return (
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["state"],
            inputs["A_log"],
            inputs["a"],
            inputs["dt_bias"],
            inputs["b"],
            inputs["scale"],
        )

    return (
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["state"],
        inputs["A_log"],
        inputs["a"],
        inputs["dt_bias"],
        inputs["b"],
        inputs["cu_seqlens"],
        inputs["scale"],
    )


def _load_cuda_override_runner(workspace: str, input_path: str) -> tuple[object, KernelRunner]:
    workspace_layout = resolve_workspace(workspace)
    target = resolve_build_target(workspace_layout)
    if target.language != "cuda" or target.problem_kind != "decode":
        raise ValueError("--input override is only supported for decode CUDA workspaces.")

    source_path = Path(input_path)
    if not source_path.is_absolute():
        source_path = workspace_layout.root / source_path
    source_path = source_path.resolve()
    extension_name = workspace_layout.name.replace("/", "_").replace(".", "root")

    ext = load(
        name=f"kachua_gdn_cuda_ext_profile_{extension_name}",
        sources=[str(source_path)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )

    def run_decode(q, k, v, state, A_log, a, dt_bias, b, scale=None):
        scale_value = 1.0 / math.sqrt(q.shape[-1]) if scale in (None, 0) else float(scale)
        output = torch.empty((q.shape[0], v.shape[2], v.shape[3]), dtype=v.dtype, device=q.device)
        new_state = torch.empty_like(state)
        ext.launch_gdn(q, k, v, state, A_log, a, dt_bias, b, scale_value, output, new_state)
        return output, new_state

    return target, run_decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one workspace kernel for local profiling")
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace root relative to the repo root (default: .)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Legacy CUDA source override relative to the selected workspace.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-seq-len", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--cu-seqlens-len", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    workspace_layout = resolve_workspace(args.workspace)
    if args.input:
        target, runner = _load_cuda_override_runner(args.workspace, args.input)
    else:
        runtime = load_kernel_runtime(workspace_layout)
        target = runtime.target
        runner = runtime.runner

    shape = build_profile_shape(
        target,
        batch_size=args.batch_size,
        total_seq_len=args.total_seq_len,
        num_seqs=args.num_seqs,
        cu_seqlens_len=args.cu_seqlens_len,
    )
    inputs = build_profile_inputs(target, shape, seed=args.seed, device="cuda")
    call_args = _build_call_args(target, inputs)

    print(f"Profiling workspace {workspace_layout.name}: {target.language} {target.problem_kind}")
    runner(*call_args)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        runner(*call_args)
    torch.cuda.synchronize()

    runner(*call_args)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
