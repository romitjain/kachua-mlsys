"""
Benchmark the current solution with FlashInfer's pure GPU timer.

This mirrors the reference repo's CUPTI path and measures only GPU kernel time,
excluding Python dispatch and benchmark-framework overhead.
"""

import json
import math
import os
import statistics
import sys
import warnings
from pathlib import Path
from typing import Callable

import safetensors.torch
import torch
from dotenv import load_dotenv

try:
    import tomllib
except ImportError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# FlashInfer writes JIT logs and cache files under this workspace root.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _is_trace_root(path: Path) -> bool:
    return (path / "definitions").is_dir() and (path / "workloads").is_dir()


def resolve_trace_set_path(path: str | Path) -> Path:
    root = Path(path).expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Trace set path does not exist: {root}")

    if _is_trace_root(root):
        return root

    for child in sorted(root.iterdir()):
        if child.is_dir() and _is_trace_root(child):
            return child

    raise FileNotFoundError(
        f"Could not find flashinfer trace-set under '{root}'. "
        "Expected 'definitions/' and 'workloads/' either there or one level below."
    )


def get_trace_set_path() -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    fib_dataset_path = os.environ.get("FIB_DATASET_PATH")
    if not fib_dataset_path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return resolve_trace_set_path(fib_dataset_path)


def load_kernel() -> tuple[str, str, Callable]:
    config = load_config()
    solution_name = config["solution"]["name"]
    definition_name = config["solution"]["definition"]
    language = config["build"]["language"]
    binding = config["build"].get("binding")

    if language == "cuda" and binding == "torch":
        import solution.cuda.binding as binding_module

        def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None):
            T = q.shape[0]
            num_v_heads = v.shape[1]
            V = v.shape[2]
            K = k.shape[2]
            num_seqs = int(cu_seqlens.numel() - 1)

            if scale is None or scale == 0:
                scale_ = 1.0 / math.sqrt(K)
            else:
                scale_ = float(scale)

            state_ = state
            if state_ is None:
                state_ = torch.zeros(
                    (num_seqs, num_v_heads, V, K),
                    dtype=torch.float32,
                    device=q.device,
                )

            out = torch.empty((T, num_v_heads, V), dtype=torch.bfloat16, device=q.device)
            new_state = torch.empty_like(state_)

            ext = binding_module._load_extension()
            ext.launch_gdn(q, k, v, state_, A_log, a, dt_bias, b, cu_seqlens, scale_, out, new_state)
            return out, new_state

        return solution_name, definition_name, kernel

    if language == "triton":
        import importlib.util

        entry_file, entry_function = config["build"]["entry_point"].split("::", maxsplit=1)
        kernel_path = PROJECT_ROOT / "solution" / "triton" / entry_file
        mod_name = "triton_kernel"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, str(kernel_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        entry_fn = getattr(mod, entry_function)

        def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None):
            T = q.shape[0]
            num_v_heads = v.shape[1]
            V = v.shape[2]
            K = k.shape[2]
            num_seqs = int(cu_seqlens.numel() - 1)
            scale_ = 1.0 / math.sqrt(K) if scale is None or scale == 0 else float(scale)
            out = torch.empty((T, num_v_heads, V), dtype=torch.bfloat16, device=q.device)
            new_state = torch.empty(
                (num_seqs, num_v_heads, V, K), dtype=torch.float32, device=q.device,
            )
            entry_fn(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale_, out, new_state)
            return out, new_state

        return solution_name, definition_name, kernel

    raise NotImplementedError(
        f"bench_fi_timing.py supports prefill CUDA torch bindings and Triton, got "
        f"language={language!r}, binding={binding!r}."
    )


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in trace definition: {dtype_name}")
    return DTYPE_MAP[dtype_name]


def _resolve_shape(shape_spec: list | None, axes_def: dict, workload_axes: dict) -> list | None:
    if shape_spec is None:
        return None

    resolved = []
    for dim in shape_spec:
        if isinstance(dim, int):
            resolved.append(dim)
        elif dim in workload_axes:
            resolved.append(workload_axes[dim])
        else:
            axis_def = axes_def.get(dim)
            if not axis_def or axis_def.get("type") != "const":
                raise KeyError(f"Could not resolve dimension '{dim}' from trace definition")
            resolved.append(axis_def["value"])
    return resolved


def _rand_tensor(shape: list, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=device)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.randint(-16, 16, shape, dtype=dtype, device=device)
    raise ValueError(f"Unsupported random dtype: {dtype}")


def _find_definition_path(trace_root: Path, definition_name: str) -> Path:
    matches = sorted(trace_root.joinpath("definitions").rglob(f"{definition_name}.json"))
    if not matches:
        raise FileNotFoundError(f"Definition '{definition_name}' not found under {trace_root}")
    return matches[0]


def _find_workload_path(trace_root: Path, definition_name: str) -> Path:
    matches = sorted(trace_root.joinpath("workloads").rglob(f"{definition_name}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"Workloads for '{definition_name}' not found under {trace_root}")
    return matches[0]


def _load_definition(trace_root: Path, definition_name: str) -> dict:
    return json.loads(_find_definition_path(trace_root, definition_name).read_text())


def _load_workload(trace_root: Path, definition_name: str, workload_idx: int) -> dict:
    workload_path = _find_workload_path(trace_root, definition_name)
    with open(workload_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == workload_idx:
                return json.loads(line)["workload"]
    raise IndexError(f"Workload index {workload_idx} out of range for '{definition_name}'")


def _count_workloads(trace_root: Path, definition_name: str) -> int:
    workload_path = _find_workload_path(trace_root, definition_name)
    with open(workload_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_workload_tensors(
    definition_name: str,
    *,
    workload_idx: int = 0,
    device: str = "cuda",
) -> tuple[dict, list[str]]:
    resolved_trace_set_path = get_trace_set_path()
    definition = _load_definition(resolved_trace_set_path, definition_name)
    workload = _load_workload(resolved_trace_set_path, definition_name, workload_idx)

    tensors = {}
    input_names = list(definition["inputs"].keys())
    workload_axes = workload["axes"]
    axes_def = definition["axes"]

    for input_name, input_def in definition["inputs"].items():
        workload_input = workload["inputs"].get(input_name)
        dtype = _dtype_from_name(input_def["dtype"])
        shape = _resolve_shape(input_def["shape"], axes_def, workload_axes)

        if workload_input is None:
            tensors[input_name] = None
            continue

        if workload_input["type"] == "scalar":
            tensors[input_name] = workload_input["value"]
            continue

        if workload_input["type"] == "safetensors":
            tensor_path = Path(workload_input["path"])
            if not tensor_path.is_absolute():
                tensor_path = resolved_trace_set_path / tensor_path
            tensor = safetensors.torch.load_file(str(tensor_path))[workload_input["tensor_key"]]
            tensors[input_name] = tensor.to(device=device, dtype=dtype, non_blocking=True)
            continue

        if workload_input["type"] == "random":
            if shape is None:
                raise ValueError(f"Random scalar input '{input_name}' is not supported")
            tensors[input_name] = _rand_tensor(shape, dtype, device)
            continue

        raise ValueError(f"Unsupported workload input type: {workload_input['type']}")

    return tensors, input_names


def run_benchmark(
    *,
    workload_idx: int = 0,
    device: str = "cuda",
) -> dict:
    from flashinfer.testing import bench_gpu_time

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; bench_fi_timing.py requires a CUDA device.")

    solution_name, definition_name, kernel = load_kernel()
    tensors, input_names = load_workload_tensors(
        definition_name,
        workload_idx=workload_idx,
        device=device,
    )
    call_args = tuple(tensors[name] for name in input_names)

    kernel(*call_args)
    torch.cuda.synchronize()

    timing_backend = "cupti"
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            times_ms = bench_gpu_time(
                fn=lambda: kernel(*call_args),
                cold_l2_cache=True,
                enable_cupti=True,
            )
        if any("falling back to cuda events" in str(w.message).lower() for w in caught_warnings):
            timing_backend = "cuda_graph"# "cuda_events"
    except Exception as exc:
        exc_text = str(exc).lower()
        if "cupti" not in exc_text and "cuda 13.0 or later driver is supported" not in exc_text:
            raise
        timing_backend = "cuda_graph" # "cuda_events"
        fallback_backend = "CUDA graph timing" # "CUDA events"
        print(f"CUPTI timing unavailable ({exc}). Falling back to {fallback_backend}.")
        times_ms = bench_gpu_time(
            fn=lambda: kernel(*call_args),
            cold_l2_cache=True,
            enable_cupti=False,
            use_cuda_graph=True,
        )
    times_us = [t * 1000.0 for t in times_ms]

    result = {
        "solution_name": solution_name,
        "definition_name": definition_name,
        "workload_idx": workload_idx,
        "timing_backend": timing_backend,
        "median_us": statistics.median(times_us),
        "mean_us": statistics.mean(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "iters": len(times_us),
    }

    print("===" * 10)
    print(f"Workload index: {workload_idx}")
    print(f"Solution: {solution_name}")
    print(f"Definition: {definition_name}")
    print(
        f"{solution_name} [{timing_backend}]: mean={result['mean_us']:.2f} median={result['median_us']:.2f} us "
        f"(min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, n={result['iters']})"
    )
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the current solution using FlashInfer bench_gpu_time"
    )
    parser.add_argument(
        "--workload-idx",
        type=int,
        default=0,
        help="Workload index inside the selected definition, or -1 for 'all' (default: 0).",
    )
    args = parser.parse_args()

    workload_idx = int(args.workload_idx)
    if workload_idx < 0:
        _, definition_name, _ = load_kernel()
        trace_root = get_trace_set_path()
        num_workloads = _count_workloads(trace_root, definition_name)
        print(f"Running timing for all {num_workloads} workloads")
        for workload_idx in range(num_workloads):
            run_benchmark(
                workload_idx=workload_idx,
                device="cuda:0",
            )
        return

    run_benchmark(
        workload_idx=workload_idx,
        device="cuda:0",
    )


if __name__ == "__main__":
    main()
