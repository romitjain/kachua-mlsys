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


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


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


def get_trace_set_path(explicit_path: str | None = None) -> Path:
    if explicit_path is not None:
        return resolve_trace_set_path(explicit_path)

    _load_env_file(PROJECT_ROOT / ".env")
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

    if language != "cuda" or binding != "torch":
        raise NotImplementedError(
            "bench_fi_timing.py currently supports the current CUDA torch binding only."
        )

    import solution.cuda.binding as binding_module

    def kernel(q, k, v, state, A_log, a, dt_bias, b, scale=None):
        B, _, _, K = q.shape
        _, _, num_v_heads, V = v.shape

        if scale is None or scale == 0:
            scale_ = 1.0 / math.sqrt(K)
        else:
            scale_ = float(scale)

        out = torch.empty((B, num_v_heads, V), dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty_like(state)

        ext = binding_module._load_extension()
        ext.launch_gdn(q, k, v, state, A_log, a, dt_bias, b, scale_, out, new_state)
        return out, new_state

    return solution_name, definition_name, kernel


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


def load_workload_tensors(
    definition_name: str,
    *,
    trace_set_path: str | None = None,
    workload_idx: int = 0,
    device: str = "cuda",
) -> tuple[dict, list[str]]:
    resolved_trace_set_path = get_trace_set_path(trace_set_path)
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
    trace_set_path: str | None = None,
    workload_idx: int = 0,
    device: str = "cuda",
) -> dict:
    from flashinfer.testing import bench_gpu_time

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; bench_fi_timing.py requires a CUDA device.")

    solution_name, definition_name, kernel = load_kernel()
    tensors, input_names = load_workload_tensors(
        definition_name,
        trace_set_path=trace_set_path,
        workload_idx=workload_idx,
        device=device,
    )
    call_args = tuple(tensors[name] for name in input_names)

    print(f"Solution: {solution_name}")
    print(f"Definition: {definition_name}")
    print(f"Workload index: {workload_idx}")
    if device.startswith("cuda"):
        print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"Device: {device}")

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
            timing_backend = "cuda_events"
    except Exception as exc:
        exc_text = str(exc).lower()
        if "cupti" not in exc_text and "cuda 13.0 or later driver is supported" not in exc_text:
            raise
        timing_backend = "cuda_events"
        print(f"CUPTI timing unavailable ({exc}). Falling back to CUDA events.")
        times_ms = bench_gpu_time(
            fn=lambda: kernel(*call_args),
            cold_l2_cache=True,
            enable_cupti=False,
        )
    times_us = [t * 1000.0 for t in times_ms]

    result = {
        "solution_name": solution_name,
        "definition_name": definition_name,
        "workload_idx": workload_idx,
        "timing_backend": timing_backend,
        "median_us": statistics.median(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "iters": len(times_us),
    }

    print(
        f"{solution_name} [{timing_backend}]: median={result['median_us']:.2f} us "
        f"(min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, n={result['iters']})"
    )
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the current solution using FlashInfer bench_gpu_time"
    )
    parser.add_argument(
        "--trace-set-path",
        default=None,
        help="Path to flashinfer-trace root. Defaults to FIB_DATASET_PATH.",
    )
    parser.add_argument(
        "--workload-idx",
        type=int,
        default=0,
        help="Workload index inside the selected definition (default: 0).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device string for generated inputs (default: cuda).",
    )
    args = parser.parse_args()

    run_benchmark(
        trace_set_path=args.trace_set_path,
        workload_idx=args.workload_idx,
        device=args.device,
    )


if __name__ == "__main__":
    main()
