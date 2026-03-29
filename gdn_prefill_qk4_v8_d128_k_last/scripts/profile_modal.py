"""Profile the active config.toml kernel on Modal with real contest workloads."""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_WORKSPACE = Path("/workspace")
LOCAL_PROFILE_ROOT = PROJECT_ROOT / "profiles"
PROFILE_VOLUME_NAME = "kachua-gdn-profiles"
TRACE_VOLUME_NAME = "flashinfer-trace"
TRACE_SET_PATH = "/data"
DEFAULT_PREFILL_SHAPES = (
    (6, 1, 2),
    (134, 1, 2),
    (82, 3, 4),
    (401, 4, 5),
    (4124, 15, 16),
    (8192, 20, 21),
    (8192, 32, 33),
    (8192, 57, 58),
)


@dataclass(frozen=True)
class ProfileTarget:
    """Describe the implementation selected by config.toml."""

    backend: str
    stage: str
    definition: str
    entry_file: str
    entry_function: str
    ncu_kernel_regex: str
    source_files: tuple[str, ...]
    binding: str | None = None


def _is_trace_root(path: Path) -> bool:
    """Return True if path looks like a flashinfer trace-set root."""
    return (path / "definitions").is_dir() and (path / "workloads").is_dir()


def _resolve_trace_set_path(base_path: str | Path) -> Path:
    """Resolve trace-set root from a mount or dataset directory."""
    base = Path(base_path)
    if _is_trace_root(base):
        return base
    if not base.exists():
        raise FileNotFoundError(f"Trace root does not exist: {base}")
    for child in sorted(base.iterdir()):
        if child.is_dir() and _is_trace_root(child):
            return child
    raise FileNotFoundError(
        f"Could not find flashinfer trace-set under '{base}'. "
        "Expected 'definitions/' and 'workloads/' at mount root or one level below."
    )


def _candidate_trace_roots(dataset_root: Path | None) -> list[Path]:
    """Return local filesystem candidates for the contest dataset."""
    dataset_name = dataset_root.name if dataset_root else "mlsys26-contest"
    candidates: list[Path] = []
    if dataset_root is not None:
        candidates.append(dataset_root)
    for base in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        candidates.append(base / dataset_name)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def resolve_local_trace_root(dataset_root: Path | None = None) -> Path:
    """Resolve the local contest dataset path from the checkout or worktree."""
    for candidate in _candidate_trace_roots(dataset_root):
        try:
            return _resolve_trace_set_path(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Could not find mlsys26-contest locally. Checked explicit hint and repo parents."
    )


def load_config() -> dict[str, Any]:
    """Load the repository config.toml."""
    with open(PROJECT_ROOT / "config.toml", "rb") as config_file:
        return tomllib.load(config_file)


def definition_stage(definition: str) -> str:
    """Infer the workload stage from the definition name."""
    return "prefill" if "prefill" in definition else "decode"


def kernel_regex_for_stage(stage: str) -> str:
    """Return the Nsight Compute kernel regex for the active stage."""
    kernel_name = "gdn_prefill_kernel" if stage == "prefill" else "gdn_decode_kernel"
    return f"regex:.*{kernel_name}.*"


def source_dir_for_language(language: str) -> str:
    """Map config language to the solution source directory."""
    source_dirs = {"cuda": "cuda", "triton": "triton", "cute": "cute"}
    try:
        return source_dirs[language]
    except KeyError as error:
        raise ValueError(f"Unsupported language in config.toml: {language}") from error


def resolve_profile_target() -> ProfileTarget:
    """Resolve the active config.toml entry to a profile target."""
    config = load_config()
    build = config["build"]
    solution = config["solution"]
    definition = solution["definition"]
    stage = definition_stage(definition)
    entry_file, entry_function = build["entry_point"].split("::", maxsplit=1)
    language = build["language"]

    if language == "cuda":
        binding = build.get("binding")
        source_files = (f"solution/cuda/{entry_file}", "solution/cuda/binding.py")
        return ProfileTarget(
            backend="cuda",
            stage=stage,
            definition=definition,
            entry_file=entry_file,
            entry_function="kernel",
            ncu_kernel_regex=kernel_regex_for_stage(stage),
            source_files=source_files,
            binding=binding,
        )

    source_dir = source_dir_for_language(language)
    return ProfileTarget(
        backend=language,
        stage=stage,
        definition=definition,
        entry_file=entry_file,
        entry_function=entry_function,
        ncu_kernel_regex=kernel_regex_for_stage(stage),
        source_files=(f"solution/{source_dir}/{entry_file}",),
    )


def read_profile_sources(target: ProfileTarget) -> dict[str, str]:
    """Load only the files needed by the active target."""
    sources = {"config.toml": (PROJECT_ROOT / "config.toml").read_text()}
    for relative_path in target.source_files:
        sources[relative_path] = (PROJECT_ROOT / relative_path).read_text()
    return sources


def workload_jsonl_path(dataset_root: Path | None, definition: str) -> Path:
    """Return the workload JSONL path for the selected definition."""
    root = resolve_local_trace_root(dataset_root)
    return root / "workloads" / "gdn" / f"{definition}.jsonl"


def load_workload_entries(dataset_root: Path | None, definition: str) -> list[dict[str, Any]]:
    """Load workload entries for the selected definition."""
    workload_path = workload_jsonl_path(dataset_root, definition)
    with workload_path.open("r", encoding="utf-8") as workload_file:
        return [json.loads(line) for line in workload_file if line.strip()]


def workload_axes_tuple(entry: dict[str, Any]) -> tuple[int, ...]:
    """Return the axes tuple used to identify a workload."""
    axes = entry["workload"]["axes"]
    if "batch_size" in axes:
        return (int(axes["batch_size"]),)
    return (
        int(axes["total_seq_len"]),
        int(axes["num_seqs"]),
        int(axes["len_cu_seqlens"]),
    )


def default_decode_batches(entries: list[dict[str, Any]]) -> list[int]:
    """Return sorted decode batch sizes present in the dataset."""
    return sorted({int(workload_axes_tuple(entry)[0]) for entry in entries})


def default_prefill_shapes(entries: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Return the representative prefill shapes that exist in the dataset."""
    available = {tuple(int(value) for value in workload_axes_tuple(entry)) for entry in entries}
    return [shape for shape in DEFAULT_PREFILL_SHAPES if shape in available]


def _select_first_entry_by_axes(
    entries: list[dict[str, Any]],
    requested_axes: list[tuple[int, ...]],
) -> list[dict[str, Any]]:
    """Return the first matching workload entry for each requested axes tuple."""
    selected: list[dict[str, Any]] = []
    for axes in requested_axes:
        match = next((entry for entry in entries if workload_axes_tuple(entry) == axes), None)
        if match is None:
            raise ValueError(f"No workload found for axes {axes}")
        selected.append(match)
    return selected


def select_profile_workloads(
    entries: list[dict[str, Any]],
    decode_batches: list[int] | None = None,
    prefill_shapes: list[tuple[int, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Select one canonical workload entry per requested decode batch or prefill shape."""
    if decode_batches is not None and prefill_shapes is not None:
        raise ValueError("Specify decode batches or prefill shapes, not both")
    if decode_batches is not None:
        return _select_first_entry_by_axes(entries, [(batch,) for batch in decode_batches])
    if prefill_shapes is not None:
        requested = [tuple(int(value) for value in shape) for shape in prefill_shapes]
        return _select_first_entry_by_axes(entries, requested)
    raise ValueError("No profile workload selection provided")


def parse_decode_batches(raw_value: str) -> list[int]:
    """Parse a comma-separated decode batch list."""
    return [int(part) for part in raw_value.split(",") if part.strip()]


def parse_prefill_shapes(raw_value: str) -> list[tuple[int, int, int]]:
    """Parse a comma-separated list of total_seq_len:num_seqs:len_cu_seqlens tuples."""
    shapes: list[tuple[int, int, int]] = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        total_seq_len, num_seqs, len_cu_seqlens = (int(part) for part in stripped.split(":"))
        shapes.append((total_seq_len, num_seqs, len_cu_seqlens))
    return shapes


def default_warmup_for_stage(stage: str) -> int:
    """Return the default warmup count for the selected stage."""
    return 1 if stage == "prefill" else 5


def workload_axis_label(entry: dict[str, Any]) -> str:
    """Return a stable workload label for artifact names."""
    axes = workload_axes_tuple(entry)
    if len(axes) == 1:
        return f"b{axes[0]}"
    total_seq_len, num_seqs, len_cu_seqlens = axes
    return f"t{total_seq_len}_n{num_seqs}_c{len_cu_seqlens}"


def to_volume_path(remote_path: str) -> str:
    """Translate a container mount path into a Modal volume path."""
    path = Path(remote_path)
    if path.is_absolute() and path.parts[:2] == ("/", "profiles"):
        return "/" + str(Path(*path.parts[2:]))
    return remote_path


def runner_prelude(
    target: dict[str, Any],
    trace_root: str,
    workload_json: str,
    warmup: int,
) -> str:
    """Return the runner prelude with embedded config for the selected workload."""
    return textwrap.dedent(
        f"""
        WORKSPACE = {str(REMOTE_WORKSPACE)!r}
        TRACESET_ROOT = {trace_root!r}
        ENTRY_FILE = {target["entry_file"]!r}
        ENTRY_FUNCTION = {target["entry_function"]!r}
        BACKEND = {target["backend"]!r}
        STAGE = {target["stage"]!r}
        WARMUP = {warmup}
        WORKLOAD_JSON = {workload_json!r}
        """
    )


def runner_base() -> str:
    """Return the shared Python runner used under Nsight Compute."""
    return textwrap.dedent(
        """
        import ctypes
        import importlib.util
        import json
        import os
        import sys
        import types
        from pathlib import Path

        import torch
        from safetensors.torch import load_file


        def load_module(module_name: str, module_path: str):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module


        def prepend_env_path(name: str, value: str):
            current = os.environ.get(name)
            os.environ[name] = f"{value}:{current}" if current else value


        spec = importlib.util.find_spec("tvm_ffi")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("apache-tvm-ffi must be installed in the profiling image")

        tvm_ffi_root = Path(next(iter(spec.submodule_search_locations))).resolve()
        for include_dir in (
            tvm_ffi_root / "include",
            tvm_ffi_root / "3rdparty/dlpack/include",
        ):
            prepend_env_path("CPATH", str(include_dir))
            prepend_env_path("CPLUS_INCLUDE_PATH", str(include_dir))

        ctypes.CDLL(str(tvm_ffi_root / "core.abi3.so"), mode=ctypes.RTLD_GLOBAL)

        stub_tvm_ffi = types.ModuleType("tvm_ffi")
        stub_tvm_ffi.register_global_func = lambda _name: (lambda function: function)
        sys.modules["tvm_ffi"] = stub_tvm_ffi

        workload = json.loads(WORKLOAD_JSON)
        tensor_cache: dict[str, dict[str, torch.Tensor]] = {}


        def load_input(spec: dict[str, object]):
            kind = spec["type"]
            if kind == "scalar":
                return spec["value"]
            if kind != "safetensors":
                raise ValueError(f"Unsupported workload input type: {kind}")
            relative_path = Path(str(spec["path"]))
            tensor_path = relative_path
            if not tensor_path.is_absolute():
                tensor_path = Path(TRACESET_ROOT) / tensor_path
            resolved = str(tensor_path.resolve())
            if resolved not in tensor_cache:
                tensor_cache[resolved] = load_file(resolved, device="cuda")
            return tensor_cache[resolved][str(spec["tensor_key"])]


        inputs = {
            name: load_input(spec)
            for name, spec in workload["workload"]["inputs"].items()
        }
        device = "cuda"


        def allocate_outputs():
            if STAGE == "decode":
                batch_size = inputs["q"].shape[0]
                num_v_heads = inputs["v"].shape[2]
                head_dim = inputs["v"].shape[3]
                output = torch.empty(
                    (batch_size, num_v_heads, head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                new_state = torch.empty_like(inputs["state"])
                return output, new_state

            total_seq_len = inputs["q"].shape[0]
            num_v_heads = inputs["v"].shape[1]
            head_dim = inputs["v"].shape[2]
            num_seqs = inputs["cu_seqlens"].numel() - 1
            output = torch.empty(
                (total_seq_len, num_v_heads, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            new_state = torch.empty(
                (num_seqs, num_v_heads, head_dim, inputs["k"].shape[-1]),
                dtype=torch.float32,
                device=device,
            )
            return output, new_state


        def binding_args():
            if STAGE == "decode":
                return (
                    inputs["q"],
                    inputs["k"],
                    inputs["v"],
                    inputs.get("state"),
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
                inputs.get("state"),
                inputs["A_log"],
                inputs["a"],
                inputs["dt_bias"],
                inputs["b"],
                inputs["cu_seqlens"],
                inputs["scale"],
            )


        def kernel_args(output, new_state):
            if STAGE == "decode":
                return (
                    inputs["q"],
                    inputs["k"],
                    inputs["v"],
                    inputs.get("state"),
                    inputs["A_log"],
                    inputs["a"],
                    inputs["dt_bias"],
                    inputs["b"],
                    inputs["scale"],
                    output,
                    new_state,
                )
            return (
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs.get("state"),
                inputs["A_log"],
                inputs["a"],
                inputs["dt_bias"],
                inputs["b"],
                inputs["cu_seqlens"],
                inputs["scale"],
                output,
                new_state,
            )
        """
    )


def runner_backend_body(target: dict[str, Any]) -> str:
    """Return the backend-specific runner body."""
    backend = target["backend"]
    if backend == "cuda" and target.get("binding") == "torch":
        return textwrap.dedent(
            """
            import torch.utils.cpp_extension as cpp_extension

            _original_load = cpp_extension.load


            def profile_load(*args, **kwargs):
                extra_cuda_cflags = list(kwargs.get("extra_cuda_cflags") or [])
                if "-lineinfo" not in extra_cuda_cflags:
                    extra_cuda_cflags.append("-lineinfo")
                kwargs["extra_cuda_cflags"] = extra_cuda_cflags
                return _original_load(*args, **kwargs)


            cpp_extension.load = profile_load
            binding = load_module(
                "cuda_binding",
                str(Path(WORKSPACE) / "solution/cuda/binding.py"),
            )
            run_kernel = getattr(binding, ENTRY_FUNCTION)


            def invoke():
                run_kernel(*binding_args())


            invoke()
            torch.cuda.synchronize()

            for _ in range(WARMUP):
                invoke()
            torch.cuda.synchronize()

            invoke()
            torch.cuda.synchronize()
            """
        )

    if backend == "cuda":
        return textwrap.dedent(
            """
            import subprocess as _subprocess

            del sys.modules["tvm_ffi"]
            import tvm_ffi

            kernel_src = str(Path(WORKSPACE) / "solution/cuda" / ENTRY_FILE)
            so_path = "/tmp/gdn_kernel.so"
            nvcc_cmd = [
                "nvcc",
                kernel_src,
                "-shared",
                "-o",
                so_path,
                "-Xcompiler",
                "-fPIC",
                "-I" + str(tvm_ffi_root / "include"),
                "-I" + str(tvm_ffi_root / "3rdparty" / "dlpack" / "include"),
                "-std=c++17",
                "-O3",
                "-lineinfo",
                "-arch=native",
            ]
            nvcc_result = _subprocess.run(nvcc_cmd, capture_output=True, text=True)
            if nvcc_result.returncode != 0:
                print("NVCC STDOUT:", nvcc_result.stdout)
                print("NVCC STDERR:", nvcc_result.stderr)
                raise RuntimeError(
                    f"nvcc failed with exit code {nvcc_result.returncode}"
                )

            kernel_module = tvm_ffi.load_module(so_path)
            kernel_func = kernel_module[ENTRY_FUNCTION]


            def invoke():
                output, new_state = allocate_outputs()
                kernel_func(*kernel_args(output, new_state))


            invoke()
            torch.cuda.synchronize()

            for _ in range(WARMUP):
                invoke()
            torch.cuda.synchronize()

            invoke()
            torch.cuda.synchronize()
            """
        )

    backend_dir = "triton" if backend == "triton" else "cute"
    return textwrap.dedent(
        f"""
        kernel_module = load_module(
            "{backend}_kernel",
            str(Path(WORKSPACE) / "solution/{backend_dir}" / ENTRY_FILE),
        )
        run_kernel = getattr(kernel_module, ENTRY_FUNCTION)


        def invoke():
            output, new_state = allocate_outputs()
            run_kernel(*kernel_args(output, new_state))


        invoke()
        torch.cuda.synchronize()

        for _ in range(WARMUP):
            invoke()
        torch.cuda.synchronize()

        invoke()
        torch.cuda.synchronize()
        """
    )


def build_runner_source(
    target: dict[str, Any],
    trace_root: str,
    workload_json: str,
    warmup: int,
) -> str:
    """Build the Python runner executed under Nsight Compute inside Modal."""
    return textwrap.dedent(
        runner_prelude(target, trace_root, workload_json, warmup)
        + runner_base()
        + runner_backend_body(target)
    )


def parse_number(value: str) -> float | None:
    """Parse a numeric Nsight metric value."""
    head = value.split(";", maxsplit=1)[0].strip()
    if not head or head.lower() == "n/a":
        return None
    try:
        return float(head)
    except ValueError:
        return None


def parse_raw_metrics(raw_csv: str) -> dict[str, dict[str, str]]:
    """Parse the Nsight raw CSV into a metric dictionary."""
    metrics: dict[str, dict[str, str]] = {}
    reader = csv.reader(io.StringIO(raw_csv))
    rows = list(reader)
    if len(rows) < 3:
        return metrics
    names = rows[0]
    units = rows[1]
    values = rows[2]
    width = min(len(names), len(units), len(values))
    for index in range(width):
        name = names[index].strip()
        value = values[index].strip()
        if not name or not value:
            continue
        metrics[name] = {"unit": units[index].strip(), "value": value}
    return metrics


def pick_metric(
    metrics: dict[str, dict[str, str]],
    names: tuple[str, ...],
) -> dict[str, str] | None:
    """Return the first present metric from the preferred name list."""
    for name in names:
        if name in metrics:
            return {"name": name, **metrics[name]}
    return None


def summarize_metrics(metrics: dict[str, dict[str, str]]) -> dict[str, object]:
    """Summarize the most useful Nsight metrics for quick inspection."""
    summary: dict[str, object] = {
        "kernel_name": metrics.get("Kernel Name", {}).get("value"),
        "launch__grid_size": pick_metric(metrics, ("launch__grid_size",)),
        "launch__block_size": pick_metric(metrics, ("launch__block_size",)),
        "launch__registers_per_thread": pick_metric(
            metrics, ("launch__registers_per_thread",)
        ),
        "gpu__time_duration.sum": pick_metric(metrics, ("gpu__time_duration.sum",)),
        "sm__throughput": pick_metric(
            metrics,
            (
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__throughput.avg.pct_of_peak_sustained_active",
            ),
        ),
        "dram__throughput": pick_metric(
            metrics,
            (
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            ),
        ),
        "l2_hit_rate": pick_metric(
            metrics,
            (
                "lts__t_sector_hit_rate.pct",
                "lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct",
            ),
        ),
        "achieved_occupancy": pick_metric(
            metrics, ("sm__warps_active.avg.pct_of_peak_sustained_active",)
        ),
    }

    stall_metrics: list[dict[str, object]] = []
    for name, metric in metrics.items():
        lower = name.lower()
        if "stall" not in lower and "stalled" not in lower:
            continue
        numeric = parse_number(metric["value"])
        if numeric is None or math.isnan(numeric):
            continue
        stall_metrics.append(
            {
                "name": name,
                "unit": metric["unit"],
                "value": metric["value"],
                "numeric_value": numeric,
            }
        )

    stall_metrics.sort(key=lambda item: float(item["numeric_value"]), reverse=True)
    summary["top_stall_metrics"] = stall_metrics[:10]
    return summary


def download_artifact(remote_path: str, local_target_dir: Path) -> tuple[Path | None, bool]:
    """Download one artifact from the Modal volume to the local profiles directory."""
    volume_path = to_volume_path(remote_path)
    local_path = local_target_dir / Path(remote_path).name
    download = subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "get",
            "--force",
            PROFILE_VOLUME_NAME,
            volume_path,
            str(local_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if download.returncode == 0:
        return local_path, False
    print(f"FAILED to download {remote_path} to {local_path}")
    if download.stdout:
        print(download.stdout)
    if download.stderr:
        print(download.stderr)
    return None, True


def cleanup_remote_artifacts(remote_paths: list[str]) -> None:
    """Remove downloaded artifacts from the Modal volume."""
    for remote_path in remote_paths:
        cleanup = subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "volume",
                "rm",
                PROFILE_VOLUME_NAME,
                to_volume_path(remote_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if cleanup.returncode != 0 and cleanup.stderr:
            print(cleanup.stderr)


try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    app = modal.App("kachua-gdn-profile")
    profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
    trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
    profile_dir = Path("/profiles")
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .uv_pip_install(
            "apache-tvm-ffi",
            "flashinfer-bench",
            "flashinfer-python",
            "safetensors",
            "torch",
            "triton",
        )
        .apt_install("wget", "gnupg", "build-essential", "ninja-build")
        .run_commands(
            "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/"
            "ubuntu2204/x86_64/3bf863cc.pub | "
            "gpg --batch --yes --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
            "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] "
            "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' "
            "> /etc/apt/sources.list.d/cuda.list",
            "apt-get update && apt-get install -y nsight-compute-2026.1.0",
        )
    )

    @app.function(
        image=image,
        gpu="B200:1",
        timeout=1800,
        volumes={str(profile_dir): profile_volume, TRACE_SET_PATH: trace_volume},
    )
    def run_profile(
        target_json: str,
        source_files: dict[str, str],
        workload_json: str,
        ncu_set: str,
        warmup: int,
        run_id: str,
    ) -> dict[str, str | int]:
        import glob

        target = json.loads(target_json)
        trace_root = _resolve_trace_set_path(TRACE_SET_PATH)
        runner_source = build_runner_source(
            target=target,
            trace_root=str(trace_root),
            workload_json=workload_json,
            warmup=warmup,
        )

        for relative_path, content in source_files.items():
            out_path = REMOTE_WORKSPACE / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content)

        runner_path = REMOTE_WORKSPACE / "runner.py"
        runner_path.write_text(runner_source)

        target_dir = profile_dir / target["backend"] / Path(target["entry_file"]).stem / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        report_stem = target_dir / "profile"

        ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
        command = [
            ncu,
            "--set",
            ncu_set,
            "--target-processes",
            "all",
            "--launch-skip",
            str(warmup + 1),
            "--launch-count",
            "1",
            "--kernel-name",
            target["ncu_kernel_regex"],
            "--export",
            str(report_stem),
            "python",
            str(runner_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(REMOTE_WORKSPACE),
        )

        report_path = report_stem.with_suffix(".ncu-rep")
        raw_csv_path = report_stem.with_suffix(".raw.csv")
        details_path = report_stem.with_suffix(".details.txt")
        summary_path = report_stem.with_suffix(".summary.json")
        raw_csv_stdout = ""
        details_stdout = ""
        summary_json = ""
        extraction_stderr = ""

        if result.returncode == 0:
            raw_result = subprocess.run(
                [ncu, "--import", str(report_path), "--page", "raw", "--csv"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REMOTE_WORKSPACE),
            )
            raw_csv_stdout = raw_result.stdout
            extraction_stderr += raw_result.stderr
            raw_csv_path.write_text(raw_csv_stdout)

            details_result = subprocess.run(
                [ncu, "--import", str(report_path), "--page", "details"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REMOTE_WORKSPACE),
            )
            details_stdout = details_result.stdout
            extraction_stderr += details_result.stderr
            details_path.write_text(details_stdout)

            metrics = parse_raw_metrics(raw_csv_stdout)
            summary = summarize_metrics(metrics)
            summary_json = json.dumps(summary, indent=2, sort_keys=True)
            summary_path.write_text(summary_json)

        profile_volume.commit()
        return {
            "backend": target["backend"],
            "entry_file": target["entry_file"],
            "exit_code": result.returncode,
            "report_path": str(report_path),
            "raw_csv_path": str(raw_csv_path),
            "details_path": str(details_path),
            "summary_path": str(summary_path),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "raw_csv_excerpt": "\n".join(raw_csv_stdout.splitlines()[:40]),
            "details_excerpt": "\n".join(details_stdout.splitlines()[:60]),
            "summary_json": summary_json,
            "extraction_stderr": extraction_stderr,
        }

    @app.local_entrypoint()
    def main(
        dataset_root: str = "",
        decode_batches: str = "",
        prefill_shapes: str = "",
        warmup: int = 0,
        ncu_set: str = "basic",
    ) -> None:
        target = resolve_profile_target()
        dataset_hint = Path(dataset_root) if dataset_root else None
        entries = load_workload_entries(dataset_hint, target.definition)
        if target.stage == "decode":
            batches = (
                parse_decode_batches(decode_batches)
                if decode_batches
                else default_decode_batches(entries)
            )
            selected = select_profile_workloads(entries, decode_batches=batches)
        else:
            shapes = (
                parse_prefill_shapes(prefill_shapes)
                if prefill_shapes
                else default_prefill_shapes(entries)
            )
            selected = select_profile_workloads(entries, prefill_shapes=shapes)

        effective_warmup = warmup or default_warmup_for_stage(target.stage)
        target_payload = json.dumps(
            {
                "backend": target.backend,
                "stage": target.stage,
                "definition": target.definition,
                "entry_file": target.entry_file,
                "entry_function": target.entry_function,
                "binding": target.binding,
                "ncu_kernel_regex": target.ncu_kernel_regex,
            }
        )
        source_files = read_profile_sources(target)

        print(f"Profiling {target.definition} from {target.backend} backend")
        print(f"Selected {len(selected)} workload(s) with warmup={effective_warmup}")
        for entry in selected:
            label = workload_axis_label(entry)
            run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{label}"
            print(f"\nProfiling workload {label} ({entry['workload']['uuid'][:8]}...)")
            result = run_profile.remote(
                target_json=target_payload,
                source_files=source_files,
                workload_json=json.dumps(entry),
                ncu_set=ncu_set,
                warmup=effective_warmup,
                run_id=run_id,
            )

            stem = Path(target.entry_file).stem
            local_target_dir = LOCAL_PROFILE_ROOT / target.backend / stem / run_id
            local_target_dir.mkdir(parents=True, exist_ok=True)
            remote_paths = [
                str(result["report_path"]),
                str(result["raw_csv_path"]),
                str(result["details_path"]),
                str(result["summary_path"]),
            ]
            downloaded_paths: list[Path] = []
            download_failed = False
            for remote_path in remote_paths:
                local_path, failed = download_artifact(remote_path, local_target_dir)
                if local_path is not None:
                    downloaded_paths.append(local_path)
                download_failed = download_failed or failed

            if not download_failed:
                cleanup_remote_artifacts(remote_paths)

            print(
                f"Profiled {target.backend} from "
                f"{target.entry_file}::{target.entry_function}"
            )
            print(f"ncu exit code: {result['exit_code']}")
            for local_path in downloaded_paths:
                print(f"downloaded profile artifact to {local_path}")
            if download_failed:
                print(f"report remains on Modal Volume at {result['report_path']}")
            if result["stdout"]:
                print(result["stdout"])
            if result.get("summary_json"):
                print("SUMMARY JSON:")
                print(result["summary_json"])
            if result.get("raw_csv_excerpt"):
                print("RAW CSV EXCERPT:")
                print(result["raw_csv_excerpt"])
            if result.get("details_excerpt"):
                print("DETAILS EXCERPT:")
                print(result["details_excerpt"])
            if result.get("extraction_stderr"):
                print("EXTRACTION STDERR:")
                print(result["extraction_stderr"])
            if result["stderr"]:
                print("STDERR:")
                print(result["stderr"])
