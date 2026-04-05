"""Profile workspace kernels on Modal with Nsight Compute using synthetic inputs."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from scripts.workspace_utils import BuildTarget, WorkspaceLayout


REMOTE_WORKSPACE = Path("/workspace")
PROFILE_VOLUME_NAME = "kachua-gdn-profiles"
MODAL_CLI = [sys.executable, "-m", "modal"]


@dataclass(frozen=True)
class ProfileTarget:
    """Describe one workspace implementation selected for profiling."""

    build_target: BuildTarget
    runner_function: str
    ncu_kernel_regex: str
    source_files: tuple[str, ...]

    @property
    def backend(self) -> str:
        return self.build_target.backend

    @property
    def workspace(self) -> WorkspaceLayout:
        return self.build_target.workspace

    @property
    def problem_kind(self) -> str:
        return self.build_target.problem_kind

    @property
    def entry_file(self) -> str:
        return self.build_target.entry_file

    @property
    def entry_function(self) -> str:
        return self.build_target.entry_function

    @property
    def binding(self) -> str | None:
        return self.build_target.binding


def kernel_regex_for_target(target: BuildTarget) -> str:
    """Return the Nsight Compute kernel-name regex for the selected target."""
    if target.problem_kind == "decode":
        return r"regex:.*gdn_v[0-9]+.*"
    if target.problem_kind == "prefill":
        return r"regex:.*gdn_prefill_kernel.*"
    raise ValueError(f"Unsupported problem kind: {target.problem_kind}")


def resolve_profile_target(workspace: WorkspaceLayout) -> ProfileTarget:
    """Resolve one workspace config to a profile target."""
    from scripts.workspace_utils import resolve_build_target

    target = resolve_build_target(workspace)
    kernel_regex = kernel_regex_for_target(target)

    if target.language == "cuda":
        source_files = (
            f"solution/cuda/{target.entry_file}",
            "solution/cuda/binding.py",
        )
        runner_function = (
            "kernel" if target.binding == "torch" else target.entry_function
        )
        return ProfileTarget(
            build_target=target,
            runner_function=runner_function,
            ncu_kernel_regex=kernel_regex,
            source_files=source_files,
        )

    if target.language == "triton":
        return ProfileTarget(
            build_target=target,
            runner_function=target.entry_function,
            ncu_kernel_regex=kernel_regex,
            source_files=(f"solution/triton/{target.entry_file}",),
        )

    if target.language == "cute":
        return ProfileTarget(
            build_target=target,
            runner_function=target.entry_function,
            ncu_kernel_regex=kernel_regex,
            source_files=(f"solution/cute/{target.entry_file}",),
        )

    raise ValueError(f"Unsupported language in config.toml: {target.language}")


def read_profile_sources(target: ProfileTarget) -> dict[str, str]:
    """Load only the files needed by the active target."""
    from scripts.workspace_utils import read_workspace_sources

    sources = read_workspace_sources(target.workspace, ("config.toml",))
    sources.update(read_workspace_sources(target.workspace, target.source_files))
    return sources


def build_runner_source(
    target: ProfileTarget,
    shape,
    warmup: int,
    seed: int,
) -> str:
    """Build the Python runner executed under Nsight Compute inside Modal."""
    base = """
import importlib.util
import ctypes
import os
import sys
import types
from pathlib import Path

import torch


def load_module(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepend_env_path(name: str, value: str):
    current = os.environ.get(name)
    os.environ[name] = f"{value}:{current}" if current else value


def build_prefill_cu_seqlens(total_seq_len: int, num_seqs: int, expected_len: int, device: str):
    actual_len = num_seqs + 1
    if expected_len != actual_len:
        raise ValueError(
            "cu_seqlens length mismatch: "
            f"expected {actual_len}, got {expected_len}"
        )
    lengths = torch.full((num_seqs,), total_seq_len // num_seqs, dtype=torch.int64)
    lengths[: total_seq_len % num_seqs] += 1
    cu_seqlens = torch.zeros(actual_len, dtype=torch.int64, device=device)
    cu_seqlens[1:] = torch.cumsum(lengths.to(device), dim=0)
    return cu_seqlens


def build_decode_inputs(device: str):
    q = torch.randn(BATCH_SIZE, 1, NUM_Q_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(BATCH_SIZE, 1, NUM_K_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(BATCH_SIZE, 1, NUM_V_HEADS, V_DIM, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        BATCH_SIZE,
        NUM_V_HEADS,
        V_DIM,
        K_DIM,
        device=device,
        dtype=torch.float32,
    )
    A_log = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    a = torch.randn(BATCH_SIZE, 1, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    b = torch.randn(BATCH_SIZE, 1, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "cu_seqlens": None,
    }


def build_prefill_inputs(device: str):
    q = torch.randn(TOTAL_SEQ_LEN, NUM_Q_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(TOTAL_SEQ_LEN, NUM_K_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(TOTAL_SEQ_LEN, NUM_V_HEADS, V_DIM, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        NUM_SEQS,
        NUM_V_HEADS,
        V_DIM,
        K_DIM,
        device=device,
        dtype=torch.float32,
    )
    A_log = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    a = torch.randn(TOTAL_SEQ_LEN, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
    b = torch.randn(TOTAL_SEQ_LEN, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
    cu_seqlens = build_prefill_cu_seqlens(
        TOTAL_SEQ_LEN,
        NUM_SEQS,
        CU_SEQLENS_LEN,
        device,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "cu_seqlens": cu_seqlens,
    }


def allocate_outputs(problem_kind: str, state: torch.Tensor, device: str):
    if problem_kind == "decode":
        output = torch.empty(
            (BATCH_SIZE, NUM_V_HEADS, V_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        return output, torch.empty_like(state)

    output = torch.empty(
        (TOTAL_SEQ_LEN, NUM_V_HEADS, V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    new_state = torch.empty(
        (NUM_SEQS, NUM_V_HEADS, V_DIM, K_DIM),
        dtype=torch.float32,
        device=device,
    )
    return output, new_state


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


device = "cuda"
K_DIM = 128
V_DIM = 128
NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
torch.manual_seed(SEED)
scale = 1.0 / (K_DIM ** 0.5)

if PROBLEM_KIND == "decode":
    inputs = build_decode_inputs(device)
elif PROBLEM_KIND == "prefill":
    inputs = build_prefill_inputs(device)
else:
    raise ValueError(f"Unsupported problem kind: {PROBLEM_KIND}")

q = inputs["q"]
k = inputs["k"]
v = inputs["v"]
state = inputs["state"]
A_log = inputs["A_log"]
a = inputs["a"]
dt_bias = inputs["dt_bias"]
b = inputs["b"]
cu_seqlens = inputs["cu_seqlens"]
"""

    if target.backend == "cuda" and target.binding == "torch":
        body = """
import torch.utils.cpp_extension as cpp_extension

_original_load = cpp_extension.load


def profile_load(*args, **kwargs):
    extra_cuda_cflags = list(kwargs.get("extra_cuda_cflags") or [])
    if "-lineinfo" not in extra_cuda_cflags:
        extra_cuda_cflags.append("-lineinfo")
    kwargs["extra_cuda_cflags"] = extra_cuda_cflags
    return _original_load(*args, **kwargs)


cpp_extension.load = profile_load

binding = load_module("cuda_binding", str(Path(WORKSPACE) / "solution/cuda/binding.py"))
run_kernel = getattr(binding, RUNNER_FUNCTION)


def invoke():
    if PROBLEM_KIND == "decode":
        run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale)
        return
    run_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)


invoke()
torch.cuda.synchronize()

for _ in range(WARMUP):
    invoke()
torch.cuda.synchronize()

invoke()
torch.cuda.synchronize()
"""
    elif target.backend == "cuda":
        body = """
import subprocess as _sp

del sys.modules["tvm_ffi"]
import tvm_ffi

kernel_src = str(Path(WORKSPACE) / "solution/cuda" / ENTRY_FILE)
so_path = "/tmp/gdn_kernel.so"
nvcc_cmd = [
    "nvcc", kernel_src,
    "-shared", "-o", so_path,
    "-Xcompiler", "-fPIC",
    "-I" + str(tvm_ffi_root / "include"),
    "-I" + str(tvm_ffi_root / "3rdparty" / "dlpack" / "include"),
    "-std=c++17", "-O3", "-lineinfo",
    "-arch=native",
]
nvcc_result = _sp.run(nvcc_cmd, capture_output=True, text=True)
if nvcc_result.returncode != 0:
    print("NVCC STDOUT:", nvcc_result.stdout)
    print("NVCC STDERR:", nvcc_result.stderr)
    raise RuntimeError(f"nvcc failed with exit code {nvcc_result.returncode}")

_kernel_mod = tvm_ffi.load_module(so_path)
kernel_func = _kernel_mod[RUNNER_FUNCTION]

output, new_state = allocate_outputs(PROBLEM_KIND, state, device)


def invoke():
    if PROBLEM_KIND == "decode":
        kernel_func(q, k, v, state, A_log, a, dt_bias, b, float(scale), output, new_state)
        return
    kernel_func(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        float(scale),
        output,
        new_state,
    )


invoke()
torch.cuda.synchronize()

for _ in range(WARMUP):
    invoke()
torch.cuda.synchronize()

invoke()
torch.cuda.synchronize()
"""
    else:
        backend_dir = "triton" if target.backend == "triton" else "cute"
        body = f"""
kernel_module = load_module(
    "{target.backend}_kernel",
    str(Path(WORKSPACE) / "solution/{backend_dir}" / ENTRY_FILE),
)
run_kernel = getattr(kernel_module, RUNNER_FUNCTION)


def invoke():
    output, new_state = allocate_outputs(PROBLEM_KIND, state, device)
    if PROBLEM_KIND == "decode":
        run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)
        return
    run_kernel(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        scale,
        output,
        new_state,
    )


invoke()
torch.cuda.synchronize()

for _ in range(WARMUP):
    invoke()
torch.cuda.synchronize()

invoke()
torch.cuda.synchronize()
"""

    prelude = textwrap.dedent(
        f"""
        WORKSPACE = {str(REMOTE_WORKSPACE)!r}
        ENTRY_FILE = {target.entry_file!r}
        RUNNER_FUNCTION = {target.runner_function!r}
        PROBLEM_KIND = {target.problem_kind!r}
        BATCH_SIZE = {shape.batch_size}
        TOTAL_SEQ_LEN = {shape.total_seq_len}
        NUM_SEQS = {shape.num_seqs}
        CU_SEQLENS_LEN = {shape.cu_seqlens_len}
        WARMUP = {warmup}
        SEED = {seed}
        """
    )
    return textwrap.dedent(prelude + base + body)


def to_volume_path(remote_path: str) -> str:
    """Translate a container mount path into a Modal volume path."""
    path = Path(remote_path)
    if path.is_absolute() and path.parts[:2] == ("/", "profiles"):
        return "/" + str(Path(*path.parts[2:]))
    return remote_path


try:
    import modal
except ModuleNotFoundError:  # pragma: no cover
    modal = None


if modal is not None:
    app = modal.App("kachua-gdn-profile")
    profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
    profile_dir = Path("/profiles")
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
        )
        .uv_pip_install(
            "apache-tvm-ffi",
            "flashinfer-bench",
            "flashinfer-python",
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
        volumes={str(profile_dir): profile_volume},
    )
    def run_profile(
        target_json: str,
        source_files: dict[str, str],
        runner_source: str,
        ncu_set: str,
        warmup: int,
        run_id: str,
    ) -> dict[str, str | int]:
        import csv
        import glob
        import io
        import math
        import subprocess

        target = json.loads(target_json)

        def parse_number(value: str) -> float | None:
            head = value.split(";", maxsplit=1)[0].strip()
            if not head or head.lower() == "n/a":
                return None
            try:
                return float(head)
            except ValueError:
                return None

        def parse_raw_metrics(raw_csv: str) -> dict[str, dict[str, str]]:
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
                unit = units[index].strip()
                value = values[index].strip()
                if not name or not value:
                    continue
                metrics[name] = {"unit": unit, "value": value}
            return metrics

        def pick_metric(
            metrics: dict[str, dict[str, str]],
            names: tuple[str, ...],
        ) -> dict[str, str] | None:
            for name in names:
                if name in metrics:
                    return {"name": name, **metrics[name]}
            return None

        def summarize_metrics(metrics: dict[str, dict[str, str]]) -> dict[str, object]:
            summary: dict[str, object] = {
                "kernel_name": metrics.get("Kernel Name", {}).get("value"),
                "launch__grid_size": pick_metric(metrics, ("launch__grid_size",)),
                "launch__block_size": pick_metric(metrics, ("launch__block_size",)),
                "launch__registers_per_thread": pick_metric(
                    metrics,
                    ("launch__registers_per_thread",),
                ),
                "gpu__time_duration.sum": pick_metric(
                    metrics, ("gpu__time_duration.sum",)
                ),
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
                    metrics,
                    ("sm__warps_active.avg.pct_of_peak_sustained_active",),
                ),
            }

            stall_metrics: list[dict[str, object]] = []
            for name, metric in metrics.items():
                lower = name.lower()
                if "stalled" not in lower and "stall" not in lower:
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

            stall_metrics.sort(
                key=lambda item: float(item["numeric_value"]), reverse=True
            )
            summary["top_stall_metrics"] = stall_metrics[:10]
            return summary

        for relative_path, content in source_files.items():
            output_path = REMOTE_WORKSPACE / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)

        runner_path = REMOTE_WORKSPACE / "runner.py"
        runner_path.write_text(runner_source)

        target_dir = (
            profile_dir / target["backend"] / Path(target["entry_file"]).stem / run_id
        )
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
            raw_command = [ncu, "--import", str(report_path), "--page", "raw", "--csv"]
            raw_result = subprocess.run(
                raw_command,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REMOTE_WORKSPACE),
            )
            raw_csv_stdout = raw_result.stdout
            extraction_stderr += raw_result.stderr
            raw_csv_path.write_text(raw_csv_stdout)

            details_command = [ncu, "--import", str(report_path), "--page", "details"]
            details_result = subprocess.run(
                details_command,
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

    def download_profile_artifacts(
        workspace_layout: WorkspaceLayout,
        target: ProfileTarget,
        run_id: str,
        result: dict[str, str | int],
    ) -> tuple[list[Path], bool]:
        local_target_dir = (
            workspace_layout.profiles_dir
            / target.backend
            / Path(target.entry_file).stem
            / run_id
        )
        local_target_dir.mkdir(parents=True, exist_ok=True)
        remote_paths = [
            result["report_path"],
            result.get("raw_csv_path"),
            result.get("details_path"),
            result.get("summary_path"),
        ]
        downloaded_paths: list[Path] = []
        download_failed = False

        for remote_path in remote_paths:
            if not remote_path:
                continue
            volume_path = to_volume_path(str(remote_path))
            local_path = local_target_dir / Path(str(remote_path)).name
            download = subprocess.run(
                [
                    *MODAL_CLI,
                    "volume",
                    "get",
                    "--force",
                    PROFILE_VOLUME_NAME,
                    volume_path,
                    str(local_path),
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            if download.returncode != 0:
                download_failed = True
                print(f"FAILED to download {remote_path} to {local_path}")
                if download.stdout:
                    print(download.stdout)
                if download.stderr:
                    print(download.stderr)
                continue
            downloaded_paths.append(local_path)

        if not download_failed:
            for remote_path in remote_paths:
                if not remote_path:
                    continue
                volume_path = to_volume_path(str(remote_path))
                cleanup = subprocess.run(
                    [
                        *MODAL_CLI,
                        "volume",
                        "rm",
                        PROFILE_VOLUME_NAME,
                        volume_path,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                )
                if cleanup.returncode != 0 and cleanup.stderr:
                    print(cleanup.stderr)

        return downloaded_paths, download_failed

    def print_profile_result(
        workspace_layout: WorkspaceLayout,
        target: ProfileTarget,
        shape,
        result: dict[str, str | int],
        downloaded_paths: list[Path],
        download_failed: bool,
    ) -> None:
        shape_summary = (
            f"batch_size={shape.batch_size}"
            if target.problem_kind == "decode"
            else (
                f"total_seq_len={shape.total_seq_len}, "
                f"num_seqs={shape.num_seqs}, "
                f"cu_seqlens_len={shape.cu_seqlens_len}"
            )
        )
        print(
            f"Profiled workspace {workspace_layout.name}: "
            f"{target.backend} {target.problem_kind} "
            f"from {target.entry_file}::{target.entry_function}"
        )
        print(f"shape: {shape_summary}")
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
        if result["stderr"]:
            print(result["stderr"])
        if result["extraction_stderr"]:
            print(result["extraction_stderr"])

    def profile_shape(
        workspace_layout: WorkspaceLayout,
        target: ProfileTarget,
        *,
        shape,
        warmup: int,
        seed: int,
        ncu_set: str,
    ) -> None:
        from datetime import datetime
        from scripts.profile_utils import format_run_id

        run_id = format_run_id(target.build_target, shape, f"{datetime.now():%Y%m%d_%H%M%S}")

        result = run_profile.remote(
            target_json=json.dumps(
                {
                    "backend": target.backend,
                    "entry_file": target.entry_file,
                    "ncu_kernel_regex": target.ncu_kernel_regex,
                }
            ),
            source_files=read_profile_sources(target),
            runner_source=build_runner_source(
                target,
                shape,
                warmup=warmup,
                seed=seed,
            ),
            ncu_set=ncu_set,
            warmup=warmup,
            run_id=run_id,
        )
        downloaded_paths, download_failed = download_profile_artifacts(
            workspace_layout,
            target,
            run_id,
            result,
        )
        print_profile_result(
            workspace_layout,
            target,
            shape,
            result,
            downloaded_paths,
            download_failed,
        )

    @app.local_entrypoint()
    def main(
        workspace: str = ".",
        batch_size: int = 1,
        total_seq_len: int = 128,
        num_seqs: int = 4,
        cu_seqlens_len: int = 0,
        warmup: int = 5,
        seed: int = 7,
        ncu_set: str = "basic",
    ) -> None:
        from scripts.profile_utils import build_profile_shape
        from scripts.workspace_utils import resolve_workspace

        workspace_layout = resolve_workspace(workspace)
        target = resolve_profile_target(workspace_layout)
        shape = build_profile_shape(
            target.build_target,
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            num_seqs=num_seqs,
            cu_seqlens_len=cu_seqlens_len,
        )
        profile_shape(
            workspace_layout,
            target,
            shape=shape,
            warmup=warmup,
            seed=seed,
            ncu_set=ncu_set,
        )
