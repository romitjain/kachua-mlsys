"""Profile the active config.toml kernel on Modal with Nsight Compute."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_WORKSPACE = Path("/workspace")
LOCAL_PROFILE_ROOT = PROJECT_ROOT / "profiles"
PROFILE_VOLUME_NAME = "kachua-gdn-profiles"


@dataclass(frozen=True)
class ProfileTarget:
    """Describe the implementation selected by config.toml."""

    backend: str
    entry_file: str
    entry_function: str
    ncu_kernel_regex: str
    source_files: tuple[str, ...]


def load_config() -> dict:
    """Load the repository config.toml."""
    with open(PROJECT_ROOT / "config.toml", "rb") as config_file:
        return tomllib.load(config_file)


def resolve_profile_target() -> ProfileTarget:
    """Resolve the active config.toml entry to a profile target."""
    build = load_config()["build"]
    entry_file, entry_function = build["entry_point"].split("::", maxsplit=1)
    language = build["language"]

    if language == "cuda":
        stem = Path(entry_file).stem
        if stem == "gdn_v1":
            kernel_regex = "regex:.*gdn_v1.*"
        elif stem == "gdn_v2":
            kernel_regex = "regex:.*gdn_v2.*"
        else:
            kernel_regex = "regex:.*gdn_decode_kernel.*"
        return ProfileTarget(
            backend="cuda",
            entry_file=entry_file,
            entry_function="kernel",
            ncu_kernel_regex=kernel_regex,
            source_files=("solution/cuda/binding.py", f"solution/cuda/{entry_file}"),
        )

    if language == "triton":
        return ProfileTarget(
            backend="triton",
            entry_file=entry_file,
            entry_function=entry_function,
            ncu_kernel_regex="regex:.*gdn_decode_kernel.*",
            source_files=(f"solution/triton/{entry_file}",),
        )

    raise ValueError(f"Unsupported language in config.toml: {language}")


def read_profile_sources(target: ProfileTarget) -> dict[str, str]:
    """Load only the files needed by the active target."""
    sources: dict[str, str] = {"config.toml": (PROJECT_ROOT / "config.toml").read_text()}
    for relative_path in target.source_files:
        sources[relative_path] = (PROJECT_ROOT / relative_path).read_text()
    return sources


def build_runner_source(target: ProfileTarget, batch_size: int, warmup: int, seed: int) -> str:
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


torch.manual_seed(SEED)
device = "cuda"
B = BATCH_SIZE
K_DIM = 128
V_DIM = 128
NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8

q = torch.randn(B, 1, NUM_Q_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
k = torch.randn(B, 1, NUM_K_HEADS, K_DIM, device=device, dtype=torch.bfloat16)
v = torch.randn(B, 1, NUM_V_HEADS, V_DIM, device=device, dtype=torch.bfloat16)
state = torch.randn(B, NUM_V_HEADS, V_DIM, K_DIM, device=device, dtype=torch.float32)
A_log = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
a = torch.randn(B, 1, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
dt_bias = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32)
b = torch.randn(B, 1, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
scale = 1.0 / (K_DIM ** 0.5)
"""

    if target.backend == "cuda":
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
run_kernel = getattr(binding, ENTRY_FUNCTION)

run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale)
torch.cuda.synchronize()

for _ in range(WARMUP):
    run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale)
torch.cuda.synchronize()

run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale)
torch.cuda.synchronize()
"""
    else:
        body = """
kernel_module = load_module(
    "triton_kernel",
    str(Path(WORKSPACE) / "solution/triton" / ENTRY_FILE),
)
run_kernel = getattr(kernel_module, ENTRY_FUNCTION)

def invoke():
    output = torch.empty((B, NUM_V_HEADS, V_DIM), dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)
    run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)

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
        ENTRY_FUNCTION = {target.entry_function!r}
        BATCH_SIZE = {batch_size}
        WARMUP = {warmup}
        SEED = {seed}
        """
    )
    return textwrap.dedent(prelude + base + body)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup launches before profiling")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic inputs")
    parser.add_argument("--ncu-set", default="basic", help="Nsight Compute metric set")
    return parser.parse_args()


def to_volume_path(remote_path: str) -> str:
    """Translate a container mount path into a Modal volume path."""
    path = Path(remote_path)
    if path.is_absolute() and path.parts[:2] == ("/", "profiles"):
        return "/" + str(Path(*path.parts[2:]))
    return remote_path


try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    app = modal.App("kachua-gdn-profile")
    profile_volume = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
    profile_dir = Path("/profiles")
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
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
    ) -> dict[str, str | int]:
        import csv
        import glob
        import io
        import math
        import subprocess
        from uuid import uuid4

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
            metrics: dict[str, dict[str, str]], names: tuple[str, ...]
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
                    metrics, ("launch__registers_per_thread",)
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
                    metrics, ("sm__warps_active.avg.pct_of_peak_sustained_active",)
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

            stall_metrics.sort(key=lambda item: float(item["numeric_value"]), reverse=True)
            summary["top_stall_metrics"] = stall_metrics[:10]
            return summary

        for relative_path, content in source_files.items():
            out_path = REMOTE_WORKSPACE / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content)

        runner_path = REMOTE_WORKSPACE / "runner.py"
        runner_path.write_text(runner_source)

        target_dir = profile_dir / target["backend"] / Path(target["entry_file"]).stem
        target_dir.mkdir(parents=True, exist_ok=True)
        report_stem = target_dir / uuid4().hex

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
            raw_command = [
                ncu,
                "--import",
                str(report_path),
                "--page",
                "raw",
                "--csv",
            ]
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

            details_command = [
                ncu,
                "--import",
                str(report_path),
                "--page",
                "details",
            ]
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

    @app.local_entrypoint()
    def main(
        batch_size: int = 1,
        warmup: int = 5,
        seed: int = 7,
        ncu_set: str = "basic",
    ) -> None:
        target = resolve_profile_target()
        result = run_profile.remote(
            target_json=json.dumps(
                {
                    "backend": target.backend,
                    "entry_file": target.entry_file,
                    "ncu_kernel_regex": target.ncu_kernel_regex,
                }
            ),
            source_files=read_profile_sources(target),
            runner_source=build_runner_source(target, batch_size=batch_size, warmup=warmup, seed=seed),
            ncu_set=ncu_set,
            warmup=warmup,
        )

        local_target_dir = LOCAL_PROFILE_ROOT / target.backend / Path(target.entry_file).stem
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
                        "uv",
                        "run",
                        "modal",
                        "volume",
                        "rm",
                        PROFILE_VOLUME_NAME,
                        volume_path,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )
                if cleanup.returncode != 0 and cleanup.stderr:
                    print(cleanup.stderr)

        print(f"Profiled {target.backend} from config entry {target.entry_file}::{target.entry_function}")
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
