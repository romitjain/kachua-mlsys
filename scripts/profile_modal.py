"""Profile the active config.toml kernel on Modal with Nsight Compute."""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_WORKSPACE = Path("/workspace")


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


stub_tvm_ffi = types.ModuleType("tvm_ffi")
stub_tvm_ffi.register_global_func = lambda _name: (lambda function: function)
sys.modules.setdefault("tvm_ffi", stub_tvm_ffi)


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


try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    app = modal.App("kachua-gdn-profile")
    profile_volume = modal.Volume.from_name("kachua-gdn-profiles", create_if_missing=True)
    profile_dir = Path("/profiles")
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .uv_pip_install("flashinfer-bench", "flashinfer-python", "torch", "triton")
        .apt_install("wget", "gnupg", "build-essential", "ninja-build")
        .run_commands(
            "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/"
            "ubuntu2204/x86_64/3bf863cc.pub | "
            "gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
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
        import glob
        import subprocess
        from uuid import uuid4

        target = json.loads(target_json)

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
        profile_volume.commit()
        return {
            "backend": target["backend"],
            "entry_file": target["entry_file"],
            "exit_code": result.returncode,
            "report_path": str(report_stem.with_suffix(".ncu-rep")),
            "stdout": result.stdout,
            "stderr": result.stderr,
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

        print(f"Profiled {target.backend} from config entry {target.entry_file}::{target.entry_function}")
        print(f"ncu exit code: {result['exit_code']}")
        print(f"report saved to Modal Volume at {result['report_path']}")
        if result["stdout"]:
            print(result["stdout"])
        if result["stderr"]:
            print("STDERR:")
            print(result["stderr"])
