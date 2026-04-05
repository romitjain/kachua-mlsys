"""
Run FlashInfer's pure GPU timer on Modal B200.

This matches the reference repo's Modal CUPTI path rather than the benchmark
latency path used by scripts/run_modal.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from scripts.workspace_utils import read_workspace_sources, read_workspace_tree, resolve_workspace


trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "flashinfer-bench",
        "torch",
        "triton",
        "flashinfer-python",
        "pandas",
        "cupti-python",
        "python-dotenv",
    )
    .env({"FIB_DATASET_PATH": TRACE_SET_PATH})
    .add_local_dir("scripts", remote_path="/root/scripts")
)


app = modal.App("fi-timing-gdn")


def build_workspace_files(workspace: str) -> dict[str, str]:
    """Serialize one selected workspace for remote Modal timing."""
    workspace_layout = resolve_workspace(workspace)
    files = read_workspace_sources(
        workspace_layout,
        ("config.toml", "scripts/pack_solution.py"),
    )
    files.update(read_workspace_tree(workspace_layout, "solution"))
    return files


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
    retries=2,
)
def run_fi_timing(workspace_files: dict[str, str], workload_idx: int = 0):
    sys.path.insert(0, "/root")

    from scripts.bench_fi_timing import _count_workloads, get_trace_set_path, load_kernel, run_benchmark
    from scripts.workspace_utils import resolve_workspace

    remote_workspace_root = Path("/root/workspace")
    for relative_path, content in workspace_files.items():
        output_path = remote_workspace_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    workspace_layout = resolve_workspace(remote_workspace_root)
    _, definition_name, runtime = load_kernel(workspace_layout)
    if workload_idx < 0:
        trace_root = get_trace_set_path(workspace_layout)
        num_workloads = _count_workloads(trace_root, definition_name)
        print(f"Running timing for all {num_workloads} workloads from workspace {workspace_layout.name}")
        return [
            run_benchmark(
                workspace=workspace_layout,
                workload_idx=index,
                runtime=runtime,
            )
            for index in range(num_workloads)
        ]

    return run_benchmark(
        workspace=workspace_layout,
        workload_idx=workload_idx,
        runtime=runtime,
    )


@app.local_entrypoint()
def main(workspace: str = ".", workload_idx: int = 0) -> None:
    workspace_layout = resolve_workspace(workspace)
    print(
        "Running FlashInfer bench_gpu_time on Modal B200 "
        f"for workspace {workspace_layout.name} (workload_idx={workload_idx})"
    )
    result = run_fi_timing.remote(build_workspace_files(workspace), workload_idx)
    if workload_idx < 0:
        print(f"Completed Modal timing for {len(result)} workloads")
        return
    print(
        f"{result['solution_name']}: median={result['median_us']:.2f} us "
        f"(min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, n={result['iters']})"
    )
