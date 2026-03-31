"""
Run FlashInfer's pure GPU timer on Modal B200.

This matches the reference repo's Modal CUPTI path rather than the benchmark
latency path used by scripts/run_modal.py.
"""

import sys
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    .add_local_file("config.toml", remote_path="/root/config.toml")
    .add_local_dir("solution", remote_path="/root/solution")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


app = modal.App("fi-timing-gdn")


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
    retries=2,
)
def run_fi_timing(workload_idx: int = 0):
    sys.path.insert(0, "/root")

    from scripts.bench_fi_timing import _count_workloads, get_trace_set_path, load_kernel, run_benchmark

    if workload_idx < 0:
        _, definition_name, _ = load_kernel()
        trace_root = get_trace_set_path()
        num_workloads = _count_workloads(trace_root, definition_name)
        print(f"Running timing for all {num_workloads} workloads")
        return [run_benchmark(workload_idx=idx) for idx in range(num_workloads)]

    return run_benchmark(workload_idx=workload_idx)


@app.local_entrypoint()
def main(workload_idx: int = 0):
    print(f"Running FlashInfer bench_gpu_time on Modal B200 (workload_idx={workload_idx})")
    result = run_fi_timing.remote(workload_idx)
    if workload_idx < 0:
        print(f"Completed Modal timing for {len(result)} workloads")
        return
    print(
        f"{result['solution_name']}: median={result['median_us']:.2f} us "
        f"(min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, n={result['iters']})"
    )
