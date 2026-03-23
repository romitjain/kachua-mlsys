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

from scripts.modal_config import TRACE_SET_PATH, image, trace_volume


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

    from scripts.bench_fi_timing import run_benchmark

    return run_benchmark(trace_set_path=TRACE_SET_PATH, workload_idx=workload_idx)


@app.local_entrypoint()
def main(workload_idx: int = 0):
    print(f"Running FlashInfer bench_gpu_time on Modal B200 (workload_idx={workload_idx})")
    result = run_fi_timing.remote(workload_idx)
    print(
        f"{result['solution_name']}: median={result['median_us']:.2f} us "
        f"(min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, n={result['iters']})"
    )
