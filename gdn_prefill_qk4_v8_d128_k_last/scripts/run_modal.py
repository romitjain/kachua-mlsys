"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("flashinfer-bench")

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
    )
)


def _is_trace_root(path: Path) -> bool:
    return (path / "definitions").is_dir() and (path / "workloads").is_dir()


def _resolve_trace_set_path(base_path: str) -> Path:
    base = Path(base_path)
    if _is_trace_root(base):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and _is_trace_root(child):
            return child
    raise FileNotFoundError(
        f"Could not find flashinfer trace-set under '{base}'. "
        "Expected 'definitions/' and 'workloads/' either at mount root "
        "or one level below it."
    )


def read_source_files() -> dict[str, str]:
    """Read config.toml and solution source files into a dict."""
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    language = config["build"]["language"]
    source_dir = PROJECT_ROOT / "solution" / language
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = {"config.toml": config_path.read_text()}
    for src in sorted(source_dir.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts:
            rel = str(src.relative_to(PROJECT_ROOT))
            files[rel] = src.read_text()
    return files


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    config_toml: str,
    source_files: dict[str, str],
) -> dict:
    """Pack solution on Modal and run benchmark on B200."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    config = tomllib.loads(config_toml)
    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    workspace = Path("/workspace")
    source_dir = workspace / "solution" / language
    source_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in source_files.items():
        if rel_path.startswith(f"solution/{language}/"):
            out = workspace / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)

    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=build_config["entry_point"],
        binding=build_config.get("binding"),
    )
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    bench_config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    trace_set_path = _resolve_trace_set_path(TRACE_SET_PATH)
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        available = ", ".join(sorted(trace_set.definitions))
        raise ValueError(
            f"Definition '{solution.definition}' not found. Available: {available}"
        )

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for '{solution.definition}'")

    import time
    total = len(workloads)
    print(f"[bench] Packed solution: {solution.name} ({solution.definition})")
    print(f"[bench] Found {total} workloads. Running one-at-a-time with progress...")
    results = {definition.name: {}}
    t0 = time.time()

    for wi, workload in enumerate(workloads):
        wl_start = time.time()
        single_trace_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: [workload]},
            traces={definition.name: []},
        )
        benchmark = Benchmark(single_trace_set, bench_config)
        result_trace_set = benchmark.run_all(dump_traces=True)
        wl_elapsed = time.time() - wl_start

        traces = result_trace_set.traces.get(definition.name, [])
        for trace in traces:
            if not trace.evaluation:
                continue
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

            status = entry["status"]
            latency = f"{entry.get('latency_ms', 0) * 1000:.1f}µs" if entry.get("latency_ms") else "?"
            speedup = f"{entry.get('speedup_factor', 0):.1f}x" if entry.get("speedup_factor") else "?"
            print(f"[bench] [{wi+1}/{total}] {status} | {latency} | {speedup} ({wl_elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"[bench] Done. {total} workloads in {total_elapsed:.1f}s")

    return results



def print_results(results: dict):
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")
            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms'] * 1000:.3f} µs", end="")
            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")
            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")
            print()


@app.local_entrypoint()
def main():
    """Read source files locally, pack and benchmark on Modal."""
    source_files = read_source_files()
    config_toml = source_files["config.toml"]

    print("Sending source files to Modal for packing + benchmark...")
    results = run_benchmark.remote(config_toml, source_files)

    if not results:
        print("No results returned!")
        return
    print_results(results)
