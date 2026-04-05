"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

from scripts.workspace_utils import pack_workspace_solution, resolve_workspace


def _is_trace_root(path: Path) -> bool:
    """Return True if path looks like a flashinfer trace-set root."""
    return (path / "definitions").is_dir() and (path / "workloads").is_dir()


def get_trace_set_path() -> Path:
    """Get the trace-set root from FIB_DATASET_PATH."""
    raw_path = os.environ.get("FIB_DATASET_PATH")
    if not raw_path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )

    trace_root = Path(raw_path).expanduser()
    if not trace_root.exists():
        raise FileNotFoundError(f"Trace set path does not exist: {trace_root}")
    if _is_trace_root(trace_root):
        return trace_root
    for child in sorted(trace_root.iterdir()):
        if child.is_dir() and _is_trace_root(child):
            return child

    raise FileNotFoundError(
        f"Could not find flashinfer trace-set under '{trace_root}'. "
        "Expected 'definitions/' and 'workloads/' either there or one level below."
    )


def run_benchmark(solution: Solution, config: BenchmarkConfig | None = None) -> dict:
    """Run benchmark locally and return results."""
    benchmark_config = config or BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    trace_set = TraceSet.from_path(get_trace_set_path())

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    result_trace_set = Benchmark(bench_trace_set, benchmark_config).run_all(dump_traces=True)
    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

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

    return results


def print_results(results: dict) -> None:
    """Print benchmark results in a formatted way."""
    for definition_name, traces in results.items():
        print(f"\n{definition_name}:")
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


def main() -> None:
    """Pack one workspace locally and benchmark it."""
    parser = argparse.ArgumentParser(description="Run a workspace benchmark locally")
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace root relative to the repo root (default: .)",
    )
    args = parser.parse_args()

    workspace_layout = resolve_workspace(args.workspace)
    print(f"Packing solution from workspace: {workspace_layout.name}")
    solution_path = pack_workspace_solution(workspace_layout)

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    results = run_benchmark(solution)
    if not results:
        print("No results returned!")
        return
    print_results(results)


if __name__ == "__main__":
    main()
