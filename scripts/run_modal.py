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
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal


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
        "nvidia-cutlass",
    )
)


def _is_trace_root(path: Path) -> bool:
    """Return True if path looks like a flashinfer trace-set root."""
    return (path / "definitions").is_dir() and (path / "workloads").is_dir()


def _resolve_trace_set_path(base_path: str) -> Path:
    """Resolve mounted trace-set root (supports /data or /data/<dataset_dir>)."""
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


@app.function(image=image, gpu="B200:1", timeout=14400, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    config_toml: str,
    source_files: dict[str, str],
    config: dict | None = None,
) -> dict:
    """Pack the active workspace on Modal and run the benchmark there."""
    import tomllib as tomllib_remote

    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    config_data = tomllib_remote.loads(config_toml)
    build_config = config_data["build"]
    solution_config = config_data["solution"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for relative_path, content in source_files.items():
            output_path = Path(tmpdir) / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)

        language = build_config["language"]
        build_language = "triton" if language == "cute" else language
        spec = BuildSpec(
            language=build_language,
            target_hardware=["cuda"],
            entry_point=build_config["entry_point"],
            binding=build_config.get("binding"),
        )
        solution = pack_solution_from_files(
            path=tmpdir,
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )

    benchmark_config = BenchmarkConfig(**config) if config else BenchmarkConfig(
        warmup_runs=3,
        iterations=100,
        num_trials=5,
    )
    trace_set_path = _resolve_trace_set_path(TRACE_SET_PATH)
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        available = ", ".join(sorted(trace_set.definitions))
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set at '{trace_set_path}'. "
            f"Available definitions: {available or '<none>'}"
        )

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

    results["_solution_json"] = solution.model_dump_json(indent=2)
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


@app.local_entrypoint()
def main(workspace: str = ".") -> None:
    """Read one workspace locally and benchmark it on Modal."""
    from scripts.workspace_utils import read_workspace_tree, resolve_build_target, resolve_workspace

    workspace_layout = resolve_workspace(workspace)
    build_target = resolve_build_target(workspace_layout)
    source_root = build_target.source_dir.relative_to(workspace_layout.root).as_posix()
    source_files = {
        Path(relative_path).relative_to(source_root).as_posix(): content
        for relative_path, content in read_workspace_tree(workspace_layout, source_root).items()
    }
    config_toml = workspace_layout.config_path.read_text(encoding="utf-8")

    print(f"Read {len(source_files)} files from {source_root}/")
    print(f"Entry point: {build_target.entry_file}::{build_target.entry_function}")

    print("\nRunning pack + benchmark on Modal B200...")
    results = run_benchmark.remote(config_toml, source_files)
    if not results:
        print("No results returned!")
        return

    solution_json = results.pop("_solution_json", None)
    if solution_json is not None:
        workspace_layout.solution_json_path.write_text(solution_json, encoding="utf-8")
        print(f"\nSolution saved to {workspace_layout.solution_json_path}")

    print_results(results)
