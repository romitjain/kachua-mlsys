"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

All flashinfer_bench imports happen on the remote Modal container (Linux + CUDA),
so this script can be launched from macOS via `uv run modal run scripts/run_modal.py`.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import modal
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(source_files: dict, build_config: dict, solution_config: dict) -> dict:
    """Pack solution + run benchmark on Modal B200.

    All flashinfer_bench imports happen here (remote Linux container).
    """
    import tempfile

    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    # Write source files to a temp directory and pack
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in source_files.items():
            fpath = Path(tmpdir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        spec = BuildSpec(
            language=build_config["language"],
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

    print(f"Packed: {solution.name} ({solution.definition})")

    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

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

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
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


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
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


def _read_source_files(source_dir: Path) -> dict:
    """Read all files from the source directory into a dict."""
    source_files = {}
    for f in sorted(source_dir.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(source_dir))
            source_files[rel] = f.read_text()
    return source_files


LANGUAGE_DIRS = {"triton": "triton", "cuda": "cuda"}


@app.local_entrypoint()
def main():
    """Read source files locally, pack + benchmark on Modal."""
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    build_config = config["build"]
    solution_config = config["solution"]
    language = build_config["language"]

    dir_name = LANGUAGE_DIRS.get(language)
    if dir_name is None:
        raise ValueError(f"Unsupported language: {language}")

    source_dir = PROJECT_ROOT / "solution" / dir_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_files = _read_source_files(source_dir)
    print(f"Read {len(source_files)} files from solution/{dir_name}/")

    print("\nRunning pack + benchmark on Modal B200...")
    results = run_benchmark.remote(source_files, build_config, solution_config)

    if not results:
        print("No results returned!")
        return

    solution_json = results.pop("_solution_json", None)
    if solution_json:
        output_path = PROJECT_ROOT / "solution.json"
        output_path.write_text(solution_json)
        print(f"\nSolution saved to {output_path}")

    print_results(results)
