"""Pack and benchmark the active submission on Modal B200 from macOS or Linux."""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_VOLUME_NAME = "flashinfer-trace"
TRACE_SET_PATH = "/data"

app = modal.App("flashinfer-bench")
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",
).uv_pip_install(
    "cupti-python",
    "flashinfer-bench",
    "flashinfer-python",
    "nvidia-cutlass",
    "pandas",
    "torch",
    "triton",
)


def _is_trace_root(path: Path) -> bool:
    """Return True if path looks like a flashinfer trace-set root."""
    return (path / "definitions").is_dir() and (path / "workloads").is_dir()


def _resolve_trace_set_path(base_path: str | Path) -> Path:
    """Resolve the mounted trace-set root from /data or /data/<dataset_dir>."""
    base = Path(base_path)
    if _is_trace_root(base):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and _is_trace_root(child):
            return child
    raise FileNotFoundError(
        f"Could not find flashinfer trace-set under '{base}'. "
        "Expected 'definitions/' and 'workloads/' either at mount root or one level below."
    )


def _read_source_files(source_dir: Path) -> dict[str, str]:
    """Read all text files in the active solution directory."""
    source_files: dict[str, str] = {}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        source_files[str(path.relative_to(source_dir))] = path.read_text()
    return source_files


def _source_dir_for_language(language: str) -> Path:
    """Return the local source directory for the selected language."""
    if language not in {"cuda", "triton", "cute"}:
        raise ValueError(f"Unsupported language: {language}")
    return PROJECT_ROOT / "solution" / language


def _benchmark_config_for_definition(definition: str):
    """Return the contest-like benchmark config for the selected definition."""
    from flashinfer_bench import BenchmarkConfig

    if "prefill" in definition:
        return BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=3)
    return BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(config_toml: str, source_files: dict[str, str]) -> dict:
    """Pack the source remotely and run the benchmark on Modal."""
    from flashinfer_bench import Benchmark, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    config = tomllib.loads(config_toml)
    build_config = config["build"]
    solution_config = config["solution"]

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        for relative_path, content in source_files.items():
            temp_path = temp_root / relative_path
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(content)

        language = build_config["language"]
        build_language = "triton" if language == "cute" else language
        solution = pack_solution_from_files(
            path=tmpdir,
            spec=BuildSpec(
                language=build_language,
                target_hardware=["cuda"],
                entry_point=build_config["entry_point"],
                binding=build_config.get("binding"),
            ),
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )

    print(f"Packed: {solution.name} ({solution.definition})")
    trace_set = TraceSet.from_path(_resolve_trace_set_path(TRACE_SET_PATH))
    if solution.definition not in trace_set.definitions:
        available = ", ".join(sorted(trace_set.definitions))
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set. "
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
    benchmark = Benchmark(bench_trace_set, _benchmark_config_for_definition(solution.definition))
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}
    for trace in traces:
        if trace.evaluation is None:
            continue
        entry = {"status": trace.evaluation.status.value, "solution": trace.solution}
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
    """Print benchmark results in a compact form."""
    for definition_name, traces in results.items():
        print(f"\n{definition_name}:")
        for workload_uuid, result in traces.items():
            print(f"  Workload {workload_uuid[:8]}...: {result.get('status')}", end="")
            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms'] * 1000:.3f} µs", end="")
            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")
            if result.get("max_abs_error") is not None:
                print(
                    " | "
                    f"abs_err={result['max_abs_error']:.2e}, "
                    f"rel_err={result.get('max_rel_error', 0):.2e}",
                    end="",
                )
            print()


@app.local_entrypoint()
def main() -> None:
    """Read config + sources locally, then pack and benchmark remotely."""
    config_path = PROJECT_ROOT / "config.toml"
    config_toml = config_path.read_text()
    config = tomllib.loads(config_toml)
    language = config["build"]["language"]
    source_dir = _source_dir_for_language(language)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_files = _read_source_files(source_dir)
    print(f"Read {len(source_files)} files from solution/{language}/")
    print(f"Entry point: {config['build']['entry_point']}")
    print("\nRunning pack + benchmark on Modal B200...")
    results = run_benchmark.remote(config_toml, source_files)
    if not results:
        print("No results returned!")
        return

    solution_json = results.pop("_solution_json", None)
    if solution_json:
        output_path = PROJECT_ROOT / "solution.json"
        output_path.write_text(solution_json)
        print(f"\nSolution saved to {output_path}")

    print_results(results)
