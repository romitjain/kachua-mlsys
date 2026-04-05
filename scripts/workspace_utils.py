"""Workspace helpers for root-level development scripts."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
IGNORED_WORKSPACE_DIRS = {"__pycache__"}
IGNORED_WORKSPACE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class WorkspaceLayout:
    """Describe one submission workspace inside the repository."""

    repo_root: Path
    root: Path
    name: str
    config_path: Path
    solution_dir: Path
    scripts_dir: Path
    pack_solution_path: Path
    solution_json_path: Path
    profiles_dir: Path


@dataclass(frozen=True)
class BuildTarget:
    """Describe the active build target from one workspace config."""

    workspace: WorkspaceLayout
    solution_name: str
    definition_name: str
    language: str
    backend: str
    binding: str | None
    entry_file: str
    entry_function: str
    problem_kind: str
    source_dir: Path
    entry_path: Path


def _module_name(prefix: str, module_path: Path) -> str:
    digest = hashlib.sha1(
        str(module_path).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", module_path.stem)
    return f"kachua_{prefix}_{stem}_{digest}"


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _should_skip_workspace_path(workspace: WorkspaceLayout, path: Path) -> bool:
    relative_parts = path.relative_to(workspace.root).parts
    if any(part in IGNORED_WORKSPACE_DIRS for part in relative_parts):
        return True
    return path.suffix in IGNORED_WORKSPACE_SUFFIXES


def _validate_workspace_root(repo_root: Path, workspace_root: Path) -> None:
    try:
        workspace_root.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Workspace must stay inside the repository: {workspace_root}"
        ) from exc


def resolve_workspace(workspace: str | Path | None = None) -> WorkspaceLayout:
    """Resolve a workspace argument to a validated repository workspace."""
    raw_path = REPO_ROOT if workspace in (None, ".", "") else Path(workspace)
    workspace_root = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
    workspace_root = workspace_root.resolve()
    _validate_workspace_root(REPO_ROOT, workspace_root)
    _require_path(workspace_root, "Workspace")

    config_path = workspace_root / "config.toml"
    solution_dir = workspace_root / "solution"
    scripts_dir = workspace_root / "scripts"
    pack_solution_path = scripts_dir / "pack_solution.py"

    _require_path(config_path, "Workspace config.toml")
    _require_path(solution_dir, "Workspace solution directory")
    _require_path(pack_solution_path, "Workspace pack_solution.py")

    name = "."
    if workspace_root != REPO_ROOT:
        name = workspace_root.relative_to(REPO_ROOT).as_posix()

    return WorkspaceLayout(
        repo_root=REPO_ROOT,
        root=workspace_root,
        name=name,
        config_path=config_path,
        solution_dir=solution_dir,
        scripts_dir=scripts_dir,
        pack_solution_path=pack_solution_path,
        solution_json_path=workspace_root / "solution.json",
        profiles_dir=workspace_root / "profiles",
    )


def load_workspace_config(workspace: WorkspaceLayout) -> dict[str, Any]:
    """Load the config.toml for the selected workspace."""
    with open(workspace.config_path, "rb") as config_file:
        return tomllib.load(config_file)


def resolve_problem_kind(definition_name: str) -> str:
    """Resolve whether the active definition is decode or prefill."""
    if "_decode_" in definition_name:
        return "decode"
    if "_prefill_" in definition_name:
        return "prefill"
    raise ValueError(
        f"Unsupported definition '{definition_name}'. "
        "Expected a decode or prefill definition."
    )


def resolve_build_target(workspace: WorkspaceLayout) -> BuildTarget:
    """Resolve the active workspace config to one runnable build target."""
    config = load_workspace_config(workspace)
    build = config["build"]
    solution = config["solution"]
    language = build["language"]
    entry_file, entry_function = build["entry_point"].split("::", maxsplit=1)
    source_dir = workspace.solution_dir / language
    entry_path = source_dir / entry_file

    _require_path(source_dir, f"Workspace {language} source directory")
    _require_path(entry_path, "Workspace entry file")

    return BuildTarget(
        workspace=workspace,
        solution_name=solution["name"],
        definition_name=solution["definition"],
        language=language,
        backend=language,
        binding=build.get("binding"),
        entry_file=entry_file,
        entry_function=entry_function,
        problem_kind=resolve_problem_kind(solution["definition"]),
        source_dir=source_dir,
        entry_path=entry_path,
    )


def load_module_from_path(prefix: str, module_path: Path) -> ModuleType:
    """Load a Python module from an explicit file path."""
    module_name = _module_name(prefix, module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def pack_workspace_solution(workspace: WorkspaceLayout) -> Path:
    """Pack one workspace by delegating to its local pack_solution.py."""
    module = load_module_from_path("pack_solution", workspace.pack_solution_path)
    solution_path = module.pack_solution()
    return Path(solution_path)


def read_workspace_sources(
    workspace: WorkspaceLayout,
    relative_paths: tuple[str, ...],
) -> dict[str, str]:
    """Read a selected set of workspace files into a string map."""
    sources: dict[str, str] = {}
    for relative_path in relative_paths:
        sources[relative_path] = (workspace.root / relative_path).read_text(encoding="utf-8")
    return sources


def read_workspace_tree(workspace: WorkspaceLayout, relative_root: str) -> dict[str, str]:
    """Read one workspace subtree into a string map."""
    root = workspace.root / relative_root
    _require_path(root, f"Workspace subtree '{relative_root}'")
    sources: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if _should_skip_workspace_path(workspace, path):
            continue
        sources[str(path.relative_to(workspace.root))] = path.read_text(encoding="utf-8")
    return sources
