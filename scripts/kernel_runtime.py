"""Backend-agnostic kernel loaders for development scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from scripts.workspace_utils import (
    BuildTarget,
    WorkspaceLayout,
    load_module_from_path,
    resolve_build_target,
)


KernelRunner = Callable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class KernelRuntime:
    """Describe one loaded kernel implementation."""

    target: BuildTarget
    runner: KernelRunner


def _allocate_decode_outputs(
    q: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = q.shape[0]
    num_v_heads = v.shape[2]
    v_dim = v.shape[3]
    output = torch.empty((batch_size, num_v_heads, v_dim), dtype=v.dtype, device=q.device)
    return output, torch.empty_like(state)


def _allocate_prefill_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_seq_len = q.shape[0]
    num_v_heads = v.shape[1]
    v_dim = v.shape[2]
    k_dim = k.shape[2]
    num_seqs = int(cu_seqlens.numel() - 1)
    output = torch.empty((total_seq_len, num_v_heads, v_dim), dtype=v.dtype, device=q.device)
    if state is not None:
        return output, torch.empty_like(state)
    new_state = torch.empty(
        (num_seqs, num_v_heads, v_dim, k_dim),
        dtype=torch.float32,
        device=q.device,
    )
    return output, new_state


def _load_cuda_torch_runner(target: BuildTarget) -> KernelRunner:
    binding_path = target.source_dir / "binding.py"
    binding_module = load_module_from_path("cuda_binding", binding_path)
    return binding_module.kernel


def _load_triton_like_runner(target: BuildTarget) -> KernelRunner:
    kernel_module = load_module_from_path(f"{target.language}_kernel", target.entry_path)
    entry_function = getattr(kernel_module, target.entry_function)

    if target.problem_kind == "decode":
        def run_decode(q, k, v, state, A_log, a, dt_bias, b, scale=None):
            output, new_state = _allocate_decode_outputs(q, v, state)
            entry_function(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)
            return output, new_state

        return run_decode

    def run_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None):
        output, new_state = _allocate_prefill_outputs(q, k, v, state, cu_seqlens)
        entry_function(
            q,
            k,
            v,
            state,
            A_log,
            a,
            dt_bias,
            b,
            cu_seqlens,
            scale,
            output,
            new_state,
        )
        return output, new_state

    return run_prefill


def load_kernel_runtime(workspace: WorkspaceLayout) -> KernelRuntime:
    """Load the active kernel implementation for one workspace."""
    target = resolve_build_target(workspace)

    if target.language == "cuda":
        if target.binding != "torch":
            raise NotImplementedError(
                "Workspace-aware timing/profile helpers currently support CUDA torch "
                f"binding only, got binding={target.binding!r}."
            )
        return KernelRuntime(target=target, runner=_load_cuda_torch_runner(target))

    if target.language in {"triton", "cute"}:
        return KernelRuntime(target=target, runner=_load_triton_like_runner(target))

    raise NotImplementedError(f"Unsupported workspace language: {target.language}")
