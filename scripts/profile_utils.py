"""Shared profiling helpers for local and Modal script entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.workspace_utils import BuildTarget

if TYPE_CHECKING:
    import torch


K_DIM = 128
V_DIM = 128
NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8


@dataclass(frozen=True)
class ProfileShape:
    """Describe the synthetic shape used for one profiling run."""

    batch_size: int
    total_seq_len: int
    num_seqs: int
    cu_seqlens_len: int
def require_positive(name: str, value: int) -> None:
    """Require a positive integer CLI argument."""
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def resolve_cu_seqlens_len(num_seqs: int, cu_seqlens_len: int) -> int:
    """Resolve and validate the prefill cu_seqlens length."""
    expected = num_seqs + 1
    if cu_seqlens_len <= 0:
        return expected
    if cu_seqlens_len != expected:
        raise ValueError(
            "cu_seqlens_len must equal num_seqs + 1 for packed prefill inputs. "
            f"Expected {expected}, got {cu_seqlens_len}."
        )
    return cu_seqlens_len


def build_profile_shape(
    target: BuildTarget,
    *,
    batch_size: int,
    total_seq_len: int,
    num_seqs: int,
    cu_seqlens_len: int,
) -> ProfileShape:
    """Build the synthetic shape for the selected decode or prefill target."""
    resolved_len = resolve_cu_seqlens_len(num_seqs, cu_seqlens_len)
    if target.problem_kind == "decode":
        require_positive("batch_size", batch_size)
        return ProfileShape(
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            num_seqs=num_seqs,
            cu_seqlens_len=resolved_len,
        )

    require_positive("total_seq_len", total_seq_len)
    require_positive("num_seqs", num_seqs)
    return ProfileShape(
        batch_size=batch_size,
        total_seq_len=total_seq_len,
        num_seqs=num_seqs,
        cu_seqlens_len=resolved_len,
    )


def format_run_id(target: BuildTarget, shape: ProfileShape, timestamp: str) -> str:
    """Format a readable run id for one profiling invocation."""
    if target.problem_kind == "decode":
        return f"{timestamp}_b{shape.batch_size}"
    return (
        f"{timestamp}_t{shape.total_seq_len}_n{shape.num_seqs}_c{shape.cu_seqlens_len}"
    )


def build_prefill_cu_seqlens(
    total_seq_len: int,
    num_seqs: int,
    expected_len: int,
    device: str,
) -> torch.Tensor:
    """Build a balanced packed cu_seqlens vector for synthetic prefill profiling."""
    import torch

    actual_len = num_seqs + 1
    if expected_len != actual_len:
        raise ValueError(
            f"cu_seqlens length mismatch: expected {actual_len}, got {expected_len}"
        )

    lengths = torch.full((num_seqs,), total_seq_len // num_seqs, dtype=torch.int64)
    lengths[: total_seq_len % num_seqs] += 1
    cu_seqlens = torch.zeros(actual_len, dtype=torch.int64, device=device)
    cu_seqlens[1:] = torch.cumsum(lengths.to(device), dim=0)
    return cu_seqlens


def build_profile_inputs(
    target: BuildTarget,
    shape: ProfileShape,
    *,
    seed: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor | float | None]:
    """Build synthetic decode or prefill inputs for local profiling."""
    import torch

    torch.manual_seed(seed)
    scale = 1.0 / (K_DIM**0.5)

    if target.problem_kind == "decode":
        q = torch.randn(
            shape.batch_size,
            1,
            NUM_Q_HEADS,
            K_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            shape.batch_size,
            1,
            NUM_K_HEADS,
            K_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            shape.batch_size,
            1,
            NUM_V_HEADS,
            V_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        state = torch.randn(
            shape.batch_size,
            NUM_V_HEADS,
            V_DIM,
            K_DIM,
            device=device,
            dtype=torch.float32,
        )
        a = torch.randn(
            shape.batch_size,
            1,
            NUM_V_HEADS,
            device=device,
            dtype=torch.bfloat16,
        )
        b = torch.randn(
            shape.batch_size,
            1,
            NUM_V_HEADS,
            device=device,
            dtype=torch.bfloat16,
        )
        return {
            "q": q,
            "k": k,
            "v": v,
            "state": state,
            "A_log": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32),
            "a": a,
            "dt_bias": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32),
            "b": b,
            "cu_seqlens": None,
            "scale": scale,
        }

    q = torch.randn(
        shape.total_seq_len,
        NUM_Q_HEADS,
        K_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        shape.total_seq_len,
        NUM_K_HEADS,
        K_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        shape.total_seq_len,
        NUM_V_HEADS,
        V_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        shape.num_seqs,
        NUM_V_HEADS,
        V_DIM,
        K_DIM,
        device=device,
        dtype=torch.float32,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32),
        "a": torch.randn(
            shape.total_seq_len,
            NUM_V_HEADS,
            device=device,
            dtype=torch.bfloat16,
        ),
        "dt_bias": torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32),
        "b": torch.randn(
            shape.total_seq_len,
            NUM_V_HEADS,
            device=device,
            dtype=torch.bfloat16,
        ),
        "cu_seqlens": build_prefill_cu_seqlens(
            shape.total_seq_len,
            shape.num_seqs,
            shape.cu_seqlens_len,
            device,
        ),
        "scale": scale,
    }
