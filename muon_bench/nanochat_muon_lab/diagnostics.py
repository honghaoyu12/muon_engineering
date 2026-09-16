"""Small, framework-independent diagnostics for Muon-family experiments.

These helpers define metrics used in EXPERIMENT_LADDER.md. They intentionally do not hook into
NanoChat automatically: instrumentation frequency and tensor capture can materially affect timing.
Use them on sampled tensors outside performance-critical windows or behind explicit profiler gates.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .muon_math import orthogonality_error, split_heads


def tensor_rms(x: Tensor) -> Tensor:
    """Root-mean-square magnitude ``sqrt(mean(x**2))``, computed in fp32."""
    if x.numel() == 0:
        raise ValueError("tensor_rms requires a non-empty tensor")
    return x.float().square().mean().sqrt()


def finite_summary(x: Tensor) -> dict[str, float | bool | int]:
    """Return basic finite-value diagnostics without changing the input tensor."""
    x32 = x.float()
    finite = torch.isfinite(x32)
    finite_count = int(finite.sum().item())
    total = x.numel()
    max_abs = float(x32[finite].abs().max().item()) if finite_count else float("nan")
    return {
        "all_finite": bool(finite_count == total),
        "finite_count": finite_count,
        "numel": int(total),
        "max_abs_finite": max_abs,
    }


def block_frobenius_stats(
    x: Tensor,
    *,
    num_heads: int,
    heads_per_group: int = 1,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Frobenius-norm dispersion across row-stacked attention head/group blocks.

    ``x`` may have arbitrary leading batch dimensions followed by ``(rows, cols)``. The function
    returns tensors over the leading batch dimensions. ``cv`` is population standard deviation
    divided by mean, not sample standard deviation.
    """
    blocks, _ = split_heads(x, num_heads=num_heads, heads_per_group=heads_per_group)
    norms = blocks.float().norm(dim=(-2, -1))
    mean = norms.mean(dim=-1)
    std = norms.std(dim=-1, unbiased=False)
    return {
        "mean": mean,
        "std": std,
        "cv": std / mean.clamp_min(eps),
        "min": norms.min(dim=-1).values,
        "max": norms.max(dim=-1).values,
    }


def polar_residual(x: Tensor) -> Tensor:
    """Dimensionless semi-orthogonality residual ``||Gram-I||_F / sqrt(r)``."""
    return orthogonality_error(x)


def selection_counts(indices: Tensor, selectable_size: int) -> Tensor:
    """Count how often each selectable coordinate appears in batched top-k indices.

    The returned length-``selectable_size`` int64 tensor aggregates every leading batch dimension
    and every selected coordinate in ``indices``. This is appropriate for starvation/coverage
    diagnostics; keep per-layer/per-matrix histograms separately when layer identity matters.
    """
    if selectable_size <= 0:
        raise ValueError("selectable_size must be positive")
    idx = indices.to(dtype=torch.int64).reshape(-1)
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= selectable_size):
        raise ValueError("selection index outside [0, selectable_size)")
    return torch.bincount(idx, minlength=selectable_size)[:selectable_size]
