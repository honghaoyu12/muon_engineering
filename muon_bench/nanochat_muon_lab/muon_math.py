"""Pure-PyTorch Muon-family building blocks used by the NanoChat benchmark package.

This module deliberately contains no NanoChat-specific code.  It is therefore useful for:
  * unit tests on CPU;
  * numerical audits against reference implementations;
  * understanding exactly which transformation each experiment enables.

The high-performance Hopper/Blackwell path can optionally call the official
``gram-newton-schulz`` package from ``research_optimizer.py``.  The GNS function here is a
correctness/reference implementation of the algorithm described by Dao-AILab; it is NOT meant
as the final performance kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
from torch import Tensor

# Keller-Jordan's quintic coefficients, repeated for the requested number of steps.
KELLER_JORDAN_COEFF = (3.4445, -4.7750, 2.0315)

# Current NanoChat Polar Express coefficients (5-step schedule, as of 2026-09-16).
POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

# Five-step NS coefficient sequence used by current Microsoft Dion Muon and historical
# modded-nanogpt-style Muon recipes. The official GNS README also uses this sequence in an
# autotuning example, but it is NOT the same thing as the Polar Express coefficient schedule.
DION_NS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

MuonPlusMode = Literal["none", "frob_snap", "row", "col", "row_col", "col_row"]


def coefficients(name: str, ns_steps: int = 5) -> tuple[tuple[float, float, float], ...]:
    """Return the polynomial coefficient schedule requested by an experiment."""
    if ns_steps < 1:
        raise ValueError(f"ns_steps must be >=1, got {ns_steps}")
    if name == "keller":
        return tuple(KELLER_JORDAN_COEFF for _ in range(ns_steps))
    if name in {"polar_express", "nanochat_pe"}:
        if ns_steps > len(POLAR_EXPRESS_COEFFS):
            raise ValueError(
                f"nanochat_pe is a fixed 5-step schedule; requested ns_steps={ns_steps}. "
                "Do not invent additional steps by repeating its final polynomial."
            )
        return POLAR_EXPRESS_COEFFS[:ns_steps]
    if name in {"dion_ns", "modded_ns", "dao_gns_pe", "gns_pe"}:
        # The last two names are legacy aliases kept only for old configs. New configs should
        # use ``dion_ns``. This is a fixed five-step schedule, not an indefinitely repeatable one.
        if ns_steps > len(DION_NS_COEFFS):
            raise ValueError(
                f"dion_ns is a fixed 5-step schedule; requested ns_steps={ns_steps}. "
                "Do not invent additional steps by repeating its final polynomial."
            )
        return DION_NS_COEFFS[:ns_steps]
    raise ValueError(f"Unknown coefficient schedule: {name!r}")


def _ensure_matrix_batch(x: Tensor) -> None:
    if x.ndim < 2:
        raise ValueError(f"Muon transformation requires at least 2 dimensions; got {x.shape}")


def standard_newton_schulz(
    x: Tensor,
    coeffs: Sequence[tuple[float, float, float]],
    *,
    eps: float = 1e-7,
    safety_factor: float = 1.0,
    norm_in_fp32: bool = True,
) -> Tensor:
    """Reference standard Newton--Schulz / Polar-Express polynomial evaluation.

    ``x`` may have arbitrary batch dimensions followed by ``(m, n)``.  The smaller Gram matrix
    is used: tall matrices are transposed internally.  ``safety_factor`` is exposed because
    current NanoChat normalizes by ``1.01 * ||X||_F + eps`` while the original Keller reference
    normally uses a direct Frobenius normalization.  For an attribution-clean KJ baseline use 1.0.

    ``norm_in_fp32=False`` reproduces the reference Keller/NanoChat finite-precision behavior after
    casting the NS working tensor to bf16: the Frobenius norm itself is then evaluated from the
    bf16 tensor rather than silently upgrading the reference path to fp32.
    """
    _ensure_matrix_batch(x)
    was_tall = x.shape[-2] > x.shape[-1]
    y = x.mT if was_tall else x
    norm_source = y.float() if norm_in_fp32 else y
    y = y / (norm_source.norm(dim=(-2, -1), keepdim=True).to(y.dtype) * safety_factor + eps)
    for a, b, c in coeffs:
        gram = y @ y.mT
        poly = b * gram + c * (gram @ gram)
        y = a * y + poly @ y
    return y.mT if was_tall else y


def gram_newton_schulz_reference(
    x: Tensor,
    coeffs: Sequence[tuple[float, float, float]],
    *,
    eps: float = 1e-7,
    reset_after: Iterable[int] = (2,),
) -> Tensor:
    """Pure-PyTorch reference implementation of Gram Newton--Schulz (GNS).

    This follows the Dao-AILab algorithm.  ``reset_after=(2,)`` means: after two polynomial
    iterations, materialize ``Q @ X``, recompute its Gram matrix, and reset ``Q`` to identity.
    This is the restart recommended for a five-step half-precision GNS run.

    This reference intentionally uses ordinary PyTorch matmuls.  The *performance* experiment
    should use the official ``gram-newton-schulz`` kernels on supported GPUs.
    """
    _ensure_matrix_batch(x)
    reset_after = set(int(i) for i in reset_after)
    was_tall = x.shape[-2] > x.shape[-1]
    y = x.mT if was_tall else x
    y = y / (y.float().norm(dim=(-2, -1), keepdim=True).to(y.dtype) + eps)

    n = y.shape[-2]
    gram = y @ y.mT
    eye = torch.eye(n, dtype=y.dtype, device=y.device)
    # Broadcast identity over any batch dimensions without allocating a separate copy per matrix.
    q = eye.expand(*y.shape[:-2], n, n).clone()

    completed = 0
    for a, b, c in coeffs:
        if completed in reset_after and completed > 0:
            y = q @ y
            gram = y @ y.mT
            q = eye.expand(*y.shape[:-2], n, n).clone()

        gram2 = gram @ gram
        z = b * gram + c * gram2
        q = q @ z + a * q
        rz = gram @ z + a * gram
        gram = z @ rz + a * rz
        completed += 1

    y = q @ y
    return y.mT if was_tall else y


def muoneq_rows(x: Tensor, eps: float = 1e-6) -> Tensor:
    """MuonEq-R style transient row equilibration used before the polar map.

    Every row is rescaled to the RMS row norm implied by the matrix Frobenius norm:
        target = ||X||_F / sqrt(number_of_rows).
    This transformation has no persistent optimizer state.
    """
    _ensure_matrix_batch(x)
    target = x.float().norm(dim=(-2, -1), keepdim=True) / (x.shape[-2] ** 0.5)
    row_norm = x.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    return x * (target / row_norm).to(x.dtype)


def frobenius_snap(x: Tensor, eps: float = 1e-6) -> Tensor:
    """NanoChat's Muon+-inspired global Frobenius correction.

    An exactly semi-orthogonal ``m x n`` matrix has Frobenius norm ``sqrt(min(m,n))``.
    """
    target = min(x.shape[-2], x.shape[-1]) ** 0.5
    current = x.float().norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return x * (target / current).to(x.dtype)


def muonplus_normalize(x: Tensor, mode: MuonPlusMode, eps: float = 1e-8) -> Tensor:
    """Post-polar Muon+ normalization modes.

    ``frob_snap`` is the light-weight NanoChat-inspired correction.  ``row``/``col`` and their
    compositions implement the normalization families used in Muon+ experiments.  We perform the
    norm calculation in fp32 and cast back, avoiding bf16 underflow in small row/column norms.
    """
    if mode == "none":
        return x
    if mode == "frob_snap":
        return frobenius_snap(x, eps=max(eps, 1e-6))

    y = x.float()

    def row_norm(z: Tensor) -> Tensor:
        return z / (z.square().sum(dim=-1, keepdim=True) + eps).sqrt()

    def col_norm(z: Tensor) -> Tensor:
        return z / (z.square().sum(dim=-2, keepdim=True) + eps).sqrt()

    if mode == "row":
        y = row_norm(y)
    elif mode == "col":
        y = col_norm(y)
    elif mode == "row_col":
        y = col_norm(row_norm(y))
    elif mode == "col_row":
        y = row_norm(col_norm(y))
    else:
        raise ValueError(f"Unknown Muon+ normalization mode: {mode}")
    return y.to(x.dtype)


def normuon_rows(
    update: Tensor,
    variance: Tensor,
    beta2: float,
    *,
    eps: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """NorMuon-style per-row second-moment normalization with norm preservation.

    ``variance`` must be broadcast-compatible with ``update[..., :, :1]`` and is updated in fp32.
    The transformed update is globally rescaled to preserve the *pre-normalization* Frobenius norm.
    """
    u32 = update.float()
    v_mean = u32.square().mean(dim=-1, keepdim=True)
    new_variance = variance.float().lerp(v_mean, 1.0 - beta2)
    step_scale = new_variance.clamp_min(eps).rsqrt()
    scaled = u32 * step_scale
    old_norm = u32.norm(dim=(-2, -1), keepdim=True)
    new_norm = scaled.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    scaled = scaled * (old_norm / new_norm)
    return scaled.to(update.dtype), new_variance.to(variance.dtype)


def factored_normuon(
    update: Tensor,
    variance: Tensor,
    beta2: float,
    *,
    reduction_dim: Literal[-1, -2],
    eps: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """NanoChat-style factored NorMuon normalization.

    This exactly captures the shape-aware idea in current NanoChat: keep one second-moment scalar
    per row for tall matrices or per column for wide matrices, then norm-preserve globally.
    """
    u32 = update.float()
    v_mean = u32.square().mean(dim=reduction_dim, keepdim=True)
    red_size = update.shape[reduction_dim]
    old_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_size).sqrt()
    new_variance = variance.float().lerp(v_mean, 1.0 - beta2)
    scale = new_variance.clamp_min(eps).rsqrt()
    scaled_sq_sum = (v_mean * red_size) * scale.square()
    new_norm = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps)
    final_scale = scale * old_norm / new_norm
    return (u32 * final_scale).to(update.dtype), new_variance.to(variance.dtype)


def split_heads(x: Tensor, num_heads: int, heads_per_group: int = 1) -> tuple[Tensor, tuple[int, int]]:
    """View a row-stacked attention projection as independent head groups.

    Expected weight convention is NanoChat's Q/K/V convention:
        x.shape[-2] == num_heads * head_dim.

    Returns a tensor with a new batch dimension for groups and metadata needed by ``merge_heads``.
    If ``heads_per_group == num_heads``, this is equivalent to full-matrix orthogonalization.
    """
    _ensure_matrix_batch(x)
    rows, cols = x.shape[-2:]
    if rows % num_heads != 0:
        raise ValueError(f"rows={rows} is not divisible by num_heads={num_heads}")
    if num_heads % heads_per_group != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by heads_per_group={heads_per_group}"
        )
    head_dim = rows // num_heads
    n_groups = num_heads // heads_per_group
    grouped_rows = heads_per_group * head_dim
    y = x.reshape(*x.shape[:-2], n_groups, grouped_rows, cols)
    return y, (rows, cols)


def merge_heads(x: Tensor, original_shape: tuple[int, int]) -> Tensor:
    """Inverse of ``split_heads`` for the final two matrix dimensions."""
    rows, cols = original_shape
    return x.reshape(*x.shape[:-3], rows, cols)


@dataclass(frozen=True)
class FractionalSelection:
    """Canonicalized Dion3 selection result.

    Dion3 selects along the smaller matrix dimension.  If columns are smaller, the matrix is
    transposed so that selection is always represented as row selection.  ``transposed`` records
    whether the caller must transpose the final update back.
    """

    indices: Tensor
    transposed: bool


def canonicalize_for_fractional_rows(x: Tensor) -> tuple[Tensor, bool]:
    """Orient a 2-D matrix so Dion3 selection runs along its smaller dimension."""
    if x.ndim != 2:
        raise ValueError(f"Fractional Dion3 helper currently expects a single 2-D matrix, got {x.shape}")
    if x.shape[0] <= x.shape[1]:
        return x, False
    return x.mT, True


def select_top_fraction_rows(x: Tensor, fraction: float) -> FractionalSelection:
    """Select ceil(fraction * rows) rows with largest L1 momentum magnitude."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must lie in (0,1], got {fraction}")
    oriented, transposed = canonicalize_for_fractional_rows(x)
    k = max(1, int(torch.ceil(torch.tensor(fraction * oriented.shape[0])).item()))
    scores = oriented.float().abs().sum(dim=-1)
    indices = torch.topk(scores, k=k, largest=True, sorted=False).indices
    return FractionalSelection(indices=indices, transposed=transposed)


def kj_lr_scale(rows: int, cols: int) -> float:
    """Original Keller-Jordan shape correction for a matrix in PyTorch (out,in) convention."""
    return max(1.0, rows / cols) ** 0.5


def moonshot_rms_lr_scale(rows: int, cols: int) -> float:
    """Moonlight/Kimi large-scale RMS-matching factor often used with Muon."""
    return 0.2 * max(rows, cols) ** 0.5


def orthogonality_error(x: Tensor) -> Tensor:
    """Dimensionless Frobenius residual of the relevant semi-orthogonality constraint."""
    if x.shape[-2] <= x.shape[-1]:
        gram = x.float() @ x.float().mT
    else:
        gram = x.float().mT @ x.float()
    n = gram.shape[-1]
    eye = torch.eye(n, device=gram.device, dtype=gram.dtype)
    return (gram - eye).norm(dim=(-2, -1)) / (n**0.5)
