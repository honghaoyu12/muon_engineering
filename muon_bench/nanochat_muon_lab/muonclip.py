"""Architecture-aware MuonClip helpers.

Important NanoChat caveat
-------------------------
Current NanoChat normalizes Q and K *activations* after projection. A positive scalar rescaling of
an entire Q/K head weight is therefore nearly canceled by that activation normalization. For that
reason this module is NOT automatically called by ``ResearchMuonAdamW`` on stock NanoChat.

It is provided for two uses:
  1. a NanoChat architecture ablation in which post-projection QK normalization is disabled; or
  2. porting the optimizer recipe to an architecture where Q/K weight rescaling affects logits.

The caller must supply observed per-query-head maximum attention logits. Collecting those logits is
architecture/kernel specific and intentionally stays out of the optimizer core.
"""
from __future__ import annotations

import torch
from torch import Tensor


def clip_factors(max_logits: Tensor, tau: float = 100.0, eps: float = 1e-12) -> Tensor:
    """Return gamma_h = min(1, tau / max_logit_h) for nonnegative per-head maxima."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    x = max_logits.float()
    if not torch.isfinite(x).all():
        raise ValueError("max_logits contains NaN/Inf values")
    if (x < 0).any():
        raise ValueError("max_logits must be nonnegative")
    x = x.clamp_min(eps)
    return torch.clamp(tau / x, max=1.0)


def _head_view(weight: Tensor, num_heads: int) -> Tensor:
    """View NanoChat-style [num_heads*head_dim, d_model] weight as [heads, head_dim, d_model]."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D projection weight, got {tuple(weight.shape)}")
    if weight.shape[0] % num_heads != 0:
        raise ValueError(
            f"Output rows {weight.shape[0]} are not divisible by num_heads={num_heads}"
        )
    return weight.view(num_heads, weight.shape[0] // num_heads, weight.shape[1])


@torch.no_grad()
def apply_muonclip_weights_(
    q_weight: Tensor,
    k_weight: Tensor,
    max_logits: Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int | None = None,
    tau: float = 100.0,
    gqa_mode: str = "query_only",
) -> Tensor:
    """Apply Kimi-style weight rescaling when the architecture makes it meaningful.

    MHA (num_q_heads == num_kv_heads)
        Balanced clipping rescales both Q_h and K_h by sqrt(gamma_h), so their dot product scales
        by gamma_h.

    GQA (num_q_heads > num_kv_heads)
        A K head is shared by multiple query heads. Independent balanced Q/K rescaling would couple
        those query heads. The conservative default is therefore ``query_only``: Q_h is multiplied
        by gamma_h and K is unchanged.

    Returns the gamma vector for logging.
    """
    if num_kv_heads is None:
        num_kv_heads = num_q_heads
    if max_logits.numel() != num_q_heads:
        raise ValueError(
            f"Expected one max logit per query head ({num_q_heads}), got {max_logits.numel()}"
        )

    if q_weight.ndim != 2 or k_weight.ndim != 2:
        raise ValueError("Q/K weights must both be 2-D projection matrices")
    if q_weight.shape[1] != k_weight.shape[1]:
        raise ValueError(
            f"Q/K input dimensions must match; got {q_weight.shape[1]} and {k_weight.shape[1]}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    q = _head_view(q_weight, num_q_heads)
    k = _head_view(k_weight, num_kv_heads)
    gamma = clip_factors(max_logits.reshape(num_q_heads), tau=tau).to(q.device)

    if num_q_heads == num_kv_heads:
        scale = gamma.sqrt().to(q.dtype).view(num_q_heads, 1, 1)
        q.mul_(scale)
        k.mul_(scale.to(k.dtype))
        return gamma

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA requires num_q_heads divisible by num_kv_heads; got {num_q_heads}/{num_kv_heads}"
        )
    if gqa_mode != "query_only":
        raise ValueError(
            "Only gqa_mode='query_only' is implemented intentionally: shared K heads make naive "
            "per-query balanced clipping ambiguous."
        )
    q.mul_(gamma.to(q.dtype).view(num_q_heads, 1, 1))
    return gamma


def rms_normalize_last_dim(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Small helper used to demonstrate why stock NanoChat QK normalization cancels scalar clipping."""
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)
