"""NanoChat-compatible modular Muon-family benchmark optimizer.

Design goal
-----------
Keep NanoChat's *existing* optimizer-owned gradient synchronization / ZeRO-2-style communication,
but replace only the Muon compute phase with an auditable modular implementation.  This makes
optimizer comparisons much cleaner than converting NanoChat to a different distributed stack.

This is a correctness-first research implementation.  The official GNS kernels can be enabled,
but the higher-level Python composition is intentionally explicit.  Once a winning recipe is
identified, fuse/batch it for production.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from nanochat.optim import MuonAdamW

from .muon_math import (
    coefficients,
    factored_normuon,
    gram_newton_schulz_reference,
    kj_lr_scale,
    merge_heads,
    moonshot_rms_lr_scale,
    muoneq_rows,
    muonplus_normalize,
    normuon_rows,
    split_heads,
    standard_newton_schulz,
)


def _canonicalize_group_aliases(group: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize legacy spelling without changing algorithmic semantics."""
    g = group
    if "rank_fraction" in g:
        legacy = float(g["rank_fraction"])
        if "ortho_fraction" in g and float(g["ortho_fraction"]) != legacy:
            raise ValueError(
                f"Conflicting ortho_fraction={g['ortho_fraction']} and deprecated "
                f"rank_fraction={g['rank_fraction']}"
            )
        g.setdefault("ortho_fraction", legacy)
        g.pop("rank_fraction", None)
    if g.get("update_rule") == "dion3":
        g["update_rule"] = "fractional_ef"
    coeff_aliases = {
        "polar_express": "nanochat_pe",
        "modded_ns": "dion_ns",
        "dao_gns_pe": "dion_ns",
        "gns_pe": "dion_ns",
    }
    if g.get("coefficients") in coeff_aliases:
        g["coefficients"] = coeff_aliases[g["coefficients"]]
    if "gns_reset_after" in g:
        g["gns_reset_after"] = tuple(sorted(set(int(x) for x in g["gns_reset_after"])))
    return g


def _defaults(group: dict[str, Any]) -> dict[str, Any]:
    """Fill research-only fields and canonicalize deprecated aliases.

    ``ortho_fraction`` is the canonical selected-dimension fraction. ``rank_fraction`` existed in
    an early draft; accept it only as a compatibility alias. ``update_rule="dion3"`` is likewise
    accepted only as a legacy alias for the canonical ``fractional_ef`` rule. The latter name is
    deliberate: current Microsoft Dion uses ``Dion3`` for a combined fractional+NorMuon family,
    while this harness must also test the fractional error-feedback mechanism by itself.
    """
    g = _canonicalize_group_aliases(group)
    g.setdefault("update_rule", "classic")            # classic | fractional_ef
    g.setdefault("nesterov", True)
    g.setdefault("orthogonalizer", "standard")       # standard | gns_reference | gns_official
    g.setdefault("coefficients", "keller")           # keller | nanochat_pe | dion_ns
    g.setdefault("gns_reset_after", ())
    g.setdefault("polar_dtype", "bf16")               # standard/reference: bf16 | fp32 | native; official: backend
    g.setdefault("polar_eps", None)                    # None => schedule/backend-appropriate reference default
    g.setdefault("muoneq", False)
    g.setdefault("muonplus_mode", "none")            # none | frob_snap | row | col | row_col | col_row
    g.setdefault("normuon", False)
    g.setdefault("normuon_mode", "row")              # row | factored
    g.setdefault("beta2", 0.9)
    g.setdefault("weight_decay_mode", "decoupled")   # decoupled | cautious
    g.setdefault("lr_scale", "keller")               # keller | moonshot_rms | none
    g.setdefault("num_heads", None)                   # Q/K/V only
    g.setdefault("heads_per_group", None)             # 1=per head; num_heads=full matrix
    g.setdefault("ortho_fraction", 1.0)               # selected fraction of smaller dimension
    g.setdefault("fraction_lr_compensation", False)
    g.setdefault("logical_role", "matrix")
    return g


class ResearchMuonAdamW(MuonAdamW):
    """Subclass current NanoChat ``MuonAdamW`` and override only Muon's compute phase.

    The parent class still owns:
      * AdamW semantics for non-Muon parameter groups;
      * single-GPU behavior;
      * asynchronous reduce-scatter/all-reduce;
      * per-rank matrix ownership for Muon groups;
      * all-gather and copy-back.

    Therefore this class remains compatible with NanoChat's no-DDP training loop.
    """

    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            if group.get("kind") == "muon":
                _defaults(group)
                self._validate_group(group)
        super().__init__(param_groups)
        self._official_gns_cache: dict[tuple, Any] = {}

    @staticmethod
    def _validate_group(group: dict) -> None:
        _canonicalize_group_aliases(group)
        allowed = {
            "update_rule": {"classic", "fractional_ef"},
            "orthogonalizer": {"standard", "gns_reference", "gns_official"},
            "coefficients": {"keller", "nanochat_pe", "dion_ns"},
            "muonplus_mode": {"none", "frob_snap", "row", "col", "row_col", "col_row"},
            "normuon_mode": {"row", "factored"},
            "weight_decay_mode": {"decoupled", "cautious"},
            "lr_scale": {"keller", "moonshot_rms", "none"},
        }
        for key, values in allowed.items():
            if group[key] not in values:
                raise ValueError(f"Unknown {key}={group[key]!r}; choose from {sorted(values)}")
        polar_dtype = group.get("polar_dtype", "bf16")
        if group["orthogonalizer"] == "gns_official":
            if polar_dtype != "backend":
                raise ValueError(
                    "orthogonalizer='gns_official' owns its internal working precision; "
                    "set polar_dtype='backend' instead of claiming bf16/fp32 control from this wrapper."
                )
        elif polar_dtype not in {"bf16", "fp32", "native"}:
            raise ValueError("polar_dtype must be one of {'bf16','fp32','native'} for standard/reference backends")

        polar_eps = group.get("polar_eps")
        if polar_eps is not None and float(polar_eps) <= 0:
            raise ValueError(f"polar_eps must be positive or None, got {polar_eps}")

        resets = tuple(group.get("gns_reset_after", ()))
        if group["orthogonalizer"] == "standard":
            if resets:
                raise ValueError("gns_reset_after is only meaningful for GNS backends; use () with standard NS")
        else:
            ns_steps = int(group["ns_steps"])
            bad_resets = [r for r in resets if r < 1 or r >= ns_steps]
            if bad_resets:
                raise ValueError(
                    f"GNS restart indices must satisfy 1 <= r < ns_steps={ns_steps}; got {bad_resets}"
                )

        if group["update_rule"] == "fractional_ef" and group["nesterov"]:
            raise ValueError("fractional_ef mode must not also use Keller-style Nesterov.")
        f = float(group["ortho_fraction"])
        if not (0.0 < f <= 1.0):
            raise ValueError(f"ortho_fraction must be in (0,1], got {f}")
        if f < 1.0 and group.get("num_heads") not in (None, 1):
            raise ValueError(
                "This benchmark intentionally does not combine fractional_ef with head-wise "
                "attention in the first implementation. Test those ingredients separately first."
            )
        if group["normuon_mode"] == "factored" and group.get("num_heads") not in (None, 1):
            raise ValueError("Use row NorMuon with head-wise groups; factored mode is full-matrix only.")
        if group["update_rule"] == "fractional_ef" and group["normuon"] and group["normuon_mode"] != "row":
            raise ValueError("fractional_ef currently supports NorMuon only in row mode in this benchmark implementation.")
        if group["weight_decay_mode"] == "cautious" and group["update_rule"] == "fractional_ef":
            raise ValueError(
                "For attribution-clean fractional error-feedback benchmarks use global decoupled WD. "
                "Cautious WD + fractional selection is a later interaction experiment."
            )
        nh = group.get("num_heads")
        hpg = group.get("heads_per_group")
        if nh is not None:
            nh = int(nh)
            hpg = 1 if hpg is None else int(hpg)
            if nh <= 0 or hpg <= 0 or nh % hpg != 0:
                raise ValueError(f"heads_per_group={hpg} must divide num_heads={nh}")

    def _official_gns(self, group: dict):
        ns_epsilon = 1e-7 if group.get("polar_eps") is None else float(group["polar_eps"])
        key = (
            group["coefficients"],
            int(group["ns_steps"]),
            tuple(group["gns_reset_after"]),
            ns_epsilon,
        )
        if key not in self._official_gns_cache:
            try:
                from gram_newton_schulz import GramNewtonSchulz
            except ImportError as exc:
                raise RuntimeError(
                    "orthogonalizer='gns_official' requires the official gram-newton-schulz package. "
                    "Install it only on a supported CUDA/PyTorch environment; use gns_reference for CPU tests."
                ) from exc
            # Official GNS documents ``List[List[float]]``. Avoid relying on permissive tuple handling.
            coeffs = [list(c) for c in coefficients(group["coefficients"], group["ns_steps"])]
            self._official_gns_cache[key] = GramNewtonSchulz(
                ns_epsilon=ns_epsilon,
                ns_coefficients=coeffs,
                gram_newton_schulz_reset_iterations=list(group["gns_reset_after"]),
            )
        return self._official_gns_cache[key]

    def _orthogonalize(self, x: Tensor, group: dict) -> Tensor:
        coeffs = coefficients(group["coefficients"], group["ns_steps"])
        backend = group["orthogonalizer"]
        if backend in {"standard", "gns_reference"}:
            dtype_mode = group.get("polar_dtype", "bf16")
            if dtype_mode == "bf16":
                work = x.bfloat16()
            elif dtype_mode == "fp32":
                work = x.float()
            elif dtype_mode == "native":
                work = x
            else:
                raise ValueError(f"Unsupported polar_dtype={dtype_mode!r} for backend={backend}")
            if backend == "standard":
                is_nanochat_pe = group["coefficients"] in {"nanochat_pe", "polar_express"}
                safety = 1.01 if is_nanochat_pe else 1.0
                # Current NanoChat PE uses +1e-6 in its Frobenius normalization; the current
                # Keller reference uses +1e-7. Keep those reference conventions distinct.
                eps = group.get("polar_eps")
                if eps is None:
                    eps = 1e-6 if is_nanochat_pe else 1e-7
                return standard_newton_schulz(
                    work, coeffs, eps=float(eps), safety_factor=safety, norm_in_fp32=False
                )
            # Reference GNS has its own normalization semantics. Do not silently import the
            # NanoChat PE safety/epsilon convention merely because the coefficient table is PE.
            eps = 1e-7 if group.get("polar_eps") is None else float(group["polar_eps"])
            return gram_newton_schulz_reference(
                work, coeffs, eps=eps, reset_after=group["gns_reset_after"]
            )
        if backend == "gns_official":
            # The official package owns its half-precision/kernel policy. It accepts batched matrices.
            out = self._official_gns(group)(x)
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"Official GNS returned {type(out)!r}, expected torch.Tensor")
            if out.shape != x.shape:
                raise RuntimeError(f"Official GNS changed tensor shape from {tuple(x.shape)} to {tuple(out.shape)}")
            return out
        raise ValueError(f"Unknown orthogonalizer={backend!r}")

    def _project(self, x: Tensor, group: dict) -> Tensor:
        """MuonEq -> optional head grouping -> polar map -> Muon+ post normalization."""
        num_heads = group.get("num_heads")
        heads_per_group = group.get("heads_per_group")
        if num_heads is not None:
            if heads_per_group is None:
                heads_per_group = 1
            blocks, original_shape = split_heads(x, int(num_heads), int(heads_per_group))
        else:
            blocks, original_shape = x, None

        if group["muoneq"]:
            # In head-wise mode each independent optimizer block gets its own equilibration target.
            blocks = muoneq_rows(blocks)
        blocks = self._orthogonalize(blocks, group)
        blocks = muonplus_normalize(blocks, group["muonplus_mode"])

        return merge_heads(blocks, original_shape) if original_shape is not None else blocks

    def _variance_shape(self, shape: tuple[int, int], group: dict) -> tuple[int, int]:
        rows, cols = shape
        if group["normuon_mode"] == "row":
            return rows, 1
        if rows >= cols:
            return rows, 1
        return 1, cols

    def _apply_normuon(self, update: Tensor, variance: Tensor, group: dict) -> tuple[Tensor, Tensor]:
        if not group["normuon"]:
            return update, variance

        beta2 = float(group["beta2"])
        if group["normuon_mode"] == "factored":
            red_dim = -1 if update.shape[-2] >= update.shape[-1] else -2
            return factored_normuon(update, variance, beta2, reduction_dim=red_dim)

        # Row NorMuon. If attention is grouped, norm preservation is performed independently
        # for each optimizer block rather than across the concatenated Q/K/V head matrix.
        num_heads = group.get("num_heads")
        if num_heads is None:
            return normuon_rows(update, variance, beta2)
        hpg = int(group.get("heads_per_group") or 1)
        u_blocks, original_shape = split_heads(update, int(num_heads), hpg)
        rows_per_group = u_blocks.shape[-2]
        v_blocks = variance.reshape(*variance.shape[:-2], u_blocks.shape[-3], rows_per_group, 1)
        u_blocks, v_blocks = normuon_rows(u_blocks, v_blocks, beta2)
        return merge_heads(u_blocks, original_shape), v_blocks.reshape_as(variance)

    @staticmethod
    def _lr_scale(group: dict, rows: int, cols: int) -> float:
        mode = group["lr_scale"]
        # Per-head/group scaling is based on the matrix that is actually orthogonalized.
        if group.get("num_heads") is not None:
            num_heads = int(group["num_heads"])
            hpg = int(group.get("heads_per_group") or 1)
            head_dim = rows // num_heads
            rows = hpg * head_dim
        if mode == "keller":
            return kj_lr_scale(rows, cols)
        if mode == "moonshot_rms":
            return moonshot_rms_lr_scale(rows, cols)
        if mode == "none":
            return 1.0
        raise ValueError(f"Unknown lr_scale={mode}")

    @staticmethod
    def _decay_and_update(param: Tensor, update: Tensor, raw_lr: float, scaled_lr: float, wd: float, mode: str) -> None:
        # Keller's reference semantics: matrix-shape scaling belongs to the UPDATE, not WD.
        if mode == "decoupled":
            if wd:
                param.mul_(1.0 - raw_lr * wd)
            param.add_(update, alpha=-scaled_lr)
        elif mode == "cautious":
            # Match NanoChat/Cautious-WD semantics: the same effective Muon LR that scales the
            # update also scales the masked weight-decay term. This is intentionally different
            # from the Keller-reference decoupled-WD branch above.
            if wd:
                mask = (update * param) >= 0
                param.sub_(scaled_lr * wd * param * mask)
            param.add_(update, alpha=-scaled_lr)
        else:
            raise ValueError(f"Unknown weight_decay_mode={mode}")

    @torch.no_grad()
    def _classic_batch(
        self,
        grad: Tensor,
        param: Tensor,
        momentum: Tensor,
        variance: Tensor | None,
        group: dict,
    ) -> None:
        """Vectorized classic Muon update for a batch of same-shape matrices.

        NanoChat groups matrices by shape before reduce-scatter. Keeping that batch intact is
        important for realistic optimizer timing, especially for official GNS and per-head Q/K/V.
        """
        beta = float(group["momentum"])
        momentum.lerp_(grad, 1.0 - beta)
        precondition_input = torch.lerp(grad, momentum, beta) if group["nesterov"] else momentum
        update = self._project(precondition_input, group).to(param.dtype)
        if variance is not None:
            update, new_v = self._apply_normuon(update, variance, group)
            variance.copy_(new_v)

        raw_lr = float(group["lr"])
        scale = self._lr_scale(group, param.shape[-2], param.shape[-1])
        self._decay_and_update(
            param,
            update,
            raw_lr=raw_lr,
            scaled_lr=raw_lr * scale,
            wd=float(group["weight_decay"]),
            mode=group["weight_decay_mode"],
        )

    @torch.no_grad()
    def _classic_one(
        self,
        grad: Tensor,
        param: Tensor,
        momentum: Tensor,
        variance: Tensor | None,
        group: dict,
    ) -> None:
        """Single-matrix wrapper retained for tests and numerical auditing."""
        self._classic_batch(grad, param, momentum, variance, group)

    @torch.no_grad()
    def _fractional_ef_batch(
        self,
        grad: Tensor,
        param: Tensor,
        error_buffer: Tensor,
        variance: Tensor | None,
        group: dict,
    ) -> None:
        """Batched selected-submatrix fractional error feedback for same-shape matrices.

        This implements the fractional selected-coordinate/error-feedback mechanism as an
        independently testable rule. It is not claimed to be API/state-equivalent to the current
        ``microsoft.dion.Dion3`` class, which currently bundles NorMuon-family behavior.

        ``grad``/``param`` may be either a single matrix or ``(..., rows, cols)``. Selection is
        performed independently for every matrix along the smaller dimension. Unselected residual
        coordinates are retained exactly; selected rows are multiplied by ``momentum`` only after
        their update is applied.
        """
        error_buffer.add_(grad)
        rows, cols = param.shape[-2:]
        transpose = rows > cols
        m = error_buffer.mT if transpose else error_buffer
        p = param.mT if transpose else param

        f = float(group["ortho_fraction"])
        k = max(1, math.ceil(f * m.shape[-2]))
        scores = m.float().abs().sum(dim=-1)
        idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False).indices
        gather_idx = idx.unsqueeze(-1).expand(*idx.shape, m.shape[-1])
        selected = torch.gather(m, dim=-2, index=gather_idx)

        # Ordering is intentional: selection uses the un-equilibrated residual magnitude;
        # MuonEq, polar iteration, and post-polar normalization apply only after selection.
        update = self._project(selected, group).to(param.dtype)

        if variance is not None:
            v_idx = idx.unsqueeze(-1).expand(*idx.shape, variance.shape[-1])
            selected_v = torch.gather(variance, dim=-2, index=v_idx)
            update, selected_v_new = normuon_rows(update, selected_v, float(group["beta2"]))
            variance.scatter_(dim=-2, index=v_idx, src=selected_v_new)

        # Keep the empirical fractional-LR compensation separate from weight decay. Otherwise
        # changing f would silently retune regularization as well as the update magnitude.
        raw_lr = float(group["lr"])
        update_lr = raw_lr / math.sqrt(f) if group["fraction_lr_compensation"] else raw_lr
        scale = self._lr_scale(group, rows, cols)
        wd = float(group["weight_decay"])
        if wd:
            # Fractional EF uses global decoupled WD, including coordinates not selected this step.
            param.mul_(1.0 - raw_lr * wd)

        p.scatter_add_(dim=-2, index=gather_idx, src=update * (-(update_lr * scale)))

        # Error feedback: decay only rows that were selected and consumed this step.
        selected_residual = torch.gather(m, dim=-2, index=gather_idx)
        selected_residual.mul_(float(group["momentum"]))
        m.scatter_(dim=-2, index=gather_idx, src=selected_residual)

    @torch.no_grad()
    def _fractional_ef_one(
        self,
        grad: Tensor,
        param: Tensor,
        error_buffer: Tensor,
        variance: Tensor | None,
        group: dict,
    ) -> None:
        """Single-matrix wrapper retained for tests and numerical auditing."""
        self._fractional_ef_batch(grad, param, error_buffer, variance, group)

    # Private compatibility wrappers for v0.3-era tests/notebooks. New code should use the
    # canonical ``fractional_ef`` terminology above.
    def _dion3_batch(self, *args, **kwargs):
        return self._fractional_ef_batch(*args, **kwargs)

    def _dion3_one(self, *args, **kwargs):
        return self._fractional_ef_one(*args, **kwargs)

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """NanoChat communication-compatible replacement for the parent Muon compute phase."""
        if info["future"] is not None:
            info["future"].wait()

        params = group["params"]
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"]
        first = params[0]
        shape, device, dtype = first.shape, first.device, first.dtype
        start_idx = rank * chunk_size
        num_owned = max(0, min(chunk_size, len(params) - start_idx))

        state = self.state[first]
        if "research_momentum" not in state:
            state["research_momentum"] = torch.zeros(
                chunk_size, *shape, dtype=dtype, device=device
            )
        need_variance = bool(group["normuon"])
        if need_variance and "research_variance" not in state:
            if group["update_rule"] == "fractional_ef":
                # Canonical fractional orientation has min(m,n) selectable rows.
                vshape = (min(shape), 1)
            else:
                vshape = self._variance_shape(shape, group)
            state["research_variance"] = torch.zeros(
                chunk_size, *vshape, dtype=torch.float32, device=device
            )

        stacked_owned = None
        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            variance = state.get("research_variance")
            variance_owned = variance[:num_owned] if variance is not None else None
            momentum_owned = state["research_momentum"][:num_owned]
            grad_owned = grad_chunk[:num_owned]
            if group["update_rule"] == "classic":
                self._classic_batch(grad_owned, stacked_owned, momentum_owned, variance_owned, group)
            else:
                self._fractional_ef_batch(grad_owned, stacked_owned, momentum_owned, variance_owned, group)

        # Preserve parent class's exact gather/copy-back contract.
        if info["stacked_grads"] is None:
            gather_list.append(dict(future=None, stacked_params=stacked_owned, params=params))
            return

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            updated_params[:num_owned].copy_(stacked_owned)
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = torch.distributed.all_gather_into_tensor(
            stacked_params, updated_params, async_op=True
        ).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))
