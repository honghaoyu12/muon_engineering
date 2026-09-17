"""Parameter grouping and benchmark presets for current NanoChat GPT.

Use ``setup_research_optimizer(model, preset=...)`` instead of ``model.setup_optimizer(...)``.
The implementation has two grouping modes:

* shape-only: mirrors NanoChat's native Muon communication batches and is used for clean baseline /
  polynomial experiments;
* role-aware: splits q/k/v/o/mlp/etc. only when a feature actually requires role metadata.

This distinction is important because changing group boundaries can change collective sizes and Python /
kernel batching even if the mathematical update is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .research_optimizer import ResearchMuonAdamW
from .runtime_state import (
    build_initial_grouping_definition,
    build_live_group_signature,
    initialize_runtime_grouping_state,
    synchronize_runtime_grouping_state_,
)


LOGICAL_ROLES = {"q", "k", "v", "o", "mlp_up", "mlp_down", "ve_gate"}
_ALLOWED_OVERRIDE_KEYS = {
    "nesterov", "orthogonalizer", "coefficients", "polar_dtype", "polar_eps",
    "gns_reset_after", "muoneq", "muonplus_mode", "normuon", "normuon_mode", "beta2",
    "weight_decay_mode", "lr_scale", "ortho_fraction", "fraction_lr_compensation",
    "per_head_roles", "heads_per_group", "fractional_roles", "dion3_roles", "role_split",
}


def _preset(name: str) -> dict[str, Any]:
    """Return feature defaults for a named benchmark rung.

    ``dion_ns`` and ``nanochat_pe`` are coefficient schedules. GNS is an evaluation backend. The
    names are intentionally kept separate so a systems change is not silently bundled with a
    polynomial change.
    """
    base = dict(
        update_rule="classic",
        nesterov=True,
        orthogonalizer="standard",
        coefficients="keller",
        polar_dtype="bf16",
        polar_eps=None,
        gns_reset_after=(),
        muoneq=False,
        muonplus_mode="none",
        normuon=False,
        normuon_mode="row",
        beta2=0.9,
        weight_decay_mode="decoupled",
        lr_scale="keller",
        ortho_fraction=0.25,
        fraction_lr_compensation=False,
        per_head_roles=(),
        heads_per_group=1,
        fractional_roles=(),
        role_split=False,
    )
    presets = {
        # Current KellerJordan/Muon matrix-transform reference inside NanoChat's training harness.
        "kj_reference": {},
        "kj_moonlight_scale": dict(lr_scale="moonshot_rms"),
        # Same grouping and KJ coefficients; only the solver backend changes.
        "kj_gns": dict(orthogonalizer="gns_official", polar_dtype="backend", gns_reset_after=(2,)),
        # Polynomial branch, still shape-only so group boundaries are not yet a confound.
        "dion_gns": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="dion_ns", gns_reset_after=(2,),
        ),
        "nanochat_pe_gns": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="nanochat_pe", gns_reset_after=(2,),
        ),
        # Control for the communication/batching effect of splitting by role before head geometry.
        "role_split_dion_gns": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="dion_ns", gns_reset_after=(2,), role_split=True,
        ),
        # First attention geometry change.
        "head_dion_gns": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="dion_ns", gns_reset_after=(2,), role_split=True,
            per_head_roles=("q", "k", "v"),
        ),
        "head_eq_dion_gns": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="dion_ns", gns_reset_after=(2,), role_split=True,
            per_head_roles=("q", "k", "v"), muoneq=True,
        ),
        "head_eq_norm_dion_gns": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="dion_ns", gns_reset_after=(2,), role_split=True,
            per_head_roles=("q", "k", "v"), muoneq=True,
            muonplus_mode="frob_snap", normuon=True,
        ),
        "production_candidate": dict(
            orthogonalizer="gns_official", polar_dtype="backend",
            coefficients="dion_ns", gns_reset_after=(2,), role_split=True,
            per_head_roles=("q", "k", "v"), muoneq=True,
            muonplus_mode="frob_snap", normuon=True,
            fractional_roles=("mlp_up", "mlp_down"), fraction_lr_compensation=True,
        ),
    }
    aliases = {
        "gns_pe": "dion_gns",
        "head_gns_pe": "head_dion_gns",
        "head_eq_gns_pe": "head_eq_dion_gns",
        "head_eq_norm_gns_pe": "head_eq_norm_dion_gns",
    }
    name = aliases.get(name, name)
    if name not in presets:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(presets)}")
    out = deepcopy(base)
    out.update(presets[name])
    return out


def _canonicalize_override(override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return {}
    out = dict(override)
    # Deprecated early-draft alias. Keep it only at the external config boundary.
    if "rank_fraction" in out:
        legacy = out.pop("rank_fraction")
        if "ortho_fraction" in out and float(out["ortho_fraction"]) != float(legacy):
            raise ValueError(
                f"Conflicting ortho_fraction={out['ortho_fraction']} and deprecated rank_fraction={legacy}"
            )
        out.setdefault("ortho_fraction", legacy)
    if "dion3_roles" in out:
        legacy_roles = tuple(out.pop("dion3_roles"))
        if "fractional_roles" in out and tuple(out["fractional_roles"]) != legacy_roles:
            raise ValueError(
                f"Conflicting fractional_roles={out['fractional_roles']} and deprecated "
                f"dion3_roles={legacy_roles}"
            )
        out.setdefault("fractional_roles", legacy_roles)
    coeff_aliases = {
        "polar_express": "nanochat_pe",
        "modded_ns": "dion_ns",
        "dao_gns_pe": "dion_ns",
        "gns_pe": "dion_ns",
    }
    if out.get("coefficients") in coeff_aliases:
        out["coefficients"] = coeff_aliases[out["coefficients"]]
    unknown = set(out) - _ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Unknown Muon-lab override key(s): {sorted(unknown)}")
    for key in ("per_head_roles", "fractional_roles"):
        if key in out:
            out[key] = tuple(out[key])
    if "gns_reset_after" in out:
        out["gns_reset_after"] = tuple(sorted(set(int(x) for x in out["gns_reset_after"])))
    return out


def _validate_cfg(cfg: dict[str, Any], model) -> None:
    for key in ("per_head_roles", "fractional_roles"):
        roles = set(cfg.get(key, ()))
        bad = roles - LOGICAL_ROLES
        if bad:
            raise ValueError(f"Unknown logical role(s) in {key}: {sorted(bad)}")
    if set(cfg.get("per_head_roles", ())) - {"q", "k", "v"}:
        raise ValueError("per_head_roles currently supports only q/k/v")

    hpg = int(cfg.get("heads_per_group", 1))
    if hpg <= 0:
        raise ValueError("heads_per_group must be positive")
    for role in cfg.get("per_head_roles", ()):
        nh = model.config.n_head if role == "q" else model.config.n_kv_head
        if nh % hpg != 0:
            raise ValueError(f"heads_per_group={hpg} must divide {role} num_heads={nh}")
    if hpg != 1 and not cfg.get("per_head_roles"):
        raise ValueError("heads_per_group has no effect unless per_head_roles is non-empty")
    overlap = set(cfg.get("per_head_roles", ())) & set(cfg.get("fractional_roles", ()))
    if overlap:
        raise ValueError(
            f"The initial benchmark does not combine per-head and fractional_ef on the same role(s): {sorted(overlap)}"
        )
    if cfg.get("fraction_lr_compensation") and not cfg.get("fractional_roles"):
        raise ValueError("fraction_lr_compensation has no effect unless fractional_roles is non-empty")
    if cfg.get("fractional_roles") and cfg.get("weight_decay_mode") == "cautious":
        raise ValueError("cautious weight decay + fractional_ef is not implemented in the attribution-clean ladder")
    if cfg.get("fractional_roles") and cfg.get("normuon") and cfg.get("normuon_mode") != "row":
        raise ValueError("fractional_ef currently supports NorMuon only in row mode")

    f = float(cfg["ortho_fraction"])
    if not (0.0 < f <= 1.0):
        raise ValueError(f"ortho_fraction must be in (0,1], got {f}")
    beta2 = float(cfg["beta2"])
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must be in [0,1), got {beta2}")

    # Any role-dependent algorithm requires role-aware groups. Make this explicit rather than
    # allowing a mixed shape-only group to carry contradictory metadata.
    if cfg.get("per_head_roles") or cfg.get("fractional_roles"):
        cfg["role_split"] = True


def _collect_roles(model) -> dict[str, list]:
    roles: dict[str, list] = {
        "q": [], "k": [], "v": [], "o": [], "mlp_up": [], "mlp_down": [], "ve_gate": []
    }
    for block in model.transformer.h:
        roles["q"].append(block.attn.c_q.weight)
        roles["k"].append(block.attn.c_k.weight)
        roles["v"].append(block.attn.c_v.weight)
        roles["o"].append(block.attn.c_proj.weight)
        roles["mlp_up"].append(block.mlp.c_fc.weight)
        roles["mlp_down"].append(block.mlp.c_proj.weight)
        if block.attn.ve_gate is not None:
            roles["ve_gate"].append(block.attn.ve_gate.weight)
    return roles


def _parameter_group_signature(param_groups: list[dict]) -> list[dict[str, Any]]:
    """Return a stable, JSON-serializable signature of optimizer grouping/hyperparameters.

    This is intentionally richer than the feature preset. It catches resume hazards such as a
    changed effective LR (for example from batch-LR scaling), changed AdamW settings, or changed
    matrix group boundaries even when the named Muon preset itself is unchanged.
    """
    shared_fields = ("kind", "logical_role", "lr", "weight_decay", "eps")
    muon_fields = (
        "momentum", "ns_steps", "beta2", "update_rule", "nesterov", "orthogonalizer",
        "coefficients", "polar_dtype", "polar_eps", "gns_reset_after", "muoneq",
        "muonplus_mode", "normuon", "normuon_mode", "weight_decay_mode", "lr_scale",
        "ortho_fraction", "fraction_lr_compensation", "num_heads", "heads_per_group",
    )
    out: list[dict[str, Any]] = []
    for group in param_groups:
        entry: dict[str, Any] = {
            key: group.get(key) for key in shared_fields if key in group
        }
        if "betas" in group:
            entry["betas"] = list(group["betas"])
        if group.get("kind") == "muon":
            for key in muon_fields:
                value = group.get(key)
                if isinstance(value, tuple):
                    value = list(value)
                entry[key] = value
        entry["param_shapes"] = [list(p.shape) for p in group.get("params", ())]
        out.append(entry)
    return out


def setup_research_optimizer(
    model,
    *,
    preset: str = "kj_reference",
    unembedding_lr: float = 0.008,
    embedding_lr: float = 0.3,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
    scalar_lr: float = 0.5,
    ortho_fraction: float = 0.25,
    beta2: float = 0.9,
    override: dict[str, Any] | None = None,
):
    """Build ``ResearchMuonAdamW`` for the current NanoChat GPT architecture.

    Configuration precedence is deliberately unambiguous:

        preset defaults < explicit setup arguments (``ortho_fraction``, ``beta2``) < JSON override

    ``rank_fraction`` is accepted only as a deprecated key inside ``override`` and is canonicalized
    to ``ortho_fraction``.
    """
    cfg = _preset(preset)
    cfg["ortho_fraction"] = float(ortho_fraction)
    cfg["beta2"] = float(beta2)
    cfg.update(_canonicalize_override(override))
    _validate_cfg(cfg, model)

    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # Mirror current NanoChat AdamW groups. Matrix experiments must not silently retune these.
    value_embeds_params = list(model.value_embeds.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]
    smear_params = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]

    param_groups: list[dict] = [
        dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
             betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01,
             betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind="adamw", params=x0_params, lr=scalar_lr,
             betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=smear_params, lr=0.2,
             betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]

    roles = _collect_roles(model)
    native_matrix_params = list(model.transformer.h.parameters())
    classified = [p for ps in roles.values() for p in ps]
    native_ids = {id(p) for p in native_matrix_params}
    classified_ids = {id(p) for p in classified}
    if classified_ids != native_ids:
        missing = [tuple(p.shape) for p in native_matrix_params if id(p) not in classified_ids]
        extra = [tuple(p.shape) for p in classified if id(p) not in native_ids]
        raise RuntimeError(
            "NanoChat architecture changed: role-aware grouping no longer covers transformer.h exactly. "
            f"Missing shapes={missing}; extra shapes={extra}. Update setup.py before benchmarking."
        )

    common = dict(
        kind="muon", lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=cfg["beta2"],
        weight_decay=weight_decay, update_rule=cfg["update_rule"], nesterov=cfg["nesterov"],
        orthogonalizer=cfg["orthogonalizer"], coefficients=cfg["coefficients"],
        polar_dtype=cfg["polar_dtype"], polar_eps=cfg.get("polar_eps"),
        gns_reset_after=tuple(cfg["gns_reset_after"]),
        muoneq=cfg["muoneq"], muonplus_mode=cfg["muonplus_mode"], normuon=cfg["normuon"],
        normuon_mode=cfg["normuon_mode"], weight_decay_mode=cfg["weight_decay_mode"],
        lr_scale=cfg["lr_scale"], ortho_fraction=1.0, fraction_lr_compensation=False,
        num_heads=None, heads_per_group=None,
    )

    if not cfg["role_split"]:
        # Exact native grouping boundary: shape only. This is important for clean KJ/GNS/polynomial
        # comparisons because changing group boundaries changes collective and batching behavior.
        for shape in sorted({tuple(p.shape) for p in native_matrix_params}):
            ps = [p for p in native_matrix_params if tuple(p.shape) == shape]
            g = dict(common)
            g.update(params=ps, logical_role="mixed")
            param_groups.append(g)
    else:
        per_head_roles = set(cfg.get("per_head_roles", ()))
        fractional_roles = set(cfg.get("fractional_roles", ()))
        for role, params in roles.items():
            if not params:
                continue
            for shape in sorted({tuple(p.shape) for p in params}):
                ps = [p for p in params if tuple(p.shape) == shape]
                g = dict(common)
                g.update(params=ps, logical_role=role)
                if role in per_head_roles:
                    g["num_heads"] = model.config.n_head if role == "q" else model.config.n_kv_head
                    g["heads_per_group"] = int(cfg["heads_per_group"])
                if role in fractional_roles:
                    g["update_rule"] = "fractional_ef"
                    g["nesterov"] = False
                    g["ortho_fraction"] = float(cfg["ortho_fraction"])
                    g["fraction_lr_compensation"] = bool(cfg["fraction_lr_compensation"])
                    # Do NOT silently enable NorMuon; fractional_ef inherits the parent adaptive setting.
                param_groups.append(g)

    optimizer = ResearchMuonAdamW(param_groups)
    optimizer.muon_lab_preset = preset
    optimizer.muon_lab_role_split = bool(cfg["role_split"])
    optimizer.muon_lab_config = deepcopy(cfg)
    optimizer.muon_lab_param_group_signature = _parameter_group_signature(optimizer.param_groups)
    optimizer.muon_lab_initial_grouping = build_initial_grouping_definition(optimizer)
    initialize_runtime_grouping_state(optimizer)
    optimizer.muon_lab_live_group_signature = build_live_group_signature(optimizer)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def set_attention_grouping(optimizer, *, heads_per_group_q: int, heads_per_group_kv: int) -> None:
    """Change Q/K/V grouping at a phase boundary while retaining row-aligned optimizer state.

    Validation is transactional: all affected groups are checked before any group is mutated.
    """
    proposals = []
    for group in optimizer.param_groups:
        role = group.get("logical_role")
        nh = group.get("num_heads")
        if nh is None or role not in {"q", "k", "v"}:
            continue
        hpg = heads_per_group_q if role == "q" else heads_per_group_kv
        hpg = int(hpg)
        if hpg <= 0 or int(nh) % hpg != 0:
            raise ValueError(f"heads_per_group={hpg} must divide {role} num_heads={nh}")
        proposals.append((group, hpg))
    for group, hpg in proposals:
        group["heads_per_group"] = hpg
    synchronize_runtime_grouping_state_(optimizer)
    optimizer.muon_lab_live_group_signature = build_live_group_signature(optimizer)
