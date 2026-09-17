"""Schema-versioned optimizer runtime state and dynamic-grouping validation.

The helpers are intentionally duck-typed: they operate on optimizer param_groups/state and tensor
metadata without importing PyTorch. This keeps the state contract independently unit-testable.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from .provenance import canonicalize


RUNTIME_STATE_SCHEMA_VERSION = 1
GROUP_SIGNATURE_SCHEMA_VERSION = 1
EXPECTED_STATE_SCHEMA_VERSION = 1
OBSERVED_STATE_SCHEMA_VERSION = 1

_RUNTIME_STATE_ATTR = "muon_lab_runtime_grouping_state"
_GROUPING_FIELDS = {
    "q": "q_heads_per_group",
    "k": "k_heads_per_group",
    "v": "v_heads_per_group",
}
_SHARED_STRUCTURAL_FIELDS = (
    "kind",
    "logical_role",
    "betas",
    "eps",
)
_MUON_STRUCTURAL_FIELDS = (
    "ns_steps",
    "beta2",
    "update_rule",
    "nesterov",
    "orthogonalizer",
    "coefficients",
    "polar_dtype",
    "polar_eps",
    "gns_reset_after",
    "muoneq",
    "muonplus_mode",
    "normuon",
    "normuon_mode",
    "weight_decay_mode",
    "lr_scale",
    "ortho_fraction",
    "fraction_lr_compensation",
    "num_heads",
    "heads_per_group",
)


class RuntimeStateError(ValueError):
    """Base error for invalid or incompatible optimizer runtime state."""


class RuntimeStateSchemaError(RuntimeStateError):
    """Raised when a runtime artifact has an unsupported schema."""


class GroupSignatureMismatch(RuntimeStateError):
    """Raised when a live optimizer no longer matches its structural signature."""


class DuplicateTransitionError(RuntimeStateError):
    """Raised when a grouping transition would be executed more than once."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(size) for size in shape]


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _device_class(value: Any) -> str | None:
    device = getattr(value, "device", None)
    if device is None:
        return None
    device_type = getattr(device, "type", None)
    return str(device_type if device_type is not None else device).split(":", 1)[0]


def _param_descriptor(parameter: Any, index: int) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"parameter_index": index}
    shape = _shape_list(parameter)
    if shape is not None:
        descriptor["shape"] = shape
    dtype = _dtype_name(parameter)
    if dtype is not None:
        descriptor["dtype"] = dtype
    return descriptor


def _extract_grouping_from_groups(param_groups: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    values: dict[str, int | None] = {field: None for field in _GROUPING_FIELDS.values()}
    seen_roles: set[str] = set()
    for group_index, group in enumerate(param_groups):
        role = group.get("logical_role")
        if role not in _GROUPING_FIELDS or group.get("num_heads") is None:
            continue
        num_heads = group.get("num_heads")
        heads_per_group = group.get("heads_per_group")
        if not _is_int(num_heads) or num_heads <= 0:
            raise RuntimeStateError(
                f"Group {group_index} role={role} has invalid num_heads={num_heads!r}"
            )
        if heads_per_group is None:
            heads_per_group = 1
        if (
            not _is_int(heads_per_group)
            or heads_per_group <= 0
            or num_heads % heads_per_group != 0
        ):
            raise RuntimeStateError(
                f"Group {group_index} heads_per_group={heads_per_group!r} "
                f"must divide {role} num_heads={num_heads}"
            )
        field = _GROUPING_FIELDS[role]
        if role in seen_roles and values[field] != heads_per_group:
            raise RuntimeStateError(
                f"Live {role} groups disagree on heads_per_group: "
                f"{values[field]} versus {heads_per_group}"
            )
        values[field] = heads_per_group
        seen_roles.add(role)
    return values


def _validate_runtime_grouping_state(state: Mapping[str, Any]) -> None:
    version = state.get("schema_version")
    if not _is_int(version) or version != RUNTIME_STATE_SCHEMA_VERSION:
        raise RuntimeStateSchemaError(
            f"Unsupported runtime grouping schema_version={version!r}; "
            f"expected {RUNTIME_STATE_SCHEMA_VERSION}"
        )
    if state.get("artifact_type") != "runtime_grouping_state":
        raise RuntimeStateError("Runtime grouping artifact_type must be 'runtime_grouping_state'")
    for field in _GROUPING_FIELDS.values():
        value = state.get(field)
        if value is not None and (not _is_int(value) or value <= 0):
            raise RuntimeStateError(f"{field} must be a positive integer or None")
    for field in ("active_stage", "transition_count"):
        value = state.get(field)
        if not _is_int(value) or value < 0:
            raise RuntimeStateError(f"{field} must be a nonnegative integer")
    last_step = state.get("last_transition_step")
    if last_step is not None and (not _is_int(last_step) or last_step < 0):
        raise RuntimeStateError("last_transition_step must be a nonnegative integer or None")
    if state["transition_count"] == 0 and last_step is not None:
        raise RuntimeStateError("last_transition_step requires transition_count > 0")
    if state["active_stage"] != state["transition_count"]:
        raise RuntimeStateError(
            "active_stage and transition_count must agree for grouping transitions"
        )


def build_initial_grouping_definition(optimizer: Any) -> dict[str, Any]:
    """Capture immutable step-zero grouping from the live optimizer."""
    return {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "artifact_type": "initial_grouping",
        **_extract_grouping_from_groups(optimizer.param_groups),
    }


def initialize_runtime_grouping_state(optimizer: Any) -> dict[str, Any]:
    """Initialize explicit mutable grouping state from live optimizer groups."""
    state = {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "artifact_type": "runtime_grouping_state",
        **_extract_grouping_from_groups(optimizer.param_groups),
        "active_stage": 0,
        "transition_count": 0,
        "last_transition_step": None,
    }
    setattr(optimizer, _RUNTIME_STATE_ATTR, deepcopy(state))
    return state


def get_active_grouping_state(optimizer: Any) -> dict[str, Any]:
    """Return explicit runtime bookkeeping cross-checked against live groups."""
    live = _extract_grouping_from_groups(optimizer.param_groups)
    recorded = getattr(optimizer, _RUNTIME_STATE_ATTR, None)
    if recorded is None:
        return initialize_runtime_grouping_state(optimizer)
    if not isinstance(recorded, Mapping):
        raise RuntimeStateError(f"{_RUNTIME_STATE_ATTR} must be a mapping")
    _validate_runtime_grouping_state(recorded)
    for field, value in live.items():
        if recorded.get(field) != value:
            raise RuntimeStateError(
                f"Recorded {field}={recorded.get(field)!r} disagrees with live value={value!r}"
            )
    return deepcopy(dict(recorded))


def synchronize_runtime_grouping_state_(
    optimizer: Any,
    *,
    transition_step: int | None = None,
) -> dict[str, Any]:
    """Record a grouping mutation already performed by a legacy low-level caller.

    New integrations should use transition_attention_grouping_ so validation and mutation are one
    operation. This adapter exists while the generated v0.5 trainer still calls the original
    set_attention_grouping helper on every step after its phase boundary.
    """
    live = _extract_grouping_from_groups(optimizer.param_groups)
    recorded = getattr(optimizer, _RUNTIME_STATE_ATTR, None)
    if recorded is None:
        return initialize_runtime_grouping_state(optimizer)
    if not isinstance(recorded, Mapping):
        raise RuntimeStateError(f"{_RUNTIME_STATE_ATTR} must be a mapping")
    _validate_runtime_grouping_state(recorded)
    if all(recorded.get(field) == value for field, value in live.items()):
        return deepcopy(dict(recorded))

    if transition_step is not None:
        if not _is_int(transition_step) or transition_step < 0:
            raise RuntimeStateError("transition_step must be a nonnegative integer or None")
        previous_step = recorded["last_transition_step"]
        if previous_step is not None and transition_step <= previous_step:
            raise RuntimeStateError(
                f"transition_step={transition_step} must be greater than previous step={previous_step}"
            )
    updated = {
        **dict(recorded),
        **live,
        "active_stage": recorded["active_stage"] + 1,
        "transition_count": recorded["transition_count"] + 1,
        "last_transition_step": transition_step,
    }
    _validate_runtime_grouping_state(updated)
    setattr(optimizer, _RUNTIME_STATE_ATTR, deepcopy(updated))
    return deepcopy(updated)


def _grouping_proposals(
    param_groups: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> list[tuple[dict[str, Any], int]]:
    proposals: list[tuple[dict[str, Any], int]] = []
    roles_present: set[str] = set()
    for group_index, raw_group in enumerate(param_groups):
        if not isinstance(raw_group, dict):
            raise RuntimeStateError(f"Optimizer group {group_index} must be mutable")
        role = raw_group.get("logical_role")
        if role not in _GROUPING_FIELDS or raw_group.get("num_heads") is None:
            continue
        roles_present.add(role)
        num_heads = raw_group.get("num_heads")
        target = state.get(_GROUPING_FIELDS[role])
        if target is None:
            raise RuntimeStateError(f"Runtime grouping state omits active role {role}")
        if (
            not _is_int(num_heads)
            or num_heads <= 0
            or not _is_int(target)
            or target <= 0
            or num_heads % target != 0
        ):
            raise RuntimeStateError(
                f"heads_per_group={target!r} must divide {role} num_heads={num_heads!r}"
            )
        proposals.append((raw_group, target))
    for role, field in _GROUPING_FIELDS.items():
        if role not in roles_present and state.get(field) is not None:
            raise RuntimeStateError(
                f"Runtime grouping specifies {field} but optimizer has no grouped {role} parameters"
            )
    return proposals


def apply_active_grouping_state_(
    optimizer: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Transactionally restore grouping and transition bookkeeping."""
    _validate_runtime_grouping_state(state)
    proposals = _grouping_proposals(optimizer.param_groups, state)
    for group, heads_per_group in proposals:
        group["heads_per_group"] = heads_per_group
    restored = deepcopy(dict(state))
    setattr(optimizer, _RUNTIME_STATE_ATTR, restored)
    get_active_grouping_state(optimizer)
    return deepcopy(restored)


def validate_saved_grouping_state(
    runtime_state: Mapping[str, Any],
    saved_optimizer_state: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> None:
    """Cross-check explicit checkpoint state against serialized optimizer param groups."""
    _validate_runtime_grouping_state(runtime_state)
    if isinstance(saved_optimizer_state, Mapping):
        param_groups = saved_optimizer_state.get("param_groups")
        if param_groups is None:
            raise RuntimeStateError("Saved optimizer state is missing param_groups")
    else:
        param_groups = saved_optimizer_state
    if not isinstance(param_groups, Sequence):
        raise RuntimeStateError("Saved optimizer param_groups must be a sequence")
    saved = _extract_grouping_from_groups(param_groups)
    for field, value in saved.items():
        if runtime_state.get(field) != value:
            raise RuntimeStateError(
                f"Checkpoint runtime {field}={runtime_state.get(field)!r} disagrees "
                f"with saved optimizer param_groups value={value!r}"
            )


def transition_attention_grouping_(
    optimizer: Any,
    *,
    heads_per_group_q: int,
    heads_per_group_kv: int,
    transition_step: int | None,
    target_stage: int | None = None,
    allow_idempotent: bool = False,
) -> dict[str, Any]:
    """Apply one validated stage transition and prevent duplicate execution."""
    current = get_active_grouping_state(optimizer)
    if not _is_int(heads_per_group_q) or not _is_int(heads_per_group_kv):
        raise RuntimeStateError("heads_per_group_q and heads_per_group_kv must be integers")
    desired = {
        "q_heads_per_group": (
            heads_per_group_q if current["q_heads_per_group"] is not None else None
        ),
        "k_heads_per_group": (
            heads_per_group_kv if current["k_heads_per_group"] is not None else None
        ),
        "v_heads_per_group": (
            heads_per_group_kv if current["v_heads_per_group"] is not None else None
        ),
    }
    if all(value is None for value in desired.values()):
        raise RuntimeStateError("Optimizer has no grouped Q/K/V parameter groups")
    if all(current[field] == value for field, value in desired.items()):
        if allow_idempotent and target_stage is None:
            return current
        raise DuplicateTransitionError(
            "Grouping transition already applied; refusing duplicate execution"
        )

    if transition_step is not None:
        if not _is_int(transition_step) or transition_step < 0:
            raise RuntimeStateError("transition_step must be a nonnegative integer or None")
        previous_step = current["last_transition_step"]
        if previous_step is not None and transition_step <= previous_step:
            raise RuntimeStateError(
                f"transition_step={transition_step} must be greater than previous step={previous_step}"
            )
    next_stage = current["active_stage"] + 1
    if target_stage is not None:
        if not _is_int(target_stage) or target_stage != next_stage:
            raise RuntimeStateError(
                f"target_stage must advance exactly once: expected {next_stage}, got {target_stage!r}"
            )
        next_stage = target_stage

    updated = {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "artifact_type": "runtime_grouping_state",
        **desired,
        "active_stage": next_stage,
        "transition_count": current["transition_count"] + 1,
        "last_transition_step": transition_step,
    }
    return apply_active_grouping_state_(optimizer, updated)


def build_live_group_signature(optimizer: Any) -> dict[str, Any]:
    """Describe structural group identity while excluding scheduled mutable values."""
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        entry: dict[str, Any] = {"group_index": group_index}
        for field in _SHARED_STRUCTURAL_FIELDS:
            if field in group:
                entry[field] = group[field]
        if group.get("kind") == "muon":
            for field in _MUON_STRUCTURAL_FIELDS:
                if field in group:
                    entry[field] = group[field]
        entry["parameters"] = [
            _param_descriptor(parameter, parameter_index)
            for parameter_index, parameter in enumerate(group.get("params", ()))
        ]
        groups.append(canonicalize(entry))
    return {
        "schema_version": GROUP_SIGNATURE_SCHEMA_VERSION,
        "artifact_type": "live_parameter_group_signature",
        "groups": groups,
    }


def validate_live_group_signature(
    optimizer: Any,
    expected_signature: Mapping[str, Any],
) -> None:
    """Compare current structural identity with a checkpoint signature."""
    if expected_signature.get("schema_version") != GROUP_SIGNATURE_SCHEMA_VERSION:
        raise RuntimeStateSchemaError(
            f"Unsupported group signature schema_version="
            f"{expected_signature.get('schema_version')!r}"
        )
    actual = build_live_group_signature(optimizer)
    if canonicalize(expected_signature) != canonicalize(actual):
        raise GroupSignatureMismatch(
            "Live optimizer parameter-group structure does not match checkpoint metadata"
        )


def validate_live_runtime_state(
    optimizer: Any,
    *,
    expected_grouping_state: Mapping[str, Any],
    expected_group_signature: Mapping[str, Any],
) -> None:
    """Validate grouping and structural identity before or after optimizer-state load."""
    _validate_runtime_grouping_state(expected_grouping_state)
    live = get_active_grouping_state(optimizer)
    if canonicalize(live) != canonicalize(expected_grouping_state):
        raise RuntimeStateError("Live runtime grouping state does not match checkpoint metadata")
    validate_live_group_signature(optimizer, expected_group_signature)


def _expected_variance_shape(
    parameter_shape: Sequence[int],
    *,
    update_rule: str,
    normuon_mode: str,
) -> list[int]:
    if len(parameter_shape) != 2:
        raise RuntimeStateError(
            f"Muon state schema requires matrix parameters, got shape={list(parameter_shape)}"
        )
    rows, columns = parameter_shape
    if update_rule == "fractional_ef":
        return [min(rows, columns), 1]
    if normuon_mode == "factored" and rows < columns:
        return [1, columns]
    return [rows, 1]


def build_expected_state_schema(
    optimizer: Any,
    *,
    world_size: int = 1,
    ownership_policy: str = "contiguous_padded_matrix_chunks",
) -> dict[str, Any]:
    """Describe state that may be allocated later without claiming it already exists."""
    if not _is_int(world_size) or world_size <= 0:
        raise RuntimeStateError("world_size must be a positive integer")
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = list(group.get("params", ()))
        entry: dict[str, Any] = {
            "group_index": group_index,
            "kind": group.get("kind"),
            "logical_role": group.get("logical_role"),
            "parameter_shapes": [_shape_list(parameter) for parameter in parameters],
        }
        if group.get("kind") == "muon":
            update_rule = group.get("update_rule", "classic")
            chunk_size = math.ceil(len(parameters) / world_size) if parameters else 0
            owners = [
                min(parameter_index // max(chunk_size, 1), world_size - 1)
                for parameter_index in range(len(parameters))
            ]
            states = [
                {
                    "name": "research_momentum",
                    "semantics": (
                        "error_feedback" if update_rule == "fractional_ef" else "momentum"
                    ),
                    "dtype_policy": "parameter_dtype",
                    "allocation": "lazy_first_parameter_sharded_batch",
                }
            ]
            if bool(group.get("normuon")):
                first_shape = _shape_list(parameters[0]) if parameters else None
                states.append(
                    {
                        "name": "research_variance",
                        "semantics": "second_moment",
                        "dtype_policy": "float32",
                        "allocation": "lazy_first_parameter_sharded_batch",
                        "per_matrix_shape": (
                            _expected_variance_shape(
                                first_shape,
                                update_rule=update_rule,
                                normuon_mode=group.get("normuon_mode", "row"),
                            )
                            if first_shape is not None
                            else None
                        ),
                    }
                )
            entry.update(
                {
                    "update_rule": update_rule,
                    "orthogonalizer": group.get("orthogonalizer"),
                    "coefficients": group.get("coefficients"),
                    "normuon_mode": (
                        group.get("normuon_mode") if bool(group.get("normuon")) else None
                    ),
                    "num_heads": group.get("num_heads"),
                    "heads_per_group": group.get("heads_per_group"),
                    "chunk_size": chunk_size,
                    "owner_rank_by_parameter_index": owners,
                    "expected_states": states,
                }
            )
        else:
            entry["expected_states"] = [
                {
                    "name": "exp_avg",
                    "semantics": "first_moment",
                    "dtype_policy": "optimizer_defined",
                    "allocation": "lazy_per_parameter",
                },
                {
                    "name": "exp_avg_sq",
                    "semantics": "second_moment",
                    "dtype_policy": "optimizer_defined",
                    "allocation": "lazy_per_parameter",
                },
            ]
        groups.append(canonicalize(entry))
    return {
        "schema_version": EXPECTED_STATE_SCHEMA_VERSION,
        "artifact_type": "expected_optimizer_state_schema",
        "world_size": world_size,
        "ownership_policy": ownership_policy,
        "active_grouping": get_active_grouping_state(optimizer),
        "groups": groups,
    }


def _finite_status(value: Any) -> bool:
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        result = isfinite()
        all_values = getattr(result, "all", None)
        if callable(all_values):
            result = all_values()
        item = getattr(result, "item", None)
        return bool(item() if callable(item) else result)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    raise RuntimeStateError(
        f"Cannot perform finite-value check for state type {type(value).__name__}"
    )


def _observed_state_value(name: str, value: Any, *, check_finite: bool) -> dict[str, Any]:
    shape = _shape_list(value)
    if shape is not None:
        record: dict[str, Any] = {
            "name": name,
            "kind": "tensor",
            "shape": shape,
            "dtype": _dtype_name(value),
            "device_class": _device_class(value),
        }
    elif value is None or isinstance(value, (bool, str, int, float)):
        record = {
            "name": name,
            "kind": "scalar",
            "value_type": type(value).__name__,
        }
    else:
        record = {
            "name": name,
            "kind": "object",
            "value_type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    if check_finite and (
        shape is not None or isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        record["finite"] = _finite_status(value)
    return record


def build_observed_state_signature(
    optimizer: Any,
    *,
    rank: int = 0,
    world_size: int = 1,
    check_finite: bool = False,
) -> dict[str, Any]:
    """Describe rank-local state that actually exists immediately before checkpointing."""
    if (
        not _is_int(world_size)
        or world_size <= 0
        or not _is_int(rank)
        or rank < 0
        or rank >= world_size
    ):
        raise RuntimeStateError(f"Invalid rank/world_size combination: {rank}/{world_size}")

    locations: dict[int, tuple[int, int]] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group.get("params", ())):
            locations[id(parameter)] = (group_index, parameter_index)

    entries: list[dict[str, Any]] = []
    ownership: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameter_count = len(group.get("params", ()))
        chunk_size = math.ceil(parameter_count / world_size) if parameter_count else 0
        start = rank * chunk_size
        stop = min(start + chunk_size, parameter_count)
        ownership.append(
            {
                "group_index": group_index,
                "chunk_size": chunk_size,
                "owned_parameter_indices": list(range(start, stop)),
                "padded_slots": max(0, chunk_size - max(0, stop - start)),
            }
        )
    state = getattr(optimizer, "state", {})
    if not isinstance(state, Mapping):
        raise RuntimeStateError("optimizer.state must be a mapping")
    for parameter, values in state.items():
        location = locations.get(id(parameter))
        if location is None:
            raise RuntimeStateError("Optimizer state contains an unmapped parameter")
        if not isinstance(values, Mapping):
            raise RuntimeStateError("Per-parameter optimizer state must be a mapping")
        group_index, parameter_index = location
        state_values = [
            _observed_state_value(str(name), value, check_finite=check_finite)
            for name, value in sorted(values.items(), key=lambda item: str(item[0]))
        ]
        entries.append(
            {
                "group_index": group_index,
                "parameter_index": parameter_index,
                "owner_rank": rank,
                "states": state_values,
            }
        )
    entries.sort(key=lambda item: (item["group_index"], item["parameter_index"]))
    return {
        "schema_version": OBSERVED_STATE_SCHEMA_VERSION,
        "artifact_type": "observed_optimizer_state_signature",
        "rank": rank,
        "world_size": world_size,
        "finite_values_checked": bool(check_finite),
        "active_grouping": get_active_grouping_state(optimizer),
        "ownership": ownership,
        "entries": canonicalize(entries),
    }
