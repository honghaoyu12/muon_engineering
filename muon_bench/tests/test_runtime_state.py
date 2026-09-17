from types import SimpleNamespace

import pytest

from nanochat_muon_lab.runtime_state import (
    DuplicateTransitionError,
    GroupSignatureMismatch,
    RuntimeStateError,
    apply_active_grouping_state_,
    build_expected_state_schema,
    build_live_group_signature,
    build_observed_state_signature,
    get_active_grouping_state,
    initialize_runtime_grouping_state,
    synchronize_runtime_grouping_state_,
    transition_attention_grouping_,
    validate_live_group_signature,
    validate_live_runtime_state,
    validate_saved_grouping_state,
)


class FakeFiniteResult:
    def __init__(self, value=True):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, *shape, dtype="float32", device_type="cpu", finite=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self._finite = finite

    def isfinite(self):
        return FakeFiniteResult(self._finite)


def make_optimizer():
    q = [FakeTensor(8, 4), FakeTensor(8, 4)]
    k = [FakeTensor(4, 4), FakeTensor(4, 4)]
    v = [FakeTensor(4, 4), FakeTensor(4, 4)]
    scalar = [FakeTensor(1)]
    groups = [
        {
            "kind": "muon",
            "logical_role": "q",
            "num_heads": 4,
            "heads_per_group": 1,
            "params": q,
            "lr": 0.02,
            "weight_decay": 0.1,
            "update_rule": "classic",
            "orthogonalizer": "standard",
            "coefficients": "keller",
        },
        {
            "kind": "muon",
            "logical_role": "k",
            "num_heads": 2,
            "heads_per_group": 1,
            "params": k,
            "lr": 0.02,
            "weight_decay": 0.1,
            "update_rule": "classic",
            "orthogonalizer": "standard",
            "coefficients": "keller",
        },
        {
            "kind": "muon",
            "logical_role": "v",
            "num_heads": 2,
            "heads_per_group": 1,
            "params": v,
            "lr": 0.02,
            "weight_decay": 0.1,
            "update_rule": "classic",
            "orthogonalizer": "standard",
            "coefficients": "keller",
        },
        {"kind": "adamw", "logical_role": "scalar", "params": scalar, "lr": 0.1},
    ]
    return SimpleNamespace(param_groups=groups, state={})


def test_runtime_state_is_explicit_and_live_grouping_is_cross_checked():
    optimizer = make_optimizer()
    initial = initialize_runtime_grouping_state(optimizer)
    assert initial["q_heads_per_group"] == 1
    assert get_active_grouping_state(optimizer) == initial
    optimizer.param_groups[0]["heads_per_group"] = 2
    with pytest.raises(RuntimeStateError, match="disagrees"):
        get_active_grouping_state(optimizer)


def test_grouping_transition_is_transactional_and_duplicate_safe():
    optimizer = make_optimizer()
    transition = transition_attention_grouping_(
        optimizer,
        heads_per_group_q=2,
        heads_per_group_kv=2,
        transition_step=10,
        target_stage=1,
    )
    assert transition["transition_count"] == 1
    assert transition["last_transition_step"] == 10
    assert [group["heads_per_group"] for group in optimizer.param_groups[:3]] == [2, 2, 2]
    assert transition_attention_grouping_(
        optimizer,
        heads_per_group_q=2,
        heads_per_group_kv=2,
        transition_step=10,
        allow_idempotent=True,
    ) == transition
    with pytest.raises(DuplicateTransitionError):
        transition_attention_grouping_(
            optimizer,
            heads_per_group_q=2,
            heads_per_group_kv=2,
            transition_step=10,
        )

    before = [group["heads_per_group"] for group in optimizer.param_groups[:3]]
    with pytest.raises(RuntimeStateError, match="must divide"):
        transition_attention_grouping_(
            optimizer,
            heads_per_group_q=4,
            heads_per_group_kv=3,
            transition_step=20,
        )
    assert [group["heads_per_group"] for group in optimizer.param_groups[:3]] == before


def test_legacy_grouping_sync_records_one_transition_and_is_idempotent():
    optimizer = make_optimizer()
    initialize_runtime_grouping_state(optimizer)
    optimizer.param_groups[0]["heads_per_group"] = 2
    optimizer.param_groups[1]["heads_per_group"] = 2
    optimizer.param_groups[2]["heads_per_group"] = 2

    state = synchronize_runtime_grouping_state_(optimizer)
    assert state["q_heads_per_group"] == 2
    assert state["k_heads_per_group"] == 2
    assert state["v_heads_per_group"] == 2
    assert state["active_stage"] == state["transition_count"] == 1
    assert synchronize_runtime_grouping_state_(optimizer) == state


def test_restore_cross_checks_saved_optimizer_groups_and_post_load_signature():
    optimizer = make_optimizer()
    state = initialize_runtime_grouping_state(optimizer)
    saved = {"param_groups": [dict(group) for group in optimizer.param_groups]}
    validate_saved_grouping_state(state, saved)
    saved["param_groups"][1]["heads_per_group"] = 2
    with pytest.raises(RuntimeStateError, match="disagrees"):
        validate_saved_grouping_state(state, saved)

    optimizer = make_optimizer()
    state = transition_attention_grouping_(
        optimizer,
        heads_per_group_q=2,
        heads_per_group_kv=2,
        transition_step=10,
    )
    signature = build_live_group_signature(optimizer)
    # Scheduled mutable values are deliberately excluded from this structural signature.
    optimizer.param_groups[0]["lr"] = 0.001
    validate_live_runtime_state(
        optimizer,
        expected_grouping_state=state,
        expected_group_signature=signature,
    )
    optimizer.param_groups[0]["heads_per_group"] = 4
    with pytest.raises(GroupSignatureMismatch):
        validate_live_group_signature(optimizer, signature)


def test_expected_schema_is_lazy_and_observed_signature_describes_allocated_state():
    optimizer = make_optimizer()
    initialize_runtime_grouping_state(optimizer)
    expected = build_expected_state_schema(optimizer, world_size=2)
    q_schema = expected["groups"][0]
    assert q_schema["chunk_size"] == 1
    assert q_schema["expected_states"][0]["allocation"].startswith("lazy")
    assert "observed" not in q_schema

    parameter = optimizer.param_groups[0]["params"][0]
    optimizer.state[parameter] = {
        "research_momentum": FakeTensor(8, 4, finite=True),
        "step": 3,
    }
    observed = build_observed_state_signature(
        optimizer, rank=0, world_size=2, check_finite=True
    )
    assert observed["entries"][0]["states"][0]["name"] == "research_momentum"
    assert observed["entries"][0]["states"][0]["finite"] is True
    assert observed["entries"][0]["owner_rank"] == 0
    assert observed["ownership"][0]["owned_parameter_indices"] == [0]


def test_apply_active_grouping_state_validates_before_mutating():
    optimizer = make_optimizer()
    initialize_runtime_grouping_state(optimizer)
    invalid = {
        "schema_version": 1,
        "artifact_type": "runtime_grouping_state",
        "q_heads_per_group": 2,
        "k_heads_per_group": 3,
        "v_heads_per_group": 3,
        "active_stage": 1,
        "transition_count": 1,
        "last_transition_step": 10,
    }
    before = [group["heads_per_group"] for group in optimizer.param_groups[:3]]
    with pytest.raises(RuntimeStateError, match="must divide"):
        apply_active_grouping_state_(optimizer, invalid)
    assert [group["heads_per_group"] for group in optimizer.param_groups[:3]] == before
