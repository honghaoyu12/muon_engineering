import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from nanochat_muon_lab.provenance import (
    ProvenanceError,
    ResumeValidationError,
    SchemaVersionError,
    aggregate_rank_local_signatures,
    append_runtime_event,
    build_fingerprint,
    build_run_manifest,
    canonical_json,
    canonicalize,
    build_seed_policy,
    derive_rank_seed,
    fingerprint_payload,
    prepare_run_directory,
    read_runtime_events,
    state_compatibility_fingerprint,
    trajectory_fingerprint,
    validate_resume_mode,
    verify_fingerprint,
    write_immutable_manifest,
)


class Mode(Enum):
    RESEARCH = "research"


@dataclass
class Config:
    beta: float


def test_canonical_json_normalizes_nested_values_and_is_order_independent():
    left = {"z": (1, 2), "a": {3, 1}, "mode": Mode.RESEARCH, "cfg": Config(0.9)}
    right = {"cfg": {"beta": 0.9}, "mode": "research", "a": [1, 3], "z": [1, 2]}
    assert canonical_json(left) == canonical_json(right)
    assert canonicalize(-0.0) == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Path("/absolute/path")])
def test_canonicalize_rejects_unstable_values(value):
    with pytest.raises(ProvenanceError):
        canonicalize(value)


def test_fingerprints_verify_and_fail_closed_on_mutation():
    record = state_compatibility_fingerprint({"world_size": 2, "roles": ["q", "k"]})
    verify_fingerprint(record, expected_type="state_compatibility")
    changed = dict(record)
    changed["payload"] = {"world_size": 4, "roles": ["q", "k"]}
    with pytest.raises(ProvenanceError, match="digest mismatch"):
        verify_fingerprint(changed)
    with pytest.raises(SchemaVersionError):
        verify_fingerprint({**record, "schema_version": 99})
    assert trajectory_fingerprint({"steps": 10})["fingerprint_type"] == "training_trajectory"


def test_resume_modes_enforce_optimizer_and_trajectory_contract():
    assert validate_resume_mode(
        "exact",
        state_compatible=True,
        trajectory_compatible=True,
        optimizer_state_present=True,
    ) == "exact"
    assert validate_resume_mode(
        "trajectory-change",
        state_compatible=True,
        trajectory_compatible=False,
        optimizer_state_present=True,
    ) == "trajectory-change"
    assert validate_resume_mode(
        "weights-only",
        state_compatible=False,
        trajectory_compatible=False,
        optimizer_state_present=False,
    ) == "weights-only"
    with pytest.raises(ResumeValidationError):
        validate_resume_mode(
            "exact",
            state_compatible=True,
            trajectory_compatible=False,
            optimizer_state_present=True,
        )
    with pytest.raises(ResumeValidationError):
        validate_resume_mode(
            "weights-only",
            state_compatible=True,
            trajectory_compatible=True,
            optimizer_state_present=True,
        )


def test_seed_policy_separates_shared_initialization_from_rank_local_streams():
    policy = build_seed_policy(
        1234,
        world_size=2,
        controlled_streams=("torch_cuda", "torch_cpu"),
        rank_local_namespaces=("dropout", "data"),
    )
    assert policy["model_initialization"]["seed"] == 1234
    assert policy["model_initialization"]["scope"] == "shared_across_ranks"
    assert policy["rank_local_streams"]["begin_after"] == "synchronized_model_initialization"
    rank_seeds = policy["rank_local_streams"]["ranks"]
    assert rank_seeds[0]["seeds"]["data"] != rank_seeds[1]["seeds"]["data"]
    assert derive_rank_seed(1234, 0, namespace="data") == rank_seeds[0]["seeds"]["data"]
    assert policy["resume_rule"] == "restore_rng_positions_do_not_reseed"
    with pytest.raises(ProvenanceError):
        build_seed_policy(-1, world_size=2)
    with pytest.raises(ProvenanceError):
        build_seed_policy(1, world_size=1, controlled_streams=([],))
    with pytest.raises(ProvenanceError):
        build_seed_policy(1, world_size=1, rank_local_namespaces=())


def _manifest(run_id="run-1"):
    return build_run_manifest(
        run_id,
        code={"nanochat_sha": "abc"},
        environment={"world_size": 1},
        model={"depth": 2},
        optimizer={"preset": "kj_reference"},
        trajectory={"num_iterations": 10},
        data={"dataset": "fixture"},
    )


def test_run_directory_refuses_implicit_reuse(tmp_path):
    run = prepare_run_directory(tmp_path, "run-1")
    assert run.is_dir()
    (run / "output").write_text("data")
    with pytest.raises(FileExistsError):
        prepare_run_directory(tmp_path, "run-1")
    assert prepare_run_directory(tmp_path, "run-1", resume=True) == run
    with pytest.raises(ProvenanceError):
        prepare_run_directory(tmp_path, "../escape")


def test_manifest_is_atomic_immutable_and_barrier_runs(tmp_path):
    path = tmp_path / "run" / "run_manifest.json"
    calls = []
    manifest = _manifest()
    digest = write_immutable_manifest(path, manifest, barrier=lambda: calls.append("barrier"))
    assert calls == ["barrier"]
    assert path.exists()
    assert json.loads(path.read_text()) == manifest
    assert digest == fingerprint_payload(manifest)
    with pytest.raises(FileExistsError):
        write_immutable_manifest(path, manifest)
    # Nonzero ranks synchronize, verify the rank-0 artifact, and never overwrite it.
    write_immutable_manifest(path, manifest, rank=1, barrier=lambda: calls.append("rank1"))
    assert calls[-1] == "rank1"


def test_runtime_events_are_canonical_monotonic_and_rank_zero_only(tmp_path):
    path = tmp_path / "runtime_events.jsonl"
    append_runtime_event(path, {"sequence_number": 0, "step": 0, "event": "start"})
    append_runtime_event(path, {"sequence_number": 1, "step": 5, "event": "checkpoint"})
    assert [event["sequence_number"] for event in read_runtime_events(path)] == [0, 1]
    with pytest.raises(ProvenanceError, match="must increase"):
        append_runtime_event(path, {"sequence_number": 1, "event": "duplicate"})
    with pytest.raises(ProvenanceError, match="rank 0"):
        append_runtime_event(path, {"sequence_number": 2, "event": "worker"}, rank=1)
    path.write_text(path.read_text() + '{"schema_version":1,"artifact_type":"runtime_event","sequence_number":1,"event":"bad"}\n')
    with pytest.raises(ProvenanceError):
        read_runtime_events(path)

    invalid = tmp_path / "invalid_events.jsonl"
    invalid.write_text('{"schema_version":1,"artifact_type":"wrong","sequence_number":0,"event":"bad"}\n')
    with pytest.raises(ProvenanceError, match="artifact_type"):
        append_runtime_event(invalid, {"sequence_number": 1, "event": "next"})


def test_rank_signature_aggregation_rejects_missing_rank():
    signatures = {
        0: {"schema_version": 1, "rank": 0, "entries": []},
        1: {"schema_version": 1, "rank": 1, "entries": []},
    }
    aggregate = aggregate_rank_local_signatures(signatures, world_size=2)
    assert [item["rank"] for item in aggregate["signatures"]] == [0, 1]
    with pytest.raises(ProvenanceError, match="Incomplete"):
        aggregate_rank_local_signatures({0: signatures[0]}, world_size=2)
