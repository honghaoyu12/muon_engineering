"""Deterministic provenance records for Muon benchmark runs.

This module deliberately has no PyTorch dependency. It defines the stable serialization,
fingerprinting, resume-mode, and append-only artifact contracts that the NanoChat trainer can
call during the M2 integration pass.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePath
import re
from typing import Any, Callable, Mapping
from uuid import uuid4


PROVENANCE_SCHEMA_VERSION = 1
RESUME_MODES = frozenset({"exact", "trajectory-change", "weights-only"})
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ProvenanceError(ValueError):
    """Base error for invalid or incompatible provenance."""


class SchemaVersionError(ProvenanceError):
    """Raised when an artifact uses an unsupported schema version."""


class ResumeValidationError(ProvenanceError):
    """Raised when requested resume semantics do not match the saved state."""


def _normalize_dtype(value: Any) -> str | None:
    cls = type(value)
    module = cls.__module__
    name = cls.__name__.lower()
    if (module == "torch" and name == "dtype") or (
        module.startswith("numpy") and "dtype" in name
    ):
        return str(value).removeprefix("torch.")
    return None


def canonicalize(value: Any) -> Any:
    """Convert supported values to a deterministic JSON-compatible representation.

    Object repr output and absolute Path values are rejected because they commonly contain
    process-specific addresses or machine-specific directory prefixes.
    """
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProvenanceError("Canonical JSON does not permit NaN or infinite floats")
        return 0.0 if value == 0.0 else value
    if isinstance(value, PurePath):
        if value.is_absolute():
            raise ProvenanceError(f"Absolute paths are not stable provenance: {value}")
        return value.as_posix()
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(
                    f"Canonical JSON mapping keys must be strings, got {type(key).__name__}"
                )
            out[key] = canonicalize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))

    dtype = _normalize_dtype(value)
    if dtype is not None:
        return dtype
    cls = type(value)
    if cls.__module__ == "torch" and cls.__name__ == "device":
        raise ProvenanceError("Device objects are not stable provenance; record a device class")
    raise ProvenanceError(
        f"Unsupported canonical JSON value {cls.__module__}.{cls.__qualname__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize a value using the project's single canonical JSON encoding."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def validate_schema_version(
    artifact: Mapping[str, Any],
    *,
    expected: int = PROVENANCE_SCHEMA_VERSION,
    artifact_name: str = "artifact",
) -> None:
    """Fail closed when a schema version is absent or unsupported."""
    actual = artifact.get("schema_version")
    if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
        raise SchemaVersionError(
            f"Unsupported {artifact_name} schema_version={actual!r}; expected {expected}"
        )


def fingerprint_payload(
    artifact: Mapping[str, Any],
    *,
    expected_schema_version: int = PROVENANCE_SCHEMA_VERSION,
) -> str:
    """Hash a schema-versioned artifact's canonical UTF-8 bytes."""
    validate_schema_version(artifact, expected=expected_schema_version)
    return hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()


def build_fingerprint(
    fingerprint_type: str,
    payload: Mapping[str, Any],
    *,
    schema_version: int = PROVENANCE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return an auditable fingerprint record containing normalized inputs and SHA-256."""
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != PROVENANCE_SCHEMA_VERSION
    ):
        raise SchemaVersionError(
            f"Unsupported fingerprint schema_version={schema_version!r}; "
            f"expected {PROVENANCE_SCHEMA_VERSION}"
        )
    if not isinstance(payload, Mapping):
        raise ProvenanceError("Fingerprint payload must be a mapping")
    if not fingerprint_type or not isinstance(fingerprint_type, str):
        raise ProvenanceError("fingerprint_type must be a non-empty string")
    envelope = {
        "schema_version": schema_version,
        "fingerprint_type": fingerprint_type,
        "payload": canonicalize(payload),
    }
    return {**envelope, "sha256": fingerprint_payload(envelope)}


def state_compatibility_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint immutable optimizer/model topology plus checkpoint-local grouping."""
    return build_fingerprint("state_compatibility", payload)


def trajectory_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint resolved future training-schedule and data identity."""
    return build_fingerprint("training_trajectory", payload)


def verify_fingerprint(
    record: Mapping[str, Any],
    *,
    expected_type: str | None = None,
) -> None:
    """Validate a fingerprint record and raise on corruption or schema drift."""
    validate_schema_version(record, artifact_name="fingerprint")
    if expected_type is not None and record.get("fingerprint_type") != expected_type:
        raise ProvenanceError(
            f"Expected fingerprint_type={expected_type!r}, got "
            f"{record.get('fingerprint_type')!r}"
        )
    supplied = record.get("sha256")
    if not isinstance(supplied, str):
        raise ProvenanceError("Fingerprint record is missing a string sha256")
    envelope = {key: value for key, value in record.items() if key != "sha256"}
    expected = fingerprint_payload(envelope)
    if not hmac.compare_digest(supplied, expected):
        raise ProvenanceError(
            f"Fingerprint digest mismatch: recorded={supplied}, calculated={expected}"
        )


def validate_resume_mode(
    mode: str,
    *,
    state_compatible: bool,
    trajectory_compatible: bool,
    optimizer_state_present: bool,
) -> str:
    """Enforce the optimizer-state contract for each explicit resume mode."""
    if mode not in RESUME_MODES:
        raise ResumeValidationError(
            f"Unknown resume mode {mode!r}; choose from {sorted(RESUME_MODES)}"
        )
    if mode == "weights-only":
        if optimizer_state_present:
            raise ResumeValidationError(
                "weights-only resume must not load saved optimizer state"
            )
        return mode

    if not optimizer_state_present:
        raise ResumeValidationError(f"{mode} resume requires optimizer state")
    if not state_compatible:
        raise ResumeValidationError(
            f"{mode} resume requires a matching state-compatibility fingerprint"
        )
    if mode == "exact" and not trajectory_compatible:
        raise ResumeValidationError(
            "exact resume requires a matching training-trajectory fingerprint"
        )
    return mode


def _validate_seed_integer(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value >= 2**63
    ):
        raise ProvenanceError(f"{name} must be an integer in [0, 2**63)")
    return value


def derive_rank_seed(base_seed: int, rank: int, *, namespace: str = "training") -> int:
    """Derive a stable rank-local seed without changing the shared model seed."""
    base_seed = _validate_seed_integer(base_seed, "base_seed")
    rank = _validate_seed_integer(rank, "rank")
    if not isinstance(namespace, str) or not namespace:
        raise ProvenanceError("Seed namespace must be a non-empty string")
    material = canonical_json(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "base_seed": base_seed,
            "rank": rank,
            "namespace": namespace,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def build_seed_policy(
    base_seed: int,
    *,
    world_size: int,
    controlled_streams: tuple[str, ...] = ("torch_cpu", "torch_cuda"),
    rank_local_namespaces: tuple[str, ...] = ("training",),
) -> dict[str, Any]:
    """Represent shared model initialization and post-sync rank-local RNG streams."""
    base_seed = _validate_seed_integer(base_seed, "base_seed")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ProvenanceError("world_size must be a positive integer")
    if not isinstance(controlled_streams, (list, tuple)) or not controlled_streams:
        raise ProvenanceError("controlled_streams must be a non-empty sequence")
    if any(not isinstance(stream, str) or not stream for stream in controlled_streams):
        raise ProvenanceError("controlled_streams must contain non-empty strings")
    if not isinstance(rank_local_namespaces, (list, tuple)) or not rank_local_namespaces:
        raise ProvenanceError("rank_local_namespaces must be a non-empty sequence")
    if any(not isinstance(namespace, str) or not namespace for namespace in rank_local_namespaces):
        raise ProvenanceError("rank_local_namespaces must contain non-empty strings")
    streams = sorted(set(controlled_streams))
    namespaces = sorted(set(rank_local_namespaces))
    rank_local = [
        {
            "rank": rank,
            "seeds": {
                namespace: derive_rank_seed(base_seed, rank, namespace=namespace)
                for namespace in namespaces
            },
        }
        for rank in range(world_size)
    ]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_type": "seed_policy",
        "base_seed": base_seed,
        "controlled_streams": streams,
        "model_initialization": {
            "scope": "shared_across_ranks",
            "seed": base_seed,
        },
        "rank_local_streams": {
            "begin_after": "synchronized_model_initialization",
            "derivation": "sha256-v1",
            "namespaces": namespaces,
            "ranks": rank_local,
        },
        "resume_rule": "restore_rng_positions_do_not_reseed",
        "world_size": world_size,
    }


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ProvenanceError(
            "run_id must be 1-128 characters using only letters, digits, '.', '_', or '-'"
        )


def prepare_run_directory(
    base_directory: str | os.PathLike[str],
    run_id: str,
    *,
    resume: bool = False,
) -> Path:
    """Create or validate a unique run directory without silently reusing prior output."""
    _validate_run_id(run_id)
    path = Path(base_directory) / run_id
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"Resume run directory does not exist: {path}")
        return path
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Refusing to reuse nonempty run directory: {path}")
        return path
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_run_manifest(
    run_id: str,
    *,
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
    model: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable, schema-versioned identity record for one run."""
    _validate_run_id(run_id)
    return canonicalize(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "artifact_type": "run_manifest",
            "run_id": run_id,
            "code": code,
            "environment": environment,
            "model": model,
            "optimizer": optimizer,
            "trajectory": trajectory,
            "data": data,
        }
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_immutable_manifest(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    rank: int = 0,
    barrier: Callable[[], None] | None = None,
) -> str:
    """Publish a manifest once from rank 0, then optionally synchronize all ranks."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ProvenanceError("rank must be a nonnegative integer")
    validate_schema_version(manifest, artifact_name="run manifest")
    if manifest.get("artifact_type") != "run_manifest":
        raise ProvenanceError("Manifest artifact_type must be 'run_manifest'")
    digest = fingerprint_payload(manifest)
    destination = Path(path)

    if rank == 0:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        payload = (canonical_json(manifest) + "\n").encode("utf-8")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite immutable run manifest: {destination}"
                ) from exc
            _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
    if barrier is not None:
        barrier()
    if not destination.is_file():
        raise ProvenanceError(f"Published run manifest is missing after synchronization: {destination}")
    try:
        published = json.loads(destination.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"Published run manifest is invalid JSON: {exc}") from exc
    published_digest = fingerprint_payload(published)
    if not hmac.compare_digest(digest, published_digest):
        raise ProvenanceError(
            f"Published run manifest digest mismatch: expected={digest}, "
            f"calculated={published_digest}"
        )
    return digest


def read_runtime_events(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read and validate an event log, including strictly increasing sequence numbers."""
    source = Path(path)
    if not source.exists():
        return []
    events: list[dict[str, Any]] = []
    previous = -1
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProvenanceError(
                    f"Invalid runtime event JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ProvenanceError(f"Runtime event line {line_number} is not an object")
            validate_schema_version(event, artifact_name="runtime event")
            if event.get("artifact_type") != "runtime_event":
                raise ProvenanceError(
                    f"Runtime event line {line_number} has invalid artifact_type"
                )
            sequence = event.get("sequence_number")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= previous:
                raise ProvenanceError(
                    f"Runtime event sequence must increase at line {line_number}; "
                    f"previous={previous}, current={sequence!r}"
                )
            previous = sequence
            events.append(event)
    return events


def append_runtime_event(
    path: str | os.PathLike[str],
    event: Mapping[str, Any],
    *,
    rank: int = 0,
) -> dict[str, Any]:
    """Append one canonical event from rank 0 and fsync it before returning."""
    if rank != 0:
        raise ProvenanceError("Only rank 0 may append runtime events")
    reserved = {"schema_version", "artifact_type"}
    overlap = reserved.intersection(event)
    if overlap:
        raise ProvenanceError(f"Runtime event cannot override reserved fields: {sorted(overlap)}")
    record = canonicalize(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "artifact_type": "runtime_event",
            **event,
        }
    )
    sequence = record.get("sequence_number")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ProvenanceError("Runtime event sequence_number must be a nonnegative integer")
    if not isinstance(record.get("event"), str) or not record["event"]:
        raise ProvenanceError("Runtime event requires a non-empty event string")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        handle.seek(0)
        previous = -1
        for existing in handle:
            if not existing.strip():
                continue
            try:
                parsed = json.loads(existing)
            except json.JSONDecodeError as exc:
                raise ProvenanceError(f"Existing runtime event log is corrupt: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ProvenanceError("Existing runtime event log contains a non-object")
            validate_schema_version(parsed, artifact_name="runtime event")
            if parsed.get("artifact_type") != "runtime_event":
                raise ProvenanceError("Existing runtime event has invalid artifact_type")
            existing_sequence = parsed.get("sequence_number")
            if (
                not isinstance(existing_sequence, int)
                or isinstance(existing_sequence, bool)
                or existing_sequence <= previous
            ):
                raise ProvenanceError("Existing runtime event sequence is not monotonic")
            previous = existing_sequence
        if sequence <= previous:
            raise ProvenanceError(
                f"Runtime event sequence must increase; previous={previous}, current={sequence}"
            )
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def aggregate_rank_local_signatures(
    signatures_by_rank: Mapping[int, Mapping[str, Any]],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Build a deterministic aggregate and reject missing or unexpected rank shards."""
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ProvenanceError("world_size must be a positive integer")
    actual = set(signatures_by_rank)
    expected = set(range(world_size))
    if actual != expected:
        raise ProvenanceError(
            f"Incomplete rank signatures: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    signatures = []
    for rank in range(world_size):
        signature = signatures_by_rank[rank]
        if not isinstance(signature, Mapping):
            raise ProvenanceError(f"Rank {rank} signature must be a mapping")
        if "schema_version" not in signature:
            raise ProvenanceError(f"Rank {rank} signature is missing schema_version")
        validate_schema_version(signature, artifact_name=f"rank {rank} signature")
        recorded_rank = signature.get("rank")
        if recorded_rank is not None and recorded_rank != rank:
            raise ProvenanceError(
                f"Rank signature key/content mismatch: key={rank}, recorded={recorded_rank}"
            )
        signatures.append({"rank": rank, "signature": canonicalize(signature)})
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_type": "rank_local_state_signatures",
        "world_size": world_size,
        "signatures": signatures,
    }
