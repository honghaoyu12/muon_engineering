"""Build a content-addressed manifest for project-owned NanoChat data artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .provenance import canonical_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path, *, split: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if split is not None:
        record["split"] = split
    return record


def build_dataset_manifest(
    data_dir: str | os.PathLike[str],
    *,
    tokenizer_dir: str | os.PathLike[str] | None = None,
    source: dict[str, Any] | None = None,
    validation_filename: str | None = None,
) -> dict[str, Any]:
    data_root = Path(data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    shards = sorted(
        path for path in data_root.iterdir()
        if path.is_file() and path.suffix == ".parquet" and not path.name.endswith(".tmp")
    )
    if not shards:
        raise ValueError(f"No parquet shards found in {data_root}")
    validation_filename = validation_filename or shards[-1].name
    files = [
        _file_record(path, data_root, split="validation" if path.name == validation_filename else "train")
        for path in shards
    ]
    tokenizer_files: list[dict[str, Any]] = []
    if tokenizer_dir is not None:
        tokenizer_root = Path(tokenizer_dir).expanduser().resolve()
        if not tokenizer_root.is_dir():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_root}")
        tokenizer_files = [
            _file_record(path, tokenizer_root)
            for path in sorted(tokenizer_root.iterdir())
            if path.is_file()
        ]
        if not tokenizer_files:
            raise ValueError(f"No tokenizer artifacts found in {tokenizer_root}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "dataset_manifest",
        "source": source or {},
        "data_files": files,
        "tokenizer_files": tokenizer_files,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def write_manifest(path: str | os.PathLike[str], manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-name", default="karpathy/climbmix-400b-shuffle")
    parser.add_argument("--source-url", default="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main")
    parser.add_argument("--validation-filename")
    args = parser.parse_args(argv)
    manifest = build_dataset_manifest(
        args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        source={"name": args.source_name, "url": args.source_url},
        validation_filename=args.validation_filename,
    )
    write_manifest(args.output, manifest)
    print(f"Wrote dataset manifest: {args.output}")
    print(f"Data files: {len(manifest['data_files'])}; tokenizer files: {len(manifest['tokenizer_files'])}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
