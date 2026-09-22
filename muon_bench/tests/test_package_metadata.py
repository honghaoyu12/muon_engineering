from pathlib import Path
import re

import nanochat_muon_lab


def test_package_versions_are_consistent():
    root = Path(__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert match.group(1) == nanochat_muon_lab.__version__ == "0.6.0"


def test_preflight_is_json_serializable():
    from nanochat_muon_lab.preflight import collect_preflight
    import json

    report = collect_preflight()
    assert report["artifact_type"] == "preflight_report"
    json.dumps(report)


def test_dataset_manifest_hashes_files(tmp_path):
    from nanochat_muon_lab.dataset_manifest import build_dataset_manifest

    data = tmp_path / "data"
    tokenizer = tmp_path / "tokenizer"
    data.mkdir()
    tokenizer.mkdir()
    (data / "shard_00000.parquet").write_bytes(b"train")
    (data / "shard_06542.parquet").write_bytes(b"validation")
    (tokenizer / "tokenizer.pkl").write_bytes(b"tokenizer")

    manifest = build_dataset_manifest(data, tokenizer_dir=tokenizer)
    assert manifest["artifact_type"] == "dataset_manifest"
    assert [item["split"] for item in manifest["data_files"]] == ["train", "validation"]
    assert manifest["tokenizer_files"][0]["path"] == "tokenizer.pkl"
    assert len(manifest["manifest_sha256"]) == 64
