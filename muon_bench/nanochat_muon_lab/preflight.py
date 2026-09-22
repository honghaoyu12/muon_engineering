"""Non-mutating environment and checkout preflight for NanoChat Muon runs."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _find_repo_root() -> Path | None:
    candidates = []
    configured = os.environ.get("NANOCHAT_REPO_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([Path.cwd(), Path.cwd() / "nanochat", Path(__file__).resolve().parent.parent])
    for candidate in candidates:
        candidate = candidate.resolve()
        if not ((candidate / "nanochat" / "optim.py").is_file() or (candidate / "optim.py").is_file()):
            continue
        if (candidate / ".git").exists() or (candidate.parent / ".git").exists():
            return candidate
    return None


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    result: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "distributed_available": bool(torch.distributed.is_available()),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            result["gpus"].append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            })
    try:
        from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
        result["compute_dtype"] = str(COMPUTE_DTYPE).removeprefix("torch.")
        result["compute_dtype_reason"] = COMPUTE_DTYPE_REASON
    except Exception as exc:
        result["compute_dtype_error"] = str(exc)
    try:
        from nanochat.flash_attention import HAS_FA3, USE_FA3
        result["flash_attention"] = {"has_fa3": bool(HAS_FA3), "use_fa3": bool(USE_FA3)}
    except Exception as exc:
        result["flash_attention"] = {"error": str(exc)}
    return result


def _gns_report() -> dict[str, Any]:
    spec = importlib.util.find_spec("gram_newton_schulz")
    try:
        version = importlib.metadata.version("gram-newton-schulz")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"module_found": spec is not None, "distribution_version": version}


def collect_preflight() -> dict[str, Any]:
    """Collect identities and capabilities without changing files or runtime state."""
    repo_root = _find_repo_root()
    git = {}
    if repo_root is not None:
        git = {
            "root": repo_root.name,
            "commit": _run_git(repo_root, "rev-parse", "HEAD"),
            "status": "clean" if _run_git(repo_root, "status", "--porcelain") == "" else "dirty",
        }
    try:
        import nanochat_muon_lab
        package_version = nanochat_muon_lab.__version__
    except ImportError:
        package_version = None
    return {
        "schema_version": 1,
        "artifact_type": "preflight_report",
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "package_version": package_version,
        "nanochat_git": git,
        "torch": _torch_report(),
        "gns": _gns_report(),
        "environment": {
            "NANOCHAT_BASE_DIR": os.environ.get("NANOCHAT_BASE_DIR"),
            "NANOCHAT_DTYPE": os.environ.get("NANOCHAT_DTYPE"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = collect_preflight()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"NanoChat Muon preflight (schema {report['schema_version']})")
        print(f"package={report['package_version']} python={report['python']}")
        git = report["nanochat_git"]
        print(f"nanochat={git.get('commit', 'unknown')} status={git.get('status', 'unknown')}")
        torch = report["torch"]
        print(f"torch={torch.get('version', 'unavailable')} cuda={torch.get('cuda_runtime')} available={torch.get('cuda_available')}")
        print(f"compute_dtype={torch.get('compute_dtype', 'unknown')} world_size={torch.get('world_size', 1)}")
        print(f"gpus={len(torch.get('gpus', []))} gns={report['gns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
