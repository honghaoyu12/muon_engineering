from pathlib import Path
import re

import nanochat_muon_lab


def test_package_versions_are_consistent():
    root = Path(__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert match.group(1) == nanochat_muon_lab.__version__ == "0.6.0"
