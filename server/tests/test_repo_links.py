"""Every link to the project points at its home, github.com/cB-Sigma/bolo-speech (the old repository is gone)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OLD = re.compile(
    r"(github\.com|githubusercontent\.com)/cb-sigma/bolo(?![-a-z0-9_])|cb-sigma\.github\.io/bolo(?![-a-z0-9_])", re.I
)


def tracked_text_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    skip = ("CHANGELOG.md",)
    return [ROOT / f for f in out.splitlines() if not f.endswith(skip) and not f.startswith("server/src/bolo/_ui/")]


def test_no_link_points_at_the_old_repository() -> None:
    found = []
    for path in tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if OLD.search(line):
                found.append(f"{path.relative_to(ROOT)}:{n}")
    assert not found, "links to the old repository: " + ", ".join(found)
