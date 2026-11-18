"""``bolo-torch install <extra>``: an extra installed without replacing the torch build (fakes only, nothing is
installed)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from bolo_torch import install
from bolo_torch.cli import main


def test_a_cuda_build_of_torch_is_kept() -> None:
    """Windows with CUDA torch: `pip install "bolo-torch[pyannote]"` replaced it with PyPI's CPU torch (the newest
    torchaudio needs a newer torch). The command pins the installed build, holds torchaudio and torchvision to its
    release and adds its index."""
    cmd = install.install_command(["pyannote"], "2.9.0+cu128", "c.txt", python="py")
    assert cmd == [
        "py", "-m", "pip", "install", "bolo-torch[pyannote]", "torch==2.9.0+cu128", "-c", "c.txt",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ]  # fmt: skip
    assert install.constraints("2.9.0+cu128") == ["torch==2.9.0+cu128", "torchaudio<2.10", "torchvision<0.25"]
    assert install.constraints("2.9.0+cu128", "2.9.0+cu128", None) == [
        "torch==2.9.0+cu128",
        "torchaudio==2.9.0+cu128",
        "torchvision<0.25",
    ]
    # torch 2.15: torchaudio's last release (2.11) is allowed
    assert install.constraints("2.15.0+cpu")[1:] == ["torchaudio<2.16", "torchvision<0.31"]
    assert install.build_index("2.8.0+rocm6.4") == "https://download.pytorch.org/whl/rocm6.4"
    assert install.build_index("2.15.0+cpu") == "https://download.pytorch.org/whl/cpu"
    assert install.build_index("2.9.0+xpu") == "https://download.pytorch.org/whl/xpu"
    # a PyPI build (Linux's CUDA wheel has no local version): pinned, no other index
    assert install.install_command(["pyannote", "sepformer"], "2.9.0", "c.txt", python="py")[4:] == [
        "bolo-torch[pyannote,sepformer]",
        "torch==2.9.0",
        "-c",
        "c.txt",
    ]
    # an unknown local part is no PyTorch index; no torch yet: pip chooses
    assert install.build_index("2.9.0+mybuild") is None
    assert install.install_command(["pyannote"], None, "c.txt", python="py") == [
        "py", "-m", "pip", "install", "bolo-torch[pyannote]",
    ]  # fmt: skip


def test_the_cli_prints_the_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(install, "installed", lambda name: "2.9.0+cu128" if name == "torch" else None)
    monkeypatch.setattr(install, "extras", lambda: ["pyannote", "sepformer"])
    ran: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: ran.append(cmd))
    assert main(["install", "pyannote", "--dry-run"]) == 0
    out = capsys.readouterr()
    assert '"bolo-torch[pyannote]" torch==2.9.0+cu128 -c ' in out.out
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in out.out
    assert "keeping torch 2.9.0+cu128 (torch==2.9.0+cu128, torchaudio<2.10, torchvision<0.25)" in out.err
    assert not ran
    assert main(["install", "nosuch"]) == 2 and "no extra nosuch" in capsys.readouterr().err
    assert sys.executable in install.install_command(["pyannote"], None)
