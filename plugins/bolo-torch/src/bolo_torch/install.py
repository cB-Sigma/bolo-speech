"""``bolo-torch install <extra>``: install an extra without replacing the torch build already in the environment.

A plain ``pip install "bolo-torch[pyannote]"`` lets pip pick the newest torchaudio, which needs a newer torch than
the one installed; pip then upgrades torch from PyPI, whose Windows (and macOS) wheels are CPU only, so a CUDA build
of torch is replaced with a CPU one and the models run on the CPU. This command installs the extra with the
installed torch pinned exactly (``torch==2.9.0+cu128``) and torchaudio and torchvision held to the releases that go with
it (the ones installed, else ``torchaudio<2.10`` and ``torchvision<0.25`` for torch 2.9) in a constraints file, so they are
only installed when the extra needs them; and, for a build from PyTorch's own index (a local version such as
``+cu128``, ``+rocm6.4``, ``+xpu`` or ``+cpu``), with that index as an extra index, so they come in the same build.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys

#: PyTorch's package index; a build's local version names its folder (``cu128``, ``rocm6.4``, ``xpu``, ``cpu``)
PYTORCH_INDEX = "https://download.pytorch.org/whl/"
_BUILD = re.compile(r"^(cu\d+|rocm\d+(\.\d+)*|xpu|cpu)$")


def installed_torch() -> str | None:
    """The version of the torch installed in this environment (``2.9.0+cu128``), or None."""
    return installed("torch")


def build_index(torch_version: str) -> str | None:
    """The PyTorch index folder the build ``torch_version`` comes from, or None (a PyPI build has no local part)."""
    _, _, local = torch_version.partition("+")
    return PYTORCH_INDEX + local if _BUILD.match(local) else None


def extras() -> list[str]:
    """The extras this bolo-torch declares."""
    try:
        return sorted(importlib.metadata.metadata("bolo-torch").get_all("Provides-Extra") or [])
    except importlib.metadata.PackageNotFoundError:
        return []


def installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def constraints(torch_version: str, torchaudio: str | None = None, torchvision: str | None = None) -> list[str]:
    """pip constraints that keep ``torch_version`` and torchaudio and torchvision no newer than its release: the
    versions installed, else below the next release of that torch (torchaudio is numbered as torch, and its last
    release, 2.11, needs torch 2.11 or later; torchvision is 0.(minor + 15) for torch 2). The newest torchaudio,
    left free, needs a newer torch than a CUDA build installed earlier, and pip then replaces that torch with
    PyPI's (CPU only on Windows)."""
    lines = [f"torch=={torch_version}"]
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if torchaudio:
        lines.append(f"torchaudio=={torchaudio}")
    elif match:
        lines.append(f"torchaudio<{match[1]}.{int(match[2]) + 1}")
    if torchvision:
        lines.append(f"torchvision=={torchvision}")
    elif match and match[1] == "2":
        lines.append(f"torchvision<0.{int(match[2]) + 16}")
    return lines


def install_command(
    names: list[str],
    torch_version: str | None,
    constraints_file: str | None = None,
    package: str = "bolo-torch",
    python: str = sys.executable,
) -> list[str]:
    """The pip command that installs ``package[names]`` and keeps ``torch_version`` (None: no torch yet, so pip
    chooses as it would), with :func:`constraints` written to ``constraints_file``."""
    cmd = [python, "-m", "pip", "install", f"{package}[{','.join(names)}]"]
    if torch_version:
        cmd.append(f"torch=={torch_version}")
        if constraints_file:
            cmd += ["-c", constraints_file]
        index = build_index(torch_version)
        if index:
            cmd += ["--extra-index-url", index]
    return cmd


def restore_command(before: dict[str, str | None], after: str | None, python: str = sys.executable) -> list[str] | None:
    """When an install replaced a torch from PyTorch's index (``before``: the torch, torchaudio and torchvision
    installed then) with another build anyway, the pip command that puts that build back; None when it is still
    there."""
    torch = before.get("torch")
    if not torch or after == torch:
        return None
    index = build_index(torch)
    if index is None or index.endswith("/cpu"):
        return None
    reqs = [f"{name}=={version}" for name, version in before.items() if version]
    return [python, "-m", "pip", "install", *reqs, "--index-url", index, "--extra-index-url", "https://pypi.org/simple"]
