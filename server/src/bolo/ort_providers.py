"""ONNX Runtime execution providers for bolo's ``device`` setting.

The engines that run ONNX graphs themselves (Nemotron diarization, DiCoW, the SNAC decoder of the
Orpheus-family voices, the end-of-turn model and in-process speech-to-speech) ask here which
providers to use. A device this ONNX Runtime build cannot serve falls back to the CPU with one
warning; the CPU is always last in the list, so operators a provider lacks still run.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

CPU = "CPUExecutionProvider"

#: ``device`` values (before an optional ``:index``) and the provider each one asks for
PROVIDER_BY_DEVICE = {
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "mps": "CoreMLExecutionProvider",
    "metal": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}

#: tried in this order for ``device = "auto"``
AUTO_ORDER = ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")

_WITH_DEVICE_ID = {
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
}
_warned: set[str] = set()


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return [CPU]
    return list(ort.get_available_providers())


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message)


def default_dml_index() -> int | None:
    """DirectML's ``device_id`` for ``dml`` with no number: Windows' high-performance adapter (bolo.winhw)."""
    from bolo import winhw

    return winhw.default_dml_index()


def _dml(index: int | None) -> Any:
    return ("DmlExecutionProvider", {"device_id": index}) if index is not None else "DmlExecutionProvider"


def providers_for(device: str | None, available: list[str] | None = None) -> list[Any]:
    """The ``providers`` argument for ``onnxruntime.InferenceSession`` on ``device``.

    ``cpu`` (the default) gives the CPU only; ``cuda``, ``cuda:1``, ``rocm``, ``dml``, ``coreml``
    (also ``mps`` and ``metal``) or ``openvino`` give that provider first and the CPU after it;
    ``auto`` takes the first accelerator this build has. ``dml`` with no number runs on Windows'
    high-performance adapter (the discrete GPU of a hybrid laptop), not on DXGI's first adapter, which
    is DirectML's own default and on such a laptop the integrated GPU; ``dml:N`` pins adapter N.
    """
    kind = (device or "cpu").strip().lower()
    name, _, index = kind.partition(":")
    have = available if available is not None else available_providers()
    if name in ("", "cpu"):
        return [CPU]
    if name == "auto":
        chosen = next((p for p in AUTO_ORDER if p in have), None)
        if chosen == "DmlExecutionProvider":
            return [_dml(default_dml_index()), CPU]
        return [chosen, CPU] if chosen else [CPU]
    wanted = PROVIDER_BY_DEVICE.get(name)
    if wanted is None:
        _warn_once(kind, f"device {device!r} is not known to ONNX Runtime engines; they run on the CPU")
        return [CPU]
    if wanted not in have:
        hint = "install onnxruntime-gpu in place of onnxruntime" if wanted == "CUDAExecutionProvider" else ""
        _warn_once(
            kind,
            f"device {device!r} asks for {wanted}, which this ONNX Runtime build does not have "
            f"(it has {', '.join(have)}); the ONNX Runtime engines run on the CPU" + (f": {hint}" if hint else ""),
        )
        return [CPU]
    if index.isdigit() and wanted in _WITH_DEVICE_ID:
        return [(wanted, {"device_id": int(index)}), CPU]
    if wanted == "DmlExecutionProvider":
        return [_dml(default_dml_index()), CPU]
    return [wanted, CPU]


def sherpa_provider(device: str | None) -> str:
    """sherpa-onnx's ``provider`` string for ``device`` (bolo uses ``cpu``, ``cuda`` and ``coreml``).

    sherpa-onnx knows a ``directml`` provider only when it was built with DirectML, and its wheels on PyPI are not
    (asked for it, they fall back to the CPU saying "DirectML is for Windows only"), so ``device = "dml"`` runs
    sherpa-onnx engines on the CPU, with one notice."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    if name in ("cuda", "gpu"):
        return "cuda"
    if name in ("coreml", "mps", "metal"):
        return "coreml"
    if name in ("dml", "directml"):
        _warn_once(
            "sherpa-dml",
            "device 'dml': sherpa-onnx's wheels have no DirectML provider, so its engines (most speech recognition "
            "and voices) run on the CPU; DirectML serves the engines that use ONNX Runtime directly",
        )
    return "cpu"


#: the PyPI distributions that each install the same ``onnxruntime`` module
ORT_PACKAGES = (
    "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-qnn",
    "onnxruntime-rocm", "onnxruntime-silicon",
)  # fmt: skip

DIRECTML_RECIPE = (
    "pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml, then pip install onnxruntime-directml "
    "(after bolo and its extras: installing faster-whisper again brings plain onnxruntime back)"
)


def installed_ort_packages(versions: dict[str, str | None] | None = None) -> dict[str, str]:
    """The ONNX Runtime distributions installed here, with their versions."""
    if versions is None:
        from importlib import metadata

        versions = {}
        for name in ORT_PACKAGES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    return {name: v for name, v in versions.items() if v}


def ort_problems(device: str | None, packages: dict[str, str], providers: list[str] | None) -> list[str]:
    """What ``bolo doctor`` warns about: several ONNX Runtime packages sharing the one module, and a device whose
    provider this ONNX Runtime lacks (``dml`` without onnxruntime-directml, ``cuda`` without onnxruntime-gpu)."""
    out: list[str] = []
    if len(packages) > 1:
        listed = ", ".join(f"{name} {version}" for name, version in sorted(packages.items()))
        out.append(
            f"several ONNX Runtime packages are installed ({listed}); they all install the same `onnxruntime` "
            "module and overwrite each other's files: keep one (uninstall all of them, then install the one you want)"
        )
    name = (device or "cpu").strip().lower().partition(":")[0]
    wanted = PROVIDER_BY_DEVICE.get(name)
    if wanted and providers is not None and wanted not in providers and packages:
        if wanted == "DmlExecutionProvider":
            out.append(f"device {device!r} needs DirectML in ONNX Runtime, which this one lacks: {DIRECTML_RECIPE}")
        elif wanted == "CUDAExecutionProvider":
            out.append(
                f"device {device!r}: this ONNX Runtime has no CUDA provider; pip uninstall -y onnxruntime, then "
                "pip install onnxruntime-gpu (faster-whisper and sherpa-onnx bring their own CUDA support)"
            )
        else:
            out.append(f"device {device!r}: this ONNX Runtime has no {wanted}")
    return out
