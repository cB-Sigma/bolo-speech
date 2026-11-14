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


def providers_for(device: str | None, available: list[str] | None = None) -> list[Any]:
    """The ``providers`` argument for ``onnxruntime.InferenceSession`` on ``device``.

    ``cpu`` (the default) gives the CPU only; ``cuda``, ``cuda:1``, ``rocm``, ``dml``, ``coreml``
    (also ``mps`` and ``metal``) or ``openvino`` give that provider first and the CPU after it;
    ``auto`` takes the first accelerator this build has.
    """
    kind = (device or "cpu").strip().lower()
    name, _, index = kind.partition(":")
    have = available if available is not None else available_providers()
    if name in ("", "cpu"):
        return [CPU]
    if name == "auto":
        chosen = next((p for p in AUTO_ORDER if p in have), None)
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
    return [wanted, CPU]


def sherpa_provider(device: str | None) -> str:
    """sherpa-onnx's ``provider`` string for ``device`` (it knows ``cpu``, ``cuda`` and ``coreml``)."""
    name = (device or "cpu").strip().lower().partition(":")[0]
    if name in ("cuda", "gpu"):
        return "cuda"
    if name in ("coreml", "mps", "metal"):
        return "coreml"
    return "cpu"
