"""Model families served by the plugin. Each module imports its libraries only when a model is loaded."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, Voice


def voices_from_options(options: dict[str, Any]) -> list[dict[str, Any]]:
    raw = options.get("voices") or []
    return [dict(v) for v in raw if isinstance(v, dict) and v.get("id")]


def to_voices(entries: Iterable[dict[str, Any]]) -> tuple[Voice, ...]:
    return tuple(
        Voice(str(v["id"]), str(v.get("name", v["id"])), str(v.get("language", "")), str(v.get("gender", "")), i)
        for i, v in enumerate(entries)
    )


def peak_safe(audio: Any) -> FloatArray:
    """float32 mono, scaled down only if it would clip."""
    out = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out
