"""Voice blends: one voice made from the style vectors of several voices of the same model.

Kokoro gives each voice a table of style vectors (one 256-wide row per input length). A blend is the weighted
sum of those tables: ``af_bella+af_sky`` takes equal parts, ``af_bella(0.7)+af_sky(0.3)`` weighs them. Weights
are normalised to sum to 1, so ``af_bella(2)+af_sky`` is two parts Bella to one part Sky; a voice without a
weight counts 1. The text is read in the language of the voice with the largest weight (the first on a tie).

This module only parses and checks a blend and mixes arrays; the engines decide how the mixed style reaches the
model (:mod:`bolo.engines.sherpa`, :mod:`bolo.engines.kokoro_onnx`).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bolo.engines.base import Voice
from bolo.errors import InvalidRequest

#: model kinds whose voices can be blended
BLEND_KINDS = ("kokoro", "kokoro-onnx")
#: the most voices one blend may mix
MAX_PARTS = 8

_PART_RE = re.compile(r"^\s*([^()+\s]+)\s*(?:\(\s*([^()]*?)\s*\))?\s*$")


def model_blends(kind: str, features: Sequence[str] = ()) -> bool:
    """True for a model whose voices can be blended: a Kokoro kind, or a catalog entry with ``voice-blending``."""
    return kind in BLEND_KINDS or "voice-blending" in features


class UnknownVoice(InvalidRequest):
    code = "voice_not_found"


@dataclass(frozen=True)
class Blend:
    """A checked blend: the model's voices and their normalised weights, in the order given."""

    parts: tuple[tuple[Voice, float], ...]

    @property
    def key(self) -> str:
        """The canonical spelling, ``af_bella(0.7)+af_sky(0.3)``: the cache key and what the headers report."""
        return "+".join(f"{v.id}({_fmt(w)})" for v, w in self.parts)

    @property
    def lead(self) -> Voice:
        """The voice with the largest weight (the first on a tie): its language reads the text."""
        best = self.parts[0]
        for part in self.parts[1:]:
            if part[1] > best[1] + 1e-9:
                best = part
        return best[0]

    def to_dict(self) -> dict[str, Any]:
        return {"blend": self.key, "voices": [{"voice": v.id, "weight": round(w, 4)} for v, w in self.parts]}


def _fmt(weight: float) -> str:
    text = f"{weight:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def looks_like_blend(value: str | None) -> bool:
    """True for a voice field written as a blend (``a+b``, ``a(0.7)+b(0.3)`` or a single ``a(1)``)."""
    return bool(value) and ("+" in str(value) or "(" in str(value))


def parse_blend(value: str) -> list[tuple[str, float]]:
    """``"af_bella(0.7)+af_sky(0.3)"`` -> ``[("af_bella", 0.7), ("af_sky", 0.3)]`` (weights as written; a voice
    without a weight gets 1). Raises :class:`InvalidRequest` for a malformed blend."""
    parts: list[tuple[str, float]] = []
    for raw in str(value).split("+"):
        m = _PART_RE.match(raw)
        if not m:
            raise InvalidRequest(
                f"cannot read the voice blend {value!r}: write voices joined by '+', each optionally with a weight "
                "in parentheses, for example af_bella(0.7)+af_sky(0.3)",
                param="voice",
            )
        name, weight_text = m.group(1), m.group(2)
        weight = 1.0
        if weight_text is not None:
            try:
                weight = float(weight_text)
            except ValueError:
                raise InvalidRequest(
                    f"the weight of '{name}' in the voice blend is not a number", param="voice"
                ) from None
            if not np.isfinite(weight) or weight <= 0:
                raise InvalidRequest(f"the weight of '{name}' in the voice blend must be above 0", param="voice")
        parts.append((name, weight))
    if len(parts) > MAX_PARTS:
        raise InvalidRequest(f"a voice blend mixes at most {MAX_PARTS} voices", param="voice")
    return parts


def near_matches(name: str, voices: Sequence[Voice], n: int = 4) -> list[str]:
    """Voice ids that look like ``name`` (closest first)."""
    ids = [v.id for v in voices]
    lowered = {i.lower(): i for i in ids}
    close = difflib.get_close_matches(name.lower(), list(lowered), n=n, cutoff=0.5)
    if not close:
        # "bella" -> af_bella: a match on the part after the prefix
        close = [i for i in lowered if i.split("_", 1)[-1] == name.lower()][:n]
    return [lowered[c] for c in close]


def unknown_voice_message(name: str, voices: Sequence[Voice], model: str) -> str:
    near = near_matches(name, voices)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    listed = ", ".join(v.id for v in voices[:12]) + (", …" if len(voices) > 12 else "")
    return f"voice '{name}' is not available for model '{model}'.{hint} Voices: {listed}"


def resolve_blend(value: str, voices: Sequence[Voice], model: str = "") -> Blend:
    """Check a blend against a model's voices (ids or display names, any case) and normalise its weights.
    The same voice named twice adds its weights up. Raises :class:`UnknownVoice` naming the near matches."""
    parsed = parse_blend(value)
    by_key: dict[str, Voice] = {}
    for v in voices:
        by_key.setdefault(v.id.lower(), v)
        if v.name:
            by_key.setdefault(v.name.lower(), v)
    found: dict[str, tuple[Voice, float]] = {}
    for name, weight in parsed:
        voice = by_key.get(name.lower())
        if voice is None:
            raise UnknownVoice(unknown_voice_message(name, voices, model or "this model"), param="voice")
        prev = found.get(voice.id)
        found[voice.id] = (voice, weight + (prev[1] if prev else 0.0))
    total = sum(w for _v, w in found.values())
    return Blend(tuple((v, w / total) for v, w in found.values()))


def mix_styles(table: np.ndarray, blend: Blend) -> np.ndarray:
    """The weighted sum of the style tables of a blend: ``table[sid]`` is one voice's table."""
    out = np.zeros(table.shape[1:], dtype=np.float64)
    for voice, weight in blend.parts:
        out += weight * table[voice.sid].astype(np.float64)
    return out.astype(np.float32)
