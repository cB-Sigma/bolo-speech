"""Built-in samples for the playground and the benchmark.

Audio samples are short WAV files shipped in this package. They were
synthesised with open models through bolo itself (see ``index.json`` for
the model and voice of each) so they can be redistributed freely. One
sample is not speech at all: ``tone-bursts`` is generated on the fly and
exists so benchmarks and tests run without any file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

from bolo.audio.decode import decode_audio
from bolo.audio.encode import encode_audio
from bolo.engines.base import FloatArray
from bolo.errors import InvalidRequest

TONE_ID = "tone-bursts"
DEFAULT_AUDIO = "en-short"


@lru_cache(maxsize=1)
def index() -> list[dict[str, Any]]:
    root = resources.files("bolo.samples")
    path = root / "index.json"
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    out: list[dict[str, Any]] = []
    for e in entries:
        if e.get("kind") == "audio" and not (root / e["file"]).is_file():
            continue
        out.append(e)
    out.append(
        {
            "id": TONE_ID,
            "kind": "audio",
            "language": "",
            "seconds": 4.0,
            "file": f"{TONE_ID}.wav",
            "text": "",
            "description": "Synthetic tone bursts separated by silence (not speech). For load tests without a model download.",
        }
    )
    return out


def listing() -> list[dict[str, Any]]:
    out = []
    for e in index():
        item = {k: v for k, v in e.items() if k != "file"}
        item["url"] = f"/api/v1/samples/{e['file']}" if e.get("kind") == "audio" else None
        out.append(item)
    return out


def tone_bursts(seconds: float = 4.0, sample_rate: int = 16000) -> FloatArray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    carrier = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    gate = ((t % 2.0) > 0.25) & ((t % 2.0) < 1.45)
    return (carrier * gate).astype(np.float32)


def read_file(name: str) -> bytes:
    if name == f"{TONE_ID}.wav":
        return encode_audio(tone_bursts(), 16000, "wav")
    for e in index():
        if e.get("file") == name:
            return (resources.files("bolo.samples") / name).read_bytes()
    raise InvalidRequest(f"unknown sample file '{name}'")


def get(sample_id: str) -> dict[str, Any]:
    sid = sample_id.removeprefix("builtin:")
    for e in index():
        if e["id"] == sid:
            return e
    raise InvalidRequest(f"unknown sample '{sample_id}'; see GET /api/v1/samples", param="sample")


def load_audio(sample_id: str) -> tuple[FloatArray, int, dict[str, Any]]:
    e = get(sample_id)
    if e.get("kind") != "audio":
        raise InvalidRequest(f"sample '{sample_id}' is not an audio sample", param="sample")
    samples, rate = decode_audio(read_file(e["file"]), filename=e["file"])
    return samples, rate, e


def load_text(sample_id: str) -> str:
    return str(get(sample_id).get("text", ""))
