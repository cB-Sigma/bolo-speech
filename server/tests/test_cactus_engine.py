"""Whistle on Cactus Compute's runtime (bolo.engines.cactus), with a stand-in for the cactus-compute package."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bolo.custom import detect_kind
from bolo.engines.base import LoadOptions, STTRequest
from bolo.engines.cactus import CactusEngine, _keywords


class _Whistle:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, language: str | None = None, keywords: Any = None, word_timestamps: bool = False):
        self.calls.append({"n": len(audio), "dtype": audio.dtype, "language": language, "keywords": keywords})
        out: dict[str, Any] = {"text": " Guten Morgen. ", "language": language or "de"}
        if word_timestamps:
            out["words"] = [
                {"word": " Guten", "start": 0.1, "end": 0.4, "probability": 0.9},
                {"word": "Morgen.", "start": 0.4, "end": 0.8, "probability": 0.8},
            ]
        return out


@pytest.fixture
def fake_cactus(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {}

    def whistle(weights: str) -> _Whistle:
        state["model"] = _Whistle(weights)
        return state["model"]

    module = types.SimpleNamespace(
        __version__="4.2.6", Whistle=whistle, set_threads=lambda n: state.setdefault("threads", n)
    )
    monkeypatch.setitem(sys.modules, "cactus", module)
    monkeypatch.delenv("CACTUS_TELEMETRY", raising=False)
    return state


def test_whistle_transcribes_with_language_words_and_keywords(tmp_path: Path, fake_cactus: dict[str, Any]) -> None:
    import os

    (tmp_path / "whistle.cact").write_bytes(b"x")
    engine = CactusEngine()
    assert engine.status().available and engine.status().version == "4.2.6"
    assert os.environ["CACTUS_TELEMETRY"] == "0"  # the package's usage counts stay off
    model = engine.load("cactus-whistle", "stt", tmp_path, {"languages": ["en", "de"]}, LoadOptions(threads=3))
    assert fake_cactus["threads"] == 3 and fake_cactus["model"].weights.endswith("whistle.cact")

    audio = np.zeros(16000, np.float64)
    plain, timed, _other = model.transcribe([
        STTRequest(audio),
        STTRequest(audio, language="de-DE", word_timestamps=True, hotwords="Zintrava :3, Bolo"),
        STTRequest(audio, language="ja"),
    ])  # fmt: skip
    assert (plain.text, plain.language, plain.words) == ("Guten Morgen.", "de", [])
    assert [w.w for w in timed.words] == ["Guten", "Morgen."] and timed.words[0].p == 0.9
    assert timed.segments[0].words == timed.words
    calls = fake_cactus["model"].calls
    assert calls[0]["dtype"] == np.float32 and calls[0]["language"] is None
    assert calls[1]["language"] == "de" and calls[1]["keywords"] == ["Zintrava", "Bolo"]
    assert calls[2]["language"] is None  # a language it does not know: it detects instead


def test_engine_without_the_package_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cactus", None)  # import fails
    status = CactusEngine().status()
    assert not status.available and "bolo-speech[cactus]" in (status.reason or "")


def test_keywords_and_kind_detection() -> None:
    assert _keywords(None) is None and _keywords(" , ") is None
    assert _keywords("alpha; beta / gamma\ndelta") == ["alpha", "beta", "gamma", "delta"]
    assert detect_kind("Cactus-Compute/whistle", ["whistle.cact", "config.json", "LICENSE"]) == "cactus-whistle"
