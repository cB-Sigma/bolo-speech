"""Qwen3-ASR (and Rumble-2-Flash, built on it) and Cohere Transcribe through sherpa-onnx, with a stand-in for the
sherpa-onnx module: the files each kind is built from, how the language reaches the model and how its answer is read."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bolo.custom import detect_kind
from bolo.engines.base import LoadOptions, STTRequest
from bolo.engines.sherpa import qwen3_asr_language_name, qwen3_asr_split


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text
        self.lang = ""
        self.tokens: list[str] = []
        self.timestamps: list[float] = []


class _Stream:
    def __init__(self, rec: _Recognizer) -> None:
        self.rec = rec
        self.options: dict[str, str] = {}
        self.result = _Result("")

    def set_option(self, key: str, value: str) -> None:
        self.options[key] = value

    def accept_waveform(self, rate: int, samples: Any) -> None:
        pass


class _Recognizer:
    def __init__(self, kind: str, kwargs: dict[str, Any]) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.streams: list[_Stream] = []

    def create_stream(self) -> _Stream:
        s = _Stream(self)
        self.streams.append(s)
        return s

    def decode_streams(self, streams: list[_Stream]) -> None:
        for s in streams:
            if self.kind == "qwen3":
                lang = s.options.get("language")
                s.result = _Result("Guten Morgen." if lang else "language German. Guten Morgen.")
            else:
                s.result = _Result(f"[{s.options.get('language', self.kwargs['language'])}] Hello.")


@pytest.fixture
def fake_sherpa(monkeypatch: pytest.MonkeyPatch) -> list[_Recognizer]:
    built: list[_Recognizer] = []

    def qwen3(**kw: Any) -> _Recognizer:
        built.append(_Recognizer("qwen3", kw))
        return built[-1]

    def cohere(**kw: Any) -> _Recognizer:
        built.append(_Recognizer("cohere", kw))
        return built[-1]

    offline = types.SimpleNamespace(from_qwen3_asr=qwen3, from_cohere_transcribe=cohere)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", types.SimpleNamespace(OfflineRecognizer=offline))
    return built


def _files(root: Path, *names: str) -> Path:
    for n in names:
        (root / n).parent.mkdir(parents=True, exist_ok=True)
        (root / n).write_bytes(b"x")
    return root


def _audio() -> np.ndarray:
    return np.zeros(16000, np.float32)


def test_qwen3_asr_language_names_and_prefix() -> None:
    assert qwen3_asr_language_name("de") == "German"
    assert qwen3_asr_language_name("zh-CN") == "Chinese"
    assert qwen3_asr_language_name("tl") == "Filipino"
    assert qwen3_asr_language_name("yue") == "Cantonese"
    assert qwen3_asr_split("language German. Guten Morgen.") == ("de", "Guten Morgen.")
    assert qwen3_asr_split("language English<asr_text>Hello there.") == ("en", "Hello there.")
    assert qwen3_asr_split("language Cantonese<asr_text>你好") == ("yue", "你好")
    assert qwen3_asr_split("language None") == (None, "")
    assert qwen3_asr_split("Hello there.") == (None, "Hello there.")
    # a transcript that happens to begin with the word "language" and no name after it is left alone
    assert qwen3_asr_split("language is hard") == (None, "language is hard")


def test_qwen3_asr_detects_or_takes_the_language(tmp_path: Path, fake_sherpa: list[_Recognizer]) -> None:
    from bolo.engines.sherpa import SherpaOfflineSTT

    path = _files(
        tmp_path, "conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer/vocab.json",
        "tokenizer/merges.txt",
    )  # fmt: skip
    model = SherpaOfflineSTT("qwen3_asr", path, {}, LoadOptions(threads=2))
    (rec,) = fake_sherpa
    assert Path(rec.kwargs["conv_frontend"]).name == "conv_frontend.onnx"
    assert Path(rec.kwargs["encoder"]).name == "encoder.int8.onnx"
    assert Path(rec.kwargs["tokenizer"]).name == "tokenizer"
    assert rec.kwargs["max_new_tokens"] == 200 and rec.kwargs["num_threads"] == 2
    assert model.max_chunk_s == 20.0 and not model.supports_word_timestamps

    auto, fixed = model.transcribe([STTRequest(_audio()), STTRequest(_audio(), language="de")])
    assert (auto.text, auto.language) == ("Guten Morgen.", "de")  # the prefix is read and removed
    assert (fixed.text, fixed.language) == ("Guten Morgen.", "de")
    assert [s.options for s in rec.streams] == [{}, {"language": "German"}]
    assert len(fake_sherpa) == 1  # one recogniser for every language


def test_cohere_transcribe_is_told_the_language_of_each_stream(tmp_path: Path, fake_sherpa: list[_Recognizer]) -> None:
    from bolo.engines.sherpa import SherpaOfflineSTT

    path = _files(tmp_path, "encoder.int8.onnx", "encoder.int8.onnx.data", "decoder.int8.onnx", "tokens.txt")
    model = SherpaOfflineSTT("cohere_transcribe", path, {}, LoadOptions(threads=1))
    assert [r.kwargs["language"] for r in fake_sherpa] == ["en"]  # the default
    assert Path(fake_sherpa[0].kwargs["encoder"]).name == "encoder.int8.onnx"
    assert fake_sherpa[0].kwargs["use_punct"] is True
    plain, fr = model.transcribe([STTRequest(_audio()), STTRequest(_audio(), language="fr-FR")])
    assert (plain.text, plain.language) == ("[en] Hello.", "en")  # no language given: English
    assert (fr.text, fr.language) == ("[fr] Hello.", "fr")
    assert len(fake_sherpa) == 1  # one recogniser (2.9 GB) for every language


def test_detect_kind_of_the_new_layouts() -> None:
    qwen = ["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer/vocab.json"]
    assert detect_kind("TypeWhisper/Rumble-2-Flash-sherpa-onnx-int8", qwen) == "qwen3_asr"
    cohere = ["encoder.int8.onnx", "encoder.int8.onnx.data", "decoder.int8.onnx", "tokens.txt"]
    assert detect_kind("csukuangfj2/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01", cohere) == (
        "cohere_transcribe"
    )
