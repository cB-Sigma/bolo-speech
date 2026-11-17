"""The plugin served as a real subprocess and driven by bolo's own plugin host (fake model backends)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from bolo.catalog import Catalog
from bolo.engines.base import (
    ConversationRequest,
    ConversationTurn,
    DenoiseModel,
    DiarizationModel,
    LoadOptions,
    SeparationModel,
    SpeechToSpeechModel,
    TTSModel,
)
from bolo.errors import GatedModel, InvalidRequest
from bolo.plugin.host import PluginConfig, PluginEngine

from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED, TorchEngine
from fakes import tone

FAKE_PLUGIN = [sys.executable, str(Path(__file__).parent / "fake_plugin.py")]


@pytest.fixture(scope="module")
def engine() -> Iterator[PluginEngine]:
    eng = PluginEngine(PluginConfig("torch", FAKE_PLUGIN, idle_s=-1))
    yield eng
    eng.shutdown()


def test_hello_and_tts_through_the_host(engine: PluginEngine, tmp_path: Path) -> None:
    voices = [
        {"id": "divya", "name": "Divya", "language": "hi-IN"},
        {"id": "rohit", "name": "Rohit", "language": "hi-IN"},
    ]
    model = engine.load("parler-tts", "tts", tmp_path, {"voices": voices, "model_id": "x"}, LoadOptions(threads=1))
    assert isinstance(model, TTSModel)
    assert engine.process.hello["name"] == "torch" and engine.process.hello["protocol"] == 1
    assert engine.process.hello["kinds"] == [
        "parler-tts", "sepformer", "gated", "dicow", "pyannote-diarization", "qwen-omni", "demucs-denoiser",
    ]  # fmt: skip
    assert model.sample_rate == 44100 and [v.id for v in model.voices] == ["divya", "rohit"]
    chunks = list(model.synthesize("पहला वाक्य। दूसरा वाक्य।", "rohit"))
    assert len(chunks) == 2 and all(c.dtype == np.float32 for c in chunks)
    model.close()


def test_placement_goes_to_the_plugin_and_the_split_comes_back(engine: PluginEngine, tmp_path: Path) -> None:
    # CI has CPU torch only: the device stays cpu; the placement still travels to the loader and back
    opts = LoadOptions(threads=1, device="cpu", max_memory={"0": 4000, "cpu": 9000})
    model = engine.load("parler-tts", "tts", tmp_path, {"model_id": "split"}, opts)
    report = getattr(model, "placement_report", None)
    assert report == {"gpu_bytes": 4000, "cpu_bytes": 1000, "device_map": {"model": "0"}}
    model.close()
    whole = engine.load("parler-tts", "tts", tmp_path, {"model_id": "whole"}, LoadOptions(threads=1))
    assert getattr(whole, "placement_report", None) is None  # no split asked for: none reported
    whole.close()


def test_blind_separation_through_the_host(engine: PluginEngine, tmp_path: Path) -> None:
    model = engine.load("sepformer", "separation", tmp_path, {"model_sample_rate": 8000}, LoadOptions(threads=1))
    assert isinstance(model, SeparationModel) and model.mode == "blind" and model.sample_rate == 16000
    mix = tone(0.8)
    tracks = model.separate(mix)
    assert len(tracks) == 2 and all(len(t) == len(mix) for t in tracks)
    model.close()


def test_diarization_through_the_host(engine: PluginEngine, tmp_path: Path) -> None:
    tracks = [[1.5, 2.5, "SPEAKER_00"], [0.2, 1.4, "SPEAKER_01"], [2.4, 3.0, "SPEAKER_01"]]
    opts = LoadOptions(threads=1, device="cpu")
    model = engine.load("pyannote-diarization", "diarization", tmp_path, {"tracks": tracks}, opts)
    assert isinstance(model, DiarizationModel) and model.task == "diarization"
    assert model.sample_rate == 16000 and model.consolidate_default is False  # pyannote's speaker count stands
    result = model.diarize_detailed(tone(3.0), num_speakers=2, threshold=0.5)
    # bolo's turns: sorted by start, speakers numbered in order of first appearance
    assert [(t.start, t.end, t.speaker) for t in result.turns] == [(0.2, 1.4, 0), (1.5, 2.5, 1), (2.4, 3.0, 0)]
    assert result.activity is None
    vec = model.embed(tone(1.0))
    assert vec.dtype == np.float32 and vec.shape == (4,) and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    model.close()


def test_denoise_through_the_host(engine: PluginEngine, tmp_path: Path) -> None:
    model = engine.load("demucs-denoiser", "denoise", tmp_path, {}, LoadOptions(threads=1))
    assert isinstance(model, DenoiseModel) and model.task == "denoise" and model.sample_rate == 16000
    x = tone(0.7)
    y = model.denoise(x)
    assert len(y) == len(x) and np.allclose(y, 0.5 * x, atol=1e-6)
    model.close()


def test_speech_to_speech_through_the_host(engine: PluginEngine, tmp_path: Path) -> None:
    opts = LoadOptions(threads=1, device="cpu")
    model = engine.load("qwen-omni", "speech-to-speech", tmp_path, {"family": "qwen3-omni"}, opts)
    assert isinstance(model, SpeechToSpeechModel) and model.sample_rate == 24000 and not model.supports_tools
    assert [v.id for v in model.voices] == ["Ethan", "Chelsie", "Aiden"]
    turns = [ConversationTurn("user", audio=tone(1.0)), ConversationTurn("assistant", text="Hi."),
             ConversationTurn("user", text="And?")]  # fmt: skip
    chunks = list(model.respond(ConversationRequest(turns, voice="aiden")))
    assert len(chunks) == 1 and chunks[0].text == "w7 w8 w9"
    assert chunks[0].audio is not None and len(chunks[0].audio) == 12000 and chunks[0].audio.dtype == np.float32
    with pytest.raises(InvalidRequest, match="unknown voice"):
        list(model.respond(ConversationRequest(turns, voice="nobody")))
    model.close()


def test_errors_keep_their_meaning(engine: PluginEngine, tmp_path: Path) -> None:
    with pytest.raises(GatedModel, match="accept its conditions"):
        engine.load("gated", "tts", tmp_path, {}, LoadOptions())
    with pytest.raises(InvalidRequest, match="cannot load SraVaani"):
        engine.load("sravaani", "stt", tmp_path, {}, LoadOptions())
    with pytest.raises(InvalidRequest, match="full-duplex"):
        engine.load("moshi", "speech-to-speech", tmp_path, {}, LoadOptions())
    with pytest.raises(InvalidRequest, match="has no model kind"):
        engine.load("nope", "tts", tmp_path, {}, LoadOptions())
    model = engine.load("parler-tts", "tts", tmp_path, {}, LoadOptions())
    with pytest.raises(Exception, match="unknown voice"):
        list(model.synthesize("Hello.", "nobody"))
    assert engine.process.running and engine.process.crashes == 0


def test_catalog_entries_use_kinds_the_plugin_knows() -> None:
    entries = [s for s in Catalog.load().all() if s.engine == "plugin:torch"]
    assert entries
    for spec in entries:
        assert spec.kind in LOADERS or spec.kind in NOT_IMPLEMENTED, spec.id
        assert "bolo-torch" in str(spec.options.get("plugin_install", "")), spec.id
    assert TorchEngine().kinds == tuple(LOADERS)
    seps = {s.id for s in entries if s.task == "separation"}
    assert {"sepformer-whamr16k", "sepformer-whamr", "pixit-ami"} <= seps
    diarizers = {s.id: s for s in entries if s.task == "diarization" and s.runnable}
    assert set(diarizers) == {"pyannote-speaker-diarization-3.1", "pyannote-speaker-diarization-community-1"}
    for spec in [*diarizers.values(), *(s for s in entries if s.id == "pixit-ami")]:
        # pinned pipeline and sub-models, with the hashes the plugin checks again before pyannote reads anything
        assert spec.gated and len(spec.source.revision or "") == 40, spec.id
        assert all(len(h) == 64 for h in spec.options["sha256"].values()) and "config.yaml" in spec.options["sha256"]
        for url in spec.source.urls:
            assert "/resolve/main/" not in url["url"] and url["sha256"] == spec.options["sha256"][url["path"]]
