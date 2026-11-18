"""The voxtral-llamacpp kind (Voxtral speech-to-text on llama-server), with a fake llama-server (no downloads)."""

from __future__ import annotations

import base64
import io
import json
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.catalog import Catalog, ModelSpec
from bolo.config import Settings
from bolo.engines import voxtral_llamacpp as vox
from bolo.engines.base import LoadOptions, STTRequest
from bolo.engines.tts_server import TTSServerEngine
from bolo.errors import EngineUnavailable, InvalidRequest
from conftest import LiveServer, wav_file
from fake_voxtral_server import MARKER, FakeVoxtralServer
from fakebin import fake_program

HERE = Path(__file__).parent


@pytest.fixture
def server() -> Iterator[FakeVoxtralServer]:
    s = FakeVoxtralServer().start()
    yield s
    s.stop()


def model_for(server: FakeVoxtralServer, **options: Any) -> vox.VoxtralLlamaCpp:
    model = vox.VoxtralLlamaCpp("voxtral-test", {"max_chunk_s": 30.0, **options}, base_url=server.url)
    model.connect()
    return model


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_prompts_follow_the_transcription_and_chat_formats() -> None:
    assert vox.build_prompt("<m>", "transcribe", "fr") == "[INST]<m>[/INST]lang:fr[TRANSCRIBE]"
    assert vox.build_prompt("<m>", "transcribe", None) == "[INST]<m>[/INST][TRANSCRIBE]"
    translate = vox.build_prompt("<m>", "translate", "de")
    assert translate.startswith("[INST]<m>Translate this audio into English.") and translate.endswith("[/INST]")
    assert "[TRANSCRIBE]" not in translate and "<s>" not in translate  # the server adds BOS itself
    into = vox.build_prompt("<m>", "translate", "en", "fr")
    assert "Translate this audio into French. Reply with the French translation only" in into
    assert vox.token_budget(30.0, 12.0, 4096) == 392 and vox.token_budget(0.1, 12.0, 4096) == 33
    assert vox.token_budget(3600.0, 12.0, 4096) == 4096


def test_audio_goes_as_a_16khz_pcm16_wav() -> None:
    data = vox.wav_bytes(np.array([0.0, 0.5, -1.5], dtype=np.float32))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 3)
        assert np.frombuffer(w.readframes(3), "<i2").tolist() == [0, 16383, -32767]


def test_transcription_against_the_fake_server(server: FakeVoxtralServer) -> None:
    model = model_for(server)
    assert model.marker == MARKER and model.supports_translate and not model.supports_word_timestamps
    results = model.transcribe(
        [STTRequest(tone(2.0), language="fr-FR"), STTRequest(tone(1.0)), STTRequest(tone(0.01), language="en")]
    )
    assert [r.text for r in results] == ["heard 2.00 s lang fr", "heard 1.00 s", ""]
    assert results[0].language == "fr-FR" and results[1].language is None and not results[0].segments
    assert len(server.requests) == 2  # nothing is sent for a piece shorter than 50 ms
    body = server.requests[0]["body"]
    assert body["temperature"] == 0.0 and body["top_k"] == 1 and body["n_predict"] == 56
    assert body["cache_prompt"] is False and body["stream"] is False and body["stop"] == ["\n\n"]
    sent = base64.b64decode(body["prompt"]["multimodal_data"][0])
    with wave.open(io.BytesIO(sent)) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 32000

    translated = model.transcribe([STTRequest(tone(1.5), language="de", task="translate", temperature=0.4)])[0]
    assert translated.text == "translated 1.50 s" and translated.language == "en"
    body = server.requests[-1]["body"]
    assert body["temperature"] == 0.4 and "top_k" not in body
    model.close()


def test_options_and_errors(server: FakeVoxtralServer) -> None:
    model = model_for(server, translate=False, tokens_per_s=2.0, max_new_tokens=40, request={"seed": 7}, stop=[])
    with pytest.raises(InvalidRequest, match="cannot translate"):
        model.transcribe([STTRequest(tone(1.0), task="translate")])
    model.transcribe([STTRequest(tone(10.0))])
    assert server.requests[-1]["body"]["n_predict"] == 40 and server.requests[-1]["body"]["seed"] == 7
    assert "stop" not in server.requests[-1]["body"]
    model.marker = "<wrong>"  # a marker the server does not know: refused with 400
    with pytest.raises(InvalidRequest, match="refused the request"):
        model.transcribe([STTRequest(tone(1.0))])
    model.close()


def test_a_server_without_the_audio_encoder_is_refused() -> None:
    server = FakeVoxtralServer(audio=False).start()
    try:
        with pytest.raises(EngineUnavailable, match="without an audio encoder"):
            model_for(server)
    finally:
        server.stop()


def test_unreachable_and_credentials() -> None:
    server = FakeVoxtralServer(api_key="secret").start()
    try:
        model = vox.VoxtralLlamaCpp("v", {}, base_url=server.url, api_key="wrong")
        with pytest.raises(EngineUnavailable, match="credentials"):
            model._post({"prompt": {}})
        model.close()
        model = vox.VoxtralLlamaCpp("v", {}, base_url=server.url, api_key="secret")
        model.connect()
        assert model.transcribe([STTRequest(tone(1.0))])[0].text == "heard 1.00 s"
        model.close()
    finally:
        server.stop()
    model = vox.VoxtralLlamaCpp("v", {}, base_url=server.url, timeout_s=2.0)
    with pytest.raises(EngineUnavailable, match="not reachable"):
        model.connect()


def entry(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "task": "stt",
        "engine": "tts-server",
        "kind": "voxtral-llamacpp",
        "family": "Voxtral",
        "languages": ["en", "fr"],
        "license": "Apache-2.0",
        "source": {"repo": "example/voxtral"},
        "options": {
            "runtime": "llama-server",
            "model_file": "m.gguf",
            "mmproj_file": "mm.gguf",
            "context_size": 4096,
            "max_chunk_s": 5.0,
            "queue_timeout_s": 0,
        },
    }


def test_transcription_api_through_an_operator_run_server(settings: Settings, server: FakeVoxtralServer) -> None:
    settings.tts_servers = [f"voxtral-remote={server.url}"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(entry("voxtral-remote")))
    srv.start()
    try:
        info = httpx.get(f"{srv.url}/api/v1/models/voxtral-remote").json()
        assert info["installed"] is True and info["available"] is True
        # the pieces of a long recording may wait as long as the model needs (catalog option queue_timeout_s: 0)
        manager, catalog = srv.core.manager, srv.core.catalog
        assert manager.queue_timeout(catalog.resolve("voxtral-remote")) == 0.0  # type: ignore[arg-type]
        assert manager.queue_timeout(catalog.resolve("fake-stt")) == settings.queue_timeout_s  # type: ignore[arg-type]
        r = httpx.post(
            f"{srv.url}/v1/audio/transcriptions",
            data={"model": "voxtral-remote", "language": "fr", "response_format": "verbose_json"},
            files={"file": wav_file(2.0)},
            timeout=30,
        )
        assert r.status_code == 200, r.text
        doc = r.json()
        assert doc["text"] == "heard 2.00 s lang fr" and len(doc["segments"]) == 1
        assert doc["segments"][0]["start"] == 0.0 and doc["segments"][0]["end"] == pytest.approx(2.0, abs=0.01)
        # long audio: cut at the pauses into pieces of at most max_chunk_s (5 s), one request each
        before = len(server.requests)
        r = httpx.post(
            f"{srv.url}/v1/audio/transcriptions",
            data={"model": "voxtral-remote", "response_format": "verbose_json"},
            files={"file": wav_file(12.0, gap_at=(3.5, 7.5))},
            timeout=30,
        )
        assert r.status_code == 200, r.text
        segments = r.json()["segments"]
        sent = server.requests[before:]
        assert len(sent) >= 3 and len(segments) == len(sent)
        assert all(s["text"].startswith("heard ") for s in segments)
        assert all(s["end"] - s["start"] <= 5.0 + 0.01 for s in segments)
        assert [s["start"] for s in segments] == sorted(s["start"] for s in segments)
        r = httpx.post(
            f"{srv.url}/v1/audio/translations",
            data={"model": "voxtral-remote"},
            files={"file": wav_file(1.0)},
            timeout=30,
        )
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "translated 1.00 s"
    finally:
        srv.stop()


def fake_binary(tmp_path: Path) -> Path:
    return fake_program(tmp_path / "bin" / "llama-server", f"import sys\nsys.path.insert(0, {str(HERE)!r})\n"
                        "from fake_voxtral_server import main\nmain(sys.argv[1:])\n")  # fmt: skip


def test_managed_llama_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv_log = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(argv_log))
    settings = Settings(data_dir=tmp_path / "data", llama_server_bin=str(fake_binary(tmp_path)))
    engine = TTSServerEngine()
    engine.configure(settings)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_bytes(b"GGUF")
    (model_dir / "mm.gguf").write_bytes(b"GGUF")
    spec = {**entry("managed")["options"], "model_id": "managed", "languages": ["en", "fr"]}
    model = engine.load("voxtral-llamacpp", "stt", model_dir, spec, LoadOptions(threads=2))
    assert isinstance(model, vox.VoxtralLlamaCpp) and model.process is not None
    assert model.languages == ("en", "fr") and model.max_chunk_s == 5.0
    proc = model.process
    try:
        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        assert argv[argv.index("-m") + 1] == str(model_dir / "m.gguf")
        assert argv[argv.index("-mm") + 1] == str(model_dir / "mm.gguf")
        assert argv[argv.index("-c") + 1] == "4096" and "--api-key" in argv
        assert model.transcribe([STTRequest(tone(1.0), language="en")])[0].text == "heard 1.00 s lang en"
        # the server dies (audio starting at exactly 0.5): engine_unavailable, then a restart after the back-off
        crash = np.full(16000, 0.5, dtype=np.float32)
        with pytest.raises(EngineUnavailable):
            model.transcribe([STTRequest(crash)])
        time.sleep(0.2)
        with pytest.raises(EngineUnavailable, match="restarting in"):
            model.transcribe([STTRequest(tone(1.0))])
        assert proc.crashes == 1
        time.sleep(1.1)
        assert model.transcribe([STTRequest(tone(1.0))])[0].text == "heard 1.00 s" and proc.restarts == 1
    finally:
        model.close()
    assert not proc.running

    with pytest.raises(EngineUnavailable, match="speech-to-text"):
        engine.load("voxtral-llamacpp", "tts", model_dir, spec, LoadOptions())
    with pytest.raises(EngineUnavailable, match="runs on llama-server"):
        engine.load("voxtral-llamacpp", "stt", model_dir, {**spec, "runtime": "audio.cpp"}, LoadOptions())
    with pytest.raises(EngineUnavailable, match="mmproj_file"):
        engine.load("voxtral-llamacpp", "stt", model_dir, {**spec, "mmproj_file": None}, LoadOptions())


def test_catalog_entries() -> None:
    specs = {s.id: s for s in Catalog.load(None).all() if s.kind == "voxtral-llamacpp"}
    assert set(specs) == {"voxtral-mini-3b-llamacpp", "voxtral-small-24b-llamacpp"}
    for s in specs.values():
        assert s.task == "stt" and s.engine == "tts-server" and s.requires == ["llama-server"]
        assert s.license == "Apache-2.0" and s.commercial_use and not s.gated and s.sample_rate == 16000
        assert "translate" in s.features and "timestamps" not in s.features
        assert set(s.languages) == {"en", "es", "fr", "pt", "hi", "de", "nl", "it"}
        assert s.options["model_file"] in s.source.include and s.options["context_size"] == 4096
        assert s.options["queue_timeout_s"] == 0 and s.options["max_chunk_s"] == 15.0
        assert len(s.source.revision) == 40
        files = [*s.source.include, *(u["path"] for u in s.source.urls)]
        assert s.options["mmproj_file"] in files
        assert all(len(u["sha256"]) == 64 and u["size"] > 0 for u in s.source.urls)
    mini = specs["voxtral-mini-3b-llamacpp"]
    assert (
        mini.size_bytes < 3_200_000_000
        and mini.source.repo == "ggml-org/Voxtral-Mini-3B-2507-GGUF"
        and "word timestamps are unavailable" in mini.description
    )
    assert specs["voxtral-small-24b-llamacpp"].options["hardware"]["gpu"] == "required"
    assert "voxtral-llamacpp" in TTSServerEngine.kinds
