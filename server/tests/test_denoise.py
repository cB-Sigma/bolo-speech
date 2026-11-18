"""Noise suppression before recognition (``denoise``), with the fake denoiser and recogniser."""

from __future__ import annotations

import io
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.denoise import OFF, DenoiseOptions, mix, parse_denoise
from bolo.engines.fake import FakeDenoiser
from bolo.errors import InvalidRequest
from bolo.obs.trace import Trace
from bolo.realtime import RealtimeError, RealtimeSession
from conftest import LiveServer

SR = 16000


def noisy(seconds: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    return (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


def wav(x: np.ndarray, sr: int = SR) -> tuple[str, io.BytesIO, str]:
    return ("noisy.wav", io.BytesIO(encode_audio(x, sr, "wav")), "audio/wav")


def calls(core: Bolo, model: str = "fake-denoise") -> list[tuple[int, int]]:
    entry = core.manager.loaded.get(model)
    if entry is None:
        return []
    denoiser = entry.model
    assert isinstance(denoiser, FakeDenoiser)
    return denoiser.calls


@pytest.fixture
def dn_settings(settings: Settings) -> Settings:
    settings.default_denoise = "fake-denoise"
    return settings


@pytest.fixture
async def dn_core(dn_settings: Settings):  # type: ignore[no-untyped-def]
    c = Bolo(dn_settings)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


@pytest.fixture
def dn_live(dn_settings: Settings):  # type: ignore[no-untyped-def]
    server = LiveServer(dn_settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_parse_denoise() -> None:
    assert parse_denoise(None) is None and parse_denoise("") is None
    assert parse_denoise(True) == DenoiseOptions() and parse_denoise("on") == DenoiseOptions()
    assert parse_denoise(False) == OFF and parse_denoise("off") == OFF and parse_denoise(0) == OFF
    assert parse_denoise("0.5") == DenoiseOptions(strength=0.5) and parse_denoise(0.25).strength == 0.25
    assert parse_denoise("gtcrn") == DenoiseOptions(model="gtcrn")
    assert parse_denoise({"model": "gtcrn", "strength": 0.7}) == DenoiseOptions(model="gtcrn", strength=0.7)
    assert parse_denoise({"enabled": False}) == OFF
    assert parse_denoise(None, strength="0.4") == DenoiseOptions(strength=0.4)
    assert parse_denoise("gtcrn", strength=0.4) == DenoiseOptions(model="gtcrn", strength=0.4)
    assert parse_denoise("off", strength=0.4) == OFF  # off stays off
    for bad in (1.5, -0.1, "2", [1], {"strength": "x"}, {"model": 3}):
        with pytest.raises(InvalidRequest):
            parse_denoise(bad)
    with pytest.raises(InvalidRequest):
        parse_denoise(True, strength="loud")


def test_mix_weights_the_enhanced_audio() -> None:
    a, b = np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
    assert np.allclose(mix(a, b, 1.0), 0.0) and np.allclose(mix(a, b, 0.25), 0.75)


def test_setting_is_checked_at_start(tmp_path: Any) -> None:
    assert load_settings(overrides={"data_dir": tmp_path}, env={}).denoise == "off"
    assert load_settings(overrides={"data_dir": tmp_path, "denoise": "0.5"}, env={}).denoise == "0.5"
    assert load_settings(env={"BOLO_DENOISE": "on", "BOLO_DATA_DIR": str(tmp_path)}).denoise == "on"
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_settings(overrides={"data_dir": tmp_path, "denoise": "3"}, env={})


def test_fake_denoiser_removes_broadband_noise() -> None:
    x = noisy()
    y = FakeDenoiser({}).denoise(x)
    clean = (0.3 * np.sin(2 * np.pi * 300 * np.arange(len(x)) / SR)).astype(np.float32)
    assert len(y) == len(x)
    assert np.mean((y - clean) ** 2) < 0.2 * np.mean((x - clean) ** 2)


async def test_off_by_default(dn_core: Bolo) -> None:
    trace = Trace("stt", "internal", "")
    await dn_core.transcribe(noisy(), SR, trace=trace)
    assert "denoise" not in trace.extra and calls(dn_core) == []


async def test_request_turns_it_on(dn_core: Bolo) -> None:
    trace = Trace("stt", "internal", "")
    await dn_core.transcribe(noisy(), SR, trace=trace, denoise=DenoiseOptions())
    record = trace.extra["denoise"]
    assert record["model"] == "fake-denoise" and record["status"] == "applied" and record["strength"] == 1.0
    assert "denoise" in [p.name for p in trace.phases]
    assert calls(dn_core) == [(2 * SR, SR)]


async def test_runs_at_the_models_own_rate(dn_core: Bolo) -> None:
    """A 48 kHz model gets 48 kHz audio; the recogniser still gets audio at its own rate and length."""
    trace = Trace("stt", "internal", "")
    t = await dn_core.transcribe(noisy(sr=8000), 8000, trace=trace, denoise=DenoiseOptions(model="fake-denoise-48k"))
    assert calls(dn_core, "fake-denoise-48k") == [(2 * 48000, 48000)]
    assert trace.extra["denoise"]["sample_rate"] == 48000 and t.duration == pytest.approx(2.0)


async def test_defaults_pin_catalog_and_setting(dn_core: Bolo) -> None:
    spec = dn_core.resolve("fake-stt", "stt")
    assert dn_core.denoise_plan(spec, None) is None
    dn_core.settings.denoise = "0.5"
    assert dn_core.denoise_plan(spec, None) == DenoiseOptions(model="fake-denoise", strength=0.5)
    dn_core.store.set_model_settings("fake-stt", {"denoise": "off"})  # the pin wins over the setting
    assert dn_core.denoise_plan(spec, None) is None
    dn_core.store.set_model_settings("fake-stt", {"denoise": "fake-denoise-48k"})
    assert dn_core.denoise_plan(spec, None) == DenoiseOptions(model="fake-denoise-48k")
    assert dn_core.denoise_plan(spec, OFF) is None  # the request wins over the pin
    dn_core.store.set_model_settings("fake-stt", {"denoise": "no-such-model"})  # a stale pin never fails requests
    assert dn_core.denoise_plan(spec, None) is None
    with pytest.raises(Exception) as err:
        dn_core.denoise_plan(spec, DenoiseOptions(model="no-such-model"))
    assert getattr(err.value, "param", None) == "denoise"
    with pytest.raises(InvalidRequest):
        dn_core.denoise_plan(spec, DenoiseOptions(model="fake-stt"))  # not a denoise model


async def test_a_failing_denoiser_is_skipped(dn_core: Bolo) -> None:
    dn_core.catalog.resolve("fake-denoise").options["fail"] = True  # type: ignore[union-attr]
    trace = Trace("stt", "internal", "")
    t = await dn_core.transcribe(noisy(), SR, trace=trace, denoise=DenoiseOptions())
    assert t.text and trace.extra["denoise"]["status"] == "skipped"
    assert "fake denoiser failure" in trace.extra["denoise"]["detail"]


def test_http_transcriptions(dn_live: LiveServer) -> None:
    url = dn_live.url + "/v1/audio/transcriptions"
    r = httpx.post(url, files={"file": wav(noisy())}, data={"model": "fake-stt"}, timeout=30)
    assert r.status_code == 200 and "x-bolo-denoise" not in r.headers
    r = httpx.post(url, files={"file": wav(noisy())}, data={"model": "fake-stt", "denoise": "true"}, timeout=30)
    assert r.status_code == 200 and r.headers["x-bolo-denoise"] == "fake-denoise"
    trace = httpx.get(dn_live.url + f"/api/v1/requests/{r.headers['x-request-id']}").json()
    assert trace["denoise"]["status"] == "applied"
    r = httpx.post(url, files={"file": wav(noisy())}, data={"model": "fake-stt", "denoise": "0.5"}, timeout=30)
    assert r.headers["x-bolo-denoise"] == "fake-denoise; strength=0.5"
    r = httpx.post(
        url,
        files={"file": wav(noisy())},
        data={"model": "fake-stt", "denoise": "fake-denoise-48k", "denoise_strength": "0.3"},
        timeout=30,
    )
    assert r.headers["x-bolo-denoise"] == "fake-denoise-48k; strength=0.3"
    for bad in ("1.5", "no-such-model"):
        r = httpx.post(url, files={"file": wav(noisy())}, data={"model": "fake-stt", "denoise": bad}, timeout=30)
        assert r.status_code in (400, 404) and r.json()["error"]["param"] == "denoise"
    # a model pin through the API, then a request that says nothing
    r = httpx.put(dn_live.url + "/api/v1/models/fake-stt/settings", json={"denoise": 0.5}, timeout=30)
    assert r.status_code == 200 and r.json()["settings"]["denoise"] == 0.5
    r = httpx.post(url, files={"file": wav(noisy())}, data={"model": "fake-stt"}, timeout=30)
    assert r.headers["x-bolo-denoise"] == "fake-denoise; strength=0.5"
    r = httpx.put(dn_live.url + "/api/v1/models/fake-stt/settings", json={"denoise": "nothing-here"}, timeout=30)
    assert r.status_code in (400, 404)
    info = httpx.get(dn_live.url + "/api/v1/info").json()["capabilities"]["denoise"]
    assert info["default_model"] == "fake-denoise" and info["default_on"] is False


async def test_realtime_session_config(dn_core: Bolo) -> None:
    sent: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    session = RealtimeSession(dn_core, send, transcription=True)
    assert session.session_object()["audio"]["input"]["noise_reduction"] is None
    session.configure({"type": "transcription", "audio": {"input": {"noise_reduction": {"type": "near_field"}}}})
    audio_in = session.session_object()["audio"]["input"]
    assert audio_in["noise_reduction"] == {"type": "near_field"} and audio_in["denoise"]["enabled"] is True
    session.configure({"audio": {"input": {"denoise": {"model": "fake-denoise-48k", "strength": 0.5}}}})
    assert session.denoise == DenoiseOptions(model="fake-denoise-48k", strength=0.5)
    session.configure({"audio": {"input": {"noise_reduction": None}}})
    assert session.denoise is None
    with pytest.raises(RealtimeError):
        session.configure({"audio": {"input": {"noise_reduction": {"type": "loud"}}}})
    with pytest.raises(RealtimeError) as err:
        session.configure({"audio": {"input": {"denoise": "no-such-model"}}})
    assert err.value.param == "session.audio.input.denoise"
    session.configure({"audio": {"input": {"transcription": {"model": "fake-stt"}, "denoise": True}}})
    item = {"id": "item_1", "content": [{"type": "input_audio"}]}
    assert await session._transcribe(item, noisy(), Trace("stt", "realtime", ""))
    assert calls(dn_core) == [(2 * SR, SR)]
    assert any(e["type"] == "conversation.item.input_audio_transcription.completed" for e in sent)


def test_http_denoise_route(dn_live: LiveServer) -> None:
    """``POST /v1/audio/denoise``: the enhanced recording, whole or streamed, in the format asked for."""
    from bolo.audio import decode_audio

    url = dn_live.url + "/v1/audio/denoise"
    x = noisy()
    r = httpx.post(url, files={"file": wav(x)}, timeout=30)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav") and r.headers["x-bolo-denoise"] == "fake-denoise"
    out, rate = decode_audio(r.content)
    assert rate == SR and abs(len(out) - len(x)) <= 1
    assert float(np.std(out - x)) > 0.01  # the noise went
    trace = httpx.get(dn_live.url + f"/api/v1/requests/{r.headers['x-request-id']}").json()
    assert trace["denoise"]["model"] == "fake-denoise"

    r = httpx.post(url, files={"file": wav(x)}, data={"strength": "0.5", "response_format": "flac"}, timeout=30)
    assert r.status_code == 200 and r.headers["x-bolo-denoise"] == "fake-denoise; strength=0.5"
    assert r.content[:4] == b"fLaC"

    # streamed: the model has a live form, so the audio comes back piece by piece, as long as the input
    r = httpx.post(
        url, files={"file": wav(x)}, data={"stream": "true", "response_format": "pcm", "model": "fake-denoise-48k"},
        timeout=30,
    )  # fmt: skip
    assert r.status_code == 200, r.text
    assert len(r.content) == 2 * len(x)
    r = httpx.post(url, files={"file": wav(x)}, data={"stream": "true", "sample_rate": "8000"}, timeout=30)
    out, rate = decode_audio(r.content)
    assert rate == 8000 and abs(len(out) - len(x) // 2) <= 32

    for data, param in (({"model": "fake-stt"}, "model"), ({"strength": "2"}, "strength")):
        r = httpx.post(url, files={"file": wav(x)}, data=data, timeout=30)
        assert r.status_code == 400 and r.json()["error"]["param"] == param

    # ElevenLabs' voice-isolation route: multipart "audio", MP3 back by default
    r = httpx.post(dn_live.url + "/v1/audio-isolation", files={"audio": wav(x)}, timeout=30)
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/mpeg")
    r = httpx.post(
        dn_live.url + "/v1/audio-isolation/stream", files={"audio": wav(x)}, data={"output_format": "pcm_16000"},
        timeout=30,
    )  # fmt: skip
    assert r.status_code == 200 and len(r.content) == 2 * len(x)
