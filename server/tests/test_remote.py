"""The remote engine against stand-ins for the cloud APIs (tests/fake_providers.py): request mapping, auth
headers, streaming, error mapping, fallback, key redaction, cost estimates, and bolo's features on top."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import io
import json
import logging
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
import websockets

from bolo.catalog import Catalog, ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import parse_fallbacks
from bolo.providers import KIND_BY_NAME, KINDS, ProviderConfig, ProviderRegistry, redact_provider, scrub
from bolo.remote.sarvam import guess_language, sarvam_language
from conftest import HTTP_TIMEOUT, LiveServer, speechlike, two_speakers, wav_bytes
from fake_providers import KEY, FakeLiveServer, FakeProviders
from fake_realtime_server import FakeRealtimeServer
from test_realtime import b64_pcm24
from test_s2s import connect, until, update
from test_ws import pcm16, stream_audio

PRICES = {
    "whisper-1": {"per_minute": 0.006},
    "gpt-4o-mini-tts": {"per_1k_chars": 0.015, "per_minute": 0.01},
    "*": {"per_request": 0.001, "currency": "EUR"},
}


@pytest.fixture
def cloud() -> Iterator[FakeProviders]:
    server = FakeProviders().start()
    try:
        yield server
    finally:
        server.stop()


def with_providers(settings: Settings, cloud: FakeProviders, live_url: str | None = None) -> Settings:
    settings.providers = {
        "openai": {"kind": "openai-compatible", "base_url": cloud.url + "/v1", "api_key": KEY, "prices": PRICES},
        "sarvam": {"base_url": cloud.url, "api_key": KEY, "prices": {"saaras:v4": {"per_hour": 30, "currency": "INR"}}},
        "elevenlabs": {"base_url": cloud.url, "api_key": KEY},
        "deepgram": {"base_url": live_url or cloud.url, "api_key": KEY},
        "soniox": {"base_url": live_url or cloud.url + "/soniox", "api_key": KEY},
    }
    return settings


@pytest.fixture
def srv(settings: Settings, cloud: FakeProviders) -> Iterator[LiveServer]:
    server = LiveServer(with_providers(settings, cloud))
    server.start()
    try:
        yield server
    finally:
        server.stop()


@contextlib.contextmanager
def another(settings: Settings, tmp_path: Path, **changes: Any) -> Iterator[LiveServer]:
    """A second server next to a fixture's (fixtures share one Settings object), with its own data directory."""
    server = LiveServer(dataclasses.replace(settings, data_dir=tmp_path / "other", **changes))
    server.start()
    try:
        yield server
    finally:
        server.stop()


def transcribe(server: LiveServer, model: str, seconds: float = 2.0, **data: Any) -> httpx.Response:
    endpoint = data.pop("endpoint", "transcriptions")
    return httpx.post(
        f"{server.url}/v1/audio/{endpoint}",
        data={"model": model, **data},
        files={"file": ("a.wav", wav_bytes(seconds), "audio/wav")},
        timeout=60,
    )


def speak(server: LiveServer, model: str, text: str = "Hello there.", **body: Any) -> httpx.Response:
    return httpx.post(
        f"{server.url}/v1/audio/speech",
        json={"model": model, "input": text, "response_format": "wav", **body},
        timeout=60,
    )


def wav_seconds(data: bytes) -> tuple[float, int]:
    with sf.SoundFile(io.BytesIO(data)) as f:
        return len(f) / f.samplerate, int(f.samplerate)


def trace_of(server: LiveServer, response: httpx.Response) -> dict[str, Any]:
    doc: dict[str, Any] = httpx.get(f"{server.url}/api/v1/requests/{response.headers['x-request-id']}").json()
    return doc


# -- configuration ------------------------------------------------------------------------------------------------
def test_providers_in_the_file_and_the_environment(tmp_path: Path) -> None:
    config = tmp_path / "bolo.toml"
    config.write_text(
        'fallbacks = ["whisper-base=openai-whisper-1"]\n'
        "[providers.openai]\n"
        'api_key = "sk-file"\n'
        "[providers.openai.prices]\n"
        '"whisper-1" = { per_minute = 0.006 }\n'
        "[providers.lan-bolo]\n"
        'kind = "openai-compatible"\n'
        'base_url = "http://10.0.0.5:8000/v1"\n'
        "[providers.lan-bolo.options]\n"
        "timeout_s = 20\n"
    )
    env = {
        "BOLO_PROVIDER_SARVAM_API_KEY": "sv-env",
        "BOLO_PROVIDER_MY_DG_KIND": "deepgram",
        "BOLO_PROVIDER_MY_DG_API_KEY": "dg-env",
        "BOLO_PROVIDER_OPENAI_BASE_URL": "https://eu.api.openai.com/v1",
    }
    s = load_settings(config, env=env)
    s.data_dir = tmp_path / "data"
    reg = ProviderRegistry(s)
    by_name = {p.name: p for p in reg.all()}
    assert set(by_name) == {"openai", "lan-bolo", "sarvam", "my-dg"}
    # the environment adds to the file's table: the key stays, the address changes
    assert by_name["openai"].api_key == "sk-file" and by_name["openai"].base_url == "https://eu.api.openai.com/v1"
    assert (
        by_name["openai"].kind == "openai-compatible" and by_name["openai"].prices["whisper-1"]["per_minute"] == 0.006
    )
    assert by_name["sarvam"].kind == "sarvam" and by_name["sarvam"].base_url == "https://api.sarvam.ai"
    assert by_name["my-dg"].kind == "deepgram" and by_name["my-dg"].configured
    # a self-hosted OpenAI-compatible server needs no key; a vendor does
    assert by_name["lan-bolo"].configured and by_name["lan-bolo"].options == {"timeout_s": 20}
    assert not ProviderConfig.from_dict("openai", {}).configured
    assert parse_fallbacks(s.fallbacks) == {"whisper-base": ["openai-whisper-1"]}
    # nothing that is shown carries a key
    shown = json.dumps(s.public_dict(), default=str) + json.dumps(reg.public())
    assert "sk-file" not in shown and "sv-env" not in shown and "dg-env" not in shown
    assert s.public_dict()["providers"]["openai"]["api_key"] == "***"
    one = load_settings(env={"BOLO_PROVIDERS": '{"elevenlabs": {"api_key": "el"}}', "BOLO_CONFIG": str(config)})
    assert one.providers == {"elevenlabs": {"api_key": "el"}}
    with pytest.raises(ValueError, match="JSON object"):
        load_settings(config, env={"BOLO_PROVIDERS": "[1]"})
    with pytest.raises(ValueError, match="fallbacks"):
        parse_fallbacks(["nonsense"])
    # a provider or fallback entry that cannot be read stops the server at start-up, naming the setting
    with pytest.raises(ValueError, match="providers: provider 'x': kind must be one of"):
        load_settings(env={"BOLO_PROVIDERS": '{"x": {"kind": "azure"}}', "BOLO_CONFIG": str(config)})
    with pytest.raises(ValueError, match="fallbacks: expected"):
        load_settings(env={"BOLO_FALLBACKS": "whisper-base", "BOLO_CONFIG": str(config)})


def test_provider_validation_and_redaction() -> None:
    from bolo.errors import InvalidRequest

    for name, entry, match in [
        ("Bad Name", {"kind": "sarvam"}, "invalid provider name"),
        ("x", {"kind": "azure"}, "kind must be one of"),
        ("x", {}, "kind must be one of"),
        ("x", {"kind": "sarvam", "base_url": "ftp://h"}, "base_url"),
        ("x", {"kind": "sarvam", "secret": 1}, "unknown fields"),
        ("x", {"kind": "sarvam", "prices": {"m": {"per_token": 1}}}, "price"),
    ]:
        with pytest.raises(InvalidRequest, match=match):
            ProviderConfig.from_dict(name, entry)
    shown = redact_provider({"api_key": "k", "options": {"organization": "o", "extra_token": "t"}})
    assert shown == {"api_key": "***", "options": {"organization": "o", "extra_token": "***"}}
    assert scrub("bad key sk-abcdef given", "sk-abcdef") == "bad key *** given"
    cfg = ProviderConfig.from_dict("openai", {"api_key": "k", "prices": PRICES})
    assert cfg.estimate("whisper-1", audio_seconds=90) == {"amount": 0.009, "currency": "USD", "estimated": True}
    assert cfg.estimate("gpt-4o-mini-tts", audio_seconds=60, chars=2000)["amount"] == pytest.approx(0.04)
    assert cfg.estimate("anything-else", requests=3) == {"amount": 0.003, "currency": "EUR", "estimated": True}
    assert ProviderConfig.from_dict("sarvam", {"api_key": "k"}).estimate("saaras:v4", audio_seconds=10) is None


# -- the catalog ----------------------------------------------------------------------------------------------
#: the provider names the catalog's cloud entries look for
PROVIDER_NAMES = KIND_BY_NAME.keys()


def test_cloud_entries_of_the_catalog() -> None:
    cloud_models = [m for m in Catalog.load().all() if m.cloud]
    assert len(cloud_models) >= 18
    for m in cloud_models:
        assert m.engine == "remote" and m.kind == f"remote-{m.task}" and m.validated == "untested"
        assert m.license_url and m.license_url.startswith("https://") and "cloud API" in m.license
        assert m.options["provider"] in PROVIDER_NAMES and m.options["model"]
        api = m.to_api()
        assert api["cloud"] is True and api["notice"] == "paid cloud API; audio leaves this server"
        assert "leaves this server" in m.description and (m.languages or m.task == "denoise")
    ids = {m.id for m in cloud_models}
    assert {"openai-whisper-1", "openai-gpt-4o-mini-tts", "sarvam-saaras-v4", "sarvam-bulbul-v3",
            "elevenlabs-flash-v2.5", "elevenlabs-scribe-v3", "deepgram-nova-3", "soniox-stt-async-v6",
            "soniox-stt-rt-v6"} <= ids  # fmt: skip
    bulbul = next(m for m in cloud_models if m.id == "sarvam-bulbul-v4-flash")
    assert len(bulbul.voices) > 200 and {"shubh_enhi_ads", "kavitha_te_narration"} <= {v["id"] for v in bulbul.voices}


def test_cloud_models_are_installed_when_their_provider_is_configured(
    srv: LiveServer, settings: Settings, tmp_path: Path
) -> None:
    listed = {m["id"]: m for m in httpx.get(f"{srv.url}/api/v1/catalog").json()["models"]}
    whisper = listed["openai-whisper-1"]
    assert whisper["installed"] is True and whisper["cloud"] is True and whisper["size_bytes"] == 0
    assert whisper["provider"] == {
        "name": "openai", "kind": "openai-compatible", "model": "whisper-1", "configured": True,
        "price": {"per_minute": 0.006}, "key_env": "BOLO_PROVIDER_OPENAI_API_KEY",
    }  # fmt: skip
    assert whisper["license_url"] == "https://openai.com/policies/services-agreement/"
    assert listed["fake-stt"]["cloud"] is False and listed["fake-stt"]["provider"] is None
    served = [m["id"] for m in httpx.get(f"{srv.url}/v1/models").json()["data"]]
    assert "sarvam-bulbul-v3" in served and "deepgram-nova-3" in served
    # without providers: listed, not installed, and using one says what to configure
    with another(settings, tmp_path, providers={}) as live:
        bare = {m["id"]: m for m in httpx.get(f"{live.url}/api/v1/catalog").json()["models"]}
        assert bare["openai-whisper-1"]["installed"] is False
        assert bare["openai-whisper-1"]["provider"]["configured"] is False
        r = transcribe(live, "openai-whisper-1")
        assert r.status_code == 503 and r.json()["error"]["code"] == "provider_not_configured"
        assert "BOLO_PROVIDER_OPENAI_API_KEY" in r.json()["error"]["message"]
        job = httpx.post(f"{live.url}/api/v1/models/pull", json={"id": "openai-whisper-1"}).json()["job_id"]
        assert "not configured" in httpx.get(f"{live.url}/api/v1/jobs/{job}").json()["error"]


# -- OpenAI-compatible ----------------------------------------------------------------------------------------
def test_openai_transcription_mapping_trace_and_cost(srv: LiveServer, cloud: FakeProviders) -> None:
    r = transcribe(
        srv, "openai-whisper-1", language="hi-IN", prompt="Names:", hotwords="Bolo, Sarvam",
        response_format="verbose_json", **{"timestamp_granularities[]": ["word", "segment"]},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "heard 2.0 seconds in hi" and body["language"] == "hindi"
    assert [w["word"] for w in body["words"]] == ["heard", "2.0", "seconds", "in", "hi"]
    assert body["segments"][0]["avg_logprob"] == -0.2
    sent = cloud.last("/v1/audio/transcriptions")
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    form = sent["form"]
    assert form["model"] == "whisper-1" and form["language"] == "hi" and form["response_format"] == "verbose_json"
    assert form["prompt"] == "Names:, Bolo, Sarvam" and form["timestamp_granularities[]"] == ["word", "segment"]
    assert form["file.filename"] == "audio.wav" and form["file.content_type"] == "audio/wav"
    assert r.headers["x-bolo-model"] == "openai-whisper-1" and "x-bolo-fallback" not in r.headers
    assert r.headers["x-bolo-estimated-cost"] == "0.0002 USD"
    trace = trace_of(srv, r)
    provider = trace["provider"]
    assert provider["name"] == "openai" and provider["model"] == "whisper-1" and provider["requests"] == 1
    assert provider["bytes_sent"] > 60_000 and provider["bytes_received"] > 100 and provider["latency_ms"] > 0
    assert trace["cost"] == {"amount": 0.0002, "currency": "USD", "estimated": True}
    stats = httpx.get(f"{srv.url}/api/v1/stats").json()["providers"]
    assert stats[0]["provider"] == "openai" and stats[0]["requests"] == 1 and stats[0]["audio_seconds"] == 2.0
    assert stats[0]["estimated_cost"] == {"USD": 0.0002}
    metrics = httpx.get(f"{srv.url}/metrics").text
    assert 'bolo_provider_requests_total{provider="openai",model="whisper-1"} 1' in metrics
    assert 'bolo_provider_estimated_cost_total{currency="USD",provider="openai",model="whisper-1"} 0.0002' in metrics


def test_openai_translation_streaming_and_unsupported_options(srv: LiveServer, cloud: FakeProviders) -> None:
    r = transcribe(srv, "openai-whisper-1", endpoint="translations")
    assert r.status_code == 200 and r.json()["text"] == "translated 2.0 seconds"
    assert cloud.last("/v1/audio/translations")["form"]["model"] == "whisper-1"
    # gpt-4o-transcribe: bolo asks OpenAI for a stream and joins the deltas
    r = transcribe(srv, "openai-gpt-4o-transcribe", language="en")
    assert r.status_code == 200 and r.json()["text"] == "heard 2.0 seconds in en"
    form = cloud.last("/v1/audio/transcriptions")["form"]
    assert form["stream"] == "true" and form["model"] == "gpt-4o-transcribe" and form["response_format"] == "json"
    # hotwords go where the model takes them
    transcribe(srv, "openai-gpt-transcribe", hotwords="Bolo\nNew Delhi")
    assert cloud.last("/v1/audio/transcriptions")["form"]["keywords[]"] == ["Bolo", "New Delhi"]
    # what a model cannot do is a 400 that names the option, not a silent drop
    r = transcribe(srv, "openai-gpt-4o-transcribe", endpoint="translations")
    assert r.status_code == 400 and "cannot translate" in r.json()["error"]["message"]
    r = transcribe(
        srv, "openai-gpt-4o-transcribe", response_format="verbose_json", **{"timestamp_granularities[]": "word"}
    )
    assert r.status_code == 400 and r.json()["error"]["param"] == "timestamp_granularities[]"
    assert "word timestamps is not supported" in r.json()["error"]["message"]


def test_openai_speech_streams_and_is_resampled(srv: LiveServer, cloud: FakeProviders) -> None:
    text = "Hello there. How are you today?"
    r = speak(srv, "openai-gpt-4o-mini-tts", text, voice="coral", speed=1.25)
    assert r.status_code == 200, r.text
    seconds, rate = wav_seconds(r.content)
    assert rate == 24000 and seconds == pytest.approx(len(text) * 0.05, rel=0.1)
    sent = [q["json"] for q in cloud.requests if q["path"] == "/v1/audio/speech"]
    assert [s["input"] for s in sent] == ["Hello there.", "How are you today?"]  # a sentence at a time
    assert sent[0] == {"model": "gpt-4o-mini-tts", "input": "Hello there.", "voice": "coral", "response_format": "pcm",
                       "speed": 1.25}  # fmt: skip
    assert r.headers["x-bolo-model"] == "openai-gpt-4o-mini-tts"
    trace = trace_of(srv, r)
    assert trace["provider"]["requests"] == 2 and trace["provider"]["first_byte_ms"] > 0
    # 31 characters at 0.015 per 1000, plus the audio minutes
    assert trace["cost"]["amount"] == pytest.approx(31 * 0.015 / 1000 + seconds / 60 * 0.01, rel=0.05)
    assert trace["normalizer"] == "model"  # this entry leaves normalisation to OpenAI
    # telephony: bolo re-encodes whatever the provider sent
    r = speak(srv, "openai-tts-1", "One two three.", response_format="mulaw")
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "8000"
    assert len(r.content) == pytest.approx(14 * 0.05 * 8000, rel=0.1)
    r = speak(srv, "openai-tts-1", voice="nobody")
    assert r.status_code == 400 and "voice 'nobody' is not available" in r.json()["error"]["message"]
    # gpt-4o-mini-tts follows a request's instructions; tts-1 does not, and they are not sent there
    r = speak(srv, "openai-gpt-4o-mini-tts", "Hi.", voice="coral", instructions="Sound tired.")
    assert r.status_code == 200 and r.headers["x-bolo-instructions"] == "applied"
    assert cloud.last("/v1/audio/speech")["json"]["instructions"] == "Sound tired."
    r = speak(srv, "openai-tts-1", "Hi.", instructions="Sound tired.")
    assert r.status_code == 200 and r.headers["x-bolo-instructions"] == "ignored"
    assert "instructions" not in cloud.last("/v1/audio/speech")["json"]


def test_text_is_prepared_before_it_leaves(srv: LiveServer, cloud: FakeProviders) -> None:
    # Bulbul gets bolo's normalisation (the default for voices that read text) ...
    r = speak(srv, "sarvam-bulbul-v3", "It costs $5 today.", language="en")
    assert r.status_code == 200, r.text
    sent = cloud.last("/text-to-speech")["json"]["text"]
    normalizer = trace_of(srv, r)["normalizer"]
    assert "five" in sent.lower() and normalizer in ("uccharan", "light")
    assert "$" not in sent or normalizer == "light"  # the light normaliser (no uccharan) spells the digits only
    # ... unless the request or the model says the provider should do it
    speak(srv, "sarvam-bulbul-v3", "It costs $5 today.", language="en", normalize="model")
    assert "$5" in cloud.last("/text-to-speech")["json"]["text"]
    httpx.put(f"{srv.url}/api/v1/models/sarvam-bulbul-v3/settings", json={"text_normalizer": "none"})
    speak(srv, "sarvam-bulbul-v3", "It costs $5 today.", language="en")
    assert "$5" in cloud.last("/text-to-speech")["json"]["text"]
    info = httpx.get(f"{srv.url}/api/v1/text-input/openai-tts-1").json()
    assert json.dumps(info).count('"reads": "text"') >= 1


# -- Sarvam -------------------------------------------------------------------------------------------------------
def test_sarvam_language_helpers() -> None:
    from bolo.remote.sarvam import STT_LANGUAGES, TTS_LANGUAGES

    assert sarvam_language("hi", STT_LANGUAGES) == "hi-IN" and sarvam_language("or", STT_LANGUAGES) == "od-IN"
    assert sarvam_language("en_US", TTS_LANGUAGES) == "en-IN" and sarvam_language("fr", STT_LANGUAGES) is None
    assert sarvam_language("ur", TTS_LANGUAGES) is None
    assert guess_language("नमस्ते दुनिया") == "hi" and guess_language("வணக்கம்") == "ta" and guess_language("hello") == "en"


def test_sarvam_transcription(srv: LiveServer, cloud: FakeProviders) -> None:
    r = transcribe(srv, "sarvam-saaras-v4", language="hi", hotwords="Sarvam, नई दिल्ली", response_format="verbose_json")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "heard 2.0 seconds in hi-IN" and body["language"] == "hindi"
    assert len(body["segments"]) == 2 and body["segments"][1]["end"] == pytest.approx(2.0, abs=0.01)
    sent = cloud.last("/speech-to-text")
    assert sent["headers"]["api-subscription-key"] == KEY and "Authorization" not in sent["headers"]
    form = sent["form"]
    assert form["model"] == "saaras:v4" and form["language_code"] == "hi-IN" and form["with_timestamps"] == "true"
    assert json.loads(form["keyterms"]) == ["Sarvam", "नई दिल्ली"]
    assert trace_of(srv, r)["cost"] == {"amount": round(30 * 2 / 3600, 8), "currency": "INR", "estimated": True}
    # no language: Sarvam detects it
    r = transcribe(srv, "sarvam-saaras-v4")
    assert cloud.last("/speech-to-text")["form"]["language_code"] == "unknown" and r.status_code == 200
    # translation is a mode of the same route
    r = transcribe(srv, "sarvam-saaras-v3", endpoint="translations")
    assert r.json()["text"] == "translated 2.0 seconds" and cloud.last("/speech-to-text")["form"]["mode"] == "translate"
    for kwargs, param, match in [
        ({"language": "fr"}, "language", "does not transcribe 'fr'"),
        ({"prompt": "context"}, "prompt", "a prompt is not supported"),
        ({"response_format": "verbose_json", "timestamp_granularities[]": "word"}, "timestamp_granularities[]", "word"),
    ]:
        r = transcribe(srv, "sarvam-saaras-v4", **kwargs)
        assert r.status_code == 400 and r.json()["error"]["param"] == param and match in r.json()["error"]["message"]
    r = transcribe(srv, "sarvam-saaras-v3", hotwords="x")  # keyterms are a saaras:v4 feature
    assert r.status_code == 400 and r.json()["error"]["param"] == "hotwords"


def test_sarvam_long_audio_is_cut_at_pauses(srv: LiveServer, cloud: FakeProviders) -> None:
    """The REST route takes 30 s: a 70 s recording goes up as several requests, none longer than that."""
    audio = wav_bytes(70.0, gap_at=(12.0, 25.0, 38.0, 51.0, 63.0))
    r = httpx.post(
        f"{srv.url}/v1/audio/transcriptions",
        data={"model": "sarvam-saaras-v4", "language": "hi"},
        files={"file": ("a.wav", audio, "audio/wav")},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    heard = [float(part.split()[0]) for part in r.json()["text"].split("heard ")[1:]]
    assert len(heard) >= 3 and max(heard) <= 30.0 and 60.0 <= sum(heard) <= 70.5
    assert trace_of(srv, r)["provider"]["requests"] == len(heard) == cloud.count("/speech-to-text")


def test_sarvam_speech(srv: LiveServer, cloud: FakeProviders) -> None:
    r = speak(srv, "sarvam-bulbul-v3", "नमस्ते।", voice="priya", speed=1.5, language="hi")
    assert r.status_code == 200, r.text
    seconds, rate = wav_seconds(r.content)
    assert rate == 24000 and seconds > 0.2
    sent = cloud.last("/text-to-speech")
    assert sent["headers"]["api-subscription-key"] == KEY
    body = sent["json"]
    assert body["language_code"] == "hi-IN" and body["speaker"] == "priya" and body["model"] == "bulbul:v3"
    assert body["pace"] == 1.5 and body["speech_sample_rate"] == 24000 and body["output_audio_codec"] == "wav"
    # no language in the request: taken from the script of the text
    speak(srv, "sarvam-bulbul-v3", "வணக்கம்.")
    assert cloud.last("/text-to-speech")["json"]["language_code"] == "ta-IN"
    # Bulbul v2 answers at 22.05 kHz; a persona voice of v4 Flash brings its language
    r = speak(srv, "sarvam-bulbul-v2", "Hello.", voice="karun")
    assert wav_seconds(r.content)[1] == 22050 and cloud.last("/text-to-speech")["json"]["language_code"] == "en-IN"
    speak(srv, "sarvam-bulbul-v4-flash", "नमस्ते।", voice="ritu_hi_customer")
    body = cloud.last("/text-to-speech")["json"]
    assert (
        body["speaker"] == "ritu_hi_customer"
        and body["language_code"] == "hi-IN"
        and body["model"] == "bulbul:v4-flash"
    )
    r = speak(srv, "sarvam-bulbul-v3", "Hello.", speed=3.0)
    assert (
        r.status_code == 400 and r.json()["error"]["param"] == "speed" and "0.5 to 2.0" in r.json()["error"]["message"]
    )
    r = speak(srv, "sarvam-bulbul-v3", "Bonjour.", language="fr", normalize=False)
    assert r.status_code == 400 and r.json()["error"]["param"] == "language"


# -- ElevenLabs ---------------------------------------------------------------------------------------------------
def test_elevenlabs_speech_and_transcription(srv: LiveServer, cloud: FakeProviders) -> None:
    # the account's voices are read when the model loads; a voice can be named
    r = speak(srv, "elevenlabs-flash-v2.5", "Hello there.", voice="Arjun", speed=1.1, language="hi-IN")
    assert r.status_code == 200, r.text
    assert wav_seconds(r.content) == (pytest.approx(12 * 0.05, rel=0.1), 24000)
    sent = cloud.last("/v1/text-to-speech/")
    assert sent["path"] == "/v1/text-to-speech/v_arjun_02/stream" and sent["query"] == [("output_format", "pcm_24000")]
    assert sent["headers"]["xi-api-key"] == KEY and "Authorization" not in sent["headers"]
    assert sent["json"] == {"text": "Hello there.", "model_id": "eleven_flash_v2_5", "language_code": "hi",
                            "voice_settings": {"speed": 1.1}}  # fmt: skip
    listed = httpx.get(f"{srv.url}/api/v1/models/elevenlabs-flash-v2.5").json()["voices"]
    assert {"id": "v_rachel_01", "name": "Rachel", "language": "en", "gender": "f"} in listed  # once loaded
    # multilingual v2 takes no language_code
    speak(srv, "elevenlabs-multilingual-v2", "Hello.", voice="v_rachel_01", language="en")
    assert "language_code" not in cloud.last("/v1/text-to-speech/")["json"]
    r = transcribe(
        srv, "elevenlabs-scribe-v3", language="en", hotwords="Bolo", response_format="verbose_json",
        **{"timestamp_granularities[]": "word"},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "heard 2.0 seconds" and [w["word"] for w in r.json()["words"]] == [
        "heard",
        "2.0",
        "seconds",
    ]
    form = cloud.last("/v1/speech-to-text")["form"]
    assert form["model_id"] == "scribe_v3" and form["language_code"] == "en" and form["keyterms"] == "Bolo"
    assert form["timestamps_granularity"] == "word"
    r = transcribe(srv, "elevenlabs-scribe-v3", prompt="context")
    assert r.status_code == 400 and r.json()["error"]["param"] == "prompt"


# -- Deepgram -----------------------------------------------------------------------------------------------------
def test_deepgram_prerecorded(srv: LiveServer, cloud: FakeProviders) -> None:
    r = transcribe(
        srv, "deepgram-nova-3", language="en-IN", hotwords="Bolo, New Delhi", response_format="verbose_json",
        **{"timestamp_granularities[]": "word"},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "Heard 2.0 seconds" and r.json()["words"][0]["word"] == "Heard"
    sent = cloud.last("/v1/listen")
    assert sent["headers"]["Authorization"] == f"Token {KEY}" and sent["headers"]["Content-Type"] == "audio/wav"
    assert sent["body"][:4] == b"RIFF" and sent["bytes"] > 60_000
    assert sent["query"] == [
        ("model", "nova-3"), ("language", "en-IN"), ("smart_format", "true"), ("keyterm", "Bolo"),
        ("keyterm", "New Delhi"), ("utterances", "true"),
    ]  # fmt: skip
    r = transcribe(srv, "deepgram-nova-2", hotwords="Bolo", response_format="verbose_json")
    query = dict(cloud.last("/v1/listen")["query"])
    assert query["detect_language"] == "true" and query["keywords"] == "Bolo" and r.json()["language"] == "hindi"
    r = transcribe(srv, "deepgram-nova-3", endpoint="translations")
    assert r.status_code == 400 and "cannot translate" in r.json()["error"]["message"]


@pytest.fixture
def live_socket() -> Iterator[FakeLiveServer]:
    server = FakeLiveServer().start()
    try:
        yield server
    finally:
        server.stop()


async def run_live(server: LiveServer, model: str, extra: str = "") -> list[dict[str, Any]]:
    async with websockets.connect(f"{server.ws_url}/api/v1/stt/stream?model={model}&vad=false{extra}") as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready" and ready["mode"] == "native", ready
        audio = np.concatenate([speechlike(2.0), np.zeros(16000, dtype=np.float32)])
        await stream_audio(ws, pcm16(audio), 3200, delay=0.03)
        await ws.send(json.dumps({"type": "stop"}))
        return [json.loads(m) async for m in ws]


async def test_deepgram_live_socket(settings: Settings, cloud: FakeProviders, live_socket: FakeLiveServer) -> None:
    server = LiveServer(with_providers(settings, cloud, live_socket.url))
    server.start()
    try:
        messages = await run_live(server, "deepgram-nova-3", "&language=en")
        # the socket is closed when the session ends, not left to idle out
        for _ in range(100):
            if "CloseStream" in live_socket.control:
                break
            await asyncio.sleep(0.05)
        stats = httpx.get(f"{server.url}/api/v1/stats").json()["providers"]
    finally:
        server.stop()
    partials = [m["text"] for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    assert partials and partials[0].startswith("hearing ")
    assert len(finals) == 1 and finals[0]["text"].startswith("heard 2.") and finals[0]["text"].endswith(" seconds")
    assert messages[-1]["type"] == "done"
    path = live_socket.paths[0]
    assert path.startswith("/v1/listen?model=nova-3&language=en&smart_format=true&encoding=linear16&sample_rate=16000")
    assert "interim_results=true" in path and KEY not in path  # the key travels in a header only
    assert live_socket.headers[0]["Authorization"] == f"Token {KEY}"
    assert "CloseStream" in live_socket.control
    assert stats and stats[0]["provider"] == "deepgram" and stats[0]["audio_seconds"] >= 2.9


async def test_sarvam_live_socket(settings: Settings, cloud: FakeProviders, live_socket: FakeLiveServer) -> None:
    with_providers(settings, cloud)
    settings.providers["sarvam"]["base_url"] = live_socket.url
    server = LiveServer(settings)
    server.start()
    try:
        messages = await run_live(server, "sarvam-saaras-v4-streaming", "&language=hi")
    finally:
        server.stop()
    finals = [m for m in messages if m["type"] == "final"]
    assert len(finals) == 1 and finals[0]["text"].startswith("heard 2.")
    path = live_socket.paths[0]
    assert path.startswith("/speech-to-text/ws?language-code=hi-IN&model=saaras%3Av4&sample_rate=16000")
    assert live_socket.headers[0]["Api-Subscription-Key"] == KEY


# -- Soniox -------------------------------------------------------------------------------------------------------
def test_soniox_recorded_audio(srv: LiveServer, cloud: FakeProviders) -> None:
    r = transcribe(
        srv, "soniox-stt-async-v6", language="en", prompt="A call about bolo.", hotwords="Bolo, New Delhi",
        response_format="verbose_json", **{"timestamp_granularities[]": "word"},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    doc = r.json()
    # Soniox's pieces of words ("He", "ard") are joined into words; a leading space starts the next one
    assert doc["text"] == "Heard 2.0 seconds." and [w["word"] for w in doc["words"]] == ["Heard", "2.0", "seconds."]
    assert doc["language"] == "english" and doc["words"][2]["end"] == pytest.approx(2.0, abs=0.01)
    upload = cloud.last("/soniox/v1/files", "POST")
    assert upload["headers"]["Authorization"] == f"Bearer {KEY}" and upload["form"]["file.content_type"] == "audio/wav"
    body = cloud.last("/soniox/v1/transcriptions", "POST")["json"]
    assert body == {
        "model": "stt-async-v6", "language_hints": ["en"], "enable_language_identification": True,
        "context": {"text": "A call about bolo.", "terms": ["Bolo", "New Delhi"]}, "file_id": body["file_id"],
    }  # fmt: skip
    # queued, processing, completed: two polls; then the transcription and the file are deleted
    polls = [q for q in cloud.requests if q["method"] == "GET" and q["path"].count("/") == 4]
    assert len(polls) == 2 and not cloud.soniox_jobs and not cloud.soniox_files
    assert [q["method"] for q in cloud.requests if q["method"] == "DELETE"] == ["DELETE", "DELETE"]
    trace = trace_of(srv, r)
    # upload, create, two polls, transcript (the deletes are not counted)
    assert trace["provider"]["kind"] == "soniox" and trace["provider"]["requests"] == 5
    # no language: Soniox identifies it per token; the language most of the text is in is answered
    r = transcribe(srv, "soniox-stt-async-v6", response_format="verbose_json")
    assert (
        r.json()["language"] == "hindi"
        and "language_hints" not in cloud.last("/soniox/v1/transcriptions", "POST")["json"]
    )
    # the live entry sends recorded audio to the async model of the same generation
    r = transcribe(srv, "soniox-stt-rt-v6")
    assert r.status_code == 200 and cloud.last("/soniox/v1/transcriptions", "POST")["json"]["model"] == "stt-async-v6"
    # translation to English: Soniox's translated tokens
    r = transcribe(srv, "soniox-stt-async-v6", endpoint="translations")
    assert r.status_code == 200 and r.json()["text"] == "Translated 2.0 seconds"
    assert cloud.last("/soniox/v1/transcriptions", "POST")["json"]["translation"] == {
        "type": "one_way",
        "target_language": "en",
    }
    test = httpx.post(f"{srv.url}/api/v1/providers/soniox/test").json()
    assert test["ok"] and test["checked"] == "credentials" and "2 models" in test["message"]


def test_soniox_labels_speakers_itself(srv: LiveServer, cloud: FakeProviders) -> None:
    r = transcribe(srv, "soniox-stt-async-v6", language="en", response_format="diarized_json")
    assert r.status_code == 200, r.text
    segments = r.json()["segments"]
    assert [(s["speaker"], s["text"]) for s in segments] == [("A", "Heard"), ("B", "2.0 seconds.")]
    assert segments[1]["start"] == pytest.approx(0.667, abs=0.01)
    assert cloud.last("/soniox/v1/transcriptions", "POST")["json"]["enable_speaker_diarization"] is True
    assert trace_of(srv, r)["diarization_model"] == "provider"
    r = transcribe(
        srv, "soniox-stt-async-v6", language="en", diarize="true", response_format="verbose_json",
        **{"timestamp_granularities[]": ["word", "segment"]},
    )  # fmt: skip
    assert [w.get("speaker") for w in r.json()["words"]] == ["A", "B", "B"]
    # without diarization, nothing about speakers is asked for or answered
    r = transcribe(srv, "soniox-stt-async-v6", language="en", response_format="verbose_json")
    assert "enable_speaker_diarization" not in cloud.last("/soniox/v1/transcriptions", "POST")["json"]
    assert all("speaker" not in s for s in r.json()["segments"])
    # a named diarization model is bolo's: the recording is cut at its turns as for any recogniser
    r = transcribe(srv, "soniox-stt-async-v6", language="en", diarize="true", diarization_model="fake-diarization")
    assert r.status_code == 200, r.text
    assert "enable_speaker_diarization" not in cloud.last("/soniox/v1/transcriptions", "POST")["json"]


def test_soniox_errors(settings: Settings, cloud: FakeProviders) -> None:
    from bolo.errors import ProviderTimeout
    from bolo.remote import RemoteRequest, make_adapter

    with_providers(settings, cloud)
    server = LiveServer(settings)
    server.start()
    try:
        cloud.soniox_fail = ("audio_decode_error", f"the audio could not be decoded (key {KEY})")
        r = transcribe(server, "soniox-stt-async-v6")
        assert r.status_code == 400 and "audio_decode_error" in r.text and KEY not in r.text
        cloud.soniox_fail = ("internal_error", "try again")
        r = transcribe(server, "soniox-stt-async-v6")
        assert r.status_code == 502 and r.json()["error"]["code"] == "provider_error"
        assert not cloud.soniox_files  # cleaned up after a failure too
        cloud.retry_after = "3"
        cloud.fail = [429]
        r = transcribe(server, "soniox-stt-async-v6")
        assert r.status_code == 429 and r.headers["retry-after"] == "3"
        cloud.fail = [402]
        r = transcribe(server, "soniox-stt-async-v6")
        assert r.status_code == 502 and "(402)" in r.json()["error"]["message"]
        cloud.fail = [401]
        r = transcribe(server, "soniox-stt-async-v6")
        assert r.status_code == 502 and r.json()["error"]["code"] == "provider_auth" and KEY not in r.text
    finally:
        server.stop()
    # a transcription that takes longer than the model allows
    cfg = ProviderConfig.from_dict("soniox", settings.providers["soniox"])
    adapter = make_adapter(cfg, "stt-async-v6", {"poll_timeout_s": 0.1})
    try:
        usage = adapter.new_usage()
        with pytest.raises(ProviderTimeout, match="had not finished"):
            adapter.transcribe(RemoteRequest(wav_bytes(1.0), "audio/wav", 1.0), usage)
    finally:
        adapter.close()
    assert not cloud.soniox_files  # the upload is removed even so (the transcription cannot be, while it runs)


async def test_soniox_live_socket(settings: Settings, cloud: FakeProviders, live_socket: FakeLiveServer) -> None:
    with_providers(settings, cloud)
    settings.providers["soniox"]["base_url"] = live_socket.url
    server = LiveServer(settings)
    server.start()
    try:
        messages = await run_live(server, "soniox-stt-rt-v6", "&language=en")
        for _ in range(100):
            if "end" in live_socket.control:
                break
            await asyncio.sleep(0.05)
        stats = httpx.get(f"{server.url}/api/v1/stats").json()["providers"]
    finally:
        server.stop()
    partials = [m["text"] for m in messages if m["type"] == "partial"]
    finals = [m for m in messages if m["type"] == "final"]
    # final tokens are held until Soniox's <end>, shown with the non-final ones in the meantime
    assert partials and partials[0] == "heard hearing"
    assert len(finals) == 1 and finals[0]["text"].startswith("heard 2.") and finals[0]["text"].endswith(" seconds")
    assert (
        live_socket.paths[0] == "/transcribe-websocket" and live_socket.headers[0]["Authorization"] == f"Bearer {KEY}"
    )
    assert live_socket.configs[0] == {
        "model": "stt-rt-v6", "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
        "language_hints": ["en"], "enable_endpoint_detection": True,
    }  # fmt: skip
    assert KEY not in json.dumps(live_socket.configs) and "end" in live_socket.control
    assert stats and any(s["provider"] == "soniox" and s["audio_seconds"] >= 2.9 for s in stats)


def test_soniox_tokens() -> None:
    from bolo.remote.soniox import SonioxLive, tokens_to_words, transcript_to_result

    words = tokens_to_words([{"text": "東", "start_ms": 0, "end_ms": 100}, {"text": "京", "start_ms": 100,
                             "end_ms": 200}, {"text": "Hel", "start_ms": 300, "end_ms": 350},
                             {"text": "lo", "start_ms": 350, "end_ms": 400}, {"text": ",", "start_ms": 400,
                             "end_ms": 410}, {"text": " you", "start_ms": 500, "end_ms": 600}])  # fmt: skip
    assert [(w.w, w.t0, w.t1) for w, _ in words] == [
        ("東", 0.0, 0.1), ("京", 0.1, 0.2), ("Hello,", 0.3, 0.41), ("you", 0.5, 0.6),
    ]  # fmt: skip
    result = transcript_to_result({"text": "", "tokens": [
        {"text": "नमस्ते", "start_ms": 0, "end_ms": 500, "language": "hi", "speaker": "2"},
        {"text": " ok", "start_ms": 600, "end_ms": 700, "language": "en", "speaker": "2"},
        {"text": "<end>", "is_final": True},
    ]}, None, False)  # fmt: skip
    assert result.text == "नमस्ते ok" and result.language == "hi" and result.segments[0].speaker == "B"
    live = object.__new__(SonioxLive)
    live._final = []
    assert live.parse(json.dumps({"tokens": [], "error_code": 503, "error_type": "service_unavailable",
                                  "error_message": "Cannot continue request (code 11)."}))[0].kind == "error"  # fmt: skip
    events = live.parse(json.dumps({"tokens": [{"text": "Hi", "is_final": True}, {"text": " the", "is_final": False}]}))
    assert [(e.kind, e.text) for e in events] == [("interim", "Hi the")]
    events = live.parse(json.dumps({"tokens": [{"text": " there", "is_final": True, "start_ms": 10, "end_ms": 20},
                                               {"text": "<fin>", "is_final": True}]}))  # fmt: skip
    assert [(e.kind, e.text) for e in events] == [("final", "Hi there"), ("endpoint", "")]


# -- errors and keys ----------------------------------------------------------------------------------------------
def test_provider_errors_are_mapped(srv: LiveServer, cloud: FakeProviders) -> None:
    cloud.retry_after = "7"
    for status, expect, code in [
        (500, 502, "provider_error"),
        (503, 502, "provider_error"),
        (504, 504, "provider_timeout"),
        (429, 429, "overloaded"),
        (404, 502, "provider_error"),
        (422, 400, "invalid_request"),
    ]:
        cloud.fail = [status]
        r = transcribe(srv, "openai-whisper-1")
        assert (r.status_code, r.json()["error"]["code"]) == (expect, code), (status, r.text)
        assert KEY not in r.text and "***" in r.text  # the stand-in echoes the key in its message
        if status == 429:
            assert r.headers["retry-after"] == "7"
    for model, path in [("sarvam-saaras-v4", "/speech-to-text"), ("deepgram-nova-3", "/v1/listen"),
                        ("elevenlabs-scribe-v3", "/v1/speech-to-text"), ("soniox-stt-async-v6", "/soniox/v1/files")]:  # fmt: skip
        cloud.fail = [500]
        r = transcribe(srv, model)
        assert r.status_code == 502 and "simulated failure 500" in r.json()["error"]["message"], (model, path)
    r = transcribe(srv, "openai-whisper-1", prompt="bad request")
    assert r.status_code == 400 and "refused the request: the prompt is not acceptable" in r.json()["error"]["message"]
    cloud.fail = [500]
    r = speak(srv, "openai-tts-1")
    assert r.status_code == 502 and r.json()["error"]["code"] == "provider_error"
    stats = {(s["provider"], s["model"]): s for s in httpx.get(f"{srv.url}/api/v1/stats").json()["providers"]}
    assert stats[("openai", "whisper-1")]["errors"] == 7 and KEY not in json.dumps(list(stats.values()))
    assert 'bolo_provider_errors_total{provider="openai",model="tts-1"} 1' in httpx.get(f"{srv.url}/metrics").text


def test_a_wrong_key_and_a_timeout(settings: Settings, cloud: FakeProviders, caplog: pytest.LogCaptureFixture) -> None:
    with_providers(settings, cloud)
    wrong = "not-a-real-key-0123456789"
    settings.providers["openai"]["api_key"] = wrong
    settings.provider_timeout_s = 0.5
    server = LiveServer(settings)
    server.start()
    try:
        with caplog.at_level(logging.DEBUG):
            r = transcribe(server, "openai-whisper-1")
            assert r.status_code == 502 and r.json()["error"]["code"] == "provider_auth"
            assert wrong not in r.text and "refused bolo's API key (401)" in r.text
            cloud.delay_s = 1.2
            r = transcribe(server, "sarvam-saaras-v4")
            assert r.status_code == 504 and r.json()["error"]["code"] == "provider_timeout"
            cloud.delay_s = 0.0
            test = httpx.post(f"{server.url}/api/v1/providers/openai/test").json()
            assert test["ok"] is False and test["code"] == "provider_auth" and wrong not in json.dumps(test)
        everything = "\n".join(
            [
                caplog.text,
                httpx.get(f"{server.url}/api/v1/info").text,
                httpx.get(f"{server.url}/api/v1/providers").text,
                httpx.get(f"{server.url}/api/v1/catalog").text,
                httpx.get(f"{server.url}/api/v1/requests").text,
                httpx.get(f"{server.url}/api/v1/stats").text,
                httpx.get(f"{server.url}/metrics").text,
            ]
        )
        assert wrong not in everything and KEY not in everything
    finally:
        server.stop()


# -- fallback -----------------------------------------------------------------------------------------------------
def test_fallback_from_cloud_to_local_and_back(settings: Settings, cloud: FakeProviders) -> None:
    with_providers(settings, cloud)
    settings.fallbacks = ["openai-whisper-1=no-such-model,fake-stt", "openai-tts-1=fake-tts"]
    server = LiveServer(settings)
    # a local model whose server is down, with a cloud model behind it
    server.core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "down-tts", "task": "tts", "engine": "tts-server", "kind": "tts-server", "family": "Test",
                "languages": ["en"], "sample_rate": 24000, "license": "MIT", "source": {"local": "/nonexistent"},
                "options": {"runtime": "llama-server", "fallback": ["openai-gpt-4o-mini-tts"]},
            }
        )
    )  # fmt: skip
    settings.tts_servers = ["down-tts=http://127.0.0.1:9"]
    server.start()
    try:
        cloud.fail = [500]
        r = transcribe(server, "openai-whisper-1")
        assert r.status_code == 200 and r.json()["text"].startswith("the quick brown fox")
        assert r.headers["x-bolo-model"] == "fake-stt"
        assert r.headers["x-bolo-fallback"] == "openai-whisper-1 -> fake-stt; reason=provider_error"
        trace = trace_of(server, r)
        assert trace["model"] == "fake-stt" and trace["status"] == "ok"
        assert trace["fallback"]["requested"] == "openai-whisper-1" and trace["fallback"]["answered"] == "fake-stt"
        assert trace["fallback"]["attempts"][0]["code"] == "provider_error" and KEY not in json.dumps(trace)
        # healthy again: the cloud model answers, and no header claims otherwise
        r = transcribe(server, "openai-whisper-1")
        assert r.headers["x-bolo-model"] == "openai-whisper-1" and "x-bolo-fallback" not in r.headers
        # a request the provider calls wrong is the caller's to fix: no fallback
        r = transcribe(server, "openai-whisper-1", prompt="bad request")
        assert r.status_code == 400 and "x-bolo-fallback" not in r.headers
        # timeouts and rate limits fall back too
        cloud.fail = [429]
        assert transcribe(server, "openai-whisper-1").headers["x-bolo-fallback"].endswith("reason=overloaded")
        # speech: cloud -> local, with a voice the local model does not have
        cloud.fail = [503]
        r = speak(server, "openai-tts-1", voice="coral")
        assert r.status_code == 200 and r.headers["x-bolo-model"] == "fake-tts"
        assert r.headers["x-bolo-fallback"] == "openai-tts-1 -> fake-tts; reason=provider_error"
        assert trace_of(server, r)["fallback"]["attempts"][0]["model"] == "openai-tts-1"
        # speech: local -> cloud (the catalog entry's own fallback list)
        r = speak(server, "down-tts", "Hello there.", voice="alloy")
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-fallback"] == "down-tts -> openai-gpt-4o-mini-tts; reason=engine_unavailable"
        assert cloud.last("/v1/audio/speech")["json"]["model"] == "gpt-4o-mini-tts"
        # pinned through the API: wins over the setting; [] switches the fallback off
        put = httpx.put(f"{server.url}/api/v1/models/openai-whisper-1/settings", json={"fallback": []})
        assert put.status_code == 200 and put.json()["settings"] == {"fallback": []}
        cloud.fail = [500]
        assert transcribe(server, "openai-whisper-1").status_code == 502
        bad = httpx.put(f"{server.url}/api/v1/models/openai-whisper-1/settings", json={"fallback": ["fake-tts"]})
        assert bad.status_code == 400 and "cannot be used for stt" in bad.json()["error"]["message"]
        entry = httpx.get(f"{server.url}/api/v1/models/openai-tts-1").json()
        assert entry["fallback"] == ["fake-tts"]
        stats = httpx.get(f"{server.url}/api/v1/stats").json()["fallbacks"]
        assert {"requested": "openai-whisper-1", "answered": "fake-stt", "count": 2} in stats
        assert (
            'bolo_fallbacks_total{requested="openai-tts-1",answered="fake-tts"} 1'
            in httpx.get(f"{server.url}/metrics").text
        )
    finally:
        server.stop()


# -- the providers API --------------------------------------------------------------------------------------------
def test_providers_api(settings: Settings, cloud: FakeProviders) -> None:
    web = httpx.Client(timeout=HTTP_TIMEOUT)  # httpx's default 5 s is too short on a busy machine
    settings.providers = {"sarvam": {"base_url": cloud.url, "api_key": KEY}}
    server = LiveServer(settings)
    server.start()
    api = f"{server.url}/api/v1/providers"
    try:
        listing = web.get(api).json()
        assert listing["editable"] is True and listing["kinds"][0]["kind"] == "openai-compatible"
        assert listing["providers"][0]["source"] == "settings" and listing["providers"][0]["editable"] is False
        assert web.get(f"{server.url}/api/v1/catalog").json()["models"] and listing["notice"].startswith(
            "paid cloud API"
        )
        # add: the key goes in, and never comes back
        r = web.put(f"{api}/openai", json={"base_url": cloud.url + "/v1", "api_key": KEY, "prices": PRICES})
        assert r.status_code == 200, r.text
        assert r.json() == {
            "name": "openai", "kind": "openai-compatible", "base_url": cloud.url + "/v1", "configured": True,
            "api_key_set": True, "options": {}, "prices": PRICES, "source": "stored", "editable": True,
            "terms_url": None,
        }  # fmt: skip
        stored = settings.data_dir / "providers.json"
        assert json.loads(stored.read_text(encoding="utf-8"))["openai"]["api_key"] == KEY
        # owner-only on POSIX; Windows has no mode bits (the file inherits the folder's ACL, the user's profile)
        assert sys.platform == "win32" or stat.S_IMODE(stored.stat().st_mode) == 0o600
        assert KEY not in web.get(api).text and KEY not in web.get(f"{api}/openai").text
        assert web.get(f"{server.url}/api/v1/info").json()["providers"] == [
            {"name": "sarvam", "kind": "sarvam", "configured": True},
            {"name": "openai", "kind": "openai-compatible", "configured": True},
        ]
        # test: one cheap request with the stored key
        test = web.post(f"{api}/openai/test").json()
        assert test["ok"] is True and test["checked"] == "credentials" and "2 models" in test["message"]
        assert cloud.last("/v1/models")["headers"]["Authorization"] == f"Bearer {KEY}"
        assert web.post(f"{api}/sarvam/test").json()["checked"] == "reachability"
        assert transcribe(server, "openai-whisper-1").status_code == 200
        # edit without a key keeps the key; the loaded model is dropped so the change takes effect
        r = web.put(f"{api}/openai", json={"options": {"timeout_s": 30}})
        assert r.json()["api_key_set"] is True and r.json()["options"] == {"timeout_s": 30}
        assert r.json()["base_url"] == cloud.url + "/v1"
        assert web.get(f"{server.url}/api/v1/models/openai-whisper-1").json()["loaded"] is False
        assert transcribe(server, "openai-whisper-1").status_code == 200
        r = web.put(f"{api}/openai", json={"clear_api_key": True})
        assert r.json()["api_key_set"] is False
        assert transcribe(server, "openai-whisper-1").status_code in (200, 502)  # a keyless self-hosted address
        # what the API refuses
        assert web.put(f"{api}/sarvam", json={"api_key": "x"}).status_code == 400  # from the configuration
        assert web.put(f"{api}/new", json={"kind": "nope"}).status_code == 400
        assert web.put(f"{api}/new", json={"kind": "sarvam", "surprise": 1}).status_code == 400
        assert web.delete(f"{api}/sarvam").status_code == 400 and web.delete(f"{api}/ghost").status_code == 404
        assert web.post(f"{api}/ghost/test").status_code == 404
        assert web.delete(f"{api}/openai").status_code == 204
        assert "openai" not in json.loads(stored.read_text(encoding="utf-8"))
        assert web.get(f"{server.url}/api/v1/models/openai-whisper-1").json()["installed"] is False
    finally:
        web.close()
        server.stop()
    # a stored provider is there again after a restart
    settings.providers = {}
    reg = ProviderRegistry(settings)
    reg.put("deepgram", {"api_key": "dg-1"})
    assert ProviderRegistry(settings).get("deepgram").api_key == "dg-1"  # type: ignore[union-attr]


def test_providers_cannot_be_changed_on_an_exposed_server(settings: Settings) -> None:
    from bolo.errors import InvalidRequest

    settings.host = "0.0.0.0"
    with pytest.raises(InvalidRequest, match="non-loopback"):
        ProviderRegistry(settings).put("openai", {"api_key": "k"})
    settings.api_key = "server-key"
    assert ProviderRegistry(settings).put("openai", {"api_key": "k"}).configured
    settings.allow_provider_edit = False
    with pytest.raises(InvalidRequest, match="allow_provider_edit"):
        ProviderRegistry(settings).delete("openai")


def test_adhoc_provider_models(settings: Settings, cloud: FakeProviders) -> None:
    with_providers(settings, cloud)
    settings.providers["lan"] = {
        "kind": "openai-compatible", "base_url": cloud.url + "/v1", "api_key": KEY,
        "options": {"models": {"big-whisper": {"response_format": "verbose_json", "word_timestamps": True}}},
    }  # fmt: skip
    server = LiveServer(settings)
    server.start()
    try:
        r = transcribe(server, "openai:some-new-transcribe", language="en")
        assert r.status_code == 200 and r.json()["text"] == "heard 2.0 seconds in en"
        assert cloud.last("/v1/audio/transcriptions")["form"]["model"] == "some-new-transcribe"
        assert r.headers["x-bolo-model"] == "openai:some-new-transcribe"
        # a provider's own model names may hold colons; options can be set per model on the provider
        r = transcribe(server, "sarvam:saarika:v2.5", language="ta")
        assert r.status_code == 200 and cloud.last("/speech-to-text")["form"]["model"] == "saarika:v2.5"
        r = transcribe(
            server, "lan:big-whisper", response_format="verbose_json", **{"timestamp_granularities[]": "word"}
        )
        assert r.status_code == 200 and len(r.json()["words"]) == 3
        r = speak(server, "openai:tts-next", voice="anything")
        assert r.status_code == 200 and cloud.last("/v1/audio/speech")["json"]["voice"] == "anything"
        entry = httpx.get(f"{server.url}/api/v1/models/openai:tts-next").json()
        assert (
            entry["cloud"] is True and entry["adhoc"] is True and entry["installed"] is True and entry["task"] == "tts"
        )
        assert httpx.post(f"{server.url}/v1/audio/speech", json={"model": "nobody:x", "input": "hi"}).status_code == 404
        r = speak(server, "openai:some-new-transcribe")  # already known as a speech-to-text model
        assert r.status_code == 400 and "is a stt model" in r.json()["error"]["message"]
    finally:
        server.stop()


def test_another_bolo_as_an_openai_compatible_provider(settings: Settings, live: LiveServer, tmp_path: Path) -> None:
    """The openai-compatible adapter against a real implementation of that API: a second bolo."""
    front = dataclasses.replace(
        settings,
        data_dir=tmp_path / "front",
        providers={
            "upstream": {
                "kind": "openai-compatible",
                "base_url": live.url + "/v1",  # no key: a self-hosted server needs none
                "options": {
                    "models": {
                        "fake-stt": {"response_format": "verbose_json", "word_timestamps": True},
                        "fake-tts": {"send_language": True},
                    }
                },
            }
        },
    )
    with another(front, tmp_path) as server:
        r = transcribe(
            server, "upstream:fake-stt", response_format="verbose_json", **{"timestamp_granularities[]": "word"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["text"].startswith("the quick brown fox") and len(r.json()["words"]) >= 4
        assert r.headers["x-bolo-model"] == "upstream:fake-stt"
        r = speak(server, "upstream:fake-tts", "Hello there.", voice="beta", response_format="mulaw")
        assert r.status_code == 200, r.text
        # the upstream's 24 kHz PCM (it says so in x-sample-rate), re-encoded here to 8 kHz mu-law
        assert r.headers["x-sample-rate"] == "8000" and len(r.content) == pytest.approx(12 * 0.06 * 8000, rel=0.2)
        r = speak(server, "upstream:no-such-voice-model", voice="beta")
        assert r.status_code == 502 and "does not know 'no-such-voice-model'" in r.json()["error"]["message"]
    # the upstream bolo saw ordinary requests
    upstream = [t["model"] for t in httpx.get(f"{live.url}/api/v1/requests").json()["requests"]]
    assert "fake-stt" in upstream and "fake-tts" in upstream


# -- bolo's features on top ---------------------------------------------------------------------------------------
def test_diarization_with_a_cloud_recogniser(srv: LiveServer, cloud: FakeProviders) -> None:
    from bolo.audio.encode import encode_audio

    audio = encode_audio(two_speakers(), 16000, "wav")
    r = httpx.post(
        f"{srv.url}/v1/audio/transcriptions",
        data={"model": "openai-gpt-4o-mini-transcribe", "response_format": "diarized_json"},
        files={"file": ("a.wav", audio, "audio/wav")},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    segments = r.json()["segments"]
    assert [s["speaker"] for s in segments] == ["A", "B", "A", "B"]
    assert all(s["text"].startswith("heard 1.") for s in segments)
    # bolo's diarizer cut the recording; each turn went to the provider on its own
    assert cloud.count("/v1/audio/transcriptions") == 4 and trace_of(srv, r)["provider"]["requests"] == 4


async def test_realtime_cascade_with_cloud_stt_and_tts(settings: Settings, cloud: FakeProviders, llm: Any) -> None:
    with_providers(settings, cloud)
    settings.llm_base_url, settings.llm_model = llm.url, "fake-llm"
    settings.default_realtime_stt = settings.default_stt = "openai-gpt-4o-mini-transcribe"
    settings.default_realtime_tts = "openai-gpt-4o-mini-tts"
    server = LiveServer(settings)
    server.start()
    try:
        ws = await connect(server)
        try:
            updated = await update(
                ws,
                {
                    "type": "realtime",
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 300}},
                        "output": {"format": {"type": "audio/pcmu"}, "voice": "alloy"},
                    },
                },
            )
            assert updated["type"] == "session.updated", updated
            for chunk in b64_pcm24(2.0):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            events = await until(ws, "response.done")
        finally:
            await ws.close()
        done = next(e for e in events if e["type"] == "response.done")["response"]
        assert done["status"] == "completed", done
        heard = next(e for e in events if e["type"].endswith("input_audio_transcription.completed"))["transcript"]
        assert heard.startswith("heard 2.")  # the cloud recogniser's words went to the language model
        assert llm.requests[-1]["messages"][-1]["content"] == heard
        mulaw = sum(len(e["delta"]) for e in events if e["type"] == "response.output_audio.delta") * 3 / 4
        assert mulaw > 8000  # the cloud voice's 24 kHz audio, re-encoded to 8 kHz mu-law for the caller
        assert cloud.count("/v1/audio/speech") >= 1 and cloud.last("/v1/audio/speech")["json"]["voice"] == "alloy"
        # one billed request for the turn: no interim decodes of the growing utterance go to a cloud recogniser
        assert cloud.count("/v1/audio/transcriptions") == 1
        trace = httpx.get(server.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
        assert trace["pipeline"] == "cascade" and trace["stt_model"] == "openai-gpt-4o-mini-transcribe"
    finally:
        server.stop()


async def test_cloud_speech_to_speech(settings: Settings, cloud: FakeProviders) -> None:
    upstream = FakeRealtimeServer(api_key=KEY, openai=True).start()
    with_providers(settings, cloud)
    settings.providers["openai"]["base_url"] = upstream.url + "/v1"
    settings.s2s_model = "openai:gpt-realtime-2.1"
    server = LiveServer(settings)
    server.start()
    try:
        caps = httpx.get(f"{server.url}/api/v1/info").json()["capabilities"]["realtime"]["pipelines"]
        assert caps["speech-to-speech"]["configured"] is True and caps["speech-to-speech"]["available"] is True
        ws = await connect(server)
        try:
            updated = await update(
                ws,
                {
                    "type": "realtime", "pipeline": "speech-to-speech", "instructions": "Be brief.",
                    "audio": {"input": {"transcription": {"model": "fake-stt"},
                                        "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}},
                              "output": {"voice": "marin"}},
                },
            )  # fmt: skip
            assert updated["type"] == "session.updated", updated
            for chunk in b64_pcm24(2.0):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            events = await until(ws, "response.done")
        finally:
            await ws.close()
        done = next(e for e in events if e["type"] == "response.done")["response"]
        transcript = done["output"][0]["content"][0]["transcript"]
        # the stand-in measures what it was sent at 24 kHz: bolo resampled the turn's 16 kHz audio
        assert done["status"] == "completed" and 2.0 <= float(transcript.split()[2]) <= 2.8, transcript
        assert any(e["type"] == "response.output_audio.delta" for e in events)
        assert upstream.paths == ["/v1/realtime?model=gpt-realtime-2.1"]
        session = upstream.events[-1][0]["session"]
        assert session["type"] == "realtime" and session["output_modalities"] == ["audio"]
        assert session["audio"]["input"] == {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}
        assert session["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"}
        create = next(e for e in upstream.events[-1] if e["type"] == "response.create")
        assert create["response"]["output_modalities"] == ["audio"]  # never text and audio together
        trace = httpx.get(server.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
        assert trace["pipeline"] == "speech-to-speech" and trace["status"] == "ok"
        described = httpx.get(f"{server.url}/api/v1/info").json()["capabilities"]["realtime"]["pipelines"]
        assert described["speech-to-speech"]["backend"] == "openai-realtime" and KEY not in json.dumps(described)
    finally:
        server.stop()
        upstream.stop()


async def test_pseudo_streaming_does_not_multiply_cloud_requests(srv: LiveServer, cloud: FakeProviders) -> None:
    """A recogniser without a live socket, on the STT WebSocket: the final decode only (each decode is billed),
    unless the model option interim_decodes says otherwise."""

    async def run(model: str) -> list[dict[str, Any]]:
        async with websockets.connect(f"{srv.ws_url}/api/v1/stt/stream?model={model}") as ws:
            ready = json.loads(await ws.recv())
            assert ready["mode"] == "pseudo", ready
            audio = np.concatenate([speechlike(2.0), np.zeros(16000, dtype=np.float32)])
            await stream_audio(ws, pcm16(audio), 3200, delay=0.05)
            await ws.send(json.dumps({"type": "stop"}))
            return [json.loads(m) async for m in ws]

    messages = await run("openai-whisper-1")
    finals = [m for m in messages if m["type"] == "final"]
    assert len(finals) == 1 and finals[0]["text"].startswith("heard ")
    assert not [m for m in messages if m["type"] == "partial"]
    assert cloud.count("/v1/audio/transcriptions") == 1
    srv.core.catalog.resolve("openai-gpt-4o-mini-transcribe").options["interim_decodes"] = True  # type: ignore[union-attr]
    before = cloud.count("/v1/audio/transcriptions")
    messages = await run("openai-gpt-4o-mini-transcribe")
    assert cloud.count("/v1/audio/transcriptions") - before >= 2


# -- what the console asked for -------------------------------------------------------------------------------
def test_console_flags(live: LiveServer, settings: Settings, tmp_path: Path, llm: Any) -> None:
    info = httpx.get(f"{live.url}/api/v1/info").json()
    assert info["auto_pull"] is False and info["providers"] == []
    caps = info["capabilities"]
    assert caps["wordlists"]["endpoint"] == "/api/v1/wordlists" and isinstance(caps["wordlists"]["available"], bool)
    assert caps["wordlists"]["request_fields"] == ["wordlists", "pronunciations"]
    assert caps["providers"]["kinds"] == list(KINDS) and KINDS[:5] == (
        "openai-compatible", "sarvam", "elevenlabs", "deepgram", "soniox",
    )  # fmt: skip
    assert caps["providers"]["editable"] is True and caps["providers"]["configured"] == []
    pipelines = caps["realtime"]["pipelines"]
    assert pipelines["cascade"] == {"configured": False, "available": False}
    assert pipelines["audio-llm"]["configured"] is False and pipelines["audio-llm"]["available"] is False
    s2s = pipelines["speech-to-speech"]
    assert s2s["configured"] is True and s2s["available"] is False and "llama-server" in s2s["reason"]
    with another(settings, tmp_path, llm_base_url=llm.url, llm_model="fake-llm", auto_pull=True) as agent:
        info = httpx.get(f"{agent.url}/api/v1/info").json()
        with_llm = info["capabilities"]["realtime"]["pipelines"]
        assert with_llm["cascade"] == {"configured": True, "available": True}
        assert with_llm["audio-llm"]["available"] is True and info["auto_pull"] is True
    listed = {m["id"]: m for m in httpx.get(f"{live.url}/api/v1/catalog").json()["models"]}
    # every entry the catalog lists runs somewhere: the flag stays in the API for custom entries
    assert listed["diarizen-wavlm-large-s80-md-v2"]["runnable"] is True
    assert listed["fake-stt"]["runnable"] is True and listed["fake-stt"]["not_runnable_reason"] is None
    # entries whose files are not published yet are listed, with their status
    pending = [m for m in listed.values() if m["status"] == "pending"]
    assert pending == [m for m in pending if m["installed"] is False]
    assert len(pending) == len([s for s in Catalog.load().all() if s.status == "pending"])


def test_cli_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    from bolo.cli import main

    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    assert main(["providers", "list"]) == 0
    assert "no cloud providers are configured" in capsys.readouterr().out
    monkeypatch.setenv("BOLO_PROVIDER_OPENAI_API_KEY", KEY)
    monkeypatch.setenv("BOLO_PROVIDER_OPENAI_BASE_URL", cloud.url + "/v1")
    monkeypatch.setenv("BOLO_PROVIDER_ELEVENLABS_API_KEY", "wrong-key-123456")
    monkeypatch.setenv("BOLO_PROVIDER_ELEVENLABS_BASE_URL", cloud.url)
    assert main(["providers", "list"]) == 0
    out = capsys.readouterr().out
    assert "openai" in out and "openai-compatible" in out and "configured" in out and KEY not in out
    assert main(["providers", "test", "openai"]) == 0
    assert "key accepted" in capsys.readouterr().out
    assert main(["providers", "test", "elevenlabs"]) == 1
    err = capsys.readouterr().err
    assert "provider_auth" in err and "wrong-key-123456" not in err
    assert main(["providers", "test", "nobody"]) == 1 and main(["providers", "test"]) == 1
