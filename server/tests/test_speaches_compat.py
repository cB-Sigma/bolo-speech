"""Routes that mirror speaches (shapes taken from its route definitions)."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from bolo.audio.encode import encode_audio
from conftest import LiveServer, wav_file


def voice(freq: float, seconds: float = 2.0, rate: int = 16000) -> bytes:
    """A tone with a slow wobble: the fake diarizer's "voice" is its pitch."""
    t = np.arange(int(seconds * rate)) / rate
    return encode_audio(
        (0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))).astype(np.float32), rate, "wav"
    )


def embedding(live: LiveServer, audio: bytes, **data: str) -> dict:
    r = httpx.post(live.url + "/v1/audio/speech/embedding", files={"file": ("a.wav", audio)}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_registry_and_model_listing(live: LiveServer) -> None:
    registry = httpx.get(live.url + "/v1/registry").json()
    assert registry["object"] == "list" and len(registry["data"]) > 100
    entry = registry["data"][0]
    assert {"id", "object", "created", "owned_by", "language", "task"} <= set(entry) and entry["object"] == "model"
    tts = httpx.get(live.url + "/v1/registry", params={"task": "text-to-speech"}).json()["data"]
    assert tts and all(m["task"] == "text-to-speech" for m in tts)
    asr = httpx.get(live.url + "/v1/models", params={"task": "automatic-speech-recognition"}).json()["data"]
    assert {m["id"] for m in asr} == {"fake-stt", "fake-stt-streaming", "fake-stt-ta"}
    audio_models = httpx.get(live.url + "/v1/audio/models").json()
    assert audio_models["object"] == "list" and [m["id"] for m in audio_models["models"]] == ["fake-tts"]
    assert audio_models["models"][0]["sample_rate"] == 24000
    voices = httpx.get(live.url + "/v1/audio/voices").json()["voices"]
    assert {
        "name": "alpha",
        "id": "alpha",
        "language": "en-US",
        "gender": "female",
        "model_id": "fake-tts",
        "blendable": False,
    } in voices


def test_speaches_model_ids_resolve_to_catalog_entries(live: LiveServer) -> None:
    core = live.core
    assert core.catalog.resolve("Systran/faster-whisper-small").id == "faster-whisper-small"
    assert core.catalog.resolve("Systran/faster-distil-whisper-large-v3").id == "faster-whisper-distil-large-v3"
    assert core.catalog.resolve("speaches-ai/Kokoro-82M-v1.0-ONNX").id == "kokoro-v1.0"
    assert core.catalog.resolve("speaches-ai/piper-en_US-amy-low").id == "piper-en-us-amy-low"
    # speaches' default aliases
    for alias in ("whisper-1", "tts-1", "tts-1-hd"):
        assert core.resolve(alias, "stt" if alias.startswith("whisper") else "tts")


def test_download_and_delete_model_routes(live: LiveServer) -> None:
    r = httpx.post(live.url + "/v1/models/fake-stt")
    assert r.status_code == 201 and "already exists" in r.text  # speaches: 201 when nothing had to be downloaded
    assert httpx.post(live.url + "/v1/models/owner/unknown-model").status_code == 404
    assert httpx.delete(live.url + "/v1/models/owner/unknown-model").status_code == 404


def test_ps_routes(live: LiveServer) -> None:
    assert httpx.get(live.url + "/api/ps").json() == {"models": []}
    r = httpx.post(live.url + "/api/ps/fake-stt")
    assert r.status_code == 201 and r.json() == {"message": "Model 'fake-stt' loaded."}
    assert httpx.post(live.url + "/api/ps/fake-stt").status_code == 409
    assert httpx.get(live.url + "/api/ps").json() == {"models": ["fake-stt"]}
    assert httpx.post(live.url + "/api/ps/nope/nope").status_code == 404
    assert httpx.delete(live.url + "/api/ps/fake-stt").json() == {"message": "Model fake-stt unloaded."}
    assert httpx.delete(live.url + "/api/ps/fake-stt").status_code == 404


def test_vad_timestamps(live: LiveServer) -> None:
    files = {"file": wav_file(6.0, gap_at=(2.0, 4.0))}
    r = httpx.post(live.url + "/v1/audio/speech/timestamps", files=files, data={"min_silence_duration_ms": "300"})
    assert r.status_code == 200
    segments = r.json()
    assert len(segments) == 3 and all(set(s) == {"start", "end"} for s in segments)  # milliseconds, like speaches
    assert segments[0]["start"] < 150 and abs(segments[0]["end"] - 2000) < 250
    assert abs(segments[1]["start"] - 2800) < 250 and abs(segments[2]["end"] - 6000) < 250
    # a speaches client passes its fixed silero model id: accepted and ignored
    r = httpx.post(
        live.url + "/v1/audio/speech/timestamps", files={"file": wav_file(1.0)}, data={"model": "silero_vad_v5"}
    )
    assert r.status_code == 200 and len(r.json()) == 1
    assert httpx.post(live.url + "/v1/audio/speech/timestamps", data={}).status_code == 400


def test_speaches_environment_names_are_honoured() -> None:
    from bolo.config import load_settings

    s = load_settings(
        env={
            "API_KEY": "k",
            "PRELOAD_MODELS": '["a","b"]',
            "STT_MODEL_TTL": "-1",
            "UVICORN_PORT": "9001",
            "ALLOW_ORIGINS": "*",
            "ENABLE_UI": "false",
        }
    )
    assert s.api_key == "k" and s.preload == ["a", "b"] and s.stt_ttl_s == -1 and s.port == 9001
    assert s.cors_origins == ["*"] and s.ui is False
    s = load_settings(env={"API_KEY": "old", "BOLO_API_KEY": "new"})
    assert s.api_key == "new"  # bolo's own names win


def test_speaker_embedding_shape_and_default_model(live: LiveServer) -> None:
    doc = embedding(live, voice(150))
    assert doc["object"] == "list" and doc["model"] == "fake-diarization"  # voice cloning's embedder by default
    (item,) = doc["data"]
    assert item["object"] == "embedding" and item["index"] == 0
    vector = np.array(item["embedding"])
    assert vector.shape == (2,) and np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    assert doc["usage"] == {"prompt_tokens": 32000, "total_tokens": 32000}  # 2 s at 16 kHz, as speaches counts
    # speaches' own embedding model id is not in the catalog: the default is used; a catalog id is honoured
    assert embedding(live, voice(150), model="pyannote/wespeaker-voxceleb-resnet34-LM")["model"] == "fake-diarization"
    assert embedding(live, voice(150), model="fake-diarization")["model"] == "fake-diarization"
    trace = httpx.get(live.url + "/api/v1/requests", params={"kind": "embedding"}).json()
    assert trace and "embed" in [p["name"] for p in (trace.get("requests") or trace)[0]["phases"]]


def test_speaker_embedding_is_deterministic_and_tells_speakers_apart(live: LiveServer) -> None:
    def vec(audio: bytes) -> np.ndarray:
        return np.array(embedding(live, audio)["data"][0]["embedding"])

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    first = vec(voice(130))
    assert np.array_equal(first, vec(voice(130)))
    same = cos(first, vec(voice(140, seconds=3.0, rate=24000)))  # another clip of the same "speaker"
    different = cos(first, vec(voice(420)))
    assert same > 0.99 and different < 0.5


def test_speaker_embedding_errors(live: LiveServer) -> None:
    url = live.url + "/v1/audio/speech/embedding"
    assert httpx.post(url, data={"model": "fake-diarization"}).status_code == 400
    r = httpx.post(url, files={"file": ("a.wav", voice(150, seconds=0.2))})
    assert r.status_code == 400 and "half a second" in r.json()["error"]["message"]
    r = httpx.post(url, files={"file": ("a.wav", voice(150))}, data={"model": "fake-stt"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "model"
