"""Voice blends: parsing and checking, mixing style tables, speech with a blend, saved blends in the voice library."""

from __future__ import annotations

import io
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bolo.api import create_app
from bolo.audio import decode_audio
from bolo.config import Settings
from bolo.engines.base import Voice
from bolo.engines.blend import (
    UnknownVoice,
    looks_like_blend,
    mix_styles,
    near_matches,
    parse_blend,
    resolve_blend,
)
from bolo.errors import InvalidRequest
from bolo.voices import VoiceStore

VOICES = (
    Voice("af_bella", "Bella", "en-US", "f", 2),
    Voice("af_sky", "Sky", "en-US", "f", 7),
    Voice("am_adam", "Adam", "en-US", "m", 11),
    Voice("hf_alpha", "Alpha", "hi-IN", "f", 30),
)

#: the fake engine's voice with option blending (a custom entry, so that the fake catalog other tests count stays)
BLEND_MODEL = """\
- id: fake-tts-blend
  task: tts
  engine: fake
  kind: fake-tts
  family: Fake
  languages: [en, hi]
  tier: realtime
  sample_rate: 24000
  license: Apache-2.0
  builtin: true
  features: [voice-blending]
  options: {blending: true}
  voices:
  - {id: alpha, name: Alpha, language: en-US, gender: f, sid: 0}
  - {id: beta, name: Beta, language: en-GB, gender: m, sid: 1}
  - {id: gamma, name: Gamma, language: hi-IN, gender: f, sid: 2}
  description: Deterministic test voice that blends voices by pitch.
"""


@pytest.fixture
def client(settings: Settings) -> TestClient:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "custom.yaml").write_text(BLEND_MODEL)
    with TestClient(create_app(settings)) as c:
        yield c  # type: ignore[misc]


def pitch(wav: bytes) -> float:
    x, rate = decode_audio(wav)
    return float(np.argmax(np.abs(np.fft.rfft(x))) * rate / len(x))


# -- parsing and checking ----------------------------------------------------------------------------------------


def test_parse_and_normalise() -> None:
    assert parse_blend("af_bella+af_sky") == [("af_bella", 1.0), ("af_sky", 1.0)]
    assert parse_blend(" af_bella (0.7) + af_sky(0.3) ") == [("af_bella", 0.7), ("af_sky", 0.3)]
    equal = resolve_blend("af_bella+af_sky", VOICES)
    assert equal.key == "af_bella(0.5)+af_sky(0.5)"
    weighted = resolve_blend("af_bella(0.7)+af_sky(0.3)", VOICES)
    assert weighted.key == "af_bella(0.7)+af_sky(0.3)" and weighted.lead.id == "af_bella"
    # weights are relative; a voice without one counts 1; display names and any case work; repeats add up
    assert resolve_blend("Bella(2)+AF_SKY", VOICES).key == "af_bella(0.667)+af_sky(0.333)"
    assert resolve_blend("af_sky(1)+af_bella(1)+af_sky(2)", VOICES).key == "af_sky(0.75)+af_bella(0.25)"
    # the language of the heaviest voice reads the text (the first on a tie)
    assert resolve_blend("af_bella(0.4)+hf_alpha(0.6)", VOICES).lead.id == "hf_alpha"
    assert resolve_blend("hf_alpha+af_bella", VOICES).lead.id == "hf_alpha"
    assert resolve_blend("af_bella+af_sky", VOICES).to_dict()["voices"] == [
        {"voice": "af_bella", "weight": 0.5},
        {"voice": "af_sky", "weight": 0.5},
    ]
    assert looks_like_blend("a+b") and looks_like_blend("a(1)") and not looks_like_blend("af_bella")


@pytest.mark.parametrize("bad", ["af_bella+", "af_bella(x)+af_sky", "af_bella(0)+af_sky", "af_bella(-1)", "a(b)c"])
def test_malformed_blends_are_refused(bad: str) -> None:
    with pytest.raises(InvalidRequest):
        resolve_blend(bad, VOICES)


def test_unknown_voices_name_near_matches() -> None:
    with pytest.raises(UnknownVoice) as err:
        resolve_blend("af_bela+af_sky", VOICES, "kokoro")
    assert err.value.status == 400 and err.value.param == "voice"
    assert "'af_bela'" in err.value.message and "Did you mean: af_bella" in err.value.message
    assert near_matches("adam", VOICES) == ["am_adam"]
    with pytest.raises(InvalidRequest, match="at most 8"):
        resolve_blend("+".join(["af_bella"] * 9), VOICES)


def test_mix_styles_is_the_weighted_sum() -> None:
    table = np.zeros((40, 3, 1, 4), dtype=np.float32)
    table[2] = 1.0
    table[7] = 3.0
    mixed = mix_styles(table, resolve_blend("af_bella(3)+af_sky(1)", VOICES))
    assert mixed.shape == (3, 1, 4) and mixed.dtype == np.float32
    assert np.allclose(mixed, 0.75 * 1.0 + 0.25 * 3.0)


def test_sherpa_blend_instance_keeps_blends_in_slots(tmp_path: Path) -> None:
    """sherpa-onnx takes voices from a file of exactly the model's voice count: blends go into a second instance's
    file, rebuilt only for a new blend, the least recently used blend making room."""
    from bolo.engines.sherpa import SherpaTTS

    rows = 3
    table = np.arange(rows * 2 * 4, dtype=np.float32).reshape(rows, 2, 1, 4)
    voices_file = tmp_path / "voices.bin"
    table.tofile(voices_file)
    built: list[np.ndarray] = []

    def build(parts: dict[str, Any]) -> object:
        built.append(np.fromfile(parts["kokoro"].voices, dtype=np.float32).reshape(rows, -1))
        return SimpleNamespace(n=len(built))

    tts = SherpaTTS.__new__(SherpaTTS)
    tts.num_speakers = rows
    tts._so = SimpleNamespace(OfflineTtsKokoroModelConfig=SimpleNamespace)
    tts._cfg = {"kokoro": SimpleNamespace(model="m.onnx", voices=str(voices_file), tokens="t.txt", lexicon="")}
    tts._build = build
    tts._blend_tts = None
    tts._blend_lock = threading.Lock()
    tts._blends = OrderedDict()
    tts._blend_slots = {}
    voices = (Voice("a", "", "en-US", "", 0), Voice("b", "", "en-US", "", 1), Voice("c", "", "en-US", "", 2))
    ab = resolve_blend("a+b", voices)
    inst, sid = tts._blend_voice(ab)
    assert sid == 0 and len(built) == 1
    assert np.allclose(built[0][0], (table[0] + table[1]).reshape(-1) / 2)
    assert np.allclose(built[0][1:], table[1:].reshape(rows - 1, -1))  # unused rows keep the model's voices
    assert tts._blend_voice(ab) == (inst, 0) and len(built) == 1  # known blend: no rebuild
    tts._blend_voice(resolve_blend("a(3)+c", voices))
    tts._blend_voice(resolve_blend("b+c", voices))
    assert len(built) == 3 and sorted(tts._blend_slots.values()) == [0, 1, 2]
    tts._blend_voice(ab)  # used again: now the most recent
    _inst, sid = tts._blend_voice(resolve_blend("a(1)+b(3)", voices))  # a fourth: "a(3)+c" (oldest) makes room
    assert sid == 1 and "a(0.75)+c(0.25)" not in tts._blend_slots and len(tts._blend_slots) == 3
    tts.close()
    assert tts._blend_tts is None


# -- speech -------------------------------------------------------------------------------------------------------


def test_speech_with_a_blend(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts-blend", "input": "Hello there.", "voice": "alpha+gamma", "response_format": "wav"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-blend"] == "alpha(0.5)+gamma(0.5)" and "x-bolo-cloned" not in r.headers
    # the fake model speaks at the weighted mean of the voices' pitches (220 and 330 Hz)
    assert abs(pitch(r.content) - 275) < 10
    r = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts-blend", "input": "Hello.", "voice": "alpha(0.8)+gamma(0.2)", "response_format": "wav"},
    )
    assert r.headers["x-bolo-blend"] == "alpha(0.8)+gamma(0.2)" and abs(pitch(r.content) - 242) < 10
    core = client.app.state.core  # type: ignore[attr-defined]
    trace = core.traces.recent(1)[0]
    assert trace.extra["voice"]["blend"] == "alpha(0.8)+gamma(0.2)" and trace.language == "en-US"
    # a single voice still works as before
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-blend", "input": "Hi.", "voice": "beta"})
    assert r.status_code == 200 and "x-bolo-blend" not in r.headers


def test_blend_errors(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-blend", "input": "Hi.", "voice": "alpha+betta"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "voice" and "'betta'" in err["message"] and "Did you mean: beta" in err["message"]
    r = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "Hi.", "voice": "alpha+beta"})
    assert r.status_code == 400
    assert "cannot blend voices" in r.json()["error"]["message"]
    assert "fake-tts-blend" in r.json()["error"]["message"]
    # an unknown single voice names its near matches too
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-blend", "input": "Hi.", "voice": "gama"})
    assert r.status_code == 400 and "Did you mean: gamma" in r.json()["error"]["message"]


def test_voices_endpoint_marks_blendable_voices(client: TestClient) -> None:
    voices = client.get("/v1/audio/voices", params={"model": "fake-tts-blend"}).json()["voices"]
    assert {v["id"] for v in voices} == {"alpha", "beta", "gamma"} and all(v["blendable"] for v in voices)
    plain = client.get("/v1/audio/voices", params={"model": "fake-tts"}).json()["voices"]
    assert not any(v["blendable"] for v in plain)
    model = client.get("/api/v1/models/fake-tts-blend").json()
    assert "voice-blending" in model["features"]


# -- saved blends ---------------------------------------------------------------------------------------------------


def test_saved_blends(client: TestClient) -> None:
    r = client.post(
        "/api/v1/voices/blends",
        json={"name": "Warm Mix", "blend": "alpha(0.7)+gamma(0.3)", "model": "fake-tts-blend", "description": "x"},
    )
    assert r.status_code == 201, r.text
    saved = r.json()
    assert saved["id"] == "warm-mix" and saved["kind"] == "blend" and saved["blend"] == "alpha(0.7)+gamma(0.3)"
    assert saved["default_model"] == "fake-tts-blend" and saved["language"] == "en-US"
    assert saved["audio_url"] is None and "consent" not in saved  # a blend imitates no one: no consent needed
    listing = client.get("/api/v1/voices").json()
    assert listing["voices"] == [] and [b["id"] for b in listing["blends"]] == ["warm-mix"]
    assert listing["blending_models"] == ["fake-tts-blend"]
    # spoken by naming it; without a model, the blend's model answers
    r = client.post("/v1/audio/speech", json={"input": "Hello.", "voice": "warm-mix", "response_format": "wav"})
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-model"] == "fake-tts-blend" and r.headers["x-bolo-voice"] == "warm-mix"
    assert r.headers["x-bolo-blend"] == "alpha(0.7)+gamma(0.3)" and "x-bolo-cloned" not in r.headers
    assert abs(pitch(r.content) - 253) < 10
    # with a model that cannot blend: 400 naming the models that can
    r = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "Hi.", "voice": "warm-mix"})
    assert r.status_code == 400 and "warm-mix" in r.json()["error"]["message"]
    # listed with its model by /v1/audio/voices
    voices = client.get("/v1/audio/voices", params={"model": "fake-tts-blend"}).json()["voices"]
    assert {"id": "warm-mix", "blend": "alpha(0.7)+gamma(0.3)"}.items() <= next(
        v for v in voices if v["id"] == "warm-mix"
    ).items()
    # no recording: no audio, no similarity score
    assert client.get("/api/v1/voices/warm-mix/audio").status_code == 400
    assert client.get("/api/v1/voices/warm-mix/score").status_code == 400
    # changed: checked against the model's voices again
    r = client.patch("/api/v1/voices/warm-mix", json={"blend": "beta+gamma"})
    assert r.status_code == 200 and r.json()["blend"] == "beta(0.5)+gamma(0.5)"
    assert r.json()["voices"] == [{"voice": "beta", "weight": 0.5}, {"voice": "gamma", "weight": 0.5}]
    assert client.patch("/api/v1/voices/warm-mix", json={"blend": "beta+delta"}).status_code == 400
    assert client.patch("/api/v1/voices/warm-mix", json={"transcript": "x"}).status_code == 400
    # refused: unknown voices, a model that cannot blend, an id taken
    r = client.post("/api/v1/voices/blends", json={"name": "B", "blend": "alpa+beta", "model": "fake-tts-blend"})
    assert r.status_code == 400 and "Did you mean: alpha" in r.json()["error"]["message"]
    r = client.post("/api/v1/voices/blends", json={"name": "B", "blend": "alpha+beta", "model": "fake-tts"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "model"
    r = client.post(
        "/api/v1/voices/blends", json={"name": "Warm Mix", "blend": "alpha+beta", "model": "fake-tts-blend"}
    )
    assert r.status_code == 409
    assert client.delete("/api/v1/voices/warm-mix").json()["deleted"] is True
    assert client.get("/api/v1/voices").json()["blends"] == []


def test_blends_in_the_store_export_and_import(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "a")
    store.create_blend(
        name="Mix", blend="alpha(0.5)+beta(0.5)", voices=[{"voice": "alpha", "weight": 0.5}], model="kokoro",
        created_by={"auth": "none"},
    )  # fmt: skip
    assert store.all() == [] and [v.id for v in store.blends()] == ["mix"] and store.find("MIX") is not None
    with pytest.raises(InvalidRequest):
        store.get("mix").reference_bytes()
    data = store.export_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert sorted(zf.namelist()) == ["manifest.json", "mix/voice.json"]
    other = VoiceStore(tmp_path / "b")
    assert other.import_zip(data, imported_by={"auth": "none"}) == ["mix"]  # no consent asked for a blend
    assert other.get("mix").is_blend and other.get("mix").blend == "alpha(0.5)+beta(0.5)"
