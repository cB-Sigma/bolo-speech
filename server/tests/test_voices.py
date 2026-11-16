"""Saved voices: the store, cleaning, consent, speech with a saved voice, similarity, OpenAI's routes and the CLI."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bolo import similarity
from bolo.api import create_app
from bolo.audio import decode_audio
from bolo.audio.encode import encode_audio
from bolo.bench import run_bench
from bolo.cli import main as cli
from bolo.config import Settings
from bolo.core import Bolo
from bolo.errors import InvalidRequest
from bolo.voices import (
    ConsentRequired,
    VoiceExists,
    VoiceNotFound,
    VoiceStore,
    clean_reference,
    consistency,
    parse_consent,
    slug,
)
from conftest import LiveServer, speechlike


def tone_wav(
    seconds: float = 8.0, freq: float = 220.0, rate: int = 16000, gap_at: tuple[float, ...] = (2.5, 5.5)
) -> bytes:
    """A modulated tone with pauses (a steady sound without any is reported as noisy, like real noise)."""
    t = np.arange(int(seconds * rate)) / rate
    x = 0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    for g in gap_at:
        x[int(g * rate) : int((g + 0.8) * rate)] = 0.0
    return encode_audio(x.astype(np.float32), rate, "wav")


#: the fake engine's voice with option cloning: it "clones" a reference by speaking at its pitch (a custom entry, so
#: that the fake catalog other tests count stays as it is)
CLONE_MODEL = """\
- id: fake-tts-clone
  task: tts
  engine: fake
  kind: fake-tts
  family: Fake
  languages: [en]
  tier: realtime
  sample_rate: 24000
  license: Apache-2.0
  builtin: true
  features: [voice-cloning]
  options: {cloning: true}
  voices: [{id: alpha, name: Alpha, language: en-US, gender: f}]
  description: Deterministic test voice that clones a reference by its pitch.
"""


def add_clone_model(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "custom.yaml").write_text(CLONE_MODEL)


@pytest.fixture(autouse=True)
def clone_model(settings: Settings) -> None:
    add_clone_model(settings.data_dir)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings)) as c:
        yield c  # type: ignore[misc]


def create(client: TestClient, name: str = "Priya", wav: bytes | None = None, **data: str) -> dict:
    form = {"name": name, "consent": "self", **data}
    r = client.post("/api/v1/voices", data=form, files={"file": ("ref.wav", wav or tone_wav(), "audio/wav")})
    assert r.status_code == 201, r.text
    return r.json()


# -- cleaning and the store ---------------------------------------------------------------------------------------
def test_cleaning_trims_normalises_and_warns() -> None:
    rate = 16000
    quiet = np.zeros(rate * 2, dtype=np.float32)
    voice = 0.05 * speechlike(6.0, rate, gap_at=(2.5,))
    noise = (np.random.default_rng(0).standard_normal(rate * 2) * 1e-4).astype(np.float32)
    cleaned = clean_reference([(np.concatenate([quiet + noise, voice, quiet + noise]), rate)])
    assert cleaned.sample_rate == 24000
    assert 5.5 <= cleaned.analysis["duration_s"] <= 6.0  # 2 s of silence each side cut to 0.15 s, a pause to 0.4 s
    assert cleaned.analysis["gain_db"] > 10  # a quiet recording brought up to about -20 dBFS
    assert np.max(np.abs(cleaned.samples)) <= 0.9
    assert cleaned.warnings == []
    # short: accepted with a warning; too short: refused; nothing but silence: refused
    short = clean_reference([(speechlike(3.0, rate), rate)])
    assert any("short" in w for w in short.warnings)
    assert any("noisy" in w for w in short.warnings)  # one steady sound throughout: no background to compare
    with pytest.raises(InvalidRequest, match=r"only 1\.0 s of speech"):
        clean_reference([(speechlike(1.0, rate), rate)])
    with pytest.raises(InvalidRequest, match="no speech"):
        clean_reference([(np.zeros(rate * 3, dtype=np.float32), rate)])
    # noisy and clipped recordings earn their warnings
    loud_noise = (np.random.default_rng(1).standard_normal(rate * 8) * 0.08).astype(np.float32)
    noisy = speechlike(8.0, rate, gap_at=(2.0, 5.0)) + loud_noise
    assert any("noisy" in w for w in clean_reference([(noisy, rate)]).warnings)
    clipped = np.clip(speechlike(6.0, rate, gap_at=(3.0,)) * 4, -1, 1)
    assert any("clipped" in w for w in clean_reference([(clipped, rate)]).warnings)
    # long pauses are shortened, several clips joined, and a minute is the most that is kept
    paused = clean_reference([(speechlike(8.0, rate, gap_at=(2.0,)), rate)])
    assert paused.analysis["duration_s"] < 8.0
    two = clean_reference([(speechlike(4.0, rate), rate), (speechlike(4.0, rate), 16000)])
    assert two.analysis["clips"] == 2 and 8.0 <= two.analysis["duration_s"] <= 8.8  # joined with a 0.3 s pause
    long = clean_reference([(speechlike(70.0, rate, gap_at=(55.0,)), rate)])
    assert long.analysis["duration_s"] <= 60.0 and any("longer than 60 s" in w for w in long.warnings)


def test_the_noise_warning_spares_clean_synthetic_speech() -> None:
    """A clean synthetic reference, whose DC offset comes and goes with the voice (as in bolo's own sample clips),
    was reported as noisy: removing one mean for the whole clip turned its digital silence into a "background" 27
    dB under the speech. Real noise 10 dB under the speech still earns the warning."""
    rate = 16000
    t = np.arange(rate * 8) / rate
    voice = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    voice += 0.25 * np.sin(2 * np.pi * 540 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    words = (np.sin(2 * np.pi * 1.3 * t) > -0.3).astype(np.float64)  # words and short pauses between them
    for start in (2.0, 5.0):
        words[int(start * rate) : int((start + 0.5) * rate)] = 0.0
    clean = ((voice + 0.06) * words).astype(np.float32)  # an offset only while speaking, digital silence between
    cleaned = clean_reference([(clean, rate)])
    assert not any("noisy" in w for w in cleaned.warnings), cleaned.analysis
    assert cleaned.analysis["snr_db"] > 30
    speech_rms = float(np.sqrt(np.mean(clean[words > 0] ** 2)))
    noise = np.random.default_rng(3).standard_normal(len(clean)) * speech_rms * 10 ** (-10 / 20)
    noisy = clean_reference([((clean + noise).astype(np.float32), rate)])
    assert any("noisy" in w for w in noisy.warnings) and noisy.analysis["snr_db"] < 16
    quieter = np.random.default_rng(4).standard_normal(len(clean)) * speech_rms * 10 ** (-30 / 20)
    assert not any("noisy" in w for w in clean_reference([((clean + quieter).astype(np.float32), rate)]).warnings)


def test_bundled_reference_clips_are_not_called_noisy() -> None:
    from bolo import samples

    entry = samples.get("en-long")
    audio, rate = decode_audio(samples.read_file(str(entry["file"])), filename="en-long.wav")
    assert not any("noisy" in w for w in clean_reference([(audio, rate)]).warnings)


def test_consent_is_required_and_recorded(tmp_path: Path) -> None:
    assert parse_consent("self") == "self" and parse_consent("Permission") == "permission"
    for missing in (None, "", "yes", "true", False, "maybe"):
        with pytest.raises(ConsentRequired):
            parse_consent(missing)
    store = VoiceStore(tmp_path)
    cleaned = clean_reference([(speechlike(6.0), 16000)])
    with pytest.raises(ConsentRequired):
        store.create(name="Asha", cleaned=cleaned, consent="", created_by={})
    voice = store.create(name="Asha K.", cleaned=cleaned, consent="permission", created_by={"name": "ops"})
    assert voice.id == "asha-k" and slug("Ásha  K!") == "asha-k"
    consent = voice.meta["consent"]
    assert consent["basis"] == "permission" and consent["by"] == {"name": "ops"} and consent["at"]
    assert "permission" in consent["statement"]
    with pytest.raises(VoiceExists):
        store.create(name="Asha K.", cleaned=cleaned, consent="self", created_by={})
    with pytest.raises(InvalidRequest, match="OpenAI's voice names"):
        store.create(name="Nova", cleaned=cleaned, consent="self", created_by={})
    assert store.find("ASHA-K") is not None and store.find("../etc") is None and store.find("nobody") is None
    store.update("asha-k", {"description": "warm", "language": "hi"})
    assert store.get("asha-k").meta["description"] == "warm"
    store.delete("asha-k")
    with pytest.raises(VoiceNotFound):
        store.get("asha-k")


def test_export_and_import(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "a")
    cleaned = clean_reference([(speechlike(6.0), 16000)])
    store.create(name="One", cleaned=cleaned, consent="self", created_by={}, transcript="hello")
    store.create(name="Two", cleaned=cleaned, consent="permission", created_by={})
    data = store.export_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert set(zf.namelist()) == {
            "manifest.json", "one/voice.json", "one/reference.wav", "two/voice.json", "two/reference.wav"
        }  # fmt: skip
    other = VoiceStore(tmp_path / "b")
    assert sorted(other.import_zip(data, imported_by={"name": "me"})) == ["one", "two"]
    assert other.get("one").transcript == "hello" and other.get("two").meta["consent"]["basis"] == "permission"
    assert other.import_zip(data, imported_by={}) == []  # existing ids are skipped unless replace
    assert len(other.import_zip(data, imported_by={}, replace=True)) == 2
    # an export whose voices carry no consent record needs one at import
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(buf, "w") as dst:
        for name in src.namelist():
            body = src.read(name)
            if name == "one/voice.json":
                meta = json.loads(body)
                meta.pop("consent")
                body = json.dumps(meta).encode()
            dst.writestr(name, body)
    third = VoiceStore(tmp_path / "c")
    with pytest.raises(ConsentRequired, match="'one'"):
        third.import_zip(buf.getvalue(), imported_by={})
    assert "one" in third.import_zip(buf.getvalue(), imported_by={}, consent="self")
    assert third.get("one").meta["consent"]["basis"] == "self"
    with pytest.raises(InvalidRequest, match=r"not a \.zip"):
        third.import_zip(b"nope", imported_by={})


# -- HTTP ---------------------------------------------------------------------------------------------------------
def test_voice_routes(client: TestClient) -> None:
    r = client.post("/api/v1/voices", data={"name": "Priya"}, files={"file": ("a.wav", tone_wav(), "audio/wav")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "consent_required"
    r = client.post("/api/v1/voices", data={"name": "Priya", "consent": "sure"}, files={"file": ("a.wav", b"x")})
    assert r.status_code == 400 and r.json()["error"]["param"] == "consent"
    voice = create(client, transcript="the words", language="en", description="calm", created_by="asha")
    assert voice["id"] == "priya" and voice["consent"]["by"]["name"] == "asha"
    assert voice["reference"]["sample_rate"] == 24000 and voice["warnings"] == []
    assert voice["reference"]["speaker_consistency"] == pytest.approx(1.0)  # the fake embedder: one pitch
    listing = client.get("/api/v1/voices").json()
    assert [v["id"] for v in listing["voices"]] == ["priya"] and "fake-tts-clone" in listing["cloning_models"]
    assert listing["similarity"]["same_speaker"] == similarity.SAME_SPEAKER
    assert client.get("/api/v1/voices/priya").json()["transcript"] == "the words"
    audio = client.get("/api/v1/voices/priya/audio")
    assert audio.headers["content-type"] == "audio/wav" and decode_audio(audio.content)[1] == 24000
    r = client.patch("/api/v1/voices/priya", json={"default_model": "fake-tts"})
    assert r.status_code == 400 and "cannot clone" in r.json()["error"]["message"]
    r = client.patch("/api/v1/voices/priya", json={"default_model": "fake-tts-clone", "description": "warm"})
    assert r.json()["default_model"] == "fake-tts-clone" and r.json()["description"] == "warm"
    assert client.post("/api/v1/voices", data={"name": "Priya", "consent": "self"},
                       files={"file": ("a.wav", tone_wav())}).status_code == 409  # fmt: skip
    assert client.get("/api/v1/voices/nobody").status_code == 404
    export = client.get("/api/v1/voices/export")
    assert export.headers["content-type"] == "application/zip"
    assert client.delete("/api/v1/voices/priya").json()["deleted"] is True
    assert client.get("/api/v1/voices").json()["voices"] == []
    r = client.post("/api/v1/voices/import", files={"file": ("v.zip", export.content)})
    assert r.json() == {"imported": ["priya"]}


def test_two_people_in_one_reference_are_flagged(client: TestClient) -> None:
    rate = 16000
    # the fake embedder hears 110 Hz and 490 Hz as two orthogonal "voices"; the second speaks for the last 3 s
    a = decode_audio(tone_wav(9.0, 110.0, gap_at=(4.0,)))[0]
    b = decode_audio(tone_wav(3.0, 490.0, gap_at=()))[0]
    voice = create(client, "Mixed", encode_audio(np.concatenate([a, b]), rate, "wav"))
    assert any("different people" in w for w in voice["warnings"])
    assert voice["reference"]["speaker_consistency"] < 0.4
    assert consistency([np.array([1.0, 0.0])]) is None


def test_speech_with_a_saved_voice(client: TestClient, settings: Settings) -> None:
    create(client, transcript="what the reference says")
    r = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts-clone", "input": "Hello there.", "voice": "priya", "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.headers["x-bolo-voice"] == "priya" and r.headers["x-bolo-cloned"] == "true"
    # the reference's pitch came through: the fake model speaks at it
    x, rate = decode_audio(r.content)
    assert abs(np.argmax(np.abs(np.fft.rfft(x))) * rate / len(x) - 220) < 10
    core: Bolo = client.app.state.core  # type: ignore[attr-defined]
    trace = core.traces.recent(1)[0]
    assert trace.extra["voice"] == {"saved": "priya", "cloned": True, "consent": "self"}
    model = core.manager.loaded["fake-tts-clone"].model
    assert model.extras[-1].reference_text == "what the reference says"  # type: ignore[attr-defined]
    # a model voice of the same name wins over the library, unless the request sends OpenAI's {"id": ...}
    create(client, "Alpha", tone_wav(freq=300.0))
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-clone", "input": "Hi.", "voice": "alpha"})
    assert "x-bolo-voice" not in r.headers
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-clone", "input": "Hi.", "voice": {"id": "alpha"}})
    assert r.headers["x-bolo-voice"] == "alpha"
    # a model that cannot clone answers 400 with the cloning models that are installed
    r = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "Hi.", "voice": "priya"})
    assert r.status_code == 400
    assert "saved voice" in r.json()["error"]["message"] and "fake-tts-clone" in r.json()["error"]["message"]
    # without a model, the saved voice's default model answers (else the default model, which cannot clone)
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "priya"})
    assert r.status_code == 400
    client.patch("/api/v1/voices/priya", json={"default_model": "fake-tts-clone"})
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "priya"})
    assert r.status_code == 200 and r.headers["x-bolo-model"] == "fake-tts-clone"
    # raw reference audio is marked as cloned too, without a voice name
    import base64

    ref = base64.b64encode(tone_wav(3.0)).decode()
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-clone", "input": "Hi.", "reference_audio": ref})
    assert r.headers["x-bolo-cloned"] == "true" and "x-bolo-voice" not in r.headers
    # model options: read by the model, refused when it does not know them
    r = client.post(
        "/v1/audio/speech", json={"model": "fake-tts-clone", "input": "Hi.", "voice": "priya", "options": {"x": 1}}
    )
    assert r.status_code == 400 and "does not take the option(s) x" in r.json()["error"]["message"]
    r = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "Hi.", "options": {"pitch": 300}})
    assert r.status_code == 400 and r.json()["error"]["param"] == "options"
    r = client.post("/v1/audio/speech", json={"model": "fake-tts-clone", "input": "Hi.", "voice": {"name": "x"}})
    assert r.status_code == 400


def test_similarity_score(client: TestClient) -> None:
    create(client)
    r = client.get("/api/v1/voices/priya/score", params={"model": "fake-tts-clone", "audio": "true"})
    body = r.json()
    assert body["similarity"] == pytest.approx(1.0) and body["verdict"] == "same speaker"
    assert body["embedding_model"] == "fake-diarization" and body["calibrated"] is False
    assert body["audio"].startswith("data:audio/wav;base64,") and body["rtf"] is not None
    assert client.get("/api/v1/voices/priya").json()["scores"]["fake-tts-clone"]["similarity"] == pytest.approx(1.0)
    r = client.get("/api/v1/voices/priya/score", params={"model": "fake-tts"})
    assert r.status_code == 400
    # scored at creation with score_model
    voice = create(client, "Low", tone_wav(freq=150.0), score_model="fake-tts-clone")
    assert voice["score"]["model"] == "fake-tts-clone" and voice["score"]["similarity"] > 0.9
    failed = create(client, "Other", score_model="fake-tts")
    assert "cannot clone" in failed["score"]["error"]


def test_thresholds() -> None:
    assert similarity.verdict(0.8) == "same speaker" and similarity.verdict(0.35) == "similar"
    assert similarity.verdict(0.1) == "different" and similarity.verdict(None) is None
    assert similarity.cosine(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == 0.0
    assert similarity.cosine(np.zeros(2), np.ones(2)) == 0.0
    cal = similarity.CALIBRATION
    assert cal["same_speaker"]["min"] < similarity.SAME_SPEAKER <= cal["same_speaker"]["median"]
    assert cal["different_speaker"]["p95"] < similarity.SAME_SPEAKER
    assert similarity.test_sentence("de-DE").startswith("Der") and similarity.test_sentence("xx").startswith("The")


async def test_bench_reports_speaker_similarity(core: Bolo) -> None:
    core.voices.create(name="Bench", cleaned=clean_reference([(speechlike(6.0), 16000)]), consent="self", created_by={})
    result = await run_bench(core, kind="tts", model="fake-tts-clone", concurrency=[1], duration_s=0.3, voice="bench")
    sim = result["speaker_similarity"]
    assert sim["similarity"] == pytest.approx(1.0) and sim["reference"] == "saved voice bench"
    plain = await run_bench(core, kind="tts", model="fake-tts", concurrency=[1], duration_s=0.3)
    assert "speaker_similarity" not in plain
    none = await run_bench(core, kind="tts", model="fake-tts-clone", concurrency=[1], duration_s=0.3)
    assert none["speaker_similarity"]["similarity"] is None


# -- OpenAI's custom voices ---------------------------------------------------------------------------------------
def test_openai_sdk_custom_voices(live: LiveServer) -> None:
    from openai import BadRequestError, OpenAI

    client = OpenAI(base_url=live.url + "/v1", api_key="x")
    consent = client.audio.voice_consents.create(
        name="Asha's consent", language="en-US", recording=("consent.wav", tone_wav(3.0), "audio/wav")
    )
    assert consent.id.startswith("cons_") and consent.object == "audio.voice_consent"
    assert [c.id for c in client.audio.voice_consents.list()] == [consent.id]
    assert client.audio.voice_consents.retrieve(consent.id).name == "Asha's consent"
    assert client.audio.voice_consents.update(consent.id, name="Asha").name == "Asha"
    voice = client.audio.voices.create(
        name="Asha", consent=consent.id, audio_sample=("asha.wav", tone_wav(), "audio/wav")
    )
    assert voice.id == "asha" and voice.object == "audio.voice" and voice.type == "audio_sample"
    stored = live.core.voices.get("asha").meta["consent"]
    assert stored["basis"] == "recording" and stored["recording"] == consent.id
    with client.audio.speech.with_streaming_response.create(
        model="fake-tts-clone", voice={"id": "asha"}, input="Hello.", response_format="wav"
    ) as resp:
        assert resp.headers["x-bolo-voice"] == "asha" and len(resp.read()) > 1000
    with pytest.raises(BadRequestError):
        client.audio.voices.create(name="Nobody", consent="", audio_sample=("a.wav", tone_wav(), "audio/wav"))
    with pytest.raises(BadRequestError):
        client.audio.voices.create(name="Prompted", consent="self", type="prompt", prompt="a deep voice")
    assert client.audio.voices.create(
        name="Self", consent="self", audio_sample=("a.wav", tone_wav(), "audio/wav")
    ).id == "self"  # fmt: skip
    client.audio.voices.delete("asha")
    assert live.core.voices.find("asha") is None
    client.audio.voice_consents.delete(consent.id)
    assert client.audio.voice_consents.list().data == []


# -- CLI ----------------------------------------------------------------------------------------------------------
def test_cli_voices(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLO_DEFAULT_DIARIZATION", "fake-diarization")
    monkeypatch.setenv("BOLO_AUTO_PULL", "false")
    add_clone_model(tmp_path / "data")
    data = ["--data-dir", str(tmp_path / "data"), "--fake-engine"]
    ref = tmp_path / "me.wav"
    ref.write_bytes(tone_wav())
    assert cli(["voices", "add", "Me", str(ref), *data]) == 2  # no consent
    assert "explicit consent" in capsys.readouterr().err
    assert cli(["voices", "add", "Me", str(ref), "--consent", "self", "--transcript", "hello", *data]) == 0
    assert "saved voice me" in capsys.readouterr().out
    assert cli(["voices", "list", *data]) == 0
    assert "self" in capsys.readouterr().out
    assert cli(["voices", "show", "me", *data]) == 0
    assert json.loads(capsys.readouterr().out)["transcript"] == "hello"
    assert cli(["voices", "add", "Me", str(ref), "--consent", "self", *data]) == 1  # exists
    capsys.readouterr()
    out = tmp_path / "voices.zip"
    assert cli(["voices", "export", "-o", str(out), *data]) == 0
    assert cli(["voices", "score", "me", "--model", "fake-tts-clone", *data]) == 0
    assert "speaker similarity 1.000" in capsys.readouterr().out
    assert cli(["voices", "rm", "me", *data]) == 0
    assert cli(["voices", "import", str(out), *data]) == 0
    assert "imported 1 voice(s): me" in capsys.readouterr().out
    assert cli(["voices", "ls", "--json", *data]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == "me"
