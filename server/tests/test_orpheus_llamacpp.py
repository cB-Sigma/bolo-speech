"""The orpheus-llamacpp kind (Orpheus, Veena, Svara, snorTTS on llama-server + SNAC in onnxruntime), with a fake
llama-server that streams token ids and a fake SNAC session (no downloads, no onnxruntime)."""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from bolo.catalog import Catalog, ModelSpec
from bolo.config import Settings
from bolo.engines import orpheus_llamacpp as orph
from bolo.engines.base import LoadOptions, SynthesisExtras
from bolo.engines.tts_server import TTSServerEngine
from bolo.errors import EngineUnavailable, InvalidRequest
from conftest import LiveServer
from fake_orpheus_server import AUDIO_OFFSET, END_OF_SPEECH, FakeOrpheusServer, frame_tokens
from fakebin import fake_program

HERE = Path(__file__).parent
SPF = orph.SAMPLES_PER_FRAME


class Input:
    def __init__(self, name: str, type_: str = "tensor(int64)") -> None:
        self.name = name
        self.type = type_


class FakeSnac:
    """A SNAC decoder stand-in: each frame decodes to 2048 samples of ``level-0 code / 10000``."""

    def __init__(self, names: tuple[str, ...] = ("codes_0", "codes_1", "codes_2")) -> None:
        self.names = names
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[Input]:
        return [Input(n) for n in self.names]

    def run(self, output_names: list[str] | None, feeds: dict[str, Any]) -> list[np.ndarray]:
        self.calls.append(feeds)
        level0 = feeds[self.names[0]][0]
        assert len(feeds[self.names[1]][0]) == 2 * len(level0) and len(feeds[self.names[2]][0]) == 4 * len(level0)
        audio = np.repeat(level0.astype(np.float32) / 10000.0, SPF)
        return [audio.reshape(1, 1, -1)]


def expected_audio(frames: int) -> np.ndarray:
    return np.repeat(np.arange(1, frames + 1, dtype=np.float32) / 10000.0, SPF)


SVARA_VOICES = [
    {"id": "hi_male", "name": "Hindi (Male)", "language": "hi-IN", "gender": "m"},
    {"id": "bn_female", "name": "Bengali (Female)", "language": "bn-IN", "gender": "f"},
]


def model_for(server: FakeOrpheusServer, fmt: str = "svara", **options: Any) -> orph.OrpheusLlamaCpp:
    spec: dict[str, Any] = {"prompt_format": fmt, "voices": SVARA_VOICES if fmt == "svara" else [], **options}
    return orph.OrpheusLlamaCpp("svara-test", spec, orph.SnacDecoder(FakeSnac()), base_url=server.url)


@pytest.fixture
def server() -> Iterator[FakeOrpheusServer]:
    s = FakeOrpheusServer().start()
    yield s
    s.stop()


# -- prompt and token layout --------------------------------------------------------------------------------------
def test_prompts_frames_and_levels() -> None:
    def tok(text: str) -> list[int]:
        return [ord(c) for c in text]

    assert orph.build_prompt("veena", tok, "hi", "kavya") == [
        orph.START_OF_HUMAN, *tok("<spk_kavya> hi"), orph.END_OF_HUMAN, orph.START_OF_AI, orph.START_OF_SPEECH,
    ]  # fmt: skip
    assert orph.build_prompt("svara", tok, "hi", "Hindi (Male)") == [
        orph.BOS, orph.START_OF_HUMAN, orph.SVARA_AUDIO, *tok("Hindi (Male): hi"), orph.END_OF_HUMAN,
        orph.END_OF_TURN, orph.START_OF_AI, orph.START_OF_SPEECH,
    ]  # fmt: skip
    assert orph.build_prompt("orpheus", tok, "hi", "tara") == [
        orph.START_OF_HUMAN, orph.BOS, *tok("tara: hi"), orph.END_OF_TURN, orph.END_OF_HUMAN, orph.START_OF_AI,
        orph.START_OF_SPEECH,
    ]  # fmt: skip
    with pytest.raises(InvalidRequest):
        orph.build_prompt("bark", tok, "hi", "x")

    tokens = [orph.START_OF_SPEECH, *frame_tokens(0), 42, *frame_tokens(1), *frame_tokens(2)[:3]]
    frames = orph.frames_from_tokens(tokens)
    assert frames == [[1, 1, 2, 3, 4, 5, 6], [2, 8, 9, 10, 11, 12, 13]]  # non-audio ids skipped, partial dropped
    assert orph.snac_levels(frames) == [[1, 2], [1, 4, 8, 11], [2, 3, 5, 6, 9, 10, 12, 13]]
    # a frame whose tokens sit in the wrong position slot is dropped
    shifted = frame_tokens(0)[1:] + frame_tokens(0)[:1]
    assert orph.frames_from_tokens(shifted) == []
    assert orph.token_budget("a" * 10, 2048) == (20 + 12) * 7
    assert orph.token_budget("a" * 1000, 2048) == 2048


def test_decoder_takes_inputs_in_graph_order() -> None:
    session = FakeSnac(names=("audio_codes.0", "audio_codes.1", "audio_codes.2"))  # the onnx-community naming
    dec = orph.SnacDecoder(session)
    out = dec.decode([[5], [1, 2], [1, 2, 3, 4]])
    assert out.dtype == np.float32 and len(out) == SPF and out[0] == pytest.approx(5 / 10000)
    assert session.calls[0]["audio_codes.0"].dtype == np.int64
    with pytest.raises(EngineUnavailable, match="expected 3"):
        orph.SnacDecoder(FakeSnac(names=("a", "b")))


# -- streaming ------------------------------------------------------------------------------------------------
def test_streaming_windows_match_the_reference(server: FakeOrpheusServer) -> None:
    model = model_for(server)
    session = model.decoder.session
    assert isinstance(session, FakeSnac)
    tokens = [t for j in range(10) for t in frame_tokens(j)]
    chunks = list(model.stream_tokens(iter(tokens), lambda: False))
    # 4 frames once 5 are complete (one held back), 4 more at 9, the last 2 at the end
    assert [len(c) // SPF for c in chunks] == [4, 4, 2]
    np.testing.assert_allclose(np.concatenate(chunks), expected_audio(10))
    # each later window starts 3 frames of left context early; that audio is decoded and dropped
    windows = [list(c["codes_0"][0]) for c in session.calls]
    assert windows == [[1, 2, 3, 4], [2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 10]]


def test_synthesis_against_the_fake_server(server: FakeOrpheusServer) -> None:
    model = model_for(server)
    model.connect()
    text = "Namaste duniya, kaise ho?"
    chunks = list(model.synthesize(text, "bn_female"))
    audio = np.concatenate(chunks)
    frames = len(text) // 2
    np.testing.assert_allclose(audio, expected_audio(frames))
    tokenize, completion = server.requests[-2], server.requests[-1]
    assert tokenize["path"] == "/tokenize" and tokenize["body"] == {
        "content": f"Bengali (Female): {text}", "add_special": False, "parse_special": True,
    }  # fmt: skip
    body = completion["body"]
    assert completion["path"] == "/completion" and body["stream"] is True and body["return_tokens"] is True
    assert body["prompt"][:3] == [orph.BOS, orph.START_OF_HUMAN, orph.SVARA_AUDIO]
    assert body["prompt"][-4:] == [orph.END_OF_HUMAN, orph.END_OF_TURN, orph.START_OF_AI, orph.START_OF_SPEECH]
    # Svara's recommended sampling, temperature before top-p, penalty over the whole sequence
    assert (body["temperature"], body["top_p"], body["repeat_penalty"]) == (0.75, 0.9, 1.1)
    assert body["samplers"] == ["penalties", "temperature", "top_p"] and body["top_k"] == 0
    assert body["repeat_last_n"] == len(body["prompt"]) + body["n_predict"]
    assert body["n_predict"] == orph.token_budget(text, 2048)
    # the stream is left at END_OF_SPEECH: the tokens the server keeps sending afterwards are never decoded
    time.sleep(0.2)
    assert server.sent_after_eos < 40 * 7

    # the default voice is the first; a voice can be named by its display name; an unknown one is refused
    list(model.synthesize("Hello.", None))
    assert server.requests[-2]["body"]["content"].startswith("Hindi (Male): ")
    list(model.synthesize("Hello.", "Bengali (Female)"))
    assert server.requests[-2]["body"]["content"].startswith("Bengali (Female): ")
    with pytest.raises(InvalidRequest, match="no voice"):
        list(model.synthesize("Hello.", "klingon"))
    with pytest.raises(InvalidRequest, match="one speed"):
        list(model.synthesize("Hello.", speed=1.2))
    with pytest.raises(InvalidRequest, match="reference audio"):
        list(model.synthesize("Hello.", extras=SynthesisExtras(reference_audio=b"RIFF")))
    with pytest.raises(InvalidRequest, match="bad prompt"):
        list(model.synthesize("bad request"))
    model.close()


def test_other_formats_sampling_and_overrides(server: FakeOrpheusServer) -> None:
    veena = orph.OrpheusLlamaCpp(
        "veena", {"prompt_format": "veena", "voices": [{"id": "kavya", "name": "Kavya"}], "seed": 7},
        orph.SnacDecoder(FakeSnac()), base_url=server.url,
    )  # fmt: skip
    assert len(np.concatenate(list(veena.synthesize("Hello there friend", "Kavya")))) > 0
    assert server.requests[-2]["body"]["content"] == "<spk_kavya> Hello there friend"
    body = server.requests[-1]["body"]
    assert (body["temperature"], body["top_p"], body["repeat_penalty"], body["seed"]) == (0.4, 0.9, 1.05, 7)
    assert body["prompt"][0] == orph.START_OF_HUMAN and body["prompt"][-1] == orph.START_OF_SPEECH

    snor = orph.OrpheusLlamaCpp(
        "snor",
        {"prompt_format": "orpheus", "voices": [{"id": "hi_159", "name": "Hindi 159", "speaker": "hindi159"}],
         "temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.05, "max_chars": 30, "request": {"min_p": 0.01}},
        orph.SnacDecoder(FakeSnac()), base_url=server.url,
    )  # fmt: skip
    list(snor.synthesize("One sentence here. And a second one follows."))
    contents = [r["body"]["content"] for r in server.requests if r["path"] == "/tokenize"][-2:]
    assert contents == ["hindi159: One sentence here.", "hindi159: And a second one follows."]  # max_chars split
    body = server.requests[-1]["body"]
    assert (body["temperature"], body["repeat_penalty"], body["min_p"]) == (0.4, 1.05, 0.01)
    assert body["prompt"][:2] == [orph.START_OF_HUMAN, orph.BOS]

    # Svara voices in the Orpheus layout: the display name is the speaker
    svara = orph.OrpheusLlamaCpp(
        "svara-orpheus", {"prompt_format": "orpheus", "speaker_field": "name", "voices": SVARA_VOICES},
        orph.SnacDecoder(FakeSnac()), base_url=server.url,
    )  # fmt: skip
    list(svara.synthesize("Kemon acho?", "bn_female"))
    assert server.requests[-2]["body"]["content"] == "Bengali (Female): Kemon acho?"

    with pytest.raises(EngineUnavailable, match="prompt_format"):
        orph.OrpheusLlamaCpp("x", {"prompt_format": "nope"}, orph.SnacDecoder(FakeSnac()), base_url=server.url)


def test_cancelling_stops_generation_and_closes_the_stream() -> None:
    server = FakeOrpheusServer(delay_s=0.01, tokens_per_event=1).start()
    try:
        model = model_for(server)
        stop = {"now": False}
        got = []
        for chunk in model.synthesize("x" * 120, "hi_male", cancelled=lambda: stop["now"]):
            got.append(chunk)
            stop["now"] = True
        assert len(got) == 1  # nothing after the cancel, not even the held-back tail
        deadline = time.time() + 5
        while server.disconnects == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert server.disconnects == 1  # the server saw the client leave (llama-server then frees the slot)
    finally:
        server.stop()


def test_unreachable_and_credentials() -> None:
    model = orph.OrpheusLlamaCpp(
        "gone", {"prompt_format": "svara", "voices": SVARA_VOICES}, orph.SnacDecoder(FakeSnac()),
        base_url="http://127.0.0.1:9",
    )  # fmt: skip
    with pytest.raises(EngineUnavailable, match="not reachable"):
        model.connect()
    with pytest.raises(EngineUnavailable, match="not reachable"):
        list(model.synthesize("Hello."))
    locked = FakeOrpheusServer(api_key="sesame").start()
    try:
        with pytest.raises(EngineUnavailable, match="credentials"):
            list(model_for(locked).synthesize("Hello."))
        ok = orph.OrpheusLlamaCpp(
            "ok", {"prompt_format": "svara", "voices": SVARA_VOICES}, orph.SnacDecoder(FakeSnac()),
            base_url=locked.url, api_key="sesame",
        )  # fmt: skip
        assert len(list(ok.synthesize("Hello there."))) > 0
    finally:
        locked.stop()


# -- the SNAC decoder file ------------------------------------------------------------------------------------
def test_finding_and_fetching_the_decoder(tmp_path: Path) -> None:
    payload = b"fake onnx graph"
    sha = hashlib.sha256(payload).hexdigest()
    server = FakeOrpheusServer(files={"dec.onnx": payload, "bad.onnx": b"tampered"}).start()
    try:
        model_dir = tmp_path / "model"
        (model_dir / "snac").mkdir(parents=True)
        info = {"file": "snac/dec.onnx", "url": f"{server.url}/files/dec.onnx", "sha256": sha}
        # pulled: the file inside the model folder
        (model_dir / "snac" / "dec.onnx").write_bytes(payload)
        assert orph.find_decoder(model_dir, {"snac": info}, tmp_path / "data", download=False) == (
            model_dir / "snac" / "dec.onnx"
        )
        # an explicit path wins; a missing one is an error
        assert orph.find_decoder(model_dir, {"snac": info, "snac_path": str(model_dir / "snac" / "dec.onnx")},
                                 None, download=False).name == "dec.onnx"  # fmt: skip
        with pytest.raises(EngineUnavailable, match="does not exist"):
            orph.find_decoder(model_dir, {"snac_path": str(tmp_path / "nope.onnx")}, None, download=False)
        # served by an operator-run server: nothing was pulled, the decoder is fetched once and checked
        empty = tmp_path / "empty"
        with pytest.raises(EngineUnavailable, match="pull the model again"):
            orph.find_decoder(empty, {"snac": info}, tmp_path / "data", download=False)
        got = orph.find_decoder(empty, {"snac": info}, tmp_path / "data", download=True)
        assert got.read_bytes() == payload and got.is_relative_to(tmp_path / "data" / "components" / "snac")
        assert orph.find_decoder(empty, {"snac": info}, tmp_path / "data", download=False) == got  # cached
        bad = {**info, "url": f"{server.url}/files/bad.onnx", "sha256": "0" * 64}
        with pytest.raises(EngineUnavailable, match="SHA-256"):
            orph.find_decoder(empty, {"snac": bad}, tmp_path / "data", download=True)
        assert not list((tmp_path / "data" / "components" / "snac" / ("0" * 16)).glob("*"))
        with pytest.raises(EngineUnavailable, match="name its SNAC decoder"):
            orph.find_decoder(empty, {}, tmp_path / "data", download=True)
    finally:
        server.stop()


# -- through the engine and the HTTP API --------------------------------------------------------------------------
def entry(model_id: str, decoder: Path) -> dict[str, Any]:
    return {
        "id": model_id,
        "task": "tts",
        "engine": "tts-server",
        "kind": "orpheus-llamacpp",
        "family": "Svara",
        "languages": ["hi", "bn"],
        "sample_rate": 24000,
        "license": "Apache-2.0",
        "requires": ["llama-server"],
        "source": {"local": "/nonexistent"},
        "voices": SVARA_VOICES,
        "options": {
            "runtime": "llama-server",
            "model_file": "m.gguf",
            "prompt_format": "svara",
            "server_args": ["-c", "4096", "-np", "1"],
            "snac_path": str(decoder),
        },
    }


@pytest.fixture
def fake_onnx(monkeypatch: pytest.MonkeyPatch) -> list[FakeSnac]:
    sessions: list[FakeSnac] = []

    def open_session(path: Path, threads: int, device: str = "cpu") -> FakeSnac:
        assert path.is_file() and threads >= 1
        sessions.append(FakeSnac())
        return sessions[-1]

    monkeypatch.setattr(orph, "open_session", open_session)
    return sessions


def test_speech_through_an_operator_run_server(
    settings: Settings, server: FakeOrpheusServer, fake_onnx: list[FakeSnac], tmp_path: Path
) -> None:
    decoder = tmp_path / "decoder.onnx"
    decoder.write_bytes(b"onnx")
    settings.tts_servers = [f"svara-remote={server.url}"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(entry("svara-remote", decoder)))
    srv.start()
    try:
        info = httpx.get(f"{srv.url}/api/v1/models/svara-remote").json()
        assert info["installed"] is True and info["available"] is True and info["requires"] == []
        r = httpx.post(
            f"{srv.url}/v1/audio/speech",
            json={"model": "svara-remote", "input": "Namaste duniya.", "voice": "bn_female", "response_format": "wav"},
            timeout=30,
        )
        assert r.status_code == 200, r.text
        audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
        assert rate == 24000 and len(audio) >= SPF
        assert server.requests[-2]["body"]["content"].startswith("Bengali (Female): ")
        voices = httpx.get(f"{srv.url}/v1/audio/voices").json()["voices"]
        assert {"hi_male", "bn_female"} <= {v["id"] for v in voices if v["model_id"] == "svara-remote"}
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "svara-remote", "input": "Hi.", "speed": 2.0})
        assert r.status_code == 400 and r.json()["error"]["param"] == "speed"
        assert len(fake_onnx) == 1
    finally:
        srv.stop()


def fake_binary(tmp_path: Path) -> Path:
    return fake_program(tmp_path / "bin" / "llama-server", f"import sys\nsys.path.insert(0, {str(HERE)!r})\n"
                        "from fake_orpheus_server import main\nmain(sys.argv[1:])\n")  # fmt: skip


def test_managed_llama_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_onnx: list[FakeSnac]) -> None:
    argv_log = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(argv_log))
    settings = Settings(data_dir=tmp_path / "data", llama_server_bin=str(fake_binary(tmp_path)))
    engine = TTSServerEngine()
    engine.configure(settings)
    model_dir = tmp_path / "model"
    (model_dir / "snac").mkdir(parents=True)
    (model_dir / "m.gguf").write_bytes(b"GGUF")
    (model_dir / "snac" / "dec.onnx").write_bytes(b"onnx")
    spec = {**entry("managed", tmp_path / "unused")["options"], "model_id": "managed", "voices": SVARA_VOICES}
    spec.pop("snac_path")
    spec["snac"] = {"file": "snac/dec.onnx"}
    model = engine.load("orpheus-llamacpp", "tts", model_dir, spec, LoadOptions(threads=2))
    assert isinstance(model, orph.OrpheusLlamaCpp) and model.process is not None
    proc = model.process
    try:
        import json

        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        assert argv[argv.index("-m") + 1] == str(model_dir / "m.gguf")
        assert argv[argv.index("-c") + 1] == "4096" and "--api-key" in argv and "-mm" not in argv
        assert proc.running and proc.api_key
        out = np.concatenate(list(model.synthesize("Hello there.", "hi_male")))
        np.testing.assert_allclose(out, expected_audio(len("Hello there.") // 2))
        assert engine.processes() and engine.processes()[0]["running"]
        # the server dies: engine_unavailable, then a restart after the back-off
        with pytest.raises(EngineUnavailable):
            list(model.synthesize("crash now"))
        time.sleep(0.2)
        with pytest.raises(EngineUnavailable, match="restarting in"):
            list(model.synthesize("Hello."))
        assert proc.crashes == 1
        time.sleep(1.1)
        assert len(np.concatenate(list(model.synthesize("Hello again.")))) > 0 and proc.restarts == 1
    finally:
        model.close()
    assert not proc.running

    with pytest.raises(EngineUnavailable, match="runs on llama-server"):
        engine.load("orpheus-llamacpp", "tts", model_dir, {**spec, "runtime": "audio.cpp"}, LoadOptions())


def test_catalog_entries() -> None:
    specs = [s for s in Catalog.load(None).all() if s.kind == "orpheus-llamacpp"]
    ids = {s.id for s in specs}
    assert {"svara-tts-v1-llamacpp", "svara-tts-v1-q8-llamacpp", "veena-llamacpp", "veena-q8-llamacpp"} <= ids
    for s in specs:
        assert s.engine == "tts-server" and s.requires == ["llama-server"] and s.sample_rate == 24000
        assert s.options["prompt_format"] in orph.FORMATS and s.options["model_file"] in s.source.include
        assert s.options["hardware"]["gpu"] == "recommended" and s.options["hardware"]["vram_gb"] > 0
        snac = s.options["snac"]
        assert len(snac["sha256"]) == 64 and any(u["path"] == snac["file"] for u in s.source.urls)
        assert len(s.source.revision) == 40 and s.voices and s.license
    svara = next(s for s in specs if s.id == "svara-tts-v1-llamacpp")
    assert "IndicTTS" in svara.description and svara.options["model_file"].endswith("Q4_K_M.gguf")
    assert TTSServerEngine.kinds == ("tts-server", "orpheus-llamacpp", "voxtral-llamacpp", "ten-turn-llamacpp")


def test_audio_token_range_is_the_reference_one() -> None:
    assert orph.is_audio_token(AUDIO_OFFSET) and orph.is_audio_token(AUDIO_OFFSET + 7 * 4096 - 1)
    assert not orph.is_audio_token(AUDIO_OFFSET + 7 * 4096) and not orph.is_audio_token(END_OF_SPEECH)
