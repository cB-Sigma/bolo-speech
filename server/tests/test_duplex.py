"""Full-duplex models (bolo.duplex): the Ogg Opus framing, the Kyutai protocol client, the Realtime bridge against a
fake Kyutai server (tests/fake_kyutai_server.py), and the managed servers' install and launch commands."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import io
import json
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.config import Settings
from bolo.duplex import DuplexAudio, DuplexConfig, DuplexEnd, DuplexText
from bolo.duplex import runtime as rt
from bolo.duplex.kyutai import KyutaiBackend, chat_url
from bolo.duplex.ogg_opus import OggOpusReader, OggOpusWriter, ogg_crc, ogg_page
from bolo.errors import EngineUnavailable, Overloaded
from conftest import LiveServer, speechlike
from fake_kyutai_server import FakeKyutaiServer, main
from fakebin import fake_program
from test_s2s import connect, until, update

HERE = Path(__file__).parent
MODEL = "moshiko-pytorch-q8"


def tone(seconds: float, rate: int = 24000, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# -- Ogg Opus -------------------------------------------------------------------------------------------------------
def test_ogg_page_and_crc() -> None:
    page = ogg_page(b"x" * 600, serial=7, sequence=3, granule=960, flags=2)
    assert page[:4] == b"OggS" and page[26] == 3 and list(page[27:30]) == [255, 255, 90]
    zeroed = bytearray(page)
    zeroed[22:26] = b"\0\0\0\0"
    assert ogg_crc(bytes(zeroed)) == int.from_bytes(page[22:26], "little")
    with pytest.raises(ValueError):
        ogg_page(b"x" * 70000, serial=0, sequence=0, granule=0)


def test_ogg_opus_round_trip_in_any_split() -> None:
    x = tone(1.0)
    writer = OggOpusWriter(24000)
    data = b"".join(writer.append(x[i : i + 1000]) for i in range(0, len(x), 1000))
    assert data.startswith(b"OggS") and b"OpusHead" in data[:64] and b"OpusTags" in data[:200]
    reader = OggOpusReader(24000)
    out = np.concatenate([reader.append(data[i : i + 37]) for i in range(0, len(data), 37)])
    assert abs(len(out) - len(x)) <= 1920 and reader.bad_pages == 0
    spectrum = np.abs(np.fft.rfft(out[4800:]))
    assert abs(np.argmax(spectrum) * 24000 / len(out[4800:]) - 440) < 5
    # a damaged page is skipped, the stream goes on
    reader2 = OggOpusReader(24000)
    broken = bytearray(data)
    broken[300] ^= 0xFF
    assert len(reader2.append(bytes(broken))) > 0 and reader2.bad_pages == 1


def test_ogg_opus_reads_and_is_read_by_kyutais_sphn() -> None:
    sphn = pytest.importorskip("sphn")
    x = tone(1.0)
    ours = b"".join(OggOpusWriter(24000).append(x[i : i + 1920]) for i in range(0, len(x), 1920))
    theirs = np.asarray(sphn.OpusStreamReader(24000).append_bytes(ours)).reshape(-1)
    assert len(theirs) >= len(x) - 1920 and float(np.abs(theirs).max()) > 0.2
    writer = sphn.OpusStreamWriter(24000)
    stream = b"".join(writer.append_pcm(x[i : i + 1920]) for i in range(0, len(x), 1920))
    back = OggOpusReader(24000).append(stream)
    assert len(back) >= len(x) - 2 * 1920 and float(np.abs(back).max()) > 0.2


def test_chat_url() -> None:
    assert chat_url("http://h:8998") == "ws://h:8998/api/chat"
    assert chat_url("https://h:8998/") == "wss://h:8998/api/chat"
    assert chat_url("ws://h/api/chat") == "ws://h/api/chat"
    assert chat_url("h:1") == "ws://h:1/api/chat"


# -- the protocol client --------------------------------------------------------------------------------------------
@pytest.fixture
def fake() -> Iterator[FakeKyutaiServer]:
    server = FakeKyutaiServer().start()
    try:
        yield server
    finally:
        server.stop()


async def _feed(session: Any, audio: np.ndarray, chunk: int = 1920) -> None:
    for i in range(0, len(audio), chunk):
        await session.push(audio[i : i + chunk])


async def _eventually(condition: Any, timeout_s: float = 10.0) -> None:
    """Wait until ``condition()`` holds (the fake servers update their state from another thread)."""
    async with asyncio.timeout(timeout_s):
        while not condition():
            await asyncio.sleep(0.02)


def _speech_then_quiet(speech_s: float = 0.8, quiet_s: float = 2.0) -> np.ndarray:
    return np.concatenate([speechlike(speech_s, 24000), np.zeros(int(quiet_s * 24000), dtype=np.float32)])


@pytest.mark.parametrize("use_sphn", [True, False])
@pytest.mark.parametrize("rust", [False, True])
async def test_kyutai_session(use_sphn: bool, rust: bool) -> None:
    if use_sphn:
        pytest.importorskip("sphn")
    server = FakeKyutaiServer(use_sphn=use_sphn, rust_handshake=rust).start()
    try:
        backend = KyutaiBackend(server.url, model=MODEL, query={"text_temperature": 0.7})
        session = await backend.open(DuplexConfig(options={"audio_topk": 100, "ignored": 1}))
        await _feed(session, _speech_then_quiet())
        text, audio = "", 0
        async with asyncio.timeout(10):
            async for event in session.events():
                if isinstance(event, DuplexText):
                    text += event.text
                elif isinstance(event, DuplexAudio):
                    assert event.sample_rate == 24000
                    audio += len(event.samples)
                # the server answers each 80 ms of input with 80 ms of output: 2 s back means 2 s of input reached it
                if text.endswith(".") and audio >= 24000 * 2:
                    break
        assert text == " Hello there."
        await session.close()
        await session.close()  # twice is fine
        state = server.connections[0]
        await _eventually(lambda: state["closed"])
        assert state["samples"] >= 24000 * 2
        assert state["query"] == {"text_temperature": "0.7", "audio_topk": "100"}
        assert backend.active == 0 and backend.sessions_served == 1
    finally:
        server.stop()


async def test_kyutai_sessions_are_limited_and_a_busy_server_is_reported(fake: FakeKyutaiServer) -> None:
    backend = KyutaiBackend(fake.url, model=MODEL, max_sessions=1)
    first = await backend.open(DuplexConfig())
    with pytest.raises(Overloaded, match="as many as it holds"):
        await backend.open(DuplexConfig())
    # a second client of the server itself (Kyutai's Python servers hold one connection; the next one waits)
    other = KyutaiBackend(fake.url, model=MODEL, handshake_timeout_s=0.5)
    with pytest.raises(Overloaded, match="one conversation at a time"):
        await other.open(DuplexConfig())
    assert other.active == 0
    await first.close()
    second = await backend.open(DuplexConfig())
    await second.close()
    unreachable = KyutaiBackend("http://127.0.0.1:9", model=MODEL)
    with pytest.raises(EngineUnavailable, match="not reachable"):
        await unreachable.open(DuplexConfig())


async def test_kyutai_server_error_and_end() -> None:
    server = FakeKyutaiServer(error="out of memory").start()
    try:
        session = await KyutaiBackend(server.url, model=MODEL).open(DuplexConfig())
        events = [e async for e in session.events()]
        assert events[-1] == DuplexEnd("failed", "out of memory")
        await session.close()
    finally:
        server.stop()
    server = FakeKyutaiServer(max_steps=5).start()
    try:
        session = await KyutaiBackend(server.url, model=MODEL).open(DuplexConfig())
        await _feed(session, np.zeros(24000, dtype=np.float32))
        async with asyncio.timeout(10):
            events = [e async for e in session.events()]
        assert isinstance(events[-1], DuplexEnd) and events[-1].reason == "ended"
        await session.close()
    finally:
        server.stop()


async def test_personaplex_prompts_go_in_the_query(fake: FakeKyutaiServer) -> None:
    backend = KyutaiBackend(
        fake.url, model="personaplex-7b-v1", voice_param="voice_prompt", voice_suffix=".pt", default_voice="NATF2",
        instructions_param="text_prompt",
    )  # fmt: skip
    session = await backend.open(DuplexConfig(instructions="You run a bakery.", voice="alloy"))
    await session.close()
    session = await backend.open(DuplexConfig(voice="VARM1"))
    await session.close()
    assert fake.connections[0]["query"] == {"voice_prompt": "NATF2.pt", "text_prompt": "You run a bakery."}
    assert fake.connections[1]["query"] == {"voice_prompt": "VARM1.pt"}  # an empty text_prompt is sent as nothing


# -- Realtime sessions ------------------------------------------------------------------------------------------------
@pytest.fixture
def duplex(settings: Settings, fake: FakeKyutaiServer) -> Iterator[LiveServer]:
    settings.s2s_model = MODEL
    settings.duplex_server_url = fake.url
    settings.duplex_silence_ms = 400
    settings.duplex_jitter_ms = 100
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def b64_pcm24(audio: np.ndarray) -> list[str]:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    return [base64.b64encode(pcm[i : i + 4800]).decode() for i in range(0, len(pcm), 4800)]


SESSION = {
    "type": "realtime", "pipeline": "speech-to-speech",
    "audio": {"input": {"transcription": {"model": "fake-stt"},
                        "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}}},
}  # fmt: skip


async def test_realtime_session_with_a_full_duplex_model(duplex: LiveServer, fake: FakeKyutaiServer) -> None:
    info = httpx.get(duplex.url + "/api/v1/info").json()
    s2s = info["capabilities"]["realtime"]["pipelines"]["speech-to-speech"]
    assert s2s["full_duplex"] is True and s2s["available"] is True and "full-duplex" in s2s["features"]
    ws = await connect(duplex)
    try:
        updated = await update(ws, SESSION)
        assert updated["session"]["full_duplex"] is True
        for chunk in b64_pcm24(_speech_then_quiet(1.0, 0.6)):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        # the client stops sending: bolo feeds the model silence on the clock, so it still answers and finishes
        events = await until(ws, "response.done", timeout=20)
        events += (
            await until(ws, "conversation.item.input_audio_transcription.completed", timeout=20)
            if not any(e["type"] == "conversation.item.input_audio_transcription.completed" for e in events)
            else []
        )
        # requests a full-duplex model does not take
        await ws.send(json.dumps({"type": "response.create"}))
        refused = await until(ws, "error")
        assert refused[-1]["error"]["code"] == "unsupported"
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
        assert (await until(ws, "error"))[-1]["error"]["code"] == "unsupported"
        await ws.send(json.dumps({"type": "response.cancel"}))
        assert (await until(ws, "error"))[-1]["error"]["code"] == "response_cancel_not_active"
    finally:
        await ws.close()
    types = [e["type"] for e in events]
    order = [
        "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
        "response.created", "response.output_item.added", "response.content_part.added",
        "response.output_audio.done", "response.output_audio_transcript.done", "response.content_part.done",
        "response.output_item.done", "response.done",
    ]  # fmt: skip
    positions = [types.index(t) for t in order]
    assert positions == sorted(positions), types
    deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_audio_transcript.delta")
    assert deltas == "Hello there."  # the leading space of the first token is dropped
    done = next(e for e in events if e["type"] == "response.done")["response"]
    assert done["status"] == "completed" and done["output"][0]["content"][0]["transcript"] == "Hello there."
    audio = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
    seconds = len(audio) / 2 / 24000
    assert 0.9 <= seconds <= 2.5  # 12 frames of tone, the pre-roll and the pause that ends the response
    assert "response.created" not in types[: types.index("input_audio_buffer.speech_stopped")]
    state = fake.connections[0]
    await _eventually(lambda: state["closed"])
    assert state["replies"] == 1
    assert state["samples"] > len(_speech_then_quiet(1.0, 0.6)) - 1920  # padding on top of what the client sent
    trace = httpx.get(duplex.url + "/api/v1/requests", params={"kind": "agent"}).json()["requests"][0]
    assert trace["status"] == "ok" and trace["llm_model"] == MODEL and trace["full_duplex"] is True


async def test_barge_in_is_the_models_and_cancel_is_the_clients(duplex: LiveServer, fake: FakeKyutaiServer) -> None:
    fake.speak_frames = 40  # a long answer
    ws = await connect(duplex)
    try:
        await update(ws, SESSION)
        for chunk in b64_pcm24(_speech_then_quiet(1.0, 0.8)):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        await until(ws, "response.output_audio.delta", timeout=20)
        # the user talks over it: bolo reports the speech but does not cancel the model's response
        for chunk in b64_pcm24(speechlike(1.0, 24000)):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        events = await until(ws, "input_audio_buffer.speech_started", timeout=20)
        events += await until(ws, "response.done", timeout=20)
        done = events[-1]["response"]
        assert done["status"] == "completed"  # it ended because the model stopped talking, not because bolo cut it
        # the next answer, cancelled by the client
        for chunk in b64_pcm24(np.zeros(int(1.0 * 24000), dtype=np.float32)):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        await until(ws, "response.created", timeout=20)
        await ws.send(json.dumps({"type": "response.cancel"}))
        cancelled = (await until(ws, "response.done", timeout=20))[-1]["response"]
        assert cancelled["status"] == "cancelled" and cancelled["status_details"]["reason"] == "client_cancelled"
    finally:
        await ws.close()
    assert fake.connections[0]["barge_ins"] == 1 and fake.connections[0]["replies"] == 2


async def test_text_modality_and_a_second_session_when_busy(duplex: LiveServer, fake: FakeKyutaiServer) -> None:
    ws = await connect(duplex)
    other = await connect(duplex)
    try:
        await update(ws, {**SESSION, "output_modalities": ["text"]})
        await update(other, SESSION)
        for chunk in b64_pcm24(_speech_then_quiet(1.0, 1.5)):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        events = await until(ws, "response.done", timeout=20)
        await other.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64_pcm24(tone(0.1))[0]}))
        busy = await until(other, "error", timeout=20)
    finally:
        await ws.close()
        await other.close()
    assert "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta") == "Hello there."
    assert not any(e["type"] == "response.output_audio.delta" for e in events)
    assert busy[-1]["error"]["code"] == "overloaded"


def test_turn_based_models_are_not_duplex(settings: Settings) -> None:
    from bolo.core import Bolo

    core = Bolo(settings)
    assert core.s2s.full_duplex is False
    settings.s2s_model = "hibiki-1b-pytorch-bf16"
    assert core.s2s.full_duplex is True
    described = core.s2s.describe()
    assert described["full_duplex"] is True and described["available"] is False
    assert "bolo components install moshi" in described["reason"]


# -- catalog --------------------------------------------------------------------------------------------------------
def test_catalog_entries() -> None:
    from bolo.catalog import Catalog

    specs = [s for s in Catalog.load(None).all() if s.engine == "duplex"]
    ids = {s.id for s in specs}
    assert {"moshiko-pytorch-bf16", "moshika-mlx-q4", "moshiko-candle-q8", "hibiki-2b-mlx-bf16",
            "hibiki-zero-3b-pytorch-bf16", "personaplex-7b-v1"} <= ids  # fmt: skip
    for s in specs:
        assert s.task == "speech-to-speech" and "full-duplex" in s.features and s.runnable, s.id
        server, build = rt.server_of(s)
        assert build in rt.SERVERS[server].builds and s.requires == [server], s.id
        names = {u["path"] for u in s.source.urls} or set(s.source.include)
        assert {s.options[k] for k in ("model_file", "mimi_file", "tokenizer_file") if k in s.options} <= names, s.id
        assert all(u["sha256"] for u in s.source.urls) and s.license and s.license_url, s.id
    assert next(s for s in specs if s.id == "hibiki-zero-3b-pytorch-bf16").commercial_use is False
    assert next(s for s in specs if s.id == "personaplex-7b-v1").gated is True


# -- installing and starting the servers -----------------------------------------------------------------------------
def _install_fake(settings: Settings, server: str, build: str) -> Path:
    root = rt.install_dir(settings, server, build)
    b = rt.SERVERS[server].builds[build]
    exe = rt._cargo_binary(root, b.binary) if b.binary else rt._venv_python(root)
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    (root / ".bolo-complete").write_text("{}", encoding="utf-8")
    return exe


def _model_dir(tmp_path: Path, spec: Any) -> Path:
    path = tmp_path / "models" / spec.id
    path.mkdir(parents=True)
    for key in ("model_file", "mimi_file", "tokenizer_file", "config_file"):
        if spec.options.get(key):
            (path / spec.options[key]).parent.mkdir(parents=True, exist_ok=True)
            (path / spec.options[key]).write_text("{}", encoding="utf-8")
    return path


def test_launch_commands(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.catalog import Catalog

    monkeypatch.setattr(rt, "platform_key", lambda: "linux-x86_64")
    catalog = Catalog.load(None)
    spec = catalog.resolve("moshiko-pytorch-bf16")
    assert spec is not None
    with pytest.raises(FileNotFoundError, match="bolo components install moshi"):
        rt.launch(settings, spec, tmp_path, 1234, [])
    exe = _install_fake(settings, "moshi", "pytorch")
    path = _model_dir(tmp_path, spec)
    settings.device = "cuda:1"
    launch = rt.launch(settings, spec, path, 1234, ["--half"])
    argv = launch.process.argv
    assert argv[:3] == [str(exe), "-I", "-u"] and Path(argv[3]).read_text(encoding="utf-8") == rt.LAUNCHER
    run = json.loads(argv[4])
    assert run["module"] == "moshi.server" and run["host"] == "127.0.0.1" and run["port"] == 1234 and not run["cpu"]
    args = run["args"]
    assert args[args.index("--device") + 1] == "cuda:1" and args[-1] == "--half" and args[:2] == ["--host", "127.0.0.1"]
    config = json.loads(Path(args[args.index("--config-path") + 1]).read_text(encoding="utf-8"))
    assert config["moshi_name"] == "model.safetensors" and config["dim"] == 4096  # Moshi 7B, written by bolo
    assert launch.max_sessions == 1 and not launch.restart_after_session and launch.process.base_url.startswith("http:")
    assert launch.process.env["HF_HUB_OFFLINE"] == "1"
    settings.device = "dml"
    with pytest.raises(ValueError, match="NVIDIA GPU"):
        rt.launch(settings, spec, path, 1234, [])
    # the q8 checkpoint brings its config; the launcher answers Hub lookups of it from disk
    settings.device = "auto"
    q8 = catalog.resolve("moshika-pytorch-q8")
    assert q8 is not None
    run = json.loads(rt.launch(settings, q8, _model_dir(tmp_path, q8), 5, []).process.argv[4])
    assert run["files"]["config.json"].endswith("config.json") and "--config-path" in run["args"]
    # MLX: quantised weights, restarted after every conversation
    monkeypatch.setattr(rt, "platform_key", lambda: "darwin-arm64")
    _install_fake(settings, "moshi", "mlx")
    mlx = catalog.resolve("moshiko-mlx-q4")
    assert mlx is not None
    launch = rt.launch(settings, mlx, _model_dir(tmp_path, mlx), 6, [])
    run = json.loads(launch.process.argv[4])
    assert run["module"] == "moshi_mlx.local_web" and run["args"][run["args"].index("-q") + 1] == "4"
    assert launch.restart_after_session
    # candle: a JSON config, TLS on loopback, several sessions
    monkeypatch.setattr(rt, "platform_key", lambda: "linux-x86_64")
    binary = _install_fake(settings, "moshi", "candle")
    candle = catalog.resolve("moshiko-candle-q8")
    assert candle is not None
    settings.duplex_max_sessions = 3
    settings.device = "cpu"
    launch = rt.launch(settings, candle, _model_dir(tmp_path, candle), 7, [])
    assert launch.process.argv[0] == str(binary) and launch.process.argv[-2:] == ["standalone", "--cpu"]
    cfg = json.loads(Path(launch.process.argv[2]).read_text(encoding="utf-8"))
    assert cfg["addr"] == "127.0.0.1" and cfg["port"] == 7 and cfg["lm_model_file"].endswith("model.q8.gguf")
    assert (Path(cfg["static_dir"]) / "index.html").is_file()
    assert launch.process.base_url == "https://127.0.0.1:7" and launch.max_sessions == 3


def test_personaplex_and_hibiki_zero_launch(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bolo.catalog import Catalog

    monkeypatch.setattr(rt, "platform_key", lambda: "windows-x86_64")
    settings.device = "cuda"
    catalog = Catalog.load(None)
    _install_fake(settings, "personaplex", "pytorch")
    spec = catalog.resolve("personaplex-7b-v1")
    assert spec is not None
    path = _model_dir(tmp_path, spec)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in ("voices/NATF2.pt", "voices/VARM1.pt", "../evil.pt", "voices/sub/x.pt"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tf.addfile(info, io.BytesIO(b"abc"))
    (path / "voices.tgz").write_bytes(buf.getvalue())
    run = json.loads(rt.launch(settings, spec, path, 9, []).process.argv[4])
    voices = Path(run["args"][run["args"].index("--voice-prompt-dir") + 1])
    assert sorted(p.name for p in voices.iterdir() if not p.name.startswith(".")) == ["NATF2.pt", "VARM1.pt"]
    assert json.loads(Path(run["files"]["config.json"]).read_text(encoding="utf-8")) == {}
    assert "--config-path" not in run["args"]
    _install_fake(settings, "hibiki-zero", "pytorch")
    hz = catalog.resolve("hibiki-zero-3b-pytorch-bf16")
    assert hz is not None
    run = json.loads(rt.launch(settings, hz, _model_dir(tmp_path, hz), 10, ["--bf16"]).process.argv[4])
    assert run["module"] == "hibiki_zero.run" and run["args"][0] == "serve" and run["args"][-1] == "--bf16"
    assert run["args"][run["args"].index("--model-weight") + 1].endswith(".safetensors")


def test_install_steps(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "platform_key", lambda: "windows-x86_64")
    monkeypatch.setattr(rt.shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("uv", "git", "cargo") else None)
    commands: list[list[str]] = []
    fetched: list[str] = []

    def run(cmd: list[str]) -> tuple[int, str]:
        commands.append(cmd)
        if cmd[0].endswith("uv") and cmd[1] == "venv":
            python = rt._venv_python(Path(cmd[-1]))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
        if cmd[-1] == "HEAD":
            return 0, "16419a2ef7b98611f58c9df07477d00e5ad61d5f\n"
        return 0, ""

    def fetch(pinned: rt.Pinned, folder: Path) -> Path:
        fetched.append(pinned.sha256)
        out = folder / pinned.name
        out.write_bytes(b"wheel")
        return out

    assert rt.install(settings, "moshi", None, fetch, run) is None
    assert rt.installed_flavours(settings, "moshi") == ["pytorch"]
    torch = next(c for c in commands if "torch==2.9.1" in c)
    assert torch[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu128"]
    package = next(c for c in commands if any(a.endswith("moshi-0.2.13-py3-none-any.whl") for a in c))
    assert "bitsandbytes==0.49.2" in package and fetched == [rt.SERVERS["moshi"].builds["pytorch"].wheel.sha256]  # type: ignore[union-attr]
    commands.clear()
    assert rt.install(settings, "personaplex", None, fetch, run) is None
    assert any(c[:2] == ["git", "clone"] for c in commands) and any("torch==2.4.1" in c for c in commands)
    # a failing step leaves nothing behind and says what failed
    assert "installing torch==2.9.1 failed (1): no space" in str(
        rt.install(settings, "hibiki-zero", None, fetch, lambda c: (1, "no space") if "torch==2.9.1" in c else run(c))
    )
    assert rt.installed_flavours(settings, "hibiki-zero") == []
    assert "mlx build" in str(rt.install(settings, "moshi", "mlx", fetch, run))  # not on Windows


def test_components_list_and_install_api(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.core import Bolo

    monkeypatch.setattr(rt, "platform_key", lambda: "linux-x86_64")
    core = Bolo(settings)
    listed = {c["id"]: c for c in core.components.status()}
    moshi = listed["moshi"]
    assert moshi["kind"] == "duplex-server" and not moshi["installed"] and moshi["wanted_flavour"] == "pytorch"
    assert set(moshi["builds"]) == {"pytorch", "mlx", "candle"} and moshi["required_by_count"] >= 14
    assert listed["personaplex"]["license"].startswith("MIT") and "hibiki-zero" in listed
    from bolo.errors import InvalidRequest

    with pytest.raises(InvalidRequest, match="no build 'cuda'"):
        core.components.start_install("moshi", False, "cuda")
    spec = core.catalog.resolve("moshiko-pytorch-q8")
    assert spec is not None
    requires, _, available = core.components.requirements(spec)
    assert requires == ["moshi"] and available is False


async def test_managed_server_end_to_end(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """bolo starts the model's server (here the fake, as the server's "python"), waits for it, bridges a session,
    and stops it."""
    from bolo.core import Bolo
    from bolo.duplex.manager import build_backend

    monkeypatch.setattr(rt, "platform_key", lambda: "linux-x86_64")
    settings.s2s_model = MODEL
    core = Bolo(settings)
    spec = core.catalog.resolve(MODEL)
    assert spec is not None
    root = rt.install_dir(settings, "moshi", "pytorch")
    python = rt._venv_python(root)
    launcher = fake_program(
        python,
        f"import sys\nsys.path.insert(0, {str(HERE)!r})\nfrom fake_kyutai_server import main\nmain(sys.argv[1:])\n",
    )
    if launcher != python:  # Windows: the fake is python.exe.cmd, which bolo starts in the interpreter's place
        monkeypatch.setattr(rt, "_venv_python", lambda root: launcher)
    (root / ".bolo-complete").write_text("{}", encoding="utf-8")
    model_dir = _model_dir(tmp_path, spec)
    monkeypatch.setattr(core.store, "is_installed", lambda name: True)
    monkeypatch.setattr(core.store, "model_dir", lambda name: model_dir)
    backend = await build_backend(core, spec)
    try:
        assert backend.describe()["managed"] is True
        session = await backend.open(DuplexConfig())
        started = time.monotonic()
        await _feed(session, _speech_then_quiet())
        text = ""
        async with asyncio.timeout(15):
            async for event in session.events():
                if isinstance(event, DuplexText):
                    text += event.text
                if text.endswith("."):
                    break
        assert text == " Hello there." and time.monotonic() - started < 15
        await session.close()
    finally:
        await backend.close()
    assert not backend.process.running  # type: ignore[attr-defined]


def test_fake_server_main_is_importable() -> None:
    assert callable(main)


# -- Freeze-Omni (Socket.IO) and FLM-Audio ---------------------------------------------------------------------------
@pytest.fixture
def freeze() -> Iterator[Any]:
    pytest.importorskip("socketio")
    from fake_freeze_omni_server import FakeFreezeOmniServer

    server = FakeFreezeOmniServer().start()
    try:
        yield server
    finally:
        server.stop()


def test_socketio_url_and_attachments() -> None:
    from bolo.duplex.socketio import _fill, socketio_url

    assert socketio_url("https://h:1/") == "wss://h:1/socket.io/?EIO=4&transport=websocket"
    assert _fill(["audio", {"_placeholder": True, "num": 0}], [b"x"]) == ["audio", b"x"]


async def test_freeze_omni_session(freeze: Any) -> None:
    from bolo.duplex import DuplexInterrupt
    from bolo.duplex.freeze_omni import FreezeOmniBackend

    backend = FreezeOmniBackend(freeze.url, model="freeze-omni")
    session = await backend.open(DuplexConfig(instructions="You are a tour guide."))
    assert freeze.prompts == ["You are a tour guide."]
    speech = np.concatenate([speechlike(0.5, 16000), np.zeros(8000, dtype=np.float32)])
    await _feed(session, speech, 320)
    audio = 0
    async with asyncio.timeout(10):
        events = session.events()
        while audio < 2400 * 3:
            await _feed(session, np.zeros(640, dtype=np.float32), 320)  # its server answers one piece per event
            event = await anext(events)
            if isinstance(event, DuplexAudio):
                assert event.sample_rate == 24000
                audio += len(event.samples)
        await _feed(session, speechlike(0.1, 16000), 320)
        while not isinstance(event, DuplexInterrupt):
            event = await anext(events)
    await session.close()
    state = freeze.sessions[0]
    await _eventually(lambda: state["closed"])
    assert state["started"] == 1 and state["interrupted"] == 1 and backend.active == 0


async def test_freeze_omni_in_a_realtime_session(settings: Settings, freeze: Any) -> None:
    settings.s2s_model = "freeze-omni"
    settings.duplex_server_url = freeze.url
    settings.duplex_silence_ms = 400
    srv = LiveServer(settings)
    srv.start()
    try:
        ws = await connect(srv)
        try:
            await update(ws, {**SESSION, "instructions": "Be brief."})
            for chunk in b64_pcm24(_speech_then_quiet(0.6, 1.5)):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            await until(ws, "response.output_audio.delta", timeout=20)
            for chunk in b64_pcm24(speechlike(0.3, 24000)):  # the caller talks over it: the model stops itself
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
            done = (await until(ws, "response.done", timeout=20))[-1]["response"]
        finally:
            await ws.close()
    finally:
        srv.stop()
    assert done["status"] == "cancelled" and done["status_details"]["reason"] == "turn_detected"
    assert freeze.prompts == ["Be brief."] and freeze.sessions[0]["interrupted"] == 1


def test_flm_audio_and_freeze_omni_launch(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.catalog import Catalog

    monkeypatch.setattr(rt, "platform_key", lambda: "windows-x86_64")
    settings.device = "cuda"
    catalog = Catalog.load(None)
    _install_fake(settings, "flm-audio", "pytorch")
    flm = catalog.resolve("flm-audio")
    assert flm is not None and flm.commercial_use is False
    path = _model_dir(tmp_path, flm)
    run = json.loads(rt.launch(settings, flm, path, 11, []).process.argv[4])
    assert run["module"] == "flmaudio.server" and run["cwd"].endswith(
        str(Path("flm-audio") / "41d768c9" / "pytorch" / "src")
    )
    assert run["args"][run["args"].index("--model-path") + 1] == str(path)
    _install_fake(settings, "freeze-omni", "pytorch")
    fz = catalog.resolve("freeze-omni")
    assert fz is not None and fz.kind == "freeze-omni"
    path = _model_dir(tmp_path, fz)
    settings.duplex_max_sessions = 2
    launch = rt.launch(settings, fz, path, 12, [])
    run = json.loads(launch.process.argv[4])
    args = run["args"]
    assert run["module"] == "bin/server.py" and args[:4] == ["--ip", "127.0.0.1", "--port", "12"]
    assert (
        args[args.index("--llm_path") + 1] == str(path / "Qwen2-7B-Instruct")
        and args[args.index("--max_users") + 1] == "2"
    )
    assert launch.process.base_url == "https://127.0.0.1:12" and launch.max_sessions == 2
    assert any(u["path"].startswith("Qwen2-7B-Instruct/model-") for u in fz.source.urls)


def test_install_from_a_checkout(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "platform_key", lambda: "linux-x86_64")
    monkeypatch.setattr(rt.shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("uv", "git") else None)
    commands: list[list[str]] = []

    def run(cmd: list[str]) -> tuple[int, str]:
        commands.append(cmd)
        if cmd[1] == "venv":
            python = rt._venv_python(Path(cmd[-1]))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
        if cmd[-1] == "HEAD":
            return 0, "163a24880e533b2a07038fb8dcfe02dbbb8457e6\n"
        return 0, ""

    assert rt.install(settings, "freeze-omni", None, lambda p, f: f / p.name, run) is None
    venv = next(c for c in commands if c[1] == "venv")
    assert venv[venv.index("--python") + 1] == "3.10"
    clone = next(c for c in commands if c[:2] == ["git", "clone"])
    assert clone[-1].endswith(str(Path("freeze-omni") / "163a2488" / "pytorch" / "src"))  # kept: it runs from there
    reqs = next(c for c in commands if "-r" in c)
    assert reqs[reqs.index("-r") + 1].endswith("requirements.txt")
    assert "torch==2.2.0" in next(c for c in commands if "--index-url" in c)
    check = commands[-1]
    assert "sys.path.insert" in check[-1] and "import flask_socketio, torchaudio" in check[-1]


# -- MiniCPM-o 4.5: bolo's own server around the model's duplex API ---------------------------------------------------
class FakeMiniCPMDuplex:
    """MiniCPMODuplex as its remote code shapes it: prepare(), then streaming_prefill / streaming_generate per second."""

    def __init__(self) -> None:
        self.prepared: list[dict[str, Any]] = []
        self.pieces: list[int] = []

    def prepare(self, prefix_system_prompt: Any = None, ref_audio: Any = None, prompt_wav_path: Any = None) -> str:
        self.prepared.append({"prompt": prefix_system_prompt, "voice": prompt_wav_path, "ref": len(ref_audio)})
        return ""

    def streaming_prefill(self, audio_waveform: Any = None, frame_list: Any = None) -> None:
        self.pieces.append(len(audio_waveform))

    def streaming_generate(self, **kwargs: Any) -> dict[str, Any]:
        speak = len(self.pieces) == 2  # it answers in its second step
        audio = (0.3 * np.sin(np.arange(24000) / 5)).astype(np.float32) if speak else None
        return {
            "is_listen": not speak,
            "text": "Hi there." if speak else "",
            "audio_waveform": audio,
            "end_of_turn": speak,
        }


def _minicpmo_server() -> Any:
    import importlib.util

    path = Path(rt.__file__).with_name("servers") / "minicpmo_duplex.py"
    spec = importlib.util.spec_from_file_location("minicpmo_duplex_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_minicpmo_duplex_server_speaks_kyutais_protocol() -> None:
    pytest.importorskip("sphn")
    from bolo.engines.tts_server import free_port

    server = _minicpmo_server()
    fake = FakeMiniCPMDuplex()
    conversations = server.Conversations(
        fake, {"ht": "/m/assets/HT.wav", "system": "/m/assets/system.wav"}, "ht", load_audio=lambda p: np.zeros(16000)
    )
    port = free_port()
    task = asyncio.create_task(server.run(conversations, "127.0.0.1", port))
    try:
        backend = KyutaiBackend(
            f"http://127.0.0.1:{port}", model="minicpm-o-4.5", voice_param="voice", default_voice="ht",
            instructions_param="text_prompt",
        )  # fmt: skip
        for _ in range(50):
            try:
                session = await backend.open(DuplexConfig(instructions="Be a tour guide.", voice="system"))
                break
            except EngineUnavailable:
                await asyncio.sleep(0.1)
        await _feed(session, tone(2.2, 24000))
        text, audio = "", 0
        async with asyncio.timeout(10):
            async for event in session.events():
                if isinstance(event, DuplexText):
                    text += event.text
                elif isinstance(event, DuplexAudio):
                    audio += len(event.samples)
                if text and audio >= 48000 - 4000:
                    break
        await session.close()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert fake.prepared == [{"prompt": "Be a tour guide.", "voice": "/m/assets/system.wav", "ref": 16000}]
    assert fake.pieces[:2] == [16560, 16000]  # the first piece is 1035 ms, then one second at a time
    assert text == "Hi there."


def test_minicpmo_remote_code_is_opt_in_and_checked(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from bolo.core import Bolo
    from bolo.duplex.manager import readiness, remote_code_problem

    monkeypatch.setattr(rt, "platform_key", lambda: "linux-x86_64")
    settings.s2s_model = "minicpm-o-4.5"
    core = Bolo(settings)
    spec = core.catalog.resolve("minicpm-o-4.5")
    assert spec is not None and spec.options["remote_code"]["files"]
    _install_fake(settings, "minicpm-o", "pytorch")
    monkeypatch.setattr(core.store, "is_installed", lambda name: True)
    configured, available, reason = readiness(core, spec)
    assert configured and not available and "allow_remote_code=true" in str(reason)
    core.store.set_model_settings(spec.id, {"allow_remote_code": True})
    assert readiness(core, spec) == (True, True, None)
    folder = tmp_path / "minicpm"
    pinned = {**spec.options["remote_code"]["files"], **spec.options["sha256"]}
    assert "unreviewed" not in str(remote_code_problem(core, spec, folder))  # missing files come first
    for name in pinned:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("changed", encoding="utf-8")
    assert "not the reviewed file" in str(remote_code_problem(core, spec, folder))
    # the same entry pinned to the files as they are now: accepted, until an unlisted .py file appears
    digest = {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in pinned}
    code = {n: d for n, d in digest.items() if n in spec.options["remote_code"]["files"]}
    options = {**spec.options, "remote_code": {**spec.options["remote_code"], "files": code},
               "sha256": {n: d for n, d in digest.items() if n not in code}}  # fmt: skip
    repinned = dataclasses.replace(spec, options=options)
    assert remote_code_problem(core, repinned, folder) is None
    (folder / "evil.py").write_text("", encoding="utf-8")
    assert "unreviewed code" in str(remote_code_problem(core, repinned, folder))


def test_minicpmo_launch_and_install(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo.catalog import Catalog

    monkeypatch.setattr(rt, "platform_key", lambda: "windows-x86_64")
    settings.device = "cuda"
    spec = Catalog.load(None).resolve("minicpm-o-4.5")
    assert spec is not None
    _install_fake(settings, "minicpm-o", "pytorch")
    path = _model_dir(tmp_path, spec)
    run = json.loads(rt.launch(settings, spec, path, 13, []).process.argv[4])
    assert run["module"].endswith("minicpmo_duplex.py") and Path(run["module"]).is_file()
    assert (
        run["args"][run["args"].count("--voice") and run["args"].index("--voice") + 1] == "ht=assets/HT_ref_audio.wav"
    )
    monkeypatch.setattr(rt.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "uv" else None)
    commands: list[list[str]] = []

    def run_step(cmd: list[str]) -> tuple[int, str]:
        commands.append(cmd)
        if cmd[1] == "venv":
            python = rt._venv_python(Path(cmd[-1]))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
        return 0, ""

    import shutil

    shutil.rmtree(rt.install_dir(settings, "minicpm-o", "pytorch"))
    assert rt.install(settings, "minicpm-o", None, lambda p, f: f / p.name, run_step) is None
    torch = next(c for c in commands if "--index-url" in c)
    assert "torch==2.8.0" in torch and "torchaudio==2.8.0" in torch
    assert any("minicpmo-utils[all]==1.0.6" in c and "transformers==4.51.0" in c for c in commands)
