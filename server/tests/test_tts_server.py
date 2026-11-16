"""The tts-server engine against fake llama-server / audiocpp_server processes (no downloads)."""

from __future__ import annotations

import base64
import io
import os
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from bolo import tts_runtimes
from bolo.catalog import ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.engines.base import LoadOptions, SynthesisExtras
from bolo.engines.tts_server import ServerProcess, ServerTTS, TTSServerEngine
from bolo.errors import ComponentMissing, EngineUnavailable, InvalidRequest
from conftest import LiveServer, wav_bytes
from fake_tts_server import FakeTTSServer

HERE = Path(__file__).parent


def entry(model_id: str, runtime: str = "llama-server", **options: object) -> dict[str, object]:
    return {
        "id": model_id,
        "task": "tts",
        "engine": "tts-server",
        "kind": "tts-server",
        "family": "Test",
        "languages": ["en", "hi"],
        "sample_rate": 24000,
        "license": "MIT",
        "requires": [runtime],
        "features": ["voice-cloning"],
        "source": {"local": "/nonexistent"},
        "voices": [
            {"id": "default", "name": "Model default", "server_voice": ""},
            {"id": "heart", "name": "Heart (English reference clip)", "reference": "en-short"},
        ],
        "options": {"runtime": runtime, "cloning": True, "send_language": True, **options},
    }


@pytest.fixture
def llama() -> Iterator[FakeTTSServer]:
    server = FakeTTSServer("llama", rate=16000, voices=[]).start()
    yield server
    server.stop()


@pytest.fixture
def audiocpp() -> Iterator[FakeTTSServer]:
    server = FakeTTSServer("audiocpp", rate=22050, voices=["narrator"], model_id="ac-model").start()
    yield server
    server.stop()


# -- connect mode through the HTTP API ------------------------------------------------------------------------
def test_speech_through_an_operator_run_llama_server(settings: Settings, llama: FakeTTSServer) -> None:
    settings.tts_servers = [f"remote-llama={llama.url}"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(entry("remote-llama")))
    srv.start()
    try:
        info = httpx.get(f"{srv.url}/api/v1/models/remote-llama").json()
        # served remotely: nothing to download, the runtime binary is not needed
        assert info["installed"] is True and info["available"] is True and info["requires"] == []
        r = httpx.post(
            f"{srv.url}/v1/audio/speech",
            json={"model": "remote-llama", "input": "Hello there.", "response_format": "wav", "language": "hi"},
            timeout=30,
        )
        assert r.status_code == 200, r.text
        audio, rate = sf.read(io.BytesIO(r.content))
        assert rate == 24000  # resampled from the server's 16 kHz to the catalog rate
        assert 0.4 < len(audio) / rate < 1.2
        sent = llama.requests[-1]
        assert sent["path"] == "/tts" and sent["body"]["stream"] is True and sent["body"]["response_format"] == "pcm16"
        assert sent["body"]["lang"] == "hi" and "voice" not in sent["body"]  # "default": the model's own voice
        assert "speaker_ref_b64" not in sent["body"]

        # a reference-clip voice is sent as the voice to imitate
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "remote-llama", "input": "Hi.", "voice": "heart"})
        assert r.status_code == 200
        assert base64.b64decode(llama.requests[-1]["body"]["speaker_ref_b64"])[:4] == b"RIFF"

        # cloning passthrough: the caller's own recording
        ref = base64.b64encode(wav_bytes(2.0)).decode()
        r = httpx.post(
            f"{srv.url}/v1/audio/speech",
            json={
                "model": "remote-llama",
                "input": "Hi.",
                "voice": "alloy",
                "reference_audio": f"data:audio/wav;base64,{ref}",
            },
        )
        assert r.status_code == 200
        sent_ref = base64.b64decode(llama.requests[-1]["body"]["speaker_ref_b64"])
        assert sf.info(io.BytesIO(sent_ref)).duration == pytest.approx(2.0, abs=0.05)

        # streaming as SSE works the same way
        with httpx.stream(
            "POST", f"{srv.url}/v1/audio/speech",
            json={"model": "remote-llama", "input": "Stream me.", "response_format": "pcm", "stream_format": "sse"},
        ) as resp:  # fmt: skip
            events = [line for line in resp.iter_lines() if line.startswith("data: ")]
        assert any("speech.audio.delta" in e for e in events) and "speech.audio.done" in events[-1]

        # the server's own errors keep their meaning
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "remote-llama", "input": "bad request"})
        assert r.status_code == 400 and "not a voice" in r.json()["error"]["message"]
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "remote-llama", "input": "Hi.", "speed": 1.5})
        assert r.status_code == 400 and r.json()["error"]["param"] == "speed"
        r = httpx.post(
            f"{srv.url}/v1/audio/speech", json={"model": "remote-llama", "input": "Hi.", "reference_audio": "@@"}
        )
        assert r.status_code == 400 and r.json()["error"]["param"] == "reference_audio"

        # voices listing
        voices = httpx.get(f"{srv.url}/v1/audio/voices").json()["voices"]
        assert {"default", "heart"} <= {v["id"] for v in voices if v["model_id"] == "remote-llama"}

        # native WebSocket streaming
        from websockets.sync.client import connect

        with connect(f"{srv.ws_url}/api/v1/tts/stream?model=remote-llama&format=pcm_s16le&sample_rate=16000") as ws:
            ready = ws.recv()
            assert '"ready"' in str(ready)
            ws.send('{"type":"text","text":"Hello from the socket."}')
            ws.send('{"type":"stop"}')
            got = sum(len(m) for m in ws if isinstance(m, bytes))
        assert got > 16000  # more than half a second of 16 kHz PCM16
    finally:
        srv.stop()


def test_server_gone_is_engine_unavailable(settings: Settings) -> None:
    settings.tts_servers = ["remote-llama=http://127.0.0.1:9"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(entry("remote-llama")))
    srv.start()
    try:
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "remote-llama", "input": "Hi."}, timeout=30)
        assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable"
        assert "not reachable" in r.json()["error"]["message"]
    finally:
        srv.stop()


async def test_runtime_missing_is_component_missing(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    core.catalog.add(ModelSpec.from_dict(entry("managed-llama")))
    info = core.model_api(core.catalog.resolve("managed-llama"))  # type: ignore[arg-type]
    assert info["requires"] == ["llama-server"] and info["available"] is False
    with pytest.raises(ComponentMissing, match="bolo components install llama-server"):
        await core.synthesize_all("Hello.", model="managed-llama")
    listing = {c["id"]: c for c in core.components.status()}
    assert listing["llama-server"]["kind"] == "tts-server" and listing["llama-server"]["installed"] is False
    assert listing["llama-server"]["license"] == "MIT" and listing["audio.cpp"]["license"] == "Apache-2.0"
    assert listing["llama-server"]["required_by_count"] >= 1


def test_audiocpp_connect_offline_and_streaming(audiocpp: FakeTTSServer) -> None:
    spec = {**entry("ac", "audio.cpp", server_model="ac-model", speed=True, request={"exaggeration": 0.7})}
    options = dict(spec["options"])  # type: ignore[call-overload]
    options.update(voices=spec["voices"], model_id="ac", sample_rate=24000)
    model = ServerTTS("audio.cpp", "ac", options, base_url=audiocpp.url)
    model.connect()
    assert "narrator" in {v.id for v in model.voices}  # listed by the server
    out = np.concatenate(
        list(model.synthesize("Hello world.", "narrator", 1.25, extras=SynthesisExtras(language="hi-IN")))
    )
    assert len(out) == pytest.approx(24000 * 0.05 * 12, rel=0.02)
    body = audiocpp.requests[-1]["body"]
    assert body == {
        "model": "ac-model", "input": "Hello world.", "response_format": "wav", "language": "hi", "voice": "narrator",
        "speed": 1.25, "options": {"exaggeration": 0.7},
    }  # fmt: skip
    list(model.synthesize("Hi.", "heart"))
    body = audiocpp.requests[-1]["body"]
    assert body["voice_ref"]["type"] == "base64" and body["reference_text"].startswith("Good morning")
    # unknown OpenAI voice names are dropped rather than refused by the server
    list(model.synthesize("Hi.", "alloy"))
    assert "voice" not in audiocpp.requests[-1]["body"]
    # families that declare a "language" request option get it there as well
    model.spec["language_option"] = True
    list(model.synthesize("Hi.", extras=SynthesisExtras(language="hi")))
    assert audiocpp.requests[-1]["body"]["options"] == {"exaggeration": 0.7, "language": "hi"}
    model.close()

    streaming = ServerTTS(
        "audio.cpp", "ac", {**options, "stream": True, "server_sample_rate": 22050}, base_url=audiocpp.url
    )
    pieces = list(streaming.synthesize("Streaming text."))
    assert len(pieces) >= 2 and sum(len(p) for p in pieces) == pytest.approx(24000 * 0.05 * 15, rel=0.02)
    assert audiocpp.requests[-1]["body"]["stream_format"] == "audio"
    wrong = ServerTTS("audio.cpp", "ac", {**options, "server_model": "other"}, base_url=audiocpp.url)
    with pytest.raises(EngineUnavailable, match="does not serve 'other'"):
        list(wrong.synthesize("Hi."))
    streaming.close()
    wrong.close()


def test_cloning_refused_by_models_that_cannot(llama: FakeTTSServer) -> None:
    model = ServerTTS("llama-server", "x", {"cloning": False, "sample_rate": 16000}, base_url=llama.url)
    with pytest.raises(InvalidRequest, match="cannot imitate"):
        list(model.synthesize("Hi.", extras=SynthesisExtras(reference_audio=wav_bytes(1.0))))
    # a float PCM answer and an exact rate need no resampling
    model.spec["request"] = {"response_format": "pcm"}
    out = np.concatenate(list(model.synthesize("Hello.")))
    assert len(out) == int(16000 * 0.05 * 6)
    model.close()


# -- managed mode: bolo starts the binary ---------------------------------------------------------------------
def fake_binary(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "bin" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(HERE)!r})\n"
                    "from fake_tts_server import main\nmain(sys.argv[1:])\n")  # fmt: skip
    path.chmod(0o755)
    return path


def test_managed_llama_server_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LOAD_DELAY", "0.6")  # /health answers 503 "Loading model" first
    monkeypatch.setenv("FAKE_VOICES", "female,male")
    settings = Settings(data_dir=tmp_path / "data", llama_server_bin=str(fake_binary(tmp_path, "llama-server")))
    engine = TTSServerEngine()
    engine.configure(settings)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_bytes(b"GGUF")
    (model_dir / "mm.gguf").write_bytes(b"GGUF")
    spec = {
        "runtime": "llama-server",
        "model_file": "m.gguf",
        "mmproj_file": "mm.gguf",
        "model_id": "managed",
        "sample_rate": 16000,
    }
    model = engine.load("tts-server", "tts", model_dir, spec, LoadOptions(threads=2))
    assert isinstance(model, ServerTTS) and model.process is not None
    proc = model.process
    try:
        assert proc.running and proc.api_key  # started with a random API key, which bolo sends
        assert {v.id for v in model.voices} == {"female", "male"}  # from GET /props
        assert "-mm" in proc.argv and "--api-key" in proc.argv and "127.0.0.1" in proc.argv
        out = np.concatenate(list(model.synthesize("Hello.", "male")))
        assert len(out) == int(16000 * 0.05 * 6)
        # the process dies: the request fails as engine_unavailable, the next one waits for the back-off
        with pytest.raises(EngineUnavailable):
            list(model.synthesize("crash now"))
        time.sleep(0.2)
        with pytest.raises(EngineUnavailable, match="restarting in"):
            list(model.synthesize("Hello."))
        assert proc.crashes == 1
        time.sleep(1.1)
        out = np.concatenate(list(model.synthesize("Hello again.")))
        assert len(out) > 0 and proc.restarts == 1 and proc.running
    finally:
        model.close()
    assert not proc.running


def test_managed_audiocpp_writes_its_config(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", audiocpp_server_bin=str(fake_binary(tmp_path, "audiocpp_server")))
    engine = TTSServerEngine()
    engine.configure(settings)
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "voice.gguf").write_bytes(b"GGUF")
    spec = {
        "runtime": "audio.cpp", "family": "pocket_tts", "model_file": "voice.gguf", "model_id": "pocket",
        "load_options": {"language": "english"}, "sample_rate": 24000,
    }  # fmt: skip
    model = engine.load("tts-server", "tts", model_dir, spec, LoadOptions(threads=3, device="cpu"))
    try:
        import json

        config = json.loads((tmp_path / "data" / "run" / "tts-server" / "pocket.json").read_text())
        assert config["backend"] == "cpu" and config["threads"] == 3 and config["host"] == "127.0.0.1"
        assert config["models"] == [
            {"id": "pocket", "family": "pocket_tts", "path": str(model_dir / "voice.gguf"), "task": "tts",
             "mode": "offline", "load_options": {"language": "english"}},
        ]  # fmt: skip
        assert len(np.concatenate(list(model.synthesize("Hello.")))) == int(24000 * 0.05 * 6)
    finally:
        engine.shutdown()


def test_server_that_never_starts(tmp_path: Path) -> None:
    script = tmp_path / "broken"
    script.write_text("#!/bin/sh\necho 'error: unknown model architecture' >&2\nexit 2\n")
    script.chmod(0o755)
    proc = ServerProcess("broken", [str(script)], 1, start_timeout_s=5)
    with pytest.raises(EngineUnavailable, match="unknown model architecture"):
        proc.ensure_started()
    with pytest.raises(EngineUnavailable, match="restarting in"):
        proc.ensure_started()


# -- settings, binaries, release archives -------------------------------------------------------------------
def test_settings_and_binary_discovery(tmp_path: Path) -> None:
    s = load_settings(
        overrides={"tts_servers": "llama-server=http://a:1/, voxcpm2-audiocpp=http://b:2", "data_dir": tmp_path},
        env={},
    )
    assert s.tts_servers == ["llama-server=http://a:1/", "voxcpm2-audiocpp=http://b:2"]
    spec = ModelSpec.from_dict(entry("any-llama"))
    assert tts_runtimes.remote_url(s, spec) == "http://a:1"
    spec = ModelSpec.from_dict(entry("voxcpm2-audiocpp", "audio.cpp"))
    assert tts_runtimes.remote_url(s, spec) == "http://b:2"
    with pytest.raises(ValueError):
        tts_runtimes.parse_server_setting("no-url")
    assert tts_runtimes.find_binary(Settings(data_dir=tmp_path), "audio.cpp")[0] is None or os.environ.get("PATH")
    installed = tts_runtimes.install_dir(s, "llama-server") / "llama-b1" / "llama-server"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\necho 'version: 1 (abc)'\n")
    installed.chmod(0o755)
    path, source = tts_runtimes.find_binary(Settings(data_dir=tmp_path), "llama-server")
    assert path == installed and source == "bolo"
    assert tts_runtimes.binary_version(installed) == "1 (abc)"


def test_release_archive_extraction(tmp_path: Path) -> None:
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as tf:
        for name, data, mode in (
            ("llama-b1/llama-server", b"#!/bin/sh\n", 0o755),
            ("llama-b1/libllama.so.1.0", b"x", 0o644),
        ):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("llama-b1/libllama.so")
        link.type, link.linkname = tarfile.SYMTYPE, "libllama.so.1.0"
        tf.addfile(link)
    dest = tmp_path / "out"
    tts_runtimes.extract_release(good, dest)
    assert os.access(dest / "llama-b1" / "llama-server", os.X_OK)
    assert (dest / "llama-b1" / "libllama.so").resolve() == (dest / "llama-b1" / "libllama.so.1.0").resolve()
    for linkname in ("/etc/passwd", "../../evil"):
        bad = tmp_path / "bad.tar.gz"
        with tarfile.open(bad, "w:gz") as tf:
            link = tarfile.TarInfo("x/lib.so")
            link.type, link.linkname = tarfile.SYMTYPE, linkname
            tf.addfile(link)
        with pytest.raises(ValueError, match="unsafe"):
            tts_runtimes.extract_release(bad, tmp_path / "bad")


async def test_installing_a_runtime_checks_the_checksum(
    core: Bolo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        data = b"#!/bin/sh\necho 'version: 9'\n"
        info = tarfile.TarInfo("llama-b9/llama-server")
        info.size, info.mode = len(data), 0o755
        tf.addfile(info, io.BytesIO(data))
    payload = archive.getvalue()
    served = FakeArchive(payload).start()
    try:
        key = tts_runtimes.platform_key()
        release = tts_runtimes.Release(served.url + "/llama.tar.gz", hashlib.sha256(payload).hexdigest(), len(payload))
        runtime = tts_runtimes.RUNTIME_INFO["llama-server"]
        monkeypatch.setitem(runtime.releases, key, tts_runtimes.Release(release.url, "0" * 64, len(payload)))
        monkeypatch.setenv("PATH", "/nonexistent")
        error = core.components.install_blocking(component="llama-server")
        assert error and "checksum mismatch" in error and not core.components.is_installed("llama-server")
        monkeypatch.setitem(runtime.releases, key, release)
        job = core.components.start_install("llama-server", accept_license=False)
        for _ in range(100):
            if job.state in ("done", "error"):
                break
            import asyncio

            await asyncio.sleep(0.05)
        assert job.state == "done", job.error
        assert core.components.is_installed("llama-server")
        listing = {c["id"]: c for c in core.components.status()}
        assert listing["llama-server"]["source"] == "bolo" and listing["llama-server"]["version"] == runtime.version
    finally:
        served.stop()


class FakeArchive:
    def __init__(self, payload: bytes) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> FakeArchive:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# -- Tontaube ---------------------------------------------------------------------------------------------------
def tontaube_spec(**extra: object) -> dict[str, object]:
    return {
        "runtime": "tontaube", "cloning": True, "sample_rate": 24000, "model_id": "tontaube-test",
        "voices": [
            {"id": "marcus", "name": "Marcus", "language": "en", "server_voice": "samples/Marcus.mp3"},
            {"id": "juri", "name": "Juri", "language": "de", "server_voice": "samples/Juri.mp3", "style": "audiobook"},
        ],
        **extra,
    }  # fmt: skip


def test_tontaube_requests_voices_styles_and_sliders() -> None:
    server = FakeTTSServer("tontaube", api_key="k").start()
    try:
        model = ServerTTS("tontaube", "tontaube-test", tontaube_spec(), base_url=server.url, api_key="k")
        model.connect()  # GET /readyz
        assert model.request_options[0] == "style" and model.supports_cloning
        out = np.concatenate(list(model.synthesize("Hello there.", "marcus")))
        assert len(out) == int(24000 * 0.05 * 12)
        assert server.requests[-1]["body"] == {
            "text": "Hello there.", "language": "en", "tag": "conversational", "format": "wav",
            "voice_path": "samples/Marcus.mp3",
        }  # fmt: skip
        # a voice's own style and language; sliders, temperature and seed from the request options
        extras = SynthesisExtras(options={"rate": 1, "pitch_range": 5, "legato": 0, "seed": 7})
        list(model.synthesize("Guten Tag.", "juri", extras=extras))
        body = server.requests[-1]["body"]
        assert body["tag"] == "audiobook" and body["language"] == "de" and body["seed"] == 7
        assert body["controls"] == {"rate": 1, "pitch_range": 5, "legato": 0}
        # a reference recording replaces the bundled voice; style chosen per request
        extras = SynthesisExtras(language="nl", reference_audio=wav_bytes(2.0), options={"style": "agentic"})
        list(model.synthesize("Hallo.", None, extras=extras))
        body = server.requests[-1]["body"]
        assert body["voice_audio_b64"] and "voice_path" not in body and body["tag"] == "agentic"
        assert body["language"] == "nl"
        for bad, match in (
            (SynthesisExtras(options={"rate": 9}), "rate is a step from 0 to 4"),
            (SynthesisExtras(options={"style": "shouting"}), "style must be one of"),
            (SynthesisExtras(language="hi"), "not 'hi'"),
        ):
            with pytest.raises(InvalidRequest, match=match):
                list(model.synthesize("Hi.", "marcus", extras=bad))
        with pytest.raises(InvalidRequest, match="option rate"):
            list(model.synthesize("Hi.", "marcus", 1.5))
        model.close()
        # a wrong key: the server's 401 is engine_unavailable with a hint
        wrong = ServerTTS("tontaube", "tontaube-test", tontaube_spec(), base_url=server.url, api_key="nope")
        with pytest.raises(EngineUnavailable, match="credentials"):
            list(wrong.synthesize("Hi.", "marcus"))
        wrong.close()
    finally:
        server.stop()


def test_managed_tontaube_runs_uv_with_the_downloaded_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = fake_binary(tmp_path, "uv").parent
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_ENV_FILE", str(tmp_path / "env.json"))
    monkeypatch.setenv("FAKE_LOAD_DELAY", "0.4")  # /readyz answers 503 while the model loads
    checkout = tmp_path / "tontaube"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'tontaube-inference'\n")
    settings = Settings(data_dir=tmp_path / "data", tontaube_dir=str(checkout))
    assert tts_runtimes.find_binary(settings, "tontaube") == (checkout, "settings")
    engine = TTSServerEngine()
    engine.configure(settings)
    weights = tmp_path / "weights"
    (weights / "cb0").mkdir(parents=True)
    model = engine.load("tts-server", "tts", weights, tontaube_spec(), LoadOptions(threads=3, device="cpu"))
    try:
        assert isinstance(model, ServerTTS) and model.process is not None
        argv = model.process.argv
        assert argv[1:6] == ["run", "--python", "3.12", "--frozen", "--directory"] and argv[6] == str(checkout)
        assert argv[7:9] == ["tontaube", "serve"] and model.process.health_path == "/readyz"
        import json

        env = json.loads((tmp_path / "env.json").read_text())
        assert env["MODEL_PATH"] == str(weights) and env["LITE_DEVICE"] == "cpu" and env["LITE_CPU_THREADS"] == "3"
        assert env["REQUIRE_API_KEY"] == "1" and env["TTS_API_KEY"] == model.process.api_key
        assert len(np.concatenate(list(model.synthesize("Hello.", "marcus")))) == int(24000 * 0.05 * 6)
    finally:
        engine.shutdown()


def test_tontaube_component_is_a_pinned_checkout(tmp_path: Path, core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    commit = tts_runtimes.RUNTIME_INFO["tontaube"].git.commit  # type: ignore[union-attr]

    def fake_run(cmd: list[str]) -> tuple[int, str]:
        calls.append(cmd)
        if cmd[1] == "clone":
            Path(cmd[-1]).mkdir(parents=True)
            (Path(cmd[-1]) / "pyproject.toml").write_text("")
            return 0, ""
        return 0, commit + "\n"

    monkeypatch.setattr(core.components, "_run", fake_run)
    status = {c["id"]: c for c in core.components.runtime_status()}["tontaube"]
    assert not status["installed"] and status["release"]["commit"] == commit
    assert core.components.install_runtime_blocking("tontaube") is None
    assert calls[0][:6] == ["git", "clone", "--quiet", "--depth", "1", "--branch"] and "v1.1.0" in calls[0]
    assert core.components.is_installed("tontaube")
    # another commit behind the tag is refused
    monkeypatch.setattr(core.components, "_run", lambda cmd: (0, "") if cmd[1] == "clone" else (0, "0" * 40))
    import shutil

    shutil.rmtree(tts_runtimes.install_dir(core.settings, "tontaube"))
    assert "not the pinned commit" in str(core.components.install_runtime_blocking("tontaube"))
    assert not core.components.is_installed("tontaube")
