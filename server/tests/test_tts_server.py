"""The tts-server engine against fake llama-server / audiocpp_server processes (no downloads)."""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
from fakebin import fake_program, printing_program

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
    return fake_program(tmp_path / "bin" / name, f"import sys\nsys.path.insert(0, {str(HERE)!r})\n"
                        "from fake_tts_server import main\nmain(sys.argv[1:])\n")  # fmt: skip


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

        config = json.loads((tmp_path / "data" / "run" / "tts-server" / "pocket.json").read_text(encoding="utf-8"))
        assert config["backend"] == "cpu" and config["threads"] == 3 and config["host"] == "127.0.0.1"
        assert config["models"] == [
            {"id": "pocket", "family": "pocket_tts", "path": str(model_dir / "voice.gguf"), "task": "tts",
             "mode": "offline", "load_options": {"language": "english"}},
        ]  # fmt: skip
        assert len(np.concatenate(list(model.synthesize("Hello.")))) == int(24000 * 0.05 * 6)
    finally:
        engine.shutdown()


def test_audiocpp_backend_names(tmp_path: Path) -> None:
    """audio.cpp knows cpu, cuda, hip, vulkan and metal and refuses anything else, so dml runs it on the CPU."""
    (tmp_path / "voice.gguf").write_bytes(b"GGUF")
    spec = {"model_file": "voice.gguf", "family": "pocket_tts"}
    for device, backend in (
        ("cuda:1", "cuda"),
        ("rocm", "hip"),
        ("coreml", "metal"),
        ("dml:0", "cpu"),
        ("auto", "cpu"),
    ):
        config = TTSServerEngine._audiocpp_config("v", tmp_path, spec, LoadOptions(threads=1, device=device), 1)
        assert config["backend"] == backend, device


def test_server_that_never_starts(tmp_path: Path) -> None:
    script = printing_program(tmp_path / "broken", stderr="error: unknown model architecture\n", code=2)
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
    installed = printing_program(
        tts_runtimes.install_dir(s, "llama-server") / "llama-b1" / "llama-server", "version: 1 (abc)\n"
    )
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


def fake_release(kind: str, version: str = "9", library: str = "ggml-base.dll") -> bytes:
    """A release archive holding a llama-server that prints its version: a tarball with an executable script, or
    on Windows (and for ``kind="zip"``) a flat zip like llama.cpp's Windows builds, with ``library`` next to it."""
    data = f"#!/bin/sh\necho 'version: {version}'\n".encode()
    name = "llama-server"
    if os.name == "nt":
        data, name = f"@echo version: {version}\r\n".encode(), "llama-server.cmd"
    archive = io.BytesIO()
    if kind == "zip" or os.name == "nt":
        with zipfile.ZipFile(archive, "w") as zf:
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o755 << 16
            zf.writestr(info, data)
            zf.writestr(library, b"MZ")
    else:
        with tarfile.open(fileobj=archive, mode="w:gz") as tf:
            info = tarfile.TarInfo(f"llama-b9/{name}")
            info.size, info.mode = len(data), 0o755
            tf.addfile(info, io.BytesIO(data))
    return archive.getvalue()


async def wait_for(job: Any) -> None:
    import asyncio

    for _ in range(200):
        if job.state in ("done", "error"):
            return
        await asyncio.sleep(0.05)


async def test_installing_a_runtime_checks_the_checksum(
    core: Bolo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    payload = fake_release("tar")
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
        await wait_for(job)
        assert job.state == "done", job.error
        assert core.components.is_installed("llama-server")
        listing = {c["id"]: c for c in core.components.status()}
        assert listing["llama-server"]["source"] == "bolo" and listing["llama-server"]["version"] == runtime.version
    finally:
        served.stop()


async def test_installing_the_cuda_build_with_its_runtime(
    core: Bolo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """device = "cuda" picks the CUDA build where one is pinned (Windows), and its CUDA runtime archive is unpacked
    next to it, in the CUDA build's own folder."""
    import hashlib

    main = fake_release("zip", library="ggml-cuda.dll")
    cudart = io.BytesIO()
    with zipfile.ZipFile(cudart, "w") as zf:
        zf.writestr("cudart64_12.dll", b"MZ cudart")
    extra = cudart.getvalue()
    served = {"/main.zip": main, "/cudart.zip": extra}
    server = FakeArchive(served).start()
    try:
        key = "testos-x86_64"  # a platform without real pins, so that only these releases exist
        monkeypatch.setattr(tts_runtimes, "platform_key", lambda: key)
        runtime = tts_runtimes.RUNTIME_INFO["llama-server"]
        cpu = tts_runtimes.Release(server.url + "/cpu.zip", "0" * 64, 1)
        cuda = tts_runtimes.Release(
            server.url + "/main.zip", hashlib.sha256(main).hexdigest(), len(main),
            extras=(tts_runtimes.Release(server.url + "/cudart.zip", hashlib.sha256(extra).hexdigest(), len(extra)),),
        )  # fmt: skip
        monkeypatch.setitem(runtime.releases, key, cpu)
        monkeypatch.setitem(runtime.releases, key + "-cuda", cuda)
        assert tts_runtimes.release_for("llama-server", "cpu") is cpu
        assert tts_runtimes.release_for("llama-server", "cuda:1") is cuda
        assert tts_runtimes.release_for("llama-server", "vulkan") is cpu  # no Vulkan build pinned here
        assert tts_runtimes.release_for("llama-server", "cpu", flavour="cuda") is cuda
        assert tts_runtimes.pinned_flavours("llama-server") == ["cpu", "cuda"]
        assert tts_runtimes.missing_build_note("llama-server", "cuda") is None
        assert tts_runtimes.missing_build_note("llama-server", "cpu") is None
        note = tts_runtimes.missing_build_note("llama-server", "vulkan")
        assert note and "no prebuilt vulkan build" in note and "--flavour cuda" in note and "llama_server_bin" in note
        monkeypatch.setenv("PATH", "/nonexistent")
        core.settings.device = "cuda"
        job = core.components.start_install("llama-server", accept_license=False)
        await wait_for(job)
        assert job.state == "done", job.error
        folder = tts_runtimes.install_dir(core.settings, "llama-server", "cuda")
        assert (folder / "cudart64_12.dll").read_bytes() == b"MZ cudart" and (folder / "ggml-cuda.dll").is_file()
        path, source = tts_runtimes.find_binary(core.settings, "llama-server")
        assert source == "bolo" and path is not None and path.parent == folder
        assert tts_runtimes.binary_version(path) == "9"
    finally:
        server.stop()


def _serve_builds(monkeypatch: pytest.MonkeyPatch, runtime_id: str = "llama-server") -> tuple[Any, dict[str, Any]]:
    """Fake CPU, CUDA and Vulkan builds of ``runtime_id`` (each printing its own version) pinned for a test
    platform."""
    import hashlib

    archives = {
        "cpu": fake_release("zip", "cpu-1", "ggml-cpu.dll"),
        "cuda": fake_release("zip", "cuda-1", "ggml-cuda.dll"),
        "vulkan": fake_release("zip", "vulkan-1", "ggml-vulkan.dll"),
    }
    server = FakeArchive({f"/{k}.zip": v for k, v in archives.items()}).start()
    key = "testos-x86_64"
    monkeypatch.setattr(tts_runtimes, "platform_key", lambda: key)
    runtime = tts_runtimes.RUNTIME_INFO[runtime_id]
    for flavour, payload in archives.items():
        release = tts_runtimes.Release(f"{server.url}/{flavour}.zip", hashlib.sha256(payload).hexdigest(), len(payload))
        monkeypatch.setitem(runtime.releases, key if flavour == "cpu" else f"{key}-{flavour}", release)
    monkeypatch.setenv("PATH", "/nonexistent")
    return server, archives


@pytest.mark.parametrize("runtime_id", ["llama-server", "audio.cpp"])
async def test_each_build_in_its_own_folder(core: Bolo, monkeypatch: pytest.MonkeyPatch, runtime_id: str) -> None:
    """The CPU build installed first does not make the CUDA build "already installed": each build has its own
    folder, the one matching ``device`` is started, and ``flavour`` picks one explicitly."""
    info = tts_runtimes.RUNTIME_INFO[runtime_id]
    monkeypatch.setitem(  # the fake archives hold "llama-server": let audio.cpp's binary have that name here
        tts_runtimes.RUNTIME_INFO, runtime_id, tts_runtimes.Runtime(**{**info.__dict__, "binary": "llama-server"})
    )
    server, _ = _serve_builds(monkeypatch, runtime_id)
    try:
        s = core.settings
        s.device = "cpu"
        assert core.components.runtime_needs_install(runtime_id)
        assert core.components.install_blocking(component=runtime_id) is None
        assert not core.components.runtime_needs_install(runtime_id)
        s.device = "cuda:0"
        assert core.components.runtime_needs_install(runtime_id)  # the CPU build is not the CUDA build
        assert tts_runtimes.flavour_for(runtime_id, s.device) == "cuda"
        path, _ = tts_runtimes.find_binary(s, runtime_id)  # until it is installed, the CPU build runs
        assert path is not None and tts_runtimes.binary_version(path) == "cpu-1"
        job = core.components.start_install(runtime_id, accept_license=False)
        await wait_for(job)
        assert job.state == "done", job.error
        assert sorted(tts_runtimes.installed_flavours(s, runtime_id)) == ["cpu", "cuda"]
        path, _ = tts_runtimes.find_binary(s, runtime_id)
        assert path is not None and tts_runtimes.binary_version(path) == "cuda-1"
        assert path.is_relative_to(tts_runtimes.install_dir(s, runtime_id, "cuda"))
        s.device = "cpu"
        path, _ = tts_runtimes.find_binary(s, runtime_id)
        assert path is not None and tts_runtimes.binary_version(path) == "cpu-1"
        s.device = "auto"  # the GPU build that was installed
        assert tts_runtimes.binary_version(tts_runtimes.find_binary(s, runtime_id)[0]) == "cuda-1"  # type: ignore[arg-type]
        s.device = "cpu"
        with pytest.raises(InvalidRequest, match="unknown flavour"):
            core.components.start_install(runtime_id, accept_license=False, flavour="metal")
        job = core.components.start_install(runtime_id, accept_license=False, flavour="vulkan")
        await wait_for(job)
        assert job.state == "done", job.error
        assert tts_runtimes.find_binary(s, runtime_id, "vulkan")[0] is not None
        listing = {c["id"]: c for c in core.components.status()}[runtime_id]
        assert listing["flavour"] == "cpu" and listing["flavours"] == ["cpu", "cuda", "vulkan"]
        assert listing["wanted_flavour"] == "cpu" and not listing["installable"]
    finally:
        server.stop()


def test_an_install_from_before_the_build_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Files unpacked right in the version folder (bolo before the per-build folders) count as the build their
    libraries show, and move into that build's folder before another build is installed."""
    monkeypatch.setattr(tts_runtimes, "platform_key", lambda: "windows-x86_64")
    monkeypatch.setenv("PATH", "")
    settings = Settings(data_dir=tmp_path, device="cuda")
    base = tts_runtimes.install_dir(settings, "llama-server")
    binary = printing_program(base / "llama-server", "version: old\n")
    (base / "ggml-cuda.dll").write_bytes(b"MZ")
    assert tts_runtimes.installed_flavours(settings, "llama-server") == {"cuda": binary}
    assert tts_runtimes.find_binary(settings, "llama-server") == (binary, "bolo")
    tts_runtimes.migrate_legacy(settings, "llama-server")
    moved = tts_runtimes.installed_flavours(settings, "llama-server")["cuda"]
    assert moved.parent == base / "cuda" and (base / "cuda" / "ggml-cuda.dll").is_file()
    assert moved.name == binary.name  # (on Windows the stand-in's launcher names its script by an absolute path)
    from bolo import hardware

    state = hardware._server_runtime(hardware.Host(), settings, "llama-server", deep=False)
    assert state["flavour"] == "cuda" and state["gpu"] is True  # bolo recommend counts it as a GPU build
    other = Settings(data_dir=tmp_path / "plain", device="cpu")
    printing_program(tts_runtimes.install_dir(other, "llama-server") / "llama-b1" / "llama-server", "version: 1\n")
    assert list(tts_runtimes.installed_flavours(other, "llama-server")) == ["cpu"]


def test_cli_installs_the_build_device_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`BOLO_DEVICE=cuda bolo components install llama-server` after the CPU build installs the CUDA build;
    `--flavour` overrides; `bolo components list` names the build."""
    from bolo.cli import main

    server, _ = _serve_builds(monkeypatch)
    try:
        monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("BOLO_DEVICE", "cpu")
        assert main(["components", "install", "llama-server"]) == 0
        assert main(["components", "install", "llama-server"]) == 0
        assert "llama-server (cpu build) is already installed" in capsys.readouterr().out
        monkeypatch.setenv("BOLO_DEVICE", "cuda")
        assert main(["components", "install", "llama-server"]) == 0
        assert "already installed" not in capsys.readouterr().out
        assert main(["components", "list"]) == 0
        assert "cuda build" in capsys.readouterr().out
        assert main(["components", "install", "llama-server", "--flavour", "vulkan"]) == 0
        assert main(["components", "install", "espeak-ng", "--flavour", "cuda"]) == 1
        assert "--flavour is for llama-server" in capsys.readouterr().err
        data = Settings(data_dir=tmp_path)
        assert sorted(tts_runtimes.installed_flavours(data, "llama-server")) == ["cpu", "cuda", "vulkan"]
    finally:
        server.stop()


def test_windows_release_pins() -> None:
    """The Windows builds are pinned for the same llama.cpp and audio.cpp releases as Linux and macOS."""
    for runtime in ("llama-server", "audio.cpp"):
        info = tts_runtimes.RUNTIME_INFO[runtime]
        for flavour in ("windows-x86_64", "windows-x86_64-cuda", "windows-x86_64-vulkan"):
            release = info.releases[flavour]
            assert info.version in release.url and release.url.endswith(".zip") and len(release.sha256) == 64
        cuda = info.releases["windows-x86_64-cuda"]
        assert "12.4" in cuda.url and len(cuda.extras) == 1 and "cudart" in cuda.extras[0].url
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_runtimes, "platform_key", lambda: "windows-x86_64")
        assert "win-cuda-12.4" in str(tts_runtimes.release_for("llama-server", "cuda").url)  # type: ignore[union-attr]
        assert "win-vulkan" in str(tts_runtimes.release_for("llama-server", "vulkan").url)  # type: ignore[union-attr]
        assert "win-cpu-x64" in str(tts_runtimes.release_for("llama-server", "dml").url)  # type: ignore[union-attr]
        assert "cpu-portable" in str(tts_runtimes.release_for("audio.cpp", "cpu").url)  # type: ignore[union-attr]


def test_windows_binary_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows the components folder and the setting find llama-server.exe for "llama-server" (PATHEXT)."""
    monkeypatch.setattr(tts_runtimes, "_windows", lambda: True)
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    settings = Settings(data_dir=tmp_path)
    folder = tts_runtimes.install_dir(settings, "llama-server")
    folder.mkdir(parents=True)
    (folder / "llama-server-impl.dll").write_bytes(b"MZ")
    (folder / "llama-server.exe").write_bytes(b"MZ")
    monkeypatch.setenv("PATH", "")
    assert tts_runtimes.find_binary(settings, "llama-server") == (folder / "llama-server.exe", "bolo")
    assert tts_runtimes.executable_names("audiocpp_server")[:2] == ["audiocpp_server.com", "audiocpp_server.exe"]
    configured = Settings(data_dir=tmp_path, llama_server_bin=str(folder / "llama-server"))
    assert tts_runtimes.find_binary(configured, "llama-server") == (folder / "llama-server.exe", "settings")
    assert not tts_runtimes.is_executable(folder / "llama-server-impl.dll")


class FakeArchive:
    def __init__(self, payload: bytes | dict[str, bytes]) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                body = payload if isinstance(payload, bytes) else payload.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

        env = json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))
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


def test_audiocpp_families_that_refuse_the_reference_text() -> None:
    spec = {"cloning": True, "sample_rate": 24000, "reference_text": False, "model_id": "x"}
    model = ServerTTS("audio.cpp", "x", spec, base_url="http://127.0.0.1:9")
    extras = SynthesisExtras(reference_audio=wav_bytes(1.0), reference_text="the words")
    body = model.build_request("Hi.", None, 1.0, extras)
    assert body["voice_ref"]["type"] == "base64" and "reference_text" not in body
    model.spec["reference_text"] = True
    assert model.build_request("Hi.", None, 1.0, extras)["reference_text"] == "the words"
    model.close()


async def test_a_gpu_device_without_a_pinned_gpu_build_says_so(core: Bolo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux has no prebuilt CUDA build of llama.cpp: device = "cuda" installs the CPU build, and says so (job
    warning, log and command line) instead of reporting a plain install; the Vulkan build is named as the way to
    the GPU."""
    import hashlib

    assert "linux-x86_64-vulkan" in tts_runtimes.RUNTIME_INFO["llama-server"].releases  # the real pin
    payload = fake_release("zip", "cpu-1", "ggml-cpu.dll")
    vk = fake_release("zip", "vulkan-1", "ggml-vulkan.dll")
    server = FakeArchive({"/cpu.zip": payload, "/vulkan.zip": vk}).start()
    try:
        key = "testos-x86_64"
        monkeypatch.setattr(tts_runtimes, "platform_key", lambda: key)
        runtime = tts_runtimes.RUNTIME_INFO["llama-server"]
        monkeypatch.setitem(
            runtime.releases,
            key,
            tts_runtimes.Release(f"{server.url}/cpu.zip", hashlib.sha256(payload).hexdigest(), len(payload)),
        )
        monkeypatch.setitem(
            runtime.releases,
            key + "-vulkan",
            tts_runtimes.Release(f"{server.url}/vulkan.zip", hashlib.sha256(vk).hexdigest(), len(vk)),
        )
        monkeypatch.setenv("PATH", "/nonexistent")
        core.settings.device = "cuda"
        assert tts_runtimes.flavour_for("llama-server", "cuda") == "cpu"
        assert tts_runtimes.flavour_for("llama-server", "vulkan") == "vulkan"
        job = core.components.start_install("llama-server", accept_license=False)
        await wait_for(job)
        assert job.state == "done", job.error
        (warning,) = job.warnings
        assert "no prebuilt cuda build" in warning["message"] and "--flavour vulkan" in warning["message"]
        assert tts_runtimes.installed_flavours(core.settings, "llama-server").keys() == {"cpu"}
    finally:
        server.stop()


def test_an_unknown_audiocpp_voice_is_a_400_with_the_voices(settings: Settings, audiocpp: FakeTTSServer) -> None:
    """audio.cpp takes a voice it does not know for a file path on its machine and answered 500: bolo refuses it
    first, naming the voices there are. GET /v1/audio/voices?model= lists that model's voices only."""
    settings.tts_servers = [f"remote-ac={audiocpp.url}", f"remote-llama={audiocpp.url}"]
    srv = LiveServer(settings)
    srv.core.catalog.add(ModelSpec.from_dict(entry("remote-ac", "audio.cpp", server_model="ac-model")))
    other = entry("remote-llama")
    other["voices"] = [{"id": "other-voice", "name": "Another model's voice", "server_voice": "x"}]
    srv.core.catalog.add(ModelSpec.from_dict(other))
    srv.start()
    try:
        before = len(audiocpp.requests)
        r = httpx.post(
            f"{srv.url}/v1/audio/speech", json={"model": "remote-ac", "input": "Hi.", "voice": "nobody"}, timeout=30
        )
        assert r.status_code == 400, r.text
        error = r.json()["error"]
        assert error["param"] == "voice" and "'nobody'" in error["message"]
        assert "default" in error["message"] and "narrator" in error["message"]  # the catalog's and the server's
        assert not [q for q in audiocpp.requests[before:] if q["path"] == "/v1/audio/speech"]  # never sent
        r = httpx.post(
            f"{srv.url}/v1/audio/speech", json={"model": "remote-ac", "input": "Hi.", "voice": "narrator"}, timeout=30
        )
        assert r.status_code == 200 and audiocpp.requests[-1]["body"]["voice"] == "narrator"
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "remote-ac", "input": "Hi.", "voice": "alloy"})
        assert r.status_code == 200  # a stock OpenAI name is still dropped, not refused

        listed = httpx.get(f"{srv.url}/v1/audio/voices", params={"model": "remote-ac"}).json()["voices"]
        assert listed and {v["model_id"] for v in listed} == {"remote-ac"}
        assert "other-voice" not in {v["id"] for v in listed}
        listed = httpx.get(f"{srv.url}/v1/audio/voices", params={"model": "remote-llama"}).json()["voices"]
        assert {v["id"] for v in listed} == {"other-voice"}
        everything = httpx.get(f"{srv.url}/v1/audio/voices").json()["voices"]
        assert {"remote-ac", "remote-llama"} <= {v["model_id"] for v in everything}
        r = httpx.get(f"{srv.url}/v1/audio/voices", params={"model": "no-such-model"})
        assert r.status_code == 404
    finally:
        srv.stop()


def test_check_voice_without_named_voices() -> None:
    model = ServerTTS("audio.cpp", "ac", {"runtime": "audio.cpp", "model_id": "ac"}, base_url="http://127.0.0.1:9")
    try:
        with pytest.raises(InvalidRequest, match="no named voices"):
            model.build_request("Hi.", "C:/voices/me.wav", 1.0, None)
        assert "voice" not in model.build_request("Hi.", None, 1.0, None)
    finally:
        model.close()


def test_instructions_go_to_the_option_the_audiocpp_family_reads() -> None:
    """A request's instructions become the family's instruction option (``instruct``, or IndexTTS-2's
    ``emotion_text`` with ``use_emotion_text``); an explicit option of the same name wins; Maya1's default is sent
    when a request has none; families without the option, and llama-server, never get it."""
    model = ServerTTS("audio.cpp", "x", {"model_id": "x", "instructions": "instruct"}, base_url="http://127.0.0.1:9")
    assert model.supports_instructions
    body = model.build_request("Hi.", None, 1.0, SynthesisExtras(instructions="Whisper, slowly."))
    assert body["options"] == {"instruct": "Whisper, slowly."}
    body = model.build_request(
        "Hi.", None, 1.0, SynthesisExtras(instructions="Whisper.", options={"instruct": "Shout."})
    )
    assert body["options"] == {"instruct": "Shout."}
    assert "options" not in model.build_request("Hi.", None, 1.0, SynthesisExtras())
    model.close()

    spec = {"model_id": "x", "instructions": {"option": "emotion_text", "with": {"use_emotion_text": True}}}
    model = ServerTTS("audio.cpp", "x", spec, base_url="http://127.0.0.1:9")
    body = model.build_request("Hi.", None, 1.0, SynthesisExtras(instructions="Sad."))
    assert body["options"] == {"emotion_text": "Sad.", "use_emotion_text": True}
    model.close()

    spec = {"model_id": "x", "instructions": "instruct", "default_instructions": "A calm voice."}
    model = ServerTTS("audio.cpp", "x", spec, base_url="http://127.0.0.1:9")
    assert model.build_request("Hi.", None, 1.0, None)["options"] == {"instruct": "A calm voice."}
    model.close()

    for runtime in ("audio.cpp", "llama-server"):
        model = ServerTTS(runtime, "x", {"model_id": "x"}, base_url="http://127.0.0.1:9")
        assert not model.supports_instructions
        assert "options" not in model.build_request("Hi.", None, 1.0, SynthesisExtras(instructions="Sad."))
        model.close()


def test_instructions_through_the_speech_api(settings: Settings, audiocpp: FakeTTSServer) -> None:
    """``instructions`` reach a model that follows them (``x-bolo-instructions: applied``) and are reported as
    ignored by one that does not; /v1/models lists the feature."""
    settings.tts_servers = [f"steer={audiocpp.url}", f"plain={audiocpp.url}"]
    srv = LiveServer(settings)
    steer = entry("steer", "audio.cpp", server_model="ac-model", instructions="instruct")
    steer["features"] = ["voice-cloning", "instructions"]
    srv.core.catalog.add(ModelSpec.from_dict(steer))
    srv.core.catalog.add(ModelSpec.from_dict(entry("plain", "audio.cpp", server_model="ac-model")))
    srv.start()
    try:
        info = httpx.get(f"{srv.url}/api/v1/models/steer").json()
        assert "instructions" in info["features"]
        body = {"input": "Hello.", "response_format": "wav", "instructions": "Cheerful and quick."}
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "steer", **body}, timeout=30)
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-instructions"] == "applied"
        assert audiocpp.requests[-1]["body"]["options"]["instruct"] == "Cheerful and quick."
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "plain", **body}, timeout=30)
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-instructions"] == "ignored"
        assert "instruct" not in (audiocpp.requests[-1]["body"].get("options") or {})
        r = httpx.post(f"{srv.url}/v1/audio/speech", json={"model": "plain", "input": "Hi.", "response_format": "wav"})
        assert "x-bolo-instructions" not in r.headers
    finally:
        srv.stop()


def test_dialogue_in_one_take_and_stitched(settings: Settings, audiocpp: FakeTTSServer) -> None:
    """A model with ``dialogue: vibevoice`` gets the whole script as ``Speaker N:`` lines and a recording per speaker
    (files the server reads); any other model speaks turn by turn, joined with pauses. ``mode`` forces either."""
    settings.tts_servers = [f"vv={audiocpp.url}", f"plain={audiocpp.url}"]
    srv = LiveServer(settings)
    vv = entry("vv", "audio.cpp", server_model="ac-model", dialogue="vibevoice", cloning=False)
    vv["features"] = ["dialogue"]
    srv.core.catalog.add(ModelSpec.from_dict(vv))
    srv.core.catalog.add(ModelSpec.from_dict(entry("plain", "audio.cpp", server_model="ac-model")))
    srv.start()
    turns = [
        {"speaker": "Asha", "text": "Hello there."},
        {"speaker": "Ravi", "text": "Hi."},
        {"speaker": "Asha", "text": "Shall we start?"},
    ]
    try:
        r = httpx.post(
            f"{srv.url}/v1/audio/dialogue",
            json={"model": "vv", "turns": turns, "voices": {"Ravi": "en-long"}},
            timeout=60,
        )
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-dialogue"] == "native" and r.headers["x-bolo-speakers"] == "Asha, Ravi"
        sent = audiocpp.requests[-1]
        assert sent["body"]["input"] == "Speaker 1: Hello there.\nSpeaker 2: Hi.\nSpeaker 1: Shall we start?"
        assert len(sent["voice_samples"]) == 2 and all(n > 1000 for n in sent["voice_samples"])
        assert "voice_ref" not in sent["body"]
        audio, rate = sf.read(io.BytesIO(r.content))
        assert rate == 24000 and len(audio) > 0

        # a voice with no recording: auto stitches, native refuses
        body = {"model": "vv", "turns": turns, "voices": {"Ravi": "default"}}
        r = httpx.post(f"{srv.url}/v1/audio/dialogue", json=body, timeout=60)
        assert r.status_code == 200 and r.headers["x-bolo-dialogue"] == "stitched"
        r = httpx.post(f"{srv.url}/v1/audio/dialogue", json={**body, "mode": "native"}, timeout=60)
        assert r.status_code == 400 and "recording of every voice" in r.json()["error"]["message"]

        # stitched: one request per turn, the pause between them, segments in the header
        n = len(audiocpp.requests)
        script = "Asha: Hello there.\nRavi: Hi.\nAsha: Shall we start?"
        r = httpx.post(
            f"{srv.url}/v1/audio/dialogue",
            json={"model": "plain", "script": script, "voices": {"Asha": "heart"}, "pause_ms": 500},
            timeout=60,
        )
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-dialogue"] == "stitched"
        assert [q["body"]["input"] for q in audiocpp.requests[n:]] == ["Hello there.", "Hi.", "Shall we start?"]
        assert "voice_ref" in audiocpp.requests[n]["body"] and "voice_ref" not in audiocpp.requests[n + 1]["body"]
        segments = json.loads(r.headers["x-bolo-segments"])
        assert [s["speaker"] for s in segments] == ["Asha", "Ravi", "Asha"]
        assert segments[1]["start"] == pytest.approx(segments[0]["end"] + 0.5, abs=0.01)
        audio, rate = sf.read(io.BytesIO(r.content))
        assert len(audio) / rate == pytest.approx(segments[-1]["end"], abs=0.01)

        r = httpx.post(f"{srv.url}/v1/audio/dialogue", json={"model": "plain", "turns": turns, "mode": "native"})
        assert r.status_code == 400 and "does not speak dialogues in one take" in r.json()["error"]["message"]
        r = httpx.post(f"{srv.url}/v1/audio/dialogue", json={"model": "plain", "turns": [], "script": ""})
        assert r.status_code == 400
        r = httpx.post(
            f"{srv.url}/v1/audio/dialogue", json={"model": "plain", "turns": turns, "voices": {"Nobody": "heart"}}
        )
        assert r.status_code == 400 and "Nobody" in r.json()["error"]["message"]
    finally:
        srv.stop()


def test_voice_conversion_through_audiocpp(settings: Settings, audiocpp: FakeTTSServer) -> None:
    """``POST /v1/audio/voice-conversion`` sends the source inline to ``/v1/tasks/run`` and the target recording as
    a file the server reads (a saved voice, a reference clip or an upload); a model with voices of its own (RVC) takes
    one of them as ``voice_id``; the ElevenLabs route answers too."""
    settings.tts_servers = [f"vc={audiocpp.url}", f"rvc={audiocpp.url}"]
    srv = LiveServer(settings)
    base = {
        "task": "voice-conversion", "engine": "tts-server", "kind": "tts-server", "family": "Test", "languages": [],
        "sample_rate": 16000, "license": "MIT", "requires": ["audio.cpp"], "source": {"local": "/nonexistent"},
    }  # fmt: skip
    srv.core.catalog.add(
        ModelSpec.from_dict({**base, "id": "vc", "options": {"runtime": "audio.cpp", "server_model": "ac-model"}})
    )
    srv.core.catalog.add(
        ModelSpec.from_dict(
            {
                **base, "id": "rvc", "voices": [{"id": "default"}, {"id": "chocola"}],
                "options": {"runtime": "audio.cpp", "server_model": "ac-model", "vc_voices": True},
            }
        )
    )  # fmt: skip
    srv.start()
    source = wav_bytes(1.0)
    url = f"{srv.url}/v1/audio/voice-conversion"
    try:
        r = httpx.post(url, files={"file": ("a.wav", source)}, data={"model": "vc", "voice": "en-short"}, timeout=30)
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-target"] == "clip=en-short" and r.headers["x-bolo-model"] == "vc"
        sent = audiocpp.requests[-1]
        assert sent["path"] == "/v1/tasks/run" and sent["voice_ref_bytes"] > 1000
        audio, rate = sf.read(io.BytesIO(r.content))
        assert rate == 16000 and len(audio) == pytest.approx(8000, abs=10)

        files = {"file": ("a.wav", source), "reference_audio": ("ref.wav", source)}
        data = {"model": "vc", "options": '{"seed": 7, "fast": true}'}
        r = httpx.post(url, files=files, data=data, timeout=30)
        assert r.status_code == 200 and r.headers["x-bolo-target"] == "reference=upload"
        assert audiocpp.requests[-1]["body"]["request"]["options"] == {"seed": "7", "fast": "true"}

        r = httpx.post(url, files={"file": ("a.wav", source)}, data={"model": "vc", "voice": "nobody"}, timeout=30)
        assert r.status_code == 400 and "from a recording" in r.json()["error"]["message"]

        r = httpx.post(url, files={"file": ("a.wav", source)}, data={"model": "rvc", "voice": "chocola"}, timeout=30)
        assert r.status_code == 200 and r.headers["x-bolo-target"] == "voice=chocola"
        sent = audiocpp.requests[-1]["body"]["request"]
        assert sent["options"] == {"voice_id": "chocola"} and "voice_ref" not in sent
        r = httpx.post(url, files={"file": ("a.wav", source)}, data={"model": "rvc", "voice": "en-short"}, timeout=30)
        assert r.status_code == 400 and "its own voices" in r.json()["error"]["message"]

        r = httpx.post(
            f"{srv.url}/v1/speech-to-speech/en-long",
            files={"audio": ("a.wav", source)},
            data={"model_id": "vc", "output_format": "pcm_16000"},
            timeout=30,
        )
        assert r.status_code == 200 and len(r.content) == pytest.approx(16000, abs=40)
        r = httpx.post(url, data={"model": "vc"}, timeout=30)
        assert r.status_code == 400
    finally:
        srv.stop()


def test_plain_instructions_for_a_model_with_a_closed_list_of_items(
    settings: Settings, audiocpp: FakeTTSServer
) -> None:
    """OmniVoice understands only its own items: plain words are put into them ("Whisper very softly" -> whisper),
    and words that fit none are dropped with a 200 and ``x-bolo-instructions: ignored; ...`` instead of the server's
    refusal (a 503 that said the engine was down). A family that refuses instructions bolo did not expect is asked
    again without them."""
    from bolo.engines.instruct_vocab import omnivoice

    assert omnivoice("Whisper very softly").items == ["whisper"]
    assert omnivoice("An old man with a deep voice").items == ["low pitch", "male", "elderly"]
    assert omnivoice("female, low pitch, british accent").items == ["female", "low pitch", "british accent"]
    assert omnivoice("a little girl").items == ["female", "child"]
    assert omnivoice("angry").items == [] and omnivoice("angry").unplaced == ["angry"]
    assert omnivoice("Speak English with an Indian accent").items == ["indian accent"]

    audiocpp.instruct_items = {"whisper", "male", "female", "low pitch", "elderly"}
    settings.tts_servers = [f"omni={audiocpp.url}", f"strict={audiocpp.url}"]
    srv = LiveServer(settings)
    omni = entry(
        "omni", "audio.cpp", server_model="ac-model", instructions={"option": "instruct", "vocabulary": "omnivoice"}
    )
    omni["features"] = ["voice-cloning", "instructions"]
    strict = entry("strict", "audio.cpp", server_model="ac-model", instructions="instruct")
    strict["features"] = ["voice-cloning", "instructions"]
    srv.core.catalog.add(ModelSpec.from_dict(omni))
    srv.core.catalog.add(ModelSpec.from_dict(strict))
    srv.start()
    url = f"{srv.url}/v1/audio/speech"
    try:
        for words, sent, header in [
            ("whisper", "whisper", "applied"),
            ("Whisper very softly", "whisper", "applied; as: whisper"),
            ("An old man, whispering angrily", "whisper, male, elderly", "applied; as: whisper, male, elderly"),
            ("Speak angrily, with a lot of emotion", None, "ignored; OmniVoice understands only these items"),
            ("happy", None, "ignored; OmniVoice understands only these items"),
        ]:
            r = httpx.post(
                url, json={"model": "omni", "input": "Hello.", "response_format": "wav", "instructions": words}
            )
            assert r.status_code == 200, (words, r.text)
            assert r.headers["x-bolo-instructions"].startswith(header), (words, r.headers["x-bolo-instructions"])
            assert (audiocpp.requests[-1]["body"].get("options") or {}).get("instruct") == sent, words
        assert (
            "(not understood: angrily)"
            in httpx.post(
                url,
                json={"model": "omni", "input": "Hi.", "response_format": "wav", "instructions": "an old man, angrily"},
            ).headers["x-bolo-instructions"]
        )
        # no vocabulary in the catalog: the server refuses, bolo asks again without the instructions
        r = httpx.post(
            url, json={"model": "strict", "input": "Hello.", "response_format": "wav", "instructions": "angry"}
        )
        assert r.status_code == 200, r.text
        assert r.headers["x-bolo-instructions"].startswith("ignored; the model refused them: unsupported OmniVoice")
        assert "instruct" not in (audiocpp.requests[-1]["body"].get("options") or {})
        assert audiocpp.requests[-2]["body"]["options"]["instruct"] == "angry"
    finally:
        srv.stop()


def test_instructions_only_without_a_reference_recording() -> None:
    """FireRedTTS3 follows instructions when it designs a voice (no reference): with a reference they are noted as
    ignored rather than sent where the model would read them as the reference's transcript."""
    spec = {
        "model_id": "x", "cloning": True,
        "instructions": {"option": "instruct", "with": {"template_name": "voice_design"}, "without_reference": True},
    }  # fmt: skip
    model = ServerTTS("audio.cpp", "x", spec, base_url="http://127.0.0.1:9")
    try:
        extras = SynthesisExtras(instructions="A calm old man.")
        assert model.build_request("Hi.", None, 1.0, extras)["options"] == {
            "instruct": "A calm old man.", "template_name": "voice_design"
        }  # fmt: skip
        extras = SynthesisExtras(instructions="A calm old man.", reference_audio=wav_bytes(1.0))
        assert "options" not in model.build_request("Hi.", None, 1.0, extras)
        assert extras.instructions_note and extras.instructions_note.startswith("ignored; x follows instructions only")
    finally:
        model.close()
