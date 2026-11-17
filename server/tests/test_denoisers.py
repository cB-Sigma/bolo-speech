"""The denoisers of the catalog: their entries, the frame-by-frame runners (DeepFilterNet 3, RNNoise), files pinned
per platform, the plugin op ``denoise`` and ElevenLabs' voice isolator."""

from __future__ import annotations

import io
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from bolo.catalog import Catalog, ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.denoise import DenoiseOptions, LiveDenoiser
from bolo.engines.base import DenoiseModel, FrameDenoiseStream, OverlapDenoiseStream
from bolo.engines.denoisers import FRAME, DeepFilterNetStream, RNNoise, rnnoise_library
from bolo.errors import DownloadError
from bolo.obs.trace import Trace
from bolo.store import ModelStore, Source, extract_archive, platform_matches, platform_tag
from conftest import LiveServer
from fake_providers import KEY, FakeProviders

SR = 48000


def speech(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t)).astype(np.float32)


# -- the catalog ------------------------------------------------------------------------------------------------
def test_every_denoiser_states_its_licence_and_pins_its_files() -> None:
    specs = [s for s in Catalog.load().all() if s.task == "denoise"]
    ids = {s.id for s in specs}
    assert {"gtcrn-simple", "dpdfnet2", "deepfilternet3", "rnnoise", "elevenlabs-voice-isolator"} <= ids
    for spec in specs:
        assert spec.license and spec.license != "unknown", spec.id
        if spec.cloud or not spec.runnable:
            continue
        assert spec.source.urls, spec.id
        for u in spec.source.urls:
            assert len(str(u.get("sha256", ""))) == 64 and u.get("size"), (spec.id, u["path"])
        assert spec.sample_rate in (8000, 16000, 48000) and spec.size_bytes > 0, spec.id


def test_platform_tags() -> None:
    assert platform_matches("linux-x86_64", "linux-x86_64")
    assert platform_matches(["linux-aarch64", "linux-x86_64"], "linux-x86_64")
    assert platform_matches("darwin", "darwin-aarch64") and not platform_matches("darwin", "linux-x86_64")
    assert not platform_matches("win32-x86_64", "linux-x86_64")
    assert platform_tag().count("-") >= 1


def test_files_for_another_platform_are_left_out(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    here, there = platform_tag(), "plan9-mips"
    files = store.plan(
        Source(urls=[{"path": "a.whl", "url": "https://x/a", "platform": here}, {"path": "b", "url": "https://x/b",
                                                                                     "platform": there}])
    )  # fmt: skip
    assert [f.path for f in files] == ["a.whl"]
    with pytest.raises(DownloadError, match="plan9-mips"):
        store.plan(Source(urls=[{"path": "b", "url": "https://x/b", "platform": there}]))


def test_a_wheel_is_unpacked_like_a_zip(tmp_path: Path) -> None:
    wheel = tmp_path / "pkg.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("pkg-1.0.data/purelib/pkg/librnnoise.so", b"\x7fELF")
        z.writestr("pkg-1.0.dist-info/METADATA", b"Name: pkg")
    dest = tmp_path / "model"
    written = extract_archive(wheel, dest, "pkg-linux.whl")
    assert "pkg-1.0.data/purelib/pkg/librnnoise.so" in written
    if sys.platform.startswith("linux"):
        assert rnnoise_library(dest).name == "librnnoise.so"


# -- DeepFilterNet 3, one ONNX graph run frame by frame ----------------------------------------------------------
class _DelayGraph:
    """Stands in for the torchDF graph: hands back its input ``delay`` samples late, keeps the delay in its state."""

    def __init__(self, delay: int) -> None:
        self.delay = delay
        self.frames = 0

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        frame, state = feeds["input_frame"], feeds["states"]
        assert frame.shape == (FRAME,) and feeds["atten_lim_db"].shape == ()
        self.frames += 1
        line = np.concatenate([state[: self.delay], frame])
        out, rest = line[:FRAME], line[FRAME:]
        new = np.zeros_like(state)
        new[: self.delay] = rest
        return [out, new, np.zeros(1, np.float32)]


def test_deepfilternet_output_lines_up_with_the_input() -> None:
    model = DeepFilterNetStream.__new__(DeepFilterNetStream)
    graph = _DelayGraph(1440)
    model._session = graph  # type: ignore[assignment]
    model._state_size, model._delay, model.sample_rate = 4096, 1440, SR
    model._limit = np.array(0.0, np.float32)
    import threading

    model._lock = threading.Lock()
    x = speech(0.123)
    y = model.denoise(x)
    assert len(y) == len(x) and np.allclose(y, x)
    assert graph.frames * FRAME >= len(x) + 1440


# -- RNNoise, through its C library -----------------------------------------------------------------------------
class _FakeRNNoise:
    """The four C functions bolo calls, with RNNoise's habit of answering two frames late."""

    def __init__(self, delay: int = 2 * FRAME) -> None:
        self.pending = np.zeros(delay, np.float32)
        self.created = self.destroyed = 0

    def rnnoise_create(self, _model: Any) -> int:
        self.created += 1
        self.pending[:] = 0
        return 1

    def rnnoise_destroy(self, _state: Any) -> None:
        self.destroyed += 1

    def rnnoise_process_frame(self, _state: Any, out: Any, inp: Any) -> float:
        src = np.ctypeslib.as_array(inp, shape=(FRAME,)).copy()
        dst = np.ctypeslib.as_array(out, shape=(FRAME,))
        assert np.abs(src).max() > 1.0 or not src.any()  # 16-bit sample values, not -1..1
        line = np.concatenate([self.pending, src])
        dst[:] = line[:FRAME]
        self.pending = line[FRAME:]
        return 0.0


def test_rnnoise_output_lines_up_with_the_input() -> None:
    model = RNNoise.__new__(RNNoise)
    lib = _FakeRNNoise()
    model._lib, model._delay, model.sample_rate = lib, 2 * FRAME, SR  # type: ignore[assignment]
    x = speech(0.31)
    y = model.denoise(x)
    assert len(y) == len(x) and np.allclose(y, x, atol=1e-6)
    assert lib.created == lib.destroyed == 1


# -- a denoiser in a plugin (op ``denoise``) -------------------------------------------------------------------
async def test_plugin_denoiser(settings: Settings) -> None:
    settings.plugins = [f"ref={sys.executable} -m bolo.plugin.reference"]
    settings.plugin_entry_points = False
    core = Bolo(settings)
    core.catalog.add(
        ModelSpec.from_dict(
            {
                "id": "plug-denoise", "task": "denoise", "engine": "plugin:ref", "kind": "fake-denoise",
                "family": "Fake", "languages": [], "license": "Apache-2.0", "builtin": True, "sample_rate": 48000,
                "options": {"sample_rate": 48000},
            }
        )
    )  # fmt: skip
    await core.start()
    try:
        rng = np.random.default_rng(0)
        t = np.arange(32000) / 16000
        noisy = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
        trace = Trace("stt", "internal", "")
        result = await core.transcribe(noisy, 16000, trace=trace, denoise=DenoiseOptions(model="plug-denoise"))
        assert result.text and trace.extra["denoise"]["status"] == "applied"
        assert trace.extra["denoise"]["sample_rate"] == 48000
        assert type(core.manager.loaded["plug-denoise"].model).__name__ == "PluginDenoise"
    finally:
        await core.stop()


# -- ElevenLabs' voice isolator ---------------------------------------------------------------------------------
@pytest.fixture
def cloud() -> Iterator[FakeProviders]:
    server = FakeProviders().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def cloud_srv(settings: Settings, cloud: FakeProviders) -> Iterator[LiveServer]:
    settings.providers = {"elevenlabs": {"base_url": cloud.url, "api_key": KEY}}
    server = LiveServer(settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_elevenlabs_voice_isolator(cloud_srv: LiveServer, cloud: FakeProviders) -> None:
    from bolo.audio.encode import encode_audio

    t = np.arange(32000) / 16000
    audio = encode_audio((0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 16000, "wav")
    r = httpx.post(
        f"{cloud_srv.url}/v1/audio/transcriptions",
        data={"model": "fake-stt", "denoise": "elevenlabs-voice-isolator"},
        files={"file": ("a.wav", io.BytesIO(audio), "audio/wav")},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.headers["x-bolo-denoise"] == "elevenlabs-voice-isolator"
    sent = cloud.last("/v1/audio-isolation")
    assert sent["headers"]["xi-api-key"] == KEY and sent["form"]["file_format"] == "pcm_s16le_16"
    assert sent["files"]["audio"] == 2 * 32000  # 16-bit PCM at 16 kHz


# -- live streams (the speech-to-text WebSocket) ----------------------------------------------------------------
class _Identity(DenoiseModel):
    def __init__(self, rate: int) -> None:
        self.sample_rate = rate

    def denoise(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=np.float32).copy()


def _pieces(x: np.ndarray, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 2000))
        out.append(x[i : i + n])
        i += n
    return out


def test_frame_stream_cuts_the_delay_and_returns_every_sample() -> None:
    delayed = _DelayGraph(700)
    state = [np.zeros(4096, np.float32)]

    def step(frame: np.ndarray) -> np.ndarray:
        out, state[0], _ = delayed.run(None, {"input_frame": frame, "states": state[0], "atten_lim_db": np.array(0.0)})
        return out

    stream = FrameDenoiseStream(step, FRAME, 700)
    x = speech(0.37)
    got = [stream.process(p) for p in _pieces(x)]
    y = np.concatenate([*got, stream.finish()])
    assert len(y) == len(x) and np.allclose(y, x)
    assert sum(len(g) for g in got) >= len(x) - 700 - FRAME  # at most the delay and a frame held back


def test_overlap_stream_for_models_without_a_streaming_form() -> None:
    stream = OverlapDenoiseStream(_Identity(SR), context_s=0.1)
    x = speech(0.3)
    y = np.concatenate([stream.process(p) for p in _pieces(x)] + [stream.finish()])
    assert not stream.stateful and np.allclose(y, x)


@pytest.mark.parametrize("native", [8000, 16000, 48000])
def test_live_denoiser_stays_in_step_through_resampling(native: int) -> None:
    rate = 16000
    t = np.arange(rate) / rate
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    live = LiveDenoiser(_Identity(native), rate, 1.0)
    y = np.concatenate([live.process(p) for p in _pieces(x, 2)] + [live.finish()])
    assert len(y) == len(x)
    middle = slice(800, len(x) - 800)  # the resampler's edges aside, sample k out is sample k in
    assert np.max(np.abs(y[middle] - x[middle])) < 0.02
    half = LiveDenoiser(_Identity(native), rate, 0.5)
    z = np.concatenate([half.process(p) for p in _pieces(x, 3)] + [half.finish()])
    assert len(z) == len(x) and live.summary()["chunks"] > 3


async def test_stt_stream_with_denoise(dn_ws: LiveServer) -> None:
    import json

    import websockets

    url = dn_ws.ws_url + "/api/v1/stt/stream?model=fake-stt&vad=false&interim=false&denoise=fake-denoise-48k"
    async with websockets.connect(url) as ws:
        ready = json.loads(await ws.recv())
        assert ready["denoise"]["model"] == "fake-denoise-48k" and ready["denoise"]["status"] == "applied"
        assert ready["denoise"]["sample_rate"] == 48000 and ready["denoise"]["strength"] == 1.0
        t = np.arange(32000) / 16000
        audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        for i in range(0, len(audio), 1600):
            await ws.send((audio[i : i + 1600] * 32767).astype("<i2").tobytes())
        await ws.send(json.dumps({"type": "stop"}))
        messages = [json.loads(m) async for m in ws]
    metrics = [m for m in messages if m["type"] == "metrics"][-1]
    assert metrics["audio_seconds"] == pytest.approx(2.0, abs=0.01)  # nothing lost to the denoiser
    assert metrics["denoise"]["chunks"] >= 1 and metrics["denoise"]["mean_ms"] >= 0
    # a request's denoise=false wins; strength alone uses the default denoiser
    async with websockets.connect(dn_ws.ws_url + "/api/v1/stt/stream?model=fake-stt&denoise=off") as ws:
        assert json.loads(await ws.recv())["denoise"] is None
    async with websockets.connect(dn_ws.ws_url + "/api/v1/stt/stream?model=fake-stt") as ws:
        assert json.loads(await ws.recv())["denoise"] is None  # off by default
        await ws.send(json.dumps({"type": "config", "denoise": 0.5}))
        ready = json.loads(await ws.recv())
        assert ready["denoise"]["model"] == "fake-denoise" and ready["denoise"]["strength"] == 0.5


@pytest.fixture
def dn_ws(settings: Settings) -> Iterator[LiveServer]:
    settings.default_denoise = "fake-denoise"
    server = LiveServer(settings)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.parametrize(("rate_in", "rate_out"), [(16000, 48000), (16000, 8000), (48000, 16000), (8000, 16000)])
def test_low_delay_resampler_is_in_step_and_holds_back_little(rate_in: int, rate_out: int) -> None:
    from bolo.audio.resample import LowDelayResampler

    r = LowDelayResampler(rate_in, rate_out)
    x = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(rate_in) / rate_in)).astype(np.float32)
    step = rate_in // 50
    outs = [r.process(x[i : i + step]) for i in range(0, len(x), step)]
    y = np.concatenate([*outs, r.finish()])
    assert len(y) == rate_out
    ref = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(len(y)) / rate_out)
    assert np.max(np.abs(y[200:-200] - ref[200:-200])) < 1e-3
    held = 16 * rate_out // min(rate_in, rate_out)  # the filter's delay: 16 samples of the lower rate
    assert len(outs[0]) >= rate_out // 50 - held - 1
    with pytest.raises(ValueError):
        LowDelayResampler(16000, 44100)
