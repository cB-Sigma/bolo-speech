from __future__ import annotations

import asyncio
import functools
import io
import os
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn

from bolo.api import create_app
from bolo.audio.encode import encode_audio
from bolo.config import Settings
from bolo.core import Bolo
from fake_llm import FakeLLM

_handle_error = socketserver.BaseServer.handle_error


def _quiet_handle_error(self: socketserver.BaseServer, request: object, client_address: object) -> None:
    """The fake HTTP servers: a client that hangs up (ConnectionResetError, WinError 10054 on Windows) is not news."""
    if isinstance(sys.exc_info()[1], ConnectionError):
        return
    _handle_error(self, request, client_address)  # type: ignore[arg-type]


socketserver.BaseServer.handle_error = _quiet_handle_error  # type: ignore[method-assign]

#: Time limits of the tests, in one place. What a test waits for (a server starting, an answer from it) takes well
#: under a second on a quiet machine and many times that on a busy CI runner, so the limits are generous: they are
#: reached only when something is wrong. BOLO_TEST_TIMEOUT_SCALE multiplies them (2 on a slow or shared machine).
TIMEOUT_SCALE = float(os.environ.get("BOLO_TEST_TIMEOUT_SCALE") or 1) or 1.0
#: how long a test waits for something to happen (a server to start, a thread to finish, an event)
WAIT_S = 60.0 * TIMEOUT_SCALE
#: a test's HTTP requests to the servers it starts; httpx's own default (5 s) is too short on a busy machine
HTTP_TIMEOUT = httpx.Timeout(30.0 * TIMEOUT_SCALE)


def _with_test_timeout(call: object) -> object:
    @functools.wraps(call)  # type: ignore[arg-type]
    def with_timeout(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("timeout", HTTP_TIMEOUT)
        return call(*args, **kwargs)  # type: ignore[operator]

    return with_timeout


# httpx.get(...) and the like without a timeout, in the tests, get HTTP_TIMEOUT; an explicit one is kept. bolo's own
# requests always say how long they wait, so this changes nothing in the code under test.
for _name in ("request", "stream", "get", "options", "head", "post", "put", "patch", "delete"):
    setattr(httpx, _name, _with_test_timeout(getattr(httpx, _name)))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("BOLO_TEST_REAL") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test: set BOLO_TEST_REAL=1 (needs network and engines)")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_host_hf_token(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A developer's Hugging Face login (HF_TOKEN, or the token `hf auth login` saved) must not reach the tests: a
    test that a gated model needs a token failed on a Mac where `hf auth login` had run."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "BOLO_HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path_factory.getbasetemp() / "no-hf-token"))


@pytest.fixture(autouse=True)
def _sherpa_build_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Which sherpa-onnx wheel this machine has (PyPI's CPU build, or none) must not move tests' sherpa-onnx models
    to the CPU: tests that are about it say which build they mean."""
    from bolo import ort_providers

    monkeypatch.setattr(ort_providers, "sherpa_gpu_build", lambda: None)


def speechlike(seconds: float, sample_rate: int = 16000, gap_at: tuple[float, ...] = ()) -> np.ndarray:
    """A loud modulated tone (the energy VAD treats it as speech) with optional 0.8 s gaps."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    for g in gap_at:
        x[int(g * sample_rate) : int((g + 0.8) * sample_rate)] = 0.0
    return x.astype(np.float32)


def wav_bytes(seconds: float = 2.0, sample_rate: int = 16000, gap_at: tuple[float, ...] = ()) -> bytes:
    return encode_audio(speechlike(seconds, sample_rate, gap_at), sample_rate, "wav")


def wav_file(seconds: float = 2.0, **kw: object) -> io.BytesIO:
    f = io.BytesIO(wav_bytes(seconds, **kw))  # type: ignore[arg-type]
    f.name = "audio.wav"
    return f


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        fake_engine=True,
        default_stt="fake-stt",
        default_realtime_stt="fake-stt-streaming",
        default_tts="fake-tts",
        default_realtime_tts="fake-tts",
        default_vad="fake-vad",
        default_diarization="fake-diarization",
        default_punctuation="fake-punctuation",
        default_itn="fake-itn",
        default_pii="fake-pii",
        # the fake diarizer's "voices" are two-dimensional and its clips short: no folding of small speakers
        diarization_min_speaker_s=0.0,
        diarization_merge_similarity=0.0,
        auto_pull=False,
    )


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    server = FakeLLM().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def agent(settings: Settings, llm: FakeLLM) -> Iterator[LiveServer]:
    """A live server with the fake language model connected."""
    settings.llm_base_url = llm.url
    settings.llm_model = "fake-llm"
    settings.llm_api_key = "llm-secret"
    srv = LiveServer(settings)
    srv.llm = llm  # type: ignore[attr-defined]
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def two_speakers(turns: int = 4, seconds: float = 1.2, gap: float = 0.6, sample_rate: int = 16000) -> np.ndarray:
    """Alternating bursts at 150 Hz and 420 Hz: two "speakers" the fake diarizer tells apart by pitch."""
    parts = []
    for i in range(turns):
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        freq = 150.0 if i % 2 == 0 else 420.0
        parts.append((0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))).astype(np.float32))
        parts.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
    return np.concatenate(parts)


@pytest.fixture
async def core(settings: Settings) -> Iterator[Bolo]:
    c = Bolo(settings)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


class LiveServer:
    def __init__(self, settings: Settings) -> None:
        self.core = Bolo(settings)
        self.app = create_app(core=self.core)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", ws="websockets")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + WAIT_S
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("server did not start: it stopped during start-up")
            if time.monotonic() > deadline:
                raise RuntimeError(f"server did not start in {WAIT_S:.0f} s")
            time.sleep(0.02)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=WAIT_S)

    def call(self, coro: object) -> object:
        """Run a coroutine on the server's event loop (to inspect state safely)."""
        loop = self.core.events._loop
        assert loop is not None
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=WAIT_S)  # type: ignore[arg-type]


@pytest.fixture
def live(settings: Settings) -> Iterator[LiveServer]:
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()
