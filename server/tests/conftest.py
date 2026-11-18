from __future__ import annotations

import asyncio
import io
import os
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("BOLO_TEST_REAL") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test: set BOLO_TEST_REAL=1 (needs network and engines)")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)


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
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def call(self, coro: object) -> object:
        """Run a coroutine on the server's event loop (to inspect state safely)."""
        loop = self.core.events._loop
        assert loop is not None
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)  # type: ignore[arg-type]


@pytest.fixture
def live(settings: Settings) -> Iterator[LiveServer]:
    srv = LiveServer(settings)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()
