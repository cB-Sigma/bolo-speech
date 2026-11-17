"""Tencent's Freeze-Omni server (``bin/server.py`` of VITA-MLLM/Freeze-Omni) as a :class:`DuplexBackend`.

Its server speaks Socket.IO (Flask-SocketIO 5.3), with the events of its own web page (``web/resources/demo.html``)
at commit 163a2488:

* client → server: ``prompt_text`` (a system prompt; answered with ``prompt_success``), ``recording-started`` (a new
  conversation state), ``audio`` with a JSON string ``{"sample_rate": 16000, "audio": [bytes of 16-bit PCM]}``,
  ``recording-stopped``;
* server → client: ``audio`` (16-bit PCM at 24 kHz, a binary attachment; the server sends at most one piece per
  ``audio`` event it receives, so the caller's audio has to keep coming), ``stop_tts`` (the model stopped
  speaking: the caller interrupted it), ``too_many_users`` and ``out_time`` (the session's time is up).

The model runs its own voice-activity detection and decides from the caller's audio whether to keep listening,
answer or stop: full-duplex behaviour from a turn-based core. It sends no text.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np

from bolo.duplex.base import (
    DuplexAudio,
    DuplexBackend,
    DuplexConfig,
    DuplexEnd,
    DuplexEvent,
    DuplexInterrupt,
    DuplexSession,
)
from bolo.duplex.kyutai import LOOPBACK
from bolo.duplex.socketio import SocketIOClient
from bolo.engines.base import FloatArray
from bolo.errors import EngineUnavailable, Overloaded

if TYPE_CHECKING:
    from bolo.engines.tts_server import ServerProcess

INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 320  # 20 ms of input per audio event (its web page sends 256 samples)


class FreezeOmniSession(DuplexSession):
    input_rate = INPUT_RATE
    frame_samples = CHUNK

    def __init__(self, client: SocketIOClient, where: str) -> None:
        self.client = client
        self.where = where
        self._pending = np.zeros(0, dtype=np.float32)
        self._queue: asyncio.Queue[DuplexEvent] = asyncio.Queue()
        self._closing = False
        self._ended = False
        self._task = asyncio.create_task(self._read(), name="bolo-duplex-freeze-omni")

    async def _read(self) -> None:
        end = DuplexEnd("ended")
        try:
            while True:
                event = await self.client.next_event()
                if event is None:
                    break
                name, args = event
                if name == "audio" and args and isinstance(args[0], bytes | bytearray):
                    pcm = np.frombuffer(bytes(args[0]), dtype="<i2").astype(np.float32) / 32768.0
                    if len(pcm):
                        await self._queue.put(DuplexAudio(pcm, OUTPUT_RATE))
                elif name == "stop_tts":
                    await self._queue.put(DuplexInterrupt())
                elif name == "out_time":
                    end = DuplexEnd("ended", "the server's session time ran out")
                    break
                elif name == "too_many_users":
                    end = DuplexEnd("failed", "the Freeze-Omni server has as many users as it takes")
                    break
        except asyncio.CancelledError:
            end = DuplexEnd("closed")
            raise
        finally:
            if self._closing:
                end = DuplexEnd("closed")
            self._ended = True
            self._queue.put_nowait(end)

    async def push(self, samples: FloatArray) -> None:
        if self._ended or self._closing or not len(samples):
            return
        self._pending = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32)])
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            try:
                await self.client.emit("audio", json.dumps({"sample_rate": INPUT_RATE, "audio": list(pcm)}))
            except Exception as exc:
                raise EngineUnavailable(f"the Freeze-Omni server at {self.where} closed the session: {exc}") from exc

    async def events(self) -> AsyncIterator[DuplexEvent]:
        while True:
            event = await self._queue.get()
            yield event
            if isinstance(event, DuplexEnd):
                return

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        with contextlib.suppress(Exception):
            await self.client.emit("recording-stopped")
        await self.client.close()
        if not self._task.done():
            self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task


class FreezeOmniBackend(DuplexBackend):
    name = "freeze-omni"

    def __init__(
        self, url: str | None = None, *, process: ServerProcess | None = None, model: str | None = None,
        max_sessions: int = 1, runtime: str | None = None, connect_timeout_s: float = 30.0,
    ) -> None:  # fmt: skip
        if url is None and process is None:
            raise ValueError("a URL or a process is needed")
        self.url = url
        self.process = process
        self.model = model
        self.runtime = runtime
        self.max_sessions = max(1, int(max_sessions))
        self.connect_timeout_s = connect_timeout_s
        self.active = 0
        self.sessions_served = 0
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return self.process.base_url if self.process is not None else str(self.url)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(url=self.base_url, managed=self.process is not None, runtime=self.runtime, active_sessions=self.active,
                   sessions_served=self.sessions_served)  # fmt: skip
        if self.process is not None:
            out["process"] = self.process.status()
        return out

    def _ssl(self) -> ssl.SSLContext | None:
        if not self.base_url.startswith("https://"):
            return None
        ctx = ssl.create_default_context()
        if (urlsplit(self.base_url).hostname or "") in LOOPBACK:
            # its server serves https with a certificate it makes itself; bolo only skips the check on loopback
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def open(self, config: DuplexConfig) -> DuplexSession:
        async with self._lock:
            if self.active >= self.max_sessions:
                raise Overloaded(
                    f"the full-duplex model '{self.model}' is in {self.active} conversation(s), as many as it holds at "
                    f"once ({self.max_sessions}); try again when one ends"
                )
            self.active += 1
        try:
            if self.process is not None:
                await asyncio.to_thread(self.process.ensure_started)
            try:
                client = await SocketIOClient.connect(
                    self.base_url, ssl_context=self._ssl(), timeout_s=self.connect_timeout_s
                )
            except EngineUnavailable:
                raise
            except Exception as exc:
                raise EngineUnavailable(
                    f"the Freeze-Omni server at {self.base_url} is not reachable ({type(exc).__name__}: {exc})"
                ) from exc
            try:
                if config.instructions:
                    await client.emit("prompt_text", config.instructions)
                    await self._expect(client, "prompt_success")
                await client.emit("recording-started")
            except BaseException:
                await client.close()
                raise
        except BaseException:
            async with self._lock:
                self.active -= 1
            raise
        self.sessions_served += 1
        return _Released(FreezeOmniSession(client, self.base_url), self)

    async def _expect(self, client: SocketIOClient, name: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.connect_timeout_s
        while True:
            event = await asyncio.wait_for(client.next_event(), max(0.01, deadline - loop.time()))
            if event is None:
                raise EngineUnavailable(f"the Freeze-Omni server at {self.base_url} closed the connection")
            if event[0] == "too_many_users":
                raise Overloaded("the Freeze-Omni server has as many users as it takes (--max_users)")
            if event[0] == name:
                return

    async def _release(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)

    async def close(self) -> None:
        if self.process is not None:
            await asyncio.to_thread(self.process.stop)


class _Released(DuplexSession):
    def __init__(self, inner: FreezeOmniSession, backend: FreezeOmniBackend) -> None:
        self.inner = inner
        self.backend = backend
        self.input_rate = inner.input_rate
        self.frame_samples = inner.frame_samples
        self._released = False

    async def push(self, samples: FloatArray) -> None:
        await self.inner.push(samples)

    def events(self) -> AsyncIterator[DuplexEvent]:
        return self.inner.events()

    async def close(self) -> None:
        await self.inner.close()
        if not self._released:
            self._released = True
            await self.backend._release()
