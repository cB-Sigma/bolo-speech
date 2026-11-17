"""A client of Kyutai's full-duplex WebSocket protocol (``/api/chat``), as a :class:`~bolo.duplex.base.DuplexBackend`.

Kyutai's servers for Moshi speak it (the PyTorch ``moshi.server``, the MLX ``moshi_mlx.local_web`` and the Rust
``moshi-backend``), and so do the servers of models built on Moshi (Hibiki, PersonaPlex). The protocol, from
Kyutai's ``rust/protocol.md`` and the servers' code (moshi 0.2.13, moshi_mlx 0.3.0, moshi-backend 0.6.4):

* one WebSocket per conversation; every message is binary, its first byte the kind, the rest the payload;
* ``0`` handshake (server to client), sent once the server has given this connection the model; the Python
  servers send the byte alone, the Rust one adds two little-endian ``u32`` (protocol and model version, 0);
* ``1`` audio, both ways: the bytes of an Ogg Opus stream (mono, 24 kHz; :mod:`bolo.duplex.ogg_opus`), the
  headers first, then pages as audio is produced. The model steps once per 80 ms (1920 samples) of input;
* ``2`` text (server to client): one text token of what the model says, UTF-8, with its leading space;
* ``3`` control and ``6`` ping: unused in full streaming; ``4`` metadata (JSON, Rust server); ``5`` error (UTF-8).

The Python servers serve one connection at a time (a lock: a second connection waits, without its handshake,
until the first one ends) and the MLX server keeps the model's state from one connection to the next; the Rust
server takes several connections at once (each one more stream on the GPU), reads sampling options from the
query string (``text_temperature``, ``text_topk``, ``audio_temperature``, ``audio_topk``, ``max_steps``, ``pad_mult``,
``repetition_penalty``, ``repetition_penalty_context``, ``text_seed``, ``audio_seed``) and ends a connection after
360 s. The message kinds below are the protocol's (Copyright (c) Kyutai, MIT and Apache-2.0; NOTICE).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit

from bolo.duplex.base import DuplexAudio, DuplexBackend, DuplexConfig, DuplexEnd, DuplexEvent, DuplexSession, DuplexText
from bolo.duplex.ogg_opus import OggOpusReader, OggOpusWriter
from bolo.engines.base import FloatArray
from bolo.errors import EngineUnavailable, Overloaded

if TYPE_CHECKING:
    from bolo.engines.tts_server import ServerProcess

log = logging.getLogger("bolo.duplex")

MSG_HANDSHAKE, MSG_AUDIO, MSG_TEXT, MSG_CONTROL, MSG_METADATA, MSG_ERROR, MSG_PING = range(7)
SAMPLE_RATE = 24000
FRAME = 1920  # 80 ms: one model step (Mimi runs at 12.5 frames a second)
#: query-string options the Rust server reads (the Python servers ignore the query string)
QUERY_OPTIONS = (
    "text_temperature", "text_topk", "audio_temperature", "audio_topk", "max_steps", "pad_mult", "repetition_penalty",
    "repetition_penalty_context", "text_seed", "audio_seed",
)  # fmt: skip
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def chat_url(url: str) -> str:
    """``http://host:port`` (or ``https``, ``ws``, ``wss``; with or without ``/api/chat``) → the WebSocket URL."""
    url = url.strip().rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if url.startswith(http):
            url = ws + url[len(http) :]
    if not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    return url if urlsplit(url).path not in ("", "/") else url + "/api/chat"


class KyutaiSession(DuplexSession):
    input_rate = SAMPLE_RATE
    frame_samples = FRAME

    def __init__(self, ws: Any, where: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.ws = ws
        self.where = where
        self.metadata = metadata or {}
        self._writer = OggOpusWriter(SAMPLE_RATE)
        self._reader = OggOpusReader(SAMPLE_RATE)
        self._queue: asyncio.Queue[DuplexEvent] = asyncio.Queue()
        self._closing = False
        self._ended = False
        self._send_lock = asyncio.Lock()
        self._task = asyncio.create_task(self._read(), name="bolo-duplex-kyutai")

    async def _read(self) -> None:
        end = DuplexEnd("ended")
        try:
            async for message in self.ws:
                if isinstance(message, str) or not message:
                    continue
                kind, payload = message[0], bytes(message[1:])
                if kind == MSG_AUDIO:
                    pcm = self._reader.append(payload)
                    if len(pcm):
                        await self._queue.put(DuplexAudio(pcm, SAMPLE_RATE))
                elif kind == MSG_TEXT:
                    text = payload.decode("utf-8", errors="replace")
                    if text:
                        await self._queue.put(DuplexText(text))
                elif kind == MSG_ERROR:
                    end = DuplexEnd(
                        "failed", payload.decode("utf-8", errors="replace") or "the server reported an error"
                    )
                    break
                elif kind == MSG_METADATA:
                    with contextlib.suppress(ValueError):
                        self.metadata = json.loads(payload)
                # handshake (again), control, ping and unknown kinds are ignored, as the protocol says
        except asyncio.CancelledError:
            end = DuplexEnd("closed")
            raise
        except Exception as exc:  # the connection broke
            if not self._closing:
                end = DuplexEnd("failed", f"the connection to {self.where} broke ({type(exc).__name__}: {exc})")
        finally:
            if self._closing:
                end = DuplexEnd("closed")
            self._ended = True
            self._queue.put_nowait(end)

    async def push(self, samples: FloatArray) -> None:
        if self._ended or self._closing or not len(samples):
            return
        data = self._writer.append(samples)
        if data:
            async with self._send_lock:
                try:
                    await self.ws.send(bytes([MSG_AUDIO]) + data)
                except Exception as exc:
                    raise EngineUnavailable(
                        f"the full-duplex server at {self.where} closed the session: {exc}"
                    ) from exc

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
            await asyncio.wait_for(self.ws.close(), 5.0)
        if not self._task.done():
            self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task


class KyutaiBackend(DuplexBackend):
    """A server that speaks Kyutai's protocol: one bolo started (:class:`ServerProcess`) or one at ``url``."""

    name = "kyutai"

    def __init__(
        self,
        url: str | None = None,
        *,
        process: ServerProcess | None = None,
        model: str | None = None,
        runtime: str | None = None,
        max_sessions: int = 1,
        restart_after_session: bool = False,
        handshake_timeout_s: float = 30.0,
        query: dict[str, Any] | None = None,
        voice_param: str | None = None,
        voice_suffix: str = "",
        default_voice: str | None = None,
        instructions_param: str | None = None,
    ) -> None:
        if url is None and process is None:
            raise ValueError("a URL or a process is needed")
        self.url = url
        self.process = process
        self.model = model
        self.runtime = runtime
        self.max_sessions = max(1, int(max_sessions))
        self.restart_after_session = restart_after_session
        self.handshake_timeout_s = handshake_timeout_s
        #: fixed query parameters (the catalog entry's), and where the session's voice and instructions go
        #: (PersonaPlex: voice_prompt=<voice>.pt, text_prompt=<instructions>)
        self.query = {k: v for k, v in (query or {}).items() if v is not None}
        self.voice_param = voice_param
        self.voice_suffix = voice_suffix
        self.default_voice = default_voice
        self.instructions_param = instructions_param
        self.active = 0
        self.sessions_served = 0
        self._lock = asyncio.Lock()

    @property
    def ws_url(self) -> str:
        return chat_url(self.process.base_url if self.process is not None else str(self.url))

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            url=self.ws_url, managed=self.process is not None, runtime=self.runtime, active_sessions=self.active,
            sessions_served=self.sessions_served, restart_after_session=self.restart_after_session,
        )  # fmt: skip
        if self.process is not None:
            out["process"] = self.process.status()
        return out

    def _ssl(self, url: str) -> ssl.SSLContext | None:
        if not url.startswith("wss://"):
            return None
        host = urlsplit(url).hostname or ""
        if self.process is not None and host in LOOPBACK:
            # the Rust server bolo started serves TLS with a certificate it made itself, on the loopback address
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return ssl.create_default_context()

    async def open(self, config: DuplexConfig) -> DuplexSession:
        from websockets.asyncio.client import connect

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
            url = self._url(config)
            session = await self._connect(connect, url)
        except BaseException:
            async with self._lock:
                self.active -= 1
            raise
        self.sessions_served += 1
        return _Tracked(session, self)

    def _url(self, config: DuplexConfig) -> str:
        from bolo.s2s import OPENAI_VOICES

        options: dict[str, Any] = {**self.query, **{k: v for k, v in config.options.items() if k in QUERY_OPTIONS}}
        if self.voice_param:
            voice = config.voice if config.voice and config.voice.lower() not in OPENAI_VOICES else self.default_voice
            options[self.voice_param] = f"{voice}{self.voice_suffix}" if voice else ""
        if self.instructions_param:
            options[self.instructions_param] = config.instructions or ""
        return self.ws_url + (("&" if "?" in self.ws_url else "?") + urlencode(options) if options else "")

    async def _connect(self, connect: Any, url: str) -> KyutaiSession:
        where = url.split("?", 1)[0]
        ws: Any = None
        try:
            ws = await connect(url, ssl=self._ssl(url), max_size=None, open_timeout=self.handshake_timeout_s)
            first = await asyncio.wait_for(ws.recv(), self.handshake_timeout_s)
        except TimeoutError as exc:
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
            raise Overloaded(
                f"the full-duplex server at {where} did not answer the handshake within {self.handshake_timeout_s:.0f} s: "
                "Kyutai's Python and MLX servers hold one conversation at a time, so another client may be using it"
            ) from exc
        except Exception as exc:
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
            if self.process is not None:
                self.process.check()
            raise EngineUnavailable(
                f"the full-duplex server at {where} is not reachable ({type(exc).__name__}: {exc})"
            ) from exc
        if isinstance(first, str) or not first or first[0] != MSG_HANDSHAKE:
            with contextlib.suppress(Exception):
                await ws.close()
            kind = "text" if isinstance(first, str) else (first[:1].hex() or "empty")
            if not isinstance(first, str) and first and first[0] == MSG_ERROR:
                raise EngineUnavailable(
                    f"the full-duplex server at {where} refused: {bytes(first[1:]).decode(errors='replace')}"
                )
            raise EngineUnavailable(f"the full-duplex server at {where} did not start with a handshake (got {kind})")
        return KyutaiSession(ws, where)

    async def _release(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)
            restart = self.restart_after_session and self.process is not None and self.active == 0
        if restart:
            await self.session_closed()

    async def session_closed(self) -> None:
        """The MLX server keeps the model's state between connections: bolo stops it after a conversation, and the
        next one starts it afresh."""
        if self.process is not None and self.restart_after_session:
            await asyncio.to_thread(self.process.stop)

    async def close(self) -> None:
        if self.process is not None:
            await asyncio.to_thread(self.process.stop)


class _Tracked(DuplexSession):
    """A session that gives its slot back to the backend when it closes."""

    def __init__(self, inner: KyutaiSession, backend: KyutaiBackend) -> None:
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
