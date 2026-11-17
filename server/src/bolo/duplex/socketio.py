"""A minimal Socket.IO client (protocol 5 over Engine.IO 4, WebSocket transport only) for full-duplex model servers
that speak Socket.IO (Freeze-Omni's Flask-SocketIO server).

Only what those servers use: connecting to the default namespace, emitting events with JSON arguments, receiving
events whose arguments are JSON or binary attachments, and answering the server's pings. Written from the protocol
documents (socket.io-protocol v5 and engine.io-protocol v4); the tests run it against python-socketio's own server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from typing import Any

from bolo.errors import EngineUnavailable

#: Engine.IO packet types (as the first character of a text frame)
EIO_OPEN, EIO_CLOSE, EIO_PING, EIO_PONG, EIO_MESSAGE = "0", "1", "2", "3", "4"
#: Socket.IO packet types (after the Engine.IO message type)
SIO_CONNECT, SIO_DISCONNECT, SIO_EVENT, SIO_ACK, SIO_CONNECT_ERROR, SIO_BINARY_EVENT = "0", "1", "2", "3", "4", "5"


def socketio_url(base: str) -> str:
    """``https://host:port`` → ``wss://host:port/socket.io/?EIO=4&transport=websocket``."""
    base = base.rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            base = ws + base[len(http) :]
    return base + "/socket.io/?EIO=4&transport=websocket"


def _fill(value: Any, attachments: list[bytes]) -> Any:
    """Put binary attachments back where the packet has ``{"_placeholder": true, "num": n}``."""
    if isinstance(value, dict):
        if value.get("_placeholder") is True and isinstance(value.get("num"), int):
            return attachments[value["num"]]
        return {k: _fill(v, attachments) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, attachments) for v in value]
    return value


class SocketIOClient:
    """One connection. :meth:`events` yields ``(name, args)`` for every event the server emits."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.sid: str | None = None
        self._queue: asyncio.Queue[tuple[str, list[Any]] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.closed = False

    @classmethod
    async def connect(
        cls, url: str, *, ssl_context: ssl.SSLContext | None = None, timeout_s: float = 30.0
    ) -> SocketIOClient:
        from websockets.asyncio.client import connect

        ws = await connect(socketio_url(url), ssl=ssl_context, max_size=None, open_timeout=timeout_s)
        client = cls(ws)
        try:
            opened = await asyncio.wait_for(ws.recv(), timeout_s)
            if not isinstance(opened, str) or not opened.startswith(EIO_OPEN):
                raise EngineUnavailable(f"not an Engine.IO server (it said {str(opened)[:40]!r})")
            await ws.send(EIO_MESSAGE + SIO_CONNECT)
            while True:
                frame = await asyncio.wait_for(ws.recv(), timeout_s)
                if isinstance(frame, str) and frame == EIO_PING:
                    await ws.send(EIO_PONG)
                    continue
                if isinstance(frame, str) and frame.startswith(EIO_MESSAGE + SIO_CONNECT):
                    with contextlib.suppress(ValueError):
                        client.sid = (json.loads(frame[2:] or "{}") or {}).get("sid")
                    break
                if isinstance(frame, str) and frame.startswith(EIO_MESSAGE + SIO_CONNECT_ERROR):
                    raise EngineUnavailable(f"the Socket.IO server refused the connection: {frame[2:]}")
        except BaseException:
            with contextlib.suppress(Exception):
                await ws.close()
            raise
        client._task = asyncio.create_task(client._read(), name="bolo-socketio")
        return client

    async def _read(self) -> None:
        pending: tuple[list[Any], int] | None = None  # a binary event waiting for its attachments
        attachments: list[bytes] = []
        try:
            async for frame in self.ws:
                if isinstance(frame, bytes):
                    if pending is None:
                        continue
                    attachments.append(frame)
                    if len(attachments) == pending[1]:
                        data = _fill(pending[0], attachments)
                        await self._queue.put((str(data[0]), list(data[1:])))
                        pending, attachments = None, []
                    continue
                if frame == EIO_PING:
                    await self.ws.send(EIO_PONG)
                    continue
                if frame.startswith(EIO_CLOSE) or not frame.startswith(EIO_MESSAGE):
                    if frame.startswith(EIO_CLOSE):
                        break
                    continue
                kind, body = frame[1:2], frame[2:]
                if kind == SIO_DISCONNECT:
                    break
                if kind == SIO_EVENT:
                    start = body.find("[")
                    data = json.loads(body[start:]) if start >= 0 else []
                    if data:
                        await self._queue.put((str(data[0]), list(data[1:])))
                elif kind == SIO_BINARY_EVENT:
                    count, _, rest = body.partition("-")
                    start = rest.find("[")
                    data = json.loads(rest[start:]) if start >= 0 else []
                    pending, attachments = (data, int(count)), []
        except Exception:  # the connection broke: the queue's end says so
            pass
        finally:
            self.closed = True
            self._queue.put_nowait(None)

    async def emit(self, event: str, *args: Any) -> None:
        if self.closed:
            raise EngineUnavailable("the Socket.IO connection is closed")
        await self.ws.send(EIO_MESSAGE + SIO_EVENT + json.dumps([event, *args], separators=(",", ":")))

    async def next_event(self) -> tuple[str, list[Any]] | None:
        """The next event, or None once the connection ended."""
        return await self._queue.get()

    async def close(self) -> None:
        if not self.closed:
            with contextlib.suppress(Exception):
                await self.ws.send(EIO_MESSAGE + SIO_DISCONNECT)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.ws.close(), 5.0)
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
