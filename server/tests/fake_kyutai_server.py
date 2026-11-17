"""A stand-in for Kyutai's full-duplex WebSocket server (``/api/chat``: moshi.server, moshi_mlx.local_web,
moshi-backend, and the PersonaPlex and Hibiki-Zero servers) in tests.

It speaks the real framing: binary messages whose first byte is the kind (0 handshake, 1 audio, 2 text, 4 metadata,
5 error), audio as an Ogg Opus stream (24 kHz mono) written and read with Kyutai's own ``sphn`` library when it is
installed (as the Python servers do), else with bolo's :mod:`bolo.duplex.ogg_opus`. Like the real servers it
steps once per 1920 samples (80 ms) of input and answers each step with 1920 samples of output, so it produces
audio only as fast as it is fed.

The "model": after at least 3 loud input frames followed by 4 quiet ones it speaks ``reply`` (one text token every
other frame, a 330 Hz tone meanwhile, silence otherwise) for ``speak_frames`` frames; loud input while it speaks
makes it stop (its own barge-in). Like the Python servers it holds one connection at a time (``lock``): a second
one waits without a handshake. Run it in a thread (``FakeKyutaiServer().start()``) or as a process
(:func:`main`, with the launcher's JSON argument bolo gives a managed server).
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import threading
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

RATE = 24000
FRAME = 1920


def codec(use_sphn: bool = True) -> tuple[Any, Any]:
    """(writer, reader) of Ogg Opus at 24 kHz: Kyutai's sphn when available, else bolo's own."""
    if use_sphn:
        try:
            import sphn

            writer, reader = sphn.OpusStreamWriter(RATE), sphn.OpusStreamReader(RATE)
            return writer.append_pcm, reader.append_bytes
        except ImportError:
            pass
    from bolo.duplex.ogg_opus import OggOpusReader, OggOpusWriter

    w, r = OggOpusWriter(RATE), OggOpusReader(RATE)
    return w.append, r.append


class FakeKyutaiServer:
    def __init__(
        self,
        *,
        port: int = 0,
        reply: tuple[str, ...] = (" Hello", " there", "."),
        speak_frames: int = 12,
        lock: bool = True,
        rust_handshake: bool = False,
        max_steps: int | None = None,
        use_sphn: bool = True,
        error: str | None = None,
    ) -> None:
        self.port = port
        self.reply = reply
        self.speak_frames = speak_frames
        self.lock_connections = lock
        self.rust_handshake = rust_handshake
        self.max_steps = max_steps
        self.use_sphn = use_sphn
        #: send this as an error message (kind 5) right after the handshake
        self.error = error
        #: per connection: {"query", "samples", "steps", "barge_ins", "replies", "closed"}
        self.connections: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Future[None] | None = None
        self._lock: asyncio.Lock | None = None
        self.thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeKyutaiServer:
        self.thread.start()
        self._ready.wait(10)
        return self

    def stop(self) -> None:
        if self.loop is not None and self._stop is not None and not self._stop.done():
            self.loop.call_soon_threadsafe(self._stop.set_result, None)
        self.thread.join(5)

    def _check(self, connection: ServerConnection, request: Request) -> Response | None:
        if not request.path.startswith("/api/chat"):
            # Kyutai's Python servers answer 404 here with --static none (bolo's readiness check takes any answer)
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._stop = self.loop.create_future()
        self._lock = asyncio.Lock()
        async with serve(self._session, "127.0.0.1", self.port, process_request=self._check, max_size=None) as server:
            self.port = int(next(iter(server.sockets)).getsockname()[1])
            self._ready.set()
            await self._stop

    async def _session(self, ws: ServerConnection) -> None:
        assert self._lock is not None
        query = {k: v[0] for k, v in parse_qs(urlsplit(ws.request.path if ws.request else "").query).items()}
        state: dict[str, Any] = {
            "query": query,
            "samples": 0,
            "steps": 0,
            "barge_ins": 0,
            "replies": 0,
            "closed": False,
        }
        self.connections.append(state)
        if self.lock_connections:
            async with self._lock:
                await self._serve(ws, state)
        else:
            await self._serve(ws, state)

    async def _serve(self, ws: ServerConnection, state: dict[str, Any]) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        write, read = codec(self.use_sphn)
        try:
            await ws.send(b"\x00" + (struct.pack("<II", 0, 0) if self.rust_handshake else b""))
            if self.rust_handshake:
                await ws.send(b"\x04" + json.dumps({"instance_name": "fake"}).encode())
            if self.error:
                await ws.send(b"\x05" + self.error.encode())
                return
            pending = np.zeros(0, dtype=np.float32)
            heard, quiet, speaking, said = 0, 0, 0, 0
            t = 0
            async for message in ws:
                if isinstance(message, str) or not message or message[0] != 1:
                    continue  # the servers ignore everything but audio
                pcm = np.asarray(read(bytes(message[1:])), dtype=np.float32).reshape(-1)
                state["samples"] += len(pcm)
                pending = np.concatenate([pending, pcm])
                while len(pending) >= FRAME:
                    frame, pending = pending[:FRAME], pending[FRAME:]
                    state["steps"] += 1
                    loud = float(np.sqrt(np.mean(frame * frame))) > 0.02
                    if loud:
                        heard, quiet = heard + 1, 0
                        if speaking:
                            speaking = 0  # the caller talks over it: it stops
                            state["barge_ins"] += 1
                    else:
                        quiet += 1
                        if heard >= 3 and quiet >= 4 and not speaking:
                            speaking, said, heard = self.speak_frames, 0, 0
                            state["replies"] += 1
                    if speaking:
                        n = np.arange(t, t + FRAME)
                        out = (0.3 * np.sin(2 * np.pi * 330 * n / RATE)).astype(np.float32)
                        index = self.speak_frames - speaking
                        if index % 2 == 0 and said < len(self.reply):
                            await ws.send(b"\x02" + self.reply[said].encode())
                            said += 1
                        speaking -= 1
                    else:
                        out = np.zeros(FRAME, dtype=np.float32)
                    t += FRAME
                    data = write(out)
                    if len(data):
                        await ws.send(b"\x01" + bytes(data))
                    if self.max_steps is not None and state["steps"] >= self.max_steps:
                        await ws.close()
                        return
        except Exception:  # the client went away
            pass
        finally:
            state["closed"] = True
            self.active -= 1


def main(argv: list[str]) -> None:
    """As a process: the last argument is the JSON bolo's launcher takes ({"module", "args", "host", "port", ...})."""
    spec = json.loads(argv[-1])
    server = FakeKyutaiServer(port=int(spec["port"]))
    print(f"fake {spec['module']} {' '.join(spec['args'])}", flush=True)
    server.start()
    server.thread.join()


if __name__ == "__main__":
    main(sys.argv[1:])
