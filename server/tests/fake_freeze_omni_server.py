"""A stand-in for Freeze-Omni's Socket.IO server (VITA-MLLM/Freeze-Omni ``bin/server.py``) in tests, built on
python-socketio's own server so that bolo's Socket.IO client is checked against a real implementation.

Events as the real server's: ``prompt_text`` → ``prompt_success``, ``recording-started``, ``audio`` (a JSON string
``{"sample_rate", "audio": [bytes]}``), ``recording-stopped``; it emits ``audio`` (16-bit PCM at 24 kHz, binary) at
most once per ``audio`` event it receives, ``stop_tts`` when the caller talks over it, and ``too_many_users`` past
``max_users``. The "model": 0.3 s of loud input followed by 0.2 s of quiet makes it answer with ``answer_chunks``
pieces of 0.1 s of tone.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any

import numpy as np
import socketio
import uvicorn


class FakeFreezeOmniServer:
    def __init__(self, *, max_users: int = 5, answer_chunks: int = 200) -> None:
        self.max_users = max_users
        self.answer_chunks = answer_chunks
        self.prompts: list[str] = []
        self.users: dict[str, dict[str, Any]] = {}
        self.sessions: list[dict[str, Any]] = []
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = int(s.getsockname()[1])
        sio = socketio.AsyncServer(async_mode="asgi", max_http_buffer_size=10_000_000)
        self.sio = sio
        sio.on("connect", self._connect)
        sio.on("disconnect", self._disconnect)
        sio.on("prompt_text", self._prompt)
        sio.on("recording-started", self._started)
        sio.on("recording-stopped", self._stopped)
        sio.on("audio", self._audio)
        config = uvicorn.Config(socketio.ASGIApp(sio), host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeFreezeOmniServer:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(5)

    async def _connect(self, sid: str, environ: dict[str, Any], auth: Any = None) -> None:
        if len(self.users) >= self.max_users:
            await self.sio.emit("too_many_users", to=sid)
            return
        state = {"samples": 0, "loud": 0, "quiet": 0, "answering": 0, "started": 0, "stopped": 0, "interrupted": 0,
                 "answers": 0, "closed": False}  # fmt: skip
        self.users[sid] = state
        self.sessions.append(state)

    async def _disconnect(self, sid: str, *args: Any) -> None:
        state = self.users.pop(sid, None)
        if state is not None:
            state["closed"] = True

    async def _prompt(self, sid: str, text: str) -> None:
        self.prompts.append(text)
        await self.sio.emit("prompt_success", to=sid)

    async def _started(self, sid: str) -> None:
        if sid in self.users:
            self.users[sid]["started"] += 1

    async def _stopped(self, sid: str) -> None:
        if sid in self.users:
            self.users[sid]["stopped"] += 1

    async def _audio(self, sid: str, data: str) -> None:
        state = self.users.get(sid)
        if state is None:
            return
        payload = json.loads(data)
        pcm = np.frombuffer(bytes(payload["audio"]), dtype="<i2").astype(np.float32) / 32768.0
        state["samples"] += len(pcm)
        loud = len(pcm) > 0 and float(np.sqrt(np.mean(pcm * pcm))) > 0.02
        if loud:
            state["loud"] += len(pcm)
            state["quiet"] = 0
            if state["answering"]:
                state["answering"] = 0
                state["interrupted"] += 1
                await self.sio.emit("stop_tts", to=sid)
        else:
            state["quiet"] += len(pcm)
            if state["loud"] >= 0.3 * 16000 and state["quiet"] >= 0.2 * 16000 and not state["answering"]:
                state["answering"], state["loud"] = self.answer_chunks, 0
                state["answers"] += 1
        if state["answering"]:
            state["answering"] -= 1
            t = np.arange(2400) / 24000
            tone = (0.3 * np.sin(2 * np.pi * 330 * t) * 32767).astype("<i2").tobytes()
            await self.sio.emit("audio", tone, to=sid)
        await asyncio.sleep(0)
