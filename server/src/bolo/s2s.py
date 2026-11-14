"""Speech-to-speech backends for Realtime sessions (the ``speech-to-speech`` pipeline).

bolo's Realtime session (:mod:`bolo.realtime`) owns everything the client sees: authentication, the event set,
telephony codecs and resampling, turn detection and barge-in, conversation items, tool-call items, transcripts
of the user (from bolo's own speech recogniser) and traces. A speech-to-speech model behind it only has to:

* take the user's audio of a turn (16 kHz mono float32, as bolo hears it) or a text message,
* end the turn (:meth:`S2SConversation.commit`),
* produce a response as a stream of :data:`S2SEvent` (text, audio at the model's rate, tool calls, the end),
* stop a response early (barge-in), and take the results of tool calls.

That is :class:`S2SConversation`, opened per Realtime session by an :class:`S2SBackend`. Backends:

* :class:`LlamaRealtimeBackend`: a llama.cpp ``llama-server`` speaking its WebSocket ``/v1/realtime``
  (LFM3-Audio and LFM3-Omni conversations with tool calls; tools/server/README.md, "Realtime conversation").
  Either one the operator runs (setting ``s2s_server_url``) or one bolo starts for the catalog model
  ``s2s_model`` with the llama-server binary of the ``tts-server`` engine (start, ``/health``, crash back-off as
  for TTS servers). Each Realtime session holds one upstream session, which holds one llama-server slot: the
  managed server is started with ``--parallel s2s_parallel``.

An in-process backend (onnxruntime / onnxruntime-genai exports of a model and its audio codec) implements the
same two classes; nothing in the session logic depends on llama-server.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.engines.base import FloatArray
from bolo.errors import ComponentMissing, EngineUnavailable, InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.engines.tts_server import ServerProcess

log = logging.getLogger("bolo.s2s")

#: OpenAI voice names that stock clients send; a backend with its own voices never sees them
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar"}  # fmt: skip
INPUT_RATE = 16000


# -- what a response is made of ----------------------------------------------------------------------------------
@dataclass
class TextDelta:
    text: str


@dataclass
class AudioDelta:
    samples: FloatArray
    sample_rate: int


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class ResponseEnd:
    #: "completed", "incomplete" (output token limit), "cancelled" or "failed"
    status: str = "completed"
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None


S2SEvent = TextDelta | AudioDelta | ToolCall | ResponseEnd


@dataclass
class S2SConfig:
    """The parts of a Realtime session configuration a speech-to-speech model takes (before the first turn)."""

    instructions: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = "auto"
    voice: str | None = None
    max_output_tokens: Any = "inf"


class S2SConversation(abc.ABC):
    """One conversation with a speech-to-speech model, for one Realtime session."""

    #: sample rate of the audio in :class:`AudioDelta`
    output_rate: int = 24000

    @abc.abstractmethod
    async def append_audio(self, samples16: FloatArray) -> None:
        """Audio of the current user turn, 16 kHz mono."""

    @abc.abstractmethod
    async def commit(self) -> None:
        """End the user turn (its audio so far is the user's message)."""

    @abc.abstractmethod
    async def add_text(self, text: str) -> None:
        """A user message as text."""

    @abc.abstractmethod
    async def add_tool_output(self, call_id: str, output: str) -> None:
        """The result of a tool call of the previous response."""

    @abc.abstractmethod
    def respond(
        self, *, modalities: list[str], max_output_tokens: Any = None, tool_choice: Any = None
    ) -> AsyncGenerator[S2SEvent, None]:
        """Generate the response to the conversation so far. The last event is a :class:`ResponseEnd`.

        Cancelling the task that iterates stops the response in the model too (barge-in)."""

    @abc.abstractmethod
    async def close(self) -> None: ...


class S2SBackend(abc.ABC):
    name: str = "s2s"

    @abc.abstractmethod
    async def open(self, config: S2SConfig) -> S2SConversation: ...

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}

    async def close(self) -> None:
        return None


# -- llama-server /v1/realtime ------------------------------------------------------------------------------------
def realtime_url(url: str) -> str:
    """``http://host:port[/v1]`` (or ``ws://…``) → ``ws://host:port/v1/realtime``."""
    url = url.strip().rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    elif not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    if url.endswith("/v1/realtime"):
        return url
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url + "/v1/realtime"


def _pcm16(data: bytes) -> FloatArray:
    out: FloatArray = (np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0).astype(np.float32)
    return out


def _b64_pcm16(samples: FloatArray) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode()


class LlamaRealtimeConversation(S2SConversation):
    """An upstream ``/v1/realtime`` session of llama-server (realtime conversation mode)."""

    def __init__(self, ws: Any, output_rate: int, where: str) -> None:
        self.ws = ws
        self.output_rate = output_rate
        self.where = where
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        #: responses that were cancelled and whose response.done has not come back yet: their events are dropped
        self._stale = 0
        #: a response.create was sent and its response.done has not come back
        self._active = False
        self._closed = False
        self._reader = asyncio.create_task(self._read(), name="bolo-s2s-upstream")

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = str(event.get("type", ""))
                if kind.startswith("response.") or kind == "error":
                    if kind == "error" and not self._active and not self._stale:
                        log.warning("%s: %s", self.where, (event.get("error") or {}).get("message"))
                        continue
                    await self._events.put(event)
                else:
                    log.debug("%s: %s", self.where, kind)
        except Exception as exc:  # the connection broke
            log.warning("%s: connection lost (%s: %s)", self.where, type(exc).__name__, exc)
        finally:
            self._closed = True
            await self._events.put(None)

    async def _send(self, event: dict[str, Any]) -> None:
        if self._closed:
            raise EngineUnavailable(f"the speech-to-speech server at {self.where} closed the session")
        try:
            await self.ws.send(json.dumps(event))
        except Exception as exc:
            raise EngineUnavailable(f"the speech-to-speech server at {self.where} is not reachable: {exc}") from exc

    async def append_audio(self, samples16: FloatArray) -> None:
        if len(samples16):
            await self._send({"type": "input_audio_buffer.append", "audio": _b64_pcm16(samples16)})

    async def commit(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

    async def add_text(self, text: str) -> None:
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
        await self._send({"type": "conversation.item.create", "item": item})

    async def add_tool_output(self, call_id: str, output: str) -> None:
        item = {"type": "function_call_output", "call_id": call_id, "output": output}
        await self._send({"type": "conversation.item.create", "item": item})

    async def respond(
        self, *, modalities: list[str], max_output_tokens: Any = None, tool_choice: Any = None
    ) -> AsyncGenerator[S2SEvent, None]:
        params: dict[str, Any] = {"output_modalities": ["text", "audio"] if "audio" in modalities else ["text"]}
        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens
        if tool_choice in ("auto", "none", "required"):
            params["tool_choice"] = tool_choice
        self._active = True
        await self._send({"type": "response.create", "response": params})
        started = False
        try:
            while True:
                event = await self._events.get()
                if event is None:
                    raise EngineUnavailable(f"the speech-to-speech server at {self.where} closed the session")
                kind = event.get("type")
                if self._stale:  # what is left of a cancelled response
                    if kind in ("response.done", "error"):
                        self._stale -= 1
                    continue
                if kind == "error":
                    message = str((event.get("error") or {}).get("message") or "error")
                    self._active = False
                    yield ResponseEnd("failed", error=message)
                    return
                if kind == "response.created":
                    started = True
                elif kind == "response.output_text.delta" and event.get("delta"):
                    yield TextDelta(str(event["delta"]))
                elif kind == "response.output_audio.delta" and event.get("delta"):
                    yield AudioDelta(_pcm16(base64.b64decode(str(event["delta"]))), self.output_rate)
                elif kind == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        yield ToolCall(
                            str(item.get("call_id") or item.get("id") or ""),
                            str(item.get("name") or ""),
                            str(item.get("arguments") or "{}"),
                        )
                elif kind == "response.done":
                    response = event.get("response") or {}
                    usage = {k: int(v) for k, v in (response.get("usage") or {}).items() if isinstance(v, int)}
                    status = str(response.get("status") or "completed")
                    self._active = False
                    yield ResponseEnd(
                        status if status in ("completed", "incomplete", "cancelled", "failed") else "completed", usage
                    )
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # barge-in or the client's response.cancel: stop the model and drop the rest of this response
            if self._active:
                self._active = False
                self._stale += 1
                with contextlib.suppress(Exception):
                    await asyncio.shield(self._send({"type": "response.cancel"}))
            raise
        finally:
            if not started:
                log.debug("%s: response ended before response.created", self.where)

    async def close(self) -> None:
        self._closed = True
        with contextlib.suppress(Exception):
            await self.ws.close()
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._reader


class LlamaRealtimeBackend(S2SBackend):
    """llama-server's WebSocket ``/v1/realtime`` (operator-run, or a :class:`ServerProcess` bolo started)."""

    name = "llama-server"

    def __init__(
        self,
        url: str | None = None,
        *,
        api_key: str | None = None,
        process: ServerProcess | None = None,
        model: str | None = None,
        connect_timeout_s: float = 30.0,
    ) -> None:
        if url is None and process is None:
            raise ValueError("a URL or a process is needed")
        self.url = url
        self.process = process
        self.api_key = api_key or (process.api_key if process is not None else None)
        self.model = model
        self.connect_timeout_s = connect_timeout_s

    @property
    def ws_url(self) -> str:
        base = self.process.base_url if self.process is not None else str(self.url)
        return realtime_url(base)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": self.name, "url": self.ws_url, "model": self.model}
        out["managed"] = self.process is not None
        if self.process is not None:
            out["process"] = self.process.status()
        return out

    async def open(self, config: S2SConfig) -> S2SConversation:
        from websockets.asyncio.client import connect

        if self.process is not None:
            await asyncio.to_thread(self.process.ensure_started)
        url = self.ws_url
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        ws: Any = None
        try:
            ws = await connect(url, additional_headers=headers, max_size=None, open_timeout=self.connect_timeout_s)
            created = await asyncio.wait_for(self._next(ws, ("session.created",)), self.connect_timeout_s)
        except Exception as exc:
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
            if self.process is not None:
                self.process.check()
            if isinstance(exc, EngineUnavailable):
                raise
            raise EngineUnavailable(
                f"the speech-to-speech server at {url} is not reachable ({type(exc).__name__}: {exc})"
            ) from exc
        session = created.get("session") or {}
        out_fmt = ((session.get("audio") or {}).get("output") or {}).get("format") or {}
        output_rate = int(out_fmt.get("rate") or 24000)
        update: dict[str, Any] = {
            "type": "realtime",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": INPUT_RATE}}},
            "tools": [
                {
                    "type": "function",
                    "name": t.get("name"),
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                }
                for t in config.tools
            ],
        }
        if config.instructions:
            update["instructions"] = config.instructions
        if config.tool_choice in ("auto", "none", "required"):
            update["tool_choice"] = config.tool_choice
        if isinstance(config.max_output_tokens, int):
            update["max_output_tokens"] = config.max_output_tokens
        if config.voice and config.voice.lower() not in OPENAI_VOICES:
            update["audio"]["output"] = {"voice": config.voice}
        try:
            await ws.send(json.dumps({"type": "session.update", "session": update}))
            reply = await asyncio.wait_for(self._next(ws, ("session.updated", "error")), self.connect_timeout_s)
        except Exception as exc:
            with contextlib.suppress(Exception):
                await ws.close()
            raise EngineUnavailable(f"the speech-to-speech server at {url} did not take the session: {exc}") from exc
        if reply.get("type") == "error":
            with contextlib.suppress(Exception):
                await ws.close()
            message = str((reply.get("error") or {}).get("message") or "error")
            raise InvalidRequest(f"the speech-to-speech model refused the session configuration: {message}")
        return LlamaRealtimeConversation(ws, output_rate, url)

    @staticmethod
    async def _next(ws: Any, kinds: tuple[str, ...]) -> dict[str, Any]:
        while True:
            raw = await ws.recv()
            with contextlib.suppress(ValueError):
                event = json.loads(raw)
                if isinstance(event, dict) and event.get("type") in kinds:
                    return event
                if isinstance(event, dict) and event.get("type") == "error" and "error" not in kinds:
                    message = str((event.get("error") or {}).get("message") or "error")
                    raise EngineUnavailable(f"the speech-to-speech server answered: {message}")

    async def close(self) -> None:
        if self.process is not None:
            await asyncio.to_thread(self.process.stop)


# -- which backend a server uses ----------------------------------------------------------------------------------
class S2SManager:
    """Builds the backend of the ``speech-to-speech`` pipeline once, from the settings (see docs/configuration.md)."""

    def __init__(self, core: Bolo) -> None:
        self.core = core
        self._backend: S2SBackend | None = None
        self._lock = asyncio.Lock()

    async def backend(self) -> S2SBackend:
        async with self._lock:
            if self._backend is None:
                self._backend = await self._build()
            return self._backend

    def describe(self) -> dict[str, Any]:
        s = self.core.settings
        out: dict[str, Any] = {"model": s.s2s_model, "url": s.s2s_server_url}
        if self._backend is not None:
            out.update(self._backend.describe())
        return out

    async def _build(self) -> S2SBackend:
        s = self.core.settings
        if s.s2s_server_url:
            return LlamaRealtimeBackend(s.s2s_server_url, api_key=s.s2s_server_api_key, model=s.s2s_model or None)
        if not s.s2s_model:
            raise EngineUnavailable(
                "the speech-to-speech pipeline needs a model: set s2s_model (a llama-server model of the catalog, "
                "e.g. lfm3-audio-2b-llamacpp) or s2s_server_url (a llama-server you run)"
            )
        spec = self.core.manager.resolve(s.s2s_model)
        options = spec.options
        if spec.engine != "tts-server" or str(options.get("runtime") or "llama-server") != "llama-server":
            raise InvalidRequest(
                f"model '{spec.id}' does not run on llama-server; s2s_model must name a llama-server model such as "
                "lfm3-audio-2b-llamacpp",
                param="s2s_model",
            )
        if not self.core.store.is_installed(spec.id):
            await self.core._auto_pull(spec)
        return LlamaRealtimeBackend(
            process=self._process(spec.id, self.core.store.model_dir(spec.id), options), model=spec.id
        )

    def _process(self, name: str, path: Path, options: dict[str, Any]) -> ServerProcess:
        import os

        from bolo.engines.tts_server import ServerProcess, free_port
        from bolo.tts_runtimes import LLAMA, RUNTIME_INFO, find_binary

        s = self.core.settings
        binary, _source = find_binary(s, LLAMA)
        if binary is None:
            raise ComponentMissing(
                f"the speech-to-speech model '{name}' runs on {RUNTIME_INFO[LLAMA].name}, which is not installed: run "
                f"`bolo components install llama-server`, set llama_server_bin, or set s2s_server_url to a running "
                "llama-server"
            )
        model_file = path / str(options.get("model_file") or "")
        if not options.get("model_file") or not model_file.is_file():
            raise EngineUnavailable(f"model '{name}': the GGUF file {model_file} is missing; pull the model again")
        port = free_port()
        key = secrets.token_hex(16)
        argv = [str(binary), "-m", str(model_file), "--host", "127.0.0.1", "--port", str(port), "--api-key", key]
        if options.get("mmproj_file"):
            argv += ["-mm", str(path / str(options["mmproj_file"]))]
        argv += ["-t", str(s.resolved_threads()), "--parallel", str(max(1, s.s2s_parallel))]
        extra = [str(a) for a in options.get("server_args") or []] + list(s.s2s_server_args)
        if s.device != "cpu" and not any(a in ("-ngl", "--n-gpu-layers", "--gpu-layers") for a in extra):
            argv += ["-ngl", "99"]
        env = {
            "LD_LIBRARY_PATH": os.pathsep.join(
                p for p in (str(binary.parent), os.environ.get("LD_LIBRARY_PATH", "")) if p
            )
        }
        return ServerProcess(
            f"s2s:{name}", argv + extra, port, start_timeout_s=s.tts_server_start_timeout_s, env=env, api_key=key
        )

    async def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            await backend.close()
