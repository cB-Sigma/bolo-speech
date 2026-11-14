"""A stand-in for llama-server's WebSocket ``/v1/realtime`` (realtime conversation mode) in tests.

The events follow llama.cpp tools/server/README.md ("Realtime conversation") and server-realtime.cpp: no server
VAD, the client commits; ``response.create`` streams ``response.output_text.delta`` and
``response.output_audio.delta`` (base64 PCM16 at ``audio.output.format.rate``), tool calls as ``function_call``
items, and ``response.done``.

Replies are a function of the conversation:

* after a ``function_call_output``          -> "The tool said: <output>."
* a user text with "weather" and tools set  -> a ``get_weather`` tool call
* a user text                               -> "You wrote: <text>."
* audio                                     -> "I heard <seconds> seconds." (one word per ``word_delay_s``)

Each word comes with 0.1 s of tone. Run in a thread (``FakeRealtimeServer().start()``) or as a process with
llama-server's command line (``-m m.gguf --port N --api-key K ...``), where ``GET /health`` answers too.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
from http import HTTPStatus
from typing import Any

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

OUT_RATE = 24000


def tone_b64(seconds: float = 0.1) -> str:
    t = np.arange(int(OUT_RATE * seconds)) / OUT_RATE
    return base64.b64encode((0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2").tobytes()).decode()


class FakeRealtimeServer:
    def __init__(
        self, api_key: str | None = None, port: int = 0, word_delay_s: float = 0.0, openai: bool = False
    ) -> None:
        #: behave like OpenAI's Realtime API where it differs from llama-server: 24 kHz PCM input, and a
        #: response that is audio (text in ``response.output_audio_transcript.delta``) or text, never both
        self.openai = openai
        self.paths: list[str] = []
        self.api_key = api_key
        self.port = port
        self.word_delay_s = word_delay_s
        #: every client event, per connection (in order)
        self.events: list[list[dict[str, Any]]] = []
        self.sessions: list[dict[str, Any]] = []
        self.cancelled = 0
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Future[None] | None = None
        self.thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeRealtimeServer:
        self.thread.start()
        self._ready.wait(10)
        return self

    def stop(self) -> None:
        if self.loop is not None and self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set_result, None)
        self.thread.join(5)

    def _check(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.path.startswith("/health"):
            return connection.respond(HTTPStatus.OK, '{"status":"ok"}\n')
        if self.api_key and request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Invalid API Key\n")
        if not request.path.startswith("/v1/realtime"):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        self.paths.append(request.path)
        return None

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._stop = self.loop.create_future()
        async with serve(self._session, "127.0.0.1", self.port, process_request=self._check) as server:
            self.port = int(next(iter(server.sockets)).getsockname()[1])
            self._ready.set()
            await self._stop

    async def _session(self, ws: ServerConnection) -> None:
        log: list[dict[str, Any]] = []
        self.events.append(log)
        state: dict[str, Any] = {"tools": [], "audio_s": 0.0, "turn": False, "text": None, "tool_output": None}
        self.sessions.append(state)
        rate = 24000 if self.openai else 16000
        task: asyncio.Task[None] | None = None
        cancel = asyncio.Event()

        async def send(event: dict[str, Any]) -> None:
            await ws.send(json.dumps({**event, "event_id": "event_x"}))

        def session() -> dict[str, Any]:
            return {
                "object": "realtime.session", "type": "realtime", "model": "fake-lfm3",
                "instructions": state.get("instructions", ""), "tools": state["tools"],
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": rate}, "turn_detection": None},
                          "output": {"format": {"type": "audio/pcm", "rate": OUT_RATE}, "voice": state.get("voice")}},
            }  # fmt: skip

        async def respond(response_id: str) -> None:
            await send({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}})
            output: list[dict[str, Any]] = []
            if state["tool_output"] is not None:
                reply = f"The tool said: {state['tool_output']}."
            elif state["text"] and "weather" in state["text"] and state["tools"]:
                call = {"id": "item_call", "object": "realtime.item", "type": "function_call", "status": "completed",
                        "name": "get_weather", "call_id": "call_0", "arguments": '{"city": "Pune"}'}  # fmt: skip
                await send({"type": "response.output_item.added", "response_id": response_id, "output_index": 0,
                            "item": {**call, "arguments": "", "status": "in_progress"}})  # fmt: skip
                await send({"type": "response.function_call_arguments.done", "response_id": response_id,
                            "item_id": "item_call", "output_index": 0, "call_id": "call_0", "name": "get_weather",
                            "arguments": call["arguments"]})  # fmt: skip
                await send({"type": "response.output_item.done", "response_id": response_id, "output_index": 0,
                            "item": call})  # fmt: skip
                await send({"type": "response.done", "response": {"id": response_id, "status": "completed",
                            "output": [call], "usage": {"output_tokens": 9}}})  # fmt: skip
                return
            elif state["text"]:
                reply = f"You wrote: {state['text']}."
            else:
                reply = f"I heard {state['audio_s']:.1f} seconds."
            state.update(turn=False, text=None, tool_output=None, audio_s=0.0)
            item = {"id": "item_msg", "object": "realtime.item", "type": "message", "role": "assistant"}
            await send(
                {"type": "response.output_item.added", "response_id": response_id, "output_index": 0, "item": item}
            )
            status = "completed"
            for i, word in enumerate(reply.split()):
                if cancel.is_set():
                    status = "cancelled"
                    break
                base = {"response_id": response_id, "item_id": "item_msg", "output_index": 0}
                delta = word if i == 0 else " " + word
                audio = "audio" in state.get("modalities", ["audio"])
                if self.openai and audio:
                    await send({"type": "response.output_audio_transcript.delta", **base, "content_index": 0,
                                "delta": delta})  # fmt: skip
                else:
                    await send({"type": "response.output_text.delta", **base, "content_index": 0, "delta": delta})
                if audio or not self.openai:
                    await send({"type": "response.output_audio.delta", **base, "content_index": 1, "delta": tone_b64()})
                await asyncio.sleep(self.word_delay_s)
            output.append({**item, "status": status})
            await send({"type": "response.output_item.done", "response_id": response_id, "output_index": 0,
                        "item": output[0]})  # fmt: skip
            await send({"type": "response.done", "response": {"id": response_id, "status": status, "output": output,
                        "usage": {"output_tokens": len(reply.split())}}})  # fmt: skip

        await send({"type": "session.created", "session": session()})
        async for raw in ws:
            event = json.loads(raw)
            log.append(event)
            kind = event.get("type")
            if kind == "session.update":
                upd = event.get("session") or {}
                voice = ((upd.get("audio") or {}).get("output") or {}).get("voice")
                if voice == "robot":
                    await send({"type": "error", "error": {"type": "invalid_request_error", "code": None,
                                "message": 'session.update: unknown voice "robot"', "param": None}})  # fmt: skip
                    continue
                rate = int(((upd.get("audio") or {}).get("input") or {}).get("format", {}).get("rate", rate))
                state.update(
                    tools=upd.get("tools", state["tools"]), instructions=upd.get("instructions", ""), voice=voice
                )
                await send({"type": "session.updated", "session": session()})
            elif kind == "input_audio_buffer.append":
                state["audio_s"] += len(base64.b64decode(event["audio"])) / 2 / rate
                state["turn"] = True
            elif kind == "input_audio_buffer.commit":
                await send({"type": "input_audio_buffer.committed", "item_id": "item_user", "previous_item_id": None})
            elif kind == "conversation.item.create":
                item = event["item"]
                if item.get("type") == "function_call_output":
                    state["tool_output"] = item["output"]
                else:
                    state["text"] = "".join(p.get("text", "") for p in item.get("content", []))
                    state["turn"] = True
                await send({"type": "conversation.item.added", "item": item, "previous_item_id": None})
            elif kind == "response.create":
                if not state["turn"] and state["tool_output"] is None:
                    await send({"type": "error", "error": {"type": "invalid_request_error", "code": "conversation_empty",
                                "message": "response.create: commit audio or create a message first", "param": None}})  # fmt: skip
                    continue
                cancel.clear()
                state["modalities"] = (event.get("response") or {}).get("output_modalities") or ["audio"]
                task = asyncio.create_task(respond(f"resp_{len(log)}"))
            elif kind == "response.cancel":
                self.cancelled += 1
                cancel.set()
        if task is not None:
            task.cancel()


def main(argv: list[str]) -> None:
    args = dict(zip(argv[::2], argv[1::2]))
    server = FakeRealtimeServer(api_key=args.get("--api-key"), port=int(args["--port"]))
    print(f"fake realtime server on {server.port} ({' '.join(argv)})", flush=True)
    asyncio.run(server._main())


if __name__ == "__main__":
    main(sys.argv[1:])
