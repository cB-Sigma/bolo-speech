"""Pieces shared by the streaming speech-to-speech models: token streams out of ``generate`` running in a thread,
cancellation inside generation, text decoded as it grows, tool calls in ``<tool_call>`` blocks, temporary WAV files
for code that reads audio from a path."""

from __future__ import annotations

import contextlib
import json
import queue
import re
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

_END = object()


class TokenQueue:
    """What ``generate(streamer=...)`` feeds: each new token id, in order (the prompt is skipped).

    The same shape as transformers' ``BaseStreamer`` (``put`` and ``end``), which is all ``generate`` calls."""

    def __init__(self, skip_prompt: bool = True) -> None:
        self.skip_prompt = skip_prompt
        self._prompt_seen = False
        self._q: queue.Queue[Any] = queue.Queue()

    def put(self, value: Any) -> None:
        if self.skip_prompt and not self._prompt_seen:
            self._prompt_seen = True
            return
        ids = value.tolist() if hasattr(value, "tolist") else list(value)
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            if len(ids) > 1:
                raise ValueError("only one conversation at a time")
            ids = ids[0]
        for token in ids if isinstance(ids, list) else [ids]:
            self._q.put(int(token))

    def end(self) -> None:
        self._q.put(_END)

    def fail(self, exc: BaseException) -> None:
        self._q.put(exc)

    def __iter__(self) -> Iterator[int]:
        while True:
            item = self._q.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield int(item)


class _Halt:
    """Ends ``generate`` once ``halted()`` is true (a barge-in, or the reader went away)."""

    def __init__(self, halted: Callable[[], bool]) -> None:
        self.halted = halted

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        stop = bool(self.halted())
        try:
            import torch

            return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)
        except (ImportError, AttributeError, TypeError):  # fakes without torch tensors
            return stop


def stopping_criteria(halted: Callable[[], bool]) -> Any:
    """``generate``'s ``stopping_criteria``: a ``StoppingCriteriaList`` holding :class:`_Halt` (a plain list when
    transformers is not installed, as in the tests)."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return [_Halt(halted)]

    class Halt(_Halt, StoppingCriteria):  # type: ignore[misc]
        pass

    return StoppingCriteriaList([Halt(halted)])


def generate_in_thread(generate: Callable[..., Any], kwargs: dict[str, Any], streamer: TokenQueue) -> threading.Thread:
    """Run ``generate(**kwargs, streamer=streamer)``; an error ends the stream with that error."""

    def run() -> None:
        try:
            generate(**kwargs, streamer=streamer)
        except BaseException as exc:
            streamer.fail(exc)
        else:
            streamer.end()

    thread = threading.Thread(target=run, name="bolo-torch-generate", daemon=True)
    thread.start()
    return thread


class GrowingText:
    """Text of a growing list of token ids, handed out as new pieces (a piece is held back while it ends in a
    partial UTF-8 character)."""

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self.decode = decode
        self.ids: list[int] = []
        self.sent = ""

    def add(self, token: int) -> str:
        self.ids.append(token)
        return self.flush(final=False)

    def flush(self, final: bool = True) -> str:
        text = self.decode(self.ids)
        if not final and text.endswith("�"):
            return ""
        # when the tokenizer re-segments what was already sent, only the growth goes out (nothing is sent twice)
        new = text[len(self.sent) :]
        if len(text) >= len(self.sent):
            self.sent = text
        return new


TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def parse_tool_call(body: str, number: int) -> dict[str, str] | None:
    """A ``<tool_call>`` body as ``{"call_id", "name", "arguments"}``: JSON ``{"name", "arguments"}``, or the
    ``function`` / name / arguments lines that Step-Audio 2 writes."""
    body = body.strip()
    name = args = None
    with contextlib.suppress(ValueError):
        data = json.loads(body)
        if isinstance(data, dict) and data.get("name"):
            name = str(data["name"])
            raw = data.get("arguments", data.get("parameters", {}))
            args = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    if name is None:
        lines = [x.strip() for x in body.splitlines() if x.strip()]
        if lines and lines[0] == "function":
            lines = lines[1:]
        if not lines:
            return None
        name = lines[0]
        args = "\n".join(lines[1:]) or "{}"
    return {"call_id": f"call_{number}", "name": name, "arguments": args or "{}"}


class ToolCallFilter:
    """Splits the model's text into what is said and the tool calls in ``<tool_call>...</tool_call>`` blocks."""

    def __init__(self) -> None:
        self.buffer = ""
        self.calls = 0

    def feed(self, text: str, final: bool = False) -> tuple[str, list[dict[str, str]]]:
        self.buffer += text
        calls: list[dict[str, str]] = []
        out = ""
        while True:
            start = self.buffer.find("<tool_call>")
            if start < 0:
                # keep a possible beginning of "<tool_call>" for the next piece
                keep = 0 if final else _partial_tag(self.buffer)
                out += self.buffer[: len(self.buffer) - keep]
                self.buffer = self.buffer[len(self.buffer) - keep :]
                break
            out += self.buffer[:start]
            end = self.buffer.find("</tool_call>", start)
            if end < 0:
                self.buffer = self.buffer[start:]
                if final:
                    self.buffer = ""
                break
            call = parse_tool_call(self.buffer[start + len("<tool_call>") : end], self.calls)
            if call is not None:
                calls.append(call)
                self.calls += 1
            self.buffer = self.buffer[end + len("</tool_call>") :]
        return out, calls


def _partial_tag(text: str) -> int:
    tag = "<tool_call>"
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class WavFiles:
    """Temporary 16 kHz WAV files for inference code that reads audio from a path; removed on close."""

    def __init__(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="bolo-torch-")
        self.count = 0

    def write(self, audio: Any, rate: int = 16000) -> str:
        import soundfile as sf

        self.count += 1
        path = Path(self._dir.name) / f"turn-{self.count}.wav"
        sf.write(str(path), np.asarray(audio, dtype=np.float32).reshape(-1), rate, subtype="FLOAT")
        return str(path)

    def close(self) -> None:
        self._dir.cleanup()

    def __enter__(self) -> WavFiles:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def no_grad() -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - the fakes of the tests
        return contextlib.nullcontext()
    return torch.inference_mode()


def system_text(*parts: str | None) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Realtime API function tools (``{"type": "function", "name", ...}``) in the chat-completions shape."""
    out = []
    for tool in tools:
        if tool.get("type", "function") != "function" or not tool.get("name"):
            continue
        fn = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
        out.append({"type": "function", "function": fn})
    return out
