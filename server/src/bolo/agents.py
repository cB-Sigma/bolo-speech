"""Pieces shared by the voice-agent front ends (Realtime API, chat completions with audio).

``speak_stream`` turns a stream of text deltas from a language model into
interleaved text and audio: sentences are cut off the token stream as they
complete and synthesised while the model is still writing, so the first
audio is ready long before the reply is finished.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from bolo.obs.trace import Trace
from bolo.text.segment import IncrementalSegmenter

if TYPE_CHECKING:
    from bolo.core import Bolo


async def speak_stream(
    core: Bolo,
    deltas: AsyncIterator[str],
    *,
    model: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    transport: str = "agent",
    phonemizer: str | None = None,
    romanised: str | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ``("text", delta)``, ``("sentence", text)`` and ``("audio", SynthChunk)`` items as they become available.

    ``sentence`` precedes the audio of that sentence. Closing the generator
    stops both the reading of the model and any synthesis in progress.
    """
    out: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    sentences: asyncio.Queue[str | None] = asyncio.Queue()

    async def read() -> None:
        segmenter = IncrementalSegmenter(first_max_chars=80)
        try:
            async for delta in deltas:
                if not delta:
                    continue
                await out.put(("text", delta))
                for sentence in segmenter.push(delta):
                    await sentences.put(sentence)
            for sentence in segmenter.flush():
                await sentences.put(sentence)
        finally:
            await sentences.put(None)  # also when the model fails: the speaker must not wait forever

    async def speak() -> None:
        while True:
            sentence = await sentences.get()
            if sentence is None:
                break
            if not any(ch.isalnum() for ch in sentence):
                continue  # nothing pronounceable ("...", emoji)
            await out.put(("sentence", sentence))
            trace = Trace("tts", transport, "")
            gen = core.synthesize(
                sentence, model=model, voice=voice, speed=speed, trace=trace, phonemizer=phonemizer,
                romanised=romanised, romanised_strict=False,
            )  # fmt: skip
            try:
                async for chunk in gen:
                    trace.mark_ttfa()
                    await out.put(("audio", chunk))
            finally:
                with contextlib.suppress(Exception):
                    await gen.aclose()
                core.complete_tts(trace)

    async def run() -> None:
        try:
            reader = asyncio.ensure_future(read())
            try:
                await speak()
                await reader
            finally:
                reader.cancel()
            await out.put(("end", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await out.put(("error", exc))

    task = asyncio.ensure_future(run())
    try:
        while True:
            kind, value = await out.get()
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield kind, value
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
