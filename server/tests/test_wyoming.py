from __future__ import annotations

import socket

import numpy as np
import pytest

from bolo.core import Bolo
from conftest import speechlike

wyoming = pytest.importorskip("wyoming")

from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStop  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.client import AsyncTcpClient  # noqa: E402
from wyoming.error import Error  # noqa: E402
from wyoming.info import Describe, Info  # noqa: E402
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice  # noqa: E402

from bolo.wyoming_server import start_wyoming  # noqa: E402


@pytest.fixture
async def wyoming_port(core: Bolo):  # type: ignore[no-untyped-def]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = await start_wyoming(core, "127.0.0.1", port)
    try:
        yield port
    finally:
        await server.stop()


async def read_until(client: AsyncTcpClient, *types: type) -> list:  # type: ignore[type-arg]
    events = []
    while True:
        event = await client.read_event()
        assert event is not None, "server closed the connection"
        events.append(event)
        if Error.is_type(event.type):
            return events
        if any(t.is_type(event.type) for t in types):  # type: ignore[attr-defined]
            return events


async def test_describe_lists_models_and_voices(wyoming_port: int) -> None:
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event((await read_until(client, Info))[-1])
    assert info.asr[0].name == "bolo" and {m.name for m in info.asr[0].models} == {"fake-stt", "fake-stt-streaming"}
    assert info.asr[0].models[0].installed and info.asr[0].models[0].languages
    voices = info.tts[0].voices
    assert {v.name for v in voices} == {"fake-tts:en-US", "fake-tts:en-GB", "fake-tts:hi-IN"}
    hindi = next(v for v in voices if v.name == "fake-tts:hi-IN")
    assert hindi.languages == ["hi-IN"] and [s.name for s in hindi.speakers] == ["gamma"]
    assert info.tts[0].supports_synthesize_streaming is True
    assert info.asr[0].supports_transcript_streaming is True


async def test_transcribe_round_trip(wyoming_port: int, core: Bolo) -> None:
    pcm = (speechlike(2.0) * 32767).astype("<i2").tobytes()
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(Transcribe(name="fake-stt", language="en").event())
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        for off in range(0, len(pcm), 4096):
            await client.write_event(AudioChunk(audio=pcm[off : off + 4096], rate=16000, width=2, channels=1).event())
        await client.write_event(AudioStop().event())
        events = await read_until(client, TranscriptStop)
    types = [e.type for e in events]
    # Home Assistant's order for transcript streaming: start, chunk(s), the compatible transcript, stop
    assert types[0] == "transcript-start" and types[-2:] == ["transcript", "transcript-stop"]
    chunks = [TranscriptChunk.from_event(e).text for e in events if TranscriptChunk.is_type(e.type)]
    transcript = Transcript.from_event(events[-2])
    assert chunks and chunks[-1] == transcript.text
    assert transcript.text == "the quick brown fox jumps" and transcript.language == "en"
    trace = core.traces.recent()[0]
    assert trace.transport == "wyoming" and trace.kind == "stt" and trace.audio_seconds == 2.0


async def test_synthesize_round_trip(wyoming_port: int, core: Bolo) -> None:
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        voice = SynthesizeVoice(name="fake-tts:en-GB", speaker="beta")
        await client.write_event(Synthesize(text="Hello there. How are you?", voice=voice).event())
        events = await read_until(client, AudioStop)
    assert AudioStart.is_type(events[0].type)
    start = AudioStart.from_event(events[0])
    assert (start.rate, start.width, start.channels) == (24000, 2, 1)
    audio = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    assert len(audio) == pytest.approx(len("Hello there. How are you?") * 0.06 * 24000 * 2, rel=0.05)
    assert np.abs(np.frombuffer(audio, dtype="<i2")).max() > 1000
    trace = core.traces.recent(kind="tts")[0]
    assert trace.transport == "wyoming" and trace.ttfa_ms is not None


async def test_streaming_synthesis(wyoming_port: int) -> None:
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(SynthesizeStart(voice=SynthesizeVoice(name="fake-tts")).event())
        for piece in ["Hello the", "re. And mo", "re text"]:
            await client.write_event(SynthesizeChunk(text=piece).event())
        await client.write_event(Synthesize(text="Hello there. And more text").event())  # compatibility duplicate
        await client.write_event(SynthesizeStop().event())
        events = await read_until(client, SynthesizeStopped)
    types = [e.type for e in events]
    assert types[0] == "audio-start" and types.count("audio-start") == 1
    assert types[-2:] == ["audio-stop", "synthesize-stopped"]
    audio = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    assert len(audio) == pytest.approx(len("Hello there.And more text") * 0.06 * 24000 * 2, rel=0.08)


async def test_errors_are_reported_without_dropping_the_connection(wyoming_port: int) -> None:
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(
            Synthesize(text="Hello.", voice=SynthesizeVoice(name="fake-tts", speaker="nobody")).event()
        )
        error = Error.from_event((await read_until(client, AudioStop))[-1])
        assert "voice" in error.text and error.code == "invalid_request"
        await client.write_event(Describe().event())  # still usable
        assert Info.is_type((await read_until(client, Info))[-1].type)
