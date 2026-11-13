#!/usr/bin/env python3
"""bolo behind a LiveKit Agents session, in three ways.

    realtime    openai.realtime.RealtimeModel pointed at bolo's /v1/realtime: bolo does turn
                detection, speech-to-text, the language model call and text-to-speech.
    cascade     LiveKit's VAD with openai.STT, openai.LLM and openai.TTS, all pointed at bolo
                (plain HTTP; the LLM calls go through bolo to your language model).
    cascade-rt  the same, with openai.STT(use_realtime=True): a Realtime transcription session
                that delivers interim transcripts while the caller speaks.

To stay runnable without a LiveKit server, the session gets a wav file (16 kHz mono) as its
audio input and a collector as its audio output. In a real agent pass room=ctx.room to
session.start() instead.

Run with livekit-agents 1.8.11 and livekit-plugins-openai 1.8.11 against bolo with
moonshine-tiny-en, a Piper voice and a small language model behind Ollama: all three modes
worked without changes to bolo (docs/voice-agents.md has what was observed). "realtime" and
"cascade" need a language model configured in bolo (--llm-base-url).

    pip install livekit-agents livekit-plugins-openai livekit-plugins-silero
    bolo serve --llm-base-url http://127.0.0.1:11434/v1 --llm-model <model>
    python livekit_agent.py realtime question.wav reply.wav
"""

import asyncio
import json
import os
import sys
import time
import wave

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.voice import io
from livekit.plugins import openai

BOLO = os.environ.get("BOLO_URL", "http://127.0.0.1:8000") + "/v1"
MODE, WAV = sys.argv[1], sys.argv[2]
T0 = time.monotonic()
log = {"mode": MODE, "user": [], "agent": [], "errors": [], "states": []}


class WavInput(io.AudioInput):
    def __init__(self, path, tail_s=40.0):
        super().__init__(label="wav")
        with wave.open(path) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1
            self.pcm = w.readframes(w.getnframes())
            self.speech_s = w.getnframes() / 16000
        self.pcm = b"\0" * 32000 + self.pcm + b"\0" * int(32000 * tail_s)
        self.pos = 0
        self.start = None

    async def __anext__(self):
        if self.start is None:
            self.start = time.monotonic()
            log["speech_end_t"] = round(self.start - T0 + 1.0 + self.speech_s, 3)
        step = 320 * 2
        if self.pos >= len(self.pcm):
            await asyncio.sleep(3600)
        chunk = self.pcm[self.pos : self.pos + step].ljust(step, b"\0")
        self.pos += step
        wait = self.start + self.pos / 2 / 16000 - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        return rtc.AudioFrame(data=chunk, sample_rate=16000, num_channels=1, samples_per_channel=320)


class Collect(io.AudioOutput):
    def __init__(self):
        super().__init__(
            label="collect",
            capabilities=io.AudioOutputCapabilities(pause=False),
            sample_rate=24000,
        )
        self.buf = bytearray()
        self.seg = 0.0

    async def capture_frame(self, frame):
        await super().capture_frame(frame)
        if not self.buf:
            log["first_audio_t"] = round(time.monotonic() - T0, 3)
        self.buf += bytes(frame.data)
        self.seg += frame.duration

    def flush(self):
        super().flush()
        pos, self.seg = self.seg, 0.0
        self.on_playback_finished(playback_position=pos, interrupted=False)

    def clear_buffer(self):
        pass


async def main():
    if MODE == "realtime":
        from openai.types.realtime import AudioTranscription

        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                base_url=BOLO,
                api_key="not-needed",
                model="gpt-realtime",
                voice="alloy",
                input_audio_transcription=AudioTranscription(model="moonshine-tiny-en"),
            )
        )
    else:
        from livekit.plugins import silero

        stt = openai.STT(
            base_url=BOLO,
            api_key="not-needed",
            model="moonshine-tiny-en",
            language="en",
            **({"use_realtime": True} if MODE == "cascade-rt" else {}),
        )
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=stt,
            llm=openai.LLM(
                base_url=BOLO,
                api_key="not-needed",
                model=os.environ.get("LLM_MODEL", "qwen2.5:0.5b"),
            ),
            tts=openai.TTS(
                base_url=BOLO,
                api_key="not-needed",
                model=os.environ.get("BOLO_TTS", "piper-en-us-amy-low"),
                voice="alloy",
            ),
        )
    out = Collect()
    session.input.audio = WavInput(WAV)
    session.output.audio = out
    def t() -> float:
        return round(time.monotonic() - T0, 3)

    session.on(
        "user_input_transcribed",
        lambda ev: log["user"].append([t(), ev.is_final, ev.transcript]),
    )
    session.on(
        "conversation_item_added",
        lambda ev: (
            log["agent"].append([t(), ev.item.role, ev.item.text_content])
            if getattr(ev.item, "role", "") == "assistant"
            else None
        ),
    )
    session.on("error", lambda ev: log["errors"].append(str(ev.error)[:300]))
    session.on("agent_state_changed", lambda ev: log["states"].append([t(), ev.new_state]))
    await session.start(agent=Agent(instructions="You are a helpful voice assistant. Answer in one short sentence."))
    deadline = time.monotonic() + float(os.environ.get("RUN_S", "50"))
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        if log["agent"] and time.monotonic() - T0 > log["agent"][-1][0] + 4:
            break
    await session.aclose()
    log["agent_audio_s"] = round(len(out.buf) / 2 / 24000, 2)
    if out.buf and len(sys.argv) > 3:
        with wave.open(sys.argv[3], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(bytes(out.buf))
    if "first_audio_t" in log:
        log["first_audio_after_speech_end_ms"] = round((log["first_audio_t"] - log["speech_end_t"]) * 1000)
    print("RESULT " + json.dumps(log, ensure_ascii=False))


async def wrapped():
    from livekit.agents.utils import http_context

    async with http_context.open():
        await main()


asyncio.run(wrapped())
