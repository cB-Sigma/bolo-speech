#!/usr/bin/env python3
"""bolo as speech-to-text and text-to-speech for a LiveKit agent.

    !!  UNTESTED  !!
This file has not been run by the bolo maintainers. LiveKit's OpenAI plugin
accepts a custom `base_url`, and bolo implements the endpoints it calls
(/v1/audio/transcriptions and /v1/audio/speech), so the plugin can be pointed
at bolo unchanged. Names follow the LiveKit Agents documentation at the time
of writing; check them against the version you install.

    pip install "livekit-agents[openai,silero]"
    bolo serve
"""

from __future__ import annotations

import os

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

BOLO = os.environ.get("BOLO_URL", "http://127.0.0.1:8000") + "/v1"


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    session = AgentSession(
        vad=silero.VAD.load(),  # LiveKit's VAD decides where an utterance ends
        stt=openai.STT(base_url=BOLO, api_key="not-needed", model="moonshine-tiny-en", language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(base_url=BOLO, api_key="not-needed", model="kokoro-v1.0-int8", voice="af_heart"),
    )
    await session.start(room=ctx.room, agent=Agent(instructions="You are a concise, friendly voice assistant."))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
