#!/usr/bin/env python3
"""bolo as the speech layer of a Pipecat voice bot.

    !!  UNTESTED  !!
This file has not been run by the bolo maintainers. It shows the intended
wiring: Pipecat already ships OpenAI-compatible STT and TTS services, and
bolo speaks that protocol, so no bolo-specific plugin is needed. Class names
and parameters follow Pipecat's documentation at the time of writing and may
have moved; check them against the Pipecat version you install.

    pip install "pipecat-ai[openai,silero]"
    bolo serve
"""

from __future__ import annotations

import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService

BOLO = os.environ.get("BOLO_URL", "http://127.0.0.1:8000") + "/v1"

# Pipecat segments speech with its own VAD and posts each utterance to
# /v1/audio/transcriptions: a realtime-tier model keeps that round trip short.
stt = OpenAISTTService(base_url=BOLO, api_key="not-needed", model="moonshine-tiny-en")

# /v1/audio/speech streams raw PCM as it is synthesised.
tts = OpenAITTSService(base_url=BOLO, api_key="not-needed", model="kokoro-v1.0-int8", voice="af_heart", sample_rate=24000)

llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")


def build_pipeline(transport, context_aggregator) -> Pipeline:
    """`transport` is any Pipecat transport (Daily, a WebSocket server, a telephony serializer, ...)."""
    return Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
