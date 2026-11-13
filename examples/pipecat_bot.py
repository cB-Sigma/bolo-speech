#!/usr/bin/env python3
"""bolo as the speech layer of a Pipecat voice bot, in three ways.

    realtime      Pipecat's OpenAIRealtimeLLMService talks to bolo's /v1/realtime: bolo does
                  turn detection, speech-to-text, the language model call and text-to-speech.
    realtime-stt  Pipecat's OpenAIRealtimeSTTService uses a Realtime transcription session
                  (interim and final transcripts); the rest of the pipeline is yours.
    cascade       Pipecat's own VAD and turn detection, with bolo behind its stock
                  OpenAI-compatible STT, LLM and TTS services (plain HTTP).

To stay runnable without a room or a telephone line, the "caller" is a 16 kHz mono wav file fed
in at real-time pace and the bot's speech is written to another wav file. In a real bot, replace
Feeder and Sink by transport.input() and transport.output().

Run with Pipecat 1.16.0 against bolo with moonshine-tiny-en, a Piper or Kokoro voice and a small
language model behind Ollama (docs/voice-agents.md has what was observed). Notes from that run:

* Pipecat's OpenAI TTS service accepts OpenAI voice names only (alloy, ash, ...); bolo maps them
  onto the voices of whichever model is configured.
* The same service gives up on a sentence when no audio arrives for three seconds
  (stop_frame_timeout_s) and logs "completed with no audio"; the audio is still played when it
  arrives. On a slow machine prefer a realtime-tier voice (Piper) to Kokoro.
* "realtime" and "cascade" need a language model configured in bolo (--llm-base-url).

    pip install "pipecat-ai[openai,silero]"
    bolo serve --llm-base-url http://127.0.0.1:11434/v1 --llm-model <model>
    python pipecat_bot.py realtime question.wav reply.wav
"""

import asyncio
import json
import os
import sys
import time
import wave

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameProcessor

BOLO = os.environ.get("BOLO_URL", "http://127.0.0.1:8000")
MODE, WAV = sys.argv[1], sys.argv[2]
T0 = time.monotonic()
log = {
    "mode": MODE,
    "events": [],
    "user": [],
    "bot_text": "",
    "bot_audio_s": 0.0,
    "errors": [],
}


class Feeder(FrameProcessor):
    """Plays a 16 kHz mono wav into the pipeline at real-time pace, then silence."""

    def __init__(self, path, tail_s=25.0):
        super().__init__()
        self.path, self.tail_s, self._task = path, tail_s, None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._task is None:
            self._task = asyncio.create_task(self._feed())

    async def _feed(self):
        await asyncio.sleep(1.5)
        with wave.open(self.path) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1
            pcm = w.readframes(w.getnframes())
        pcm += b"\0" * int(16000 * 2 * self.tail_s)
        step = 320 * 2
        start = time.monotonic()
        speech_end = start + w.getnframes() / 16000
        log["speech_end_t"] = round(speech_end - T0, 3)
        for i in range(0, len(pcm), step):
            await self.push_frame(InputAudioRawFrame(audio=pcm[i : i + step], sample_rate=16000, num_channels=1))
            wait = start + (i + step) / 2 / 16000 - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            if log.get("done_t") and time.monotonic() - T0 > log["done_t"] + 2.0:
                break


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.audio = bytearray()
        self.rate = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        t = round(time.monotonic() - T0, 3)
        if isinstance(frame, OutputAudioRawFrame):
            if not self.audio:
                log["first_audio_t"] = t
            self.audio += frame.audio
            self.rate = frame.sample_rate
            log["last_audio_t"] = t
        elif isinstance(frame, TranscriptionFrame):
            log["user"].append(frame.text)
            log.setdefault("user_final_t", t)
        elif isinstance(frame, InterimTranscriptionFrame):
            log.setdefault("first_interim_t", t)
            log["interims"] = log.get("interims", 0) + 1
        elif isinstance(frame, (TTSTextFrame,)):
            log["bot_text"] += frame.text + " "
        elif isinstance(frame, LLMTextFrame):
            log.setdefault("first_llm_text_t", t)
            log["llm_text"] = log.get("llm_text", "") + frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            log["done_t"] = t
        elif isinstance(frame, ErrorFrame):
            log["errors"].append(str(frame.error))
        await self.push_frame(frame, direction)


async def main():
    sink = Sink()
    feeder = Feeder(WAV)
    context = LLMContext(
        [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Answer in one short sentence.",
            }
        ]
    )
    if MODE == "realtime":
        from pipecat.services.openai.realtime import events
        from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

        llm = OpenAIRealtimeLLMService(
            api_key="not-needed",
            base_url=BOLO.replace("http", "ws") + "/v1/realtime",
            settings=OpenAIRealtimeLLMService.Settings(
                model="gpt-realtime",
                system_instruction="You are a helpful voice assistant. Answer in one short sentence.",
                session_properties=events.SessionProperties(
                    audio=events.AudioConfiguration(
                        input=events.AudioInput(transcription=events.InputAudioTranscription(model="moonshine-tiny-en"))
                    )
                ),
            ),
        )
        pair = LLMContextAggregatorPair(context)
        stages = [feeder, pair.user(), Sink(), llm, sink, pair.assistant()]
    elif MODE == "realtime-stt":
        from pipecat.services.openai.stt import OpenAIRealtimeSTTService

        stt = OpenAIRealtimeSTTService(
            api_key="not-needed",
            base_url=BOLO.replace("http", "ws") + "/v1/realtime",
            settings=OpenAIRealtimeSTTService.Settings(model="moonshine-tiny-en"),
            turn_detection={"type": "server_vad"},
        )
        stages = [feeder, Sink(), stt, sink]
    else:  # cascade: Pipecat's own VAD, bolo for STT and TTS, the language model through bolo's proxy
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.stt import OpenAISTTService
        from pipecat.services.openai.tts import OpenAITTSService

        stt = OpenAISTTService(
            base_url=BOLO + "/v1",
            api_key="not-needed",
            settings=OpenAISTTService.Settings(model="moonshine-tiny-en"),
        )
        tts = OpenAITTSService(
            base_url=BOLO + "/v1",
            api_key="not-needed",
            settings=OpenAITTSService.Settings(
                model=os.environ.get("BOLO_TTS", "piper-en-us-amy-low"),
                voice=os.environ.get("BOLO_VOICE", "alloy"),
            ),
            sample_rate=24000,
        )
        llm = OpenAILLMService(
            base_url=BOLO + "/v1",
            api_key="not-needed",
            settings=OpenAILLMService.Settings(model=os.environ.get("LLM_MODEL", "qwen2.5:0.5b")),
        )
        pair = LLMContextAggregatorPair(context)
        stages = [
            feeder,
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            pair.user(),
            Sink(),
            llm,
            tts,
            sink,
            pair.assistant(),
        ]
    task = PipelineTask(
        Pipeline(stages),
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
    )
    runner = PipelineRunner(handle_sigint=False)

    async def stop():
        deadline = time.monotonic() + float(os.environ.get("RUN_S", "45"))
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if MODE != "realtime-stt" and log.get("done_t") and time.monotonic() - T0 > log["done_t"] + 6:
                break
            if MODE == "realtime-stt" and log["user"] and time.monotonic() - T0 > log["user_final_t"] + 3:
                break
        await task.cancel()

    stopper = asyncio.create_task(stop())
    await runner.run(task)
    stopper.cancel()
    log["bot_audio_s"] = round(len(sink.audio) / 2 / (sink.rate or 24000), 2)
    if sink.audio and len(sys.argv) > 3:
        with wave.open(sys.argv[3], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sink.rate)
            w.writeframes(bytes(sink.audio))
    for k in ("first_interim_t", "user_final_t", "first_llm_text_t", "first_audio_t"):
        if k in log and "speech_end_t" in log:
            log[k[: -len("_t")] + "_after_speech_end_ms"] = round((log[k] - log["speech_end_t"]) * 1000)
    print("RESULT " + json.dumps(log, ensure_ascii=False))


asyncio.run(main())
