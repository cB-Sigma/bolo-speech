#!/usr/bin/env python3
"""One turn of a voice conversation over the OpenAI Realtime API, from a file.

The server needs a language model:

    bolo serve --llm-base-url http://127.0.0.1:11434/v1 --llm-model llama3.2

Then:

    pip install openai soundfile numpy
    python examples/realtime_agent.py [question.wav] [--out reply.wav]

Without a file the question is synthesised by bolo itself. The audio is sent
at the pace of a live microphone, the server's voice-activity detection ends
the turn, and the reply is written to a WAV file together with the latency
figures bolo recorded for the turn.

How well the conversation goes depends entirely on the language model you
connected; bolo only supplies the ears and the voice.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import urllib.request

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI

URL = os.environ.get("BOLO_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("BOLO_API_KEY", "not-needed")
RATE = 24000  # the Realtime API's audio/pcm format: 24 kHz, 16-bit, mono


def http(path: str, body: dict | None = None) -> bytes:
    request = urllib.request.Request(URL + path, headers={"Authorization": f"Bearer {KEY}"})
    if body is not None:
        request.data = json.dumps(body).encode()
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def load_question(path: str | None) -> bytes:
    if path is None:
        wav = http("/v1/audio/speech", {"input": "What is the tallest mountain on Earth?", "response_format": "wav"})
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    else:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != RATE:  # linear interpolation is plenty for speech recognition
        positions = np.arange(int(len(mono) * RATE / rate)) * (rate / RATE)
        mono = np.interp(positions, np.arange(len(mono)), mono).astype(np.float32)
    mono = np.concatenate([mono, np.zeros(RATE, dtype=np.float32)])  # a second of silence ends the turn
    return (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", help="audio file with the question (default: synthesised)")
    parser.add_argument("--out", default="reply.wav")
    parser.add_argument("--voice", default=None, help="a voice of the server's text-to-speech model")
    parser.add_argument("--timeout", type=float, default=120.0, help="give up after this many seconds")
    parser.add_argument("--instructions", default="You are a helpful voice assistant. Answer in one or two sentences.")
    args = parser.parse_args()

    pcm = load_question(args.file)
    client = AsyncOpenAI(base_url=URL + "/v1", api_key=KEY)
    reply = bytearray()
    async with client.realtime.connect(model="gpt-realtime") as conn:
        session: dict = {"type": "realtime", "instructions": args.instructions}
        if args.voice:
            session["audio"] = {"output": {"voice": args.voice}}
        await conn.session.update(session=session)

        async def microphone() -> None:
            step = RATE // 10 * 2  # 100 ms of 16-bit samples
            for i in range(0, len(pcm), step):
                await conn.input_audio_buffer.append(audio=base64.b64encode(pcm[i : i + step]).decode())
                await asyncio.sleep(0.1)

        async def listen() -> None:
            async for event in conn:
                if event.type == "input_audio_buffer.speech_started":
                    print("(speech detected)")
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    print("you:      ", event.transcript)
                elif event.type == "response.output_audio.delta":
                    reply.extend(base64.b64decode(event.delta))
                elif event.type == "response.output_audio_transcript.done":
                    print("assistant:", event.transcript)
                elif event.type == "response.function_call_arguments.done":
                    print("tool call:", event.name, event.arguments)
                elif event.type == "error":
                    print("error:    ", event.error.message)
                    break
                elif event.type == "response.done":
                    print("status:   ", event.response.status)
                    break

        feeder = asyncio.create_task(microphone())
        try:
            await asyncio.wait_for(listen(), timeout=args.timeout)
        except TimeoutError:
            print(f"no complete turn within {args.timeout:.0f} s (was any speech detected?)")
        feeder.cancel()

    if reply:
        sf.write(args.out, np.frombuffer(bytes(reply), dtype="<i2"), RATE)
        print(f"wrote {args.out} ({len(reply) / 2 / RATE:.1f} s)")
    traces = json.loads(http("/api/v1/requests?kind=agent&limit=1"))["requests"]
    if traces:
        t = traces[0]
        print(
            f"latency: transcript {t.get('final_ms')} ms, first token {t.get('first_token_ms')} ms, "
            f"first audio {t.get('ttfa_ms')} ms after the end of speech"
        )


if __name__ == "__main__":
    asyncio.run(main())
