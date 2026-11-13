#!/usr/bin/env python3
"""bolo through the official OpenAI Python SDK.

    pip install openai
    bolo serve            # in another terminal
    python examples/openai_sdk.py recording.wav
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BOLO_URL", "http://127.0.0.1:8000") + "/v1",
    api_key=os.environ.get("BOLO_API_KEY", "not-needed"),
)


def main() -> None:
    # 1. Text to speech. "tts-1" and OpenAI voice names work; so do bolo's own ids.
    speech = Path("hello.wav")
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input="Hello from bolo. This sentence was synthesised on your own machine.",
        response_format="wav",
    ) as response:
        response.stream_to_file(speech)
    print(f"wrote {speech}")

    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else speech

    # 2. Plain transcription. "whisper-1" maps to the server's default STT model.
    with audio.open("rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    print("text:", result.text)

    # 3. Word timestamps (the model must support them: faster-whisper, Parakeet, Zipformer, ...).
    with audio.open("rb") as f:
        verbose = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json", timestamp_granularities=["word", "segment"]
        )
    print(f"language={verbose.language} duration={verbose.duration:.1f}s segments={len(verbose.segments or [])}")
    for word in (verbose.words or [])[:5]:
        print(f"  {word.start:6.2f} {word.end:6.2f} {word.word}")

    # 4. Streaming: text arrives segment by segment while a long file is decoded.
    with audio.open("rb") as f:
        for event in client.audio.transcriptions.create(model="whisper-1", file=f, stream=True):
            if event.type == "transcript.text.delta":
                print(event.delta, end="", flush=True)
    print()

    # 5. A specific model, a language hint, and subtitles.
    with audio.open("rb") as f:
        srt = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="srt", language="en")
    print(srt)


if __name__ == "__main__":
    main()
