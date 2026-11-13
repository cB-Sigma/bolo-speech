#!/usr/bin/env python3
"""Audio in and audio out through /v1/chat/completions with the openai SDK.

The server needs a language model (`bolo serve --llm-base-url ... --llm-model ...`).
bolo transcribes the `input_audio` part, sends the conversation as text to
that model, and speaks the answer.

    pip install openai
    python examples/chat_audio.py question.wav [reply.wav]
"""

from __future__ import annotations

import base64
import os
import sys

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BOLO_URL", "http://127.0.0.1:8000") + "/v1",
    api_key=os.environ.get("BOLO_API_KEY", "not-needed"),
)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1], "rb") as f:
        data = base64.b64encode(f.read()).decode()
    fmt = "mp3" if sys.argv[1].lower().endswith(".mp3") else "wav"
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",  # replaced by the server's llm_model when that is set
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {"role": "system", "content": "Answer in one or two sentences."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": fmt}}]},
        ],
    )
    message = completion.choices[0].message
    if message.audio is None:
        print("assistant:", message.content)
        return
    print("assistant:", message.audio.transcript)
    out = sys.argv[2] if len(sys.argv) > 2 else "reply.wav"
    with open(out, "wb") as f:
        f.write(base64.b64decode(message.audio.data))
    print("wrote", out)


if __name__ == "__main__":
    main()
