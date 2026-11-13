#!/usr/bin/env python3
"""Streaming synthesis over bolo's WebSocket, with text arriving in pieces
(as it would from a language model).

    pip install websockets
    python examples/ws_tts_stream.py "First sentence. Second sentence, a bit longer." -o out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import wave

import websockets

URL = os.environ.get("BOLO_URL", "http://127.0.0.1:8000").replace("http", "ws", 1)
KEY = os.environ.get("BOLO_API_KEY")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("-o", "--output", default="out.wav")
    parser.add_argument("--model", default="")
    parser.add_argument("--voice", default="")
    args = parser.parse_args()
    query = "format=pcm_s16le"
    for key in ("model", "voice"):
        if getattr(args, key):
            query += f"&{key}={getattr(args, key)}"
    if KEY:
        query += f"&api_key={KEY}"
    async with websockets.connect(f"{URL}/api/v1/tts/stream?{query}") as ws:
        ready = json.loads(await ws.recv())
        if ready["type"] == "error":
            raise SystemExit(ready["message"])
        print(f"model {ready['model']}, {ready['sample_rate']} Hz")

        async def feed() -> None:
            words = args.text.split(" ")
            for i in range(0, len(words), 3):  # three words at a time, like tokens from an LLM
                await ws.send(json.dumps({"type": "text", "text": " ".join(words[i : i + 3]) + " "}))
                await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "stop"}))

        feeder = asyncio.create_task(feed())
        pcm = bytearray()
        async for message in ws:
            if isinstance(message, bytes):
                pcm.extend(message)  # play this immediately in a real application
                continue
            msg = json.loads(message)
            if msg["type"] == "chunk" and msg["ttfa_ms"] is not None:
                print(f"  first audio for {msg['text']!r} after {msg['ttfa_ms']:.0f} ms")
            elif msg["type"] == "metrics":
                print(f"  {msg['audio_seconds']:.1f}s of audio, real-time factor {msg['rtf']}")
            elif msg["type"] == "error":
                raise SystemExit(msg["message"])
        await feeder
    with wave.open(args.output, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(ready["sample_rate"])
        out.writeframes(bytes(pcm))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
