#!/usr/bin/env python3
"""Telephony audio end to end: 8 kHz G.711 μ-law, 20 ms (160-byte) frames.

This is the shape of audio a SIP/RTP gateway or a media-stream webhook
(Twilio, Plivo, Exotel, ...) gives you. The script

  1. asks bolo to speak a prompt as μ-law (what you would play to the caller),
  2. streams that μ-law back in RTP-sized frames at real-time pace, as if it
     were the caller's voice, and prints the transcript,

so it needs no phone line. It has NOT been run against a real telephony
network; jitter, packet loss and DTMF are yours to handle.

    pip install websockets httpx
    python examples/telephony_mulaw.py
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import websockets

HTTP = os.environ.get("BOLO_URL", "http://127.0.0.1:8000")
WS = HTTP.replace("http", "ws", 1)
FRAME = 160  # 20 ms of 8 kHz μ-law
MULAW_SILENCE = b"\xff"


async def main() -> None:
    prompt = "Thank you for calling. Your order will be delivered tomorrow."
    response = httpx.post(
        f"{HTTP}/v1/audio/speech",
        json={"input": prompt, "response_format": "mulaw"},  # raw G.711 μ-law at 8000 Hz
        timeout=120,
    )
    response.raise_for_status()
    ulaw = response.content
    print(f"synthesised {len(ulaw) / 8000:.1f}s of μ-law audio ({len(ulaw) // FRAME} RTP frames)")

    url = f"{WS}/api/v1/stt/stream?model=realtime&encoding=mulaw&sample_rate=8000"
    async with websockets.connect(url) as ws:
        ready = json.loads(await ws.recv())
        print(f"transcribing with {ready['model']}")

        async def caller() -> None:
            payload = ulaw + MULAW_SILENCE * 8000  # a second of line silence ends the utterance
            for off in range(0, len(payload), FRAME):
                frame = payload[off : off + FRAME]
                await ws.send(frame + MULAW_SILENCE * (FRAME - len(frame)))
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(caller())
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "partial":
                print(f"  partial ({msg['latency_ms']:.0f} ms): {msg['text']}")
            elif msg["type"] == "final":
                print(f"  FINAL   ({msg['latency_ms']:.0f} ms): {msg['text']}")
        await task


if __name__ == "__main__":
    asyncio.run(main())
