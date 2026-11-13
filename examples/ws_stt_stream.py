#!/usr/bin/env python3
"""Realtime transcription over bolo's WebSocket.

    pip install websockets soundfile numpy        # and sounddevice for --mic
    python examples/ws_stt_stream.py recording.wav
    python examples/ws_stt_stream.py --mic        # untested by the maintainers (no audio hardware)

The file is sent at real-time speed in 20 ms frames, as a microphone would.
Partials are printed in place; the stable part is shown before the "|".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import numpy as np
import websockets

URL = os.environ.get("BOLO_URL", "http://127.0.0.1:8000").replace("http", "ws", 1)
KEY = os.environ.get("BOLO_API_KEY")
RATE = 16000
FRAME = RATE // 50  # 20 ms


async def file_frames(path: str):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:  # crude but dependency-free; the server resamples properly if you send the native rate
        audio = np.interp(np.arange(0, len(audio), rate / RATE), np.arange(len(audio)), audio)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    pcm = np.concatenate([pcm, np.zeros(RATE, dtype="<i2")])  # trailing silence lets the endpoint fire
    for off in range(0, len(pcm), FRAME):
        yield pcm[off : off + FRAME].tobytes()
        await asyncio.sleep(0.02)


async def mic_frames():
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    with sd.RawInputStream(
        samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME,
        callback=lambda data, *_: loop.call_soon_threadsafe(queue.put_nowait, bytes(data)),
    ):
        while True:
            yield await queue.get()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    parser.add_argument("--mic", action="store_true")
    parser.add_argument("--model", default="realtime", help="'realtime' is the server's low-latency default")
    parser.add_argument("--language", default="")
    args = parser.parse_args()
    if not args.file and not args.mic:
        parser.error("give a file or --mic")
    query = f"model={args.model}&sample_rate={RATE}&encoding=pcm_s16le&interim=true&vad=true"
    if args.language:
        query += f"&language={args.language}"
    if KEY:
        query += f"&api_key={KEY}"
    async with websockets.connect(f"{URL}/api/v1/stt/stream?{query}") as ws:
        ready = json.loads(await ws.recv())
        if ready["type"] == "error":
            raise SystemExit(ready["message"])
        print(f"model {ready['model']} ({ready['mode']} streaming), request {ready['request_id']}")

        async def sender() -> None:
            frames = mic_frames() if args.mic else file_frames(args.file)
            async for frame in frames:
                await ws.send(frame)
            await ws.send(json.dumps({"type": "stop"}))

        send_task = asyncio.create_task(sender())
        async for message in ws:
            msg = json.loads(message)
            if msg["type"] == "partial":
                stable, rest = msg["text"][: msg["stable_chars"]], msg["text"][msg["stable_chars"] :]
                print(f"\r  ... {stable}|{rest}   ({msg['latency_ms']:.0f} ms)", end="", flush=True)
            elif msg["type"] == "final":
                print(f"\r[{msg['t0']:6.2f}-{msg['t1']:6.2f}] {msg['text']}   ({msg['latency_ms']:.0f} ms)" + " " * 20)
            elif msg["type"] == "metrics":
                last_metrics = msg
            elif msg["type"] == "error":
                print(f"\nerror: {msg['message']}")
            elif msg["type"] == "done":
                print(f"audio {last_metrics['audio_seconds']:.1f}s, real-time factor {last_metrics['rtf']}")
        await send_task


if __name__ == "__main__":
    asyncio.run(main())
