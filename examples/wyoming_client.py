#!/usr/bin/env python3
"""Talk to bolo over the Wyoming protocol, as Home Assistant does.

    bolo serve --wyoming                 # listens on tcp://127.0.0.1:10300
    pip install wyoming
    python examples/wyoming_client.py
"""

from __future__ import annotations

import asyncio
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize


async def main(host: str = "127.0.0.1", port: int = 10300) -> None:
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await client.read_event())
        print("speech-to-text models:", [m.name for p in info.asr for m in p.models])
        print("voices:", [v.name for p in info.tts for v in p.voices][:8])

        # text -> speech
        await client.write_event(Synthesize(text="Hello from bolo over Wyoming.").event())
        pcm, rate = bytearray(), 22050
        while True:
            event = await client.read_event()
            if AudioStart.is_type(event.type):
                rate = AudioStart.from_event(event).rate
            elif AudioChunk.is_type(event.type):
                pcm.extend(AudioChunk.from_event(event).audio)
            elif AudioStop.is_type(event.type):
                break
        with wave.open("wyoming.wav", "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(bytes(pcm))
        print(f"wrote wyoming.wav ({len(pcm) / 2 / rate:.1f}s)")

        # speech -> text (send the audio we just received back)
        await client.write_event(Transcribe(language="en").event())
        await client.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        for off in range(0, len(pcm), 4096):
            await client.write_event(AudioChunk(audio=bytes(pcm[off : off + 4096]), rate=rate, width=2, channels=1).event())
        await client.write_event(AudioStop().event())
        while True:
            event = await client.read_event()
            if Transcript.is_type(event.type):
                print("transcript:", Transcript.from_event(event).text)
                break


if __name__ == "__main__":
    asyncio.run(main())
