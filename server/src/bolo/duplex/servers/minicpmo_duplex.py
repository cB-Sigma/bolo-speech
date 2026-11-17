"""MiniCPM-o 4.5's full-duplex mode behind Kyutai's WebSocket protocol, so that bolo's Kyutai backend can talk to it.

This file is not imported by bolo: bolo starts it with the Python of the ``minicpm-o`` component (PyTorch,
transformers 4.51.0, minicpmo-utils, sphn and websockets; bolo.duplex.runtime). It uses only that environment.

MiniCPM-o 4.5 (openbmb/MiniCPM-o-4_5, Apache-2.0) is full-duplex in one-second steps through its own remote code
(``MiniCPMO.as_duplex()``, model card "Duplex Omni Mode"): ``prepare`` takes a system prompt and a reference voice,
then every second ``streaming_prefill`` takes the caller's audio (16 kHz; the first piece 1.035 s) and
``streaming_generate`` answers with whether the model listens or speaks, the text it says and 24 kHz audio. Here:

* ``GET /api/chat?text_prompt=...&voice=...`` opens a conversation (one at a time: the model holds one session);
  the handshake byte ``0`` is sent once ``prepare`` is done;
* audio in: kind ``1``, Ogg Opus at 24 kHz (sphn), resampled to 16 kHz and cut into the model's one-second pieces;
* audio out: kind ``1``, the model's audio as Ogg Opus at 24 kHz; text out: kind ``2``.

The model's remote code runs only because the operator opted in for this model (bolo checks the pinned files first).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

RATE = 24000
MODEL_RATE = 16000


def resample(x: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    if rate_in == rate_out or not len(x):
        same: np.ndarray = x.astype(np.float32)
        return same
    n = round(len(x) * rate_out / rate_in)
    out: np.ndarray = np.interp(np.arange(n) * (rate_in / rate_out), np.arange(len(x)), x).astype(np.float32)
    return out


def as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    flat: np.ndarray = np.asarray(value, dtype=np.float32).reshape(-1)
    return flat


class Conversations:
    """The model and the one conversation it holds at a time."""

    def __init__(self, duplex: Any, voices: dict[str, str], default_voice: str, *, load_audio: Any = None) -> None:
        self.duplex = duplex
        self.voices = voices
        self.default_voice = default_voice
        self.lock = asyncio.Lock()
        self.step_lock = threading.Lock()
        self.load_audio = load_audio or self._load_audio

    @staticmethod
    def _load_audio(path: str) -> np.ndarray:
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return resample(data.mean(axis=1), int(rate), MODEL_RATE)

    def prepare(self, prompt: str | None, voice: str | None) -> None:
        path = self.voices.get(voice or "", self.voices[self.default_voice])
        with self.step_lock:
            self.duplex.prepare(
                prefix_system_prompt=prompt or None, ref_audio=self.load_audio(path), prompt_wav_path=path
            )

    def step(self, audio16: np.ndarray) -> dict[str, Any]:
        with self.step_lock:
            self.duplex.streaming_prefill(audio_waveform=audio16, frame_list=[])
            result: dict[str, Any] = self.duplex.streaming_generate(
                prompt_wav_path=None, max_new_speak_tokens_per_chunk=20, decode_mode="sampling"
            )
            return result

    async def serve(self, ws: Any) -> None:
        import sphn

        path = ws.request.path if getattr(ws, "request", None) is not None else ""
        query = {k: v[0] for k, v in parse_qs(urlsplit(path).query).items()}
        async with self.lock:
            await asyncio.to_thread(self.prepare, query.get("text_prompt"), query.get("voice"))
            writer, reader = sphn.OpusStreamWriter(RATE), sphn.OpusStreamReader(RATE)
            await ws.send(b"\x00")
            pending = np.zeros(0, dtype=np.float32)
            need = int(MODEL_RATE * 1.035)  # the model's first piece is 1035 ms, the others 1000 ms
            async for message in ws:
                if isinstance(message, str) or not message or message[0] != 1:
                    continue
                pcm = np.asarray(reader.append_bytes(bytes(message[1:])), dtype=np.float32).reshape(-1)
                pending = np.concatenate([pending, resample(pcm, RATE, MODEL_RATE)])
                while len(pending) >= need:
                    piece, pending = pending[:need], pending[need:]
                    need = MODEL_RATE
                    result = await asyncio.to_thread(self.step, piece)
                    text = "" if result.get("is_listen") else str(result.get("text") or "")
                    if text:
                        await ws.send(b"\x02" + text.encode("utf-8"))
                    out = as_numpy(result.get("audio_waveform"))
                    if not len(out):
                        out = np.zeros(RATE, dtype=np.float32)  # it listens: a second of silence keeps the clock
                    data = writer.append_pcm(out)
                    if len(data):
                        await ws.send(b"\x01" + bytes(data))


async def run(conversations: Conversations, host: str, port: int) -> None:
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def check(connection: Any, request: Any) -> Any:
        if not request.path.startswith("/api/chat"):
            return Response(404, "Not Found", Headers({"Content-Type": "text/plain"}), b"not found\n")
        return None

    async with serve(conversations.serve, host, port, process_request=check, max_size=None):
        print(f"MiniCPM-o duplex server on ws://{host}:{port}/api/chat", flush=True)
        await asyncio.Future()


def load(path: str, device: str) -> Any:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True,
    )  # fmt: skip
    model.eval().to(device)
    model.init_tts(model_dir=str(Path(path) / "assets" / "token2wav"))
    return model.as_duplex()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voice", action="append", default=[], help="name=relative/path.wav (a reference voice)")
    args = parser.parse_args(argv)
    voices = {}
    for item in args.voice:
        name, _, rel = item.partition("=")
        voices[name] = str(Path(args.model_path) / rel)
    default = next(iter(voices))
    print("loading MiniCPM-o 4.5 (remote code, opted in by the operator)", flush=True)
    duplex = load(args.model_path, args.device)
    conversations = Conversations(duplex, voices, default)
    print("warming up", flush=True)
    conversations.prepare(None, default)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(conversations, args.host, args.port))


if __name__ == "__main__":
    main(sys.argv[1:])
