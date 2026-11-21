"""Run every zero-shot (CLAP) model of the catalog that is installed on bolo's sample clips and synthetic ones
(docs/measurements/audio-classify.json): load time, memory, real-time factor, text-tower time (first and cached),
the top labels per clip, and a classifier trained on its embeddings (speech, music and noise) with its
cross-validated accuracy.

    BOLO_DATA_DIR=... python tools/measure_zero_shot.py > audio-classify.json
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bolo import classifiers
from bolo.config import Settings
from bolo.core import Bolo
from bolo.obs.trace import Trace
from bolo.samples import read_file

RATE = 48000
LABELS = [
    "a man speaking", "a woman speaking", "music", "a piano", "drums", "white noise", "a pure tone", "a beep",
    "silence", "a dog barking", "a siren", "rain", "hold music", "an angry voice",
]  # fmt: skip


def synthetic(kind: str, seconds: float, seed: int, rate: int = RATE) -> np.ndarray:
    """``music``: chords with harmonics and a beat (the key and tempo vary with the seed); ``noise``: white, pink
    or brown noise; ``tone``: a sine between 300 Hz and 3 kHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    if kind == "music":
        root = 196.0 * 2 ** (rng.integers(0, 12) / 12)
        out = np.zeros_like(t)
        beat = rng.uniform(1.5, 3.0)
        for i, step in enumerate(rng.choice([0, 3, 5, 7, 8, 10], 4)):
            seg = slice(i * len(t) // 4, (i + 1) * len(t) // 4)
            tt = t[seg]
            env = np.exp(-4 * ((tt - tt[0]) % (1 / beat)))
            for f in (root * 2 ** (step / 12), root * 2 ** ((step + 4) / 12), root * 2 ** ((step + 7) / 12)):
                for h in (1, 2, 3):
                    out[seg] += env * np.sin(2 * np.pi * f * h * tt) / (3 * h)
        kick = (np.sin(2 * np.pi * beat * t) > 0.97) * np.sin(2 * np.pi * 60 * t)
        return (0.5 * out / np.max(np.abs(out)) + 0.3 * kick).astype(np.float32)
    if kind == "noise":
        white = rng.normal(0, 1, len(t))
        colour = seed % 3
        if colour == 1:  # pink: 1/f power
            spec = np.fft.rfft(white)
            spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
            white = np.fft.irfft(spec, len(t))
        elif colour == 2:  # brown
            white = np.cumsum(white)
            white -= np.convolve(white, np.ones(1001) / 1001, mode="same")
        return (0.25 * white / np.std(white)).astype(np.float32)
    return (0.4 * np.sin(2 * np.pi * rng.uniform(300, 3000) * t)).astype(np.float32)


def speech_pieces(seconds: float = 2.5) -> list[tuple[np.ndarray, int]]:
    """bolo's sample clips cut into pieces (speech examples for training)."""
    out = []
    for name in ("en-short", "en-long", "hi-short"):
        x, r = sample(name)
        n = int(seconds * r)
        out.extend((x[a : a + n], r) for a in range(0, len(x) - n // 2, n))
    return out


def sample(name: str) -> tuple[np.ndarray, int]:
    import io

    import soundfile as sf

    x, r = sf.read(io.BytesIO(read_file(f"{name}.wav")), dtype="float32")
    return x, int(r)


async def measure() -> dict[str, Any]:
    import onnxruntime

    core = Bolo(Settings(data_dir=Path(os.environ["BOLO_DATA_DIR"]), auto_pull=False))
    await core.start()
    clips: dict[str, tuple[np.ndarray, int]] = {}
    for name in ("en-short", "en-long", "hi-short"):
        clips[name] = sample(name)
    clips["music (synthetic)"] = (synthetic("music", 8.0, 1), RATE)
    clips["white noise (synthetic)"] = (synthetic("noise", 6.0, 0), RATE)
    clips["tone (synthetic)"] = (synthetic("tone", 4.0, 2), RATE)
    speech = speech_pieces()
    examples = [("speech", x, r) for x, r in speech]
    n = len(speech)
    examples += [("music", synthetic("music", 2.5, 10 + i), RATE) for i in range(n)]
    examples += [("noise", synthetic("noise", 2.5, 20 + i), RATE) for i in range(n)]
    out: dict[str, Any] = {
        "what": "zero-shot audio classification (CLAP) models, one request at a time (bolo.classifiers)",
        "hardware": f"{os.cpu_count()} vCPU {platform.machine()} ({platform.processor() or 'cpu'}), no GPU",
        "packages": {"onnxruntime": onnxruntime.__version__},
        "date": time.strftime("%Y-%m-%d"),
        "load_average": round(os.getloadavg()[0], 1),
        "labels": LABELS,
        "clips": {name: round(len(x) / r, 2) for name, (x, r) in clips.items()},
        "training_examples": {"speech": n, "music": n, "noise": n},
        "models": {},
    }
    try:
        for spec in core.catalog.all():
            if spec.task != classifiers.TASK or spec.engine in ("fake", "builtin"):
                continue
            if not core.store.is_installed(spec.id):
                continue
            rss0 = core._proc.memory_info().rss
            t0 = time.perf_counter()
            async with core.manager.use(spec.id) as entry:
                load_s = time.perf_counter() - t0
                memory = entry.memory_bytes
                model = entry.model
                t1 = time.perf_counter()
                model.embed_text([f"This is a sound of {label}." for label in LABELS])
                text_first = time.perf_counter() - t1
                t1 = time.perf_counter()
                model.embed_text([f"This is a sound of {label}." for label in LABELS])
                text_cached = time.perf_counter() - t1
                wall = audio = 0.0
                tops: dict[str, Any] = {}
                for name, (x, r) in clips.items():
                    t1 = time.perf_counter()
                    trace = Trace(classifiers.TASK, "tool", spec.id)
                    result = await classifiers.zero_shot(core, x, r, spec.id, LABELS, trace)
                    wall += time.perf_counter() - t1
                    audio += len(x) / r
                    tops[name] = [[label, round(score, 3)] for label, score, _ in result.scores[:3]]
                rss = core._proc.memory_info().rss - rss0
            t1 = time.perf_counter()
            trained = await classifiers.train(core, f"measure-{spec.id}"[:64], examples, encoder=spec.id, replace=True)
            train_wall = time.perf_counter() - t1
            await classifiers.delete(core, trained["id"])
            out["models"][spec.id] = {
                "load_s": round(load_s, 2),
                "memory_mb": round(max(memory, rss) / 1e6),
                "rtf": round(wall / audio, 4),
                "audio_s": round(audio, 1),
                "text_s": {"labels": len(LABELS), "first": round(text_first, 3), "cached": round(text_cached, 5)},
                "top3": tops,
                "trained": {
                    "cv_accuracy": trained["cross_validation"]["accuracy"],
                    "per_class": trained["cross_validation"]["per_class"],
                    "embed_s": trained["embed_s"],
                    "train_s": trained["train_s"],
                    "wall_s": round(train_wall, 2),
                },
            }
            print(spec.id, out["models"][spec.id]["rtf"], file=sys.stderr)
            await core.manager.unload(spec.id)
    finally:
        await core.stop()
    return out


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure()), indent=1, ensure_ascii=False))
