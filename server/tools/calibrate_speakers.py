"""Measure speaker verification and identification with every speaker embedder (docs/measurements/speaker-verification.json).

    curl -sSL https://www.openslr.org/resources/12/dev-clean.tar.gz | python tools/calibrate_similarity.py pick DIR
    BOLO_DATA_DIR=... python tools/calibrate_speakers.py DIR [model ...] > speaker-verification.json

DIR holds five utterances (4 to 15 s) of each of 16 LibriSpeech dev-clean speakers (CC BY 4.0). Every speaker is
enrolled from every pair of their utterances (10 enrolments each, the voiceprint being the mean of the two unit
embeddings, as bolo.speakers computes it) and tested with their other three utterances (genuine trials) and with
every utterance of the 15 others (impostor trials). For each embedder this prints the score distributions, the
equal error rate, the threshold that 1 in 1000 and 1 in 100 impostor trials reach (and how many genuine trials fall
below each), and closed-set identification accuracy (each test utterance against all 16 enrolled speakers).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bolo import similarity
from bolo.audio import decode_audio
from bolo.config import Settings
from bolo.core import Bolo
from bolo.speakers import unit

MODELS = (
    "eres2net-voxceleb", "campplus-voxceleb", "campplus-zh-en", "wespeaker-campplus-voxceleb",
    "wespeaker-resnet34-voxceleb", "titanet-small", "ecapa-tdnn-voxceleb",
)  # fmt: skip


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "n": len(values), "min": round(float(values.min()), 3), "p5": round(float(np.percentile(values, 5)), 3),
        "median": round(float(np.median(values)), 3), "p95": round(float(np.percentile(values, 95)), 3),
        "max": round(float(values.max()), 3),
    }  # fmt: skip


def at_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> dict[str, float]:
    """The lowest threshold (rounded up to 0.01) that at most ``far`` of impostor trials reach."""
    ranked = np.sort(impostor)[::-1]
    allowed = math.floor(far * len(ranked))
    raw = float(ranked[allowed]) if allowed < len(ranked) else float(ranked[-1])
    threshold = math.floor(raw * 100 + 1) / 100  # strictly above the (allowed+1)-th highest impostor score
    return {
        "threshold": round(threshold, 2),
        "false_accepts": round(float(np.mean(impostor >= threshold)), 5),
        "false_rejects": round(float(np.mean(genuine < threshold)), 5),
    }


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(equal error rate, the threshold where false accepts and false rejects are closest)."""
    grid = np.linspace(-1, 1, 2001)
    fa = np.array([np.mean(impostor >= t) for t in grid])
    fr = np.array([np.mean(genuine < t) for t in grid])
    k = int(np.argmin(np.abs(fa - fr)))
    return round(float(fa[k] + fr[k]) / 2, 4), round(float(grid[k]), 3)


async def measure(root: str, models: list[str]) -> dict[str, Any]:
    meta = json.loads(Path(root, "meta.json").read_text(encoding="utf-8"))
    kept: dict[str, list[str]] = meta["kept"]
    core = Bolo(Settings(data_dir=Path(os.environ["BOLO_DATA_DIR"])))
    await core.start()
    clips = {f: decode_audio(Path(root, f).read_bytes(), filename=f) for files in kept.values() for f in files}
    out: dict[str, Any] = {
        "what": "speaker verification and identification with enrolled speakers (bolo.speakers)",
        "data": f"LibriSpeech dev-clean (CC BY 4.0): {len(kept)} speakers, {len(clips)} utterances of 4 to 15 s",
        "protocol": "each speaker enrolled from every pair of their 5 utterances (mean of the unit embeddings); "
        "genuine trials: their other 3 utterances; impostor trials: all utterances of the other speakers; "
        "identification: each held-out utterance against every speaker's voiceprint from the same pair split",
        "hardware": f"{os.cpu_count()} vCPU {platform.machine()}, no GPU",
        "date": time.strftime("%Y-%m-%d"),
        "models": {},
    }
    try:
        for model in models:
            emb = {f: unit(await similarity.embed(core, x, r, model)) for f, (x, r) in clips.items()}
            genuine, impostor = [], []
            correct = total = 0
            for split in itertools.combinations(range(5), 2):
                prints = {s: unit(np.mean([emb[files[i]] for i in split], axis=0)) for s, files in kept.items()}
                for spk, files in kept.items():
                    for i, f in enumerate(files):
                        held_out = i not in split
                        for other, vp in prints.items():
                            score = float(np.dot(emb[f], vp))
                            if other == spk and held_out:
                                genuine.append(score)
                            elif other != spk:
                                impostor.append(score)
                        if held_out:
                            best = max(prints, key=lambda s: float(np.dot(emb[f], prints[s])))
                            correct += best == spk
                            total += 1
            g, i_ = np.array(genuine), np.array(impostor)
            rate, at = eer(g, i_)
            out["models"][model] = {
                "dimensions": len(next(iter(emb.values()))),
                "genuine": stats(g),
                "impostor": stats(i_),
                "eer": rate,
                "eer_threshold": at,
                "far_0.001": at_far(g, i_, 0.001),
                "far_0.01": at_far(g, i_, 0.01),
                "identification_top1": round(correct / total, 4),
                "identification_trials": total,
            }
            print(model, json.dumps(out["models"][model]), file=sys.stderr)
    finally:
        await core.stop()
    return out


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure(sys.argv[1], sys.argv[2:] or list(MODELS))), indent=1))
