"""Calibrate bolo's speaker-similarity thresholds on real speech (docs/measurements/speaker-similarity-calibration.json).

    curl -sSL https://www.openslr.org/resources/12/dev-clean.tar.gz | python tools/calibrate_similarity.py pick DIR
    BOLO_DATA_DIR=... python tools/calibrate_similarity.py score DIR

``pick`` streams LibriSpeech dev-clean (CC BY 4.0) and keeps five 4-15 s utterances of each of the first 16
speakers (the archive itself is never stored). ``score`` embeds every clip with the default speaker-embedding model
(pyannote-3.0-eres2net must be installed) and prints the same-speaker and different-speaker statistics.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

from bolo import similarity
from bolo.audio import decode_audio
from bolo.config import Settings
from bolo.core import Bolo
from bolo.voices import consistency, speaker_windows


def pick(out: str, want_speakers: int = 16, per: int = 5) -> None:
    kept: dict[str, list[str]] = collections.defaultdict(list)
    texts: dict[str, str] = {}
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if m.name.endswith(".trans.txt"):
                f = tf.extractfile(m)
                assert f is not None
                for line in f.read().decode().splitlines():
                    k, _, t = line.partition(" ")
                    texts[k] = t
                continue
            if not m.name.endswith(".flac"):
                continue
            spk = m.name.split("/")[-3]
            if (spk not in kept and len(kept) >= want_speakers) or len(kept[spk]) >= per:
                continue
            if not 60000 < m.size < 400000:
                continue
            f = tf.extractfile(m)
            assert f is not None
            name = Path(m.name).name
            Path(out, name).write_bytes(f.read())
            kept[spk].append(name)
    Path(out, "meta.json").write_text(json.dumps({"kept": kept, "texts": texts}))


def stats(values: list[float]) -> dict[str, float]:
    v = np.array(values)
    return {
        "n": len(v), "min": round(float(v.min()), 3), "p5": round(float(np.percentile(v, 5)), 3),
        "median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3),
        "max": round(float(v.max()), 3),
    }  # fmt: skip


async def score(root: str) -> dict[str, object]:
    meta = json.loads(Path(root, "meta.json").read_text())
    core = Bolo(Settings(data_dir=Path(os.environ["BOLO_DATA_DIR"]), auto_pull=False))
    await core.start()
    try:
        emb, clips = {}, {}
        for spk, files in meta["kept"].items():
            for f in files:
                x, r = decode_audio(Path(root, f).read_bytes(), filename=f)
                emb[(spk, f)] = await similarity.embed(core, x, r)
                clips[(spk, f)] = (x, r)
        same, diff = [], []
        for a, b in itertools.combinations(list(emb), 2):
            (same if a[0] == b[0] else diff).append(similarity.cosine(emb[a], emb[b]))
        out: dict[str, object] = {"same_speaker": stats(same), "different_speaker": stats(diff)}
        for t in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6):
            out[f"{t:.2f}"] = {
                "same_speaker": round(float(np.mean(np.array(same) >= t)), 3),
                "different_speaker": round(float(np.mean(np.array(diff) >= t)), 3),
            }
        speakers = list(meta["kept"])
        one, two = [], []
        for i, spk in enumerate(speakers):
            a = [clips[(spk, f)] for f in meta["kept"][spk][:2]]
            other = speakers[(i + 1) % len(speakers)]
            b = clips[(other, meta["kept"][other][0])]
            for joined, into in ((np.concatenate([a[0][0], a[1][0]]), one), (np.concatenate([a[0][0], b[0]]), two)):
                vectors = [await similarity.embed(core, w, 16000) for w in speaker_windows(joined, 16000)]
                low = consistency(vectors)
                if low is not None:
                    into.append(low)
        out["window_consistency"] = {"one_speaker": stats(one), "two_speakers": stats(two)}
        return out
    finally:
        await core.stop()


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        pick(sys.argv[2])
    else:
        print(json.dumps(asyncio.run(score(sys.argv[2])), indent=2))
