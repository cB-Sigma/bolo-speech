"""Run every audio-tagging and emotion model of the catalog that is installed on a set of clips
(docs/measurements/audio-classifiers.json): load time, real-time factor, and the top classes per clip.

    BOLO_DATA_DIR=... python tools/measure_classifiers.py CLIP [CLIP ...] > audio-classifiers.json
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

from bolo.audio import decode_audio
from bolo.classify import classify
from bolo.config import Settings
from bolo.core import Bolo
from bolo.obs.trace import Trace


async def measure(paths: list[str]) -> dict[str, Any]:
    import onnxruntime
    import sherpa_onnx

    core = Bolo(Settings(data_dir=Path(os.environ["BOLO_DATA_DIR"]), auto_pull=False))
    await core.start()
    clips = {Path(p).name: decode_audio(Path(p).read_bytes(), filename=p) for p in paths}
    out: dict[str, Any] = {
        "what": "audio tagging and speech emotion models, one request at a time (bolo.classify)",
        "hardware": f"{os.cpu_count()} vCPU {platform.machine()}, no GPU",
        "packages": {"sherpa-onnx": sherpa_onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "date": time.strftime("%Y-%m-%d"),
        "load_average": round(os.getloadavg()[0], 1),
        "clips": {name: round(len(x) / r, 2) for name, (x, r) in clips.items()},
        "models": {},
    }
    try:
        for spec in core.catalog.all():
            if spec.task not in ("audio-tagging", "emotion") or spec.engine == "fake":
                continue
            if not core.store.is_installed(spec.id):
                continue
            t0 = time.perf_counter()
            async with core.manager.use(spec.id):
                load_s = time.perf_counter() - t0
                wall = audio = 0.0
                tops: dict[str, Any] = {}
                for name, (x, r) in clips.items():
                    t1 = time.perf_counter()
                    result = await classify(core, x, r, spec.id, spec.task, Trace(spec.task, "tool", spec.id))
                    wall += time.perf_counter() - t1
                    audio += len(x) / r
                    tops[name] = [[label, round(score, 3)] for label, score in result.scores[:3]]
            out["models"][spec.id] = {
                "task": spec.task,
                "load_s": round(load_s, 2),
                "rtf": round(wall / audio, 4),
                "audio_s": round(audio, 1),
                "top3": tops,
            }
            print(spec.id, out["models"][spec.id]["rtf"], file=sys.stderr)
            await core.manager.unload(spec.id)
    finally:
        await core.stop()
    return out


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure(sys.argv[1:])), indent=1, ensure_ascii=False))
