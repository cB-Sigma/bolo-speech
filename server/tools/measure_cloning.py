"""Clone real voices with every given model and measure it (docs/measurements/voice-cloning-*.json).

    BOLO_DATA_DIR=... python tools/measure_cloning.py --refs DIR --out FILE MODEL [MODEL ...]

``DIR`` is what ``tools/calibrate_similarity.py pick`` writes (LibriSpeech dev-clean, CC BY 4.0). Two of its
speakers become saved voices (three utterances each, with their transcripts, through the same cleaning as
``POST /api/v1/voices``). For each model and voice the script speaks a test sentence in the model's first
language and records:

* ``similarity``: speaker similarity of the output to the reference (bolo.similarity; 0.50 and above is what the
  same real person scores, see docs/measurements/speaker-similarity-calibration.json);
* ``impostor``: the highest similarity of the output to the other 14 speakers (a clone that resembles everyone
  equally has not cloned anyone);
* ``stock_similarity``: the same model's first catalog voice compared with the reference (the score without
  cloning);
* ``rtf``, ``ttfa_ms`` (time to the first audio), ``audio_s``, after one warm-up request;
* ``wer``: the output transcribed back (moonshine-tiny-en for English, whisper-base otherwise) against the text.

Models must be installed (``bolo pull``) along with pyannote-3.0-eres2net, moonshine-tiny-en and whisper-base.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from bolo import similarity
from bolo.audio import decode_audio
from bolo.config import load_settings
from bolo.core import Bolo
from bolo.obs.trace import Trace
from bolo.voices import clean_reference

VOICES = ("2277", "84")  # LibriSpeech dev-clean speakers that become the saved voices


def cpu_name() -> str:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.platform()


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\s']", " ", text).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return round(d[len(h)] / max(1, len(r)), 3)


async def speak(core: Bolo, text: str, model: str, voice: str | None, custom: bool, language: str) -> dict[str, Any]:
    trace = Trace("tts", "measure", model)
    parts = []
    rate = 24000
    started = time.perf_counter()
    try:
        async for chunk in core.synthesize(
            text, model=model, voice=voice, custom_voice=custom, trace=trace, language=language
        ):
            trace.mark_ttfa()
            parts.append(chunk.samples)
            rate = chunk.sample_rate
    finally:
        core.complete_tts(trace)
    wall = time.perf_counter() - started
    audio = np.concatenate(parts)
    return {"audio": audio, "rate": rate, "wall": wall, "ttfa_ms": trace.ttfa_ms}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    meta = json.loads(Path(args.refs, "meta.json").read_text(encoding="utf-8"))
    core = Bolo(load_settings(None, {"auto_pull": False, "threads": args.threads}))  # BOLO_* variables apply
    await core.start()
    results: dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": f"{os.cpu_count()} vCPU {platform.machine()} ({cpu_name()})",
        "threads": args.threads,
        "references": "LibriSpeech dev-clean (CC BY 4.0) speakers " + ", ".join(VOICES) + ", three utterances each",
        "models": {},
    }
    try:
        embeddings: dict[str, list[Any]] = {}
        for spk, files in meta["kept"].items():
            embeddings[spk] = []
            for f in files:
                x, r = decode_audio(Path(args.refs, f).read_bytes(), filename=f)
                embeddings[spk].append(await similarity.embed(core, x, r))
        for spk in VOICES:
            vid = f"libri-{spk}"
            if core.voices.find(vid) is None:
                files = meta["kept"][spk][:3]
                clips = [decode_audio(Path(args.refs, f).read_bytes(), filename=f) for f in files]
                text = " ".join(meta["texts"][f.removesuffix(".flac")] for f in files).lower()
                core.voices.create(
                    name=vid, cleaned=clean_reference(clips), consent="permission", transcript=text, language="en",
                    created_by={"auth": "none", "name": "measure_cloning (LibriSpeech, CC BY 4.0)", "client": None},
                )  # fmt: skip
        for model in args.models:
            spec = core.resolve(model, "tts")
            lang = (spec.languages or ["en"])[0]
            sentence = similarity.test_sentence(lang)
            asr = "moonshine-tiny-en" if lang == "en" else "whisper-base"
            entry: dict[str, Any] = {"language": lang, "text": sentence, "asr": asr, "voices": {}}
            try:
                await speak(core, "Hello.", spec.id, f"libri-{VOICES[0]}", True, lang)  # load + warm up
                stock = spec.voices[0]["id"] if spec.voices else None
                for spk in VOICES:
                    vid = f"libri-{spk}"
                    out = await speak(core, sentence, spec.id, vid, True, lang)
                    seconds = len(out["audio"]) / out["rate"]
                    emb = await similarity.embed(core, out["audio"], out["rate"])
                    ref_x, ref_r = decode_audio(core.voices.get(vid).reference_bytes())
                    ref = await similarity.embed(core, ref_x, ref_r)
                    others = [
                        max(similarity.cosine(emb, e) for e in vecs) for s, vecs in embeddings.items() if s != spk
                    ]
                    text = (await core.transcribe(out["audio"], out["rate"], model=asr, language=lang)).text
                    row: dict[str, Any] = {
                        "similarity": round(similarity.cosine(emb, ref), 3),
                        "impostor": round(max(others), 3),
                        "rtf": round(out["wall"] / seconds, 3) if seconds else None,
                        "ttfa_ms": round(out["ttfa_ms"]) if out["ttfa_ms"] is not None else None,
                        "audio_s": round(seconds, 2),
                        "wer": wer(sentence, text),
                        "heard": text,
                    }
                    if stock:
                        base = await speak(core, sentence, spec.id, stock, False, lang)
                        stock_emb = await similarity.embed(core, base["audio"], base["rate"])
                        row["stock_voice"] = stock
                        row["stock_similarity"] = round(similarity.cosine(stock_emb, ref), 3)
                    entry["voices"][vid] = row
                    print(model, vid, {k: v for k, v in row.items() if k != "heard"}, flush=True)
                sims = [v["similarity"] for v in entry["voices"].values()]
                entry["cloned"] = all(s >= similarity.SAME_SPEAKER for s in sims)
                entry["similarity_mean"] = round(float(np.mean(sims)), 3)
            except Exception as exc:  # a model that cannot run here is recorded, not fatal
                entry["error"] = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
                print(model, "error", entry["error"], flush=True)
            finally:
                await core.manager.unload(spec.id)
            results["models"][spec.id] = entry
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        await core.stop()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--refs", required=True)
    p.add_argument("--out")
    p.add_argument("--threads", type=int, default=4)
    asyncio.run(run(p.parse_args()))
