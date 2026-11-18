#!/usr/bin/env python3
"""Word error rate of recognisers on noisy speech with and without a denoiser, through a running bolo server.

The repository's two English samples (``en-short``, ``en-long``; 47 words) are mixed with white and pink noise
(numpy, seed 7) and with LibreOffice's gallery sounds ``applause.wav`` and ``train.wav`` (looped, resampled to
16 kHz) at 10, 5 and 0 dB SNR over each clip, and sent to ``POST /v1/audio/transcriptions`` with ``denoise`` off,
at full strength and at strength 0.5. WER is over both clips together, lower case, punctuation removed (jiwer).
Results go to a JSON file that the next run resumes from::

    python tools/measure_denoise.py --url http://127.0.0.1:8000 --out denoise-runs.json off dpdfnet2 deepfilternet3

Each denoiser is pulled by the server on first use (auto_pull); ``--delete`` removes it afterwards. ``off`` measures
the recognisers without denoising. ``--rtf`` also loads each denoiser in this process from ``--data-dir`` and
measures its CPU time per second of audio (one thread, 74 s of speech) and the time per chunk of the live stream's
denoiser (20 ms chunks, 15 s of speech; 3 s for a model that overlaps chunks), and writes them to the same file.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "src" / "bolo" / "samples"
GALLERY = Path("/usr/lib/libreoffice/share/gallery/sounds")
RECOGNISERS = ["moonshine-v2-tiny-en", "whisper-base"]
NOISES = ["white", "pink", "applause", "train"]
SNRS = [10, 5, 0]


def clips() -> list[tuple[str, np.ndarray, str]]:
    index = json.loads((SAMPLES / "index.json").read_text())
    out = []
    for item in index:
        if item["id"] in ("en-short", "en-long"):
            x, _ = sf.read(SAMPLES / item["file"], dtype="float32")
            out.append((item["id"], x, item["text"]))
    return out


def noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "pink":
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.arange(len(spec), dtype=np.float64)
        f[0] = 1.0
        return np.fft.irfft(spec / np.sqrt(f), n).astype(np.float32)
    import soxr

    y, sr = sf.read(GALLERY / f"{kind}.wav", dtype="float32", always_2d=True)
    y = soxr.resample(y.mean(axis=1), sr, 16000).astype(np.float32)
    return np.tile(y, n // len(y) + 1)[:n]


def mix(x: np.ndarray, nz: np.ndarray, snr: float) -> np.ndarray:
    ps, pn = float(np.mean(x**2)), float(np.mean(nz**2))
    return (x + nz * np.sqrt(ps / (pn * 10 ** (snr / 10)))).astype(np.float32)


def conditions() -> dict[str, list[tuple[str, np.ndarray, str]]]:
    rng = np.random.default_rng(7)
    cs = clips()
    out = {"clean": cs}
    for kind in NOISES:
        nz = {cid: noise(kind, len(x), rng) for cid, x, _ in cs}
        for snr in SNRS:
            out[f"{kind}@{snr}dB"] = [(cid, mix(x, nz[cid], snr), t) for cid, x, t in cs]
    return out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def transcribe(client: Any, url: str, model: str, audio: np.ndarray, denoise: str, strength: float | None) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV", subtype="FLOAT")
    data = {"model": model, "response_format": "json", "language": "en", "denoise": denoise}
    if strength is not None:
        data["denoise_strength"] = str(strength)
    r = client.post(url + "/v1/audio/transcriptions", data=data, files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    r.raise_for_status()
    header = r.headers.get("x-bolo-denoise")
    if denoise != "off" and (header is None or header.startswith("skipped")):
        raise SystemExit(f"{denoise} did not run: {header}")
    return str(r.json()["text"])


def wer_runs(url: str, denoiser: str, results: dict[str, Any], save: Any) -> None:
    import httpx
    import jiwer

    settings = {"off": ("off", None)} if denoiser == "off" else {"on": (denoiser, None), "half": (denoiser, 0.5)}
    conds = conditions()
    with httpx.Client(timeout=900) as client:
        for rec in RECOGNISERS:
            for cname, items in conds.items():
                for label, (dn, st) in settings.items():
                    slot = results.setdefault(denoiser, {}).setdefault(rec, {}).setdefault(cname, {})
                    if label in slot:
                        continue
                    refs = [norm(t) for _, _, t in items]
                    hyps = [norm(transcribe(client, url, rec, audio, dn, st)) for _, audio, _ in items]
                    slot[label] = round(jiwer.wer(" ".join(refs), " ".join(hyps)), 3)
                    print(denoiser, rec, cname, label, slot[label], flush=True)
                    save()


def rtf_run(model_id: str, data_dir: Path) -> dict[str, Any]:
    from bolo.audio.resample import resample
    from bolo.catalog import Catalog
    from bolo.denoise import LiveDenoiser
    from bolo.engines import EngineRegistry
    from bolo.engines.base import DenoiseModel, LoadOptions

    spec = Catalog.load().resolve(model_id)
    assert spec is not None
    if spec.engine == "plugin:torch":  # the PyTorch plugin's engine, in this process (bolo-torch installed here)
        from bolo_torch.engine import TorchEngine

        engine: Any = TorchEngine()
    else:
        engine = EngineRegistry(settings=None).get(spec.engine)
    assert engine is not None
    path = data_dir / "models" / model_id
    model = engine.load(spec.kind, spec.task, path, dict(spec.options), LoadOptions(threads=1))
    assert isinstance(model, DenoiseModel)
    speech = np.concatenate([x for _, x, _ in clips()] * 5)
    audio = resample(speech, 16000, model.sample_rate)
    model.denoise(audio[: model.sample_rate])
    runs = []
    for _ in range(3):
        c0, w0 = time.process_time(), time.perf_counter()
        if spec.engine.startswith("plugin:"):  # PyTorch models in 15 s pieces (a non-causal one needs GBs for 74 s)
            piece = 15 * model.sample_rate
            for i in range(0, len(audio), piece):
                model.denoise(audio[i : i + piece])
        else:
            model.denoise(audio)
        runs.append((time.process_time() - c0, time.perf_counter() - w0))
    seconds = len(speech) / 16000
    live = LiveDenoiser(model, 16000, 1.0)
    clip = speech[: 16000 * (15 if live.stream.stateful else 3)]  # an overlapping model: 3 s are enough
    for i in range(0, len(clip), 320):
        live.process(clip[i : i + 320])
    live.finish()
    return {
        "audio_s": round(seconds, 1),
        "rtf_cpu": round(min(r[0] for r in runs) / seconds, 4),
        "rtf_wall": round(min(r[1] for r in runs) / seconds, 4),
        "live_20ms": live.summary(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("denoisers", nargs="+", help="denoise model ids, or off")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rtf", action="store_true", help="also measure CPU time and live chunk time in this process")
    ap.add_argument("--data-dir", type=Path, help="the server's data_dir (for --rtf)")
    ap.add_argument("--delete", action="store_true", help="delete each denoiser from the server afterwards")
    args = ap.parse_args()
    results: dict[str, Any] = json.loads(args.out.read_text()) if args.out.exists() else {}

    def save() -> None:
        args.out.write_text(json.dumps(results, indent=1))

    for denoiser in args.denoisers:
        wer_runs(args.url, denoiser, results, save)
        if denoiser != "off" and args.rtf:
            if args.data_dir is None:
                raise SystemExit("--rtf needs --data-dir")
            results.setdefault("speed", {})[denoiser] = rtf_run(denoiser, args.data_dir)
            print(denoiser, results["speed"][denoiser], flush=True)
            save()
        if denoiser != "off" and args.delete:
            import httpx

            httpx.delete(f"{args.url}/api/v1/models/{denoiser}", timeout=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
