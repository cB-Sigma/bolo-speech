#!/usr/bin/env python3
"""Make the example reference voices the live demo's "Clone a voice" page offers.

Each one is synthetic speech from a voice the demo itself speaks with, under terms that allow
redistribution, so nobody's own recording is shipped:

- SYSPIN voices (IISc / SPIRE Lab; models MIT, trained on the SYSPIN corpus, CC BY 4.0), as the demo
  publishes them on GitHub Pages (``voices/<id>/model.onnx`` and ``voice.json``): Indian English female
  and Hindi male. They read characters, so no phonemiser is needed.
- Kokoro-82M (Apache-2.0), onnx-community's 8-bit ONNX export at a pinned revision, with uccharan
  (Apache-2.0) for the phonemes: an American English male voice.

    python -m venv /tmp/v && /tmp/v/bin/pip install numpy onnx onnxruntime uccharan==0.6.1
    /tmp/v/bin/python ui/scripts/clone/make_examples.py --out ui/public/clone

With ``--calibration DIR`` it also writes two short clips per voice for six voices (``<voice>-<n>.wav``),
the pairs ``calibrate_similarity.mjs`` uses to set the page's speaker-similarity thresholds.

Writes ``<id>.mp3`` (mono, 24 kHz, 48 kbit/s; needs ffmpeg) and ``index.json`` (title, language,
transcript, credits for each). The voices draw a little noise inside the graph, so a re-run gives
slightly different audio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "voices"))
from build import coqui_ids  # noqa: E402

RATE = 24000
PAGES = "https://cb-sigma.github.io/bolo/voices/"
KOKORO = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231/"

CREDITS = {
    "syspin": "SYSPIN voice by IISc / SPIRE Lab (model MIT; SYSPIN corpus CC BY 4.0), as the bolo demo publishes it",
    "kokoro": "Kokoro-82M by hexgrad (Apache-2.0), onnx-community ONNX export; phonemes by uccharan (Apache-2.0)",
}

EXAMPLES = [
    {
        "id": "syspin-en-female",
        "title": "Indian English, female (SYSPIN)",
        "language": "en",
        "voice": "syspin-en-female",
        "sentences": [
            "Good morning, and thank you for calling.",
            "I am reading this short passage so that you can hear how my voice sounds.",
            "The weather has been pleasant all week, and we are hoping for a quiet weekend at home.",
        ],
    },
    {
        "id": "kokoro-am-michael",
        "title": "American English, male (Kokoro)",
        "language": "en",
        "voice": "kokoro:am_michael",
        "sentences": [
            "Every morning I walk to the little bakery at the end of our street.",
            "The owner already knows my order, so it only takes a minute.",
            "On the way back I usually call my sister and tell her about my plans for the day.",
        ],
    },
    {
        "id": "syspin-hi-male",
        "title": "Hindi, male (SYSPIN)",
        "language": "hi",
        "voice": "syspin-hi-male",
        "sentences": [
            "नमस्ते, आज मैं आपको अपने गाँव के बारे में बताना चाहता हूँ।",
            "हमारे गाँव में एक बड़ा तालाब है और उसके पास एक पुराना पेड़ है।",
            "शाम को बच्चे वहाँ खेलते हैं और बड़े लोग बातें करते हैं।",
        ],
    },
]


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def trim(x: np.ndarray, floor: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(x) > floor * float(np.abs(x).max() or 1))
    return x[max(0, loud[0] - 240) : loud[-1] + 240] if len(loud) else x


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


class Voices:
    def __init__(self, work: Path) -> None:
        import onnxruntime as ort

        self.ort, self.work = ort, work
        self.sessions: dict[str, object] = {}

    def say(self, voice: str, text: str) -> np.ndarray:
        if voice.startswith("kokoro:"):
            return self.kokoro(voice.split(":", 1)[1], text)
        model = fetch(PAGES + f"{voice}/model.onnx", self.work / f"{voice}.onnx")
        cfg = json.loads(fetch(PAGES + f"{voice}/voice.json", self.work / f"{voice}.json").read_text(encoding="utf-8"))
        if voice not in self.sessions:
            self.sessions[voice] = self.ort.InferenceSession(str(model))
        ids = coqui_ids(cfg, text)
        inf = cfg["inference"]
        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([inf["noise_scale"], inf["length_scale"], inf["noise_scale_w"]], dtype=np.float32),
        }
        out = self.sessions[voice].run(None, feed)[0].reshape(-1)  # type: ignore[attr-defined]
        return resample(out, cfg["sample_rate"], RATE)

    def kokoro(self, name: str, text: str) -> np.ndarray:
        import uccharan

        model = fetch(KOKORO + "onnx/model_quantized.onnx", self.work / "kokoro-q8.onnx")
        style = fetch(KOKORO + f"voices/{name}.bin", self.work / f"{name}.bin")
        if "kokoro" not in self.sessions:
            self.sessions["kokoro"] = self.ort.InferenceSession(str(model))
        ids = uccharan.kokoro_ids(uccharan.phonemize(text, lang="en-us", profile="kokoro"))
        styles = np.fromfile(style, dtype=np.float32).reshape(-1, 256)
        row = styles[min(max(len(ids) - 2, 0), len(styles) - 1)][None, :]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": row.astype(np.float32),
            "speed": np.array([1.0], dtype=np.float32),
        }
        return self.sessions["kokoro"].run(None, feed)[0].reshape(-1).astype(np.float32)  # type: ignore[attr-defined]


CALIBRATION_VOICES = [
    "syspin-en-female",
    "syspin-en-male",
    "syspin-hi-female",
    "syspin-hi-male",
    "kokoro:am_michael",
    "kokoro:af_heart",
]
CALIBRATION_TEXT = {
    "en": [
        "The train to the city leaves every twenty minutes from the second platform.",
        "Please remember to bring your identity card and a copy of the letter.",
    ],
    "hi": [
        "शहर जाने वाली रेलगाड़ी हर बीस मिनट में दूसरे प्लेटफ़ॉर्म से जाती है।",
        "कृपया अपना पहचान पत्र और पत्र की एक प्रति साथ लाना याद रखें।",
    ],
}


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def calibration(voices: Voices, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for voice in CALIBRATION_VOICES:
        lang = "hi" if "-hi-" in voice else "en"
        for n, text in enumerate(CALIBRATION_TEXT[lang], start=1):
            a = trim(voices.say(voice, text))
            write_wav(out / f"{voice.replace(':', '-')}-{n}.wav", 0.8 * a / float(np.abs(a).max() or 1))
    print(f"calibration clips in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None, help="where downloaded voices are kept (default: a temp dir)")
    ap.add_argument("--calibration", type=Path, default=None, help="also write the similarity calibration clips here")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.work or Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    voices = Voices(work)
    index = []
    for ex in EXAMPLES:
        pieces = []
        for s in ex["sentences"]:
            a = trim(voices.say(ex["voice"], s))
            pieces += [0.8 * a / float(np.abs(a).max() or 1), np.zeros(int(0.35 * RATE), dtype=np.float32)]
        audio = np.concatenate(pieces)
        wav = work / f"{ex['id']}.wav"
        write_wav(wav, audio)
        mp3 = args.out / f"{ex['id']}.mp3"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", str(RATE), "-b:a", "48k", str(mp3)],
            check=True,
        )
        source = "kokoro" if ex["voice"].startswith("kokoro:") else "syspin"
        index.append(
            {
                "id": ex["id"],
                "title": ex["title"],
                "language": ex["language"],
                "file": mp3.name,
                "seconds": round(len(audio) / RATE, 2),
                "transcript": " ".join(ex["sentences"]),
                "credit": CREDITS[source],
                "synthetic": True,
            }
        )
        print(f"{mp3}: {len(audio) / RATE:.1f} s, {mp3.stat().st_size // 1024} KB")
    if args.calibration:
        calibration(voices, args.calibration)
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
