#!/usr/bin/env python3
"""Make the conversations the live demo's diarization page offers as ready examples.

Every voice is one the demo itself speaks with, under terms that allow redistribution:

- SYSPIN voices (IISc / SPIRE Lab; models MIT, trained on the SYSPIN corpus, CC BY 4.0), as the
  voice build (``ui/scripts/voices/build.py``) publishes them: Indian English female and male,
  Hindi female and male. They read characters, so no phonemiser is needed.
- Kokoro-82M (Apache-2.0), onnx-community's 8-bit ONNX export at a pinned revision, with uccharan
  (Apache-2.0) for the phonemes: a third English voice.

Lines are placed on one 16 kHz track the way the server's tests build conversations: each line after
the previous one with a pause, or starting before it ends (overlap, interruption). The reference
who-spoke-when is written next to the audio from those positions (the voice's own leading and trailing
silence trimmed), so the page can score a result (DER) against it.

    python -m venv /tmp/v && /tmp/v/bin/pip install numpy onnx onnxruntime uccharan==0.5.0
    python ui/scripts/voices/build.py --out /tmp/voices --only syspin-en-female,syspin-en-male,syspin-hi-female,syspin-hi-male
    /tmp/v/bin/python ui/scripts/diarization/make_examples.py --voices /tmp/voices --out ui/public/diarization

Writes ``<id>.mp3`` (mono, 16 kHz, 32 kbit/s; needs ffmpeg) and ``<id>.json`` (reference turns,
transcript per line, credits). The voices draw a little noise inside the graph, so a re-run gives
slightly different audio; its reference is written from the same run.
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

RATE = 16000
KOKORO = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231/"

# (speaker, voice, text, gap before the line in seconds; negative: starts that long before the last line ends)
EXAMPLES = {
    "two-speakers-en": {
        "title": "Two speakers, English, taking turns",
        "language": "en",
        "lines": [
            ("A", "syspin-en-female", "Good morning. Did you get a chance to look at the delivery schedule?", 0.0),
            ("B", "syspin-en-male", "I did. The first shipment leaves on Monday, and the second one on Thursday.", 0.7),
            ("A", "syspin-en-female", "That works for us. Can the second one include the spare parts as well?", 0.6),
            ("B", "syspin-en-male", "Yes, I will add them to the order today and send you the updated invoice.", 0.8),
            ("A", "syspin-en-female", "Perfect. Please copy the warehouse team on that email.", 0.6),
            ("B", "syspin-en-male", "Will do. Anything else before we close?", 0.7),
            ("A", "syspin-en-female", "No, that is everything. Thank you for your help.", 0.6),
        ],
    },
    "three-speakers-overlap-en": {
        "title": "Three speakers, English, with overlaps and interruptions",
        "language": "en",
        "lines": [
            ("A", "syspin-en-female", "Let us start with the budget. We are about ten percent over this quarter.", 0.0),
            ("B", "syspin-en-male", "Most of that is the new servers. We had to buy them early.", 0.5),
            ("C", "kokoro:am_michael", "Sorry, can I jump in? The servers were planned for next year.", -0.9),
            ("B", "syspin-en-male", "They were, but the old ones failed in March.", 0.3),
            ("A", "syspin-en-female", "Right, and that is in the report I sent on Friday.", -0.6),
            ("C", "kokoro:am_michael", "Okay. Then we should move some of next year's money forward.", 0.6),
            ("A", "syspin-en-female", "Agreed.", -0.4),
            ("B", "syspin-en-male", "I can prepare the numbers by tomorrow afternoon.", 0.5),
            ("C", "kokoro:am_michael", "Great, send them to both of us.", 0.6),
        ],
    },
    "two-speakers-hi": {
        "title": "Two speakers, Hindi",
        "language": "hi",
        "lines": [
            ("A", "syspin-hi-female", "नमस्ते, क्या आप कल की बैठक में आ रहे हैं?", 0.0),
            ("B", "syspin-hi-male", "हाँ, मैं आऊँगा। बैठक कितने बजे शुरू होगी?", 0.7),
            ("A", "syspin-hi-female", "सुबह दस बजे। सबसे पहले हम नए दफ़्तर के बारे में बात करेंगे।", 0.6),
            ("B", "syspin-hi-male", "ठीक है। क्या मुझे कुछ साथ लाना होगा?", 0.7),
            ("A", "syspin-hi-female", "बस पिछले महीने की रिपोर्ट ले आइए।", 0.5),
            ("B", "syspin-hi-male", "ज़रूर, मैं आज शाम तक उसे तैयार कर लूँगा। धन्यवाद।", 0.7),
        ],
    },
}

CREDITS = {
    "syspin": "SYSPIN voices by IISc / SPIRE Lab (models MIT; SYSPIN corpus CC BY 4.0), as published with the bolo demo",
    "kokoro": "Kokoro-82M by hexgrad (Apache-2.0), onnx-community ONNX export; phonemes by uccharan (Apache-2.0)",
}


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def trim(x: np.ndarray, floor: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(x) > floor * float(np.abs(x).max() or 1))
    return x[max(0, loud[0] - 160) : loud[-1] + 160] if len(loud) else x


class Voices:
    def __init__(self, voices_dir: Path, work: Path) -> None:
        import onnxruntime as ort

        self.ort, self.dir, self.work = ort, voices_dir, work
        self.sessions: dict[str, object] = {}

    def say(self, voice: str, text: str) -> np.ndarray:
        if voice.startswith("kokoro:"):
            return self.kokoro(voice.split(":", 1)[1], text)
        cfg = json.loads((self.dir / voice / "voice.json").read_text(encoding="utf-8"))
        if voice not in self.sessions:
            self.sessions[voice] = self.ort.InferenceSession(str(self.dir / voice / "model.onnx"))
        ids = coqui_ids(cfg, text)
        inf = cfg["inference"]
        # No noise in the duration predictor (stable timing), half the usual noise in the voice.
        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([inf["noise_scale"] * 0.5, inf["length_scale"], 0.0], dtype=np.float32),
        }
        out = self.sessions[voice].run(None, feed)[0].reshape(-1)  # type: ignore[attr-defined]
        return resample(out, cfg["sample_rate"], RATE)

    def kokoro(self, name: str, text: str) -> np.ndarray:
        import uccharan

        model = self.work / "kokoro-q8.onnx"
        style = self.work / f"{name}.bin"
        for url, dest in ((KOKORO + "onnx/model_quantized.onnx", model), (KOKORO + f"voices/{name}.bin", style)):
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
        if "kokoro" not in self.sessions:
            self.sessions["kokoro"] = self.ort.InferenceSession(str(model))
        tokens = uccharan.phonemize(text, lang="en-us", profile="kokoro")
        ids = uccharan.kokoro_ids(tokens)
        styles = np.fromfile(style, dtype=np.float32).reshape(-1, 256)
        row = styles[min(max(len(ids) - 2, 0), len(styles) - 1)][None, :]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": row.astype(np.float32),
            "speed": np.array([1.0], dtype=np.float32),
        }
        out = self.sessions["kokoro"].run(None, feed)[0].reshape(-1)  # type: ignore[attr-defined]
        return resample(out, 24000, RATE)


def render(spec: dict, voices: Voices) -> tuple[np.ndarray, list[dict]]:
    pieces: list[tuple[int, np.ndarray, str, str]] = []
    cursor = 0
    for speaker, voice, text, gap in spec["lines"]:
        audio = trim(voices.say(voice, text))
        audio = 0.8 * audio / float(np.abs(audio).max() or 1)
        start = max(0, cursor + int(gap * RATE))
        pieces.append((start, audio, speaker, text))
        cursor = start + len(audio)
    track = np.zeros(max(s + len(a) for s, a, _, _ in pieces) + RATE // 2, dtype=np.float32)
    turns = []
    for start, audio, speaker, text in pieces:
        track[start : start + len(audio)] += audio
        turns.append({"speaker": speaker, "start": round(start / RATE, 3), "end": round((start + len(audio)) / RATE, 3), "text": text})
    track /= max(1.0, float(np.abs(track).max()) / 0.95)
    return track, turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", required=True, help="output of ui/scripts/voices/build.py with the SYSPIN voices")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        voices = Voices(Path(args.voices), Path(tmp))
        for eid, spec in EXAMPLES.items():
            track, turns = render(spec, voices)
            wav = Path(tmp) / f"{eid}.wav"
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes((np.clip(track, -1, 1) * 32767).astype("<i2").tobytes())
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", str(RATE), "-b:a", "32k",
                 "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(out / f"{eid}.mp3")],
                check=True,
            )  # fmt: skip
            used = {v.split(":")[0] if ":" in v else "syspin" for _, v, _, _ in spec["lines"]}
            meta = {
                "id": eid,
                "title": spec["title"],
                "language": spec["language"],
                "seconds": round(len(track) / RATE, 2),
                "speakers": sorted({t["speaker"] for t in turns}),
                "turns": turns,
                "credits": [CREDITS[u] for u in sorted(used)],
                "made_with": "ui/scripts/diarization/make_examples.py",
            }
            (out / f"{eid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            size = (out / f"{eid}.mp3").stat().st_size
            print(f"{eid}: {meta['seconds']} s, {len(turns)} lines, {size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
