"""How live transcription starts an utterance: first word right, a stray token before it, word error rate.

    BOLO_DATA_DIR=... python tools/measure_stream_onset.py --out FILE [--session PATH] [--seeds 1,2,3,4] [--only TEXT,...]
    python tools/measure_stream_onset.py --table LABEL=FILE [LABEL=FILE ...]

Each take goes through a live session's own processor (``bolo.streaming.session``) with the real Silero VAD, one
send at a time, each handled before the next arrives (a server keeping up with real time), for
zipformer-en-20m-streaming (VAD-gated streaming) and moonshine-tiny-en (pseudo-streaming). ``--session`` swaps in
another ``session.py`` (one from an earlier commit: ``git show REV:server/src/bolo/streaming/session.py``) so that
everything else stays the same. The takes:

* ``laptop``: what was measured on a Windows laptop with bolo 0.6.2, the samples sent as files in 100 ms pieces.
  en-short, en-long, en-ask-not (ui/src/mock/assets) and a LibriSpeech utterance, each with 0.5 s of digital silence
  before it, without it ("trimmed"), with an 80 ms burst of noise 0.3 s before the speech, and the whole clip 30 and
  40 dB down; each as it is and over a room at -70 and -60 dBFS. "normal" in the table is the first three.
* ``quietword``: simulated microphone takes with a quiet first word (as in 8ccf736): speech at -26 to -36 dBFS,
  its first word 14-24 dB quieter and then a 0.15-0.3 s pause, 0.4-1.5 s of room (-66 to -58 dBFS) before,
  100 and 250 ms sends, a third of them after a second of audio held back; 24 takes per seed, over five clips.
* ``soft300``: the same, but only the first 300 ms quieter, with no pause after it.

A take has a stray token when the transcript's alignment with the reference starts with an inserted word ("o good
morning", "16. Speech servers"). Models must be installed (``bolo pull``), with silero-vad; the LibriSpeech
utterances are fetched into ``BOLO_DATA_DIR/validation`` (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

SR = 16000
REPO = Path(__file__).resolve().parents[2]
MODELS = ("zipformer-en-20m-streaming", "moonshine-tiny-en")
LIBRISPEECH_REPO = "csukuangfj/sherpa-onnx-whisper-tiny"

# clip: (where it is, reference, (start, end) of its first word in seconds)
CLIPS: dict[str, tuple[Callable[[Path], Path], str, tuple[float, float]]] = {
    "en-short": (
        lambda _: REPO / "server/src/bolo/samples/en-short.wav",
        "good morning could you tell me when my order will arrive",
        (0.06, 0.88),  # "Good morning." then a pause
    ),
    "en-long": (
        lambda _: REPO / "server/src/bolo/samples/en-long.wav",
        "speech servers are judged by how quickly the first sound arrives a good one starts speaking before it has "
        "finished reading keeps a steady pace under load and tells you honestly how long each step took",
        (0.06, 0.32),  # "Speech"
    ),
    "ask-not": (
        lambda _: REPO / "ui/src/mock/assets/en-ask-not.wav",
        "and so my fellow americans ask not what your country can do for you ask what you can do for your country",
        (0.32, 0.62),  # "And"
    ),
    "libri0": (
        lambda data: data / "validation" / LIBRISPEECH_REPO.replace("/", "--") / "0.wav",
        "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels",
        (0.40, 0.74),  # "After"
    ),
    "libri1": (
        lambda data: data / "validation" / LIBRISPEECH_REPO.replace("/", "--") / "1.wav",
        "god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was "
        "on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to "
        "be finally a blessed soul in heaven",
        (0.34, 0.66),  # "God"
    ),
}


def clip_audio(name: str, data: Path) -> np.ndarray:
    from bolo.audio import decode_audio, resample

    path = CLIPS[name][0](data)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{LIBRISPEECH_REPO}/resolve/main/test_wavs/{path.name}"
        path.write_bytes(httpx.get(url, follow_redirects=True, timeout=60).content)
    x, rate = decode_audio(path.read_bytes(), filename=path.name)
    return (resample(x, rate, SR) if rate != SR else x).astype(np.float64)


def gain(db: float) -> float:
    return float(10 ** (db / 20))


def level(x: np.ndarray) -> float:
    """RMS of the speech (samples above 1% of the peak), as tests/test_real.py's browser_microphone measures it."""
    return float(np.sqrt(np.mean(x[np.abs(x) > 0.01 * np.abs(x).max()] ** 2)))


@dataclass
class Take:
    name: str
    group: str
    clip: str
    audio: np.ndarray
    send_ms: int = 100
    held_s: float = 0.0  # this much audio arrives in one piece first (held while the stream opened)
    meta: dict[str, float] = field(default_factory=dict)


def laptop_takes(data: Path) -> list[Take]:
    takes = []
    for clip in ("en-short", "en-long", "ask-not", "libri0"):
        x = clip_audio(clip, data)
        for i, room_dbfs in enumerate((None, -70.0, -60.0)):
            rng = np.random.default_rng(100 + i)
            for case in ("lead0.5", "trimmed", "burst", "-30dB", "-40dB"):
                body, lead = x, 0.5
                if case == "trimmed":
                    body, lead = x[np.nonzero(np.abs(x) > 1e-4)[0][0] :], 0.0
                elif case == "burst":
                    lead = 1.0
                elif case.endswith("dB"):
                    body = x * gain(float(case[:-2]))
                lead_n = int(lead * SR)
                a = np.concatenate([np.zeros(lead_n), body, np.zeros(int(1.2 * SR))])
                if room_dbfs is not None:  # the room under everything; only the speech is made quieter
                    a = a + rng.standard_normal(len(a)) * gain(room_dbfs)
                if case == "burst":
                    b0 = lead_n + int(np.nonzero(np.abs(body) > 1e-3)[0][0]) - int(0.38 * SR)
                    a[b0 : b0 + int(0.08 * SR)] += rng.standard_normal(int(0.08 * SR)) * gain(-30.0)
                takes.append(Take(f"{clip}/{case}/room{room_dbfs}", f"laptop:{case}", clip, a))
    return takes


def quiet_word_takes(data: Path, seed: int, pause: bool) -> list[Take]:
    rng = np.random.default_rng(seed)
    takes = []
    names = list(CLIPS)
    for i in range(24):
        clip = names[i % len(names)]
        x = clip_audio(clip, data)
        w0, w1 = (int(t * SR) for t in CLIPS[clip][2])
        speech_dbfs = rng.uniform(-36, -26)
        drop = rng.uniform(14, 24)
        before = rng.uniform(0.4, 1.5)
        room_dbfs = rng.uniform(-66, -58)
        x = x / level(x) * gain(speech_dbfs)
        if pause:
            body = np.concatenate([x[w0:w1] * gain(-drop), np.zeros(int(rng.uniform(0.15, 0.3) * SR)), x[w1:]])
        else:
            body = x[w0:].copy()
            n, ramp = int(0.3 * SR), int(0.05 * SR)
            envelope = np.ones(len(body))
            envelope[:n] = gain(-drop)
            envelope[n : n + ramp] = np.linspace(gain(-drop), 1, ramp)
            body = body * envelope
        a = np.concatenate([np.zeros(int(before * SR)), body, np.zeros(int(1.2 * SR))])
        a = a + rng.standard_normal(len(a)) * gain(room_dbfs)
        group = f"{'quietword' if pause else 'soft300'}:s{seed}"
        takes.append(
            Take(
                f"{group}/{i:02d}/{clip}",
                group,
                clip,
                a,
                100 if i % 2 == 0 else 250,
                1.0 if i % 3 == 2 else 0.0,
                {"speech_dbfs": speech_dbfs, "drop_db": drop},
            )
        )
    return takes


def normalise(text: str) -> str:
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " "))
    return " ".join(w.strip("'") for w in text.split() if w.strip("'"))


def score(reference: str, hypothesis: str) -> dict[str, Any]:
    import jiwer

    r, h = normalise(reference), normalise(hypothesis)
    if not h:
        return {"first": False, "stray": False, "wer": 1.0, "text": hypothesis}
    out = jiwer.process_words(r, h)
    chunks = out.alignments[0]
    return {
        "first": h.split()[0] == r.split()[0],
        "stray": bool(chunks) and chunks[0].type == "insert",
        "wer": float(out.wer),
        "text": hypothesis,
    }


async def transcribe(core: Any, session: Any, model: str, take: Take) -> str:
    from bolo.engines.base import VadParams
    from bolo.obs.trace import Trace

    finals: list[str] = []

    async def send(msg: dict[str, Any]) -> None:
        if msg["type"] == "final":
            finals.append(msg["text"])

    s = session.STTSession(core, send, session.SessionConfig(model=model, interim=False, word_timestamps=False))
    s.trace = Trace("stt", "bench", s.spec.id)
    params = VadParams(
        min_silence_s=core.settings.endpoint_silence_ms / 1000, max_speech_s=core.settings.max_utterance_s
    )
    async with core.manager.use(s.spec.id) as entry:
        detector, _ = await core.vad_detector(params, fallback=True)
        kind = session._NativeProcessor if entry.model.streaming else session._PseudoProcessor
        proc = kind(s, entry, entry.model, detector)
        pcm = np.clip(take.audio, -1, 1)  # what 16-bit PCM carries
        pcm = (np.where(pcm < 0, np.round(pcm * 32768), np.round(pcm * 32767)) / 32768).astype(np.float32)
        off = int(take.held_s * SR)
        if off:
            await proc.audio(pcm[:off], time.perf_counter())
        step = int(take.send_ms * SR / 1000)
        for o in range(off, len(pcm), step):
            await proc.audio(pcm[o : o + step], time.perf_counter())
        await proc.finalize(time.perf_counter(), reason="stop")
    return " ".join(finals)


async def measure(args: argparse.Namespace) -> None:
    import bolo.streaming

    if args.session:  # another session.py, everything else as it is
        spec = importlib.util.spec_from_file_location("bolo.streaming.session", args.session)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["bolo.streaming.session"] = module
        spec.loader.exec_module(module)
        bolo.streaming.session = module  # type: ignore[attr-defined]
    from bolo.config import Settings
    from bolo.core import Bolo
    from bolo.streaming import session

    data = Path(os.environ.get("BOLO_DATA_DIR", Path.home() / ".cache" / "bolo"))
    groups = args.groups.split(",")
    takes = laptop_takes(data) if "laptop" in groups else []
    for seed in (int(s) for s in args.seeds.split(",")):
        takes += [t for pause, g in ((True, "quietword"), (False, "soft300")) if g in groups
                  for t in quiet_word_takes(data, seed, pause)]  # fmt: skip
    if args.only:  # e.g. "en-short/burst/,en-short/-40dB/roomNone,en-long/-40dB/roomNone": the laptop's cases alone
        takes = [t for t in takes if any(part in t.name for part in args.only.split(","))]
    core = Bolo(Settings(data_dir=data, auto_pull=False, stt_ttl_s=-1, tts_ttl_s=-1))
    await core.start()
    rows = []
    try:
        for model in args.models.split(","):
            await core.manager.load(model)
            for take in takes:
                row = score(CLIPS[take.clip][1], await transcribe(core, session, model, take))
                rows.append({"model": model, "take": take.name, "group": take.group, **row, **take.meta})
                flags = ("first" if row["first"] else "     ") + (" stray" if row["stray"] else "      ")
                print(f"{model[:10]} {flags} {row['wer']:.2f} {take.name:32s} | {row['text'][:80]}", flush=True)
    finally:
        await core.stop()
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def bucket(group: str) -> str:
    if group.startswith("laptop:"):
        case = group.split(":", 1)[1]
        return "normal" if case in ("lead0.5", "trimmed", "burst") else case
    return group.split(":", maxsplit=1)[0]


def table(files: list[str]) -> None:
    runs = [(f.split("=", 1)[0], json.loads(Path(f.split("=", 1)[1]).read_text(encoding="utf-8"))) for f in files]
    for model in MODELS:
        print(f"\n{model}\n{'':10s}" + "".join(f" | {label:^24s}" for label, _ in runs))
        print(f"{'':10s}" + " | first    stray  WER     " * len(runs))
        for g in ("normal", "-30dB", "-40dB", "quietword", "soft300", "all"):
            cells = []
            for _, rows in runs:
                sel = [r for r in rows if r["model"] == model and g in ("all", bucket(r["group"]))]
                n = len(sel) or 1
                cells.append(
                    f"{sum(r['first'] for r in sel):3d}/{len(sel):<4d} {sum(r['stray'] for r in sel):3d}"
                    f"    {sum(r['wer'] for r in sel) / n:.3f}  "
                )
            print(f"{g:10s}" + "".join(f" | {c}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="where the results go (JSON)")
    ap.add_argument("--session", help="the session.py to measure instead of the installed one")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--groups", default="laptop,quietword,soft300")
    ap.add_argument("--seeds", default="1,2,3,4", help="room and level draws for the quietword and soft300 takes")
    ap.add_argument("--only", help="only the takes whose name has one of these (comma-separated) in it")
    ap.add_argument("--table", nargs="+", metavar="LABEL=FILE", help="print a before/after table of results")
    args = ap.parse_args()
    if args.table:
        table(args.table)
    elif args.out:
        asyncio.run(measure(args))
    else:
        ap.error("--out or --table")


if __name__ == "__main__":
    main()
