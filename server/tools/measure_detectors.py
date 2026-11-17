"""Measure voice-activity detectors, end-of-turn models and speaker embedders on the CPU (docs/measurements/).

    BOLO_DATA_DIR=... python tools/measure_detectors.py vad --out FILE MODEL [MODEL ...]
    BOLO_DATA_DIR=... python tools/measure_detectors.py turn --out FILE MODEL [MODEL ...]
    BOLO_DATA_DIR=... python tools/measure_detectors.py embed --librispeech DIR --out FILE MODEL [MODEL ...]

Every model goes through bolo's own code path (``bolo pull``, the model manager, the same detector objects a
Realtime session gets). ``--remove`` deletes each downloaded model after it was measured.

**vad**: bolo's three sample clips (``bolo/samples``, synthesised speech) and the three clone samples of the web
console (``bolo/_ui/clone``) are joined into one 68 s stream with 1 to 3 s of silence between them, and the stream
is measured clean and with added noise:

* white noise at 10 dB and 0 dB SNR, pink noise at 5 dB, low rumble (brown noise) at 0 dB, the speech 30 dB
  quieter (-45 dBFS speech in faint white noise), and non-speech sounds in the gaps only: harmonic tones that
  glide in pitch (a stand-in for music) and short clicks.

The reference: the 10 ms frames of each clip within 40 dB of the clip's peak, with pauses shorter than 0.3 s
filled in (the detectors are asked to bridge those). Detectors run with threshold 0.5, 0.3 s minimum silence,
0.25 s minimum speech and no padding, fed 20 ms at a time as in a live session. Reported per condition: frame
accuracy, speech recall (share of speech frames detected), false alarms (share of non-speech frames called
speech), the number of segments (the reference has six), and how long after the audio the start and the end of
each clip were reported (median; the end includes the 0.3 s of silence the detector waits for). ``rtf`` is the
detector's time over the stream's duration, clean condition.

**turn**: the repository has no end-of-turn test data, so the set is scripted here: 20 English and 6 Hindi
sentences a voice agent might hear, each with an unfinished beginning ("Can you book me a table for"). The audio is
synthesised by an installed voice (``--voice-model``, Kokoro by default); the unfinished audio is the finished
sentence cut at its quietest 20 ms frame near the beginning's share of the words, so it has mid-sentence prosody.
Models that listen get the audio with 0.2 s of silence after it, as a Realtime session hands it over; models that
read get the text, as written and with the final punctuation removed (recognisers differ in whether they write it).
Reported: accuracy at 0.5, the share of each class called right, AUC (how well finished and unfinished are told
apart at any threshold), median probabilities and milliseconds per call (``load_average`` says how busy the machine
was).

**embed**: speaker embedders on the repository's clips (whose voices are known: the Kokoro voices of the samples,
the console's cloning samples) plus three sentences in each of eight Kokoro voices, and, with ``--librispeech``,
real read speech (LibriSpeech dev-clean, 16 speakers, 5 utterances each, as ``tools/calibrate_similarity.py pick``
writes it). Every pair of clips is scored (cosine of the embeddings): same speaker against different speakers,
with the equal error rate, AUC and the shares at or above bolo's 0.5 "same speaker" threshold. ``rtf``: embedding
time over the audio's duration.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bolo.audio import decode_audio
from bolo.audio.resample import resample
from bolo.config import load_settings
from bolo.core import Bolo
from bolo.engines.base import VadParams

SR = 16000
HERE = Path(__file__).resolve().parent.parent / "src" / "bolo"
CLIPS = [
    HERE / "samples" / "en-short.wav",
    HERE / "samples" / "en-long.wav",
    HERE / "samples" / "hi-short.wav",
    HERE / "_ui" / "clone" / "kokoro-am-michael.mp3",
    HERE / "_ui" / "clone" / "syspin-en-female.mp3",
    HERE / "_ui" / "clone" / "syspin-hi-male.mp3",
]
GAPS_S = [1.5, 2.0, 3.0, 1.0, 2.5, 2.0, 1.5]
FRAME = 160  # 10 ms reference frames


def cpu_name() -> str:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.platform()


def hardware() -> str:
    return f"{os.cpu_count()} vCPU {platform.machine()} ({cpu_name()}), no GPU"


# ---------------------------------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------------------------------
def load_clip(path: Path) -> np.ndarray:
    x, rate = decode_audio(path.read_bytes(), filename=path.name)
    return resample(np.asarray(x, dtype=np.float32), rate, SR)


def speech_frames(x: np.ndarray) -> np.ndarray:
    """10 ms frames within 40 dB of the clip's peak, with pauses shorter than 0.3 s filled in."""
    frames = len(x) // FRAME
    rms = np.sqrt(np.mean(x[: frames * FRAME].reshape(frames, FRAME).astype(np.float64) ** 2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms / rms.max()) > -40.0
    idx = np.nonzero(loud)[0]
    for a, b in itertools.pairwise(idx):
        if 1 < b - a <= 30:
            loud[a:b] = True
    return loud


def build_stream() -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """The test stream, each clip's (first, last) speech sample and the 10 ms speech reference."""
    parts: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    marks: list[np.ndarray] = []
    for i, path in enumerate(CLIPS):
        gap = np.zeros(int(GAPS_S[i] * SR), dtype=np.float32)
        x = load_clip(path)
        x = x / max(1e-6, float(np.sqrt(np.mean(x**2)))) * 10 ** (-20 / 20)  # every clip at -20 dBFS RMS
        x = x[: len(x) // FRAME * FRAME].astype(np.float32)
        loud = speech_frames(x)
        pos = sum(len(p) for p in parts) + len(gap)
        idx = np.nonzero(loud)[0]
        spans.append((pos + int(idx[0]) * FRAME, pos + (int(idx[-1]) + 1) * FRAME))
        parts += [gap, x]
        marks += [np.zeros(len(gap) // FRAME, dtype=bool), loud]
    tail = np.zeros(int(GAPS_S[-1] * SR), dtype=np.float32)
    parts.append(tail)
    marks.append(np.zeros(len(tail) // FRAME, dtype=bool))
    return np.concatenate(parts), spans, np.concatenate(marks)


def coloured(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    spectrum /= np.sqrt(f) if kind == "pink" else f
    out = np.fft.irfft(spectrum, n)
    return out / out.std()


def at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, spans: list[tuple[int, int]]) -> np.ndarray:
    power = np.mean(np.concatenate([speech[a:b] for a, b in spans]) ** 2)
    noise = noise * np.sqrt(power / np.mean(noise**2) / 10 ** (snr_db / 10))
    return (speech + noise).astype(np.float32)


def gap_sounds(stream: np.ndarray, spans: list[tuple[int, int]], kind: str, rng: np.random.Generator) -> np.ndarray:
    out = stream.copy()
    edges = [0, *[x for span in spans for x in span], len(stream)]
    for a, b in zip(edges[0::2], edges[1::2], strict=True):
        a, b = a + SR // 5, b - SR // 5  # keep clear of the speech
        if b - a < SR // 2:
            continue
        t = np.arange(b - a) / SR
        if kind == "tones":
            f0 = 220 * 2 ** (np.cumsum(rng.uniform(-1, 1, b - a)) / SR)  # a gliding pitch
            phase = 2 * np.pi * np.cumsum(f0) / SR
            sound = sum(np.sin(k * phase) / k for k in range(1, 6))
            out[a:b] += (0.05 * sound * np.minimum(1, np.minimum(t, t[::-1]) * 20)).astype(np.float32)
        else:
            for c in rng.integers(a, b - 400, 6):
                out[c : c + 400] += (0.3 * rng.standard_normal(400) * np.exp(-np.arange(400) / 60)).astype(np.float32)
    return out


def conditions(stream: np.ndarray, spans: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    n = len(stream)
    quiet = stream * 10 ** (-30 / 20)
    return {
        "clean": stream,
        "white noise 10 dB SNR": at_snr(stream, coloured(n, "white", rng), 10, spans),
        "white noise 0 dB SNR": at_snr(stream, coloured(n, "white", rng), 0, spans),
        "pink noise 5 dB SNR": at_snr(stream, coloured(n, "pink", rng), 5, spans),
        "low rumble (brown noise) 0 dB SNR": at_snr(stream, coloured(n, "brown", rng), 0, spans),
        "quiet speech (-50 dBFS) in faint white noise, 20 dB SNR": at_snr(quiet, coloured(n, "white", rng), 20, spans),
        "gliding harmonic tones in the gaps": gap_sounds(stream, spans, "tones", rng),
        "clicks in the gaps": gap_sounds(stream, spans, "clicks", rng),
    }


def score_vad(detector: Any, audio: np.ndarray, spans: list[tuple[int, int]], ref: np.ndarray) -> dict[str, Any]:
    step = 320
    events: list[tuple[str, int, int]] = []  # kind, sample, samples fed when it was reported
    started = time.perf_counter()
    for off in range(0, len(audio), step):
        for ev in detector.accept(audio[off : off + step]):
            events.append((ev.kind, ev.sample, min(off + step, len(audio))))
    for ev in detector.flush():
        events.append((ev.kind, ev.sample, len(audio)))
    took = time.perf_counter() - started
    frames = len(audio) // FRAME
    hyp = np.zeros(frames, dtype=bool)
    ref = ref[:frames]
    segments: list[tuple[int, int, int, int]] = []
    open_at: tuple[int, int] | None = None
    for kind, sample, fed in events:
        if kind == "speech_start":
            open_at = (sample, fed)
        elif kind == "speech_end" and open_at is not None:
            segments.append((open_at[0], sample, open_at[1], fed))
            hyp[open_at[0] // FRAME : sample // FRAME] = True
            open_at = None
    start_delay, end_delay = [], []
    for a, b in spans:
        hits = [s for s in segments if s[0] < b and s[1] > a]
        if hits and hits[0][0] >= a - SR // 2 and hits[-1][1] <= b + SR:  # a segment that is about this clip
            start_delay.append((hits[0][2] - a) / SR)
            end_delay.append((hits[-1][3] - b) / SR)
    speech, silence = ref.sum(), (~ref).sum()
    return {
        "frame_accuracy": round(float((hyp == ref).mean()), 4),
        "speech_recall": round(float((hyp & ref).sum() / speech), 4),
        "false_alarm": round(float((hyp & ~ref).sum() / silence), 4),
        "segments": len(segments),
        "start_reported_after_s": round(float(np.median(start_delay)), 3) if start_delay else None,
        "end_reported_after_s": round(float(np.median(end_delay)), 3) if end_delay else None,
        "clips_missed": sum(1 for a, b in spans if not any(s[0] < b and s[1] > a for s in segments)),
        "seconds": round(took, 4),
    }


async def measure_vad(core: Bolo, names: list[str], remove: bool) -> dict[str, Any]:
    stream, spans, ref = build_stream()
    conds = conditions(stream, spans)
    params = VadParams(threshold=0.5, min_silence_s=0.3, min_speech_s=0.25, max_speech_s=30.0, speech_pad_s=0.0)
    out: dict[str, Any] = {}
    for name in names:
        spec = core.resolve(name, "vad")
        if not spec.builtin and not core.store.is_installed(spec.id):
            await core.pull_and_wait(spec.id)
        rss = _rss()
        load_started = time.perf_counter()
        await core.vad_detector(params, spec.id)  # loads the model
        load_s = time.perf_counter() - load_started
        results: dict[str, Any] = {}
        for label, audio in conds.items():
            detector, used = await core.vad_detector(params, spec.id)
            assert used == spec.id, used
            results[label] = score_vad(detector, audio, spans, ref)
        clean = results["clean"]
        noisy = [r for k, r in results.items() if k != "clean"]
        out[spec.id] = {
            "load_s": round(load_s, 3),
            "memory_mb": round(max(0, _rss() - rss) / 1e6),
            "rtf": round(clean["seconds"] / (len(stream) / SR), 5),
            "frame_accuracy_clean": clean["frame_accuracy"],
            "frame_accuracy_noisy_mean": round(float(np.mean([r["frame_accuracy"] for r in noisy])), 4),
            "conditions": results,
        }
        print(spec.id, json.dumps({k: v for k, v in out[spec.id].items() if k != "conditions"}), file=sys.stderr)
        await core.manager.unload(spec.id)
        if remove and not spec.builtin:
            await core.remove(spec.id)
    return {
        "stream_s": round(len(stream) / SR, 2),
        "clips": [str(p.relative_to(HERE)) for p in CLIPS],
        "reference": "10 ms frames of each clip within 40 dB of the clip's peak, pauses under 0.3 s filled in",
        "reference_speech_s": round(float(ref.sum()) * FRAME / SR, 2),
        "params": {"threshold": 0.5, "min_silence_s": 0.3, "min_speech_s": 0.25, "speech_pad_s": 0.0},
        "feed": "20 ms pieces, as a live session",
        "models": out,
    }


# ---------------------------------------------------------------------------------------------------
# end of turn
# ---------------------------------------------------------------------------------------------------
#: (finished sentence, an unfinished beginning of it)
TURN_EN = [
    ("Can you book me a table for two at seven tonight?", "Can you book me a table for"),
    ("I'd like to change my delivery address.", "I'd like to change my"),
    ("What's the weather going to be like tomorrow morning?", "What's the weather going to"),
    ("Please send the invoice to my work email.", "Please send the invoice to"),
    ("My order number is four five two nine.", "My order number is"),
    ("I think that's everything I needed, thanks.", "I think that's everything I"),
    ("How long does shipping usually take to Delhi?", "How long does shipping usually"),
    ("Cancel my subscription starting next month.", "Cancel my subscription starting"),
    ("Could you remind me to call my mother at six?", "Could you remind me to"),
    ("The package arrived but the box was damaged.", "The package arrived but the"),
    ("Is there a pharmacy open near me right now?", "Is there a pharmacy open near"),
    ("I want to transfer five hundred rupees to Priya.", "I want to transfer five hundred"),
    ("Turn off the lights in the living room.", "Turn off the lights in the"),
    ("Yes, that works for me.", "Yes, that works for"),
    ("No, I meant the blue one, not the red one.", "No, I meant the blue one, not the"),
    ("Can you explain why my bill went up this month?", "Can you explain why my"),
    ("Set an alarm for six thirty tomorrow.", "Set an alarm for"),
    ("I'm calling about the job interview on Friday.", "I'm calling about the"),
    ("Please read me the last message from Rahul.", "Please read me the last"),
    ("That's all for now, goodbye.", "That's all for now, and"),
]
TURN_HI = [
    ("क्या आप कल सुबह की ट्रेन का टिकट बुक कर सकते हैं?", "क्या आप कल सुबह की ट्रेन का"),
    ("मेरा ऑर्डर अभी तक नहीं आया है।", "मेरा ऑर्डर अभी तक"),
    ("बिजली का बिल इस महीने बहुत ज़्यादा आया है।", "बिजली का बिल इस महीने"),
    ("मुझे अपना पता बदलना है।", "मुझे अपना पता"),
    ("आज शाम को बारिश होगी क्या?", "आज शाम को"),
    ("धन्यवाद, बस इतना ही चाहिए था।", "धन्यवाद, बस इतना ही चाहिए था और"),
]
TURN_VOICES = {"en": ["af_heart", "am_michael"], "hi": ["hf_alpha"]}


def quiet_cut(audio: np.ndarray, share: float) -> int:
    """The quietest 20 ms frame within 15 % (of the speech) around ``share`` of the way through the speech."""
    frames = len(audio) // 320
    rms = np.sqrt(np.mean(audio[: frames * 320].reshape(frames, 320).astype(np.float64) ** 2, axis=1))
    loud = np.nonzero(rms > rms.max() * 0.03)[0]
    first, last = int(loud[0]), int(loud[-1])
    centre = first + share * (last - first)
    lo, hi = int(centre - 0.15 * (last - first)), int(centre + 0.15 * (last - first))
    k = lo + int(np.argmin(rms[lo:hi]))
    return (k + 1) * 320


async def turn_set(core: Bolo, voice_model: str) -> list[dict[str, Any]]:
    """Finished and unfinished utterances: audio (16 kHz, 0.2 s of silence after, as a Realtime session hands
    over) and text."""
    items: list[dict[str, Any]] = []
    tail = np.zeros(int(0.2 * SR), dtype=np.float32)
    for lang, pairs in (("en", TURN_EN), ("hi", TURN_HI)):
        voices = TURN_VOICES[lang]
        for i, (full, part) in enumerate(pairs):
            audio, rate = await core.synthesize_all(full, model=voice_model, voice=voices[i % len(voices)])
            x = resample(np.asarray(audio, dtype=np.float32), rate, SR)
            share = len(part.split()) / len(full.split())
            cut = x[: quiet_cut(x, share)]
            items.append({"lang": lang, "finished": True, "text": full, "audio": np.concatenate([x, tail])})
            items.append({"lang": lang, "finished": False, "text": part, "audio": np.concatenate([cut, tail])})
    return items


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def turn_scores(items: list[dict[str, Any]], probs: list[float]) -> dict[str, Any]:
    fin = [p for it, p in zip(items, probs, strict=True) if it["finished"]]
    unf = [p for it, p in zip(items, probs, strict=True) if not it["finished"]]
    correct = [(p >= 0.5) == it["finished"] for it, p in zip(items, probs, strict=True)]
    return {
        "accuracy": round(float(np.mean(correct)), 3),
        "finished_called_finished": round(float(np.mean([p >= 0.5 for p in fin])), 3),
        "unfinished_called_unfinished": round(float(np.mean([p < 0.5 for p in unf])), 3),
        "auc": round(auc(fin, unf), 3),
        "median_probability_finished": round(float(np.median(fin)), 3),
        "median_probability_unfinished": round(float(np.median(unf)), 3),
    }


def strip_end(text: str) -> str:
    return text.rstrip(" .?!।,")


async def measure_turn(core: Bolo, names: list[str], remove: bool, voice_model: str) -> dict[str, Any]:
    items = await turn_set(core, voice_model)
    await core.manager.unload(voice_model)
    out: dict[str, Any] = {}
    for name in names:
        spec = core.resolve(name, "turn")
        if not spec.builtin and not core.store.is_installed(spec.id):
            await core.pull_and_wait(spec.id)
        hears = str(spec.options.get("input", "audio"))
        result: dict[str, Any] = {"input": hears}
        async with core.manager.use(spec.id) as entry:
            loaded = core.manager.loaded.get(spec.id)
            if loaded is not None:
                result["load_s"] = round(loaded.load_ms / 1000, 2)
                result["memory_mb"] = round(loaded.memory_bytes / 1e6)
            op = "turn_text" if hears == "text" else "turn"

            async def run(values: list[Any], entry: Any = entry, op: str = op) -> tuple[list[float], list[float]]:
                probs, ms = [], []
                await entry.queue.submit(op, values[0])  # warm-up
                for v in values:
                    t0 = time.perf_counter()
                    p, _ = await entry.queue.submit(op, v)
                    ms.append((time.perf_counter() - t0) * 1000)
                    probs.append(float(p))
                return probs, ms

            for lang in ("en", "hi"):
                subset = [it for it in items if it["lang"] == lang]
                if lang not in spec.languages and spec.languages:
                    continue
                if hears == "text":
                    probs, ms = await run([it["text"] for it in subset])
                    bare, _ = await run([strip_end(it["text"]) for it in subset])
                    result[lang] = {
                        "as written (finished sentences end in . or ?)": turn_scores(subset, probs),
                        "final punctuation removed": turn_scores(subset, bare),
                        "median_ms": round(float(np.median(ms)), 1),
                        "probabilities": [round(p, 3) for p in probs],
                    }
                else:
                    probs, ms = await run([it["audio"] for it in subset])
                    result[lang] = {
                        **turn_scores(subset, probs),
                        "median_ms": round(float(np.median(ms)), 1),
                        "probabilities": [round(p, 3) for p in probs],
                    }
        print(spec.id, json.dumps(result, ensure_ascii=False)[:600], file=sys.stderr)
        await core.manager.unload(spec.id)
        if remove and not spec.builtin:
            await core.remove(spec.id)
        out[spec.id] = result
    return {
        "set": {
            "en": f"{len(TURN_EN)} finished sentences and an unfinished beginning of each (scripted for this test)",
            "hi": f"{len(TURN_HI)} finished Hindi sentences and an unfinished beginning of each",
            "audio": f"synthesised by {voice_model} (voices {TURN_VOICES}); the unfinished audio is the finished "
            "sentence cut at its quietest 20 ms frame near the beginning's share of the words; 0.2 s of silence "
            "after each, as a Realtime session hands it over",
            "text": "the sentence as written (finished ones end in . or ?, as recognisers write them) and with the "
            "final punctuation removed",
        },
        "items": [{"lang": it["lang"], "finished": it["finished"], "text": it["text"]} for it in items],
        "threshold": 0.5,
        "models": out,
    }


# ---------------------------------------------------------------------------------------------------
# speaker embedders
# ---------------------------------------------------------------------------------------------------
#: the repository's clips and whose voice each is
EMBED_CLIPS = {
    "samples/en-short.wav": "kokoro:af_heart",
    "samples/en-long.wav": "kokoro:am_michael",
    "samples/hi-short.wav": "kokoro:hf_alpha",
    "_ui/clone/kokoro-am-michael.mp3": "kokoro:am_michael",
    "_ui/clone/syspin-en-female.mp3": "syspin-en-female",
    "_ui/clone/syspin-hi-male.mp3": "syspin-hi-male",
}
EMBED_VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bf_emma", "bm_george", "hf_alpha"]
EMBED_SENTENCES = [
    "The train to the city leaves every twenty minutes from the second platform.",
    "Please remember to bring your identity card and a copy of the signed form.",
    "We walked along the river until the sun went down behind the old mill.",
]


async def embed_set(core: Bolo, voice_model: str, librispeech: str | None) -> list[tuple[str, str, np.ndarray]]:
    """(source, speaker, 16 kHz audio) for every clip."""
    clips: list[tuple[str, str, np.ndarray]] = []
    for rel, speaker in EMBED_CLIPS.items():
        clips.append((f"repo:{rel}", speaker, load_clip(HERE / rel)))
    for voice in EMBED_VOICES:
        for i, text in enumerate(EMBED_SENTENCES):
            audio, rate = await core.synthesize_all(text, model=voice_model, voice=voice)
            clips.append((f"{voice_model}:{voice}:{i}", f"kokoro:{voice}", resample(np.asarray(audio), rate, SR)))
    if librispeech:
        meta = json.loads(Path(librispeech, "meta.json").read_text(encoding="utf-8"))
        for spk, files in meta["kept"].items():
            for f in files:
                clips.append((f"librispeech:{f}", f"librispeech:{spk}", load_clip(Path(librispeech, f))))
    return clips


def eer(same: list[float], diff: list[float]) -> float:
    """Equal error rate: the threshold where as many same-speaker pairs fall below as different ones reach it."""
    best = 1.0
    for t in sorted(set(same) | set(diff)):
        miss = float(np.mean(np.array(same) < t))
        false = float(np.mean(np.array(diff) >= t))
        best = min(best, max(miss, false))
    return round(best, 3)


def pair_stats(vectors: dict[int, np.ndarray], clips: list[tuple[str, str, np.ndarray]], group: str) -> dict[str, Any]:
    from bolo import similarity

    def kind(source: str) -> str:
        return "librispeech" if source.startswith("librispeech:") else "synthetic"

    keys = [i for i, c in enumerate(clips) if kind(c[0]) == group]
    same, diff = [], []
    for a, b in itertools.combinations(keys, 2):
        (same if clips[a][1] == clips[b][1] else diff).append(similarity.cosine(vectors[a], vectors[b]))
    if not same or not diff:
        return {}

    def stats(v: list[float]) -> dict[str, float]:
        x = np.array(v)
        return {
            "pairs": len(v), "min": round(float(x.min()), 3), "median": round(float(np.median(x)), 3),
            "p95": round(float(np.percentile(x, 95)), 3), "max": round(float(x.max()), 3),
        }  # fmt: skip

    return {
        "same_speaker": stats(same),
        "different_speaker": stats(diff),
        "eer": eer(same, diff),
        "auc": round(auc(same, diff), 4),
        "same_at_or_above_0.5": round(float(np.mean(np.array(same) >= 0.5)), 3),
        "different_at_or_above_0.5": round(float(np.mean(np.array(diff) >= 0.5)), 3),
    }


async def measure_embed(
    core: Bolo, names: list[str], remove: bool, voice_model: str, librispeech: str | None
) -> dict[str, Any]:
    from bolo import similarity

    clips = await embed_set(core, voice_model, librispeech)
    await core.manager.unload(voice_model)
    total_s = sum(len(c[2]) for c in clips) / SR
    out: dict[str, Any] = {}
    for name in names:
        spec = similarity.embedder(core, name)
        if not core.store.is_installed(spec.id):
            await core.pull_and_wait(spec.id)
        await similarity.embed(core, clips[0][2], SR, spec.id)  # loads the model
        loaded = core.manager.loaded.get(spec.id)
        vectors: dict[int, np.ndarray] = {}
        started = time.perf_counter()
        for i, (_src, _spk, audio) in enumerate(clips):
            vectors[i] = await similarity.embed(core, audio, SR, spec.id)
        took = time.perf_counter() - started
        result: dict[str, Any] = {
            "dimensions": len(vectors[0]),
            "rtf": round(took / total_s, 4),
            "load_s": round(loaded.load_ms / 1000, 2) if loaded else None,
            "memory_mb": round(loaded.memory_bytes / 1e6) if loaded else None,
            "repository clips and Kokoro voices": pair_stats(vectors, clips, "synthetic"),
            "LibriSpeech dev-clean": pair_stats(vectors, clips, "librispeech") if librispeech else None,
        }
        print(spec.id, json.dumps(result)[:500], file=sys.stderr)
        out[spec.id] = result
        await core.manager.unload(spec.id)
        if remove:
            await core.remove(spec.id)
    return {
        "clips": {
            "repository": EMBED_CLIPS,
            "kokoro": f"{voice_model}, voices {EMBED_VOICES}, three sentences each",
            "librispeech": "LibriSpeech dev-clean (CC BY 4.0): 16 speakers, 5 utterances of 4 to 15 s each, as picked by "
            "tools/calibrate_similarity.py"
            if librispeech
            else None,
        },
        "audio_s": round(total_s, 1),
        "models": out,
    }


def _rss() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["vad", "turn", "embed"])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove", action="store_true", help="delete each model after measuring it")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--voice-model", default="kokoro-v1.0-int8", help="installed voice for the synthesised test speech")
    ap.add_argument("--librispeech", help="embed: a folder written by tools/calibrate_similarity.py pick")
    args = ap.parse_args()
    settings = load_settings()
    settings.threads = args.threads
    core = Bolo(settings)
    await core.start()
    try:
        if args.what == "vad":
            result = await measure_vad(core, args.models, args.remove)
        elif args.what == "turn":
            result = await measure_turn(core, args.models, args.remove, args.voice_model)
        else:
            result = await measure_embed(core, args.models, args.remove, args.voice_model, args.librispeech)
    finally:
        await core.stop()
    import importlib.metadata as md

    versions = {}
    for pkg in ("sherpa-onnx", "onnxruntime", "webrtcvad-wheels", "tokenizers"):
        with contextlib.suppress(md.PackageNotFoundError):
            versions[pkg] = md.version(pkg)
    result = {
        "what": args.what,
        "hardware": hardware(),
        "python": platform.python_version(),
        "packages": versions,
        "date": time.strftime("%Y-%m-%d"),
        "load_average": round(os.getloadavg()[0], 1),
        **result,
    }
    path = Path(args.out)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if previous.get("models"):
        result["models"] = {**previous["models"], **result["models"]}
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
