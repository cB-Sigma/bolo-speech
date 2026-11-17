#!/usr/bin/env python3
"""Download catalog models, run them on this machine and record what was measured.

This is how a catalog entry earns ``validated: cpu``: the model is pulled,
loaded and run through the same code paths the server uses, and the
real-time factor (processing time / audio duration, best of three runs
after a warm-up) is written to a JSON report. ``apply_validation.py`` then
copies the report into the catalog YAML.

Speech-to-text models are fed real speech from the ``test_wavs`` folder of
their own model repository when there is one (word error rate is reported
when a reference transcript is available), otherwise speech synthesised by
an already validated voice. Text-to-speech models synthesise a sentence in
their language, and English voices are transcribed back to check
intelligibility.

Usage:
    python tools/validate_models.py --out report.json [--keep] MODEL [MODEL ...]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bolo.audio import decode_audio
from bolo.config import load_settings
from bolo.core import Bolo
from bolo.errors import BoloError
from bolo.obs.trace import Trace
from bolo.text.normalize import word_error_rate

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog. Speech servers are judged by how quickly the first sound arrives.",
    "hi": "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।",
    "ne": "नमस्ते, तपाईंलाई कस्तो छ? आज मौसम राम्रो छ।",
    "ur": "السلام علیکم، آپ کیسے ہیں؟ آج موسم بہت اچھا ہے۔",
    "bn": "নমস্কার, আপনি কেমন আছেন? আজ আবহাওয়া খুব ভালো।",
    "gu": "નમસ્તે, તમે કેમ છો? આજે હવામાન ખૂબ સરસ છે.",
    "de": "Guten Morgen! Wie geht es Ihnen heute? Das Wetter ist sehr schön.",
    "es": "Hola, ¿cómo estás? Hoy hace muy buen tiempo.",
    "fr": "Bonjour, comment allez-vous aujourd'hui ? Il fait très beau.",
    "it": "Buongiorno, come stai oggi? Il tempo è molto bello.",
    "pt": "Olá, como você está hoje? O tempo está muito bom.",
    "ru": "Здравствуйте, как у вас дела? Сегодня очень хорошая погода.",
    "zh": "你好，你今天怎么样？今天天气很好。",
}


def hardware(hw: Any = None, settings: Any = None) -> str:
    """The machine, for the report: "16 vCPU x86_64 (AMD Ryzen 7 5800H with Radeon Graphics), no GPU". The CPU's name
    comes from bolo.hardware (on Windows from the registry, not /proc/cpuinfo); a GPU that was found is named, with
    the ``device`` the models ran on."""
    if hw is None:
        from bolo import hardware as hwmod

        hw = hwmod.detect(settings)
    cpu = hw.cpu
    cores = cpu.logical_cores or os.cpu_count()
    arch = cpu.architecture or platform.machine()
    text = f"{cores} vCPU {arch} ({cpu.model or platform.processor() or 'unknown CPU'})"
    gpus = [g.name or f"{g.vendor} GPU" for g in hw.gpus if not g.integrated]
    if not gpus:
        return f"{text}, no GPU"
    device = str(getattr(hw, "device", "") or "cpu")
    used = "not used" if device == "cpu" else f"device {device}"
    return f"{text}, GPU {', '.join(gpus)} ({used})"


def sample_speech(synth: dict[str, Any], lang: str = "en") -> tuple[np.ndarray, int, str, str]:
    """Speech made by an installed voice for the validators that need some: ``lang``'s, else any. A run without
    any (no ``--en-voice`` installed) cannot check those models and says how to fix it."""
    if lang in synth:
        return synth[lang]  # type: ignore[no-any-return]
    if synth:
        return next(iter(synth.values()))  # type: ignore[no-any-return]
    raise RuntimeError(
        "no test speech: no voice to make it is installed (pull the --en-voice model, kokoro-v1.0-int8 by default, "
        "or name an installed English voice with --en-voice)"
    )


def pick_en_voice(core: Bolo, wanted: str | None) -> str | None:
    """The English voice that makes test speech: ``wanted`` when it is installed, else the first installed English
    text-to-speech model, else None (the models that need speech are then skipped with a reason)."""
    if wanted:
        spec = core.catalog.resolve(wanted)
        if spec is not None and core.manager.is_installed(spec):
            return spec.id
    for spec in core.catalog.all():
        if spec.task == "tts" and "en" in spec.languages and not spec.cloud and core.manager.is_installed(spec):
            return spec.id
    return None


def fetch_test_wavs(core: Bolo, repo: str, revision: str, cache: Path) -> list[tuple[Path, str | None]]:
    """Download up to two test wavs (and their transcript, if any) from a model repository."""
    try:
        files = core.store.hub.list_files(repo, revision)
    except Exception:
        return []
    wavs = [
        f for f in files if f.path.startswith("test_wavs/") and f.path.endswith(".wav") and (f.size or 0) < 2_000_000
    ]
    wavs = sorted(wavs, key=lambda f: f.path)[:2]
    trans: dict[str, str] = {}
    for f in files:
        if f.path in ("test_wavs/trans.txt", "test_wavs/transcript.txt"):
            text = httpx.get(f.url, follow_redirects=True, timeout=30).text
            for line in text.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    trans[parts[0].removesuffix(".wav")] = parts[1]
    out: list[tuple[Path, str | None]] = []
    for f in wavs:
        dest = cache / repo.replace("/", "--") / Path(f.path).name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(httpx.get(f.url, follow_redirects=True, timeout=60).content)
        out.append((dest, trans.get(dest.stem)))
    return out


async def validate_stt(core: Bolo, spec: Any, cache: Path, synth: dict[str, Any]) -> dict[str, Any]:
    clips: list[tuple[np.ndarray, int, str | None, str]] = []
    for path, ref in fetch_test_wavs(core, spec.source.repo, spec.source.revision, cache):
        samples, rate = decode_audio(path.read_bytes(), filename=path.name)
        if rate >= 16000:
            clips.append((samples, rate, ref, f"{spec.source.repo}/test_wavs/{path.name}"))
    if not clips:
        lang = spec.languages[0] if spec.languages else "en"
        for candidate in (lang, "en"):
            if candidate in synth and (candidate in spec.languages or len(spec.languages) > 10):
                audio, rate, text, voice = synth[candidate]
                clips.append((audio, rate, text, f"synthesised by {voice}"))
                break
    if not clips:
        raise RuntimeError("no test audio available for this model's language")
    language = spec.options.get("language") or (spec.languages[0] if len(spec.languages) == 1 else None)
    await core.transcribe(clips[0][0], clips[0][1], model=spec.id, language=language)  # warm-up (includes loading)
    best_rtf = float("inf")
    texts: list[str] = []
    wers: list[float] = []
    audio_s = sum(len(c[0]) / c[1] for c in clips)
    for run in range(3):
        elapsed = 0.0
        for samples, rate, ref, _src in clips:
            trace = Trace("stt", "bench", spec.id)
            result = await core.transcribe(samples, rate, model=spec.id, language=language, trace=trace)
            elapsed += (trace.total_ms or 0.0) / 1000.0
            if run == 0:
                texts.append(result.text)
                if ref:
                    wers.append(word_error_rate(ref, result.text, language or "en"))
        best_rtf = min(best_rtf, elapsed / audio_s)
    if not any(t.strip() for t in texts):
        raise RuntimeError("the model produced no text")
    out: dict[str, Any] = {
        "rtf": round(best_rtf, 4),
        "audio_s": round(audio_s, 1),
        "text": texts[0][:120],
        "source": clips[0][3],
    }
    if wers:
        out["wer"] = round(sum(wers) / len(wers), 3)
    return out


async def validate_tts(core: Bolo, spec: Any, stt_model: str | None) -> dict[str, Any]:
    lang = spec.languages[0] if spec.languages else "en"
    text = SENTENCES.get(lang)
    if text is None:
        raise RuntimeError(f"no validation sentence for language '{lang}'")
    await core.synthesize_all(text, model=spec.id)  # warm-up (includes loading)
    best_rtf, best_ttfa = float("inf"), float("inf")
    audio = np.zeros(0, dtype=np.float32)
    rate = spec.sample_rate
    for _ in range(3):
        trace = Trace("tts", "bench", spec.id)
        parts = []
        try:
            async for chunk in core.synthesize(text, model=spec.id, trace=trace):
                trace.mark_ttfa()
                parts.append(chunk.samples)
                rate = chunk.sample_rate
        finally:
            core.complete_tts(trace)
        audio = np.concatenate(parts)
        best_rtf = min(best_rtf, trace.rtf or float("inf"))
        best_ttfa = min(best_ttfa, trace.ttfa_ms or float("inf"))
    seconds = len(audio) / rate
    if seconds < 0.5 or float(np.abs(audio).max()) < 0.01:
        raise RuntimeError("the voice produced no audible audio")
    out: dict[str, Any] = {
        "rtf": round(best_rtf, 4),
        "ttfa_ms": round(best_ttfa),
        "audio_s": round(seconds, 1),
        "sample_rate": rate,
    }
    if stt_model and lang == "en":
        heard = await core.transcribe(audio, rate, model=stt_model, language="en")
        out["round_trip_wer"] = round(word_error_rate(text, heard.text, "en"), 3)
        out["round_trip_stt"] = stt_model
    return out


async def validate_vad(core: Bolo, spec: Any, synth: dict[str, Any]) -> dict[str, Any]:
    from bolo.audio.resample import resample
    from bolo.streaming.vad import detect_segments

    audio, rate, _text, _voice = sample_speech(synth)
    speech = resample(audio, rate, 16000)
    gap = np.zeros(16000, dtype=np.float32)
    clip = np.concatenate([gap, speech, gap, speech, gap])
    detector, name = await core.vad_detector(model=spec.id)
    if name != spec.id:
        raise RuntimeError(f"expected detector {spec.id}, got {name}")
    t0 = time.perf_counter()
    segments = detect_segments(detector, clip)
    elapsed = time.perf_counter() - t0
    if not 2 <= len(segments) <= 6:
        raise RuntimeError(f"expected speech in two regions, got {len(segments)} segments")
    return {
        "rtf": round(elapsed / (len(clip) / 16000), 5),
        "audio_s": round(len(clip) / 16000, 1),
        "segments": len(segments),
    }


async def validate_diarization(core: Bolo, spec: Any, synth: dict[str, Any]) -> dict[str, Any]:
    """Speech from the test voices (two languages when there are two: two voices) with pauses: speaker turns come
    back, and the real-time factor of the whole pipeline is recorded."""
    from bolo.audio.resample import resample
    from bolo.diarize import DiarizeOptions

    clips = [resample(audio, rate, 16000) for audio, rate, _text, _voice in synth.values()] or [
        resample(*sample_speech(synth)[:2], 16000)
    ]
    gap = np.zeros(16000, dtype=np.float32)
    parts: list[np.ndarray] = [gap]
    for clip in clips * 2:
        parts += [clip, gap]
    audio16 = np.concatenate(parts)
    await core.diarize(audio16, 16000, DiarizeOptions(model=spec.id))  # warm-up (includes loading)
    best = float("inf")
    turns: list[Any] = []
    for _ in range(3):
        t0 = time.perf_counter()
        turns = await core.diarize(audio16, 16000, DiarizeOptions(model=spec.id))
        best = min(best, time.perf_counter() - t0)
    if not turns:
        raise RuntimeError("the model found no speaker turns in speech")
    seconds = len(audio16) / 16000
    return {
        "rtf": round(best / seconds, 4),
        "audio_s": round(seconds, 1),
        "turns": len(turns),
        "speakers": len({t.speaker for t in turns}),
    }


async def validate_turn(core: Bolo, spec: Any, synth: dict[str, Any]) -> dict[str, Any]:
    """The end-of-turn model on a finished sentence: a probability comes back; its time per call is recorded."""
    from bolo.audio.resample import resample

    audio, rate, text, _voice = sample_speech(synth)
    speech = np.ascontiguousarray(resample(audio, rate, 16000)[-8 * 16000 :], dtype=np.float32)
    op, item = ("turn_text", text) if core.turn_reads_text(spec) else ("turn", speech)  # a model that reads text
    async with core.manager.use(spec.id) as entry:
        await entry.queue.submit(op, item)  # warm-up
        best = float("inf")
        probability = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            probability, _ = await entry.queue.submit(op, item)
            best = min(best, time.perf_counter() - t0)
    if not 0.0 <= float(probability) <= 1.0:
        raise RuntimeError(f"expected a probability, got {probability!r}")
    seconds = len(speech) / 16000
    return {
        "rtf": round(best / seconds, 5),
        "audio_s": round(seconds, 1),
        "ms": round(best * 1000, 1),
        "probability": round(float(probability), 3),
    }


async def validate_speaker_embedding(core: Bolo, spec: Any, synth: dict[str, Any]) -> dict[str, Any]:
    """Embeddings of the two halves of the test speech (one voice) and of the speech played 40 % faster and lower
    (another "voice"): unit vectors come back, the halves score higher than the changed voice."""
    from bolo import similarity
    from bolo.audio.resample import resample

    audio, rate, _text, _voice = sample_speech(synth)
    speech = np.ascontiguousarray(resample(audio, rate, 16000), dtype=np.float32)
    half = len(speech) // 2
    other = resample(speech, 16000, 11500)  # played at 16 kHz: slower and lower
    await similarity.embed(core, speech, 16000, spec.id)  # warm-up (includes loading)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        a = await similarity.embed(core, speech[:half], 16000, spec.id)
        b = await similarity.embed(core, speech[half:], 16000, spec.id)
        best = min(best, time.perf_counter() - t0)
    c = await similarity.embed(core, other, 16000, spec.id)
    same, changed = similarity.cosine(a, b), similarity.cosine(a, c)
    if not same > changed:
        raise RuntimeError(f"two halves of one voice scored {same:.3f}, below the changed voice ({changed:.3f})")
    return {
        "rtf": round(best / (len(speech) / 16000), 5),
        "audio_s": round(len(speech) / 16000, 1),
        "dimensions": len(a),
        "same_voice": round(same, 3),
        "changed_voice": round(changed, 3),
    }


#: which validator checks a model of each task
VALIDATORS = {
    "stt": "stt", "tts": "tts", "vad": "vad", "diarization": "diarization", "turn": "turn",
    "speaker-embedding": "speaker-embedding",
}  # fmt: skip


async def validate(
    core: Bolo, spec: Any, args: argparse.Namespace, cache: Path, synth: dict[str, Any]
) -> dict[str, Any]:
    kind = VALIDATORS.get(spec.task)
    if kind == "stt":
        return await validate_stt(core, spec, cache, synth)
    if kind == "tts":
        return await validate_tts(core, spec, args.stt)
    if kind == "vad":
        return await validate_vad(core, spec, synth)
    if kind == "diarization":
        return await validate_diarization(core, spec, synth)
    if kind == "turn":
        return await validate_turn(core, spec, synth)
    if kind == "speaker-embedding":
        return await validate_speaker_embedding(core, spec, synth)
    raise RuntimeError(f"no validation for {spec.task} models yet")


async def cleanup(core: Bolo, model_id: str) -> str | None:
    """Remove a model pulled for the run. A folder Windows will not let go (an antivirus scanner holding a file
    that was just written: WinError 32) is recorded and the run goes on; None when it was removed."""
    try:
        await core.remove(model_id)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"[:1000]
    except BoloError as exc:
        return exc.message[:1000]  # long: it names the folder that is left, and Windows paths are long
    return None


async def main_async(args: argparse.Namespace) -> None:
    settings = load_settings(overrides={"threads": args.threads} if args.threads else None)
    settings.stt_ttl_s = settings.tts_ttl_s = -1
    core = Bolo(settings)
    await core.start()
    out_path = Path(args.out)
    report: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    cache = settings.data_dir / "validation"
    synth: dict[str, Any] = {}
    hw = hardware(settings=settings)
    try:
        en_voice = pick_en_voice(core, args.en_voice)
        if en_voice is None:
            print("note: no English voice is installed: STT models without test wavs, VAD, diarization and turn models "
                  "are skipped (pull kokoro-v1.0-int8, or name one with --en-voice)", flush=True)  # fmt: skip
        elif en_voice != args.en_voice:
            print(f"note: --en-voice {args.en_voice} is not installed; test speech comes from {en_voice}", flush=True)
        for lang, voice_model in (("en", en_voice), ("hi", args.hi_voice)):
            if voice_model and core.manager.is_installed(core.manager.resolve(voice_model)):
                audio, rate = await core.synthesize_all(SENTENCES[lang], model=voice_model)
                synth[lang] = (audio, rate, SENTENCES[lang], voice_model)
                await core.manager.unload(voice_model)
        for name in args.models:
            spec = core.manager.resolve(name)
            if spec.id in report and report[spec.id].get("ok") and not args.force:
                continue
            free = shutil.disk_usage(settings.data_dir).free
            if free < spec.size_bytes + args.min_free_gb * 1e9:
                print(f"{spec.id}: skipped, only {free / 1e9:.1f} GB free", flush=True)
                continue
            was_installed = core.manager.is_installed(spec)
            entry: dict[str, Any] = {"ok": False}
            t0 = time.perf_counter()
            try:
                job = await core.pull_and_wait(spec.id)
                if job.state != "done":
                    raise RuntimeError(job.error or "pull failed")
                entry.update(await validate(core, spec, args, cache, synth))
                loaded = core.manager.loaded.get(spec.id)
                if loaded:
                    entry["load_s"] = round(loaded.load_ms / 1000.0, 2)
                    entry["memory_mb"] = round(loaded.memory_bytes / 1e6)
                entry.update(ok=True, threads=settings.resolved_threads(), hardware=hw, date=dt.date.today().isoformat(),
                             engine=f"{spec.engine} {core.engines.status(spec.engine).version}")  # fmt: skip
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
            entry["took_s"] = round(time.perf_counter() - t0, 1)
            for other in list(core.manager.loaded):
                if other != args.stt:
                    with contextlib.suppress(Exception):
                        await core.manager.unload(other)
            if not was_installed and not args.keep:
                problem = await cleanup(core, spec.id)
                if problem:
                    entry["cleanup_error"] = problem
            report[spec.id] = entry
            out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{spec.id}: {json.dumps(entry, ensure_ascii=False)}", flush=True)
    finally:
        await core.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--keep", action="store_true", help="keep models that were downloaded for the run")
    p.add_argument("--force", action="store_true", help="re-run models already in the report")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--stt", default="moonshine-tiny-en", help="installed STT model for the TTS round-trip check")
    p.add_argument(
        "--en-voice",
        default="kokoro-v1.0-int8",
        help="installed English voice used to make test speech (else the first installed English voice)",
    )
    p.add_argument("--hi-voice", default="piper-hi-in-rohan-medium")
    p.add_argument("--min-free-gb", type=float, default=2.0)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
