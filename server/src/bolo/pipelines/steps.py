"""What each step type does (registered into :data:`bolo.pipelines.runner.HANDLERS`). Every model step calls the
core operation its endpoint calls; none has engine logic of its own."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np

from bolo.errors import EngineUnavailable, InvalidRequest, LLMError
from bolo.obs.trace import Trace
from bolo.pipelines.runner import (
    Audio,
    Run,
    Runner,
    Scope,
    StopRun,
    as_text,
    as_transcript,
    flag,
    handler,
    integer,
    label_by_turns,
    llm_complete,
    need_audio,
    need_text,
    number,
    parse_json_object,
    plain,
    text_list,
    transcript_outputs,
    wav_part,
)

Inputs = dict[str, Any]
Options = dict[str, Any]
Entry = dict[str, Any]


def _trace(kind: str, model: str | None) -> Trace:
    return Trace(kind, "pipeline", model or "")


def _opt(r: Runner, run: Run, options: Options, name: str, scope: Scope) -> Any:
    value = options.get(name)
    return None if value is None else r.resolve(run, value, scope)


def _record(core: Any, trace: Trace, status: str = "ok", error: str | None = None) -> None:
    trace.finish(status, error)
    core.traces.record(trace)


# -- audio ----------------------------------------------------------------------------------------------------
@handler("denoise")
async def _denoise(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, depth: int) -> dict[str, Any]:
    from bolo.denoise import enhance, mix

    audio = need_audio(i)
    model = r.core.denoiser_before_stt(e["model"]).id
    strength = number(_opt(r, run, o, "strength", scope), "strength")
    if strength is not None and not 0.0 < strength <= 1.0:
        raise InvalidRequest("strength must be above 0 and at most 1")
    trace = _trace("denoise", model)
    trace.audio_seconds = audio.seconds
    try:
        enhanced, _ = await enhance(r.core, audio.samples, audio.rate, model, trace)
    except Exception as exc:
        _record(r.core, trace, "error", str(exc))
        raise
    _record(r.core, trace)
    return {"audio": Audio(mix(audio.samples, enhanced, 1.0 if strength is None else strength), audio.rate)}


@handler("vad")
async def _vad(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, depth: int) -> dict[str, Any]:
    from bolo.audio.resample import resample
    from bolo.engines.base import VadParams
    from bolo.streaming.vad import detect_segments

    audio = need_audio(i)

    def ms(name: str, default: float) -> float:
        v = number(_opt(r, run, o, name, scope), name)
        return default if v is None else v

    params = VadParams(
        threshold=number(_opt(r, run, o, "threshold", scope), "threshold") or 0.5,
        min_silence_s=ms("min_silence_ms", 500.0) / 1000.0,
        min_speech_s=max(ms("min_speech_ms", 0.0) / 1000.0, 0.03),
        max_speech_s=3600.0,
        speech_pad_s=ms("speech_pad_ms", 0.0) / 1000.0,
    )
    samples16 = await asyncio.to_thread(resample, audio.samples, audio.rate, 16000)
    detector, name = await r.core.vad_detector(params, e["model"])
    found = await asyncio.to_thread(detect_segments, detector, samples16)
    segments = [{"start": round(a / 16000, 3), "end": round(b / 16000, 3)} for a, b in found]
    return {"segments": segments, "speech_seconds": round(sum(s["end"] - s["start"] for s in segments), 3),
            "count": len(segments), "_model": name}  # fmt: skip


def _turns(turns: Any, label: Any) -> list[dict[str, Any]]:
    return [{"speaker": label(t.speaker), "start": round(t.start, 3), "end": round(t.end, 3)} for t in turns]


@handler("diarization")
async def _diarization(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.diarize import DiarizeOptions

    audio = need_audio(i)
    opts = DiarizeOptions(
        model=e["model"],
        num_speakers=integer(_opt(r, run, o, "num_speakers", scope), "num_speakers"),
        min_speakers=integer(_opt(r, run, o, "min_speakers", scope), "min_speakers"),
        max_speakers=integer(_opt(r, run, o, "max_speakers", scope), "max_speakers"),
        threshold=number(_opt(r, run, o, "threshold", scope), "threshold"),
    )
    turns = await r.core.diarize(audio.samples, audio.rate, opts, _trace("diarization", e["model"]))
    out = _turns(turns, opts.label)
    speakers = list(dict.fromkeys(t["speaker"] for t in out))
    return {"turns": out, "speakers": speakers, "num_speakers": len(speakers)}


@handler("separation")
async def _separation(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.diarize import DiarizeOptions
    from bolo.separate import SeparateOptions

    audio = need_audio(i)
    model = r.core.resolve(e["model"], "separation").id
    opts = DiarizeOptions(model=_opt(r, run, o, "diarization_model", scope) or None)
    separate = SeparateOptions(
        model=model,
        min_overlap_s=number(_opt(r, run, o, "min_overlap_s", scope), "min_overlap_s"),
        allow_narrowband=bool(flag(_opt(r, run, o, "allow_narrowband", scope))),
        return_audio=True,
    )
    turns = await r.core.diarize(audio.samples, audio.rate, opts, _trace("diarization", opts.model), separate=separate)
    result = separate.result.to_dict(opts.label, audio.seconds, audio=True) if separate.result is not None else {}
    return {"turns": _turns(turns, opts.label), "separation": result}


@handler("speaker-identification")
async def _identify(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo import speakers

    audio = need_audio(i)
    model = speakers.embedder_id(r.core, e["model"])
    names = text_list(_opt(r, run, o, "speakers", scope))
    candidates = [r.core.speakers.find(n) for n in names] if names else r.core.speakers.all()
    body = await speakers.identify(r.core, audio.samples, audio.rate, candidates, model,
                                   number(_opt(r, run, o, "threshold", scope), "threshold"))  # fmt: skip
    best = body["speaker"]
    return {"speaker": best["name"] if best else "", "speaker_id": best["id"] if best else "",
            "similarity": body["similarity"], "match": best is not None, "_model": model}  # fmt: skip


@handler("speaker-enrolment")
async def _enrol(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo import speakers
    from bolo.speakers import SpeakerNotFound

    audio = need_audio(i)
    name = need_text(i, "name").strip()
    now = time_iso()
    clip = speakers.prepare_clip(audio.samples, audio.rate)
    try:
        speaker = r.core.speakers.find(name)
    except SpeakerNotFound:
        speaker = r.core.speakers.create(name, None, None, now)
    r.core.speakers.add_clip(speaker, clip, now)
    speaker = r.core.speakers.get(speaker.id)
    return {"speaker_id": speaker.id, "clips": len(speaker.clips())}


def time_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -- speech to text ---------------------------------------------------------------------------------------------
@handler("stt")
async def _stt(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.diarize import DiarizeOptions

    audio = need_audio(i)
    diarize = DiarizeOptions() if flag(_opt(r, run, o, "diarize", scope)) else None
    t = await r.core.transcribe(
        audio.samples, audio.rate,
        model=e["model"],
        language=_opt(r, run, o, "language", scope) or None,
        prompt=_opt(r, run, o, "prompt", scope) or None,
        hotwords=_opt(r, run, o, "hotwords", scope) or None,
        temperature=number(_opt(r, run, o, "temperature", scope), "temperature") or 0.0,
        vad_filter=flag(_opt(r, run, o, "vad_filter", scope)),
        word_timestamps=bool(flag(_opt(r, run, o, "word_timestamps", scope))),
        diarize=diarize,
        trace=_trace("stt", e["model"]),
    )  # fmt: skip
    turns = i.get("turns")
    if isinstance(turns, list) and turns:
        label_by_turns(t, turns)
    return {**transcript_outputs(t), "_model": t.model}


@handler("translation")
async def _translation(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    target = str(_opt(r, run, o, "target_language", scope) or "en")
    source = _opt(r, run, o, "source_language", scope) or None
    text = i.get("text")
    explicit_audio = "audio" in e["inputs"]
    if text not in (None, "") and not explicit_audio:
        from bolo.text.iso639 import name as language_name

        content = as_text(text)
        target_name = language_name(target) or target
        system = (
            f"Translate the user's text into {target_name}"
            + (f" from {language_name(source) or source}" if source else "")
            + ". Answer with the translation only, keeping speaker labels and line breaks."
        )
        out = await llm_complete(r.core, [{"role": "system", "content": system}, {"role": "user", "content": content}])
        return {"text": out.strip(), "language": target, "_model": r.core.settings.llm_model or "llm"}
    audio = need_audio(i)
    t = await r.core.transcribe(audio.samples, audio.rate, model=e["model"], language=source, task="translate",
                                target_language=target, trace=_trace("stt", e["model"]))  # fmt: skip
    return {"text": t.text, "language": target, "_model": t.model}


@handler("align")
async def _align(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    audio = need_audio(i)
    text = need_text(i)
    result = await r.core.align(audio.samples, audio.rate, text, language=_opt(r, run, o, "language", scope) or None,
                                model=e["model"])  # fmt: skip
    words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in result.words]
    return {"words": words, "text": result.text}


# -- text models ------------------------------------------------------------------------------------------------
@handler("punctuation")
async def _punctuation(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.punctuation import PunctuationOptions, apply

    t = as_transcript(i.get("text"))
    model = r.core.resolve(e["model"], "punctuation").id
    trace = _trace("punctuation", model)
    await apply(r.core, t, PunctuationOptions("on", model), trace)
    record = trace.extra.get("punctuation") or {}
    if record.get("status") == "skipped":
        raise EngineUnavailable(f"punctuation with {model} did not run: {record.get('reason')}")
    return {"text": t.text, "transcript": t}


@handler("itn")
async def _itn(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo import itn

    t = as_transcript(i.get("text"))
    opts = itn.ItnOptions(True, e["model"])
    spec = itn.choose(r.core, opts, t.language)
    trace = _trace("itn", spec.id)
    await itn.apply(r.core, t, spec, opts, trace)
    record = trace.extra.get("itn") or {}
    if record.get("status") == "skipped":
        raise EngineUnavailable(f"itn with {spec.id} did not run: {record.get('reason')}")
    return {"text": t.text, "transcript": t, "_model": spec.id}


@handler("pii")
async def _pii(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.redact import RedactOptions, apply, parse_entities

    t = as_transcript(i.get("text"))
    spec = r.core.resolve(e["model"], "pii")
    entities = parse_entities(_opt(r, run, o, "entities", scope) or None, "entities")
    await apply(r.core, t, RedactOptions(True, spec.id, entities), spec, _trace("pii", spec.id))
    redactions = t.redactions or []
    return {"text": t.text, "transcript": t, "redactions": redactions, "count": len(redactions)}


# -- audio analysis ---------------------------------------------------------------------------------------------
@handler("lid")
async def _lid(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    audio = need_audio(i)
    result = await r.core.identify_language(audio.samples, audio.rate, e["model"])
    return {"language": result.language, "probability": round(result.probability, 4), "languages": result.top(5),
            "_model": result.model}  # fmt: skip


async def _classify(r: Runner, e: Entry, i: Inputs, task: str) -> Any:
    from bolo.classify import classify

    audio = need_audio(i)
    model = r.core.resolve(e["model"], task).id
    trace = _trace(task, model)
    try:
        result = await classify(r.core, audio.samples, audio.rate, model, task, trace)
    except Exception as exc:
        _record(r.core, trace, "error", str(exc))
        raise
    _record(r.core, trace)
    return result


@handler("audio-tagging")
async def _tags(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.classify import tags_body

    result = await _classify(r, e, i, "audio-tagging")
    threshold = number(_opt(r, run, o, "threshold", scope), "threshold")
    body = tags_body(result, integer(_opt(r, run, o, "top_k", scope), "top_k") or 10,
                     0.1 if threshold is None else threshold, text_list(_opt(r, run, o, "labels", scope)) or None)  # fmt: skip
    tags = body["tags"]
    return {"tags": tags, "top": tags[0]["label"] if tags else ""}


@handler("classify")
async def _zero_shot(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.classifiers import TASK, body, classify_request

    audio = need_audio(i)
    classifier = as_text(_opt(r, run, o, "classifier", scope) or "").strip() or None
    prompt = _opt(r, run, o, "prompt", scope)
    trace = _trace(TASK, classifier or e["model"])
    try:
        result = await classify_request(
            r.core, audio.samples, audio.rate, trace, model=e["model"] or None, classifier=classifier,
            labels=text_list(_opt(r, run, o, "labels", scope)), prompt=None if prompt is None else as_text(prompt),
        )  # fmt: skip
    except Exception as exc:
        _record(r.core, trace, "error", str(exc))
        raise
    _record(r.core, trace)
    out = body(result)
    return {"label": out["label"] or "", "score": out["score"], "scores": out["scores"],
            "_model": result.classifier or result.model}  # fmt: skip


@handler("emotion")
async def _emotion(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.classify import emotion_body

    body = emotion_body(await _classify(r, e, i, "emotion"), 1000)
    return {"emotion": body["emotion"], "score": body["score"], "emotions": body["emotions"]}


@handler("intent")
async def _intent(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.intent import understand

    audio = need_audio(i)
    model = r.core.resolve(e["model"], "intent").id
    trace = _trace("intent", model)
    try:
        result, _ = await understand(r.core, audio.samples, audio.rate, model, trace)
    except Exception as exc:
        _record(r.core, trace, "error", str(exc))
        raise
    _record(r.core, trace)
    return {"intent": result.intent or "", "scenario": result.scenario or "", "action": result.action or "",
            "slots": [dict(s) for s in result.slots], "text": result.text}  # fmt: skip


@handler("audio-llm")
async def _audio_llm(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.audio_llm import client_for

    audio = need_audio(i)
    prompt = need_text(i, "prompt")
    client = client_for(r.core, e["model"] or "audio-llm")
    if client is None:
        raise LLMError("no audio language model is configured: set audio_llm_model (or give a catalog audio-llm model)")
    content = [{"type": "text", "text": prompt}, await asyncio.to_thread(wav_part, audio)]
    model = e["model"] or getattr(client, "model", None)
    text = await llm_complete(r.core, [{"role": "user", "content": content}], model=model, client=client,
                              max_tokens=integer(_opt(r, run, o, "max_tokens", scope), "max_tokens"),
                              temperature=number(_opt(r, run, o, "temperature", scope), "temperature"))  # fmt: skip
    return {"text": text.strip(), "_model": model or "audio-llm"}


# -- decisions --------------------------------------------------------------------------------------------------
def _choose(answer: Any, choices: list[str]) -> str | None:
    said = as_text(answer).strip().strip(".").lower()
    for c in choices:
        if said == c.lower():
            return c
    for c in choices:  # "Yes, it was read" -> yes
        if said.startswith(c.lower()):
            return c
    return None


def _decision_model(r: Runner, wanted: str | None) -> str | None:
    """The decision model a step uses: the one named; for ``auto``, ``default_decision`` when it is installed (else
    None: the language model answers); ``llm``: always the language model."""
    if wanted == "llm":
        return None
    if wanted:
        return r.core.resolve(wanted, "decision").id
    default = r.core.settings.default_decision
    if not default:
        return None
    try:
        spec = r.core.resolve(default, "decision")
        return spec.id if spec.builtin or r.core.manager.is_installed(spec) else None
    except Exception:
        return None


async def _decide_with_model(
    r: Runner, model: str, question: str, choices: list[str], text: Any, audio: Audio | None
) -> dict[str, Any]:
    """One question to a decision model (bolo.decide): yes/no, or one of the choices, with its probability."""
    from bolo.decide import decide

    yes_no = [c.lower() for c in choices] == ["yes", "no"]
    q: dict[str, Any] = {"type": "noul", "instructions": question}
    if not yes_no:
        q = {"type": "choice", "instructions": question, "criteria": dict.fromkeys(choices)}
    content = as_text(text).strip() or None
    trace = _trace("decision", model)
    result = await decide(r.core, model, content, audio.samples if audio else None, audio.rate if audio else None,
                          {"q": q}, trace)  # fmt: skip
    _record(r.core, trace)
    a = result.answers["q"]
    if yes_no:
        p = float(a["probability"])
        return {"answer": "yes" if a["answer"] else "no", "confidence": round(max(p, 1 - p), 4), "reason": "",
                "_model": model}  # fmt: skip
    return {"answer": str(a["answer"]), "confidence": float(a["confidence"]), "reason": "", "_model": model}


@handler("decisions")
async def _decisions(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    """Several questions in one pass of a decision model (bolo.decide): ``answers`` in its shape, and the probability
    of each answer given (yes/no: of the answer, a choice: of the option, a score: of the nearest level)."""
    from bolo.decide import decide, parse_questions

    questions = parse_questions(r.resolve(run, o.get("questions"), scope))
    model = r.core.resolve(e["model"] or None, "decision").id
    text = as_text(i.get("text")).strip() or None
    audio = i.get("audio") if isinstance(i.get("audio"), Audio) else None
    trace = _trace("decision", model)
    result = await decide(r.core, model, text, audio.samples if audio else None, audio.rate if audio else None,
                          questions, trace)  # fmt: skip
    _record(r.core, trace)
    return {"answers": result.answers, "probabilities": answer_probabilities(result.answers), "model": result.model}


def answer_probabilities(answers: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The probability of each answer given, by question (bolo.decide's answers)."""
    out: dict[str, float] = {}
    for qid, a in answers.items():
        if a["type"] == "yes_no":
            p = float(a["probability"])
            out[qid] = p if a["answer"] else round(1.0 - p, 4)
        elif a["type"] == "choice":
            out[qid] = float(a["probabilities"].get(a["answer"], 0.0))
        else:
            probs = a.get("probabilities") or []
            out[qid] = float(probs[a["level"]]) if len(probs) > a["level"] else 0.0
    return out


@handler("decision")
async def _decision(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    question = as_text(_opt(r, run, o, "question", scope)).strip()
    if not question:
        raise InvalidRequest("the decision's question is empty")
    choices = text_list(_opt(r, run, o, "choices", scope)) or ["yes", "no"]
    model = _decision_model(r, e["model"])
    if model is not None:
        text = i.get("text")
        heard = i.get("audio") if "audio" in e["inputs"] or text in (None, "") else None
        return await _decide_with_model(r, model, question, choices, text, heard if isinstance(heard, Audio) else None)
    instructions = as_text(_opt(r, run, o, "instructions", scope)).strip()
    text = i.get("text")
    audio = i.get("audio") if "audio" in e["inputs"] or text in (None, "") else None
    system = (
        "You answer one question about the " + ("recording" if audio is not None else "text") + " you are given. "
        f"Answer with one of: {', '.join(choices)}. Reply with a JSON object only: "
        '{"answer": "<one of the choices>", "confidence": <0 to 1>, "reason": "<one short sentence>"}.'
        + (f" {instructions}" if instructions else "")
    )
    if audio is not None:
        from bolo.audio_llm import client_for

        audio = need_audio(i)
        client = client_for(r.core, e["model"] or "audio-llm")
        if client is None:
            raise LLMError("a decision about a recording needs an audio language model (audio_llm_model)")
        user: Any = [{"type": "text", "text": f"Question: {question}"}, await asyncio.to_thread(wav_part, audio)]
        reply = await llm_complete(r.core, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   client=client, model=getattr(client, "model", None), json_mode=True)  # fmt: skip
        used = getattr(client, "model", None) or "audio-llm"
    else:
        content = need_text(i)
        user = f"Question: {question}\n\nText:\n{content}"
        reply = await llm_complete(r.core, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   json_mode=True)  # fmt: skip
        used = r.core.settings.llm_model or "llm"
    doc = parse_json_object(reply)
    answer = _choose(doc.get("answer") if doc else reply, choices)
    if answer is None:
        raise LLMError(f"the model's answer is not one of {', '.join(choices)}: {reply[:200]!r}")
    confidence = number((doc or {}).get("confidence"), "confidence") if doc else None
    return {"answer": answer, "confidence": 1.0 if confidence is None else max(0.0, min(1.0, confidence)),
            "reason": as_text((doc or {}).get("reason")).strip(), "_model": used}  # fmt: skip


# -- speech out -------------------------------------------------------------------------------------------------
@handler("tts")
async def _tts(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    text = need_text(i)
    watermark = flag(_opt(r, run, o, "watermark", scope))
    trace = _trace("tts", e["model"])
    samples, rate = await r.core.synthesize_all(
        text,
        model=e["model"],
        voice=_opt(r, run, o, "voice", scope) or None,
        speed=number(_opt(r, run, o, "speed", scope), "speed") or 1.0,
        language=_opt(r, run, o, "language", scope) or None,
        instructions=_opt(r, run, o, "instructions", scope) or None,
        watermark=watermark,
        trace=trace,
    )
    r.core.complete_tts(trace)
    return {"audio": Audio(samples, rate), "_model": trace.model or e["model"]}


@handler("voice-conversion")
async def _convert(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.voice_conversion import convert_voice

    audio = need_audio(i)
    trace = _trace("voice-conversion", e["model"])
    out = await convert_voice(r.core, audio.samples, audio.rate, model=e["model"],
                              voice=as_text(_opt(r, run, o, "voice", scope)) or None, trace=trace)  # fmt: skip
    _record(r.core, trace)
    return {"audio": Audio(out.samples, out.sample_rate), "_model": out.model}


@handler("watermark")
async def _watermark(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.watermark import marker

    audio = need_audio(i)
    model = r.core.resolve(e["model"] or r.core.settings.watermark_model, "watermark").id
    async with marker(r.core, model) as mark:
        if mark is None or not mark.active:
            raise EngineUnavailable(f"the watermark model '{model}' cannot run: {getattr(mark, 'reason', '')}")
        marked = await mark.mark(audio.samples, audio.rate)
    return {"audio": Audio(marked, audio.rate), "_model": model}


@handler("watermark-detect")
async def _detect(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.watermark import detect

    audio = need_audio(i)
    model = r.core.resolve(e["model"] or r.core.settings.watermark_model, "watermark").id
    found = await detect(r.core, audio.samples, audio.rate, model)
    return {"detected": found.marked, "score": round(found.score, 4), "_model": model}


# -- text and control -------------------------------------------------------------------------------------------
@handler("template")
async def _template(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    return {"text": r.text(run, o.get("template"), scope)}


@handler("llm")
async def _llm(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    prompt = r.text(run, o.get("prompt"), scope)
    if not prompt.strip():
        raise InvalidRequest("the prompt is empty")
    messages: list[dict[str, Any]] = []
    system = r.text(run, o.get("system"), scope) if o.get("system") else ""
    want_json = bool(flag(_opt(r, run, o, "json", scope)))
    if want_json:
        system = (system + "\n" if system else "") + "Reply with a JSON object only."
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = await llm_complete(r.core, messages, model=e["model"], json_mode=want_json,
                              max_tokens=integer(_opt(r, run, o, "max_tokens", scope), "max_tokens"),
                              temperature=number(_opt(r, run, o, "temperature", scope), "temperature"))  # fmt: skip
    doc = parse_json_object(text) if want_json else None
    if want_json and doc is None:
        raise LLMError(f"the language model did not answer with a JSON object: {text[:200]!r}")
    return {"text": text.strip(), "json": doc or {}, "_model": e["model"] or r.core.settings.llm_model or "llm"}


@handler("filter")
async def _filter(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    items = i.get("items")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise InvalidRequest("filter takes a list")
    kept = [item for item in items if r.holds(run, o["where"], scope.child({"item": item}))]
    return {"items": kept, "count": len(kept)}


@handler("condition")
async def _condition(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    taken = "then" if r.holds(run, o["if"], scope) else "else"
    other = "else" if taken == "then" else "then"
    for nested in e.get(other) or []:
        r._mark(run, nested, "skipped", d + 1, scope)
        _empty(scope, nested)
    await r._steps(run, o.get(taken) or [], e.get(taken) or [], scope, d + 1)
    return {"branch": taken}


def _empty(scope: Scope, entry: Entry) -> None:
    scope.values[entry["id"]] = {}
    for key in ("then", "else", "default"):
        for nested in entry.get(key) or []:
            _empty(scope, nested)
    for nested_list in (entry.get("cases") or {}).values():
        for nested in nested_list:
            _empty(scope, nested)


@handler("switch")
async def _switch(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    value = as_text(i.get("value")).strip().lower()
    cases: dict[str, Any] = o.get("cases") or {}
    taken = next((c for c in cases if str(c).strip().lower() == value), None)
    branches = [(c, cases[c], (e.get("cases") or {}).get(c, [])) for c in cases]
    branches.append(("default", o.get("default") or [], e.get("default") or []))
    for name, _, entries in branches:
        if name != (taken or "default"):
            for nested in entries:
                r._mark(run, nested, "skipped", d + 1, scope)
                _empty(scope, nested)
    for name, raws, entries in branches:
        if name == (taken or "default"):
            await r._steps(run, raws, entries, scope, d + 1)
    return {"branch": taken or "default"}


@handler("split-speakers")
async def _split(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    audio = need_audio(i)
    turns = i.get("turns")
    if not isinstance(turns, list) or not turns:
        raise InvalidRequest("split-speakers needs speaker turns: add a diarization step before it")
    min_s = number(_opt(r, run, o, "min_seconds", scope), "min_seconds") or 0.0
    by: dict[str, list[dict[str, Any]]] = {}
    for t in turns:
        by.setdefault(str(t.get("speaker")), []).append(t)
    out = []
    for label, own in by.items():
        parts = [audio.samples[int(float(t["start"]) * audio.rate) : int(float(t["end"]) * audio.rate)] for t in own]
        samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        seconds = len(samples) / audio.rate
        if seconds < min_s or not len(samples):
            continue
        speaker = {"label": label, "audio": Audio(samples, audio.rate), "seconds": round(seconds, 3), "turns": own}
        child = scope.child({"speaker": speaker}, suffix=f"-{_safe(label)}", isolated=True)
        await r._steps(run, o.get("steps") or [], e.get("steps") or [], child, d + 1)
        outputs = {
            n["id"]: r.serialise(run, n["id"], child.values.get(n["id"]) or {}, child) for n in e.get("steps") or []
        }
        out.append({"speaker": label, "seconds": round(seconds, 3), "outputs": outputs})
    return {"speakers": out, "count": len(out)}


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label)[:24] or "speaker"


@handler("stop")
async def _stop(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    reason = r.text(run, o.get("reason"), scope) if o.get("reason") else f"stopped by step {e['id']}"
    scope.values[e["id"]] = {}
    raise StopRun(reason)


# -- actions ----------------------------------------------------------------------------------------------------
@handler("webhook")
async def _webhook(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    import httpx

    from bolo.pipelines import webhooks

    url = r.text(run, o.get("url"), scope).strip()
    if not url.startswith(("http://", "https://")):
        raise InvalidRequest(f"the webhook url must be an http:// or https:// address, not {url[:80]!r}")
    method = str(o.get("method") or "POST").upper()
    if "body" in e["inputs"]:
        body_value = plain(i.get("body"))
    else:
        body_value = {
            "run": run.id,
            "pipeline": run.pipeline,
            "step": e["id"],
            "steps": {k: _jsonable(r, run, k, v, scope) for k, v in scope.values.items() if v},
        }
    body = json.dumps(body_value, ensure_ascii=False).encode()
    headers = {str(k): r.text(run, v, scope) for k, v in (o.get("headers") or {}).items()}
    try:
        resp = await webhooks.send(r.core.settings, url, body, event="pipeline.step", run_id=run.id, method=method,
                                   headers=headers, timeout_s=number(o.get("timeout_s"), "timeout_s"))  # fmt: skip
    except httpx.HTTPError as exc:
        raise EngineUnavailable(f"the webhook {url} could not be reached: {type(exc).__name__}: {exc}") from exc
    text = resp.text[:20000]
    try:
        doc = resp.json()
    except ValueError:
        doc = None
    if resp.status_code >= 400:
        raise EngineUnavailable(f"the webhook {url} answered HTTP {resp.status_code}: {text[:200]}")
    return {"status": resp.status_code, "ok": True, "body": text, "json": doc if isinstance(doc, dict) else {}}


def _jsonable(r: Runner, run: Run, step_id: str, outputs: dict[str, Any], scope: Scope) -> Any:
    return {k: plain(v) for k, v in outputs.items() if k not in ("transcript", "_model")}


@handler("save")
async def _save(r: Runner, run: Run, e: Entry, i: Inputs, o: Options, scope: Scope, d: int) -> dict[str, Any]:
    from bolo.audio.encode import encode_audio
    from bolo.pipelines.files import check_name, extension
    from bolo.text.formats import to_srt, to_vtt

    name = check_name(r.text(run, o.get("name"), scope).strip())
    if scope.suffix:
        stem, ext = name.rsplit(".", 1)
        name = f"{stem}{scope.suffix}.{ext}"
    value = i.get("value")
    ext = extension(name)
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".opus"):
        if not isinstance(value, Audio):
            raise InvalidRequest(f"{name}: only audio is saved as {ext}")
        fmt = {".ogg": "opus"}.get(ext, ext[1:])
        data = await asyncio.to_thread(encode_audio, np.asarray(value.samples, dtype=np.float32), value.rate, fmt)
    elif ext in (".srt", ".vtt"):
        segments = _segments_of(value)
        data = (to_srt(segments) if ext == ".srt" else to_vtt(segments)).encode()
    elif ext == ".json":
        data = json.dumps(plain(value), ensure_ascii=False, indent=1).encode()
    else:
        data = as_text(value).encode()
    url = r.write_file(run, name, data)
    return {"file": name, "url": url}


def _segments_of(value: Any) -> list[Any]:
    from bolo.core import Transcript
    from bolo.engines.base import Segment

    if isinstance(value, Transcript):
        return value.segments
    if isinstance(value, list):
        return [Segment(float(s.get("start", 0)), float(s.get("end", 0)), str(s.get("text", "")),
                        speaker=s.get("speaker")) for s in value if isinstance(s, dict)]  # fmt: skip
    raise InvalidRequest("subtitles (.srt, .vtt) are made from a transcript or a list of segments")
