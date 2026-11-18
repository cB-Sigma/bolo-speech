"""Multi-speaker dialogue: turns of (speaker, text) spoken with a voice per speaker.

Two ways to speak one:

* **in one take** (``native``), by a model that speaks a whole script itself (``TTSModel.supports_dialogue``:
  VibeVoice on audio.cpp). Every speaker needs a recording of their voice: a saved voice, or one of bolo's
  reference clips (``en-short``, ``en-long``, ``hi-short``). Turn-taking, breaths and overlap come from the model.
* **stitched** (``stitched``), by any text-to-speech model: each turn is synthesised on its own in its speaker's
  voice (a voice of the model, a saved voice, a blend) and the turns are joined with a pause between them.

``mode="auto"`` (the default) speaks in one take when the model can and every speaker has a recording, and
stitches otherwise.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import FloatArray, SynthesisExtras, TTSModel
from bolo.errors import InvalidRequest
from bolo.obs.trace import Trace

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo

MODES = ("auto", "native", "stitched")
#: most turns in one request, and most characters over all of them
MAX_TURNS = 200
MAX_CHARS = 20000
#: bolo's reference clips that can stand for a speaker in a one-take dialogue, alternated by default
DEFAULT_REFERENCES = ("en-short", "en-long")

_SCRIPT_LINE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*(.+?)\s*$")


@dataclass
class Turn:
    speaker: str
    text: str
    #: direction for this turn only (stitched dialogues; a one-take dialogue takes the request's)
    instructions: str | None = None


@dataclass
class DialogueResult:
    samples: FloatArray
    sample_rate: int
    mode: str
    model: str
    speakers: list[str]
    #: per turn, where it starts and ends in the audio (stitched dialogues only)
    segments: list[dict[str, Any]] = field(default_factory=list)


def parse_script(script: str) -> list[Turn]:
    """``Name: line`` per line; a line without a name continues the previous turn."""
    turns: list[Turn] = []
    for raw in script.splitlines():
        if not raw.strip():
            continue
        m = _SCRIPT_LINE.match(raw)
        if m and not m.group(1).strip().startswith(("http", "[")):
            turns.append(Turn(m.group(1).strip(), m.group(2)))
        elif turns:
            turns[-1].text += " " + raw.strip()
        else:
            raise InvalidRequest("each line of a script starts with the speaker's name and a colon", param="script")
    return turns


def check_turns(turns: list[Turn]) -> list[str]:
    """The speakers, in order of first appearance; raises for an empty or oversized dialogue."""
    if not turns:
        raise InvalidRequest("a dialogue needs at least one turn", param="turns")
    if len(turns) > MAX_TURNS:
        raise InvalidRequest(f"a dialogue has at most {MAX_TURNS} turns", param="turns")
    if sum(len(t.text) for t in turns) > MAX_CHARS:
        raise InvalidRequest(f"a dialogue has at most {MAX_CHARS} characters", param="turns")
    speakers: list[str] = []
    for t in turns:
        if not t.speaker.strip():
            raise InvalidRequest("every turn names its speaker", param="turns")
        if not t.text.strip():
            raise InvalidRequest(f"a turn of {t.speaker!r} has no text", param="turns")
        if t.speaker not in speakers:
            speakers.append(t.speaker)
    return speakers


def _recording(core: Bolo, voice: str | None, index: int) -> tuple[bytes, str] | None:
    """A speaker's recording for a one-take dialogue: a saved voice or a reference clip; None when the voice is
    neither (then the dialogue is stitched)."""
    from bolo import samples
    from bolo.core import reference_wav

    if not voice:
        clip = DEFAULT_REFERENCES[index % len(DEFAULT_REFERENCES)]
        entry = samples.get(clip)
        return reference_wav(samples.read_file(str(entry["file"]))), str(entry.get("text", ""))
    saved = core.voices.find(voice)
    if saved is not None:
        return reference_wav(saved.reference_bytes()), saved.transcript or ""
    clips = {str(e["id"]): e for e in samples.index() if e.get("kind") == "audio"}
    found = clips.get(voice)
    if found is not None:
        return reference_wav(samples.read_file(str(found["file"]))), str(found.get("text", ""))
    return None


async def speak_dialogue(
    core: Bolo,
    turns: list[Turn],
    voices: dict[str, str],
    *,
    model: str | None = None,
    mode: str = "auto",
    pause_ms: int = 300,
    language: str | None = None,
    speed: float = 1.0,
    instructions: str | None = None,
    trace: Trace | None = None,
) -> DialogueResult:
    """Speak ``turns`` with ``voices`` (speaker -> voice). See the module docstring for ``mode``."""
    if mode not in MODES:
        raise InvalidRequest(f"mode must be one of {', '.join(MODES)}", param="mode")
    if not 0 <= pause_ms <= 5000:
        raise InvalidRequest("pause_ms must be between 0 and 5000", param="pause_ms")
    speakers = check_turns(turns)
    unknown = sorted(set(voices) - set(speakers))
    if unknown:
        raise InvalidRequest(f"voices names speakers with no turn: {', '.join(unknown)}", param="voices")
    spec = core.resolve(model, "tts")
    trace = trace or Trace("tts", "http", spec.id)
    trace.model = spec.id
    trace.chars = sum(len(t.text) for t in turns)
    trace.extra["dialogue"] = {"turns": len(turns), "speakers": len(speakers)}
    if mode != "stitched":
        result = await _native(core, spec, turns, speakers, voices, language, instructions, mode, trace)
        if result is not None:
            return result
    return await _stitched(core, spec.id, turns, speakers, voices, pause_ms, language, speed, instructions, trace)


async def _native(
    core: Bolo,
    spec: ModelSpec,
    turns: list[Turn],
    speakers: list[str],
    voices: dict[str, str],
    language: str | None,
    instructions: str | None,
    mode: str,
    trace: Trace,
) -> DialogueResult | None:
    """One take, or None (auto) when the model or a speaker's voice does not allow it."""
    model_id = spec.id
    if not spec.options.get("dialogue"):
        if mode == "native":
            raise InvalidRequest(f"model '{model_id}' does not speak dialogues in one take", param="mode")
        return None
    recordings: list[tuple[bytes, str]] = []
    for k, who in enumerate(speakers):
        rec = _recording(core, voices.get(who), k)
        if rec is None:
            if mode == "native":
                raise InvalidRequest(
                    f"a one-take dialogue needs a recording of every voice: {voices.get(who)!r} ({who}) is neither "
                    "a saved voice nor a reference clip (en-short, en-long, hi-short)",
                    param="voices",
                )
            return None
        recordings.append(rec)
    core.require_components(spec)
    async with core.manager.use(model_id) as entry:
        tts = entry.model
        assert isinstance(tts, TTSModel)
        if not tts.supports_dialogue or len(speakers) > tts.max_dialogue_speakers:
            if mode == "native":
                limit = tts.max_dialogue_speakers
                raise InvalidRequest(
                    f"model '{model_id}' speaks dialogues in one take only from a server on this machine"
                    if not tts.supports_dialogue
                    else f"model '{model_id}' takes at most {limit} speakers in one take",
                    param="mode",
                )
            return None
        extras = SynthesisExtras(language=language, instructions=(instructions or "").strip() or None)
        if extras.instructions:
            trace.extra["instructions"] = "applied" if tts.supports_instructions else "ignored"
        script = [(speakers.index(t.speaker), t.text) for t in turns]
        with trace.phase("synthesize"):
            parts = await asyncio.to_thread(lambda: list(tts.synthesize_dialogue(script, recordings, None, extras)))
        rate = tts.sample_rate
    audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    trace.extra["dialogue"]["mode"] = "native"
    return DialogueResult(audio, rate, "native", model_id, speakers)


async def _stitched(
    core: Bolo,
    model_id: str,
    turns: list[Turn],
    speakers: list[str],
    voices: dict[str, str],
    pause_ms: int,
    language: str | None,
    speed: float,
    instructions: str | None,
    trace: Trace,
) -> DialogueResult:
    pieces: list[FloatArray] = []
    segments: list[dict[str, Any]] = []
    rate = 0
    at = 0
    applied: set[str] = set()
    with trace.phase("synthesize"):
        for k, turn in enumerate(turns):
            sub = Trace("tts", "http", model_id)
            audio, turn_rate = await core.synthesize_all(
                turn.text, model=model_id, voice=voices.get(turn.speaker) or None, language=language, speed=speed,
                instructions=turn.instructions or instructions, detect_language=language is None, trace=sub,
            )  # fmt: skip
            if sub.extra.get("instructions"):
                applied.add(str(sub.extra["instructions"]))
            core.complete_tts(sub)
            if not rate:
                rate = turn_rate
            elif turn_rate != rate:
                audio = resample(audio, turn_rate, rate)
            if k and pause_ms:
                gap = np.zeros(int(rate * pause_ms / 1000), dtype=np.float32)
                pieces.append(gap)
                at += len(gap)
            segments.append(
                {
                    "turn": k,
                    "speaker": turn.speaker,
                    "start": round(at / rate, 3),
                    "end": round((at + len(audio)) / rate, 3),
                }
            )
            pieces.append(audio.astype(np.float32))
            at += len(audio)
    if applied:
        trace.extra["instructions"] = (
            "applied" if applied == {"applied"} else "ignored" if applied == {"ignored"} else "partly"
        )
    trace.extra["dialogue"]["mode"] = "stitched"
    out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return DialogueResult(out, rate or 24000, "stitched", model_id, speakers, segments)
