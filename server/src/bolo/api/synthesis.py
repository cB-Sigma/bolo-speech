"""Speech synthesis beyond one voice: multi-speaker dialogues (``POST /v1/audio/dialogue``)."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from bolo.api.common import get_core
from bolo.audio.encode import FORMATS, StreamEncoder, encode_audio, normalize_format
from bolo.dialogue import MODES, Turn, parse_script, speak_dialogue
from bolo.errors import BoloError, InvalidRequest
from bolo.keepalive import request_keep_alive
from bolo.obs.trace import Trace

router = APIRouter()


class DialogueTurn(BaseModel):
    speaker: str = Field(max_length=40, description="Who speaks this turn (any name; the same name, the same voice).")
    text: str = Field(description="What they say.")
    instructions: str | None = Field(
        default=None, max_length=4096, description="Direction for this turn only (stitched dialogues)."
    )


class DialogueRequest(BaseModel):
    model: str | None = Field(default=None, description="Any text-to-speech model; default `default_tts`.")
    turns: list[DialogueTurn] | None = Field(default=None, description="The dialogue, turn by turn.")
    script: str | None = Field(
        default=None,
        description="Instead of `turns`: one `Name: text` line per turn (a line without a name continues the turn).",
    )
    voices: dict[str, str] = Field(
        default_factory=dict,
        description="Speaker name -> voice: a voice of the model, a saved voice, a blend (stitched), or one of the "
        "reference clips `en-short`, `en-long`, `hi-short` (one take). A speaker without one gets the model's "
        "default voice (stitched) or a reference clip (one take).",
    )
    mode: str = Field(
        default="auto",
        description="`auto`: in one take when the model speaks dialogues itself (`features` has `dialogue`) and "
        "every voice has a recording, stitched otherwise; `native`: in one take or an error; `stitched`: each turn "
        "on its own, joined with `pause_ms` between them.",
    )
    pause_ms: int = Field(default=300, description="Silence between turns of a stitched dialogue (0 to 5000).")
    language: str | None = None
    speed: float = 1.0
    instructions: str | None = Field(
        default=None, max_length=4096, description="Direction for the whole dialogue, for models that follow it."
    )
    response_format: str = "wav"
    sample_rate: int | None = None
    keep_alive: str | int | float | None = None


@router.post("/v1/audio/dialogue", tags=["native"], summary="Speak a multi-speaker dialogue")
async def create_dialogue(request: Request, body: DialogueRequest) -> Response:
    """Extension. A dialogue (turns of speaker and text, a voice per speaker) as one recording.

    Models that speak a whole script themselves (`features` contains `dialogue`: VibeVoice on audio.cpp) do it
    in one take, with a recording of each voice (a saved voice or a reference clip); any other model speaks each
    turn on its own and the turns are joined. Headers: `x-bolo-dialogue` (`native` or `stitched`),
    `x-bolo-speakers` (the speakers in order), `x-bolo-model`, `x-bolo-segments` (stitched: JSON
    `[{turn, speaker, start, end}]`, in seconds)."""
    core = get_core(request)
    fmt = normalize_format(body.response_format)
    if fmt not in FORMATS:
        raise InvalidRequest(f"response_format must be one of {', '.join(FORMATS)}", param="response_format")
    if body.mode not in MODES:
        raise InvalidRequest(f"mode must be one of {', '.join(MODES)}", param="mode")
    if body.sample_rate is not None and not 8000 <= body.sample_rate <= 48000:
        raise InvalidRequest("sample_rate must be between 8000 and 48000", param="sample_rate")
    if body.turns and body.script:
        raise InvalidRequest("send turns or a script, not both", param="script")
    turns = (
        [Turn(t.speaker.strip(), t.text, t.instructions) for t in body.turns]
        if body.turns
        else parse_script(body.script or "")
    )
    request_keep_alive(body.keep_alive, ("tts",))
    trace = Trace("tts", "http", body.model or "")
    try:
        result = await speak_dialogue(
            core, turns, {k.strip(): v for k, v in body.voices.items()}, model=body.model, mode=body.mode,
            pause_ms=body.pause_ms, language=body.language, speed=body.speed, instructions=body.instructions,
            trace=trace,
        )  # fmt: skip
        trace.audio_seconds = len(result.samples) / result.sample_rate if result.sample_rate else 0.0
        data = await asyncio.to_thread(encode_audio, result.samples, result.sample_rate, fmt, body.sample_rate)
    except BoloError as exc:
        trace.finish("error", exc.message)
        if trace.model:
            core.traces.record(trace)
        raise
    core.complete_tts(trace)
    encoder = StreamEncoder(fmt, result.sample_rate, body.sample_rate)
    headers = {
        "x-request-id": trace.id,
        "x-bolo-model": result.model,
        "x-bolo-dialogue": result.mode,
        "x-bolo-speakers": ", ".join(result.speakers),
        "x-sample-rate": str(encoder.sample_rate),
    }
    if result.segments:
        headers["x-bolo-segments"] = json.dumps(result.segments, ensure_ascii=True, separators=(",", ":"))
    if trace.extra.get("instructions"):
        headers["x-bolo-instructions"] = str(trace.extra["instructions"])
    return Response(data, media_type=encoder.media_type, headers=headers)
