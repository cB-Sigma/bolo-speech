"""Decisions (task ``decision``): typed questions about a recording, a text or both, answered by a decision model.

A decision model reads the state and each question in one pass and answers from its distribution over the options:
no text is generated, so an answer is always one of the options, with its probability. Three types of question:

* ``yes_no`` (``noul`` in llama.cpp's ``/v1/systemone``): the answer is true or false, with the probability of yes;
* ``choice``: one option of a list (``options``: a list of names, or names mapped to a description), with the
  probability of each option and a confidence (0 when every option is as likely);
* ``score``: a level on a scale (``levels``: 2 to 10 descriptions, lowest first), with the expected level (it can lie
  between two levels), the nearest level and its description, and the probability of each level.

The request (``POST /v1/audio/decisions``, ``bolo decide``) gives ``questions`` as an object that maps an id to a
question: ``{"type": "choice", "question": "Which team?", "options": {"billing": "Charges, refunds"}}``. The names of
``/v1/systemone`` are accepted too (``instructions``, ``criteria``, type ``noul``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import DecisionRequest, FloatArray
from bolo.errors import InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

RATE = 16000
MAX_QUESTIONS = 32
MAX_OPTIONS = 255
MAX_TEXT = 32000
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
#: bolo's type names, and llama.cpp's
TYPES = {"yes_no": "noul", "yes-no": "noul", "noul": "noul", "choice": "choice", "score": "score"}


@dataclass
class Decision:
    model: str
    duration: float | None
    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any] = field(default_factory=dict)


def _text(value: Any, what: str, qid: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequest(f"question '{qid}': {what} must be a non-empty string", param="questions")
    if len(value) > 2000:
        raise InvalidRequest(f"question '{qid}': {what} is longer than 2000 characters", param="questions")
    return value.strip()


def parse_questions(raw: Any) -> dict[str, dict[str, Any]]:
    """bolo's questions (a JSON object, or its text) in ``/v1/systemone``'s shape; :class:`InvalidRequest` when
    anything does not hold."""
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise InvalidRequest(f"questions is not valid JSON: {exc}", param="questions") from exc
    if not isinstance(raw, dict) or not raw:
        raise InvalidRequest("questions must be an object that maps an id to a question", param="questions")
    if len(raw) > MAX_QUESTIONS:
        raise InvalidRequest(f"at most {MAX_QUESTIONS} questions per request", param="questions")
    out: dict[str, dict[str, Any]] = {}
    for qid, q in raw.items():
        if not isinstance(qid, str) or not _ID.match(qid):
            raise InvalidRequest(f"question id {qid!r}: letters, digits, '_', '-' and '.', 1 to 64", param="questions")
        if not isinstance(q, dict):
            raise InvalidRequest(f"question '{qid}' must be an object", param="questions")
        kind = TYPES.get(str(q.get("type") or "").lower())
        if kind is None:
            raise InvalidRequest(f"question '{qid}': type must be yes_no, choice or score", param="questions")
        instructions = _text(q.get("question", q.get("instructions")), "question", qid)
        one: dict[str, Any] = {"type": kind, "instructions": instructions}
        if kind == "choice":
            options = q.get("options", q.get("criteria"))
            if isinstance(options, list):
                options = {str(o): None for o in options}
            if not isinstance(options, dict) or not 2 <= len(options) <= MAX_OPTIONS:
                raise InvalidRequest(
                    f"question '{qid}': options must name 2 to {MAX_OPTIONS} answers (a list, or names mapped to a "
                    "description)",
                    param="questions",
                )  # fmt: skip
            criteria: dict[str, str | None] = {}
            for name, about in options.items():
                key = _text(str(name), "an option", qid)
                criteria[key] = None if about in (None, "") else _text(about, f"the description of '{key}'", qid)
            if len(criteria) != len(options):
                raise InvalidRequest(f"question '{qid}': two options have the same name", param="questions")
            one["criteria"] = criteria
        elif kind == "score":
            levels = q.get("levels", q.get("criteria"))
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise InvalidRequest(f"question '{qid}': levels must list 2 to 10 descriptions, lowest first",
                                     param="questions")  # fmt: skip
            one["criteria"] = [_text(level, "a level", qid) for level in levels]
        out[qid] = one
    return out


async def decide(
    core: Bolo,
    model: str,
    text: str | None,
    samples: FloatArray | None,
    rate: int | None,
    questions: dict[str, dict[str, Any]],
    trace: Trace,
) -> Decision:
    """Ask ``questions`` (as :func:`parse_questions` returns them) about ``text`` and/or the recording."""
    text = (text or "").strip() or None
    if text is not None and len(text) > MAX_TEXT:
        raise InvalidRequest(f"text is longer than {MAX_TEXT} characters", param="text")
    x: FloatArray | None = None
    duration = None
    if samples is not None:
        x = np.asarray(samples, dtype=np.float32)
        if rate and rate != RATE:
            x = resample(x, rate, RATE)
        duration = len(x) / RATE
        if len(x) < RATE // 10:
            raise InvalidRequest("the recording is shorter than 0.1 s", param="file")
    if text is None and x is None:
        raise InvalidRequest("give a recording (file), a text, or both", param="file")
    async with core.manager.use(model) as entry:
        if not entry.queue.handles("decide"):
            raise InvalidRequest(f"model '{model}' is not a decision model", param="model")
        longest = float(getattr(entry.model, "max_audio_s", 30.0) or 30.0)
        if duration is not None and duration > longest + 0.05:
            raise InvalidRequest(
                f"model '{model}' hears at most {longest:g} s of audio; this recording is {duration:.1f} s",
                param="file",
            )
        with trace.phase("decide"):
            raw, _ = await entry.queue.submit("decide", DecisionRequest(text, x, questions))
    return Decision(model, duration, _answers(raw.get("answers") or {}, questions), dict(raw.get("usage") or {}))


def _answers(raw: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``/v1/systemone``'s answers in bolo's shape."""
    out: dict[str, dict[str, Any]] = {}
    for qid, q in questions.items():
        a = raw.get(qid)
        if not isinstance(a, dict):
            raise RuntimeError(f"the model gave no answer to question '{qid}'")
        if q["type"] == "noul":
            p = float(a.get("noul", 0.0))
            out[qid] = {"type": "yes_no", "answer": p >= 0.5, "probability": round(p, 4)}
        elif q["type"] == "choice":
            probs = {str(k): round(float(v), 4) for k, v in dict(a.get("probabilities") or {}).items()}
            out[qid] = {
                "type": "choice", "answer": str(a.get("choice")), "confidence": round(float(a.get("confidence", 0)), 4),
                "probabilities": probs,
            }  # fmt: skip
        else:
            levels = list(q["criteria"])
            score = float(a.get("score", 0.0))
            level = min(len(levels) - 1, max(0, round(score)))
            probs_raw = dict(a.get("probabilities") or {})
            out[qid] = {
                "type": "score", "score": round(score, 4), "level": level, "label": levels[level],
                "confidence": round(float(a.get("confidence", 0)), 4),
                "probabilities": [round(float(probs_raw.get(str(i), 0.0)), 4) for i in range(len(levels))],
            }  # fmt: skip
    return out


def body(result: Decision) -> dict[str, Any]:
    out: dict[str, Any] = {"object": "audio.decision", "model": result.model, "answers": result.answers}
    if result.duration is not None:
        out["duration"] = round(result.duration, 3)
    if result.usage:
        out["usage"] = {"input_tokens": int(result.usage.get("input_tokens", 0))}
    return out
