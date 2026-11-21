"""Spoken intent (task ``intent``): what a speaker wants, straight from the audio.

A spoken-language-understanding model hears one utterance ("wake me up at seven tomorrow") and writes its meaning:
the intent (``alarm_set``), and for SLURP-style models its scenario (``alarm``) and action (``set``), with the slots
it filled (``time``: ``seven``, ``date``: ``tomorrow``). There is no transcript in between, so a slot's value is the
model's own spelling of the words it heard.

SpeechBrain's direct SLU models write their semantics as a Python dict literal
(``{'scenario': 'alarm', 'action': 'set', 'entities': [{'type': 'time', 'filler': 'seven'}]}``);
:func:`parse_semantics` reads it without evaluating code. The SLURP model writes ``|`` where the literal has commas
(``{'scenario': 'alarm'| 'action': 'set'| ...}``, as its training recipe does); bolo reads both. A model that writes
something it cannot read gives ``intent: null`` and its text.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import FloatArray, IntentModel, IntentResult
from bolo.errors import InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

RATE = 16000
_FIELD = re.compile(r"""['"](scenario|action|intent)['"]\s*:\s*['"]([^'"]*)['"]""")
_ENTITY = re.compile(r"""['"]type['"]\s*:\s*['"]([^'"]*)['"]\s*[,|]\s*['"]filler['"]\s*:\s*['"]([^'"]*)['"]""")


def _slots(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):  # {'number1': 37.67, 'op': ' minus '} (Timers and Such)
        return [{"type": str(k), "value": str(v).strip()} for k, v in value.items()]
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and "type" in item:
            out.append({"type": str(item["type"]), "value": str(item.get("filler", item.get("value", ""))).strip()})
    return out


def parse_semantics(text: str) -> IntentResult:
    """An :class:`IntentResult` from a SpeechBrain SLU model's output text."""
    text = text.strip()
    data = None
    # the SLURP recipe writes "|" where the dict has commas (its tokenizer keeps commas for the fillers)
    for candidate in (text, text.replace("| ", ", ")):
        try:
            data = ast.literal_eval(candidate)
            break
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    if isinstance(data, dict):
        scenario = data.get("scenario")
        action = data.get("action")
        intent = data.get("intent")
        slots = _slots(data.get("entities")) + _slots(data.get("slots"))
        known = {"scenario", "action", "intent", "entities", "slots"}
        # other plain fields are slots too (Fluent Speech Commands: {'action': ..., 'object': ..., 'location': ...})
        slots += [
            {"type": str(k), "value": str(v)}
            for k, v in data.items()
            if k not in known and not isinstance(v, (dict, list))
        ]
    else:  # a broken literal (an unclosed quote): take what can be read
        found = dict(_FIELD.findall(text))
        scenario, action, intent = found.get("scenario"), found.get("action"), found.get("intent")
        slots = [{"type": t, "value": v.strip()} for t, v in _ENTITY.findall(text)]
    scenario = str(scenario) if scenario not in (None, "") else None
    action = str(action) if action not in (None, "") else None
    if not intent and (scenario or action):
        intent = "_".join(x for x in (scenario, action) if x)
    return IntentResult(str(intent) if intent else None, scenario, action, slots, text)


async def understand(
    core: Bolo, samples: FloatArray, rate: int, model: str, trace: Trace
) -> tuple[IntentResult, float]:
    """The ``intent`` model ``model`` (resolved) on ``samples``; and the recording's length in seconds."""
    x = np.asarray(samples, dtype=np.float32)
    x = resample(x, rate, RATE) if rate != RATE else x
    duration = len(x) / RATE
    if len(x) < RATE // 10:
        raise InvalidRequest("the recording is shorter than 0.1 s", param="file")
    async with core.manager.use(model) as entry:
        if not entry.queue.handles("intent"):
            raise InvalidRequest(f"model '{model}' does not understand spoken intents", param="model")
        limit = float(getattr(entry.model, "max_audio_s", 30.0) or 30.0)
        if duration > limit:
            raise InvalidRequest(
                f"the recording is {duration:.1f} s long; '{model}' hears one utterance of at most {limit:g} s",
                param="file",
            )
        with trace.phase("intent"):
            result, _ = await entry.queue.submit("intent", np.ascontiguousarray(x, dtype=np.float32))
    if not isinstance(result, IntentResult):
        result = IntentResult.from_dict(dict(result))
    return result, duration


def body(result: IntentResult, model: str, duration: float) -> dict[str, Any]:
    return {"object": "audio.intent", "model": model, "duration": round(duration, 3), **result.to_dict()}


__all__ = ["IntentModel", "IntentResult", "body", "parse_semantics", "understand"]
