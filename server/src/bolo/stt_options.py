"""Decoding options of speech-to-text requests (bolo extensions of ``/v1/audio/transcriptions``).

Two groups:

* **engine options** (:data:`ENGINE_OPTIONS`) go to the recogniser in ``STTRequest.options``. A model lists the
  ones it honours in ``STTModel.decode_options``; bolo drops the others and says so (the response's
  ``x-bolo-ignored-options`` header, the request log and the server log).
* **bolo options** are applied by bolo itself, for every engine: ``vad_filter`` and ``vad_parameters`` (bolo's
  own silence removal, which cuts long audio before the recogniser hears it), ``chunk_length`` (the longest piece
  bolo hands over; faster-whisper also uses it for its own windows) and ``max_repeats`` (the repetition guard,
  :func:`trim_repeats`).

Each option can be set per request, pinned for a model (``PUT /api/v1/models/{id}/settings``) or given in the
model's catalog ``options``; the request wins over the pin, the pin over the catalog.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from bolo.engines.base import Segment, VadParams, Word


@dataclass(frozen=True)
class OptionSpec:
    kind: str  # "bool" | "int" | "float"
    lo: float | None = None
    hi: float | None = None
    lo_open: bool = False  # the lower bound itself is not allowed
    #: bolo option: applied by bolo, for every engine (never put in STTRequest.options unless the model asks)
    core: bool = False

    def describe(self) -> str:
        if self.kind == "bool":
            return "true or false"
        lo = "" if self.lo is None else (f"> {self.lo:g}" if self.lo_open else f">= {self.lo:g}")
        hi = "" if self.hi is None else f"<= {self.hi:g}"
        what = "an integer" if self.kind == "int" else "a number"
        return f"{what} {' and '.join(x for x in (lo, hi) if x)}".strip()


#: options the recogniser applies (faster-whisper names and meanings)
ENGINE_OPTIONS: dict[str, OptionSpec] = {
    "condition_on_previous_text": OptionSpec("bool"),
    "repetition_penalty": OptionSpec("float", 0.0, 10.0, lo_open=True),
    "no_repeat_ngram_size": OptionSpec("int", 0, 100),
    "compression_ratio_threshold": OptionSpec("float", 0.0, 100.0, lo_open=True),
    "log_prob_threshold": OptionSpec("float", -100.0, 0.0),
    "no_speech_threshold": OptionSpec("float", 0.0, 1.0),
    "temperature_increment_on_fallback": OptionSpec("float", 0.0, 1.0),
    "hallucination_silence_threshold": OptionSpec("float", 0.0, 3600.0, lo_open=True),
    "chunk_length": OptionSpec("int", 1, 30),
}

#: options bolo applies itself (besides ``chunk_length``, which is both)
CORE_OPTIONS: dict[str, OptionSpec] = {
    "vad_filter": OptionSpec("bool", core=True),
    "max_repeats": OptionSpec("int", 0, 10000, core=True),
}

#: every name a request, a pin or a catalog entry may set (``vad_parameters`` is an object, checked on its own)
DECODE_SETTINGS: tuple[str, ...] = (*ENGINE_OPTIONS, *CORE_OPTIONS, "vad_parameters")

#: the repetition guard's default: a run of more than this many back-to-back copies of the same phrase is cut to
#: one copy (0 turns the guard off)
DEFAULT_MAX_REPEATS = 4

#: ``vad_parameters`` keys (faster-whisper's names): key -> (spec, VadParams field, factor to seconds)
VAD_KEYS: dict[str, tuple[OptionSpec, str, float]] = {
    "threshold": (OptionSpec("float", 0.0, 1.0), "threshold", 1.0),
    "min_speech_duration_ms": (OptionSpec("int", 0, 60_000), "min_speech_s", 0.001),
    "max_speech_duration_s": (OptionSpec("float", 0.0, 3600.0, lo_open=True), "max_speech_s", 1.0),
    "min_silence_duration_ms": (OptionSpec("int", 0, 60_000), "min_silence_s", 0.001),
    "speech_pad_ms": (OptionSpec("int", 0, 10_000), "speech_pad_s", 0.001),
}
#: accepted for compatibility, not used by bolo's detectors
VAD_IGNORED_KEYS = ("neg_threshold",)

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class OptionError(ValueError):
    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


def coerce(name: str, spec: OptionSpec, raw: Any) -> Any:
    """``raw`` (a form string, or a JSON value) as the option's type, range-checked; raises :class:`OptionError`."""
    bad = OptionError(name, f"{name} must be {spec.describe()}; got {raw!r}")
    if spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise bad
    if isinstance(raw, bool):
        raise bad
    try:
        number = float(str(raw).strip()) if not isinstance(raw, (int, float)) else float(raw)
    except ValueError as exc:
        raise bad from exc
    if not math.isfinite(number):
        raise bad
    if spec.kind == "int":
        if number != int(number):
            raise bad
        number = int(number)
    if spec.lo is not None and (number <= spec.lo if spec.lo_open else number < spec.lo):
        raise bad
    if spec.hi is not None and number > spec.hi:
        raise bad
    return number


def parse_vad_parameters(raw: Any) -> dict[str, Any]:
    """``vad_parameters`` as an object (or its JSON text), checked: faster-whisper's key names and ranges."""
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OptionError("vad_parameters", f"vad_parameters must be a JSON object: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise OptionError("vad_parameters", 'vad_parameters must be an object such as {"threshold": 0.5}')
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in VAD_IGNORED_KEYS:
            out[key] = value
            continue
        if key not in VAD_KEYS:
            raise OptionError(
                "vad_parameters",
                f"unknown vad_parameters key {key!r}; accepted: {', '.join([*VAD_KEYS, *VAD_IGNORED_KEYS])}",
            )
        if value is None:
            continue
        out[key] = coerce(f"vad_parameters.{key}", VAD_KEYS[key][0], value)
    return out


def check(values: dict[str, Any]) -> dict[str, Any]:
    """Every decode option of ``values`` (other keys pass through untouched) coerced and range-checked."""
    out = dict(values)
    for name, value in values.items():
        if value is None or name not in DECODE_SETTINGS:
            continue
        if name == "vad_parameters":
            out[name] = parse_vad_parameters(value)
        else:
            out[name] = coerce(name, ENGINE_OPTIONS.get(name) or CORE_OPTIONS[name], value)
    return out


def from_form(form: Any) -> dict[str, Any]:
    """The decode options of a multipart form. ``vad_parameters`` comes as JSON text or as bracketed fields
    (``vad_parameters[threshold]``, which is how the OpenAI SDKs send a nested ``extra_body`` object)."""
    raw: dict[str, Any] = {}
    for name in DECODE_SETTINGS:
        value = form.get(name)
        if value not in (None, "") and isinstance(value, str):
            raw[name] = value
    nested = {
        key[len("vad_parameters[") : -1]: value
        for key, value in form.multi_items()
        if key.startswith("vad_parameters[") and key.endswith("]") and isinstance(value, str)
    }
    if nested:
        merged = parse_vad_parameters(raw.get("vad_parameters", "")) if "vad_parameters" in raw else {}
        merged.update(nested)
        raw["vad_parameters"] = merged
    return check(raw)


def vad_params(base: VadParams, given: dict[str, Any]) -> VadParams:
    """``base`` with the ``vad_parameters`` the request gave."""
    changes = {VAD_KEYS[k][1]: float(v) * VAD_KEYS[k][2] for k, v in given.items() if k in VAD_KEYS}
    return replace(base, **changes)


# ----------------------------------------------------------------------
# the repetition guard
# ----------------------------------------------------------------------

#: longest phrase (in words; twice as many characters for text written without spaces) the guard looks for
MAX_NGRAM = 8
_NORM = re.compile(r"[^\w']+", re.UNICODE)


def _norm(token: str) -> str:
    return _NORM.sub("", token.lower())


def _longest_run(keys: list[str], max_repeats: int, max_n: int) -> tuple[int, int, int] | None:
    """The first run of more than ``max_repeats`` back-to-back copies of one n-gram: (start, n, copies)."""
    size = len(keys)
    i = 0
    while i < size:
        best: tuple[int, int, int] | None = None
        for n in range(1, max_n + 1):
            if i + n * (max_repeats + 1) > size:
                break
            gram = keys[i : i + n]
            if not any(gram):
                continue
            copies = 1
            while keys[i + copies * n : i + (copies + 1) * n] == gram:
                copies += 1
            if copies > max_repeats and (best is None or copies * n > best[1] * best[2]):
                best = (i, n, copies)
        if best is not None:
            return best
        i += 1
    return None


def trim_repeats(segment: Segment, max_repeats: int = DEFAULT_MAX_REPEATS) -> tuple[Segment, int]:
    """Cut every run of more than ``max_repeats`` back-to-back copies of one phrase (1 to :data:`MAX_NGRAM` words)
    in ``segment`` down to one copy: Whisper's loop ("the the the ...") on silence and long audio.

    Returns the segment (the same object when nothing was cut) and the number of words removed. Words with
    timestamps follow the text, and the segment then ends where its last kept word ends.
    """
    if max_repeats <= 0 or not segment.text:
        return segment, 0
    # text written without spaces (Chinese, Japanese, Thai): the phrase is counted in characters
    by_chars = " " not in segment.text.strip() and any(ord(c) >= 0x0E00 for c in segment.text)
    tokens = list(segment.text.strip()) if by_chars else segment.text.split()
    keys = [_norm(t) for t in tokens]
    keep = [True] * len(tokens)
    removed = 0
    start = 0
    while True:
        run = _longest_run(keys[start:], max_repeats, 2 * MAX_NGRAM if by_chars else MAX_NGRAM)
        if run is None:
            break
        at, n, copies = run
        at += start
        for k in range(at + n, at + n * copies):
            keep[k] = False
        removed += n * (copies - 1)
        start = at + n * copies
    if not removed:
        return segment, 0
    kept_tokens = [t for t, k in zip(tokens, keep) if k]
    text = ("" if by_chars else " ").join(kept_tokens)
    words = segment.words
    end = segment.end
    if words and not by_chars:
        # when the engine split words differently from the spaces, keep as many words as there are kept tokens
        words = [w for w, k in zip(words, keep) if k] if len(words) == len(tokens) else words[: len(kept_tokens)]
        if words:
            end = min(end, words[-1].t1)
    return replace(segment, text=text, words=[Word(w.w, w.t0, w.t1) for w in words], end=end), removed


def drop_repeated_segments(segments: list[Segment], max_repeats: int = DEFAULT_MAX_REPEATS) -> list[Segment]:
    """A run of more than ``max_repeats`` back-to-back segments with the same text ("Thank you." over and over on
    silence) keeps its first segment only."""
    if max_repeats <= 0:
        return segments
    keys = [" ".join(_norm(t) for t in seg.text.split()) for seg in segments]
    out: list[Segment] = []
    i = 0
    while i < len(segments):
        j = i
        while j + 1 < len(segments) and keys[i] and keys[j + 1] == keys[i]:
            j += 1
        out.extend(segments[i : i + 1] if j - i + 1 > max_repeats else segments[i : j + 1])
        i = j + 1
    return out
