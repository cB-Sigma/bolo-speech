"""Punctuation and casing restoration after speech recognition (``punctuation``).

Many recognisers write no punctuation and no capital letters (sherpa-onnx's Zipformers and most CTC models). A
punctuation model (``task: punctuation`` in the catalog) adds them to the transcript afterwards; the words stay
the same, so word timings still fit.

``punctuation`` takes, wherever it is accepted (a request, a model pin, the server setting):

* ``"auto"``: restore it only when the recogniser writes none (its catalog entry has no ``punctuation`` feature);
* ``true`` / ``"on"``: restore it whatever the recogniser wrote;
* ``false`` / ``"off"``: never;
* a model id or alias of a punctuation model: ``"on"`` with that model (else ``default_punctuation``).

The first of request, model pin and server setting that says anything decides; the server's own default is
``"off"``. A punctuation model only runs on a transcript in a language it knows (``languages`` of its entry); on
another one, or when it cannot run, the transcript is returned as it was and the response says it was skipped
(``x-bolo-punctuation``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from bolo.errors import BoloError, InvalidRequest
from bolo.text.iso639 import same_language

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo, Transcript
    from bolo.obs.trace import Trace

log = logging.getLogger("bolo")

_ON = {"true", "on", "yes", "1"}
_OFF = {"false", "off", "no", "none", "0"}
MODES = ("off", "auto", "on")


@dataclass(frozen=True)
class PunctuationOptions:
    """What ``punctuation`` asked for: ``mode`` off, auto or on; ``model`` None means ``default_punctuation``."""

    mode: str = "on"
    model: str | None = None


def parse_punctuation(value: Any, param: str = "punctuation") -> PunctuationOptions | None:
    """``punctuation`` as given (see the module's docstring); None when it says nothing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return PunctuationOptions("on" if value else "off")
    if not isinstance(value, str):
        raise InvalidRequest(f"{param} must be auto, on, off or a punctuation model id", param=param)
    text = value.strip()
    low = text.lower()
    if low in _ON:
        return PunctuationOptions("on")
    if low in _OFF:
        return PunctuationOptions("off")
    if low == "auto":
        return PunctuationOptions("auto")
    return PunctuationOptions("on", text)


def plan(core: Bolo, spec: ModelSpec, requested: PunctuationOptions | None) -> PunctuationOptions | None:
    """The punctuation to restore after ``spec`` transcribes (None: none). An unknown model in the request is an
    error; in a pin or the setting it is logged and the next source decides."""
    opts = requested
    if opts is None:
        for source, raw in (
            ("model pin", core.store.model_settings(spec.id).get("punctuation")),
            ("setting", core.settings.punctuation),
        ):
            try:
                opts = parse_punctuation(raw)
            except BoloError as exc:
                log.warning("ignoring the punctuation %s of '%s': %s", source, spec.id, exc.message)
                continue
            if opts is not None:
                break
    if opts is None or opts.mode == "off":
        return None
    if opts.mode == "auto" and "punctuation" in spec.features:
        return None  # the recogniser writes its own
    try:
        model = core.resolve(opts.model, "punctuation")
    except BoloError as exc:
        if requested is not None:
            exc.param = "punctuation"
            raise
        log.warning("punctuation model %r cannot be used: %s", opts.model, exc.message)
        return None
    return PunctuationOptions(opts.mode, model.id)


def _speaks(spec: ModelSpec, language: str | None) -> bool:
    if not spec.languages or not language:
        return not language or not spec.languages
    return any(same_language(code, language) for code in spec.languages)


async def apply(core: Bolo, transcript: Transcript, opts: PunctuationOptions, trace: Trace) -> None:
    """Restore punctuation in ``transcript`` (its text, each segment and the words) in place. Never raises for the
    punctuation model's sake: when it cannot run, the transcript stays as it was and the trace says so."""
    assert opts.model is not None
    spec = core.resolve(opts.model, "punctuation")
    record: dict[str, Any] = {"model": spec.id, "mode": opts.mode}
    trace.extra["punctuation"] = record
    language = transcript.language
    if not _speaks(spec, language):
        record.update(status="skipped", reason=f"{spec.id} does not know '{language}'")
        return
    try:
        with trace.phase("punctuation"):
            async with core.manager.use(spec.id) as entry:
                pieces = transcript.segments or []
                texts = [seg.text for seg in pieces] if pieces else [transcript.text]
                if "casing" in spec.features:  # it writes the capitals itself, and reads all-capital text as noise
                    texts = [t.lower() if t.isupper() else t for t in texts]
                done: list[str] = []
                for text in texts:
                    out, _ = await entry.queue.submit("punctuate", text)
                    done.append(str(out))
    except BoloError as exc:
        log.warning("punctuation: %s could not run, the transcript is kept as it was: %s", spec.id, exc.message)
        record.update(status="skipped", reason=exc.message)
        return
    except Exception as exc:
        log.warning("punctuation: %s failed, the transcript is kept as it was: %s", spec.id, exc)
        record.update(status="skipped", reason=f"{type(exc).__name__}: {exc}")
        return
    if pieces:
        for i, (seg, text) in enumerate(zip(pieces, done)):
            words = _rewrite_words(seg.words, text)
            pieces[i] = replace(seg, text=text, words=words)
        transcript.text = " ".join(seg.text for seg in pieces).strip()
        if transcript.words:
            transcript.words = _rewrite_words(transcript.words, transcript.text)
    else:
        transcript.text = done[0]
    record["status"] = "ok"


def _rewrite_words(words: list[Any], text: str) -> list[Any]:
    """Word timings with the punctuated spelling of each word, when the text still has one token per word."""
    tokens = text.split()
    if not words or len(tokens) != len(words):
        return words
    return [replace(w, w=token) for w, token in zip(words, tokens)]


def header(record: Any) -> str | None:
    """``x-bolo-punctuation``: ``punct-en-cnn-bilstm`` or ``skipped; model=...; reason=...``."""
    if not isinstance(record, dict) or not record.get("model"):
        return None
    if record.get("status") == "skipped":
        return f"skipped; model={record['model']}"
    return str(record["model"])
