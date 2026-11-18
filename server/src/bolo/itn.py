"""Inverse text normalisation after speech recognition (``itn``): "twenty five dollars" -> "$25".

Recognisers trained on spoken-form text (sherpa-onnx's Zipformers, most CTC models, the streaming models) write
numbers, money, dates, times and measures as words. An inverse-normalisation model (``task: itn`` in the catalog)
rewrites them in written form after recognition, before punctuation is restored: WeTextProcessing's grammars per
language (``itn-en-wetext``, ``itn-zh-wetext``, ``itn-hi-wetext``) or sherpa-onnx's Chinese number rules
(``itn-zh-number``).

``itn`` takes, wherever it is accepted (a request, a model pin, the server setting):

* ``true`` / ``"on"``: the ``default_itn`` model when it knows the transcript's language, else an installed ``itn``
  model that does;
* ``false`` / ``"off"``: none, whatever the defaults say;
* a model id or alias of an ``itn`` model.

The first of request, model pin and server setting that says anything decides; the server's own default is
``"off"``. A transcript in a language the model does not know is returned as it was, and the response says it was
skipped (``x-bolo-itn``). Word timings follow the text: a word that is not changed keeps its own; the figures that
replace spoken words share the time of the words they were said in.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from bolo.errors import BoloError, InvalidRequest
from bolo.text.iso639 import normalize as normalize_language
from bolo.text.iso639 import same_language

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo, Transcript
    from bolo.engines.base import Word
    from bolo.obs.trace import Trace

log = logging.getLogger("bolo")

_ON = {"true", "on", "yes", "1"}
_OFF = {"false", "off", "no", "none", "0"}


@dataclass(frozen=True)
class ItnOptions:
    """What ``itn`` asked for: ``enabled`` False switches it off; ``model`` None means: chosen for the language."""

    enabled: bool = True
    model: str | None = None


def parse_itn(value: Any, param: str = "itn") -> ItnOptions | None:
    """``itn`` as given (see the module's docstring); None when it says nothing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return ItnOptions(value)
    if not isinstance(value, str):
        raise InvalidRequest(f"{param} must be true, false or an itn model id", param=param)
    text = value.strip()
    if text.lower() in _ON:
        return ItnOptions()
    if text.lower() in _OFF:
        return ItnOptions(False)
    return ItnOptions(True, text)


def plan(core: Bolo, spec: ModelSpec, requested: ItnOptions | None) -> ItnOptions | None:
    """The inverse normalisation asked for after ``spec`` transcribes (None: none). An unknown model in the request
    is an error; in a pin or the setting it is logged and the next source decides."""
    opts = requested
    if opts is None:
        for source, raw in (
            ("model pin", core.store.model_settings(spec.id).get("itn")),
            ("setting", core.settings.itn),
        ):
            try:
                opts = parse_itn(raw)
            except BoloError as exc:
                log.warning("ignoring the itn %s of '%s': %s", source, spec.id, exc.message)
                continue
            if opts is not None:
                break
    if opts is None or not opts.enabled:
        return None
    if opts.model is not None:
        try:
            core.resolve(opts.model, "itn")
        except BoloError as exc:
            if requested is not None:
                exc.param = "itn"
                raise
            log.warning("itn model %r cannot be used: %s", opts.model, exc.message)
            return None
    return opts


def language_of(transcript: Transcript, spec: ModelSpec) -> str | None:
    """The transcript's language, else the recogniser's only language."""
    code = transcript.language or (spec.languages[0] if len(spec.languages) == 1 else None)
    return normalize_language(code) if code else None


def _speaks(spec: ModelSpec, language: str | None) -> bool:
    return language is not None and any(same_language(code, language) for code in spec.languages)


def choose(core: Bolo, opts: ItnOptions, language: str | None) -> ModelSpec:
    """The ``itn`` model for a transcript in ``language``: the one asked for; else ``default_itn`` when it knows the
    language, else an installed one that does, else ``default_itn`` (which then skips it)."""
    if opts.model is not None:
        return core.resolve(opts.model, "itn")
    default = core.resolve(None, "itn")
    if _speaks(default, language):
        return default
    for spec in core.catalog.all():
        if spec.task == "itn" and _speaks(spec, language):
            try:
                if core.manager.is_installed(spec):
                    return spec
            except Exception:
                continue
    return default


def rewrite_words(words: list[Word], text: str) -> list[Word]:
    """Words timed for ``text``, the written form of what ``words`` say: unchanged words keep their times, and the
    written words that replace a stretch of spoken ones share that stretch's time evenly."""
    if not words:
        return words
    spoken = [w.w for w in words]
    written = text.split()
    out: list[Word] = []
    matcher = difflib.SequenceMatcher(a=[w.lower() for w in spoken], b=[w.lower() for w in written], autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            out.extend(replace(words[a0 + k], w=written[b0 + k]) for k in range(a1 - a0))
            continue
        if b1 == b0:
            continue  # words the grammars dropped ("and" in "one hundred and five")
        if a1 > a0:
            start, end = words[a0].t0, words[a1 - 1].t1
        else:  # written words with no spoken counterpart: at the end of what came before
            start = end = out[-1].t1 if out else words[0].t0
        step = (end - start) / (b1 - b0)
        out.extend(
            replace(words[min(a0, len(words) - 1)], w=written[b0 + k], t0=start + k * step, t1=start + (k + 1) * step)
            for k in range(b1 - b0)
        )
    return out


async def apply(core: Bolo, transcript: Transcript, spec: ModelSpec, opts: ItnOptions, trace: Trace) -> None:
    """Inverse-normalise ``transcript`` in place (its text, segments and words). Never raises for the model's sake:
    when it cannot run, the transcript stays as it was and the trace says so."""
    language = language_of(transcript, spec)
    model = choose(core, opts, language)
    record: dict[str, Any] = {"model": model.id, "language": language}
    trace.extra["itn"] = record
    if not _speaks(model, language):
        record.update(status="skipped", reason=f"{model.id} does not know '{language}'")
        return
    pieces = transcript.segments or []
    texts = [seg.text for seg in pieces] if pieces else [transcript.text]
    try:
        with trace.phase("itn"):
            async with core.manager.use(model.id) as entry:
                done: list[str] = []
                for text in texts:
                    upper = text.isupper()  # the English Zipformers write in capitals; the grammars read small letters
                    out, _ = await entry.queue.submit("itn", text.lower() if upper else text)
                    done.append(str(out).upper() if upper else str(out))
    except BoloError as exc:
        log.warning("itn: %s could not run, the transcript is kept as it was: %s", model.id, exc.message)
        record.update(status="skipped", reason=exc.message)
        return
    except Exception as exc:
        log.warning("itn: %s failed, the transcript is kept as it was: %s", model.id, exc)
        record.update(status="skipped", reason=f"{type(exc).__name__}: {exc}")
        return
    if pieces:
        for i, (seg, text) in enumerate(zip(pieces, done)):
            pieces[i] = replace(seg, text=text, words=rewrite_words(seg.words, text))
        transcript.text = " ".join(seg.text for seg in pieces).strip()
        if transcript.words:
            transcript.words = rewrite_words(transcript.words, transcript.text)
    else:
        transcript.text = done[0]
    record["status"] = "ok"


def header(record: Any) -> str | None:
    """``x-bolo-itn``: ``itn-en-wetext`` (the model that ran) or ``skipped; model=...; language=ta``."""
    if not isinstance(record, dict) or not record.get("model"):
        return None
    if record.get("status") == "skipped":
        return f"skipped; model={record['model']}; language={record.get('language')}"
    return str(record["model"]) if record.get("status") == "ok" else None
