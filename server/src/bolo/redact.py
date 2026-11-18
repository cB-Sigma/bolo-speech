"""Redaction of personal information in transcripts (``redact``).

A personal-information model (``task: pii`` in the catalog) finds names, telephone numbers, e-mail addresses, card
and account numbers, ID numbers, addresses and the like in the transcript after recognition (and after ``itn`` and
``punctuation``, so that it sees numbers as figures); each one found is replaced by its type in brackets
(``[PERSON]``, ``[PHONE_NUMBER]``). Three models: ``pii-presidio-patterns`` (Presidio's pattern recognisers, no
model), ``pii-presidio-spacy-en`` (the same with a spaCy English pipeline for names, places and organisations) and
``pii-gliner-multi`` (GLiNER multi PII, names and more in many languages).

``redact`` takes, wherever it is accepted (a request, a model pin, the server setting): ``true`` / ``"on"`` (the
``default_pii`` model), ``false`` / ``"off"``, or a model id or alias of a ``pii`` model. ``redact_entities`` (a
request field, or the ``redact_entities`` setting) narrows it to some entity types (``PERSON,PHONE_NUMBER``); unset,
every type the model knows is masked. The first of request, model pin and server setting that says anything
decides; the server's own default is ``"off"``.

Every redacted response says what was done, so that it can be audited: ``x-bolo-redact`` names the model and counts
each type masked, and ``verbose_json`` lists each redaction (``redactions``: its type, where its placeholder is in
``text``, the score, and its time when the words have times). The masked text itself is never returned or logged.
When the model cannot run, the request fails (503) rather than return unredacted text.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from bolo.errors import BoloError, EngineUnavailable, InvalidRequest

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo, Transcript
    from bolo.engines.base import PiiSpan, Segment, Word
    from bolo.obs.trace import Trace

log = logging.getLogger("bolo")

_ON = {"true", "on", "yes", "1"}
_OFF = {"false", "off", "no", "none", "0"}


@dataclass(frozen=True)
class RedactOptions:
    """What ``redact`` asked for: ``enabled`` False switches it off; ``model`` None means ``default_pii``;
    ``entities`` empty means every type the model knows."""

    enabled: bool = True
    model: str | None = None
    entities: tuple[str, ...] = ()


def parse_entities(value: Any, param: str = "redact_entities") -> tuple[str, ...]:
    """``PERSON, phone_number`` (or a list) as entity type names in capitals."""
    if value is None:
        return ()
    items = value if isinstance(value, (list, tuple)) else str(value).replace(";", ",").split(",")
    out = tuple(str(x).strip().upper().replace(" ", "_") for x in items if str(x).strip())
    for name in out:
        if not name.replace("_", "").isalnum():
            raise InvalidRequest(f"{param}: not an entity type: {name!r}", param=param)
    return out


def parse_redact(value: Any, param: str = "redact") -> RedactOptions | None:
    """``redact`` as given (see the module's docstring); None when it says nothing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return RedactOptions(value)
    if not isinstance(value, str):
        raise InvalidRequest(f"{param} must be true, false or a pii model id", param=param)
    text = value.strip()
    if text.lower() in _ON:
        return RedactOptions()
    if text.lower() in _OFF:
        return RedactOptions(False)
    return RedactOptions(True, text)


def plan(
    core: Bolo, spec: ModelSpec, requested: RedactOptions | None, entities: tuple[str, ...] = ()
) -> tuple[RedactOptions, ModelSpec] | None:
    """The redaction to apply to what ``spec`` transcribes (None: none), with the pii model resolved and the entity
    types to mask (the request's, else the ``redact_entities`` setting, else all). A model that cannot be resolved
    is an error wherever it was named: text that should be redacted is never returned as it was."""
    opts = requested
    if opts is None:
        for raw in (core.store.model_settings(spec.id).get("redact"), core.settings.redact):
            opts = parse_redact(raw, "redact")
            if opts is not None:
                break
    if opts is None or not opts.enabled:
        return None
    types = entities or parse_entities(core.settings.redact_entities or None)
    try:
        model = core.resolve(opts.model, "pii")
    except BoloError as exc:
        exc.param = "redact"
        raise
    return replace(opts, model=model.id, entities=types), model


def _locate_words(text: str, words: list[Word]) -> list[tuple[int, int] | None]:
    """Where each word is in ``text`` (character offsets), or None when it cannot be found in order."""
    out: list[tuple[int, int] | None] = []
    pos = 0
    for w in words:
        token = w.w.strip()
        at = text.find(token, pos) if token else -1
        if at < 0:
            out.append(None)
            continue
        out.append((at, at + len(token)))
        pos = at + len(token)
    return out


def mask(text: str, spans: list[PiiSpan], words: list[Word]) -> tuple[str, list[Word], list[dict[str, Any]]]:
    """``text`` with each span replaced by ``[TYPE]``; the words with the masked ones merged into that placeholder
    (keeping the time they covered); and one record per redaction (type, its placeholder's place, score, time)."""
    located = _locate_words(text, words)
    out_text: list[str] = []
    records: list[dict[str, Any]] = []
    cursor = 0
    length = 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < cursor:
            continue
        out_text.append(text[cursor : span.start])
        length += span.start - cursor
        placeholder = f"[{span.type}]"
        covered = [w for w, at in zip(words, located) if at is not None and at[0] < span.end and at[1] > span.start]
        record: dict[str, Any] = {
            "type": span.type,
            "start": length,
            "end": length + len(placeholder),
            "score": round(float(span.score), 3),
        }
        if covered:
            record["time_start"] = round(covered[0].t0, 3)
            record["time_end"] = round(covered[-1].t1, 3)
        records.append(record)
        out_text.append(placeholder)
        length += len(placeholder)
        cursor = span.end
    out_text.append(text[cursor:])
    new_text = "".join(out_text)
    new_words: list[Word] = []
    pending: dict[int, Word] = {}
    for w, at in zip(words, located):
        hit = None if at is None else next(
            (i for i, s in enumerate(spans) if at[0] < s.end and at[1] > s.start), None
        )  # fmt: skip
        if hit is None:
            new_words.append(w)
        elif hit in pending:
            pending[hit].t1 = max(pending[hit].t1, w.t1)
        else:
            pending[hit] = replace(w, w=f"[{spans[hit].type}]")
            new_words.append(pending[hit])
    return new_text, new_words, records


async def apply(core: Bolo, transcript: Transcript, opts: RedactOptions, model: ModelSpec, trace: Trace) -> None:
    """Redact ``transcript`` in place (text, segments, words) and record what was masked. Raises
    :class:`EngineUnavailable` when the model cannot run: a transcript that should be redacted is never returned
    as it was."""
    from bolo.text.iso639 import normalize as normalize_language

    language = normalize_language(transcript.language) if transcript.language else None
    pieces: list[Segment] = transcript.segments or []
    texts = [seg.text for seg in pieces] if pieces else [transcript.text]
    try:
        with trace.phase("redact"):
            async with core.manager.use(model.id) as entry:
                known = tuple(getattr(entry.model, "entities", ()))
                unknown = [e for e in opts.entities if known and e not in known]
                if unknown:
                    raise InvalidRequest(
                        f"{model.id} does not find {', '.join(unknown)}; it finds {', '.join(known)}",
                        param="redact_entities",
                    )
                found: list[list[PiiSpan]] = []
                for text in texts:
                    spans, _ = await entry.queue.submit("pii", (text, language, opts.entities))
                    found.append(list(spans))
    except InvalidRequest:
        raise
    except BoloError as exc:
        raise EngineUnavailable(f"redaction with {model.id} could not run: {exc.message}") from exc
    except Exception as exc:
        raise EngineUnavailable(f"redaction with {model.id} failed: {type(exc).__name__}: {exc}") from exc
    records: list[dict[str, Any]] = []
    if pieces:
        offset = 0
        for i, (seg, spans) in enumerate(zip(pieces, found)):
            text, words, done = mask(seg.text, spans, seg.words)
            pieces[i] = replace(seg, text=text, words=words)
            for r in done:
                records.append({**r, "start": r["start"] + offset, "end": r["end"] + offset, "segment": i})
            offset += len(text.strip()) + 1 if text.strip() else 0
        transcript.text = " ".join(seg.text.strip() for seg in pieces if seg.text.strip())
        transcript.words = [w for seg in pieces for w in seg.words] if transcript.words else transcript.words
    else:
        transcript.text, transcript.words, records = mask(transcript.text, found[0], transcript.words)
    transcript.redactions = records
    counts = Counter(r["type"] for r in records)
    trace.extra["redact"] = {"model": model.id, "counts": dict(sorted(counts.items())), "entities": list(opts.entities)}


def header(record: Any) -> str | None:
    """``x-bolo-redact``: ``pii-gliner-multi; PERSON=2, PHONE_NUMBER=1`` (``none`` when nothing was found)."""
    if not isinstance(record, dict) or not record.get("model"):
        return None
    counts = record.get("counts") or {}
    found = ", ".join(f"{k}={v}" for k, v in counts.items()) or "none"
    return f"{record['model']}; {found}"
