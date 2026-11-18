"""Spoken-language identification (task ``lid``) and routing a transcription to a recogniser for the language.

``POST /v1/audio/language`` answers which languages a recording is likely in, with probabilities
(:meth:`bolo.core.Bolo.identify_language`).

Routing: a transcription whose model is ``auto`` and which names no language is heard by the language identifier
first (``default_lid``); the audio then goes to a recogniser for that language (:func:`route`):

1. the ``language_routes`` setting, ``<language>=<model>``, when it names one for the language;
2. otherwise the best *installed* recogniser that speaks it: the one this machine's recommendations name for the
   language, then one made for that language over one that covers many, then the measured ones, then
   ``default_stt``;
3. otherwise ``default_stt``.

A recogniser the catalog marks as not to be used for a language (``options.not_recommended_for``, with the reason)
is never chosen for it in step 2. When the recogniser bolo recommends for the language is not installed, the
response names it (``recommended=``) and the server log says so; with ``auto_pull`` on, it is downloaded and
answers the request (``pulled=true``), and the request falls back as above if the download fails.

A detected language below ``lid_min_probability`` is not trusted: ``default_stt`` answers, not told the language.
When the language identifier cannot run, the request still goes on, with ``default_stt``. A request that names a
model other than ``auto``, or a language, is never routed. The decision is in the ``x-bolo-route`` and
``x-bolo-language`` response headers and in the request trace (``route``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import FloatArray, LidModel
from bolo.errors import BoloError
from bolo.text.iso639 import name as language_name
from bolo.text.iso639 import normalize, same_language

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

log = logging.getLogger("bolo")

RATE = 16000
#: the model names that ask for routing
ROUTING_NAMES = ("auto",)


@dataclass
class LidResult:
    """What a language identifier heard: every language it knows with its probability, most likely first."""

    model: str
    languages: list[tuple[str, float]]
    #: seconds of audio the model looked at (at most its ``max_s``) and in the recording
    heard_s: float = 0.0
    duration_s: float = 0.0
    ms: float = 0.0

    @property
    def language(self) -> str | None:
        return self.languages[0][0] if self.languages else None

    @property
    def probability(self) -> float:
        return float(self.languages[0][1]) if self.languages else 0.0

    def top(self, k: int) -> list[dict[str, Any]]:
        return [
            {"language": code, "name": language_name(code), "probability": round(float(p), 4)}
            for code, p in self.languages[: max(1, k)]
        ]

    def to_dict(self, top_k: int = 5) -> dict[str, Any]:
        return {
            "model": self.model,
            "language": self.language,
            "name": language_name(self.language) if self.language else None,
            "probability": round(self.probability, 4),
            "languages": self.top(top_k),
            "duration": round(self.duration_s, 3),
            "heard": round(self.heard_s, 3),
        }


@dataclass
class Route:
    """How a routed transcription was answered."""

    requested: str
    model: str
    #: route (language_routes), recommended (best installed recogniser), default (none speaks the language),
    #: unsure (probability under lid_min_probability), lid-failed (the identifier could not run)
    reason: str
    lid: str | None = None
    language: str | None = None
    probability: float | None = None
    #: the language the recogniser was told (None: it decided for itself)
    steer: str | None = None
    #: a recogniser bolo recommends for the language that is not installed
    recommended: str | None = None
    detail: str | None = None
    candidates: list[str] = field(default_factory=list)
    lid_ms: float | None = None
    #: the recommended recogniser was not installed and was downloaded for this request (auto_pull)
    pulled: bool = False

    def header(self) -> str:
        """``x-bolo-route``: ``auto -> indicconformer-ta-community; reason=recommended; language=ta;
        probability=0.97; lid=voxlingua107-ecapa``."""
        parts = [f"{self.requested} -> {self.model}", f"reason={self.reason}"]
        if self.language:
            parts.append(f"language={self.language}")
        if self.probability is not None:
            parts.append(f"probability={self.probability:.3g}")
        if self.lid:
            parts.append(f"lid={self.lid}")
        if self.recommended:
            parts.append(f"recommended={self.recommended}")
        if self.pulled:
            parts.append("pulled=true")
        return "; ".join(parts)

    def language_header(self) -> str | None:
        """``x-bolo-language`` on a routed transcription: ``ta; source=detected; probability=0.97; applied=true``."""
        if not self.language:
            return None
        p = f"; probability={self.probability:.3g}" if self.probability is not None else ""
        return f"{self.language}; source=detected{p}; applied={'true' if self.steer else 'false'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "model": self.model,
            "reason": self.reason,
            "lid": self.lid,
            "language": self.language,
            "probability": None if self.probability is None else round(self.probability, 4),
            "steer": self.steer,
            "recommended": self.recommended,
            "detail": self.detail,
            "candidates": self.candidates,
            "lid_ms": self.lid_ms,
            "pulled": self.pulled,
        }


def route_headers(record: dict[str, Any]) -> dict[str, str]:
    """The response headers of a routed request, from its trace record (``Route.to_dict``)."""
    try:
        decision = Route(**record)
    except TypeError:
        return {}
    headers = {"x-bolo-route": decision.header()}
    said = decision.language_header()
    if said:
        headers["x-bolo-language"] = said
    return headers


def parse_routes(entries: list[str]) -> dict[str, str]:
    """``language_routes`` entries (``ta=indicconformer-ta-community``) as language -> model."""
    out: dict[str, str] = {}
    for item in entries:
        language, sep, model = str(item).partition("=")
        if sep and language.strip() and model.strip():
            out[normalize(language)] = model.strip()
    return out


def is_routing_request(core: Bolo, model: str | None, language: str | None) -> bool:
    """A transcription to route: the model is ``auto``, no language was given, and routing is on."""
    return (
        bool(core.settings.language_routing)
        and (model or "").strip().lower() in ROUTING_NAMES
        and (not language or language.strip().lower() == "auto")
    )


async def identify(
    core: Bolo, samples: FloatArray, sample_rate: int, model: str | None = None, trace: Trace | None = None
) -> LidResult:
    """Run a language identifier (``model``, else ``default_lid``) on mono audio at ``sample_rate``."""
    spec = core.resolve(model, "lid")
    duration = len(samples) / sample_rate if sample_rate else 0.0
    t0 = trace.now() if trace is not None else 0.0
    loop = asyncio.get_running_loop()
    started = loop.time()
    async with core.manager.use(spec.id) as entry:
        lid = entry.model
        assert isinstance(lid, LidModel)
        keep = int(lid.max_s * sample_rate) if lid.max_s > 0 else len(samples)
        audio = np.asarray(samples[:keep], dtype=np.float32)  # only what the model looks at is resampled
        if sample_rate != RATE:
            audio = await asyncio.to_thread(resample, audio, sample_rate, RATE)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        languages, _ = await entry.queue.submit("lid", audio)
    ms = (loop.time() - started) * 1000.0
    if trace is not None:
        trace.add_phase("lid", t0, trace.now())
    return LidResult(
        model=spec.id,
        languages=[(str(code), float(p)) for code, p in languages],
        heard_s=len(audio) / RATE,
        duration_s=duration,
        ms=ms,
    )


def _speaks(spec: ModelSpec, language: str) -> bool:
    return any(same_language(code, language) for code in spec.languages)


def _works_for(spec: ModelSpec, language: str) -> bool:
    return _speaks(spec, language) and spec.not_recommended(language) is None


def recommended_for(core: Bolo, language: str) -> str | None:
    """The local recogniser this machine's recommendations name first for ``language`` (installed or not), from
    what the advisor already knows (never computed on the request path)."""
    best: str | None = None
    for spec in core.catalog.all():
        if spec.task != "stt" or spec.cloud or spec.builtin or not _works_for(spec, language):
            continue
        fit = core.advisor.cached_fit(spec)
        if fit is not None and any(same_language(m, language) for m in fit.default_for if m != "*"):
            return spec.id
        if best is None and fit is not None and any(same_language(m, language) for m in fit.recommended_for):
            best = spec.id
    return best


def installed_recognisers(core: Bolo, language: str) -> list[ModelSpec]:
    """Installed local recognisers that speak ``language``, best first (see the module's docstring)."""
    engines: dict[str, bool] = {}
    found: list[ModelSpec] = []
    for spec in core.catalog.all():
        if spec.task != "stt" or spec.cloud or not spec.runnable or not spec.languages:
            continue
        if not _works_for(spec, language):
            continue
        if spec.engine not in engines:
            try:
                engines[spec.engine] = bool(core.engines.status(spec.engine).available)
            except Exception:
                engines[spec.engine] = False
        if not engines[spec.engine]:
            continue
        try:
            if not core.manager.is_installed(spec):
                continue
        except Exception:
            continue
        found.append(spec)
    default = core.settings.default_stt

    def key(spec: ModelSpec) -> tuple[Any, ...]:
        fit = core.advisor.cached_fit(spec)
        marked = fit is not None and any(same_language(m, language) for m in fit.default_for if m != "*")
        liked = fit is not None and any(same_language(m, language) for m in fit.recommended_for if m != "*")
        count = len(spec.languages)
        breadth = 0 if count <= 3 else (1 if count <= 25 else 2)
        return (
            not marked,
            not liked,
            breadth,
            spec.builtin,
            spec.validated != "cpu",
            spec.id != default and default not in spec.aliases,
            spec.id,
        )

    return sorted(found, key=key)


async def route(core: Bolo, samples: FloatArray, sample_rate: int, requested: str, trace: Trace) -> Route:
    """Decide which recogniser hears a routed transcription (see the module's docstring). Never raises for the
    language identifier's sake: when it cannot run, ``default_stt`` answers."""
    s = core.settings
    default = core.resolve(None, "stt").id
    try:
        result = await identify(core, samples, sample_rate, None, trace)
    except asyncio.CancelledError:
        raise
    except BoloError as exc:
        log.warning("language routing: the language identifier could not run, using %s: %s", default, exc.message)
        return Route(requested, default, "lid-failed", lid=s.default_lid, detail=exc.message)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log.warning("language routing: the language identifier failed, using %s: %s", default, detail)
        return Route(requested, default, "lid-failed", lid=s.default_lid, detail=detail)
    decision = decide(core, result, requested)
    if decision.recommended:
        await _pull_recommended(core, decision)
    return decision


async def _pull_recommended(core: Bolo, decision: Route) -> None:
    """The recogniser recommended for the detected language is not installed: say so, and with ``auto_pull``
    download it and let it answer. A download that cannot start or fails leaves ``decision`` as it was."""
    best = decision.recommended
    assert best is not None
    language = language_name(decision.language) if decision.language else decision.language
    if not core.settings.auto_pull:
        log.warning(
            "language routing: %s is recommended for %s but is not installed (auto_pull is off), so %s answers; "
            "run `bolo pull %s`",
            best, language, decision.model, best,
        )  # fmt: skip
        return
    log.warning("language routing: %s is recommended for %s and is not installed; pulling it now", best, language)
    try:
        spec = core.resolve(best, "stt")
        status = core.engines.status(spec.engine)
        if not status.available:
            raise BoloError(f"its engine {spec.engine} is not available here ({status.reason or 'not installed'})")
        await core.ensure_installed(spec)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        detail = exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}"
        log.warning("language routing: could not pull %s, %s answers: %s", best, decision.model, detail)
        decision.detail = f"could not pull {best}: {detail}"
        return
    decision.model, decision.reason, decision.steer = spec.id, "recommended", decision.language
    decision.pulled = True
    decision.detail = None


def decide(core: Bolo, result: LidResult, requested: str = "auto") -> Route:
    """The recogniser a routed transcription of audio in which ``result`` was heard goes to."""
    s = core.settings
    default = core.resolve(None, "stt").id
    language, p = result.language, result.probability
    base = Route(requested, default, "default", lid=result.model, language=language, probability=p)
    base.lid_ms = round(result.ms, 1)
    if language is None:
        base.reason = "unsure"
        return base
    if p < s.lid_min_probability:
        base.reason = "unsure"
        base.detail = f"probability {p:.3g} is under lid_min_probability {s.lid_min_probability:g}"
        return base
    routes = parse_routes(s.language_routes)
    target = next((m for lang, m in routes.items() if same_language(lang, language)), None)
    if target:
        try:
            spec = core.resolve(target, "stt")
        except BoloError as exc:
            log.warning("language_routes: %s=%s cannot be used: %s", language, target, exc.message)
        else:
            base.model, base.reason = spec.id, "route"
            base.steer = language if _speaks(spec, language) else None
            return base
    found = installed_recognisers(core, language)
    base.candidates = [spec.id for spec in found[:5]]
    best = recommended_for(core, language)
    if found:
        base.model, base.reason, base.steer = found[0].id, "recommended", language
    else:
        base.detail = f"no installed recogniser speaks '{language}'"
    if best and best != base.model and not any(spec.id == best for spec in found):
        base.recommended = best
    return base
