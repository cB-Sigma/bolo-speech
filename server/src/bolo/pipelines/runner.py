"""Running pipelines: each step calls the same core operation its endpoint calls (bolo.core, bolo.denoise,
bolo.classify, ...), so models load, queue and unload through the scheduler as for any request.

A run keeps every step's timing, outputs and error; when a step fails the outputs made before it are kept.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.errors import BoloError, InvalidRequest, LLMError
from bolo.pipelines import conditions
from bolo.pipelines.definition import REF_RE, STEP_TYPES, Report, single_ref, validate
from bolo.pipelines.files import MEDIA_TYPES, check_name, extension

if TYPE_CHECKING:
    from bolo.core import Bolo, Transcript
    from bolo.pipelines.store import PipelineStore

log = logging.getLogger("bolo.pipelines")


@dataclass
class Audio:
    samples: Any
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate if self.rate else 0.0


class StopRun(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else ""


def new_run_id() -> str:
    return "run_" + secrets.token_hex(6)


@dataclass
class Scope:
    values: dict[str, dict[str, Any]]
    extra: dict[str, Any] = field(default_factory=dict)
    #: per-speaker steps write their files with this suffix
    suffix: str = ""

    def child(self, extra: dict[str, Any] | None = None, suffix: str = "", isolated: bool = False) -> Scope:
        values = dict(self.values) if isolated else self.values
        return Scope(values, {**self.extra, **(extra or {})}, suffix or self.suffix)


@dataclass
class Run:
    id: str
    pipeline: str | None
    version: int | None
    definition: dict[str, Any]
    inputs: dict[str, Any]
    vars: dict[str, Any]
    folder: Path | None
    trigger: str | None = None
    status: str = "queued"  # queued, running, ok, error, cancelled
    steps: list[dict[str, Any]] = field(default_factory=list)
    outputs: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    files: list[str] = field(default_factory=list)
    callback: dict[str, Any] | None = None
    job_id: str | None = None
    stop_reason: str | None = None
    #: what a live run did to its session (reply, respond, transfer, hangup), in order
    actions: list[dict[str, Any]] = field(default_factory=list)
    report: Report | None = None
    #: called after each step (progress for jobs and the console)
    on_step: Callable[[Run], None] | None = None

    def url(self, name: str) -> str:
        return f"/api/v1/pipelines/runs/{self.id}/files/{name}"

    def to_dict(self) -> dict[str, Any]:
        ms = (self.finished - self.started) * 1000.0 if self.finished and self.started else None
        out: dict[str, Any] = {
            "id": self.id,
            "object": "pipeline.run",
            "pipeline": self.pipeline,
            "version": self.version,
            "status": self.status,
            "created_at": _iso(self.created),
            "started_at": _iso(self.started) or None,
            "finished_at": _iso(self.finished) or None,
            "ms": round(ms, 1) if ms is not None else None,
            "input": {
                k: v
                for k, v in self.inputs.items()
                if k not in ("audio", "history") and not k.startswith("_") and v is not None
            },
            "steps": self.steps,
            "outputs": self.outputs,
            "error": self.error,
            "files": [{"name": n, "url": self.url(n)} for n in self.files],
        }
        if isinstance(self.inputs.get("audio"), Audio):
            out["input"]["duration"] = round(self.inputs["audio"].seconds, 3)
        if self.vars:
            out["vars"] = self.vars
        if self.trigger:
            out["trigger"] = self.trigger
        if self.callback is not None:
            out["callback"] = self.callback
        if self.job_id:
            out["job_id"] = self.job_id
        if self.stop_reason:
            out["stopped"] = self.stop_reason
        if self.actions:
            out["actions"] = self.actions
        if self.report is not None and self.report.warnings:
            out["warnings"] = [w.to_dict() for w in self.report.warnings]
        return out


# -- values -------------------------------------------------------------------------------------------------
def transcript_dict(t: Transcript) -> dict[str, Any]:
    return {
        "text": t.text,
        "language": t.language,
        "duration": round(t.duration, 3),
        "segments": [_segment(s) for s in t.segments],
        "words": [w.to_dict() for w in t.words],
        **({"redactions": t.redactions} if t.redactions is not None else {}),
    }


def _segment(s: Any) -> dict[str, Any]:
    out = {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
    if s.speaker:
        out["speaker"] = s.speaker
    return out


def as_text(value: Any) -> str:
    from bolo.core import Transcript

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    if isinstance(value, Transcript):
        if any(s.speaker for s in value.segments):
            return "\n".join(f"{s.speaker or '?'}: {s.text.strip()}" for s in value.segments)
        return value.text
    if isinstance(value, Audio):
        raise InvalidRequest("audio cannot be written into a text")
    return json.dumps(plain(value), ensure_ascii=False)


def plain(value: Any) -> Any:
    """``value`` as JSON data (audio as its length; files are written by :meth:`Runner.serialise`)."""
    from bolo.core import Transcript

    if isinstance(value, Transcript):
        return transcript_dict(value)
    if isinstance(value, Audio):
        return {"type": "audio", "seconds": round(value.seconds, 3), "sample_rate": value.rate}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value


def descend(value: Any, parts: list[str]) -> Any:
    for p in parts:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(p)
        elif isinstance(value, (list, tuple)):
            value = value[int(p)] if p.lstrip("-").isdigit() and -len(value) <= int(p) < len(value) else None
        elif hasattr(value, p) and not p.startswith("_"):
            value = getattr(value, p)
            if p == "segments" and isinstance(value, list):
                value = [_segment(s) for s in value]
        else:
            return None
    return value


# -- the runner ---------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, core: Bolo, store: PipelineStore | None = None) -> None:
        self.core = core
        self.store = store

    # references -------------------------------------------------------------------------------------
    def lookup(self, run: Run, path: str, scope: Scope) -> Any:
        parts = path.split(".")
        ns, rest = parts[0], parts[1:]
        if ns == "input":
            return descend(run.inputs, rest)
        if ns == "vars":
            return descend(run.vars, rest)
        if ns == "steps":
            return descend(scope.values.get(rest[0]) if rest else None, rest[1:])
        if ns == "run":
            return descend({"id": run.id}, rest)
        if ns == "pipeline":
            return descend({"id": run.pipeline, "version": run.version}, rest)
        if ns in scope.extra:
            return descend(scope.extra[ns], rest)
        raise InvalidRequest(f"unknown reference '{path}'")

    def resolve(self, run: Run, value: Any, scope: Scope) -> Any:
        ref = single_ref(value)
        if ref is not None:
            return self.lookup(run, ref, scope)
        if isinstance(value, str):
            if "{{" not in value:
                return value
            return REF_RE.sub(lambda m: as_text(self.lookup(run, m.group(1), scope)), value)
        if isinstance(value, list):
            return [self.resolve(run, v, scope) for v in value]
        if isinstance(value, dict):
            return {k: self.resolve(run, v, scope) for k, v in value.items()}
        return value

    def text(self, run: Run, value: Any, scope: Scope) -> str:
        return as_text(self.resolve(run, value, scope))

    def holds(self, run: Run, cond: dict[str, Any], scope: Scope) -> bool:
        return conditions.evaluate(cond, lambda v: self.resolve(run, v, scope), as_text)

    # files ---------------------------------------------------------------------------------------------
    def write_file(self, run: Run, name: str, data: bytes) -> str:
        check_name(name)
        if run.folder is None:
            return ""
        run.folder.mkdir(parents=True, exist_ok=True)
        (run.folder / name).write_bytes(data)
        if name not in run.files:
            run.files.append(name)
        return run.url(name)

    def serialise(self, run: Run, step_id: str, outputs: dict[str, Any], scope: Scope) -> dict[str, Any]:
        from bolo.audio.encode import encode_audio

        out: dict[str, Any] = {}
        for key, value in outputs.items():
            if isinstance(value, Audio):
                desc = plain(value)
                if run.folder is not None and len(value.samples):
                    name = f"{step_id}{scope.suffix}" + ("" if key == "audio" else f"-{key}") + ".wav"
                    data = encode_audio(np.asarray(value.samples, dtype=np.float32), value.rate, "wav")
                    desc.update(file=name, url=self.write_file(run, name, data))
                out[key] = desc
            elif key == "transcript":
                continue  # its fields are outputs of their own
            else:
                out[key] = plain(value)
        return out

    # running -------------------------------------------------------------------------------------------
    def prepare(
        self,
        definition: dict[str, Any],
        *,
        audio: Audio | None = None,
        text: str | None = None,
        filename: str | None = None,
        variables: dict[str, Any] | None = None,
        trigger: str | None = None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> Run:
        """Validate ``definition`` (raises :class:`PipelineInvalid`) and make the run record."""
        report = validate(definition, self.core)
        report.raise_if_invalid()
        merged = {**(definition.get("vars") or {}), **(variables or {})}
        run_id = new_run_id()
        folder = self.store.run_dir(run_id) if self.store is not None else None
        inputs: dict[str, Any] = {"audio": audio, "text": text, "filename": filename, **(extra_inputs or {})}
        if audio is not None:
            inputs["duration"] = round(audio.seconds, 3)
        run = Run(run_id, definition.get("id"), definition.get("version"), definition, inputs, merged, folder,
                  trigger=trigger, report=report)  # fmt: skip
        return run

    async def execute(self, run: Run) -> Run:
        """Run every step; never raises for a step's sake (the run's ``status`` and ``error`` say)."""
        assert run.report is not None
        run.status = "running"
        run.started = time.time()
        scope = Scope({})
        try:
            await self._steps(run, run.definition["steps"], run.report.steps, scope)
            run.status = "ok"
        except StopRun as stop:
            run.status = "ok"
            run.stop_reason = stop.reason
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.error = {"step": None, "message": "the run was cancelled", "code": "cancelled"}
            self._finish(run, scope)
            raise
        except _StepFailed as failed:
            run.status = "error"
            run.error = failed.error
        self._finish(run, scope)
        return run

    def _finish(self, run: Run, scope: Scope) -> None:
        if run.status in ("ok", "error"):
            try:
                run.outputs = self._outputs(run, scope)
            except Exception as exc:
                run.outputs = None
                if run.status == "ok":
                    run.status = "error"
                    run.error = {"step": None, "message": f"the outputs could not be made: {_message(exc)}",
                                 "code": "outputs"}  # fmt: skip
        run.finished = time.time()
        if self.store is not None:
            try:
                self.store.save_run(run.to_dict())
                self.store.prune()
            except OSError as exc:
                log.warning("pipeline run %s could not be saved: %s", run.id, exc)

    def _outputs(self, run: Run, scope: Scope) -> dict[str, Any]:
        wanted = run.definition.get("outputs")
        if not wanted:
            last = next((s for s in reversed(run.steps) if s["status"] == "ok" and s.get("depth", 0) == 0), None)
            return dict(last["output"]) if last and isinstance(last.get("output"), dict) else {}
        out: dict[str, Any] = {}
        for name, value in wanted.items():
            got = self.resolve(run, value, scope)
            if isinstance(got, Audio):
                desc = self.serialise(run, f"output-{name}", {"audio": got}, scope)["audio"]
                out[name] = desc
            else:
                out[name] = plain(got)
        return out

    async def _steps(
        self, run: Run, raws: list[dict[str, Any]], entries: list[dict[str, Any]], scope: Scope, depth: int = 0
    ) -> None:
        for i, (raw, entry) in enumerate(zip(raws, entries)):
            try:
                await self._step(run, raw, entry, scope, depth)
            except (_StepFailed, StopRun) as exc:
                status = "pending" if depth == 0 and isinstance(exc, _StepFailed) else "skipped"
                for later in entries[i + 1 :]:
                    self._mark(run, later, status, depth, scope)
                raise

    def _mark(self, run: Run, entry: dict[str, Any], status: str, depth: int, scope: Scope) -> None:
        record = {"id": entry["id"], "type": entry["type"], "model": entry.get("model"), "status": status,
                  "ms": None, "output": None, "error": None, "depth": depth}  # fmt: skip
        if "speaker" in scope.extra:
            record["speaker"] = scope.extra["speaker"].get("label")
        run.steps.append(record)
        for key in ("then", "else", "default", "steps"):
            for nested in entry.get(key) or []:
                self._mark(run, nested, status, depth + 1, scope)
        for nested_list in (entry.get("cases") or {}).values():
            for nested in nested_list:
                self._mark(run, nested, status, depth + 1, scope)

    async def _step(self, run: Run, raw: dict[str, Any], entry: dict[str, Any], scope: Scope, depth: int) -> None:
        st = STEP_TYPES[entry["type"]]
        record: dict[str, Any] = {"id": entry["id"], "type": st.name, "model": entry.get("model"), "status": "running",
                                  "ms": None, "output": None, "error": None, "depth": depth}  # fmt: skip
        if "speaker" in scope.extra:
            record["speaker"] = scope.extra["speaker"].get("label")
        when = raw.get("when")
        if when is not None and not self.holds(run, when, scope):
            scope.values[entry["id"]] = {}
            self._mark(run, entry, "skipped", depth, scope)
            return
        run.steps.append(record)
        started = time.perf_counter()
        try:
            inputs = self._inputs(run, entry, scope)
            options = raw.get("options") or {}
            handler = HANDLERS[st.name]
            outputs = await handler(self, run, entry, inputs, options, scope, depth)
            scope.values[entry["id"]] = outputs
            record["output"] = self.serialise(run, entry["id"], outputs, scope)
            record["status"] = "ok"
            if entry.get("model") is None and outputs.get("_model"):
                record["model"] = outputs.pop("_model")
            record["output"].pop("_model", None)
        except (StopRun, _StepFailed):
            record["status"] = "ok"
            raise
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            if not isinstance(exc, BoloError):
                log.exception("pipeline step %s failed", entry["id"])
            record["status"] = "error"
            record["error"] = {"message": _message(exc), "code": getattr(exc, "code", "internal_error")}
            scope.values[entry["id"]] = {}
            if raw.get("on_error") != "continue":
                raise _StepFailed({"step": entry["id"], **record["error"]}) from exc
        finally:
            record["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            if run.on_step is not None:
                run.on_step(run)

    def _inputs(self, run: Run, entry: dict[str, Any], scope: Scope) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in entry["inputs"].items():
            out[name] = self.resolve(run, value, scope)
        for name, candidates in (entry.get("defaults") or {}).items():
            for c in candidates:
                got = self.resolve(run, c, scope)
                if got not in (None, "", []):
                    out[name] = got
                    out.setdefault("_defaulted", []).append(name)
                    break
        return out


class _StepFailed(Exception):
    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error["message"])
        self.error = error


def _message(exc: BaseException) -> str:
    return exc.message if isinstance(exc, BoloError) else f"{type(exc).__name__}: {exc}"


# -- helpers for handlers -----------------------------------------------------------------------------------
def need_audio(inputs: dict[str, Any], name: str = "audio") -> Audio:
    value = inputs.get(name)
    if not isinstance(value, Audio):
        raise InvalidRequest(f"this step needs audio ('{name}'), and the run has none: send a file or url")
    if not len(value.samples):
        raise InvalidRequest(f"the audio '{name}' is empty")
    return value


def need_text(inputs: dict[str, Any], name: str = "text") -> str:
    value = inputs.get(name)
    text = as_text(value)
    if not text.strip():
        raise InvalidRequest(f"this step needs a text ('{name}'), and it is empty")
    return text


def as_transcript(value: Any) -> Transcript:
    from bolo.core import Transcript

    if isinstance(value, Transcript):
        return copy.deepcopy(value)
    text = as_text(value)
    if not text.strip():
        raise InvalidRequest("this step needs a text, and it is empty")
    return Transcript(text=text, language=None, duration=0.0)


def number(v: Any, name: str) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise InvalidRequest(f"{name} must be a number, not {v!r}") from None


def integer(v: Any, name: str) -> int | None:
    n = number(v, name)
    return None if n is None else int(n)


def flag(v: Any) -> bool | None:
    if v in (None, ""):
        return None
    return conditions.truthy(v)


def text_list(v: Any) -> list[str]:
    if v in (None, ""):
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [as_text(x).strip() for x in v if as_text(x).strip()]


def transcript_outputs(t: Transcript) -> dict[str, Any]:
    return {
        "text": t.text,
        "language": t.language,
        "duration": round(t.duration, 3),
        "segments": [_segment(s) for s in t.segments],
        "words": [w.to_dict() for w in t.words],
        "transcript": t,
    }


def label_by_turns(t: Transcript, turns: list[dict[str, Any]]) -> None:
    """Each segment gets the speaker who talks most during it."""
    from dataclasses import replace

    for i, seg in enumerate(t.segments):
        best, most = None, 0.0
        for turn in turns:
            try:
                a, b = float(turn["start"]), float(turn["end"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(b, seg.end) - max(a, seg.start)
            if overlap > most:
                best, most = str(turn.get("speaker")), overlap
        if best is not None:
            t.segments[i] = replace(seg, speaker=best)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            doc = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(doc, dict):
            return doc
    return None


async def llm_complete(
    core: Bolo,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    client: Any = None,
) -> str:
    client = client or core.llm
    body: dict[str, Any] = {"model": model or core.settings.llm_model, "messages": messages}
    if max_tokens:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    doc = await client.chat(body)
    try:
        content = doc["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMError("the language model's answer has no message") from None
    return str(content)


def wav_part(audio: Audio) -> dict[str, Any]:
    from bolo.audio.encode import encode_audio
    from bolo.audio.resample import resample

    x = resample(np.asarray(audio.samples, dtype=np.float32), audio.rate, 16000)
    data = encode_audio(x, 16000, "wav")
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"}}


Handler = Callable[[Runner, Run, dict[str, Any], dict[str, Any], dict[str, Any], Scope, int], Awaitable[dict[str, Any]]]
HANDLERS: dict[str, Handler] = {}


def handler(name: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        HANDLERS[name] = fn
        return fn

    return register


def media_type(name: str) -> str:
    return MEDIA_TYPES.get(extension(name), "application/octet-stream")


# the handlers register themselves
from bolo.pipelines import live as _live  # noqa: E402,F401
from bolo.pipelines import steps as _steps  # noqa: E402,F401
