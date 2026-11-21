"""Pipeline definitions: step types, references between steps, and validation (docs/pipelines.md).

A definition is plain data (JSON, or YAML): named steps of a type, each with a model, inputs and options. Inputs and
string options may hold references to earlier steps' outputs (``{{steps.stt.text}}``). :func:`validate` checks a
definition completely before anything runs and lists every problem with the path of the field it is in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo

#: value types (``transcript`` is accepted wherever text is)
TYPES = ("audio", "text", "transcript", "turns", "number", "boolean", "list", "object", "any")
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
PIPELINE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RESERVED_IDS = ("runs", "templates", "validate", "step-types", "run")
REF_RE = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")
PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$")
MODES = ("file", "live")
MAX_STEPS = 200


@dataclass(frozen=True)
class Slot:
    """An input of a step type. ``default``: ``audio`` (the closest earlier audio, else the run's), ``text`` (the
    closest earlier text, else the run's), ``turns`` (the closest earlier diarization), or None (no default)."""

    name: str
    type: str
    required: bool = True
    default: str | None = None
    help: str = ""


@dataclass(frozen=True)
class Option:
    name: str
    type: str  # text, number, integer, boolean, list, object, steps, condition, cases
    help: str = ""
    required: bool = False
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepType:
    name: str
    #: the catalog task of the step's model (None: no model; "llm": the configured language model)
    kind: str | None
    group: str  # model, text, control, action
    help: str
    inputs: tuple[Slot, ...] = ()
    options: tuple[Option, ...] = ()
    outputs: dict[str, str] = field(default_factory=dict)
    #: may run per user turn of a live pipeline
    live: bool = False
    #: only in live pipelines
    live_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "kind": self.kind,
            "group": self.group,
            "help": self.help,
            "inputs": [
                {"name": s.name, "type": s.type, "required": s.required, "default": s.default, "help": s.help}
                for s in self.inputs
            ],
            "options": [
                {"name": o.name, "type": o.type, "help": o.help, "required": o.required, "choices": list(o.choices)}
                for o in self.options
            ],
            "outputs": dict(self.outputs),
            "live": self.live,
            "live_only": self.live_only,
        }


AUDIO = Slot("audio", "audio", default="audio")
TEXT = Slot("text", "text", default="text")
TRANSCRIPT_OUT = {"text": "text", "transcript": "transcript"}

STEP_TYPES: dict[str, StepType] = {
    t.name: t
    for t in (
        StepType("denoise", "denoise", "model", "Remove background noise (speech enhancement).", (AUDIO,),
                 (Option("strength", "number", "0 to 1: the denoised audio mixed with the original"),),
                 {"audio": "audio"}),
        StepType("vad", "vad", "model", "Where speech is (voice activity).", (AUDIO,),
                 (Option("threshold", "number"), Option("min_silence_ms", "number"), Option("min_speech_ms", "number"),
                  Option("speech_pad_ms", "number")),
                 {"segments": "list", "speech_seconds": "number", "count": "number"}),
        StepType("diarization", "diarization", "model", "Who spoke when.", (AUDIO,),
                 (Option("num_speakers", "integer"), Option("min_speakers", "integer"), Option("max_speakers", "integer"),
                  Option("threshold", "number")),
                 {"turns": "turns", "speakers": "list", "num_speakers": "number"}),
        StepType("separation", "separation", "model", "Separate overlapped speech into one track per speaker.", (AUDIO,),
                 (Option("diarization_model", "text"), Option("min_overlap_s", "number"),
                  Option("allow_narrowband", "boolean")),
                 {"turns": "turns", "separation": "object"}),
        StepType("speaker-identification", "speaker-embedding", "model", "Which enrolled speaker is this?", (AUDIO,),
                 (Option("speakers", "list", "enrolled speaker ids or names (default: all)"), Option("threshold", "number")),
                 {"speaker": "text", "speaker_id": "text", "similarity": "number", "match": "boolean"}),
        StepType("speaker-enrolment", "speaker-embedding", "model", "Enrol the recording as a speaker (or add to one).",
                 (AUDIO, Slot("name", "text", help="the speaker's name")), (),
                 {"speaker_id": "text", "clips": "number"}),
        StepType("stt", "stt", "model", "Speech to text.",
                 (AUDIO, Slot("turns", "turns", required=False,
                              help="speaker turns: each segment is labelled with its speaker")),
                 (Option("language", "text"), Option("prompt", "text"), Option("hotwords", "text"),
                  Option("temperature", "number"), Option("diarize", "boolean", "diarize inside the recogniser"),
                  Option("vad_filter", "boolean"), Option("word_timestamps", "boolean")),
                 {"text": "text", "language": "text", "duration": "number", "segments": "list", "words": "list",
                  "transcript": "transcript"}, live=False),
        StepType("translation", "stt", "model",
                 "Translate: a text through the language model, or a recording through a recogniser that translates.",
                 (Slot("text", "text", required=False, default="text"),
                  Slot("audio", "audio", required=False, default="audio")),
                 (Option("target_language", "text", "default en"), Option("source_language", "text")),
                 {"text": "text", "language": "text"}, live=True),
        StepType("align", "align", "model", "Word timestamps of a known text in the recording.", (AUDIO, TEXT),
                 (Option("language", "text"),), {"words": "list", "text": "text"}),
        StepType("punctuation", "punctuation", "model", "Restore punctuation and capitals.", (TEXT,), (),
                 dict(TRANSCRIPT_OUT), live=True),
        StepType("itn", "itn", "model", "Write numbers, money, dates and times as figures.", (TEXT,), (),
                 dict(TRANSCRIPT_OUT), live=True),
        StepType("pii", "pii", "model", "Mask personal information.", (TEXT,),
                 (Option("entities", "list", "PERSON, PHONE_NUMBER, ... (default: all the model finds)"),),
                 {**TRANSCRIPT_OUT, "redactions": "list", "count": "number"}, live=True),
        StepType("lid", "lid", "model", "Which language is spoken.", (AUDIO,), (),
                 {"language": "text", "probability": "number", "languages": "list"}, live=True),
        StepType("audio-tagging", "audio-tagging", "model", "Sound events and tags (music, speech, applause, ...).",
                 (AUDIO,), (Option("top_k", "integer"), Option("threshold", "number"), Option("labels", "list")),
                 {"tags": "list", "top": "text"}),
        StepType("emotion", "emotion", "model", "How the speaker sounds.", (AUDIO,), (),
                 {"emotion": "text", "score": "number", "emotions": "list"}, live=True),
        StepType("audio-llm", "audio-llm", "model", "Ask an audio language model about the recording.",
                 (AUDIO, Slot("prompt", "text", help="the question")),
                 (Option("max_tokens", "integer"), Option("temperature", "number")), {"text": "text"}),
        StepType("decision", "decision", "model",
                 "A question with a fixed set of answers, about a text or a recording.",
                 (Slot("text", "text", required=False, default="text"),
                  Slot("audio", "audio", required=False, default="audio")),
                 (Option("question", "text", required=True), Option("choices", "list", "default: yes, no"),
                  Option("instructions", "text", "for the language model")),
                 {"answer": "text", "confidence": "number", "reason": "text"}, live=True),
        StepType("decisions", "decision", "model",
                 "Several typed questions (yes/no, a choice, a score) about a text and a recording, in one pass of a "
                 "decision model (bolo.decide).",
                 (Slot("text", "text", required=False, default="text"),
                  Slot("audio", "audio", required=False, default="audio")),
                 (Option("questions", "object", "id -> {type: yes_no | choice | score, question, options | levels}",
                         required=True),),
                 {"answers": "object", "probabilities": "object", "model": "text"}, live=True),
        StepType("tts", "tts", "model", "Text to speech.", (TEXT,),
                 (Option("voice", "text", "a model voice or a saved (cloned) voice"), Option("speed", "number"),
                  Option("language", "text"), Option("instructions", "text"), Option("watermark", "boolean")),
                 {"audio": "audio"}),
        StepType("voice-conversion", "voice-conversion", "model", "Say the same words in another voice.", (AUDIO,),
                 (Option("voice", "text", required=True),), {"audio": "audio"}),
        StepType("watermark", "watermark", "model", "Add an inaudible watermark.", (AUDIO,), (), {"audio": "audio"}),
        StepType("watermark-detect", "watermark", "model", "Is the recording watermarked?", (AUDIO,), (),
                 {"detected": "boolean", "score": "number"}),
        # -- text and control --------------------------------------------------------------------
        StepType("template", None, "text", "Build a text from references.", (),
                 (Option("template", "text", required=True),), {"text": "text"}, live=True),
        StepType("llm", "llm", "text", "Ask the configured language model.", (),
                 (Option("prompt", "text", required=True), Option("system", "text"),
                  Option("json", "boolean", "ask for a JSON object and parse it"), Option("max_tokens", "integer"),
                  Option("temperature", "number")),
                 {"text": "text", "json": "object"}, live=True),
        StepType("filter", None, "control", "Keep the elements of a list for which a condition holds.",
                 (Slot("items", "list"),), (Option("where", "condition", required=True),),
                 {"items": "list", "count": "number"}, live=True),
        StepType("condition", None, "control", "Run some steps or others.", (),
                 (Option("if", "condition", required=True), Option("then", "steps"), Option("else", "steps")),
                 {"branch": "text"}, live=True),
        StepType("switch", None, "control", "Run the steps of the case a value matches.", (Slot("value", "any"),),
                 (Option("cases", "cases", required=True), Option("default", "steps")), {"branch": "text"}, live=True),
        StepType("split-speakers", None, "control", "Run steps once per speaker, on that speaker's audio.",
                 (AUDIO, Slot("turns", "turns", default="turns")),
                 (Option("steps", "steps", required=True), Option("min_seconds", "number")),
                 {"speakers": "list", "count": "number"}),
        StepType("stop", None, "control", "End the run here.", (), (Option("reason", "text"),), {}, live=True),
        # -- actions -----------------------------------------------------------------------------
        StepType("webhook", None, "action", "Call a URL (signed).", (Slot("body", "any", required=False),),
                 (Option("url", "text", required=True), Option("method", "text", choices=("POST", "PUT", "GET")),
                  Option("headers", "object"), Option("timeout_s", "number")),
                 {"status": "number", "ok": "boolean", "body": "text", "json": "object"}, live=True),
        StepType("save", None, "action", "Write a value to a file of the run.", (Slot("value", "any", default="any"),),
                 (Option("name", "text", required=True),), {"file": "text", "url": "text"}),
        StepType("reply", None, "action", "Say a text to the caller.", (TEXT,), (), {"text": "text"},
                 live=True, live_only=True),
        StepType("respond", None, "action",
                 "Let the session's language model answer (streamed, with its tools), with more instructions for "
                 "this turn if given.", (), (Option("instructions", "text"),), {"text": "text"},
                 live=True, live_only=True),
        StepType("tag", None, "action", "Tag the call or the session (its record and traces).", (),
                 (Option("value", "text", required=True),), {}, live=True, live_only=True),
        StepType("notify", None, "action", "Tell the session's client (a pipeline.action event), nothing more.", (),
                 (Option("message", "text"),), {}, live=True, live_only=True),
        StepType("transfer", None, "action", "Transfer the call to a person (a number or SIP address).", (),
                 (Option("to", "text", "default: the agent's transfer_number"), Option("say", "text")), {},
                 live=True, live_only=True),
        StepType("hangup", None, "action", "End the call or session.", (), (Option("say", "text"),), {},
                 live=True, live_only=True),
    )
}  # fmt: skip

#: step types whose outputs feed the default ``audio`` and ``text`` of later steps
AUDIO_PRODUCERS = {"denoise", "tts", "voice-conversion", "watermark"}


def nested_keys(type_name: str) -> list[str]:
    """Option names of a control step that hold steps."""
    return {"condition": ["then", "else"], "split-speakers": ["steps"], "switch": ["default"]}.get(type_name, [])


# -- references --------------------------------------------------------------------------------------------
def refs_in(value: Any) -> list[str]:
    """Every reference path in a value (strings, and lists and objects of them)."""
    if isinstance(value, str):
        return [m.group(1) for m in REF_RE.finditer(value)]
    if isinstance(value, list):
        return [r for v in value for r in refs_in(v)]
    if isinstance(value, dict):
        return [r for v in value.values() for r in refs_in(v)]
    return []


def single_ref(value: Any) -> str | None:
    """The path when ``value`` is exactly one reference (it then keeps its type)."""
    if isinstance(value, str):
        m = REF_RE.fullmatch(value.strip())
        if m:
            return m.group(1)
    return None


def accepts(slot_type: str, value_type: str) -> bool:
    if slot_type == "any" or value_type == "any":
        return True
    if slot_type == value_type:
        return True
    if slot_type == "text":
        return value_type in ("transcript", "number", "boolean")
    if slot_type == "list":
        return value_type == "turns"
    if slot_type == "turns":
        return value_type == "list"
    if slot_type == "object":
        return value_type == "transcript"
    return False


# -- validation --------------------------------------------------------------------------------------------
@dataclass
class Problem:
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


@dataclass
class _Seen:
    type: StepType
    #: may not have run (inside a branch)
    conditional: bool


@dataclass
class _Scope:
    steps: dict[str, _Seen]
    last_audio: list[str]
    last_text: list[str]
    last_turns: str | None
    extra: dict[str, str]  # namespace -> type of every path under it ("speaker", "item")
    live: bool
    vars: dict[str, Any]

    def child(self, extra: dict[str, str] | None = None) -> _Scope:
        return _Scope(dict(self.steps), list(self.last_audio), list(self.last_text), self.last_turns,
                      {**self.extra, **(extra or {})}, self.live, self.vars)  # fmt: skip


@dataclass
class Report:
    errors: list[Problem] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "steps": self.steps,
        }

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise PipelineInvalid(self)


class PipelineInvalid(InvalidRequest):
    def __init__(self, report: Report) -> None:
        first = report.errors[0]
        more = f" (and {len(report.errors) - 1} more)" if len(report.errors) > 1 else ""
        super().__init__(f"the pipeline is not valid: {first.path}: {first.message}{more}", param=first.path)
        self.report = report


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "any"


class Validator:
    def __init__(self, core: Bolo | None) -> None:
        self.core = core
        self.report = Report()
        self.ids: set[str] = set()
        #: ids of steps that run per speaker (not visible outside their split-speakers step)
        self.hidden: set[str] = set()
        #: every id written in the definition (to tell "comes later" from "does not exist")
        self.declared: set[str] = set()

    def error(self, path: str, message: str) -> None:
        self.report.errors.append(Problem(path, message))

    def warn(self, path: str, message: str) -> None:
        self.report.warnings.append(Problem(path, message))

    # -- references --------------------------------------------------------------------------------
    def ref_type(self, ref: str, scope: _Scope, path: str) -> str | None:
        """The type of what ``ref`` names, or None (an error was recorded)."""
        if not PATH_RE.match(ref):
            self.error(path, f"'{{{{{ref}}}}}' is not a reference (namespace.name, e.g. steps.stt.text)")
            return None
        parts = ref.split(".")
        ns = parts[0]
        if ns == "input":
            known = {"audio": "audio", "text": "text", "filename": "text", "duration": "number", "call": "object",
                     "session": "object"}  # fmt: skip
            if scope.live:
                known.update(history="list", turn="number", session="object")
            if len(parts) < 2 or parts[1] not in known:
                self.error(path, f"input has {', '.join(known)}; '{ref}' is not one of them")
                return None
            return known[parts[1]] if len(parts) == 2 else "any"
        if ns == "vars":
            if len(parts) < 2:
                self.error(path, "name a variable: vars.<name>")
                return None
            if parts[1] not in scope.vars:
                self.warn(path, f"'{parts[1]}' is not in the pipeline's vars: it must be given when the pipeline runs")
                return "any"
            return _value_type(scope.vars[parts[1]]) if len(parts) == 2 else "any"
        if ns in ("run", "pipeline"):
            return "any"
        if ns in scope.extra:
            return scope.extra[ns] if len(parts) == 1 else ("audio" if ref == "speaker.audio" else "any")
        if ns == "steps":
            if len(parts) < 3:
                self.error(path, f"'{ref}': name a step and one of its outputs (steps.<id>.<output>)")
                return None
            seen = scope.steps.get(parts[1])
            if seen is None:
                later = parts[1] in self.ids or parts[1] in self.declared
                self.error(path, f"step '{parts[1]}' " + (
                    "is not available here (it comes later, or inside a branch or fan-out this step is not in)"
                    if later else "does not exist"))  # fmt: skip
                return None
            outputs = seen.type.outputs
            if parts[2] not in outputs:
                names = ", ".join(outputs) or "none"
                self.error(path, f"step '{parts[1]}' ({seen.type.name}) has no output '{parts[2]}' (it has: {names})")
                return None
            if seen.conditional:
                self.warn(path, f"step '{parts[1]}' may not have run (it is in a branch); its outputs may be empty")
            return outputs[parts[2]] if len(parts) == 3 else "any"
        namespaces = ["input", "vars", "steps", "run", "pipeline", *scope.extra]
        self.error(path, f"unknown namespace '{ns}' (use {', '.join(namespaces)})")
        return None

    def check_text(self, value: Any, scope: _Scope, path: str) -> None:
        """References in a string option or a template: every one must exist and be writable as text."""
        for ref in refs_in(value):
            t = self.ref_type(ref, scope, path)
            if t == "audio" and single_ref(value) is None:
                self.error(path, f"'{{{{{ref}}}}}' is audio and cannot be written into a text")

    def slot_value(self, slot: Slot, given: Any, scope: _Scope, path: str) -> tuple[Any, str]:
        """Check an input; returns what it resolves to (for the report) and the type of the value."""
        ref = single_ref(given)
        if ref is not None:
            t = self.ref_type(ref, scope, path)
            if t is not None and not accepts(slot.type, t):
                self.error(path, f"'{slot.name}' takes {slot.type}, but '{{{{{ref}}}}}' is {t}")
            return given, t or "any"
        if isinstance(given, str):
            self.check_text(given, scope, path)
            if slot.type not in ("text", "any"):
                self.error(path, f"'{slot.name}' takes {slot.type}: give a reference ({{{{steps.<id>.<output>}}}})")
            return given, "text"
        t = _value_type(given)
        if not accepts(slot.type, t):
            self.error(path, f"'{slot.name}' takes {slot.type}, not {t}")
        return given, t

    def default_for(self, slot: Slot, scope: _Scope) -> list[str]:
        if slot.default == "audio":
            return [*scope.last_audio[-1:], "input.audio"]
        if slot.default == "text":
            return [*scope.last_text[-1:], "input.text"]
        if slot.default == "turns":
            return [scope.last_turns] if scope.last_turns else []
        if slot.default == "any":
            return [*scope.last_text[-1:], *scope.last_audio[-1:]][:1] or ["input.text"]
        return []

    # -- models ---------------------------------------------------------------------------------------
    def check_model(self, st: StepType, model: Any, path: str) -> str | None:
        if st.kind is None:
            if model not in (None, "", "auto"):
                self.error(path, f"a {st.name} step takes no model")
            return None
        if model is not None and not isinstance(model, str):
            self.error(path, "model must be a model id or 'auto'")
            return None
        wanted = None if model in (None, "", "auto") else model
        if st.kind == "llm":
            if self.core is not None and not self.core.llm.configured:
                self.warn(path, "no language model is configured (llm_base_url): this step will fail")
            if wanted:
                return wanted  # a model name of the configured endpoint
            return self.core.settings.llm_model if self.core is not None else None
        if self.core is None:
            return wanted
        if st.name == "translation" and wanted is None:
            return None  # the language model (text) or the default recogniser (audio)
        if st.kind == "decision" and wanted in (None, "llm"):
            return wanted  # chosen when it runs: the default decision model if installed, else the language model
        if st.kind == "audio-llm" and wanted is None:
            return None  # the audio_llm_* settings
        try:
            spec = self.core.resolve(wanted, st.kind)
        except BoloError as exc:
            if wanted is None:
                self.warn(path, f"no default {st.kind} model: {exc.message}")
                return None
            self.error(path, exc.message)
            return None
        if spec.task != st.kind:
            self.error(path, f"'{spec.id}' is a {spec.task} model; a {st.name} step needs a {st.kind} model")
            return spec.id
        try:
            installed = self.core.manager.is_installed(spec)
        except Exception:
            installed = True
        if not installed:
            how = "it is downloaded on first use" if self.core.settings.auto_pull else f"run: bolo pull {spec.id}"
            self.warn(path, f"'{spec.id}' is not installed; {how}")
        return spec.id

    # -- options --------------------------------------------------------------------------------------
    def check_options(self, st: StepType, options: Any, scope: _Scope, path: str, step_id: str) -> None:
        if options is None:
            options = {}
        if not isinstance(options, dict):
            self.error(path, "options must be an object")
            return
        known = {o.name: o for o in st.options}
        for name in options:
            if name not in known:
                hint = f" (it has: {', '.join(known)})" if known else ""
                self.error(f"{path}.{name}", f"a {st.name} step has no option '{name}'{hint}")
        for opt in st.options:
            p = f"{path}.{opt.name}"
            if opt.name not in options or options[opt.name] is None:
                if opt.required:
                    self.error(p, f"a {st.name} step needs '{opt.name}'")
                continue
            v = options[opt.name]
            if opt.type == "steps":
                if not isinstance(v, list):
                    self.error(p, "must be a list of steps")
                continue
            if opt.type == "cases":
                if not isinstance(v, dict) or not v:
                    self.error(p, "must be an object: case value -> list of steps")
                continue
            if opt.type == "condition":
                self.check_condition(v, scope.child({"item": "any"}) if st.name == "filter" else scope, p)
                continue
            if single_ref(v) is not None or (isinstance(v, str) and refs_in(v) and opt.type == "text"):
                self.check_text(v, scope, p)
                continue
            ok = {
                "text": isinstance(v, str),
                "number": isinstance(v, (int, float)) and not isinstance(v, bool),
                "integer": isinstance(v, int) and not isinstance(v, bool),
                "boolean": isinstance(v, bool),
                "list": isinstance(v, (list, str)),
                "object": isinstance(v, dict),
            }.get(opt.type, True)
            if not ok:
                self.error(p, f"must be {'an' if opt.type[0] in 'aeiou' else 'a'} {opt.type}")
            elif opt.choices and str(v).upper() not in opt.choices:
                self.error(p, f"must be one of {', '.join(opt.choices)}")
            else:
                self.check_text(v, scope, p)
        if st.name == "webhook" and isinstance(options.get("url"), str):
            url = options["url"]
            if not refs_in(url) and not url.startswith(("http://", "https://")):
                self.error(f"{path}.url", "must be an http:// or https:// address")
        if st.name == "save" and isinstance(options.get("name"), str) and not refs_in(options["name"]):
            from bolo.pipelines.files import check_name

            try:
                check_name(options["name"])
            except InvalidRequest as exc:
                self.error(f"{path}.name", exc.message)

    def check_condition(self, cond: Any, scope: _Scope, path: str) -> None:
        if not isinstance(cond, dict) or not cond:
            self.error(path, 'a condition is an object: {"value": "{{...}}", "equals": ...}, all, any or not')
            return
        if "all" in cond or "any" in cond:
            key = "all" if "all" in cond else "any"
            items = cond[key]
            if not isinstance(items, list) or not items:
                self.error(f"{path}.{key}", "must be a non-empty list of conditions")
                return
            for i, c in enumerate(items):
                self.check_condition(c, scope, f"{path}.{key}[{i}]")
            return
        if "not" in cond:
            self.check_condition(cond["not"], scope, f"{path}.not")
            return
        if "value" not in cond:
            self.error(path, "a condition needs 'value' (a reference) and an operator")
            return
        self.check_text(cond["value"], scope, f"{path}.value")
        from bolo.pipelines.conditions import OPERATORS

        ops = [k for k in cond if k != "value"]
        if len(ops) != 1 or ops[0] not in OPERATORS:
            self.error(path, f"a condition needs exactly one operator of: {', '.join(OPERATORS)}")
            return
        op = ops[0]
        arg = cond[op]
        if op in ("in", "not_in") and not isinstance(arg, list):
            self.error(f"{path}.{op}", "must be a list")
        not_number = isinstance(arg, bool) or not isinstance(arg, (int, float))
        if op in ("gt", "gte", "lt", "lte") and not_number and single_ref(arg) is None:
            self.error(f"{path}.{op}", "must be a number")
        if op == "matches":
            try:
                re.compile(str(arg))
            except re.error as exc:
                self.error(f"{path}.{op}", f"not a regular expression: {exc}")
        if op in ("exists", "truthy") and not isinstance(arg, bool):
            self.error(f"{path}.{op}", "must be true or false")
        self.check_text(arg, scope, f"{path}.{op}")

    # -- steps ----------------------------------------------------------------------------------------
    def steps(self, steps: Any, scope: _Scope, path: str, conditional: bool, out: list[dict[str, Any]]) -> None:
        if not isinstance(steps, list):
            self.error(path, "must be a list of steps")
            return
        counter: dict[str, int] = {}
        for i, raw in enumerate(steps):
            p = f"{path}[{i}]"
            if not isinstance(raw, dict):
                self.error(p, "a step is an object with at least a type")
                continue
            for key in raw:
                if key not in ("id", "type", "model", "inputs", "options", "when", "on_error", "name", "notes"):
                    self.error(f"{p}.{key}", f"unknown field '{key}' (a step has id, type, model, inputs, options, "
                                             "when, on_error)")  # fmt: skip
            type_name = raw.get("type")
            st = STEP_TYPES.get(type_name) if isinstance(type_name, str) else None
            if st is None:
                self.error(f"{p}.type", f"unknown step type {type_name!r} (types: {', '.join(STEP_TYPES)})")
                continue
            sid = raw.get("id")
            if sid is None:
                base = st.name.replace("-", "_")
                counter[base] = counter.get(base, 0) + 1
                sid = base if counter[base] == 1 else f"{base}_{counter[base]}"
                while sid in self.ids:
                    counter[base] += 1
                    sid = f"{base}_{counter[base]}"
            elif not isinstance(sid, str) or not ID_RE.match(sid):
                self.error(f"{p}.id", "an id starts with a letter and holds letters, digits, '_' and '-' (64 at most)")
                continue
            if sid in self.ids:
                self.error(f"{p}.id", f"the id '{sid}' is used twice")
                continue
            self.ids.add(sid)
            if scope.live and not st.live:
                self.error(f"{p}.type", f"a {st.name} step cannot run per turn of a live pipeline")
            if st.live_only and not scope.live:
                self.error(f"{p}.type", f"a {st.name} step only runs in a live pipeline (mode: live)")
            if raw.get("on_error") not in (None, "fail", "continue"):
                self.error(f"{p}.on_error", "must be 'fail' or 'continue'")
            if raw.get("when") is not None:
                self.check_condition(raw["when"], scope, f"{p}.when")
            model = self.check_model(st, raw.get("model"), f"{p}.model")
            inputs = raw.get("inputs") or {}
            resolved: dict[str, Any] = {}
            defaults: dict[str, list[str]] = {}
            if not isinstance(inputs, dict):
                self.error(f"{p}.inputs", "inputs must be an object: slot -> reference or value")
                inputs = {}
            slots = {s.name: s for s in st.inputs}
            for name in inputs:
                if name not in slots:
                    have = ", ".join(slots) or "none"
                    self.error(f"{p}.inputs.{name}", f"a {st.name} step has no input '{name}' (it has: {have})")
            for slot in st.inputs:
                if slot.name in inputs and inputs[slot.name] is not None:
                    value, _ = self.slot_value(slot, inputs[slot.name], scope, f"{p}.inputs.{slot.name}")
                    resolved[slot.name] = value
                    continue
                candidates = self.default_for(slot, scope)
                if candidates:
                    defaults[slot.name] = ["{{" + c + "}}" for c in candidates]
                elif slot.required:
                    hint = {"turns": "add a diarization step before it, or give inputs.turns"}.get(
                        slot.type, f"give inputs.{slot.name}"
                    )
                    self.error(f"{p}.inputs.{slot.name}", f"a {st.name} step needs {slot.type} '{slot.name}': {hint}")
            options = raw.get("options") or {}
            self.check_options(st, options, scope, f"{p}.options", sid)
            entry: dict[str, Any] = {"id": sid, "type": st.name, "model": model, "inputs": resolved,
                                     "defaults": defaults, "outputs": dict(st.outputs), "path": p}  # fmt: skip
            # nested steps
            if isinstance(options, dict):
                if st.name == "condition":
                    for key in ("then", "else"):
                        if isinstance(options.get(key), list):
                            entry[key] = []
                            self.steps(options[key], scope.child(), f"{p}.options.{key}", True, entry[key])
                elif st.name == "switch" and isinstance(options.get("cases"), dict):
                    entry["cases"] = {}
                    for case, nested in options["cases"].items():
                        entry["cases"][case] = []
                        self.steps(nested, scope.child(), f"{p}.options.cases.{case}", True, entry["cases"][case])
                    if isinstance(options.get("default"), list):
                        entry["default"] = []
                        self.steps(options["default"], scope.child(), f"{p}.options.default", True, entry["default"])
                elif st.name == "split-speakers" and isinstance(options.get("steps"), list):
                    child = scope.child({"speaker": "object"})
                    child.last_audio.append("speaker.audio")
                    entry["steps"] = []
                    before = set(self.ids)
                    self.steps(options["steps"], child, f"{p}.options.steps", conditional, entry["steps"])
                    self.hidden |= self.ids - before
            out.append(entry)
            # nested branch steps become visible after the branch (maybe empty)
            for key in ("then", "else", "default"):
                for nested in entry.get(key, []):
                    self._expose(nested, scope)
            for nested_list in (entry.get("cases") or {}).values():
                for nested in nested_list:
                    self._expose(nested, scope)
            scope.steps[sid] = _Seen(st, conditional)
            if "audio" in st.outputs and st.name in AUDIO_PRODUCERS:
                scope.last_audio.append(f"steps.{sid}.audio")
            if st.outputs.get("text") == "text":
                scope.last_text.append(f"steps.{sid}.text")
            if "turns" in st.outputs:
                scope.last_turns = f"steps.{sid}.turns"

    def _expose(self, entry: dict[str, Any], scope: _Scope) -> None:
        st = STEP_TYPES[entry["type"]]
        scope.steps[entry["id"]] = _Seen(st, True)
        for key in ("then", "else", "default"):
            for nested in entry.get(key, []):
                self._expose(nested, scope)
        for nested_list in (entry.get("cases") or {}).values():
            for nested in nested_list:
                self._expose(nested, scope)


def validate(definition: Any, core: Bolo | None = None) -> Report:
    """Check ``definition`` completely; never raises for the definition's sake."""
    v = Validator(core)
    if not isinstance(definition, dict):
        v.error("", "a pipeline is an object with steps")
        return v.report
    known = ("id", "name", "description", "mode", "vars", "steps", "outputs", "triggers", "version", "created_at",
             "updated_at", "template", "tags")  # fmt: skip
    for key in definition:
        if key not in known:
            v.error(key, f"unknown field '{key}' (a pipeline has {', '.join(known[:8])})")
    pid = definition.get("id")
    if pid is not None and (not isinstance(pid, str) or not PIPELINE_ID_RE.match(pid) or pid in RESERVED_IDS):
        v.error("id", "an id holds lower-case letters, digits, '-' and '_' (64 at most), and is not "
                      + ", ".join(RESERVED_IDS))  # fmt: skip
    for key in ("name", "description"):
        if definition.get(key) is not None and not isinstance(definition[key], str):
            v.error(key, "must be a text")
    mode = definition.get("mode", "file")
    if mode not in MODES:
        v.error("mode", f"must be one of {', '.join(MODES)}")
    variables = definition.get("vars") or {}
    if not isinstance(variables, dict):
        v.error("vars", "must be an object: name -> default value")
        variables = {}
    steps = definition.get("steps")
    if not isinstance(steps, list) or not steps:
        v.error("steps", "a pipeline needs at least one step")
        return v.report
    if count_steps(steps) > MAX_STEPS:
        v.error("steps", f"a pipeline has at most {MAX_STEPS} steps")
        return v.report
    scope = _Scope({}, [], [], None, {}, mode == "live", variables)
    v.declared = declared_ids(steps)
    v.steps(steps, scope, "steps", False, v.report.steps)
    hidden = v.hidden
    outputs = definition.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, dict):
            v.error("outputs", "must be an object: name -> reference or template")
        else:
            for name, value in outputs.items():
                for ref in refs_in(value):
                    parts = ref.split(".")
                    if len(parts) > 1 and parts[0] == "steps" and parts[1] in hidden:
                        v.error(f"outputs.{name}", f"step '{parts[1]}' runs per speaker: use the split-speakers "
                                                   "step's speakers output")  # fmt: skip
                        continue
                    v.ref_type(ref, scope, f"outputs.{name}")
    triggers = definition.get("triggers")
    if triggers is not None:
        from bolo.pipelines.triggers import check_triggers

        watch = list(core.settings.pipeline_watch_dirs) if core is not None else None
        for problem in check_triggers(triggers, mode, watch):
            v.error(problem.path, problem.message)
    return v.report


def declared_ids(steps: Any) -> set[str]:
    out: set[str] = set()
    for s in steps if isinstance(steps, list) else []:
        if isinstance(s, dict):
            if isinstance(s.get("id"), str):
                out.add(s["id"])
            o = s.get("options")
            if isinstance(o, dict):
                for key in ("then", "else", "steps", "default"):
                    out |= declared_ids(o.get(key))
                if isinstance(o.get("cases"), dict):
                    for nested in o["cases"].values():
                        out |= declared_ids(nested)
    return out


def count_steps(steps: Any) -> int:
    if not isinstance(steps, list):
        return 0
    n = 0
    for s in steps:
        n += 1
        if isinstance(s, dict) and isinstance(s.get("options"), dict):
            o = s["options"]
            for key in ("then", "else", "steps", "default"):
                n += count_steps(o.get(key))
            if isinstance(o.get("cases"), dict):
                n += sum(count_steps(x) for x in o["cases"].values())
    return n


def with_ids(steps: list[Any]) -> list[Any]:
    """The steps with the ids validation gives the ones that have none (same numbering)."""
    v = Validator(None)
    out: list[dict[str, Any]] = []
    v.steps(steps, _Scope({}, [], [], None, {}, False, {}), "steps", False, out)
    return out


# -- parsing -----------------------------------------------------------------------------------------------
def parse_text(text: str | bytes, hint: str = "") -> Any:
    """A definition from JSON or YAML text (``hint``: a file name or content type)."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if "yaml" in hint or hint.endswith(".yml"):
        return _yaml(text)
    try:
        return json.loads(text)
    except ValueError:
        if "json" in hint:
            raise InvalidRequest("the pipeline is not valid JSON") from None
        return _yaml(text)


def _yaml(text: str) -> Any:
    import yaml

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InvalidRequest(f"the pipeline is neither JSON nor YAML: {exc}") from None


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64]
    return s or "pipeline"
