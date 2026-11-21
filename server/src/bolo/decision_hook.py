"""Live decisions: questions asked after every user turn of a Realtime session or a phone call (bolo.decide).

Live decisions are a shortcut for a live pipeline (bolo.pipelines.live): :func:`definition` turns a hook into a
pipeline of one ``decisions`` step and one action step per rule (``tag``, ``notify``, ``transfer``, ``hangup``), and
:func:`run` runs it with the same runner, conditions and ``pipeline.action`` events as a saved live pipeline. The one
difference is when: a live pipeline answers the turn, a hook runs beside the answer and never speaks of its own.

A hook is a decision model, named questions (as for ``POST /v1/audio/decisions``) and optional rules that turn an
answer into an action. After each user turn has its final transcript, the questions are asked about that turn (the
transcript and, unless ``audio`` is false, the turn's audio, at most the model's limit) in the background, so the
reply is never held up. The answers go to the session as the event ``conversation.item.decisions.completed`` (or
``.failed``), to the webhook the server is configured with, and the rules that match are carried out where they can
be: on a phone call ``end_call``, ``transfer`` (to the rule's ``value`` or the agent's transfer number) and ``tag``
(the rule's ``value`` on the call record); in other sessions the matched actions are reported in the event and
``tag`` goes on the session's traces.

The configuration (Realtime: ``session.decisions``; phone agents: ``decisions``)::

    {"model": "d1-omni-600m-llamacpp",            # optional: default_decision
     "questions": {"escalate": {"type": "yes_no", "question": "Does the caller ask for a person?"},
                   "department": {"type": "choice", "question": "Which department?",
                                  "options": ["billing", "technical", "sales"]}},
     "audio": true,                                # also send the turn's audio (default true)
     "actions": [{"question": "escalate", "answer": true, "min_probability": 0.8, "do": "transfer"},
                 {"question": "department", "answer": "billing", "do": "tag", "value": "billing"}]}

A rule matches when its question's answer is ``answer`` (yes_no: true or false; choice: an option; score: a level's
label or index), when given, and the probability of that answer is at least ``min_probability`` (yes_no: of the
answer given; choice: of the chosen option), when given; ``min_score`` asks a score question for a level of at
least this value instead. Each rule becomes the ``when`` of its step; rules that end the call run after the others.

Webhooks: ``decision_webhook_url`` (a setting; phone agents can name their own) receives a POST of the event as
JSON, signed like pipeline webhooks (``X-Bolo-Signature`` with ``webhook_secret``, ``X-Bolo-Event: decisions``).
Clients of a session never choose the URL: the server posts only where its operator said.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.decide import parse_questions
from bolo.engines.base import FloatArray
from bolo.errors import BoloError, InvalidRequest

if TYPE_CHECKING:
    from bolo.config import Settings
    from bolo.core import Bolo

log = logging.getLogger("bolo.decisions")

ACTIONS = ("end_call", "transfer", "tag", "notify")
MAX_RULES = 32
SR = 16000


@dataclass
class Rule:
    question: str
    do: str
    answer: Any = None
    min_probability: float | None = None
    min_score: float | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"question": self.question, "do": self.do}
        for k in ("answer", "min_probability", "min_score", "value"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out


@dataclass
class Hook:
    questions: dict[str, dict[str, Any]]
    #: the request's questions as given (bolo's names), echoed back by to_dict
    raw_questions: dict[str, Any]
    model: str | None = None
    audio: bool = True
    rules: list[Rule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "questions": self.raw_questions, "audio": self.audio,
            "actions": [r.to_dict() for r in self.rules],
        }  # fmt: skip


def parse_hook(raw: Any, *, param: str = "decisions") -> Hook | None:
    """A hook from its configuration; None for null or ``{}`` (no hook). :class:`InvalidRequest` names ``param``."""
    if raw in (None, {}, False):
        return None
    if not isinstance(raw, dict):
        raise InvalidRequest(f"{param} must be an object with questions", param=param)
    unknown = set(raw) - {"model", "questions", "audio", "actions"}
    if unknown:
        raise InvalidRequest(f"{param} has unknown fields: {', '.join(sorted(unknown))}", param=param)
    try:
        questions = parse_questions(raw.get("questions"))
    except InvalidRequest as exc:
        raise InvalidRequest(exc.message, param=f"{param}.questions") from exc
    rules_raw = raw.get("actions") or []
    if not isinstance(rules_raw, list) or len(rules_raw) > MAX_RULES:
        raise InvalidRequest(f"{param}.actions must be a list of at most {MAX_RULES} rules", param=f"{param}.actions")
    rules: list[Rule] = []
    for i, r in enumerate(rules_raw):
        where = f"{param}.actions[{i}]"
        if not isinstance(r, dict):
            raise InvalidRequest(f"{where} must be an object", param=where)
        extra = set(r) - {"question", "answer", "min_probability", "min_score", "do", "value"}
        if extra:
            raise InvalidRequest(f"{where} has unknown fields: {', '.join(sorted(extra))}", param=where)
        qid = r.get("question")
        if qid not in questions:
            raise InvalidRequest(f"{where}: question {qid!r} is not one of the questions", param=where)
        do = r.get("do")
        if do not in ACTIONS:
            raise InvalidRequest(f"{where}: do must be one of {', '.join(ACTIONS)}", param=where)
        if do == "tag" and not str(r.get("value") or "").strip():
            raise InvalidRequest(f"{where}: a tag rule needs its value (the tag)", param=where)
        rule = Rule(str(qid), str(do), r.get("answer"), value=str(r["value"]).strip() if r.get("value") else None)
        for k in ("min_probability", "min_score"):
            if r.get(k) is not None:
                try:
                    setattr(rule, k, float(r[k]))
                except (TypeError, ValueError) as exc:
                    raise InvalidRequest(f"{where}: {k} must be a number", param=where) from exc
        kind = questions[rule.question]["type"]
        if rule.min_score is not None and kind != "score":
            raise InvalidRequest(f"{where}: min_score is for score questions", param=where)
        if kind == "noul" and rule.answer is not None and not isinstance(rule.answer, bool):
            raise InvalidRequest(f"{where}: the answer of a yes_no question is true or false", param=where)
        if (
            kind == "choice"
            and rule.answer is not None
            and str(rule.answer) not in questions[rule.question]["criteria"]
        ):
            raise InvalidRequest(f"{where}: {rule.answer!r} is not an option of '{rule.question}'", param=where)
        rules.append(rule)
    model = raw.get("model")
    given = raw.get("questions")
    shown_q: dict[str, Any] = json.loads(given) if isinstance(given, (str, bytes)) else dict(given or {})
    hook = Hook(questions, shown_q, str(model) if model else None, raw.get("audio") is not False, rules)
    from bolo.pipelines.definition import validate

    report = validate(definition(hook))
    if not report.valid:
        first = report.errors[0]
        raise InvalidRequest(f"{param}: {first.message} ({first.path})", param=param)
    return hook


#: what each rule action is in the implicit live pipeline
STEP_FOR = {"end_call": "hangup", "transfer": "transfer", "tag": "tag", "notify": "notify"}
#: rules that end the call run after the others, so that tags and notices of the same turn are not lost
ENDING = ("end_call", "transfer")
DECISIONS_STEP = "decisions"


def _condition(hook: Hook, rule: Rule) -> dict[str, Any] | None:
    """The ``when`` of a rule's step: the pipeline condition the rule's answer, probability and score make."""
    q, kind = rule.question, hook.questions[rule.question]["type"]
    ref = f"{{{{steps.{DECISIONS_STEP}.answers.{q}."
    parts: list[dict[str, Any]] = []
    if rule.answer is not None:
        if kind == "score":
            field_ = "level" if isinstance(rule.answer, int) and not isinstance(rule.answer, bool) else "label"
            parts.append({"value": ref + field_ + "}}", "equals": rule.answer})
        else:
            parts.append({"value": ref + "answer}}", "equals": rule.answer if kind == "noul" else str(rule.answer)})
    if rule.min_probability is not None:
        parts.append({"value": f"{{{{steps.{DECISIONS_STEP}.probabilities.{q}}}}}", "gte": rule.min_probability})
    if rule.min_score is not None:
        parts.append({"value": ref + "score}}", "gte": rule.min_score})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"all": parts}


def ordered_rules(hook: Hook) -> list[tuple[int, Rule]]:
    """The rules in the order their steps run: tags and notices first, then those that end the call."""
    indexed = list(enumerate(hook.rules))
    return [x for x in indexed if x[1].do not in ENDING] + [x for x in indexed if x[1].do in ENDING]


def definition(hook: Hook) -> dict[str, Any]:
    """The live pipeline a hook is a shortcut for: a ``decisions`` step with the hook's questions, then one action
    step per rule (``tag``, ``notify``, ``transfer``, ``hangup``), each with the rule as its ``when``."""
    inputs: dict[str, Any] = {"text": "{{input.text}}"}
    if hook.audio:
        inputs["audio"] = "{{input.audio}}"
    steps: list[dict[str, Any]] = [
        {"id": DECISIONS_STEP, "type": "decisions", "model": hook.model, "inputs": inputs,
         "options": {"questions": hook.raw_questions}},
    ]  # fmt: skip
    for i, rule in ordered_rules(hook):
        step: dict[str, Any] = {"id": f"rule-{i}", "type": STEP_FOR[rule.do]}
        when = _condition(hook, rule)
        if when is not None:
            step["when"] = when
        if rule.do == "tag":
            step["options"] = {"value": rule.value}
        elif rule.do == "transfer" and rule.value:
            step["options"] = {"to": rule.value}
        steps.append(step)
    return {"id": "session-decisions", "name": "Live decisions", "mode": "live", "steps": steps}


@dataclass
class HookResult:
    model: str
    answers: dict[str, dict[str, Any]]
    #: the rules whose step ran (their condition held), in the hook's order
    matched: list[Rule]
    run: Any


async def run(
    core: Bolo, hook: Hook, target: Any, *, text: str, audio: FloatArray | None, session: dict[str, Any], turn: int
) -> HookResult:
    """One turn through the hook's live pipeline (bolo.pipelines.live.run_turn): the same runner, steps and
    ``pipeline.action`` events as a saved live pipeline. ``target`` carries out the actions."""
    from bolo.pipelines.live import run_turn
    from bolo.pipelines.runner import Audio

    if hook.model is None:
        hook.model = core.resolve(None, "decision").id
    heard = Audio(np.ascontiguousarray(audio, dtype=np.float32), SR) if audio is not None and len(audio) else None
    if heard is not None:
        spec = core.resolve(hook.model, "decision")
        longest = float(spec.options.get("max_audio_s") or 30.0)
        heard = Audio(heard.samples[-int(longest * SR) :], SR)
    result = await run_turn(core, definition(hook), target, text=text, audio=heard, history=[], turn=turn,
                            session=session, keep=False)  # fmt: skip
    steps = {s["id"]: s for s in result.steps}
    first = steps.get(DECISIONS_STEP) or {}
    if first.get("status") != "ok":
        message = (result.error or {}).get("message") or "the decision model gave no answers"
        code = (result.error or {}).get("code") or "decision_failed"
        error = BoloError(message)
        error.code = str(code)
        raise error
    output = first.get("output") or {}
    ran = [rule for i, rule in enumerate(hook.rules) if (steps.get(f"rule-{i}") or {}).get("status") == "ok"]
    return HookResult(str(output.get("model") or hook.model), dict(output.get("answers") or {}), ran, result)


def event_fields(item_id: str, result: HookResult) -> dict[str, Any]:
    """The fields of ``conversation.item.decisions.completed`` (and the webhook's body)."""
    return {
        "item_id": item_id, "model": result.model, "answers": result.answers,
        "actions": [r.to_dict() for r in result.matched],
    }  # fmt: skip


async def post_webhook(settings: Settings, url: str | None, payload: dict[str, Any], session_id: str) -> None:
    """POST ``payload`` to the operator's webhook, signed as pipeline webhooks are (bolo.pipelines.webhooks:
    ``X-Bolo-Signature: t=<unix seconds>,v1=<HMAC-SHA256 of "<t>.<body>">`` with ``webhook_secret``), with
    ``X-Bolo-Event: decisions``. Failures are logged, never raised: a hook must not break a call."""
    if not url:
        return
    from bolo.pipelines.webhooks import send

    body = json.dumps(payload, ensure_ascii=False).encode()
    try:
        resp = await send(settings, url, body, event="decisions", run_id=session_id)
        if resp.status_code >= 400:
            log.warning("decision webhook %s answered %s", url, resp.status_code)
    except Exception as exc:
        log.warning("decision webhook %s failed: %s: %s", url, type(exc).__name__, exc)


def spawn_webhook(settings: Settings, url: str | None, payload: dict[str, Any], session_id: str) -> None:
    if url:
        task = asyncio.get_running_loop().create_task(post_webhook(settings, url, payload, session_id))
        _pending.add(task)
        task.add_done_callback(_pending.discard)


#: webhook tasks in flight (kept so that they are not garbage-collected before they finish)
_pending: set[asyncio.Task[None]] = set()
