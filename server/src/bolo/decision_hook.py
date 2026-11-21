"""Live decisions: questions asked after every user turn of a Realtime session or a phone call (bolo.decide).

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
least this value instead. The module is independent of the transport (a pipeline step can use :func:`evaluate` and
:func:`matched` the same way).

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

from bolo.decide import Decision, decide, parse_questions
from bolo.engines.base import FloatArray
from bolo.errors import InvalidRequest

if TYPE_CHECKING:
    from bolo.config import Settings
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

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
    shown: dict[str, Any] = json.loads(given) if isinstance(given, (str, bytes)) else dict(given or {})
    return Hook(questions, shown, str(model) if model else None, raw.get("audio") is not False, rules)


def matched(hook: Hook, answers: dict[str, dict[str, Any]]) -> list[Rule]:
    """The rules whose condition holds for ``answers`` (bolo.decide's shape), in order."""
    out: list[Rule] = []
    for rule in hook.rules:
        a = answers.get(rule.question)
        if not a:
            continue
        if a["type"] == "yes_no":
            if rule.answer is not None and a["answer"] is not rule.answer:
                continue
            p = a["probability"] if a["answer"] else 1.0 - a["probability"]
        elif a["type"] == "choice":
            if rule.answer is not None and a["answer"] != str(rule.answer):
                continue
            p = float(a["probabilities"].get(a["answer"], 0.0))
        else:
            if rule.answer is not None and rule.answer not in (a["label"], a["level"]):
                continue
            if rule.min_score is not None and a["score"] < rule.min_score:
                continue
            p = float(a["probabilities"][a["level"]]) if a["probabilities"] else 0.0
        if rule.min_probability is not None and p < rule.min_probability:
            continue
        out.append(rule)
    return out


async def evaluate(
    core: Bolo, hook: Hook, text: str | None, audio: FloatArray | None, trace: Trace
) -> tuple[Decision, list[Rule]]:
    """Ask the hook's questions about one turn: its transcript and, when the hook sends audio, its last seconds."""
    spec = core.resolve(hook.model, "decision")
    samples = None
    if hook.audio and audio is not None and len(audio) >= SR // 10:
        spec_max = float(spec.options.get("max_audio_s") or 30.0)
        samples = np.ascontiguousarray(audio[-int(spec_max * SR) :], dtype=np.float32)
    result = await decide(core, spec.id, text or None, samples, SR if samples is not None else None,
                          hook.questions, trace)  # fmt: skip
    return result, matched(hook, result.answers)


def event_fields(item_id: str, result: Decision, rules: list[Rule]) -> dict[str, Any]:
    """The fields of ``conversation.item.decisions.completed`` (and the webhook's body)."""
    return {
        "item_id": item_id, "model": result.model, "answers": result.answers,
        "actions": [r.to_dict() for r in rules],
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
