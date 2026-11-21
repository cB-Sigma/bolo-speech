"""Live pipelines (``mode: live``): a pipeline as the behaviour of a phone number or a Realtime session, run once per
user turn (docs/pipelines.md, "Live pipelines").

The session transcribes the turn as usual; the pipeline gets its text (``input.text``), its audio (``input.audio``)
and the conversation so far (``input.history``). Its actions act on the session through a :class:`LiveTarget`:

* ``reply`` says a fixed text (synthesised and streamed as the session's own answers are);
* ``respond`` lets the session's language model answer as usual (streamed, with its tools), optionally with more
  instructions for this turn;
* ``transfer`` and ``hangup`` end the call (in a Realtime session, the client is told with a ``pipeline.action``
  event).

The steps before the first speaking action are the only ones that add to the time before the caller hears the
answer; the answer itself streams.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

from bolo.errors import InvalidRequest
from bolo.pipelines.runner import (
    Audio,
    Run,
    Runner,
    Scope,
    StopRun,
    handler,
    llm_complete,
    need_text,
)

if TYPE_CHECKING:
    from bolo.core import Bolo

#: what a live session's actions are, in the run record
SPEAKING = ("reply", "respond")


class LiveTarget(Protocol):
    """What a live pipeline acts on: a Realtime session (and through it a phone call), or the simulator."""

    async def say(self, text: str) -> str: ...

    async def respond(self, instructions: str | None) -> str: ...

    async def action(self, kind: str, **fields: Any) -> None: ...


def _target(run: Run) -> LiveTarget:
    target = run.inputs.get("_target")
    if target is None:
        raise InvalidRequest("this step only runs in a live session (a call, a Realtime session or the simulator)")
    return target  # type: ignore[no-any-return]


def _act(run: Run, kind: str, **fields: Any) -> None:
    run.actions.append({"action": kind, **fields})


@handler("reply")
async def _reply(r: Runner, run: Run, e: dict[str, Any], i: dict[str, Any], o: dict[str, Any], scope: Scope,
                 d: int) -> dict[str, Any]:  # fmt: skip
    text = need_text(i).strip()
    said = await _target(run).say(text)
    _act(run, "reply", text=said)
    return {"text": said}


@handler("respond")
async def _respond(r: Runner, run: Run, e: dict[str, Any], i: dict[str, Any], o: dict[str, Any], scope: Scope,
                   d: int) -> dict[str, Any]:  # fmt: skip
    extra = r.text(run, o.get("instructions"), scope).strip() if o.get("instructions") else None
    said = await _target(run).respond(extra or None)
    _act(run, "respond", text=said, **({"instructions": extra} if extra else {}))
    return {"text": said}


@handler("transfer")
async def _transfer(r: Runner, run: Run, e: dict[str, Any], i: dict[str, Any], o: dict[str, Any], scope: Scope,
                    d: int) -> dict[str, Any]:  # fmt: skip
    target = _target(run)
    to = r.text(run, o.get("to"), scope).strip() if o.get("to") else ""
    say = r.text(run, o.get("say"), scope).strip() if o.get("say") else ""
    if say:
        await target.say(say)
    await target.action("transfer", to=to or None, reason=f"pipeline step {e['id']}")
    _act(run, "transfer", to=to or None)
    raise StopRun(f"transferred by step {e['id']}")


@handler("hangup")
async def _hangup(r: Runner, run: Run, e: dict[str, Any], i: dict[str, Any], o: dict[str, Any], scope: Scope,
                  d: int) -> dict[str, Any]:  # fmt: skip
    target = _target(run)
    say = r.text(run, o.get("say"), scope).strip() if o.get("say") else ""
    if say:
        await target.say(say)
    await target.action("hangup", reason=f"pipeline step {e['id']}")
    _act(run, "hangup")
    raise StopRun(f"ended by step {e['id']}")


async def run_turn(
    core: Bolo,
    definition: dict[str, Any],
    target: LiveTarget,
    *,
    text: str,
    audio: Audio | None,
    history: list[dict[str, Any]],
    turn: int,
    session: dict[str, Any],
    keep: bool = True,
) -> Run:
    """One user turn through a live pipeline. ``keep``: write the run to the history."""
    runner = Runner(core, core.pipelines.store if keep else None)
    run = runner.prepare(definition, audio=audio, text=text, trigger="turn",
                         extra_inputs={"history": history, "turn": turn, "session": session, "_target": target})  # fmt: skip
    await runner.execute(run)
    return run


# -- the simulator ------------------------------------------------------------------------------------------
class SimulatedSession:
    """A conversation without audio: replies are recorded; ``respond`` asks the language model (not streamed)."""

    def __init__(self, core: Bolo, instructions: str | None) -> None:
        self.core = core
        self.instructions = instructions
        self.history: list[dict[str, Any]] = []
        self.said: list[str] = []
        self.ended: dict[str, Any] | None = None

    async def say(self, text: str) -> str:
        self.said.append(text)
        self.history.append({"role": "assistant", "content": text})
        return text

    async def respond(self, instructions: str | None) -> str:
        system = "\n\n".join(x for x in (self.instructions, instructions) if x)
        messages = ([{"role": "system", "content": system}] if system else []) + self.history
        text = (await llm_complete(self.core, messages)).strip()
        return await self.say(text)

    async def action(self, kind: str, **fields: Any) -> None:
        self.ended = {"action": kind, **{k: v for k, v in fields.items() if v is not None}}


async def simulate(
    core: Bolo, definition: dict[str, Any], turns: list[str], *, instructions: str | None = None
) -> dict[str, Any]:
    """Play typed turns through a live pipeline; each turn's run (steps, outputs, actions) and the conversation."""
    if not turns:
        raise InvalidRequest("give at least one turn", param="turns")
    session = SimulatedSession(core, instructions)
    out: list[dict[str, Any]] = []
    started = time.time()
    for n, text in enumerate(turns):
        if session.ended is not None:
            break
        session.history.append({"role": "user", "content": text})
        before = len(session.said)
        run = await run_turn(core, definition, session, text=text, audio=None, history=list(session.history[:-1]),
                             turn=n, session={"id": "simulation", "transport": "simulator"}, keep=False)  # fmt: skip
        doc = run.to_dict()
        doc["actions"] = run.actions
        out.append({"user": text, "agent": " ".join(session.said[before:]), "run": doc})
    return {
        "object": "pipeline.simulation",
        "turns": out,
        "ended": session.ended,
        "transcript": [{"role": m["role"], "text": m["content"]} for m in session.history],
        "ms": round((time.time() - started) * 1000.0, 1),
    }
