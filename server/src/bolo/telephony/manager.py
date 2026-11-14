"""The telephony manager: one per server, owned by :class:`bolo.core.Bolo` as ``core.telephony``.

It holds the configuration, the provider adapters, the calls that are being set up or are in progress, and the
call records. The HTTP and WebSocket routes (:mod:`bolo.api.telephony`) and the adapters go through it.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bolo.errors import BoloError, InvalidRequest, ModelNotFound
from bolo.providers import scrub
from bolo.telephony.adapters import ADAPTERS, Adapter, StreamStart, TelephonyNotConfigured, require_number
from bolo.telephony.config import PROVIDERS, VALIDATION_FIELDS, AgentConfig, TelephonyConfig
from bolo.telephony.records import CallRecord, CallStore
from bolo.telephony.session import CallSession

if TYPE_CHECKING:
    from bolo.core import Bolo

log = logging.getLogger("bolo.telephony")

#: how long a token minted by a webhook stays valid for the stream to connect and the call to run
PENDING_TTL_S = 6 * 3600.0
#: the stream must connect this soon after the webhook that announced it
CONNECT_WINDOW_S = 120.0
#: how long after a call ended its token still answers the provider's "what now" and status webhooks
AFTER_WINDOW_S = 600.0
SUMMARY_PROMPT = (
    "Summarise this phone call in two or three plain sentences: who called and why, what was said or agreed, "
    "and anything that still has to be done. Do not add anything that is not in the transcript."
)
COMPLIANCE_NOTICE = (
    "Outbound calling is regulated. Consent, calling hours, caller identification, do-not-call registers (for "
    "example TRAI's TCCCPR and the DND registry in India, the TCPA in the United States) and the disclosure "
    "that a machine is speaking are the operator's responsibility; bolo checks none of them."
)

#: the public base URL of the request being handled, when ``public_base_url`` is not set (signature opt-out only)
_request_base: contextvars.ContextVar[str | None] = contextvars.ContextVar("bolo_telephony_base", default=None)


@dataclass
class PendingCall:
    """A call bolo knows about: announced by a webhook or placed by bolo, with or without its stream yet."""

    token: str
    provider: str
    agent: AgentConfig
    record: CallRecord
    created: float = field(default_factory=time.monotonic)
    #: "agent" (talk to whoever answers) or "voicemail" (a machine answered: leave the message and hang up)
    mode: str = "agent"
    voicemail_stage: str = "end"
    session: CallSession | None = None
    closed: float | None = None  # when its record was closed (the provider may still ask what to do next)


class TelephonyManager:
    def __init__(self, core: Bolo) -> None:
        self.core = core
        self.config = TelephonyConfig(core.settings)
        self.store = CallStore(self.config.call_buffer)
        self.adapters: dict[str, Adapter] = {name: cls(self) for name, cls in ADAPTERS.items()}
        self.pending: dict[str, PendingCall] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    # -- addresses ----------------------------------------------------------------------------------------
    def set_request_base(self, base: str | None) -> None:
        _request_base.set(base)

    def base_url(self) -> str:
        base = self.config.public_base_url or _request_base.get()
        if base:
            return base
        s = self.core.settings
        host = "127.0.0.1" if s.host in ("0.0.0.0", "::") else s.host
        return f"http://{host}:{s.port}"

    def url(self, path: str) -> str:
        return self.base_url() + path

    def ws_url(self, path: str) -> str:
        base = self.base_url()
        return (
            "wss://" + base[len("https://") :] if base.startswith("https://") else "ws://" + base[len("http://") :]
        ) + path

    # -- exposure -----------------------------------------------------------------------------------------
    def problems(self, provider: str) -> list[str]:
        """Why the provider's endpoints cannot be exposed with validation on (empty: they can)."""
        cfg = self.config.provider(provider)
        out = [f"telephony.providers.{provider}.{key} is not set" for key in cfg.missing(VALIDATION_FIELDS[provider])]
        if not self.config.public_base_url:
            out.append("telephony.public_base_url is not set (the address the provider reaches this server at)")
        return out

    def adapter(self, provider: str) -> Adapter:
        """The adapter of ``provider``, or a clear refusal when its endpoints must not be exposed."""
        name = provider.strip().lower()
        adapter = self.adapters.get(name) if name in PROVIDERS else None
        if adapter is None:
            raise ModelNotFound(f"unknown telephony provider '{provider}' (known: {', '.join(PROVIDERS)})")
        if not self.config.validate_signatures:
            return adapter
        problems = self.problems(adapter.name)
        if problems:
            what = "request signatures" if adapter.protection == "signature" else "the credentials it sends"
            raise TelephonyNotConfigured(
                f"the {adapter.label} endpoints are closed: bolo cannot check {what} because "
                f"{'; '.join(problems)}. Configure them (docs/telephony.md), or set "
                "telephony.validate_signatures = false to accept unauthenticated requests (not recommended)."
            )
        return adapter

    # -- calls being set up -------------------------------------------------------------------------------
    def _prune(self) -> None:
        now = time.monotonic()
        for token, pending in list(self.pending.items()):
            ended = pending.closed is not None and now - pending.closed > AFTER_WINDOW_S
            if ended or (pending.session is None and now - pending.created > PENDING_TTL_S):
                del self.pending[token]

    def _new_pending(self, provider: str, agent: AgentConfig, record: CallRecord) -> PendingCall:
        self._prune()
        pending = PendingCall(secrets.token_urlsafe(24).replace("-", "a").replace("_", "b"), provider, agent, record)
        self.pending[pending.token] = pending
        self.store.add(record)
        return pending

    def inbound_call(
        self, provider: str, call_id: str | None, from_number: Any, to_number: Any, agent: str | None = None
    ) -> PendingCall | None:
        """Register a call a provider announced. None when no agent answers the called number."""
        existing = self.pending_by_call(provider, call_id)
        if existing is not None:
            return existing
        chosen = self.config.agent_for(to_number, agent)
        if chosen is None:
            self.core.log(
                "warning",
                f"telephony: a {provider} call to {to_number or 'an unknown number'} was refused: no agent is mapped "
                "to that number and there is no telephony.default_agent",
            )
            return None
        record = CallRecord(
            provider, "inbound", chosen.name, str(from_number) if from_number else None,
            str(to_number) if to_number else None, call_id,
        )  # fmt: skip
        record.event("ringing")
        return self._new_pending(provider, chosen, record)

    def pending_by_call(self, provider: str, call_id: str | None) -> PendingCall | None:
        if not call_id:
            return None
        return next(
            (p for p in self.pending.values() if p.provider == provider and p.record.provider_call_id == call_id), None
        )

    def claim_stream(self, provider: str, start: StreamStart) -> PendingCall | None:
        """The call behind a stream that connected without a token (Exotel; any provider with validation off)."""
        pending = self.pending_by_call(provider, start.call_id)
        if pending is not None and pending.session is None:
            return pending
        explicit = start.params.get("agent")
        return self.inbound_call(
            provider, start.call_id, start.from_number, start.to_number or start.params.get("to"), explicit
        )

    def open_stream(
        self,
        provider: str,
        token: str | None,
        headers: Mapping[str, str],
        url: str,
        send_text: Callable[[str], Awaitable[None]],
        close: Callable[[], Awaitable[None]],
        *,
        trusted: bool = False,
    ) -> CallSession:
        """Admit a media-stream connection. Raises a :class:`BoloError` that says why it was refused."""
        adapter = self.adapters[provider] if trusted else self.adapter(provider)
        pending: PendingCall | None = None
        if token:
            pending = self.pending.get(token)
            if pending is None or pending.provider != adapter.name:
                raise InvalidRequest("unknown or expired stream token")
            if pending.session is not None or pending.record.status == "ended":
                raise InvalidRequest("this stream token was used already")
            if (
                not trusted
                and time.monotonic() - pending.created > CONNECT_WINDOW_S
                and pending.record.direction == "inbound"
            ):
                raise InvalidRequest("the stream token expired before the stream connected")
        if not trusted and self.config.validate_signatures:
            checked = adapter.verify_stream(headers, url)
            if checked is False:
                raise InvalidRequest(f"the {adapter.label} stream did not present valid credentials")
            if pending is None and checked is not True:
                raise InvalidRequest(
                    f"a {adapter.label} stream must use the URL bolo's voice webhook hands out (it carries a "
                    "one-time token)"
                )
        session = CallSession(self, adapter, pending, send_text, close)
        if pending is not None:
            pending.session = session
        return session

    # -- outcomes -----------------------------------------------------------------------------------------
    def answered_by(self, pending: PendingCall, kind: str, stage: str, detail: str | None = None) -> str:
        """Note who answered an outbound call. Returns "agent", "hangup" or "message"."""
        record = pending.record
        record.answered_by = kind
        record.event("answered_by", who=kind, **({"detail": detail} if detail else {}))
        if kind != "machine" or pending.agent.voicemail == "ignore":
            return "agent"
        if pending.agent.voicemail == "leave_message":
            pending.mode, pending.voicemail_stage = "voicemail", stage
            return "message"
        return "hangup"

    async def machine_detected(self, pending: PendingCall, kind: str, stage: str, detail: str | None = None) -> None:
        """A provider reported an answering machine after the call was set up (Plivo, Telnyx)."""
        first = pending.record.answered_by != "machine"
        decision = self.answered_by(pending, kind, stage, detail) if first else (
            "message" if pending.mode == "voicemail" else "agent"
        )  # fmt: skip
        if not first and stage == "end":
            pending.voicemail_stage = "end"
        session = pending.session
        if session is not None and session.started:
            if first or stage == "end":
                await session.on_machine(decision, stage)
        elif decision == "hangup":
            adapter = self.adapters[pending.provider]
            if adapter.ends_by == "api":
                with contextlib.suppress(BoloError):
                    await adapter.hangup(pending)
            await self.finish(pending, "voicemail")

    async def finish(self, pending: PendingCall, outcome: str, error: str | None = None) -> None:
        """The provider says the call is over (or it never connected)."""
        session = pending.session
        if session is not None and session.started:
            await session.finalize(outcome, error)
            return
        if outcome == "caller_hangup" and pending.record.direction == "outbound":
            outcome = "no_answer"  # it ended before any audio was exchanged
        self.close_record(pending, outcome, error)

    def close_record(self, pending: PendingCall, outcome: str, error: str | None = None) -> None:
        record = pending.record
        if not record.end(outcome, scrub(error, *self.config.secrets()) if error else None):
            return
        pending.closed = time.monotonic()
        self.store.count(record)
        self.publish(record, "ended")
        self.core.log(
            "info",
            f"telephony: {record.provider} call {record.id} ({record.direction}, agent {record.agent}) ended: "
            f"{record.outcome}" + (f" after {record.duration_s:.0f} s" if record.duration_s else ""),
        )
        if pending.agent.summary and record.transcript and self.core.llm.configured:
            self._spawn(self._summarise(record, pending.agent))

    def _spawn(self, coro: Any) -> None:
        with contextlib.suppress(RuntimeError):  # no running loop (a record closed during shutdown)
            task = asyncio.get_running_loop().create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return
        coro.close()

    async def _summarise(self, record: CallRecord, agent: AgentConfig) -> None:
        body = {
            "model": agent.llm_model or self.core.settings.llm_model or "default",
            "messages": [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": record.transcript_text()[:12000]},
            ],
        }
        try:
            doc = await self.core.llm.chat(body)
            text = str((((doc.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
            record.summary = text or None
        except BoloError as exc:
            record.summary_error = exc.message
        except Exception as exc:
            record.summary_error = f"{type(exc).__name__}: {exc}"
        self.publish(record, "summary")

    def publish(self, record: CallRecord, event: str) -> None:
        self.core.events.publish_threadsafe("call", {"event": event, "call": record.summary_dict()})

    # -- outbound -----------------------------------------------------------------------------------------
    async def place_call(
        self, provider: str, to: str, agent: str | None = None, from_number: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CallRecord:  # fmt: skip
        adapter = self.adapter(provider)
        chosen = self.config.agents.get((agent or self.config.default_agent or "").strip().lower())
        if chosen is None:
            raise InvalidRequest(
                f"unknown agent '{agent}'" if agent else "name the agent (there is no telephony.default_agent)",
                param="agent",
            )
        if not self.config.public_base_url:
            raise TelephonyNotConfigured(
                "placing a call needs telephony.public_base_url: the provider must reach this server for the audio"
            )
        to = require_number(to, "to")
        caller = require_number(from_number or adapter.cfg.get("from_number"), "from")
        record = CallRecord(adapter.name, "outbound", chosen.name, caller, to, metadata=dict(metadata or {}))
        record.event("dialing")
        pending = self._new_pending(adapter.name, chosen, record)
        try:
            record.provider_call_id = await adapter.place_call(pending, to, caller)
        except BoloError as exc:
            self.close_record(pending, "failed", exc.message)
            raise
        self.publish(record, "dialing")
        return record

    async def hangup(self, call_id: str) -> CallRecord:
        record = self.store.get(call_id)
        if record is None:
            raise ModelNotFound(f"unknown call '{call_id}' (calls are kept in memory and rotate)")
        pending = next((p for p in self.pending.values() if p.record is record), None)
        if pending is None or record.status == "ended":
            return record
        if pending.session is not None and pending.session.started:
            await pending.session.end("agent_hangup")
        else:
            adapter = self.adapters[pending.provider]
            if adapter.ends_by == "api":
                await adapter.hangup(pending)
            self.close_record(pending, "canceled")
        return record

    # -- views --------------------------------------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        cfg = self.config
        providers = []
        for name in PROVIDERS:
            adapter = self.adapters[name]
            settings = cfg.provider(name)
            problems = self.problems(name) if cfg.validate_signatures else []
            providers.append(
                {
                    "name": name,
                    "label": adapter.label,
                    "configured": name in cfg.providers,
                    "exposed": not problems,
                    "problems": problems,
                    "protection": adapter.protection,
                    "can_validate": settings.can_validate,
                    "can_place_calls": settings.can_call and bool(cfg.public_base_url),
                    "machine_detection": adapter.machine_detection,
                    "ends_by": adapter.ends_by,
                    "settings": settings.public(),
                    "docs_url": adapter.docs_url,
                    **adapter.setup(),
                }
            )
        return {
            "public_base_url": cfg.public_base_url,
            "validate_signatures": cfg.validate_signatures,
            "default_agent": cfg.default_agent,
            "providers": providers,
            "agents": [a.to_dict() for a in cfg.agents.values()],
            "numbers": cfg.numbers_public(),
            "editable": cfg.edit_blocked_reason() is None,
            "edit_blocked_reason": cfg.edit_blocked_reason(),
            "llm_configured": self.core.llm.configured,
            "stats": self.stats(),
            "compliance_notice": COMPLIANCE_NOTICE,
        }

    def capability(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "endpoint": "/api/v1/telephony",
            "providers": list(PROVIDERS),
            "configured": [name for name in PROVIDERS if name in cfg.providers],
            # providers whose webhook and stream endpoints answer (requests can be checked, or checking is off)
            "exposed": [name for name in PROVIDERS if not cfg.validate_signatures or not self.problems(name)],
            "validate_signatures": cfg.validate_signatures,
            "agents": len(cfg.agents),
            "outbound": True,
            "simulator": "/api/v1/telephony/simulate",
            "sip": {
                "enabled": bool(self.core.settings.sip),
                "listening": self.core.sip is not None,
                "port": self.core.sip.port if self.core.sip is not None else self.core.settings.sip_port,
            },
        }

    def stats(self) -> dict[str, Any]:
        return {
            "active_calls": self.store.active(),
            "calls_total": self.store.total,
            "by_outcome": [
                {"provider": provider, "outcome": outcome, "calls": count}
                for (provider, outcome), count in sorted(self.store.by_outcome.items())
            ],
        }

    async def close(self) -> None:
        for pending in list(self.pending.values()):
            if pending.session is not None and pending.record.status != "ended":
                with contextlib.suppress(Exception):
                    await pending.session.end("error", "the server is shutting down")
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
