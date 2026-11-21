"""Telephony configuration: providers, agents and the numbers that reach them.

Everything lives under the ``telephony`` setting (a ``[telephony]`` table in the configuration file, the
``BOLO_TELEPHONY`` JSON variable, or ``BOLO_TELEPHONY_<PROVIDER>_<FIELD>`` variables for credentials)::

    [telephony]
    public_base_url = "https://voice.example.org"   # how the providers reach this server
    default_agent = "front-desk"

    [telephony.providers.twilio]
    account_sid = "AC..."
    auth_token = "..."

    [telephony.agents.front-desk]
    instructions = "You answer the phone for ..."
    greeting = "Hello, how can I help?"

    [telephony.numbers]
    "+14155550100" = "front-desk"

Agents and numbers can also be added through the API; those are stored in ``<data_dir>/telephony.json``.
Provider credentials come from the configuration only and are never returned by any API or logged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.errors import InvalidRequest, ModelNotFound

if TYPE_CHECKING:
    from bolo.config import Settings

log = logging.getLogger("bolo.telephony")

PROVIDERS = ("twilio", "plivo", "telnyx", "exotel")
#: fields each provider understands; the ones in SECRET_FIELDS are never shown
PROVIDER_FIELDS: dict[str, tuple[str, ...]] = {
    "twilio": ("account_sid", "auth_token", "api_key_sid", "api_key_secret", "from_number", "api_base_url"),
    "plivo": ("auth_id", "auth_token", "from_number", "api_base_url"),
    "telnyx": ("api_key", "public_key", "connection_id", "from_number", "api_base_url"),
    "exotel": (
        "account_sid", "api_key", "api_token", "subdomain", "app_id", "from_number", "stream_username",
        "stream_password", "sample_rate", "api_base_url",
    ),
}  # fmt: skip
SECRET_FIELDS = frozenset({"auth_token", "api_key_secret", "api_key", "api_token", "stream_password"})
#: what a provider needs before bolo can check that a request really comes from it
VALIDATION_FIELDS: dict[str, tuple[str, ...]] = {
    "twilio": ("auth_token",),
    "plivo": ("auth_token",),
    "telnyx": ("public_key",),
    "exotel": ("stream_username", "stream_password"),
}
#: what a provider needs before bolo can place calls (and, for Telnyx, answer them)
API_FIELDS: dict[str, tuple[str, ...]] = {
    "twilio": ("account_sid", "auth_token"),
    "plivo": ("auth_id", "auth_token"),
    "telnyx": ("api_key", "connection_id"),
    "exotel": ("account_sid", "api_key", "api_token", "app_id"),
}
TOP_LEVEL = ("public_base_url", "validate_signatures", "default_agent", "call_buffer", "providers", "agents", "numbers")
VOICEMAIL_ACTIONS = ("hangup", "leave_message", "ignore")
DTMF_MODES = ("message", "ignore")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


def normalize_number(number: Any) -> str:
    """A phone number reduced to its digits (``+1 (415) 555-0100`` → ``14155550100``); SIP URIs lose the scheme."""
    text = str(number or "").strip()
    if text.lower().startswith(("sip:", "sips:", "tel:")):
        text = text.split(":", 1)[1].split("@", 1)[0]
    digits = re.sub(r"\D", "", text)
    return digits or text.lower()


@dataclass
class AgentConfig:
    """One voice agent that can answer or place calls. Unset fields fall back to the server's settings."""

    name: str
    instructions: str | None = None
    #: said as soon as the call is answered (after the consent announcement); none: the caller speaks first
    greeting: str | None = None
    voice: str | None = None
    language: str | None = None
    stt_model: str | None = None
    tts_model: str | None = None
    llm_model: str | None = None
    pipeline: str | None = None
    #: a saved live pipeline (docs/pipelines.md, "Live pipelines") that answers each turn of the caller
    pipeline_id: str | None = None
    #: function tools: {name, description, parameters, url}; a tool with a url is called with an HTTP POST
    tools: list[dict[str, Any]] = field(default_factory=list)
    #: longest call in seconds (0: no limit) and what is said before hanging up at the limit
    max_duration_s: float = 600.0
    max_duration_message: str | None = None
    #: seconds of silence from the caller before the prompt is said (0: never), how often, and the farewell
    silence_timeout_s: float = 10.0
    silence_prompt: str = "Are you still there?"
    silence_prompts: int = 2
    goodbye: str | None = None
    #: where transfer_call sends the caller, and what is said first
    transfer_number: str | None = None
    transfer_message: str | None = None
    #: said first on every call and never interrupted, e.g. "This call is transcribed for quality purposes."
    recording_consent: str | None = None
    #: outbound calls answered by a machine: "hangup", "leave_message" (says voicemail_message) or "ignore"
    voicemail: str = "hangup"
    voicemail_message: str | None = None
    #: key presses: "message" (given to the agent as a user message) or "ignore"
    dtmf: str = "message"
    #: write a short summary with the language model when the call ends
    summary: bool = True
    #: passed to the Realtime session as audio.input.turn_detection (e.g. {type = "semantic_vad"})
    turn_detection: dict[str, Any] | None = None
    #: romanised text in what the agent says ("off" | "auto" | "on"; unset: the voice's default), bolo.text.romanised
    romanised: str | None = None
    source: str = "settings"

    @classmethod
    def from_dict(cls, name: str, entry: Any, source: str = "settings") -> AgentConfig:
        name = str(name).strip().lower()
        if not _NAME_RE.match(name):
            raise InvalidRequest(
                f"invalid agent name {name!r}: lowercase letters, digits, '-' and '_' (at most 40)", param="name"
            )
        if not isinstance(entry, dict):
            raise InvalidRequest(f"agent '{name}' must be a table of settings", param="agents")
        known = {f.name: f for f in fields(cls) if f.name not in ("name", "source")}
        unknown = set(entry) - set(known)
        if unknown:
            raise InvalidRequest(f"agent '{name}' has unknown fields: {', '.join(sorted(unknown))}", param="agents")
        values: dict[str, Any] = {}
        for key, raw in entry.items():
            if raw is None:
                continue
            if key in ("max_duration_s", "silence_timeout_s"):
                try:
                    values[key] = max(0.0, float(raw))
                except (TypeError, ValueError) as exc:
                    raise InvalidRequest(f"agent '{name}': {key} must be a number of seconds", param=key) from exc
            elif key == "silence_prompts":
                try:
                    values[key] = max(0, int(raw))
                except (TypeError, ValueError) as exc:
                    raise InvalidRequest(f"agent '{name}': silence_prompts must be an integer", param=key) from exc
            elif key == "summary":
                values[key] = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
            elif key == "tools":
                values[key] = _tools(name, raw)
            elif key == "turn_detection":
                if not isinstance(raw, dict):
                    raise InvalidRequest(f"agent '{name}': turn_detection must be a table", param=key)
                values[key] = dict(raw)
            elif key == "romanised":
                from bolo.text.romanised import parse_mode

                values[key] = parse_mode(raw)
            else:
                values[key] = str(raw).strip() or None
        agent = cls(name=name, source=source, **values)
        if agent.voicemail not in VOICEMAIL_ACTIONS:
            raise InvalidRequest(
                f"agent '{name}': voicemail must be one of {', '.join(VOICEMAIL_ACTIONS)}", param="voicemail"
            )
        if agent.dtmf not in DTMF_MODES:
            raise InvalidRequest(f"agent '{name}': dtmf must be one of {', '.join(DTMF_MODES)}", param="dtmf")
        if agent.pipeline is not None:
            from bolo.realtime import PIPELINES

            if agent.pipeline not in PIPELINES:
                raise InvalidRequest(
                    f"agent '{name}': pipeline must be one of {', '.join(PIPELINES)}", param="pipeline"
                )
        if agent.voicemail == "leave_message" and not agent.voicemail_message:
            raise InvalidRequest(
                f"agent '{name}': voicemail = \"leave_message\" needs a voicemail_message", param="voicemail_message"
            )
        return agent

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["editable"] = self.source == "stored"
        return out

    def stored(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("name", "source")}


def _tools(agent: str, raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise InvalidRequest(f"agent '{agent}': tools must be a list", param="tools")
    out: list[dict[str, Any]] = []
    for tool in raw:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"].strip():
            raise InvalidRequest(f"agent '{agent}': every tool needs a name", param="tools")
        unknown = set(tool) - {"name", "description", "parameters", "url", "type", "timeout_s"}
        if unknown:
            raise InvalidRequest(
                f"agent '{agent}': tool '{tool['name']}' has unknown fields: {', '.join(sorted(unknown))}",
                param="tools",
            )
        url = tool.get("url")
        if url is not None and not str(url).startswith(("http://", "https://")):
            raise InvalidRequest(
                f"agent '{agent}': the url of tool '{tool['name']}' must start with http:// or https://", param="tools"
            )
        out.append({k: v for k, v in tool.items() if v is not None and k != "type"})
    return out


@dataclass
class ProviderSettings:
    """The credentials and options of one telephony provider."""

    name: str
    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        return self.values.get(key) or None

    def missing(self, keys: tuple[str, ...]) -> list[str]:
        return [k for k in keys if not self.values.get(k)]

    @property
    def can_validate(self) -> bool:
        return not self.missing(VALIDATION_FIELDS[self.name])

    @property
    def can_call(self) -> bool:
        return not self.missing(API_FIELDS[self.name])

    def secrets(self) -> list[str]:
        return [v for k, v in self.values.items() if k in SECRET_FIELDS and v]

    def public(self) -> dict[str, Any]:
        shown: dict[str, Any] = {}
        for key in PROVIDER_FIELDS[self.name]:
            value = self.values.get(key)
            shown[key] = ("***" if value else None) if key in SECRET_FIELDS else (value or None)
        return shown


def redact_telephony(value: Any) -> dict[str, Any]:
    """The ``telephony`` setting as it may be shown: every provider secret replaced."""
    if not isinstance(value, dict):
        return {}
    out = dict(value)
    providers = out.get("providers")
    if isinstance(providers, dict):
        out["providers"] = {
            str(name): {str(k): ("***" if k in SECRET_FIELDS and v else v) for k, v in entry.items()}
            if isinstance(entry, dict)
            else {}
            for name, entry in providers.items()
        }
    return out


def telephony_env(values: dict[str, Any], environ: dict[str, str]) -> None:
    """``BOLO_TELEPHONY_PUBLIC_BASE_URL``, ``_VALIDATE_SIGNATURES``, ``_DEFAULT_AGENT`` and
    ``BOLO_TELEPHONY_<PROVIDER>_<FIELD>`` add to (or override) the ``telephony`` table."""
    prefix = "BOLO_TELEPHONY_"
    doc = dict(values.get("telephony") or {})
    changed = False
    for envname, raw in environ.items():
        if not envname.startswith(prefix) or raw == "":
            continue
        rest = envname[len(prefix) :].lower()
        if rest in ("public_base_url", "default_agent"):
            doc[rest] = raw
            changed = True
        elif rest == "validate_signatures":
            doc[rest] = raw.strip().lower() not in ("0", "false", "no", "off")
            changed = True
        else:
            provider, _, name = rest.partition("_")
            if provider in PROVIDERS and name in PROVIDER_FIELDS[provider]:
                providers = {str(k): dict(v) for k, v in dict(doc.get("providers") or {}).items()}
                providers.setdefault(provider, {})[name] = raw
                doc["providers"] = providers
                changed = True
    if changed:
        values["telephony"] = doc


class TelephonyConfig:
    """The parsed ``telephony`` setting plus the agents and numbers stored through the API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        doc = dict(settings.telephony or {})
        unknown = set(doc) - set(TOP_LEVEL)
        if unknown:
            raise InvalidRequest(f"telephony: unknown settings: {', '.join(sorted(unknown))}", param="telephony")
        base = str(doc.get("public_base_url") or "").strip().rstrip("/")
        if base and not base.startswith(("http://", "https://")):
            raise InvalidRequest("telephony: public_base_url must start with https:// (or http://)", param="telephony")
        self.public_base_url: str | None = base or None
        validate = doc.get("validate_signatures", True)
        self.validate_signatures = (
            validate if isinstance(validate, bool) else str(validate).strip().lower() not in ("0", "false", "no", "off")
        )
        self.default_agent: str | None = str(doc["default_agent"]).strip().lower() if doc.get("default_agent") else None
        try:
            self.call_buffer = max(1, int(doc.get("call_buffer", 200)))
        except (TypeError, ValueError) as exc:
            raise InvalidRequest("telephony: call_buffer must be an integer", param="telephony") from exc
        self.providers: dict[str, ProviderSettings] = {}
        raw_providers = doc.get("providers") or {}
        if not isinstance(raw_providers, dict):
            raise InvalidRequest("telephony: providers must be a table", param="telephony")
        for name, entry in raw_providers.items():
            pname = str(name).strip().lower()
            if pname not in PROVIDERS:
                raise InvalidRequest(
                    f"telephony: unknown provider '{pname}' (known: {', '.join(PROVIDERS)})", param="telephony"
                )
            if not isinstance(entry, dict):
                raise InvalidRequest(f"telephony: provider '{pname}' must be a table", param="telephony")
            bad = set(entry) - set(PROVIDER_FIELDS[pname])
            if bad:
                raise InvalidRequest(
                    f"telephony: provider '{pname}' has unknown fields: {', '.join(sorted(bad))} "
                    f"(known: {', '.join(PROVIDER_FIELDS[pname])})",
                    param="telephony",
                )
            self.providers[pname] = ProviderSettings(
                pname, {str(k): str(v).strip() for k, v in entry.items() if v not in (None, "")}
            )
        self._lock = threading.Lock()
        self.agents: dict[str, AgentConfig] = {}
        raw_agents = doc.get("agents") or {}
        if not isinstance(raw_agents, dict):
            raise InvalidRequest("telephony: agents must be a table", param="telephony")
        for name, entry in raw_agents.items():
            agent = AgentConfig.from_dict(str(name), entry, "settings")
            self.agents[agent.name] = agent
        #: normalised number -> (agent name, the number as written, source)
        self.numbers: dict[str, tuple[str, str, str]] = {}
        raw_numbers = doc.get("numbers") or {}
        if not isinstance(raw_numbers, dict):
            raise InvalidRequest("telephony: numbers must be a table of number = agent", param="telephony")
        for number, agent_name in raw_numbers.items():
            self.numbers[normalize_number(number)] = (str(agent_name).strip().lower(), str(number), "settings")
        self._path = Path(settings.data_dir) / "telephony.json"
        self._load_stored()

    # -- stored agents and numbers ------------------------------------------------------------------------
    def _load_stored(self) -> None:
        if not self._path.is_file():
            return
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s", self._path, exc)
            return
        if not isinstance(doc, dict):
            return
        for name, entry in (doc.get("agents") or {}).items():
            try:
                agent = AgentConfig.from_dict(str(name), entry, "stored")
            except InvalidRequest as exc:
                log.warning("ignoring stored agent %r: %s", name, exc.message)
                continue
            self.agents.setdefault(agent.name, agent)  # the configuration wins over a stored agent
        for number, agent_name in (doc.get("numbers") or {}).items():
            self.numbers.setdefault(normalize_number(number), (str(agent_name), str(number), "stored"))

    def _save_stored(self) -> None:
        doc = {
            "agents": {a.name: a.stored() for a in self.agents.values() if a.source == "stored"},
            "numbers": {written: agent for agent, written, source in self.numbers.values() if source == "stored"},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        tmp.replace(self._path)
        with contextlib.suppress(OSError):
            self._path.chmod(0o600)

    def edit_blocked_reason(self) -> str | None:
        s = self.settings
        if s.host not in ("127.0.0.1", "localhost", "::1") and not s.api_key:
            return "agents and numbers are never changed through the API on a non-loopback address without an API key"
        return None

    def _check_edit(self) -> None:
        reason = self.edit_blocked_reason()
        if reason:
            raise InvalidRequest(reason, param="telephony")

    def put_agent(self, name: str, entry: dict[str, Any]) -> AgentConfig:
        self._check_edit()
        name = name.strip().lower()
        with self._lock:
            existing = self.agents.get(name)
            if existing is not None and existing.source != "stored":
                raise InvalidRequest(
                    f"agent '{name}' comes from the configuration (file or environment); change it there", param="name"
                )
            agent = AgentConfig.from_dict(name, entry, "stored")
            self.agents[name] = agent
            self._save_stored()
            return agent

    def delete_agent(self, name: str) -> None:
        self._check_edit()
        name = name.strip().lower()
        with self._lock:
            existing = self.agents.get(name)
            if existing is None:
                raise ModelNotFound(f"unknown agent '{name}'")
            if existing.source != "stored":
                raise InvalidRequest(
                    f"agent '{name}' comes from the configuration (file or environment); remove it there", param="name"
                )
            del self.agents[name]
            self._save_stored()

    def put_number(self, number: str, agent: str) -> None:
        self._check_edit()
        key = normalize_number(number)
        agent = agent.strip().lower()
        if not key:
            raise InvalidRequest("the number is empty", param="number")
        with self._lock:
            if agent not in self.agents:
                raise InvalidRequest(f"unknown agent '{agent}'", param="agent")
            existing = self.numbers.get(key)
            if existing is not None and existing[2] != "stored":
                raise InvalidRequest(
                    f"the number {number} is mapped in the configuration (file or environment); change it there",
                    param="number",
                )
            self.numbers[key] = (agent, str(number).strip(), "stored")
            self._save_stored()

    def delete_number(self, number: str) -> None:
        self._check_edit()
        key = normalize_number(number)
        with self._lock:
            existing = self.numbers.get(key)
            if existing is None:
                raise ModelNotFound(f"the number {number} is not mapped to an agent")
            if existing[2] != "stored":
                raise InvalidRequest(
                    f"the number {number} is mapped in the configuration (file or environment); remove it there",
                    param="number",
                )
            del self.numbers[key]
            self._save_stored()

    # -- lookups ------------------------------------------------------------------------------------------
    def provider(self, name: str) -> ProviderSettings:
        return self.providers.get(name) or ProviderSettings(name)

    def agent_for(self, number: Any, explicit: str | None = None) -> AgentConfig | None:
        """The agent that answers ``number``: one named explicitly, the number's mapping (exact, then by its
        last ten digits, so national and international spellings meet), or the default agent."""
        if explicit:
            return self.agents.get(str(explicit).strip().lower())
        key = normalize_number(number)
        entry = self.numbers.get(key)
        if entry is None and len(key) >= 10:
            entry = next((v for k, v in self.numbers.items() if len(k) >= 10 and k[-10:] == key[-10:]), None)
        if entry is not None and entry[0] in self.agents:
            return self.agents[entry[0]]
        if self.default_agent:
            return self.agents.get(self.default_agent)
        return None

    def numbers_public(self) -> list[dict[str, Any]]:
        return [
            {"number": written, "agent": agent, "source": source, "editable": source == "stored"}
            for agent, written, source in self.numbers.values()
        ]

    def secrets(self) -> list[str]:
        return [s for p in self.providers.values() for s in p.secrets()]
