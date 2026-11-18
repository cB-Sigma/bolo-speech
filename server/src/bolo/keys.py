"""API keys with scopes, per-key rate limits and quotas.

The ``api_key`` setting stays what it was: one root key with every scope. On top of it, a key store
(``<data_dir>/api-keys.json``, mode 0600) holds named keys, each with

* scopes - what the key may call (:data:`SCOPES`; ``admin`` implies all of them),
* a rate limit - a token bucket of ``rate_limit`` requests per minute with room for ``burst`` at once,
* a quota - at most ``quota`` requests per ``quota_period`` (``day``, ``month`` or ``total``),
* usage counters - requests in all, in the current period, by scope, and when the key was last used.

Only a SHA-256 hash of each key is stored; the key itself is shown once, when it is created. Authentication is
on when the ``api_key`` setting is set or the store holds at least one key (revoked and expired keys count:
revoking every key locks the server rather than opening it; only deleting them all opens it again). The store is a
plain file: ``bolo keys create`` writes it with the server stopped or running (a running server reads it again
when it changes).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bolo.errors import AuthError, BoloError, InvalidRequest, Overloaded

log = logging.getLogger("bolo.keys")

#: what a key may do. ``admin`` implies every other scope.
SCOPES: dict[str, str] = {
    "read": "Read-only routes: catalog, models, info, stats, request traces, events, voices and word lists",
    "stt": "Speech-to-text, translation, language identification, diarization, wake words and their streams",
    "tts": "Text-to-speech and its stream, saved voices",
    "realtime": "Realtime and WebRTC sessions, client secrets, chat completions, telephony",
    "manage": "Pull, load, unload and delete models; change model settings, word lists and providers; benchmarks",
    "admin": "Everything, including API key management",
}
PERIODS = ("day", "month", "total")
PREFIX = "bolo_"
FILE_NAME = "api-keys.json"
#: usage counters are written back at most this often (seconds); key changes are written at once
FLUSH_EVERY_S = 10.0


class RateLimited(Overloaded):
    code = "rate_limit_exceeded"


class QuotaExceeded(Overloaded):
    code = "insufficient_quota"


class ScopeError(BoloError):
    status = 403
    code = "insufficient_scope"
    type = "invalid_request_error"


def _now_iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def period_key(period: str, ts: float | None = None) -> str:
    """The name of the quota period ``ts`` falls in (UTC): ``2026-11-18``, ``2026-11`` or ``total``."""
    when = datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc)
    if period == "day":
        return when.strftime("%Y-%m-%d")
    if period == "month":
        return when.strftime("%Y-%m")
    return "total"


def _period_end(period: str, ts: float) -> float | None:
    when = datetime.fromtimestamp(ts, timezone.utc)
    if period == "day":
        start = when.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp() + 86400
    if period == "month":
        nxt = when.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt = nxt.replace(year=nxt.year + 1, month=1) if nxt.month == 12 else nxt.replace(month=nxt.month + 1)
        return nxt.timestamp()
    return None


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def parse_scopes(value: Any) -> list[str]:
    if value is None:
        return ["admin"]
    items = [s.strip() for s in value.split(",")] if isinstance(value, str) else list(value)
    scopes = []
    for s in items:
        s = str(s).strip().lower()
        if not s:
            continue
        if s == "*":
            s = "admin"
        if s not in SCOPES:
            raise InvalidRequest(f"unknown scope {s!r}; scopes are {', '.join(SCOPES)}", param="scopes")
        if s not in scopes:
            scopes.append(s)
    if not scopes:
        raise InvalidRequest("a key needs at least one scope", param="scopes")
    return scopes


def _positive(value: Any, param: str, *, allow_float: bool = False) -> float | int | None:
    if value is None or value in {"", 0}:
        return None
    try:
        number = float(value) if allow_float else int(value)
    except (TypeError, ValueError):
        raise InvalidRequest(f"{param} must be a positive number or null", param=param) from None
    if number <= 0 or (isinstance(number, float) and not math.isfinite(number)):
        raise InvalidRequest(f"{param} must be a positive number or null", param=param)
    return number


@dataclass
class ApiKey:
    id: str
    name: str
    hash: str
    #: the first characters of the key, shown so people can tell keys apart
    hint: str
    scopes: list[str]
    created_at: float
    rate_limit: float | None = None  # requests per minute; None: unlimited
    burst: int | None = None  # bucket size; None: the per-minute limit (at least 1)
    quota: int | None = None  # requests per quota_period; None: unlimited
    quota_period: str = "day"
    expires_at: float | None = None
    revoked_at: float | None = None
    # usage
    requests: int = 0
    period: str = ""  # the quota period the next counter belongs to (period_key)
    period_requests: int = 0
    by_scope: dict[str, int] = field(default_factory=dict)
    last_used_at: float | None = None
    rejected: int = 0
    # token bucket (not stored)
    tokens: float = -1.0
    refilled_at: float = 0.0

    @property
    def active(self) -> bool:
        return self.revoked_at is None and (self.expires_at is None or self.expires_at > time.time())

    def allows(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes or scope == "read"

    def stored(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "hash": self.hash, "hint": self.hint, "scopes": self.scopes,
            "created_at": self.created_at, "rate_limit": self.rate_limit, "burst": self.burst, "quota": self.quota,
            "quota_period": self.quota_period, "expires_at": self.expires_at, "revoked_at": self.revoked_at,
            "requests": self.requests, "period": self.period, "period_requests": self.period_requests,
            "by_scope": self.by_scope, "last_used_at": self.last_used_at, "rejected": self.rejected,
        }  # fmt: skip

    @classmethod
    def from_stored(cls, d: dict[str, Any]) -> ApiKey:
        return cls(
            id=str(d["id"]), name=str(d.get("name") or d["id"]), hash=str(d["hash"]), hint=str(d.get("hint") or ""),
            scopes=[s for s in d.get("scopes") or ["admin"] if s in SCOPES] or ["read"],
            created_at=float(d.get("created_at") or 0), rate_limit=d.get("rate_limit"), burst=d.get("burst"),
            quota=d.get("quota"), quota_period=str(d.get("quota_period") or "day"), expires_at=d.get("expires_at"),
            revoked_at=d.get("revoked_at"), requests=int(d.get("requests") or 0), period=str(d.get("period") or ""),
            period_requests=int(d.get("period_requests") or 0), by_scope=dict(d.get("by_scope") or {}),
            last_used_at=d.get("last_used_at"), rejected=int(d.get("rejected") or 0),
        )  # fmt: skip

    def public(self) -> dict[str, Any]:
        """What the API returns: never the hash."""
        now = time.time()
        used = self.period_requests if self.period == period_key(self.quota_period, now) else 0
        return {
            "id": self.id,
            "name": self.name,
            "hint": self.hint,
            "scopes": self.scopes,
            "created_at": _now_iso(self.created_at),
            "expires_at": _now_iso(self.expires_at) if self.expires_at else None,
            "revoked_at": _now_iso(self.revoked_at) if self.revoked_at else None,
            "active": self.active,
            "rate_limit": self.rate_limit,
            "burst": self.burst,
            "quota": self.quota,
            "quota_period": self.quota_period,
            "usage": {
                "requests": self.requests,
                "period": period_key(self.quota_period, now),
                "period_requests": used,
                "quota_remaining": max(0, self.quota - used) if self.quota else None,
                "by_scope": dict(self.by_scope),
                "rejected": self.rejected,
                "last_used_at": _now_iso(self.last_used_at) if self.last_used_at else None,
            },
        }


@dataclass
class Principal:
    """Who made a request: the root key (``api_key`` setting), a stored key, or nobody (authentication off)."""

    key: ApiKey | None
    root: bool = False

    @property
    def id(self) -> str | None:
        return "root" if self.root else (self.key.id if self.key else None)

    def allows(self, scope: str) -> bool:
        return self.key is None or self.key.allows(scope)


OPEN = Principal(None)
ROOT = Principal(None, root=True)


class KeyStore:
    def __init__(self, data_dir: Path | str, root_key: str | None = None) -> None:
        self.path = Path(data_dir) / FILE_NAME
        self.root_key = root_key or None
        self._lock = threading.Lock()
        self._keys: dict[str, ApiKey] = {}
        self._by_hash: dict[str, ApiKey] = {}
        self._stamp: tuple[int, int] | None = None
        self._dirty = False
        self._flushed_at = 0.0
        self._load()

    # -- file -----------------------------------------------------------------------------------------------
    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _load(self) -> None:
        stamp = self._file_stamp()
        keys: dict[str, ApiKey] = {}
        if stamp is not None:
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
                for entry in doc.get("keys", []) if isinstance(doc, dict) else []:
                    try:
                        key = ApiKey.from_stored(entry)
                    except (KeyError, TypeError, ValueError) as exc:
                        log.warning("ignoring a malformed entry in %s: %s", self.path, exc)
                        continue
                    old = self._keys.get(key.id)
                    if old is not None:  # keep the bucket across reloads
                        key.tokens, key.refilled_at = old.tokens, old.refilled_at
                    keys[key.id] = key
            except (OSError, ValueError) as exc:
                log.warning("could not read %s: %s", self.path, exc)
                return
        self._keys = keys
        self._by_hash = {k.hash: k for k in keys.values()}
        self._stamp = stamp

    def _reload_if_changed(self) -> None:
        if self._file_stamp() != self._stamp:
            if self._dirty:  # counters not yet written: write them first, then read the other writer's changes
                self._write()
            self._load()

    def _write(self) -> None:
        doc = {"version": 1, "keys": [k.stored() for k in self._keys.values()]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        with contextlib.suppress(OSError):
            tmp.chmod(0o600)
        tmp.replace(self.path)
        self._stamp = self._file_stamp()
        self._dirty = False
        self._flushed_at = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._write()

    # -- state ----------------------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether requests need a key."""
        if self.root_key:
            return True
        with self._lock:
            self._reload_if_changed()
            return bool(self._keys)

    def all(self) -> list[ApiKey]:
        with self._lock:
            self._reload_if_changed()
            return sorted(self._keys.values(), key=lambda k: k.created_at)

    def get(self, key_id: str) -> ApiKey:
        with self._lock:
            self._reload_if_changed()
            key = self._keys.get(key_id)
        if key is None:
            raise InvalidRequest(f"no API key with id {key_id!r}", param="id")
        return key

    # -- changes --------------------------------------------------------------------------------------------
    def create(
        self,
        name: str,
        scopes: Any = None,
        *,
        rate_limit: Any = None,
        burst: Any = None,
        quota: Any = None,
        quota_period: str = "day",
        expires_in_days: Any = None,
    ) -> tuple[ApiKey, str]:
        """A new key and its secret value (shown once)."""
        name = (name or "").strip()
        if not name or len(name) > 80:
            raise InvalidRequest("a key needs a name of 1 to 80 characters", param="name")
        if quota_period not in PERIODS:
            raise InvalidRequest(f"quota_period must be one of {', '.join(PERIODS)}", param="quota_period")
        days = _positive(expires_in_days, "expires_in_days", allow_float=True)
        secret = PREFIX + secrets.token_urlsafe(30)
        now = time.time()
        key = ApiKey(
            id="key_" + secrets.token_hex(6),
            name=name,
            hash=hash_key(secret),
            hint=secret[: len(PREFIX) + 4] + "…" + secret[-4:],
            scopes=parse_scopes(scopes),
            created_at=now,
            rate_limit=_positive(rate_limit, "rate_limit", allow_float=True),
            burst=_positive(burst, "burst"),  # type: ignore[arg-type]
            quota=_positive(quota, "quota"),  # type: ignore[arg-type]
            quota_period=quota_period,
            expires_at=now + float(days) * 86400 if days else None,
        )
        with self._lock:
            self._reload_if_changed()
            self._keys[key.id] = key
            self._by_hash[key.hash] = key
            self._write()
        return key, secret

    def update(self, key_id: str, changes: dict[str, Any]) -> ApiKey:
        with self._lock:
            self._reload_if_changed()
            key = self._keys.get(key_id)
            if key is None:
                raise InvalidRequest(f"no API key with id {key_id!r}", param="id")
            if "name" in changes and changes["name"] is not None:
                name = str(changes["name"]).strip()
                if not name or len(name) > 80:
                    raise InvalidRequest("a key needs a name of 1 to 80 characters", param="name")
                key.name = name
            if "scopes" in changes and changes["scopes"] is not None:
                key.scopes = parse_scopes(changes["scopes"])
            if "rate_limit" in changes:
                key.rate_limit = _positive(changes["rate_limit"], "rate_limit", allow_float=True)
                key.tokens = -1.0
            if "burst" in changes:
                key.burst = _positive(changes["burst"], "burst")  # type: ignore[assignment]
                key.tokens = -1.0
            if "quota" in changes:
                key.quota = _positive(changes["quota"], "quota")  # type: ignore[assignment]
            if changes.get("quota_period") is not None:
                if changes["quota_period"] not in PERIODS:
                    raise InvalidRequest(f"quota_period must be one of {', '.join(PERIODS)}", param="quota_period")
                if changes["quota_period"] != key.quota_period:
                    key.quota_period = changes["quota_period"]
                    key.period, key.period_requests = "", 0
            if changes.get("reset_usage"):
                key.period, key.period_requests = "", 0
            self._write()
            return key

    def revoke(self, key_id: str, *, delete: bool = False) -> None:
        with self._lock:
            self._reload_if_changed()
            key = self._keys.get(key_id)
            if key is None:
                raise InvalidRequest(f"no API key with id {key_id!r}", param="id")
            if delete:
                del self._keys[key_id]
                self._by_hash.pop(key.hash, None)
            elif key.revoked_at is None:
                key.revoked_at = time.time()
            self._write()

    # -- requests -------------------------------------------------------------------------------------------
    def authenticate(self, presented: str | None) -> Principal:
        """Who ``presented`` is. Raises :class:`AuthError` when keys are needed and it is not a valid one."""
        if not self.enabled:
            return OPEN
        if not presented:
            raise AuthError("API key required. Provide it in the Authorization header using the Bearer scheme.")
        if self.root_key and hmac.compare_digest(presented.encode(), self.root_key.encode()):
            return ROOT
        with self._lock:
            key = self._by_hash.get(hash_key(presented))
        if key is None:
            raise AuthError("Incorrect API key provided.")
        if key.revoked_at is not None:
            raise AuthError("This API key has been revoked.")
        if not key.active:
            raise AuthError("This API key has expired.")
        return Principal(key)

    def admit(self, principal: Principal, scope: str) -> None:
        """Check the scope, take a token from the key's bucket and count the request against its quota."""
        key = principal.key
        if key is None:
            return
        if not key.allows(scope):
            raise ScopeError(
                f"this API key does not have the {scope!r} scope (it has: {', '.join(key.scopes)})", param=scope
            )
        now = time.time()
        with self._lock:
            period = period_key(key.quota_period, now)
            if key.period != period:
                key.period, key.period_requests = period, 0
            if key.quota is not None and key.period_requests >= key.quota:
                key.rejected += 1
                self._dirty = True
                end = _period_end(key.quota_period, now)
                raise QuotaExceeded(
                    f"this API key has used its quota of {key.quota} requests per {key.quota_period}",
                    retry_after_s=(end - now) if end else 86400.0,
                )
            if key.rate_limit:
                capacity = float(key.burst or max(1.0, key.rate_limit))
                rate = key.rate_limit / 60.0
                if key.tokens < 0:
                    key.tokens, key.refilled_at = capacity, now
                key.tokens = min(capacity, key.tokens + (now - key.refilled_at) * rate)
                key.refilled_at = now
                if key.tokens < 1.0:
                    key.rejected += 1
                    self._dirty = True
                    raise RateLimited(
                        f"rate limit of {key.rate_limit:g} requests per minute reached for this API key",
                        retry_after_s=(1.0 - key.tokens) / rate,
                    )
                key.tokens -= 1.0
            key.requests += 1
            key.period_requests += 1
            key.by_scope[scope] = key.by_scope.get(scope, 0) + 1
            key.last_used_at = now
            self._dirty = True
            if time.monotonic() - self._flushed_at >= FLUSH_EVERY_S:
                with contextlib.suppress(OSError):
                    self._write()


# -- which scope a route needs ----------------------------------------------------------------------------
_STT = (
    "/v1/audio/transcriptions", "/v1/audio/translations", "/v1/audio/language", "/api/v1/diarize",
    "/api/v1/stt/stream", "/api/v1/wake", "/v1/audio/diarize", "/v1/audio/denoise", "/v1/audio-isolation",
    "/v1/audio/tags", "/v1/audio/emotion", "/v1/audio/speakers/verify", "/v1/audio/speakers/identify",
)  # fmt: skip
_TTS = ("/v1/audio/speech", "/api/v1/tts/stream")
_REALTIME = ("/v1/realtime", "/v1/chat/completions", "/api/v1/telephony", "/api/v1/agents")


def scope_for(method: str, path: str) -> str:
    """The scope a request needs."""
    path = "/" + path.lstrip("/")
    if path.startswith("/api/v1/keys"):
        return "admin"
    if path.startswith(_STT):
        return "stt"
    if path.startswith(_TTS):
        return "tts"
    if path.startswith("/api/v1/voices"):
        return "read" if method in ("GET", "HEAD") else "tts"
    if path.startswith(_REALTIME):
        return "realtime"
    if method in ("GET", "HEAD", "OPTIONS"):
        return "read"
    return "manage"
