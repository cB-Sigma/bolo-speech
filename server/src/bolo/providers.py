"""Cloud providers: who they are, their keys, what they cost and how they did.

A *provider* is a named account at a speech API (or any server that speaks the same protocol): a ``kind`` that
picks the adapter (:mod:`bolo.remote`), a base URL, an API key and options. Providers come from the settings
(``providers`` in the configuration file or the environment; read-only at run time) and from
``<data_dir>/providers.json`` (added through the API; the file is written with mode 0600).

Keys never leave this module in the clear: :meth:`ProviderConfig.public` and :func:`redact_provider` only say
whether one is set, and :func:`scrub` removes a key from any text that came back from a provider before it is
logged or shown.

The registry also keeps per-provider counters (requests, errors, latency, bytes, audio seconds, characters) and
turns usage into an *estimated* cost when the provider's ``prices`` table has an entry for the model. Prices
are whatever the operator wrote down; bolo ships none.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bolo.errors import InvalidRequest, ModelNotFound, ProviderNotConfigured

if TYPE_CHECKING:
    from bolo.config import Settings

log = logging.getLogger("bolo.providers")

KINDS = ("openai-compatible", "sarvam", "elevenlabs", "deepgram")
DEFAULT_BASE_URLS = {
    "openai-compatible": "https://api.openai.com/v1",
    "sarvam": "https://api.sarvam.ai",
    "elevenlabs": "https://api.elevenlabs.io",
    "deepgram": "https://api.deepgram.com",
}
#: provider names whose kind need not be spelled out
KIND_BY_NAME = {"openai": "openai-compatible", "sarvam": "sarvam", "elevenlabs": "elevenlabs", "deepgram": "deepgram"}
TERMS = {
    "openai-compatible": "https://openai.com/policies/services-agreement/",
    "sarvam": "https://www.sarvam.ai/terms-of-service",
    "elevenlabs": "https://elevenlabs.io/terms-of-use",
    "deepgram": "https://deepgram.com/terms",
}
LOOPBACK = ("127.0.0.1", "localhost", "::1")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_SECRET_RE = re.compile(r"key|token|secret|password|authorization", re.IGNORECASE)
#: price units: what one unit of each is measured in
PRICE_UNITS = ("per_minute", "per_hour", "per_1k_chars", "per_1m_chars", "per_request")


def redact_provider(entry: Any) -> dict[str, Any]:
    """A provider entry as it may be shown: the key and any secret-looking option replaced."""
    if not isinstance(entry, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "api_key":
            out[key] = "***" if value else None
        elif key == "options" and isinstance(value, dict):
            out[key] = {str(k): ("***" if _SECRET_RE.search(str(k)) and v else v) for k, v in value.items()}
        else:
            out[key] = value
    return out


def scrub(text: str, *secrets: str | None) -> str:
    """``text`` without any of ``secrets`` (provider error messages sometimes echo the key)."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


@dataclass
class ProviderConfig:
    name: str
    kind: str
    base_url: str
    api_key: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    #: remote model name (or "*") -> {per_minute, per_hour, per_1k_chars, per_1m_chars, per_request, currency}
    prices: dict[str, Any] = field(default_factory=dict)
    #: "settings" (configuration file or environment; read-only) or "stored" (added through the API)
    source: str = "settings"

    @classmethod
    def from_dict(cls, name: str, entry: dict[str, Any], source: str = "settings") -> ProviderConfig:
        name = str(name).strip().lower()
        if not _NAME_RE.match(name):
            raise InvalidRequest(
                f"invalid provider name {name!r}: lowercase letters, digits, '-' and '_' (at most 40)", param="name"
            )
        if not isinstance(entry, dict):
            raise InvalidRequest(f"provider '{name}' must be a table of settings", param="providers")
        unknown = set(entry) - {"kind", "base_url", "api_key", "options", "prices"}
        if unknown:
            raise InvalidRequest(
                f"provider '{name}' has unknown fields: {', '.join(sorted(unknown))}", param="providers"
            )
        kind = str(entry.get("kind") or KIND_BY_NAME.get(name) or "").strip().lower()
        if kind in ("openai", "openai_compatible"):
            kind = "openai-compatible"
        if kind not in KINDS:
            raise InvalidRequest(
                f"provider '{name}': kind must be one of {', '.join(KINDS)}" + (f", not {kind!r}" if kind else ""),
                param="kind",
            )
        base_url = str(entry.get("base_url") or DEFAULT_BASE_URLS[kind]).strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise InvalidRequest(f"provider '{name}': base_url must start with http:// or https://", param="base_url")
        options = entry.get("options") or {}
        prices = entry.get("prices") or {}
        if not isinstance(options, dict) or not isinstance(prices, dict):
            raise InvalidRequest(f"provider '{name}': options and prices must be tables", param="options")
        for model, price in prices.items():
            if not isinstance(price, dict) or not all(
                k == "currency" or (k in PRICE_UNITS and isinstance(v, (int, float))) for k, v in price.items()
            ):
                raise InvalidRequest(
                    f"provider '{name}': the price of '{model}' must be a table of {', '.join(PRICE_UNITS)} "
                    "(numbers) and an optional currency",
                    param="prices",
                )
        key = entry.get("api_key")
        return cls(name, kind, base_url, str(key) if key else None, dict(options), dict(prices), source)

    @property
    def is_default_host(self) -> bool:
        return self.base_url == DEFAULT_BASE_URLS[self.kind]

    @property
    def configured(self) -> bool:
        """Usable: a key is set, or it is a self-hosted OpenAI-compatible server (those often need none)."""
        return bool(self.api_key) or (self.kind == "openai-compatible" and not self.is_default_host)

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "configured": self.configured,
            "api_key_set": bool(self.api_key),
            "options": redact_provider({"options": self.options})["options"],
            "prices": self.prices,
            "source": self.source,
            "editable": self.source == "stored",
            "terms_url": TERMS[self.kind] if self.is_default_host else None,
        }

    def stored(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "options": self.options,
            "prices": self.prices,
        }

    def price_for(self, model: str) -> dict[str, Any] | None:
        price = self.prices.get(model) or self.prices.get(model.lower()) or self.prices.get("*")
        return dict(price) if isinstance(price, dict) else None

    def estimate(
        self, model: str, *, audio_seconds: float = 0.0, chars: int = 0, requests: int = 1
    ) -> dict[str, Any] | None:
        """``{"amount", "currency", "estimated": True}`` from the price table, or None without a price."""
        price = self.price_for(model)
        if not price:
            return None
        amount = (
            float(price.get("per_minute", 0.0)) * audio_seconds / 60.0
            + float(price.get("per_hour", 0.0)) * audio_seconds / 3600.0
            + float(price.get("per_1k_chars", 0.0)) * chars / 1000.0
            + float(price.get("per_1m_chars", 0.0)) * chars / 1_000_000.0
            + float(price.get("per_request", 0.0)) * requests
        )
        return {"amount": round(amount, 8), "currency": str(price.get("currency") or "USD"), "estimated": True}


@dataclass
class Usage:
    """What one request (or one piece of one) cost at a provider. Filled by the adapters, summed by the caller."""

    provider: str = ""
    kind: str = ""
    model: str = ""
    requests: int = 0
    latency_ms: float = 0.0
    #: time until the first byte of the answer, of the first request only
    first_byte_ms: float | None = None
    bytes_sent: int = 0
    bytes_received: int = 0
    audio_seconds: float = 0.0
    chars: int = 0
    cost: dict[str, Any] | None = None

    def add(self, other: Usage) -> None:
        self.provider, self.kind, self.model = other.provider, other.kind, other.model
        self.requests += other.requests
        self.latency_ms += other.latency_ms
        if self.first_byte_ms is None:
            self.first_byte_ms = other.first_byte_ms
        self.bytes_sent += other.bytes_sent
        self.bytes_received += other.bytes_received
        self.audio_seconds += other.audio_seconds
        self.chars += other.chars
        if other.cost:
            if self.cost and self.cost.get("currency") == other.cost.get("currency"):
                self.cost = {**self.cost, "amount": round(self.cost["amount"] + other.cost["amount"], 8)}
            elif not self.cost:
                self.cost = dict(other.cost)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.provider,
            "kind": self.kind,
            "model": self.model,
            "requests": self.requests,
            "latency_ms": round(self.latency_ms, 1),
            "first_byte_ms": None if self.first_byte_ms is None else round(self.first_byte_ms, 1),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }
        return out


@dataclass
class ProviderStats:
    requests: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    audio_seconds: float = 0.0
    chars: int = 0
    cost: dict[str, float] = field(default_factory=dict)  # by currency
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "latency_ms_mean": round(self.latency_ms / self.requests, 1) if self.requests else None,
            "latency_ms_total": round(self.latency_ms, 1),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "audio_seconds": round(self.audio_seconds, 2),
            "chars": self.chars,
            "estimated_cost": {c: round(a, 6) for c, a in self.cost.items()},
            "last_error": self.last_error,
        }


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._providers: dict[str, ProviderConfig] = {}
        self._stats: dict[tuple[str, str], ProviderStats] = {}
        #: bumped on every change, so loaded models know their provider's settings are stale
        self.generation = 0
        #: (requested model, model that answered) -> count
        self.fallbacks: dict[tuple[str, str], int] = {}
        for name, entry in dict(settings.providers or {}).items():
            cfg = ProviderConfig.from_dict(str(name), entry, "settings")
            self._providers[cfg.name] = cfg
        self._path = Path(settings.data_dir) / "providers.json"
        self._load_stored()

    # -- stored providers ---------------------------------------------------------------------------------
    def _load_stored(self) -> None:
        if not self._path.is_file():
            return
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s", self._path, exc)
            return
        for name, entry in (doc if isinstance(doc, dict) else {}).items():
            try:
                cfg = ProviderConfig.from_dict(str(name), entry, "stored")
            except InvalidRequest as exc:
                log.warning("ignoring stored provider %r: %s", name, exc.message)
                continue
            # the configuration wins over a stored provider of the same name
            self._providers.setdefault(cfg.name, cfg)

    def _save_stored(self) -> None:
        doc = {p.name: p.stored() for p in self._providers.values() if p.source == "stored"}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        with contextlib.suppress(OSError):
            tmp.chmod(0o600)
        tmp.replace(self._path)

    def edit_blocked_reason(self) -> str | None:
        s = self.settings
        if s.host not in LOOPBACK and not s.api_key:
            return "providers are never changed through the API on a non-loopback address without an API key"
        if s.allow_provider_edit is False:
            return "the operator disabled provider changes through the API (allow_provider_edit = false)"
        return None

    def put(self, name: str, entry: dict[str, Any]) -> ProviderConfig:
        """Add or change a stored provider. An absent or empty ``api_key`` keeps the key already stored;
        ``api_key: null`` with ``clear_api_key`` handled by the caller removes it."""
        reason = self.edit_blocked_reason()
        if reason:
            raise InvalidRequest(reason, param="providers")
        name = name.strip().lower()
        with self._lock:
            existing = self._providers.get(name)
            if existing is not None and existing.source != "stored":
                raise InvalidRequest(
                    f"provider '{name}' comes from the configuration (file or environment); change it there",
                    param="name",
                )
            merged = dict(entry)
            clear = bool(merged.pop("clear_api_key", False))
            if not merged.get("api_key"):
                merged["api_key"] = None if clear or existing is None else existing.api_key
            if existing is not None:
                merged.setdefault("kind", existing.kind)
                if "base_url" not in merged and merged["kind"] == existing.kind:
                    merged["base_url"] = existing.base_url
                merged.setdefault("options", existing.options)
                merged.setdefault("prices", existing.prices)
            cfg = ProviderConfig.from_dict(name, merged, "stored")
            self._providers[name] = cfg
            self.generation += 1
            self._save_stored()
        log.info("provider '%s' (%s at %s) saved", cfg.name, cfg.kind, cfg.base_url)
        return cfg

    def delete(self, name: str) -> None:
        reason = self.edit_blocked_reason()
        if reason:
            raise InvalidRequest(reason, param="providers")
        with self._lock:
            existing = self._providers.get(name.strip().lower())
            if existing is None:
                raise ModelNotFound(f"unknown provider '{name}'")
            if existing.source != "stored":
                raise InvalidRequest(
                    f"provider '{name}' comes from the configuration (file or environment); remove it there",
                    param="name",
                )
            del self._providers[existing.name]
            self.generation += 1
            self._save_stored()

    # -- lookup -----------------------------------------------------------------------------------------------
    def all(self) -> list[ProviderConfig]:
        return list(self._providers.values())

    def get(self, name: str) -> ProviderConfig | None:
        return self._providers.get(name.strip().lower())

    def find(self, name: str | None, kind: str | None = None) -> ProviderConfig | None:
        """The provider called ``name``; failing that, a configured provider of ``kind`` at the vendor's own
        address (so the catalog's ``provider: openai`` entries work whatever the account was called)."""
        if name:
            found = self.get(name)
            if found is not None:
                return found
        if kind:
            for cfg in self._providers.values():
                if cfg.kind == kind and cfg.is_default_host and cfg.configured:
                    return cfg
        return None

    def require(self, name: str | None, kind: str | None = None, model: str = "") -> ProviderConfig:
        cfg = self.find(name, kind)
        what = f"model '{model}' " if model else ""
        if cfg is None:
            raise ProviderNotConfigured(
                f"{what}needs the cloud provider '{name or kind}', which is not configured: add a "
                f"[providers.{name or kind}] table with its api_key to the configuration (or set "
                f"BOLO_PROVIDER_{(name or kind or '').upper().replace('-', '_')}_API_KEY), or add it on the "
                "console's Providers page"
            )
        if not cfg.configured:
            raise ProviderNotConfigured(f"{what}needs the cloud provider '{cfg.name}', which has no API key yet")
        return cfg

    def public(self) -> list[dict[str, Any]]:
        out = []
        for cfg in self._providers.values():
            entry = cfg.public()
            entry["stats"] = self.stats_for(cfg.name)
            out.append(entry)
        return out

    # -- accounting -------------------------------------------------------------------------------------------
    def record(self, usage: Usage, *, error: str | None = None) -> None:
        with self._lock:
            st = self._stats.setdefault((usage.provider, usage.model), ProviderStats())
            st.requests += usage.requests
            st.latency_ms += usage.latency_ms
            st.bytes_sent += usage.bytes_sent
            st.bytes_received += usage.bytes_received
            st.audio_seconds += usage.audio_seconds
            st.chars += usage.chars
            if usage.cost:
                currency = str(usage.cost.get("currency") or "USD")
                st.cost[currency] = st.cost.get(currency, 0.0) + float(usage.cost.get("amount") or 0.0)
            if error is not None:
                st.errors += 1
                st.last_error = error[:300]

    def note_fallback(self, requested: str, answered: str) -> None:
        with self._lock:
            self.fallbacks[(requested, answered)] = self.fallbacks.get((requested, answered), 0) + 1

    def stats_for(self, name: str) -> dict[str, Any]:
        with self._lock:
            return {model: st.to_dict() for (provider, model), st in self._stats.items() if provider == name}

    def stats(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"provider": provider, "model": model, **st.to_dict()} for (provider, model), st in self._stats.items()
            ]

    # -- connection test --------------------------------------------------------------------------------------
    def test(self, name: str) -> dict[str, Any]:
        """Ask the provider something cheap that needs the key. Never raises for a provider-side failure."""
        cfg = self.get(name)
        if cfg is None:
            raise ModelNotFound(f"unknown provider '{name}'")
        from bolo.errors import BoloError
        from bolo.remote import make_adapter

        t0 = time.perf_counter()
        adapter = make_adapter(cfg, "", {}, timeout_s=min(15.0, self.settings.provider_timeout_s))
        try:
            detail = adapter.check()
            ok, message, code = True, str(detail.get("message") or "ok"), None
        except BoloError as exc:
            detail, ok, message, code = {}, False, scrub(exc.message, cfg.api_key), exc.code
        finally:
            adapter.close()
        return {
            "name": cfg.name,
            "ok": ok,
            "message": message,
            "code": code,
            #: "credentials": the key was accepted; "reachability": only that the address answers
            "checked": detail.get("checked", "credentials") if ok else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
