"""Keep-alive: how long a model stays loaded after its last use, Ollama style.

A value is a number of seconds or a duration (``"300ms"``, ``"30s"``, ``"5m"``, ``"1h30m"``). ``0`` unloads the
model as soon as the request that asked for it ends; any negative value (``-1``, ``"-1m"``) keeps it loaded until it
is unloaded on purpose (``bolo stop``, the unload route) or evicted to make room.

A request names one with an extra field (``keep_alive`` in a form, a JSON body or a query string). The routes put
it in :data:`REQUEST_KEEP_ALIVE` for the duration of the request; the lifecycle manager applies it to the models of
the request's task that the request uses (:meth:`bolo.lifecycle.ModelManager.use`). Sessions (WebSockets, Realtime)
apply it when the session ends, so ``0`` does not reload the model between utterances.
"""

from __future__ import annotations

import contextvars
import math
import re
from dataclasses import dataclass, field
from typing import Any

FOREVER = -1.0

_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|ms|s|m|h|d)")
_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def parse_keep_alive(value: Any) -> float | None:
    """Seconds (``-1.0`` for "forever"), or None when ``value`` is None or empty. Raises ValueError."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("keep_alive must be a number of seconds or a duration such as '5m'")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        if text == "":
            return None
        if text in ("forever", "inf", "infinite", "never"):
            return FOREVER
        if _NUMBER.fullmatch(text):
            seconds = float(text)
        else:
            sign = -1.0 if text.startswith("-") else 1.0
            body = text.lstrip("+-")
            pos, total = 0, 0.0
            for m in _PART.finditer(body):
                if m.start() != pos:
                    break
                total += float(m.group(1)) * _UNITS[m.group(2)]
                pos = m.end()
            if pos != len(body) or not body:
                raise ValueError(f"keep_alive {value!r} is not a number of seconds or a duration such as '5m'")
            seconds = sign * total
    if math.isnan(seconds):
        raise ValueError("keep_alive must be a number")
    if seconds < 0 or math.isinf(seconds):
        return FOREVER
    return seconds


def format_keep_alive(seconds: float | None) -> str:
    """``-1`` -> "forever", ``300`` -> "5m", ``90`` -> "1m30s"."""
    if seconds is None:
        return "default"
    if seconds < 0:
        return "forever"
    if seconds == 0:
        return "0s"
    s = round(seconds)
    if s == 0:
        return f"{int(seconds * 1000)}ms"
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return "".join(f"{n}{u}" for n, u in ((h, "h"), (m, "m"), (sec, "s")) if n) or "0s"


def humanize_until(seconds: float | None, stopping: bool = False) -> str:
    """Ollama's UNTIL column: "4 minutes from now", "Forever", "Stopping..."."""
    if stopping:
        return "Stopping..."
    if seconds is None:
        return "Forever"
    if seconds < 1:
        return "Less than a second from now"
    if seconds < 60:
        n = int(seconds)
        return f"{n} second{'s' if n != 1 else ''} from now"
    if seconds < 120:
        return "About a minute from now"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes from now"
    if seconds < 7200:
        return "About an hour from now"
    if seconds < 86400:
        return f"{int(seconds // 3600)} hours from now"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days != 1 else ''} from now"


@dataclass
class RequestKeepAlive:
    """The keep-alive one request (or session) asked for, and which of its models it applies to."""

    seconds: float
    #: tasks whose models it applies to (a transcription request: stt; not the VAD it uses on the side)
    tasks: frozenset[str] = frozenset({"stt", "tts"})
    #: a session: applied when the session ends (``apply_session_keep_alive``), not after each use
    session: bool = False
    used: set[str] = field(default_factory=set)

    def applies(self, task: str) -> bool:
        return task in self.tasks


REQUEST_KEEP_ALIVE: contextvars.ContextVar[RequestKeepAlive | None] = contextvars.ContextVar(
    "bolo_keep_alive", default=None
)


def request_keep_alive(
    value: Any, tasks: tuple[str, ...], session: bool = False, param: str = "keep_alive"
) -> RequestKeepAlive | None:
    """Parse a request's ``keep_alive`` and make it current for this request; None when it has none."""
    from bolo.errors import InvalidRequest

    try:
        seconds = parse_keep_alive(value)
    except ValueError as exc:
        raise InvalidRequest(str(exc), param=param) from exc
    if seconds is None:
        return None
    ka = RequestKeepAlive(seconds, frozenset(tasks), session)
    REQUEST_KEEP_ALIVE.set(ka)
    return ka
