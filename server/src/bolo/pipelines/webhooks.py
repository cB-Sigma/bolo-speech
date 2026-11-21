"""Signed webhooks: ``webhook`` steps and the callbacks of asynchronous runs (docs/pipelines.md, "Webhooks and
signatures").

``X-Bolo-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256 of "<t>.<body>">`` with the ``webhook_secret`` setting, or
a secret bolo generates once into ``<data_dir>/webhook-secret``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Any

import httpx

from bolo.config import Settings

SECRET_FILE = "webhook-secret"
#: seconds before each retry of a callback that failed
RETRIES_S = (1.0, 5.0, 25.0)


def secret(settings: Settings) -> str:
    if settings.webhook_secret:
        return settings.webhook_secret
    path = Path(settings.data_dir) / SECRET_FILE
    with contextlib.suppress(OSError):
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = "whsec_" + secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value + "\n")
    return value


def signature(key: str, body: bytes, t: int | None = None) -> str:
    t = int(time.time()) if t is None else t
    mac = hmac.new(key.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


def verify(key: str, body: bytes, header: str, tolerance_s: float = 300.0) -> bool:
    """Check a ``X-Bolo-Signature`` (for receivers written in Python, and the tests)."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        t = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - t) > tolerance_s:
        return False
    expected = signature(key, body, t).split("v1=", 1)[1]
    return hmac.compare_digest(expected, parts.get("v1", ""))


async def send(
    settings: Settings,
    url: str,
    body: bytes,
    *,
    event: str,
    run_id: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> httpx.Response:
    sent = {
        "Content-Type": "application/json",
        "User-Agent": "bolo-pipelines",
        **(headers or {}),
        "X-Bolo-Event": event,
        "X-Bolo-Run": run_id,
        "X-Bolo-Signature": signature(secret(settings), body),
    }
    timeout = httpx.Timeout(timeout_s or settings.pipeline_webhook_timeout_s)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        return await client.request(method, url, content=body if method != "GET" else None, headers=sent)


async def deliver(settings: Settings, url: str, body: bytes, *, event: str, run_id: str) -> list[dict[str, Any]]:
    """POST a callback, retrying a failure (:data:`RETRIES_S`). Returns each attempt."""
    attempts: list[dict[str, Any]] = []
    for i in range(len(RETRIES_S) + 1):
        try:
            resp = await send(settings, url, body, event=event, run_id=run_id)
            attempts.append({"status": resp.status_code, "at": time.time()})
            if resp.status_code < 400:
                break
        except httpx.HTTPError as exc:
            attempts.append({"error": f"{type(exc).__name__}: {exc}", "at": time.time()})
        if i < len(RETRIES_S):
            await asyncio.sleep(RETRIES_S[i])
    return attempts
