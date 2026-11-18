"""Helpers shared by the HTTP routers."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse

from bolo.audio import decode_audio
from bolo.core import Bolo
from bolo.engines.base import FloatArray
from bolo.errors import AuthError, BoloError, InvalidRequest, Overloaded
from bolo.keys import Principal, scope_for


def get_core(request: Request) -> Bolo:
    core: Bolo = request.app.state.core
    return core


def error_body(message: str, type_: str, code: str | None, param: str | None = None) -> dict[str, Any]:
    """OpenAI-style error envelope; ``detail`` mirrors it for FastAPI/speaches clients."""
    return {"error": {"message": message, "type": type_, "param": param, "code": code}, "detail": message}


def error_response(exc: BoloError) -> JSONResponse:
    headers = {}
    if isinstance(exc, Overloaded):
        headers["Retry-After"] = str(max(1, round(exc.retry_after_s)))
    if isinstance(exc, AuthError):
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(error_body(exc.message, exc.type, exc.code, exc.param), status_code=exc.status, headers=headers)


def _presented_key(headers: Any, query: Any) -> str | None:
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return str(auth[7:].strip())
    key = headers.get("x-api-key") or query.get("api_key")
    return str(key) if key else None


def check_auth(
    core: Bolo, conn: Request | WebSocket, *, scope: str | None = None, presented: list[str | None] | None = None
) -> Principal:
    """Authenticate a request (the ``api_key`` setting or a stored key, bolo.keys), check that the key has the
    scope the route needs and count the request against the key's rate limit and quota. ``presented``: the keys
    to try instead of the one in the headers or query (the first that is valid wins)."""
    candidates = presented if presented is not None else [_presented_key(conn.headers, conn.query_params)]
    candidates = [c for c in candidates if c] or [None]
    principal: Principal | None = None
    error: AuthError | None = None
    for candidate in candidates:
        try:
            principal = core.keys.authenticate(candidate)
            break
        except AuthError as exc:
            error = error or exc
    if principal is None:
        assert error is not None
        raise error
    method = conn.scope.get("method") or "GET"
    core.keys.admit(principal, scope or scope_for(str(method), conn.url.path))
    conn.state.principal = principal
    return principal


def sse(data: dict[str, Any] | str, event: str | None = None) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


async def read_upload(core: Bolo, file: Any) -> tuple[FloatArray, int]:
    """Read and decode a multipart audio upload (in a thread: decoding can be slow)."""
    import asyncio

    limit = core.settings.max_upload_mb * 1024 * 1024
    data = await file.read()
    if len(data) > limit:
        raise InvalidRequest(f"the upload is larger than the {core.settings.max_upload_mb} MB limit", param="file")
    return await asyncio.to_thread(decode_audio, data, filename=file.filename, content_type=file.content_type)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")
