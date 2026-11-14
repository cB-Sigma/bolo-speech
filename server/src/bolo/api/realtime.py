"""``WS /v1/realtime``: the OpenAI Realtime API (see :mod:`bolo.realtime`)."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from bolo import realtime
from bolo.api.common import _presented_key, check_auth
from bolo.core import Bolo
from bolo.errors import AuthError, BoloError, EngineUnavailable, InvalidRequest, ModelNotFound
from bolo.keepalive import request_keep_alive

router = APIRouter()

KEY_PROTOCOL = "openai-insecure-api-key."


@router.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    """Query: ``model`` (reported back; the language model is the operator's ``llm_model``),
    ``intent=transcription`` for a transcription-only session.

    Authentication, when the server has an API key: ``Authorization: Bearer <key>`` (what the
    OpenAI SDKs send), or for browsers the subprotocols ``realtime`` and
    ``openai-insecure-api-key.<key>``, or ``?api_key=<key>``. In each place an ephemeral client
    secret (``ek_...`` from ``POST /v1/realtime/client_secrets``) is accepted instead of the server
    key; the session then starts with the configuration the secret was created with.
    """
    core: Bolo = ws.app.state.core
    offered = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    await ws.accept(subprotocol="realtime" if "realtime" in offered else None)

    async def send(event: dict[str, Any]) -> None:
        await ws.send_text(json.dumps(event, ensure_ascii=False))

    config: dict[str, Any] | None = None
    try:
        presented = [_presented_key(ws.headers, ws.query_params)]
        presented += [p[len(KEY_PROTOCOL) :] for p in offered if p.startswith(KEY_PROTOCOL)]
        config = next((c for c in (realtime.client_secret_session(core, k) for k in presented) if c is not None), None)
        if config is None:
            try:
                check_auth(core, ws)
            except AuthError:
                expected = core.settings.api_key
                if not any(p == KEY_PROTOCOL + str(expected) for p in offered):
                    raise
    except BoloError as exc:
        with contextlib.suppress(Exception):
            await send(
                {
                    "type": "error",
                    "event_id": realtime.new_id("event"),
                    "error": {
                        "type": exc.type,
                        "code": exc.code,
                        "message": exc.message,
                        "param": None,
                        "event_id": None,
                    },
                }
            )
            await ws.close(code=1008)
        return

    async def receive() -> str | bytes | None:
        try:
            message = await ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None
        if message["type"] == "websocket.disconnect":
            return None
        return message.get("text") or message.get("bytes") or "{}"

    params = ws.query_params
    try:
        # bolo extension (Ollama style): ?keep_alive= for the speech models the session uses, applied when it ends
        ka = request_keep_alive(params.get("keep_alive"), ("stt", "tts"), session=True)
    except BoloError as exc:
        with contextlib.suppress(Exception):
            await send({"type": "error", "event_id": realtime.new_id("event"), "error": {
                "type": exc.type, "code": exc.code, "message": exc.message, "param": exc.param, "event_id": None,
            }})  # fmt: skip
            await ws.close(code=1008)
        return
    try:
        await realtime.serve(
            core, receive, send, model=params.get("model"), transcription=params.get("intent") == "transcription",
            config=config,
        )  # fmt: skip
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if ka is not None:
            with contextlib.suppress(Exception):
                await core.manager.apply_keep_alive(ka.used, ka.seconds)
    with contextlib.suppress(Exception):
        await ws.close()


def _realtime_error(exc: realtime.RealtimeError) -> InvalidRequest:
    return InvalidRequest(exc.message, param=exc.param)


@router.post("/v1/realtime/client_secrets", tags=["openai"], summary="Create an ephemeral key for Realtime sessions")
async def client_secrets(request: Request) -> JSONResponse:
    """Body (all optional): ``{"expires_after": {"anchor": "created_at", "seconds": 600}, "session": {...}}``.

    Needs the server API key. The returned ``value`` (``ek_...``) can be handed to a browser: it opens
    Realtime sessions (WebSocket or WebRTC) until it expires and is not accepted by any other route.
    """
    core: Bolo = request.app.state.core
    check_auth(core, request)
    try:
        body = await request.json() if await request.body() else {}
    except ValueError as exc:
        raise InvalidRequest("the request body must be JSON") from exc
    if not isinstance(body, dict):
        raise InvalidRequest("the request body must be a JSON object")
    expires = body.get("expires_after") or {}
    if not isinstance(expires, dict) or expires.get("anchor", "created_at") != "created_at":
        raise InvalidRequest("expires_after.anchor must be 'created_at'", param="expires_after.anchor")
    try:
        seconds = int(expires.get("seconds", 600))
        return JSONResponse(realtime.mint_client_secret(core, body.get("session"), seconds))
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("expires_after.seconds must be an integer", param="expires_after.seconds") from exc
    except realtime.RealtimeError as exc:
        raise _realtime_error(exc) from exc


@router.post("/v1/realtime/calls", tags=["openai"], summary="Start a Realtime session over WebRTC (SDP offer/answer)")
async def create_call(request: Request) -> Response:
    """Send the SDP offer as the body (``Content-Type: application/sdp``), or as the ``sdp`` field of a
    multipart form together with an optional ``session`` field holding the session configuration as JSON.
    The answer is ``201`` with the SDP answer as the body and the call in the ``Location`` header.

    Authentication: the server API key or an ephemeral client secret. Query: ``model``,
    ``intent=transcription``. Needs the ``webrtc`` extra (aiortc).
    """
    core: Bolo = request.app.state.core
    config = realtime.client_secret_session(core, _presented_key(request.headers, request.query_params))
    if config is None:
        check_auth(core, request)
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    session: Any = None
    if content_type == "multipart/form-data":
        form = await request.form()
        sdp_field = form.get("sdp")
        offer = sdp_field if isinstance(sdp_field, str) else (await sdp_field.read()).decode() if sdp_field else ""
        raw = form.get("session")
        if raw is not None:
            text = raw if isinstance(raw, str) else (await raw.read()).decode()
            try:
                session = json.loads(text) if text.strip() else None
            except ValueError as exc:
                raise InvalidRequest("the 'session' field must be JSON", param="session") from exc
    else:
        offer = (await request.body()).decode(errors="replace")
    if "v=0" not in offer or "m=" not in offer:
        raise InvalidRequest("the body must be an SDP offer (Content-Type: application/sdp)", param="sdp")
    if session is not None and not isinstance(session, dict):
        raise InvalidRequest("the 'session' field must be a JSON object", param="session")
    merged = {**(config or {}), **(session or {})} or None
    try:
        from bolo import webrtc
    except ImportError as exc:
        raise EngineUnavailable(
            "the WebRTC transport needs aiortc, which is not installed: pip install 'bolo-speech[webrtc]'. "
            "The WebSocket transport (WS /v1/realtime) works without it."
        ) from exc
    params = request.query_params
    try:
        call_id, answer = await webrtc.create_call(
            core, offer, config=merged, model=params.get("model"), transcription=params.get("intent") == "transcription"
        )
    except realtime.RealtimeError as exc:
        raise _realtime_error(exc) from exc
    except ValueError as exc:  # aiortc could not parse the offer
        raise InvalidRequest(f"the SDP offer could not be used: {exc}", param="sdp") from exc
    return Response(
        answer, status_code=201, media_type="application/sdp", headers={"Location": f"/v1/realtime/calls/{call_id}"}
    )


@router.post("/v1/realtime/calls/{call_id}/hangup", tags=["openai"], summary="End a WebRTC Realtime call")
async def hangup(request: Request, call_id: str) -> Response:
    core: Bolo = request.app.state.core
    if realtime.client_secret_session(core, _presented_key(request.headers, request.query_params)) is None:
        check_auth(core, request)
    call = core.calls.get(call_id)
    if call is None:
        raise ModelNotFound(f"there is no call '{call_id}'")
    await call.close()
    return Response(status_code=200)
