"""``WS /v1/realtime``: the OpenAI Realtime API (see :mod:`bolo.realtime`)."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from bolo import realtime
from bolo.api.common import check_auth
from bolo.core import Bolo
from bolo.errors import AuthError, BoloError

router = APIRouter()

KEY_PROTOCOL = "openai-insecure-api-key."


@router.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    """Query: ``model`` (reported back; the language model is the operator's ``llm_model``),
    ``intent=transcription`` for a transcription-only session.

    Authentication, when the server has an API key: ``Authorization: Bearer <key>`` (what the
    OpenAI SDKs send), or for browsers the subprotocols ``realtime`` and
    ``openai-insecure-api-key.<key>``, or ``?api_key=<key>``.
    """
    core: Bolo = ws.app.state.core
    offered = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    await ws.accept(subprotocol="realtime" if "realtime" in offered else None)

    async def send(event: dict[str, Any]) -> None:
        await ws.send_text(json.dumps(event, ensure_ascii=False))

    try:
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
        await realtime.serve(
            core, receive, send, model=params.get("model"), transcription=params.get("intent") == "transcription"
        )
    except (WebSocketDisconnect, RuntimeError):
        pass
    with contextlib.suppress(Exception):
        await ws.close()
