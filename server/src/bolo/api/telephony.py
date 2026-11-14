"""Phone calls (see :mod:`bolo.telephony` and ``docs/telephony.md``).

Two groups of routes:

* ``/telephony/{provider}/...``: what the telephony providers call. They are **not** behind the server's API
  key (a provider cannot send it); every request is checked against the provider's signature or credentials
  instead, and the routes refuse to work until that is possible (or ``telephony.validate_signatures`` is
  switched off on purpose).
* ``/api/v1/telephony/...``: the operator's API (agents, numbers, calls, placing a call, the simulator),
  behind the API key like the rest of the native API.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from bolo.api.common import get_core
from bolo.core import Bolo
from bolo.errors import BoloError, InvalidRequest, ModelNotFound
from bolo.telephony.adapters import Inbound
from bolo.telephony.manager import COMPLIANCE_NOTICE, TelephonyManager

log = logging.getLogger("bolo.telephony")

#: called by providers: no API key, signatures instead
public_router = APIRouter()
#: the operator's API
router = APIRouter(prefix="/api/v1/telephony")

HOOKS = ("voice", "after", "status", "amd")


def _request_base(headers: Any, scheme: str) -> str:
    proto = (headers.get("x-forwarded-proto") or scheme).split(",")[0].strip()
    proto = {"ws": "http", "wss": "https"}.get(proto, proto)
    host = (headers.get("x-forwarded-host") or headers.get("host") or "127.0.0.1").split(",")[0].strip()
    return f"{proto}://{host}"


async def _webhook(request: Request, provider: str, hook: str, token: str | None) -> Response:
    core = get_core(request)
    manager: TelephonyManager = core.telephony
    if hook not in HOOKS:
        raise ModelNotFound(f"unknown telephony webhook '{hook}'")
    adapter = manager.adapter(provider)  # refuses when the endpoints may not be exposed
    manager.set_request_base(_request_base(request.headers, request.url.scheme))
    body = await request.body()
    form: dict[str, str] = {}
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if request.method == "POST" and content_type in ("application/x-www-form-urlencoded", "multipart/form-data"):
        form = {str(k): str(v) for k, v in (await request.form()).items() if isinstance(v, str)}
    path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    inbound = Inbound(
        request.method, manager.url(path), {k.lower(): v for k, v in request.headers.items()},
        dict(request.query_params), form, body,
    )  # fmt: skip
    if manager.config.validate_signatures and not adapter.verify(inbound):
        what = "signature" if adapter.protection == "signature" else "credentials"
        core.log(
            "warning",
            f"telephony: refused a {adapter.label} {hook} webhook with a missing or wrong {what}. If the request is "
            f"genuine, check telephony.public_base_url ({manager.config.public_base_url}) against the URL in the "
            "provider's dashboard: the signature covers the exact URL.",
        )
        headers = {"WWW-Authenticate": 'Basic realm="bolo telephony"'} if adapter.protection == "basic" else None
        return JSONResponse(
            {"error": {"message": f"invalid {adapter.label} {what}", "type": "invalid_request_error", "code": "invalid_signature"}},
            status_code=401 if adapter.protection == "basic" else 403,
            headers=headers,
        )  # fmt: skip
    reply = await adapter.webhook(hook, inbound, token)
    return Response(reply.body, status_code=reply.status, media_type=reply.media_type)


@public_router.api_route("/telephony/{provider}/{hook}", methods=["GET", "POST"], include_in_schema=False)
async def provider_webhook(request: Request, provider: str, hook: str) -> Response:
    return await _webhook(request, provider, hook, None)


@public_router.api_route("/telephony/{provider}/{hook}/{token}", methods=["GET", "POST"], include_in_schema=False)
async def provider_webhook_token(request: Request, provider: str, hook: str, token: str) -> Response:
    return await _webhook(request, provider, hook, token)


async def _stream(ws: WebSocket, provider: str, token: str | None) -> None:
    core: Bolo = ws.app.state.core
    manager: TelephonyManager = core.telephony
    manager.set_request_base(_request_base(ws.headers, ws.url.scheme))

    async def send(text: str) -> None:
        await ws.send_text(text)

    async def close() -> None:
        with contextlib.suppress(Exception):
            await ws.close()

    try:
        path = ws.url.path + (f"?{ws.url.query}" if ws.url.query else "")
        headers = {k.lower(): v for k, v in ws.headers.items()}
        session = manager.open_stream(provider, token, headers, manager.ws_url(path), send, close)
    except BoloError as exc:
        core.log("warning", f"telephony: refused a {provider} media stream: {exc.message}")
        await ws.close(code=1008)  # before accept: the handshake is answered with HTTP 403
        return
    await ws.accept()
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text:
                await session.handle_message(text)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await session.finalize("caller_hangup")
        await close()


@public_router.websocket("/telephony/{provider}/stream")
async def provider_stream(ws: WebSocket, provider: str) -> None:
    await _stream(ws, provider, None)


@public_router.websocket("/telephony/{provider}/stream/{token}")
async def provider_stream_token(ws: WebSocket, provider: str, token: str) -> None:
    await _stream(ws, provider, token)


# -- the operator's API -------------------------------------------------------------------------------------
async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json() if await request.body() else {}
    except ValueError as exc:
        raise InvalidRequest("the request body must be JSON") from exc
    if not isinstance(body, dict):
        raise InvalidRequest("the request body must be a JSON object")
    return body


@router.get("", tags=["native"], summary="Telephony: providers, webhook URLs, agents, numbers (no secrets)")
async def overview(request: Request) -> JSONResponse:
    manager: TelephonyManager = get_core(request).telephony
    manager.set_request_base(_request_base(request.headers, request.url.scheme))
    return JSONResponse(manager.overview())


@router.get("/agents", tags=["native"], summary="Voice agents that answer and place calls")
async def list_agents(request: Request) -> JSONResponse:
    config = get_core(request).telephony.config
    return JSONResponse({"agents": [a.to_dict() for a in config.agents.values()]})


@router.get("/agents/{name}", tags=["native"], summary="One voice agent")
async def get_agent(request: Request, name: str) -> JSONResponse:
    agent = get_core(request).telephony.config.agents.get(name.strip().lower())
    if agent is None:
        raise ModelNotFound(f"unknown agent '{name}'")
    return JSONResponse(agent.to_dict())


@router.put("/agents/{name}", tags=["native"], summary="Add or change a voice agent")
async def put_agent(request: Request, name: str) -> JSONResponse:
    """Stored in ``<data_dir>/telephony.json``. Agents from the configuration file cannot be changed here.
    Refused on a non-loopback bind without an API key."""
    core = get_core(request)
    body = await _json(request)
    for key in ("name", "source", "editable"):
        body.pop(key, None)
    agent = core.telephony.config.put_agent(name, body)
    core.events.publish("telephony", {"event": "agent_saved", "name": agent.name})
    return JSONResponse(agent.to_dict())


@router.delete("/agents/{name}", status_code=204, tags=["native"], summary="Remove a stored voice agent")
async def delete_agent(request: Request, name: str) -> Response:
    core = get_core(request)
    core.telephony.config.delete_agent(name)
    core.events.publish("telephony", {"event": "agent_removed", "name": name.strip().lower()})
    return Response(status_code=204)


@router.get("/numbers", tags=["native"], summary="Which agent answers which number")
async def list_numbers(request: Request) -> JSONResponse:
    config = get_core(request).telephony.config
    return JSONResponse({"numbers": config.numbers_public(), "default_agent": config.default_agent})


@router.put("/numbers/{number}", tags=["native"], summary="Map a called number to an agent")
async def put_number(request: Request, number: str) -> JSONResponse:
    core = get_core(request)
    body = await _json(request)
    if not isinstance(body.get("agent"), str):
        raise InvalidRequest('the body must name the agent: {"agent": "<name>"}', param="agent")
    core.telephony.config.put_number(number, body["agent"])
    core.events.publish("telephony", {"event": "number_saved", "number": number})
    return JSONResponse({"numbers": core.telephony.config.numbers_public()})


@router.delete("/numbers/{number}", status_code=204, tags=["native"], summary="Remove a number's mapping")
async def delete_number(request: Request, number: str) -> Response:
    core = get_core(request)
    core.telephony.config.delete_number(number)
    core.events.publish("telephony", {"event": "number_removed", "number": number})
    return Response(status_code=204)


@router.get("/calls", tags=["native"], summary="Recent calls, newest first")
async def list_calls(
    request: Request, limit: int = 50, provider: str | None = None, agent: str | None = None, status: str | None = None
) -> JSONResponse:
    store = get_core(request).telephony.store
    calls = store.recent(max(1, min(limit, 500)), provider or None, agent or None, status or None)
    return JSONResponse({"calls": [c.summary_dict() for c in calls]})


@router.get("/calls/{call_id}", tags=["native"], summary="One call: timeline, turns with latency, transcript")
async def get_call(request: Request, call_id: str) -> JSONResponse:
    record = get_core(request).telephony.store.get(call_id)
    if record is None:
        raise ModelNotFound(f"unknown call '{call_id}' (calls are kept in memory and rotate)")
    return JSONResponse(record.to_dict())


@router.post("/calls", status_code=202, tags=["native"], summary="Place an outbound call and attach an agent")
async def place_call(request: Request) -> JSONResponse:
    """Body: ``{"provider": "twilio", "to": "+14155550100", "agent": "front-desk", "from": "+1...",
    "metadata": {}}``. ``from`` defaults to the provider's ``from_number``. The answer is the call record;
    poll ``GET /api/v1/telephony/calls/{id}`` for what happened.

    Outbound calling is regulated (consent, calling hours, do-not-call registers such as TRAI's DND in India,
    disclosure that a machine is speaking). Compliance is the operator's responsibility."""
    core = get_core(request)
    body = await _json(request)
    for key in ("provider", "to"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            raise InvalidRequest(f"'{key}' is required", param=key)
    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise InvalidRequest("'metadata' must be an object", param="metadata")
    record = await core.telephony.place_call(
        body["provider"], body["to"], body.get("agent"), body.get("from") or body.get("from_number"), metadata
    )
    return JSONResponse({**record.to_dict(), "compliance_notice": COMPLIANCE_NOTICE}, status_code=202)


@router.post("/calls/{call_id}/hangup", tags=["native"], summary="End a call in progress")
async def hangup_call(request: Request, call_id: str) -> JSONResponse:
    record = await get_core(request).telephony.hangup(call_id)
    return JSONResponse(record.summary_dict())


@router.get("/scenarios", tags=["native"], summary="Built-in call-simulator scenarios")
async def scenarios(request: Request) -> JSONResponse:
    from bolo.telephony.simulator import SCENARIOS

    return JSONResponse(
        {"scenarios": [{"name": name, "description": s.get("description", "")} for name, s in SCENARIOS.items()]}
    )


@router.post("/simulate", tags=["native"], summary="Simulate a call to an agent through a fake provider")
async def simulate(request: Request) -> JSONResponse:
    """Body: ``{"agent": "front-desk", "provider": "twilio", "scenario": "basic" | {...}, "noise_db": 20}``.
    A scripted caller speaks to the agent through a fake provider connection inside this process (no network,
    no telephony account) with the models this server is configured with. The answer is the simulator's report:
    what the caller heard, latency as the caller experiences it, and whether the expectations were met. The
    call also appears among the recent calls."""
    from bolo.telephony.simulator import run_scenario

    core = get_core(request)
    body = await _json(request)
    report = await run_scenario(
        core, body.get("scenario") or "basic", agent=body.get("agent"), provider=str(body.get("provider") or "twilio"),
        noise_db=body.get("noise_db"),
    )  # fmt: skip
    return JSONResponse(report)
