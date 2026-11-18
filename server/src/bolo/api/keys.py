"""``/api/v1/keys``: API keys with scopes, rate limits and quotas (bolo.keys). Every route needs the ``admin``
scope, except that a server without any key yet lets a loopback client create the first (admin) key."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from bolo.api.common import get_core
from bolo.errors import InvalidRequest
from bolo.keys import PERIODS, SCOPES

router = APIRouter(prefix="/api/v1/keys", tags=["native"])

LOOPBACK = ("127.0.0.1", "::1", "localhost", "testclient")


class KeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="A name to tell keys apart (1 to 80 characters).")
    scopes: list[str] | None = Field(
        default=None, description=f"Any of {', '.join(SCOPES)}. Default: admin. Every key may use read routes."
    )
    rate_limit: float | None = Field(default=None, description="Requests per minute (token bucket); null: no limit.")
    burst: int | None = Field(default=None, description="Requests allowed at once; default: the per-minute limit.")
    quota: int | None = Field(default=None, description="Requests per quota_period; null: no quota.")
    quota_period: str = Field(default="day", description=f"One of {', '.join(PERIODS)} (UTC).")
    expires_in_days: float | None = Field(default=None, description="Days until the key stops working; null: never.")


class KeyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    scopes: list[str] | None = None
    rate_limit: float | None = None
    burst: int | None = None
    quota: int | None = None
    quota_period: str | None = None
    reset_usage: bool = Field(default=False, description="Start the current quota period's count again from zero.")


def _info(request: Request) -> dict[str, Any]:
    core = get_core(request)
    principal = getattr(request.state, "principal", None)
    return {
        "auth": core.keys.enabled,
        "root_key": bool(core.keys.root_key),
        "scopes": SCOPES,
        "periods": list(PERIODS),
        "you": getattr(principal, "id", None),
    }


@router.get("", summary="API keys and their usage (the keys themselves are never returned)")
async def list_keys(request: Request) -> JSONResponse:
    core = get_core(request)
    return JSONResponse({**_info(request), "data": [k.public() for k in core.keys.all()]})


@router.post("", status_code=201, summary="Create an API key (its value is returned once)")
async def create_key(request: Request, body: KeyRequest) -> JSONResponse:
    core = get_core(request)
    if not core.keys.enabled:
        # the first key turns authentication on: it must be able to manage keys, and only this machine may make it
        host = request.client.host if request.client else ""
        if host not in LOOPBACK:
            raise InvalidRequest("the first API key can only be created from this machine (or with `bolo keys create`)")
        if body.scopes is not None and "admin" not in [s.lower() for s in body.scopes] and "*" not in body.scopes:
            raise InvalidRequest(
                "the first API key turns authentication on, so it needs the admin scope; create scoped keys with it",
                param="scopes",
            )
    key, secret = core.keys.create(
        body.name,
        body.scopes,
        rate_limit=body.rate_limit,
        burst=body.burst,
        quota=body.quota,
        quota_period=body.quota_period,
        expires_in_days=body.expires_in_days,
    )
    core.log("info", f"API key {key.id} ({key.name}) created with scopes {', '.join(key.scopes)}")
    return JSONResponse({**key.public(), "key": secret}, status_code=201)


@router.get("/{key_id}", summary="One API key and its usage")
async def get_key(request: Request, key_id: str) -> JSONResponse:
    return JSONResponse(get_core(request).keys.get(key_id).public())


@router.patch("/{key_id}", summary="Change a key's name, scopes, rate limit or quota")
async def update_key(request: Request, key_id: str, body: KeyUpdate) -> JSONResponse:
    core = get_core(request)
    key = core.keys.update(key_id, body.model_dump(exclude_unset=True))
    return JSONResponse(key.public())


@router.delete("/{key_id}", summary="Revoke a key (?delete=true removes it and its counters)")
async def revoke_key(request: Request, key_id: str, delete: bool = False) -> Response:
    core = get_core(request)
    if delete and not core.keys.root_key and [k.id for k in core.keys.all()] == [key_id]:
        raise InvalidRequest(
            "deleting the last key would turn authentication off; revoke it, or delete it with `bolo keys delete`"
        )
    core.keys.revoke(key_id, delete=delete)
    core.log("info", f"API key {key_id} {'deleted' if delete else 'revoked'}")
    return Response(status_code=204)
