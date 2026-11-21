"""Application factory."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bolo import __version__
from bolo.api import analysis, chat, native, openai, realtime, speaches, static, synthesis, telephony, voices, ws
from bolo.api import keys as keys_api
from bolo.api import pipelines as pipelines_api
from bolo.api import speakers as speakers_api
from bolo.api import wake as wake_api
from bolo.api import wordlists as wordlists_api
from bolo.api.common import check_auth, error_body, error_response
from bolo.config import Settings
from bolo.core import Bolo
from bolo.errors import BoloError

log = logging.getLogger("bolo.api")

DESCRIPTION = """
bolo serves speech-to-text, text-to-speech and voice-activity detection.

* **openai** - drop-in `/v1/audio/*` and `/v1/models`
* **speaches** - model-management routes compatible with speaches
* **native** - `/api/v1/*`: catalog, pulls, cloud providers, stats, request traces, events, benchmark
* **voices** - `/api/v1/voices`: saved voices for the models that clone voices (consent required)
* **pipelines** - `/api/v1/pipelines` and `/v1/pipelines/{id}/run`: chains of steps, run at once or as jobs
* WebSockets (not shown by OpenAPI): `/api/v1/stt/stream`, `/api/v1/tts/stream` - see docs/api.md
"""


async def require_auth(request: Request) -> None:
    check_auth(request.app.state.core, request)


def create_app(settings: Settings | None = None, core: Bolo | None = None) -> FastAPI:
    core = core or Bolo(settings)
    settings = core.settings

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await core.start()
        wyoming_server: Any = None
        if settings.wyoming:
            try:
                from bolo.wyoming_server import start_wyoming

                wyoming_server = await start_wyoming(core, settings.wyoming_host, settings.wyoming_port)
                core.log("info", f"Wyoming protocol listening on tcp://{settings.wyoming_host}:{settings.wyoming_port}")
            except Exception as exc:
                core.log("error", f"could not start the Wyoming server: {exc}")
        if settings.sip:
            try:
                from bolo.telephony.sip import start_sip

                core.sip = await start_sip(core.telephony)
                core.log("info", f"SIP listening on udp://{settings.sip_host}:{core.sip.port}")
            except ImportError:
                core.log("error", "the SIP endpoint needs sipmessage: pip install 'bolo-speech[sip]'")
            except Exception as exc:
                core.log("error", f"could not start the SIP endpoint: {exc}")
        try:
            yield
        finally:
            if core.sip is not None:
                with contextlib.suppress(Exception):
                    await core.sip.stop()
                core.sip = None
            if wyoming_server is not None:
                with contextlib.suppress(Exception):
                    await wyoming_server.stop()
            await core.stop()

    app = FastAPI(
        title="bolo",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    )
    app.state.core = core

    from bolo.api.console import console_origin_regex

    # with the console on a port of its own, the API answers its cross-origin requests
    console_origins = None if "*" in settings.cors_origins else console_origin_regex(settings)
    if settings.cors_origins or console_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_origin_regex=console_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "x-request-id",
                "x-sample-rate",
                "openai-processing-ms",
                "x-bolo-model",
                "x-bolo-fallback",
                "x-bolo-estimated-cost",
                "x-bolo-separation",
                "x-bolo-ignored-options",
                "x-bolo-word-aligner",
                "x-bolo-denoise",
                "x-bolo-route",
                "x-bolo-voice",
                "x-bolo-cloned",
                "x-bolo-blend",
                "x-bolo-language",
                "x-bolo-pipeline-run",
            ],
        )

    @app.exception_handler(BoloError)
    async def _bolo_error(_request: Request, exc: BoloError) -> JSONResponse:
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(p) for p in first.get("loc", []) if p not in ("body", "query")]
        message = (
            f"{'.'.join(loc)}: {first.get('msg', 'invalid request')}"
            if loc
            else str(first.get("msg", "invalid request"))
        )
        return JSONResponse(
            error_body(message, "invalid_request_error", "invalid_request", loc[0] if loc else None), status_code=400
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse(error_body(str(exc.detail), kind, None), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(
            error_body(f"{type(exc).__name__}: {exc}", "server_error", "internal_error"), status_code=500
        )

    protected = [Depends(require_auth)]
    app.include_router(openai.router, dependencies=protected)
    app.include_router(speaches.router, dependencies=protected)
    app.include_router(native.router, dependencies=protected)
    app.include_router(wordlists_api.router, dependencies=protected)
    app.include_router(voices.router, dependencies=protected)
    app.include_router(analysis.router, dependencies=protected)
    app.include_router(synthesis.router, dependencies=protected)
    app.include_router(speakers_api.router, dependencies=protected)
    app.include_router(keys_api.router, dependencies=protected)
    app.include_router(wake_api.router, dependencies=protected)
    app.include_router(pipelines_api.router, dependencies=protected)
    app.include_router(native.root_router)
    app.include_router(chat.router, dependencies=protected)
    app.include_router(ws.router)
    app.include_router(wake_api.ws_router)
    app.include_router(realtime.router)
    app.include_router(telephony.router, dependencies=protected)
    app.include_router(telephony.public_router)
    if settings.console_mode() == "same":  # "separate": bolo.api.console serves it on console_port
        app.include_router(static.router)
    return app
