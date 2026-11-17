"""The web console on a port of its own (``console_port``).

By default one port serves both the API and the console. With ``console_port`` set, the API keeps ``port`` and a
second, console-only app listens on ``console_port``: the API port can then be exposed on its own (a firewall rule,
a proxy) while the console stays private, or the other way round. The console page is told where the API is
(``<meta name="bolo-api">``), and the API answers the console's cross-origin requests (CORS).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Iterator
from types import FrameType
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI

from bolo import __version__
from bolo.config import Settings

if TYPE_CHECKING:
    from bolo.core import Bolo

_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_ANY = ("0.0.0.0", "::", "")


def console_host(settings: Settings) -> str:
    return settings.console_host or settings.host


def api_pointer(settings: Settings) -> str:
    """What the console page is told about the API: ``console_api_url``, else ``:<port>`` (the page's own host at
    the API's port)."""
    if settings.console_api_url:
        return settings.console_api_url.rstrip("/") + "/"
    return f":{settings.port}"


def console_origin_regex(settings: Settings) -> str | None:
    """Origins the API accepts cross-origin requests from because the console is served from them: any host name
    at ``console_port`` (only loopback names when the console listens on loopback). None unless the console has a
    port of its own."""
    if settings.console_mode() != "separate":
        return None
    host = console_host(settings)
    if host in _LOOPBACK:
        names = r"(localhost|127\.0\.0\.1|\[::1\])"
    elif host in _ANY:
        names = r"([^/:]+|\[[0-9a-fA-F:.]+\])"
    else:
        names = re.escape(f"[{host}]" if ":" in host else host)
    return rf"^https?://{names}:{int(settings.console_port or 0)}$"


def create_console_app(core: Bolo) -> FastAPI:
    """A console-only app: the static bundle, with the page told where the API is."""
    from bolo.api import static

    app = FastAPI(title="bolo console", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.core = core
    app.state.console_api = api_pointer(core.settings)
    app.include_router(static.router)
    return app


class _ConsoleServer(uvicorn.Server):
    """The console's server: signals are left to the API's server, which stops this one with it."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class _ApiServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, follower: uvicorn.Server) -> None:
        super().__init__(config)
        self.follower = follower

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self.follower.should_exit = True
        if self.should_exit:
            self.follower.force_exit = True
        super().handle_exit(sig, frame)


def servers(api: uvicorn.Config, console: uvicorn.Config) -> tuple[uvicorn.Server, uvicorn.Server]:
    """The API's server and the console's, linked: a signal that stops the first stops the second."""
    follower = _ConsoleServer(console)
    return _ApiServer(api, follower), follower


async def serve_both(api: uvicorn.Server, console: uvicorn.Server) -> None:
    """Run both servers until the API's stops; the console's then stops too."""
    task = asyncio.create_task(console.serve(), name="bolo-console")
    try:
        await api.serve()
    finally:
        console.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await task


def uvicorn_kwargs(settings: Settings) -> dict[str, Any]:
    return {"log_level": settings.log_level.lower(), "ws_max_size": 16 * 1024 * 1024}
