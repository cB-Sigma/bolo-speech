"""The web console on a port of its own (``console_port``): both apps started, as ``bolo serve`` runs them."""

from __future__ import annotations

import asyncio
import re
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from bolo.api import create_app
from bolo.api.console import api_pointer, console_origin_regex, create_console_app, serve_both, servers
from bolo.api.static import has_bundle
from bolo.cli import build_parser
from bolo.config import Settings, load_settings
from bolo.core import Bolo


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class Pair:
    """The API's server and the console's, run by :func:`serve_both` on a thread of their own."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.core = Bolo(settings)
        api = uvicorn.Config(create_app(core=self.core), host="127.0.0.1", port=settings.port, log_level="warning")
        console = uvicorn.Config(
            create_console_app(self.core), host="127.0.0.1", port=int(settings.console_port or 0), log_level="warning"
        )
        self.api, self.console = servers(api, console)
        self.thread = threading.Thread(target=lambda: asyncio.run(serve_both(self.api, self.console)), daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 10
        while not (self.api.started and self.console.started):
            if time.time() > deadline:
                raise RuntimeError("servers did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        self.api.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def pair(settings: Settings) -> Iterator[Pair]:
    settings.port, settings.console_port = free_port(), free_port()
    p = Pair(settings)
    p.start()
    try:
        yield p
    finally:
        p.stop()


def test_console_modes() -> None:
    assert Settings().console_mode() == "same"
    assert Settings(console_port=8000).console_mode() == "same"
    assert Settings(console_port=8001).console_mode() == "separate"
    assert Settings(console_port=8000, console_host="10.0.0.2").console_mode() == "separate"
    assert Settings(console_port=0).console_mode() == "off"
    assert Settings(ui=False, console_port=8001).console_mode() == "off"


def test_settings_env_and_flags(tmp_path: Path) -> None:
    env = {
        "BOLO_CONSOLE_PORT": "8001",
        "BOLO_CONSOLE_API_URL": "https://api.example.test",
        "BOLO_DATA_DIR": str(tmp_path),
    }
    s = load_settings(env=env)
    assert s.console_port == 8001 and s.console_api_url == "https://api.example.test"
    assert api_pointer(s) == "https://api.example.test/" and api_pointer(Settings(port=9000)) == ":9000"
    with pytest.raises(ValueError, match="console_port"):
        load_settings(overrides={"data_dir": tmp_path, "console_port": 70000}, env={})
    args = build_parser().parse_args(["serve", "--console-port", "8001", "--console-host", "0.0.0.0"])
    assert args.console_port == 8001 and args.console_host == "0.0.0.0"
    assert build_parser().parse_args(["serve", "--no-console"]).ui is False
    assert build_parser().parse_args(["serve", "--console-port", "0"]).console_port == 0


def test_cors_origins_follow_the_console_address() -> None:
    assert console_origin_regex(Settings()) is None
    loopback = console_origin_regex(Settings(console_port=8001))
    assert loopback is not None
    assert re.match(loopback, "http://127.0.0.1:8001") and re.match(loopback, "http://localhost:8001")
    assert not re.match(loopback, "http://evil.example:8001") and not re.match(loopback, "http://127.0.0.1:8002")
    anywhere = console_origin_regex(Settings(host="0.0.0.0", console_port=8001))
    assert anywhere is not None
    assert re.match(anywhere, "https://box.lan:8001") and re.match(anywhere, "http://[fd00::1]:8001")
    assert not re.match(anywhere, "http://box.lan:8001/x")
    one = console_origin_regex(Settings(host="0.0.0.0", console_host="10.0.0.2", console_port=8001))
    assert one is not None and re.match(one, "http://10.0.0.2:8001") and not re.match(one, "http://10.0.0.3:8001")


def test_api_and_console_on_their_own_ports(pair: Pair) -> None:
    api = f"http://127.0.0.1:{pair.settings.port}"
    console = f"http://127.0.0.1:{pair.settings.console_port}"
    # the API port: the API, no console
    assert httpx.get(api + "/v1/models").status_code == 200
    assert httpx.get(api + "/").status_code == 404
    assert httpx.get(api + "/assets/whatever.js").status_code == 404
    # the console port: the console, told where the API is; no API
    page = httpx.get(console + "/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    if has_bundle():
        assert f'<meta name="bolo-api" content=":{pair.settings.port}" />' in page.text
    refused = httpx.get(console + "/api/v1/info")
    assert refused.status_code == 404 and f":{pair.settings.port}" in refused.json()["error"]["message"]
    assert httpx.get(console + "/docs").status_code == 404
    # the console's requests to the API are allowed cross-origin; another port's are not
    preflight = httpx.options(
        api + "/api/v1/info",
        headers={
            "Origin": console,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert preflight.status_code == 200 and preflight.headers["access-control-allow-origin"] == console
    r = httpx.get(api + "/api/v1/info", headers={"Origin": console})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == console
    other = httpx.get(api + "/api/v1/info", headers={"Origin": f"http://127.0.0.1:{pair.settings.port + 1}"})
    assert "access-control-allow-origin" not in other.headers or pair.settings.port + 1 == pair.settings.console_port


def test_stopping_the_api_stops_the_console(settings: Settings) -> None:
    settings.port, settings.console_port = free_port(), free_port()
    p = Pair(settings)
    p.start()
    p.api.handle_exit(2, None)  # what SIGINT does
    p.thread.join(timeout=10)
    assert not p.thread.is_alive() and p.console.should_exit


def test_one_port_and_no_console(settings: Settings, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as client:  # the default: both on one port, no meta
        page = client.get("/")
        assert page.status_code == 200 and 'name="bolo-api"' not in page.text
        assert "access-control-allow-origin" not in client.get("/v1/models", headers={"Origin": "http://x:1"}).headers
    off = Settings(**{**settings.__dict__, "console_port": 0, "data_dir": tmp_path / "off"})
    with TestClient(create_app(off)) as client:
        assert client.get("/").status_code == 404 and client.get("/v1/models").status_code == 200
