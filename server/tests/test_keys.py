"""API keys with scopes, rate limits and quotas (bolo.keys, /api/v1/keys, bolo keys)."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
import openai
import pytest

from bolo.config import Settings
from bolo.errors import AuthError
from bolo.keys import KeyStore, QuotaExceeded, RateLimited, ScopeError, period_key, scope_for
from conftest import LiveServer, wav_bytes


def test_scope_of_routes() -> None:
    assert scope_for("POST", "/v1/audio/transcriptions") == "stt"
    assert scope_for("GET", "/api/v1/stt/stream") == "stt"
    assert scope_for("POST", "/v1/audio/speech") == "tts"
    assert scope_for("GET", "/v1/realtime") == "realtime"
    assert scope_for("POST", "/v1/chat/completions") == "realtime"
    assert scope_for("GET", "/v1/models") == "read"
    assert scope_for("POST", "/api/v1/models/pull") == "manage"
    assert scope_for("GET", "/api/v1/keys") == "admin"
    assert scope_for("POST", "/api/v1/voices") == "tts"
    assert scope_for("GET", "/api/v1/voices") == "read"
    for path in ("/v1/audio/tags", "/v1/audio/emotion", "/v1/audio/denoise", "/v1/audio/speakers/identify"):
        assert scope_for("POST", path) == "stt"
    assert scope_for("POST", "/v1/audio/speakers") == "manage"  # enrolling


def test_store_scopes_bucket_and_quota(tmp_path: Path) -> None:
    store = KeyStore(tmp_path)
    assert not store.enabled
    assert store.authenticate(None).key is None  # open
    key, secret = store.create("caller", ["stt"], rate_limit=60, burst=2, quota=3, quota_period="day")
    assert store.enabled and secret.startswith("bolo_")
    stored = (tmp_path / "api-keys.json").read_text(encoding="utf-8")
    assert secret not in stored and key.hash in stored  # only the hash is kept
    with pytest.raises(AuthError):
        store.authenticate(None)
    with pytest.raises(AuthError):
        store.authenticate("bolo_wrong")
    who = store.authenticate(secret)
    assert who.key is not None and who.key.id == key.id
    with pytest.raises(ScopeError):
        store.admit(who, "tts")
    store.admit(who, "read")  # every key reads
    store.admit(who, "stt")
    with pytest.raises(RateLimited) as limited:  # burst of 2 spent
        store.admit(who, "stt")
    assert 0 < limited.value.retry_after_s <= 1.0
    key.tokens = 5.0  # refilled
    store.admit(who, "stt")
    with pytest.raises(QuotaExceeded):  # 3 per day
        store.admit(who, "stt")
    info = key.public()
    assert info["usage"]["period_requests"] == 3 and info["usage"]["quota_remaining"] == 0
    assert info["usage"]["by_scope"] == {"read": 1, "stt": 2} and info["usage"]["rejected"] == 2
    assert info["usage"]["period"] == period_key("day")
    store.update(key.id, {"reset_usage": True, "quota": None})
    key.tokens = 5.0
    store.admit(who, "stt")
    store.revoke(key.id)
    with pytest.raises(AuthError, match="revoked"):
        store.authenticate(secret)
    assert store.enabled  # a revoked key keeps authentication on
    store.revoke(key.id, delete=True)
    assert not store.enabled


def test_store_reads_changes_made_elsewhere(tmp_path: Path) -> None:
    server = KeyStore(tmp_path)
    cli = KeyStore(tmp_path)
    _, secret = cli.create("from the cli", "realtime,read")
    time.sleep(0.01)
    assert server.authenticate(secret).key is not None
    key = server.authenticate(secret).key
    assert key is not None and key.scopes == ["realtime", "read"]
    server.admit(server.authenticate(secret), "realtime")
    server.flush()
    assert json.loads((tmp_path / "api-keys.json").read_text(encoding="utf-8"))["keys"][0]["requests"] == 1


def test_root_key_still_works(tmp_path: Path) -> None:
    store = KeyStore(tmp_path, root_key="sekrit")
    assert store.enabled
    who = store.authenticate("sekrit")
    assert who.root and who.allows("admin")


def test_keys_over_http(settings: Settings) -> None:
    srv = LiveServer(settings)
    srv.start()
    try:
        base = srv.url
        # no key yet: open, and a loopback client may create the first key, which must be an admin key
        assert httpx.get(base + "/api/v1/info").json()["auth_required"] is False
        refused = httpx.post(base + "/api/v1/keys", json={"name": "x", "scopes": ["stt"]})
        assert refused.status_code == 400
        admin = httpx.post(base + "/api/v1/keys", json={"name": "ops"}).json()
        assert admin["scopes"] == ["admin"] and admin["key"].startswith("bolo_")
        assert httpx.get(base + "/api/v1/info").status_code == 401
        assert httpx.get(base + "/v1/models").status_code == 401
        h = {"Authorization": f"Bearer {admin['key']}"}
        stt = httpx.post(
            base + "/api/v1/keys", headers=h, json={"name": "app", "scopes": ["stt"], "rate_limit": 2, "burst": 2}
        ).json()
        listing = httpx.get(base + "/api/v1/keys", headers=h).json()
        assert [k["name"] for k in listing["data"]] == ["ops", "app"] and "key" not in listing["data"][1]
        assert listing["you"] == admin["id"]
        client = openai.OpenAI(base_url=base + "/v1", api_key=stt["key"], max_retries=0)
        assert client.audio.transcriptions.create(model="fake-stt", file=io.BytesIO(wav_bytes(1.0))).text
        with pytest.raises(openai.PermissionDeniedError) as denied:
            client.audio.speech.create(model="fake-tts", voice="a", input="hi")
        assert denied.value.body["code"] == "insufficient_scope"  # type: ignore[index]
        assert httpx.get(base + "/v1/models", headers={"Authorization": f"Bearer {stt['key']}"}).status_code == 200
        r = httpx.get(base + "/v1/models", headers={"Authorization": f"Bearer {stt['key']}"})
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limit_exceeded"
        assert int(r.headers["retry-after"]) >= 1
        assert httpx.get(base + "/api/v1/keys", headers={"Authorization": f"Bearer {stt['key']}"}).status_code == 403
        one = httpx.get(base + f"/api/v1/keys/{stt['id']}", headers=h).json()
        assert one["usage"]["requests"] == 2 and one["usage"]["rejected"] >= 1
        changed = httpx.patch(base + f"/api/v1/keys/{stt['id']}", headers=h, json={"rate_limit": None, "quota": 1})
        assert changed.json()["rate_limit"] is None and changed.json()["quota"] == 1
        r = httpx.get(base + "/v1/models", headers={"Authorization": f"Bearer {stt['key']}"})
        assert r.status_code == 429 and r.json()["error"]["code"] == "insufficient_quota"
        assert httpx.delete(base + f"/api/v1/keys/{stt['id']}", headers=h).status_code == 204
        assert httpx.get(base + "/v1/models", headers={"Authorization": f"Bearer {stt['key']}"}).status_code == 401
    finally:
        srv.stop()


def test_scoped_key_opens_a_stream_and_realtime(settings: Settings) -> None:
    from websockets.sync.client import connect

    store = KeyStore(settings.data_dir)
    _, stt = store.create("stt", ["stt"])
    srv = LiveServer(settings)
    srv.start()
    try:
        with connect(f"{srv.ws_url}/api/v1/stt/stream?model=fake-stt-streaming&api_key={stt}") as ws:
            assert json.loads(ws.recv())["type"] in ("ready", "session.created", "started")
        with connect(f"{srv.ws_url}/v1/realtime", subprotocols=["realtime", f"openai-insecure-api-key.{stt}"]) as ws:  # type: ignore[list-item]
            event = json.loads(ws.recv())
            assert event["type"] == "error" and event["error"]["code"] == "insufficient_scope"
    finally:
        srv.stop()


def test_cli_keys(settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from bolo.cli import main

    data = str(settings.data_dir)
    assert main(["keys", "create", "batch", "--data-dir", data, "--scopes", "stt,tts", "--quota", "100"]) == 0
    secret = capsys.readouterr().out.strip().splitlines()[-1]
    assert secret.startswith("bolo_")
    assert main(["keys", "--data-dir", data]) == 0
    out = capsys.readouterr().out
    assert "batch" in out and "stt,tts" in out and "100/day" in out
    key_id = KeyStore(settings.data_dir).all()[0].id
    assert main(["keys", "update", key_id, "--data-dir", data, "--rate-limit", "30", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rate_limit"] == 30
    assert main(["keys", "revoke", key_id, "--data-dir", data]) == 0
    assert "revoked" in capsys.readouterr().out
