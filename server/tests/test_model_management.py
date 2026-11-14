"""Model management like Ollama's (``bolo show``, ``rm``, ``cp``, ``list``), presets, and the server's log file
(``log_file``, ``bolo logs``, ``bolo doctor``)."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest

from bolo import cli, logs
from bolo.catalog import ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from conftest import LiveServer


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BOLO_AUTO_PULL", "0")
    monkeypatch.setenv("BOLO_URL", "http://127.0.0.1:9")  # no server: the commands act on the data directory
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("BOLO_LOG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def install_local(core: Bolo, model_id: str = "local-voice") -> ModelSpec:
    """A non-built-in model with two files on disk (one a licence) and a manifest, as `bolo pull` leaves it."""
    core.settings.probe_models = False  # the probe process has no fake engine
    spec = ModelSpec.from_dict({
        "id": model_id, "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Fake", "languages": ["en"],
        "license": "Apache-2.0", "license_url": "https://example.org/licence", "source": {"repo": "x/y"},
        "voices": [{"id": "alpha"}],
    })  # fmt: skip
    core.catalog.add_custom(spec)
    d = core.store.model_dir(spec.id)
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"\0" * 2048)
    (d / "LICENSE").write_text("Apache License, Version 2.0\n")
    files = [{"path": "model.onnx", "size": 2048}, {"path": "LICENSE", "size": 28}]
    manifest = {"id": spec.id, "files": files, "size_bytes": 2076, "installed_at": time.time() - 3 * 86400}
    (d / "manifest.json").write_text(json.dumps({**manifest, "complete": True}))
    return spec


# -- presets (bolo cp) ---------------------------------------------------------------------------------------------
async def test_a_preset_uses_its_models_files_and_has_its_own_settings(settings: Settings) -> None:
    core = Bolo(settings)
    await core.start()
    try:
        base = install_local(core)
        core.store.set_model_settings(base.id, {"keep_alive": "1m"})
        preset = core.copy_model(base.id, "fast-voice")
        assert (
            preset.preset_of == base.id
            and preset.custom
            and core.store.model_dir("fast-voice") == core.store.model_dir(base.id)
        )
        assert core.manager.is_installed(preset) and core.store.model_settings("fast-voice") == {"keep_alive": "1m"}
        core.store.set_model_settings("fast-voice", {"num_parallel": 2})
        assert core.store.model_settings(base.id) == {"keep_alive": "1m"}  # the source keeps its own
        entry = await core.manager.load("fast-voice")
        assert entry.queue.workers == 2 and (await core.manager.load(base.id)).queue.workers == 1
        assert core.show_api(base)["presets"] == ["fast-voice"]
        # a preset of a preset points at the model with the files; built-in models can be copied too
        assert core.copy_model("fast-voice", "faster-voice").preset_of == base.id
        assert core.copy_model("fake-tts", "my-fake").builtin
        from bolo.errors import InvalidRequest

        with pytest.raises(InvalidRequest, match="already exists"):
            core.copy_model(base.id, "fast-voice")
        with pytest.raises(InvalidRequest):
            core.copy_model(base.id, "Not A Valid Id")
    finally:
        await core.stop()
    # kept in custom.yaml: a new server knows the presets and where their files are
    again = Bolo(settings)
    spec = again.manager.resolve("fast-voice")
    assert spec.preset_of == "local-voice" and again.manager.is_installed(spec)
    assert again.manager.resolve("my-fake").builtin
    # deleting a preset leaves the files; deleting the model removes them
    assert await again.remove("fast-voice") and again.catalog.resolve("fast-voice") is None
    assert again.store.is_installed("local-voice") and again.store.model_settings("fast-voice") == {}
    assert await again.remove("local-voice") and not again.store.is_installed("local-voice")


def test_copy_show_and_delete_routes(settings: Settings) -> None:
    srv = LiveServer(settings)
    install_local(srv.core)
    srv.start()
    try:
        url = srv.url + "/api/v1/models"
        r = httpx.post(url + "/copy", json={"source": "local-voice", "destination": "slow-voice"})
        assert r.status_code == 201 and r.json()["preset_of"] == "local-voice" and r.json()["installed"]
        assert httpx.post(url + "/copy", json={"source": "local-voice", "destination": "slow-voice"}).status_code == 400
        assert httpx.post(url + "/copy", json={"source": "nope", "destination": "x"}).status_code == 404
        r = httpx.put(url + "/slow-voice/settings", json={"kv_cache_type": "Q8_0", "flash_attention": "on",
                                                          "num_parallel": 2, "max_queue": 5, "context_size": 2048})  # fmt: skip
        assert r.status_code == 200 and r.json()["settings"]["kv_cache_type"] == "q8_0"
        assert httpx.put(url + "/slow-voice/settings", json={"kv_cache_type": "q3"}).status_code == 400
        assert httpx.put(url + "/slow-voice/settings", json={"num_parallel": 0}).status_code == 400
        doc = httpx.get(url + "/local-voice/show").json()
        assert [f["path"] for f in doc["files"]] == ["model.onnx", "LICENSE"] and doc["presets"] == ["slow-voice"]
        assert doc["modified_at"].endswith("Z") and doc["effective"]["num_parallel"] == 1 and "license_text" not in doc
        assert "Apache License" in httpx.get(url + "/local-voice/show?license=true").json()["license_text"]
        assert httpx.get(url + "/slow-voice/show").json()["effective"]["max_queue"] == 5
        assert httpx.delete(url + "/slow-voice").status_code == 204
        assert srv.core.store.is_installed("local-voice")
        assert httpx.delete(url + "/fake-tts").status_code == 400  # built in
    finally:
        srv.stop()


# -- the command line ----------------------------------------------------------------------------------------------
def test_cli_cp_list_show_rm(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    core = Bolo(load_settings())
    install_local(core)
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["ID", "TASK", "ENGINE", "SIZE", "MODIFIED", "LICENCE"]
    assert "local-voice" in out and "3 days ago" in out and "2 KB" in out
    assert cli.main(["cp", "local-voice", "my-voice"]) == 0
    assert "copied local-voice to my-voice" in capsys.readouterr().out
    assert cli.main(["models", "config", "my-voice", "--set", "num_parallel=3", "--set", "kv-cache-type=Q4_0"]) == 0
    assert json.loads(capsys.readouterr().out)["settings"] == {"num_parallel": 3, "kv_cache_type": "q4_0"}
    assert cli.main(["models", "config", "my-voice", "--set", "kv_cache_type=q3"]) == 1
    assert "kv_cache_type" in capsys.readouterr().err
    assert cli.main(["models", "config", "my-voice", "--set", "nonsense=1"]) == 1
    capsys.readouterr()
    assert cli.main(["models", "config", "my-voice", "--unset", "kv_cache_type"]) == 0
    assert json.loads(capsys.readouterr().out)["settings"] == {"num_parallel": 3}
    assert cli.main(["list"]) == 0
    assert "preset of local-voice" in capsys.readouterr().out
    assert cli.main(["show", "my-voice"]) == 0
    out = capsys.readouterr().out
    assert "preset of" in out and "num_parallel" in out and "model.onnx" in out and "Licence" in out
    assert cli.main(["show", "local-voice", "--license"]) == 0
    out = capsys.readouterr().out
    assert "local-voice: Apache-2.0  https://example.org/licence" in out and "Apache License, Version 2.0" in out
    assert cli.main(["rm", "my-voice"]) == 0
    assert "a preset: the model's files stay" in capsys.readouterr().out
    assert cli.main(["rm", "local-voice", "my-voice"]) == 1
    captured = capsys.readouterr()
    assert "removed local-voice" in captured.out and "my-voice: not installed" in captured.err


def test_cli_rm_refuses_a_loaded_model_unless_forced(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    srv = LiveServer(settings)
    install_local(srv.core)
    srv.start()
    try:
        base = ["--url", srv.url, "--data-dir", str(settings.data_dir)]
        assert httpx.post(srv.url + "/api/v1/models/local-voice/load").status_code == 200
        assert cli.main(["rm", "local-voice", *base]) == 1
        assert "is loaded in the server" in capsys.readouterr().err
        assert srv.core.store.is_installed("local-voice")
        assert cli.main(["cp", "local-voice", "copy-voice", *base]) == 0  # through the server: it knows the preset
        assert srv.core.catalog.resolve("copy-voice") is not None
        capsys.readouterr()
        assert cli.main(["rm", "local-voice", "--force", *base]) == 0
        assert "removed local-voice" in capsys.readouterr().out
        assert "local-voice" not in srv.core.manager.loaded and not srv.core.store.is_installed("local-voice")
    finally:
        srv.stop()


def test_ago() -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    assert cli._ago(None) == "-" and cli._ago("garbage") == "-"
    assert cli._ago((now - timedelta(seconds=5)).isoformat()) == "just now"
    assert cli._ago((now - timedelta(hours=1, minutes=5)).isoformat()) == "1 hour ago"
    assert cli._ago((now - timedelta(days=15)).isoformat()) == "2 weeks ago"


# -- logs ----------------------------------------------------------------------------------------------------------
def test_log_locations() -> None:
    home = {"HOME": "/home/u"}
    assert logs.default_log_path(home, "linux") == Path("/home/u/.local/state/bolo/logs/server.log")
    assert logs.default_log_path({**home, "XDG_STATE_HOME": "/s"}, "linux") == Path("/s/bolo/logs/server.log")
    assert logs.default_log_path(home, "darwin") == Path("/home/u/Library/Logs/bolo/server.log")
    win = logs.default_log_path({"LOCALAPPDATA": "C:/Users/u/AppData/Local"}, "win32")
    assert win == Path("C:/Users/u/AppData/Local/bolo/logs/server.log")
    assert logs.running_as_service({"INVOCATION_ID": "abc", "JOURNAL_STREAM": "8:123"}) == "systemd"
    assert logs.running_as_service({"XPC_SERVICE_NAME": "org.example.bolo"}) == "launchd"
    assert logs.running_as_service({"XPC_SERVICE_NAME": "0"}) is None and logs.running_as_service({}) is None
    service = {**home, "INVOCATION_ID": "abc", "JOURNAL_STREAM": "8:1"}
    assert logs.resolve_log_file("", home, system="linux") is None  # off by default
    assert logs.resolve_log_file("", service, system="linux") == logs.default_log_path(home, "linux")  # a service
    assert logs.resolve_log_file("off", service) is None
    assert logs.resolve_log_file("", service, serving=False) is None
    assert logs.resolve_log_file("auto", home, system="darwin") == Path("/home/u/Library/Logs/bolo/server.log")
    assert logs.resolve_log_file("/var/log/bolo.log", home) == Path("/var/log/bolo.log")
    assert load_settings(env={"BOLO_LOG_FILE": "auto", "BOLO_LOG_MAX_MB": "5"}).log_max_mb == 5
    with pytest.raises(ValueError, match="log_max_mb"):
        load_settings(env={"BOLO_LOG_MAX_MB": "0"})


def test_file_logging_rotates_and_is_followed(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "server.log"
    root = logging.getLogger()
    before = list(root.handlers), root.level
    try:
        got = logs.configure(Settings(log_file=str(path), log_max_mb=1, log_backups=2))
        assert got == path and path == logs.ACTIVE_LOG_FILE
        lines: list[str] = []
        stop = threading.Event()
        reader = threading.Thread(target=lambda: lines.extend(logs.follow(path, 0.01, stop.is_set)), daemon=True)
        reader.start()
        time.sleep(0.1)
        log = logging.getLogger("bolo.test")
        log.info("first line")
        for i in range(1100):  # a little over 1 MB: one rotation
            log.info("x" * 1000 + f" {i}")
        log.info("after the rotation")
        deadline = time.time() + 5
        while not any("after the rotation" in line for line in lines) and time.time() < deadline:
            time.sleep(0.02)
        stop.set()
        reader.join(timeout=2)
        assert (tmp_path / "logs" / "server.log.1").is_file() and not (tmp_path / "logs" / "server.log.3").exists()
        assert any("first line" in line for line in lines) and any("after the rotation" in line for line in lines)
        assert logs.tail(path, 1)[0].endswith("after the rotation")
    finally:
        for h in [h for h in root.handlers if h not in before[0]]:
            root.removeHandler(h)
            h.close()
        root.setLevel(before[1])
        logs.ACTIVE_LOG_FILE = None


def test_cli_logs_and_doctor(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["logs"]) == 1
    err = capsys.readouterr().err
    assert "no log file is configured" in err and "journalctl -u bolo" in err and "docker logs" in err
    log_file = cli_env / "bolo.log"
    log_file.write_text("".join(f"line {i}\n" for i in range(10)))
    monkeypatch.setenv("BOLO_LOG_FILE", str(log_file))
    assert cli.main(["logs", "-n", "3"]) == 0
    assert capsys.readouterr().out.splitlines() == ["line 7", "line 8", "line 9"]
    assert cli.main(["logs", "--lines", "0"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 10
    assert cli.main(["doctor"]) == 0
    assert f"logs: {log_file} (log_file; rotated at 10 MB, 3 old files kept)" in capsys.readouterr().out
    monkeypatch.delenv("BOLO_LOG_FILE")
    assert cli.main(["doctor"]) == 0
    assert "logs: standard error of `bolo serve`" in capsys.readouterr().out


def test_logs_asks_the_running_server(settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    srv = LiveServer(settings)
    srv.start()
    path = tmp_path / "server.log"
    path.write_text("hello from the server\n")
    try:
        assert httpx.get(srv.url + "/api/v1/info").json()["log_file"] is None
        logs.ACTIVE_LOG_FILE = path
        assert cli.main(["logs", "--url", srv.url]) == 0
        assert capsys.readouterr().out.strip() == "hello from the server"
    finally:
        logs.ACTIVE_LOG_FILE = None
        srv.stop()


def test_every_setting_is_documented() -> None:
    """Each setting has a BOLO_* variable (automatic) and a row in docs/configuration.md."""
    from dataclasses import fields

    docs = (Path(__file__).resolve().parents[2] / "docs" / "configuration.md").read_text()
    missing = [f.name for f in fields(Settings) if f"`{f.name}`" not in docs]
    assert not missing, f"settings missing from docs/configuration.md: {missing}"
