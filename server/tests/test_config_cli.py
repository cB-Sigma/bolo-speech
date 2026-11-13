from __future__ import annotations

import json
from pathlib import Path

import pytest

from bolo import __version__
from bolo.cli import main
from bolo.config import Settings, load_settings
from conftest import wav_bytes


def test_defaults() -> None:
    s = load_settings(env={})
    assert (s.host, s.port) == ("127.0.0.1", 8000)
    assert s.api_key is None and s.cors_origins == [] and s.auto_pull is True
    assert s.wyoming is False and s.wyoming_port == 10300
    assert s.resolved_threads() >= 1
    assert Settings(workers=2, threads=0).resolved_threads() >= 1
    assert Settings(threads=3).resolved_threads() == 3


def test_precedence_file_env_flags(tmp_path: Path) -> None:
    cfg = tmp_path / "bolo.toml"
    cfg.write_text(
        """
port = 9000
default-stt = "whisper-small"
preload = ["moonshine-tiny-en", "silero-vad"]

[scheduler]
max_batch = 4
workers = 2

[server]
cors_origins = ["http://localhost:5173"]
"""
    )
    s = load_settings(cfg, env={})
    assert s.port == 9000 and s.default_stt == "whisper-small" and s.preload == ["moonshine-tiny-en", "silero-vad"]
    assert s.max_batch == 4 and s.workers == 2 and s.cors_origins == ["http://localhost:5173"]
    s = load_settings(
        cfg, env={"BOLO_PORT": "9100", "BOLO_PRELOAD": "a, b", "BOLO_WYOMING": "yes", "BOLO_DATA_DIR": str(tmp_path)}
    )
    assert s.port == 9100 and s.preload == ["a", "b"] and s.wyoming is True and s.data_dir == tmp_path
    s = load_settings(cfg, overrides={"port": 9200, "threads": None}, env={"BOLO_PORT": "9100"})
    assert s.port == 9200  # flags beat the environment, which beats the file
    assert load_settings(env={"BOLO_CONFIG": str(cfg)}).port == 9000


def test_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.toml", env={})
    bad = tmp_path / "bad.toml"
    bad.write_text("no_such_setting = 1\n")
    with pytest.raises(ValueError, match="unknown setting"):
        load_settings(bad, env={})
    with pytest.raises(ValueError, match="boolean"):
        load_settings(env={"BOLO_WYOMING": "maybe"})
    with pytest.raises(ValueError, match="workers"):
        load_settings(env={"BOLO_WORKERS": "0"})
    assert Settings(api_key="x", hf_token="y").public_dict()["api_key"] == "***"


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BOLO_DEFAULT_STT", "fake-stt")
    monkeypatch.setenv("BOLO_DEFAULT_TTS", "fake-tts")
    monkeypatch.setenv("BOLO_DEFAULT_VAD", "fake-vad")
    monkeypatch.setenv("BOLO_AUTO_PULL", "0")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out.strip() == f"bolo {__version__}"
    assert main([]) == 2
    assert "serve" in capsys.readouterr().out


def test_cli_list_show(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    assert "no models installed" in capsys.readouterr().out
    assert main(["list", "--fake-engine"]) == 0
    out = capsys.readouterr().out
    assert "fake-stt" in out and "moonshine" not in out
    assert main(["list", "--all", "--task", "vad", "--json"]) == 0
    models = json.loads(capsys.readouterr().out)["models"]
    assert {m["id"] for m in models} >= {"silero-vad"} and all(m["task"] == "vad" for m in models)
    assert main(["list", "-a", "-q", "hindi"]) == 0
    assert "piper-hi-in-rohan-medium" in capsys.readouterr().out
    assert main(["show", "moonshine-tiny"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == "moonshine-tiny-en" and shown["source"]["repo"] and shown["kind"] == "moonshine"
    assert main(["show", "nope"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_cli_transcribe_and_speak_with_fake_engine(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (cli_env / "in.wav").write_bytes(wav_bytes(2.0))
    assert main(["transcribe", "in.wav", "--fake-engine"]) == 0
    assert capsys.readouterr().out.strip() == "the quick brown fox jumps"
    assert main(["transcribe", "in.wav", "--fake-engine", "-f", "json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["model"] == "fake-stt" and doc["words"][0]["w"] == "the" and doc["timing"]["rtf"] > 0
    assert main(["transcribe", "in.wav", "--fake-engine", "-f", "srt"]) == 0
    assert "00:00:00,000 --> 00:00:02,000" in capsys.readouterr().out
    assert main(["speak", "Hello there.", "-o", "out.wav", "--fake-engine", "--voice", "beta", "-v"]) == 0
    captured = capsys.readouterr()
    assert "wrote out.wav (0.7s, wav)" in captured.out and "first audio after" in captured.err
    assert (cli_env / "out.wav").read_bytes()[:4] == b"RIFF"
    assert main(["speak", "Hello there.", "-o", "out.ulaw", "--fake-engine", "-f", "mulaw"]) == 0
    assert len((cli_env / "out.ulaw").read_bytes()) == pytest.approx(0.72 * 8000, rel=0.02)
    assert main(["transcribe", "missing.wav", "--fake-engine"]) == 1
    assert main(["transcribe", "in.wav"]) == 1  # the fake models do not exist without the flag
    assert "does not exist" in capsys.readouterr().err


def test_cli_bench_doctor_rm_pull(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--fake-engine", "-m", "fake-stt", "-c", "1,3", "-d", "0.5", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [p["concurrency"] for p in result["points"]] == [1, 3] and result["points"][1]["ok"] > 0
    assert main(["bench", "--fake-engine", "--kind", "tts", "-m", "fake-tts", "-c", "1", "-d", "0.3"]) == 0
    assert "TTFA P50" in capsys.readouterr().out
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"bolo {__version__}" in out and "engines:" in out and "resampler:" in out and "encode:" in out
    assert main(["rm", "moonshine-tiny-en"]) == 1
    assert "not installed" in capsys.readouterr().err
    assert main(["pull", "fake-stt", "--fake-engine"]) == 0
    assert "already installed" in capsys.readouterr().out
    assert main(["ps", "--url", "http://127.0.0.1:9"]) == 1
    assert "could not reach" in capsys.readouterr().err


def test_cli_ps_against_live_server(live, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    import httpx

    httpx.post(live.url + "/v1/audio/speech", json={"input": "Hello there.", "response_format": "pcm"})
    assert main(["ps", "--url", live.url]) == 0
    out = capsys.readouterr().out
    assert "fake-tts" in out and "uptime" in out
