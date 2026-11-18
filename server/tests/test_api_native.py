from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from bolo import __version__
from conftest import LiveServer, wav_file


def _built_in() -> set[str]:
    """Catalog entries that need no download (bolo's own detectors; WebRTC VAD once its package is installed)."""
    import importlib.util

    ids = {"energy-vad", "silence-only"}
    if importlib.util.find_spec("webrtcvad") is not None:
        ids |= {"webrtc-vad", "webrtc-vad-aggressive"}
    return ids


BUILT_IN = _built_in()


def test_info_and_health(live: LiveServer) -> None:
    info = httpx.get(live.url + "/api/v1/info").json()
    assert info["name"] == "bolo" and info["version"] == __version__
    assert {"name", "version", "engines", "device", "cpu_count", "platform", "auth_required", "data_dir"} <= set(info)
    assert all(set(e) == {"id", "version", "available", "reason"} for e in info["engines"])
    assert info["platform"].count("-") == 1 and info["cpu_count"] >= 1
    assert httpx.get(live.url + "/healthz").json() == {"status": "ok"}
    assert httpx.get(live.url + "/readyz").json() == {"status": "ready"}
    assert httpx.get(live.url + "/health").json() == {"message": "OK"}


def test_catalog_shape_and_filters(live: LiveServer) -> None:
    models = httpx.get(live.url + "/api/v1/catalog").json()["models"]
    assert len(models) > 100
    contract_keys = {
        "id", "aliases", "task", "engine", "family", "languages", "tier", "streaming", "sample_rate", "size_bytes",
        "params", "license", "gated", "voices", "features", "installed", "loaded", "pulling", "description",
        "source_url", "validated",
    }  # fmt: skip
    for m in models:
        assert contract_keys <= set(m), m["id"]
        assert m["tier"] in ("realtime", "balanced", "accuracy")
        assert m["validated"] in ("cpu", "untested")
        assert m["task"] in (
            "stt",
            "tts",
            "vad",
            "diarization",
            "turn",
            "separation",
            "denoise",
            "lid",
            "speech-to-speech",
            "audio-llm",
            "speaker-embedding",
            "audio-tagging",
            "emotion",
            "voice-conversion",
            "align",
            "punctuation",
            "echo-cancellation",
        )
        assert isinstance(m["requires"], list) and isinstance(m["available"], bool)
    tts = httpx.get(live.url + "/api/v1/catalog", params={"task": "tts"}).json()["models"]
    assert tts and all(m["task"] == "tts" for m in tts)
    fake_tts = next(m for m in tts if m["id"] == "fake-tts")
    assert fake_tts["voices"][0] == {"id": "alpha", "name": "Alpha", "language": "en-US", "gender": "f"}
    installed = httpx.get(live.url + "/api/v1/catalog", params={"installed": "true"}).json()["models"]
    assert {m["id"] for m in installed} == {
        "fake-stt",
        "fake-stt-streaming",
        "fake-tts",
        "fake-vad",
        "fake-turn",
        "fake-turn-text",
        "fake-speaker-embedding",
        "fake-diarization",
        "fake-diarization-overlap",
        "fake-separation",
        "fake-separation-blind",
        "fake-denoise",
        "fake-denoise-48k",
        "fake-s2s",
        "fake-lid",
        "fake-align",
        "fake-punctuation",
        "fake-stt-ta",
        "fake-tagging",
        "fake-emotion",
    } | BUILT_IN
    not_installed = httpx.get(live.url + "/api/v1/catalog", params={"installed": "false", "task": "vad"}).json()[
        "models"
    ]
    assert not_installed and all(not m["installed"] for m in not_installed)
    hindi = httpx.get(live.url + "/api/v1/catalog", params={"q": "hindi piper"}).json()["models"]
    assert hindi and all("hi" in m["languages"] for m in hindi)
    assert httpx.get(live.url + "/api/v1/catalog", params={"q": "zzzz-nothing"}).json() == {"models": []}
    one = httpx.get(live.url + "/api/v1/models/fake-realtime").json()
    assert one["id"] == "fake-stt-streaming" and one["streaming"] is True


def test_load_unload_and_loaded_flag(live: LiveServer) -> None:
    r = httpx.post(live.url + "/api/v1/models/fake-stt/load")
    assert r.status_code == 200 and r.json()["loaded"] is True
    cat = {m["id"]: m for m in httpx.get(live.url + "/api/v1/catalog", params={"installed": "true"}).json()["models"]}
    assert cat["fake-stt"]["loaded"] is True and cat["fake-tts"]["loaded"] is False
    assert httpx.post(live.url + "/api/v1/models/fake-stt/unload").json() == {
        "id": "fake-stt",
        "loaded": False,
        "unloaded": True,
    }
    assert httpx.post(live.url + "/api/v1/models/nope/load").status_code == 404
    assert httpx.delete(live.url + "/api/v1/models/fake-stt").status_code == 400  # built-in models cannot be deleted


def test_pull_job_for_installed_and_unknown_models(live: LiveServer) -> None:
    r = httpx.post(live.url + "/api/v1/models/pull", json={"id": "fake-stt"})
    assert r.status_code == 202
    job = httpx.get(live.url + f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert {"id", "kind", "model", "state", "progress", "bytes_done", "bytes_total", "error"} <= set(job)
    assert job["kind"] == "pull" and job["state"] == "done" and job["progress"] == 1.0
    assert httpx.post(live.url + "/api/v1/models/pull", json={"id": "nope"}).status_code == 404
    assert httpx.get(live.url + "/api/v1/jobs/j_999999").status_code == 404
    assert httpx.post(live.url + "/api/v1/models/pull", json={}).status_code == 400


def test_requests_traces_and_stats(live: LiveServer) -> None:
    for seconds in (1.0, 2.0):
        httpx.post(live.url + "/v1/audio/transcriptions", files={"file": wav_file(seconds)}, data={"model": "fake-stt"})
    httpx.post(live.url + "/v1/audio/speech", json={"input": "Hello there.", "response_format": "pcm"})
    traces = httpx.get(live.url + "/api/v1/requests").json()["requests"]
    assert [t["kind"] for t in traces] == ["tts", "stt", "stt"]  # newest first
    keys = {
        "id", "kind", "transport", "model", "started_at", "status", "error", "audio_seconds", "chars", "language",
        "phases", "first_partial_ms", "final_ms", "ttfa_ms", "total_ms", "rtf", "batch_size", "text_preview",
    }  # fmt: skip
    assert all(keys <= set(t) for t in traces)
    stt = traces[1]
    assert stt["audio_seconds"] == 2.0 and stt["transport"] == "http" and stt["status"] == "ok"
    assert {p["name"] for p in stt["phases"]} <= {"queue", "resample", "vad", "decode", "finalize"}
    assert stt["final_ms"] is not None and stt["ttfa_ms"] is None and stt["rtf"] > 0
    assert stt["text_preview"] == "the quick brown fox jumps"
    tts = traces[0]
    assert tts["ttfa_ms"] is not None and tts["chars"] == len("Hello there.")
    assert httpx.get(live.url + "/api/v1/requests", params={"kind": "tts"}).json()["requests"] == [tts]
    assert len(httpx.get(live.url + "/api/v1/requests", params={"limit": 1}).json()["requests"]) == 1
    assert httpx.get(live.url + "/api/v1/requests", params={"model": "fake-vad"}).json()["requests"] == []
    assert httpx.get(live.url + f"/api/v1/requests/{stt['id']}").json() == stt
    stats = httpx.get(live.url + "/api/v1/stats").json()
    assert stats["totals"] == {
        "requests": 3,
        "errors": 0,
        "audio_seconds_in": 3.0,
        "audio_seconds_out": tts["audio_seconds"],
    }
    by_id = {m["id"]: m for m in stats["models"]}
    assert by_id["fake-stt"]["requests"] == 2 and by_id["fake-stt"]["latency_ms"]["final"]["p50"] > 0
    assert by_id["fake-tts"]["latency_ms"]["ttfa"]["p50"] > 0


def test_metrics_exposition(live: LiveServer) -> None:
    httpx.post(live.url + "/v1/audio/transcriptions", files={"file": wav_file(1.0)}, data={"model": "fake-stt"})
    text = httpx.get(live.url + "/metrics").text
    assert "# TYPE bolo_requests_total counter" in text
    assert "bolo_requests_total 1" in text
    assert 'bolo_model_loaded{model="fake-stt",task="stt"} 1' in text
    assert 'bolo_latency_ms{metric="final",quantile="0.95",model="fake-stt",task="stt"}' in text
    assert 'bolo_audio_seconds_total{direction="in"} 1.0' in text
    for line in text.splitlines():  # every sample line parses as "name{labels} number"
        if line and not line.startswith("#"):
            float(line.rsplit(" ", 1)[1])


def test_samples(live: LiveServer) -> None:
    from bolo.audio import decode_audio

    samples = httpx.get(live.url + "/api/v1/samples").json()["samples"]
    by_id = {s["id"]: s for s in samples}
    assert by_id["hi-text-short"]["kind"] == "text" and by_id["hi-text-short"]["language"] == "hi"
    tone = by_id["tone-bursts"]
    assert tone["kind"] == "audio" and tone["url"] == "/api/v1/samples/tone-bursts.wav"
    for s in samples:
        if s["kind"] == "audio":
            r = httpx.get(live.url + s["url"])
            assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
            audio, rate = decode_audio(r.content, filename="x.wav")
            assert len(audio) / rate == pytest.approx(s["seconds"], abs=0.2), s["id"]
    assert httpx.get(live.url + "/api/v1/samples/nope.wav").status_code == 400


def wait_job(live: LiveServer, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = httpx.get(live.url + f"/api/v1/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_bench_stt_and_tts(live: LiveServer) -> None:
    r = httpx.post(
        live.url + "/api/v1/bench", json={"kind": "stt", "model": "fake-stt", "concurrency": [1, 4], "duration_s": 1.0}
    )
    assert r.status_code == 202
    job = wait_job(live, r.json()["job_id"])
    assert job["state"] == "done", job
    points = job["result"]["points"]
    assert [p["concurrency"] for p in points] == [1, 4]
    for p in points:
        assert {
            "concurrency",
            "ok",
            "errors",
            "first_partial_ms",
            "final_ms",
            "ttfa_ms",
            "rtf",
            "throughput_x_realtime",
        } <= set(p)
        assert p["ok"] > 0 and p["errors"] == 0 and set(p["final_ms"]) == {"p50", "p95"} and p["ttfa_ms"] is None
    # the fake engine decodes a batch in the time of one item: four clients get about four times the throughput
    # (asserted with a margin: on a busy test machine the HTTP overhead around the fake delays grows)
    assert points[1]["throughput_x_realtime"] > 2.0 * points[0]["throughput_x_realtime"]
    assert points[1]["mean_batch"] > 2
    r = httpx.post(
        live.url + "/api/v1/bench",
        json={
            "kind": "tts",
            "model": "fake-tts",
            "concurrency": [2],
            "duration_s": 0.6,
            "sample": "builtin:en-text-short",
        },
    )
    job = wait_job(live, r.json()["job_id"])
    p = job["result"]["points"][0]
    assert p["ok"] > 0 and p["ttfa_ms"]["p50"] > 0 and p["first_partial_ms"] is None and p["throughput_x_realtime"] > 1
    r = httpx.post(
        live.url + "/api/v1/bench",
        json={"kind": "stt", "model": "fake-stt-streaming", "concurrency": [2], "duration_s": 0.5, "mode": "realtime"},
    )
    job = wait_job(live, r.json()["job_id"])
    p = job["result"]["points"][0]
    assert job["result"]["mode"] == "realtime" and p["ok"] >= 2
    assert p["first_partial_ms"]["p50"] < 300 and p["final_ms"]["p50"] < 300
    assert 1.0 < p["throughput_x_realtime"] <= 2.0  # paced at 1x: two clients cannot exceed 2x
    bad = httpx.post(live.url + "/api/v1/bench", json={"kind": "stt", "model": "fake-stt", "concurrency": [0]})
    assert wait_job(live, bad.json()["job_id"])["state"] == "error"
    assert httpx.post(live.url + "/api/v1/bench", json={"kind": "nope"}).status_code == 400


def test_events_stream(live: LiveServer) -> None:
    events: list[dict] = []
    ready = threading.Event()

    def listen() -> None:
        with httpx.stream("GET", live.url + "/api/v1/events", timeout=20) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                    ready.set()
                    kinds = {e["type"] for e in events}
                    if {"stats", "request", "model", "job"} <= kinds and sum(e["type"] == "stats" for e in events) >= 2:
                        return

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    assert ready.wait(5)
    httpx.post(live.url + "/v1/audio/transcriptions", files={"file": wav_file(1.0)}, data={"model": "fake-stt"})
    job_id = httpx.post(
        live.url + "/api/v1/bench", json={"kind": "stt", "model": "fake-stt", "concurrency": [1], "duration_s": 0.3}
    ).json()["job_id"]
    thread.join(timeout=15)
    assert not thread.is_alive()
    first = events[0]
    assert first["type"] == "stats" and "uptime_s" in first and "models" in first  # same shape as /stats
    request = next(e for e in events if e["type"] == "request")
    assert request["kind"] == "stt" and request["model"] == "fake-stt" and "phases" in request
    model = next(e for e in events if e["type"] == "model")
    assert model == {"type": "model", "id": "fake-stt", "event": "loaded", "processor": "100% CPU"}
    job = next(e for e in events if e["type"] == "job")
    assert job["id"] == job_id and job["kind"] == "bench"


def test_console_placeholder_and_spa_fallback(live: LiveServer, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from bolo.api import static

    monkeypatch.setattr(static, "ui_dir", lambda: tmp_path / "empty")
    r = httpx.get(live.url + "/")
    assert r.status_code == 200 and "web console bundle is not included" in r.text
    assert r.headers["cache-control"] == "no-cache"
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html>console</html>")
    (ui / "assets" / "app-abc123.js").write_text("console.log(1)")
    (ui / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(static, "ui_dir", lambda: ui)
    r = httpx.get(live.url + "/")
    assert r.text == "<html>console</html>" and r.headers["cache-control"] == "no-cache"
    r = httpx.get(live.url + "/models/some/deep/link")  # client-side routes fall back to the shell
    assert r.status_code == 200 and r.text == "<html>console</html>"
    r = httpx.get(live.url + "/assets/app-abc123.js")
    assert r.text == "console.log(1)" and "immutable" in r.headers["cache-control"]
    assert "javascript" in r.headers["content-type"]
    assert httpx.get(live.url + "/favicon.svg").headers["cache-control"] == "public, max-age=3600"
    assert httpx.get(live.url + "/assets/missing.js").status_code == 404
    assert httpx.get(live.url + "/../../etc/passwd").status_code in (200, 404)  # never escapes the bundle
    assert "root:" not in httpx.get(live.url + "/..%2f..%2f..%2fetc%2fpasswd").text
    r = httpx.get(live.url + "/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert httpx.get(live.url + "/v1/nope").status_code == 404


def test_overload_returns_429_with_retry_after(settings) -> None:  # type: ignore[no-untyped-def]
    import concurrent.futures

    settings.max_queue = 1
    settings.max_batch = 1
    srv = LiveServer(settings)
    srv.start()
    try:
        httpx.post(srv.url + "/api/v1/models/fake-stt/load")
        srv.core.manager.loaded["fake-stt"].model.base_delay_s = 0.5  # type: ignore[attr-defined]

        def call(_: int) -> httpx.Response:
            return httpx.post(
                srv.url + "/v1/audio/transcriptions",
                files={"file": wav_file(1.0)},
                data={"model": "fake-stt"},
                timeout=30,
            )

        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            responses = list(pool.map(call, range(6)))
        codes = sorted(r.status_code for r in responses)
        assert codes.count(200) >= 2 and codes.count(429) >= 2
        rejected = next(r for r in responses if r.status_code == 429)
        assert int(rejected.headers["retry-after"]) >= 1
        assert rejected.json()["error"]["type"] == "rate_limit_error"
    finally:
        srv.stop()


def test_console_is_cross_origin_isolated_by_default(live: LiveServer, settings, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """COOP/COEP let the in-browser recogniser use WebAssembly threads; the API itself is unaffected."""
    from bolo.api import static

    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html>console</html>")
    (ui / "assets" / "worker-abc.js").write_text("self.onmessage = () => {}")
    monkeypatch.setattr(static, "ui_dir", lambda: ui)
    for path in ("/", "/index.html", "/assets/worker-abc.js"):
        r = httpx.get(live.url + path)
        assert r.status_code == 200, path
        assert r.headers["cross-origin-opener-policy"] == "same-origin", path
        assert r.headers["cross-origin-embedder-policy"] == "credentialless", path
    api = httpx.get(live.url + "/api/v1/info")
    assert api.status_code == 200 and "cross-origin-embedder-policy" not in api.headers
    settings.cross_origin_isolation = False
    off = LiveServer(settings)
    off.start()
    try:
        assert "cross-origin-opener-policy" not in httpx.get(off.url + "/").headers
    finally:
        off.stop()


def test_bundled_console_is_served_when_present(live: LiveServer) -> None:
    from bolo.api import static

    r = httpx.get(live.url + "/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    if static.has_bundle():
        assert "<div id=" in r.text and "web console bundle is not included" not in r.text
        import re

        assets = re.findall(r'(?:src|href)="\./?(assets/[^"]+)"', r.text)
        assert assets
        for asset in assets:
            a = httpx.get(live.url + "/" + asset)
            assert a.status_code == 200 and "immutable" in a.headers["cache-control"], asset
