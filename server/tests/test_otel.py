"""OpenTelemetry export of request traces: spans in an in-memory exporter, switching on, a missing extra."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.obs import otel
from bolo.obs.trace import Trace
from conftest import LiveServer, wav_file


def test_switched_on_by_the_setting_or_an_endpoint_variable() -> None:
    assert otel.wanted(None, {}) is False
    assert otel.wanted(None, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318"}) is True
    assert otel.wanted(None, {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://c:4318/v1/traces"}) is True
    assert otel.wanted(None, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://c:4318", "OTEL_SDK_DISABLED": "true"}) is False
    assert otel.wanted(False, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318"}) is False
    assert otel.wanted(True, {}) is True
    assert load_settings(env={"BOLO_OTEL": "1"}, config_file=None).otel is True
    assert load_settings(env={}, config_file=None).otel is None
    assert otel.setup(None, environ={}) is None  # off: nothing is imported or started


def test_missing_extra_gives_one_clear_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)  # as if the otel extra were not installed
    with caplog.at_level(logging.WARNING, logger="bolo.otel"):
        assert otel.setup(True, environ={}) is None
    (record,) = [r for r in caplog.records if r.name == "bolo.otel"]
    assert record.levelno == logging.WARNING
    assert 'pip install "bolo-speech[otel]"' in record.getMessage() and "/metrics" in record.getMessage()


async def test_server_starts_and_serves_without_the_extra(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    settings.otel = True
    with caplog.at_level(logging.WARNING, logger="bolo.otel"):
        core = Bolo(settings)
    assert core.otel is None and len([r for r in caplog.records if r.name == "bolo.otel"]) == 1
    await core.start()
    try:
        assert (await core.transcribe(*_speech(), model="fake-stt")).text
    finally:
        await core.stop()


def _speech() -> tuple[Any, int]:
    from conftest import speechlike

    return speechlike(1.5), 16000


@pytest.fixture
def spans() -> Iterator[Any]:
    pytest.importorskip("opentelemetry.sdk", reason="needs the otel extra")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    yield InMemorySpanExporter()


def _attach(core: Bolo, exporter: Any) -> otel.OtelExport:
    export = otel.setup(None, core._device_of, exporter=exporter, environ={"OTEL_SERVICE_NAME": "bolo-test"})
    assert export is not None
    core.otel = export
    core.traces.listeners.append(export.export)
    return export


def test_request_trace_becomes_a_span_with_a_child_per_phase(live: LiveServer, spans: Any) -> None:
    _attach(live.core, spans)
    r = httpx.post(live.url + "/v1/audio/transcriptions", files={"file": wav_file(2.0)}, data={"model": "fake-stt"})
    assert r.status_code == 200
    (trace,) = [t for t in live.core.traces.recent(5) if t.kind == "stt"]
    finished = spans.get_finished_spans()
    (root,) = [s for s in finished if s.parent is None]
    children = [s for s in finished if s.parent is not None]
    assert root.name == "bolo.stt" and root.resource.attributes["service.name"] == "bolo-test"
    attrs = root.attributes
    assert attrs["bolo.request_id"] == trace.id and attrs["bolo.model"] == "fake-stt"
    assert attrs["bolo.kind"] == "stt" and attrs["bolo.transport"] == "http" and attrs["bolo.status"] == "ok"
    assert attrs["bolo.device"] == "cpu" and attrs["bolo.audio_seconds"] == pytest.approx(2.0, abs=0.01)
    assert "bolo.text_preview" not in attrs and not any("text" in k for k in attrs)  # what was said stays here
    assert root.status.is_ok
    assert [c.name for c in children] == [p.name for p in trace.phases] and children
    for child, phase in zip(children, trace.phases):
        assert child.parent.span_id == root.context.span_id and child.context.trace_id == root.context.trace_id
        assert child.start_time == root.start_time + int(phase.start_ms * 1e6)
        assert (child.end_time - child.start_time) / 1e6 == pytest.approx(phase.dur_ms, abs=0.01)
        assert root.start_time <= child.start_time and child.end_time <= root.end_time
    assert (root.end_time - root.start_time) / 1e6 == pytest.approx(trace.total_ms, abs=0.01)


def test_speech_and_failed_requests(live: LiveServer, spans: Any) -> None:
    export = _attach(live.core, spans)
    r = httpx.post(live.url + "/v1/audio/speech", json={"model": "fake-tts", "input": "Hello there.", "voice": "alpha"})
    assert r.status_code == 200
    (tts,) = [s for s in spans.get_finished_spans() if s.name == "bolo.tts"]
    assert tts.attributes["bolo.chars"] == len("Hello there.") and tts.attributes["bolo.model"] == "fake-tts"
    assert "bolo.ttfa_ms" in tts.attributes
    assert {"synthesize_first"} <= {s.name for s in spans.get_finished_spans() if s.parent is not None}
    spans.clear()
    failed = Trace("stt", "http", "fake-stt")
    with failed.phase("decode"):
        pass
    failed.finish("error", "engine_unavailable: the model crashed")
    live.core.traces.record(failed)
    (root,) = [s for s in spans.get_finished_spans() if s.parent is None]
    assert not root.status.is_ok and root.status.description == "engine_unavailable: the model crashed"
    assert root.attributes["error.type"] == "engine_unavailable" and "bolo.device" not in root.attributes  # not loaded
    assert export.exported == 2


def test_an_exporter_that_fails_does_not_touch_requests(live: LiveServer, spans: Any) -> None:
    export = _attach(live.core, spans)

    def broken(_trace: Trace) -> None:
        raise RuntimeError("collector exploded")

    export._export = broken  # type: ignore[method-assign]
    r = httpx.post(live.url + "/v1/audio/transcriptions", files={"file": wav_file(1.0)}, data={"model": "fake-stt"})
    assert r.status_code == 200 and live.core.traces.recent(1)[0].status == "ok"
