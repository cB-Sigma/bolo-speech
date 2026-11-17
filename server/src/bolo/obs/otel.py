"""OpenTelemetry export of request traces (optional: the ``otel`` extra).

Every finished :class:`~bolo.obs.trace.Trace` becomes one span named
``bolo.<kind>`` (``bolo.stt``, ``bolo.tts``, ``bolo.agent`` ...) with a child
span per phase (``queue``, ``decode``, ``synthesize_first`` ...), placed on the
wall clock where the phase ran. Attributes carry the model id, the device it
ran on, the transport, the sizes (seconds of audio, characters) and the
latencies bolo measured. Text (transcripts, what was spoken) is never exported.

The spans are built after the request has finished, from the trace bolo keeps
anyway, so the request path does no extra work; they go out over OTLP through a
batching processor. Prometheus ``/metrics``, ``/api/v1/requests`` and the event
stream are unaffected.

Switched on by the setting ``otel`` (``BOLO_OTEL``); unset, it is on when one
of the standard endpoint variables (``OTEL_EXPORTER_OTLP_ENDPOINT``,
``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``) is set. Everything else comes from the
standard variables the OpenTelemetry SDK reads: ``OTEL_SERVICE_NAME`` (default
``bolo``), ``OTEL_RESOURCE_ATTRIBUTES``, ``OTEL_EXPORTER_OTLP_PROTOCOL``
(``http/protobuf`` by default; ``grpc`` needs
``opentelemetry-exporter-otlp-proto-grpc``), ``OTEL_EXPORTER_OTLP_HEADERS``,
``OTEL_TRACES_SAMPLER`` and the batch processor's ``OTEL_BSP_*``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bolo.obs.trace import Trace

log = logging.getLogger("bolo.otel")

ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")
INSTALL_HINT = 'pip install "bolo-speech[otel]"'
#: string values of ``Trace.extra`` that are exported (ids and names of models and settings, never text)
EXTRA_STRINGS = frozenset(
    {
        "stt_model",
        "tts_model",
        "llm_model",
        "voice",
        "pipeline",
        "provider",
        "diarization_model",
        "separation_model",
        "separation",
    }
)
#: numeric fields of a trace exported as ``bolo.<name>``
NUMBERS = (
    "audio_seconds",
    "chars",
    "first_partial_ms",
    "final_ms",
    "ttfa_ms",
    "total_ms",
    "rtf",
    "batch_size",
    "compute_ms",
)


def wanted(setting: bool | None, environ: Mapping[str, str] | None = None) -> bool:
    """Whether export is switched on: the setting when it is set, else an OTLP endpoint in the environment."""
    env = os.environ if environ is None else environ
    if setting is not None:
        return setting
    if env.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    return any(env.get(name) for name in ENDPOINT_VARS)


def _redact(url: str) -> str:
    """An endpoint without the credentials a URL may carry (``user:pass@``) or its query string."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    if not parts.scheme or not host:
        return ""
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def describe(
    setting: bool | None, export: OtelExport | None, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """What ``GET /api/v1/info`` says about export (``capabilities.opentelemetry``): whether it is switched on and
    running, and where the spans go. Read-only; never a header value or a credential."""
    env = os.environ if environ is None else environ
    switched_on = wanted(setting, env)
    installed = importlib.util.find_spec("opentelemetry") is not None and (
        importlib.util.find_spec("opentelemetry.sdk") is not None
    )
    protocol = (
        env.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or env.get("OTEL_EXPORTER_OTLP_PROTOCOL") or ""
    ).strip() or "http/protobuf"
    raw = next((env[name] for name in ENDPOINT_VARS if env.get(name)), None)
    endpoint = _redact(raw) if raw else None
    if endpoint is None and switched_on:
        endpoint = "http://localhost:4317" if protocol == "grpc" else "http://localhost:4318"
    return {
        "enabled": export is not None,
        "switched_on": switched_on,
        "setting": setting,
        "installed": installed,
        "install": None if installed else INSTALL_HINT,
        "endpoint": endpoint if switched_on else None,
        "endpoint_source": ("environment" if raw else "default") if switched_on else None,
        "protocol": protocol if switched_on else None,
        "service_name": (env.get("OTEL_SERVICE_NAME") or "bolo") if switched_on else None,
        "exported": export.exported if export is not None else 0,
    }


def attributes(trace: Trace, device: str | None) -> dict[str, Any]:
    """The request span's attributes (OpenTelemetry takes str, bool, int and float values only)."""
    out: dict[str, Any] = {
        "bolo.request_id": trace.id,
        "bolo.kind": trace.kind,
        "bolo.transport": trace.transport,
        "bolo.model": trace.model,
        "bolo.status": trace.status,
    }
    if device:
        out["bolo.device"] = device
    if trace.language:
        out["bolo.language"] = trace.language
    for name in NUMBERS:
        value = getattr(trace, name)
        if value is not None and not (name == "compute_ms" and not value):
            out[f"bolo.{name}"] = value
    for key, value in trace.extra.items():
        if isinstance(value, (bool, int, float)) or (isinstance(value, str) and key in EXTRA_STRINGS):
            out[f"bolo.{key}"] = value
    if trace.error:
        out["error.type"] = trace.error.split(":")[0][:120]
    return out


class OtelExport:
    """Turns finished traces into spans. ``tracer`` is an OpenTelemetry tracer; ``device_of`` says where a model
    ran (or None)."""

    def __init__(self, provider: Any, device_of: Callable[[str], str | None] | None = None) -> None:
        from opentelemetry.trace import SpanKind, Status, StatusCode

        from bolo import __version__

        self.provider = provider
        self.tracer = provider.get_tracer("bolo", __version__)
        self.device_of = device_of or (lambda _model: None)
        self._kind = SpanKind.SERVER
        self._ok = Status(StatusCode.OK)
        self._status = Status
        self._error = StatusCode.ERROR
        self.exported = 0

    def export(self, trace: Trace) -> None:
        """A trace listener (:attr:`bolo.obs.trace.TraceStore.listeners`): never raises."""
        try:
            self._export(trace)
        except Exception:  # an exporter problem must not touch the request path
            log.debug("could not export trace %s", trace.id, exc_info=True)

    def _export(self, trace: Trace) -> None:
        from opentelemetry import trace as ot

        start = trace.started_ns
        total_ms = (
            trace.total_ms
            if trace.total_ms is not None
            else max((p.start_ms + p.dur_ms for p in trace.phases), default=0.0)
        )
        end = start + int(total_ms * 1e6)
        try:
            device = self.device_of(trace.model)
        except Exception:
            device = None
        span = self.tracer.start_span(
            f"bolo.{trace.kind}", kind=self._kind, start_time=start, attributes=attributes(trace, device)
        )
        context = ot.set_span_in_context(span)
        for phase in trace.phases:
            p_start = start + int(phase.start_ms * 1e6)
            child = self.tracer.start_span(
                phase.name,
                context=context,
                start_time=p_start,
                attributes={"bolo.phase": phase.name, "bolo.model": trace.model},
            )
            child.end(end_time=p_start + int(phase.dur_ms * 1e6))
            end = max(end, p_start + int(phase.dur_ms * 1e6))
        if trace.status == "ok":
            span.set_status(self._ok)
        else:
            span.set_status(self._status(self._error, trace.error or trace.status))
        span.end(end_time=end)
        self.exported += 1

    def shutdown(self) -> None:
        try:
            self.provider.shutdown()
        except Exception:
            log.debug("OpenTelemetry shutdown failed", exc_info=True)


def _otlp_exporter(environ: Mapping[str, str]) -> Any:
    protocol = (
        environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or ""
    ).strip()
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
        except ImportError as exc:
            raise ImportError(
                "OTEL_EXPORTER_OTLP_PROTOCOL=grpc needs opentelemetry-exporter-otlp-proto-grpc "
                "(the otel extra brings the http/protobuf exporter)"
            ) from exc
        return GrpcExporter()
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def setup(
    setting: bool | None,
    device_of: Callable[[str], str | None] | None = None,
    *,
    exporter: Any = None,
    environ: Mapping[str, str] | None = None,
) -> OtelExport | None:
    """An :class:`OtelExport` when export is switched on and the SDK is installed, else None (with one warning
    when it is switched on but cannot run). ``exporter``: a span exporter to use instead of OTLP (tests), sent
    spans one at a time."""
    env = os.environ if environ is None else environ
    if exporter is None and not wanted(setting, env):
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        from bolo import __version__

        resource = Resource.create(
            {"service.name": env.get("OTEL_SERVICE_NAME") or "bolo", "service.version": __version__}
        )
        provider = TracerProvider(resource=resource)
        if exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        else:
            provider.add_span_processor(BatchSpanProcessor(_otlp_exporter(env)))
    except ImportError as exc:
        missing = str(exc) if "OTEL_EXPORTER_OTLP_PROTOCOL" in str(exc) else f"{exc.name or exc} is not installed"
        log.warning(
            "OpenTelemetry export is switched on (setting otel or OTEL_EXPORTER_OTLP_ENDPOINT) but cannot start: "
            "%s. Install it with %s. Request traces, /metrics and /api/v1/requests work as before.",
            missing,
            INSTALL_HINT,
        )
        return None
    endpoint = next((env[name] for name in ENDPOINT_VARS if env.get(name)), None)
    log.info("exporting request traces to OpenTelemetry%s", f" at {endpoint}" if endpoint else "")
    return OtelExport(provider, device_of)
