"""Prometheus text exposition (no client library needed for a handful of gauges)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bolo import __version__

if TYPE_CHECKING:
    from bolo.core import Bolo


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._seen: set[str] = set()

    def add(self, name: str, kind: str, help_: str, value: float | int | None, **labels: str) -> None:
        if value is None:
            return
        if name not in self._seen:
            self._seen.add(name)
            self.lines.append(f"# HELP {name} {help_}")
            self.lines.append(f"# TYPE {name} {kind}")
        label_str = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        self.lines.append(f"{name}{{{label_str}}} {value}" if label_str else f"{name} {value}")


def render_metrics(core: Bolo) -> str:
    w = _Writer()
    stats = core.stats()
    w.add("bolo_info", "gauge", "Server version.", 1, version=__version__)
    w.add("bolo_uptime_seconds", "gauge", "Seconds since the server started.", stats["uptime_s"])
    w.add("bolo_active_streams", "gauge", "Open realtime streams.", stats["active_streams"])
    w.add("bolo_queue_depth", "gauge", "Jobs waiting in all model queues.", stats["queue_depth"])
    w.add(
        "bolo_process_cpu_percent",
        "gauge",
        "Process CPU use as a percentage of all cores.",
        stats["system"]["cpu_percent"],
    )
    w.add("bolo_process_resident_memory_bytes", "gauge", "Resident set size.", stats["system"]["rss_bytes"])
    w.add("bolo_process_threads", "gauge", "Threads in the server process.", stats["system"]["threads"])
    w.add("bolo_requests_total", "counter", "Finished requests.", stats["totals"]["requests"])
    w.add(
        "bolo_request_errors_total",
        "counter",
        "Requests that did not finish with status ok.",
        stats["totals"]["errors"],
    )
    w.add(
        "bolo_audio_seconds_total",
        "counter",
        "Seconds of audio processed.",
        stats["totals"]["audio_seconds_in"],
        direction="in",
    )
    w.add(
        "bolo_audio_seconds_total",
        "counter",
        "Seconds of audio processed.",
        stats["totals"]["audio_seconds_out"],
        direction="out",
    )
    for item in stats.get("providers", []):
        labels = {"provider": str(item["provider"]), "model": str(item["model"])}
        w.add(
            "bolo_provider_requests_total", "counter", "Requests sent to cloud providers.", item["requests"], **labels
        )
        w.add("bolo_provider_errors_total", "counter", "Requests a cloud provider failed.", item["errors"], **labels)
        w.add(
            "bolo_provider_latency_seconds_total",
            "counter",
            "Time spent waiting for cloud providers.",
            round(item["latency_ms_total"] / 1000.0, 3),
            **labels,
        )
        w.add(
            "bolo_provider_bytes_sent_total", "counter", "Bytes sent to cloud providers.", item["bytes_sent"], **labels
        )
        w.add(
            "bolo_provider_bytes_received_total",
            "counter",
            "Bytes received from cloud providers.",
            item["bytes_received"],
            **labels,
        )
        w.add(
            "bolo_provider_audio_seconds_total",
            "counter",
            "Seconds of audio sent to (speech-to-text) or received from (text-to-speech) cloud providers.",
            item["audio_seconds"],
            **labels,
        )
        w.add("bolo_provider_characters_total", "counter", "Characters sent to cloud voices.", item["chars"], **labels)
        for currency, amount in item["estimated_cost"].items():
            w.add(
                "bolo_provider_estimated_cost_total",
                "counter",
                "Estimated spend at cloud providers, from the operator's price table (not a bill).",
                amount,
                currency=str(currency),
                **labels,
            )
    for item in stats.get("fallbacks", []):
        w.add(
            "bolo_fallbacks_total",
            "counter",
            "Requests answered by a fallback model.",
            item["count"],
            requested=str(item["requested"]),
            answered=str(item["answered"]),
        )
    for status, count in stats["separation"]["requests"].items():
        w.add(
            "bolo_separation_requests_total",
            "counter",
            "Requests that asked for separate_overlaps, by outcome (applied, partial, skipped).",
            count,
            status=status,
        )
    w.add(
        "bolo_separation_audio_seconds_total",
        "counter",
        "Seconds of audio sent through the separation model (window length x speakers).",
        stats["separation"]["audio_seconds"],
    )
    for m in stats["models"]:
        labels = {"model": m["id"], "task": m["task"]}
        w.add("bolo_model_loaded", "gauge", "1 when the model is in memory.", int(m["loaded"]), **labels)
        w.add("bolo_model_memory_bytes", "gauge", "Estimated memory held by the model.", m["memory_bytes"], **labels)
        w.add("bolo_model_active_requests", "gauge", "Requests and streams using the model now.", m["active"], **labels)
        w.add("bolo_model_requests_total", "counter", "Finished requests per model.", m["requests"], **labels)
        w.add("bolo_model_request_errors_total", "counter", "Failed requests per model.", m["errors"], **labels)
        for metric, values in m["latency_ms"].items():
            for q, v in (values or {}).items():
                w.add(
                    "bolo_latency_ms", "gauge", "Rolling latency quantiles (see docs/api.md for definitions).", v,
                    metric=metric, quantile=str(int(q[1:]) / 100), **labels,
                )  # fmt: skip
        for q, v in (m["rtf"] or {}).items():
            w.add(
                "bolo_real_time_factor",
                "gauge",
                "Rolling real-time factor quantiles.",
                v,
                quantile=str(int(q[1:]) / 100),
                **labels,
            )
        queue = m.get("queue")
        if queue:
            w.add("bolo_model_queue_depth", "gauge", "Jobs waiting for this model.", queue["depth"], **labels)
            w.add("bolo_model_inflight", "gauge", "Jobs being decoded now.", queue["inflight"], **labels)
            w.add("bolo_model_batches_total", "counter", "Engine calls made.", queue["batches"], **labels)
            w.add("bolo_model_mean_batch_size", "gauge", "Mean jobs per engine call.", queue["mean_batch"], **labels)
            w.add(
                "bolo_model_queue_rejected_total",
                "counter",
                "Jobs rejected because the queue was full.",
                queue["rejected"],
                **labels,
            )
            for q, v in (queue["queue_ms"] or {}).items():
                w.add(
                    "bolo_queue_time_ms",
                    "gauge",
                    "Rolling queue-time quantiles.",
                    v,
                    quantile=str(int(q[1:]) / 100),
                    **labels,
                )
    return "\n".join(w.lines) + "\n"
