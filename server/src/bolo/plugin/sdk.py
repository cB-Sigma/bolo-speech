"""Plugin side of the protocol: serve a bolo :class:`~bolo.engines.base.Engine` over stdio.

A plugin package implements bolo's ordinary engine interface (``Engine.load`` returning an
``STTModel`` or ``TTSModel``) in its own process, with whatever libraries it needs, and ends its
``__main__`` with::

    from bolo.plugin.sdk import serve
    serve(MyEngine())

The plugin process needs bolo installed (for these classes); bolo's server never imports the
plugin's libraries. See docs/plugins.md.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any

import numpy as np

from bolo.engines.base import Engine, LoadOptions, Model, SeparationModel, STTModel, STTRequest, TTSModel
from bolo.plugin.protocol import PROTOCOL_VERSION, audio_bytes, audio_from_bytes, read_message, write_message


class _Server:
    def __init__(self, engine: Engine, stdin: IO[bytes], stdout: IO[bytes]) -> None:
        self.engine = engine
        self.stdin = stdin
        self.stdout = stdout
        self.write_lock = threading.Lock()
        self.models: dict[str, Model] = {}
        self.cancelled: set[str] = set()
        self.next_handle = 0
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("BOLO_PLUGIN_WORKERS", "4")))

    def send(self, message: dict[str, Any], payload: bytes | None = None) -> None:
        with self.write_lock:
            write_message(self.stdout, message, payload)

    def reply(self, rid: str, **members: Any) -> None:
        self.send({"id": rid, "ok": True, **members})

    def fail(self, rid: str, code: str, message: str) -> None:
        self.send({"id": rid, "ok": False, "error": {"code": code, "message": message}})

    def run(self) -> int:
        while True:
            got = read_message(self.stdin)
            if got is None:
                return 0
            message, payload = got
            rid = str(message.get("id", ""))
            op = message.get("op")
            if op == "hello":
                version = min(int(message.get("protocol", 1)), PROTOCOL_VERSION)
                self.reply(
                    rid, protocol=version, name=self.engine.name or self.engine.id,
                    version=self.engine.status().version, kinds=list(self.engine.kinds),
                )  # fmt: skip
            elif op == "health":
                self.reply(rid, rss_mb=_rss_mb(), models=len(self.models))
            elif op == "cancel":
                self.cancelled.add(str(message.get("target")))
                self.reply(rid)
            elif op == "shutdown":
                self.reply(rid)
                return 0
            else:
                self.pool.submit(self.handle, rid, op, message, payload)

    def handle(self, rid: str, op: Any, message: dict[str, Any], payload: bytes) -> None:
        try:
            if op == "load":
                self.load(rid, message)
            elif op == "unload":
                model = self.models.pop(str(message.get("handle")), None)
                if model is not None:
                    model.close()
                self.reply(rid)
            elif op == "transcribe":
                self.transcribe(rid, message, payload)
            elif op == "synthesize":
                self.synthesize(rid, message)
            elif op in ("separate", "enrol", "extract", "release"):
                self.separation(rid, op, message, payload)
            else:
                self.fail(rid, "unknown_op", f"unknown op {op!r}")
        except Exception as exc:
            self.fail(rid, type(exc).__name__, str(exc)[:2000])
        finally:
            self.cancelled.discard(rid)

    def load(self, rid: str, m: dict[str, Any]) -> None:
        opts = LoadOptions(threads=int(m.get("threads", 1)), device=str(m.get("device", "cpu")))
        model = self.engine.load(
            str(m["kind"]), str(m["task"]), Path(str(m["path"])), dict(m.get("options") or {}), opts
        )
        self.next_handle += 1
        handle = f"h{self.next_handle}"
        self.models[handle] = model
        info: dict[str, Any] = {}
        if isinstance(model, TTSModel):
            info["voices"] = [v.to_dict() for v in model.voices]
        if isinstance(model, SeparationModel):
            info.update(mode=model.mode, min_sample_s=model.min_sample_s)
        if isinstance(model, STTModel):
            info.update(
                languages=list(model.languages), max_batch=model.max_batch,
                supports_translate=model.supports_translate, supports_word_timestamps=model.supports_word_timestamps,
            )  # fmt: skip
        self.reply(rid, handle=handle, sample_rate=model.sample_rate, task=model.task, info=info)

    def model(self, m: dict[str, Any]) -> Model:
        model = self.models.get(str(m.get("handle")))
        if model is None:
            raise KeyError(f"no loaded model with handle {m.get('handle')!r}")
        return model

    def transcribe(self, rid: str, m: dict[str, Any], payload: bytes) -> None:
        model = self.model(m)
        if not isinstance(model, STTModel):
            raise TypeError("this model does not transcribe")
        audio = audio_from_bytes(payload)
        requests: list[STTRequest] = []
        offset = 0
        for r in m.get("requests", []):
            n = int(r["n"])
            requests.append(
                STTRequest(
                    audio[offset : offset + n], r.get("language"), r.get("task", "transcribe"), r.get("prompt"),
                    bool(r.get("word_timestamps")), float(r.get("temperature", 0.0)), r.get("hotwords"),
                )
            )  # fmt: skip
            offset += n
        results = model.transcribe(requests)
        out = [
            {
                "text": res.text,
                "language": res.language,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text, "words": [w.to_dict() for w in s.words]}
                    for s in res.segments
                ],
                "words": [w.to_dict() for w in res.words],
            }
            for res in results
        ]
        self.reply(rid, results=out)

    def separation(self, rid: str, op: str, m: dict[str, Any], payload: bytes) -> None:
        model = self.model(m)
        if not isinstance(model, SeparationModel):
            raise TypeError("this model does not separate speech")
        key = str(m.get("key", ""))
        if op == "separate":
            tracks = model.separate(audio_from_bytes(payload))
            n = max((len(t) for t in tracks), default=0)
            joined = [np.pad(t, (0, n - len(t))) for t in tracks]
            data = audio_bytes(np.concatenate(joined).astype(np.float32) if joined else np.zeros(0, np.float32))
            self.send({"id": rid, "ok": True, "tracks": len(tracks)}, data)
        elif op == "enrol":
            model.enrol(key, audio_from_bytes(payload))
            self.reply(rid)
        elif op == "extract":
            self.send({"id": rid, "ok": True}, audio_bytes(model.extract(key, audio_from_bytes(payload))))
        else:
            model.release(key)
            self.reply(rid)

    def synthesize(self, rid: str, m: dict[str, Any]) -> None:
        model = self.model(m)
        if not isinstance(model, TTSModel):
            raise TypeError("this model does not synthesise")
        seq = 0
        for chunk in model.synthesize(
            str(m.get("text", "")), m.get("voice"), float(m.get("speed", 1.0)), lambda: rid in self.cancelled
        ):
            if rid in self.cancelled:
                break
            self.send({"id": rid, "chunk": seq}, audio_bytes(chunk))
            seq += 1
        self.reply(rid, sample_rate=model.sample_rate, chunks=seq, cancelled=rid in self.cancelled)


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None


def serve(engine: Engine, stdin: IO[bytes] | None = None, stdout: IO[bytes] | None = None) -> int:
    """Run the plugin loop until bolo closes the pipe or asks for shutdown. Returns an exit code."""
    out = stdout or sys.stdout.buffer
    # Anything a library prints must not corrupt the protocol stream: send stray stdout to stderr.
    if stdout is None:
        sys.stdout = sys.stderr
    return _Server(engine, stdin or sys.stdin.buffer, out).run()
