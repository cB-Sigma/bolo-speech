"""bolo's side of out-of-process engine plugins.

A plugin engine appears in the registry as ``plugin:<name>`` and is used by catalog entries with
``engine: plugin:<name>``. Its models implement the ordinary :class:`STTModel` / :class:`TTSModel`
interfaces, so the scheduler, queues, traces and metrics treat them like built-in models; every
call is forwarded to the plugin process (see :mod:`bolo.plugin.protocol`).

Process lifecycle (:class:`PluginProcess`):

* started on the first ``load`` (not when bolo starts), with ``hello`` to agree on the protocol;
* a crash fails the requests in flight with ``engine_unavailable``; the next call restarts the
  process after a back-off (1 s, doubling up to 60 s while crashes keep coming) and loads the
  models again transparently;
* stopped when no model has been loaded on it for ``plugin_idle_s`` seconds;
* optional address-space limit (``plugin_memory_limit_mb``, POSIX ``RLIMIT_AS``) and lower
  scheduling priority (``plugin_nice``).
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bolo.engines.base import (
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    Segment,
    SeparationModel,
    SpeakerRequest,
    STTModel,
    STTRequest,
    STTResult,
    TTSModel,
    Voice,
    Word,
)
from bolo.errors import EngineUnavailable, GatedModel, InvalidRequest
from bolo.plugin.protocol import (
    PROTOCOL_VERSION,
    audio_bytes,
    audio_from_bytes,
    read_message,
    speaker_payload,
    write_message,
)

log = logging.getLogger("bolo.plugin")

ENTRY_POINT_GROUP = "bolo.engines"
PREFIX = "plugin:"


@dataclass
class PluginConfig:
    name: str
    command: list[str]
    source: str = "settings"  # "settings" | "entry-point"
    idle_s: float = 300.0
    memory_limit_mb: int = 0
    nice: int = 0
    request_timeout_s: float = 600.0
    env: dict[str, str] = field(default_factory=dict)


def parse_plugin_setting(item: str) -> tuple[str, list[str]]:
    """``"name=command args"`` → ``(name, argv)``."""
    name, sep, command = item.partition("=")
    if not sep or not name.strip() or not command.strip():
        raise ValueError(f"a plugin is configured as 'name=command', got {item!r}")
    return name.strip(), shlex.split(command)


def discover_entry_points() -> dict[str, list[str]]:
    """Plugins installed as Python packages: entry points in the group ``bolo.engines``.

    The entry point names a module (``bolo_torch.plugin``), started as ``python -m <module>`` with
    bolo's own interpreter, or ``module:function`` returning the command line to start.
    """
    from importlib.metadata import entry_points

    found: dict[str, list[str]] = {}
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - broken metadata of another package
        return found
    for ep in eps:
        try:
            found[ep.name] = list(ep.load()()) if ep.attr else [sys.executable, "-m", ep.module]
        except Exception as exc:
            log.warning("plugin entry point %s could not be read: %s", ep.name, exc)
    return found


class _Call:
    """A request in flight: its final reply, or a stream of chunks then the final reply."""

    def __init__(self) -> None:
        self.items: queue.Queue[tuple[dict[str, Any], bytes] | BaseException] = queue.Queue()


class PluginProcess:
    def __init__(self, config: PluginConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._calls: dict[str, _Call] = {}
        self._ids = itertools.count(1)
        self.generation = 0  # increases with every (re)start; models reload when it changes
        self.crashes = 0
        self.restarts = 0
        self._not_before = 0.0
        self._backoff = 1.0
        self.hello: dict[str, Any] = {}
        self.last_error: str | None = None
        self.handles: set[str] = set()
        self._last_active = time.monotonic()
        self._idle_timer: threading.Timer | None = None

    # -- process ------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _preexec(self) -> None:  # pragma: no cover - runs in the child
        if self.config.nice:
            os.nice(self.config.nice)
        if self.config.memory_limit_mb:
            import resource

            limit = self.config.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def ensure_started(self) -> None:
        with self._lock:
            if self.running:
                return
            wait = self._not_before - time.monotonic()
            if wait > 0:
                raise EngineUnavailable(
                    f"plugin '{self.config.name}' crashed ({self.last_error or 'exited'}); restarting in {wait:.0f} s"
                )
            env = {**os.environ, **self.config.env, "PYTHONUNBUFFERED": "1"}
            try:
                self._proc = subprocess.Popen(
                    self.config.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    preexec_fn=self._preexec if os.name == "posix" else None,  # noqa: PLW1509
                )
            except OSError as exc:
                self.last_error = str(exc)
                raise EngineUnavailable(f"plugin '{self.config.name}' could not be started: {exc}") from exc
            self.generation += 1
            if self.generation > 1:
                self.restarts += 1
            self.handles.clear()
            proc = self._proc
            threading.Thread(
                target=self._read, args=(proc,), name=f"bolo-plugin-{self.config.name}", daemon=True
            ).start()
            threading.Thread(target=self._stderr, args=(proc,), daemon=True).start()
        try:
            self.hello = self.request({"op": "hello", "protocol": PROTOCOL_VERSION}, timeout=60)
        except EngineUnavailable:
            self.stop()
            raise
        if int(self.hello.get("protocol", 0)) not in range(1, PROTOCOL_VERSION + 1):
            self.stop()
            raise EngineUnavailable(f"plugin '{self.config.name}' speaks protocol {self.hello.get('protocol')}")
        self._backoff = 1.0

    def _stderr(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            log.info("[plugin %s] %s", self.config.name, line.decode(errors="replace").rstrip())

    def _read(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        error: str | None = None
        try:
            while True:
                got = read_message(proc.stdout)
                if got is None:
                    break
                message, payload = got
                call = self._calls.get(str(message.get("id")))
                if call is not None:
                    call.items.put((message, payload))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        code = proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None
                stopping = getattr(proc, "_bolo_stopping", False)
                if not stopping:
                    self.crashes += 1
                    self.last_error = error or f"exit code {code}"
                    self._not_before = time.monotonic() + self._backoff
                    self._backoff = min(60.0, self._backoff * 2)
                    log.error("plugin %s stopped unexpectedly (%s)", self.config.name, self.last_error)
            failure = EngineUnavailable(f"plugin '{self.config.name}' stopped ({error or f'exit code {code}'})")
            for call in list(self._calls.values()):
                call.items.put(failure)

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            self.handles.clear()
        if proc is None:
            return
        proc._bolo_stopping = True  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            assert proc.stdin is not None
            with self._write_lock:
                write_message(proc.stdin, {"id": "0", "op": "shutdown"})
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # -- requests -------------------------------------------------------------
    def _send(self, message: dict[str, Any], payload: bytes | None = None) -> tuple[str, _Call]:
        rid = str(next(self._ids))
        call = _Call()
        self._calls[rid] = call
        proc = self._proc
        if proc is None or proc.stdin is None:
            self._calls.pop(rid, None)
            raise EngineUnavailable(f"plugin '{self.config.name}' is not running")
        try:
            with self._write_lock:
                write_message(proc.stdin, {**message, "id": rid}, payload)
        except OSError as exc:
            self._calls.pop(rid, None)
            raise EngineUnavailable(f"plugin '{self.config.name}' is not accepting requests: {exc}") from exc
        self._last_active = time.monotonic()
        return rid, call

    def _next(self, call: _Call, timeout: float | None) -> tuple[dict[str, Any], bytes]:
        try:
            item = call.items.get(timeout=timeout or self.config.request_timeout_s)
        except queue.Empty as exc:
            raise EngineUnavailable(f"plugin '{self.config.name}' did not answer in time") from exc
        if isinstance(item, BaseException):
            raise item
        return item

    @staticmethod
    def _check(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("ok") is False:
            err = message.get("error") or {}
            code, text = str(err.get("code", "error")), str(err.get("message", ""))
            # errors with a meaning to the client keep their status (a gated download is 401, not 503)
            if code == GatedModel.code:
                raise GatedModel(text)
            if code == InvalidRequest.code:
                raise InvalidRequest(text)
            raise EngineUnavailable(f"plugin error ({code}): {text}")
        return message

    def request(
        self, message: dict[str, Any], payload: bytes | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        rid, call = self._send(message, payload)
        try:
            reply, _ = self._next(call, timeout)
            return self._check(reply)
        finally:
            self._calls.pop(rid, None)

    def request_with_payload(
        self, message: dict[str, Any], payload: bytes | None = None, timeout: float | None = None
    ) -> tuple[dict[str, Any], bytes]:
        """Like :meth:`request`, for requests whose final reply carries audio (``separate``, ``extract``)."""
        rid, call = self._send(message, payload)
        try:
            reply, data = self._next(call, timeout)
            return self._check(reply), data
        finally:
            self._calls.pop(rid, None)

    def stream(
        self, message: dict[str, Any], cancelled: Callable[[], bool] | None = None
    ) -> Iterator[tuple[dict[str, Any], bytes]]:
        """Chunks of a streaming request; the final reply is yielded last."""
        rid, call = self._send(message)
        sent_cancel = False
        try:
            while True:
                try:
                    reply, payload = self._next(call, 0.2 if cancelled is not None else None)
                except EngineUnavailable as exc:
                    if cancelled is not None and "in time" in str(exc):
                        if cancelled() and not sent_cancel:
                            sent_cancel = True
                            with contextlib.suppress(EngineUnavailable):
                                self._send({"op": "cancel", "target": rid})
                        continue
                    raise
                if "chunk" in reply:
                    if cancelled is not None and cancelled() and not sent_cancel:
                        sent_cancel = True
                        with contextlib.suppress(EngineUnavailable):
                            self._send({"op": "cancel", "target": rid})
                    yield reply, payload
                    continue
                yield self._check(reply), payload
                return
        finally:
            self._calls.pop(rid, None)

    # -- idle shutdown ----------------------------------------------------------
    def model_unloaded(self, handle: str) -> None:
        self.handles.discard(handle)
        if not self.handles and self.config.idle_s >= 0:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(self.config.idle_s, self._idle_check)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _idle_check(self) -> None:
        if not self.handles and time.monotonic() - self._last_active >= self.config.idle_s - 0.05:
            log.info("stopping idle plugin %s", self.config.name)
            self.stop()

    def status(self) -> dict[str, Any]:
        return {
            "name": self.config.name, "source": self.config.source, "command": self.config.command,
            "running": self.running, "pid": self._proc.pid if self._proc else None, "generation": self.generation,
            "crashes": self.crashes, "restarts": self.restarts, "last_error": self.last_error,
            "loaded_models": len(self.handles), "protocol": self.hello.get("protocol"),
        }  # fmt: skip


class _PluginModelMixin:
    """Keeps a model handle valid across plugin restarts (it is loaded again when the generation changes)."""

    process: PluginProcess
    _load_message: dict[str, Any]
    _handle: str | None
    _generation: int

    def _ensure_loaded(self) -> str:
        proc = self.process
        proc.ensure_started()
        if self._handle is None or self._generation != proc.generation:
            reply = proc.request(self._load_message)
            self._handle = str(reply["handle"])
            self._generation = proc.generation
            proc.handles.add(self._handle)
            self._apply_load_reply(reply)
        return self._handle

    def _apply_load_reply(self, reply: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        if self._generation == self.process.generation and self.process.running:
            with contextlib.suppress(EngineUnavailable):
                self.process.request({"op": "unload", "handle": handle}, timeout=30)
        self.process.model_unloaded(handle)


class PluginSTT(_PluginModelMixin, STTModel):
    def __init__(self, process: PluginProcess, load_message: dict[str, Any]) -> None:
        self.process, self._load_message, self._handle, self._generation = process, load_message, None, 0
        self._ensure_loaded()

    def _apply_load_reply(self, reply: dict[str, Any]) -> None:
        info = reply.get("info") or {}
        self.sample_rate = int(reply.get("sample_rate") or 16000)
        self.max_batch = max(1, int(info.get("max_batch", 1)))
        self.languages = tuple(info.get("languages") or ())
        self.supports_translate = bool(info.get("supports_translate"))
        self.supports_word_timestamps = bool(info.get("supports_word_timestamps"))
        self.speaker_conditioned = bool(info.get("speaker_conditioned"))

    def transcribe_speakers(self, request: SpeakerRequest) -> list[STTResult]:
        if not self.speaker_conditioned:
            raise InvalidRequest("this plugin model is not conditioned on speaker diarization")
        handle = self._ensure_loaded()
        probs = request.activity.probs
        frames, speakers = (int(probs.shape[0]), int(probs.shape[1])) if probs.ndim == 2 else (0, 0)
        message = {
            "op": "speakers", "handle": handle, "n": len(request.samples), "language": request.language,
            "task": request.task, "prompt": request.prompt, "word_timestamps": request.word_timestamps,
            "temperature": request.temperature, "frame_s": request.activity.frame_s, "frames": frames,
            "speakers": speakers,
        }  # fmt: skip
        # one decode per speaker over the whole recording: allow far longer than a batch of short chunks
        seconds = len(request.samples) / (self.sample_rate or 16000)
        timeout = max(self.process.config.request_timeout_s, 30.0 * seconds * max(1, speakers))
        reply = self.process.request(message, speaker_payload(request.samples, probs), timeout=timeout)
        out = _results(reply)
        if len(out) != speakers:
            raise EngineUnavailable(
                f"plugin '{self.process.config.name}' returned {len(out)} results for {speakers} speakers"
            )
        return out

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        handle = self._ensure_loaded()
        meta = [
            {"n": len(r.samples), "language": r.language, "task": r.task, "prompt": r.prompt,
             "word_timestamps": r.word_timestamps, "temperature": r.temperature, "hotwords": r.hotwords}
            for r in requests
        ]  # fmt: skip
        import numpy as np

        audio = np.concatenate([r.samples for r in requests]) if requests else np.zeros(0, np.float32)
        reply = self.process.request({"op": "transcribe", "handle": handle, "requests": meta}, audio_bytes(audio))
        out = _results(reply)
        if len(out) != len(requests):
            raise EngineUnavailable(
                f"plugin '{self.process.config.name}' returned {len(out)} results for {len(requests)}"
            )
        return out


class PluginTTS(_PluginModelMixin, TTSModel):
    def __init__(self, process: PluginProcess, load_message: dict[str, Any]) -> None:
        self.process, self._load_message, self._handle, self._generation = process, load_message, None, 0
        self._ensure_loaded()

    def _apply_load_reply(self, reply: dict[str, Any]) -> None:
        info = reply.get("info") or {}
        self.sample_rate = int(reply.get("sample_rate") or 24000)
        self.voices = tuple(
            Voice(str(v["id"]), str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i)
            for i, v in enumerate(info.get("voices") or [])
        )

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        handle = self._ensure_loaded()
        message = {"op": "synthesize", "handle": handle, "text": text, "voice": voice, "speed": speed}
        for reply, payload in self.process.stream(message, cancelled):
            if "chunk" in reply and payload:
                yield audio_from_bytes(payload)


class PluginSeparation(_PluginModelMixin, SeparationModel):
    """A separation model in a plugin: ``info.mode`` "blind" (op ``separate``) or "target" (``enrol``,
    ``extract``, ``release``)."""

    def __init__(self, process: PluginProcess, load_message: dict[str, Any]) -> None:
        self.process, self._load_message, self._handle, self._generation = process, load_message, None, 0
        self._ensure_loaded()

    def _apply_load_reply(self, reply: dict[str, Any]) -> None:
        info = reply.get("info") or {}
        self.sample_rate = int(reply.get("sample_rate") or 16000)
        self.mode = str(info.get("mode") or "blind")
        self.min_sample_s = float(info.get("min_sample_s") or 3.0)

    def separate(self, mixture: FloatArray) -> list[FloatArray]:
        handle = self._ensure_loaded()
        reply, payload = self.process.request_with_payload({"op": "separate", "handle": handle}, audio_bytes(mixture))
        audio = audio_from_bytes(payload)
        count = int(reply.get("tracks", 0))
        n = len(audio) // count if count else 0
        return [audio[i * n : (i + 1) * n] for i in range(count)]

    def enrol(self, key: str, sample: FloatArray) -> None:
        handle = self._ensure_loaded()
        self.process.request({"op": "enrol", "handle": handle, "key": key}, audio_bytes(sample))

    def extract(self, key: str, mixture: FloatArray) -> FloatArray:
        handle = self._ensure_loaded()
        _, payload = self.process.request_with_payload(
            {"op": "extract", "handle": handle, "key": key}, audio_bytes(mixture)
        )
        return audio_from_bytes(payload)

    def release(self, key: str) -> None:
        if self._handle is None or self._generation != self.process.generation or not self.process.running:
            return
        with contextlib.suppress(EngineUnavailable):
            self.process.request({"op": "release", "handle": self._handle, "key": key}, timeout=30)


class PluginEngine(Engine):
    """The registry's face of one plugin (``plugin:<name>``)."""

    def __init__(self, config: PluginConfig) -> None:
        self.config = config
        self.id = PREFIX + config.name
        self.name = config.name
        self.process = PluginProcess(config)
        self.kinds = ()

    def status(self) -> EngineStatus:
        if not self.config.command:
            return EngineStatus(self.id, False, None, "no command configured")
        exe = self.config.command[0]
        if not (shutil.which(exe) or Path(exe).is_file()):
            return EngineStatus(self.id, False, None, f"plugin command not found: {exe}")
        version = self.process.hello.get("version")
        return EngineStatus(self.id, True, str(version) if version else "not started", None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        options = {k: v for k, v in spec.items() if _jsonable(v)}
        message = {
            "op": "load", "kind": kind, "task": task, "path": str(path), "options": options,
            "threads": opts.threads, "device": opts.device, "model": str(spec.get("model_id", path.name)),
        }  # fmt: skip
        if task == "stt":
            return PluginSTT(self.process, message)
        if task == "tts":
            return PluginTTS(self.process, message)
        if task == "separation":
            return PluginSeparation(self.process, message)
        raise EngineUnavailable(f"plugins can serve stt, tts and separation models, not {task!r}")

    def shutdown(self) -> None:
        self.process.stop()


def _results(reply: dict[str, Any]) -> list[STTResult]:
    out: list[STTResult] = []
    for res in reply.get("results", []):
        words = [_word(w) for w in res.get("words", [])]
        segments = [
            Segment(float(s["start"]), float(s["end"]), str(s["text"]), [_word(w) for w in s.get("words", [])])
            for s in res.get("segments", [])
        ]
        out.append(STTResult(str(res.get("text", "")), res.get("language"), segments, words))
    return out


def _word(w: dict[str, Any]) -> Word:
    return Word(str(w["w"]), float(w["t0"]), float(w["t1"]))


def _jsonable(value: Any) -> bool:
    import json

    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
