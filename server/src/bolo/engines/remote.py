"""The ``remote`` engine: speech-to-text and text-to-speech forwarded to a cloud API.

A catalog entry with ``engine: remote`` names a provider (``options.provider``, an entry of the ``providers``
setting) and the provider's model (``options.model``). The provider's ``kind`` picks the adapter
(:mod:`bolo.remote`): ``openai-compatible``, ``sarvam``, ``elevenlabs``, ``deepgram`` or ``soniox``. Nothing is
downloaded: such a model counts as installed as soon as its provider is configured.

To the rest of bolo these are ordinary models. Recognition gets 16 kHz audio (bolo decoded, resampled and, for
long recordings or diarization, cut it first) and sends each piece as a WAV file; synthesis gets the prepared
text sentence by sentence and returns float audio at the catalog's sample rate, resampled from whatever the
provider sent. So diarization, word lists and text normalisation, telephony codecs, the Realtime cascade,
traces and fallback all work as they do for local models.

Every request to a provider is counted (:class:`bolo.providers.Usage`): time, bytes both ways, seconds of
audio, characters, and an estimated cost when the provider's ``prices`` table knows the model.

Model options shared by every adapter (each adapter documents its own): ``provider``, ``provider_kind``,
``model``, ``upload_format`` (``wav`` or ``flac``), ``max_chunk_s``, ``timeout_s``, ``ignore_unsupported`` (drop
request options the provider lacks instead of answering 400), ``list_voices`` (read the account's voices when
loading), ``default_voice``, ``live`` (use the provider's streaming socket), ``workers`` (parallel requests;
default 4), ``request`` (extra provider fields).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from bolo.audio.encode import encode_audio
from bolo.engines.base import (
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    StreamResult,
    STTModel,
    STTRequest,
    STTResult,
    STTStream,
    SynthesisExtras,
    TTSModel,
    Voice,
    Word,
)
from bolo.errors import BoloError, EngineUnavailable, ProviderError
from bolo.providers import ProviderConfig, ProviderRegistry, Usage, scrub
from bolo.remote import Adapter, LiveProtocol, RemoteRequest, make_adapter
from bolo.remote.base import float_to_pcm16

if TYPE_CHECKING:
    from bolo.config import Settings

log = logging.getLogger("bolo.remote")

ENGINE_ID = "remote"
#: a live socket nobody has sent audio to for this long is closed (it reconnects on the next audio)
IDLE_CLOSE_S = 20.0
#: longest wait for the provider's last words after the end of input
SETTLE_S = 3.0


class RemoteSTT(STTModel):
    max_batch = 1

    def __init__(self, registry: ProviderRegistry, adapter: Adapter, name: str, spec: dict[str, Any]) -> None:
        self.registry = registry
        self.adapter = adapter
        self.name = name
        self.spec = spec
        self.sample_rate = 16000
        self.max_chunk_s = float(spec.get("max_chunk_s") or adapter.max_chunk_s)
        self.supports_translate = adapter.supports_translate
        self.supports_word_timestamps = adapter.supports_word_timestamps
        #: the provider labels speakers itself: a request's diarization is left to it (see Bolo._transcribe_once)
        self.native_diarization = adapter.supports_diarization and spec.get("provider_diarization") is not False
        self.upload_format = "flac" if str(spec.get("upload_format") or "wav") == "flac" else "wav"
        try:
            self.streaming = adapter.live(None) is not None
        except BoloError:
            self.streaming = False

    @property
    def provider(self) -> ProviderConfig:
        return self.adapter.provider

    def close(self) -> None:
        self.adapter.close()

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            seconds = len(r.samples) / self.sample_rate
            usage = self.adapter.new_usage()
            usage.audio_seconds = seconds
            req = RemoteRequest(
                encode_audio(r.samples, self.sample_rate, self.upload_format),
                f"audio/{self.upload_format}",
                seconds,
                r.language,
                r.task,
                r.prompt,
                r.word_timestamps,
                r.temperature,
                r.hotwords,
                r.diarize and self.native_diarization,
            )
            try:
                result = self.adapter.transcribe(req, usage)
            except BoloError as exc:
                self.registry.record(usage, error=exc.message)
                raise
            usage.cost = self.provider.estimate(self.adapter.model, audio_seconds=seconds, requests=usage.requests)
            self.registry.record(usage)
            result.usage = usage
            out.append(result)
        return out

    def create_stream(self, language: str | None = None) -> STTStream:
        protocol = self.adapter.live(language)
        if protocol is None:
            raise NotImplementedError("this model has no streaming socket")
        return RemoteStream(self, protocol)

    def decode_streams(self, streams: list[STTStream]) -> None:
        for s in streams:
            if isinstance(s, RemoteStream):
                s.settle()


class RemoteStream(STTStream):
    """One live recognition socket at a provider. A writer thread owns the connection and sends what
    :meth:`accept` queues; a reader thread turns the provider's messages into text."""

    def __init__(self, model: RemoteSTT, protocol: LiveProtocol) -> None:
        self.model = model
        self.protocol = protocol
        self._lock = threading.Lock()
        self._out: queue.Queue[bytes | str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._finals: list[str] = []
        self._words: list[Word] = []
        self._interim = ""
        self._endpoint = False
        self._error: str | None = None
        self._finishing = False
        self._settled = threading.Event()
        self._discard = False  # the provider's open segment belongs to an utterance bolo already closed
        self._sent = 0  # samples sent on this connection
        self._base = 0  # where the current utterance began, in samples of this connection
        self._usage = model.adapter.new_usage()

    # -- the connection -----------------------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._out = queue.Queue()
        self._sent = self._base = 0
        self._usage = self.model.adapter.new_usage()
        self._thread = threading.Thread(target=self._run, args=(self._out,), name="bolo-remote-live", daemon=True)
        self._thread.start()

    def _fail(self, message: str) -> None:
        with self._lock:
            self._error = scrub(message, self.model.provider.api_key)
        self._settled.set()

    def _run(self, out: queue.Queue[bytes | str | None]) -> None:
        from websockets.sync.client import connect

        where = self.model.adapter.where
        t0 = time.perf_counter()
        try:
            ws = connect(self.protocol.url, additional_headers=self.protocol.headers, open_timeout=10, max_size=None)
        except Exception as exc:
            self._fail(f"{where}: the streaming socket could not be opened ({type(exc).__name__}: {exc})")
            self.model.registry.record(self._usage, error="connect failed")
            return
        self._usage.requests = 1
        self._usage.first_byte_ms = (time.perf_counter() - t0) * 1000.0
        try:
            hello = self.protocol.start()
            if hello is not None:
                ws.send(hello)
        except Exception as exc:
            self._fail(f"{where}: the streaming socket refused the start message ({type(exc).__name__}: {exc})")
            self.model.registry.record(self._usage, error="start failed")
            with contextlib.suppress(Exception):
                ws.close()
            return
        reader = threading.Thread(target=self._read, args=(ws,), name="bolo-remote-live-read", daemon=True)
        reader.start()
        last = last_ping = time.monotonic()
        error: str | None = None
        try:
            while reader.is_alive():
                try:
                    frame = out.get(timeout=0.5)
                except queue.Empty:
                    now = time.monotonic()
                    if now - last > IDLE_CLOSE_S:
                        break
                    keep = self.protocol.keepalive_s
                    if keep is not None and now - max(last, last_ping) > keep:
                        ping = self.protocol.keepalive()
                        if ping:
                            ws.send(ping)
                        last_ping = now
                    continue
                if frame is None:
                    break
                ws.send(frame)
                self._usage.bytes_sent += len(frame)
                last = time.monotonic()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._fail(f"{where}: the streaming socket broke ({error})")
        finally:
            with contextlib.suppress(Exception):
                bye = self.protocol.close()
                if bye is not None:
                    ws.send(bye)
            with contextlib.suppress(Exception):
                ws.close()
            self._usage.latency_ms = (time.perf_counter() - t0) * 1000.0
            self._usage.audio_seconds = self._sent / self.protocol.sample_rate
            self._usage.cost = self.model.provider.estimate(
                self.model.adapter.model, audio_seconds=self._usage.audio_seconds
            )
            self.model.registry.record(self._usage, error=error)
            self._settled.set()

    def _read(self, ws: Any) -> None:
        try:
            for message in ws:
                self._usage.bytes_received += len(message)
                for event in self.protocol.parse(message):
                    self._on_event(event.kind, event.text, event.words)
        except Exception as exc:  # the socket closed or broke; the writer reports what matters
            log.debug("live socket of %s ended: %s", self.model.name, exc)
        finally:
            self._settled.set()

    def _on_event(self, kind: str, text: str, words: list[tuple[str, float, float]] | None) -> None:
        with self._lock:
            if kind == "error":
                self._error = f"{self.model.adapter.where}: {text}"
            elif self._discard:
                if kind in ("final", "endpoint"):
                    self._discard = False
                    self._interim = ""
            elif kind == "interim":
                self._interim = text
            elif kind == "final":
                self._finals.append(text)
                self._interim = ""
                base = self._base / self.protocol.sample_rate
                self._words.extend(Word(w, max(0.0, a - base), max(0.0, b - base)) for w, a, b in words or [])
            elif kind == "endpoint":
                self._endpoint = True
        if kind in ("final", "endpoint", "error"):
            self._settled.set()

    # -- STTStream ----------------------------------------------------------------------------------------
    def _raise_if_failed(self) -> None:
        with self._lock:
            error, self._error = self._error, None
        if error:
            raise ProviderError(error)

    def accept(self, samples: FloatArray) -> None:
        self._raise_if_failed()
        if not len(samples):
            return
        self._ensure()
        self._sent += len(samples)
        self._out.put(self.protocol.audio(float_to_pcm16(np.asarray(samples, dtype=np.float32))))

    def finish(self) -> None:
        self._finishing = True
        self._settled.clear()
        if self._thread is not None and self._thread.is_alive():
            frame = self.protocol.finalize()
            if frame:
                self._out.put(frame)

    def settle(self) -> None:
        """After the end of input: wait (briefly) for the provider's last words."""
        if self._finishing:
            self._finishing = False
            if self._thread is not None and self._thread.is_alive():
                self._settled.wait(SETTLE_S)
        self._raise_if_failed()

    def result(self) -> StreamResult:
        with self._lock:
            text = " ".join(p for p in [*self._finals, self._interim] if p).strip()
            return StreamResult(text, list(self._words))

    def is_endpoint(self) -> bool:
        return self._endpoint

    def reset(self) -> None:
        with self._lock:
            self._discard = bool(self._interim)
            self._finals, self._words, self._interim = [], [], ""
            self._endpoint = False
            self._base = self._sent

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._out.put(None)

    def __del__(self) -> None:  # a session that went away without closing its stream
        with contextlib.suppress(Exception):
            self.close()


class RemoteTTS(TTSModel):
    accepts_extras = True

    def __init__(self, registry: ProviderRegistry, adapter: Adapter, name: str, spec: dict[str, Any]) -> None:
        self.registry = registry
        self.adapter = adapter
        self.name = name
        self.spec = spec
        self.sample_rate = int(spec.get("sample_rate") or 24000)
        voices: list[Voice] = []
        self._voice_info: dict[str, dict[str, Any]] = {}
        for i, v in enumerate(spec.get("voices") or []):
            vid = str(v["id"])
            voices.append(Voice(vid, str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i))
            self._voice_info[vid.lower()] = dict(v)
        self.voices = tuple(voices)

    @property
    def provider(self) -> ProviderConfig:
        return self.adapter.provider

    def connect(self) -> None:
        """Read the account's voices where the provider lists them (``list_voices``); never fatal."""
        if not self.spec.get("list_voices"):
            return
        known = {v.id.lower() for v in self.voices}
        extra = list(self.voices)
        with contextlib.suppress(BoloError):
            for v in self.adapter.voices():
                if v.id.lower() not in known:
                    extra.append(Voice(v.id, v.name, v.language, v.gender, len(extra)))
                    known.add(v.id.lower())
        self.voices = tuple(extra)

    def close(self) -> None:
        self.adapter.close()

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        extras: SynthesisExtras | None = None,
    ) -> Iterator[FloatArray]:
        info = self._voice_info.get((voice or "").lower(), {})
        send_voice = str(info["provider_voice"]) if info.get("provider_voice") else voice
        language = (extras.language if extras else None) or str(info.get("language") or "") or None
        usage = self.adapter.new_usage()
        usage.chars = len(text)
        samples = 0
        error: str | None = None
        try:
            for piece in self.adapter.synthesize(text, send_voice, speed, language, self.sample_rate, usage, cancelled):
                samples += len(piece)
                yield piece
        except BoloError as exc:
            error = exc.message
            raise
        except httpx.HTTPError as exc:  # the connection broke while the audio was arriving
            mapped = self.adapter.transport_error(exc)
            error = mapped.message
            raise mapped from exc
        finally:
            usage.audio_seconds = samples / self.sample_rate
            usage.cost = self.provider.estimate(
                self.adapter.model, audio_seconds=usage.audio_seconds, chars=usage.chars, requests=usage.requests
            )
            self.registry.record(usage, error=error)
            sink = extras.usage if extras is not None else None
            if isinstance(sink, Usage):
                sink.add(usage)


class RemoteEngine(Engine):
    id = ENGINE_ID
    name = "Cloud speech APIs (OpenAI-compatible, Sarvam, ElevenLabs, Deepgram, Soniox)"
    license = "Apache-2.0 (bolo's client); each provider's terms apply to its API"
    url = "https://github.com/cb-Sigma/bolo/blob/main/docs/providers.md"
    kinds = ("remote-stt", "remote-tts")

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.registry: ProviderRegistry | None = None

    def configure(self, settings: Settings) -> None:
        self.settings = settings

    def attach(self, registry: ProviderRegistry) -> None:
        self.registry = registry

    def _registry(self) -> ProviderRegistry:
        if self.registry is None:
            if self.settings is None:
                from bolo.config import Settings

                self.settings = Settings()
            self.registry = ProviderRegistry(self.settings)
        return self.registry

    def status(self) -> EngineStatus:
        # the engine is an HTTP client; whether a model's provider is configured is a per-model question
        return EngineStatus(self.id, True, f"httpx {httpx.__version__}", None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        registry = self._registry()
        name = str(spec.get("model_id") or path.name)
        if task not in ("stt", "tts"):
            raise EngineUnavailable(f"the remote engine serves speech-to-text and text-to-speech, not {task!r}")
        provider = registry.require(
            str(spec.get("provider") or "") or None, str(spec.get("provider_kind") or "") or None, model=name
        )
        remote_model = str(spec.get("model") or "")
        if not remote_model:
            raise EngineUnavailable(f"model '{name}' does not say which model of '{provider.name}' it is")
        timeout = float(registry.settings.provider_timeout_s)
        adapter = make_adapter(provider, remote_model, spec, timeout_s=timeout)
        if task == "stt":
            return RemoteSTT(registry, adapter, name, spec)
        model = RemoteTTS(registry, adapter, name, spec)
        model.connect()
        return model
