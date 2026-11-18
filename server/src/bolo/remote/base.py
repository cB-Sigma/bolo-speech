"""What every provider adapter shares: the HTTP client, error mapping, byte and latency accounting, audio decoding.

An adapter turns bolo's request fields into one provider's and the provider's answer back into bolo's types. It
runs in a worker thread of the model's queue (``httpx.Client``, blocking), like every other engine.

Errors are mapped once, here:

=========================  ===============================================================================
provider answered          bolo raises
=========================  ===============================================================================
400, 413, 415, 422         :class:`InvalidRequest` (400) with the provider's message
401, 403                   :class:`ProviderAuthError` (502, ``provider_auth``)
404                        :class:`ProviderError` (502): the provider does not serve that model or voice
408, 504, a timeout        :class:`ProviderTimeout` (504, ``provider_timeout``)
429                        :class:`Overloaded` (429) with the provider's ``Retry-After``
other 5xx, broken link     :class:`ProviderError` (502, ``provider_error``)
=========================  ===============================================================================

Everything but the 400s lets a model's ``fallback`` list take over. The API key is removed from every message.
"""

from __future__ import annotations

import abc
import contextlib
import io
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from bolo.audio.resample import StreamResampler, resample
from bolo.engines.base import FloatArray, STTResult, Voice
from bolo.errors import BoloError, InvalidRequest, Overloaded, ProviderAuthError, ProviderError, ProviderTimeout
from bolo.providers import ProviderConfig, Usage, scrub

#: shortest piece of audio handed on while streaming (seconds); smaller network reads are joined
MIN_CHUNK_S = 0.08

# ISO 639-1 -> English name, both ways (verbose_json answers with names)
LANGUAGE_CODES = {
    "english": "en", "hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te", "marathi": "mr",
    "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa", "urdu": "ur", "nepali": "ne",
    "odia": "or", "assamese": "as", "sanskrit": "sa", "sindhi": "sd", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru", "ukrainian": "uk",
    "polish": "pl", "turkish": "tr", "arabic": "ar", "persian": "fa", "hebrew": "he", "chinese": "zh",
    "cantonese": "yue", "japanese": "ja", "korean": "ko", "vietnamese": "vi", "thai": "th", "indonesian": "id",
    "malay": "ms", "swedish": "sv", "danish": "da", "norwegian": "no", "finnish": "fi", "czech": "cs",
    "greek": "el", "hungarian": "hu", "romanian": "ro", "catalan": "ca", "swahili": "sw",
}  # fmt: skip


def base_language(language: str | None) -> str | None:
    """``hi-IN`` / ``hi_IN`` / ``HI`` -> ``hi``."""
    if not language:
        return None
    return language.replace("_", "-").split("-")[0].lower() or None


def pcm16_to_float(data: bytes) -> FloatArray:
    out: FloatArray = (np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0).astype(np.float32)
    return out


def float_to_pcm16(samples: FloatArray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def decode_whole(data: bytes, out_rate: int, *, raw_rate: int | None = None) -> FloatArray:
    """A complete audio answer (WAV, or anything bolo's decoder reads; raw PCM16 with ``raw_rate``) at ``out_rate``."""
    if raw_rate is not None and data[:4] != b"RIFF":
        return resample(pcm16_to_float(data[: len(data) - len(data) % 2]), raw_rate, out_rate)
    if data[:4] == b"RIFF":
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(data)) as f:
            x = f.read(dtype="float32", always_2d=True).mean(axis=1).astype(np.float32)
            rate = int(f.samplerate)
    else:
        from bolo.audio.decode import decode_audio

        x, rate = decode_audio(data)
    return resample(x, rate, out_rate)


def split_seconds(x: FloatArray, rate: int) -> Iterator[FloatArray]:
    """One second per piece, so encoders and sockets downstream can start early."""
    for i in range(0, len(x), rate):
        yield x[i : i + rate]


def read_pcm_stream(
    chunks: Iterator[bytes], rate: int, out_rate: int, cancelled: Callable[[], bool] | None
) -> Iterator[FloatArray]:
    """Raw PCM16 mono arriving in arbitrary network reads -> float32 pieces at ``out_rate``."""
    resampler = StreamResampler(rate, out_rate)
    pending = b""
    min_bytes = int(rate * MIN_CHUNK_S) * 2
    for data in chunks:
        pending += data
        if len(pending) < min_bytes:
            continue
        usable = len(pending) - len(pending) % 2
        chunk, pending = pending[:usable], pending[usable:]
        out = resampler.process(pcm16_to_float(chunk))
        if len(out):
            yield out
        if cancelled is not None and cancelled():
            return
    usable = len(pending) - len(pending) % 2
    tail = resampler.process(pcm16_to_float(pending[:usable]), last=True)
    if len(tail):
        yield tail


def error_message(body: bytes, status: int) -> str:
    """The human-readable part of a provider's error body (the shapes of the families bolo talks to)."""
    with contextlib.suppress(ValueError, AttributeError, TypeError):
        doc = json.loads(body)
        if isinstance(doc, dict):
            err = doc.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str) and err:
                return err
            detail = doc.get("detail")
            if isinstance(detail, dict) and detail.get("message"):
                return str(detail["message"])
            if isinstance(detail, str) and detail:
                return detail
            if isinstance(detail, list) and detail:
                first = detail[0]
                return str(first.get("msg") or first) if isinstance(first, dict) else str(first)
            for key in ("err_msg", "message", "reason"):
                if doc.get(key):
                    return str(doc[key])
    text = body.decode(errors="replace").strip()
    return text[:300] or f"HTTP {status}"


@dataclass
class RemoteRequest:
    """One utterance for a provider: bolo's :class:`STTRequest` fields, the audio already encoded."""

    audio: bytes  # a WAV (PCM16 mono) or FLAC file
    content_type: str
    seconds: float
    language: str | None = None
    task: str = "transcribe"
    prompt: str | None = None
    word_timestamps: bool = False
    temperature: float = 0.0
    hotwords: str | None = None
    #: label speakers at the provider (only sent to adapters with ``supports_diarization``)
    diarize: bool = False

    @property
    def filename(self) -> str:
        return "audio.flac" if self.content_type == "audio/flac" else "audio.wav"

    def hotword_list(self) -> list[str]:
        """``hotwords`` as terms: split at commas and line breaks (a single term may hold spaces)."""
        if not self.hotwords:
            return []
        return [t.strip() for t in self.hotwords.replace("\n", ",").split(",") if t.strip()]


@dataclass
class LiveEvent:
    #: "interim" (text so far of the open segment), "final" (a finished segment), "endpoint" (the speaker
    #: stopped) or "error"
    kind: str
    text: str = ""
    words: list[tuple[str, float, float]] | None = None


class LiveProtocol(abc.ABC):
    """A provider's streaming speech-to-text WebSocket, reduced to what :class:`bolo.engines.remote.RemoteStream`
    needs. Audio goes in as PCM16 mono at ``sample_rate``."""

    url: str = ""
    headers: dict[str, str]
    sample_rate: int = 16000
    #: seconds without audio after which ``keepalive`` is sent (None: the provider needs none)
    keepalive_s: float | None = None

    @abc.abstractmethod
    def audio(self, pcm: bytes) -> bytes | str:
        """The frame to send for a piece of audio (bytes: a binary frame; str: a text frame)."""

    def keepalive(self) -> str | None:
        return None

    def start(self) -> str | None:
        """The first frame after connecting (a provider's start or configuration message), if it needs one."""
        return None

    def finalize(self) -> str | None:
        """Ask for the text of everything sent so far (end of input)."""
        return None

    def close(self) -> str | None:
        return None

    @abc.abstractmethod
    def parse(self, message: bytes | str) -> list[LiveEvent]: ...


class Adapter(abc.ABC):
    kind: str = ""
    #: longest utterance sent in one request (seconds); bolo cuts longer audio at pauses
    max_chunk_s: float = 600.0
    supports_translate: bool = False
    supports_word_timestamps: bool = False
    #: the provider labels speakers itself (a request's diarization is then left to it)
    supports_diarization: bool = False

    def __init__(self, provider: ProviderConfig, model: str, spec: dict[str, Any], *, timeout_s: float = 60.0) -> None:
        self.provider = provider
        self.model = model
        self.spec = spec
        self.base_url = provider.base_url.rstrip("/")
        self.lenient = bool(spec.get("ignore_unsupported", provider.options.get("ignore_unsupported", False)))
        timeout = float(spec.get("timeout_s") or provider.options.get("timeout_s") or timeout_s)
        self.client = httpx.Client(
            headers={**self.auth_headers(), "User-Agent": "bolo"}, timeout=httpx.Timeout(timeout, connect=10.0)
        )

    # -- identity ---------------------------------------------------------------------------------------------
    @property
    def where(self) -> str:
        return f"provider '{self.provider.name}' ({self.kind})"

    @abc.abstractmethod
    def auth_headers(self) -> dict[str, str]: ...

    def close(self) -> None:
        self.client.close()

    def new_usage(self) -> Usage:
        return Usage(self.provider.name, self.kind, self.model)

    # -- options a provider does not have -----------------------------------------------------------------
    def unsupported(self, param: str, what: str) -> None:
        """A request option this provider (or this model) has no equivalent for: a clear 400, unless the model
        was told to drop such options (``ignore_unsupported``)."""
        if self.lenient:
            return
        raise InvalidRequest(
            f"{what} is not supported by '{self.model}' at {self.where}; leave it out (or set the model option "
            "ignore_unsupported to have bolo drop it)",
            param=param,
        )

    # -- HTTP -------------------------------------------------------------------------------------------------
    def _clean(self, text: str) -> str:
        return scrub(text, self.provider.api_key)

    def transport_error(self, exc: Exception) -> BoloError:
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeout(f"{self.where} did not answer in time ({type(exc).__name__})")
        return ProviderError(self._clean(f"{self.where} is not reachable ({type(exc).__name__}: {exc})"))

    def http_error(self, status: int, body: bytes, headers: httpx.Headers | None = None) -> BoloError:
        message = self._clean(error_message(body, status))
        if status in (400, 413, 415, 422):
            return InvalidRequest(f"{self.where} refused the request: {message}")
        if status in (401, 403):
            return ProviderAuthError(f"{self.where} refused bolo's API key ({status}): {message}")
        if status == 404:
            return ProviderError(f"{self.where} does not know '{self.model}' or the address is wrong (404): {message}")
        if status in (408, 504):
            return ProviderTimeout(f"{self.where} timed out ({status}): {message}")
        if status == 429:
            retry = 1.0
            with contextlib.suppress(ValueError, TypeError):
                retry = float((headers.get("retry-after") if headers is not None else None) or 1.0)
            return Overloaded(f"{self.where} is rate-limiting or out of quota (429): {message}", retry_after_s=retry)
        return ProviderError(f"{self.where} failed ({status}): {message}")

    @staticmethod
    def _sent(request: httpx.Request) -> int:
        with contextlib.suppress(ValueError, TypeError):
            return int(request.headers.get("content-length") or 0)
        return 0

    def request(self, method: str, url: str, usage: Usage | None = None, **kwargs: Any) -> httpx.Response:
        """One complete request. Raises the mapped error for a failure; counts bytes and time into ``usage``."""
        t0 = time.perf_counter()
        try:
            resp = self.client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise self.transport_error(exc) from exc
        if usage is not None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            usage.requests += 1
            usage.latency_ms += elapsed
            if usage.first_byte_ms is None:
                usage.first_byte_ms = elapsed
            usage.bytes_sent += self._sent(resp.request)
            usage.bytes_received += len(resp.content)
        if resp.status_code >= 400:
            raise self.http_error(resp.status_code, resp.content, resp.headers)
        return resp

    @contextlib.contextmanager
    def stream(self, method: str, url: str, usage: Usage | None = None, **kwargs: Any) -> Iterator[httpx.Response]:
        """A request whose answer is read as it arrives (``iter_bytes`` / ``iter_lines``)."""
        t0 = time.perf_counter()
        try:
            with self.client.stream(method, url, **kwargs) as resp:
                if usage is not None:
                    usage.requests += 1
                    if usage.first_byte_ms is None:
                        usage.first_byte_ms = (time.perf_counter() - t0) * 1000.0
                    usage.bytes_sent += self._sent(resp.request)
                try:
                    if resp.status_code >= 400:
                        raise self.http_error(resp.status_code, resp.read(), resp.headers)
                    yield resp
                finally:
                    if usage is not None:
                        usage.latency_ms += (time.perf_counter() - t0) * 1000.0
                        usage.bytes_received += resp.num_bytes_downloaded
        except httpx.HTTPError as exc:
            raise self.transport_error(exc) from exc

    def json(self, resp: httpx.Response) -> dict[str, Any]:
        try:
            doc = resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.where} answered with something that is not JSON") from exc
        if not isinstance(doc, dict):
            raise ProviderError(f"{self.where} answered with an unexpected JSON value")
        return doc

    # -- what an adapter can do (each overrides what its provider offers) -------------------------------------
    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        raise InvalidRequest(f"{self.where} has no speech-to-text API in bolo", param="model")

    def synthesize(
        self,
        text: str,
        voice: str | None,
        speed: float,
        language: str | None,
        out_rate: int,
        usage: Usage,
        cancelled: Callable[[], bool] | None = None,
        instructions: str | None = None,
    ) -> Iterator[FloatArray]:
        raise InvalidRequest(f"{self.where} has no text-to-speech API in bolo", param="model")

    def isolate(self, samples: FloatArray, rate: int, usage: Usage) -> FloatArray:
        """The voice of a recording with the background removed (speech enhancement), at ``rate``."""
        raise InvalidRequest(f"{self.where} has no voice-isolation API in bolo", param="denoise")

    def voices(self) -> list[Voice]:
        """Voices the account can use, when the provider lists them (else none: the catalog's are used)."""
        return []

    def live(self, language: str | None) -> LiveProtocol | None:
        """The provider's streaming speech-to-text socket for this model, if it has one."""
        return None

    @abc.abstractmethod
    def check(self) -> dict[str, Any]:
        """A cheap request that proves the address answers (and, where possible, that the key is accepted)."""
