"""Microsoft's MAI speech models in Azure: MAI-Transcribe-2-Streaming (speech-to-text) and MAI-Voice-2.1 /
MAI-Voice-2.1-Flash (text-to-speech).

Checked against Microsoft Learn on 2026-11-21 (azure/ai-services/speech-service: mai-transcribe-2-streaming,
mai-transcribe-2-streaming-realtime, mai-voices):

* **MAI-Transcribe-2-Streaming** (public preview, released 2026-10-01): ``wss://<resource>.services.ai.azure.com
  /mai/v1/realtime?intent=transcription`` with the key in an ``api-key`` header (or a Microsoft Entra token as
  ``Authorization: Bearer``; bolo uses the key). ``session.created`` on connect; then, before any audio,
  ``session.update`` with ``session.type: transcription`` and ``session.audio.input``: ``format`` (``audio/pcm`` at
  16000 or 24000), ``transcription.model`` (the *deployment* name in Foundry), ``transcription.language`` (null:
  detected; 60 languages), ``turn_detection: null`` and ``noise_reduction: null`` (the only values: there is no
  server-side speech detection; the client commits). Settings cannot change after the first audio. Audio as
  ``input_audio_buffer.append`` (base64 PCM16, 10 to 20 ms pieces for the lowest latency). Answers:
  ``conversation.item.input_audio_transcription.delta`` (newly finalised text, appended as it is),
  ``…intermediate`` (MAI's own: the provisional rest, replacing the previous one), ``input_audio_buffer.committed``
  and, after ``input_audio_buffer.commit``, ``…completed`` with the whole ``transcript`` since the previous commit;
  ``error`` and ``…failed`` for failures. A session lasts at most an hour.
* **MAI-Voice-2.1** and **MAI-Voice-2.1-Flash** (public preview): Azure Speech's text-to-speech REST route,
  ``POST https://<region>.tts.speech.microsoft.com/cognitiveservices/v1`` with ``Ocp-Apim-Subscription-Key``,
  ``Content-Type: application/ssml+xml`` and ``X-Microsoft-OutputFormat`` (Microsoft's example:
  ``audio-24khz-160kbitrate-mono-mp3``), the SSML naming the voice with the model as a suffix
  (``<voice name="en-US-Harper:MAI-Voice-2.1-Flash">``). 97 managed voices in 23 languages; every voice takes both
  models. Regions: francecentral, eastasia, southeastasia, eastus, canadacentral, eastus2, westus, westeurope,
  northeurope, westus2, westus3, centralindia, swedencentral, japaneast.

Provider settings: ``base_url`` is the Foundry resource (``https://<resource>.services.ai.azure.com``; there is no
default to fall back on), option ``region`` (or ``tts_url``) the Speech endpoint for the voices, option
``deployments`` maps a model to its deployment name (default: the model's own name).

Not verified (no account was used): whether one resource key serves both the realtime socket and the Speech route
(the pages describe a Foundry resource for each; give the voices their own provider of this kind if your keys
differ), the shape of ``error`` events, whether ``completed`` carries a language, and SSML ``prosody`` with MAI voices
(bolo does not send it: ``speed`` other than 1 answers 400).
"""

from __future__ import annotations

import base64
import contextlib
import json
import time
from collections.abc import Callable, Iterator
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from bolo.engines.base import FloatArray, STTResult
from bolo.errors import InvalidRequest, ProviderAuthError, ProviderError, ProviderTimeout
from bolo.providers import Usage
from bolo.remote.base import (
    Adapter,
    LiveEvent,
    LiveProtocol,
    RemoteRequest,
    base_language,
    decode_whole,
    pcm16_to_float,
    split_seconds,
)

#: the placeholder ``base_url`` of the kind: an Azure resource has its own host
PLACEHOLDER_HOST = "your-resource.services.ai.azure.com"
TRANSCRIPTION = "conversation.item.input_audio_transcription."
#: Microsoft's example output format of the Speech route (MP3; bolo decodes it)
DEFAULT_TTS_FORMAT = "audio-24khz-160kbitrate-mono-mp3"
#: a pause this long after speech ends a segment (MAI has no server-side speech detection; the client commits)
COMMIT_SILENCE_S = 0.5
#: a segment is committed after this long even without a pause
MAX_SEGMENT_S = 30.0
#: louder than this (absolute sample) counts as speech for the pause detection
SPEECH_LEVEL = 0.01


class AzureMAI(Adapter):
    kind = "azure-mai"
    max_chunk_s = 600.0

    def auth_headers(self) -> dict[str, str]:
        return {"api-key": self.provider.api_key} if self.provider.api_key else {}

    def option(self, name: str, default: Any = None) -> Any:
        if name in self.spec:
            return self.spec[name]
        return self.provider.options.get(name, default)

    @property
    def deployment(self) -> str:
        deployments = self.option("deployments") or {}
        return str(self.spec.get("deployment") or deployments.get(self.model) or self.model)

    def realtime_url(self) -> str:
        if self.option("realtime_url"):
            return str(self.option("realtime_url"))
        if PLACEHOLDER_HOST in self.base_url:
            raise InvalidRequest(
                f"{self.where} needs the address of your Azure resource: set its base_url to "
                "https://<resource>.services.ai.azure.com (BOLO_PROVIDER_AZURE_MAI_BASE_URL)",
                param="base_url",
            )
        base = self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return base + "/mai/v1/realtime?intent=transcription"

    def tts_url(self) -> str:
        if self.option("tts_url"):
            return str(self.option("tts_url"))
        region = str(self.option("region") or "").strip()
        if not region:
            raise InvalidRequest(
                f"{self.where} needs the Azure region of your Speech resource for MAI voices: set the provider "
                "option region (for example eastus), or tts_url",
                param="region",
            )
        return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def session_update(self, language: str | None) -> dict[str, Any]:
        code = base_language(language)
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "transcription": {"model": self.deployment, "language": code},
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }

    def check(self) -> dict[str, Any]:
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect

        try:
            with connect(self.realtime_url(), additional_headers=self.auth_headers(), open_timeout=10) as ws:
                first = json.loads(ws.recv(timeout=10))
        except InvalidStatus as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise ProviderAuthError(f"{self.where} refused bolo's API key ({status})") from exc
            raise ProviderError(f"{self.where} refused the connection ({status})") from exc
        except (OSError, TimeoutError, ValueError) as exc:
            raise ProviderError(self._clean(f"{self.where} is not reachable ({type(exc).__name__}: {exc})")) from exc
        if not isinstance(first, dict) or first.get("type") != "session.created":
            raise ProviderError(f"{self.where} answered the realtime socket with something else than session.created")
        return {"checked": "credentials", "message": "the key was accepted (the transcription socket opened)"}

    # -- recorded audio: the realtime socket, one commit -------------------------------------------------------
    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect

        if req.task == "translate":
            raise InvalidRequest(f"'{self.model}' at {self.where} cannot translate", param="model")
        for name, given in (("prompt", req.prompt), ("hotwords", req.hotwords), ("temperature", req.temperature)):
            if given:
                self.unsupported(name, f"`{name}`")
        samples = decode_whole(req.audio, 16000)
        limit = float(self.option("timeout_s") or 60.0) + req.seconds
        t0 = time.perf_counter()
        try:
            with connect(self.realtime_url(), additional_headers=self.auth_headers(), open_timeout=10) as ws:
                usage.requests += 1
                usage.first_byte_ms = (time.perf_counter() - t0) * 1000.0
                ws.send(json.dumps(self.session_update(req.language)))
                for piece in split_seconds(samples, 16000):
                    data = (np.clip(piece, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    frame = json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode()})
                    usage.bytes_sent += len(frame)
                    ws.send(frame)
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                deadline = time.monotonic() + limit
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ProviderTimeout(f"{self.where} did not finish the transcript in {limit:.0f} s")
                    raw = ws.recv(timeout=remaining)
                    usage.bytes_received += len(raw)
                    event = json.loads(raw)
                    kind = str(event.get("type") or "")
                    if kind in ("error", TRANSCRIPTION + "failed"):
                        message = self._clean(str((event.get("error") or {}).get("message") or event))
                        raise ProviderError(f"{self.where} could not transcribe the audio: {message}")
                    if kind == TRANSCRIPTION + "completed":
                        text = str(event.get("transcript") or "").strip()
                        break
        except InvalidStatus as exc:
            raise self.http_error(exc.response.status_code, bytes(exc.response.body or b"")) from exc
        except TimeoutError as exc:
            raise ProviderTimeout(f"{self.where} did not finish the transcript in {limit:.0f} s") from exc
        except OSError as exc:
            raise self.transport_error(exc) from exc
        finally:
            usage.latency_ms += (time.perf_counter() - t0) * 1000.0
        return STTResult(text, base_language(req.language))

    def live(self, language: str | None, diarize: bool = False) -> LiveProtocol | None:
        if self.spec.get("live") is False:
            return None
        return MAITranscribeLive(self, language)

    # -- MAI-Voice: Azure Speech's text-to-speech route -----------------------------------------------------
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
        name = voice or self.option("default_voice")
        if not name:
            raise InvalidRequest(f"{self.where} needs a voice (for example en-US-Harper)", param="voice")
        if speed != 1.0:
            self.unsupported("speed", "a speed other than 1")
        full = name if ":" in name else f"{name}:{self.model}"
        locale = "-".join(name.split("-")[:2]) if name.count("-") >= 2 else (language or "en-US")
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang={quoteattr(locale)}>'
            f"<voice name={quoteattr(full)}>{escape(text)}</voice></speak>"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.provider.api_key or "",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": str(self.option("output_format") or DEFAULT_TTS_FORMAT),
            "User-Agent": "bolo",
        }
        resp = self.request("POST", self.tts_url(), usage, content=ssml.encode("utf-8"), headers=headers)
        if not resp.content:
            raise ProviderError(f"{self.where} answered with no audio")
        fmt = headers["X-Microsoft-OutputFormat"]
        if fmt.startswith("raw-"):  # raw-24khz-16bit-mono-pcm, raw-22050hz-16bit-mono-pcm: headerless PCM16
            spec_rate = fmt.split("-")[1]
            digits = int("".join(ch for ch in spec_rate if ch.isdigit()) or 24)
            rate = digits * 1000 if spec_rate.endswith("khz") else digits
            audio = decode_whole(resp.content, out_rate, raw_rate=rate)
        else:
            audio = decode_whole(resp.content, out_rate)
        yield from split_seconds(audio, out_rate)


class MAITranscribeLive(LiveProtocol):
    """MAI-Transcribe-2-Streaming's socket. ``delta`` events are finalised text and ``intermediate`` the provisional
    rest: their sum is the interim text. MAI ends a segment only on the client's ``commit``, so the protocol commits
    at pauses it hears itself (``commit_silence_s`` of quiet after speech; at the latest every ``max_segment_s``),
    as Microsoft's guide suggests, and at the end of input; ``completed`` is then the final."""

    def __init__(self, adapter: AzureMAI, language: str | None) -> None:
        self.url = adapter.realtime_url()
        self.headers = adapter.auth_headers()
        self.sample_rate = 16000
        self.config = adapter.session_update(language)
        self.commit_silence_s = float(adapter.option("commit_silence_s") or COMMIT_SILENCE_S)
        self.max_segment_s = float(adapter.option("max_segment_s") or MAX_SEGMENT_S)
        self._done = ""  # finalised text of the open segment
        self._voiced = False
        self._quiet_s = 0.0
        self._segment_s = 0.0

    def start(self) -> str:
        self._done, self._voiced, self._quiet_s, self._segment_s = "", False, 0.0, 0.0
        return json.dumps(self.config)

    def audio(self, pcm: bytes) -> list[bytes | str]:
        frames: list[bytes | str] = [
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        ]
        x = pcm16_to_float(pcm)
        seconds = len(x) / self.sample_rate
        self._segment_s += seconds
        if len(x) and float(np.abs(x).max()) > SPEECH_LEVEL:
            self._voiced, self._quiet_s = True, 0.0
        elif self._voiced:
            self._quiet_s += seconds
        if self._voiced and (self._quiet_s >= self.commit_silence_s or self._segment_s >= self.max_segment_s):
            frames.append(self.finalize())
        return frames

    def finalize(self) -> str:
        self._voiced, self._quiet_s, self._segment_s = False, 0.0, 0.0
        return json.dumps({"type": "input_audio_buffer.commit"})

    def parse(self, message: bytes | str) -> list[LiveEvent]:
        with contextlib.suppress(ValueError):
            event = json.loads(message)
            if not isinstance(event, dict):
                return []
            kind = str(event.get("type") or "")
            if kind in ("error", TRANSCRIPTION + "failed"):
                error = event.get("error")
                message_text = error.get("message") if isinstance(error, dict) else error
                return [LiveEvent("error", str(message_text or kind))]
            if kind == TRANSCRIPTION + "delta":
                self._done += str(event.get("delta") or "")
                return [LiveEvent("interim", self._done.strip())] if self._done.strip() else []
            if kind == TRANSCRIPTION + "intermediate":
                text = (self._done + str(event.get("intermediate") or "")).strip()
                return [LiveEvent("interim", text)] if text else []
            if kind == TRANSCRIPTION + "completed":
                text = str(event.get("transcript") or self._done).strip()
                self._done = ""
                return [LiveEvent("final", text), LiveEvent("endpoint")] if text else [LiveEvent("endpoint")]
        return []
