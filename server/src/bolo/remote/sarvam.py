"""Sarvam AI: Indian-language speech-to-text (Saaras), translation to English, and text-to-speech (Bulbul).

Checked against https://docs.sarvam.ai (API reference pages for speech-to-text, speech-to-text-translate,
text-to-speech and the two WebSocket references) on 2026-11-14:

* ``POST https://api.sarvam.ai/speech-to-text`` (multipart): ``file``, ``model`` (``saaras:v4`` default,
  ``saaras:v3``), ``mode`` (``transcribe``, ``translate``, ``verbatim``, ``translit``, ``codemix``),
  ``language_code`` (``unknown`` or one of 23 ``xx-IN`` codes), ``with_timestamps`` (chunk-level times; no
  word times), ``keyterms`` (a JSON array, up to 50 terms of 64 characters; ``saaras:v4`` only). Meant for audio
  under 30 seconds: bolo cuts longer recordings at pauses. Answer: ``transcript``, ``language_code``,
  ``timestamps``.
* ``POST https://api.sarvam.ai/speech-to-text-translate`` (the older translate route, ``saaras:v2.5``, takes
  ``prompt``): used when the model option ``legacy_translate`` is set.
* ``POST https://api.sarvam.ai/text-to-speech`` (JSON): ``text`` (2500 characters; 1500 for ``bulbul:v2``),
  ``language_code`` (11 codes), ``speaker``, ``model`` (``bulbul:v2``, ``bulbul:v3``, ``bulbul:v4-flash``),
  ``pace``, ``speech_sample_rate``, ``output_audio_codec``. Answer: ``audios`` (base64 WAV).
* ``wss://api.sarvam.ai/speech-to-text/ws`` (streaming recognition; model option ``live``): audio as JSON
  ``{"audio": {"data": <base64>, "sample_rate": "16000", "encoding": "audio/wav"}}``, ``{"type": "flush"}``,
  answers ``{"type": "data", "data": {"transcript": ...}}`` and VAD ``events``. One transcript per utterance; no
  interim text. bolo's use of it has only been run against a stand-in for the service.
* Header ``api-subscription-key``.

Not verified: Saarika (``saarika:v2.5``) is no longer in the reference; it can still be asked for by name
(``sarvam:saarika:v2.5``) and Sarvam decides. The language field of text-to-speech was ``target_language_code``
in earlier versions of the API; the option ``language_field`` switches back.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlencode

from bolo.engines.base import FloatArray, Segment, STTResult
from bolo.errors import InvalidRequest, ProviderError
from bolo.providers import Usage
from bolo.remote.base import Adapter, LiveEvent, LiveProtocol, RemoteRequest, base_language, decode_whole, split_seconds

#: language_code values of speech-to-text (besides "unknown")
STT_LANGUAGES = (
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN", "en-IN", "gu-IN", "as-IN",
    "ur-IN", "ne-IN", "kok-IN", "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN",
)  # fmt: skip
TTS_LANGUAGES = ("bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN")
TTS_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
#: pace limits by model family
PACE = {"bulbul:v2": (0.3, 3.0)}
DEFAULT_PACE = (0.5, 2.0)


def sarvam_language(language: str | None, allowed: tuple[str, ...]) -> str | None:
    """bolo's language (``hi``, ``hi-IN``, ``or``) as Sarvam's code, or None when Sarvam has no such language."""
    base = base_language(language)
    if base is None:
        return None
    base = {"or": "od"}.get(base, base)
    code = f"{base}-IN"
    return code if code in allowed else None


#: Unicode blocks of the scripts Bulbul reads -> the language assumed when a request names none
_SCRIPTS = (
    (0x0900, 0x097F, "hi"), (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"), (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
)  # fmt: skip


def guess_language(text: str) -> str:
    """The language of ``text`` by its script (Devanagari is taken as Hindi); Latin text is Indian English."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for low, high, lang in _SCRIPTS:
            if low <= cp <= high:
                counts[lang] = counts.get(lang, 0) + 1
                break
    return max(counts, key=lambda k: counts[k]) if counts else "en"


def bolo_language(code: Any) -> str | None:
    base = base_language(str(code)) if code else None
    return {"od": "or"}.get(base, base) if base else None


class Sarvam(Adapter):
    kind = "sarvam"
    max_chunk_s = 30.0
    supports_translate = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_chunk_s = float(self.spec.get("max_chunk_s") or 30.0)
        self.supports_translate = bool(self.spec.get("translate", True))

    def auth_headers(self) -> dict[str, str]:
        return {"api-subscription-key": self.provider.api_key} if self.provider.api_key else {}

    def check(self) -> dict[str, Any]:
        # Sarvam documents no free, read-only route. An empty text-to-speech request costs nothing: a key that
        # is refused answers 403 (reported as a failure); a 400/422 for the empty body proves the address only.
        try:
            self.request("POST", self.base_url + "/text-to-speech", json={})
        except InvalidRequest:
            pass
        return {
            "checked": "reachability",
            "message": "the address answers and did not refuse the key; Sarvam has no free request that proves "
            "the key, so it is confirmed by the first real request",
        }

    # -- speech to text -----------------------------------------------------------------------------------
    def _stt_language(self, language: str | None) -> str:
        if not language:
            return "unknown"
        code = sarvam_language(language, STT_LANGUAGES)
        if code is None:
            raise InvalidRequest(
                f"{self.where} does not transcribe '{language}'. Languages: "
                + ", ".join(sorted({bolo_language(c) or c for c in STT_LANGUAGES})),
                param="language",
            )
        return code

    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        spec = self.spec
        translate = req.task == "translate"
        if translate and not self.supports_translate:
            raise InvalidRequest(f"'{self.model}' at {self.where} cannot translate", param="model")
        if req.word_timestamps:
            self.unsupported("timestamp_granularities[]", "word timestamps (Sarvam returns phrase-level times)")
        files = {"file": (req.filename, req.audio, req.content_type)}
        data: dict[str, Any] = {"model": self.model}
        if translate and spec.get("legacy_translate"):
            if req.prompt:
                data["prompt"] = req.prompt
            if req.hotword_list():
                self.unsupported("hotwords", "hotwords")
            doc = self.json(
                self.request("POST", self.base_url + "/speech-to-text-translate", usage, data=data, files=files)
            )
            return STTResult(str(doc.get("transcript") or "").strip(), "en")
        if req.prompt:
            self.unsupported("prompt", "a prompt")
        data["language_code"] = self._stt_language(req.language)
        mode = "translate" if translate else str(spec.get("mode") or "")
        if mode:
            data["mode"] = mode
        terms = req.hotword_list()
        if terms:
            if not spec.get("keyterms"):
                self.unsupported("hotwords", "hotwords (keyterms need saaras:v4)")
            else:
                data["keyterms"] = json.dumps([t[:64] for t in terms[:50]], ensure_ascii=False)
        if spec.get("with_timestamps", True):
            data["with_timestamps"] = "true"
        data.update(spec.get("request") or {})
        doc = self.json(self.request("POST", self.base_url + "/speech-to-text", usage, data=data, files=files))
        text = str(doc.get("transcript") or "").strip()
        segments: list[Segment] = []
        stamps = doc.get("timestamps")
        if isinstance(stamps, dict):
            chunks = stamps.get("words") or []
            starts = stamps.get("start_time_seconds") or []
            ends = stamps.get("end_time_seconds") or []
            for chunk, start, end in zip(chunks, starts, ends):
                if str(chunk).strip():
                    segments.append(Segment(float(start), float(end), str(chunk).strip()))
        language = "en" if translate else (bolo_language(doc.get("language_code")) or base_language(req.language))
        return STTResult(text, language, segments)

    def live(self, language: str | None) -> LiveProtocol | None:
        if not self.spec.get("live"):
            return None
        return SarvamLive(self, self._stt_language(language))

    # -- text to speech -----------------------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: str | None,
        speed: float,
        language: str | None,
        out_rate: int,
        usage: Usage,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        spec = self.spec
        code = sarvam_language(language or guess_language(text), TTS_LANGUAGES)
        if code is None:
            raise InvalidRequest(
                f"{self.where} does not speak '{language}'. Languages: "
                + ", ".join(bolo_language(c) or c for c in TTS_LANGUAGES),
                param="language",
            )
        limit = int(spec.get("max_chars") or (1500 if self.model == "bulbul:v2" else 2500))
        if len(text) > limit:
            raise InvalidRequest(
                f"'{self.model}' at {self.where} takes at most {limit} characters per piece of text; this one has "
                f"{len(text)} without a sentence break",
                param="input",
            )
        low, high = PACE.get(self.model, DEFAULT_PACE)
        if not low <= speed <= high:
            raise InvalidRequest(
                f"'{self.model}' at {self.where} speaks at speeds {low} to {high} (Sarvam's pace); got {speed}",
                param="speed",
            )
        rate = int(spec.get("provider_sample_rate") or out_rate)
        if rate not in TTS_RATES:
            rate = 24000
        body: dict[str, Any] = {
            "text": text,
            str(spec.get("language_field") or "language_code"): code,
            "model": self.model,
            "speech_sample_rate": rate,
            "output_audio_codec": "wav",
        }
        speaker = voice or spec.get("default_voice")
        if speaker:
            body["speaker"] = str(speaker).lower()
        if speed != 1.0:
            body["pace"] = speed
        if "enable_preprocessing" in spec:
            body["enable_preprocessing"] = bool(spec["enable_preprocessing"])
        body.update(spec.get("request") or {})
        doc = self.json(self.request("POST", self.base_url + "/text-to-speech", usage, json=body))
        audios = doc.get("audios") or []
        if not audios:
            raise ProviderError(f"{self.where} answered without audio")
        for item in audios:
            try:
                data = base64.b64decode(str(item))
            except ValueError as exc:
                raise ProviderError(f"{self.where} answered with audio that is not base64") from exc
            yield from split_seconds(decode_whole(data, out_rate, raw_rate=rate), out_rate)
            if cancelled is not None and cancelled():
                return


class SarvamLive(LiveProtocol):
    def __init__(self, adapter: Sarvam, language_code: str) -> None:
        base = adapter.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        params = {
            "language-code": language_code,
            "model": adapter.model,
            "sample_rate": "16000",
            "input_audio_codec": "pcm_s16le",
            "vad_signals": "true",
            "flush_signal": "true",
        }
        if adapter.spec.get("mode"):
            params["mode"] = str(adapter.spec["mode"])
        self.url = f"{base}/speech-to-text/ws?{urlencode(params)}"
        self.headers = {"Api-Subscription-Key": adapter.provider.api_key or ""}
        self.sample_rate = 16000

    def audio(self, pcm: bytes) -> str:
        data = base64.b64encode(pcm).decode()
        return json.dumps({"audio": {"data": data, "sample_rate": "16000", "encoding": "audio/wav"}})

    def finalize(self) -> str:
        return json.dumps({"type": "flush"})

    def parse(self, message: bytes | str) -> list[LiveEvent]:
        try:
            doc = json.loads(message)
        except ValueError:
            return []
        if not isinstance(doc, dict):
            return []
        raw = doc.get("data")
        data: dict[str, Any] = raw if isinstance(raw, dict) else {}
        kind = doc.get("type")
        if kind == "data":
            text = str(data.get("transcript") or "").strip()
            # one transcript per utterance: it is final, and the utterance is over
            return [LiveEvent("final", text), LiveEvent("endpoint")] if text else []
        if kind == "events":
            return [LiveEvent("endpoint")] if data.get("signal_type") == "END_SPEECH" else []
        if kind == "error":
            return [LiveEvent("error", str(data.get("error") or data.get("message") or "error"))]
        return []
