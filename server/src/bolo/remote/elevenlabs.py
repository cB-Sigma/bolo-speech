"""ElevenLabs: text-to-speech and speech-to-text (Scribe).

Checked against https://elevenlabs.io/docs/api-reference (text-to-speech/stream, speech-to-text/convert,
voices/search) and the models overview on 2026-11-14:

* ``POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_24000`` (JSON):
  ``text``, ``model_id``, ``language_code`` (ISO 639-1; not for ``eleven_multilingual_v2``),
  ``voice_settings.speed``, ``apply_text_normalization`` (``auto``, ``on``, ``off``). The answer is the audio,
  streamed; ``pcm_*`` formats are 16-bit mono (8, 16, 22.05, 24, 32, 44.1 and 48 kHz; 44.1 kHz needs a Pro plan).
* ``POST https://api.elevenlabs.io/v1/speech-to-text`` (multipart): ``model_id`` (``scribe_v3``,
  ``scribe_v2``), ``file``, ``language_code``, ``timestamps_granularity`` (``word``), ``keyterms``,
  ``temperature``. Answer: ``text``, ``language_code``, ``words`` (``text``, ``start``, ``end``, ``type``).
* ``GET https://api.elevenlabs.io/v2/voices`` lists the account's voices (``voice_id``, ``name``, ``labels``).
* ``POST https://api.elevenlabs.io/v1/audio-isolation`` (multipart): ``audio``, ``file_format`` (``pcm_s16le_16``:
  raw 16-bit mono 16 kHz PCM, lower latency; or ``other``). The answer is the isolated voice as an audio file
  (checked against the API reference on 2026-11-17; the reference does not name the answer's format, so bolo
  decodes whatever arrives). ElevenLabs bills it per minute of audio and accepts up to 500 MB and one hour.
* Header ``xi-api-key``.

Not verified: the limits of ``voice_settings.speed`` (the reference gives none; ElevenLabs answers 400 outside
them, which bolo passes on), and the ids of the stock voices, which is why the catalog lists none: the voices
of the account are read when the model is loaded, and a voice can be named by id or by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

from bolo.engines.base import FloatArray, STTResult, Voice, Word
from bolo.errors import BoloError, InvalidRequest, ProviderError
from bolo.providers import Usage
from bolo.remote.base import Adapter, RemoteRequest, base_language, decode_whole, float_to_pcm16, read_pcm_stream

PCM_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


class ElevenLabs(Adapter):
    kind = "elevenlabs"
    supports_word_timestamps = True

    def auth_headers(self) -> dict[str, str]:
        return {"xi-api-key": self.provider.api_key} if self.provider.api_key else {}

    def check(self) -> dict[str, Any]:
        doc = self.json(self.request("GET", self.base_url + "/v2/voices", params={"page_size": 1}))
        total = doc.get("total_count")
        return {
            "checked": "credentials",
            "message": "the key was accepted" + (f"; {total} voices are available" if total is not None else ""),
        }

    def voices(self) -> list[Voice]:
        out: list[Voice] = []
        token: str | None = None
        for _ in range(5):  # at most 500 voices
            params: dict[str, Any] = {"page_size": 100}
            if token:
                params["next_page_token"] = token
            try:
                doc = self.json(self.request("GET", self.base_url + "/v2/voices", params=params))
            except BoloError:
                break  # a key without the voices permission still synthesises with a voice id
            for v in doc.get("voices") or []:
                if not isinstance(v, dict) or not v.get("voice_id"):
                    continue
                raw = v.get("labels")
                labels: dict[str, Any] = raw if isinstance(raw, dict) else {}
                gender = {"female": "f", "male": "m"}.get(str(labels.get("gender", "")).lower(), "")
                out.append(
                    Voice(
                        str(v["voice_id"]),
                        str(v.get("name") or ""),
                        str(labels.get("language") or ""),
                        gender,
                        len(out),
                    )
                )
            token = doc.get("next_page_token") if doc.get("has_more") else None
            if not token:
                break
        return out

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
        voice_id = voice or spec.get("default_voice")
        if not voice_id:
            raise InvalidRequest(
                f"{self.where} needs a voice: send the id (or name) of one of your ElevenLabs voices as `voice`",
                param="voice",
            )
        rate = int(spec.get("provider_sample_rate") or out_rate)
        if rate not in PCM_RATES:
            rate = 24000
        body: dict[str, Any] = {"text": text, "model_id": self.model}
        code = base_language(language)
        if code and spec.get("send_language"):
            body["language_code"] = code
        if speed != 1.0:
            body["voice_settings"] = {"speed": speed}
        if spec.get("apply_text_normalization"):
            body["apply_text_normalization"] = str(spec["apply_text_normalization"])
        body.update(spec.get("request") or {})
        url = f"{self.base_url}/v1/text-to-speech/{quote(str(voice_id), safe='')}/stream"
        with self.stream("POST", url, usage, params={"output_format": f"pcm_{rate}"}, json=body) as resp:
            yield from read_pcm_stream(resp.iter_bytes(), rate, out_rate, cancelled)

    # -- speech to text -----------------------------------------------------------------------------------
    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        if req.task == "translate":
            raise InvalidRequest(f"'{self.model}' at {self.where} cannot translate", param="model")
        if req.prompt:
            self.unsupported("prompt", "a prompt")
        data: dict[str, Any] = {"model_id": self.model, "timestamps_granularity": "word"}
        code = base_language(req.language)
        if code:
            data["language_code"] = code
        terms = req.hotword_list()
        if terms:
            data["keyterms"] = [t[:50] for t in terms[:1000]]
        if req.temperature:
            data["temperature"] = str(req.temperature)
        data.update(self.spec.get("request") or {})
        files = {"file": (req.filename, req.audio, req.content_type)}
        doc = self.json(self.request("POST", self.base_url + "/v1/speech-to-text", usage, data=data, files=files))
        words = [
            Word(str(w.get("text", "")).strip(), float(w.get("start") or 0.0), float(w.get("end") or 0.0))
            for w in doc.get("words") or []
            if isinstance(w, dict) and w.get("type", "word") == "word" and str(w.get("text", "")).strip()
        ]
        language = base_language(str(doc.get("language_code") or "")) or base_language(req.language)
        # ISO 639-3 answers ("eng", "hin") stay as they are; clients that need 639-1 send the language
        return STTResult(str(doc.get("text") or "").strip(), language, [], words)

    # -- voice isolation ----------------------------------------------------------------------------------
    def isolate(self, samples: FloatArray, rate: int, usage: Usage) -> FloatArray:
        if rate != 16000:
            raise InvalidRequest(f"{self.where}: voice isolation is sent 16 kHz audio, got {rate} Hz", param="denoise")
        data: dict[str, Any] = {"file_format": "pcm_s16le_16"}
        data.update(self.spec.get("request") or {})
        files = {"audio": ("audio.pcm", float_to_pcm16(samples), "application/octet-stream")}
        resp = self.request("POST", self.base_url + "/v1/audio-isolation", usage, data=data, files=files)
        if not resp.content:
            raise ProviderError(f"{self.where} answered the voice isolation with no audio")
        return decode_whole(resp.content, rate)
