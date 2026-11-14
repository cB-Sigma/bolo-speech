"""OpenAI's audio API, and every server that copies it (another bolo, speaches, vLLM, LocalAI, ...).

Checked against the OpenAI API reference as typed in openai-python 3.51.0 (generated from OpenAI's OpenAPI
description) on 2026-11-14:

* ``POST {base_url}/audio/transcriptions`` (multipart): ``file``, ``model``, ``language`` (ISO 639-1),
  ``prompt``, ``temperature``, ``response_format`` (``json``; ``verbose_json`` with
  ``timestamp_granularities[]`` for ``whisper-1`` only), ``stream`` (server-sent events
  ``transcript.text.delta`` / ``transcript.text.done``; ignored by ``whisper-1``), ``keywords`` (``gpt-transcribe``).
* ``POST {base_url}/audio/translations`` (``whisper-1``).
* ``POST {base_url}/audio/speech`` (JSON): ``model``, ``input`` (at most 4096 characters), ``voice``,
  ``instructions``, ``response_format`` (``pcm`` is 24 kHz 16-bit mono), ``speed`` (0.25 to 4.0). The answer is
  chunked, so audio is passed on while the provider is still synthesising.
* ``Authorization: Bearer <key>``.

Model options (catalog ``options`` of an ``engine: remote`` entry):

``response_format``  ``json`` (default) or ``verbose_json`` (segments, language; needed for word timestamps)
``word_timestamps``  the model returns word times with ``verbose_json`` (``whisper-1``)
``translate``        the model serves ``/audio/translations``
``stream``           ask for a streamed transcription and join the deltas
``prompt``           false: the model takes no ``prompt``
``hotwords_field``   ``keywords`` sends hotwords as ``keywords[]``; default: appended to the prompt
``instructions``     speech: sent as ``instructions`` (voice direction of ``gpt-4o-mini-tts``)
``speed``            false: the model speaks at one speed only
``send_language``    speech: also send ``language`` (a bolo extension another bolo understands)
``pcm_sample_rate``  the rate of the server's ``pcm`` answers when it sends no ``x-sample-rate`` header (24000)
``request``          extra fields merged into every request body
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from bolo.engines.base import FloatArray, Segment, STTResult, Word
from bolo.errors import InvalidRequest, ProviderError
from bolo.providers import Usage
from bolo.remote.base import LANGUAGE_CODES, Adapter, RemoteRequest, base_language, read_pcm_stream

_RATE_RE = re.compile(r"rate=(\d+)")


class OpenAICompatible(Adapter):
    kind = "openai-compatible"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.supports_translate = bool(self.spec.get("translate"))
        self.supports_word_timestamps = bool(self.spec.get("word_timestamps"))
        # 25 MB per upload at OpenAI: 16 kHz PCM16 WAV is 32 kB/s, so 10 minutes stay well below it
        self.max_chunk_s = float(self.spec.get("max_chunk_s") or 600.0)

    def auth_headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.provider.api_key}"} if self.provider.api_key else {}
        if self.provider.options.get("organization"):
            headers["OpenAI-Organization"] = str(self.provider.options["organization"])
        if self.provider.options.get("project"):
            headers["OpenAI-Project"] = str(self.provider.options["project"])
        return headers

    def check(self) -> dict[str, Any]:
        doc = self.json(self.request("GET", self.base_url + "/models"))
        count = len(doc.get("data") or [])
        return {"checked": "credentials", "message": f"the key was accepted; {count} models are listed"}

    # -- speech to text -----------------------------------------------------------------------------------
    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        spec = self.spec
        translate = req.task == "translate"
        if translate and not self.supports_translate:
            raise InvalidRequest(
                f"'{self.model}' at {self.where} cannot translate; use a model with /audio/translations (whisper-1)",
                param="model",
            )
        fmt = str(spec.get("response_format") or "json")
        data: dict[str, Any] = {"model": self.model, "response_format": fmt}
        language = base_language(req.language)
        if language and not translate:
            data["language"] = language
        prompt = req.prompt
        terms = req.hotword_list()
        if terms and spec.get("hotwords_field") == "keywords":
            data["keywords[]"] = terms
        elif terms:
            if spec.get("prompt") is False:
                self.unsupported("hotwords", "hotwords")
            else:
                prompt = ", ".join([*([prompt] if prompt else []), *terms])
        if prompt:
            if spec.get("prompt") is False:
                self.unsupported("prompt", "a prompt")
            else:
                data["prompt"] = prompt
        if req.temperature:
            data["temperature"] = str(req.temperature)
        if req.word_timestamps:
            if not self.supports_word_timestamps or fmt != "verbose_json":
                self.unsupported("timestamp_granularities[]", "word timestamps")
            else:
                data["timestamp_granularities[]"] = ["word", "segment"]
        data.update(spec.get("request") or {})
        files = {"file": (req.filename, req.audio, req.content_type)}
        url = self.base_url + ("/audio/translations" if translate else "/audio/transcriptions")
        if spec.get("stream") and not translate:
            return self._transcribe_stream(url, {**data, "stream": "true"}, files, usage)
        doc = self.json(self.request("POST", url, usage, data=data, files=files))
        return self._result(doc, req)

    def _transcribe_stream(self, url: str, data: dict[str, Any], files: dict[str, Any], usage: Usage) -> STTResult:
        parts: list[str] = []
        final: str | None = None
        with self.stream("POST", url, usage, data=data, files=files) as resp:
            if not resp.headers.get("content-type", "").startswith("text/event-stream"):
                # a model (whisper-1) or server that ignores `stream` answers with plain JSON
                doc = json.loads(resp.read())
                return STTResult(str(doc.get("text") or "").strip())
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                kind = event.get("type")
                if kind == "transcript.text.delta":
                    parts.append(str(event.get("delta") or ""))
                elif kind == "transcript.text.done":
                    final = str(event.get("text") or "")
                elif kind == "error" or event.get("error"):
                    err = event.get("error") or {}
                    message = str(err.get("message") if isinstance(err, dict) else err)
                    raise ProviderError(self._clean(f"{self.where} failed while streaming: {message}"))
        return STTResult((final if final is not None else "".join(parts)).strip())

    @staticmethod
    def _result(doc: dict[str, Any], req: RemoteRequest) -> STTResult:
        text = str(doc.get("text") or "").strip()
        language = doc.get("language")
        if isinstance(language, str):
            language = LANGUAGE_CODES.get(language.lower(), language if len(language) <= 3 else None)
        words = [
            Word(str(w.get("word", "")), float(w.get("start") or 0.0), float(w.get("end") or 0.0))
            for w in doc.get("words") or []
            if isinstance(w, dict)
        ]
        segments = [
            Segment(
                float(s.get("start") or 0.0),
                float(s.get("end") or 0.0),
                str(s.get("text") or "").strip(),
                [w for w in words if float(s.get("start") or 0.0) - 0.01 <= w.t0 < float(s.get("end") or 0.0) + 0.01],
                s.get("avg_logprob"),
                s.get("no_speech_prob"),
            )
            for s in doc.get("segments") or []
            if isinstance(s, dict)
        ]
        return STTResult(text, language or base_language(req.language), segments, words)

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
        voice = voice or spec.get("default_voice")
        if not voice:
            raise InvalidRequest(f"'{self.model}' at {self.where} needs a voice", param="voice")
        body: dict[str, Any] = {"model": self.model, "input": text, "voice": voice, "response_format": "pcm"}
        if speed != 1.0:
            if spec.get("speed") is False:
                self.unsupported("speed", "a speed other than 1.0")
            else:
                body["speed"] = speed
        if spec.get("instructions"):
            body["instructions"] = str(spec["instructions"])
        if language and spec.get("send_language"):
            body["language"] = language
        body.update(spec.get("request") or {})
        with self.stream("POST", self.base_url + "/audio/speech", usage, json=body) as resp:
            match = _RATE_RE.search(resp.headers.get("content-type", ""))
            rate = int(
                resp.headers.get("x-sample-rate") or (match.group(1) if match else spec.get("pcm_sample_rate") or 24000)
            )
            yield from read_pcm_stream(resp.iter_bytes(), rate, out_rate, cancelled)
