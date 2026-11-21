"""AssemblyAI: speech-to-text, recorded files (pre-recorded API) and live (Streaming API v3).

Checked against https://www.assemblyai.com/docs (pre-recorded: files/upload, transcripts/submit, transcripts/get,
transcripts/delete, transcripts/list, select-the-speech-model, the assemblyai-stt-3-7 migration guide,
label-speakers, supported-languages; streaming: select-the-speech-model, message-sequence, turn-detection,
label-speakers-and-separate-channels, and the AsyncAPI description api-reference/specs/streaming.yaml) on
2026-11-21:

* Recorded audio is three steps. ``POST https://api.assemblyai.com/v2/upload`` (the raw file as
  ``application/octet-stream``) answers ``upload_url``; ``POST /v2/transcript`` (JSON) with ``audio_url``,
  ``speech_models`` (``["assemblyai-stt-3-7"]``), ``language_code`` or ``language_detection: true``,
  ``speaker_labels`` (and ``speakers_expected``), ``keyterms_prompt`` (up to 1000 terms on STT-3.7) and
  ``prompt`` answers ``id`` and ``status`` (``queued``, ``processing``, ``completed``, ``error`` with ``error``);
  ``GET /v2/transcript/{id}`` is polled (60 requests a minute per key and transcript; 429 with ``Retry-After``)
  and, once completed, carries ``text``, ``language_code``, ``words`` (``text``, ``start`` and ``end`` in ms,
  ``confidence``, ``speaker``) and, with speaker labels, ``utterances`` (``speaker``, ``start``, ``end``,
  ``text``, ``words``). bolo then deletes the transcript (``DELETE /v2/transcript/{id}``); the model option
  ``keep_files`` leaves it at AssemblyAI. STT-3.7 keeps disfluencies by default (option ``request``:
  ``{"disfluencies": false}`` to drop them).
* ``wss://streaming.assemblyai.com/v3/ws`` (``streaming.us.`` / ``streaming.eu.`` for data residency): the key in
  the ``Authorization`` header (no ``Bearer``), the session's settings as query parameters (``speech_model``:
  ``assemblyai-stt-3-7-realtime`` or ``assemblyai-stt-3-7-realtime-agent``; ``sample_rate``, ``encoding``
  ``pcm_s16le``, ``language_codes`` as a JSON list, ``speaker_labels``, ``max_speakers``, ``min_turn_silence``,
  ``max_turn_silence``), audio as binary frames of 20 ms to 1 s (50 to 100 ms recommended), and the JSON messages
  ``ForceEndpoint`` (end the turn now), ``KeepAlive`` and ``Terminate``. Answers: ``Begin``, ``SpeechStarted``,
  ``Turn`` (``transcript``, ``end_of_turn``, ``turn_is_formatted``, ``words`` with ``start`` / ``end`` in ms
  from the start of the stream, and with speaker labels ``speaker_label`` per turn and ``speaker`` per final
  word), ``Termination`` last, and ``Error`` (``error_code``, ``error``) before the socket closes. A turn is
  complete on the message where both ``end_of_turn`` and ``turn_is_formatted`` are true; each ``Turn`` replaces
  the previous one of the same turn.
* Speaker labels are ``A``, ``B``, … (``UNKNOWN`` where the model could not tell), the same as bolo's.
* Header ``Authorization: <key>``. ``GET /v2/transcript?limit=1`` lists transcripts (bolo's key check).

Not verified (no account was used): whether a bad key is refused at the WebSocket handshake or with an ``Error``
message, and the exact text of error bodies beyond the documented ``error`` field.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any
from urllib.parse import urlencode

from bolo.engines.base import Segment, STTResult, Word
from bolo.errors import BoloError, InvalidRequest, ProviderError, ProviderTimeout
from bolo.providers import Usage
from bolo.remote.base import Adapter, LiveEvent, LiveProtocol, RemoteRequest, base_language

#: the live socket's address at AssemblyAI (the REST API is at api.assemblyai.com)
REALTIME_URL = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_ASYNC_MODEL = "assemblyai-stt-3-7"
DEFAULT_LIVE_MODEL = "assemblyai-stt-3-7-realtime"
#: AssemblyAI takes 20 ms to 1 s of audio per frame: shorter pieces are joined to 50 ms (at 16 kHz, PCM16)
MIN_FRAME_BYTES = 1600
MAX_FRAME_BYTES = 32000


def ms(value: Any) -> float:
    with contextlib.suppress(TypeError, ValueError):
        return float(value) / 1000.0
    return 0.0


def speaker_of(value: Any) -> str | None:
    """AssemblyAI's ``A``, ``B``, … are bolo's labels; ``UNKNOWN`` and nothing are no label."""
    text = str(value or "").strip()
    return text if text and text.upper() != "UNKNOWN" else None


def transcript_to_result(doc: dict[str, Any], fallback_language: str | None) -> STTResult:
    """A completed transcript as bolo's result: words, segments (one per utterance, with its speaker, when speaker
    labels were asked for), and the language."""
    words = [
        Word(str(w.get("text") or "").strip(), ms(w.get("start")), ms(w.get("end")), _confidence(w))
        for w in doc.get("words") or []
        if isinstance(w, dict) and str(w.get("text") or "").strip()
    ]
    segments: list[Segment] = []
    for u in doc.get("utterances") or []:
        if not isinstance(u, dict) or not str(u.get("text") or "").strip():
            continue
        seg_words = [
            Word(str(w.get("text") or "").strip(), ms(w.get("start")), ms(w.get("end")), _confidence(w))
            for w in u.get("words") or []
            if isinstance(w, dict) and str(w.get("text") or "").strip()
        ]
        segments.append(
            Segment(
                ms(u.get("start")),
                ms(u.get("end")),
                str(u["text"]).strip(),
                seg_words,
                speaker=speaker_of(u.get("speaker")),
            )
        )
    language = base_language(str(doc.get("language_code") or "")) or base_language(fallback_language)
    probability = doc.get("language_confidence")
    result = STTResult(str(doc.get("text") or "").strip(), language, segments, words)
    if isinstance(probability, (int, float)):
        result.language_probability = float(probability)
    return result


def _confidence(w: dict[str, Any]) -> float | None:
    value = w.get("confidence")
    return float(value) if isinstance(value, (int, float)) else None


class AssemblyAI(Adapter):
    kind = "assemblyai"
    #: AssemblyAI takes long files in one request; bolo cuts recordings longer than an hour at pauses
    max_chunk_s = 3600.0
    supports_word_timestamps = True
    supports_diarization = True
    supports_live_diarization = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_chunk_s = float(self.spec.get("max_chunk_s") or 3600.0)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": self.provider.api_key} if self.provider.api_key else {}

    def option(self, name: str, default: Any = None) -> Any:
        """A model option, else the provider's option of the same name."""
        if name in self.spec:
            return self.spec[name]
        return self.provider.options.get(name, default)

    def check(self) -> dict[str, Any]:
        self.json(self.request("GET", self.base_url + "/v2/transcript", params={"limit": 1}))
        return {"checked": "credentials", "message": "the key was accepted"}

    # -- which AssemblyAI model -------------------------------------------------------------------------------
    @property
    def async_model(self) -> str:
        """The pre-recorded model (recorded audio of a realtime entry goes to the pre-recorded API)."""
        if self.spec.get("async_model"):
            return str(self.spec["async_model"])
        if "-realtime" in self.model:
            return self.model.split("-realtime")[0]
        return self.model or DEFAULT_ASYNC_MODEL

    @property
    def live_model(self) -> str:
        if self.spec.get("live_model"):
            return str(self.spec["live_model"])
        return self.model if "-realtime" in self.model else DEFAULT_LIVE_MODEL

    # -- recorded audio -----------------------------------------------------------------------------------
    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        if req.task == "translate":
            raise InvalidRequest(f"'{self.model}' at {self.where} cannot translate", param="model")
        body: dict[str, Any] = {"speech_models": [self.async_model]}
        code = base_language(req.language)
        if code:
            body["language_code"] = code
        else:
            body["language_detection"] = True
        if req.diarize or self.option("speaker_labels", False):
            body["speaker_labels"] = True
            if self.option("speakers_expected"):
                body["speakers_expected"] = int(self.option("speakers_expected"))
        terms = req.hotword_list()
        if terms:
            body["keyterms_prompt"] = terms[:1000]
        if req.prompt:
            body["prompt"] = req.prompt
        if req.temperature:
            self.unsupported("temperature", "a sampling temperature")
        body.update(self.spec.get("request") or {})

        headers = {"Content-Type": "application/octet-stream"}
        upload = self.json(
            self.request("POST", self.base_url + "/v2/upload", usage, content=req.audio, headers=headers)
        )
        audio_url = str(upload.get("upload_url") or "")
        if not audio_url:
            raise ProviderError(f"{self.where} accepted the upload but returned no upload_url")
        body["audio_url"] = audio_url
        job = self.json(self.request("POST", self.base_url + "/v2/transcript", usage, json=body))
        job_id = str(job.get("id") or "")
        if not job_id:
            raise ProviderError(f"{self.where} accepted the transcription but returned no id")
        try:
            doc = self._wait(job_id, job, req.seconds, usage)
        finally:
            if not self.option("keep_files", False):
                with contextlib.suppress(BoloError):
                    self.request("DELETE", f"{self.base_url}/v2/transcript/{job_id}")
        return transcript_to_result(doc, req.language)

    def _wait(self, job_id: str, job: dict[str, Any], seconds: float, usage: Usage) -> dict[str, Any]:
        """Poll the transcript until it is done (0.25 s, doubling to 3 s between polls: AssemblyAI allows 60 a
        minute per transcript)."""
        limit = float(self.option("poll_timeout_s") or max(120.0, 60.0 + seconds))
        deadline = time.monotonic() + limit
        pause = float(self.option("poll_interval_s") or 0.25)
        status = str(job.get("status") or "queued")
        while True:
            if status == "completed":
                return job
            if status == "error":
                message = self._clean(str(job.get("error") or "no message"))
                raise ProviderError(f"{self.where} could not transcribe the audio: {message}")
            if time.monotonic() + pause > deadline:
                raise ProviderTimeout(f"{self.where} had not finished the transcription after {limit:.0f} s ({status})")
            time.sleep(pause)
            pause = min(3.0, pause * 2)
            job = self.json(self.request("GET", f"{self.base_url}/v2/transcript/{job_id}", usage))
            status = str(job.get("status") or "")

    # -- live -----------------------------------------------------------------------------------------------
    def live(self, language: str | None, diarize: bool = False) -> LiveProtocol | None:
        if self.spec.get("live") is False:
            return None
        return AssemblyAILive(self, language, diarize)


class AssemblyAILive(LiveProtocol):
    """AssemblyAI's Streaming API v3. Each ``Turn`` replaces the text of its turn: interim text until the turn is
    complete (``end_of_turn`` and ``turn_is_formatted``), then one final with its words and speaker, and an
    endpoint (AssemblyAI's turn detection ended the turn, or ``ForceEndpoint`` did)."""

    def __init__(self, adapter: AssemblyAI, language: str | None, diarize: bool) -> None:
        url = adapter.option("realtime_url")
        if not url:
            if adapter.provider.is_default_host:
                url = REALTIME_URL
            else:
                base = adapter.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
                url = base + "/v3/ws"
        self.sample_rate = 16000
        params: dict[str, Any] = {
            "speech_model": adapter.live_model,
            "sample_rate": self.sample_rate,
            "encoding": "pcm_s16le",
        }
        code = base_language(language)
        codes = [code] if code else [str(c) for c in adapter.option("language_codes") or []]
        if codes:
            params["language_codes"] = json.dumps(codes)
        if diarize or adapter.option("live_speaker_labels", False):
            params["speaker_labels"] = "true"
            if adapter.option("max_speakers"):
                params["max_speakers"] = int(adapter.option("max_speakers"))
        for name in ("min_turn_silence", "max_turn_silence", "mode"):
            if adapter.option(name) is not None:
                params[name] = adapter.option(name)
        params.update(adapter.spec.get("live_request") or {})
        self.url = f"{url}?{urlencode(params)}"
        self.headers = adapter.auth_headers()
        self.params = params
        self._held = b""

    def audio(self, pcm: bytes) -> bytes:
        self._held += pcm
        if len(self._held) < MIN_FRAME_BYTES:
            return b""
        frame, self._held = self._held[:MAX_FRAME_BYTES], self._held[MAX_FRAME_BYTES:]
        return frame

    def flush(self) -> bytes | None:
        # a last piece shorter than 20 ms is padded with silence rather than refused
        held, self._held = self._held, b""
        if not held:
            return None
        return held + bytes(max(0, 640 - len(held)))

    def finalize(self) -> str:
        return json.dumps({"type": "ForceEndpoint"})

    def close(self) -> str:
        return json.dumps({"type": "Terminate"})

    def parse(self, message: bytes | str) -> list[LiveEvent]:
        try:
            doc = json.loads(message)
        except ValueError:
            return []
        if not isinstance(doc, dict):
            return []
        kind = doc.get("type")
        if kind == "Error" or (doc.get("error") and kind not in ("Turn", "Begin")):
            return [LiveEvent("error", f"{doc.get('error_code', '')} {doc.get('error') or 'error'}".strip())]
        if kind != "Turn":
            return []
        text = str(doc.get("transcript") or "").strip()
        if not (doc.get("end_of_turn") and doc.get("turn_is_formatted")):
            return [LiveEvent("interim", text)] if text else []
        events: list[LiveEvent] = []
        if text:
            words = [
                (str(w.get("text") or "").strip(), ms(w.get("start")), ms(w.get("end")))
                for w in doc.get("words") or []
                if isinstance(w, dict) and str(w.get("text") or "").strip()
            ]
            events.append(LiveEvent("final", text, words, speaker_of(doc.get("speaker_label"))))
        events.append(LiveEvent("endpoint"))
        return events
