"""Soniox: speech-to-text, recorded files (async API) and live (WebSocket API).

Checked against https://docs.soniox.com (API reference: files/upload_file, files/delete_file,
transcriptions/create_transcription, get_transcription, get_transcription_transcript, delete_transcription,
models/get_models, stt/websocket-api, errors; guides: async transcription, models, supported languages, context,
manual finalization, endpoint detection, connection keepalive, websocket authentication) on 2026-11-17:

* Recorded audio is three steps. ``POST https://api.soniox.com/v1/files`` (multipart ``file``) answers ``id``;
  ``POST /v1/transcriptions`` (JSON) with ``model`` (``stt-async-v6``), ``file_id``, ``language_hints``,
  ``language_hints_strict``, ``enable_speaker_diarization``, ``enable_language_identification``, ``context``
  (``{"text", "terms", "general"}``, about 10,000 characters) and ``translation``
  (``{"type": "one_way", "target_language": "en"}``) answers ``id`` and ``status`` (``queued``, ``processing``,
  ``completed``, ``error`` with ``error_type`` / ``error_message``); ``GET /v1/transcriptions/{id}`` is polled;
  ``GET /v1/transcriptions/{id}/transcript`` answers ``text`` and ``tokens`` (``text``, ``start_ms``,
  ``end_ms``, ``confidence``, ``speaker``, ``language``, ``translation_status``). bolo then deletes the
  transcription and the file (``DELETE /v1/transcriptions/{id}``, ``DELETE /v1/files/{id}``); the model option
  ``keep_files`` leaves them at Soniox.
* Tokens are pieces of words: a token that starts with a space starts a new word. ``speaker`` is ``"1"``,
  ``"2"``, … (bolo's labels ``A``, ``B``, …); ``language`` is set per token with language identification.
* ``wss://stt-rt.soniox.com/transcribe-websocket`` (live, model ``stt-rt-v6``): the key in the
  ``Authorization: Bearer`` header (the ``api_key`` field of the start message is deprecated), a JSON start
  message (``model``, ``audio_format: pcm_s16le``, ``sample_rate``, ``num_channels``, ``language_hints``,
  ``enable_endpoint_detection``, ``max_endpoint_delay_ms``), binary audio frames, the control messages
  ``{"type": "finalize"}`` and ``{"type": "keepalive"}`` (at least every 20 s without audio), and an empty text
  frame to end the stream. Answers: ``{"tokens": [...], "final_audio_proc_ms", "total_audio_proc_ms"}``, where
  final tokens arrive once and non-final ones are re-sent each time; the tokens ``<end>`` (endpoint detection)
  and ``<fin>`` (after ``finalize``); ``finished: true`` last; errors as ``error_code``, ``error_type``,
  ``error_message``.
* ``Authorization: Bearer <key>``. ``GET /v1/models`` lists the models the key can use.
* Errors: 400 invalid request, 401 key, 402 balance or budget exhausted, 408 timeout, 429 rate or concurrency
  limit, 500 and 503 (retry). REST error bodies carry ``message``; no ``Retry-After`` is documented.

Not verified (no account was used): the exact shape of tokens in the async transcript beyond the fields named
above, how translation tokens are ordered in an async transcript, and whether a WebSocket refuses the
connection or sends an error message for a bad key.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import Counter
from typing import Any

from bolo.diarize import speaker_label
from bolo.engines.base import Segment, STTResult, Word
from bolo.errors import BoloError, InvalidRequest, ProviderError, ProviderTimeout
from bolo.providers import Usage
from bolo.remote.base import Adapter, LiveEvent, LiveProtocol, RemoteRequest, base_language

#: the live socket's address at Soniox (the REST API is at api.soniox.com)
REALTIME_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
DEFAULT_ASYNC_MODEL = "stt-async-v6"
DEFAULT_LIVE_MODEL = "stt-rt-v6"
#: characters a token may end a segment with
SENTENCE_END = (".", "?", "!", "。", "？", "！", "।", "॥")
#: special tokens of the live socket
MARKERS = ("<end>", "<fin>")


def _no_space_script(ch: str) -> bool:
    """Characters of scripts written without spaces between words (Chinese, Japanese, Thai): one word each."""
    cp = ord(ch)
    return 0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x0E00 <= cp <= 0x0E7F


def soniox_speaker(value: Any) -> str | None:
    """Soniox's ``"1"``, ``"2"`` -> bolo's ``"A"``, ``"B"``; any other label is kept."""
    if value is None or value == "":
        return None
    text = str(value)
    return speaker_label(int(text) - 1) if text.isdigit() and int(text) > 0 else text


def tokens_to_words(tokens: list[dict[str, Any]]) -> list[tuple[Word, list[dict[str, Any]]]]:
    """Tokens (pieces of words, with times in ms) -> words, each with the tokens it was made of."""
    out: list[tuple[Word, list[dict[str, Any]]]] = []
    text = ""
    parts: list[dict[str, Any]] = []
    starts_word = True
    for tok in tokens:
        piece = str(tok.get("text") or "")
        if not piece:
            continue
        new = starts_word or piece[0].isspace() or _no_space_script(piece.strip()[:1] or " ")
        if new and text.strip():
            out.append((_word(text, parts), parts))
            text, parts = "", []
        text += piece
        parts.append(tok)
        starts_word = piece[-1].isspace() or _no_space_script(piece[-1])
    if text.strip():
        out.append((_word(text, parts), parts))
    return out


def _word(text: str, parts: list[dict[str, Any]]) -> Word:
    timed = [p for p in parts if p.get("start_ms") is not None]
    t0 = float(timed[0].get("start_ms") or 0) / 1000.0 if timed else 0.0
    t1 = float(timed[-1].get("end_ms") or timed[-1].get("start_ms") or 0) / 1000.0 if timed else 0.0
    return Word(text.strip(), t0, max(t0, t1))


def dominant_language(tokens: list[dict[str, Any]]) -> str | None:
    """The language most of the text is in (by characters), from the tokens' ``language``."""
    counts: Counter[str] = Counter()
    for tok in tokens:
        lang = base_language(str(tok.get("language") or ""))
        if lang:
            counts[lang] += len(str(tok.get("text") or "").strip())
    return counts.most_common(1)[0][0] if counts else None


def transcript_to_result(doc: dict[str, Any], fallback_language: str | None, translated: bool) -> STTResult:
    """Soniox's transcript (``text``, ``tokens``) as bolo's result: words, segments cut at a change of speaker
    or the end of a sentence, speaker labels, and the language most of it was spoken in."""
    tokens = [t for t in doc.get("tokens") or [] if isinstance(t, dict) and str(t.get("text") or "") not in MARKERS]
    original = [t for t in tokens if t.get("translation_status") in (None, "", "none", "original")]
    if translated:
        text = "".join(str(t.get("text") or "") for t in tokens if t.get("translation_status") == "translation")
        return STTResult(text.strip(), "en", [], [])
    words = tokens_to_words(original)
    segments: list[Segment] = []
    current: Segment | None = None
    for word, parts in words:
        speaker = soniox_speaker(parts[0].get("speaker"))
        if current is None or (speaker is not None and speaker != current.speaker):
            current = Segment(word.t0, word.t1, "", [], speaker=speaker)
            segments.append(current)
        current.words.append(word)
        current.text = (current.text + "".join(str(p.get("text") or "") for p in parts)).lstrip()
        current.end = max(current.end, word.t1)
        if word.w.endswith(SENTENCE_END):
            current = None
    for seg in segments:
        seg.text = seg.text.strip()
    text = str(doc.get("text") or "").strip() or "".join(str(t.get("text") or "") for t in original).strip()
    language = dominant_language(original) or base_language(fallback_language)
    return STTResult(text, language, [s for s in segments if s.text], [w for w, _ in words])


class Soniox(Adapter):
    kind = "soniox"
    #: Soniox takes up to 5 hours in one request; bolo cuts recordings longer than an hour at pauses
    max_chunk_s = 3600.0
    supports_word_timestamps = True
    supports_translate = True
    supports_diarization = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_chunk_s = float(self.spec.get("max_chunk_s") or 3600.0)
        self.supports_translate = bool(self.spec.get("translate", True))

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.provider.api_key}"} if self.provider.api_key else {}

    def option(self, name: str, default: Any = None) -> Any:
        """A model option, else the provider's option of the same name."""
        if name in self.spec:
            return self.spec[name]
        return self.provider.options.get(name, default)

    def check(self) -> dict[str, Any]:
        doc = self.json(self.request("GET", self.base_url + "/v1/models"))
        models = [str(m.get("id")) for m in doc.get("models") or [] if isinstance(m, dict) and m.get("id")]
        return {"checked": "credentials", "message": f"the key was accepted; {len(models)} models are available"}

    # -- which Soniox model -----------------------------------------------------------------------------------
    @property
    def async_model(self) -> str:
        if self.spec.get("async_model"):
            return str(self.spec["async_model"])
        if self.model.startswith("stt-rt-"):
            return "stt-async-" + self.model[len("stt-rt-") :]
        return self.model or DEFAULT_ASYNC_MODEL

    @property
    def live_model(self) -> str:
        if self.spec.get("live_model"):
            return str(self.spec["live_model"])
        if self.model.startswith("stt-async-"):
            return "stt-rt-" + self.model[len("stt-async-") :]
        return self.model or DEFAULT_LIVE_MODEL

    def _hints(self, language: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        code = base_language(language)
        hints = [code] if code else [str(h) for h in self.option("language_hints") or []]
        if hints:
            out["language_hints"] = hints
            if self.option("language_hints_strict"):
                out["language_hints_strict"] = True
        return out

    # -- recorded audio -----------------------------------------------------------------------------------
    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        translate = req.task == "translate"
        if translate and not self.supports_translate:
            raise InvalidRequest(f"'{self.model}' at {self.where} cannot translate", param="model")
        body: dict[str, Any] = {"model": self.async_model, **self._hints(req.language)}
        if self.option("language_identification", True):
            body["enable_language_identification"] = True
        if req.diarize or self.option("speaker_diarization", False):
            body["enable_speaker_diarization"] = True
        context: dict[str, Any] = {}
        if req.prompt:
            context["text"] = req.prompt
        terms = req.hotword_list()
        if terms:
            context["terms"] = terms
        if self.option("context"):
            extra = self.option("context")
            context.update(extra if isinstance(extra, dict) else {"text": str(extra)})
        if context:
            body["context"] = context
        if translate:
            body["translation"] = {"type": "one_way", "target_language": str(self.option("translate_to", "en"))}
        body.update(self.spec.get("request") or {})

        files = {"file": (req.filename, req.audio, req.content_type)}
        file_id = str(self.json(self.request("POST", self.base_url + "/v1/files", usage, files=files)).get("id") or "")
        if not file_id:
            raise ProviderError(f"{self.where} accepted the upload but returned no file id")
        job_id = ""
        try:
            body["file_id"] = file_id
            job = self.json(self.request("POST", self.base_url + "/v1/transcriptions", usage, json=body))
            job_id = str(job.get("id") or "")
            if not job_id:
                raise ProviderError(f"{self.where} accepted the transcription but returned no id")
            self._wait(job_id, job, req.seconds, usage)
            doc = self.json(self.request("GET", f"{self.base_url}/v1/transcriptions/{job_id}/transcript", usage))
        finally:
            if not self.option("keep_files", False):
                self._cleanup(job_id, file_id)
        return transcript_to_result(doc, req.language, translate)

    def _wait(self, job_id: str, job: dict[str, Any], seconds: float, usage: Usage) -> None:
        """Poll the transcription until it is done (0.25 s, doubling to 2 s between polls)."""
        limit = float(self.option("poll_timeout_s") or max(120.0, 60.0 + seconds))
        deadline = time.monotonic() + limit
        pause = float(self.option("poll_interval_s") or 0.25)
        status = str(job.get("status") or "queued")
        while True:
            if status == "completed":
                return
            if status == "error":
                kind = str(job.get("error_type") or "error")
                message = self._clean(str(job.get("error_message") or "no message"))
                if kind in ("invalid_request", "audio_decode_error", "invalid_audio"):
                    raise InvalidRequest(f"{self.where} refused the audio ({kind}): {message}")
                raise ProviderError(f"{self.where} could not transcribe the audio ({kind}): {message}")
            if time.monotonic() + pause > deadline:
                raise ProviderTimeout(f"{self.where} had not finished the transcription after {limit:.0f} s ({status})")
            time.sleep(pause)
            pause = min(2.0, pause * 2)
            job = self.json(self.request("GET", f"{self.base_url}/v1/transcriptions/{job_id}", usage))
            status = str(job.get("status") or "")

    def _cleanup(self, job_id: str, file_id: str) -> None:
        """Remove the transcription and the uploaded file at Soniox (best effort; not counted as usage)."""
        if job_id:
            with contextlib.suppress(BoloError):
                self.request("DELETE", f"{self.base_url}/v1/transcriptions/{job_id}")
        if file_id:
            with contextlib.suppress(BoloError):
                self.request("DELETE", f"{self.base_url}/v1/files/{file_id}")

    # -- live -----------------------------------------------------------------------------------------------
    def live(self, language: str | None, diarize: bool = False) -> LiveProtocol | None:
        if self.spec.get("live") is False:
            return None
        return SonioxLive(self, language)


class SonioxLive(LiveProtocol):
    """Soniox's live socket. Final tokens arrive once, so they are kept until the speaker stops (``<end>``) or
    the input is finalized (``<fin>``) and then handed on as one final text; non-final tokens are interim text."""

    keepalive_s = 10.0

    def __init__(self, adapter: Soniox, language: str | None) -> None:
        url = adapter.option("realtime_url")
        if not url:
            if adapter.provider.is_default_host:
                url = REALTIME_URL
            else:
                base = adapter.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
                url = base + "/transcribe-websocket"
        self.url = str(url)
        self.headers = adapter.auth_headers()
        self.sample_rate = 16000
        config: dict[str, Any] = {
            "model": adapter.live_model,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            **adapter._hints(language),
            "enable_endpoint_detection": True,
        }
        if adapter.option("endpoint_delay_ms"):
            config["max_endpoint_delay_ms"] = int(adapter.option("endpoint_delay_ms"))
        config.update(adapter.spec.get("live_request") or {})
        self.config = config
        self._final: list[dict[str, Any]] = []

    def start(self) -> str:
        self._final = []
        return json.dumps(self.config)

    def audio(self, pcm: bytes) -> bytes:
        return pcm

    def keepalive(self) -> str:
        return json.dumps({"type": "keepalive"})

    def finalize(self) -> str:
        return json.dumps({"type": "finalize"})

    def close(self) -> str:
        return ""  # an empty text frame ends the stream

    def _flush(self) -> list[LiveEvent]:
        words = tokens_to_words(self._final)
        text = "".join(str(t.get("text") or "") for t in self._final).strip()
        self._final = []
        if not text:
            return []
        return [LiveEvent("final", text, [(w.w, w.t0, w.t1) for w, _ in words])]

    def parse(self, message: bytes | str) -> list[LiveEvent]:
        try:
            doc = json.loads(message)
        except ValueError:
            return []
        if not isinstance(doc, dict):
            return []
        if doc.get("error_code") or doc.get("error_message"):
            kind = doc.get("error_type") or "error"
            return [LiveEvent("error", f"{doc.get('error_code', '')} {kind}: {doc.get('error_message') or ''}".strip())]
        events: list[LiveEvent] = []
        interim: list[str] = []
        for tok in doc.get("tokens") or []:
            if not isinstance(tok, dict) or tok.get("translation_status") == "translation":
                continue
            piece = str(tok.get("text") or "")
            if tok.get("is_final"):
                if piece in MARKERS:
                    events += self._flush()
                    events.append(LiveEvent("endpoint"))
                elif piece:
                    self._final.append(tok)
            elif piece and piece not in MARKERS:
                interim.append(piece)
        if doc.get("finished"):
            events += self._flush()
        elif self._final or interim:
            text = ("".join(str(t.get("text") or "") for t in self._final) + "".join(interim)).strip()
            if text:
                events.append(LiveEvent("interim", text))
        return events
