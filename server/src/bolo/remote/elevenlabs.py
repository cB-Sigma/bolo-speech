"""ElevenLabs: text-to-speech and speech-to-text (Scribe).

Checked against https://elevenlabs.io/docs/api-reference (text-to-speech/stream, speech-to-text/convert,
voices/search) and the models overview on 2026-11-14:

* ``POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_24000`` (JSON):
  ``text``, ``model_id``, ``language_code`` (ISO 639-1; not for ``eleven_multilingual_v2``),
  ``voice_settings.speed``, ``apply_text_normalization`` (``auto``, ``on``, ``off``). The answer is the audio,
  streamed; ``pcm_*`` formats are 16-bit mono (8, 16, 22.05, 24, 32, 44.1 and 48 kHz; 44.1 kHz needs a Pro plan).
* ``POST https://api.elevenlabs.io/v1/speech-to-text`` (multipart): ``model_id`` (``scribe_v3``,
  ``scribe_v2``), ``file``, ``language_code``, ``timestamps_granularity`` (``word``), ``keyterms``,
  ``temperature``, ``diarize`` and ``num_speakers`` (up to 32). Answer: ``text``, ``language_code``, ``words``
  (``text``, ``start``, ``end``, ``type``, and with ``diarize`` ``speaker_id``: ``speaker_0``, ``speaker_1``, …).
* ``wss://api.elevenlabs.io/v1/speech-to-text/realtime`` (checked against the API reference and the models
  overview on 2026-11-21): the key in the ``xi-api-key`` header, the session as query parameters (``model_id``:
  ``scribe_v3_live`` or ``scribe_v2_realtime``; ``audio_format`` ``pcm_16000``, ``language_code``,
  ``commit_strategy`` ``manual`` / ``vad`` / ``turn_prediction`` (the last for models that predict turns, Scribe v3
  Live), ``include_timestamps``, ``diarize``, ``keyterms``), audio as JSON ``input_audio_chunk`` messages
  (``audio_base_64``, ``commit``, ``sample_rate``; ``commit: true`` ends the segment). Answers: ``session_started``,
  ``partial_transcript``, ``committed_transcript`` and, with ``include_timestamps``,
  ``committed_transcript_with_timestamps`` (``text``, ``words`` with ``start`` / ``end`` in seconds); with
  ``diarize``, ``diarized_transcript`` after each commit (``segment_index``, ``text``, ``words`` with their final
  ``speaker_id``; the committed transcript never waits for it) and ``speakers_merged`` (``from_speaker_id``,
  ``to_speaker_id``: apply in order); errors as ``message_type`` ``error``, ``auth_error``, ``quota_exceeded``, …
  with ``error``.
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

import base64
import json
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote, urlencode

from bolo.diarize import speaker_label
from bolo.engines.base import FloatArray, Segment, STTResult, Voice, Word
from bolo.errors import BoloError, InvalidRequest, ProviderError
from bolo.providers import Usage
from bolo.remote.base import (
    Adapter,
    LiveEvent,
    LiveProtocol,
    RemoteRequest,
    base_language,
    decode_whole,
    float_to_pcm16,
    read_pcm_stream,
)

PCM_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
#: recorded audio of a live model goes to the recorded model of the same generation
ASYNC_OF_LIVE = {"scribe_v3_live": "scribe_v3", "scribe_v2_realtime": "scribe_v2"}
#: message types of the realtime socket that report a failure
LIVE_ERRORS = ("error", "auth_error", "quota_exceeded")


def scribe_speaker(value: Any, merged: dict[str, str] | None = None) -> str | None:
    """ElevenLabs' ``speaker_0``, ``speaker_1`` -> bolo's ``A``, ``B`` (after merges); other labels are kept."""
    if value is None or value == "":
        return None
    text = str(value)
    while merged and text in merged:
        text = merged[text]
    number = text.rsplit("_", 1)[-1]
    return speaker_label(int(number)) if text.startswith("speaker_") and number.isdigit() else text


def scribe_words(items: Any) -> list[tuple[Word, str | None]]:
    """The ``word`` items of a Scribe answer (spacing and audio events left out), each with its raw speaker id."""
    out: list[tuple[Word, str | None]] = []
    for w in items or []:
        if not isinstance(w, dict) or w.get("type", "word") != "word" or not str(w.get("text", "")).strip():
            continue
        word = Word(str(w["text"]).strip(), float(w.get("start") or 0.0), float(w.get("end") or 0.0))
        out.append((word, str(w["speaker_id"]) if w.get("speaker_id") is not None else None))
    return out


def speaker_segments(words: list[tuple[Word, str | None]]) -> list[Segment]:
    """Words with speakers -> one segment per run of the same speaker."""
    segments: list[Segment] = []
    for word, raw in words:
        speaker = scribe_speaker(raw)
        if not segments or segments[-1].speaker != speaker:
            segments.append(Segment(word.t0, word.t1, "", [], speaker=speaker))
        seg = segments[-1]
        seg.words.append(word)
        seg.text = f"{seg.text} {word.w}".strip()
        seg.end = max(seg.end, word.t1)
    return segments


class ElevenLabs(Adapter):
    kind = "elevenlabs"
    supports_word_timestamps = True
    #: Scribe labels speakers (``diarize``), in recorded audio and on its realtime socket
    supports_diarization = True
    supports_live_diarization = True

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
        instructions: str | None = None,
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
        model = str(self.spec.get("async_model") or ASYNC_OF_LIVE.get(self.model, self.model))
        data: dict[str, Any] = {"model_id": model, "timestamps_granularity": "word"}
        diarize = req.diarize or bool(self.spec.get("speaker_labels"))
        if diarize:
            data["diarize"] = "true"
            if self.spec.get("num_speakers"):
                data["num_speakers"] = str(int(self.spec["num_speakers"]))
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
        words = scribe_words(doc.get("words"))
        language = base_language(str(doc.get("language_code") or "")) or base_language(req.language)
        # ISO 639-3 answers ("eng", "hin") stay as they are; clients that need 639-1 send the language
        segments = speaker_segments(words) if diarize and any(raw for _, raw in words) else []
        return STTResult(str(doc.get("text") or "").strip(), language, segments, [w for w, _ in words])

    # -- live (the realtime socket) -------------------------------------------------------------------------
    def live(self, language: str | None, diarize: bool = False) -> LiveProtocol | None:
        if self.spec.get("live") is False or (self.model not in ASYNC_OF_LIVE and not self.spec.get("live")):
            return None
        return ElevenLabsLive(self, language, diarize)

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


class ElevenLabsLive(LiveProtocol):
    """ElevenLabs' realtime speech-to-text socket (Scribe v3 Live, Scribe v2 Realtime). ``partial_transcript`` is
    interim text; a segment is final with its ``committed_transcript_with_timestamps`` (asked for always), which
    ElevenLabs sends when its commit strategy ends the segment (``turn_prediction`` for Scribe v3 Live: the model
    predicts that the speaker finished their turn) or when bolo commits at the end of input. With speaker labels
    the segment is held until its ``diarized_transcript`` (the final speakers) arrives, and handed on unlabelled if
    the next segment is committed first."""

    def __init__(self, adapter: ElevenLabs, language: str | None, diarize: bool) -> None:
        spec, provider = adapter.spec, adapter.provider
        url = spec.get("realtime_url") or provider.options.get("realtime_url")
        if not url:
            base = adapter.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            url = base + "/v1/speech-to-text/realtime"
        self.sample_rate = 16000
        strategy = spec.get("commit_strategy") or ("turn_prediction" if adapter.model == "scribe_v3_live" else "vad")
        params: dict[str, Any] = {
            "model_id": adapter.model,
            "audio_format": f"pcm_{self.sample_rate}",
            "commit_strategy": strategy,
            "include_timestamps": "true",
        }
        code = base_language(language)
        if code:
            params["language_code"] = code
        self.diarize = diarize or bool(spec.get("live_speaker_labels"))
        if self.diarize:
            params["diarize"] = "true"
        params.update(spec.get("live_request") or {})
        self.url = f"{url}?{urlencode(params)}"
        self.headers = adapter.auth_headers()
        self.params = params
        self._held: tuple[str, list[tuple[str, float, float]]] | None = None
        self._merged: dict[str, str] = {}

    def _chunk(self, pcm: bytes, commit: bool) -> str:
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode("ascii"),
                "commit": commit,
                "sample_rate": self.sample_rate,
            }
        )

    def audio(self, pcm: bytes) -> str:
        return self._chunk(pcm, False)

    def finalize(self) -> str:
        return self._chunk(b"", True)  # commit what was sent so far

    def _flush_held(self, speaker: str | None = None) -> list[LiveEvent]:
        held, self._held = self._held, None
        if held is None:
            return []
        return [LiveEvent("final", held[0], held[1], speaker), LiveEvent("endpoint")]

    def parse(self, message: bytes | str) -> list[LiveEvent]:
        try:
            doc = json.loads(message)
        except ValueError:
            return []
        if not isinstance(doc, dict):
            return []
        kind = str(doc.get("message_type") or "")
        if kind in LIVE_ERRORS or (kind.endswith("_error") or kind.endswith("_exceeded")):
            return [LiveEvent("error", f"{kind}: {doc.get('error') or doc.get('message') or ''}".strip())]
        if kind == "partial_transcript":
            text = str(doc.get("text") or "").strip()
            return [LiveEvent("interim", text)] if text else []
        if kind == "speakers_merged":
            if doc.get("from_speaker_id") and doc.get("to_speaker_id"):
                self._merged[str(doc["from_speaker_id"])] = str(doc["to_speaker_id"])
            return []
        if kind == "committed_transcript_with_timestamps":
            text = str(doc.get("text") or "").strip()
            words = [(w.w, w.t0, w.t1) for w, _ in scribe_words(doc.get("words"))]
            events = self._flush_held()  # its diarized transcript did not come: handed on unlabelled
            if not text:
                return [*events, LiveEvent("endpoint")]
            if self.diarize:
                self._held = (text, words)
                return events
            return [*events, LiveEvent("final", text, words), LiveEvent("endpoint")]
        if kind == "diarized_transcript" and self._held is not None:
            speakers = [scribe_speaker(raw, self._merged) for _, raw in scribe_words(doc.get("words"))]
            counted = Counter(s for s in speakers if s)
            return self._flush_held(counted.most_common(1)[0][0] if counted else None)
        return []
