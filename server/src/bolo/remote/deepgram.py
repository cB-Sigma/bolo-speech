"""Deepgram: speech-to-text, pre-recorded and live.

Checked against https://developers.deepgram.com (reference pages listen-pre-recorded and listen-streaming, the
pre-recorded audio guide and the models and languages overview) on 2026-11-14:

* ``POST https://api.deepgram.com/v1/listen?model=nova-3&language=hi`` with the audio as the request body
  (``Content-Type: audio/wav``). Query: ``model``, ``language`` (BCP-47, or ``multi`` with Nova-3),
  ``detect_language``, ``smart_format``, ``punctuate``, ``utterances``, ``keyterm`` (Nova-3; repeated) or
  ``keywords`` (older models). Answer: ``results.channels[0].alternatives[0]`` (``transcript``, ``words`` with
  ``word``, ``punctuated_word``, ``start``, ``end``), ``results.channels[0].detected_language``,
  ``results.utterances``, ``metadata.duration``.
* ``wss://api.deepgram.com/v1/listen`` (live): query ``encoding=linear16&sample_rate=16000&channels=1``,
  ``interim_results``, ``endpointing`` (ms), ``smart_format``; binary audio frames in, JSON control frames
  ``KeepAlive``, ``Finalize``, ``CloseStream``; ``Results`` messages (``channel.alternatives[0].transcript``,
  ``is_final``, ``speech_final``) and ``UtteranceEnd`` out.
* ``Authorization: Token <key>``. ``GET https://api.deepgram.com/v1/projects`` lists the key's projects.

Not done: Deepgram's text-to-speech (Aura) and its Flux turn-taking models.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

from bolo.engines.base import Segment, STTResult, Word
from bolo.errors import InvalidRequest
from bolo.providers import Usage
from bolo.remote.base import Adapter, LiveEvent, LiveProtocol, RemoteRequest, base_language


def _words(items: Any) -> list[Word]:
    return [
        Word(
            str(w.get("punctuated_word") or w.get("word") or ""),
            float(w.get("start") or 0.0),
            float(w.get("end") or 0.0),
            float(w["confidence"]) if isinstance(w.get("confidence"), (int, float)) else None,
        )
        for w in items or []
        if isinstance(w, dict)
    ]


class Deepgram(Adapter):
    kind = "deepgram"
    supports_word_timestamps = True

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.provider.api_key}"} if self.provider.api_key else {}

    def check(self) -> dict[str, Any]:
        doc = self.json(self.request("GET", self.base_url + "/v1/projects"))
        count = len(doc.get("projects") or [])
        return {"checked": "credentials", "message": f"the key was accepted; it belongs to {count} project(s)"}

    def _params(self, language: str | None, terms: list[str]) -> list[tuple[str, str]]:
        spec = self.spec
        params: list[tuple[str, str]] = [("model", self.model)]
        if language:
            # Deepgram takes regional tags (en-IN, pt-BR); a bare code is sent as it is
            params.append(("language", language.replace("_", "-")))
        elif spec.get("language"):
            params.append(("language", str(spec["language"])))
        elif spec.get("detect_language", True):
            params.append(("detect_language", "true"))
        if spec.get("smart_format", True):
            params.append(("smart_format", "true"))
        field = str(spec.get("hotwords_field") or ("keyterm" if self.model.startswith("nova-3") else "keywords"))
        params.extend((field, t) for t in terms[:100])
        params.extend((str(k), str(v)) for k, v in (spec.get("request") or {}).items())
        return params

    def transcribe(self, req: RemoteRequest, usage: Usage) -> STTResult:
        if req.task == "translate":
            raise InvalidRequest(f"'{self.model}' at {self.where} cannot translate", param="model")
        if req.prompt:
            self.unsupported("prompt", "a prompt")
        params = [*self._params(req.language, req.hotword_list()), ("utterances", "true")]
        doc = self.json(
            self.request(
                "POST",
                self.base_url + "/v1/listen",
                usage,
                params=params,
                content=req.audio,
                headers={"Content-Type": req.content_type},
            )
        )
        raw = doc.get("results")
        results: dict[str, Any] = raw if isinstance(raw, dict) else {}
        channels = results.get("channels") or [{}]
        channel = channels[0] if isinstance(channels[0], dict) else {}
        alt = (channel.get("alternatives") or [{}])[0]
        words = _words(alt.get("words"))
        segments = [
            Segment(float(u.get("start") or 0.0), float(u.get("end") or 0.0), str(u.get("transcript") or "").strip())
            for u in results.get("utterances") or []
            if isinstance(u, dict) and str(u.get("transcript") or "").strip()
        ]
        for seg in segments:
            seg.words = [w for w in words if seg.start - 0.01 <= w.t0 < seg.end + 0.01]
        language = base_language(str(channel.get("detected_language") or "")) or base_language(req.language)
        return STTResult(str(alt.get("transcript") or "").strip(), language, segments, words)

    def live(self, language: str | None) -> LiveProtocol | None:
        if self.spec.get("live") is False:
            return None
        return DeepgramLive(self, language)


class DeepgramLive(LiveProtocol):
    keepalive_s = 4.0

    def __init__(self, adapter: Deepgram, language: str | None) -> None:
        base = adapter.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        params = [p for p in adapter._params(language, []) if p[0] != "detect_language"]
        params += [
            ("encoding", "linear16"),
            ("sample_rate", "16000"),
            ("channels", "1"),
            ("interim_results", "true"),
            ("endpointing", str(int(adapter.spec.get("endpointing_ms") or 300))),
        ]
        self.url = f"{base}/v1/listen?{urlencode(params)}"
        self.headers = adapter.auth_headers()
        self.sample_rate = 16000

    def audio(self, pcm: bytes) -> bytes:
        return pcm

    def keepalive(self) -> str:
        return json.dumps({"type": "KeepAlive"})

    def finalize(self) -> str:
        return json.dumps({"type": "Finalize"})

    def close(self) -> str:
        return json.dumps({"type": "CloseStream"})

    def parse(self, message: bytes | str) -> list[LiveEvent]:
        try:
            doc = json.loads(message)
        except ValueError:
            return []
        if not isinstance(doc, dict):
            return []
        kind = doc.get("type")
        if kind == "Results":
            alt = ((doc.get("channel") or {}).get("alternatives") or [{}])[0]
            text = str(alt.get("transcript") or "").strip()
            events: list[LiveEvent] = []
            if doc.get("is_final"):
                if text:
                    words = [(w.w, w.t0, w.t1) for w in _words(alt.get("words"))]
                    events.append(LiveEvent("final", text, words))
            elif text:
                events.append(LiveEvent("interim", text))
            if doc.get("speech_final") or doc.get("from_finalize"):
                events.append(LiveEvent("endpoint"))
            return events
        if kind == "UtteranceEnd":
            return [LiveEvent("endpoint")]
        if kind == "Error":
            return [LiveEvent("error", str(doc.get("description") or doc.get("message") or "error"))]
        return []
