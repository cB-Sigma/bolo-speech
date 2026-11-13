"""faster-whisper engine (MIT, https://github.com/SYSTRAN/faster-whisper): Whisper on CTranslate2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import (
    Engine,
    EngineStatus,
    LoadOptions,
    Model,
    Segment,
    STTModel,
    STTRequest,
    STTResult,
    Word,
)


class FasterWhisperSTT(STTModel):
    max_chunk_s = 30.0
    supports_word_timestamps = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device = "cuda" if opts.device.startswith("cuda") else "cpu"
        compute = str(spec.get("compute_type", "int8" if device == "cpu" else "float16"))
        self.model = WhisperModel(
            str(path), device=device, compute_type=compute, cpu_threads=opts.threads, num_workers=1
        )
        self.batched = BatchedInferencePipeline(self.model)
        self.max_batch = int(spec.get("max_batch", 8))
        self.beam_size = int(spec.get("beam_size", 1))
        self.fixed_language: str | None = (
            spec.get("language") if spec.get("english_only") or spec.get("language") else None
        )
        if spec.get("english_only"):
            self.fixed_language = "en"
        #: Whisper language token a fine-tune for a language Whisper has no token for was trained with
        #: (Bhojpuri fine-tunes decode as "hi"); requests for the model's own language use it
        self.decode_language: str | None = spec.get("decode_language")
        self.supports_translate = bool(self.model.model.is_multilingual)

    def _language(self, req: STTRequest) -> str | None:
        if self.decode_language:
            return str(self.decode_language)
        lang = req.language or self.fixed_language
        return lang.split("-")[0].lower() if lang else None

    def _one(self, req: STTRequest) -> STTResult:
        segments, info = self.model.transcribe(
            req.samples,
            language=self._language(req),
            task=req.task,
            beam_size=self.beam_size,
            temperature=req.temperature,
            initial_prompt=req.prompt,
            hotwords=req.hotwords,
            word_timestamps=req.word_timestamps,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=not req.word_timestamps,
        )
        out: list[Segment] = []
        words: list[Word] = []
        for seg in segments:
            seg_words = [Word(w.word.strip(), float(w.start), float(w.end)) for w in (seg.words or [])]
            words.extend(seg_words)
            out.append(
                Segment(
                    float(seg.start), float(seg.end), seg.text.strip(), seg_words, seg.avg_logprob, seg.no_speech_prob
                )
            )
        text = " ".join(s.text for s in out).strip()
        return STTResult(text, info.language, out, words)

    def _batch(self, requests: list[STTRequest], language: str, task: str) -> list[STTResult]:
        """Decode several utterances in one batched CTranslate2 call.

        The utterances are laid end to end and handed over as explicit clips,
        which is how the batched pipeline consumes VAD segments.
        """
        sr = self.sample_rate
        gap = np.zeros(int(0.1 * sr), dtype=np.float32)
        parts: list[np.ndarray[Any, Any]] = []
        clips: list[dict[str, float]] = []
        cursor = 0
        for r in requests:
            clips.append({"start": cursor / sr, "end": (cursor + len(r.samples)) / sr})
            parts.extend([r.samples, gap])
            cursor += len(r.samples) + len(gap)
        audio = np.concatenate(parts)
        segments, info = self.batched.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=self.beam_size,
            batch_size=len(requests),
            vad_filter=False,
            clip_timestamps=clips,
            without_timestamps=True,
            condition_on_previous_text=False,
            temperature=requests[0].temperature,
            initial_prompt=requests[0].prompt,
            hotwords=requests[0].hotwords,
        )
        buckets: list[list[Segment]] = [[] for _ in requests]
        for seg in segments:
            mid = (float(seg.start) + float(seg.end)) / 2
            idx = min(
                range(len(clips)),
                key=lambda i: (
                    0.0
                    if clips[i]["start"] <= mid <= clips[i]["end"]
                    else min(abs(mid - clips[i]["start"]), abs(mid - clips[i]["end"]))
                ),
            )
            base = clips[idx]["start"]
            buckets[idx].append(
                Segment(
                    max(0.0, float(seg.start) - base),
                    float(seg.end) - base,
                    seg.text.strip(),
                    [],
                    seg.avg_logprob,
                    seg.no_speech_prob,
                )
            )
        return [STTResult(" ".join(s.text for s in b).strip(), info.language, b, []) for b in buckets]

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        results: list[STTResult | None] = [None] * len(requests)
        groups: dict[tuple[str, str], list[int]] = {}
        for i, r in enumerate(requests):
            lang = self._language(r)
            batchable = lang is not None and not r.word_timestamps and len(r.samples) >= self.sample_rate // 2
            if batchable and len(requests) > 1:
                assert lang is not None
                groups.setdefault((lang, r.task), []).append(i)
            else:
                results[i] = self._one(r)
        for (lang, task), idxs in groups.items():
            if len(idxs) == 1:
                results[idxs[0]] = self._one(requests[idxs[0]])
                continue
            for i, res in zip(idxs, self._batch([requests[i] for i in idxs], lang, task)):
                results[i] = res
        return [r for r in results if r is not None]

    def close(self) -> None:
        del self.batched
        del self.model


class FasterWhisperEngine(Engine):
    id = "faster-whisper"
    name = "faster-whisper (CTranslate2)"
    license = "MIT"
    url = "https://github.com/SYSTRAN/faster-whisper"
    kinds = ("ct2-whisper",)

    def status(self) -> EngineStatus:
        try:
            import faster_whisper

            return EngineStatus(self.id, True, str(faster_whisper.__version__), None)
        except Exception as exc:
            return EngineStatus(
                self.id,
                False,
                None,
                f"{type(exc).__name__}: {exc}. Install with: pip install 'bolo[faster-whisper]'",
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "ct2-whisper":
            raise ValueError(f"faster-whisper engine cannot load kind {kind!r}")
        return FasterWhisperSTT(path, spec, opts)
