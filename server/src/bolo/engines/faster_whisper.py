"""faster-whisper engine (MIT, https://github.com/SYSTRAN/faster-whisper): Whisper on CTranslate2."""

from __future__ import annotations

import copy
import logging
import re
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

log = logging.getLogger(__name__)

#: what CTranslate2 says when CUDA cannot be used at all here (a library missing, no driver, a build without CUDA),
#: as opposed to a CUDA error of the model itself (out of memory), which is raised as it is
_NO_CUDA = re.compile(
    r"cublas|cudnn|cudart|nvrtc|cannot be loaded|is not found|CUDA driver|no CUDA-capable|compiled with CUDA", re.I
)

#: decoding options the batched pipeline applies the same way as one-at-a-time decoding; a request with any other
#: option is decoded on its own
_BATCHABLE = ("repetition_penalty", "no_repeat_ngram_size")


def _temperatures(start: float, increment: float | None) -> float | list[float]:
    """``temperature``, or with a fallback increment the temperatures faster-whisper retries at when a window fails
    ``compression_ratio_threshold`` or ``log_prob_threshold`` (as in OpenAI's Whisper: 0.0, 0.2, ... 1.0)."""
    if not increment:
        return start
    steps = round((1.0 - start) / increment) if start < 1.0 else 0
    return [round(start + i * increment, 4) for i in range(steps + 1)] or [start]


class FasterWhisperSTT(STTModel):
    max_chunk_s = 30.0
    supports_word_timestamps = True
    decode_options = (
        "condition_on_previous_text", "repetition_penalty", "no_repeat_ngram_size", "compression_ratio_threshold",
        "log_prob_threshold", "no_speech_threshold", "temperature_increment_on_fallback",
        "hallucination_silence_threshold", "chunk_length",
    )  # fmt: skip

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device = "cuda" if opts.device.startswith("cuda") else "cpu"
        compute = str(spec.get("compute_type", "int8" if device == "cpu" else "float16"))
        index = int(opts.device.partition(":")[2] or 0) if device == "cuda" else 0

        def build(on: str, compute_type: str) -> Any:
            return WhisperModel(str(path), device=on, compute_type=compute_type, cpu_threads=opts.threads,
                                num_workers=1, device_index=index if on == "cuda" else 0)  # fmt: skip

        #: where the model runs: "cuda" or "cpu" (also after a CUDA load that fell back)
        self.device = device
        if device == "cuda":
            self.model = self._on_cuda(build, compute, spec)
        else:
            self.model = build("cpu", compute)
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

    def _on_cuda(self, build: Any, compute: str, spec: dict[str, Any]) -> Any:
        """The model on CUDA, warmed up so that a CUDA library CTranslate2 loads late (cuBLAS, at the first matrix
        product) is found missing now and not in a request. Missing, the model is loaded again on the CPU, with a
        notice (bolo.cuda_libs.ProviderFallback): requests are served slower, not refused."""
        from bolo import cuda_libs

        cuda_libs.prepare_ct2_cuda()
        try:
            model = build("cuda", compute)
            segments, _info = model.transcribe(
                np.zeros(16000, dtype=np.float32), language="en", beam_size=1, without_timestamps=True,
                condition_on_previous_text=False, vad_filter=False,
            )  # fmt: skip
            list(segments)
            return model
        except (RuntimeError, OSError, ValueError) as exc:
            if not _NO_CUDA.search(str(exc)):
                raise
            error = f"{type(exc).__name__}: {exc}"
        fallback = cuda_libs.ProviderFallback(
            "ctranslate2", "cuda", "cpu", cuda_libs.missing_libraries(error), error, cuda_libs.CT2_CUDA_RECIPE
        )
        log.warning("faster-whisper: %s", fallback.message())
        cuda_libs.record(fallback)
        self.device = "cpu"
        return build("cpu", str(spec.get("cpu_compute_type", "int8")))

    def _language(self, req: STTRequest) -> str | None:
        if self.decode_language:
            return str(self.decode_language)
        lang = req.language or self.fixed_language
        return lang.split("-")[0].lower() if lang else None

    def _one(self, req: STTRequest) -> STTResult:
        opts = req.options
        model = self.model
        kwargs: dict[str, Any] = {
            k: opts[k]
            for k in (
                "repetition_penalty", "no_repeat_ngram_size", "compression_ratio_threshold", "log_prob_threshold",
                "no_speech_threshold", "hallucination_silence_threshold",
            )
            if opts.get(k) is not None
        }  # fmt: skip
        if opts.get("chunk_length"):
            # faster-whisper keeps chunk_length on the model's feature extractor (it changes it for every later
            # call): this request gets its own copy of the model object, which shares the weights
            model = copy.copy(self.model)
            model.feature_extractor = copy.copy(self.model.feature_extractor)
            kwargs["chunk_length"] = int(opts["chunk_length"])
        # the hallucination check needs word timings inside faster-whisper's decoding loop, which decodes with
        # timestamp tokens: with it the text is the same whether or not words are asked for, and the words are
        # faster-whisper's own. Without it, words are aligned afterwards to the text of the plain decode, because
        # asking faster-whisper for words switches it to timestamp-token decoding, which gives other text (on
        # large-v3-turbo: lowercase, no punctuation)
        timed = opts.get("hallucination_silence_threshold") is not None
        align = req.word_timestamps and not timed
        windows: list[Any] = []  # the encoder output of the window being decoded (align only)
        if align:
            model = _capturing_encoder(model, windows)
        segments, info = model.transcribe(
            req.samples,
            language=self._language(req),
            task=req.task,
            beam_size=self.beam_size,
            temperature=_temperatures(req.temperature, opts.get("temperature_increment_on_fallback")),
            initial_prompt=req.prompt,
            hotwords=req.hotwords,
            word_timestamps=timed,
            condition_on_previous_text=bool(opts.get("condition_on_previous_text", False)),
            vad_filter=False,
            without_timestamps=not timed,
            **kwargs,
        )
        aligner = _Aligner(model, info, req.task, len(req.samples) / self.sample_rate) if align else None
        out: list[Segment] = []
        words: list[Word] = []
        for seg in segments:  # faster-whisper decodes a window at a time, as this loop asks for the next segment
            if req.abort.is_set():  # the client went away: stop at this window (the result is discarded)
                break
            if aligner is not None:
                seg_words = aligner.words(seg, windows[-1] if windows else None)
            elif req.word_timestamps:
                seg_words = [_word(w) for w in (seg.words or [])]
            else:
                seg_words = []
            words.extend(seg_words)
            out.append(
                Segment(
                    float(seg.start), float(seg.end), seg.text.strip(), seg_words, seg.avg_logprob, seg.no_speech_prob
                )
            )
        text = " ".join(s.text for s in out).strip()
        return STTResult(text, info.language, out, words, language_probability=_probability(info))

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
            **{k: v for k, v in requests[0].options.items() if k in _BATCHABLE and v is not None},
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
        return [
            STTResult(
                " ".join(s.text for s in b).strip(), info.language, b, [], language_probability=_probability(info)
            )
            for b in buckets
        ]

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        results: list[STTResult | None] = [None] * len(requests)
        groups: dict[tuple[str, str, tuple[tuple[str, Any], ...]], list[int]] = {}
        for i, r in enumerate(requests):
            lang = self._language(r)
            options = tuple(sorted((k, v) for k, v in r.options.items() if v is not None))
            batchable = (
                lang is not None
                and not r.word_timestamps
                and len(r.samples) >= self.sample_rate // 2
                and all(k in _BATCHABLE for k, _ in options)
            )
            if batchable and len(requests) > 1:
                assert lang is not None
                groups.setdefault((lang, r.task, options), []).append(i)
            else:
                results[i] = self._one(r)
        for (lang, task, _options), idxs in groups.items():
            if len(idxs) == 1:
                results[idxs[0]] = self._one(requests[idxs[0]])
                continue
            for i, res in zip(idxs, self._batch([requests[i] for i in idxs], lang, task)):
                results[i] = res
        return [r for r in results if r is not None]

    def close(self) -> None:
        del self.batched
        del self.model


def _word(w: Any) -> Word:
    return Word(w.word.strip(), float(w.start), float(w.end))


def _capturing_encoder(model: Any, windows: list[Any]) -> Any:
    """A copy of ``model`` (sharing the weights) that keeps the encoder output of the window it last encoded.

    faster-whisper encodes each 30-second window just before decoding it and yields that window's segments before
    it encodes the next one, so while a segment is being read, ``windows[-1]`` is the encoder output it came from.
    """
    from faster_whisper import WhisperModel

    clone = copy.copy(model)

    def encode(features: Any) -> Any:
        output = WhisperModel.encode(clone, features)
        windows[:] = [output]
        return output

    clone.encode = encode
    return clone


class _Aligner:
    """Word timings for the segments of a plain (``without_timestamps``) decode, aligned to their own tokens.

    The text is decoded exactly as without word timestamps; each window's tokens are then aligned to that window's
    encoder output with faster-whisper's cross-attention alignment (``WhisperModel.add_word_timestamps``), so the
    words are the transcript's words and the transcript does not change."""

    def __init__(self, model: Any, info: Any, task: str, duration: float) -> None:
        from faster_whisper.tokenizer import Tokenizer

        self.model = model
        self.tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=info.language)
        options = info.transcription_options
        self.prepend = str(options.prepend_punctuations)
        self.append = str(options.append_punctuations)
        self.duration = duration
        self.last_speech = 0.0
        self.last_start = 0.0

    def words(self, seg: Any, encoder_output: Any) -> list[Word]:
        model = self.model
        frames = round((float(seg.end) - float(seg.start)) * model.frames_per_second)
        if encoder_output is None or frames <= 0 or not seg.tokens:
            return []
        subsegment = {"seek": seg.seek, "start": float(seg.start), "end": float(seg.end), "tokens": list(seg.tokens)}
        self.last_speech = model.add_word_timestamps(
            [[subsegment]], self.tokenizer, encoder_output, frames, self.prepend, self.append, self.last_speech
        )
        out: list[Word] = []
        for w in subsegment.get("words", []):
            text = str(w["word"]).strip()
            if not text:
                continue
            # inside the audio and in order: the alignment can run a few frames past the end of a short window
            t0 = min(max(float(w["start"]), self.last_start, 0.0), self.duration)
            t1 = min(max(float(w["end"]), t0), self.duration)
            self.last_start = t0
            out.append(Word(text, t0, t1))
        return out


def _probability(info: Any) -> float | None:
    value = getattr(info, "language_probability", None)
    return round(float(value), 4) if value is not None else None


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
                f"{type(exc).__name__}: {exc}. Install with: pip install 'bolo-speech[faster-whisper]'",
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "ct2-whisper":
            raise ValueError(f"faster-whisper engine cannot load kind {kind!r}")
        return FasterWhisperSTT(path, spec, opts)
