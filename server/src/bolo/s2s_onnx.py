"""LFM3-Audio speech-to-speech in ONNX Runtime and NumPy, without PyTorch: an in-process S2S backend.

Liquid's LFM3-Audio-2B (``LiquidAI/LFM3-Audio-2B-ONNX``) takes text or 16 kHz speech and answers with text,
24 kHz speech, or both interleaved, and can call tools. This module runs its four ONNX graphs and reproduces the
generation procedure of the model's own code (``modeling_lfm3_audio.py``, ``feature_extraction_lfm3_audio.py``
and ``modeling_liquid_codec.py`` of ``LiquidAI/LFM3-Audio-2B``), step by step:

* features: 128-band log-mel of 16 kHz audio (25 ms periodic Hann window every 10 ms, no centering, Slaney
  filters, ``log10`` clamped at -12, then ``(x + 4) / 4``); a clip is right-padded with zeros to whole frames
  (:func:`log_mel`). Every 8 frames are one audio token of the language model.
* ``audio_encoder.onnx``: log-mel -> one 2048-wide row per audio token (encoder and projector). The encoder is
  causal, so the rows of the audio heard so far never change: while the user is still speaking, the finished
  rows are fed to the decoder (prefill), and only the last rows and the turn's closing text are left for the
  moment the turn ends.
* ``decoder*.onnx``: the LFM3 language model with its caches as inputs and outputs (a 7-tap window per
  convolution layer, keys and values per attention layer). ``input_ids`` holds text tokens, ``<|audio|>``
  placeholders (their rows come from ``audio_features``) and ``<|audio_out|>`` placeholders (their rows are the
  summed codebook embeddings of ``audio_out_codes``, one generated speech frame each). It returns the logits of
  the next text token and the final hidden state.
* ``depth_decoder.onnx``: from the hidden state, the 12 codes of the next 80 ms speech frame, MaskGIT style: the
  codebooks are revealed in a random order, 4 per pass, each pass scoring the codebooks still masked
  (:func:`sample_audio_frame`). A frame holding the end-of-audio code (2016) ends the speech.
* ``codec_decoder.onnx`` + inverse STFT: codes -> polar STFT frames (6 per speech frame) -> 24 kHz audio by
  overlap-add (1280-sample periodic Hann window, hop 320). The codec decoder is causal with a receptive field
  of 20 frames, so audio is decoded in small chunks behind a window of earlier frames
  (:class:`CodecStream`) and comes out as it is generated.
* the mode machine of ``generate()``: text by default; ``<|audio_start|>`` switches to speech until an
  end-of-audio frame; ``<|mixed_start|>`` to blocks of 6 text tokens and 12 speech frames until
  ``<|text_end|>``, after which the rest of the speech follows up to its end-of-audio frame; ``<|im_end|>`` in a
  text slot ends the response (:meth:`Lfm3Session.respond`).
* sampling: greedy text; speech codes with temperature 1.0, top-k 128 and a repetition penalty of 1.1 over the
  codes each codebook took in the last 4096 frames (the model card's values, read from
  ``generation_config.json``).
* chat format: the model's ``chat_template.jinja`` written out for the turns a Realtime session has: system
  prompt with the tool definitions (one JSON object per line inside ``<tools>``), user turns of audio and/or
  text, assistant turns, tool results in ``<tool_response>``. Tool calls
  (``<tool_call><function=name><parameter=key>value</parameter></function></tool_call>``) are parsed out of
  the text stream.

The conversation is append-only, as in llama.cpp's realtime server: every turn and response stays in the
decoder's cache, nothing is evaluated twice.

Classes: :class:`Lfm3Model` (graphs, tokenizer and settings; one per model directory, shared by all
conversations and by the text-to-speech model), :class:`Lfm3Session` (one conversation, synchronous),
:class:`OnnxS2SConversation` and :class:`OnnxS2SBackend` (the :mod:`bolo.s2s` interface around a session, with a
worker thread per conversation), and :class:`Lfm3Tts` (the same graphs as a text-to-speech model of the
``onnxruntime`` engine, kind ``lfm3_audio_onnx``). The graphs are behind the :class:`Graphs` protocol so tests run
the whole procedure against scripted stand-ins.

Needs ``onnxruntime`` and ``tokenizers`` (``pip install 'bolo-speech[s2s]'``). Settings (catalog ``options``):
``precision`` (``q4``, the default, ``q8`` or ``fp32`` on a CPU; ``q4f16``, ``q8f16`` or ``fp16`` on a GPU),
``seed`` (speech sampling; unset: different every time), ``prefill_s``, ``chunk_frames``, ``first_chunk_frames``,
``codec_context``, ``max_context``, ``max_response_slots`` and ``reasoning`` (see :class:`Lfm3Session`), and the
sampling values of ``generation_config.json`` (``audio_temperature``, ``audio_top_k``,
``audio_repetition_penalty``, ``audio_repetition_window``, ``temperature``, ``top_k``, ``repetition_penalty``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from bolo.engines.base import FloatArray, LoadOptions, SynthesisExtras, TTSModel, Voice
from bolo.errors import ComponentMissing, EngineUnavailable
from bolo.ort_providers import providers_for
from bolo.s2s import (
    AudioDelta,
    ResponseEnd,
    S2SBackend,
    S2SConfig,
    S2SConversation,
    S2SEvent,
    TextDelta,
    ToolCall,
)

IntArray = npt.NDArray[np.int64]

log = logging.getLogger("bolo.s2s.onnx")

#: the model ``kind`` of the catalog entries this module runs
KIND = "lfm3_audio_onnx"
INSTALL_HINT = "pip install 'bolo-speech[s2s]'"


#: system prompts the model was trained with (model card)
SYSTEM_MIXED = "Respond with interleaved text and audio."
SYSTEM_MIXED_TOOLS = "Respond with function calls, and interleaved text and audio."
SYSTEM_TOOLS_SUFFIX = "List of tools:"


# -- settings read from the model directory -----------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSettings:
    """``preprocessor_config.json``: the log-mel front end of the audio encoder."""

    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    log_floor: float = -12.0
    offset: float = 4.0
    divisor: float = 4.0
    #: mel frames per audio token
    encoder_stride: int = 8

    @classmethod
    def from_dir(cls, path: Path) -> FeatureSettings:
        doc = json.loads((path / "preprocessor_config.json").read_text())
        return cls(
            sample_rate=int(doc.get("sampling_rate", 16000)),
            n_mels=int(doc.get("feature_size", 128)),
            n_fft=int(doc.get("n_fft", 400)),
            win_length=int(doc.get("win_length", 400)),
            hop_length=int(doc.get("hop_length", 160)),
            log_floor=float(doc.get("log_floor", -12.0)),
            offset=float(doc.get("normalization_offset", 4.0)),
            divisor=float(doc.get("normalization_divisor", 4.0)),
            encoder_stride=int(doc.get("encoder_stride", 8)),
        )

    def pad_length(self, num_samples: int) -> int:
        """Samples after right-padding a clip so that no trailing sample is dropped."""
        if num_samples < self.n_fft:
            return self.n_fft
        return num_samples + (-(num_samples - self.n_fft) % self.hop_length)

    def complete_frames(self, num_samples: int) -> int:
        """Mel frames whose window lies inside the samples (no padding)."""
        return 0 if num_samples < self.n_fft else 1 + (num_samples - self.n_fft) // self.hop_length

    def num_tokens(self, num_frames: int) -> int:
        return -(num_frames // -self.encoder_stride)


@dataclass(frozen=True)
class CodecSettings:
    """``audio_tokenizer/config.json``: the inverse STFT after the codec decoder."""

    sample_rate: int = 24000
    n_fft: int = 1280
    hop_length: int = 320
    win_length: int = 1280
    #: STFT frames per speech frame
    frame_stack: int = 6
    codebooks: int = 12

    @classmethod
    def from_dir(cls, path: Path) -> CodecSettings:
        doc = json.loads((path / "audio_tokenizer" / "config.json").read_text())
        return cls(
            sample_rate=int(doc.get("sampling_rate", 24000)),
            n_fft=int(doc.get("stft_n_fft", 1280)),
            hop_length=int(doc.get("stft_hop_length", 320)),
            win_length=int(doc.get("stft_win_length", 1280)),
            frame_stack=int(doc.get("stft_frame_stack", 6)),
            codebooks=int(doc.get("n_q", 12)),
        )

    @property
    def frame_samples(self) -> int:
        return self.frame_stack * self.hop_length


@dataclass
class GenerationSettings:
    """``config.json`` and ``generation_config.json``: the mode machine and the sampling."""

    codebooks: int = 12
    codebook_size: int = 2017
    maskgit_schedule: tuple[int, ...] = (4, 4, 4)
    n_text: int = 6
    n_audio: int = 12
    eos_ids: tuple[int, ...] = (124900, 125014)
    #: text sampling: temperature 0 is greedy
    temperature: float = 0.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    text_repetition_window: int = 32
    audio_temperature: float = 1.0
    audio_top_k: int = 128
    audio_repetition_penalty: float = 1.1
    audio_repetition_window: int = 4096

    @property
    def audio_eos_code(self) -> int:
        return self.codebook_size - 1

    @classmethod
    def from_dir(cls, path: Path) -> GenerationSettings:
        config = json.loads((path / "config.json").read_text())
        gen = (
            json.loads((path / "generation_config.json").read_text())
            if (path / "generation_config.json").exists()
            else {}
        )
        codebooks = int(config.get("codebooks", 12))
        schedule = config.get("maskgit_schedule")
        if not schedule:
            steps = int(config.get("maskgit_steps", 3))
            schedule = [codebooks // steps] * steps
            schedule[-1] += codebooks - sum(schedule)
        eos = gen.get("eos_token_id", [124900, 125014])
        return cls(
            codebooks=codebooks,
            codebook_size=int(config.get("audio_vocab_size", 2017)),
            maskgit_schedule=tuple(int(n) for n in schedule),
            n_text=int(config.get("interleaved_n_text", 6)),
            n_audio=int(config.get("interleaved_n_audio", 12)),
            eos_ids=tuple(eos) if isinstance(eos, list) else (int(eos),),
            temperature=float(gen.get("temperature") or 0.0) if gen.get("do_sample") else 0.0,
            top_k=int(gen.get("top_k") or 0) if gen.get("do_sample") else 0,
            repetition_penalty=float(gen.get("repetition_penalty") or 1.0),
            audio_temperature=float(gen.get("audio_temperature", 1.0)),
            audio_top_k=int(gen.get("audio_top_k", 128)),
            audio_repetition_penalty=float(gen.get("audio_repetition_penalty", 1.1)),
            audio_repetition_window=int(gen.get("audio_repetition_window", 4096)),
        )


@dataclass(frozen=True)
class Tokens:
    """Ids of the control tokens (looked up in ``tokenizer.json``)."""

    bos: int
    im_start: int
    im_end: int
    audio: int
    audio_out: int
    audio_start: int
    text_end: int
    mixed_start: int
    tool_call: int

    @classmethod
    def from_lookup(cls, lookup: Callable[[str], int | None]) -> Tokens:
        names = {
            "bos": "<|startoftext|>", "im_start": "<|im_start|>", "im_end": "<|im_end|>", "audio": "<|audio|>",
            "audio_out": "<|audio_out|>", "audio_start": "<|audio_start|>", "text_end": "<|text_end|>",
            "mixed_start": "<|mixed_start|>", "tool_call": "<tool_call>",
        }  # fmt: skip
        ids: dict[str, int] = {}
        for key, name in names.items():
            token = lookup(name)
            if token is None:
                raise ValueError(f"the tokenizer has no {name} token: not an LFM3-Audio tokenizer")
            ids[key] = int(token)
        return cls(**ids)


# -- audio front end ----------------------------------------------------------------------------------------------
def _hz_to_mel(freq: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    mel = 3.0 * freq / 200.0
    log_region = freq >= 1000.0
    mel[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (27.0 / np.log(6.4))
    return mel


def _mel_to_hz(mel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    freq = 200.0 * mel / 3.0
    log_region = mel >= 15.0
    freq[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel[log_region] - 15.0))
    return freq


def mel_filter_bank(n_freq: int, n_mels: int, sample_rate: int) -> npt.NDArray[np.float64]:
    """Triangular Slaney filters with Slaney area normalisation over 0 Hz to Nyquist, ``(n_freq, n_mels)``
    (``transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")``)."""
    edges = _mel_to_hz(np.linspace(0.0, _hz_to_mel(np.array([sample_rate / 2.0]))[0], n_mels + 2))
    fft_freqs = np.linspace(0.0, sample_rate // 2, n_freq)
    widths = np.diff(edges)
    slopes = edges[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= 2.0 / (edges[2 : n_mels + 2] - edges[:n_mels])
    return filters


class LogMel:
    """The model's log-mel features, ``(n_mels, frames)``."""

    def __init__(self, settings: FeatureSettings) -> None:
        self.settings = settings
        n = settings.win_length
        self.window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)  # periodic Hann
        self.filters = mel_filter_bank(settings.n_fft // 2 + 1, settings.n_mels, settings.sample_rate)

    def __call__(self, samples: FloatArray, *, pad: bool = True) -> FloatArray:
        """``pad``: right-pad the clip to whole frames (a finished clip); without it only complete frames."""
        s = self.settings
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if pad:
            x = np.pad(x, (0, s.pad_length(len(x)) - len(x)))
        count = s.complete_frames(len(x))
        if count == 0:
            return np.zeros((s.n_mels, 0), dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(x, s.n_fft)[:: s.hop_length][:count]
        power = np.abs(np.fft.rfft(frames * self.window, axis=1)) ** 2
        with np.errstate(divide="ignore"):
            mel = np.log10(power @ self.filters)
        mel = (np.maximum(mel, s.log_floor) + s.offset) / s.divisor
        out: FloatArray = np.ascontiguousarray(mel.T, dtype=np.float32)
        return out


# -- the graphs ---------------------------------------------------------------------------------------------------
@dataclass
class DecoderState:
    """The language model's caches (graph input name -> array) and the number of positions they cover."""

    caches: dict[str, Any]
    length: int = 0


class Graphs(Protocol):
    """What the procedure needs from the four graphs (tests pass a scripted stand-in)."""

    def empty_state(self) -> DecoderState: ...

    def encode_audio(self, features: FloatArray) -> FloatArray:
        """Log-mel ``(n_mels, frames)`` -> ``(tokens, hidden)``."""

    def decode(
        self,
        state: DecoderState,
        input_ids: Sequence[int],
        audio_features: FloatArray | None = None,
        audio_out_codes: IntArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Append positions to the state. Returns the text logits and the hidden state of the last position."""

    def depth(self, hidden: FloatArray, codes: IntArray, masked: npt.NDArray[np.bool_]) -> FloatArray:
        """One pass of the depth decoder for one frame: ``(codebooks, codebook_size)`` logits."""

    def codec(self, codes: IntArray) -> FloatArray:
        """Speech frames ``(frames, codebooks)`` -> ``(frames * 6, 2 * (n_fft // 2 + 1))`` polar STFT rows:
        log-magnitudes, then phases."""


#: bundles of the model card: the decoder file with the encoder and codec decoder it is paired with
PRECISIONS: dict[str, tuple[str, str, str]] = {
    "q4": ("decoder_q4.onnx", "audio_encoder.onnx", "codec_decoder.onnx"),
    "q8": ("decoder_q8.onnx", "audio_encoder.onnx", "codec_decoder.onnx"),
    "fp32": ("decoder.onnx", "audio_encoder.onnx", "codec_decoder.onnx"),
    "q4f16": ("decoder_q4f16.onnx", "audio_encoder_fp16.onnx", "codec_decoder_fp16.onnx"),
    "q8f16": ("decoder_q8f16.onnx", "audio_encoder_fp16.onnx", "codec_decoder_fp16.onnx"),
    "fp16": ("decoder_fp16.onnx", "audio_encoder_fp16.onnx", "codec_decoder_fp16.onnx"),
}


class OnnxGraphs:
    """The four graphs in ONNX Runtime sessions. ``run`` is thread-safe; conversations share one instance."""

    def __init__(
        self,
        path: Path,
        *,
        precision: str = "q4",
        threads: int = 4,
        spinning: bool = False,
        providers: Sequence[Any] | None = None,
        graph_providers: dict[str, Sequence[Any]] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ComponentMissing(f"LFM3-Audio in ONNX needs onnxruntime: {INSTALL_HINT}") from exc
        if precision not in PRECISIONS:
            raise EngineUnavailable(f"unknown precision {precision!r}: one of {', '.join(PRECISIONS)}")
        decoder, encoder, codec = PRECISIONS[precision]
        onnx_dir = path / "onnx"
        for name in (decoder, encoder, codec, "depth_decoder.onnx"):
            if not (onnx_dir / name).exists():
                raise EngineUnavailable(f"{onnx_dir / name} is missing (precision {precision}); pull the model again")

        def session(name: str, n_threads: int, role: str) -> Any:
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, n_threads)
            options.inter_op_num_threads = 1
            if not spinning:
                # worker threads that spin while idle take the cores from the other graphs and from every other
                # process of a shared machine
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            chosen = (graph_providers or {}).get(role) or providers or ["CPUExecutionProvider"]
            return ort.InferenceSession(str(onnx_dir / name), sess_options=options, providers=list(chosen))

        # each graph can sit on its own device (bolo.placement places the largest on the GPU while they fit)
        self.decoder = session(decoder, threads, "decoder")
        self.encoder = session(encoder, threads, "encoder")
        self.depth_decoder = session("depth_decoder.onnx", threads, "depth_decoder")
        self.codec_decoder = session(codec, threads, "codec_decoder")
        self._cache_inputs = [
            (i.name, i.shape, np.float16 if "float16" in i.type else np.float32)
            for i in self.decoder.get_inputs()
            if i.name.startswith(("past_conv.", "past_key_values."))
        ]
        self._outputs = [o.name for o in self.decoder.get_outputs()]
        self._renames = {
            name: name.replace("present_conv.", "past_conv.").replace("present.", "past_key_values.")
            for name in self._outputs
            if name.startswith("present")
        }
        hidden = next(i for i in self.decoder.get_inputs() if i.name == "audio_features").shape[-1]
        self.hidden_size = int(hidden)
        self._no_features = np.zeros((0, self.hidden_size), dtype=np.float32)
        self._no_codes = np.zeros((0, 12), dtype=np.int64)
        self._keep_last = np.array([1], dtype=np.int64)

    def empty_state(self) -> DecoderState:
        caches: dict[str, Any] = {}
        for name, shape, dtype in self._cache_inputs:
            dims = [1 if i == 0 else (0 if isinstance(d, str) else int(d)) for i, d in enumerate(shape)]
            caches[name] = np.zeros(dims, dtype=dtype)
        return DecoderState(caches, 0)

    def encode_audio(self, features: FloatArray) -> FloatArray:
        out: FloatArray = self.encoder.run(None, {"input_features": features[None].astype(np.float32)})[0]
        return out

    def decode(
        self,
        state: DecoderState,
        input_ids: Sequence[int],
        audio_features: FloatArray | None = None,
        audio_out_codes: IntArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        n = len(input_ids)
        total = state.length + n
        feed = dict(state.caches)
        feed["input_ids"] = np.asarray(input_ids, dtype=np.int64)[None]
        feed["attention_mask"] = np.ones((1, total), dtype=np.int64)
        feed["position_ids"] = np.arange(state.length, total, dtype=np.int64)[None]
        feed["audio_features"] = self._no_features if audio_features is None else audio_features.astype(np.float32)
        feed["audio_out_codes"] = self._no_codes if audio_out_codes is None else audio_out_codes.astype(np.int64)
        feed["logits_to_keep"] = self._keep_last
        outputs = self.decoder.run(self._outputs, feed)
        logits = hidden = None
        for name, value in zip(self._outputs, outputs):
            if name == "logits":
                logits = value[0, -1]
            elif name == "hidden_states":
                hidden = value[0, -1]
            else:
                state.caches[self._renames[name]] = value
        state.length = total
        assert logits is not None and hidden is not None
        return logits.astype(np.float32), hidden.astype(np.float32)

    def depth(self, hidden: FloatArray, codes: IntArray, masked: npt.NDArray[np.bool_]) -> FloatArray:
        out: FloatArray = self.depth_decoder.run(
            None, {"hidden_states": hidden[None].astype(np.float32), "codes": codes[None], "masked": masked[None]}
        )[0][0]
        return out

    def codec(self, codes: IntArray) -> FloatArray:
        out: FloatArray = self.codec_decoder.run(None, {"audio_codes": np.ascontiguousarray(codes.T)[None]})[0][0]
        return out.astype(np.float32)


# -- sampling -----------------------------------------------------------------------------------------------------
def sample(
    logits: FloatArray,
    rng: np.random.Generator,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    history: Sequence[int] | IntArray | None = None,
    suppress: Sequence[int] = (),
    allowed: Sequence[int] | None = None,
) -> int:
    """One id from ``logits`` as ``Lfm3AudioForConditionalGeneration._sample`` picks it: repetition penalty over
    the ids of ``history``, suppressed ids, then the arg-max (``temperature`` 0) or temperature, top-k (ties
    kept) and a draw from the softmax. ``allowed`` restricts the choice to those ids."""
    scores = logits.astype(np.float64, copy=True)
    if repetition_penalty != 1.0 and history is not None and len(history):
        seen = np.unique(np.asarray(history, dtype=np.int64))
        seen = seen[(seen >= 0) & (seen < len(scores))]
        values = scores[seen]
        scores[seen] = np.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    if len(suppress):
        scores[list(suppress)] = -np.inf
    if allowed is not None:
        mask = np.full(len(scores), -np.inf)
        mask[list(allowed)] = 0.0
        scores = scores + mask
    if temperature <= 0.0:
        return int(np.argmax(scores))
    scores = scores / temperature
    if 0 < top_k < len(scores):
        kth = np.partition(scores, -top_k)[-top_k]
        scores[scores < kth] = -np.inf
    probs = np.exp(scores - np.max(scores))
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def sample_audio_frame(
    graphs: Graphs,
    hidden: FloatArray,
    settings: GenerationSettings,
    rng: np.random.Generator,
    history: IntArray | None = None,
) -> IntArray:
    """The codes of one speech frame from a hidden state: the codebooks are revealed in a random order over the
    passes of the MaskGIT schedule, each pass scoring the codebooks still hidden. ``history``: the earlier frames
    ``(frames, codebooks)`` for the repetition penalty."""
    n = settings.codebooks
    codes = np.zeros(n, dtype=np.int64)
    revealed = np.zeros(n, dtype=bool)
    order = rng.permutation(n)
    start = 0
    for count in settings.maskgit_schedule:
        pick = order[start : start + count]
        start += count
        logits = graphs.depth(hidden, codes, ~revealed)
        for index in pick:
            codes[index] = sample(
                logits[index],
                rng,
                temperature=settings.audio_temperature,
                top_k=settings.audio_top_k,
                repetition_penalty=settings.audio_repetition_penalty,
                history=None if history is None else history[:, index],
            )
        revealed[pick] = True
    return codes


# -- codec: codes -> audio, as they are generated ------------------------------------------------------------------
class CodecStream:
    """Incremental LiquidCodec decoding of one speech segment.

    The codec decoder graph has no cache inputs, but it is causal and a frame depends on the 20 frames before it
    at most (measured: decoding from a later start gives the same STFT rows from the 20th frame on). A chunk is
    therefore decoded together with ``context`` earlier frames and only its own rows are kept. The inverse STFT
    is the overlap-add of ``LiquidCodecModel.decode`` with a streaming cache: the first 480 samples are trimmed,
    the last 960 samples of every chunk wait for the frames that still add to them, and the end adds 480."""

    def __init__(self, graphs: Graphs, settings: CodecSettings, *, context: int = 24) -> None:
        self.graphs = graphs
        self.settings = settings
        self.context = context
        n = settings.win_length
        self.window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float64)
        self.tail = settings.win_length - settings.hop_length
        self.trim = self.tail // 2
        self.frames: list[IntArray] = []
        self.decoded = 0
        self._audio: npt.NDArray[np.float64] | None = None
        self._envelope: npt.NDArray[np.float64] | None = None
        self._started = False

    def push(self, codes: IntArray) -> None:
        self.frames.append(np.asarray(codes, dtype=np.int64))

    @property
    def pending(self) -> int:
        return len(self.frames) - self.decoded

    def _istft(self, rows: FloatArray) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        s = self.settings
        bins = s.n_fft // 2 + 1
        log_magnitude, angle = rows[:, :bins].astype(np.float64), rows[:, bins:].astype(np.float64)
        spec = np.exp(log_magnitude) * (np.cos(angle) + 1j * np.sin(angle))
        frames = np.fft.irfft(spec, s.n_fft, axis=1) * self.window
        count = len(frames)
        size = (count - 1) * s.hop_length + s.win_length
        audio = np.zeros(size)
        envelope = np.zeros(size)
        squared = self.window**2
        for i in range(count):
            begin = i * s.hop_length
            audio[begin : begin + s.win_length] += frames[i]
            envelope[begin : begin + s.win_length] += squared
        return audio, envelope

    def decode(self, *, final: bool = False) -> FloatArray:
        """Audio of the frames pushed since the last call (24 kHz). ``final`` also returns the held-back tail and
        ends the segment."""
        s = self.settings
        new = self.pending
        if new == 0 and not (final and self._started):
            return np.zeros(0, dtype=np.float32)
        if new:
            begin = max(0, self.decoded - self.context)
            rows = self.graphs.codec(np.stack(self.frames[begin:]))
            audio, envelope = self._istft(rows[-new * s.frame_stack :])
            self.decoded = len(self.frames)
        else:
            audio, envelope = np.zeros(self.tail), np.zeros(self.tail)
        if self._audio is not None and self._envelope is not None:
            audio[: self.tail] += self._audio
            envelope[: self.tail] += self._envelope
        if not self._started:
            audio, envelope = audio[self.trim :], envelope[self.trim :]
            self._started = True
        if final:
            audio, envelope = audio[: -self.trim], envelope[: -self.trim]
            self._audio = self._envelope = None
            self._started = False
            self.frames, self.decoded = [], 0
        else:
            self._audio, self._envelope = audio[-self.tail :].copy(), envelope[-self.tail :].copy()
            audio, envelope = audio[: -self.tail], envelope[: -self.tail]
        out: FloatArray = (audio / np.maximum(envelope, 1e-8)).astype(np.float32)
        return out


# -- text: token pieces and tool calls ----------------------------------------------------------------------------
class TextStream:
    """Generated token ids -> text deltas (control tokens dropped, incomplete UTF-8 held back)."""

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self._decode = decode
        self._ids: list[int] = []
        self._done = ""

    def push(self, token: int) -> str:
        self._ids.append(token)
        text = self._decode(self._ids)
        if text.endswith("�"):
            return ""
        delta = text[len(self._done) :]
        self._done = text
        if delta.endswith("\n") and len(self._ids) > 64:  # a safe place to start over: keeps decoding linear
            self._ids, self._done = [], ""
        return delta


_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """The calls in generated text (model card): values that are not JSON are plain strings."""
    calls = []
    for name, body in _CALL_RE.findall(text):
        arguments: dict[str, Any] = {}
        for key, value in _PARAM_RE.findall(body):
            try:
                arguments[key] = json.loads(value)
            except ValueError:
                arguments[key] = value
        calls.append((name.strip(), arguments))
    return calls


class OutputParser:
    """Splits the text stream into what is said (text deltas), tool calls and reasoning (dropped)."""

    _BLOCKS = (("<tool_call>", "</tool_call>"), ("<think>", "</think>"))

    def __init__(self) -> None:
        self._buffer = ""
        self._inside: tuple[str, str] | None = None
        self._said = False

    def feed(self, delta: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
        self._buffer += delta
        text, calls = "", []
        while True:
            if self._inside is None:
                found = [(self._buffer.find(o), (o, c)) for o, c in self._BLOCKS if o in self._buffer]
                if not found:
                    text += self._buffer
                    self._buffer = ""
                    break
                at, block = min(found)
                text += self._buffer[:at]
                self._buffer = self._buffer[at:]
                self._inside = block
            else:
                opening, closing = self._inside
                end = self._buffer.find(closing)
                if end < 0:
                    break
                body = self._buffer[: end + len(closing)]
                self._buffer = self._buffer[end + len(closing) :]
                self._inside = None
                if opening == "<tool_call>":
                    calls += parse_tool_calls(body)
                    self._said = False  # the blank line between calls is not speech
        if not self._said:  # no leading white space, nor white space alone between calls
            text = text.lstrip()
            self._said = bool(text)
        return text, calls


# -- chat format --------------------------------------------------------------------------------------------------
def tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    """A Realtime (flat) or chat-completions (``{"type": "function", "function": {...}}``) tool as the function
    object the chat template writes."""
    nested = tool.get("function")
    fn: dict[str, Any] = nested if isinstance(nested, dict) else tool
    return {k: v for k, v in fn.items() if k not in ("type", "defer_loading", "strict")}


def system_prompt(instructions: str | None, tools: Sequence[dict[str, Any]], *, speech: bool = True) -> str:
    """The system turn's text: the trained mode sentence, the operator's instructions, and the tool definitions
    as ``chat_template.jinja`` appends them."""
    parts: list[str] = []
    if speech:
        parts.append(SYSTEM_MIXED_TOOLS if tools else SYSTEM_MIXED)
    if instructions and instructions.strip():
        parts.append(instructions.strip())
    if tools:
        parts.append(SYSTEM_TOOLS_SUFFIX)
    prompt = "\n\n".join(parts)
    if tools:
        lines = [json.dumps(tool_definition(t), ensure_ascii=False) for t in tools]
        prompt += ("\n\n" if prompt else "") + "<tools>\n" + "\n".join(lines) + "\n</tools>"
    return prompt


USER_HEADER = "<|im_start|>user\n"
TURN_FOOTER = "<|im_end|>\n"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
TOOL_HEADER = "<|im_start|>tool\n<tool_response>\n"
TOOL_FOOTER = "\n</tool_response><|im_end|>\n"
NO_REASONING = "<think></think>"


def first_header(system: str) -> str:
    return "<|startoftext|>" + (f"<|im_start|>system\n{system}<|im_end|>\n" if system else "") + USER_HEADER


# -- the model ----------------------------------------------------------------------------------------------------
class Lfm3Model:
    """Graphs, tokenizer and settings of one model directory; shared by all conversations."""

    def __init__(
        self,
        graphs: Graphs,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        tokens: Tokens,
        *,
        features: FeatureSettings | None = None,
        codec: CodecSettings | None = None,
        generation: GenerationSettings | None = None,
    ) -> None:
        self.graphs = graphs
        self.encode = encode
        self.decode = decode
        self.tokens = tokens
        self.features = features or FeatureSettings()
        self.codec = codec or CodecSettings()
        self.generation = generation or GenerationSettings()
        self.log_mel = LogMel(self.features)
        #: one graph run at a time: conversations take turns step by step instead of fighting for the cores
        self.lock = threading.Lock()

    @classmethod
    def from_dir(
        cls,
        path: Path,
        *,
        precision: str = "q4",
        threads: int = 4,
        device: str = "cpu",
        graph_devices: dict[str, str] | None = None,
        **options: Any,
    ) -> Lfm3Model:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ComponentMissing(f"LFM3-Audio in ONNX needs the tokenizers package: {INSTALL_HINT}") from exc
        path = Path(path)
        for name in ("tokenizer.json", "config.json", "preprocessor_config.json", "audio_tokenizer/config.json"):
            if not (path / name).is_file():
                raise EngineUnavailable(f"{path / name} is missing; pull the model again")
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        generation = GenerationSettings.from_dir(path)
        for key in list(options):
            if hasattr(generation, key) and options[key] is not None:
                setattr(generation, key, type(getattr(generation, key))(options.pop(key)))
        return cls(
            OnnxGraphs(
                path,
                precision=precision,
                threads=threads,
                providers=providers_for(device),
                graph_providers={role: providers_for(d) for role, d in (graph_devices or {}).items()},
            ),
            lambda text: list(tokenizer.encode(text, add_special_tokens=False).ids),
            lambda ids: str(tokenizer.decode(ids, skip_special_tokens=True)),
            Tokens.from_lookup(tokenizer.token_to_id),
            features=FeatureSettings.from_dir(path),
            codec=CodecSettings.from_dir(path),
            generation=generation,
        )


def missing_packages() -> list[str]:
    """The Python packages this module needs that are not installed."""
    import importlib.util

    return [name for name in ("onnxruntime", "tokenizers") if importlib.util.find_spec(name) is None]


#: sampling values a catalog entry may override (the names of :class:`GenerationSettings`)
GENERATION_OPTIONS = (
    "audio_temperature", "audio_top_k", "audio_repetition_penalty", "audio_repetition_window", "temperature",
    "top_k", "repetition_penalty",
)  # fmt: skip
#: :class:`Lfm3Session` settings a catalog entry may set
SESSION_OPTIONS = (
    "prefill_s", "chunk_frames", "first_chunk_frames", "codec_context", "max_context", "max_response_slots",
    "reasoning",
)  # fmt: skip

_models: dict[tuple[str, str, int, str, str], Lfm3Model] = {}
_models_lock = threading.Lock()


def shared_model(
    path: Path,
    options: dict[str, Any],
    threads: int,
    device: str = "cpu",
    graph_devices: dict[str, str] | None = None,
) -> Lfm3Model:
    """The model of a directory, loaded once: speech-to-speech sessions and text-to-speech use the same graphs
    (about 3 GB of memory for the q4 files)."""
    precision = str(options.get("precision") or "q4")
    overrides = {k: options[k] for k in GENERATION_OPTIONS if options.get(k) is not None}
    where = device + (json.dumps(graph_devices, sort_keys=True) if graph_devices else "")
    key = (str(Path(path).resolve()), precision, threads, json.dumps(overrides, sort_keys=True), where)
    with _models_lock:
        model = _models.get(key)
        if model is None:
            model = _models[key] = Lfm3Model.from_dir(
                Path(path), precision=precision, threads=threads, device=device, graph_devices=graph_devices,
                **overrides,
            )  # fmt: skip
        return model


def graph_files(path: Path, options: dict[str, Any]) -> dict[str, int]:
    """The four graphs of a precision and their sizes (with external weights), for placement."""
    precision = str(options.get("precision") or "q4")
    if precision not in PRECISIONS:
        return {}
    decoder, encoder, codec = PRECISIONS[precision]
    out: dict[str, int] = {}
    for role, name in (("decoder", decoder), ("encoder", encoder), ("depth_decoder", "depth_decoder.onnx"),
                       ("codec_decoder", codec)):  # fmt: skip
        f = Path(path) / "onnx" / name
        if f.is_file():
            out[role] = f.stat().st_size + sum(x.stat().st_size for x in f.parent.glob(name + "_data*") if x.is_file())
            data = f.with_name(name + ".data")
            if data.is_file():
                out[role] += data.stat().st_size
    return out


def release_models() -> None:
    """Forget the loaded models (their memory is freed once no session or voice uses them)."""
    with _models_lock:
        _models.clear()


@dataclass
class ResponseStats:
    """Timings of the last response, in seconds from ``respond()`` unless said otherwise."""

    context_tokens: int = 0
    prefill_tokens: int = 0  # evaluated inside respond() before the first step
    prefill_s: float = 0.0
    first_text_s: float | None = None
    first_audio_s: float | None = None
    total_s: float = 0.0
    text_tokens: int = 0
    audio_frames: int = 0
    audio_s: float = 0.0
    decoder_s: float = 0.0
    depth_s: float = 0.0
    codec_s: float = 0.0
    status: str = ""

    def summary(self) -> dict[str, Any]:
        steps = self.text_tokens + self.audio_frames
        gen_s = max(1e-9, self.total_s - self.prefill_s)
        return {
            "status": self.status,
            "context_tokens": self.context_tokens,
            "prefill_tokens": self.prefill_tokens,
            "prefill_s": round(self.prefill_s, 3),
            "first_text_s": None if self.first_text_s is None else round(self.first_text_s, 3),
            "first_audio_s": None if self.first_audio_s is None else round(self.first_audio_s, 3),
            "total_s": round(self.total_s, 3),
            "text_tokens": self.text_tokens,
            "audio_frames": self.audio_frames,
            "audio_s": round(self.audio_s, 2),
            "steps_per_s": round(steps / gen_s, 2),
            "rtf": round(self.total_s / self.audio_s, 3) if self.audio_s else None,
            "decoder_s": round(self.decoder_s, 3),
            "depth_s": round(self.depth_s, 3),
            "codec_s": round(self.codec_s, 3),
        }


class Lfm3Session:
    """One conversation: the decoder state and the turn being filled. Synchronous and not thread-safe: one
    thread (the conversation's worker) calls it."""

    def __init__(
        self,
        model: Lfm3Model,
        config: S2SConfig | None = None,
        *,
        seed: int | None = None,
        prefill_s: float = 0.48,
        chunk_frames: int = 6,
        first_chunk_frames: int = 2,
        codec_context: int = 24,
        max_context: int = 8192,
        speech: bool = True,
        reasoning: bool = False,
        max_response_slots: int = 1500,
    ) -> None:
        self.model = model
        self.config = config or S2SConfig()
        self.rng = np.random.default_rng(seed)
        #: while the user speaks, finished audio tokens are evaluated every ``prefill_s`` seconds of new audio
        self.prefill_s = prefill_s
        self.chunk_frames = max(1, chunk_frames)
        self.first_chunk_frames = max(1, first_chunk_frames)
        self.codec_context = codec_context
        #: positions after which the next user turn starts a fresh context (system prompt only)
        self.max_context = max_context
        #: the length of a response (text tokens plus speech frames; 1500 is about 80 s of speech) when neither the
        #: session nor the request sets ``max_output_tokens``: a model that never ends its turn must not run for ever
        self.max_response_slots = max_response_slots
        #: text-only responses may open with a ``<think>`` block (never passed on); off: they are prefilled with
        #: an empty one, which is faster
        self.reasoning = reasoning
        self.system = system_prompt(self.config.instructions, self.config.tools, speech=speech)
        self.state = model.graphs.empty_state()
        self.stats = ResponseStats()
        self._pending_ids: list[int] = []  # text tokens not evaluated yet (go in with the next decoder run)
        self._started = False
        self._turn_open = False
        self._tool_results = False
        self._next_header = ""  # the footer of the last response, before the next turn
        self._audio: list[FloatArray] = []  # the clip being heard
        self._audio_len = 0
        self._audio_fed = 0  # audio tokens of the clip already evaluated
        self._audio_checked = 0  # samples at the last prefill
        self._last: tuple[FloatArray, FloatArray] | None = None  # logits and hidden state of the last position
        self._audio_history: deque[IntArray] = deque(maxlen=model.generation.audio_repetition_window)

    # -- feeding the decoder --
    def _queue(self, text: str) -> None:
        if text:
            self._pending_ids += self.model.encode(text)

    def _flush(self, audio_rows: FloatArray | None = None, extra: Sequence[int] = ()) -> None:
        """Evaluate the queued text tokens, then ``audio_rows`` (one ``<|audio|>`` position each), then
        ``extra`` tokens, in one decoder run."""
        ids = list(self._pending_ids)
        if audio_rows is not None and len(audio_rows):
            ids += [self.model.tokens.audio] * len(audio_rows)
        else:
            audio_rows = None
        ids += list(extra)
        if not ids:
            return
        self._pending_ids = []
        with self.model.lock:
            self._last = self.model.graphs.decode(self.state, ids, audio_rows)

    @property
    def context_tokens(self) -> int:
        return self.state.length + len(self._pending_ids)

    def reset(self) -> None:
        """Forget the conversation (the system prompt stays)."""
        self.state = self.model.graphs.empty_state()
        self._pending_ids = []
        self._started = self._turn_open = self._tool_results = False
        self._next_header = ""
        self._audio, self._audio_len, self._audio_fed, self._audio_checked = [], 0, 0, 0
        self._last = None
        self._audio_history.clear()

    # -- the user turn --
    def _open_turn(self) -> None:
        if self._turn_open:
            return
        if self._started and self.context_tokens >= self.max_context:
            log.warning("speech-to-speech context is full (%d positions): starting over", self.context_tokens)
            self.reset()
        self._queue(self._next_header + USER_HEADER if self._started else first_header(self.system))
        self._next_header = ""
        self._started = self._turn_open = True
        self._tool_results = False

    def append_audio(self, samples16: FloatArray) -> None:
        """Audio of the current user turn (16 kHz mono). Audio tokens that are complete are evaluated now and
        then, so that little is left to do when the turn ends."""
        samples = np.asarray(samples16, dtype=np.float32).reshape(-1)
        if not len(samples):
            return
        self._open_turn()
        self._audio.append(samples)
        self._audio_len += len(samples)
        # re-encoding the clip costs time in proportion to its length: space the prefills out as it grows
        interval = max(self.prefill_s, self._audio_len / self.model.features.sample_rate / 8.0)
        if self.prefill_s > 0 and self._audio_len - self._audio_checked >= interval * self.model.features.sample_rate:
            self._prefill_audio(final=False)

    def _prefill_audio(self, *, final: bool) -> None:
        f = self.model.features
        self._audio_checked = self._audio_len
        clip = np.concatenate(self._audio) if len(self._audio) > 1 else self._audio[0]
        self._audio = [clip]
        frames = f.complete_frames(f.pad_length(len(clip))) if final else f.complete_frames(len(clip))
        tokens = f.num_tokens(frames) if final else frames // f.encoder_stride
        if tokens <= self._audio_fed:
            return
        features = self.model.log_mel(clip, pad=final)[:, : tokens * f.encoder_stride]
        with self.model.lock:
            rows = self.model.graphs.encode_audio(features)
        self._flush(rows[self._audio_fed : tokens])
        self._audio_fed = tokens

    def commit(self) -> None:
        """End the audio of the user turn: its last tokens are evaluated."""
        if not self._audio_len:
            return
        self._prefill_audio(final=True)
        self._audio, self._audio_len, self._audio_fed, self._audio_checked = [], 0, 0, 0

    def add_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.commit()
        self._open_turn()
        self._queue(text)

    def add_tool_output(self, call_id: str, output: str) -> None:
        """The result of a tool call of the last response; the next response answers it."""
        if self._turn_open or not self._started:
            raise ValueError("a tool output must follow the response that made the call")
        self._queue(self._next_header + TOOL_HEADER + output + TOOL_FOOTER)
        self._next_header = ""
        self._tool_results = True

    # -- the response --
    def respond(
        self,
        *,
        modalities: Sequence[str] = ("text", "audio"),
        max_output_tokens: Any = None,
        tool_choice: Any = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[S2SEvent]:
        """Generate the response to the conversation so far; the last event is a :class:`ResponseEnd`.
        ``cancelled`` is asked before every step (a text token or a speech frame)."""
        stop = cancelled or (lambda: False)
        started = time.perf_counter()
        stats = self.stats = ResponseStats()
        model, t, g = self.model, self.model.tokens, self.model.generation
        speech = "audio" in modalities
        choice = tool_choice if tool_choice in ("auto", "none", "required") else self.config.tool_choice
        if choice not in ("auto", "none", "required"):
            choice = "auto"
        tools = bool(self.config.tools) and choice != "none"
        limit = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        max_slots = int(limit) if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else None
        if max_slots is None and self.max_response_slots > 0:
            max_slots = self.max_response_slots

        # close the turn and open the assistant's
        self.commit()
        if self._turn_open:
            self._queue(TURN_FOOTER)
        elif not self._tool_results:
            if not self._started:
                yield ResponseEnd("failed", error="nothing to respond to: commit audio or add a message first")
                return
            self._queue(self._next_header)
        self._queue(ASSISTANT_HEADER)
        # how the response opens (model card): a spoken answer is prefilled with <|mixed_start|>; with tools the
        # model itself opens with <tool_call> or <|mixed_start|>
        prefill: list[int] = []
        first_allowed: list[int] | None = None
        if tools and choice == "required":
            first_allowed = [t.tool_call]
        elif speech and tools:
            first_allowed = [t.tool_call, t.mixed_start]
        elif speech:
            prefill = [t.mixed_start]
        elif not tools and not self.reasoning:
            prefill = model.encode(NO_REASONING)  # a direct text answer (model card)
        text_only = not speech
        before = self.state.length
        self._flush(extra=prefill)
        stats.prefill_tokens = self.state.length - before
        stats.context_tokens = self.state.length
        stats.prefill_s = time.perf_counter() - started
        self._turn_open = self._tool_results = False
        self._next_header = TURN_FOOTER
        assert self._last is not None
        logits, hidden = self._last

        mode = "interleaved" if prefill == [t.mixed_start] else "text"
        suppress = [t.audio, t.audio_out] + ([t.mixed_start, t.audio_start] if text_only else [])
        slots_left, speaking, text_done = g.n_text, False, False
        text_stream, parser = TextStream(model.decode), OutputParser()
        text_history: deque[int] = deque(maxlen=g.text_repetition_window)
        codec = CodecStream(model.graphs, model.codec, context=self.codec_context)
        chunk = self.first_chunk_frames
        status = "completed"
        step = 0

        def audio_out(final: bool) -> Iterator[S2SEvent]:
            nonlocal chunk
            clock = time.perf_counter()
            with model.lock:
                samples = codec.decode(final=final)
            stats.codec_s += time.perf_counter() - clock
            chunk = self.chunk_frames
            if len(samples):
                if stats.first_audio_s is None:
                    stats.first_audio_s = time.perf_counter() - started
                stats.audio_s += len(samples) / model.codec.sample_rate
                yield AudioDelta(samples, model.codec.sample_rate)

        try:
            while True:
                if stop():
                    status = "cancelled"
                    break
                if max_slots is not None and step >= max_slots:
                    status = "incomplete"
                    break
                step += 1
                interleaving = mode == "interleaved"
                frame: IntArray | None = None
                token = -1
                if mode == "audio" or (interleaving and speaking):
                    clock = time.perf_counter()
                    history = np.stack(self._audio_history) if self._audio_history else None
                    with model.lock:
                        frame = sample_audio_frame(model.graphs, hidden, g, self.rng, history)
                    stats.depth_s += time.perf_counter() - clock
                    if (frame == g.audio_eos_code).any():
                        # the end of the speech: the frame is fed back as all end codes, text follows
                        frame = np.full_like(frame, g.audio_eos_code)
                        mode = "text"
                        yield from audio_out(True)
                    else:
                        self._audio_history.append(frame)
                        stats.audio_frames += 1
                        codec.push(frame)
                        if codec.pending >= chunk:
                            yield from audio_out(False)
                else:
                    token = sample(
                        logits,
                        self.rng,
                        temperature=g.temperature,
                        top_k=g.top_k,
                        repetition_penalty=g.repetition_penalty,
                        history=list(text_history),
                        suppress=suppress,
                        allowed=first_allowed if step == 1 else None,
                    )
                    text_history.append(token)
                    if token == t.text_end:
                        if not interleaving:
                            break  # outside an interleaved block it ends the turn
                        text_done = speaking = True
                    elif token == t.im_end or token in g.eos_ids:
                        break  # <|im_end|> ends the turn in any text slot
                    elif interleaving:
                        pass  # the mode markers only open a block; inside one they are ordinary text
                    elif token == t.audio_start:
                        mode = "audio"
                    elif token == t.mixed_start:
                        mode = "interleaved"
                        slots_left, speaking, text_done = g.n_text, False, False
                    stats.text_tokens += 1
                    said, calls = parser.feed(text_stream.push(token))
                    if said:
                        if stats.first_text_s is None:
                            stats.first_text_s = time.perf_counter() - started
                        yield TextDelta(said)
                    for name, arguments in calls:
                        yield ToolCall("call_" + secrets.token_hex(8), name, json.dumps(arguments, ensure_ascii=False))
                if interleaving and not text_done:
                    slots_left -= 1
                    if slots_left == 0:
                        speaking = not speaking
                        slots_left = g.n_audio if speaking else g.n_text
                        if not speaking and codec.pending:
                            yield from audio_out(False)  # the block of speech is complete: let it be heard
                clock = time.perf_counter()
                with model.lock:
                    if frame is not None:
                        logits, hidden = model.graphs.decode(self.state, [t.audio_out], None, frame[None])
                    else:
                        logits, hidden = model.graphs.decode(self.state, [token])
                self._last = (logits, hidden)
                stats.decoder_s += time.perf_counter() - clock
            if codec.pending or status != "completed":
                yield from audio_out(True)  # what was generated before the stop
        except GeneratorExit:
            status = "cancelled"
            raise
        except Exception as exc:
            log.exception("speech-to-speech response failed")
            stats.status = "failed"
            stats.total_s = time.perf_counter() - started
            yield ResponseEnd("failed", self._usage(stats), error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            stats.status = stats.status or status
            stats.total_s = time.perf_counter() - started
        yield ResponseEnd(status, self._usage(stats))

    @staticmethod
    def _usage(stats: ResponseStats) -> dict[str, int]:
        output = stats.text_tokens + stats.audio_frames
        return {
            "input_tokens": stats.context_tokens,
            "output_tokens": output,
            "total_tokens": stats.context_tokens + output,
        }


# -- text to speech -----------------------------------------------------------------------------------------------
#: language code -> the name the model was trained with (the "tts" chat template of the model's GGUF)
TTS_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "American English",
    "en-us": "American English", "en-gb": "British English", "fi": "Finnish", "fr": "French", "he": "Hebrew",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "es": "Spanish", "sv": "Swedish", "tr": "Turkish", "zh": "Chinese",
}  # fmt: skip
TTS_VOICES = ("female", "male")


def tts_prompt(text: str, voice: str | None = None, language: str | None = None) -> str:
    """The prompt of speech synthesis (model card): the assistant turn is prefilled with ``<|audio_start|>``."""
    code = (language or "en").strip().lower().replace("_", "-")
    name = TTS_LANGUAGES.get(code) or TTS_LANGUAGES.get(code.split("-")[0]) or (language or "American English")
    v = (voice or "").strip().lower()
    return (
        f"<|startoftext|><|im_start|>system\nPerform TTS in {name} using a {v if v in TTS_VOICES else TTS_VOICES[0]} "
        f"voice.<|im_end|>\n{USER_HEADER}{text.strip()}{TURN_FOOTER}{ASSISTANT_HEADER}<|audio_start|>"
    )


def synthesize(
    model: Lfm3Model,
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    rng: np.random.Generator | None = None,
    cancelled: Callable[[], bool] | None = None,
    chunk_frames: int = 6,
    first_chunk_frames: int = 2,
    max_frames: int | None = None,
) -> Iterator[FloatArray]:
    """Speak ``text``: 24 kHz audio in chunks as it is generated. Speech frames are sampled until the
    end-of-audio frame, or until ``max_frames`` (by default a budget that follows the length of the text)."""
    stop = cancelled or (lambda: False)
    rng = rng or np.random.default_rng()
    g, t = model.generation, model.tokens
    budget = max_frames if max_frames is not None else 60 + 4 * len(text)
    state = model.graphs.empty_state()
    with model.lock:
        _logits, hidden = model.graphs.decode(state, model.encode(tts_prompt(text, voice, language)))
    codec = CodecStream(model.graphs, model.codec)
    history: deque[IntArray] = deque(maxlen=g.audio_repetition_window)
    chunk = first_chunk_frames
    for _ in range(budget):
        if stop():
            return
        with model.lock:
            frame = sample_audio_frame(model.graphs, hidden, g, rng, np.stack(history) if history else None)
        if (frame == g.audio_eos_code).any():
            break
        history.append(frame)
        codec.push(frame)
        if codec.pending >= chunk:
            with model.lock:
                samples = codec.decode()
            chunk = chunk_frames
            if len(samples):
                yield samples
        with model.lock:
            _logits, hidden = model.graphs.decode(state, [t.audio_out], None, frame[None])
    with model.lock:
        samples = codec.decode(final=True)
    if len(samples):
        yield samples


class Lfm3Tts(TTSModel):
    """LFM3-Audio as a text-to-speech model (engine ``onnxruntime``): two voices, 18 languages, no cloning."""

    accepts_extras = True

    def __init__(self, model: Lfm3Model, spec: dict[str, Any] | None = None) -> None:
        spec = spec or {}
        self.model = model
        self.sample_rate = model.codec.sample_rate
        self.seed = int(spec["seed"]) if spec.get("seed") is not None else None
        self.chunk_frames = max(1, int(spec.get("chunk_frames", 6)))
        self.first_chunk_frames = max(1, int(spec.get("first_chunk_frames", 2)))
        listed = spec.get("voices") or [{"id": v, "name": v.capitalize(), "gender": v[0]} for v in TTS_VOICES]
        self.voices = tuple(
            Voice(str(v["id"]), str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i)
            for i, v in enumerate(listed)
        )

    @classmethod
    def from_dir(cls, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Lfm3Tts:
        placed = {"graph_devices": dict(opts.graph_devices)} if opts.graph_devices else {}
        return cls(shared_model(path, spec, opts.threads, opts.device, **placed), spec)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        extras: SynthesisExtras | None = None,
    ) -> Iterator[FloatArray]:
        found = self.find_voice(voice)
        if not text.strip():
            return
        yield from synthesize(
            self.model,
            text,
            voice=found.id if found is not None else None,
            language=extras.language if extras is not None else None,
            rng=np.random.default_rng(self.seed),
            cancelled=cancelled,
            chunk_frames=self.chunk_frames,
            first_chunk_frames=self.first_chunk_frames,
        )


# -- the bolo.s2s interface ---------------------------------------------------------------------------------------
class OnnxS2SConversation(S2SConversation):
    """An :class:`Lfm3Session` behind the asynchronous S2S interface.

    A worker thread runs what the session is asked to do, in order: audio as it arrives (prefill), the commit,
    messages, tool results and responses. ``append_audio`` therefore never waits for the model. Cancelling the
    task that iterates :meth:`respond` stops the response before its next step; the audio and text generated up
    to there stay in the context, as with llama-server."""

    output_rate = 24000

    def __init__(self, session: Lfm3Session) -> None:
        self.session = session
        self.output_rate = session.model.codec.sample_rate
        self._ops: queue.Queue[tuple[Any, ...] | None] = queue.Queue()
        self._error: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._work, name="bolo-s2s-onnx", daemon=True)
        self._thread.start()

    def _work(self) -> None:
        backlog: deque[tuple[Any, ...] | None] = deque()
        while True:
            if not backlog:
                backlog.append(self._ops.get())
            while True:  # everything already asked for: consecutive audio is taken in one piece
                try:
                    backlog.append(self._ops.get_nowait())
                except queue.Empty:
                    break
            op = backlog.popleft()
            if op is None:
                return
            kind = op[0]
            try:
                if kind == "audio":
                    parts = [op[1]]
                    while backlog:
                        following = backlog[0]
                        if following is None or following[0] != "audio":
                            break
                        parts.append(following[1])
                        backlog.popleft()
                    self.session.append_audio(np.concatenate(parts) if len(parts) > 1 else parts[0])
                elif kind == "commit":
                    self.session.commit()
                elif kind == "text":
                    self.session.add_text(op[1])
                elif kind == "tool":
                    self.session.add_tool_output(op[1], op[2])
                elif kind == "respond":
                    _, params, cancel, emit = op
                    if self._error is not None:
                        error, self._error = self._error, None
                        emit(ResponseEnd("failed", error=error))
                        continue
                    for event in self.session.respond(cancelled=cancel.is_set, **params):
                        emit(event)
            except Exception as exc:  # reported by the next response
                log.exception("speech-to-speech conversation: %s failed", kind)
                self._error = f"{type(exc).__name__}: {exc}"
                if kind == "respond":
                    self._error = None
                    op[3](ResponseEnd("failed", error=f"{type(exc).__name__}: {exc}"))

    def _post(self, *op: Any) -> None:
        if self._closed:
            raise RuntimeError("the speech-to-speech conversation is closed")
        self._ops.put(op)

    async def append_audio(self, samples16: FloatArray) -> None:
        if len(samples16):
            self._post("audio", np.array(samples16, dtype=np.float32))

    async def commit(self) -> None:
        self._post("commit")

    async def add_text(self, text: str) -> None:
        self._post("text", text)

    async def add_tool_output(self, call_id: str, output: str) -> None:
        self._post("tool", call_id, output)

    async def respond(
        self, *, modalities: list[str], max_output_tokens: Any = None, tool_choice: Any = None
    ) -> AsyncGenerator[S2SEvent, None]:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[S2SEvent] = asyncio.Queue()
        cancel = threading.Event()

        def emit(event: S2SEvent) -> None:
            if not cancel.is_set() or isinstance(event, ResponseEnd):
                loop.call_soon_threadsafe(events.put_nowait, event)

        params = {"modalities": list(modalities), "max_output_tokens": max_output_tokens, "tool_choice": tool_choice}
        self._post("respond", params, cancel, emit)
        try:
            while True:
                event = await events.get()
                yield event
                if isinstance(event, ResponseEnd):
                    return
        finally:
            # barge-in, the client's response.cancel, or the normal end: the worker stops before its next step
            cancel.set()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ops.put(None)
        await asyncio.to_thread(self._thread.join, 30.0)


class OnnxS2SBackend(S2SBackend):
    """LFM3-Audio in this process. The model is loaded when the first conversation opens and shared by all of
    them; ``options`` are the catalog entry's (see the module docstring). The session's ``voice`` is not used: in
    a conversation the model speaks with its own voice (the two named voices exist for text-to-speech only)."""

    name = "onnx"

    def __init__(
        self,
        model_dir: Path | str,
        *,
        options: dict[str, Any] | None = None,
        threads: int = 4,
        model_id: str | None = None,
        model: Lfm3Model | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.options = dict(options or {})
        self.threads = threads
        self.device = device
        self.model_id = model_id
        self._model = model
        self._load_lock = threading.Lock()
        self._conversations = 0

    def model(self) -> Lfm3Model:
        with self._load_lock:
            if self._model is None:
                self._model = shared_model(self.model_dir, self.options, self.threads, self.device)
            return self._model

    def session_options(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: self.options[k] for k in SESSION_OPTIONS if self.options.get(k) is not None}
        if self.options.get("seed") is not None:
            out["seed"] = int(self.options["seed"])
        return out

    async def open(self, config: S2SConfig) -> S2SConversation:
        model = await asyncio.to_thread(self.model)
        self._conversations += 1
        return OnnxS2SConversation(Lfm3Session(model, config, **self.session_options()))

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model_id,
            "managed": True,
            "in_process": True,
            "precision": str(self.options.get("precision") or "q4"),
            "threads": self.threads,
            "loaded": self._model is not None,
            "conversations_opened": self._conversations,
        }

    async def close(self) -> None:
        self._model = None
