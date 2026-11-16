"""DiCoW (diarization-conditioned Whisper) in ONNX Runtime and NumPy, without PyTorch.

BUT-FIT's DiCoW v3.3 is Whisper large-v3-turbo whose encoder is told, for every 20 ms frame, the probabilities
that there is **S**ilence, only the **T**arget speaker, only **N**on-target speakers, or the target in
**O**verlap with others (the STNO mask, built from a diarization). FDDT layers apply four learned diagonal
affine maps weighted by those probabilities before the positional embeddings and before each encoder layer. A
recording is decoded once per diarized speaker, with that speaker as the target, so each speaker gets a
transcript of their own words, also where people talk at once.

This module runs the graphs made by ``tools/export_dicow_onnx.py`` (an encoder that takes log-mel features and
the STNO mask and returns the decoder's cross-attention keys and values; a decoder with a key/value cache) and
reproduces what the bolo-torch plugin does with the model's own PyTorch code (``plugins/bolo-torch``,
``bolo_torch.models.dicow``), step by step:

* features: Whisper's 128-band log-mel (:func:`bolo.audio.features.whisper_log_mel`); recordings up to 30 s
  are padded to 30 s, longer ones are processed whole (transformers' long-form input);
* masks: speaker activity resampled to the 50 Hz encoder rate, then BUT's ``DiCoWPipeline.get_stno_mask``
  formula (:func:`stno_masks`), identical to the plugin's;
* decoding: transformers 4.55 Whisper long-form generation as the model's ``generation.py`` changes it:
  30 s windows from ``seek``, the mask window padded with silence, greedy search with the suppress-token list,
  Whisper's timestamp rules plus DiCoW's exception (end-of-text is allowed as the first token, to leave a
  silent window at once), DiCoW's ``_retrieve_segment`` for cutting a window into segments and moving
  ``seek``, and language detection on the first window when no language is given;
* text: byte-level BPE decoding of the segment tokens with the special and timestamp tokens skipped and
  transformers' ``clean_up_tokenization``.

Without diarization (plain ``transcribe``) the whole input is the target speaker, as in the plugin.

Settings (catalog ``options``): ``encoder`` / ``decoder`` (ONNX files, default the int8 graphs, else the fp32
ones), ``batch`` (speakers decoded together, default 4), ``skip_silent_s`` (speakers with less target or
overlapped speech are not decoded, default 0.2).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from bolo.engines.base import (
    FloatArray,
    LoadOptions,
    Segment,
    SpeakerActivity,
    SpeakerRequest,
    STTModel,
    STTRequest,
    STTResult,
)
from bolo.ort_providers import create_session, providers_for

SR = 16000
#: STNO mask frames per second (Whisper's 100 mel frames per second after the encoder's stride-2 convolution)
MASK_RATE = 50
WINDOW_S = 30.0
WINDOW_FRAMES = 3000  # mel frames of one encoder window
MASK_WINDOW = WINDOW_FRAMES // 2
TIME_PRECISION = 0.02  # seconds per timestamp token
INPUT_STRIDE = 2  # mel frames per encoder frame
N_MELS = 128
IntArray = npt.NDArray[np.int64]

#: Whisper's languages (tokenizer order)
WHISPER_LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
)  # fmt: skip


# -- conditioning -------------------------------------------------------------------------------------


def activity_at_mask_rate(activity: FloatArray, frame_s: float, n_frames: int) -> FloatArray:
    """Speaker activity ``(frames, speakers)`` resampled to ``n_frames`` mask frames of 20 ms: ``(speakers, n)``.

    Mask frame ``j`` takes the activity frame under its centre; frames past the end of the activity are silent.
    """
    probs = np.asarray(activity, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("speaker activity must be a (frames, speakers) matrix")
    centres = (np.arange(n_frames) + 0.5) / MASK_RATE
    index = np.floor(centres / frame_s).astype(np.int64)
    out = np.zeros((probs.shape[1], n_frames), np.float32)
    inside = index < len(probs)
    out[:, inside] = probs[index[inside]].T
    return np.clip(out, 0.0, 1.0)


def stno_masks(activity: FloatArray) -> FloatArray:
    """STNO masks ``(speakers, 4, frames)`` from per-speaker activity ``(speakers, frames)``.

    For target ``s``: silence = prod_k (1 - p_k); target = p_s * prod_{k != s} (1 - p_k);
    non-target = (1 - p_s) * (1 - prod_{k != s} (1 - p_k)); overlap = p_s - target
    (``DiCoWPipeline.get_stno_mask``). With 0/1 activity exactly one class is 1 in every frame.
    """
    p = np.asarray(activity, dtype=np.float64)
    n_spk = p.shape[0]
    silence = np.prod(1.0 - p, axis=0)
    out = np.zeros((n_spk, 4, p.shape[1]), np.float32)
    for s in range(n_spk):
        others = np.prod(np.delete(1.0 - p, s, axis=0), axis=0) if n_spk > 1 else np.ones(p.shape[1])
        target = p[s] * others
        out[s, 0] = silence
        out[s, 1] = target
        out[s, 2] = (1.0 - p[s]) * (1.0 - others)
        out[s, 3] = p[s] - target
    return out


def log_mel(audio: FloatArray) -> FloatArray:
    """Whisper large-v3 features ``(128, frames)``: 30 s windows for short input, the whole recording otherwise."""
    from bolo.audio.features import whisper_log_mel

    x = np.asarray(audio, dtype=np.float32)
    if len(x) <= WINDOW_S * SR:  # the feature extractor pads short input to 30 s (with zeros, before the STFT)
        x = np.concatenate([x, np.zeros(int(WINDOW_S * SR) - len(x), np.float32)])
    return whisper_log_mel(x, n_mels=N_MELS)


def mask_frames(num_samples: int) -> int:
    """Mask frames for a recording of ``num_samples`` (the feature frames the model sees, halved)."""
    if num_samples <= WINDOW_S * SR:
        return MASK_WINDOW
    return (num_samples // 160) // 2


# -- tokens -------------------------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character table (byte-level BPE)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_CLEAN_UP = (
    (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
    (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
)  # fmt: skip


class TokenDecoder:
    """Token ids -> text as transformers' Whisper tokenizer decodes them with ``skip_special_tokens=True``."""

    def __init__(self, vocab: dict[str, int], first_special: int, clean_up: bool = True) -> None:
        self.tokens = {i: t for t, i in vocab.items()}
        self.first_special = first_special
        self.clean_up = clean_up
        self.byte_of = {c: b for b, c in _bytes_to_unicode().items()}

    @classmethod
    def from_dir(cls, path: Path, first_special: int) -> TokenDecoder:
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        clean_up = True
        config = path / "tokenizer_config.json"
        if config.exists():
            clean_up = bool(json.loads(config.read_text(encoding="utf-8")).get("clean_up_tokenization_spaces", True))
        return cls(vocab, first_special, clean_up)

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [self.tokens.get(int(i), "") for i in ids if int(i) < self.first_special]
        raw = bytes(self.byte_of[c] for c in "".join(pieces) if c in self.byte_of)
        text = raw.decode("utf-8", errors="replace")
        if self.clean_up:
            for a, b in _CLEAN_UP:
                text = text.replace(a, b)
        return text


# -- generation settings --------------------------------------------------------------------------------


@dataclass
class GenerationSettings:
    """What the decode loop needs from ``generation_config.json`` and ``config.json``."""

    eos: int = 50257
    start: int = 50258  # <|startoftranscript|>
    no_timestamps: int = 50364
    max_length: int = 448
    vocab_size: int = 51866
    suppress: list[int] = field(default_factory=list)
    lang_to_id: dict[str, int] = field(default_factory=dict)
    task_to_id: dict[str, int] = field(default_factory=lambda: {"transcribe": 50360, "translate": 50359})

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps + 1

    @classmethod
    def from_dir(cls, path: Path) -> GenerationSettings:
        gen = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if float(gen.get("ctc_weight", 0) or 0) > 0:
            raise ValueError(
                "this model's generation config decodes jointly with its CTC head, which bolo does not run"
            )
        return cls(
            eos=int(gen["eos_token_id"]),
            start=int(gen["decoder_start_token_id"]),
            no_timestamps=int(gen["no_timestamps_token_id"]),
            max_length=int(gen.get("max_length") or config.get("max_target_positions", 448)),
            vocab_size=int(config["vocab_size"]),
            suppress=[int(t) for t in gen.get("suppress_tokens") or []],
            lang_to_id={k.strip("<|>"): int(v) for k, v in (gen.get("lang_to_id") or {}).items()},
            task_to_id={str(k): int(v) for k, v in (gen.get("task_to_id") or {}).items()} or cls().task_to_id,
        )


# -- the network --------------------------------------------------------------------------------------


class Network(Protocol):
    """The two graphs (ONNX Runtime in production, a fake in tests)."""

    def encode(self, features: FloatArray, masks: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``(batch, 128, 3000)`` features, ``(batch, 4, 1500)`` masks -> cross-attention keys and values."""
        ...

    def decode(
        self, ids: IntArray, self_k: FloatArray, self_v: FloatArray, cross_k: FloatArray, cross_v: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Logits of the last position ``(batch, vocab)`` and the self-attention cache with ``ids`` appended."""
        ...

    def empty_cache(self, batch: int) -> FloatArray:
        """A self-attention cache of length 0 for ``batch`` sequences."""
        ...


class OnnxNetwork:
    def __init__(
        self,
        encoder: Path,
        decoder: Path,
        threads: int,
        device: str = "cpu",
        decoder_device: str | None = None,
        split: dict[str, int] | None = None,
    ) -> None:
        import onnxruntime as ort

        from bolo.onnx_split import make_session

        def options() -> Any:
            o = ort.SessionOptions()
            o.intra_op_num_threads = max(1, threads)
            o.inter_op_num_threads = 1
            o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return o

        def session(path: Path, on: str, role: str) -> Any:
            budget = (split or {}).get(role)
            if budget:  # this graph is cut in two: its front on ``on``, its back on the CPU (bolo.onnx_split)
                return make_session(path, providers_for(on), options, budget)
            return create_session(str(path), options(), providers_for(on), role=role)

        # the scheduler may place the two graphs apart (bolo.placement: the larger one on the GPU when only one fits)
        self.encoder = session(encoder, device, "encoder")
        self.decoder = session(decoder, decoder_device or device, "decoder")
        names = {i.name for i in self.decoder.get_inputs()}
        if names != {"input_ids", "self_k", "self_v", "cross_k", "cross_v"}:
            raise ValueError(f"{decoder} is not a DiCoW decoder graph (inputs {sorted(names)})")
        shape = next(i.shape for i in self.decoder.get_inputs() if i.name == "self_k")
        self.layers, self.heads, self.head_dim = int(shape[0]), int(shape[2]), int(shape[4])

    def encode(self, features: FloatArray, masks: FloatArray) -> tuple[FloatArray, FloatArray]:
        k, v = self.encoder.run(
            None,
            {
                "input_features": np.ascontiguousarray(features, dtype=np.float32),
                "stno_mask": np.ascontiguousarray(masks, dtype=np.float32),
            },
        )
        return k, v

    def decode(
        self, ids: IntArray, self_k: FloatArray, self_v: FloatArray, cross_k: FloatArray, cross_v: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        logits, k, v = self.decoder.run(
            None,
            {
                "input_ids": np.ascontiguousarray(ids, dtype=np.int64),
                "self_k": self_k,
                "self_v": self_v,
                "cross_k": cross_k,
                "cross_v": cross_v,
            },
        )
        return logits, k, v

    def empty_cache(self, batch: int) -> FloatArray:
        return np.zeros((self.layers, batch, self.heads, 0, self.head_dim), np.float32)


# -- decoding -----------------------------------------------------------------------------------------


@dataclass
class DecodedSegment:
    start: float
    end: float
    tokens: list[int]


def apply_timestamp_rules(
    scores: FloatArray, generated: Sequence[Sequence[int]], gen: GenerationSettings
) -> FloatArray:
    """transformers' ``WhisperTimeStampLogitsProcessor`` with DiCoW's change: as the first token, end-of-text is
    allowed besides a timestamp (``utils.WhisperTimeStampLogitsProcessorCustom``). ``scores`` (batch, vocab) are
    the logits after the suppress-token list; ``generated`` the tokens decoded so far in this window, per row."""
    ts = gen.timestamp_begin
    out = scores.copy()
    out[:, gen.no_timestamps] = -np.inf
    first = all(len(seq) == 0 for seq in generated)
    for k, seq in enumerate(generated):
        last_was_ts = len(seq) >= 1 and seq[-1] >= ts
        penultimate_was_ts = len(seq) < 2 or seq[-2] >= ts
        if last_was_ts:
            if penultimate_was_ts:  # has to be non-timestamp
                out[k, ts:] = -np.inf
            else:  # cannot be normal text tokens
                out[k, : gen.eos] = -np.inf
        stamps = [t for t in seq if t >= ts]
        if stamps:
            last = stamps[-1] if last_was_ts and not penultimate_was_ts else stamps[-1] + 1
            out[k, ts:last] = -np.inf
    if first:
        out[:, :ts] = -np.inf
    logprobs = _log_softmax(out)
    for k in range(len(generated)):
        if _logsumexp(logprobs[k, ts:]) > float(np.max(logprobs[k, :ts])):
            out[k, :ts] = -np.inf
    if first:
        out[:, gen.eos] = scores[:, gen.eos]
    return np.asarray(out, dtype=np.float32)


def _log_softmax(x: FloatArray) -> FloatArray:
    m = np.max(x, axis=-1, keepdims=True)
    shifted = x - m
    with np.errstate(divide="ignore"):
        lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return np.asarray(shifted - lse, dtype=np.float32)


def _logsumexp(x: FloatArray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


def greedy(
    network: Network,
    cross_k: FloatArray,
    cross_v: FloatArray,
    init: IntArray,
    gen: GenerationSettings,
) -> list[list[int]]:
    """Greedy decoding of one window for every row of ``init`` (batch, prompt); tokens after the prompt, up to
    (not including) end-of-text.

    Rows are independent, so a row that has ended leaves the batch (transformers keeps feeding it padding, which
    changes nothing for the other rows)."""
    batch = len(init)
    k, v = network.empty_cache(batch), network.empty_cache(batch)
    logits, k, v = network.decode(init, k, v, cross_k, cross_v)
    generated: list[list[int]] = [[] for _ in range(batch)]
    rows = np.arange(batch)  # rows still decoding, in the order of the arrays
    length = init.shape[1]
    suppress = np.asarray(gen.suppress, np.int64)
    while True:
        scores = np.asarray(logits, np.float32).copy()
        if len(suppress):
            scores[:, suppress] = -np.inf
        scores = apply_timestamp_rules(scores, [generated[r] for r in rows], gen)
        nxt = np.argmax(scores, axis=-1)
        for r, token in zip(rows, nxt):
            generated[r].append(int(token))
        length += 1
        keep = nxt != gen.eos
        if not keep.any() or length >= gen.max_length:
            break
        if not keep.all():
            rows, nxt = rows[keep], nxt[keep]
            k, v, cross_k, cross_v = k[:, keep], v[:, keep], cross_k[:, keep], cross_v[:, keep]
        logits, k, v = network.decode(nxt[:, None].astype(np.int64), k, v, cross_k, cross_v)
    return [seq[:-1] if seq and seq[-1] == gen.eos else seq for seq in generated]


def retrieve_segments(
    seq: Sequence[int], time_offset: float, seek_num_frames: int, timestamp_begin: int
) -> tuple[list[DecodedSegment], int]:
    """A window's tokens cut into segments, and how far to move ``seek`` (mel frames): the model's own
    ``DiCoWGenerationMixin._retrieve_segment`` (transformers' version, with a different rule when the window
    has no pair of consecutive timestamps)."""
    tokens = np.asarray(seq, np.int64)
    is_ts = tokens >= timestamp_begin
    single_ending = is_ts[-2:].tolist() == [False, True]
    pairs = np.where(is_ts[:-1] & is_ts[1:])[0] + 1
    segments: list[DecodedSegment] = []
    if len(pairs) > 0:
        slices = pairs.tolist()
        if single_ending:
            slices.append(len(tokens))
        else:
            slices[-1] += 1
        last = 0
        for i, current in enumerate(slices):
            is_last = i == len(slices) - 1
            part = tokens[last:current]
            start = int(part[0]) - timestamp_begin
            end = int(part[-1 if not is_last or single_ending else -2]) - timestamp_begin
            segments.append(
                DecodedSegment(time_offset + start * TIME_PRECISION, time_offset + end * TIME_PRECISION, part.tolist())
            )
            last = current
        offset = seek_num_frames if single_ending else (int(tokens[last - 2]) - timestamp_begin) * INPUT_STRIDE
    else:
        stamps = tokens[is_ts]
        start_pos: float = 0.0
        last_pos: float = seek_num_frames // 2
        skip = False
        offset = seek_num_frames
        if len(stamps) > 1:
            start_pos = int(stamps[-2]) - timestamp_begin
            last_pos = int(stamps[-1]) - timestamp_begin
        elif len(stamps) == 1:
            start_pos = int(stamps[-1]) - timestamp_begin
            if start_pos > 200:  # the segment does not fit into the window: roll back
                offset = int(start_pos) * INPUT_STRIDE - 100
                skip = True
        elif len(tokens) > 1:
            pass  # decoding without timestamps: the output as it is
        else:
            skip = True
        if not skip:
            segments = [
                DecodedSegment(
                    time_offset + start_pos * TIME_PRECISION, time_offset + last_pos * TIME_PRECISION, tokens.tolist()
                )
            ]
            offset = seek_num_frames
    if offset <= 0:
        raise RuntimeError(f"DiCoW decoding did not advance (segment offset {offset})")
    return segments, offset


def window_inputs(
    features: FloatArray, masks: FloatArray, seek: int, max_frames: int
) -> tuple[FloatArray, FloatArray, int]:
    """One 30 s window from ``seek`` (mel frames): features padded with zeros, the mask padded with silence."""
    num = min(max_frames - seek, WINDOW_FRAMES)
    feats = np.zeros((features.shape[0], WINDOW_FRAMES), np.float32)
    part = features[:, seek : seek + num]
    feats[:, : part.shape[1]] = part
    seek_vad = seek // 2
    num_vad = min(max_frames // 2 - seek_vad, MASK_WINDOW)
    mask = np.zeros((4, MASK_WINDOW), np.float32)
    piece = masks[:, seek_vad : seek_vad + num_vad]
    mask[:, : piece.shape[1]] = piece
    mask[0, piece.shape[1] :] = 1.0
    return feats, mask, num


def decode_long_form(
    network: Network,
    features: FloatArray,
    masks: FloatArray,
    gen: GenerationSettings,
    language: str | None,
    task: str,
) -> tuple[list[list[DecodedSegment]], list[str | None]]:
    """transformers' Whisper long-form generation for several targets of the same recording (``masks``:
    ``(targets, 4, frames)``): segments per target, and the language each was decoded in."""
    n = len(masks)
    max_frames = int(features.shape[1])
    seek = [0] * n
    segments: list[list[DecodedSegment]] = [[] for _ in range(n)]
    task_id = gen.task_to_id.get(task, gen.task_to_id["transcribe"])
    lang_ids: list[int | None] = [gen.lang_to_id[language]] * n if language in gen.lang_to_id else [None] * n
    id_to_lang = {v: k for k, v in gen.lang_to_id.items()}
    while True:
        active = [i for i in range(n) if seek[i] < max_frames]
        if not active:
            break
        windows = [window_inputs(features, masks[i], seek[i], max_frames) for i in active]
        feats = np.stack([w[0] for w in windows])
        mask = np.stack([w[1] for w in windows])
        cross_k, cross_v = network.encode(feats, mask)
        undetected = [j for j, i in enumerate(active) if lang_ids[i] is None]
        if undetected:  # language detection on the first window, from <|startoftranscript|> alone
            idx = np.asarray(undetected)
            batch = len(idx)
            empty = network.empty_cache(batch)
            logits, _, _ = network.decode(
                np.full((batch, 1), gen.start, np.int64), empty, empty, cross_k[:, idx], cross_v[:, idx]
            )
            allowed = np.full(logits.shape[-1], -np.inf, np.float32)
            ids = np.asarray(sorted(gen.lang_to_id.values()), np.int64)
            allowed[ids] = 0.0
            for j, row in zip(undetected, np.asarray(logits) + allowed):
                lang_ids[active[j]] = int(np.argmax(row))
        init = np.asarray([[gen.start, lang_ids[i], task_id] for i in active], np.int64)
        sequences = greedy(network, cross_k, cross_v, init, gen)
        for j, i in enumerate(active):
            time_offset = seek[i] * TIME_PRECISION / INPUT_STRIDE
            found, offset = retrieve_segments(sequences[j], time_offset, windows[j][2], gen.timestamp_begin)
            segments[i].extend(found)
            seek[i] += offset
    return segments, [id_to_lang.get(lid) if lid is not None else None for lid in lang_ids]


# -- the model ----------------------------------------------------------------------------------------


class DiCoWOnnx(STTModel):
    """Target-speaker recognition conditioned on diarization (bolo ``speaker_conditioned`` STT model)."""

    sample_rate = SR
    speaker_conditioned = True
    supports_translate = True
    supports_word_timestamps = False
    languages = WHISPER_LANGUAGES
    max_chunk_s = WINDOW_S

    def __init__(
        self,
        network: Network,
        gen: GenerationSettings,
        tokens: TokenDecoder,
        *,
        batch: int = 4,
        skip_silent_s: float = 0.2,
    ) -> None:
        self.network = network
        self.gen = gen
        self.tokens = tokens
        self.batch = max(1, batch)
        #: a speaker with less target or overlapped speech than this (seconds) is not decoded at all
        self.skip_silent_s = skip_silent_s
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, path: Path, spec: dict[str, Any], opts: LoadOptions) -> DiCoWOnnx:
        encoder = _find(path, spec.get("encoder"), ("encoder.int8.onnx", "encoder.onnx"))
        decoder = _find(path, spec.get("decoder"), ("decoder.int8.onnx", "decoder.onnx"))
        gen = GenerationSettings.from_dir(path)
        extra: dict[str, Any] = {}
        if opts.device_for("decoder") != opts.device_for("encoder"):
            extra["decoder_device"] = opts.device_for("decoder")
        if opts.graph_split:
            extra["split"] = dict(opts.graph_split)
        network = OnnxNetwork(encoder, decoder, opts.threads, opts.device_for("encoder"), **extra)
        return cls(
            network,
            gen,
            TokenDecoder.from_dir(path, gen.eos),
            batch=int(spec.get("batch", 4)),
            skip_silent_s=float(spec.get("skip_silent_s", 0.2)),
        )

    # -- conditioned recognition -------------------------------------------------------------------
    def transcribe_speakers(self, request: SpeakerRequest) -> list[STTResult]:
        audio = np.ascontiguousarray(request.samples, dtype=np.float32)
        activity = request.activity
        n_spk = activity.num_speakers
        results = [STTResult("", request.language) for _ in range(n_spk)]
        if n_spk == 0 or len(audio) == 0:
            return results
        masks = stno_masks(activity_at_mask_rate(activity.probs, activity.frame_s, mask_frames(len(audio))))
        spoken = (masks[:, 1] + masks[:, 3]).sum(axis=1) / MASK_RATE
        active = [s for s in range(n_spk) if spoken[s] >= self.skip_silent_s]
        if not active:
            return results
        features = log_mel(audio)
        duration = len(audio) / SR
        with self._lock:
            for first in range(0, len(active), self.batch):
                group = active[first : first + self.batch]
                decoded, langs = decode_long_form(
                    self.network, features, masks[group], self.gen, request.language, request.task
                )
                for s, segs, lang in zip(group, decoded, langs):
                    results[s] = self._result(segs, request.language or lang, duration)
        return results

    # -- plain recognition (no diarization): the whole input is the target speaker -----------------------
    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            frames = max(1, int(np.ceil(len(r.samples) / SR * 100)))
            whole = SpeakerActivity(np.ones((frames, 1), np.float32), 0.01)
            out.append(self.transcribe_speakers(SpeakerRequest(r.samples, whole, r.language, r.task, r.prompt))[0])
        return out

    def _result(self, segments: Sequence[DecodedSegment], language: str | None, duration: float) -> STTResult:
        out: list[Segment] = []
        for seg in segments:
            text = self.tokens.decode(seg.tokens).strip()
            if not text:
                continue
            start = min(max(0.0, seg.start), duration)
            end = min(max(start, seg.end), duration)
            out.append(Segment(start, end, text))
        return STTResult(" ".join(s.text for s in out), language, out)


def graph_files(path: Path, spec: dict[str, Any]) -> dict[str, int]:
    """The two graphs and their sizes, for placement (empty when they are not there yet)."""
    try:
        encoder = _find(path, spec.get("encoder"), ("encoder.int8.onnx", "encoder.onnx"))
        decoder = _find(path, spec.get("decoder"), ("decoder.int8.onnx", "decoder.onnx"))
    except FileNotFoundError:
        return {}
    return {"encoder": _size(encoder), "decoder": _size(decoder)}


def _size(path: Path) -> int:
    total = path.stat().st_size
    data = path.with_name(path.name + ".data")  # external weights next to the graph
    if data.is_file():
        total += data.stat().st_size
    return total


def _find(path: Path, chosen: Any, defaults: tuple[str, ...]) -> Path:
    for name in ([str(chosen)] if chosen else []) + list(defaults):
        if (path / name).is_file():
            return path / name
    raise FileNotFoundError(
        f"no DiCoW ONNX graph ({' or '.join(([str(chosen)] if chosen else []) + list(defaults))}) in {path}"
    )
