"""DiCoW: Whisper conditioned on speaker diarization (BUT-FIT target-speaker ASR).

DiCoW (Diarization-Conditioned Whisper) is Whisper large-v3-turbo with *FDDT* layers: before the encoder's
positional embeddings, every 20 ms frame is transformed by one of four learned affine maps, weighted by the
frame's **STNO** mask -- the probabilities that in this frame there is **S**ilence, only the **T**arget speaker,
only **N**on-target speakers, or the target speaker in **O**verlap with others. The recording is decoded once
per diarized speaker, with that speaker as the target, so each speaker gets a transcript of their own words, also
where people talk at once. No speaker embedding or enrolment recording is needed.

Input: the 16 kHz mixture and per-frame speaker activity (``SpeakerActivity``: one column per speaker, values in
0..1). The STNO masks are built from it exactly as BUT's inference pipeline builds them from a diarization
(``BUTSpeechFIT/DiCoW``, ``pipeline.py``, ``DiCoWPipeline.get_stno_mask``; Apache-2.0), at the 50 Hz rate of
Whisper's encoder input after its stride-2 convolution. With 0/1 activity (speaker turns) the masks are the
reference's; with probabilities the same products give soft masks.

Decoding is the model's own generation code (the repository's Python files, imported as a package: what
``trust_remote_code`` runs; Whisper long-form decoding in 30 s windows, greedy), which needs transformers 4.55. Segment times are taken from the generator's segments before the model
re-encodes them modulo 30 s.

SE-DiCoW (``use_enrollments`` in its config) additionally attends to a self-enrolment window: the 30 s of the
recording where the target speaker talks alone the most, selected as BUT's pipeline does
(``_process_enrollment_sample``).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, Segment, SpeakerActivity, STTModel, STTRequest, STTResult

from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency, model_dir, require

try:
    from bolo.engines.base import SpeakerRequest
except ImportError as exc:  # a bolo without speaker-conditioned recognition
    raise MissingDependency(
        "DiCoW needs a bolo release with speaker-conditioned recognition (0.2.0 or later): pip install -U bolo-speech"
    ) from exc

SR = 16000
#: STNO mask frames per second (Whisper's 100 mel frames per second after the encoder's stride-2 convolution)
MASK_RATE = 50
WINDOW_S = 30.0
#: Whisper's languages (tokenizer order), for the plugin's ``info.languages``
WHISPER_LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
)  # fmt: skip


def activity_at_mask_rate(activity: FloatArray, frame_s: float, n_frames: int) -> FloatArray:
    """Speaker activity ``(frames, speakers)`` resampled to ``n_frames`` mask frames of 20 ms: ``(speakers, n)``.

    Mask frame ``j`` takes the activity frame under its centre; frames past the end of the activity are silent.
    """
    probs = np.asarray(activity, dtype=np.float32)
    if probs.ndim != 2:
        raise InvalidRequest("speaker activity must be a (frames, speakers) matrix")
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


def enrolment_window(target: FloatArray, window: int = int(WINDOW_S * MASK_RATE)) -> int:
    """Start (mask frame) of the ``window`` frames with the most target-only speech (the first such window)."""
    if len(target) <= window:
        return 0
    sums = np.convolve(np.asarray(target, np.float64), np.ones(window), mode="valid")
    return int(np.argmax(sums))


@dataclass
class DecodedSegment:
    start: float
    end: float
    text: str


class Backend:
    """What the model wrapper needs from the network (a fake in tests, transformers in production)."""

    #: SE-DiCoW: also pass a self-enrolment window per speaker
    use_enrollments: bool = False

    def mask_frames(self, num_samples: int) -> int:
        """Mask frames for a recording of ``num_samples`` (the feature frames the model sees, halved)."""
        raise NotImplementedError

    def decode(
        self, audio: FloatArray, masks: FloatArray, language: str | None, task: str, enrol_starts: list[int] | None
    ) -> list[list[DecodedSegment]]:
        """Decode ``audio`` once per mask in ``masks`` ``(speakers, 4, mask_frames)``; segments per speaker."""
        raise NotImplementedError


class DiCoW(STTModel):
    """Target-speaker recognition conditioned on diarization (bolo ``speaker_conditioned`` STT model)."""

    sample_rate = SR
    speaker_conditioned = True
    supports_translate = True
    supports_word_timestamps = False
    languages = WHISPER_LANGUAGES
    max_chunk_s = WINDOW_S

    def __init__(self, backend: Backend, *, skip_silent_s: float = 0.2) -> None:
        self.backend = backend
        #: a speaker with less target or overlapped speech than this (seconds) is not decoded at all
        self.skip_silent_s = skip_silent_s
        self._lock = threading.Lock()

    # -- conditioned recognition -------------------------------------------------------------------
    def transcribe_speakers(self, request: SpeakerRequest) -> list[STTResult]:
        audio = np.ascontiguousarray(request.samples, dtype=np.float32)
        activity = request.activity
        n_spk = activity.num_speakers
        if n_spk == 0 or len(audio) == 0:
            return [STTResult("", request.language) for _ in range(n_spk)]
        n_frames = self.backend.mask_frames(len(audio))
        masks = stno_masks(activity_at_mask_rate(activity.probs, activity.frame_s, n_frames))
        spoken = (masks[:, 1] + masks[:, 3]).sum(axis=1) / MASK_RATE
        active = [s for s in range(n_spk) if spoken[s] >= self.skip_silent_s]
        results = [STTResult("", request.language) for _ in range(n_spk)]
        if not active:
            return results
        chosen = masks[active]
        starts = [enrolment_window(m[1]) for m in chosen] if self.backend.use_enrollments else None
        with self._lock:
            decoded = self.backend.decode(audio, chosen, request.language, request.task, starts)
        duration = len(audio) / SR
        for s, segments in zip(active, decoded):
            results[s] = _result(segments, request.language, duration)
        return results

    # -- plain recognition (no diarization): the whole input is the target speaker -----------------------
    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            frames = max(1, int(np.ceil(len(r.samples) / SR * 100)))
            whole = SpeakerActivity(np.ones((frames, 1), np.float32), 0.01)
            out.append(self.transcribe_speakers(SpeakerRequest(r.samples, whole, r.language, r.task, r.prompt))[0])
        return out


def _result(segments: Sequence[DecodedSegment], language: str | None, duration: float) -> STTResult:
    out: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start = min(max(0.0, seg.start), duration)
        end = min(max(start, seg.end), duration)
        out.append(Segment(start, end, text))
    return STTResult(" ".join(s.text for s in out), language, out)


# -- transformers backend ------------------------------------------------------------------------------


def lower_upper_mapping(tokenizer: Any) -> dict[int, int]:
    """``tokenizer.upper_cased_tokens`` as BUT's ``inference.py`` builds it (lower-case token id -> upper-case)."""
    mapping: dict[int, int] = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if not token:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower = token[0] + token[1].lower() + token[2:]
        else:
            lower = token[0].lower() + token[1:]
        if lower != token and lower in vocab:
            mapping[vocab[lower]] = index
    return mapping


class TransformersBackend(Backend):
    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        tokenizer: Any,
        *,
        device: str,
        dtype: Any,
        batch: int = 4,
        generate_options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.fe = feature_extractor
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.batch = max(1, batch)
        self.generate_options = dict(generate_options or {})
        self.use_enrollments = bool(getattr(model.config, "use_enrollments", False))

    def _features(self, audio: FloatArray) -> Any:
        long = len(audio) > WINDOW_S * SR
        kwargs: dict[str, Any] = {"sampling_rate": SR, "return_tensors": "pt", "return_attention_mask": True}
        if long:  # as transformers' ASR pipeline does for Whisper long-form input
            kwargs.update(truncation=False, padding="longest")
        return self.fe(audio, **kwargs)["input_features"]

    def mask_frames(self, num_samples: int) -> int:
        if num_samples <= WINDOW_S * SR:
            return int(WINDOW_S * MASK_RATE)  # short input is padded to 30 s
        return int(self._features(np.zeros(num_samples, np.float32)).shape[-1]) // 2

    def decode(
        self, audio: FloatArray, masks: FloatArray, language: str | None, task: str, enrol_starts: list[int] | None
    ) -> list[list[DecodedSegment]]:
        import torch

        features = self._features(audio)
        n_mel = int(features.shape[-1])
        out: list[list[DecodedSegment]] = []
        for first in range(0, len(masks), self.batch):
            part = masks[first : first + self.batch]
            n = len(part)
            feats = features.repeat(n, 1, 1).to(self.device, dtype=self.dtype)
            stno = torch.from_numpy(np.ascontiguousarray(part[:, :, : n_mel // 2])).to(self.device, dtype=self.dtype)
            kwargs: dict[str, Any] = {
                "input_features": feats,
                "attention_mask": torch.ones(n, n_mel, dtype=torch.bool, device=self.device),
                "stno_mask": stno,
                "return_timestamps": True,
                "task": task,
                **self.generate_options,
            }
            if language:
                kwargs["language"] = language
            if self.use_enrollments and enrol_starts is not None:
                window = int(WINDOW_S * MASK_RATE)
                starts = enrol_starts[first : first + n]
                kwargs["enrollments"] = {
                    "input_features": torch.stack(
                        [feats[i, :, 2 * s : 2 * s + 2 * window] for i, s in enumerate(starts)]
                    ),
                    "attention_mask": torch.ones(n, 2 * window, dtype=torch.bool, device=self.device),
                    "stno_mask": torch.stack([stno[i, :, s : s + window] for i, s in enumerate(starts)]),
                }
            out.extend(self._generate(kwargs, n))
        return out

    def _generate(self, kwargs: dict[str, Any], n: int) -> list[list[DecodedSegment]]:
        import torch

        captured: dict[str, Any] = {}
        original: Callable[[Any], Any] = self.model._fix_timestamps_from_segmentation

        def capture(sequences: Any) -> Any:
            # the generator's segments carry absolute times; the model then re-encodes them modulo 30 s
            captured["segments"] = [list(s) for s in sequences["segments"]]
            return original(sequences)

        self.model._fix_timestamps_from_segmentation = capture
        try:
            with torch.inference_mode():
                tokens = self.model.generate(**kwargs)
        finally:
            del self.model._fix_timestamps_from_segmentation  # back to the class's method
        if "segments" in captured:
            return [
                [
                    DecodedSegment(
                        float(seg["start"]),
                        float(seg["end"]),
                        self.tokenizer.decode(seg["tokens"], skip_special_tokens=True),
                    )
                    for seg in segs
                ]
                for segs in captured["segments"]
            ]
        texts = self.tokenizer.batch_decode(tokens, decode_with_timestamps=True, skip_special_tokens=True)
        return [parse_timestamped(t) for t in texts[:n]]


def parse_timestamped(text: str) -> list[DecodedSegment]:
    """``<|0.00|> text<|2.40|>`` pairs of one 30 s window (short-form output) as segments."""
    import re

    return [
        DecodedSegment(float(a), float(b), body)
        for a, body, b in re.findall(r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>", text)
    ]


def cpu_has_fast_bf16(cpuinfo: str = "/proc/cpuinfo") -> bool:
    """True when this CPU computes bfloat16 natively (AMX or AVX512-BF16): torch's probes where it has them,
    else the CPU flags Linux reports. False when neither can tell (macOS, Windows)."""
    import torch

    cpu = getattr(torch._C, "_cpu", None)
    for probe in ("_is_amx_tile_supported", "_is_avx512_bf16_supported"):
        check = getattr(cpu, probe, None)
        try:
            if callable(check) and check():
                return True
        except Exception:
            continue
    try:
        with Path(cpuinfo).open(encoding="utf-8") as fh:
            flags = next((line.split(":", 1)[1].split() for line in fh if line.startswith("flags")), [])
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def import_model_code(root: Path) -> ModuleType:
    """The model repository's own Python code (``modeling_dicow`` and the modules it imports), as a package.

    This is what ``trust_remote_code=True`` runs. transformers 4.55 cannot load it from a local directory: it
    copies only the modules ``modeling_dicow.py`` imports directly, not ``decoding.py`` and ``utils.py``, which
    ``generation.py`` imports. The package is named after a hash of the code, so two revisions do not mix, and
    no ``__pycache__`` is written into the model directory.
    """
    files = sorted(root.glob("*.py"))
    if not any(f.name == "modeling_dicow.py" for f in files):
        raise InvalidRequest(f"{root} has no modeling_dicow.py: not a DiCoW model directory")
    digest = hashlib.sha256(b"".join(f.name.encode() + f.read_bytes() for f in files)).hexdigest()[:16]
    name = f"bolo_torch_dicow_{digest}"
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        hf_logging = None
    level = hf_logging.get_verbosity() if hf_logging is not None else None
    bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return importlib.import_module(f"{name}.modeling_dicow")
    finally:
        sys.dont_write_bytecode = bytecode
        if hf_logging is not None and level is not None:
            hf_logging.set_verbosity(level)  # the model code switches transformers to debug logging on import


def load(ctx: LoadContext) -> DiCoW:
    transformers = require("transformers", "dicow")
    import torch

    version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if version != (4, 55):
        raise InvalidRequest(
            f"DiCoW's model code needs transformers 4.55 (this environment has {transformers.__version__}); "
            'install it with: pip install "bolo-torch[dicow]"',
        )
    root = model_dir(ctx.path, ctx.options)
    dtype = ctx.dtype
    explicit = os.environ.get("BOLO_TORCH_DTYPE") or ctx.options.get("dtype")
    if ctx.device == "cpu" and not explicit and cpu_has_fast_bf16():
        dtype = torch.bfloat16  # with AMX / AVX512-BF16, bf16 halves memory and is much faster than fp32
    code = import_model_code(root)
    model = code.DiCoWForConditionalGeneration.from_pretrained(str(root), torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(ctx.device).eval()
    fe = transformers.AutoFeatureExtractor.from_pretrained(str(root))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root))
    tokenizer.upper_cased_tokens = lower_upper_mapping(tokenizer)
    model.set_tokenizer(tokenizer)
    options = {k: ctx.option(k) for k in ("num_beams",) if ctx.option(k) is not None}
    backend = TransformersBackend(
        model,
        fe,
        tokenizer,
        device=ctx.device,
        dtype=dtype,
        batch=int(ctx.option("batch", 4)),
        generate_options=options,
    )
    return DiCoW(backend, skip_silent_s=float(ctx.option("skip_silent_s", 0.2)))
