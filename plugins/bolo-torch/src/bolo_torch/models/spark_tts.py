"""Spark-TTS 0.5B (SparkAudio): a Qwen2.5-0.5B language model writing BiCodec tokens, English and Chinese.

What runs is the authors' recipe (github.com/SparkAudio/Spark-TTS, ``cli/SparkTTS.py`` at the pinned commit,
Apache-2.0): the language model is an ordinary Qwen2 checkpoint (``LLM/``) for transformers' ``AutoModelForCausalLM``;
the BiCodec tokenizer and decoder (``BiCodec/``, with ``wav2vec2-large-xlsr-53/`` for its features) come from the
authors' ``sparktts`` package, which is not on PyPI: bolo-torch installs it as the code bundle ``spark-tts`` (the
repository's files at the pinned commit, each checked by sha256; ``bolo-torch code install spark-tts``, or on the
first load).

Two kinds of voice, as in the reference:

* **created** voices (``controllable_tts``): a gender and a pitch level, and the speed level from the request's
  ``speed`` (``<|task_controllable_tts|>`` with ``<|gender_N|><|pitch_label_N|><|speed_label_N|>``); the model writes
  the voice's global tokens itself;
* **cloned** voices (``tts`` with a prompt): the global tokens of a reference recording (bolo's built-in clips with
  ``reference: <sample id>``, ``ref_audio`` files, or ``<name>.wav`` + ``<name>.txt`` in ``voices_dir``), with its
  transcript and semantic tokens when the transcript is known. ``speed`` does not apply to them.

Sampling as in the reference: ``do_sample``, temperature 0.8, top-k 50, top-p 0.95, at most 3000 new tokens per
sentence. 16 kHz output. Weights CC-BY-NC-SA-4.0 (not for commercial use).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, TTSModel

from bolo_torch import upstream
from bolo_torch.models import peak_safe, to_voices
from bolo_torch.models.f5 import collect_voices
from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require
from bolo_torch.text import split_sentences

BUNDLE = "spark-tts"
LEVELS = ("very_low", "low", "moderate", "high", "very_high")
GENDERS = ("female", "male")
#: files of the model repository the recipe reads
MODEL_FILES = ("config.yaml", "LLM/config.json", "BiCodec/config.yaml", "BiCodec/model.safetensors",
               "wav2vec2-large-xlsr-53/config.json")  # fmt: skip
DEFAULT_VOICES = [
    {"id": "female", "name": "Female (created)", "gender": "f", "spark_gender": "female", "pitch": "moderate"},
    {"id": "male", "name": "Male (created)", "gender": "m", "spark_gender": "male", "pitch": "moderate"},
]
SEMANTIC = re.compile(r"bicodec_semantic_(\d+)")
GLOBAL = re.compile(r"bicodec_global_(\d+)")


def speed_level(speed: float) -> str:
    """bolo's speed factor (1.0 = normal) as one of Spark-TTS's five speed levels."""
    if speed <= 0.6:
        return "very_low"
    if speed < 0.85:
        return "low"
    if speed <= 1.15:
        return "moderate"
    if speed < 1.5:
        return "high"
    return "very_high"


def control_prompt(text: str, gender: str, pitch: str, speed: str, maps: Any) -> str:
    """``SparkTTS.process_prompt_control``: a created voice."""
    if gender not in maps.GENDER_MAP or pitch not in maps.LEVELS_MAP or speed not in maps.LEVELS_MAP:
        raise InvalidRequest(f"Spark-TTS voice needs gender {GENDERS} and pitch/speed levels {LEVELS}")
    attributes = (
        f"<|gender_{maps.GENDER_MAP[gender]}|><|pitch_label_{maps.LEVELS_MAP[pitch]}|>"
        f"<|speed_label_{maps.LEVELS_MAP[speed]}|>"
    )
    return "".join([
        maps.TASK_TOKEN_MAP["controllable_tts"], "<|start_content|>", text, "<|end_content|>",
        "<|start_style_label|>", attributes, "<|end_style_label|>",
    ])  # fmt: skip


def clone_prompt(text: str, global_ids: list[int], semantic_ids: list[int] | None, prompt_text: str | None,
                 maps: Any) -> str:  # fmt: skip
    """``SparkTTS.process_prompt``: a cloned voice (with the reference's transcript and semantic tokens when known)."""
    global_tokens = "".join(f"<|bicodec_global_{i}|>" for i in global_ids)
    if prompt_text and semantic_ids is not None:
        semantic_tokens = "".join(f"<|bicodec_semantic_{i}|>" for i in semantic_ids)
        parts = [
            maps.TASK_TOKEN_MAP["tts"], "<|start_content|>", prompt_text, text, "<|end_content|>",
            "<|start_global_token|>", global_tokens, "<|end_global_token|>", "<|start_semantic_token|>",
            semantic_tokens,
        ]  # fmt: skip
    else:
        parts = [
            maps.TASK_TOKEN_MAP["tts"], "<|start_content|>", text, "<|end_content|>", "<|start_global_token|>",
            global_tokens, "<|end_global_token|>",
        ]  # fmt: skip
    return "".join(parts)


class Backend:
    """What :class:`SparkTTS` needs from the networks (a fake in tests, the real ones from :func:`load`)."""

    def generate(self, prompt: str) -> str:
        """The language model's continuation of ``prompt``, decoded with special tokens skipped."""
        raise NotImplementedError

    def tokenize(self, path: str) -> tuple[list[int], list[int]]:
        """Global and semantic BiCodec tokens of a reference recording."""
        raise NotImplementedError

    def detokenize(self, global_ids: list[int], semantic_ids: list[int]) -> Any:
        """Audio (16 kHz) from global and semantic tokens."""
        raise NotImplementedError


class SparkTTS(TTSModel):
    def __init__(self, backend: Backend, voices: list[dict[str, Any]], maps: Any, *, sample_rate: int = 16000,
                 max_chars: int = 200) -> None:  # fmt: skip
        if not voices:
            raise InvalidRequest("Spark-TTS needs at least one voice")
        self.backend = backend
        self.maps = maps
        self._voices = {str(v["id"]).lower(): v for v in voices}
        self.voices = to_voices(voices)
        self.sample_rate = sample_rate
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._clones: dict[str, tuple[list[int], list[int]]] = {}

    def voice(self, name: str | None) -> dict[str, Any]:
        if not name:
            return next(iter(self._voices.values()))
        entry = self._voices.get(name.strip().lower())
        if entry is None:
            raise InvalidRequest(f"unknown voice {name!r}")
        return entry

    def prompt(self, text: str, voice: dict[str, Any], speed: float) -> tuple[str, list[int] | None]:
        """The prompt for one sentence, and the global tokens when the voice fixes them (cloned voices)."""
        if voice.get("ref_audio"):
            key = str(voice["ref_audio"])
            if key not in self._clones:
                self._clones[key] = self.backend.tokenize(key)
            global_ids, semantic_ids = self._clones[key]
            return clone_prompt(text, global_ids, semantic_ids, voice.get("ref_text"), self.maps), global_ids
        gender = str(voice.get("spark_gender") or {"f": "female", "m": "male"}.get(str(voice.get("gender")), ""))
        pitch = str(voice.get("pitch") or "moderate")
        return control_prompt(text, gender, pitch, speed_level(speed), self.maps), None

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        entry = self.voice(voice)
        for sentence in split_sentences(text, self.max_chars):
            if cancelled is not None and cancelled():
                return
            with self._lock:
                prompt, fixed = self.prompt(sentence, entry, float(speed))
                generated = self.backend.generate(prompt)
                semantic = [int(t) for t in SEMANTIC.findall(generated)]
                global_ids = fixed if fixed is not None else [int(t) for t in GLOBAL.findall(generated)]
                if not semantic or not global_ids:
                    continue  # the model wrote no speech for this sentence
                audio = np.asarray(self.backend.detokenize(global_ids, semantic), dtype=np.float32)
            if audio.size:
                yield peak_safe(audio)


class TransformersBackend(Backend):
    def __init__(self, model: Any, tokenizer: Any, audio_tokenizer: Any, device: str,
                 sampling: dict[str, Any]) -> None:  # fmt: skip
        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.sampling = sampling

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=True, **self.sampling)
        new = out[0][len(inputs.input_ids[0]) :]
        return str(self.tokenizer.batch_decode([new], skip_special_tokens=True)[0])

    def tokenize(self, path: str) -> tuple[list[int], list[int]]:
        import torch

        with torch.inference_mode():
            global_tokens, semantic_tokens = self.audio_tokenizer.tokenize(path)
        global_ids = [int(i) for i in global_tokens.reshape(-1).tolist()]
        return global_ids, [int(i) for i in semantic_tokens.reshape(-1).tolist()]

    def detokenize(self, global_ids: list[int], semantic_ids: list[int]) -> Any:
        import torch

        g = torch.tensor(global_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, 32)
        s = torch.tensor(semantic_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, n)
        with torch.inference_mode():
            return self.audio_tokenizer.detokenize(g, s)


def voices_for(root: Path, options: dict[str, Any]) -> list[dict[str, Any]]:
    """Created voices from the catalog (or the two defaults), then cloned ones (reference clips, voices_dir)."""
    created = [dict(v) for v in options.get("voices") or [] if isinstance(v, dict) and v.get("id")
               and not v.get("reference") and not v.get("ref_audio")]  # fmt: skip
    return (created or [dict(v) for v in DEFAULT_VOICES]) + collect_voices(root, options)


def load(ctx: LoadContext) -> SparkTTS:
    transformers = require("transformers", "spark-tts")
    for module in ("einops", "einx", "omegaconf", "soxr", "soundfile"):
        require(module, "spark-tts")
    import torch

    upstream.ensure(BUNDLE)
    root = model_dir(ctx.path, ctx.options)
    missing = [f for f in MODEL_FILES if not (root / f).is_file()]
    if missing:
        raise InvalidRequest(f"{root} is not a Spark-TTS model folder (missing {', '.join(missing)})")
    audio_tokenizer_mod = require("sparktts.models.audio_tokenizer", "spark-tts")
    maps = require("sparktts.utils.token_parser", "spark-tts")
    file_utils = require("sparktts.utils.file", "spark-tts")
    sample_rate = int(file_utils.load_config(str(root / "config.yaml"))["sample_rate"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root / "LLM"))
    model = transformers.AutoModelForCausalLM.from_pretrained(str(root / "LLM"), torch_dtype=ctx.dtype)
    model.to(ctx.device).eval()
    audio_tokenizer = audio_tokenizer_mod.BiCodecTokenizer(root, device=torch.device(ctx.device))
    sampling = {
        "max_new_tokens": int(ctx.option("max_new_tokens", 3000)),
        "temperature": float(ctx.option("temperature", 0.8)),
        "top_k": int(ctx.option("top_k", 50)),
        "top_p": float(ctx.option("top_p", 0.95)),
    }
    backend = TransformersBackend(model, tokenizer, audio_tokenizer, ctx.device, sampling)
    return SparkTTS(backend, voices_for(root, ctx.options), maps, sample_rate=sample_rate,
                    max_chars=int(ctx.option("max_chars", 200)))  # fmt: skip
