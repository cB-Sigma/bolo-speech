"""F5-TTS flow matching that clones the voice of a short reference recording: IndicF5 (AI4Bharat) and the
original F5-TTS checkpoints (SWivid, option ``layout: f5-tts``).

The model repository ships its weights (``model.safetensors``: the EMA DiT and the Vocos vocoder), its vocabulary
and its own copy of the ``f5_tts`` package. The repository's ``model.py`` (``trust_remote_code``) is not used: it
looks the vocabulary up as a Hub repository id, wraps the models in ``torch.compile`` and reads audio with
``torchaudio.load``; this loader builds the same DiT (dim 1024, depth 22, heads 16) from the bundled ``f5_tts``
instead, loads the weights from the local file, and fetches only Vocos's ``config.yaml``
(``charactr/vocos-mel-24khz``, MIT, not gated).

The original checkpoints (``SWivid/F5-TTS``: ``F5TTS_v1_Base/model_1250000.safetensors`` and ``vocab.txt``,
CC-BY-NC-4.0 weights) run on the ``f5_tts`` package from PyPI (MIT, extra ``f5-original``): its ``load_model``
builds the v1 DiT from the options ``checkpoint`` and ``vocab`` (paths in the model directory) and its Vocos
loader reads the pinned ``charactr/vocos-mel-24khz`` files.

A voice is a reference clip plus its transcript: the catalog's voices (``ref_audio`` relative to the model
directory, ``ref_text``; or ``reference: <bolo sample id>``, one of bolo's built-in synthetic clips with its
text) and any ``<name>.wav`` + ``<name>.txt`` pair in ``voices_dir`` (option or ``BOLO_TORCH_VOICES_DIR``).
Clips are cut to 15 s, as F5-TTS does.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, TTSModel

from bolo_torch.models import peak_safe, to_voices, voices_from_options
from bolo_torch.runtime import InvalidRequest, LoadContext, hub_snapshot, model_dir, require
from bolo_torch.text import split_sentences

VOCOS_REPO = "charactr/vocos-mel-24khz"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
DIT_CONFIG = {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
LAYOUTS = ("indicf5", "f5-tts")
MAX_REF_S = 15.0


def batch_result(result: Any) -> Any:
    """``infer_batch_process`` returns ``(wave, rate, mel)`` in IndicF5's copy of f5_tts and yields it once in the
    PyPI package (where the function also streams)."""
    if inspect.isgenerator(result):
        return next(result)
    return result


def normalize_ref_text(text: str) -> str:
    """F5-TTS expects the reference transcript to end with a sentence break and a space."""
    text = text.strip()
    if text.endswith((". ", "。")):
        return text
    if text.endswith((".", "।", "?", "!")):
        return text + " "
    return text + ". "


class F5TTS(TTSModel):
    def __init__(
        self,
        generate: Callable[[str, dict[str, Any], float], Any],
        voices: list[dict[str, Any]],
        *,
        sample_rate: int = 24000,
        max_chars: int = 160,
    ) -> None:
        if not voices:
            raise InvalidRequest("F5-TTS needs at least one reference voice (catalog voices or voices_dir)")
        self.generate = generate
        self._voices = {str(v["id"]).lower(): v for v in voices}
        self.voices = to_voices(voices)
        self.sample_rate = sample_rate
        self.max_chars = max_chars
        self._lock = threading.Lock()

    def reference(self, voice: str | None) -> dict[str, Any]:
        if not voice:
            return next(iter(self._voices.values()))
        entry = self._voices.get(voice.strip().lower())
        if entry is None:
            raise InvalidRequest(f"unknown voice {voice!r}")
        return entry

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        ref = self.reference(voice)
        for sentence in split_sentences(text, self.max_chars):
            if cancelled is not None and cancelled():
                return
            with self._lock:
                audio = np.asarray(self.generate(sentence, ref, speed))
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            if audio.size:
                yield peak_safe(audio)


def sample_reference(sample_id: str) -> tuple[Path, str] | None:
    """A built-in bolo sample clip (``bolo.samples``) and its transcript, or None when there is no such clip."""
    from importlib import resources

    try:
        from bolo import samples

        entry = samples.get(sample_id)
    except Exception:
        return None
    path = Path(str(resources.files("bolo.samples") / str(entry.get("file", ""))))
    text = str(entry.get("text") or "")
    return (path, text) if path.is_file() and text else None


def collect_voices(root: Path, options: dict[str, Any]) -> list[dict[str, Any]]:
    voices: list[dict[str, Any]] = []
    for v in voices_from_options(options):
        if v.get("reference") and not v.get("ref_audio"):
            found = sample_reference(str(v["reference"]))
            if found is not None:
                voices.append({**v, "ref_audio": str(found[0]), "ref_text": v.get("ref_text") or found[1]})
            continue
        if not v.get("ref_audio") or not v.get("ref_text"):
            continue
        path = Path(str(v["ref_audio"]))
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            voices.append({**v, "ref_audio": str(path)})
    folder = options.get("voices_dir") or os.environ.get("BOLO_TORCH_VOICES_DIR")
    if folder and Path(folder).is_dir():
        for wav in sorted(Path(folder).glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.is_file():
                voices.append(
                    {"id": wav.stem, "name": wav.stem, "ref_audio": str(wav),
                     "ref_text": txt.read_text(encoding="utf-8").strip()}
                )  # fmt: skip
    return voices


def load(ctx: LoadContext) -> F5TTS:
    layout = str(ctx.option("layout", "indicf5"))
    if layout not in LAYOUTS:
        raise InvalidRequest(f"unknown F5-TTS layout {layout!r} (one of {', '.join(LAYOUTS)})")
    root = model_dir(ctx.path, ctx.options)
    if layout == "f5-tts":
        ui, model, vocoder = _load_package(ctx, root)
    else:
        ui, model, vocoder = _load_indicf5(ctx, root)
    return _model(ctx, root, ui, model, vocoder)


def _load_package(ctx: LoadContext, root: Path) -> tuple[Any, Any, Any]:
    """The original SWivid checkpoints, on the ``f5_tts`` package from PyPI."""
    require("f5_tts", "f5-original")
    ui = importlib.import_module("f5_tts.infer.utils_infer")
    dit = importlib.import_module("f5_tts.model").DiT
    checkpoint = root / str(ctx.option("checkpoint", "F5TTS_v1_Base/model_1250000.safetensors"))
    vocab = root / str(ctx.option("vocab", "F5TTS_v1_Base/vocab.txt"))
    if not checkpoint.is_file() or not vocab.is_file():
        raise InvalidRequest(f"{root} does not hold the F5-TTS checkpoint {checkpoint.name} and its vocab.txt")
    config = {**DIT_CONFIG, **dict(ctx.option("dit_config", {}))}
    model = ui.load_model(dit, config, str(checkpoint), mel_spec_type="vocos", vocab_file=str(vocab), device=ctx.device)
    vocos_dir = hub_snapshot(
        str(ctx.option("vocoder_repo", VOCOS_REPO)),
        str(ctx.option("vocoder_revision", VOCOS_REVISION)),
        ["config.yaml", "pytorch_model.bin"],
    )
    vocoder = ui.load_vocoder("vocos", is_local=True, local_path=str(vocos_dir), device=ctx.device)
    return ui, model, vocoder


def _load_indicf5(ctx: LoadContext, root: Path) -> tuple[Any, Any, Any]:
    import torch

    require("vocos", "f5")
    require("torchdiffeq", "f5")
    require("x_transformers", "f5")
    safetensors = require("safetensors.torch", "f5")
    if not (root / "f5_tts").is_dir() or not (root / "model.safetensors").is_file():
        raise InvalidRequest(f"{root} does not look like the IndicF5 repository (f5_tts/ and model.safetensors)")
    # the bundled f5_tts matches the weights; it must win over any other installed f5_tts
    if sys.path[:1] != [str(root)]:
        sys.path.insert(0, str(root))
    for name in [m for m in sys.modules if m == "f5_tts" or m.startswith("f5_tts.")]:
        del sys.modules[name]
    ui = importlib.import_module("f5_tts.infer.utils_infer")
    f5_model = importlib.import_module("f5_tts.model")
    f5_utils = importlib.import_module("f5_tts.model.utils")
    vocos_mod = importlib.import_module("vocos")

    state = safetensors.load_file(str(root / "model.safetensors"))
    vocab_char_map, vocab_size = f5_utils.get_tokenizer(str(root / "checkpoints" / "vocab.txt"), "custom")
    model = f5_model.CFM(
        transformer=f5_model.DiT(**DIT_CONFIG, text_num_embeds=vocab_size, mel_dim=ui.n_mel_channels),
        mel_spec_kwargs={
            "n_fft": ui.n_fft, "hop_length": ui.hop_length, "win_length": ui.win_length,
            "n_mel_channels": ui.n_mel_channels, "target_sample_rate": ui.target_sample_rate, "mel_spec_type": "vocos",
        },
        odeint_kwargs={"method": ui.ode_method},
        vocab_char_map=vocab_char_map,
    )  # fmt: skip
    skip = ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window")
    dit = {
        k.split("ema_model._orig_mod.", 1)[1]: v
        for k, v in state.items()
        if k.startswith("ema_model._orig_mod.") and not k.endswith(skip)
    }
    model.load_state_dict(dit)
    model = model.to(ctx.device, torch.float32).eval()
    vocos_dir = hub_snapshot(
        str(ctx.option("vocoder_repo", VOCOS_REPO)),
        str(ctx.option("vocoder_revision", VOCOS_REVISION)),
        ["config.yaml"],
    )
    vocoder = vocos_mod.Vocos.from_hparams(str(vocos_dir / "config.yaml"))
    vocoder.load_state_dict(
        {k.split("vocoder._orig_mod.", 1)[1]: v for k, v in state.items() if k.startswith("vocoder._orig_mod.")}
    )
    vocoder = vocoder.to(ctx.device).eval()
    del state
    return ui, model, vocoder


def _model(ctx: LoadContext, root: Path, ui: Any, model: Any, vocoder: Any) -> F5TTS:
    import soundfile as sf
    import torch

    nfe = int(ctx.option("nfe_step", 32))
    refs: dict[str, tuple[Any, int, str]] = {}

    def reference(entry: dict[str, Any]) -> tuple[Any, int, str]:
        key = str(entry["ref_audio"])
        if key not in refs:
            audio, sr = sf.read(key, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)[: int(MAX_REF_S * sr)]
            refs[key] = (
                torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0),
                int(sr),
                normalize_ref_text(str(entry["ref_text"])),
            )
        return refs[key]

    def generate(text: str, entry: dict[str, Any], speed: float) -> Any:
        audio, sr, ref_text = reference(entry)
        seconds = audio.shape[-1] / sr
        max_chars = int(len(ref_text.encode("utf-8")) / seconds * (25 - seconds))
        batches = ui.chunk_text(text, max_chars=max(max_chars, 20))
        with torch.inference_mode():
            result = ui.infer_batch_process(
                (audio, sr), ref_text, batches, model, vocoder, mel_spec_type="vocos", progress=None,
                nfe_step=nfe, speed=float(speed), device=ctx.device,
            )  # fmt: skip
            wave, _, _ = batch_result(result)
        return wave

    return F5TTS(
        generate, collect_voices(root, ctx.options), sample_rate=int(ui.target_sample_rate),
        max_chars=int(ctx.option("max_chars", 160)),
    )  # fmt: skip
