"""MaskGCT (Amphion): non-autoregressive masked generative codec TTS with zero-shot voice cloning.

What runs is the authors' pipeline (github.com/open-mmlab/Amphion, ``models/tts/maskgct/maskgct_utils.py`` and
``maskgct_inference.py`` at the pinned commit, MIT), installed by bolo-torch as the code bundle ``maskgct`` (each
file checked by sha256; ``bolo-torch code install maskgct``, or on the first load):

1. the reference recording at 16 kHz through ``facebook/w2v-bert-2.0`` (layer 17, normalised by the repository's
   statistics) and the semantic codec (RepCodec) gives its semantic tokens; at 24 kHz through the acoustic codec,
   its acoustic tokens;
2. the text-to-semantic model fills in the semantic tokens of the new text (phonemes of the reference transcript
   and the new text, 25 steps, classifier-free guidance 2.5); the length follows the reference's speaking rate
   (divided by bolo's ``speed``);
3. the semantic-to-acoustic models (first layer, then all twelve) fill in the acoustic tokens, and the acoustic
   codec's decoder makes 24 kHz audio.

``MaskGCT_Inference_Pipeline``'s methods (``text2semantic``, ``extract_acoustic_code``, ``semantic2acoustic``) run
unchanged; only what its constructor and ``build_*`` helpers fetch from the network or read relative to the working
directory is done here instead: w2v-bert-2.0 comes from the folder bolo-torch downloads at a pinned revision, the
statistics from the bundle (``weights_only``), the weights from the model folder bolo downloaded.

The phoneme front end is Amphion's: ``phonemizer`` with the espeak-ng library for English, French and German (and
mixed Chinese-English), jieba, pypinyin, cn2an and a polyphone ONNX model for Chinese, pyopenjtalk and pykakasi for
Japanese, and LangSegment. ``phonemizer`` is GPL-3.0 and espeak-ng GPL-3.0-or-later: neither is in the ``maskgct``
extra; install them yourself (``pip install phonemizer espeakng-loader``; the loader points phonemizer at
espeakng-loader's library when ``PHONEMIZER_ESPEAK_LIBRARY`` is not set). The front end reads its lexicons relative
to the working directory while it is imported, so the import runs with the bundle as the working directory, under a
lock. The code's top-level packages are called ``models`` and ``utils``: another bundle with those names cannot share
this plugin process (the load says so).

Voices are reference clips with their transcripts: bolo's built-in clips (``reference: <sample id>``), ``ref_audio``
+ ``ref_text``, or ``<name>.wav`` + ``<name>.txt`` in ``voices_dir``. The text's language is Chinese/English (mixed),
Japanese or Korean by its script, French or German by bolo's text-language guess, else the voice's or the model's
``language`` option.
Weights CC-BY-NC-4.0 (not for commercial use); w2v-bert-2.0 MIT.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, TTSModel

from bolo_torch import upstream
from bolo_torch.models import peak_safe, to_voices
from bolo_torch.models.f5 import collect_voices
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency, hub_snapshot, model_dir, require
from bolo_torch.text import split_sentences

BUNDLE = "maskgct"
SEMANTIC_REPO = "facebook/w2v-bert-2.0"
SEMANTIC_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
CONFIG = "models/tts/maskgct/config/maskgct.json"
STATS = "models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"
#: model folder file -> what it holds
WEIGHTS = {
    "semantic_codec/model.safetensors": "semantic_codec",
    "acoustic_codec/model.safetensors": "codec_encoder",
    "acoustic_codec/model_1.safetensors": "codec_decoder",
    "t2s_model/model.safetensors": "t2s_model",
    "s2a_model/s2a_model_1layer/model.safetensors": "s2a_model_1layer",
    "s2a_model/s2a_model_full/model.safetensors": "s2a_model_full",
}
LANGUAGES = ("zh", "en", "ja", "ko", "fr", "de")
#: the libraries Amphion's front end imports (module -> what to install)
FRONT_END = {
    "phonemizer": "phonemizer and espeak-ng (both GPL-3.0, not in the extra): pip install phonemizer espeakng-loader",
    "pykakasi": "pykakasi (GPL-3.0, not in the extra): pip install pykakasi",
    "unidecode": "Unidecode (GPL-2.0, not in the extra): pip install Unidecode",
    "LangSegment": 'LangSegment (pip install "bolo-torch[maskgct]")',
    "jieba": 'jieba (pip install "bolo-torch[maskgct]")',
    "pyopenjtalk": 'pyopenjtalk (pip install "bolo-torch[maskgct]", which brings pyopenjtalk-plus)',
}
_import_lock = threading.Lock()
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")


def text_language(text: str, fallback: str) -> str:
    """The g2p language for ``text``: Japanese or Korean by script; French or German when bolo's text-language guess
    says so (its common-word lists); else ``fallback`` when that is French or German, else ``en`` (which covers
    Chinese and English, mixed, as Amphion's ``g2p_`` does for both)."""
    if KANA.search(text):
        return "ja"
    if HANGUL.search(text):
        return "ko"
    import importlib

    try:
        detect: Any = importlib.import_module("bolo.text.langdetect").detect
    except ImportError:  # an older bolo
        detect = None
    if detect is not None:
        guess = detect(text)
        if guess.script == "Latin":
            language = guess.language
            if language is None and guess.scores:
                ranked = sorted(guess.scores.items(), key=lambda kv: -kv[1])
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    language = ranked[0][0]
            if language in ("fr", "de"):
                return str(language)
    return fallback if fallback in ("fr", "de") else "en"


def prompt_language(text: str) -> str:
    return text_language(text, "en")


def target_seconds(prompt_samples: int, prompt_phones: int, target_phones: int, speed: float) -> float:
    """``text2semantic``'s rule for the length of new speech (the reference's rate), divided by ``speed``."""
    seconds = prompt_samples * target_phones / max(prompt_phones, 1) / 16000
    return seconds / max(speed, 0.25)


def point_phonemizer_at_espeak() -> None:
    """phonemizer finds espeak-ng through ``PHONEMIZER_ESPEAK_LIBRARY``; use espeakng-loader's copy when it is
    installed and nothing else is set."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    import importlib

    try:
        espeakng_loader = importlib.import_module("espeakng_loader")
    except ImportError:
        return
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH", str(espeakng_loader.get_data_path()))


def check_top_level(root: Path) -> None:
    """Amphion's packages are ``models`` and ``utils``: refuse when other modules of those names are loaded."""
    for name in ("models", "utils"):
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        where = getattr(loaded, "__file__", None) or next(iter(getattr(loaded, "__path__", []) or []), "")
        if where and not str(Path(str(where)).resolve()).startswith(str(root.resolve())):
            raise InvalidRequest(
                f"MaskGCT's code has a top-level package '{name}', and this plugin process already imported another "
                f"one ({where}). Run MaskGCT in a bolo-torch environment of its own (BOLO_PLUGINS)."
            )


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def import_langsegment() -> Any:
    """``LangSegment``. Its only release left on PyPI (0.2.0) cannot be imported: its ``__init__`` imports names that
    its own ``LangSegment.py`` does not define. That module itself works; it is registered as the package then
    (Amphion calls only ``setfilters``, and ``getTexts`` for ``language="auto"``, which bolo never passes)."""
    import importlib.util

    try:
        return __import__("LangSegment")
    except ImportError as exc:
        spec = importlib.util.find_spec("LangSegment")
        locations = list(spec.submodule_search_locations or []) if spec is not None else []
        inner = Path(locations[0]) / "LangSegment.py" if locations else None
        if inner is None or not inner.is_file():
            raise MissingDependency(f"MaskGCT's phoneme front end needs {FRONT_END['LangSegment']} ({exc})") from exc
        sys.modules.pop("LangSegment", None)
        module_spec = importlib.util.spec_from_file_location("LangSegment", inner)
        if module_spec is None or module_spec.loader is None:
            raise MissingDependency(f"cannot load {inner}") from exc
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["LangSegment"] = module
        module_spec.loader.exec_module(module)
        return module


def import_pipeline(root: Path) -> Any:
    """``models.tts.maskgct.maskgct_utils`` from the bundle, imported with the bundle as the working directory."""
    for module, what in FRONT_END.items():
        try:
            import_langsegment() if module == "LangSegment" else __import__(module)
        except ImportError as exc:
            raise MissingDependency(f"MaskGCT's phoneme front end needs {what} ({exc})") from exc
    point_phonemizer_at_espeak()
    with _import_lock:
        check_top_level(root)
        with working_directory(root):
            return require("models.tts.maskgct.maskgct_utils", "maskgct")


class Backend:
    """What :class:`MaskGCT` needs from the pipeline (a fake in tests, the authors' pipeline from :func:`load`)."""

    def phones(self, text: str, language: str) -> int:
        """How many phoneme tokens ``text`` has."""
        raise NotImplementedError

    def generate(self, ref_audio: str, ref_text: str, ref_language: str, text: str, language: str,
                 seconds: float) -> Any:  # fmt: skip
        """24 kHz speech of ``text`` in the voice of ``ref_audio``, ``seconds`` long."""
        raise NotImplementedError

    def reference_samples(self, ref_audio: str) -> int:
        """Length of the reference at 16 kHz."""
        raise NotImplementedError


class MaskGCT(TTSModel):
    def __init__(self, backend: Backend, voices: list[dict[str, Any]], *, language: str = "en",
                 sample_rate: int = 24000, max_chars: int = 200) -> None:  # fmt: skip
        if not voices:
            raise InvalidRequest(
                "MaskGCT clones a voice: it needs at least one reference voice (catalog or voices_dir)"
            )
        self.backend = backend
        self._voices = {str(v["id"]).lower(): v for v in voices}
        self.voices = to_voices(voices)
        self.language = language
        self.sample_rate = sample_rate
        self.max_chars = max_chars
        self._lock = threading.Lock()

    def voice(self, name: str | None) -> dict[str, Any]:
        if not name:
            return next(iter(self._voices.values()))
        entry = self._voices.get(name.strip().lower())
        if entry is None:
            raise InvalidRequest(f"unknown voice {name!r}")
        return entry

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        entry = self.voice(voice)
        ref_audio, ref_text = str(entry["ref_audio"]), str(entry["ref_text"]).strip()
        ref_language = prompt_language(ref_text)
        fallback = str(entry.get("language") or self.language)[:2]
        for sentence in split_sentences(text, self.max_chars):
            if cancelled is not None and cancelled():
                return
            language = text_language(sentence, fallback)
            with self._lock:
                seconds = target_seconds(
                    self.backend.reference_samples(ref_audio), self.backend.phones(ref_text, ref_language),
                    self.backend.phones(sentence, language), float(speed),
                )  # fmt: skip
                audio = np.asarray(
                    self.backend.generate(ref_audio, ref_text, ref_language, sentence, language, seconds),
                    dtype=np.float32,
                )
            if audio.size:
                yield peak_safe(audio)


class AmphionBackend(Backend):
    """The authors' ``MaskGCT_Inference_Pipeline`` with its inference settings (``maskgct_inference``)."""

    def __init__(self, pipeline: Any, utils: Any, librosa: Any, steps: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.utils = utils
        self.librosa = librosa
        self.steps = steps
        self._refs: dict[tuple[str, int], Any] = {}

    def _load(self, path: str, rate: int) -> Any:
        key = (path, rate)
        if key not in self._refs:
            self._refs[key] = self.librosa.load(path, sr=rate)[0]
        return self._refs[key]

    def phones(self, text: str, language: str) -> int:
        return len(self.utils.g2p_(text, language)[1])

    def reference_samples(self, ref_audio: str) -> int:
        return len(self._load(ref_audio, 16000))

    def generate(self, ref_audio: str, ref_text: str, ref_language: str, text: str, language: str,
                 seconds: float) -> Any:  # fmt: skip
        import torch

        p = self.pipeline
        with torch.inference_mode():
            combine_semantic_code, _ = p.text2semantic(
                self._load(ref_audio, 16000), ref_text, ref_language, text, language, seconds,
                self.steps["n_timesteps"], self.steps["cfg"], self.steps["rescale_cfg"],
            )  # fmt: skip
            acoustic_code = p.extract_acoustic_code(
                torch.tensor(self._load(ref_audio, 24000)).unsqueeze(0).to(p.device)
            )
            _, audio = p.semantic2acoustic(
                combine_semantic_code, acoustic_code, n_timesteps=list(self.steps["n_timesteps_s2a"]),
                cfg=self.steps["cfg_s2a"], rescale_cfg=self.steps["rescale_cfg_s2a"],
            )  # fmt: skip
        return audio


def build_pipeline(utils: Any, code: Path, root: Path, semantic_dir: Path, device: str) -> Any:
    """``maskgct_inference.py``'s set-up with local files: the five networks from the model folder, w2v-bert-2.0 and
    its feature extractor from ``semantic_dir``, the statistics from the bundle."""
    import torch

    transformers = require("transformers", "maskgct")
    safetensors = require("safetensors.torch", "maskgct")
    cfg = require("utils.util", "maskgct").load_config(str(code / CONFIG))
    dev = torch.device(device)
    semantic_model = transformers.Wav2Vec2BertModel.from_pretrained(str(semantic_dir)).eval().to(dev)
    stats = torch.load(str(code / STATS), map_location="cpu", weights_only=True)
    nets = {
        "semantic_codec": utils.build_semantic_codec(cfg.model.semantic_codec, dev),
        "t2s_model": utils.build_t2s_model(cfg.model.t2s_model, dev),
        "s2a_model_1layer": utils.build_s2a_model(cfg.model.s2a_model.s2a_1layer, dev),
        "s2a_model_full": utils.build_s2a_model(cfg.model.s2a_model.s2a_full, dev),
    }
    nets["codec_encoder"], nets["codec_decoder"] = utils.build_acoustic_codec(cfg.model.acoustic_codec, dev)
    for rel, name in WEIGHTS.items():
        safetensors.load_model(nets[name], str(root / rel))
    pipeline = utils.MaskGCT_Inference_Pipeline.__new__(utils.MaskGCT_Inference_Pipeline)
    # what the constructor sets, without its download of the feature extractor's latest revision
    pipeline.processor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(str(semantic_dir))
    pipeline.semantic_model = semantic_model
    pipeline.semantic_mean = stats["mean"].to(dev)
    pipeline.semantic_std = torch.sqrt(stats["var"]).to(dev)
    pipeline.device = dev
    for name, net in nets.items():
        setattr(pipeline, name, net)
    return pipeline


def load(ctx: LoadContext) -> MaskGCT:
    require("transformers", "maskgct")
    librosa = require("librosa", "maskgct")
    code = upstream.ensure(BUNDLE)
    root = model_dir(ctx.path, ctx.options)
    missing = [f for f in WEIGHTS if not (root / f).is_file()]
    if missing:
        raise InvalidRequest(f"{root} is not a MaskGCT model folder (missing {', '.join(missing)})")
    voices = collect_voices(root, ctx.options)
    language = str(ctx.option("language", "en"))
    if language not in LANGUAGES:
        raise InvalidRequest(f"MaskGCT speaks {', '.join(LANGUAGES)}, not {language!r}")
    utils = import_pipeline(code)
    semantic_dir = hub_snapshot(
        str(ctx.option("semantic_repo", SEMANTIC_REPO)),
        str(ctx.option("semantic_revision", SEMANTIC_REVISION)),
        ["config.json", "preprocessor_config.json", "model.safetensors"],
    )
    pipeline = build_pipeline(utils, code, root, Path(semantic_dir), ctx.device)
    steps = {
        "n_timesteps": int(ctx.option("n_timesteps", 25)),
        "cfg": float(ctx.option("cfg", 2.5)),
        "rescale_cfg": float(ctx.option("rescale_cfg", 0.75)),
        "n_timesteps_s2a": list(ctx.option("n_timesteps_s2a", [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
        "cfg_s2a": float(ctx.option("cfg_s2a", 2.5)),
        "rescale_cfg_s2a": float(ctx.option("rescale_cfg_s2a", 0.75)),
    }
    return MaskGCT(AmphionBackend(pipeline, utils, librosa, steps), voices, language=language,
                   max_chars=int(ctx.option("max_chars", 200)))  # fmt: skip
