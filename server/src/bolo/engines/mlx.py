"""MLX: speech models on Apple silicon's GPU through Apple's array framework (``bolo-speech[mlx]``).

Optional, in process, macOS arm64 only as an extra (MLX also has Linux CPU wheels, ``mlx[cpu]``, which run these
models on the CPU: that is how they were first run). Four kinds:

* ``mlx-whisper``: Whisper in the MLX format (``weights.npz`` or ``weights.safetensors`` with ``config.json``, the
  ``mlx-community/whisper-*-mlx`` repositories) through ``mlx-whisper`` (Apple's port in ``ml-explore/mlx-examples``):
  transcribe and translate, language detection, word timestamps.
* ``parakeet-mlx``: NVIDIA's Parakeet TDT and CTC models converted to MLX (``mlx-community/parakeet-*``) through
  ``parakeet-mlx``: transcripts with word timestamps from the token alignment.
* ``kokoro`` (``engine: mlx``): Kokoro-82M through ``mlx-audio``'s port of the model. bolo phonemises the text itself
  (uccharan's Kokoro front end, as for the sherpa-onnx Kokoro voices) and gives the model the phonemes, so mlx-audio's
  own front end (misaki, which needs spaCy and espeak-ng) is not needed. Voice blends work as for sherpa-onnx.
* ``mlx-audio-stt``: speech-to-text models of ``mlx-audio`` (``mlx_audio.stt.load``; mlx-audio's own port of the model,
  no code from the model repository) that take an instruction, such as SOV-Transcribe-1. The instruction comes from
  the catalog entry (``options.prompts``: ``transcribe``, ``transcribe_language`` with ``{language_name}``, and
  ``hotwords`` with ``{hotwords}`` put in front); the model gives no timestamps, so segments are the pieces bolo cut
  the audio into (``max_chunk_s``).

Device: a model placed on the GPU (``device`` ``coreml``/``mps``/``metal``/``mlx``, or ``auto`` on a Mac) runs on
MLX's GPU stream (Metal); placed on the CPU it runs on MLX's CPU stream. MLX's streams belong to the thread that made
them ("There is no Stream(cpu, 0) in current thread" from any other), so every call into MLX, loads included, runs on
one thread of this module's (:func:`_run`), which also serialises them; each sets the device its model was placed on.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import (
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    Segment,
    STTModel,
    STTRequest,
    STTResult,
    TTSModel,
    Voice,
    Word,
)
from bolo.engines.blend import looks_like_blend, resolve_blend
from bolo.errors import EngineUnavailable

INSTALL = "pip install 'bolo-speech[mlx]' (macOS on Apple silicon)"
#: the package each kind needs (module, distribution)
PACKAGES = {
    "mlx-whisper": ("mlx_whisper", "mlx-whisper"),
    "parakeet-mlx": ("parakeet_mlx", "parakeet-mlx"),
    "kokoro": ("mlx_audio", "mlx-audio"),
    "mlx-audio-stt": ("mlx_audio", "mlx-audio"),
}
#: the one thread every call into MLX runs on (MLX's streams are per thread)
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bolo-mlx")


def _require(kind: str) -> Any:
    import importlib

    module, dist = PACKAGES[kind]
    try:
        return importlib.import_module(module)
    except Exception as exc:
        raise EngineUnavailable(
            f"MLX models of kind {kind!r} need {dist} ({type(exc).__name__}: {exc}): {INSTALL}"
        ) from exc


def on_gpu(device: str | None) -> bool:
    """Whether a model placed on ``device`` runs on MLX's GPU (Metal) stream."""
    from bolo.placement import family

    return family(device) == "metal"


def metal_available() -> bool:
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


def _run(gpu: bool, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)`` on the MLX thread with the default device set to the GPU or the CPU; waits for it."""

    def call() -> Any:
        import mlx.core as mx

        mx.set_default_device(mx.gpu if gpu else mx.cpu)
        return fn(*args, **kwargs)

    return _WORKER.submit(call).result()


def _placed(opts: LoadOptions) -> bool:
    """The model runs on the GPU: placed there and MLX reaches Metal (else the CPU, which MLX always has). Placed on
    the GPU without Metal (MLX's Linux CPU wheels, a virtual Mac without it), the load says so and the model is moved
    to the CPU (bolo.cuda_libs)."""
    if not on_gpu(opts.device):
        return False
    if metal_available():
        return True
    from bolo import cuda_libs

    cuda_libs.record(
        cuda_libs.ProviderFallback("mlx", "Metal", "cpu", [], "MLX here has no Metal device (mx.metal.is_available())")
    )
    return False


# -- Whisper -----------------------------------------------------------------------------------------------------
class MlxWhisperSTT(STTModel):
    max_chunk_s = 30.0
    supports_translate = True
    supports_word_timestamps = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        mw = _require("mlx-whisper")
        from mlx_whisper.load_models import load_model

        self._mw = mw
        self.path = str(path)
        self.gpu = _placed(opts)
        self.device = "metal" if self.gpu else "cpu"
        import mlx.core as mx

        # float16 on the GPU (mlx-whisper's default), float32 on the CPU, where float16 is slow
        self._dtype = mx.float16 if self.gpu else mx.float32
        self.model = _run(self.gpu, load_model, self.path, dtype=self._dtype)
        self.english_only = bool(spec.get("english_only")) or ".en" in Path(self.path).name
        self.languages = ("en",) if self.english_only else ()
        self.supports_translate = not self.english_only

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        from mlx_whisper.transcribe import ModelHolder

        out: list[STTResult] = []
        for r in requests:
            if r.abort.is_set():
                out.append(STTResult("", r.language))
                continue
            kwargs: dict[str, Any] = {
                "path_or_hf_repo": self.path,
                "task": "translate" if r.task == "translate" and not self.english_only else "transcribe",
                "temperature": r.temperature or 0.0,
                "word_timestamps": r.word_timestamps,
                "verbose": None,
                "fp16": self.gpu,
            }
            language = (r.language or "").split("-")[0] or ("en" if self.english_only else None)
            if language:
                kwargs["language"] = language
            if r.prompt:
                kwargs["initial_prompt"] = r.prompt
            opts = r.options
            if opts.get("condition_on_previous_text") is not None:
                kwargs["condition_on_previous_text"] = bool(opts["condition_on_previous_text"])
            for name in ("no_speech_threshold", "compression_ratio_threshold", "hallucination_silence_threshold"):
                if opts.get(name) is not None:
                    kwargs[name] = float(opts[name])
            if opts.get("log_prob_threshold") is not None:
                kwargs["logprob_threshold"] = float(opts["log_prob_threshold"])

            def run(samples: FloatArray, kwargs: dict[str, Any] = kwargs) -> Any:
                # mlx-whisper keeps the last model it loaded in ModelHolder and loads again when the path differs:
                # give it this one (on the MLX thread, so another Whisper model cannot swap it in between)
                ModelHolder.model, ModelHolder.model_path = self.model, self.path
                return self._mw.transcribe(samples, **kwargs)

            result = _run(self.gpu, run, np.asarray(r.samples, dtype=np.float32))
            segments: list[Segment] = []
            words: list[Word] = []
            for s in result.get("segments", []):
                text = str(s.get("text", "")).strip()
                if not text:
                    continue
                seg_words = [
                    Word(str(w["word"]).strip(), float(w["start"]), float(w["end"]))
                    for w in s.get("words", []) or []
                    if str(w.get("word", "")).strip()
                ]
                words.extend(seg_words)
                segments.append(
                    Segment(
                        float(s["start"]),
                        float(s["end"]),
                        text,
                        seg_words,
                        avg_logprob=s.get("avg_logprob"),
                        no_speech_prob=s.get("no_speech_prob"),
                    )
                )
            out.append(
                STTResult(str(result.get("text", "")).strip(), result.get("language") or language, segments, words)
            )
        return out

    def close(self) -> None:
        from mlx_whisper.transcribe import ModelHolder

        def forget() -> None:
            if ModelHolder.model is self.model:
                ModelHolder.model, ModelHolder.model_path = None, None

        _WORKER.submit(forget).result()
        self.model = None


# -- Parakeet ----------------------------------------------------------------------------------------------------
class ParakeetMlxSTT(STTModel):
    # the encoder attends over the whole input: long audio is cut by bolo (with overlap handled by its chunker)
    max_chunk_s = 60.0
    supports_word_timestamps = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        pm = _require("parakeet-mlx")
        import mlx.core as mx

        self.gpu = _placed(opts)
        self.device = "metal" if self.gpu else "cpu"
        # bfloat16 on the GPU (parakeet-mlx's default); float32 on the CPU
        self._dtype = mx.bfloat16 if self.gpu else mx.float32
        self.model = _run(self.gpu, pm.from_pretrained, str(path), dtype=self._dtype)
        self.languages = tuple(spec.get("languages") or ())

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            samples = np.asarray(r.samples, dtype=np.float32)
            if r.abort.is_set() or len(samples) < self.model.preprocessor_config.hop_length:
                out.append(STTResult("", r.language))
                continue

            def run(samples: FloatArray) -> Any:
                return self.model.generate(logmel(samples, self.model.preprocessor_config))[0]

            aligned = _run(self.gpu, run, samples)
            words = _words(aligned.tokens)
            segments = [
                Segment(float(s.start), float(s.end), s.text.strip(), _words(s.tokens))
                for s in aligned.sentences
                if s.text.strip()
            ]
            out.append(STTResult(aligned.text.strip(), r.language, segments, words))
        return out


def logmel(samples: FloatArray, config: Any) -> Any:
    """parakeet-mlx's log-mel features of float32 samples, as its own ``transcribe`` makes them: always from float32
    audio, whatever the model's dtype. ``get_logmel`` reads its STFT through ``mx.view(x, input dtype)`` and pairs
    the values up as real and imaginary parts, which holds only for float32 input: bfloat16 samples (the model's dtype
    on the GPU) gave twice the frequency bins and a matmul shape error against the mel filter bank."""
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    return get_logmel(mx.array(np.asarray(samples, dtype=np.float32)), config)


def _words(tokens: list[Any]) -> list[Word]:
    """SentencePiece tokens (a word starts with a space) joined into words with their times."""
    words: list[Word] = []
    for t in tokens:
        text = str(t.text)
        if words and text and not text.startswith(" "):
            last = words[-1]
            words[-1] = Word(last.w + text, last.t0, float(t.end))
        elif text.strip():
            words.append(Word(text.strip(), float(t.start), float(t.end)))
    return words


# -- instruction-following recognisers of mlx-audio ---------------------------------------------------------------
def instruction(prompts: dict[str, str], language: str | None, hotwords: str | None) -> str:
    """The instruction of one request from a catalog entry's ``prompts`` (see the module docstring)."""
    from bolo.text.iso639 import name as language_name

    code = (language or "").strip().lower().split("-", 1)[0]
    text = (prompts.get("transcribe_language") if code else None) or prompts["transcribe"]
    text = text.format(language_name=language_name(code) if code else "")
    terms = ", ".join(t.strip() for t in (hotwords or "").replace("\n", ",").split(",") if t.strip())
    if terms and prompts.get("hotwords"):
        text = prompts["hotwords"].format(hotwords=terms) + text
    return text


class MlxAudioSTT(STTModel):
    supports_translate = False
    supports_word_timestamps = False
    request_fields = ("hotwords", "temperature")

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        _require("mlx-audio-stt")
        from mlx_audio.stt import load

        self.gpu = _placed(opts)
        self.device = "metal" if self.gpu else "cpu"
        self.prompts = {str(k): str(v) for k, v in (spec.get("prompts") or {}).items()}
        if "transcribe" not in self.prompts:
            raise EngineUnavailable(f"model '{spec.get('model_id')}': the catalog entry's prompts name no 'transcribe'")
        self.languages = tuple(spec.get("languages") or ())
        self.max_chunk_s = float(spec.get("max_chunk_s") or 30.0)
        self.tokens_per_s = float(spec.get("tokens_per_s") or 12.0)
        self.max_new_tokens = int(spec.get("max_new_tokens") or 8192)
        self.model = _run(self.gpu, load, Path(path))

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            samples = np.asarray(r.samples, dtype=np.float32)
            if r.abort.is_set() or len(samples) < 800:  # 50 ms
                out.append(STTResult("", r.language))
                continue
            seconds = len(samples) / 16000
            kwargs: dict[str, Any] = {
                "prompt": instruction(self.prompts, r.language, r.hotwords),
                "max_tokens": max(16, min(self.max_new_tokens, int(seconds * self.tokens_per_s) + 32)),
                "temperature": float(r.temperature or 0.0),
            }
            result = _run(self.gpu, self.model.generate, samples, **kwargs)
            out.append(STTResult(str(getattr(result, "text", "") or "").strip(), r.language))
        return out

    def close(self) -> None:
        self.model = None


# -- Kokoro ------------------------------------------------------------------------------------------------------
#: Kokoro's phoneme input holds at most 510 symbols (512 positions with the two pads)
_KOKORO_MAX = 510


class MlxKokoroTTS(TTSModel):
    accepts_phonemizer = True
    supports_blending = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        _require("kokoro")
        from mlx_audio.tts.utils import load_model

        self.spec = spec
        self.path = path
        self.gpu = _placed(opts)
        self.device = "metal" if self.gpu else "cpu"
        self.model = _run(self.gpu, load_model, path, strict=False)
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000) or 24000)
        folder = path / "voices"
        on_disk = {p.stem for p in folder.glob("*.safetensors")}
        declared = [
            Voice(str(v["id"]), str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i)
            for i, v in enumerate(spec.get("voices", []))
            if str(v["id"]) in on_disk
        ]
        self.voices = tuple(declared) or tuple(Voice(name, name) for name in sorted(on_disk))
        if not self.voices:
            raise FileNotFoundError(f"no Kokoro voices (voices/*.safetensors) in {path}")
        self._packs: dict[str, Any] = {}

    def _pack(self, voice_id: str) -> Any:
        import mlx.core as mx

        if voice_id not in self._packs:
            weights: Any = mx.load(str(self.path / "voices" / f"{voice_id}.safetensors"))
            self._packs[voice_id] = weights["voice"]
        return self._packs[voice_id]

    def phoneme_target(self, voice: Voice | None) -> tuple[str, str]:
        from bolo.engines.sherpa import _KOKORO_ESPEAK

        prefix = voice.id[:1] if voice is not None else "a"
        return "kokoro", _KOKORO_ESPEAK.get(prefix, "en-us")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        phonemizer: str | None = None,
    ) -> Iterator[FloatArray]:
        from bolo.text import phonemize

        v = self.find_voice(voice)
        blend = None
        if v is None and voice and looks_like_blend(voice):
            blend = resolve_blend(voice, self.voices, self.path.name)
            v = blend.lead
        if v is None:
            from bolo.engines.blend import UnknownVoice

            raise UnknownVoice(f"unknown voice {voice!r} for {self.path.name}")
        profile, language = self.phoneme_target(v)
        ph, improved = phonemize.resolve(profile, language, phonemizer)
        sentences = ["".join(map(chr, cps)) for cps in ph.voice_codepoints(text, language, profile, improved)]
        pieces = [p for s in sentences for p in _fit(s.strip(), _KOKORO_MAX) if p]
        if not pieces:
            raise RuntimeError(f"the phonemiser ({ph.name}) produced nothing for {text[:60]!r} ({language})")
        lead = v

        def mix() -> Any:
            return sum(w * self._pack(x.id) for x, w in blend.parts) if blend is not None else self._pack(lead.id)

        def speak(phones: str, pack: Any) -> FloatArray:
            audio = self.model(phones, pack[len(phones) - 1], speed)
            return np.asarray(audio, dtype=np.float32).reshape(-1)

        pack = _run(self.gpu, mix)
        for phones in pieces:
            if cancelled is not None and cancelled():
                return
            samples = _run(self.gpu, speak, phones, pack)
            if len(samples):
                yield samples

    def warm_up(self) -> None:
        with contextlib.suppress(Exception):
            for _ in self.synthesize("Hello.", None, 1.0, None):
                pass


def _fit(phones: str, limit: int) -> list[str]:
    """Phonemes cut at spaces into pieces of at most ``limit`` symbols (a word longer than that is cut anyway)."""
    out: list[str] = []
    while len(phones) > limit:
        cut = phones.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        out.append(phones[:cut].strip())
        phones = phones[cut:].strip()
    out.append(phones)
    return out


# -- the engine --------------------------------------------------------------------------------------------------
class MlxEngine(Engine):
    id = "mlx"
    name = "MLX (Apple silicon)"
    license = "MIT (mlx, mlx-whisper, mlx-audio); Apache-2.0 (parakeet-mlx)"
    url = "https://github.com/ml-explore/mlx"
    kinds = ("mlx-whisper", "parakeet-mlx", "kokoro", "mlx-audio-stt")

    def status(self) -> EngineStatus:
        from importlib.metadata import version

        try:
            import mlx.core  # noqa: F401

            mlx_version = version("mlx")
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: {INSTALL}")
        found = []
        for module, dist in PACKAGES.values():
            with contextlib.suppress(Exception):
                __import__(module)
                found.append(f"{dist} {version(dist)}")
        where = "Metal" if metal_available() else "the CPU only"
        detail = f"mlx {mlx_version} ({where}{'; ' + ', '.join(found) if found else ''})"
        if not found:
            return EngineStatus(self.id, False, detail, f"MLX is installed but none of its speech packages: {INSTALL}")
        return EngineStatus(self.id, True, detail, None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind == "mlx-whisper":
            return MlxWhisperSTT(path, spec, opts)
        if kind == "parakeet-mlx":
            return ParakeetMlxSTT(path, spec, opts)
        if kind == "kokoro":
            return MlxKokoroTTS(path, spec, opts)
        if kind == "mlx-audio-stt":
            return MlxAudioSTT(path, spec, opts)
        raise ValueError(f"the MLX engine cannot load models of kind {kind!r}")

    def detect(self, path: Path) -> dict[str, Any] | None:
        if (path / "voices").is_dir() and any(path.glob("kokoro*.safetensors")):
            return {"kind": "kokoro", "task": "tts", "options": {}}
        if (path / "weights.npz").is_file() and (path / "config.json").is_file():
            return {"kind": "mlx-whisper", "task": "stt", "options": {}}
        return None
