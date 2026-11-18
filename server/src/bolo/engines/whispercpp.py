"""whisper.cpp through ``pywhispercpp`` (MIT).

Optional. Loads a single GGML/GGUF Whisper file (``ggml-*.bin``). Use it
with ``bolo pull hf:ggerganov/whisper.cpp --kind whispercpp`` and an
``include`` for the file you want, or register a local directory.

It runs on the GPU when the installed pywhispercpp was built with one (its macOS arm64 wheels have Metal; the
Linux and Windows wheels are CPU builds, ``GGML_CUDA=1`` / ``GGML_HIP=1`` / ``GGML_VULKAN=1 pip install
pywhispercpp --no-binary pywhispercpp`` builds one): :func:`gpu_families` reads the build from whisper.cpp's system
information, and the model asks for the GPU (``use_gpu``) only when it was placed on one.
"""

from __future__ import annotations

import functools
import inspect
import re
import threading
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

#: whisper.cpp parameters bolo sets per request; each call passes all of them (see WhisperCppSTT.__init__)
_PER_REQUEST = (
    "language", "initial_prompt", "translate", "temperature", "no_context", "temperature_inc", "logprob_thold",
    "no_speech_thold", "token_timestamps",
)  # fmt: skip


def initial_prompt(prompt: str | None, hotwords: str | None) -> str:
    """The request's prompt with its hotwords after it: whisper.cpp has no hotword parameter, and terms in the
    initial prompt steer Whisper towards their spelling (faster-whisper puts its ``hotwords`` there too)."""
    return " ".join(part.strip() for part in (prompt, hotwords) if part and part.strip())


#: whisper.cpp's system information names a GPU back end it was built with ("MTL : EMBED_LIBRARY = 1", "CUDA : ...")
_BACKENDS = {"metal": ("MTL", "METAL"), "cuda": ("CUDA",), "rocm": ("HIP", "ROCM"), "vulkan": ("VULKAN",)}


def families_of(system_info: str) -> frozenset[str]:
    """The bolo ``device`` families a whisper.cpp build can use, from ``whisper_print_system_info()``."""
    words = {w.upper() for w in re.findall(r"[A-Za-z]+(?=\s*:)", system_info or "")}
    return frozenset(f for f, names in _BACKENDS.items() if words & set(names))


@functools.lru_cache(maxsize=1)
def gpu_families() -> frozenset[str]:
    """:func:`families_of` the installed pywhispercpp (empty when it is not installed or is a CPU build)."""
    try:
        import _pywhispercpp as pw

        return families_of(str(pw.whisper_print_system_info()))
    except Exception:
        return frozenset()


def _on_gpu(device: str | None) -> bool:
    return (device or "cpu").split(":", 1)[0].strip().lower() not in ("", "cpu", "auto")


class WhisperCppSTT(STTModel):
    max_chunk_s = 30.0
    supports_translate = True
    #: from whisper.cpp's token timestamps (``token_timestamps``), which it computes after decoding: the text is the
    #: same with them
    supports_word_timestamps = True
    #: whisper.cpp's own parameters for these (no_context, temperature_inc, logprob_thold, no_speech_thold);
    #: it has no repetition penalty, n-gram ban, hallucination-silence check or chunk length, and its entropy
    #: threshold measures something else than a compression ratio, so those are reported as ignored
    decode_options = (
        "condition_on_previous_text", "temperature_increment_on_fallback", "log_prob_threshold", "no_speech_threshold",
    )  # fmt: skip
    #: hotwords go into the initial prompt (after the request's prompt), as faster-whisper does with them
    request_fields = ("prompt", "hotwords", "temperature")

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from pywhispercpp.model import Model as CppModel

        wanted = spec.get("file")
        files = sorted(path.glob(str(wanted) if wanted else "*.bin"))
        if not files:
            raise FileNotFoundError(f"no ggml model (*.bin) in {path}")
        self.model: Any = self._load(CppModel, files[0], opts, spec)
        # pywhispercpp keeps every parameter given to transcribe() for all later calls: a request's language, prompt
        # or option would carry over to the next request. Each call sets all of these, from the model's own
        # defaults read here unless the request gives one, under a lock (the parameters belong to the model object)
        current = self.model.get_params()
        self._defaults: dict[str, Any] = {k: current[k] for k in _PER_REQUEST if k in current}
        # whisper.cpp's own default language is "en": a multilingual model is told to detect the language instead
        # when a request gives none; an English-only model (*.en) keeps "en" and cannot translate
        self.english_only = bool(spec.get("english_only")) or ".en" in files[0].name
        self._defaults["language"] = "en" if self.english_only else "auto"
        self.supports_translate = not self.english_only
        self.languages = ("en",) if self.english_only else ()
        self._lock = threading.Lock()

    def _load(self, cpp_model: Any, file: Path, opts: LoadOptions, spec: dict[str, Any]) -> Any:
        """The model, with whisper.cpp's DTW token timestamps (word times) switched on unless the catalog entry
        turns them off (``dtw: false``). DTW needs the model's alignment heads (a preset for the standard sizes,
        else the top heads) and cannot run with flash attention, which is then off. A binding without context
        parameters loads the model without DTW; words then come from whisper.cpp's coarser token timestamps."""
        kwargs: dict[str, Any] = {"n_threads": opts.threads, "print_progress": False, "print_realtime": False}
        # the GPU only when the model was placed on one (whisper.cpp's own default is to use it when it can)
        gpu: dict[str, Any] = {"use_gpu": _on_gpu(opts.device)}
        if gpu["use_gpu"] and opts.device.partition(":")[2].isdigit():
            gpu["gpu_device"] = int(opts.device.partition(":")[2])
        #: word times come from DTW (True) or from the plain token timestamps
        self.dtw = False
        if spec.get("dtw", True):
            try:
                import _pywhispercpp as pw

                preset = _aheads_preset(pw, file.name)
                context = {"flash_attn": False, "dtw_token_timestamps": True, "dtw_aheads_preset": preset, **gpu}
                model = cpp_model(str(file), context_params=context, **kwargs)
                self.dtw = True
                return model
            except TypeError:  # pywhispercpp before context_params
                pass
        if "context_params" in inspect.signature(cpp_model.__init__).parameters:
            return cpp_model(str(file), context_params=gpu, **kwargs)
        return cpp_model(str(file), **kwargs)

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            kwargs: dict[str, Any] = {
                **self._defaults,
                "translate": r.task == "translate",
                "temperature": r.temperature,
                "token_timestamps": r.word_timestamps,
            }
            if r.language:
                kwargs["language"] = r.language.split("-")[0]
            prompt = initial_prompt(r.prompt, r.hotwords)
            if prompt:
                kwargs["initial_prompt"] = prompt
            opts = r.options
            if opts.get("condition_on_previous_text") is not None:
                kwargs["no_context"] = not opts["condition_on_previous_text"]
            for name, param in (
                ("temperature_increment_on_fallback", "temperature_inc"),
                ("log_prob_threshold", "logprob_thold"),
                ("no_speech_threshold", "no_speech_thold"),
            ):
                if opts.get(name) is not None:
                    kwargs[param] = float(opts[name])
            if r.abort.is_set():  # the client went away while this waited
                out.append(STTResult("", r.language))
                continue
            with self._lock:
                segs = self.model.transcribe(r.samples, abort_callback=r.abort.is_set, **kwargs)
                language = self._language_used(str(kwargs["language"] or ""))
                per_segment = self._words(len(segs), len(r.samples) / self.sample_rate) if r.word_timestamps else None
            segments = [
                Segment(s.t0 / 100.0, s.t1 / 100.0, s.text.strip(), per_segment[i] if per_segment else [])
                for i, s in enumerate(segs)
                if s.text.strip()
            ]
            words = [w for s in segments for w in s.words]
            out.append(STTResult(" ".join(s.text for s in segments), language, segments, words))
        return out

    def _words(self, n_segments: int, duration: float) -> list[list[Word]]:
        """The words of each segment of the last decode, from its tokens: a token that starts with a space starts a
        word; whisper.cpp's special tokens (timestamps, end of text) are left out.

        With DTW a token's time (``t_dtw``) is where it ends: a word ends at its last token that is not punctuation
        and starts where the word before it ended, but no earlier than twice the segment's median word length before
        its end (the first word after a pause would otherwise start at the pause). Without DTW, whisper.cpp's token
        timestamps (``t0``/``t1``). Centiseconds; kept in order and inside the audio."""
        import _pywhispercpp as pw

        ctx = self.model._ctx
        eot = int(pw.whisper_token_eot(ctx))
        out: list[list[Word]] = []
        last = 0.0
        for i in range(min(n_segments, int(pw.whisper_full_n_segments(ctx)))):
            pieces: list[list[Any]] = []  # one per word: bytes, start, end, DTW end, token probabilities
            for j in range(int(pw.whisper_full_n_tokens(ctx, i))):
                data = pw.whisper_full_get_token_data(ctx, i, j)
                if int(data.id) >= eot:
                    continue
                raw = bytes(pw.whisper_token_to_bytes(ctx, int(data.id)))
                t0, t1 = float(data.t0) / 100.0, float(data.t1) / 100.0
                dtw = float(data.t_dtw) / 100.0 if self.dtw and int(data.t_dtw) >= 0 else None
                if dtw is not None and not raw.strip().decode("utf-8", "replace").strip(_PUNCTUATION):
                    dtw = None  # punctuation: its DTW time is late, at the next word
                p = float(data.p)
                if not pieces or raw.startswith(b" "):
                    pieces.append([raw, t0, t1, dtw, [p]])
                else:
                    word = pieces[-1]
                    word[0] += raw
                    word[2] = t1
                    word[3] = dtw if dtw is not None else word[3]
                    word[4].append(p)
            pieces = [p for p in pieces if p[0].decode("utf-8", errors="replace").strip()]
            ends = [p[3] for p in pieces]
            if pieces and all(e is not None for e in ends):
                lengths = np.diff([last, *[float(e) for e in ends]])
                cap = float(
                    min(1.0, max(0.2, 2.0 * float(np.median(lengths[lengths > 0])) if (lengths > 0).any() else 1.0))
                )
                spans: list[tuple[float, float]] = []
                previous = last
                for e in ends:
                    end = float(e)  # type: ignore[arg-type]
                    spans.append((max(previous, end - cap), end))
                    previous = max(previous, end)
            else:
                spans = [(float(p[1]), float(p[2])) for p in pieces]
            words: list[Word] = []
            for piece, (t0, t1) in zip(pieces, spans):
                start = min(max(t0, last), duration)
                end = min(max(t1, start), duration)
                p = round(float(np.mean(piece[4])), 4)
                words.append(Word(piece[0].decode("utf-8", errors="replace").strip(), start, end, p))
                last = max(last, start) if not self.dtw else max(last, end)
            out.append(words)
        return out

    def _language_used(self, asked: str) -> str | None:
        """The language of the last decode: the one asked for, or the one whisper.cpp detected (``auto``)."""
        if asked and asked != "auto":
            return asked
        try:
            import _pywhispercpp as pw

            return str(pw.whisper_lang_str(pw.whisper_full_lang_id(self.model._ctx))) or None
        except Exception:  # an older binding without these: the language stays unknown
            return None


#: what counts as punctuation in a token (a token of only these is not a word's end)
_PUNCTUATION = "\"'“”¿?¡!.,;:…。，、！？：；()[]{}-«»"

#: alignment-head presets by model size, longest name first (``ggml-large-v3-turbo-q5_0.bin`` is large-v3-turbo)
_PRESETS = (
    ("large-v3-turbo", "LARGE_V3_TURBO"), ("large-v3", "LARGE_V3"), ("large-v2", "LARGE_V2"), ("large-v1", "LARGE_V1"),
    ("medium.en", "MEDIUM_EN"), ("medium", "MEDIUM"), ("small.en", "SMALL_EN"), ("small", "SMALL"),
    ("base.en", "BASE_EN"), ("base", "BASE"), ("tiny.en", "TINY_EN"), ("tiny", "TINY"),
)  # fmt: skip


def _aheads_preset(pw: Any, name: str) -> Any:
    """whisper.cpp's alignment-head preset for a model file: by its size in the name, else the top heads."""
    lowered = name.lower()
    for size, preset in _PRESETS:
        if size in lowered:
            return getattr(pw, f"WHISPER_AHEADS_{preset}")
    return pw.WHISPER_AHEADS_N_TOP_MOST


class WhisperCppEngine(Engine):
    id = "whisper.cpp"
    name = "whisper.cpp (pywhispercpp)"
    license = "MIT"
    url = "https://github.com/ggml-org/whisper.cpp"
    kinds = ("whispercpp",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import pywhispercpp  # noqa: F401

            return EngineStatus(self.id, True, version("pywhispercpp"), None)
        except Exception as exc:
            return EngineStatus(
                self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: pip install 'bolo-speech[whispercpp]'"
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        return WhisperCppSTT(path, spec, opts)
