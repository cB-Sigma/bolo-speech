"""whisper.cpp through ``pywhispercpp`` (MIT).

Optional. Loads a single GGML/GGUF Whisper file (``ggml-*.bin``). Use it
with ``bolo pull hf:ggerganov/whisper.cpp --kind whispercpp`` and an
``include`` for the file you want, or register a local directory.

Untested by the maintainers in this release.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model, Segment, STTModel, STTRequest, STTResult

#: whisper.cpp parameters bolo sets per request; each call passes all of them (see WhisperCppSTT.__init__)
_PER_REQUEST = (
    "language", "initial_prompt", "translate", "temperature", "no_context", "temperature_inc", "logprob_thold",
    "no_speech_thold",
)  # fmt: skip


class WhisperCppSTT(STTModel):
    max_chunk_s = 30.0
    supports_translate = True
    #: whisper.cpp's own parameters for these (no_context, temperature_inc, logprob_thold, no_speech_thold);
    #: it has no repetition penalty, n-gram ban, hallucination-silence check or chunk length, and its entropy
    #: threshold measures something else than a compression ratio, so those are reported as ignored
    decode_options = (
        "condition_on_previous_text", "temperature_increment_on_fallback", "log_prob_threshold", "no_speech_threshold",
    )  # fmt: skip

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from pywhispercpp.model import Model as CppModel

        wanted = spec.get("file")
        files = sorted(path.glob(str(wanted) if wanted else "*.bin"))
        if not files:
            raise FileNotFoundError(f"no ggml model (*.bin) in {path}")
        self.model: Any = CppModel(str(files[0]), n_threads=opts.threads, print_progress=False, print_realtime=False)
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

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            kwargs: dict[str, Any] = {
                **self._defaults,
                "translate": r.task == "translate",
                "temperature": r.temperature,
            }
            if r.language:
                kwargs["language"] = r.language.split("-")[0]
            if r.prompt:
                kwargs["initial_prompt"] = r.prompt
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
            segments = [Segment(s.t0 / 100.0, s.t1 / 100.0, s.text.strip()) for s in segs if s.text.strip()]
            out.append(STTResult(" ".join(s.text for s in segments), language, segments))
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
