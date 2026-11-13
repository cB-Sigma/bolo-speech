"""whisper.cpp through ``pywhispercpp`` (MIT).

Optional. Loads a single GGML/GGUF Whisper file (``ggml-*.bin``). Use it
with ``bolo pull hf:ggerganov/whisper.cpp --kind whispercpp`` and an
``include`` for the file you want, or register a local directory.

Untested by the maintainers in this release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model, Segment, STTModel, STTRequest, STTResult


class WhisperCppSTT(STTModel):
    max_chunk_s = 30.0
    supports_translate = True

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from pywhispercpp.model import Model as CppModel

        wanted = spec.get("file")
        files = sorted(path.glob(str(wanted) if wanted else "*.bin"))
        if not files:
            raise FileNotFoundError(f"no ggml model (*.bin) in {path}")
        self.model = CppModel(str(files[0]), n_threads=opts.threads, print_progress=False, print_realtime=False)

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            kwargs: dict[str, Any] = {"translate": r.task == "translate"}
            if r.language:
                kwargs["language"] = r.language.split("-")[0]
            if r.prompt:
                kwargs["initial_prompt"] = r.prompt
            segs = self.model.transcribe(r.samples, **kwargs)
            segments = [Segment(s.t0 / 100.0, s.t1 / 100.0, s.text.strip()) for s in segs if s.text.strip()]
            out.append(STTResult(" ".join(s.text for s in segments), r.language, segments))
        return out


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
