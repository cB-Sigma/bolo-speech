"""Moonshine through Useful Sensors' own ONNX package (MIT).

sherpa-onnx already runs the Moonshine models in the catalog, so this engine
is optional: it exists for people who have the upstream ``moonshine_onnx``
package and its model layout (``encoder_model.onnx``,
``decoder_model_merged.onnx``) and would rather not install sherpa-onnx.
Use it with ``bolo pull hf:<repo> --kind moonshine-onnx``.

Untested by the maintainers in this release (the sherpa-onnx route is the
validated one).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model, Segment, STTModel, STTRequest, STTResult


class MoonshineSTT(STTModel):
    max_chunk_s = 30.0

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import moonshine_onnx

        self._mo = moonshine_onnx
        name = str(spec.get("model_name", "moonshine/tiny"))
        self.model = moonshine_onnx.MoonshineOnnxModel(models_dir=str(path), model_name=name)
        self.tokenizer = moonshine_onnx.load_tokenizer()

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for r in requests:
            tokens = self.model.generate(r.samples[None, :])
            text = str(self.tokenizer.decode_batch(tokens)[0]).strip()
            duration = len(r.samples) / self.sample_rate
            out.append(STTResult(text, "en", [Segment(0.0, duration, text)] if text else []))
        return out


class MoonshineEngine(Engine):
    id = "moonshine"
    name = "useful-moonshine-onnx"
    license = "MIT"
    url = "https://github.com/moonshine-ai/moonshine"
    kinds = ("moonshine-onnx",)

    def status(self) -> EngineStatus:
        try:
            import moonshine_onnx

            return EngineStatus(self.id, True, str(getattr(moonshine_onnx, "__version__", "unknown")), None)
        except Exception as exc:
            return EngineStatus(
                self.id,
                False,
                None,
                f"{type(exc).__name__}: {exc}. Optional: Moonshine models run through sherpa-onnx; "
                "install with: pip install 'bolo[moonshine]'",
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        return MoonshineSTT(path, spec, opts)
