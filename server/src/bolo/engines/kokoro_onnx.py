"""Kokoro through the ``kokoro-onnx`` package.

Optional and off by default. ``kokoro-onnx`` is MIT, but it depends on
``phonemizer`` (GPL-3.0-or-later) and loads espeak-ng (GPL-3.0-or-later)
in process. It is therefore not part of ``bolo-speech[all]``; install it
yourself with ``pip install 'bolo-speech[kokoro-onnx]'`` if those terms suit
you. The catalog's Kokoro entries use sherpa-onnx instead.

Expects a directory with ``kokoro-v1.0*.onnx`` and ``voices-v1.0.bin``
(the files published with kokoro-onnx releases). Use it with
``bolo pull hf:<repo> --kind kokoro-onnx``.

Untested by the maintainers in this release.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import Engine, EngineStatus, FloatArray, LoadOptions, Model, TTSModel, Voice

_LANG = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br", "j": "ja", "z": "cmn"}


class KokoroOnnxTTS(TTSModel):
    sample_rate = 24000

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        from kokoro_onnx import Kokoro

        model = next(iter(sorted(path.glob("*.onnx"))), None)
        voices = next(iter(sorted(path.glob("voices*.bin"))), None)
        if model is None or voices is None:
            raise FileNotFoundError(f"expected a Kokoro .onnx model and a voices .bin file in {path}")
        self.kokoro = Kokoro(str(model), str(voices))
        names = sorted(self.kokoro.get_voices())
        self.voices = tuple(
            Voice(n, n.split("_", 1)[-1].title(), _LANG.get(n[:1], ""), n[1:2], i) for i, n in enumerate(names)
        )

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FloatArray]:
        v = self.find_voice(voice)
        name = v.id if v else "af_heart"
        samples, rate = self.kokoro.create(text, voice=name, speed=speed, lang=_LANG.get(name[:1], "en-us"))
        self.sample_rate = int(rate)
        yield np.asarray(samples, dtype=np.float32)


class KokoroOnnxEngine(Engine):
    id = "kokoro-onnx"
    name = "kokoro-onnx"
    license = "MIT (depends on phonemizer, GPL-3.0-or-later)"
    url = "https://github.com/thewh1teagle/kokoro-onnx"
    kinds = ("kokoro-onnx",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import kokoro_onnx  # noqa: F401

            return EngineStatus(self.id, True, version("kokoro-onnx"), None)
        except Exception as exc:
            return EngineStatus(
                self.id,
                False,
                None,
                f"{type(exc).__name__}: {exc}. Optional and not part of [all] because it pulls in GPL "
                "dependencies; Kokoro runs through sherpa-onnx by default.",
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        return KokoroOnnxTTS(path, spec, opts)
