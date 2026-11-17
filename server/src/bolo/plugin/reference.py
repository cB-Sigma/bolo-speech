"""Reference plugin: bolo's deterministic fake engine, served out of process.

``python -m bolo.plugin.reference`` speaks the plugin protocol on stdio. It is what the test-suite
uses, and the smallest example of a plugin. It loads the kinds ``fake-stt``, ``fake-tts``, ``fake-separation``,
``fake-stt-speakers`` (a speaker-conditioned recogniser) and ``fake-diarization``.

For tests of the host's crash handling, text containing "plugin crash test" makes the process exit at once
during synthesis.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, FloatArray, LoadOptions, Model, TTSModel
from bolo.engines.fake import FakeEngine


class _CrashableTTS(TTSModel):
    def __init__(self, inner: TTSModel) -> None:
        self.inner = inner
        self.voices = inner.voices
        self.sample_rate = inner.sample_rate

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        if "plugin crash test" in text.lower():
            os._exit(17)
        yield from self.inner.synthesize(text, voice, speed, cancelled)


class ReferenceEngine(Engine):
    id = "reference"
    name = "reference"
    license = "Apache-2.0"
    kinds = ("fake-stt", "fake-tts", "fake-separation", "fake-stt-speakers", "fake-diarization")

    def __init__(self) -> None:
        self.fake = FakeEngine()

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, "1.0", None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        model = self.fake.load(kind, task, path, spec, opts)
        return _CrashableTTS(model) if isinstance(model, TTSModel) else model


def main() -> int:
    from bolo.plugin.sdk import serve

    return serve(ReferenceEngine())


if __name__ == "__main__":
    raise SystemExit(main())
