"""Reference plugin: bolo's deterministic fake engine, served out of process.

``python -m bolo.plugin.reference`` speaks the plugin protocol on stdio. It is what the test-suite
uses, and the smallest example of a plugin. It loads the kinds ``fake-stt``, ``fake-tts``, ``fake-separation``,
``fake-stt-speakers`` (a speaker-conditioned recogniser), ``fake-diarization``, ``fake-denoise``, ``fake-s2s`` (a
turn-based speech-to-speech model) and ``fake-lid`` (a language identifier).

For tests of the host's crash handling, text containing "plugin crash test" makes the process exit at once
during synthesis.
"""

from __future__ import annotations

import os
import sys
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
    kinds = (
        "fake-stt", "fake-tts", "fake-separation", "fake-stt-speakers", "fake-diarization", "fake-denoise", "fake-s2s",
        "fake-lid",
    )  # fmt: skip

    def __init__(self) -> None:
        self.fake = FakeEngine()

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, "1.0", None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        # what model libraries do on first use, for the tests: write to the process's stdout at the descriptor
        # level, and ask a question on stdin (both must stay away from the protocol stream)
        if spec.get("native_print_on_load"):
            os.write(1, b"a native library writing to stdout\n")
        if spec.get("ask_on_load"):
            input("Download the model now? [y/N] ")
        if spec.get("require_utf8_mode") and not sys.flags.utf8_mode:
            raise RuntimeError("not in Python UTF-8 mode: open() without an encoding would use the locale's")
        model = self.fake.load(kind, task, path, spec, opts)
        out = _CrashableTTS(model) if isinstance(model, TTSModel) else model
        if spec.get("cpu_only_runtime") and opts.device != "cpu":  # a runtime that cannot reach the GPU asked for
            out.device_fallback = f"device {opts.device!r}: this runtime has no GPU support, so it runs on the CPU"
        return out


def main() -> int:
    from bolo.plugin.sdk import serve

    return serve(ReferenceEngine())


if __name__ == "__main__":
    raise SystemExit(main())
