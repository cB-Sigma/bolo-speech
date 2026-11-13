"""Engine plugins and their registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from bolo.engines.base import Engine, EngineStatus

# id -> "module:Class"; nothing here is imported until the engine is asked for.
BUILTIN_ENGINES: dict[str, str] = {
    "sherpa-onnx": "bolo.engines.sherpa:SherpaOnnxEngine",
    "faster-whisper": "bolo.engines.faster_whisper:FasterWhisperEngine",
    "moonshine": "bolo.engines.moonshine:MoonshineEngine",
    "kokoro-onnx": "bolo.engines.kokoro_onnx:KokoroOnnxEngine",
    "whisper.cpp": "bolo.engines.whispercpp:WhisperCppEngine",
    "onnxruntime": "bolo.engines.onnx_turn:OnnxRuntimeEngine",
    "fake": "bolo.engines.fake:FakeEngine",
}


class EngineRegistry:
    def __init__(self, include_fake: bool = False) -> None:
        self._factories: dict[str, Callable[[], Engine]] = {}
        self._engines: dict[str, Engine] = {}
        for engine_id, target in BUILTIN_ENGINES.items():
            if engine_id == "fake" and not include_fake:
                continue
            self._factories[engine_id] = self._lazy(target)

    @staticmethod
    def _lazy(target: str) -> Callable[[], Engine]:
        def factory() -> Engine:
            module_name, _, cls_name = target.partition(":")
            cls = getattr(importlib.import_module(module_name), cls_name)
            engine: Engine = cls()
            return engine

        return factory

    def register(self, engine: Engine) -> None:
        self._engines[engine.id] = engine
        self._factories[engine.id] = lambda: engine

    def ids(self) -> list[str]:
        return list(self._factories)

    def get(self, engine_id: str) -> Engine | None:
        if engine_id in self._engines:
            return self._engines[engine_id]
        factory = self._factories.get(engine_id)
        if factory is None:
            return None
        engine = factory()
        self._engines[engine_id] = engine
        return engine

    def statuses(self) -> list[EngineStatus]:
        out: list[EngineStatus] = []
        for engine_id in self._factories:
            try:
                engine = self.get(engine_id)
                assert engine is not None
                out.append(engine.status())
            except Exception as exc:
                out.append(EngineStatus(engine_id, False, None, f"{type(exc).__name__}: {exc}"))
        return out

    def status(self, engine_id: str) -> EngineStatus:
        try:
            engine = self.get(engine_id)
        except Exception as exc:
            return EngineStatus(engine_id, False, None, f"{type(exc).__name__}: {exc}")
        if engine is None:
            return EngineStatus(engine_id, False, None, "unknown engine")
        return engine.status()


__all__ = ["BUILTIN_ENGINES", "Engine", "EngineRegistry", "EngineStatus"]
