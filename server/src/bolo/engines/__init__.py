"""Engine plugins and their registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from bolo.engines.base import Engine, EngineStatus

if TYPE_CHECKING:
    from bolo.config import Settings

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
    def __init__(self, include_fake: bool = False, settings: Settings | None = None) -> None:
        self._factories: dict[str, Callable[[], Engine]] = {}
        self._engines: dict[str, Engine] = {}
        for engine_id, target in BUILTIN_ENGINES.items():
            if engine_id == "fake" and not include_fake:
                continue
            self._factories[engine_id] = self._lazy(target)
        self._add_plugins(settings)

    def _add_plugins(self, settings: Settings | None) -> None:
        """Out-of-process plugins: installed packages (entry points), then the ``plugins`` setting (wins)."""
        from bolo.plugin.host import PluginConfig, PluginEngine, discover_entry_points, parse_plugin_setting

        found: dict[str, tuple[list[str], str]] = {}
        if settings is None or settings.plugin_entry_points:
            found.update({name: (cmd, "entry-point") for name, cmd in discover_entry_points().items()})
        for item in settings.plugins if settings is not None else []:
            name, cmd = parse_plugin_setting(item)
            found[name] = (cmd, "settings")
        for name, (cmd, source) in found.items():
            config = PluginConfig(name, cmd, source)
            if settings is not None:
                config.idle_s = settings.plugin_idle_s
                config.memory_limit_mb = settings.plugin_memory_limit_mb
                config.nice = settings.plugin_nice
            self.register(PluginEngine(config))

    def plugins(self) -> list[Engine]:
        from bolo.plugin.host import PREFIX

        return [e for i, e in self._engines.items() if i.startswith(PREFIX)]

    def shutdown(self) -> None:
        """Stop plugin processes."""
        for engine in self.plugins():
            stop = getattr(engine, "shutdown", None)
            if stop is not None:
                stop()

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
            if engine_id.startswith("plugin:"):
                return EngineStatus(engine_id, False, None, f"the plugin '{engine_id[7:]}' is not installed")
            return EngineStatus(engine_id, False, None, "unknown engine")
        return engine.status()


__all__ = ["BUILTIN_ENGINES", "Engine", "EngineRegistry", "EngineStatus"]
