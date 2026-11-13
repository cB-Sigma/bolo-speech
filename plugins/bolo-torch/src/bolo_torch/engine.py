"""The plugin's engine: model kinds, and the device/dtype/thread choices made at load time."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model

from bolo_torch import __version__
from bolo_torch.runtime import InvalidRequest, LoadContext, pick_device, pick_dtype, set_threads

log = logging.getLogger("bolo_torch")

Loader = Callable[[LoadContext], Model]

#: kind -> (module, function); modules are imported only when a model of that kind is loaded
LOADERS: dict[str, tuple[str, str]] = {
    "parler-tts": ("bolo_torch.models.parler", "load"),
    "f5-tts": ("bolo_torch.models.f5", "load"),
    "orpheus-snac": ("bolo_torch.models.orpheus", "load"),
    "sepformer": ("bolo_torch.models.separation", "load_sepformer"),
    "pixit": ("bolo_torch.models.separation", "load_pixit"),
}
#: kinds the catalog lists for this plugin that are not implemented yet
NOT_IMPLEMENTED = {"sravaani": "SraVaani (TorchScript STT)", "indicconformer-official": "IndicConformer official"}


class TorchEngine(Engine):
    id = "torch"
    name = "torch"
    license = "Apache-2.0"
    url = "https://github.com/cb-Sigma/bolo/tree/main/plugins/bolo-torch"

    def __init__(self, loaders: dict[str, Loader | tuple[str, str]] | None = None) -> None:
        self.loaders: dict[str, Loader | tuple[str, str]] = dict(LOADERS if loaders is None else loaders)
        self.kinds = tuple(self.loaders)

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, __version__, None)

    def _loader(self, kind: str) -> Loader:
        entry = self.loaders.get(kind)
        if entry is None:
            if kind in NOT_IMPLEMENTED:
                raise InvalidRequest(f"bolo-torch {__version__} cannot load {NOT_IMPLEMENTED[kind]} models yet")
            raise InvalidRequest(
                f"bolo-torch {__version__} has no model kind {kind!r} (it has {', '.join(self.kinds)})"
            )
        if isinstance(entry, tuple):
            module, func = entry
            return getattr(importlib.import_module(module), func)  # type: ignore[no-any-return]
        return entry

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        loader = self._loader(kind)
        device = pick_device(opts.device, spec)
        dtype = pick_dtype(device, spec)
        threads = set_threads(opts.threads)
        ctx = LoadContext(
            kind, task, path, dict(spec), device=device, dtype=dtype, threads=threads,
            model_id=str(spec.get("model_id", path.name)),
        )  # fmt: skip
        t0 = time.perf_counter()
        model = loader(ctx)
        log.info(
            "loaded %s (%s) on %s, %s, %d threads in %.1f s",
            ctx.model_id, kind, device, str(dtype).replace("torch.", ""), threads, time.perf_counter() - t0,
        )  # fmt: skip
        return model
