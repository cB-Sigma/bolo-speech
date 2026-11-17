"""Detectors that are part of bolo itself: nothing to download, nothing to install.

* ``energy_vad``: the adaptive energy detector (:class:`bolo.streaming.vad.EnergyVad`) that is also the fallback
  when no neural detector can be used, as a model that can be chosen by name (``energy-vad``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bolo import __version__
from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model, VadDetector, VADModel, VadParams


class EnergyVAD(VADModel):
    def create_detector(self, params: VadParams | None = None) -> VadDetector:
        from bolo.streaming.vad import EnergyVad

        return EnergyVad(params)


class BuiltinEngine(Engine):
    id = "builtin"
    name = "bolo built-in detectors"
    license = "Apache-2.0"
    url = "https://github.com/cB-Sigma/bolo"
    kinds = ("energy_vad",)

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, __version__, None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind == "energy_vad":
            return EnergyVAD()
        raise ValueError(f"the builtin engine cannot load kind {kind!r}")
