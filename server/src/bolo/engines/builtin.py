"""Detectors that are part of bolo itself: nothing to download, nothing to install.

* ``energy_vad``: the adaptive energy detector (:class:`bolo.streaming.vad.EnergyVad`) that is also the fallback
  when no neural detector can be used, as a model that can be chosen by name (``energy-vad``).
* ``silence_turn``: "silence only" end-of-turn detection (``silence-only``): no model decides whether a pause ends
  the turn, the silence does (``turn_detection.silence_duration_ms``). Realtime sessions that ask for
  ``semantic_vad`` with it behave exactly like ``server_vad``; :meth:`bolo.core.Bolo.turn_model` treats it as no
  end-of-turn model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bolo import __version__
from bolo.engines.base import (
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    TurnModel,
    VadDetector,
    VADModel,
    VadParams,
)

#: the ``kind`` of the "silence only" end-of-turn entry
SILENCE_TURN = "silence_turn"


class EnergyVAD(VADModel):
    def create_detector(self, params: VadParams | None = None) -> VadDetector:
        from bolo.streaming.vad import EnergyVad

        return EnergyVad(params)


class SilenceTurn(TurnModel):
    """Every pause that lasts ``silence_duration_ms`` ends the turn."""

    def probability(self, samples: FloatArray) -> float:
        return 1.0


class BuiltinEngine(Engine):
    id = "builtin"
    name = "bolo built-in detectors"
    license = "Apache-2.0"
    url = "https://github.com/cB-Sigma/bolo"
    kinds = ("energy_vad", SILENCE_TURN)

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, __version__, None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind == "energy_vad":
            return EnergyVAD()
        if kind == SILENCE_TURN:
            return SilenceTurn()
        raise ValueError(f"the builtin engine cannot load kind {kind!r}")
