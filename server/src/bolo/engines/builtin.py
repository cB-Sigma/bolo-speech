"""Models computed by bolo itself, with no inference library: detectors that need no download, and NumPy networks.

* ``energy_vad``: the adaptive energy detector (:class:`bolo.streaming.vad.EnergyVad`) that is also the fallback
  when no neural detector can be used, as a model that can be chosen by name (``energy-vad``).
* ``silence_turn``: "silence only" end-of-turn detection (``silence-only``): no model decides whether a pause ends
  the turn, the silence does (``turn_detection.silence_duration_ms``). Realtime sessions that ask for
  ``semantic_vad`` with it behave exactly like ``server_vad``; :meth:`bolo.core.Bolo.turn_model` treats it as no
  end-of-turn model.
* ``smart_turn_v2``: Smart Turn v2 computed by bolo's own NumPy code from its weights (:mod:`bolo.engines.smart_turn_v2`).
* ``classifier_head``: an audio classifier trained here on an encoder's embeddings (:mod:`bolo.classifiers`). Its
  head is a few NumPy matrices that :mod:`bolo.classifiers` reads itself; loading it only checks the files.
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


class TrainedClassifier(Model):
    """A trained classifier as a loaded model (``bolo run``, preloading): requests go through
    :func:`bolo.classifiers.classify_trained`, which loads its encoder."""

    task = "audio-classify"

    def __init__(self, path: Path) -> None:
        for name in ("classifier.json", "head.npz"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"classifier_head: {name} is missing from {path}")


class BuiltinEngine(Engine):
    id = "builtin"
    name = "bolo built-in detectors"
    license = "Apache-2.0"
    url = "https://github.com/cB-Sigma/bolo-speech"
    kinds = ("energy_vad", SILENCE_TURN, "smart_turn_v2", "classifier_head")

    def status(self) -> EngineStatus:
        return EngineStatus(self.id, True, __version__, None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind == "energy_vad":
            return EnergyVAD()
        if kind == SILENCE_TURN:
            return SilenceTurn()
        if kind == "smart_turn_v2":
            from bolo.engines.smart_turn_v2 import SmartTurnV2

            return SmartTurnV2(path, spec, opts)
        if kind == "classifier_head":
            return TrainedClassifier(path)
        raise ValueError(f"the builtin engine cannot load kind {kind!r}")
