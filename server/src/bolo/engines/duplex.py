"""The ``duplex`` engine: the catalog's full-duplex speech-to-speech models (Moshi, Hibiki, PersonaPlex).

These models are not loaded by the model manager: each runs in a server of its own (Kyutai's or NVIDIA's, which
bolo installs as the components ``moshi`` and ``personaplex`` and starts on a loopback port), and Realtime and
WebRTC sessions talk to it through :mod:`bolo.duplex` when ``s2s_model`` names one. This engine only says whether
bolo can talk to such a server (it needs PyAV for the Opus audio) and explains how the models are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model
from bolo.errors import EngineUnavailable


class DuplexEngine(Engine):
    id = "duplex"
    name = "Full-duplex model servers (Kyutai protocol)"
    license = "MIT (Kyutai's moshi, NVIDIA's personaplex); Apache-2.0 or MIT (moshi-backend)"
    url = "https://github.com/kyutai-labs/moshi"
    kinds = ("kyutai",)

    def status(self) -> EngineStatus:
        try:
            import av
        except ImportError:
            from bolo.duplex.ogg_opus import INSTALL_HINT

            return EngineStatus(self.id, False, None, f"PyAV is not installed: {INSTALL_HINT}")
        return EngineStatus(self.id, True, f"PyAV {av.__version__}", None)

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        name = spec.get("model_id") or path.name
        raise EngineUnavailable(
            f"'{name}' is a full-duplex speech-to-speech model: it runs in its own server, used by Realtime and WebRTC "
            f'sessions of the speech-to-speech pipeline (set s2s_model = "{name}" and realtime_pipeline = '
            '"speech-to-speech"); docs/voice-agents.md, "Full-duplex models"'
        )
