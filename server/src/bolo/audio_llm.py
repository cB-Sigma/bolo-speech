"""The ``audio-llm`` pipeline's model on a llama-server that bolo starts (catalog task ``audio-llm``).

Without ``audio_llm_base_url``, ``audio_llm_model`` may name a catalog entry with ``task: audio-llm``: a GGUF
language model and its audio encoder (mmproj) that llama.cpp's server runs with ``input_audio`` content in
``/v1/chat/completions`` (libmtmd: Ultravox, Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, LFM2/LFM2.5-Audio, Voxtral,
Gemma 4 and others). bolo then pulls the model, starts a llama-server for it on a loopback port with a random API
key (the binary of the ``tts-server`` engine, placed on the GPU or the CPU like the speech-to-speech server) the
first time a turn needs it, restarts it after a crash with a back-off, and stops it with the server. The model
hears the user and writes the answer; bolo's TTS speaks it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from bolo.errors import EngineUnavailable

if TYPE_CHECKING:
    from bolo.catalog import ModelSpec
    from bolo.core import Bolo
    from bolo.engines.tts_server import ServerProcess

log = logging.getLogger("bolo.audio_llm")

#: catalog task of models that hear audio and answer in text (the ``audio-llm`` pipeline)
TASK = "audio-llm"


class ManagedAudioLLM:
    """A llama-server bolo runs for the catalog model ``audio_llm_model`` (when ``audio_llm_base_url`` is unset)."""

    def __init__(self, core: Bolo) -> None:
        self.core = core
        self.process: ServerProcess | None = None
        self.placement: Any = None
        self._lock = asyncio.Lock()

    def spec(self) -> ModelSpec | None:
        """The catalog entry the settings name, when bolo should run it (no ``audio_llm_base_url``)."""
        s = self.core.settings
        if s.audio_llm_base_url or not s.audio_llm_model:
            return None
        spec = self.core.catalog.resolve(s.audio_llm_model)
        return spec if spec is not None and spec.task == TASK else None

    def readiness(self) -> tuple[bool, str | None]:
        """(available, reason): can a session use the managed model now?"""
        from bolo.tts_runtimes import LLAMA, find_binary

        spec = self.spec()
        if spec is None:
            return False, "audio_llm_model does not name an audio-llm model of the catalog"
        if not spec.runnable:
            return False, f"model '{spec.id}' cannot run yet: {spec.not_runnable_reason}"
        if spec.status == "pending":
            return False, f"model '{spec.id}' is not published yet"
        if find_binary(self.core.settings, LLAMA)[0] is None:
            return False, "llama-server is not installed (bolo components install llama-server)"
        s = self.core.settings
        if not (self.core.store.is_installed(spec.id) or (s.auto_pull and not spec.gated)):
            return False, f"model '{spec.id}' is not installed"
        return True, None

    async def base_url(self) -> tuple[str, str | None]:
        """The managed server's ``/v1`` URL and API key, starting it (and pulling the model) first if needed."""
        from bolo.s2s import managed_llama_server

        async with self._lock:
            spec = self.spec()
            if spec is None:
                raise EngineUnavailable("audio_llm_model does not name an audio-llm model of the catalog")
            if not spec.runnable:
                raise EngineUnavailable(f"model '{spec.id}' cannot run yet: {spec.not_runnable_reason}")
            spec.require_published()
            if self.process is None or self.process.name != f"audio-llm:{spec.id}":
                if self.process is not None:
                    await asyncio.to_thread(self.process.stop)
                    self.process = None
                if not self.core.store.is_installed(spec.id):
                    await self.core._auto_pull(spec)
                s = self.core.settings
                self.process, self.placement = managed_llama_server(
                    self.core, f"audio-llm:{spec.id}", spec.id, self.core.store.model_dir(spec.id), spec.options,
                    list(s.audio_llm_server_args), parallel=s.audio_llm_parallel, what="audio language model",
                    fallback="or set audio_llm_base_url to a running server",
                )  # fmt: skip
            process = self.process
        await asyncio.to_thread(process.ensure_started)
        return process.base_url + "/v1", process.api_key

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"managed": True}
        if self.process is not None:
            out["process"] = self.process.status()
        if self.placement is not None:
            out["placement"] = self.placement.to_dict()
        return out

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            await asyncio.to_thread(process.stop)
