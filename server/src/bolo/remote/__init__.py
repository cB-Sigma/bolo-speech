"""Provider adapters of the ``remote`` engine: one class per API family, behind :class:`bolo.remote.base.Adapter`."""

from __future__ import annotations

from typing import Any

from bolo.errors import InvalidRequest
from bolo.providers import ProviderConfig
from bolo.remote.base import Adapter, LiveProtocol, RemoteRequest

ADAPTERS: dict[str, str] = {
    "openai-compatible": "bolo.remote.openai_compat:OpenAICompatible",
    "sarvam": "bolo.remote.sarvam:Sarvam",
    "elevenlabs": "bolo.remote.elevenlabs:ElevenLabs",
    "deepgram": "bolo.remote.deepgram:Deepgram",
    "soniox": "bolo.remote.soniox:Soniox",
    "assemblyai": "bolo.remote.assemblyai:AssemblyAI",
}


def make_adapter(provider: ProviderConfig, model: str, spec: dict[str, Any], *, timeout_s: float = 60.0) -> Adapter:
    import importlib

    target = ADAPTERS.get(provider.kind)
    if target is None:
        raise InvalidRequest(f"provider '{provider.name}' has the unknown kind {provider.kind!r}")
    module_name, _, cls_name = target.partition(":")
    cls: type[Adapter] = getattr(importlib.import_module(module_name), cls_name)
    return cls(provider, model, spec, timeout_s=timeout_s)


__all__ = ["ADAPTERS", "Adapter", "LiveProtocol", "RemoteRequest", "make_adapter"]
