"""Full-duplex speech models in Realtime and WebRTC sessions.

* :mod:`bolo.duplex.base`: the backend interface (:class:`DuplexBackend`, :class:`DuplexSession`, the events).
* :mod:`bolo.duplex.bridge`: one model session behind one Realtime session (audio clock, responses, transcripts,
  cancellation), shared by every backend.
* :mod:`bolo.duplex.kyutai`: the backend for servers that speak Kyutai's WebSocket protocol (Moshi, Hibiki,
  PersonaPlex), with :mod:`bolo.duplex.ogg_opus` for its audio framing.
* :mod:`bolo.duplex.freeze_omni`: the backend for Tencent's Freeze-Omni server (Socket.IO, :mod:`bolo.duplex.socketio`).
* :mod:`bolo.duplex.runtime`: installing and starting those servers (components ``moshi`` and ``personaplex``).
* :mod:`bolo.duplex.manager`: which backend serves ``s2s_model``.
"""

from bolo.duplex.base import (
    FULL_DUPLEX,
    DuplexAudio,
    DuplexBackend,
    DuplexConfig,
    DuplexEnd,
    DuplexEvent,
    DuplexInterrupt,
    DuplexSession,
    DuplexText,
)

__all__ = [
    "FULL_DUPLEX",
    "DuplexAudio",
    "DuplexBackend",
    "DuplexConfig",
    "DuplexEnd",
    "DuplexEvent",
    "DuplexInterrupt",
    "DuplexSession",
    "DuplexText",
]
