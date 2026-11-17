"""The interface a full-duplex speech model implements to serve bolo's Realtime and WebRTC sessions.

A full-duplex model listens and speaks at the same time: it takes the caller's audio continuously, in small frames
(Moshi: 80 ms), and gives back audio continuously (silence while it listens) and, for most models, the text of
what it says, token by token. Nothing in it is turn-based: it decides itself when to speak, when to stop, and
what to do when the caller talks over it.

A backend has two parts:

* :class:`DuplexBackend`: the model behind it (a server process, a remote URL, a model in memory). One per server;
  it says how many sessions it can hold at once (:attr:`DuplexBackend.max_sessions`) and opens them.
* :class:`DuplexSession`: one live conversation. :meth:`DuplexSession.push` takes the caller's audio (any amount,
  at :attr:`DuplexSession.input_rate`), :meth:`DuplexSession.events` yields what the model produces
  (:class:`DuplexAudio`, :class:`DuplexText`) until a :class:`DuplexEnd`, and :meth:`DuplexSession.close` ends the
  conversation and frees what it held. Sessions keep no history after they close.

Everything else (timing, segmenting the model's speech into Realtime responses, transcripts, cancellation, the
WebRTC track) is shared and lives in :mod:`bolo.duplex.bridge`. docs/voice-agents.md, "Full-duplex models", says
how to add a backend.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from bolo.engines.base import FloatArray

#: the catalog feature of full-duplex models (task speech-to-speech)
FULL_DUPLEX = "full-duplex"


@dataclass
class DuplexAudio:
    """Audio the model produced (mono float32 at ``sample_rate``); silence counts too."""

    samples: FloatArray
    sample_rate: int


@dataclass
class DuplexText:
    """Text of what the model says, as it produces it (a token or a piece of a word, spaces included)."""

    text: str


@dataclass
class DuplexEnd:
    """The model ended the conversation: ``"closed"`` (the session was closed), ``"ended"`` (the model stopped,
    e.g. it reached its maximum length) or ``"failed"`` (with ``error``)."""

    reason: str = "ended"
    error: str | None = None


DuplexEvent = DuplexAudio | DuplexText | DuplexEnd


@dataclass
class DuplexConfig:
    """What a session asks of the model when it opens. A backend uses what its model understands and ignores the
    rest (Moshi has no instructions and one voice per checkpoint)."""

    instructions: str | None = None
    voice: str | None = None
    #: sampling and other model options (Kyutai: text_temperature, text_topk, audio_temperature, audio_topk, ...)
    options: dict[str, Any] = field(default_factory=dict)


class DuplexSession(abc.ABC):
    """One live conversation with a full-duplex model."""

    #: the sample rate :meth:`push` takes
    input_rate: int = 24000
    #: the length of the model's frame in samples at :attr:`input_rate` (what it consumes per step)
    frame_samples: int = 1920

    @abc.abstractmethod
    async def push(self, samples: FloatArray) -> None:
        """Audio of the caller, mono float32 at :attr:`input_rate`, in any amount."""

    @abc.abstractmethod
    def events(self) -> AsyncIterator[DuplexEvent]:
        """What the model produces, in order, until (and including) one :class:`DuplexEnd`."""

    @abc.abstractmethod
    async def close(self) -> None:
        """End the conversation; :meth:`events` then ends with ``DuplexEnd("closed")``. Safe to call twice."""


class DuplexBackend(abc.ABC):
    name: str = "duplex"
    #: the catalog model it serves
    model: str | None = None
    #: sessions it holds at once (more are refused with a clear error, not queued)
    max_sessions: int = 1

    @abc.abstractmethod
    async def open(self, config: DuplexConfig) -> DuplexSession: ...

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "full_duplex": True, "max_sessions": self.max_sessions}

    async def session_closed(self) -> None:
        """Called after a session closed (a backend whose server keeps state across connections restarts it)."""
        return None

    async def close(self) -> None:
        return None
