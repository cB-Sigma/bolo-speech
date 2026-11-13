"""In-process publish/subscribe used by the ``/api/v1/events`` SSE stream."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncGenerator
from typing import Any

Event = dict[str, Any]


class EventBus:
    """Fan-out of small JSON events to any number of slow or fast subscribers.

    Each subscriber has a bounded queue. A subscriber that cannot keep up
    loses its oldest events rather than slowing publishers down.
    """

    def __init__(self, max_queue: int = 512) -> None:
        self._subs: set[asyncio.Queue[Event]] = set()
        self._max_queue = max_queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def publish(self, type_: str, payload: Event | None = None) -> None:
        """Publish from the event loop thread."""
        event: Event = {"type": type_, **(payload or {})}
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    def publish_threadsafe(self, type_: str, payload: Event | None = None) -> None:
        """Publish from any thread (download workers, engine threads)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self.publish(type_, payload)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.publish, type_, payload)

    async def subscribe(self) -> AsyncGenerator[Event, None]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subs.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            with self._lock:
                self._subs.discard(q)
