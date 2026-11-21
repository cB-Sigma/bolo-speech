"""Providers bolo uses only for cloud speech-to-speech (``s2s_model``; :mod:`bolo.s2s`): xAI's Voice Agent API and
Rime's Prism. Their adapters only check the key (``bolo providers test``); recognition and synthesis through them
answer 400, as for any adapter without those APIs.

* xAI (docs.x.ai, the voice reference, read 2026-11-21): ``Authorization: Bearer <key>``; ``GET
  https://api.x.ai/v1/tts/voices`` lists the voices and needs the key.
* Rime (docs.rime.ai, the Prism WebSocket reference, read 2026-11-21): ``Authorization: Bearer <key>`` on the
  WebSocket upgrade of ``wss://prism.api.rime.ai/v1/realtime``; a refused key is a ``401`` at the upgrade, before any
  event. Prism has no REST route, so the check opens the socket, waits for ``session.created`` and closes it without
  sending a ``session.update`` (no conversation is set up and no audio is sent).
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from bolo.errors import ProviderAuthError, ProviderError
from bolo.remote.base import Adapter


class XAI(Adapter):
    kind = "xai"

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.provider.api_key}"} if self.provider.api_key else {}

    def check(self) -> dict[str, Any]:
        doc = self.json(self.request("GET", self.base_url + "/tts/voices"))
        voices = doc.get("voices")
        count = len(voices) if isinstance(voices, list) else 0
        return {"checked": "credentials", "message": f"the key was accepted; {count} voices are listed"}


class RimePrism(Adapter):
    kind = "rime"

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.provider.api_key}"} if self.provider.api_key else {}

    def check(self) -> dict[str, Any]:
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect

        from bolo.s2s import realtime_url

        url = realtime_url(self.base_url)
        first: Any = None
        try:
            with connect(url, additional_headers=self.auth_headers(), open_timeout=10, close_timeout=2) as ws:
                with contextlib.suppress(Exception):
                    first = json.loads(ws.recv(timeout=10))
        except InvalidStatus as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise ProviderAuthError(f"{self.where} refused bolo's API key ({status})") from exc
            raise ProviderError(f"{self.where} refused the connection ({status})") from exc
        except (OSError, TimeoutError) as exc:
            raise ProviderError(self._clean(f"{self.where} is not reachable ({type(exc).__name__}: {exc})")) from exc
        if isinstance(first, dict) and first.get("type") == "error":
            message = self._clean(str((first.get("error") or {}).get("message") or "error"))
            raise ProviderError(f"{self.where} answered: {message}")
        return {"checked": "credentials", "message": "the key was accepted (the Prism socket opened)"}
