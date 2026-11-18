"""Translation into languages other than English (``target_language``), with the fake recogniser."""

from __future__ import annotations

from typing import Any

import httpx

from bolo.engines.base import STTRequest
from conftest import LiveServer, wav_bytes


def _post(live: LiveServer, route: str = "translations", **fields: Any) -> httpx.Response:
    data = {"model": "fake-stt", "response_format": "verbose_json", **{k: str(v) for k, v in fields.items()}}
    return httpx.post(
        f"{live.url}/v1/audio/{route}", data=data, files={"file": ("a.wav", wav_bytes(2.0), "audio/wav")}, timeout=30
    )


def test_target_language_reaches_the_model_it_suits(live: LiveServer) -> None:
    assert _post(live).status_code == 200  # English, as before
    r = _post(live, target_language="de")
    assert r.status_code == 400 and r.json()["error"]["param"] == "target_language"
    assert "cannot translate into German" in r.json()["error"]["message"]
    r = _post(live, route="transcriptions", target_language="de")
    assert r.status_code == 400 and r.json()["error"]["param"] == "target_language"
    entry = live.core.manager.loaded["fake-stt"]
    model = entry.model
    seen: list[STTRequest] = []
    handlers = entry.queue._handlers
    original = handlers["batch"]

    def spy(requests: list[STTRequest]) -> Any:
        seen.extend(requests)
        return original(requests)

    model.translate_targets = ("en", "de")  # type: ignore[misc]
    handlers["batch"] = spy
    try:
        r = _post(live, target_language="de", language="en")
        assert r.status_code == 200, r.text
        assert r.json()["language"] == "german"
        assert seen and all(q.task == "translate" and q.target_language == "de" for q in seen)
    finally:
        model.translate_targets = ("en",)  # type: ignore[misc]
        handlers["batch"] = original


def test_canary_translates_to_or_from_english() -> None:
    from bolo.engines.sherpa import SherpaOfflineSTT

    canary = SherpaOfflineSTT.__new__(SherpaOfflineSTT)
    canary.kind = "canary"
    assert canary.translation_problem("de", "en") is None and canary.translation_problem("en", "fr") is None
    assert "not from de into fr" in (canary.translation_problem("de", "fr") or "")
    assert "give `language`" in (canary.translation_problem(None, "en") or "")
    assert "already in en" in (canary.translation_problem("en", "en") or "")
    whisper = SherpaOfflineSTT.__new__(SherpaOfflineSTT)
    whisper.kind = "whisper"
    assert whisper.translation_problem(None, "en") is None
