"""Google's Gemini Decisions API (provider kind ``gemini``): ``gemini-decision-1``, a decision model in the cloud.

``POST {base_url}/decisions`` (base ``https://generativelanguage.googleapis.com/v1beta``) with the key in
``x-goog-api-key``. The body is ``model``, ``input`` (content blocks: ``{"type": "text", "text"}``, and the recording
as ``{"type": "audio", "data": <base64>, "mime_type": "audio/wav"}``) and ``questions``, which have the shape of
llama.cpp's ``/v1/systemone``: ``type`` ``noul``, ``choice`` or ``score``, ``instructions`` and ``criteria`` (a
``choice`` maps each option to a description; a ``score`` lists its levels). The answers come back in the same shape
as ``/v1/systemone``'s (``noul``; ``choice``, ``probabilities``, ``confidence``; ``score``, ``legend``,
``probabilities``, ``confidence``), with ``usage.total_input_tokens``; output tokens are always 0.

Written from Google's public documentation (the Gemini API's Decisions guide and the getting-started codelab, read
2026-11-21); not run against a real account.
"""

from __future__ import annotations

import base64
from typing import Any

from bolo.errors import ProviderError
from bolo.providers import Usage
from bolo.remote.base import Adapter


class Gemini(Adapter):
    kind = "gemini"

    def auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.provider.api_key} if self.provider.api_key else {}

    def check(self) -> dict[str, Any]:
        doc = self.json(self.request("GET", self.base_url + "/models"))
        models = doc.get("models")
        count = len(models) if isinstance(models, list) else 0
        return {"checked": "credentials", "message": f"the key was accepted; {count} models are listed"}

    @staticmethod
    def questions(questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """``/v1/systemone``'s questions as Gemini takes them: a choice option without a description is described
        by its name."""
        out: dict[str, dict[str, Any]] = {}
        for qid, q in questions.items():
            one: dict[str, Any] = {"type": q["type"], "instructions": q["instructions"]}
            if q["type"] == "choice":
                one["criteria"] = {str(k): (v if v else str(k)) for k, v in dict(q["criteria"]).items()}
            elif q["type"] == "score":
                one["criteria"] = list(q["criteria"])
            out[qid] = one
        return out

    def decide(
        self, text: str | None, wav: bytes | None, questions: dict[str, dict[str, Any]], usage: Usage
    ) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        if wav is not None:
            blocks.append({"type": "audio", "data": base64.b64encode(wav).decode("ascii"), "mime_type": "audio/wav"})
        body = {"model": self.model, "input": blocks, "questions": self.questions(questions)}
        doc = self.json(self.request("POST", self.base_url + "/decisions", usage, json=body))
        answers = doc.get("answers")
        if not isinstance(answers, dict):
            raise ProviderError(f"{self.where} answered without answers")
        found = doc.get("usage")
        raw_usage: dict[str, Any] = found if isinstance(found, dict) else {}
        tokens = int(raw_usage.get("total_input_tokens") or raw_usage.get("input_tokens") or 0)
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}
