"""Client for the operator's language model: any OpenAI-compatible chat-completions endpoint.

bolo does not run language models. For voice agents (the Realtime API's
conversation mode, and audio in chat completions) it talks to whatever the
operator points it at with ``llm_base_url``: Ollama, llama.cpp's server,
vLLM, LM Studio, or a hosted API.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from bolo.config import Settings
from bolo.errors import LLMError, LLMNotConfigured


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.llm_base_url)

    @property
    def base_url(self) -> str:
        if not self.settings.llm_base_url:
            raise LLMNotConfigured(
                "no language model is configured. Set llm_base_url (and llm_model) to an OpenAI-compatible "
                "chat-completions endpoint, for example BOLO_LLM_BASE_URL=http://127.0.0.1:11434/v1 for Ollama."
            )
        return self.settings.llm_base_url.rstrip("/")

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            h["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return h

    def describe(self) -> dict[str, Any]:
        return {"configured": self.configured, "base_url": self.settings.llm_base_url, "model": self.settings.llm_model}

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.settings.llm_timeout_s, connect=10.0)

    @staticmethod
    def _upstream_message(status: int, body: bytes) -> str:
        try:
            doc = json.loads(body)
            err = doc.get("error")
            detail = err.get("message") if isinstance(err, dict) else (err or doc.get("detail") or doc.get("message"))
        except (ValueError, AttributeError):
            detail = body.decode(errors="replace")[:300]
        return f"the language model endpoint answered HTTP {status}: {detail}"

    async def request(self, body: dict[str, Any]) -> httpx.Response:
        """POST a non-streaming chat completion and return the upstream response as is."""
        url = self.base_url + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                return await client.post(url, json=body, headers=self.headers())
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach the language model at {url}: {type(exc).__name__}: {exc}") from exc

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self.request({**body, "stream": False})
        if resp.status_code >= 400:
            raise LLMError(self._upstream_message(resp.status_code, resp.content))
        doc: dict[str, Any] = resp.json()
        return doc

    async def raw_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream the upstream SSE bytes untouched (used when proxying)."""
        url = self.base_url + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                async with client.stream("POST", url, json=body, headers=self.headers()) as resp:
                    if resp.status_code >= 400:
                        raise LLMError(self._upstream_message(resp.status_code, await resp.aread()))
                    async for chunk in resp.aiter_raw():
                        yield chunk
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach the language model at {url}: {type(exc).__name__}: {exc}") from exc

    async def stream_chat(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat completion as parsed ``chat.completion.chunk`` objects."""
        url = self.base_url + "/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                async with client.stream("POST", url, json={**body, "stream": True}, headers=self.headers()) as resp:
                    if resp.status_code >= 400:
                        raise LLMError(self._upstream_message(resp.status_code, await resp.aread()))
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            continue
                        if isinstance(chunk, dict):
                            if chunk.get("error"):
                                raise LLMError(self._upstream_message(200, data.encode()))
                            yield chunk
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach the language model at {url}: {type(exc).__name__}: {exc}") from exc

    async def check(self) -> dict[str, Any]:
        """Is the endpoint reachable, and does it know the configured model? (``bolo doctor``)"""
        out: dict[str, Any] = {
            **self.describe(),
            "reachable": False,
            "models": None,
            "model_found": None,
            "error": None,
        }
        if not self.configured:
            return out
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                resp = await client.get(self.base_url + "/models", headers=self.headers())
            out["reachable"] = resp.status_code < 500
            if resp.status_code == 200:
                ids = [m.get("id") for m in resp.json().get("data", []) if isinstance(m, dict)]
                out["models"] = ids[:50]
                if self.settings.llm_model:
                    out["model_found"] = self.settings.llm_model in ids
            else:
                out["error"] = f"GET /models answered HTTP {resp.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out


class AudioLLMClient(LLMClient):
    """The model of the Realtime ``audio-llm`` pipeline: a chat-completions endpoint that takes ``input_audio``
    content (llama-server with Gemma 4 E2B/E4B, Ultravox, Qwen2.5-Omni, ...). Falls back to the ``llm_*``
    settings for whatever ``audio_llm_*`` leaves unset."""

    @property
    def configured(self) -> bool:
        return bool(self.settings.audio_llm_base_url or self.settings.llm_base_url)

    @property
    def base_url(self) -> str:
        url = self.settings.audio_llm_base_url or self.settings.llm_base_url
        if not url:
            raise LLMNotConfigured(
                "the audio-llm pipeline needs an audio-understanding model: set audio_llm_base_url (and "
                "audio_llm_model) to a chat-completions endpoint that accepts input_audio, for example a llama-server "
                "started with -hf ggml-org/gemma-4-E4B-it-GGUF"
            )
        return url.rstrip("/")

    @property
    def model(self) -> str | None:
        return self.settings.audio_llm_model or self.settings.llm_model

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = self.settings.audio_llm_api_key if self.settings.audio_llm_base_url else self.settings.llm_api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def describe(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "base_url": self.settings.audio_llm_base_url or self.settings.llm_base_url,
            "model": self.model,
        }
