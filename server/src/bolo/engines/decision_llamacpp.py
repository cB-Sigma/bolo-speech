"""Decision models on llama.cpp's server (kind ``llama-decision`` of the tts-server engine): Liquid's d1-omni.

llama.cpp serves native decision models on ``POST /v1/systemone`` (its TypeSafe-compatible System One API): a
``state`` (text, or null), ``files`` (data URLs; an audio clip for d1-omni) and ``questions``, each answered from the
model's distribution over the options, without generating text. Such a model refuses ``/v1/chat/completions``
(501). bolo starts the server (``-m`` the model, ``-mm`` its audio and image encoder) or uses one the operator runs
(setting ``tts_servers``), as for every tts-server model; d1-omni evaluates a whole prompt in one batch, so the
catalog entry gives ``-b``/``-ub`` 4096 in ``server_args``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from bolo.engines.base import DecisionModel, DecisionRequest, LoadOptions
from bolo.engines.voxtral_llamacpp import wav_bytes
from bolo.errors import EngineUnavailable, InvalidRequest

if TYPE_CHECKING:
    from bolo.engines.tts_server import ServerProcess, TTSServerEngine

KIND = "llama-decision"


class LlamaDecision(DecisionModel):
    """A decision model in llama-server, asked through ``/v1/systemone``."""

    def __init__(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        base_url: str | None = None,
        process: ServerProcess | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.name = name
        self.process = process
        self._base_url = (base_url or (process.base_url if process else "")).rstrip("/")
        self.max_audio_s = float(spec.get("max_audio_s") or 30.0)
        api_key = api_key or (process.api_key if process is not None else None)
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            trust_env=process is None,  # never send loopback traffic of a managed server through a proxy
        )

    @property
    def base_url(self) -> str:
        return self.process.base_url if self.process is not None else self._base_url

    def connect(self) -> None:
        if self.process is not None:
            self.process.ensure_started()
        try:
            resp = self._client.get(self.base_url + "/health", timeout=10.0)
        except httpx.HTTPError as exc:
            raise EngineUnavailable(f"llama-server at {self.base_url} is not reachable ({exc})") from exc
        if resp.status_code >= 400:
            raise EngineUnavailable(f"llama-server at {self.base_url} answered /health with {resp.status_code}")

    def close(self) -> None:
        if self.process is not None:
            self.process.stop()
        self._client.close()

    def request_body(self, request: DecisionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"state": request.text, "questions": request.questions}
        if request.samples is not None:
            data = base64.b64encode(wav_bytes(request.samples)).decode("ascii")
            body["files"] = [f"data:audio/wav;base64,{data}"]
        return body

    def decide(self, request: DecisionRequest) -> dict[str, Any]:
        from bolo.engines.tts_server import _error_message

        if self.process is not None:
            self.process.ensure_started()
        try:
            resp = self._client.post(self.base_url + "/v1/systemone", json=self.request_body(request))
        except httpx.HTTPError as exc:
            if self.process is not None:
                self.process.check()
            raise EngineUnavailable(f"llama-server at {self.base_url} did not answer ({type(exc).__name__})") from exc
        if resp.status_code == 400:
            raise InvalidRequest(f"llama-server refused the questions for '{self.name}': {_error_message(resp)}")
        if resp.status_code == 501:
            raise EngineUnavailable(
                f"llama-server at {self.base_url} does not serve '{self.name}' as a decision model: "
                f"{_error_message(resp)}"
            )
        if resp.status_code >= 400:
            raise EngineUnavailable(
                f"llama-server at {self.base_url} failed ({resp.status_code}): {_error_message(resp)}"
            )
        doc = resp.json()
        if not isinstance(doc, dict) or not isinstance(doc.get("answers"), dict):
            raise EngineUnavailable(f"llama-server at {self.base_url} answered /v1/systemone without answers")
        return doc


def load_model(engine: TTSServerEngine, path: Path, spec: dict[str, Any], opts: LoadOptions) -> LlamaDecision:
    from bolo.tts_runtimes import LLAMA

    s = engine._settings()
    runtime = str(spec.get("runtime") or LLAMA)
    if runtime != LLAMA:
        raise EngineUnavailable(f"the {KIND} kind runs on llama-server, not {runtime!r}")
    name = str(spec.get("model_id") or path.name)
    timeout = max(float(s.tts_server_timeout_s), float(spec.get("timeout_s") or 120.0))
    url = engine.remote(spec)
    if url:
        model = LlamaDecision(name, spec, base_url=url, api_key=s.tts_server_api_key, timeout_s=timeout)
    else:
        model = LlamaDecision(name, spec, process=engine._process(runtime, name, path, spec, opts), timeout_s=timeout)
    try:
        model.connect()
    except BaseException:
        model.close()
        raise
    return model
