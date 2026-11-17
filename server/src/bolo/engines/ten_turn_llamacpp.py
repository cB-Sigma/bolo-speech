"""TEN Turn Detection on llama-server: an end-of-turn model that reads the transcript (kind ``ten-turn-llamacpp``).

TEN Turn Detection (Agora) is Qwen2.5-7B fine-tuned to answer, for the user's text, with one word: ``finished``,
``unfinished`` or ``wait`` (the user asked the agent to stop talking). Its reference inference applies the chat
template to an empty system message and the user's text and generates one token. bolo does the same through
llama-server's ``POST /v1/chat/completions`` (``max_tokens: 1``, ``temperature: 0``, ``logprobs`` with the top
candidates) and turns the first token's probabilities into the probability that the turn is over: ``finished`` and
``wait`` both end the user's turn (after ``wait`` the agent should stay quiet, which is the pipeline's business),
``unfinished`` does not. Without log-probabilities the answered word decides (1.0 or 0.0).

The GGUF runs in a llama-server bolo starts (``bolo components install llama-server``) or one the operator runs
(setting ``tts_servers``, ``"ten-turn-detection=http://host:port"``), as the other llama.cpp models.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import httpx

from bolo.engines.base import FloatArray, LoadOptions, TurnModel
from bolo.errors import EngineUnavailable

KIND = "ten-turn-llamacpp"
LABELS = ("finished", "unfinished", "wait")
#: labels that end the user's turn
ENDS_TURN = ("finished", "wait")


def label_of(token: str) -> str | None:
    """Which label a first answered token stands for (``" finished"``, ``"Fin"``, ``"un"``...), or None."""
    t = token.strip().lower()
    if not t:
        return None
    for label in ("unfinished", "finished", "wait"):  # "un..." before "finished": "unfinished" contains it
        if label.startswith(t) or t.startswith(label):
            return label
    return None


def turn_probability(choice: dict[str, Any]) -> float | None:
    """P(the turn is over) from one chat-completions choice: the first token's top log-probabilities when there are
    some, else the answered word; None when the answer is none of the labels."""
    content = ((choice.get("logprobs") or {}).get("content")) or []
    if content:
        mass = dict.fromkeys(LABELS, 0.0)
        first = content[0]
        candidates = first.get("top_logprobs") or [{"token": first.get("token", ""), "logprob": first.get("logprob")}]
        for cand in candidates:
            label = label_of(str(cand.get("token", "")))
            lp = cand.get("logprob")
            if label is not None and isinstance(lp, (int, float)):
                mass[label] += math.exp(float(lp))
        total = sum(mass.values())
        if total > 0:
            return sum(mass[k] for k in ENDS_TURN) / total
    label = label_of(str((choice.get("message") or {}).get("content") or ""))
    if label is None:
        return None
    return 1.0 if label in ENDS_TURN else 0.0


class TenTurnLlamaCpp(TurnModel):
    """TEN Turn Detection in llama-server (managed by bolo, or the operator's)."""

    hears = "text"

    def __init__(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        base_url: str | None = None,
        process: Any = None,
        api_key: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.name = name
        self.process = process
        self._base_url = (base_url or (process.base_url if process else "")).rstrip("/")
        self.system_prompt = str(spec.get("system_prompt") or "")
        self.top_logprobs = int(spec.get("top_logprobs", 20))
        api_key = api_key or (process.api_key if process is not None else None)
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            trust_env=process is None,  # never send loopback traffic of a managed server through a proxy
        )
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return self.process.base_url if self.process is not None else self._base_url

    def connect(self) -> None:
        if self.process is not None:
            self.process.ensure_started()
        try:
            resp = self._client.get(self.base_url + "/health", timeout=10.0)
        except httpx.HTTPError as exc:
            raise EngineUnavailable(
                f"llama-server for '{self.name}' at {self.base_url} is not reachable ({type(exc).__name__}: {exc})"
            ) from exc
        if resp.status_code == 503:
            raise EngineUnavailable(f"llama-server at {self.base_url} is still loading its model; try again shortly")
        if resp.status_code >= 400:
            raise EngineUnavailable(f"llama-server at {self.base_url} answered /health with {resp.status_code}")

    def close(self) -> None:
        if self.process is not None:
            self.process.stop()
        self._client.close()

    def request(self, text: str) -> dict[str, Any]:
        return {
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

    def probability(self, samples: FloatArray) -> float:
        raise NotImplementedError("this end-of-turn model reads text: use probability_text")

    def probability_text(self, text: str) -> float:
        try:
            resp = self._client.post(self.base_url + "/v1/chat/completions", json=self.request(" ".join(text.split())))
        except httpx.HTTPError as exc:
            raise EngineUnavailable(f"llama-server at {self.base_url} failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineUnavailable(f"llama-server at {self.base_url} answered {resp.status_code}: {resp.text[:300]}")
        choices = resp.json().get("choices") or [{}]
        p = turn_probability(choices[0])
        if p is None:
            raise EngineUnavailable(f"model '{self.name}' answered none of {', '.join(LABELS)}: {choices[0]!r:.200}")
        return p


def load_model(engine: Any, path: Path, spec: dict[str, Any], opts: LoadOptions) -> TenTurnLlamaCpp:
    """The ``ten-turn-llamacpp`` kind of the tts-server engine: an operator-run or a bolo-managed llama-server."""
    from bolo.tts_runtimes import LLAMA

    s = engine._settings()
    runtime = str(spec.get("runtime") or LLAMA)
    if runtime != LLAMA:
        raise EngineUnavailable(f"the {KIND} kind runs on llama-server, not {runtime!r}")
    name = str(spec.get("model_id") or path.name)
    timeout = float(spec.get("timeout_s") or 60.0)
    url = engine.remote(spec)
    if url:
        model = TenTurnLlamaCpp(name, spec, base_url=url, api_key=s.tts_server_api_key, timeout_s=timeout)
    else:
        process = engine._process(runtime, name, path, spec, opts)
        model = TenTurnLlamaCpp(name, spec, process=process, timeout_s=timeout)
    try:
        model.connect()
    except BaseException:
        model.close()
        raise
    return model
