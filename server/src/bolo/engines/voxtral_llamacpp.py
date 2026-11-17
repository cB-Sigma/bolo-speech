"""Voxtral speech-to-text without PyTorch: the model runs in llama.cpp's server, bolo builds the prompts.

Mistral's Voxtral (Mini 3B, Small 24B) is a Whisper-style audio encoder in front of a Mistral language model.
llama.cpp runs both: ``llama-server -m <model.gguf> -mm <mmproj.gguf>`` (the ``voxtral`` projector of libmtmd).
The server is the one the ``tts-server`` engine manages (:mod:`bolo.engines.tts_server`): started by bolo on a
loopback port with a random API key and stopped on unload, or run by the operator and named in the setting
``tts_servers`` (``"<model id>=<url>"``). This module is the client.

Each request is one ``POST /completion`` whose ``prompt`` is ``{"prompt_string", "multimodal_data"}``: the audio
goes as a base64 16 kHz mono WAV, and the prompt holds the server's media marker where the audio belongs (the
marker is read from ``GET /props``; recent builds make it random). The prompts follow mistral-common's request
encoding, without the BOS the server adds itself, and libmtmd adds ``[BEGIN_AUDIO]`` in front of the audio:

* transcription (mistral-common's transcription request): ``[INST]<audio>[/INST]lang:<code>[TRANSCRIBE]``, or
  without ``lang:<code>`` when the request names no language (the model then finds it, but does not say which);
* translation into English (an instruction in the chat format, as Mistral's examples ask for it):
  ``[INST]<audio>Translate this audio into English. ...[/INST]``.

The prompts are built here rather than with ``/v1/chat/completions`` because the chat template inside a GGUF is
whatever its converter put there (one widely used conversion carries another model's template).

What the model does not give: timestamps. Segment times are those of the pieces bolo hands the model (long audio is
cut at pauses by voice-activity detection into pieces of at most ``max_chunk_s``), and word timestamps are
unavailable. The OpenAI ``prompt`` is not used (transcription mode takes no text). Decoding is greedy unless the
request sets a temperature; the token budget follows the length of the audio (``tokens_per_s`` per second plus
32, at most ``max_new_tokens``), which stops a model that repeats itself, and generation stops at a blank line
(``stop``): on llama.cpp the model sometimes goes on after the transcript as if answering a chat message, and a
transcript has no blank line.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from bolo.engines.base import FloatArray, LoadOptions, STTModel, STTRequest, STTResult
from bolo.errors import EngineUnavailable, InvalidRequest

if TYPE_CHECKING:
    from bolo.engines.tts_server import ServerProcess, TTSServerEngine

log = logging.getLogger("bolo.voxtral")

KIND = "voxtral-llamacpp"
SAMPLE_RATE = 16000
#: llama.cpp's media marker before builds made it random; used when ``/props`` does not name one
DEFAULT_MARKER = "<__media__>"
TRANSLATE_INSTRUCTION = (
    "Translate this audio into English. Reply with the English translation only, without any comment."
)


def wav_bytes(samples: FloatArray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono PCM16 WAV of float samples in -1..1."""
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_prompt(marker: str, task: str, language: str | None) -> str:
    """The prompt string of one request (the server adds BOS; libmtmd adds ``[BEGIN_AUDIO]``)."""
    if task == "translate":
        return f"[INST]{marker}{TRANSLATE_INSTRUCTION}[/INST]"
    lang = f"lang:{language}" if language else ""
    return f"[INST]{marker}[/INST]{lang}[TRANSCRIBE]"


def token_budget(seconds: float, tokens_per_s: float, max_new_tokens: int) -> int:
    return max(16, min(max_new_tokens, int(seconds * tokens_per_s) + 32))


class VoxtralLlamaCpp(STTModel):
    """A Voxtral model in llama-server: transcription in any of its languages, translation into English."""

    supports_translate = True
    supports_word_timestamps = False

    def __init__(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        base_url: str | None = None,
        process: ServerProcess | None = None,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        languages: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.spec = spec
        self.process = process
        self._base_url = (base_url or (process.base_url if process else "")).rstrip("/")
        self.sample_rate = SAMPLE_RATE
        self.max_chunk_s = float(spec.get("max_chunk_s") or 15.0)
        self.languages = languages
        self.tokens_per_s = float(spec.get("tokens_per_s", 12.0))
        self.max_new_tokens = int(spec.get("max_new_tokens", 4096))
        self.translate = bool(spec.get("translate", True))
        self.supports_translate = self.translate
        self.request_extra = dict(spec.get("request") or {})
        self.marker = str(spec.get("media_marker") or "")
        #: llama.cpp's Voxtral can carry on after the transcript as if chatting ("Here's a breakdown..."), always
        #: after a blank line; a transcript has none, so generation stops there
        self.stop = [str(x) for x in spec.get("stop", ["\n\n"]) or []]
        api_key = api_key or (process.api_key if process is not None else None)
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            trust_env=process is None,  # never send loopback traffic of a managed server through a proxy
        )

    @property
    def base_url(self) -> str:
        return self.process.base_url if self.process is not None else self._base_url

    # -- setup ----------------------------------------------------------------------------------------------
    def connect(self) -> None:
        """Start (managed) or reach (operator-run) the server, check that it hears audio, read its media marker."""
        if self.process is not None:
            self.process.ensure_started()
        try:
            resp = self._client.get(self.base_url + "/health", timeout=10.0)
            if resp.status_code == 503:
                raise EngineUnavailable(
                    f"llama-server at {self.base_url} is still loading its model; try again shortly"
                )
            if resp.status_code >= 400:
                raise EngineUnavailable(f"llama-server at {self.base_url} answered /health with {resp.status_code}")
            props = self._client.get(self.base_url + "/props", timeout=10.0)
        except httpx.HTTPError as exc:
            raise EngineUnavailable(self._unreachable(exc)) from exc
        doc: dict[str, Any] = {}
        if props.status_code < 400:
            try:
                found = props.json()
                doc = found if isinstance(found, dict) else {}
            except ValueError:
                doc = {}
        modalities = doc.get("modalities")
        if isinstance(modalities, dict) and modalities.get("audio") is False:
            raise EngineUnavailable(
                f"llama-server at {self.base_url} was started without an audio encoder: model '{self.name}' needs "
                "its mmproj file (-mm)"
            )
        if not self.marker:
            self.marker = str(doc.get("media_marker") or DEFAULT_MARKER)

    def close(self) -> None:
        if self.process is not None:
            self.process.stop()
        self._client.close()

    def _unreachable(self, exc: Exception) -> str:
        where = "bolo's llama-server" if self.process else "llama-server"
        if isinstance(exc, httpx.TimeoutException):
            return f"{where} at {self.base_url} did not answer in time ({type(exc).__name__})"
        return f"{where} at {self.base_url} is not reachable ({type(exc).__name__}: {exc})"

    # -- requests ---------------------------------------------------------------------------------------------
    def completion_request(self, req: STTRequest) -> dict[str, Any]:
        language = (req.language or "").strip().lower().split("-", 1)[0] or None
        seconds = len(req.samples) / self.sample_rate
        body: dict[str, Any] = {
            "prompt": {
                "prompt_string": build_prompt(self.marker or DEFAULT_MARKER, req.task, language),
                "multimodal_data": [base64.b64encode(wav_bytes(req.samples, self.sample_rate)).decode("ascii")],
            },
            "n_predict": token_budget(seconds, self.tokens_per_s, self.max_new_tokens),
            "temperature": float(req.temperature or 0.0),
            "stream": False,
            "cache_prompt": False,
        }
        if self.stop:
            body["stop"] = list(self.stop)
        if not req.temperature:
            body["top_k"] = 1
        body.update(self.request_extra)
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.process is not None:
            self.process.ensure_started()
        for attempt in (1, 2):
            try:
                resp = self._client.post(self.base_url + "/completion", json=body)
            except (httpx.RemoteProtocolError, httpx.ConnectError) as exc:
                if self.process is not None:
                    self.process.check()
                if attempt == 1 and (self.process is None or self.process.running):
                    continue
                raise EngineUnavailable(self._unreachable(exc)) from exc
            except httpx.HTTPError as exc:
                if self.process is not None:
                    self.process.check()
                raise EngineUnavailable(self._unreachable(exc)) from exc
            if resp.status_code >= 400:
                self._raise_for(resp)
            try:
                doc = resp.json()
            except ValueError as exc:
                raise EngineUnavailable(f"llama-server at {self.base_url} answered /completion with no JSON") from exc
            if not isinstance(doc, dict):
                raise EngineUnavailable(f"llama-server at {self.base_url} answered /completion with {type(doc)}")
            return doc
        raise AssertionError("unreachable")  # pragma: no cover

    def _raise_for(self, resp: httpx.Response) -> None:
        from bolo.engines.tts_server import _error_message

        message = _error_message(resp)
        where = f"llama-server at {self.base_url}"
        if resp.status_code in (400, 422):
            raise InvalidRequest(f"{where} refused the request for '{self.name}': {message}")
        if resp.status_code in (401, 403):
            raise EngineUnavailable(f"{where} refused bolo's credentials (set tts_server_api_key): {message}")
        raise EngineUnavailable(f"{where} failed ({resp.status_code}): {message}")

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        out: list[STTResult] = []
        for req in requests:
            if req.task == "translate" and not self.supports_translate:
                raise InvalidRequest(f"model '{self.name}' cannot translate", param="model")
            if len(req.samples) < int(0.05 * self.sample_rate):
                out.append(STTResult("", req.language))
                continue
            doc = self._post(self.completion_request(req))
            text = str(doc.get("content") or "").strip()
            if doc.get("stop_type") == "limit":
                log.warning("model '%s' reached its token budget; the text may be cut short", self.name)
            language = "en" if req.task == "translate" else req.language
            out.append(STTResult(text, language))
        return out


def load_model(engine: TTSServerEngine, path: Path, spec: dict[str, Any], opts: LoadOptions) -> VoxtralLlamaCpp:
    """The ``voxtral-llamacpp`` kind of the tts-server engine: an operator-run or a bolo-managed llama-server."""
    from bolo.tts_runtimes import LLAMA

    s = engine._settings()
    runtime = str(spec.get("runtime") or LLAMA)
    if runtime != LLAMA:
        raise EngineUnavailable(f"the {KIND} kind runs on llama-server, not {runtime!r}")
    if not spec.get("mmproj_file") and engine.remote(spec) is None:
        raise EngineUnavailable(
            f"model '{spec.get('model_id')}': the catalog entry names no audio encoder (mmproj_file)"
        )
    name = str(spec.get("model_id") or path.name)
    languages = tuple(str(x) for x in spec.get("languages") or ())
    timeout = max(float(s.tts_server_timeout_s), float(spec.get("timeout_s") or 600.0))
    url = engine.remote(spec)
    if url:
        model = VoxtralLlamaCpp(
            name, spec, base_url=url, api_key=s.tts_server_api_key, timeout_s=timeout, languages=languages
        )
    else:
        process = engine._process(runtime, name, path, spec, opts)
        model = VoxtralLlamaCpp(name, spec, process=process, timeout_s=timeout, languages=languages)
    try:
        model.connect()
    except BaseException:
        model.close()
        raise
    return model
