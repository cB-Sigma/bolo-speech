"""The ``tts-server`` engine: speech synthesis forwarded to an external llama-server or audio.cpp server.

One engine, two runtimes (see :mod:`bolo.tts_runtimes` for what each server speaks). A catalog entry names its
runtime in ``options.runtime`` and how the server loads it:

* llama-server: ``model_file`` and ``mmproj_file`` (``-m`` / ``-mm``), optional ``server_args``;
* audio.cpp: ``family``, ``model_file`` (a file of the package, or the package directory when absent), ``mode``
  (``offline`` or ``streaming``), ``load_options``, ``session_options``, ``server_model`` (the id in the
  server's config; defaults to the catalog id).

Either the operator runs the server and names it in the setting ``tts_servers`` (``<model id or runtime>=<url>``;
nothing is downloaded then), or bolo starts one per loaded model on a free loopback port, with a random API key
where the runtime has one, waits for ``GET /health``, and stops it when the model is unloaded. A server that
exits unexpectedly fails the requests in flight with ``engine_unavailable`` and is started again on the next
request after a back-off (1 s, doubling up to 60 s while crashes keep coming), as engine plugins are.

Audio comes back as PCM16 (streamed where the runtime streams) or WAV, and is resampled to the catalog's
``sample_rate`` with bolo's resampler, so the rest of bolo sees an ordinary :class:`TTSModel`.

The kind ``orpheus-llamacpp`` (Orpheus, Veena, Svara, snorTTS) uses the same llama-server management but its
own client, :mod:`bolo.engines.orpheus_llamacpp`: bolo builds the prompt, streams token ids from ``/completion``
and decodes the SNAC codes with onnxruntime.

Other per-entry options: ``cloning`` (the model takes a reference recording), ``send_language`` (tell the server
the language of the text), ``language_option`` (audio.cpp: also as the model option ``language``),
``language_names`` (send English language names instead of codes), ``language_map``, ``speed`` (the
server honours ``speed``), ``stream`` (ask for streamed audio), ``request`` (extra request fields), and voices
with ``reference: <bolo sample id>``: bolo's synthetic reference clips, sent as the voice to imitate.
"""

from __future__ import annotations

import base64
import collections
import contextlib
import io
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from bolo.audio.resample import StreamResampler
from bolo.engines.base import (
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    SynthesisExtras,
    TTSModel,
    Voice,
)
from bolo.errors import ComponentMissing, EngineUnavailable, InvalidRequest
from bolo.tts_runtimes import ENGINE_ID, LLAMA, RUNTIME_INFO, RUNTIMES, find_binary, url_for

if TYPE_CHECKING:
    from bolo.config import Settings

log = logging.getLogger("bolo.tts_server")

#: OpenAI voice names: stock clients always send one. Dropped for servers that would reject an unknown name.
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"}
LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese",
    "yue": "Cantonese", "zh": "Chinese",
}  # fmt: skip
_RATE_RE = re.compile(r"rate=(\d+)")
#: shortest piece of audio handed on while streaming (seconds); smaller network reads are joined
MIN_CHUNK_S = 0.08


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerProcess:
    """A server binary bolo started for one model: start, readiness, crash back-off, stop."""

    def __init__(
        self,
        name: str,
        argv: list[str],
        port: int,
        *,
        start_timeout_s: float = 300.0,
        env: dict[str, str] | None = None,
        health_path: str = "/health",
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.argv = argv
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.start_timeout_s = start_timeout_s
        self.env = env or {}
        self.health_path = health_path
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._tail: collections.deque[str] = collections.deque(maxlen=40)
        self._stopping = False
        self.starts = 0
        self.crashes = 0
        self.restarts = 0
        self.last_error: str | None = None
        self._backoff = 1.0
        self._not_before = 0.0

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _output(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self._tail.append(line)
                log.debug("[%s] %s", self.name, line)

    def tail(self, n: int = 6) -> str:
        return " | ".join(list(self._tail)[-n:])

    def _crashed(self, reason: str) -> None:
        self.crashes += 1
        self.last_error = reason
        self._not_before = time.monotonic() + self._backoff
        self._backoff = min(60.0, self._backoff * 2)
        log.error("TTS server %s stopped unexpectedly (%s)", self.name, reason)

    def check(self) -> None:
        """Notice a server that died since the last call (counts as a crash)."""
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                self._proc = None
                if not self._stopping:
                    self._crashed(f"exit code {proc.returncode}" + (f": {self.tail()}" if self._tail else ""))

    def ensure_started(self) -> None:
        with self._lock:
            self.check()
            if self.running:
                return
            wait = self._not_before - time.monotonic()
            if wait > 0:
                raise EngineUnavailable(
                    f"TTS server '{self.name}' stopped ({self.last_error or 'exited'}); restarting in {wait:.0f} s"
                )
            self._stopping = False
            self._tail.clear()
            env = {**os.environ, **self.env}
            try:
                proc = subprocess.Popen(
                    self.argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            except OSError as exc:
                self.last_error = str(exc)
                raise EngineUnavailable(f"TTS server '{self.name}' could not be started: {exc}") from exc
            self._proc = proc
            self.starts += 1
            if self.starts > 1:
                self.restarts += 1
            threading.Thread(
                target=self._output, args=(proc,), name=f"bolo-tts-server-{self.name}", daemon=True
            ).start()
            try:
                self._wait_ready(proc)
            except EngineUnavailable:
                self._terminate(proc)
                self._proc = None
                raise
            self._backoff = 1.0
            log.info("TTS server %s ready on port %d (pid %d)", self.name, self.port, proc.pid)

    def _wait_ready(self, proc: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        url = self.base_url + self.health_path
        with httpx.Client(timeout=2.0, trust_env=False) as client:
            while True:
                code = proc.poll()
                if code is not None:
                    killed = " (killed: often the system running out of memory)" if code == -9 else ""
                    reason = f"exit code {code}{killed} while starting" + (f": {self.tail()}" if self._tail else "")
                    self._crashed(reason)
                    raise EngineUnavailable(f"TTS server '{self.name}' did not start ({reason})")
                with contextlib.suppress(httpx.HTTPError):
                    if client.get(url).status_code == 200:
                        return
                if time.monotonic() > deadline:
                    reason = f"not ready after {self.start_timeout_s:.0f} s" + (
                        f": {self.tail()}" if self._tail else ""
                    )
                    self._crashed(reason)
                    raise EngineUnavailable(f"TTS server '{self.name}' did not start ({reason})")
                time.sleep(0.2)

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            self._stopping = True
        if proc is not None:
            self._terminate(proc)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name, "running": self.running, "pid": self._proc.pid if self._proc else None,
            "port": self.port, "starts": self.starts, "crashes": self.crashes, "restarts": self.restarts,
            "last_error": self.last_error,
        }  # fmt: skip


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.read()
    except httpx.HTTPError:  # pragma: no cover - a broken connection while reading the error
        return f"HTTP {resp.status_code}"
    with contextlib.suppress(ValueError):
        doc = json.loads(body)
        err = doc.get("error") if isinstance(doc, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if isinstance(doc, dict) and doc.get("message"):
            return str(doc["message"])
    text = body.decode(errors="replace").strip()
    return text[:300] or f"HTTP {resp.status_code}"


def _language_code(language: str | None) -> str | None:
    if not language:
        return None
    return language.replace("_", "-").split("-")[0].lower() or None


def _pcm16(data: bytes) -> FloatArray:
    return (np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0).astype(np.float32)


class ServerTTS(TTSModel):
    """A TTS model served by an external process (operator-run, or :class:`ServerProcess`)."""

    accepts_extras = True

    def __init__(
        self,
        runtime: str,
        name: str,
        spec: dict[str, Any],
        *,
        base_url: str | None = None,
        process: ServerProcess | None = None,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        server_model: str | None = None,
    ) -> None:
        if runtime not in RUNTIMES:
            raise EngineUnavailable(f"unknown TTS server runtime {runtime!r} (known: {', '.join(RUNTIMES)})")
        self.runtime = runtime
        self.name = name
        self.spec = spec
        self.process = process
        self._base_url = (base_url or (process.base_url if process else "")).rstrip("/")
        self.server_model = server_model or str(spec.get("server_model") or spec.get("model_id") or name)
        self.sample_rate = int(spec.get("sample_rate") or 24000)
        self.supports_cloning = bool(spec.get("cloning"))
        self.stream = bool(spec.get("stream", runtime == LLAMA))
        api_key = api_key or (process.api_key if process is not None else None)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            trust_env=process is None,  # never send loopback traffic of a managed server through a proxy
        )
        self._references: dict[str, tuple[bytes, str]] = {}
        self._server_voices: set[str] = set()
        voices: list[Voice] = []
        self._voice_info: dict[str, dict[str, Any]] = {}
        for i, v in enumerate(spec.get("voices") or []):
            vid = str(v["id"])
            voices.append(Voice(vid, str(v.get("name", "")), str(v.get("language", "")), str(v.get("gender", "")), i))
            self._voice_info[vid.lower()] = dict(v)
        self.voices = tuple(voices)

    @property
    def base_url(self) -> str:
        return self.process.base_url if self.process is not None else self._base_url

    # -- setup ----------------------------------------------------------------------------------------------
    def connect(self) -> None:
        """Check that the server answers and pick up the voices it knows (called once after loading)."""
        if self.process is not None:
            self.process.ensure_started()
        try:
            resp = self._client.get(self.base_url + "/health", timeout=10.0)
        except httpx.HTTPError as exc:
            raise EngineUnavailable(self._unreachable(exc)) from exc
        if resp.status_code == 503:
            raise EngineUnavailable(f"the TTS server at {self.base_url} is still loading its model; try again shortly")
        if resp.status_code >= 400:
            raise EngineUnavailable(f"the TTS server at {self.base_url} answered /health with {resp.status_code}")
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError, TypeError):
            self._add_server_voices(self._list_server_voices())

    def _list_server_voices(self) -> list[str]:
        if self.runtime == LLAMA:
            props = self._client.get(self.base_url + "/props", timeout=10.0).json()
            return [str(v) for v in props.get("voices") or []]
        resp = self._client.get(self.base_url + "/v1/audio/voices", params={"model": self.server_model}, timeout=10.0)
        return [str(v.get("id", v) if isinstance(v, dict) else v) for v in resp.json().get("voices") or []]

    def _add_server_voices(self, names: list[str]) -> None:
        known = {v.id.lower() for v in self.voices}
        extra = list(self.voices)
        for name in names:
            self._server_voices.add(name.lower())
            if name.lower() not in known:
                extra.append(Voice(name, name, "", "", len(extra)))
                known.add(name.lower())
        self.voices = tuple(extra)

    def _unreachable(self, exc: Exception) -> str:
        where = f"bolo's {RUNTIME_INFO[self.runtime].binary}" if self.process else "the TTS server"
        if isinstance(exc, httpx.TimeoutException):
            return f"{where} at {self.base_url} did not answer in time ({type(exc).__name__})"
        return f"{where} at {self.base_url} is not reachable ({type(exc).__name__}: {exc})"

    def close(self) -> None:
        if self.process is not None:
            self.process.stop()
        self._client.close()

    # -- request building -------------------------------------------------------------------------------------
    def _reference(self, sample_id: str) -> tuple[bytes, str]:
        if sample_id not in self._references:
            from bolo import samples

            entry = samples.get(sample_id)
            self._references[sample_id] = (samples.read_file(str(entry["file"])), str(entry.get("text", "")))
        return self._references[sample_id]

    def _language(self, language: str | None) -> str | None:
        code = _language_code(language)
        if code is None:
            return None
        mapping = self.spec.get("language_map") or {}
        if code in mapping:
            return str(mapping[code])
        if self.spec.get("language_names"):
            return LANGUAGE_NAMES.get(code, code)
        return code

    def build_request(
        self, text: str, voice: str | None, speed: float, extras: SynthesisExtras | None
    ) -> dict[str, Any]:
        extras = extras or SynthesisExtras()
        ref_audio, ref_text = extras.reference_audio, extras.reference_text
        info = self._voice_info.get((voice or "").lower(), {})
        send_voice: str | None = None
        if ref_audio is None and info.get("reference"):
            ref_audio, sample_text = self._reference(str(info["reference"]))
            ref_text = ref_text or sample_text
        elif ref_audio is None and voice:
            # a catalog voice may name the server's voice differently, or ("") stand for the model's own voice
            send_voice = (str(info["server_voice"]) or None) if "server_voice" in info else voice
            if (
                not info
                and send_voice
                and send_voice.lower() in OPENAI_VOICES
                and send_voice.lower() not in self._server_voices
            ):
                send_voice = None  # a stock OpenAI name the server would not know
        if ref_audio is not None and not self.supports_cloning:
            raise InvalidRequest(f"model '{self.name}' cannot imitate a voice from reference audio", param="voice")
        language = self._language(extras.language) if self.spec.get("send_language") else None
        if speed != 1.0 and not self.spec.get("speed"):
            raise InvalidRequest(f"model '{self.name}' speaks at one speed only; send speed 1.0", param="speed")
        extra_fields = dict(self.spec.get("request") or {})
        if self.runtime == LLAMA:
            body: dict[str, Any] = {"input": text, "response_format": "pcm16" if self.stream else "wav"}
            if self.stream:
                body["stream"] = True
            if language:
                body["lang"] = language
            if ref_audio is not None:
                body["speaker_ref_b64"] = base64.b64encode(ref_audio).decode()
            elif send_voice:
                body["voice"] = send_voice
            body.update(extra_fields)
            return body
        body = {"model": self.server_model, "input": text}
        if self.stream:
            body.update({"response_format": "pcm", "stream_format": "audio"})
        else:
            body["response_format"] = "wav"
        if language:
            body["language"] = language
        if ref_audio is not None:
            body["voice_ref"] = {"type": "base64", "data": base64.b64encode(ref_audio).decode()}
            if ref_text:
                body["reference_text"] = ref_text
        elif send_voice:
            body["voice"] = send_voice
        if speed != 1.0:
            body["speed"] = speed
        if language and self.spec.get("language_option"):
            # families that declare a "language" request option read it there, not from the top-level field
            extra_fields.setdefault("language", language)
        if extra_fields:
            body["options"] = {str(k): v for k, v in extra_fields.items()}
        return body

    # -- synthesis --------------------------------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        extras: SynthesisExtras | None = None,
    ) -> Iterator[FloatArray]:
        body = self.build_request(text, voice, speed, extras)
        if self.process is not None:
            self.process.ensure_started()
        path = "/tts" if self.runtime == LLAMA else "/v1/audio/speech"
        for attempt in (1, 2):
            started = False
            try:
                with self._client.stream("POST", self.base_url + path, json=body) as resp:
                    if resp.status_code >= 400:
                        self._raise_for(resp)
                    for piece in self._read_audio(resp, cancelled):
                        started = True
                        yield piece
                return
            except (httpx.RemoteProtocolError, httpx.ConnectError) as exc:
                # a kept-alive connection the server had already closed: once more on a fresh one, unless audio
                # was already delivered or the server process is gone
                if self.process is not None:
                    self.process.check()
                if attempt == 1 and not started and (self.process is None or self.process.running):
                    log.debug("retrying %s once after %s", self.name, exc)
                    continue
                raise EngineUnavailable(self._unreachable(exc)) from exc
            except httpx.HTTPError as exc:
                if self.process is not None:
                    self.process.check()
                raise EngineUnavailable(self._unreachable(exc)) from exc

    def _raise_for(self, resp: httpx.Response) -> None:
        message = _error_message(resp)
        where = f"{self.runtime} at {self.base_url}"
        if resp.status_code in (400, 422):
            raise InvalidRequest(f"{where} refused the request for '{self.name}': {message}")
        if resp.status_code in (401, 403):
            raise EngineUnavailable(f"{where} refused bolo's credentials (set tts_server_api_key): {message}")
        if resp.status_code == 404:
            raise EngineUnavailable(
                f"{where} does not serve '{self.server_model}' ({message}); check the server's models and the "
                "tts_servers setting"
            )
        raise EngineUnavailable(f"{where} failed ({resp.status_code}): {message}")

    def _read_audio(self, resp: httpx.Response, cancelled: Callable[[], bool] | None) -> Iterator[FloatArray]:
        content_type = resp.headers.get("content-type", "").lower()
        is_wav = content_type.startswith("audio/wav") or content_type.startswith("audio/x-wav")
        if is_wav or not (self.stream or content_type.startswith("audio/pcm")):
            data = resp.read()
            yield from self._split(self._decode_whole(data))
            return
        match = _RATE_RE.search(content_type)
        rate = int(match.group(1)) if match else int(self.spec.get("server_sample_rate") or self.sample_rate)
        float_pcm = "encoding=float" in content_type
        width = 4 if float_pcm else 2
        resampler = StreamResampler(rate, self.sample_rate)
        pending = b""
        min_bytes = int(rate * MIN_CHUNK_S) * width
        for data in resp.iter_bytes():
            pending += data
            if len(pending) < min_bytes:
                continue
            usable = len(pending) - len(pending) % width
            chunk, pending = pending[:usable], pending[usable:]
            out = resampler.process(self._samples(chunk, float_pcm))
            if len(out):
                yield out
            if cancelled is not None and cancelled():
                return
        usable = len(pending) - len(pending) % width
        tail = resampler.process(self._samples(pending[:usable], float_pcm), last=True)
        if len(tail):
            yield tail

    @staticmethod
    def _samples(data: bytes, float_pcm: bool) -> FloatArray:
        if float_pcm:
            return np.frombuffer(data, dtype="<f4").astype(np.float32)
        return _pcm16(data)

    def _decode_whole(self, data: bytes) -> FloatArray:
        if data[:4] == b"RIFF":
            import soundfile as sf

            with sf.SoundFile(io.BytesIO(data)) as f:
                x = f.read(dtype="float32", always_2d=True).mean(axis=1).astype(np.float32)
                rate = int(f.samplerate)
        else:  # raw PCM16 at the model's rate
            x, rate = _pcm16(data), int(self.spec.get("server_sample_rate") or self.sample_rate)
        from bolo.audio.resample import resample

        return resample(x, rate, self.sample_rate)

    def _split(self, x: FloatArray) -> Iterator[FloatArray]:
        step = self.sample_rate  # one second per piece, so encoders and websockets can start early
        for i in range(0, len(x), step):
            yield x[i : i + step]


def gpu_layer_args(device: str, gpu_layers: int | None, extra: list[str]) -> list[str]:
    """``-ngl`` for a llama-server bolo starts: none when the operator set it in the arguments; the scheduler's
    count when it computed one (bolo.placement, from free accelerator memory and the GGUF); else all layers
    (``99``) for any device other than the CPU, as before."""
    from bolo.placement import operator_layers

    if operator_layers(extra) is not None:
        return []
    if gpu_layers is not None:
        return ["-ngl", str(max(0, gpu_layers))]
    if device != "cpu":
        return ["-ngl", "99"]
    return []


class TTSServerEngine(Engine):
    id = ENGINE_ID
    name = "External TTS server (llama-server, audio.cpp)"
    license = "MIT (llama.cpp) / Apache-2.0 (audio.cpp)"
    url = "https://github.com/ggml-org/llama.cpp"
    kinds = ("tts-server", "orpheus-llamacpp")

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self._processes: list[ServerProcess] = []

    def configure(self, settings: Settings) -> None:
        self.settings = settings

    def _settings(self) -> Settings:
        if self.settings is None:
            from bolo.config import Settings

            self.settings = Settings()
        return self.settings

    def status(self) -> EngineStatus:
        # the engine itself is an HTTP client; whether a model's server exists is a per-model question
        # (components llama-server / audio.cpp, or the tts_servers setting)
        return EngineStatus(self.id, True, f"httpx {httpx.__version__}", None)

    def remote(self, spec: dict[str, Any]) -> str | None:
        names = [str(spec.get("model_id") or ""), *[str(a) for a in spec.get("model_aliases") or []]]
        return url_for(self._settings(), [*names, str(spec.get("runtime") or LLAMA)])

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if task != "tts":
            raise EngineUnavailable(f"the tts-server engine serves text-to-speech models, not {task!r}")
        if kind == "orpheus-llamacpp":
            # Orpheus-family models: bolo builds the prompt and decodes SNAC itself (bolo.engines.orpheus_llamacpp)
            from bolo.engines.orpheus_llamacpp import load_model

            return load_model(self, path, spec, opts)
        s = self._settings()
        runtime = str(spec.get("runtime") or LLAMA)
        name = str(spec.get("model_id") or path.name)
        url = self.remote(spec)
        if url:
            model = ServerTTS(
                runtime, name, spec, base_url=url, api_key=s.tts_server_api_key, timeout_s=s.tts_server_timeout_s
            )
        else:
            process = self._process(runtime, name, path, spec, opts)
            model = ServerTTS(runtime, name, spec, process=process, timeout_s=s.tts_server_timeout_s)
        try:
            model.connect()
        except BaseException:
            model.close()
            raise
        return model

    def _process(self, runtime: str, name: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> ServerProcess:
        s = self._settings()
        binary, _source = find_binary(s, runtime)
        info = RUNTIME_INFO[runtime]
        if binary is None:
            raise ComponentMissing(
                f"model '{name}' runs on {info.name}, which is not installed: run `bolo components install "
                f"{runtime}`, set {info.setting} to a binary, or point the setting tts_servers at a running server "
                f'(tts_servers = ["{name}=http://host:port"])'
            )
        port = free_port()
        env = {
            "LD_LIBRARY_PATH": os.pathsep.join(
                p for p in (str(binary.parent), os.environ.get("LD_LIBRARY_PATH", "")) if p
            )
        }
        extra = [str(a) for a in spec.get("server_args") or []] + list(s.tts_server_args)
        key: str | None = None
        if runtime == LLAMA:
            model_file = path / str(spec.get("model_file") or "")
            if not spec.get("model_file") or not model_file.is_file():
                raise EngineUnavailable(f"model '{name}': the GGUF file {model_file} is missing; pull the model again")
            key = secrets.token_hex(16)
            argv = [str(binary), "-m", str(model_file), "--host", "127.0.0.1", "--port", str(port), "--api-key", key]
            if spec.get("mmproj_file"):
                argv += ["-mm", str(path / str(spec["mmproj_file"]))]
            argv += ["-t", str(max(1, opts.threads))]
            argv += gpu_layer_args(opts.device, opts.gpu_layers, extra)
        else:
            config = self._audiocpp_config(name, path, spec, opts, port)
            run_dir = Path(s.data_dir) / "run" / "tts-server"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / f"{name}.json"
            config_path.write_text(json.dumps(config, indent=2))
            argv = [str(binary), "--config", str(config_path), "--no-ui"]
        process = ServerProcess(
            f"{runtime}:{name}", argv + extra, port, start_timeout_s=s.tts_server_start_timeout_s, env=env, api_key=key
        )
        self._processes = [p for p in self._processes if p.running or p.crashes]
        self._processes.append(process)
        return process

    @staticmethod
    def _audiocpp_config(name: str, path: Path, spec: dict[str, Any], opts: LoadOptions, port: int) -> dict[str, Any]:
        model_path = path / str(spec["model_file"]) if spec.get("model_file") else path
        if not model_path.exists():
            raise EngineUnavailable(f"model '{name}': {model_path} is missing; pull the model again")
        entry: dict[str, Any] = {
            "id": str(spec.get("server_model") or name),
            "family": str(spec.get("family") or ""),
            "path": str(model_path),
            "task": "tts",
            "mode": "streaming" if spec.get("stream") else "offline",
        }
        for key in ("load_options", "session_options", "default_request_options"):
            if spec.get(key):
                entry[key] = {str(k): v for k, v in dict(spec[key]).items()}
        backend = "cpu" if opts.device in ("cpu", "", "auto") else opts.device.split(":")[0]
        return {
            "host": "127.0.0.1", "port": port, "backend": backend, "threads": max(1, opts.threads),
            "lazy_load": False, "max_loaded_models": 1, "models": [entry],
        }  # fmt: skip

    def shutdown(self) -> None:
        for process in self._processes:
            process.stop()
        self._processes.clear()

    def processes(self) -> list[dict[str, Any]]:
        return [p.status() for p in self._processes if p.running or p.crashes]
