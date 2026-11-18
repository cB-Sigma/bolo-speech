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

The runtime ``tontaube`` (TontaubeV1.1) is a Python server: ``POST /predict`` answers base64 audio in JSON. Its
request options (``SynthesisExtras.options``): ``style`` (audiobook, conversational, agentic; else the voice's
style), the sliders ``rate`` (0-4), ``pause`` (0-4), ``pitch_range`` (0-5), ``pitch_level`` (0-4), ``legato``
(0-2), and ``temperature`` and ``seed``. Catalog voices name a bundled reference (``server_voice``, sent as
``voice_path``) and its ``style``.

The kind ``orpheus-llamacpp`` (Orpheus, Veena, Svara, snorTTS) uses the same llama-server management but its
own client, :mod:`bolo.engines.orpheus_llamacpp`: bolo builds the prompt, streams token ids from ``/completion``
and decodes the SNAC codes with onnxruntime. The kind ``voxtral-llamacpp`` (Voxtral speech-to-text) does the same
with :mod:`bolo.engines.voxtral_llamacpp`: audio in, text out through ``/completion``.

Other per-entry options: ``cloning`` (the model takes a reference recording), ``reference_text`` (false: never send
the reference's transcript, for audio.cpp families that refuse it), ``send_language`` (tell the server
the language of the text), ``language_option`` (audio.cpp: also as the model option ``language``),
``language_names`` (send English language names instead of codes), ``language_map``, ``speed`` (the
server honours ``speed``), ``stream`` (ask for streamed audio), ``request`` (extra request fields), and voices
with ``reference: <bolo sample id>``: bolo's synthetic reference clips, sent as the voice to imitate.

``instructions`` (audio.cpp): the model option a request's ``instructions`` go to (``instruct`` for most families),
or ``{option, with}`` where ``with`` are further options sent alongside (IndexTTS-2: ``emotion_text`` with
``use_emotion_text: true``); ``default_instructions`` is sent when a request has none (Maya1 needs one).
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

from bolo import gpu_health, procs
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
from bolo.llama_args import has_flag
from bolo.tts_runtimes import (
    AUDIOCPP,
    CPU_FLAVOUR,
    ENGINE_ID,
    LLAMA,
    RUNTIME_INFO,
    RUNTIMES,
    TONTAUBE,
    find_binary,
    flavour_of,
    url_for,
)

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
#: Tontaube: its styles, its five sliders (highest step of each) and the languages it reads
TONTAUBE_STYLES = ("audiobook", "conversational", "agentic")
TONTAUBE_SLIDERS = {"rate": 4, "pause": 4, "pitch_range": 5, "pitch_level": 4, "legato": 2}
TONTAUBE_LANGUAGES = {"en", "de", "es", "fr", "it", "nl", "pt"}
#: shortest piece of audio handed on while streaming (seconds); smaller network reads are joined
MIN_CHUNK_S = 0.08


def server_env(binary: Path, flavour: str | None, *, gpu_lost: bool = False) -> dict[str, str]:
    """The variables a server bolo starts gets on top of bolo's environment: its folder on ``LD_LIBRARY_PATH``
    (the shared libraries next to it), what its build needs here (:func:`bolo.tts_runtimes.child_env`: no AMD
    switchable-graphics layer for a Vulkan build on a Windows machine with AMD graphics), and after the GPU was
    lost every CUDA GPU hidden (``CUDA_VISIBLE_DEVICES=-1``), so that a GPU build bolo has to fall back on runs on
    the CPU instead of touching the lost GPU."""
    from bolo.tts_runtimes import child_env

    env = {
        "LD_LIBRARY_PATH": os.pathsep.join(p for p in (str(binary.parent), os.environ.get("LD_LIBRARY_PATH", "")) if p)
    }
    added = child_env(flavour)
    if added:
        log.info(
            "starting the %s build of %s with %s", flavour, binary.name, ", ".join(f"{k}={v}" for k, v in added.items())
        )
    env.update(added)
    if gpu_lost:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


#: bolo ``device`` families and the audio.cpp backend each one means (its server knows cpu, cuda, hip, vulkan and
#: metal); anything else (dml, openvino) runs audio.cpp on the CPU
AUDIOCPP_BACKENDS = {
    "cuda": "cuda", "gpu": "cuda", "rocm": "hip", "hip": "hip", "vulkan": "vulkan",
    "metal": "metal", "coreml": "metal", "mps": "metal",
}  # fmt: skip


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
        gpu: bool = False,
        flavour: str | None = None,
        tls: bool = False,
        ready_any_status: bool = False,
    ) -> None:
        self.name = name
        #: it serves HTTPS with a certificate of its own on the loopback address (Kyutai's Rust server)
        self.tls = tls
        #: ready as soon as it answers HTTP at all (servers without a health endpoint, e.g. Kyutai's)
        self.ready_any_status = ready_any_status
        #: started for a GPU: a CUDA or Vulkan error on its console means the GPU stopped working (bolo.gpu_health)
        self.gpu = gpu
        #: the build (cpu, cuda, vulkan), for the advice when it does not start
        self.flavour = flavour
        #: the first GPU error line it printed since it started
        self.gpu_error: str | None = None
        self.api_key = api_key
        self.argv = argv
        self.port = port
        self.base_url = f"{'https' if tls else 'http'}://127.0.0.1:{port}"
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
                if self.gpu and self.gpu_error is None and gpu_health.gpu_error_line(line):
                    self.gpu_error = line
                    gpu_health.report_gpu_error(self.name, line)

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
            self.gpu_error = None
            env = {**os.environ, **self.env}
            try:
                proc = procs.spawn(
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
        # verify=False only for a server bolo started itself on the loopback address with its own certificate
        with httpx.Client(timeout=2.0, trust_env=False, verify=not self.tls) as client:
            while True:
                code = proc.poll()
                if code is not None:
                    killed = " (killed: often the system running out of memory)" if code == -9 else ""
                    reason = f"exit code {code}{killed} while starting" + (f": {self.tail()}" if self._tail else "")
                    self._crashed(reason)
                    raise EngineUnavailable(f"TTS server '{self.name}' did not start ({reason})")
                with contextlib.suppress(httpx.HTTPError):
                    status = client.get(url).status_code
                    if status == 200 or (self.ready_any_status and status < 500):
                        return
                if time.monotonic() > deadline:
                    reason = f"not ready after {self.start_timeout_s:.0f} s" + (
                        f": {self.tail()}" if self._tail else ""
                    )
                    if self.flavour == "vulkan":
                        reason += (
                            " (a Vulkan build can hang while it lists the GPUs: on a laptop with AMD and NVIDIA "
                            'graphics see docs/configuration.md, "Windows"; or use the CPU or CUDA build)'
                        )
                    self._crashed(reason)
                    raise EngineUnavailable(f"TTS server '{self.name}' did not start ({reason})")
                time.sleep(0.2)

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes]) -> None:
        """Ask the server to stop (SIGTERM; CTRL_BREAK_EVENT on Windows), then kill it after 10 s."""
        procs.stop(proc, timeout=10.0)

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
            "last_error": self.last_error, "flavour": self.flavour, "gpu_error": self.gpu_error,
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
        wanted = spec.get("instructions") if runtime == AUDIOCPP else None
        self._instruction_option = str(wanted.get("option") if isinstance(wanted, dict) else wanted or "")
        self._instruction_with = dict(wanted.get("with") or {}) if isinstance(wanted, dict) else {}
        self.supports_instructions = bool(self._instruction_option)
        self.stream = bool(spec.get("stream", runtime == LLAMA)) and runtime != TONTAUBE
        if runtime == TONTAUBE:
            self.request_options = ("style", *TONTAUBE_SLIDERS, "temperature", "seed")
        elif runtime != LLAMA:
            self.request_options = ("*",)  # audio.cpp: model options, passed through
        api_key = api_key or (process.api_key if process is not None else None)
        if runtime == TONTAUBE:
            headers = {"x-api-key": api_key} if api_key else {}
        else:
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
            resp = self._client.get(
                self.base_url + ("/readyz" if self.runtime == TONTAUBE else "/health"), timeout=10.0
            )
        except httpx.HTTPError as exc:
            raise EngineUnavailable(self._unreachable(exc)) from exc
        if resp.status_code == 503:
            raise EngineUnavailable(f"the TTS server at {self.base_url} is still loading its model; try again shortly")
        if resp.status_code >= 400:
            raise EngineUnavailable(f"the TTS server at {self.base_url} answered /health with {resp.status_code}")
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError, TypeError):
            self._add_server_voices(self._list_server_voices())

    def _list_server_voices(self) -> list[str]:
        if self.runtime == TONTAUBE:
            return []  # its bundled voices are files on the server; the catalog lists them
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
    def _check_voice(self, voice: str) -> None:
        """audio.cpp takes a voice it does not know for the path of a reference file on its own machine and answers
        500 when there is none: a voice that is neither the catalog's nor one the server listed is refused here,
        with the voices there are."""
        if voice.lower() in self._server_voices or voice.lower() in self._voice_info:
            return
        names = [v.id for v in self.voices]
        listed = f"its voices: {', '.join(names)}" if names else "it has no named voices (leave voice out)"
        raise InvalidRequest(f"model '{self.name}' has no voice {voice!r}; {listed}", param="voice")

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

    def _tontaube_request(
        self, text: str, voice: str | None, speed: float, extras: SynthesisExtras, ref_audio: bytes | None
    ) -> dict[str, Any]:
        info = self._voice_info.get((voice or "").lower(), {})
        options = dict(extras.options)
        language = _language_code(extras.language) or _language_code(str(info.get("language") or "")) or "en"
        if language not in TONTAUBE_LANGUAGES:
            raise InvalidRequest(
                f"model '{self.name}' speaks {', '.join(sorted(TONTAUBE_LANGUAGES))}, not {language!r}",
                param="language",
            )
        style = str(options.pop("style", None) or info.get("style") or self.spec.get("style") or "conversational")
        if style not in TONTAUBE_STYLES:
            raise InvalidRequest(f"style must be one of {', '.join(TONTAUBE_STYLES)}", param="options")
        if speed != 1.0:
            raise InvalidRequest(
                f"model '{self.name}' has no speed setting; use the option rate (0 much slower .. 4 much faster)",
                param="speed",
            )
        body: dict[str, Any] = {"text": text, "language": language, "tag": style, "format": "wav"}
        controls: dict[str, int] = {}
        for name, top in TONTAUBE_SLIDERS.items():
            if name in options:
                value = options.pop(name)
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= top:
                    raise InvalidRequest(f"option {name} is a step from 0 to {top}", param="options")
                controls[name] = value
        if controls:
            body["controls"] = controls
        if "temperature" in options:
            body["temperature"] = float(options.pop("temperature"))
        if "seed" in options:
            body["seed"] = int(options.pop("seed"))
        if ref_audio is not None:
            body["voice_audio_b64"] = base64.b64encode(ref_audio).decode()
        elif info.get("server_voice"):
            body["voice_path"] = str(info["server_voice"])
        body.update(dict(self.spec.get("request") or {}))
        return body

    def build_request(
        self, text: str, voice: str | None, speed: float, extras: SynthesisExtras | None
    ) -> dict[str, Any]:
        extras = extras or SynthesisExtras()
        if self.runtime == TONTAUBE:
            if extras.reference_audio is not None and not self.supports_cloning:
                raise InvalidRequest(f"model '{self.name}' cannot imitate a voice from reference audio", param="voice")
            return self._tontaube_request(text, voice, speed, extras, extras.reference_audio)
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
            elif not info and send_voice and self.runtime == AUDIOCPP:
                self._check_voice(send_voice)
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
            if ref_text and self.spec.get("reference_text", True):  # false: the family refuses the field (XTTS v2)
                body["reference_text"] = ref_text
        elif send_voice:
            body["voice"] = send_voice
        if speed != 1.0:
            body["speed"] = speed
        if language and self.spec.get("language_option"):
            # families that declare a "language" request option read it there, not from the top-level field
            extra_fields.setdefault("language", language)
        direction = extras.instructions or self.spec.get("default_instructions")
        if self._instruction_option and direction:
            extra_fields[self._instruction_option] = str(direction)
            extra_fields.update(self._instruction_with)
        extra_fields.update(extras.options)
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
        if self.runtime == TONTAUBE:
            yield from self._predict(body)
            return
        yield from self._post_speech(body, cancelled)

    def _post_speech(self, body: dict[str, Any], cancelled: Callable[[], bool] | None) -> Iterator[FloatArray]:
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

    def _predict(self, body: dict[str, Any]) -> Iterator[FloatArray]:
        """Tontaube: one POST /predict, base64 audio in JSON."""
        try:
            resp = self._client.post(self.base_url + "/predict", json=body)
        except httpx.HTTPError as exc:
            if self.process is not None:
                self.process.check()
            raise EngineUnavailable(self._unreachable(exc)) from exc
        if resp.status_code == 503:
            raise EngineUnavailable(f"the Tontaube server at {self.base_url} is still loading its model")
        if resp.status_code >= 400:
            self._raise_for(resp)
        try:
            data = base64.b64decode(resp.json()["audio_b64"])
        except (ValueError, KeyError, TypeError) as exc:
            raise EngineUnavailable(f"the Tontaube server at {self.base_url} answered without audio") from exc
        yield from self._split(self._decode_whole(data))

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
    (``99``) for any device other than the CPU, as before. :func:`bolo.llama_args.gpu_args` adds the GPU choice."""
    from bolo.llama_args import gpu_args

    return gpu_args(device, gpu_layers, extra)


def llama_server_args(
    settings: Settings, spec: dict[str, Any], opts: LoadOptions, extra: list[str], parallel: int | None = None
) -> list[str]:
    # (``parallel`` given: the speech-to-speech server, whose slot count is always passed)
    """What bolo adds to a managed llama-server's command line besides the model and the address: threads, GPU
    layers and GPUs, parallel slots, and the speed settings (flash attention, KV cache type, context and batch
    size). Anything ``extra`` (the operator's arguments) already sets is left out."""
    from bolo.llama_args import gpu_args, parallel_args, resolve_tuning, tuning_args

    out = ["-t", str(max(1, opts.threads))]
    out += gpu_args(
        opts.device, opts.gpu_layers, extra, gpus=opts.gpus, tensor_split=opts.tensor_split,
        main_gpu=opts.main_gpu, visible=opts.visible_gpus,
    )  # fmt: skip
    out += parallel_args(opts.parallel if parallel is None else parallel, extra, always=parallel is not None)
    out += tuning_args(resolve_tuning(settings, spec), extra)
    return out


class TTSServerEngine(Engine):
    id = ENGINE_ID
    name = "External TTS server (llama-server, audio.cpp)"
    license = "MIT (llama.cpp) / Apache-2.0 (audio.cpp)"
    url = "https://github.com/ggml-org/llama.cpp"
    kinds = ("tts-server", "orpheus-llamacpp", "voxtral-llamacpp", "ten-turn-llamacpp")

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
        if kind == "voxtral-llamacpp":
            # Voxtral speech-to-text: bolo builds the prompts, llama-server runs the model (bolo.engines.voxtral_llamacpp)
            if task != "stt":
                raise EngineUnavailable(f"the voxtral-llamacpp kind is a speech-to-text model, not {task!r}")
            from bolo.engines.voxtral_llamacpp import load_model as load_voxtral

            return load_voxtral(self, path, spec, opts)
        if kind == "ten-turn-llamacpp":
            # TEN Turn Detection: an end-of-turn model that reads text, on llama-server (bolo.engines.ten_turn_llamacpp)
            if task != "turn":
                raise EngineUnavailable(f"the ten-turn-llamacpp kind is an end-of-turn model, not {task!r}")
            from bolo.engines.ten_turn_llamacpp import load_model as load_ten_turn

            return load_ten_turn(self, path, spec, opts)
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
        gpu_lost = bool(opts.options.get("gpu_lost"))
        # after the GPU was lost: the CPU build when it is installed
        binary, _source = find_binary(s, runtime, device="cpu" if gpu_lost else None)
        info = RUNTIME_INFO[runtime]
        if binary is None:
            raise ComponentMissing(
                f"model '{name}' runs on {info.name}, which is not installed: run `bolo components install "
                f"{runtime}`, set {info.setting} to a binary, or point the setting tts_servers at a running server "
                f'(tts_servers = ["{name}=http://host:port"])'
            )
        port = free_port()
        flavour = flavour_of(s, runtime, binary) if runtime != TONTAUBE else None
        if gpu_lost and flavour not in (None, CPU_FLAVOUR):
            log.warning(
                "model '%s': the GPU was lost and the CPU build of %s is not installed (`bolo components install %s "
                "--flavour cpu`); starting the %s build with every GPU hidden",
                name, info.name, runtime, flavour,
            )  # fmt: skip
        env = server_env(binary, flavour, gpu_lost=gpu_lost)
        on_gpu = (opts.device or "cpu").split(":", 1)[0].lower() not in ("cpu", "", "auto")
        extra = [str(a) for a in spec.get("server_args") or []] + list(s.tts_server_args)
        key: str | None = None
        health = "/health"
        timeout = s.tts_server_start_timeout_s
        if runtime == TONTAUBE:
            argv, env, key = self._tontaube_argv(binary, path, spec, opts, port)
            process = ServerProcess(
                f"{runtime}:{name}", argv, port, start_timeout_s=max(timeout, s.tontaube_start_timeout_s), env=env,
                health_path="/readyz", api_key=key,
            )  # fmt: skip
            self._processes = [p for p in self._processes if p.running or p.crashes]
            self._processes.append(process)
            return process
        if runtime == LLAMA:
            model_file = path / str(spec.get("model_file") or "")
            if not spec.get("model_file") or not model_file.is_file():
                raise EngineUnavailable(f"model '{name}': the GGUF file {model_file} is missing; pull the model again")
            key = secrets.token_hex(16)
            argv = [str(binary), "-m", str(model_file), "--host", "127.0.0.1", "--port", str(port), "--api-key", key]
            if spec.get("mmproj_file"):
                argv += ["-mm", str(path / str(spec["mmproj_file"]))]
            argv += llama_server_args(s, spec, opts, extra)
            if gpu_lost and not has_flag(extra, ("-dev", "--device")):
                argv += ["--device", "none"]  # no GPU at all, whatever the build
        else:
            config = self._audiocpp_config(name, path, spec, opts, port)
            run_dir = Path(s.data_dir) / "run" / "tts-server"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / f"{name}.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            argv = [str(binary), "--config", str(config_path), "--no-ui"]
        process = ServerProcess(
            f"{runtime}:{name}", argv + extra, port, start_timeout_s=timeout, env=env, health_path=health, api_key=key,
            gpu=on_gpu, flavour=flavour,
        )  # fmt: skip
        self._processes = [p for p in self._processes if p.running or p.crashes]
        self._processes.append(process)
        return process

    @staticmethod
    def _tontaube_argv(
        checkout: Path, path: Path, spec: dict[str, Any], opts: LoadOptions, port: int
    ) -> tuple[list[str], dict[str, str], str]:
        """``uv run --python 3.12 --frozen tontaube serve`` in the checkout, with the weights bolo downloaded
        (``MODEL_PATH``), a random API key and the device: ``LITE_DEVICE=cpu`` on the CPU, else the GPU."""
        import shutil

        uv = shutil.which("uv")
        if uv is None:
            raise ComponentMissing(
                "the Tontaube server is started with uv, which is not on PATH (https://docs.astral.sh/uv/)"
            )
        if not (path / "cb0").is_dir():
            raise EngineUnavailable(f"{path} has no cb0/ folder; pull the model again")
        key = secrets.token_hex(16)
        env = {
            "MODEL_PATH": str(path),
            "TTS_BACKEND": str(spec.get("backend") or "lite"),
            "REQUIRE_API_KEY": "1",
            "TTS_API_KEY": key,
            "ENABLE_VERBALIZATION": "0",
        }
        device = opts.device or "auto"
        if device == "cpu":
            env["LITE_DEVICE"] = "cpu"
            env["LITE_CPU_THREADS"] = str(max(1, opts.threads))
        elif device.startswith("cuda"):
            env["LITE_DEVICE"] = "cuda"
            if ":" in device:
                env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        for k, v in dict(spec.get("env") or {}).items():
            env[str(k)] = str(v)
        argv = [
            uv, "run", "--python", "3.12", "--frozen", "--directory", str(checkout), "tontaube", "serve",
            "--host", "127.0.0.1", "--port", str(port),
        ]  # fmt: skip
        return argv, env, key

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
        kind = (opts.device or "cpu").split(":")[0].lower()
        backend = AUDIOCPP_BACKENDS.get(kind, "cpu")
        if kind not in AUDIOCPP_BACKENDS and kind not in ("cpu", "", "auto"):
            # audio.cpp refuses to start with a backend it does not know ("unsupported server backend: dml")
            log.warning("model '%s': audio.cpp has no %s backend; it runs on the CPU", name, kind)
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
