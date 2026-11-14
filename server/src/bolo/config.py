"""Server configuration.

Precedence, lowest to highest: built-in defaults, a TOML file, environment
variables (``BOLO_*``; a few ``speaches`` names are honoured as fallbacks so
existing deployments keep working), then command-line flags.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib


def default_data_dir() -> Path:
    env = os.environ.get("BOLO_DATA_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "bolo"


def default_config_paths() -> list[Path]:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return [Path("bolo.toml"), base / "bolo" / "config.toml"]


@dataclass
class Settings:
    # -- network ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    cors_origins: list[str] = field(default_factory=list)
    ui: bool = True
    cross_origin_isolation: bool = True  # COOP/COEP on the console so in-browser WebAssembly can use threads
    max_upload_mb: int = 200
    #: who turns text into phonemes for Piper, Kitten, Matcha and Kokoro voices: "auto" (uccharan where it covers
    #: the voice's language, else espeak-ng), "uccharan" (never espeak-ng) or "espeak-ng" (as in bolo 0.1)
    phonemizer: str = "auto"
    # Install optional components (uccharan, espeak-ng) from the API. Unset: allowed on a loopback bind or with an API key.
    allow_component_install: bool | None = None
    #: STUN/TURN URLs offered to WebRTC peers (e.g. "stun:stun.example.org:3478"). Empty: none is contacted,
    #: which is enough when the browser and the server can reach each other directly.
    webrtc_ice_servers: list[str] = field(default_factory=list)

    # -- storage and downloads -------------------------------------------
    data_dir: Path = field(default_factory=default_data_dir)
    auto_pull: bool = True
    hf_token: str | None = None
    hf_endpoint: str = "https://huggingface.co"

    # -- default models ---------------------------------------------------
    default_stt: str = "whisper-base"
    default_realtime_stt: str = "moonshine-tiny-en"
    default_tts: str = "kokoro-v1.0-int8"
    #: voice for latency-sensitive speech: voice agents (Realtime API, chat with audio), Wyoming, the "realtime"
    #: and "tts-1" model names. Piper "amy" starts speaking in about 0.15 s on CPU; Kokoro stays the default
    #: for /v1/audio/speech without a model and for "tts-1-hd".
    default_realtime_tts: str = "piper-en-us-amy-low"
    default_vad: str = "silero-vad"
    default_diarization: str = "pyannote-3.0-eres2net"
    #: after clustering, a "speaker" with less speech than this (seconds) is folded into the most similar one,
    #: and two speakers whose voice prints are at least this similar are merged; 0 switches a repair off
    diarization_min_speaker_s: float = 3.0
    diarization_merge_similarity: float = 0.6
    #: target-speaker extraction model used by ``separate_overlaps`` (opt-in per request)
    default_separation: str = "voxlock-int8"
    #: seconds of audio around each overlap that are separated and transcribed again
    separation_context_s: float = 2.0
    #: overlaps shorter than this (seconds) are left as they are
    separation_min_overlap_s: float = 0.5
    #: at most this much of a speaker's clean (non-overlapped) speech is used as the voice sample
    separation_sample_max_s: float = 10.0
    #: give up separating after this many seconds per request; what was not separated is transcribed as before
    separation_timeout_s: float = 300.0
    #: end-of-turn model used by Realtime sessions that ask for turn_detection.type = "semantic_vad"
    default_turn: str = "smart-turn-v3"
    preload: list[str] = field(default_factory=list)

    # -- lifecycle --------------------------------------------------------
    stt_ttl_s: float = 300.0
    tts_ttl_s: float = 300.0
    vad_ttl_s: float = -1.0  # -1: never unload
    diarization_ttl_s: float = 300.0

    # -- language model for voice agents (any OpenAI-compatible chat-completions endpoint) --
    llm_base_url: str | None = None  # e.g. http://127.0.0.1:11434/v1 (Ollama)
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float = 120.0
    llm_instructions: str = "You are a helpful voice assistant. Answer briefly, in plain spoken sentences."
    memory_budget_mb: int = 0  # 0: no budget
    max_loaded_models: int = 0  # 0: unlimited
    probe_models: bool = True  # load a newly installed model once in a child process first

    # -- scheduler --------------------------------------------------------
    threads: int = 0  # intra-op threads per worker; 0: derive from the CPU count
    workers: int = 1  # worker threads per model
    max_batch: int = 8
    batch_wait_ms: float = 0.0
    max_queue: int = 64
    queue_timeout_s: float = 30.0

    # -- realtime ---------------------------------------------------------
    partial_interval_ms: float = 350.0
    endpoint_silence_ms: float = 500.0
    max_utterance_s: float = 20.0
    agreement: int = 2

    #: what answers in Realtime conversation sessions (a session can pick another with the bolo extension
    #: session.pipeline): "cascade" (bolo's STT -> the language model -> bolo's TTS), "audio-llm" (the user's audio
    #: goes to an audio-understanding language model, bolo's TTS speaks the answer) or "speech-to-speech" (a
    #: speech-to-speech model hears and answers; bolo keeps turn detection, barge-in, codecs and transcripts)
    realtime_pipeline: str = "cascade"
    #: audio-llm: an OpenAI-compatible chat-completions endpoint whose model takes input_audio (llama-server with
    #: Gemma 4 E2B/E4B or Ultravox); unset: llm_base_url / llm_model / llm_api_key
    audio_llm_base_url: str | None = None
    audio_llm_model: str | None = None
    audio_llm_api_key: str | None = None
    #: audio-llm and speech-to-speech: also transcribe the user's turns with bolo's STT (transcript events and
    #: conversation history), in parallel with the response
    realtime_transcribe_user: bool = True
    #: speech-to-speech: the catalog model bolo serves with a llama-server it starts (needs llama-server), unless
    #: s2s_server_url names a llama-server you run (its WebSocket /v1/realtime)
    s2s_model: str = "lfm3-audio-2b-llamacpp"
    s2s_server_url: str | None = None
    s2s_server_api_key: str | None = None  # sent as "Authorization: Bearer ..." to s2s_server_url
    s2s_parallel: int = 2  # sessions the managed server holds at once (llama-server --parallel)
    s2s_server_args: list[str] = field(default_factory=list)  # extra arguments for the managed server

    # -- other protocols --------------------------------------------------
    wyoming: bool = False
    wyoming_host: str = "127.0.0.1"
    wyoming_port: int = 10300

    # -- engine plugins (out of process; docs/plugins.md) -----------------
    #: "name=command args" per plugin, e.g. "torch=bolo-torch-plugin"; installed plugin packages are found too
    plugins: list[str] = field(default_factory=list)
    plugin_entry_points: bool = True  # discover plugins installed as packages (entry-point group bolo.engines)
    plugin_idle_s: float = 300.0  # stop a plugin process when none of its models has been loaded for this long
    plugin_memory_limit_mb: int = 0  # address-space limit of a plugin process (POSIX); 0: none
    plugin_nice: int = 0  # scheduling niceness added to plugin processes

    # -- external TTS servers (engine tts-server: llama-server, audio.cpp; docs/models.md) ---------
    #: "<model id or runtime>=<url>" per operator-run server, e.g. "llama-server=http://127.0.0.1:8080" or
    #: "voxcpm2-audiocpp=http://gpu-box:8080". Models without one are served by a server bolo starts itself.
    tts_servers: list[str] = field(default_factory=list)
    tts_server_api_key: str | None = None  # sent as "Authorization: Bearer …" to operator-run servers
    llama_server_bin: str | None = None  # llama-server binary for managed models (else bolo's component, then PATH)
    audiocpp_server_bin: str | None = None  # audiocpp_server binary for managed models
    tts_server_args: list[str] = field(default_factory=list)  # extra command-line arguments for managed servers
    tts_server_start_timeout_s: float = 300.0  # how long a managed server may take to load its model
    tts_server_timeout_s: float = 300.0  # longest wait for the next bytes of a synthesis response

    # -- misc ---------------------------------------------------------------
    device: str = "cpu"
    log_level: str = "info"
    trace_buffer: int = 1000
    fake_engine: bool = False

    def resolved_threads(self) -> int:
        if self.threads > 0:
            return self.threads
        cpus = os.cpu_count() or 1
        return max(1, min(4, cpus) // max(1, self.workers))

    def public_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in {
                "api_key",
                "hf_token",
                "llm_api_key",
                "tts_server_api_key",
                "audio_llm_api_key",
                "s2s_server_api_key",
            }:
                v = "***" if v else None
            elif isinstance(v, Path):
                v = str(v)
            out[f.name] = v
        return out


# speaches environment names accepted as fallbacks (see docs/migration-from-speaches.md).
_SPEACHES_ENV = {
    "api_key": "API_KEY",
    "preload": "PRELOAD_MODELS",
    "stt_ttl_s": "STT_MODEL_TTL",
    "tts_ttl_s": "TTS_MODEL_TTL",
    "vad_ttl_s": "VAD_MODEL_TTL",
    "cors_origins": "ALLOW_ORIGINS",
    "host": "UVICORN_HOST",
    "port": "UVICORN_PORT",
    "log_level": "LOG_LEVEL",
    "ui": "ENABLE_UI",
    "threads": "WHISPER__CPU_THREADS",
    "workers": "WHISPER__NUM_WORKERS",
    "hf_token": "HF_TOKEN",
    "llm_base_url": "CHAT_COMPLETION_BASE_URL",
    "llm_api_key": "CHAT_COMPLETION_API_KEY",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _coerce(name: str, raw: Any, target: Any) -> Any:
    """Convert a TOML/env/flag value to the declared field type."""
    origin = getattr(target, "__origin__", None)
    if target is bool:
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"{name}: expected a boolean, got {raw!r}")
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    if target is Path:
        return Path(str(raw)).expanduser()
    if origin is list:
        if isinstance(raw, (list, tuple)):
            return [str(x) for x in raw]
        s = str(raw).strip()
        if s.startswith("["):
            import json

            return [str(x) for x in json.loads(s)]
        return [p.strip() for p in s.split(",") if p.strip()]
    if raw is None:
        return None
    return str(raw)


def _field_types() -> dict[str, Any]:
    hints = get_type_hints(Settings)
    out: dict[str, Any] = {}
    for name, hint in hints.items():
        args = getattr(hint, "__args__", ())
        if args and type(None) in args and getattr(hint, "__origin__", None) is not list:
            hint = next(a for a in args if a is not type(None))
        out[name] = hint
    return out


def load_settings(
    config_file: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> Settings:
    environ = dict(os.environ if env is None else env)
    types = _field_types()
    values: dict[str, Any] = {}

    # 1. TOML file
    path: Path | None = None
    explicit = config_file or environ.get("BOLO_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
    else:
        path = next((p for p in default_config_paths() if p.is_file()), None)
    if path is not None:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
        flat: dict[str, Any] = {}
        for key, val in doc.items():
            if isinstance(val, dict):  # allow [server] / [models] style sections
                for sub_key, sub_val in val.items():
                    # `[llm] base_url = …` is the same as `llm_base_url = …`
                    prefixed = f"{key}_{sub_key}".replace("-", "_")
                    # (a section name that prefixes a real setting wins: `[llm] api_key` is llm_api_key)
                    flat[prefixed if prefixed in types else sub_key] = sub_val
            else:
                flat[key] = val
        for key, val in flat.items():
            name = key.replace("-", "_")
            if name not in types:
                raise ValueError(f"{path}: unknown setting {key!r}")
            values[name] = _coerce(name, val, types[name])

    # 2. environment (speaches names first so BOLO_* wins)
    for name, envname in _SPEACHES_ENV.items():
        if envname in environ and environ[envname] != "":
            values[name] = _coerce(name, environ[envname], types[name])
    for name in types:
        envname = "BOLO_" + name.upper()
        if envname in environ:
            values[name] = _coerce(name, environ[envname], types[name])

    # 3. explicit overrides (CLI flags)
    for name, val in (overrides or {}).items():
        if val is None:
            continue
        if name not in types:
            raise ValueError(f"unknown setting {name!r}")
        values[name] = _coerce(name, val, types[name])

    settings = Settings(**values)
    if settings.workers < 1:
        raise ValueError("workers must be >= 1")
    if settings.max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    return settings
