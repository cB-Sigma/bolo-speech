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
    #: romanised text (an Indian language typed in Latin letters) for text-to-speech: "off", "auto" (written in the
    #: voice's script when the voice speaks an Indian language and the text is mostly Latin) or "on". Needs
    #: uccharan's transliterator and the language's pack; without them text is passed on unchanged with a notice
    romanised: str = "off"
    #: download a uccharan language pack (Apache-2.0: a language's word table, British English, Kokoro's English
    #: dictionaries) the first time a voice needs it. false: such a voice fails with a message that names the pack
    #: and the command that fetches it. The CC BY-SA 4.0 packs are never downloaded automatically.
    uccharan_auto_download: bool = True
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
    #: diarization model whose speaker embeddings score cloned voices (speaker similarity); unset: default_diarization
    voice_similarity_model: str | None = None
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
    #: how long a model stays loaded after its last use (Ollama's keep_alive): seconds or a duration ("5m", "1h");
    #: 0 unloads after each request, a negative value never unloads. The per-task TTLs below override it.
    keep_alive: str = "5m"
    stt_ttl_s: float | None = None  # unset: keep_alive
    tts_ttl_s: float | None = None  # unset: keep_alive
    vad_ttl_s: float | None = -1.0  # -1: never unload
    diarization_ttl_s: float | None = None  # unset: keep_alive (also separation models)

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
    #: requests one model works on at once (Ollama's OLLAMA_NUM_PARALLEL); 0: ``workers``. Per model: the catalog's
    #: ``options.num_parallel`` or a pin. A llama-server bolo starts gets ``--parallel`` with it
    num_parallel: int = 0
    max_batch: int = 8
    batch_wait_ms: float = 0.0
    #: requests waiting per model before 429 (Ollama's OLLAMA_MAX_QUEUE, which counts the whole server)
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
    #: s2s_server_url names a llama-server you run (its WebSocket /v1/realtime). "lfm3-audio-2b-onnx" runs the same
    #: model in this process with ONNX Runtime (bolo-speech[s2s]; no llama-server); "<provider>:<model>" is a cloud model
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
    tontaube_dir: str | None = None  # a checkout of github.com/craitech/tontaube (else bolo's component)
    #: how long a managed Tontaube server may take to start: its first start installs PyTorch and more
    tontaube_start_timeout_s: float = 3600.0
    tts_server_args: list[str] = field(default_factory=list)  # extra command-line arguments for managed servers
    tts_server_start_timeout_s: float = 300.0  # how long a managed server may take to load its model
    tts_server_timeout_s: float = 300.0  # longest wait for the next bytes of a synthesis response

    # -- cloud providers (engine remote; docs/providers.md) --------------------------------------
    #: name -> {kind, base_url, api_key, options, prices}. kind: openai-compatible, sarvam, elevenlabs or deepgram.
    #: In a file: a [providers.<name>] table per provider. In the environment: BOLO_PROVIDERS (JSON), or
    #: BOLO_PROVIDER_<NAME>_KIND / _BASE_URL / _API_KEY. Keys are never logged or returned by any API.
    providers: dict[str, Any] = field(default_factory=dict)
    #: "<model>=<fallback>[,<fallback>...]" per model: what answers instead when the model's engine is
    #: unavailable, overloaded, or its provider fails (5xx, timeout). Local and cloud models can be mixed.
    fallbacks: list[str] = field(default_factory=list)
    #: add, change and remove providers through the API (stored in <data_dir>/providers.json).
    #: Unset: allowed on a loopback bind or with an API key.
    allow_provider_edit: bool | None = None
    provider_timeout_s: float = 60.0  # longest wait for a provider's answer (or its next bytes while streaming)

    # -- phone calls (docs/telephony.md) ---------------------------------------------------------
    #: public_base_url, validate_signatures, default_agent, call_buffer, and the tables providers.<twilio|plivo|
    #: telnyx|exotel> (credentials), agents.<name> (voice agents) and numbers (called number = agent). In a file:
    #: a [telephony] table. In the environment: BOLO_TELEPHONY (JSON), BOLO_TELEPHONY_PUBLIC_BASE_URL,
    #: BOLO_TELEPHONY_<PROVIDER>_<FIELD>. Credentials are never logged or returned by any API.
    telephony: dict[str, Any] = field(default_factory=dict)
    #: inbound SIP (docs/telephony.md): listen for INVITEs from a trunk or PBX on sip_host:sip_port (UDP)
    sip: bool = False
    sip_host: str = "0.0.0.0"
    sip_port: int = 5060
    #: the address written into SDP answers (the one the trunk sends RTP to); unset: the address the INVITE reached
    sip_public_ip: str | None = None
    #: source addresses allowed to send INVITEs ("203.0.113.10", "10.0.0.0/8"); empty: loopback only
    sip_allowed_ips: list[str] = field(default_factory=list)
    sip_rtp_port_min: int = 40000
    sip_rtp_port_max: int = 40100

    # -- llama.cpp servers bolo starts (tts-server voices, Orpheus-family voices, the speech-to-speech server) --
    #: flash attention (``-fa``): "on", "off" or "auto"; unset: llama-server's default (auto)
    llama_flash_attention: str = ""
    #: K and V cache type (``--cache-type-k/--cache-type-v``, Ollama's OLLAMA_KV_CACHE_TYPE): f16, q8_0, q4_0, ...;
    #: unset: llama-server's default (f16)
    llama_kv_cache_type: str = ""
    llama_context_size: int = 0  # ``-c``; 0: llama-server's default (the model's own)
    llama_batch_size: int = 0  # ``-b``; 0: llama-server's default

    # -- misc ---------------------------------------------------------------
    device: str = "cpu"
    #: the GPUs bolo may use (CUDA device numbers as this process sees them, after CUDA_VISIBLE_DEVICES); empty: all
    gpus: list[int] = field(default_factory=list)
    #: spread each llama.cpp model over every allowed GPU (Ollama's OLLAMA_SCHED_SPREAD) instead of placing it on
    #: the one GPU with the most free memory when it fits there
    gpu_spread: bool = False
    #: accelerator memory left free when placing models on the GPU (for the display, other programs, spikes)
    gpu_reserve_mb: int = 512
    #: accelerator memory bolo's models may use in all; 0: as much as is free. Over it, the least recently used
    #: idle GPU-resident model is unloaded; when none can be, the new model is placed on the CPU
    gpu_memory_budget_mb: int = 0
    #: "auto": an ONNX Runtime graph that does not fit in free accelerator memory is cut in two at a layer boundary,
    #: the front on the accelerator, the back on the CPU (needs the onnx package); "off": it runs on the CPU
    onnx_split: str = "auto"
    #: attach a notice (API responses, traces, the command line, the console) when a model that is pulled or
    #: loaded is unlikely to fit this machine or to run in real time. Notices only: nothing is ever refused.
    hardware_warnings: bool = True
    log_level: str = "info"
    #: also write the server's log to a file: "auto" (the platform's log directory, docs/configuration.md "Logs"),
    #: a path, or "off". Unset: off, except when `bolo serve` runs as a service (systemd, launchd)
    log_file: str = ""
    log_max_mb: int = 10  # rotate the log file at this size
    log_backups: int = 3  # rotated files kept (server.log.1, ...)
    trace_buffer: int = 1000
    fake_engine: bool = False

    def resolved_threads(self) -> int:
        if self.threads > 0:
            return self.threads
        cpus = os.cpu_count() or 1
        return max(1, min(4, cpus) // max(1, self.parallel()))

    def parallel(self) -> int:
        """Requests per model at once by default: ``num_parallel``, else ``workers``."""
        return self.num_parallel if self.num_parallel > 0 else max(1, self.workers)

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
            elif f.name == "providers":
                from bolo.providers import redact_provider

                v = {str(name): redact_provider(entry) for name, entry in dict(v).items()}
            elif f.name == "telephony":
                from bolo.telephony.config import redact_telephony

                v = redact_telephony(v)
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
    if origin is dict:
        if isinstance(raw, dict):
            return dict(raw)
        import json

        try:
            doc = json.loads(str(raw))
        except ValueError as exc:
            raise ValueError(f"{name}: expected a JSON object") from exc
        if not isinstance(doc, dict):
            raise ValueError(f"{name}: expected a JSON object")
        return doc
    if origin is list:
        item: Any = int if getattr(target, "__args__", (str,))[0] is int else str
        if isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            s = str(raw).strip()
            if s.startswith("["):
                import json

                items = list(json.loads(s))
            else:
                items = [p.strip() for p in s.split(",") if p.strip()]
        try:
            return [item(x) for x in items]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: expected a list of numbers, got {raw!r}") from exc
    if raw is None:
        return None
    return str(raw)


def _check_cloud(settings: Settings) -> None:
    """Fail at start-up, with the setting's name, for a provider or fallback entry that cannot be read."""
    from bolo.errors import BoloError
    from bolo.providers import ProviderConfig
    from bolo.text.romanised import MODES, parse_mode

    try:
        settings.romanised = parse_mode(settings.romanised) or "off"
    except BoloError as exc:
        raise ValueError(f"romanised: expected one of {', '.join(MODES)}, got {settings.romanised!r}") from exc

    for name, entry in settings.providers.items():
        try:
            ProviderConfig.from_dict(str(name), entry)
        except BoloError as exc:
            raise ValueError(f"providers: {exc.message}") from exc
    if settings.telephony:
        from bolo.telephony.config import TelephonyConfig

        try:
            TelephonyConfig(settings)
        except BoloError as exc:
            raise ValueError(
                exc.message if exc.message.startswith("telephony") else f"telephony: {exc.message}"
            ) from exc
    for item in settings.fallbacks:
        model, sep, rest = str(item).partition("=")
        if not sep or not model.strip() or not rest.strip():
            raise ValueError(f"fallbacks: expected '<model>=<fallback>[,<fallback>...]', got {item!r}")


_PROVIDER_ENV_FIELDS = ("API_KEY", "BASE_URL", "KIND")


def _provider_env(values: dict[str, Any], environ: dict[str, str]) -> None:
    """``BOLO_PROVIDER_<NAME>_API_KEY`` / ``_BASE_URL`` / ``_KIND`` add to (or override) the ``providers`` table."""
    prefix = "BOLO_PROVIDER_"
    found: dict[str, dict[str, str]] = {}
    for envname, raw in environ.items():
        if not envname.startswith(prefix) or raw == "":
            continue
        rest = envname[len(prefix) :]
        for suffix in _PROVIDER_ENV_FIELDS:
            if rest.endswith("_" + suffix) and len(rest) > len(suffix) + 1:
                name = rest[: -len(suffix) - 1].lower().replace("_", "-")
                found.setdefault(name, {})[suffix.lower()] = raw
                break
    if not found:
        return
    providers = {str(k): dict(v) for k, v in dict(values.get("providers") or {}).items()}
    for name, entry in found.items():
        providers[name] = {**providers.get(name, {}), **entry}
    values["providers"] = providers


def _field_types() -> dict[str, Any]:
    hints = get_type_hints(Settings)
    out: dict[str, Any] = {}
    for name, hint in hints.items():
        args = getattr(hint, "__args__", ())
        if args and type(None) in args and getattr(hint, "__origin__", None) not in (list, dict):
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
            if key in ("providers", "telephony") and isinstance(val, dict):  # these tables stay nested
                flat[key] = val
            elif isinstance(val, dict):  # allow [server] / [models] style sections
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

    _provider_env(values, environ)
    from bolo.telephony.config import telephony_env

    telephony_env(values, environ)

    # 3. explicit overrides (CLI flags)
    for name, val in (overrides or {}).items():
        if val is None:
            continue
        if name not in types:
            raise ValueError(f"unknown setting {name!r}")
        values[name] = _coerce(name, val, types[name])

    settings = Settings(**values)
    _check_cloud(settings)
    from bolo.keepalive import parse_keep_alive

    if str(settings.onnx_split).lower() not in ("auto", "off"):
        raise ValueError(f"onnx_split: expected auto or off, got {settings.onnx_split!r}")
    try:
        parse_keep_alive(settings.keep_alive)
    except ValueError as exc:
        raise ValueError(f"keep_alive: {exc}") from exc
    if settings.workers < 1:
        raise ValueError("workers must be >= 1")
    if settings.num_parallel < 0:
        raise ValueError("num_parallel must be >= 0")
    if settings.max_queue < 0:
        raise ValueError("max_queue must be >= 0")
    if any(i < 0 for i in settings.gpus):
        raise ValueError(f"gpus: device numbers start at 0, got {settings.gpus}")
    from bolo.llama_args import check_tuning

    check_tuning(
        {
            "flash_attention": settings.llama_flash_attention,
            "kv_cache_type": settings.llama_kv_cache_type,
            "context_size": settings.llama_context_size,
            "batch_size": settings.llama_batch_size,
        },
        prefix="llama_",
    )
    if settings.log_max_mb < 1 or settings.log_backups < 0:
        raise ValueError("log_max_mb must be >= 1 and log_backups >= 0")
    if settings.max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    return settings
