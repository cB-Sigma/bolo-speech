# Configuration

Settings come from, lowest to highest priority:

1. built-in defaults;
2. a TOML file: `--config FILE`, else `$BOLO_CONFIG`, else `./bolo.toml`, else
   `~/.config/bolo/config.toml`;
3. environment variables `BOLO_<NAME>` (a few `speaches` names are honoured
   too, see below);
4. command-line flags of `bolo serve`.

Lists can be written as TOML arrays, JSON arrays or comma-separated strings.
Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`.

## Settings

| Setting | Default | Flag | Meaning |
| --- | --- | --- | --- |
| `host` | `127.0.0.1` | `--host` | bind address |
| `port` | `8000` | `--port` | HTTP port |
| `api_key` | none | `--api-key` | if set, required as `Authorization: Bearer <key>` (or `X-API-Key`, or `?api_key=` for WebSockets and SSE) |
| `cors_origins` | `[]` | `--cors-origin` (repeatable) | allowed browser origins; `*` allows any |
| `ui` | `true` | `--no-ui` | serve the web console at `/` |
| `cross_origin_isolation` | `true` | | send COOP/COEP headers with the console so its in-browser WebAssembly recogniser can use threads. API responses are not affected. Turn off if you embed the console in another site's frame |
| `max_upload_mb` | `200` | | largest accepted audio upload |
| `data_dir` | `~/.cache/bolo` (`$XDG_CACHE_HOME/bolo`) | `--data-dir` | models and the blob cache |
| `auto_pull` | `true` | `--no-auto-pull` | download a catalog model the first time it is requested (never for gated models) |
| `hf_token` | none (falls back to `$HF_TOKEN`) | `--hf-token` | Hugging Face token for gated or private repositories |
| `hf_endpoint` | `https://huggingface.co` | | Hub endpoint (for a mirror) |
| `default_stt` | `whisper-base` | `--default-stt` | model behind `whisper-1` and requests that name no model |
| `default_realtime_stt` | `moonshine-tiny-en` | `--default-realtime-stt` | model for the STT WebSocket when none is named (`model=realtime`) |
| `default_tts` | `kokoro-v1.0-int8` | `--default-tts` | model behind `tts-1`, `tts-1-hd` |
| `default_vad` | `silero-vad` | | detector for long-file segmentation and realtime endpointing |
| `default_diarization` | `pyannote-3.0-eres2net` | | speaker-diarization pipeline |
| `diarization_min_speaker_s` | `3` | | after clustering, a "speaker" with less speech than this in the whole recording (and never more than a tenth of the speech) is folded into the most similar speaker; `0` switches it off. See api.md, "How many speakers" |
| `diarization_merge_similarity` | `0.6` | | speakers whose voice prints have at least this cosine similarity are merged; `0` switches it off |
| `default_turn` | `smart-turn-v3` | | end-of-turn model for Realtime sessions that ask for `semantic_vad` (pulled on first use; without it `semantic_vad` is plain silence detection) |
| `webrtc_ice_servers` | `[]` | | STUN/TURN URLs for the WebRTC transport of the Realtime API (`stun:host:port`, `turn:…`). Empty: no external server is contacted and calls connect on the same machine or network only |
| `llm_base_url` | none | `--llm-base-url` | OpenAI-compatible chat-completions endpoint for voice agents (`/v1/realtime` conversations, `/v1/chat/completions`), e.g. `http://127.0.0.1:11434/v1` |
| `llm_model` | none | `--llm-model` | model name sent to that endpoint; unset: the client's model name is passed on |
| `llm_api_key` | none | `--llm-api-key` | bearer token for that endpoint |
| `llm_timeout_s` | `120` | | |
| `llm_instructions` | a short "helpful voice assistant" prompt | | default system prompt of Realtime conversations |
| `allow_component_install` | unset | | whether optional components (eSpeak NG) may be installed through the API. Unset: allowed on a loopback bind or when an API key is set. Never allowed on a non-loopback bind without a key |
| `preload` | `[]` | `--preload` (repeatable) | models to load at startup; `/readyz` waits for them |
| `stt_ttl_s`, `tts_ttl_s` | `300` | | unload after this many idle seconds; `-1` never, `0` at once |
| `vad_ttl_s` | `-1` | | same, for the VAD |
| `diarization_ttl_s` | `300` | | same, for diarization pipelines |
| `memory_budget_mb` | `0` (off) | `--memory-budget-mb` | evict idle models (LRU) to stay under this estimate. A model's memory is estimated as the growth of the process during its load, and never less than the size of its files |
| `max_loaded_models` | `0` (off) | | cap on models in memory |
| `probe_models` | `true` | | load a newly installed model once in a child process first |
| `threads` | `0` = min(4, CPUs) / `workers` | `--threads` | inference threads per engine call. If the machine also runs other work (a language model, other services), set this below the number of cores: oversubscribed inference threads slow each other down badly |
| `workers` | `1` | `--workers` | concurrent engine calls per model |
| `max_batch` | `8` | `--max-batch` | largest micro-batch |
| `batch_wait_ms` | `0` | `--batch-wait-ms` | linger for a fuller batch (adds latency; see architecture.md) |
| `max_queue` | `64` | `--max-queue` | queued jobs per model before `429` |
| `queue_timeout_s` | `30` | | fail jobs that waited longer than this |
| `partial_interval_ms` | `350` | | how often a pseudo-streaming session re-decodes for a partial |
| `endpoint_silence_ms` | `500` | | trailing silence that ends an utterance. Per stream: `?endpointing=fast\|balanced\|patient` or `?endpoint_ms=` (see realtime.md for the trade-off) |
| `max_utterance_s` | `20` | | force a final after this much continuous speech |
| `agreement` | `2` | | hypotheses that must agree before a word is shown as stable |
| `wyoming` | `false` | `--wyoming` | serve the Wyoming protocol |
| `wyoming_host`, `wyoming_port` | `127.0.0.1`, `10300` | `--wyoming-host`, `--wyoming-port` | Wyoming bind address |
| `device` | `cpu` | | `cuda` is passed to engines that support it - **untested** |
| `log_level` | `info` | `--log-level` | `debug`, `info`, `warning`, `error` |
| `trace_buffer` | `1000` | | request traces kept in memory |
| `fake_engine` | `false` | `--fake-engine` | add deterministic test models (development) |

Three more environment variables are read outside the settings object, for
the optional eSpeak NG component: `BOLO_ESPEAK_LIBRARY` (with
`BOLO_ESPEAK_DATA`) points at a specific libespeak-ng and takes precedence;
otherwise an `espeak-ng` executable on `PATH` is used, then the
`espeakng-loader` package. `BOLO_ESPEAK` (`auto`, `executable`, `library`,
`off`) restricts that search. See
[models.md](models.md#licences-and-the-espeak-ng-question).

## Example file

```toml
# bolo.toml
host = "0.0.0.0"
port = 8000
api_key = "change-me"
cors_origins = ["https://console.example.com"]

[models]
default_stt = "faster-whisper-small"
default_realtime_stt = "zipformer-en-20m-streaming"
default_tts = "kokoro-v1.0-int8"
preload = ["silero-vad", "zipformer-en-20m-streaming"]

[lifecycle]
stt_ttl_s = 900
memory_budget_mb = 3000

[scheduler]
threads = 4
max_batch = 8
max_queue = 32
```

Section names are mostly for your own orientation: `[models]`, `[scheduler]`
and so on are flattened, so every key must be one of the settings above.
The one rule: if `<section>_<key>` is a setting, that is what is set, so
`[llm]` with `base_url`, `model`, `api_key` means `llm_base_url`, `llm_model`,
`llm_api_key`.
An unknown key is an error.

## speaches environment variables

To make switching painless, these `speaches` names are read as fallbacks
(a `BOLO_*` variable for the same setting wins):

| speaches | bolo setting |
| --- | --- |
| `API_KEY` | `api_key` |
| `PRELOAD_MODELS` | `preload` |
| `STT_MODEL_TTL`, `TTS_MODEL_TTL`, `VAD_MODEL_TTL` | `stt_ttl_s`, `tts_ttl_s`, `vad_ttl_s` |
| `ALLOW_ORIGINS` | `cors_origins` |
| `UVICORN_HOST`, `UVICORN_PORT` | `host`, `port` |
| `LOG_LEVEL` | `log_level` |
| `ENABLE_UI` | `ui` |
| `WHISPER__CPU_THREADS`, `WHISPER__NUM_WORKERS` | `threads`, `workers` |
| `HF_TOKEN` | `hf_token` |
| `CHAT_COMPLETION_BASE_URL`, `CHAT_COMPLETION_API_KEY` | `llm_base_url`, `llm_api_key` |

Note that speaches binds `0.0.0.0` by default and bolo binds `127.0.0.1`: set
`UVICORN_HOST=0.0.0.0` (or `BOLO_HOST`) in a container.

## Per-model options

A catalog entry (or a custom entry in `<data_dir>/custom.yaml`) may carry
`options` that override the server-wide values for that model: `threads`,
`workers`, `max_batch`, `max_chunk_s`, and engine-specific keys such as
`beam_size` or `compute_type` for faster-whisper.
