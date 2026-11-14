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
| `default_tts` | `kokoro-v1.0-int8` | `--default-tts` | the quality voice: requests that name no model, `tts-1-hd`, `gpt-4o-mini-tts` |
| `default_realtime_tts` | `piper-en-us-amy-low` | | the low-latency voice: `tts-1`, `model=realtime`, voice agents (Realtime API conversations, `/v1/chat/completions` audio without `speech_model`) and Wyoming synthesis without a voice. A voice that only `default_tts` has (e.g. `af_heart`) still selects `default_tts`. See models.md, "Default voices" |
| `phonemizer` | `auto` | | who turns text into phonemes for Piper, Kitten, Matcha and Kokoro voices: `auto` (uccharan where it covers the voice's language: English US/GB and Hindi; espeak-ng otherwise), `uccharan` (never espeak-ng), `espeak-ng` (as in bolo 0.1). Requests (`phonemizer` field) and per-model pins (`bolo models config <id> --phonemizer …`, `PUT /api/v1/models/{id}/settings`, stored in `<data_dir>/model-settings.json`) override it. See models.md, "Phonemisers" |
| `romanised` | `off` | | romanised text for text-to-speech: an Indian language typed in Latin letters is written in the voice's script before it is read. `off`; `auto` (when the voice speaks an Indian language and the text is mostly Latin script); `on`. A request (`romanised`) and a model pin override it. Needs a uccharan release with a transliterator and the language's pack (component `uccharan-xlit-<language>`, CC BY-SA 4.0); when one is missing the text is passed on unchanged with a notice in the request's trace. See api.md, "Romanised text" |
| `default_vad` | `silero-vad` | | detector for long-file segmentation and realtime endpointing |
| `default_diarization` | `pyannote-3.0-eres2net` | | speaker-diarization model used when a request names none; `nemotron-3-diarization` is the end-to-end alternative (models.md, "Speaker diarization") |
| `diarization_min_speaker_s` | `3` | | after clustering, a "speaker" with less speech than this in the whole recording (and never more than a tenth of the speech) is folded into the most similar speaker; `0` switches it off. Not applied to `nemotron-3-diarization`, whose speakers come from the model rather than from clustering. See api.md, "How many speakers" |
| `diarization_merge_similarity` | `0.6` | | speakers whose voice prints have at least this cosine similarity are merged; `0` switches it off |
| `default_separation` | `voxlock-int8` | | model used by `separate_overlaps` (opt-in per request; pulled on first use). See api.md, "Separating overlapping speech" |
| `separation_context_s` | `2` | | seconds of audio on each side of an overlap that are separated and transcribed again |
| `separation_min_overlap_s` | `0.5` | | overlaps shorter than this are left alone |
| `separation_sample_max_s` | `10` | | at most this much of a speaker's speech without overlap is used as their voice sample (VoxLock needs 3 s and uses the loudest 3 s of it) |
| `separation_timeout_s` | `300` | | stop separating after this long per request; what was not separated is transcribed from the mixture and the response says `reason: "timeout"` |
| `default_turn` | `smart-turn-v3` | | end-of-turn model for Realtime sessions that ask for `semantic_vad` (pulled on first use; without it `semantic_vad` is plain silence detection) |
| `plugins` | `[]` | | out-of-process engine plugins, `name=command args` each (docs/plugins.md) |
| `plugin_entry_points` | `true` | | also use plugins installed as packages (entry-point group `bolo.engines`) |
| `plugin_idle_s` | `300` | | stop a plugin process when none of its models has been loaded for this long |
| `plugin_memory_limit_mb` | `0` | | address-space limit for plugin processes (POSIX); `0`: none |
| `plugin_nice` | `0` | | niceness added to plugin processes |
| `webrtc_ice_servers` | `[]` | | STUN/TURN URLs for the WebRTC transport of the Realtime API (`stun:host:port`, `turn:…`). Empty: no external server is contacted and calls connect on the same machine or network only |
| `llm_base_url` | none | `--llm-base-url` | OpenAI-compatible chat-completions endpoint for voice agents (`/v1/realtime` conversations, `/v1/chat/completions`), e.g. `http://127.0.0.1:11434/v1` |
| `llm_model` | none | `--llm-model` | model name sent to that endpoint; unset: the client's model name is passed on |
| `llm_api_key` | none | `--llm-api-key` | bearer token for that endpoint |
| `llm_timeout_s` | `120` | | |
| `llm_instructions` | a short "helpful voice assistant" prompt | | default system prompt of Realtime conversations |
| `realtime_pipeline` | `cascade` | | what answers a Realtime conversation turn: `cascade` (bolo's STT, then the `llm_*` language model, then bolo's TTS), `audio-llm` (the user's audio goes to an audio-understanding model, `audio_llm_*`; bolo's TTS speaks the answer) or `speech-to-speech` (a speech-to-speech model hears the turn and answers with speech and tool calls, `s2s_*`). A session can pick another one with the bolo extension `session.pipeline`. See voice-agents.md, "Pipelines" |
| `audio_llm_base_url`, `audio_llm_model`, `audio_llm_api_key` | unset (the `llm_*` values) | | `audio-llm`: a chat-completions endpoint whose model accepts `input_audio`, e.g. a llama-server running Gemma 4 E2B/E4B or Ultravox |
| `realtime_transcribe_user` | `true` | | `audio-llm` and `speech-to-speech`: also transcribe the user's turns with bolo's STT (transcript events and conversation history), in parallel with the response |
| `s2s_model` | `lfm3-audio-2b-llamacpp` | | `speech-to-speech`: the catalog model of a llama-server that bolo starts (needs `bolo components install llama-server` or `llama_server_bin`; the model is pulled on first use). LFM Open License: commercial use only below USD 10M yearly revenue. Or `lfm3-audio-2b-onnx`: the same model from its ONNX graphs, in bolo's own process with ONNX Runtime (`pip install 'bolo-speech[s2s]'`; no llama-server, a 2.7 GB download, about 3.4 GB of memory; without the packages a session answers 424, with missing files 503). Or `<provider>:<model>` (a provider of kind `openai-compatible`, e.g. `openai:gpt-realtime-2.1`): OpenAI's Realtime API over WebSocket answers instead; a paid cloud API, the conversation's audio leaves this server (providers.md, "Cloud speech-to-speech") |
| `s2s_server_url` | unset | | `speech-to-speech`: a llama-server you run (`http://host:port`); bolo opens its WebSocket `/v1/realtime`, one upstream session per Realtime session. Nothing is downloaded or started then |
| `s2s_server_api_key` | unset | | sent as `Authorization: Bearer …` to `s2s_server_url` |
| `s2s_parallel` | `2` | | sessions the managed server holds at once (`llama-server --parallel`; one slot per session) |
| `s2s_server_args` | `[]` | | extra command-line arguments for the managed server (bolo adds `-ngl 99` when `device` is not `cpu`) |
| `allow_component_install` | unset | | whether optional components (eSpeak NG) may be installed through the API. Unset: allowed on a loopback bind or when an API key is set. Never allowed on a non-loopback bind without a key |
| `preload` | `[]` | `--preload` (repeatable) | models to load at startup; `/readyz` waits for them |
| `stt_ttl_s`, `tts_ttl_s` | `300` | | unload after this many idle seconds; `-1` never, `0` at once |
| `vad_ttl_s` | `-1` | | same, for the VAD |
| `diarization_ttl_s` | `300` | | same, for diarization pipelines and separation models |
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
| `device` | `cpu` | | where models run: `cpu`, `cuda` (or `cuda:1`), `coreml` (also `mps`, `metal`), `rocm`, `dml`, `openvino`, or `auto` (the first accelerator found). sherpa-onnx models use `cuda` or `coreml`; faster-whisper uses `cuda`; the ONNX Runtime engines (Nemotron diarization, DiCoW ONNX, the Orpheus-family decoder, the end-of-turn model, in-process speech-to-speech) use the matching execution provider when the installed `onnxruntime` has it (for `cuda`, install `onnxruntime-gpu`), and otherwise run on the CPU with one warning; llama-server and audio.cpp get `-ngl 99` for any value other than `cpu`. **Untested on a GPU** |
| `log_level` | `info` | `--log-level` | `debug`, `info`, `warning`, `error` |
| `trace_buffer` | `1000` | | request traces kept in memory |
| `fake_engine` | `false` | `--fake-engine` | add deterministic test models (development) |
| `tts_servers` | `[]` | | external TTS servers the operator runs, as `"<model id or runtime>=<url>"`, e.g. `"llama-server=http://127.0.0.1:8080"` (every llama-server model goes there) or `"voxcpm2-audiocpp=http://gpu-box:8080"` (one model). A model id or alias wins over a runtime name. Such a model needs no download and no binary; for audio.cpp the server's config must declare the model under the id in the entry's `options.server_model` (the catalog id). See models.md, "External TTS servers" |
| `tts_server_api_key` | unset | | sent as `Authorization: Bearer …` to the servers in `tts_servers` (llama-server `--api-key`) |
| `llama_server_bin`, `audiocpp_server_bin` | unset | | the binary bolo starts for a model whose runtime has no URL in `tts_servers`. Unset: the one `bolo components install llama-server` / `audio.cpp` put in `<data_dir>/components/`, then `llama-server` / `audiocpp_server` on `PATH`. Point it at a CUDA/Vulkan/Metal build to use a GPU |
| `tts_server_args` | `[]` | | extra command-line arguments for every server bolo starts (e.g. `["-ngl", "99"]` for llama-server; bolo adds `-ngl 99` itself when `device` is not `cpu`) |
| `tts_server_start_timeout_s` | `300` | | how long a started server may take to answer `GET /health` (loading its model) |
| `tts_server_timeout_s` | `300` | | longest wait for the next bytes of a synthesis answer |
| `providers` | `{}` | | cloud providers for `engine: remote` models: one `[providers.<name>]` table each with `kind` (`openai-compatible`, `sarvam`, `elevenlabs`, `deepgram`; implied for those four names, `openai` meaning `openai-compatible`), `base_url` (default: the vendor's), `api_key`, `options` and `prices`. In the environment: `BOLO_PROVIDERS` (the table as JSON) or `BOLO_PROVIDER_<NAME>_API_KEY` / `_BASE_URL` / `_KIND` (these add to the file's table). Keys are never logged or returned by any API. Paid cloud APIs; audio leaves this server. See [providers.md](providers.md) |
| `fallbacks` | `[]` | | `"<model>=<fallback>[,<fallback>…]"` per model: what answers instead when the model's engine is unavailable, its queue is full, or its provider fails (5xx, timeout, rate limit). Local and cloud models can be mixed (`"whisper-base=openai-whisper-1"`, `"sarvam-saaras-v4=whisper-base"`). A pin through `PUT /api/v1/models/{id}/settings` wins; a catalog entry's `options.fallback` is the default. See providers.md, "Fallback" |
| `allow_provider_edit` | unset | | add, change and remove providers through the API and the console (stored in `<data_dir>/providers.json`, mode 0600). Unset: allowed on a loopback bind or with `api_key`; never on a non-loopback bind without one |
| `provider_timeout_s` | `60` | | longest wait for a cloud provider's answer, or for its next bytes while streaming (a provider or model option `timeout_s` overrides it) |
| `telephony` | `{}` | | phone calls: a `[telephony]` table with `public_base_url` (the `https://` address providers reach this server at), `validate_signatures` (default `true`; the provider endpoints stay closed until requests can be checked, unless this is `false`), `default_agent`, `call_buffer` (call records kept, default 200), and the tables `[telephony.providers.<twilio\|plivo\|telnyx\|exotel>]` (credentials), `[telephony.agents.<name>]` (voice agents) and `[telephony.numbers]` (`"<called number>" = "<agent>"`). In the environment: `BOLO_TELEPHONY` (the table as JSON), `BOLO_TELEPHONY_PUBLIC_BASE_URL`, `BOLO_TELEPHONY_DEFAULT_AGENT`, `BOLO_TELEPHONY_VALIDATE_SIGNATURES`, `BOLO_TELEPHONY_<PROVIDER>_<FIELD>` (these add to the file's table). Credentials are never logged or returned by any API. Every field: [telephony.md](telephony.md) |
| `hardware_warnings` | `true` | | attach a notice when a model that is pulled or loaded is unlikely to fit this machine or to run in real time: a `warnings` list in the answers of `POST /api/v1/models/pull` and `/load`, on the job and on request traces, a log line, a `warning:` line from `bolo pull`, a notice in the console. Notices only: nothing is ever refused, hidden or slowed down, with the setting on or off. `false` removes the notices; the `fit` on catalog entries and `bolo recommend` stay. See [recommendations.md](recommendations.md) |
| `sip` | `false` | | listen for inbound SIP calls from a trunk or PBX (UDP, G.711 RTP; needs `bolo-speech[sip]`). See telephony.md, "SIP" |
| `sip_host`, `sip_port` | `0.0.0.0`, `5060` | | the SIP listening address (UDP) |
| `sip_public_ip` | unset | | the media address written into SDP answers; unset: the local address the INVITE reached. Set it behind NAT |
| `sip_allowed_ips` | `[]` | | source addresses and networks (`"203.0.113.10"`, `"10.0.0.0/8"`) allowed to send SIP requests; empty: loopback only. There is no other authentication |
| `sip_rtp_port_min`, `sip_rtp_port_max` | `40000`, `40100` | | UDP port range for RTP, one port per call |

Four more environment variables are read outside the settings object. `BOLO_UCCHARAN=off` hides an installed
uccharan (to test the espeak-ng route). For
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

[tts]
# a llama-server you run yourself serves every llama-server model; audio.cpp models are started by bolo
servers = ["llama-server=http://127.0.0.1:8080"]

# cloud providers (paid cloud APIs; audio leaves this server): docs/providers.md
[providers.sarvam]
api_key = "…"                         # or BOLO_PROVIDER_SARVAM_API_KEY in the environment

[providers.sarvam.prices]             # optional, for cost estimates: what you pay
"saaras:v4" = { per_hour = 30, currency = "INR" }

# phone calls: docs/telephony.md
[telephony]
public_base_url = "https://voice.example.org"
default_agent = "front-desk"

[telephony.providers.twilio]
auth_token = "…"                      # or BOLO_TELEPHONY_TWILIO_AUTH_TOKEN in the environment

[telephony.agents.front-desk]
instructions = "You answer the phone for Example Clinic."
greeting = "Example Clinic, how can I help?"

[telephony.numbers]
"+14155550100" = "front-desk"
```

`fallbacks = ["whisper-base=openai-whisper-1"]` and the other top-level
settings go above the first `[section]` (TOML reads everything after a
section header as part of that section).

Section names are mostly for your own orientation: `[models]`, `[scheduler]`
and so on are flattened, so every key must be one of the settings above.
The one rule: if `<section>_<key>` is a setting, that is what is set, so
`[llm]` with `base_url`, `model`, `api_key` means `llm_base_url`, `llm_model`,
`llm_api_key`. `[providers.<name>]` tables and the `[telephony]` table are
the exceptions: they stay nested and make up the `providers` and `telephony`
settings.
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

`fallback` (a list of model ids of the same task) names what answers instead
when the model is unavailable, overloaded or, for a cloud model, its provider
fails; the setting `fallbacks` and a pin through
`PUT /api/v1/models/{id}/settings` override it. Entries with `engine: remote`
(cloud models) take `provider`, `model` and the adapter options listed in
[providers.md](providers.md#model-options).

Text handling (see api.md, "Text preparation, SSML and pronunciation word lists"):
`text_normalizer` (`uccharan`, the default for voices that read text; `light`,
bolo's own; `model`, numbers left to the model; `none`), `text_markup` (`kokoro`
when the voice's front end takes `[word](/ipa/)` markup; default `none`),
`text_latin` (`keep` Latin-script words in Indic text, the default, or
`transliterate` them into the voice's script), `text_respell` (`native`,
`latin-hyphen` or `latin-plain` respellings of word-list entries that only have
IPA; `latin-plain` when the entry also sets `lowercases: true`), `numbers` (the
language that reads digits, e.g. `hi`; default: the request language, English
next to English words) and `wordlists` (lists applied to every request for the
model) and `romanised` (`off`, `auto` or `on` for this model's voices; default: the
server's `romanised` setting). `text_normalizer`, `wordlists` and `romanised` can
also be pinned at run time with `PUT /api/v1/models/{id}/settings`, which wins
over the catalog.

Pronunciation word lists are files in `<data_dir>/wordlists/<name>.tsv`
(uccharan's TSV format; manage them with `/api/v1/wordlists`); the list named
`default` applies to every request that does not choose its own lists. They need
the `uccharan` package, which is also what enables the normalise-only mode; without
it bolo uses its light normaliser.

Orpheus-family voices on llama.cpp (`kind: orpheus-llamacpp`; see models.md,
"Orpheus-family voices on llama.cpp") read: `prompt_format` (`svara`, `veena`
or `orpheus`), `temperature`, `top_p`, `repetition_penalty` (defaults: the
model card's values for the format), `seed`, `eos_tokens`, `max_chars` (longest
piece of text per generation, 200), `max_new_tokens` (2048), `frames_per_char`
(the token budget per character of text, 2.0 frames), `stream_frames` (frames
decoded per streamed chunk, 4), `context_frames` (left context per decode, 3),
`speaker_field` (`name` or `id`: what a voice is called in the prompt when it has no `speaker`), `decoder_threads` (onnxruntime threads of the SNAC decoder, at most 2 by
default), `request` (extra `/completion` fields), `snac` (the decoder file:
`file`, `url`, `sha256`, `size`) and `snac_path` (an absolute path to another
SNAC decoder ONNX file, which wins). The llama-server options of the
`tts-server` engine apply too (`model_file`, `server_args`, the settings
`tts_servers`, `llama_server_bin`, `tts_server_args`). Note that
`tts_servers = ["llama-server=…"]` sends these models to that server as well:
name an Orpheus model by its id (`"svara-tts-v1-llamacpp=http://…"`) when your
llama-server runs another model.
