# API reference

bolo listens on `http://127.0.0.1:8000` by default. This page describes every
interface; the machine-readable schema is [`openapi.json`](openapi.json)
(also served live at `/openapi.json`, with an interactive page at `/docs`).

* [Conventions](#conventions): authentication, errors, model names
* [OpenAI-compatible endpoints](#openai-compatible-endpoints)
* [speaches-compatible endpoints](#speaches-compatible-endpoints)
* [Native API](#native-api-apiv1)
* [Realtime speech-to-text WebSocket](#ws-apiv1sttstream)
* [Streaming text-to-speech WebSocket](#ws-apiv1ttsstream)
* [Speaker diarization](#speaker-diarization)
* [Realtime API and chat completions](#realtime-api-and-chat-completions)
* [Optional components](#optional-components)
* [Latency definitions](#latency-definitions)
* [Wyoming protocol](#wyoming-protocol)
* [Operations](#operations)

## All routes

<!-- BEGIN GENERATED: routes -->
**openai**

| Method | Path | Summary |
| --- | --- | --- |
| `GET` | `/v1/models` | List installed models |
| `GET` | `/v1/models/{model_id}` | Retrieve a model |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `POST` | `/v1/audio/speech` | Synthesise speech |
| `POST` | `/v1/chat/completions` | Chat completions with audio in and out (speech front end to your LLM) |
| `POST` | `/v1/realtime/client_secrets` | Create an ephemeral key for Realtime sessions |
| `POST` | `/v1/realtime/calls` | Start a Realtime session over WebRTC (SDP offer/answer) |
| `POST` | `/v1/realtime/calls/{call_id}/hangup` | End a WebRTC Realtime call |

**speaches**

| Method | Path | Summary |
| --- | --- | --- |
| `POST` | `/v1/models/{model_id}` | Download a model (blocks until done) |
| `DELETE` | `/v1/models/{model_id}` | Delete an installed model |
| `GET` | `/v1/registry` | List every model in the catalog |
| `GET` | `/v1/audio/models` | List installed text-to-speech models |
| `GET` | `/v1/audio/voices` | List voices of installed text-to-speech models |
| `GET` | `/api/ps` | List loaded models |
| `POST` | `/api/ps/{model_id}` | Load a model into memory |
| `DELETE` | `/api/ps/{model_id}` | Unload a model from memory |
| `POST` | `/v1/audio/speech/timestamps` | Voice-activity timestamps (milliseconds) |

**native**

| Method | Path | Summary |
| --- | --- | --- |
| `GET` | `/api/v1/info` | Server, engine and platform information |
| `GET` | `/api/v1/catalog` | Model catalog with install and load state |
| `POST` | `/api/v1/models/pull` | Start downloading a model |
| `GET` | `/api/v1/jobs/{job_id}` | Poll a background job |
| `GET` | `/api/v1/jobs` | List background jobs |
| `GET` | `/api/v1/models/{model_id}/settings` | Settings pinned for one model |
| `PUT` | `/api/v1/models/{model_id}/settings` | Pin settings for one model (phonemiser) |
| `POST` | `/api/v1/models/{model_id}/load` | Load a model into memory |
| `POST` | `/api/v1/models/{model_id}/unload` | Unload a model from memory |
| `DELETE` | `/api/v1/models/{model_id}` | Delete an installed model |
| `GET` | `/api/v1/models/{model_id}` | One catalog entry |
| `POST` | `/api/v1/diarize` | Speaker diarization: who spoke when |
| `GET` | `/api/v1/components` | Optional components (uccharan, espeak-ng, engine plugins) |
| `POST` | `/api/v1/components/{component_id}/install` | Install an optional component |
| `GET` | `/api/v1/stats` | Live server statistics |
| `GET` | `/api/v1/requests` | Recent request traces, newest first |
| `GET` | `/api/v1/requests/{request_id}` | One request trace |
| `GET` | `/api/v1/events` | Server-Sent Events: stats, request, request_progress, job, model, log |
| `POST` | `/api/v1/bench` | Start a concurrency sweep |
| `GET` | `/api/v1/samples` | Built-in audio and text samples |
| `GET` | `/api/v1/samples/{name}` | Download a built-in audio sample |

**operations**

| Method | Path | Summary |
| --- | --- | --- |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/healthz` | Liveness |
| `GET` | `/health` | Liveness (speaches-compatible name) |
| `GET` | `/readyz` | Readiness (preloaded models are loaded) |

WebSockets (not part of OpenAPI): `WS /api/v1/stt/stream`, `WS /api/v1/tts/stream`.
<!-- END GENERATED: routes -->

## Conventions

**Authentication.** Off by default. With `api_key` set, send
`Authorization: Bearer <key>`. `X-API-Key: <key>` and, for browsers that
cannot set headers on WebSockets or EventSource, `?api_key=<key>` are also
accepted. `/healthz`, `/readyz`, `/health`, `/metrics` and the console's
static files are never authenticated.

**Errors.** Every error has the OpenAI envelope plus a `detail` copy of the
message:

```json
{"error": {"message": "The model 'x' does not exist. …", "type": "invalid_request_error",
           "param": "model", "code": "model_not_found"},
 "detail": "The model 'x' does not exist. …"}
```

| Status | `code` | When |
| --- | --- | --- |
| 400 | `invalid_request`, `audio_decode_error`, `unsupported_format`, `unsupported_characters` | bad parameters, undecodable audio; text with letters or digits that a character-input voice has no symbol for (the message names them) |
| 401 | `invalid_api_key`, `model_gated` | missing/wrong key; gated model without a token |
| 404 | `model_not_found`, `model_not_installed`, `not_found` | unknown model; not installed and `auto_pull` off |
| 424 | `component_missing` | the voice needs an optional component (espeak-ng) that is not installed; the message says how to install it |
| 429 | `overloaded` (`type: rate_limit_error`) | the model's queue is full, or the memory budget is exhausted. `Retry-After` is set |
| 500 | `synthesis_failed`, `internal_error` | the engine failed, or a voice produced no audio for the text (never an empty 200) |
| 502 | `download_failed`, `llm_error` | a model download failed; the configured language model could not be reached or answered with an error |
| 503 | `llm_not_configured` | `/v1/chat/completions` or a Realtime conversation without `llm_base_url` |
| 503 | `engine_unavailable` | the engine for this model is not installed, or the model failed its load probe |

**Model names.** Anywhere a model is expected you may send a catalog id
(`moonshine-tiny-en`), one of its aliases (often the upstream repository id,
for example `Systran/faster-whisper-small`), or an OpenAI name: `whisper-1`,
`gpt-4o-transcribe`, `gpt-4o-mini-transcribe` map to `default_stt`; `tts-1`,
`tts-1-hd`, `gpt-4o-mini-tts` map to `default_tts`. Omitting the model uses
the default. A model that is in the catalog but not installed is downloaded
on first use unless `auto_pull` is off (the request waits for the download).

**Response headers.** `x-request-id` is the trace id (look it up at
`/api/v1/requests/{id}`); `openai-processing-ms` on transcriptions;
`x-sample-rate` on speech.

## OpenAI-compatible endpoints

Verified with the official `openai` Python SDK in the test-suite.

### `POST /v1/audio/transcriptions`

Multipart form.

| Field | | |
| --- | --- | --- |
| `file` | required | audio. WAV, FLAC, OGG/Vorbis, OGG/Opus, MP3, AIFF decode with no extra install; M4A/AAC, WebM, MP4 and others need `bolo[av]` or `ffmpeg` on `PATH`. Raw headerless audio is accepted when the part's content type says so: `audio/basic` or `audio/x-mulaw` (8 kHz μ-law), `audio/x-alaw`, `audio/pcm;rate=16000` (16-bit little-endian) |
| `model` | optional | see model names |
| `language` | optional | ISO-639-1 code. Improves accuracy and latency for multilingual models |
| `prompt` | optional | initial prompt (Whisper models) |
| `response_format` | `json` | `json`, `text`, `srt`, `vtt`, `verbose_json`, `diarized_json` |
| `timestamp_granularities[]` | `segment` | `word` and/or `segment`; used with `verbose_json` |
| `temperature` | `0` | Whisper models |
| `stream` | `false` | `true`: Server-Sent Events (below) |
| `diarize` | `false` | extension. `true` labels segments with speakers (see [Speaker diarization](#speaker-diarization)); also switched on by `response_format=diarized_json` or the model name `gpt-4o-transcribe-diarize` |
| `num_speakers`, `min_speakers`, `max_speakers` | - | extension, with diarization: fix or bound the number of speakers |
| `separate_overlaps` | `false` | extension, with diarization: transcribe overlapped speech from one separated track per speaker (see [Separating overlapping speech](#speaker-diarization)); `separation_model`, `separation_context_s`, `separation_min_overlap_s`, `separation_allow_narrowband` |
| `vad_filter` | auto | extension. `true` forces silence removal before decoding; by default VAD is used only to split audio longer than the model's window |
| `hotwords` | - | extension, passed to engines that support it |

Responses:

```jsonc
// json
{"text": "Good morning.", "usage": {"type": "duration", "seconds": 3}}

// verbose_json
{"task": "transcribe", "language": "english", "duration": 3.0, "text": "Good morning.",
 "words":    [{"word": "Good", "start": 0.12, "end": 0.38}, …],           // if "word" was requested
 "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": 3.0, "text": "Good morning.", "tokens": [],
               "temperature": 0.0, "avg_logprob": -0.21, "compression_ratio": 0.0, "no_speech_prob": 0.01}],
 "usage": {"type": "duration", "seconds": 3}}
```

`text`, `srt` and `vtt` return plain text. Fields bolo cannot fill honestly
are present with neutral values (`tokens: []`, `compression_ratio: 0.0`;
`avg_logprob`/`no_speech_prob` are real only for faster-whisper models).
Word timestamps need a model with the `timestamps` feature; others return an
empty `words` list.

With `stream=true` the response is `text/event-stream`:

```
data: {"type":"transcript.text.delta","delta":"After early nightfall, the yellow lamps"}
data: {"type":"transcript.text.delta","delta":" would light up here and there."}
data: {"type":"transcript.text.done","text":"After early nightfall, …","usage":{"type":"duration","seconds":42}}
```

One delta is sent per decoded segment (long audio is split at pauses), so a
short clip produces a single delta. An error after streaming has started is
sent as `{"type":"error","error":{…}}`. Disconnecting cancels the remaining work.

### `POST /v1/audio/translations`

Same form without `language` and `stream`; translates to English. Needs a
multilingual Whisper model (feature `translate`); other models answer 400.

### `POST /v1/audio/speech`

JSON body.

| Field | | |
| --- | --- | --- |
| `input` | required | text. Split into sentences and synthesised one at a time |
| `model` | optional | see model names |
| `voice` | optional | a voice id of the model (`GET /api/v1/models/{id}`), or `{"id": …}`. OpenAI voice names (`alloy`, `echo`, `nova`, …) are mapped onto the model's voices so stock clients work |
| `response_format` | `mp3` | `mp3`, `opus` (Ogg), `aac` (ADTS; needs `bolo[av]` or ffmpeg), `flac`, `wav`, `pcm` (16-bit little-endian mono at the model's rate), and the telephony extensions `mulaw` / `alaw` (raw 8 kHz G.711) |
| `speed` | `1.0` | 0.25 to 4.0 |
| `stream_format` | `audio` | `sse` wraps the audio in `speech.audio.delta` events (base64) followed by `speech.audio.done` |
| `instructions` | - | accepted and ignored |
| `sample_rate` | model's | extension: resample the output (8000 to 48000) |
| `language` | voice's | extension: language for text normalisation |
| `normalize` | `true` | extension: `false` sends the text to the voice untouched |
| `phonemizer` | voice's pin, else the server's `phonemizer` setting | extension, for voices that read phonemes: `auto`, `uccharan` (= `uccharan:compatible`), `uccharan:improved`, `espeak-ng`. `424 component_missing` when the chosen one is not installed, `400` (`param: "phonemizer"`) when uccharan does not cover the voice's language. Ignored for voices that read characters or a lexicon. See models.md, "Phonemisers" |

The body is streamed (chunked transfer) while synthesis is still running:
the first sentence is audible while the rest is being generated. `pcm`,
`wav` and `mp3` are forwarded as soon as each piece exists. Ogg/Opus is
written in pages of 100 ms when PyAV is installed (`bolo[av]` or
`[webrtc]`; in our check every 100 ms of input produced output) and in pages
of roughly one second through libsndfile otherwise, in which case prefer
`pcm` or `mp3` when time-to-first-audio matters. A streamed `wav` has a header that declares an
unknown length, as is usual for streams.

Text normalisation (on by default): Unicode NFC, Markdown emphasis, links and
emoji removed, numbers spelled out for English and Hindi. Code-mixed text is
passed through as written.

Voices that read characters from a closed set (the Coqui-format VITS voices:
`coqui-*`, the SYSPIN voices) would silently skip anything they have no
symbol for. bolo checks first: digits are spelled out when a number
normaliser exists for the language (English, Hindi), punctuation the voice
does not know is replaced or dropped, and any letter or digit that is left
answers `400 unsupported_characters` naming the characters, for example
`this voice reads a fixed set of characters and has no symbol for: '4', '2'.
There is no number normaliser for language 'bn': write numbers out in words.`

### `GET /v1/models`, `GET /v1/models/{id}`

Installed models in OpenAI's list shape, with extra fields (`task`,
`language`, `aliases`, `tier`, `license`; `voices` and `sample_rate` for
text-to-speech models). `?task=automatic-speech-recognition|text-to-speech|voice-activity-detection` filters.

## speaches-compatible endpoints

For clients written against speaches; details and differences are in
[migration-from-speaches.md](migration-from-speaches.md).

| Endpoint | Behaviour |
| --- | --- |
| `GET /v1/registry?task=` | every catalog model, installed or not |
| `POST /v1/models/{id}` | download and wait: `200` downloaded, `201` already present, `401` gated, `404` unknown |
| `DELETE /v1/models/{id}` | delete from disk |
| `GET /v1/audio/models`, `GET /v1/audio/voices` | installed text-to-speech models / their voices |
| `GET /api/ps` | `{"models": [ids in memory]}` |
| `POST /api/ps/{id}` / `DELETE /api/ps/{id}` | load (`201`, `409` if loaded) / unload (`200`, `404`, `409` if busy) |
| `POST /v1/audio/speech/timestamps` | multipart `file` (+ `threshold`, `min_silence_duration_ms`, `min_speech_duration_ms`, `max_speech_duration_s`, `speech_pad_ms`) → `[{"start": ms, "end": ms}]` |
| `GET /health` | `{"message": "OK"}` |

## Native API (`/api/v1`)

### `GET /api/v1/info`

```json
{"name": "bolo", "version": "0.2.0",
 "engines": [{"id": "sherpa-onnx", "version": "2.0.1", "available": true, "reason": null},
             {"id": "whisper.cpp", "version": null, "available": false, "reason": "ModuleNotFoundError: …"}],
 "device": "cpu", "cpu_count": 4, "platform": "linux-x86_64", "python": "3.12.3",
 "auth_required": false, "data_dir": "/home/u/.cache/bolo",
 "defaults": {"stt": "whisper-base", "realtime_stt": "moonshine-tiny-en", "tts": "kokoro-v1.0-int8", "vad": "silero-vad"},
 "wyoming": {"enabled": false, "host": "127.0.0.1", "port": 10300},
 "scheduler": {"workers": 1, "threads": 4, "max_batch": 8, "batch_wait_ms": 0.0, "max_queue": 64},
 "components": [ … the list from GET /api/v1/components … ],
 "capabilities": {
   "diarization": {"available": true, "installed": false, "default_model": "pyannote-3.0-eres2net",
                   "http": {"endpoint": "/api/v1/diarize", "transcriptions": ["diarize", "diarized_json"]},
                   "websocket": "diarize=true (one speaker per utterance, labelled online)",
                   "word_timestamps": true, "known_speakers": true, "revised_labels_at_end_of_stream": true},
   "separation": {"available": true, "installed": false, "default_model": "voxlock-int8",
                  "http": {"transcriptions": "separate_overlaps=true (with diarization)", "diarize": "separate_overlaps=true"},
                  "default_on": false, "languages": ["en"], "min_sample_rate": 16000,
                  "context_s": 2.0, "min_overlap_s": 0.5},
   "realtime": {"endpoint": "/v1/realtime", "transcription_sessions": true,
                "conversation_sessions": false,          // true once a language model is configured
                "audio_formats": ["audio/pcm (24 kHz)", "audio/pcmu", "audio/pcma"], "reference": "…",
                "transports": ["websocket", "webrtc"],   // "webrtc" only when aiortc is installed
                "webrtc": {"endpoint": "/v1/realtime/calls", "available": true, "install": null, "ice_servers": []},
                "client_secrets": "/v1/realtime/client_secrets",
                "transcript_deltas": true,               // user transcripts arrive incrementally
                "turn_detection": ["server_vad", "semantic_vad"],
                "semantic_vad": {"model": "smart-turn-v3", "available": true, "installed": false},
                "sip": false},
   "chat_completions": {"endpoint": "/v1/chat/completions", "available": false, "audio": false},
   "llm": {"configured": false, "base_url": null, "model": null}}}
```

### `GET /api/v1/catalog?task=&q=&installed=`

`task`: `stt`, `tts`, `vad`, `diarization`, `turn` (end-of-turn models), `separation` (overlapped-speech separation). `q`: space-separated words that must all appear
in the id, aliases, family, engine, languages, licence or description.
`installed`: `true` / `false`.

```jsonc
{"models": [{
  "id": "moonshine-tiny-en", "aliases": ["moonshine-tiny", "UsefulSensors/moonshine-tiny"],
  "task": "stt", "engine": "sherpa-onnx", "family": "Moonshine", "languages": ["en"],
  "tier": "realtime",                 // realtime | balanced | accuracy
  "streaming": false,                 // true only for natively incremental models
  "sample_rate": 16000, "size_bytes": 123985408, "params": 27000000,
  "license": "MIT", "license_url": null, "commercial_use": true, "gated": false,
  "voices": [],                       // tts: [{"id","name","language","gender"}]
  "features": ["punctuation", "partials"],
  "installed": true, "loaded": false,
  "pulling": null,                    // or {"job_id": "j_3", "progress": 0.42}
  "description": "…", "source_url": "https://github.com/moonshine-ai/moonshine",
  "validated": "cpu",                 // cpu | untested
  "measured": {"rtf": 0.033, "wer": 0.021, "threads": 4, "hardware": "…", "date": "…", …},  // null when untested
  "custom": false,                    // true for models added with hf:
  "requires": [],                     // optional components, e.g. ["espeak-ng"]
  "available": true,                  // false while a required component is not installed
  "status": null                      // "pending": described but its files are not published yet (see below)
}]}
```

Feature tags: `partials` (interim results on the realtime WebSocket: every
speech-to-text model), `timestamps` (word times), `punctuation`, `translate`,
`language-detect`, `multi-voice`, `telephony` (the 8 kHz path was exercised
with this model).

`GET /api/v1/models/{id}` returns one such entry.

Entries with `"status": "pending"` describe models whose files the project
has not published yet (voices awaiting conversion). They are left out of the
catalog listing, `GET /api/v1/models/{id}` shows them, and pulling or using
one answers `404 model_not_published`.

### Model management

| Call | Result |
| --- | --- |
| `POST /api/v1/models/pull` `{"id": "…"}` | `202 {"job_id": "j_1"}`. `id` may be `hf:<owner>/<repo>[@rev]` with optional `"kind"`, `"task"`, or `piper:<voice>` (for example `piper:te_IN-venkatesh-medium`) for any voice of the Piper project, which is converted for sherpa-onnx after the download. Local directories are registered from the command line only (`bolo pull ./dir`) |
| `GET /api/v1/jobs/{job_id}` | `{"id","kind":"pull"\|"bench"\|"install","model","state":"queued"\|"running"\|"done"\|"error","progress":0..1,"bytes_done","bytes_total","error"}` (+ `result` for bench jobs) |
| `GET /api/v1/jobs` | all recent jobs, newest first |
| `POST /api/v1/models/{id}/load` | `200 {"id","loaded":true,"memory_bytes","load_ms"}` |
| `POST /api/v1/models/{id}/unload` | `200 {"id","loaded":false}` |
| `DELETE /api/v1/models/{id}` | `204`; unloads first |
| `GET /api/v1/models/{id}/settings` | `{"model","settings":{…}}`: what the user pinned for that model (`{}` by default) |
| `PUT /api/v1/models/{id}/settings` `{"phonemizer": "espeak-ng"}` | pins the phonemiser of a voice (`null` removes the pin); stored in `<data_dir>/model-settings.json`, shown as `settings` in catalog entries, announced as a `model` event with `"event":"settings"`. `400` for a model that reads characters, or uccharan on a language it does not cover. CLI: `bolo models config <id> --phonemizer X` / `--clear` |

Pulling an installed model yields a job that is already `done`. Pulling a
gated model without a token yields a job in state `error` that says what to do.

### `GET /api/v1/stats`

```jsonc
{"uptime_s": 123.4, "active_streams": 2, "queue_depth": 0,
 "system": {"cpu_percent": 41.0,          // this process, as a share of all cores
            "rss_bytes": 734003200, "threads": 12},
 "totals": {"requests": 120, "errors": 1, "audio_seconds_in": 812.5, "audio_seconds_out": 96.2},
 "models": [{
   "id": "moonshine-tiny-en", "task": "stt", "loaded": true,
   "memory_bytes": 210000000,             // estimate: RSS growth while loading
   "idle_s": 3.2, "ttl_s": 300, "active": 1, "requests": 80, "errors": 0,
   "latency_ms": {"first_partial": {"p50": 180, "p95": 320, "p99": 410},
                  "final":         {"p50": 260, "p95": 480, "p99": 600},
                  "ttfa": null},
   "rtf": {"p50": 0.08, "p95": 0.15},
   "queue": {"depth": 0, "inflight": 1, "workers": 1, "batches": 61, "mean_batch": 1.31, "max_batch": 4,
             "rejected": 0, "queue_ms": {"p50": 0.1, "p95": 38.0, "p99": 71.2}}
 }]}
```

Quantiles are over a rolling window (the most recent 2,048 observations, at
most 15 minutes old) and are `null` until there is data. `queue` is present
for loaded models.

### `GET /api/v1/requests?limit=100&kind=&model=` and `GET /api/v1/requests/{id}`

The most recent request traces, newest first (kept in memory, 1,000 by default).

```jsonc
{"requests": [{
  "id": "r_000042", "kind": "stt",        // stt | tts | vad | diarization | agent
  "transport": "http",                    // http | websocket | wyoming | realtime | chat | cli | bench
  "model": "moonshine-tiny-en", "started_at": "2026-11-13T10:00:00.000Z",
  "status": "ok",                         // ok | error | cancelled
  "error": null, "audio_seconds": 4.2, "chars": 61, "language": "en",
  "phases": [{"name": "resample", "start_ms": 0.0, "dur_ms": 1.9},
             {"name": "queue",    "start_ms": 2.0, "dur_ms": 0.1},
             {"name": "decode",   "start_ms": 2.1, "dur_ms": 139.0},
             {"name": "finalize", "start_ms": 141.1, "dur_ms": 0.0}],
  "first_partial_ms": null, "final_ms": 141.2, "ttfa_ms": null, "total_ms": 141.2,
  "rtf": 0.0336, "batch_size": 1, "text_preview": "After early nightfall, the yellow lamps…"}]}
```

Phase names: speech-to-text `queue`, `resample`, `vad`, `decode`, `finalize`;
text-to-speech `queue`, `normalize`, `synthesize_first`, `synthesize_rest`,
`encode`. A phase can occur more than once (a stream decodes many times);
very long streams fold later occurrences into earlier ones so a trace stays
bounded. See [latency definitions](#latency-definitions).

### `GET /api/v1/events`

Server-Sent Events; each `data:` line is one JSON object with a `type`:

| `type` | Payload | When |
| --- | --- | --- |
| `stats` | the fields of `/api/v1/stats` | once on connect, then every second |
| `request` | the fields of one trace | when a request (or stream) finishes |
| `request_progress` | `id`, `phase: "final"`, `utterance`, `model`, `first_partial_ms`, `final_ms`, `t0`, `t1`, `text_preview` | each finalised utterance of a realtime stream |
| `job` | the fields of a job | pull progress (about four per second), bench progress |
| `model` | `id`, `event`: `loaded` \| `unloaded` \| `installed` \| `removed` (+ `reason` on unload) | |
| `component` | `id`, `event`: `installed` | an optional component was installed and activated |
| `log` | `level`, `message` | notable server messages |

Comment lines (`: keep-alive`) are sent when idle. A subscriber that reads
too slowly loses its oldest events rather than slowing the server.

### `POST /api/v1/bench`

```json
{"kind": "stt", "model": "moonshine-tiny-en", "concurrency": [1, 2, 4, 8], "duration_s": 10,
 "sample": "builtin:en-short", "mode": "batch"}
```

→ `202 {"job_id": "j_7"}`; poll the job. `kind`: `stt` or `tts`. `mode`
(speech-to-text only): `batch` (default: each virtual client sends the sample
back to back) or `realtime` (each client streams it at 1× through a realtime
session). `sample`: an audio sample id for `stt` (default `en-short`), a text
sample id for `tts` (default `en-text-long`). `voice` is optional for `tts`.
The job's `result`, updated after each concurrency level:

```jsonc
{"kind": "stt", "model": "moonshine-tiny-en", "mode": "batch", "sample": "en-short", "duration_s": 10,
 "points": [{"concurrency": 1, "ok": 20, "errors": 0,
             "first_partial_ms": {"p50": 140, "p95": 180}, "final_ms": {"p50": 140, "p95": 181},
             "ttfa_ms": null, "rtf": {"p50": 0.047, "p95": 0.06},
             "throughput_x_realtime": 11.8,     // audio seconds processed per wall-clock second
             "elapsed_s": 10.1, "mean_batch": 1.0}]}
```

The load is generated inside the server process, so it measures the engine
and the scheduler, not HTTP parsing or your network.

### `GET /api/v1/samples`, `GET /api/v1/samples/{file}`

```json
{"samples": [{"id": "en-short", "kind": "audio", "language": "en", "seconds": 3.0,
              "url": "/api/v1/samples/en-short.wav", "text": "Good morning. Could you tell me when my order will arrive?",
              "description": "…"},
             {"id": "hi-text-short", "kind": "text", "language": "hi", "text": "नमस्ते, …", "url": null}]}
```

The audio samples were synthesised with open models (named in each
`description`); `tone-bursts` is a synthetic non-speech signal.

## `WS /api/v1/stt/stream`

Realtime transcription. Query parameters:

| Parameter | Default | |
| --- | --- | --- |
| `model` | `realtime` (= `default_realtime_stt`) | any speech-to-text model |
| `language` | - | hint for multilingual models |
| `encoding` | `pcm_s16le` | `pcm_s16le`, `pcm_f32le`, `mulaw`, `alaw` |
| `sample_rate` | `16000` (`8000` for `mulaw`/`alaw`) | rate of the audio you send; the server resamples |
| `interim` | `true` | send `partial` messages |
| `vad` | `true` | detect speech boundaries. With `false`, an utterance ends only on `flush`/`stop` (or the model's own endpointing for native streaming models) |
| `words` | `true` | include word timestamps in finals when the model has them |
| `diarize` | `false` | add a `speaker` label (`A`, `B`, …) to every final; optional `max_speakers`. See [Speaker diarization](#speaker-diarization) |
| `endpointing` | `balanced` | how long a pause ends an utterance: `fast` (250 ms), `balanced` (500 ms), `patient` (900 ms). The server default is `endpoint_silence_ms` |
| `endpoint_ms` | - | the same as an exact number of milliseconds (100 to 5000); wins over `endpointing` |

Client → server:

* **binary frames**: raw audio in the declared encoding, any frame size
  (20 ms frames are typical: 640 bytes of 16 kHz PCM, 160 bytes of 8 kHz G.711);
* text frames (JSON): `{"type":"config", …}` (same keys as the query
  parameters; only before any audio; the server answers with a new `ready`),
  `{"type":"flush"}` (finalise what has been said, keep the stream open),
  `{"type":"stop"}` (finalise and close), `{"type":"ping"}`.

Server → client (JSON text frames):

```jsonc
{"type":"ready","model":"moonshine-tiny-en","sample_rate":16000,"request_id":"r_000051",
 "encoding":"pcm_s16le","mode":"pseudo","vad":"silero-vad","endpoint_ms":500}   // mode: native | pseudo
{"type":"vad","event":"speech_start","t":1.23}                      // t: seconds into the stream
{"type":"partial","text":"hello wor","t0":1.1,"t1":1.9,"latency_ms":140.2,"stable_chars":6}
{"type":"final","text":"Hello world.","t0":1.1,"t1":2.5,"latency_ms":210.4,
 "words":[{"w":"Hello","t0":1.2,"t1":1.5},{"w":"world.","t0":1.6,"t1":2.1}],"language":"en"}
{"type":"vad","event":"speech_end","t":2.3}
{"type":"metrics","audio_seconds":3.2,"rtf":0.07,"queue_ms":2.0}   // after each final and before done
{"type":"error","code":"overloaded","message":"…"}
{"type":"done","request_id":"r_000051"}
```

`text[:stable_chars]` of a partial is the part that will not change within
the utterance; the rest may still be revised. `t0`/`t1` are seconds from the
start of the stream (in the time base of the audio you sent). Errors that
make the stream unusable are followed by a close: code 1000 for request
errors (so the message can be read), 1008 for authentication, 1011 for
internal errors. If you send audio faster than it can be transcribed and the
backlog passes 30 seconds, the stream ends with `overloaded`.

A connection produces one trace in `/api/v1/requests` (`request_id`), with
`utterances`, and one `request_progress` event per final.

## `WS /api/v1/tts/stream`

Streaming synthesis with text that may arrive in pieces. Query parameters:
`model`, `voice`, `format` (`pcm_s16le` default, `pcm_f32le`, `mulaw`, `alaw`,
`wav`, `mp3`, `opus`, `flac`, `aac`), `sample_rate`, `speed`, `language`.

Client → server (JSON): `{"type":"text","text":"…"}` (optionally
`"flush": true`), `{"type":"flush"}`, `{"type":"stop"}`, `{"type":"ping"}`.
Text is buffered until a sentence is complete, so you can forward language
model tokens as they come. `flush` speaks whatever is buffered and keeps the
stream open for more text; only `stop` ends it (`done` follows the last audio).
If a voice cannot produce audio for a sentence you get an `error` message,
never silence.

Server → client:

```jsonc
{"type":"ready","model":"kokoro-v1.0-int8","voice":"af_heart","format":"pcm_s16le","sample_rate":24000,"request_id":"ws_…"}
{"type":"chunk","seq":0,"t0":0.0,"dur":1.42,"ttfa_ms":118.0,"text":"Hello there."}   // then one binary frame with that audio
{"type":"chunk","seq":1,"t0":1.42,"dur":0.8,"ttfa_ms":null,"text":""}
{"type":"metrics","rtf":0.31,"audio_seconds":2.22,"ttfa_ms":117.6,"request_id":"r_000060"}   // one per sentence
{"type":"done"}
```

`ttfa_ms` is set on the first chunk of each sentence: milliseconds from the
server receiving the text that completed the sentence to that audio being
sent. Each sentence is one trace (`request_id` in `metrics`).

## Speaker diarization

Who spoke when, without PyTorch: pyannote's segmentation-3.0 model and a
speaker-embedding model (both ONNX, run by sherpa-onnx) with clustering. The
default pipeline is `pyannote-3.0-eres2net` (about 32 MB; pulled on first use).

**With a transcription** (`POST /v1/audio/transcriptions`): add `diarize=true`,
optionally `num_speakers` or `min_speakers` / `max_speakers`,
`diarization_model`, `diarization_threshold`. The recording is cut at the
speaker turns and each turn is transcribed, so every segment has exactly one
speaker, labelled `A`, `B`, `C`, … in order of first appearance:

| `response_format` | Speakers appear as |
| --- | --- |
| `verbose_json` | `"speaker": "A"` on every segment |
| `diarized_json` | OpenAI's diarized shape: `{"task","duration","text","segments":[{"type":"transcript.text.segment","id","start","end","text","speaker"}],"usage"}` |
| `srt` | `[A] text` |
| `vtt` | `<v A>text` |
| `text` | one line per turn: `A: text` |
| `json` | not shown (text only) |
| `stream=true` | one `transcript.text.segment` event per turn instead of `transcript.text.delta`, then `transcript.text.done` |

**Word timestamps** (`timestamp_granularities[]=word`) combine with
diarization: every entry of `words` then carries `"speaker"` as well. The
model must be able to produce word times (see "Word timestamps" under
models.md).

**Known speakers.** OpenAI's `known_speaker_names[]` and
`known_speaker_references[]` are supported: up to 8 names, each with a
reference recording of that person (a `data:audio/…;base64,…` URL as in
OpenAI's API, or a file part; a few seconds of clean speech). After
diarization each speaker's voice print is compared with the references
(cosine similarity of speaker embeddings); a speaker that matches a reference
with at least `identify_threshold` (default 0.5) is labelled with that name
instead of a letter, each name being used at most once. Speakers that match
nobody keep `A`, `B`, …. `/api/v1/diarize` accepts the same fields and
reports what it found:
`"known_speakers": [{"name": "Asha", "matched": true, "similarity": 0.71}]`.
This is recognition by similarity against voices *you* supply, not
verification: do not use it to authenticate anyone. Tested with synthetic
voices only (two Kokoro voices told apart and named correctly); accuracy on
real speakers, short references or telephone audio was not measured.

**On its own**: `POST /api/v1/diarize`, multipart `file` plus optional `model`,
`num_speakers`, `min_speakers`, `max_speakers`, `threshold`,
`known_speaker_names[]` / `known_speaker_references[]`, `identify_threshold`:

```json
{"model": "pyannote-3.0-eres2net", "duration": 21.5, "speakers": ["A", "B"],
 "segments": [{"start": 0.03, "end": 2.98, "speaker": "A"}, {"start": 3.7, "end": 6.4, "speaker": "B"}],
 "overlaps": [],
 "request_id": "r_000012"}
```

**Overlapping speech.** The segmentation model (pyannote segmentation-3.0) can
assign two speakers to the same moment, and the pipeline then returns turns
of different speakers that overlap in time; `overlaps` lists those stretches
(`[{"start","end","speakers":["A","B"]}]`, 0.25 s or longer). In a diarized
transcription, segments that overlap another speaker carry `"overlap": true`
in `verbose_json` and `diarized_json`. The overlapping audio is still
transcribed once, as part of the turn that started first, so the second
speaker's first words are usually missing or attributed to the first speaker.
We tried transcribing both turns whole (`DiarizeOptions.keep_overlap`, off by
default): on three synthetic two-voice mixes with 2.5 s of overlap the second
speaker's word error rate got worse in two of three (the recogniser wrote the
louder speaker's words into both texts; `docs/measurements/diarization-overlap.json`),
so it is not the default and not exposed in the API. To get both voices,
use `separate_overlaps` (next section).

**Separating overlapping speech** (opt-in, bolo extension). Add
`separate_overlaps=true` to a diarized transcription (`diarize=true` or
`response_format=diarized_json`) or to `POST /api/v1/diarize`. After
diarization:

1. each speaker's *voice sample* is taken from their own speech away from any
   overlap (the longest clean stretches, up to `separation_sample_max_s` = 10 s;
   VoxLock needs at least 3 s and uses the loudest 3 s of it);
2. every overlap of at least `separation_min_overlap_s` (0.5 s) is widened by
   `separation_context_s` (2 s) on each side, its edges moved to the quietest
   moment nearby, and touching windows joined;
3. the separation model (`separation_model`, default `voxlock-int8`) extracts
   each involved speaker's voice from the window, once per speaker;
4. inside the window each separated speaker is transcribed from their own
   track (over the span of their own turns); everything else is transcribed
   from the recording as before.

Segments transcribed from a track carry `"separated": true` (and
`"overlap": true`). A speaker may then have two consecutive segments (before
the window, inside it), and different speakers' segments overlap in time.

Optional fields: `separation_model`, `separation_context_s` (0-10),
`separation_min_overlap_s` (0-10), `separation_allow_narrowband` (see below);
on `/api/v1/diarize`, `separation_audio=true` adds each window's tracks as
WAV data URLs (`windows[].tracks`).

Every JSON response (`json`, `verbose_json`, `diarized_json`, the final
`transcript.text.done` event of a stream, `/api/v1/diarize`) then has a
`separation` object, and every response the header `x-bolo-separation`
(`applied`, or `<status>; reason=<reason>`):

```json
"separation": {
  "model": "voxlock-int8", "status": "applied", "reason": null, "detail": null, "context_s": 2.0,
  "speakers": [{"speaker": "A", "sample_s": 10.0}, {"speaker": "B", "sample_s": 10.0}],
  "windows": [{"start": 14.57, "end": 20.81,
               "overlaps": [{"start": 16.37, "end": 18.75, "speakers": ["A", "B"]}],
               "speakers": ["A", "B"], "separated": ["A", "B"], "skipped": []}],
  "separated_audio_s": 12.48, "time_ms": 9150.7, "rtf": 0.293
}
```

| `status` | `reason` | meaning |
| --- | --- | --- |
| `applied` | `null` | every speaker in every window got a track |
| `partial` | `no_voice_sample`, `timeout`, `error` | some did; the rest is transcribed from the mixture |
| `skipped` | `no_overlap` | no overlap long enough |
| `skipped` | `no_voice_sample` | no speaker in an overlap has 3 s of speech without overlap (`speakers[].skipped` says how much they had) |
| `skipped` | `timeout`, `error` | separation did not finish (`separation_timeout_s`, default 300 s) or the model failed (`detail`); the request still succeeds |

Refused with 400 (`param: "separate_overlaps"`) before any work:
`separate_overlaps` without diarization, and audio recorded below 16 kHz
(telephony) unless `separation_allow_narrowband=true`: the model is trained on
wideband speech and did not help on 8 kHz μ-law copies in our evaluation
(models.md, "Overlapping speech"). With the override the audio is resampled
and separated anyway, and the report says `"narrowband": true` (also set when
16 kHz audio has nothing above 4 kHz, i.e. an upsampled phone call).

When to use it, and what it costs, is in models.md, "Overlapping speech":
measured results by type of audio. It is off by default. A blind separator in
an engine plugin can be named in `separation_model` as well (its tracks are
given to the diarized speakers by voice similarity).

**Live** (`WS /api/v1/stt/stream?diarize=true`, optional `max_speakers`):
every `final` gets a `"speaker"` field and `ready` reports
`"diarization": "<model>"`. This is *online* labelling: when an utterance is
finalised its audio is embedded and assigned to the closest speaker heard so
far in this stream, or to a new one. It therefore assumes **one speaker per
utterance** (a pause between speakers) and cannot split overlapping speech.
It adds one embedding computation to each final (about 90 ms in our runs;
visible as the `diarize` phase and included in `latency_ms`).

Labels sent with a `final` are never changed afterwards, but when the stream
ends (`{"type":"stop"}`) all utterances are clustered once more, now that all
of them are known, and one message before `done` gives the revised picture:

```json
{"type": "speakers", "revised": 1, "speakers": ["A", "B"],
 "utterances": [{"utterance": 1, "t0": 0.4, "t1": 2.9, "speaker": "A"},
                {"utterance": 2, "t0": 3.6, "t1": 5.1, "speaker": "A", "was": "C"}]}
```

`was` is present where the label differs from the one sent earlier (the
typical repair: a speaker's first short utterance had opened a speaker of its
own). For an authoritative result on a finished recording use the HTTP
routes, which run the full pipeline on the whole file.

Traces: kind `diarization` for `/api/v1/diarize`; a `diarize` phase and the
extra fields `diarization_model` and `speakers` on transcription traces.
With `separate_overlaps`, both kinds add a `separate` phase and the fields
`separation` (as in the header), `separation_model`, `separation_windows` and
`separation_ms`; `/api/v1/stats` counts requests by outcome
(`separation.requests`) and `/metrics` has `bolo_separation_requests_total{status}`
and `bolo_separation_audio_seconds_total`.

### How many speakers

The speaker *count* is the weak point of clustering: left alone it finds too
many (a laugh, a cough or a change of loudness opens a new cluster). bolo
does two things about it, and you can do a third:

1. The default pipeline clusters with threshold 0.9 instead of
   sherpa-onnx's 0.5 (`diarization_threshold` / `threshold` per request).
2. After clustering, speakers are consolidated: two speakers whose voice
   prints have a cosine similarity of at least 0.6 are merged, and a speaker
   with less than 3 seconds of speech in the whole recording (never more than
   a tenth of the speech) is folded into the most similar one
   (`diarization_merge_similarity`, `diarization_min_speaker_s`; the
   `consolidate` phase of the trace, and `speakers_before_consolidation`
   when it changed something).
3. Pass `num_speakers` when you know it, or `min_speakers` /
   `max_speakers`. With `num_speakers` nothing is consolidated.

What it does, measured on 18 recordings with a known number of speakers
(`docs/measurements/diarization-speaker-counts.json`): the four test clips of
sherpa-onnx's segmentation model and the first five minutes (or less) of
14 VoxConverse dev recordings (broadcast and debate audio, CC-BY-4.0).

| Recording | Speakers | Found at threshold 0.5 (sherpa-onnx default) | Threshold 0.9, clustering only | Threshold 0.9 + consolidation (default) |
| --- | --- | --- | --- | --- |
| sherpa-onnx test clip `1-two-speakers-en` | 2 | 2 | 2 | **2** |
| sherpa-onnx test clip `2-two-speakers-en` | 2 | 4 | 2 | **2** |
| sherpa-onnx test clip `3-two-speakers-en` | 2 | 4 | 2 | **2** |
| sherpa-onnx test clip `0-four-speakers-zh` | 4 | 7 | 5 | **4** |
| VoxConverse dev `aisvi` (first 300 s) | 8 | 13 | 8 | **7** |
| VoxConverse dev `xypdm` (first 300 s) | 1 | 3 | 1 | **1** |
| VoxConverse dev `ldnro` (first 300 s) | 4 | 4 | 5 | **4** |
| VoxConverse dev `atgpi` (first 122 s) | 1 | 1 | 1 | **1** |
| VoxConverse dev `xxwgv` (first 300 s) | 8 | 8 | 8 | **8** |
| VoxConverse dev `qhesr` (first 154 s) | 2 | 2 | 2 | **2** |
| VoxConverse dev `ldkmv` (first 300 s) | 5 | - | 10 | **6** |
| VoxConverse dev `ypwjd` (first 260 s) | 4 | - | 5 | **4** |
| VoxConverse dev `pnook` (first 300 s) | 9 | - | 11 | **7** |
| VoxConverse dev `kctgl` (first 106 s) | 2 | - | 2 | **2** |
| VoxConverse dev `ktzmw` (first 300 s) | 4 | - | 5 | **3** |
| VoxConverse dev `kkghn` (first 300 s) | 1 | - | 2 | **1** |
| VoxConverse dev `exymw` (first 116 s) | 5 | - | 8 | **5** |
| VoxConverse dev `sldwj` (first 140 s) | 3 | - | 3 | **3** |

The exact count was found in 14 of 18 recordings with
consolidation, against 10 with clustering alone (mean
error 0.28 against 0.83 speakers); with sherpa-onnx's
default threshold it was right in 5 of the 10 recordings tried.
The consolidation settings (3 s, 0.6) were chosen on the first eight rows
(`aisvi` to `atgpi`); on the ten recordings that were not used for that
choice it found the exact count in 7 of 10, against 4 of 10 with clustering
alone. It errs towards too few speakers in busy recordings (8 → 7, 9 → 7,
4 → 3): people who speak little are folded into others. Expect worse on
recordings unlike these. Consolidation added 0.5 to 8 seconds to recordings of 16 s to 5 min
(one speaker embedding per speaker, on 2 busy threads).

Limits you should expect: somebody who says only a few words in a long
recording disappears into another speaker; two similar voices on the same
microphone can merge; overlapping speech is attributed to one speaker; the
embedding model was trained on English (VoxCeleb), and other languages,
telephone audio and children's voices were not evaluated. No diarization
error rate on real recordings was measured, only speaker counts.

## Realtime API and chat completions

Both need a language model configured with `llm_base_url` (except Realtime
transcription sessions). The full description, the supported events and the
differences from OpenAI are in [voice-agents.md](voice-agents.md).

| Endpoint | Summary |
| --- | --- |
| `WS /v1/realtime?model=…` | OpenAI Realtime API over WebSocket (GA event names, as typed in `openai` 3.50.0). `?intent=transcription` or `session.update` with `type: "transcription"` for transcription-only sessions. Audio: base64 `audio/pcm` (24 kHz PCM16), `audio/pcmu`, `audio/pcma`. User transcripts arrive as incremental deltas. `turn_detection`: `server_vad`, `semantic_vad` (with the optional end-of-turn model) or `null`. Auth: `Authorization: Bearer`, or subprotocols `realtime` + `openai-insecure-api-key.<key>`; an ephemeral `ek_…` key is accepted in the same places |
| `POST /v1/realtime/client_secrets` | mint an ephemeral key (`{"value": "ek_…", "expires_at", "session"}`) for browsers. Needs the server key. Body: `{"expires_after": {"anchor": "created_at", "seconds": 600}, "session": {…}}`, all optional |
| `POST /v1/realtime/calls` | the same sessions over **WebRTC**: SDP offer in (`application/sdp`, or multipart `sdp` + `session`), `201` with the SDP answer and `Location: /v1/realtime/calls/{call_id}`. Needs `bolo[webrtc]` (else `503 engine_unavailable`). `POST …/{call_id}/hangup` ends the call. Not implemented: SIP |
| `POST /v1/chat/completions` | `input_audio` content parts are transcribed; `modalities: ["text","audio"]` + `audio: {voice, format}` adds a spoken reply (`message.audio`, or `delta.audio` when streaming); everything else is forwarded to the language model untouched; requests without audio are proxied as they are |

Voice-agent turns are traces of kind `agent` (transport `realtime` or `chat`)
with `first_token_ms` and `ttfa_ms`; see
[voice-agents.md](voice-agents.md#latency-and-where-to-look).

## Optional components

Pieces bolo can use but does not require: two phonemisers, **uccharan**
(Apache-2.0, part of `bolo[all]`; English US/GB and Hindi) and **eSpeak NG**
(GPL-3.0-or-later, never installed by default; every other language), and
engine plugins (`plugin:<name>`, [plugins.md](plugins.md)), which are listed so
that a client can say what a model needs but are installed as packages.
Find items by `id`; `kind` is `phonemizer` or `engine-plugin`.
`POST /api/v1/components/uccharan/install` needs no licence acceptance.

`GET /api/v1/components`

```json
{"components": [{
  "id": "espeak-ng", "name": "eSpeak NG", "installed": false,
  "source": null,                       // "system-executable" | "python-package" | "configured" | null
  "version": null, "license": "GPL-3.0-or-later",
  "license_url": "https://github.com/espeak-ng/espeak-ng/blob/master/COPYING",
  "required_by_count": 82,              // catalog models that need it
  "installable": true, "install_blocked_reason": null,
  "manual_command": "pip install 'bolo[espeak]'",
  "description": "…"}]}
```

`POST /api/v1/components/espeak-ng/install` with `{"accept_license": true}` →
`202 {"job_id": "j_4"}`. The job has `kind: "install"`; progress arrives as
`job` events and, on success, a `component` event
`{"type":"component","id":"espeak-ng","event":"installed"}`. The server runs
`python -m pip install espeakng-loader` (or `uv pip install` when pip is
absent) in its own environment and then starts using the library **without a
restart**. `400` when `accept_license` is not `true`, or when installation is
not possible; `installable` is `false` and `install_blocked_reason` says why
when the environment is read-only, has neither pip nor uv, the operator set
`allow_component_install = false`, or the server listens on a non-loopback
address without an API key (never allowed). A failed download shows up as a
job in state `error`.

Catalog entries carry `requires` (for a voice that reads phonemes, the
phonemiser it would use here: `["uccharan"]` or `["espeak-ng"]`; `["plugin:torch"]`
for a model on an engine plugin; else `[]`), `available` (false while that is
missing) and `phonemizers` (the choices for the voice, `[{id, available,
default, note}]`; for Kokoro on each voice instead). Synthesis with such a
voice fails fast with `424 component_missing` and a message that names the
component and the ways to install it; over WebSockets it is an `error`
message with that code. `/api/v1/info` includes the same `components` list.
`bolo components list` and `bolo components install espeak-ng --accept-license`
do the same from the command line.

## Latency definitions

All times are measured on the server with one monotonic clock.

| Quantity | Where | Definition |
| --- | --- | --- |
| `latency_ms` | `partial`, `final` on the STT WebSocket | from the server **receiving the last audio frame that contributed to the text** until the message is handed to the socket. It excludes the trailing silence the endpointer waits for (`endpoint_silence_ms`), which is a property of the conversation, not of the server |
| `first_partial_ms` (trace) | realtime stream | `latency_ms` of the first partial of the stream |
| | HTTP transcription with `stream=true` | from the upload being fully received until the first delta is ready |
| `final_ms` (trace) | realtime stream | `latency_ms` of the last final |
| | HTTP transcription | from the upload being fully received until the full transcript is ready |
| `ttfa_ms` (trace) | text-to-speech over HTTP | from the request being parsed until the first encoded audio bytes are handed to the response |
| `total_ms` | all | the whole request (for a stream: its whole lifetime) |
| `rtf` | speech-to-text over HTTP | `total_ms` / audio duration |
| | realtime stream | time spent decoding / audio duration |
| | text-to-speech | `total_ms` / duration of the audio produced |
| `queue` phase | all | from a job being queued until the engine call that contains it starts |
| `stats.latency_ms.first_partial`, `.final` | per model | rolling quantiles of the per-**utterance** values for streams, and of the per-request values for HTTP |

Uploading time is not part of any of these: the clock starts when the server
has the audio. Client-side latency (network, audio capture, playback
buffers) comes on top.

## Wyoming protocol

`bolo serve --wyoming` listens on `tcp://127.0.0.1:10300` (`--wyoming-host`,
`--wyoming-port`). One port serves speech-to-text and text-to-speech.

| Client sends | bolo answers |
| --- | --- |
| `describe` | `info` with one ASR program (installed speech-to-text models) and one TTS program (installed voices; multi-voice models appear once per language as `<model>:<language>` with their voices as speakers) |
| `transcribe` (`name`, `language`), `audio-start`, `audio-chunk`…, `audio-stop` | `transcript-start`, `transcript-chunk` (the text so far, once per decoded piece), `transcript` (`text`, `language`), `transcript-stop`; `info` advertises `supports_transcript_streaming` |
| `synthesize` (`text`, `voice.name`, `voice.speaker`) | `audio-start`, `audio-chunk`… (16-bit PCM at the voice's rate), `audio-stop` |
| `synthesize-start`, `synthesize-chunk`…, `synthesize-stop` | audio as sentences complete, then `audio-stop`, `synthesize-stopped` |

Errors are sent as Wyoming `error` events and the connection stays usable.
The protocol has no authentication. Synthesis without a voice uses
`default_realtime_tts`. The flows Home Assistant uses (describe, transcribe with
transcript streaming, synthesize, streaming synthesis with the compatible
`synthesize` event in between) were run with the official `wyoming` 1.11 client
against a real server with real models
(`docs/measurements/wyoming-flows.json`); Home Assistant itself was not run
(too heavy for the build machine).

## Operations

| Endpoint | |
| --- | --- |
| `GET /healthz` | `200 {"status":"ok"}` as soon as the process serves requests |
| `GET /readyz` | `200 {"status":"ready"}` once every `preload` model is in memory, else `503 {"status":"starting","loading":[…]}` |
| `GET /metrics` | Prometheus text: `bolo_requests_total`, `bolo_request_errors_total`, `bolo_audio_seconds_total{direction}`, `bolo_active_streams`, `bolo_queue_depth`, `bolo_process_*`, and per model `bolo_model_loaded`, `bolo_model_memory_bytes`, `bolo_model_requests_total`, `bolo_latency_ms{metric,quantile}`, `bolo_real_time_factor{quantile}`, `bolo_model_queue_depth`, `bolo_model_mean_batch_size`, `bolo_queue_time_ms{quantile}`, `bolo_model_queue_rejected_total` |
| `GET /` | the web console (single-page app; unknown paths fall back to it). Served with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` so its in-browser recogniser can use WebAssembly threads; turn off with `cross_origin_isolation = false` |
| `GET /docs`, `GET /openapi.json` | interactive API page and schema |
