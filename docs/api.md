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
* [Phone calls](#phone-calls)
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
| `POST` | `/v1/audio/voices` | Create a custom voice from an audio sample |
| `GET` | `/v1/audio/voice_consents` | List consent recordings |
| `POST` | `/v1/audio/voice_consents` | Upload a consent recording |
| `GET` | `/v1/audio/voice_consents/{consent_id}` | One consent recording |
| `POST` | `/v1/audio/voice_consents/{consent_id}` | Rename a consent recording |
| `DELETE` | `/v1/audio/voice_consents/{consent_id}` | Delete a consent recording |
| `DELETE` | `/v1/audio/voices/{voice_id}` | Delete a custom voice |
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
| `POST` | `/v1/audio/speech/embedding` | Speaker embedding of a recording |

**native**

| Method | Path | Summary |
| --- | --- | --- |
| `GET` | `/api/v1/info` | Server, engine and platform information |
| `GET` | `/api/v1/catalog` | Model catalog with install and load state |
| `GET` | `/api/v1/hardware` | This machine: CPU, memory, disk, GPUs, runtimes, container limits |
| `POST` | `/api/v1/hardware/benchmark` | Run the micro-benchmark (a few seconds of CPU) |
| `GET` | `/api/v1/recommendations` | Catalog models for a task, ranked for this machine |
| `GET` | `/api/v1/recommendations/setup` | A set of models for one use and language |
| `POST` | `/api/v1/models/pull` | Start downloading a model |
| `GET` | `/api/v1/jobs/{job_id}` | Poll a background job |
| `GET` | `/api/v1/jobs` | List background jobs |
| `GET` | `/api/v1/models/{model_id}/settings` | Settings pinned for one model |
| `PUT` | `/api/v1/models/{model_id}/settings` | Pin settings for one model (phonemiser, text normaliser, word lists) |
| `GET` | `/api/v1/providers` | Cloud providers (keys are never returned) |
| `GET` | `/api/v1/providers/{name}` | One cloud provider |
| `PUT` | `/api/v1/providers/{name}` | Add or change a cloud provider |
| `DELETE` | `/api/v1/providers/{name}` | Remove a stored cloud provider |
| `POST` | `/api/v1/providers/{name}/test` | Check that a provider answers and accepts the key |
| `POST` | `/api/v1/models/{model_id}/load` | Load a model into memory |
| `POST` | `/api/v1/models/{model_id}/unload` | Unload a model from memory |
| `GET` | `/api/v1/ps` | Loaded models: size, GPU/CPU split, keep-alive (Ollama style) |
| `POST` | `/api/v1/models/copy` | Copy a model's settings to a new name (a preset) |
| `GET` | `/api/v1/models/{model_id}/show` | Everything about one model (bolo show) |
| `DELETE` | `/api/v1/models/{model_id}` | Delete an installed model |
| `GET` | `/api/v1/models/{model_id}` | One catalog entry |
| `POST` | `/api/v1/diarize` | Speaker diarization: who spoke when |
| `GET` | `/api/v1/components` | Optional components (uccharan, espeak-ng, TTS servers, engine plugins) |
| `POST` | `/api/v1/components/{component_id}/install` | Install an optional component |
| `GET` | `/api/v1/stats` | Live server statistics |
| `GET` | `/api/v1/requests` | Recent request traces, newest first |
| `GET` | `/api/v1/requests/{request_id}` | One request trace |
| `GET` | `/api/v1/events` | Server-Sent Events: stats, request, request_progress, job, model, log |
| `POST` | `/api/v1/bench` | Start a concurrency sweep |
| `GET` | `/api/v1/samples` | Built-in audio and text samples |
| `GET` | `/api/v1/samples/{name}` | Download a built-in audio sample |
| `GET` | `/api/v1/wordlists` | List pronunciation word lists |
| `GET` | `/api/v1/wordlists/{name}` | One word list (JSON entries, TSV or PLS) |
| `PUT` | `/api/v1/wordlists/{name}` | Create or replace a word list (TSV, PLS or JSON body) |
| `DELETE` | `/api/v1/wordlists/{name}` | Delete a word list |
| `POST` | `/api/v1/wordlists/{name}/preview` | What each voice would receive |
| `GET` | `/api/v1/text-input/{model_id}` | How a model's voices take text |
| `GET` | `/api/v1/telephony` | Telephony: providers, webhook URLs, agents, numbers (no secrets) |
| `GET` | `/api/v1/telephony/agents` | Voice agents that answer and place calls |
| `GET` | `/api/v1/telephony/agents/{name}` | One voice agent |
| `PUT` | `/api/v1/telephony/agents/{name}` | Add or change a voice agent |
| `DELETE` | `/api/v1/telephony/agents/{name}` | Remove a stored voice agent |
| `GET` | `/api/v1/telephony/numbers` | Which agent answers which number |
| `PUT` | `/api/v1/telephony/numbers/{number}` | Map a called number to an agent |
| `DELETE` | `/api/v1/telephony/numbers/{number}` | Remove a number's mapping |
| `GET` | `/api/v1/telephony/calls` | Recent calls, newest first |
| `POST` | `/api/v1/telephony/calls` | Place an outbound call and attach an agent |
| `GET` | `/api/v1/telephony/calls/{call_id}` | One call: timeline, turns with latency, transcript |
| `POST` | `/api/v1/telephony/calls/{call_id}/hangup` | End a call in progress |
| `GET` | `/api/v1/telephony/scenarios` | Built-in call-simulator scenarios |
| `POST` | `/api/v1/telephony/simulate` | Simulate a call to an agent through a fake provider |

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
| 502 | `provider_error`, `provider_auth` | a cloud provider failed (5xx, a broken connection, an unknown model) or refused bolo's key. See [providers.md](providers.md#errors) |
| 503 | `llm_not_configured` | `/v1/chat/completions` or a Realtime conversation without `llm_base_url` |
| 503 | `engine_unavailable` | the engine for this model is not installed, or the model failed its load probe |
| 503 | `provider_not_configured` | a cloud model whose provider has no entry or no key in `providers` |
| 504 | `provider_timeout` | a cloud provider did not answer within `provider_timeout_s` |

A provider's 400 is passed on as `400 invalid_request` with its message, and its 429 as
`429 overloaded` with its `Retry-After`. `engine_unavailable`, `overloaded` and the `provider_*`
errors are what a model's [fallback list](providers.md#fallback) reacts to.

**Model names.** Anywhere a model is expected you may send a catalog id
(`moonshine-tiny-en`), one of its aliases (often the upstream repository id,
for example `Systran/faster-whisper-small`), or an OpenAI name: `whisper-1`,
`gpt-4o-transcribe`, `gpt-4o-mini-transcribe` map to `default_stt`; `tts-1`,
`tts-1-hd`, `gpt-4o-mini-tts` map to `default_tts`. Omitting the model uses
the default. A model that is in the catalog but not installed is downloaded
on first use unless `auto_pull` is off (the request waits for the download).
Cloud models (catalog entries with `cloud: true`, for example
`openai-whisper-1` or `sarvam-bulbul-v3`) are used the same way once their
provider is configured, and any other model of a configured provider can be
named as `<provider>:<model>` (`openai:gpt-transcribe`,
`sarvam:saarika:v2.5`). They are paid cloud APIs and the audio leaves this
server: see [providers.md](providers.md).

**Response headers.** `x-request-id` is the trace id (look it up at
`/api/v1/requests/{id}`); `openai-processing-ms` on transcriptions;
`x-sample-rate` on speech. `x-bolo-model` on transcriptions and speech is the
model that answered; `x-bolo-fallback: <requested> -> <answered>;
reason=<code>` is added when a [fallback](providers.md#fallback) answered
instead of the model that was asked for, and `x-bolo-estimated-cost:
<amount> <currency>` on transcriptions served by a cloud model that has a
price in the provider's price table.

## OpenAI-compatible endpoints

Verified with the official `openai` Python SDK in the test-suite.

### `POST /v1/audio/transcriptions`

Multipart form.

| Field | | |
| --- | --- | --- |
| `file` | required | audio. WAV, FLAC, OGG/Vorbis, OGG/Opus, MP3, AIFF decode with no extra install; M4A/AAC, WebM, MP4 and others need `bolo-speech[av]` or `ffmpeg` on `PATH`. Raw headerless audio is accepted when the part's content type says so: `audio/basic` or `audio/x-mulaw` (8 kHz μ-law), `audio/x-alaw`, `audio/pcm;rate=16000` (16-bit little-endian) |
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
| `voice` | optional | a voice id of the model (`GET /api/v1/models/{id}`), or a **saved voice** ([voices.md](voices.md)) for a model that clones voices: `"priya"`, or OpenAI's custom-voice object `{"id": "priya"}`. A string is the model's own voice when it has one of that name, else the saved voice; the object prefers the saved voice. A saved voice with a model that cannot clone → `400` naming the cloning models installed. Without `model`, a saved voice's `default_model` answers. OpenAI voice names (`alloy`, `echo`, `nova`, …) are mapped onto the model's voices so stock clients work |
| `response_format` | `mp3` | `mp3`, `opus` (Ogg), `aac` (ADTS; needs `bolo-speech[av]` or ffmpeg), `flac`, `wav`, `pcm` (16-bit little-endian mono at the model's rate), and the telephony extensions `mulaw` / `alaw` (raw 8 kHz G.711) |
| `speed` | `1.0` | 0.25 to 4.0 |
| `stream_format` | `audio` | `sse` wraps the audio in `speech.audio.delta` events (base64) followed by `speech.audio.done` |
| `instructions` | - | accepted and ignored |
| `sample_rate` | model's | extension: resample the output (8000 to 48000) |
| `language` | voice's | extension: language for text normalisation |
| `normalize` | `true` | extension: `true` uses the voice's default normaliser (see "Text preparation" below); `"uccharan"`, `"light"` (bolo's own), `"model"` (leave numbers to the model) or `false` / `"none"` (word lists still apply) |
| `input_format` | `text` | extension: `ssml` accepts the SSML subset below; `markdown` is the same as `text`. In every format, Kokoro-style `[word](/ipa/)` sets the pronunciation of that word |
| `wordlists` | model pins, then `default` | extension: names of pronunciation word lists (`/api/v1/wordlists`), highest priority first. `[]` applies none. An unknown name → `404 wordlist_not_found` |
| `pronunciations` | - | extension: inline entries `[{"word", "ipa"?, "say"?, "lang"?, "match"?}]`, applied before any list |
| `romanised` | model pin, else the server's `romanised` setting (`off`) | extension: `"off"`, `"auto"` or `"on"` (`true` / `false` are `on` / `off`; `romanized` is accepted too). An Indian language typed in Latin letters is written in the voice's script before it is read; see "Romanised text" below. `on` → `424 component_missing` when the transliterator or the language's pack is not installed, `400` for a voice or language it cannot apply to |
| `phonemizer` | voice's pin, else the server's `phonemizer` setting | extension, for voices that read phonemes: `auto`, `uccharan` (= `uccharan:compatible`), `uccharan:improved`, `espeak-ng`. `424 component_missing` when the chosen one is not installed, `400` (`param: "phonemizer"`) when uccharan does not cover the voice's language. Ignored for voices that read characters or a lexicon. See models.md, "Phonemisers" |
| `reference_audio` | - | extension, for models that clone voices (`features` contains `voice-cloning`, today the `tts-server` models): a recording of the voice to imitate, plain base64 or a `data:audio/…;base64,` URL (WAV, FLAC, MP3, … whatever bolo can decode). Up to 30 s are used; it is sent to the server as mono 16-bit WAV. `400` (`param: "reference_audio"`) for a model that cannot clone, an empty, undecodable or shorter-than-0.5 s recording. It takes precedence over `voice` |
| `reference_text` | - | extension: what is said in `reference_audio`; some cloning models (audio.cpp CosyVoice3, IndexTTS-2, …) use it |
| `options` | - | extension: model-specific request options. Tontaube: `style` (`audiobook`, `conversational`, `agentic`), the sliders `rate` (0-4), `pause` (0-4), `pitch_range` (0-5), `pitch_level` (0-4), `legato` (0-2), `temperature`, `seed`. audio.cpp models: passed through as the server's `options`. `400` (`param: "options"`) for an option the model does not read |

Speech in a cloned voice (a saved voice, or `reference_audio`) carries the
response headers `x-bolo-cloned: true` and, for a saved voice, `x-bolo-voice:
<id>`; the request's trace (`GET /api/v1/requests/{id}`) has `extra.voice`
(`{"saved": "priya", "cloned": true, "consent": "self"}`).

The body is streamed (chunked transfer) while synthesis is still running:
the first sentence is audible while the rest is being generated. `pcm`,
`wav` and `mp3` are forwarded as soon as each piece exists. Ogg/Opus is
written in pages of 100 ms when PyAV is installed (`bolo-speech[av]` or
`[webrtc]`; in our check every 100 ms of input produced output) and in pages
of roughly one second through libsndfile otherwise, in which case prefer
`pcm` or `mp3` when time-to-first-audio matters. A streamed `wav` has a header that declares an
unknown length, as is usual for streams.

Text normalisation (on by default) depends on how the voice takes text; see
"Text preparation" below. Voices that read text themselves (the `tts-server`
models, including Orpheus-family voices on llama.cpp, and plugin voices) get
uccharan's normalise-only mode when the `uccharan` package is installed: numbers,
money (`₹`, `Rs.`, lakh/crore), dates, times, phone numbers, units, abbreviations,
URLs and email addresses written out in words, in English, Hindi and the other
South Asian languages uccharan reads, Urdu and Sindhi; Latin-script English inside
Indic text is kept. Since uccharan 0.6.1 the same
holds for its Latin-script languages (Spanish, German, French …): `25 €` is read
"fünfundzwanzig Euro" for a German voice; since 0.6.2 Sinhala and Sindhi
numbers are words too, and 0.6.4 (the minimum the extra asks for) reads clock times,
ordinals and dates and puts currency words in the plural. (uccharan 0.6.0 read digits in English in
the Latin-script languages and 0.6.1 wrote Sinhala and Sindhi numbers as lookup keys
that are no words. bolo asks the installed uccharan to write out a few numbers in the
language, once per language, and uses its normaliser only when they come back as words
of the language's script: no digits, no characters of another script, not English, not
the digits spelled one by one. Otherwise those text voices get the light normaliser
below. Voices that read phonemes get numbers read in their language by uccharan's
phonemiser either way.)
Every other voice, and every voice when uccharan is missing, gets bolo's light
normaliser: Unicode NFC, Markdown emphasis, links and emoji removed, URLs read out
("w w w dot example dot com"), numbers spelled out for English and Hindi.
Code-mixed text is passed through as written.

Voices that read characters from a closed set (the Coqui-format VITS voices:
`coqui-*`, the SYSPIN voices) would silently skip anything they have no
symbol for. bolo checks first: digits are spelled out when a number
normaliser exists for the language (English, Hindi), punctuation the voice
does not know is replaced or dropped, and any letter or digit that is left
answers `400 unsupported_characters` naming the characters, for example
`this voice reads a fixed set of characters and has no symbol for: '4', '2'.
There is no number normaliser for language 'bn': write numbers out in words.`

Models on an external TTS server (`engine: "tts-server"`, llama-server or
audio.cpp; [models.md](models.md#external-tts-servers-llama-server-audiocpp)) behave
the same way: bolo forwards each sentence to the server, streams its PCM back
(resampled to the catalog's `sample_rate`) and encodes it as asked. `language`
is passed on to multilingual models. Voices named `… (reference clip)` are
bolo's synthetic reference recordings sent as the voice to imitate; `default`
is the model's own voice. Errors: the server binary is missing → `424
component_missing` (install it with `bolo components install llama-server` /
`audio.cpp`, or point `tts_servers` at a running server); the server cannot be
reached, did not start, crashed or answers 5xx → `503 engine_unavailable` (a
crashed server is started again on a later request, after 1 s, doubling up to
60 s); a request the server refuses (an unknown voice, a speed other than
1.0 on a model that has one speed) → `400`.

Orpheus-family voices on llama.cpp (`kind: "orpheus-llamacpp"`: Svara, Veena,
Orpheus, snorTTS; [models.md](models.md#orpheus-family-voices-on-llamacpp-svara-veena-orpheus-snortts))
run on the same llama-server management, but bolo writes the prompt, streams
the generated token ids from llama-server and decodes the audio itself, so the
first audio arrives after about 5 frames (0.4 s of speech) have been generated.
`voice` is one of the entry's voices (by id or name; the first one when
omitted; another name → `400`, `param: "voice"`); `speed` must be 1.0 and
`reference_audio` is refused (`400`). Without onnxruntime installed the model
answers `424 component_missing` (`pip install "bolo-speech[orpheus]"`).

#### Text preparation, SSML and pronunciation word lists

Each voice has a **text-input record** (`GET /api/v1/text-input/{model}`) that decides what it
receives:

| `reads` | Voices | Normaliser by default | A word-list entry becomes |
| --- | --- | --- | --- |
| `phonemes` | Piper, Kokoro, Kitten, Matcha (sherpa-onnx) | `light` (the phonemiser reads numbers itself) | its IPA, converted to the voice's phone set by uccharan (`pronunciation: "phonemes"`) |
| `text` | `tts-server` models, Orpheus on llama.cpp, plugin voices | `uccharan` | its `say` form, or a respelling uccharan derives from its IPA; a voice with `text_markup: kokoro` gets `[word](/ipa/)` (`pronunciation: "respelling"` or `"markup"`) |
| `characters` | closed character sets (Coqui VITS, SYSPIN, MMS) | `light`, then the voice's own fitting | its `say` form |

The record's other fields: `latin` (`keep` or `transliterate` Latin words), `respell`
(`native`, `latin-hyphen`, `latin-plain`), `digits` (which language reads digits),
`wordlists` (pinned lists) and `romanised` (the voice's mode for romanised text, below). They come from the catalog entry's options and the model's pinned
settings (configuration.md, "Per-model options").

Pronunciation sources, the first match wins for any stretch of text: SSML `<phoneme>` and
`[word](/ipa/)` in the input, `pronunciations`, `wordlists` in the order given, lists pinned to
the model, then the list named `default`. Inside one list the longest match wins; a match in an
earlier source wins even over a longer one in a later source. Word lists apply before number
reading, so entries such as `2G` or `MH12` win over the normaliser. Entries that a voice cannot
use are skipped, not errors (an IPA-only entry on a voice without a respelling for its script).
The trace of each request (`GET /api/v1/requests/{id}`) records `normalizer`, `normalized_text`,
`pronunciations_applied` and, when romanised text came into play, `romanised`.

**Romanised text.** People type Indian languages in Latin letters ("mujhe kal office jana
hai"), and a Hindi voice reads that badly or not at all. With `romanised`, bolo first writes such
text in the voice's script ("मुझे कल office जाना है") with uccharan's transliterator: words of the
language are converted, English words stay as they are and are then handled like any Latin word
by the voice's normaliser or phonemiser.

| Mode | What happens |
| --- | --- |
| `off` (default) | text is passed on as written |
| `auto` | converted when the voice speaks an Indian language and at least 60 % of the text's letters are Latin; text that is already in the script, or mostly so, is left alone |
| `on` | converted whenever the voice speaks an Indian language and the text has Latin-letter words |

The mode comes from the request (`romanised`), else the model's pin (`PUT
/api/v1/models/{id}/settings` with `romanised`, or the catalog option `romanised`), else the
server's `romanised` setting. The language is the request's `language`, else the voice's.
Words with a pronunciation of their own (inline `pronunciations`, `[word](/ipa/)`, word-list
entries) and URLs and email addresses are not converted. The conversion runs before the
normaliser and the word lists.

It needs two optional pieces, and bolo never fails a request because one is missing unless that
request itself said `"on"`:

* uccharan with its transliterator (`uccharan.transliterate`; bolo asks for uccharan 0.6;
  0.3.0 and older have none); `capabilities.romanised.available` in `GET /api/v1/info` says what is installed;
* the language's pack, `xlit-<language>` (uccharan lists Hindi, Marathi, Bengali, Tamil, Telugu,
  Gujarati, Punjabi, Kannada and Malayalam; the installed release decides, see
  `capabilities.romanised.languages`). The packs are **CC BY-SA 4.0**, not Apache-2.0 (they are
  built from share-alike word lists), so they are never downloaded on their own: install one as
  the component `uccharan-xlit-<language>` ([Optional components](#optional-components)).

| Situation | `auto`, or `on` from a pin, the server setting, a session or an agent | `"on"` in the request |
| --- | --- | --- |
| no uccharan, or one without a transliterator | text unchanged, notice in the trace | `424 component_missing` (the message says `pip install -U uccharan`) |
| the language's pack is not installed | text unchanged, notice in the trace | `424 component_missing` (the message names the component) |
| the installed uccharan has no pack for the language | text unchanged, notice in the trace | `400 invalid_request`, `param: "romanised"` |
| the voice does not speak an Indian language | `auto`: nothing; `on`: notice in the trace | `400 invalid_request`, `param: "romanised"` |
| the transliterator raises | text unchanged, notice in the trace | the same: text unchanged, notice |

The trace of the request (`GET /api/v1/requests/{id}`) carries `romanised` whenever the option
came into play: `{"mode", "source": "request" | "voice", "language", "applied", "latin_share",
"pack", "text" (the converted text, first 500 characters), "words"}`, or with `"applied": false`
and a `"notice"` that says what is missing and how to install it. The preview route
(`POST /api/v1/wordlists/{name}/preview`, which takes `romanised` too) returns the same record
per voice and never refuses.

Where else it can be set: `romanised` as a query parameter of `WS /api/v1/tts/stream`,
`session.audio.output.romanised` in a Realtime session, `audio.romanised` in
`/v1/chat/completions`, `romanised` on a telephony agent ([telephony.md](telephony.md)),
`bolo speak --romanised auto`. In sessions, chat completions and calls the text comes from a
language model, so nothing is ever refused there: a conversion that cannot happen is a notice in
the sentence's trace. **Not validated end to end**: bolo's side is tested against a stand-in for
the transliterator, and with uccharan 0.4.0 and its Hindi pack a romanised request was converted
and spoken by one Piper Hindi voice; conversion quality is uccharan's (its
documentation has the accuracy figures), and no listening test of converted text was made.

SSML subset (`input_format: "ssml"`; the `<speak>` wrapper is optional): `<sub alias>`,
`<phoneme alphabet="ipa" ph>`, `<say-as interpret-as>` with `characters` / `spell-out`,
`cardinal`, `ordinal`, `digits` / `telephone`, `date` (`format` `dmy`, `mdy`, `ymd`), `time`,
`currency`; `<break>`, `<p>`, `<s>`, `<lang>`, and `<emphasis>` / `<prosody>` (text kept, styling
ignored). Malformed SSML → `400` with the line and column; another element → `400`; DOCTYPE
and entity declarations are refused.

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
| `POST /v1/audio/speech/embedding` | multipart `file` (+ `model`) → the speaker embedding, see below |
| `GET /health` | `{"message": "OK"}` |

### `POST /v1/audio/speech/embedding`

The speaker embedding of a recording: a unit-length vector whose cosine with
another recording's vector says how alike the two voices are. The response
has the shape of OpenAI's embeddings response, as speaches answers:

```json
{"object": "list",
 "data": [{"object": "embedding", "embedding": [0.031, -0.087, ...], "index": 0}],
 "model": "pyannote-3.0-eres2net",
 "usage": {"prompt_tokens": 48000, "total_tokens": 48000}}
```

* `file`: the audio (any format the transcription route takes; raw PCM as
  `audio/pcm` too). At least half a second; a longer clip uses its first 30 s.
* `model`: a diarization model in bolo's catalog, whose speaker-embedding
  model is used. Default: the one that scores cloned voices
  (`voice_similarity_model`, else `default_diarization`, which is
  `pyannote-3.0-eres2net`: ERes2Net from 3D-Speaker, 192 numbers). A model id
  bolo does not know, such as speaches'
  `pyannote/wespeaker-voxceleb-resnet34-LM`, uses that default; `model` in
  the response names the model that was used.
* `usage` counts samples at 16 kHz, as speaches does.

How to read a cosine: with the default model, recordings of the same person
scored 0.45 to 0.93 and different people -0.30 to 0.53 on LibriSpeech
(`docs/measurements/speaker-similarity-calibration.json`); 0.50 is the line
bolo uses for "same speaker". Another embedding model has another scale. The
request appears in the request traces as kind `embedding`.

```bash
curl -s localhost:8000/v1/audio/speech/embedding -F file=@a.wav | jq '.data[0].embedding | length'
```

## Native API (`/api/v1`)

### `GET /api/v1/info`

```json
{"name": "bolo", "version": "0.2.0",
 "engines": [{"id": "sherpa-onnx", "version": "2.0.1", "available": true, "reason": null},
             {"id": "whisper.cpp", "version": null, "available": false, "reason": "ModuleNotFoundError: …"}],
 "device": "cpu", "cpu_count": 4, "platform": "linux-x86_64", "python": "3.12.3",
 "auth_required": false, "data_dir": "/home/u/.cache/bolo",
 "auto_pull": true,                       // models that are not installed are downloaded on first use
 "providers": [{"name": "openai", "kind": "openai-compatible", "configured": true}],   // cloud providers; never a key
 "defaults": {"stt": "whisper-base", "realtime_stt": "moonshine-tiny-en", "tts": "kokoro-v1.0-int8", "vad": "silero-vad"},
 "wyoming": {"enabled": false, "host": "127.0.0.1", "port": 10300},
 "scheduler": {"workers": 1, "num_parallel": 1, "threads": 4, "max_batch": 8, "batch_wait_ms": 0.0, "max_queue": 64},
 "log_file": "/home/u/.local/state/bolo/logs/server.log",   // the file the server also logs to; null: stderr only
 "health": {"status": "ok",              // "degraded" while the GPU is lost (configuration.md, "A lost GPU")
            "gpu": {"state": "ok", "lost": false, "reason": null, "source": "nvml", "gpus": 1, "checked_at": 1794833083.9,
                    "lost_at": null, "fallback": null, "moved_to_cpu": [], "degraded": [], "notice": null,
                    "interval_s": 30.0, "checks": 12}},   // state: unchecked (no GPU model yet) | ok | lost | none | unknown
 "components": [ … the list from GET /api/v1/components … ],
 "capabilities": {
   "wordlists": {"available": true,          // false without uccharan; "install" then says how to add it
                 "endpoint": "/api/v1/wordlists", "install": null, "request_fields": ["wordlists", "pronunciations"],
                 "formats": ["tsv", "pls", "json"], "pin_per_model": true,
                 "preview": "/api/v1/wordlists/{name}/preview", "max_lists_per_request": 8},
   // romanised text for text-to-speech: "available" is false (with a "reason") when the installed uccharan has
   // no transliterator; "languages" have a pack in that uccharan, "installed" are the packs on this machine
   "romanised": {"available": true, "reason": null, "modes": ["off", "auto", "on"], "languages": ["hi"],
                 "installed": ["hi"], "default": "off", "request_field": "romanised", "pin_per_model": true},
   "providers": {"endpoint": "/api/v1/providers", "kinds": ["openai-compatible", "sarvam", "elevenlabs", "deepgram"],
                 "editable": true, "edit_blocked_reason": null, "configured": ["openai"],
                 "adhoc_models": "<provider>:<model>", "fallback": true, "cost_estimates": false},
   "diarization": {"available": true, "installed": false, "default_model": "pyannote-3.0-eres2net",
                   "http": {"endpoint": "/api/v1/diarize", "transcriptions": ["diarize", "diarized_json"]},
                   "websocket": "diarize=true (one speaker per utterance, labelled online)",
                   "word_timestamps": true, "known_speakers": true, "revised_labels_at_end_of_stream": true,
                   "speaker_conditioned_models": ["dicow-v3.3", "se-dicow"]},
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
                "sip": false,
                "pipeline": "cascade",                   // the server's default (setting realtime_pipeline)
                // per pipeline: "configured" (its settings are in place) and "available" (a session can use it now)
                "pipelines": {
                  "cascade": {"configured": false, "available": false},          // needs llm_base_url
                  "audio-llm": {"configured": false, "available": false, "base_url": null, "model": null},
                  "speech-to-speech": {"configured": true, "available": false,
                                       "reason": "llama-server is not installed (bolo components install llama-server)",
                                       "model": "lfm3-audio-2b-llamacpp", "url": null}}},
   "chat_completions": {"endpoint": "/v1/chat/completions", "available": false, "audio": false},
   "llm": {"configured": false, "base_url": null, "model": null},
   // phone calls (telephony.md): "exposed" lists the providers whose webhook and stream routes answer
   "telephony": {"endpoint": "/api/v1/telephony", "providers": ["twilio", "plivo", "telnyx", "exotel"],
                 "configured": [], "exposed": [], "validate_signatures": true, "agents": 0, "outbound": true,
                 "simulator": "/api/v1/telephony/simulate",
                 "sip": {"enabled": false, "listening": false, "port": 5060}}}}
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
  "status": null,                     // "pending": described but its files are not published yet (see below)
  "runnable": true,                   // false: no runtime bolo has can run this model yet …
  "not_runnable_reason": null,        // … and why (spark-tts-0.5b, maskgct, diarizen-wavlm-large-s80-md-v2)
  "cloud": false,                     // true: a paid cloud API (engine "remote"); nothing is downloaded
  "notice": null,                     // cloud models: "paid cloud API; audio leaves this server"
  "provider": null,                   // cloud models: {"name","kind","model","configured","price"}
  "fallback": [],                     // models that answer instead when this one cannot (providers.md, "Fallback")
  "hardware": null,                   // or {"gpu": "optional"|"recommended"|"required", "vram_gb": 2.0, "note": "…"}
  "fit": {                            // how well the model suits THIS machine; advice, never a restriction
    "level": "comfortable",           // comfortable | usable | slow | unlikely | cloud | unavailable
    "label": "comfortable",
    "expected_rtf": 0.033,            // null when nothing was measured or estimated
    "rtf_range": [0.017, 0.066],
    "rtf_basis": "measured",          // measured | estimated (from a sibling model) | unknown
    "scaled_by": "heuristic",         // benchmark | heuristic | null
    "device": "cpu",                  // cpu | gpu | cloud | null
    "memory": {"needed_bytes": 306067849, "available_bytes": 16869146624, "kind": "ram", "basis": "estimated"},
    "engine_available": true,
    "reason": "About 30x faster than real time here (…); needs about 0.3 GB of the 15.7 GB of memory.",
    "notes": [],
    "recommended": false,             // among the few recommended for its task, whatever the language
    "recommended_for": ["en"],        // languages it is among the recommended for; "*" = any
    "default_for": []                 // languages it is the default pick for
  }
}]}
```

`fit` is computed for the machine the server runs on
([recommendations.md](recommendations.md)). It never changes what can be
pulled or loaded: a model whose level is `unlikely` is listed like any other.

Feature tags: `partials` (interim results on the realtime WebSocket: every
speech-to-text model), `timestamps` (word times), `punctuation`, `translate`,
`language-detect`, `multi-voice`, `telephony` (the 8 kHz path was exercised
with this model).

`GET /api/v1/models/{id}` returns one such entry.

Entries with `"status": "pending"` describe models whose files the project
has not published yet (voices awaiting conversion). The catalog listing
includes them (`installed` is `false`; they are not in `GET /v1/models`), and
pulling or using one answers `404 model_not_published`.

Entries with `"runnable": false` are models that no runtime bolo has can run
yet; `not_runnable_reason` says why. They are listed so the catalog can say
so, and so their licence and source are on record.

Cloud entries (`"cloud": true`, `engine: "remote"`) have no files:
`installed` is `true` when their provider is configured (`provider.configured`),
`size_bytes` is 0, `license_url` links the provider's terms, and "pulling" one
only checks the provider. `"adhoc": true` marks a `<provider>:<model>` id that
was used by name and is not a curated entry.

### Model management

| Call | Result |
| --- | --- |
| `POST /api/v1/models/pull` `{"id": "…"}` | `202 {"job_id": "j_1"}`. `id` may be `hf:<owner>/<repo>[@rev]` with optional `"kind"`, `"task"`, or `piper:<voice>` (for example `piper:te_IN-venkatesh-medium`) for any voice of the Piper project, which is converted for sherpa-onnx after the download. Local directories are registered from the command line only (`bolo pull ./dir`) |
| `GET /api/v1/jobs/{job_id}` | `{"id","kind":"pull"\|"bench"\|"install","model","state":"queued"\|"running"\|"done"\|"error","progress":0..1,"bytes_done","bytes_total","error"}` (+ `result` for bench jobs) |
| `GET /api/v1/jobs` | all recent jobs, newest first |
| `POST /api/v1/models/{id}/load` | `200 {"id","loaded":true,"memory_bytes","load_ms","size_vram_bytes","vram_measured","vram_source","processor","device","placement","keep_alive_s"}` (+ `notice` when the model is not all on the accelerator that was asked for). Optional body `{"keep_alive": "10m"}` or `?keep_alive=`: how long it stays loaded once idle (`-1`: until unloaded; `0`: load and unload at once, answering `"loaded": false`, Ollama's way to unload). CLI: `bolo load <id> --keep-alive 10m` (also `bolo run`) |
| `POST /api/v1/models/{id}/unload` | `200 {"id","loaded":false,"unloaded":true\|false}` (`false`: it was not loaded). A request still using the model finishes first. CLI: `bolo stop <id>…` |
| `GET /api/v1/ps` | loaded models, Ollama style: see "Loaded models" below. CLI: `bolo ps` |
| `DELETE /api/v1/models/{id}` | `204`; unloads first. A preset: its entry and pinned settings go, the files stay. CLI: `bolo rm <id>…`, which refuses a model the running server has loaded unless `--force` (it unloads first). `bolo rm` and `bolo cp` go through the server at `--url` only when its `data_dir` (`GET /api/v1/info`) is the CLI's own; otherwise they change the CLI's data directory and say so |
| `GET /api/v1/models/{id}/show` | everything about one model, like Ollama's `/api/show`: the catalog entry plus `kind`, `source`, `options`, `path`, `files` (`[{"path","size","sha256"}]`), `modified_at`, `presets` (presets made from it), `effective` (`device`, `keep_alive_s`, `num_parallel`, `max_queue` and, for llama-server voices, `llama`: the flash attention, KV cache type, context and batch size it would start with), `speed` (`catalog`: the catalog's measurements; `server`: this server's requests, real-time factor and latency; `expected_rtf` here and its `rtf_basis`) and `loaded_placement`. `?license=true` adds `license_text`, the licence file shipped with the model (null without one). CLI: `bolo show <id>` (`--json`, `--license`) |
| `POST /api/v1/models/copy` `{"source": "kokoro-v1.0-int8", "destination": "kokoro-fast"}` | `201` and the new entry: a **preset**, like Ollama's `/api/copy`. `kokoro-fast` uses `kokoro-v1.0-int8`'s files (nothing is copied or downloaded; pulling it pulls the source) and starts with its catalog options and pinned settings, which can then be changed for the preset alone (`PUT /api/v1/models/kokoro-fast/settings`). Kept in `<data_dir>/custom.yaml` with `preset_of`; a preset of a preset points at the model with the files. `400` when the name is taken or not a valid id (lowercase letters, digits, `.`, `_`, `-`); cloud models cannot be copied. CLI: `bolo cp <source> <destination>` |
| `GET /api/v1/models/{id}/settings` | `{"model","settings":{…}}`: what the user pinned for that model (`{}` by default) |
| `PUT /api/v1/models/{id}/settings` `{"fallback": ["openai-whisper-1"]}` | pins the model's [fallback list](providers.md#fallback) (same task, local or cloud; `[]` switches a configured fallback off, `null` removes the pin) |
| `PUT /api/v1/models/{id}/settings` `{"device": "cpu", "keep_alive": "30m"}` | pins where the model runs (overrides the `device` setting for it; it still goes on the CPU when it does not fit on the accelerator) and how long it stays loaded once idle (overrides `keep_alive` and the per-task TTLs; `-1` keeps it loaded). `null` removes a pin. Takes effect at the next load |
| `PUT /api/v1/models/{id}/settings` `{"num_parallel": 2, "max_queue": 16}` | requests the model works on at once and requests that may wait before `429` (override the settings `num_parallel` and `max_queue` and the catalog's options; configuration.md, "Parallel requests and queues"). Takes effect at the next load |
| `PUT /api/v1/models/{id}/settings` `{"flash_attention": "on", "kv_cache_type": "q8_0", "context_size": 4096, "batch_size": 512}` | llama.cpp voices: what the llama-server bolo starts for the model gets (`-fa`, `--cache-type-k/-v`, `-c`, `-b`), over the `llama_*` settings and the catalog's options; arguments in `tts_server_args` win. `400` for an unknown cache type or flash-attention value. Takes effect at the next load. CLI: `bolo models config <id> --set kv_cache_type=q8_0` |
| `PUT /api/v1/models/{id}/settings` `{"phonemizer": "espeak-ng"}` | pins the phonemiser of a voice (`null` removes the pin); stored in `<data_dir>/model-settings.json`, shown as `settings` in catalog entries, announced as a `model` event with `"event":"settings"`. `400` for a model that reads characters, or uccharan on a language it does not cover. CLI: `bolo models config <id> --phonemizer X` / `--clear` |

Pulling an installed model yields a job that is already `done`. Pulling a
gated model without a token yields a job in state `error` that says what to do.

**Hardware notices.** When the model is unlikely to fit this machine or to
run in real time (`fit.level` is `slow` or `unlikely`), the answers of
`POST /api/v1/models/pull` and `POST /api/v1/models/{id}/load` carry an extra
`warnings` list, and so does the pull's job:

```jsonc
{"job_id": "j_4", "warnings": [{
  "code": "hardware_fit",             // or "disk_space": the download is larger than the free disk space
  "model": "parler-tts-tiny-v1", "level": "unlikely",
  "message": "'parler-tts-tiny-v1' is unlikely to fit or to run well on this machine. … This is a recommendation, not a restriction."
}]}
```

The status code does not change and nothing is refused: the pull has
started, the model is loaded. The key is absent when there is nothing to
say, and always absent with `hardware_warnings = false`. Requests served by
such a model have the same `warnings` on their trace.

### Keep-alive

As in Ollama, every request that uses a model can say how long the model stays loaded afterwards with
`keep_alive`: a number of seconds or a duration (`"30s"`, `"5m"`, `"1h30m"`), `0` to unload the model when the
request ends, any negative value (`-1`) to keep it loaded until it is unloaded on purpose or evicted to make room.
It is an extra field that OpenAI clients never send, so they are unaffected:

| Where | How |
| --- | --- |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | form field `keep_alive`; applies to the speech-to-text model and to the diarization and separation models the request uses (not to the VAD it uses on the side) |
| `POST /v1/audio/speech` | JSON field `keep_alive`; applies to the voice's model |
| `WS /api/v1/stt/stream`, `WS /api/v1/tts/stream`, `WS /v1/realtime` | query parameter `?keep_alive=`; applied when the session ends (so `0` unloads the models then, not between utterances); the Realtime API's applies to its speech-to-text and text-to-speech models |
| `POST /api/v1/models/{id}/load` | JSON body `{"keep_alive": …}` or `?keep_alive=` |

Without one, the model's pinned `keep_alive`, its catalog `options.keep_alive`, the per-task TTL (`stt_ttl_s`, …) or
the `keep_alive` setting (`5m`) decide, in that order. Unlike Ollama, a later request without `keep_alive` does not
reset a value an earlier request or load gave the model: send `keep_alive` again to change it. A malformed value is
`400` with `param: "keep_alive"`.

### Loaded models: `GET /api/v1/ps`

Like Ollama's `/api/ps` (bolo's `/api/ps` is speaches' route, which lists model ids, so the Ollama shape lives here):

```jsonc
{"models": [{
   "name": "whisper-large-v3-turbo", "model": "whisper-large-v3-turbo", "task": "stt",
   "size": 1717986918,                    // bytes: RAM and accelerator memory the load took (never less than the files)
   "size_vram": 1610612736,               // of which on the accelerator
   "size_ram": 107374182,
   "processor": "100% GPU",               // "100% CPU", or "35%/65% CPU/GPU" for a split model (by bytes)
   "device": "cuda:0",                    // cpu | cuda:N | coreml | ... | remote (an operator-run server) | cloud
   "split": "whole",                      // whole | layers (llama.cpp) | graphs (ONNX Runtime) | device_map (PyTorch)
   "expires_at": "2026-11-14T21:04:05Z",  // when it is unloaded if it stays idle; null: kept loaded
   "expires_in_s": 241.3, "until": "4 minutes from now",   // "Forever", "Stopping..."
   "keep_alive": "5m", "keep_alive_s": 300.0,
   "gpus": [0],                           // CUDA GPUs its weights are on ([0, 1]: llama.cpp layers spread); [] on the CPU
   "active": 0, "requests": 12, "idle_s": 58.7, "load_ms": 2110.4,
   "running": 0, "queued": 0,             // requests running and waiting in its queue
   "num_parallel": 1, "max_queue": 64,    // its queue's limits (configuration.md, "Parallel requests and queues")
   "vram_measured": true,                 // false: the accelerator share is the plan, not a measurement
   "vram_source": "process",              // process | dxgi | device | plugin | unified | plan (unknown, e.g. WDDM) | null (CPU)
   "latency_ms": {…}, "rtf": {…},         // as in /api/v1/stats
   "placement": {"device": "cuda:0", "requested": "cuda", "split": "whole", "gpu_bytes": 1610612736,
                 "cpu_bytes": 0, "processor": "100% GPU", "notice": null, "free_memory_known": true},
                 // llama.cpp: + "gpu_layers": 19, "total_layers": 29, "operator_layers": false
                 // ONNX Runtime graphs: + "graphs": {"encoder": "cuda:0", "decoder": "cpu"}
                 // always: "gpus": [0], "gpu_bytes_by_gpu": {"0": 1610612736}
                 // spread over GPUs: "gpus": [0, 1], "tensor_split": [0.6, 0.4], "main_gpu": 0
                 // the runtime did not start the GPU (a CUDA DLL missing): "device": "cpu", a "notice", and
                 // "provider_fallback": [{"runtime": "onnxruntime", "requested": "CUDAExecutionProvider",
                 //   "missing": ["cublasLt64_13.dll"], "message": "...", ...}]
   "degraded": null,                      // on a GPU that was lost: why (it is being moved to the CPU, or fails)
   "details": {"engine": "sherpa-onnx", "kind": "whisper", "family": "Whisper"}
 }],
 "system": {"cpu_percent": 12.5, "rss_bytes": 2147483648, "device": "cuda",
            "ram_total_bytes": 34359738368, "ram_available_bytes": 21474836480,
            "gpus": [{"index": 0, "vendor": "nvidia", "name": "NVIDIA GeForce RTX 3060", "backend": "cuda",
                      "total_bytes": 12884901888, "free_bytes": 9556148224, "used_bytes": 3328753664,
                      "unified": false, "utilization": 41.0,
                      "process_bytes": 2147483648,   // held by bolo and the servers and plugins it started
                      "source": "nvml",               // nvml | nvidia-smi | unified (Apple Silicon) ; [] when unknown
                      "physical_index": 0,            // NVML's number; "index" is CUDA's (after CUDA_VISIBLE_DEVICES)
                      "uuid": "GPU-5f0c…",
                      "driver_model": null,           // Windows: "WDDM" (process_bytes null), "TCC", "MCDM"
                      "own_bytes": null,              // Windows: bolo's own process's use, as DXGI says
                      "allowed": true,                // in the gpus setting (NVIDIA only)
                      "models_vram_bytes": 1610612736}],   // what bolo's loaded models hold on this GPU
            "accelerator_source": "nvml", "gpu_spread": false,
            "models_vram_bytes": 1610612736, "models_ram_bytes": 107374182,
            "gpu_health": {"state": "ok", "lost": false, …}},   // as "health.gpu" in GET /api/v1/info; also in /api/v1/stats
 "budgets": {"memory_budget_bytes": null, "gpu_memory_budget_bytes": null, "gpu_reserve_bytes": 536870912,
             "gpus": null, "gpu_spread": false, "max_loaded_models": null, "keep_alive": "5m"},
 "speech_to_speech": {"model": "lfm3-audio-2b-llamacpp", "placement": {…}}   // once its llama-server started
}
```

`processor` says where the weights are: a model placed whole on the accelerator reads `100% GPU` even though its
runtime holds some host memory too (that is in `size`). On Apple Silicon the GPU shares the machine's memory:
`unified` is true, `process_bytes` is null and `size_vram` is the plan. On Windows under WDDM the driver reports no
per-process memory: a load is measured by bolo's own DXGI use or the GPU's growth, and when neither works
`vram_source` is `"plan"` and `processor` still says where bolo placed the model (configuration.md, "Windows and
WDDM"). How models are placed: configuration.md,
"GPU and CPU placement".

`bolo ps` prints the same as a table:

```
cpu 12.5%  rss 2.0 GB  ram 12.0/32.0 GB  gpu0 NVIDIA GeForce RTX 3060 vram 3.1/12.0 GB (bolo 2.0 GB) 41% busy
MODEL                   TASK  SIZE    PROCESSOR        DEVICE  GPU  UNTIL
whisper-large-v3-turbo  stt   1.6 GB  100% GPU         cuda:0  0    4 minutes from now
orpheus-3b-llamacpp     tts   3.0 GB  35%/65% CPU/GPU  cuda:0  0    Forever
silero-vad              vad   2.0 MB  100% CPU         cpu     -    Forever
```

GPU is the CUDA number of each GPU a model is on (`0,1` for a llama.cpp model spread over two). `bolo ps -v` adds
ACTIVE (requests and sessions using the model), QUEUED (requests waiting in its queue), PARALLEL (`num_parallel`),
REQUESTS, P50 MS, RTF P50 and IDLE, and the placement notices; `--json` prints the document.

### Hardware and recommendations

Full description: [recommendations.md](recommendations.md). These routes
describe and advise; none of them restricts anything.

`GET /api/v1/hardware[?refresh=true|force]`: what bolo detected. A value it could
not read is `null`, which means "unknown", never "none". It is detected once
and kept, so a plain request answers at once. `refresh=true` looks again and
also runs the slower probes (the torch plugin's device, the server binaries'
versions, `llama-server --list-devices`, nvidia-smi); their answers are reused
for 60 s and forgotten when a component is installed, so repeating it is fast.
`refresh=force` runs the slow probes again at once.

```jsonc
{
  "system": "linux",
  "cpu": {"model": "Intel(R) Xeon(R) Processor @ 2.10GHz", "vendor": "GenuineIntel", "architecture": "x86_64",
          "physical_cores": 4, "logical_cores": 4,
          "usable_cores": 4.0,        // min(logical cores, CPU affinity, cgroup quota)
          "features": ["avx2", "avx512", "avx512_vnni", "amx", …]},   // or ["neon", "dotprod", …]
  "memory": {"total_bytes": 16869146624, "available_bytes": 12609163264,
             "limit_bytes": null,     // a cgroup limit below the machine's memory
             "usable_bytes": 16869146624},
  "disk": {"path": "/home/me/.cache/bolo", "total_bytes": 270553174016, "free_bytes": 8192065536},
  "gpus": [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4090", "vram_total_bytes": 25757220864,
            "vram_free_bytes": 24127733760, "driver": "550.54.14", "unified_memory": false,
            "backend": "cuda",        // cuda | metal | null (no runtime bolo starts uses it)
            "source": "nvidia-smi"}],
  "gpus_known": true,                 // false: GPU detection could not run; [] then means "unknown"
  "runtimes": {
    "onnxruntime": {"installed": true, "version": "1.29.0", "providers": ["CPUExecutionProvider"], "gpu": false},
    "sherpa-onnx": {"installed": true, "version": "2.0.1", "gpu": null},   // null: cannot tell
    "ctranslate2": {"installed": true, "version": "4.5.0", "cuda_devices": 0, "gpu": false},
    "plugin:torch": {"installed": false, "version": null, "torch": null, "device": null, "cuda": null, "probed": false},
    "llama-server": {"installed": false, "path": null, "source": null, "version": null, "gpu": null, "gpu_devices": null},
    "audio.cpp": {"installed": false, "path": null, "source": null, "version": null, "gpu": null, "gpu_devices": null}
  },
  "container": {"detected": false, "kind": null, "virtual_machine": true, "hypervisor": null,
                "cpu_quota": null, "memory_limit_bytes": null},
  "device": "cpu", "threads": 4,      // the `device` setting; inference threads per worker
  "notes": [],                        // e.g. a CPU quota below the thread count, a GPU the device setting does not use
  "detected_at": "2026-11-14T13:31:24Z", "deep": false,
  "summary": "4 cores (Intel(R) Xeon(R) Processor @ 2.10GHz, AMX), 15.7 GB RAM, no GPU found",
  "speed": {
    "factor": 1.0,                    // this machine against the reference machine; catalog RTFs are divided by it
    "source": "heuristic",            // "benchmark" once the micro-benchmark has run
    "spread": [0.5, 2.0],             // how far off it may be ([0.75, 1.33] with a benchmark)
    "note": "estimated from the CPU (…), not measured; `bolo recommend --benchmark` measures it",
    "benchmark": null,                // the stored result: {"workload","threads","ms","reference_ms","speed","at",…}
    "reference": "4 vCPUs of an Intel Xeon at 2.1 GHz (AVX-512, AMX), no GPU, 4 inference threads"
  }
}
```

`POST /api/v1/hardware/benchmark`: runs the micro-benchmark on the server (a
few seconds of CPU), stores it in `<data_dir>/hardware-benchmark.json` and
answers `{"benchmark": {…}, "speed": {…}}`. Fits use it from then on.

`GET /api/v1/recommendations?task=&language=&commercial=&local_only=&realtime=&max_size_mb=&limit=`:
the models of a task, best fit for this machine first. `task`:
`stt`, `tts`, `diarization`, `vad`, `turn`, `separation` or
`speech-to-speech` (`s2s`); anything else is `400`. `realtime=true` ranks for
live speech, `false` for batch work; `limit` cuts the list after ranking.
Without `task` the answer is an overview of every task:
`{"query": {…, "task": null}, "hardware", "speed", "tasks": {"stt": {"default", "recommended",
"total", "filtered_out", "note", "results"}, "tts": {…}, …}}`, each task with its first
`limit` results (3 by default) and the same filters applied.

```jsonc
{
  "query": {"task": "stt", "language": "hi", "commercial": false, "local_only": false, "realtime": null, "max_size_bytes": null},
  "hardware": "4 cores (…), 15.7 GB RAM, no GPU found",
  "speed": {…},                       // as in GET /api/v1/hardware
  "default": "indicconformer-hi-community",   // the default pick, or null when nothing runs here
  "recommended": ["indicconformer-hi-community", "…"],   // at most three, of different families
  "total": 37,
  "filtered_out": {"language": 93, "commercial": 0, "local_only": 0, "max_size": 0},   // hidden by YOUR filters
  "note": null,                       // set when nothing keeps up with real time
  "results": [{
    "rank": 1, "id": "indicconformer-hi-community", "task": "stt", "family": "IndicConformer",
    "engine": "sherpa-onnx", "tier": "balanced", "streaming": false, "languages": ["hi"],
    "size_bytes": 197000000, "installed": false,
    "license": "MIT", "commercial_use": true, "cloud": false, "validated": "cpu",
    "quality": {"metric": "wer", "label": "word error rate", "value": 0.152,
                "note": "measured by the maintainers on that model's own test audio; not comparable across models"},
    "recommended": true, "default": true,
    "fit": {…}                        // as on catalog entries
  }]
}
```

Hardware never removes a model from `results`: `total` plus the counts in
`filtered_out` is every catalog model of the task, and the models that are
unlikely to fit are at the end with their reason.

`GET /api/v1/recommendations/setup?use=&language=&commercial=&local_only=`:
one model per task for a use. `use`: `live` (STT and TTS for live speech,
plus optional VAD and end of turn), `files` (STT, plus optional diarization)
or `voiceover` (TTS). `local_only` defaults to `true` here.

```jsonc
{
  "use": "live", "label": "Live conversation", "language": "hi", "commercial": false, "local_only": true,
  "hardware": "…", "speed": {…},
  "picks": [{"task": "stt", "optional": false, "realtime": true,
             "model": {…},            // a row as in `results`, or null when nothing suits
             "alternatives": [{…}],   // the other recommended ones
             "note": null}],
  "pull": ["indicconformer-hi-community", "piper-hi-in-rohan-medium", "silero-vad"],   // chosen and not installed
  "download_bytes": 262000000
}
```

`GET /api/v1/info` announces all of this as `capabilities.recommendations`
(`restricts` is always `false`; `warnings` mirrors the `hardware_warnings`
setting).

### Cloud providers

`GET /api/v1/providers`, `GET/PUT/DELETE /api/v1/providers/{name}` and
`POST /api/v1/providers/{name}/test` list, add, change, remove and check the
cloud providers behind `engine: remote` models. Keys are write-only: no route
returns one. The routes, the stored file and who may change providers are
described in [providers.md](providers.md#the-providers-api).

```jsonc
// GET /api/v1/providers
{"providers": [{"name": "openai", "kind": "openai-compatible", "base_url": "https://api.openai.com/v1",
                "configured": true, "api_key_set": true, "options": {}, "prices": {"whisper-1": {"per_minute": 0.006}},
                "source": "settings",        // settings (file or environment; read-only) | stored (added through the API)
                "editable": false, "terms_url": "https://openai.com/policies/services-agreement/",
                "stats": {"whisper-1": {"requests": 12, "errors": 0, "latency_ms_mean": 640.2, "bytes_sent": 3841024,
                                        "bytes_received": 2210, "audio_seconds": 120.0, "chars": 0,
                                        "estimated_cost": {"USD": 0.012}, "last_error": null}}}],
 "kinds": [{"kind": "openai-compatible", "default_base_url": "https://api.openai.com/v1", "terms_url": "…"}, …],
 "editable": true, "edit_blocked_reason": null, "notice": "paid cloud API; audio leaves this server"}

// PUT /api/v1/providers/sarvam   {"api_key": "…"}            -> the provider, without the key
// POST /api/v1/providers/sarvam/test
{"name": "sarvam", "ok": true, "message": "…", "code": null, "checked": "reachability", "latency_ms": 212.4}
```

### Pronunciation word lists

Named lists in `<data_dir>/wordlists/<name>.tsv` (names `[a-z0-9_-]`, at most 64 characters),
in uccharan's format: tab-separated `word`, `lang`, `ipa`, `say`, `match`, `note`, where `match`
is `word` (whole words, any case; the default), `case`, `exact` (any substring) or `regex`, and
`lang` is a language tag or `*` (see uccharan's docs/wordlists.md). Parsing and matching need the
`uccharan` package (`424 component_missing` otherwise; listing and deleting still work). Limits:
10,000 entries and 1 MB per list, 8 lists and 500 inline entries per request.

- `GET /api/v1/wordlists` → `{"wordlists": [{"name", "entries", "languages", "bytes", "updated"}], "available"}`.
- `GET /api/v1/wordlists/{name}` → `{"name", …, "entries": [{"word", "lang", "ipa", "say", "match", "note", "line"}]}`;
  `?format=tsv` or `Accept: text/tab-separated-values` gives the file, `?format=pls&lang=hi` or
  `Accept: application/pls+xml` a W3C PLS 1.0 lexicon for one language (plus the entries for any language).
- `PUT /api/v1/wordlists/{name}` with a TSV, PLS (`alphabet="ipa"`) or JSON (`[…]` or
  `{"entries": […]}`) body replaces the list (stored as TSV) and answers its summary plus
  `warnings` (`[{"line", "message"}]`, e.g. an entry with neither `ipa` nor `say`). A parse error
  → `400` naming the line. Publishes a `wordlist` event.
- `DELETE /api/v1/wordlists/{name}` → `{"deleted": name}`; `404 wordlist_not_found` for an unknown list.
- `POST /api/v1/wordlists/{name}/preview` with `{"text", "models": [up to 8 TTS ids], "language"?,
  "voice"?, "input_format"?, "normalize"?, "romanised"?}` → `{"previews": [{"model", "voice", "language",
  "text_input", "normalizer", "text", "applied", "romanised", "phonemes"?}]}`: exactly the text each voice would
  receive (with `[word](/ipa/)` overrides for phoneme voices), the entries applied, and for phoneme
  voices the phonemes (one line per sentence) when a phonemiser is available. `name` `-` previews
  with the pinned and `default` lists only. Nothing is synthesised; send the same text to
  `/v1/audio/speech` with `wordlists` to hear it.
- `GET /api/v1/text-input/{model}?voice=` → `{"model", "voice", "text_input": {"reads", "normalizer",
  "markup", "latin", "respell", "digits", "wordlists", "romanised", "pronunciation"}}`.
- `PUT /api/v1/models/{id}/settings` also takes `text_normalizer` (`uccharan`, `light`, `model`,
  `none`), `wordlists` (a list of names, pinned in that order) and `romanised` (`off`, `auto`,
  `on`); `null` removes a pin.

### `GET /api/v1/stats`

```jsonc
{"uptime_s": 123.4, "active_streams": 2, "queue_depth": 0,
 "system": {"cpu_percent": 41.0,          // this process, as a share of all cores
            "rss_bytes": 734003200, "threads": 12,
            // the machine's memory and accelerators, as in GET /api/v1/ps
            "ram_total_bytes": 17179869184, "ram_available_bytes": 11811160064, "gpus": [],
            "accelerator_source": null, "models_vram_bytes": 0, "models_ram_bytes": 210000000},
 "totals": {"requests": 120, "errors": 1, "audio_seconds_in": 812.5, "audio_seconds_out": 96.2},
 "models": [{
   "id": "moonshine-tiny-en", "task": "stt", "loaded": true,
   "memory_bytes": 210000000,             // estimate: RSS growth while loading
   "idle_s": 3.2, "ttl_s": 300, "active": 1, "requests": 80, "errors": 0,
   "latency_ms": {"first_partial": {"p50": 180, "p95": 320, "p99": 410},
                  "final":         {"p50": 260, "p95": 480, "p99": 600},
                  "ttfa": null},
   "rtf": {"p50": 0.08, "p95": 0.15},
   // loaded models: placement and keep-alive, the fields of GET /api/v1/ps
   "size_vram_bytes": 0, "ram_bytes": 210000000, "vram_measured": false, "processor": "100% CPU", "device": "cpu",
   "placement": {…}, "keep_alive_s": 300.0, "expires_in_s": 296.8, "expires_at": "2026-11-14T21:04:05Z",
   "stopping": false,
   "queue": {"depth": 0, "inflight": 1, "workers": 1, "num_parallel": 1, "max_queue": 64, "batches": 61, "mean_batch": 1.31, "max_batch": 4,
             "rejected": 0, "queue_ms": {"p50": 0.1, "p95": 38.0, "p99": 71.2}}
 }],
 "separation": {"requests": {"applied": 0, "partial": 0, "skipped": 0}, "audio_seconds": 0.0},
 // phone calls: in progress, counted since the server started, and finished calls by provider and outcome
 "telephony": {"active_calls": 0, "calls_total": 3,
               "by_outcome": [{"provider": "twilio", "outcome": "caller_hangup", "calls": 3}]},
 // cloud providers, per provider and provider model (empty without cloud requests)
 "providers": [{"provider": "openai", "model": "whisper-1", "requests": 12, "errors": 0, "latency_ms_mean": 640.2,
                "latency_ms_total": 7682.4, "bytes_sent": 3841024, "bytes_received": 2210, "audio_seconds": 120.0,
                "chars": 0, "estimated_cost": {"USD": 0.012}, "last_error": null}],
 "fallbacks": [{"requested": "whisper-base", "answered": "openai-whisper-1", "count": 2}]}
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

Traces carry extra fields where they apply: `provider` and `cost` (a request
served by a cloud model: requests, time and bytes at the provider, and the
estimated cost when the provider's price table knows the model),
`fallback` (`{requested, answered, attempts: [{model, code, message}]}` when
another model answered; `model` is then the one that answered), and on
voice-agent traces (`kind: "agent"`) `pipeline`: which pipeline answered the
turn (`cascade`, `audio-llm`, `speech-to-speech`; `transcription` for a
transcription-only session; `cascade` for `/v1/chat/completions`). See
[providers.md](providers.md#traces-metrics-and-cost-estimates).

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

For a model that clones voices, the result also has `speaker_similarity`: one
more synthesis after the sweep, compared with the voice's reference by the
speaker-embedding model (`voice` a saved voice, or a catalog voice with a
reference clip), when that model is installed:
`{"similarity": 0.62, "verdict": "same speaker", "embedding_model":
"pyannote-3.0-eres2net", "calibrated": true, "reference": "saved voice priya"}`,
or `{"similarity": null, "note": "…"}`.

### Saved voices: `/api/v1/voices`

A library of cleaned reference recordings for the models that clone voices;
[voices.md](voices.md) explains the workflow, the consent rule and the
similarity score.

| Route | |
| --- | --- |
| `POST /api/v1/voices` | multipart: `file` (one or more recordings of the same person, also `files[]`), `name`, **`consent`** (required: `self` = "this is my voice", `permission` = "I have the speaker's permission"; anything else → `400 consent_required`), optional `id`, `transcript` (else transcribed by the default STT model when it is installed; `transcribe=false` skips that), `language`, `description`, `default_model` (a cloning model), `created_by` (a name to record), `score_model` (speak a test sentence with it and return `score`; default: `default_model`), `score_audio=true` (include the test clip). → `201` with the voice. `409 voice_exists` for an id that is taken; the id comes from the name (`"Priya S."` → `priya-s`), and OpenAI's voice names are reserved |
| `GET /api/v1/voices` | `{"voices": [...], "cloning_models": [...], "consent": {...}, "similarity": {thresholds, calibration}}` |
| `GET /api/v1/voices/{id}` | one voice |
| `PATCH /api/v1/voices/{id}` | JSON: `name`, `description`, `language`, `transcript`, `default_model` (`""` clears it) |
| `DELETE /api/v1/voices/{id}` | removes the voice and its recording |
| `GET /api/v1/voices/{id}/audio` | the cleaned reference (WAV, 24 kHz mono) |
| `GET /api/v1/voices/{id}/score?model=&text=&audio=` | speaks `text` (default: a test sentence in the voice's language) with `model` (default: the voice's default model, else the default TTS model) and compares it with the reference: `similarity`, `verdict`, `embedding_model`, `calibrated`, `rtf`, `ttfa_ms`, `audio_s`, and with `audio=true` the clip as a data URL. Kept in the voice's `scores` |
| `GET /api/v1/voices/export?ids=a,b` | a .zip of `manifest.json` and `<id>/voice.json` + `<id>/reference.wav` (all voices without `ids`) |
| `POST /api/v1/voices/import` | multipart `file` (an export), `consent` (for voices whose metadata has none; refused otherwise), `replace=true` |

A voice:

```jsonc
{"id": "priya", "object": "voice", "name": "Priya", "language": "en", "description": null,
 "transcript": "the words of the recording", "transcript_source": "auto", "default_model": "pocket-tts-en-llamacpp",
 "created_at": "2026-11-15T23:40:58Z",
 "consent": {"basis": "self", "statement": "This is my voice.",
             "by": {"auth": "api_key", "name": "asha", "client": "10.0.0.4"}, "at": "2026-11-15T23:40:58Z"},
 "reference": {"sample_rate": 24000, "duration_s": 14.2, "speech_s": 12.9, "input_s": 18.0, "clips": 1,
               "snr_db": 38.5, "level_dbfs": -24.1, "gain_db": 4.1, "speaker_consistency": 0.71},
 "warnings": [],                                  // short, noisy, clipped, several people, cut to 60 s
 "scores": {"pocket-tts-en-llamacpp": {"similarity": 0.66, "verdict": "same speaker", "rtf": 0.47, ...}},
 "audio_url": "/api/v1/voices/priya/audio"}
```

`by.auth` is `api_key` when the server has an API key (bolo has one key, so
the name is what the client sends in `created_by`), `none` otherwise, `cli`
for `bolo voices add`.

#### OpenAI's custom voices

As in the OpenAI API reference (the openai-python SDK's `client.audio.voices`
and `client.audio.voice_consents`):

| Route | |
| --- | --- |
| `POST /v1/audio/voice_consents` | multipart `recording` (the speaker reading a consent statement, up to 10 MiB), `name`, `language` (BCP 47) → `{"id": "cons_…", "object": "audio.voice_consent", "created_at", "name", "language"}`. bolo stores the recording; it does not check what is said |
| `GET /v1/audio/voice_consents`, `GET`/`POST`/`DELETE /v1/audio/voice_consents/{id}` | list, retrieve, rename (`{"name"}`), delete |
| `POST /v1/audio/voices` | multipart `audio_sample`, `name`, `consent` (a consent recording id, or bolo's `self` / `permission`) → `{"id", "object": "audio.voice", "created_at", "name", "type": "audio_sample"}`. Voices from a text prompt (`type: prompt`) are refused |
| `DELETE /v1/audio/voices/{id}` | removes the voice |

```python
consent = client.audio.voice_consents.create(name="Priya", language="en-US", recording=open("consent.wav", "rb"))
voice = client.audio.voices.create(name="Priya", consent=consent.id, audio_sample=open("priya.wav", "rb"))
client.audio.speech.create(model="pocket-tts-en-llamacpp", voice={"id": voice.id}, input="Hello.")
```

bolo has no ElevenLabs-compatible server routes (ElevenLabs is one of its
cloud providers, not a surface it imitates), so there is no ElevenLabs-style
voice creation route.

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
`wav`, `mp3`, `opus`, `flac`, `aac`), `sample_rate`, `speed`, `language`,
`phonemizer`, `normalize` (`uccharan`, `light`, `model`, `none`/`false`),
`wordlists` (comma-separated names; an empty value applies none) and
`romanised` (`off`, `auto`, `on`), as for `/v1/audio/speech`. A `romanised=on`
that cannot be done is an `error` message with code `component_missing`.

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

Who spoke when, without PyTorch. Two kinds of diarizer are in the catalog:

| model | how | download | choose it with |
| --- | --- | --- | --- |
| `pyannote-3.0-eres2net` (default) | pyannote's segmentation-3.0 model and a speaker-embedding model (both ONNX, run by sherpa-onnx), then clustering | 32 MB | nothing (default), or `diarization_model` / `model` |
| `nemotron-3-diarization` | NVIDIA's Nemotron-3-Diarization, an end-to-end model that gives every 10 ms the probability that each of up to 8 speakers is talking (ONNX Runtime) | 147 MB | `diarization_model=nemotron-3-diarization` on `/v1/audio/transcriptions`, `model=nemotron-3-diarization` on `/api/v1/diarize`, or `default_diarization` in the configuration |

Both are pulled on first use. Which one to use, with measurements, is in
models.md, "Speaker diarization"; in short, Nemotron does better with
overlapping speech and similar voices and finds the number of speakers more
often, at a fifth of the cost; on short, clean turn-taking dialogues at
telephone quality it can merge two speakers into one. `nemotron-3-diarization` needs the
`onnxruntime` Python package (installed with `bolo-speech[all]`, which brings it with
faster-whisper; otherwise `pip install onnxruntime`) and, for named speakers
and live streams, sherpa-onnx for its speaker embeddings.

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

`threshold` (`diarization_threshold` on transcriptions) means different things
to the two kinds of model: for the pyannote pipelines it is the clustering
threshold (higher: fewer speakers); for `nemotron-3-diarization` it is the
speech probability above which a speaker counts as talking (default 0.5;
lower: more speech, more false alarms). With Nemotron, `num_speakers` and
`max_speakers` fold the speakers who talk least into the others, but the
model cannot be made to hear more speakers than it does, so `min_speakers`
(and a `num_speakers` above what it found) has no effect; it handles at most
8 speakers.

```json
{"model": "pyannote-3.0-eres2net", "duration": 21.5, "speakers": ["A", "B"],
 "segments": [{"start": 0.03, "end": 2.98, "speaker": "A"}, {"start": 3.7, "end": 6.4, "speaker": "B"}],
 "overlaps": [],
 "request_id": "r_000012"}
```

**Overlapping speech.** The segmentation model (pyannote segmentation-3.0) can
assign two speakers to the same moment, and so can Nemotron (its speakers'
probabilities are independent), and the pipeline then returns turns
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
use a diarization-conditioned model or `separate_overlaps` (next two
paragraphs).

**Diarization-conditioned recognition.** With a speaker-conditioned model
(`model=dicow-v3.3`; needs the bolo-torch plugin, `pip install
"bolo-torch[dicow]"`; or the same model without PyTorch in bolo's core,
`dicow-v3.3-onnx`) a diarized transcription is not cut at speaker turns.
bolo diarizes the recording, then the model transcribes the whole recording
once per speaker, told every 20 ms whether that speaker, someone else, both or
nobody is talking, so overlapping speech is transcribed for each speaker. The
response is the usual `diarized_json` / `verbose_json` (segments with
`speaker`, `overlap`); segments of different speakers may overlap in time.
Such a model brings its own diarizer, `nemotron-3-diarization`, unless the
request names one with `diarization_model`; with a clustering diarizer
(`pyannote-*`), which gives no frame probabilities, the model is conditioned
on its speaker turns. `separate_overlaps` cannot be combined with it (400,
`param: "separate_overlaps"`). Without diarization the model transcribes like
an ordinary Whisper. On CPU it is several times slower than the default
pipeline; measurements in models.md, "Diarization-conditioned recognition".
`GET /api/v1/info` lists such models under
`capabilities.diarization.speaker_conditioned_models`.

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
does two things about it, and you can do a third (all of this is about the
pyannote pipelines; `nemotron-3-diarization` counts speakers itself and got
the count right far more often in our tests, see models.md):

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
transcription sessions, and Realtime conversations on the `speech-to-speech`
pipeline). The full description, the supported events and the
differences from OpenAI are in [voice-agents.md](voice-agents.md).

| Endpoint | Summary |
| --- | --- |
| `WS /v1/realtime?model=…` | OpenAI Realtime API over WebSocket (GA event names, as typed in `openai` 3.50.0). `?intent=transcription` or `session.update` with `type: "transcription"` for transcription-only sessions. Audio: base64 `audio/pcm` (24 kHz PCM16), `audio/pcmu`, `audio/pcma`. User transcripts arrive as incremental deltas. `turn_detection`: `server_vad`, `semantic_vad` (with the optional end-of-turn model) or `null`. Auth: `Authorization: Bearer`, or subprotocols `realtime` + `openai-insecure-api-key.<key>`; an ephemeral `ek_…` key is accepted in the same places. bolo extension `session.pipeline`: `cascade` (default, setting `realtime_pipeline`), `audio-llm` (an audio-understanding model hears the turn, bolo's TTS speaks) or `speech-to-speech` (a speech-to-speech model, first backend llama-server `/v1/realtime` with LFM3-Audio, answers with speech and tool calls; bolo keeps turn detection, barge-in, codecs and transcripts); see [voice-agents.md](voice-agents.md#pipelines-cascade-audio-in-language-model-speech-to-speech). `GET /api/v1/info` lists them under `capabilities.realtime.pipelines` |
| `POST /v1/realtime/client_secrets` | mint an ephemeral key (`{"value": "ek_…", "expires_at", "session"}`) for browsers. Needs the server key. Body: `{"expires_after": {"anchor": "created_at", "seconds": 600}, "session": {…}}`, all optional |
| `POST /v1/realtime/calls` | the same sessions over **WebRTC**: SDP offer in (`application/sdp`, or multipart `sdp` + `session`), `201` with the SDP answer and `Location: /v1/realtime/calls/{call_id}`. Needs `bolo-speech[webrtc]` (else `503 engine_unavailable`). `POST …/{call_id}/hangup` ends the call. Not implemented: OpenAI's SIP connector (for phone calls see [Phone calls](#phone-calls)) |
| `POST /v1/chat/completions` | `input_audio` content parts are transcribed; `modalities: ["text","audio"]` + `audio: {voice, format}` adds a spoken reply (`message.audio`, or `delta.audio` when streaming); everything else is forwarded to the language model untouched; requests without audio are proxied as they are |

Voice-agent turns are traces of kind `agent` (transport `realtime`, `webrtc`,
`chat` or `telephony`) with `first_token_ms` and `ttfa_ms`; see
[voice-agents.md](voice-agents.md#latency-and-where-to-look).

## Phone calls

Telephony providers and SIP trunks connected to voice agents; the full
description (setup per provider, security, agents, the simulator, what was
verified) is in [telephony.md](telephony.md).

Called by the providers. **No API key**: each request is checked against the
provider's signature (Twilio `X-Twilio-Signature`, Plivo
`X-Plivo-Signature-V3`, Telnyx `telnyx-signature-ed25519`) or the Basic
credentials it sends (Exotel), and a provider's routes answer
`503 telephony_not_configured` until that check is possible or
`telephony.validate_signatures = false`. A wrong signature answers `403`
(`401` for Exotel) with code `invalid_signature`. These routes are not in the
OpenAPI schema.

| Endpoint | Summary |
| --- | --- |
| `POST /telephony/{provider}/voice[/{token}]` | the answer webhook. Returns TwiML `<Connect><Stream>` (Twilio), Plivo XML `<Stream bidirectional keepCallAlive>`, or `200 {}` after sending Telnyx the `answer` command; `GET /telephony/exotel/voice` returns `{"url": "wss://…"}`. `{token}` marks a call bolo placed |
| `WS /telephony/{provider}/stream/{token}` | the media stream, in the provider's JSON framing. The one-time token comes from the answer webhook; without a valid one the handshake is refused (`403`). `WS /telephony/exotel/stream` takes Basic credentials instead |
| `POST /telephony/{provider}/after/{token}` | what happens once bolo closed the stream: `<Hangup/>` or `<Dial>` (Twilio, Plivo); `GET /telephony/exotel/after` answers `200` (transfer) or `302` |
| `POST /telephony/{provider}/status[/{token}]`, `POST /telephony/plivo/amd/{token}` | call-status and machine-detection callbacks |

The operator's API:

| Endpoint | Summary |
| --- | --- |
| `GET /api/v1/telephony` | `{public_base_url, validate_signatures, default_agent, providers: [{name, label, configured, exposed, problems, protection, can_validate, can_place_calls, machine_detection, ends_by, settings, webhooks: [{name, where, url, method}], stream_url, notes, docs_url}], agents, numbers, editable, edit_blocked_reason, llm_configured, stats, compliance_notice}`. `settings` shows `***` for a secret that is set, never a value |
| `GET /api/v1/telephony/agents`, `GET` / `PUT` / `DELETE /api/v1/telephony/agents/{name}` | voice agents (fields in [telephony.md](telephony.md#agents)). `PUT` replaces the agent; agents from the configuration are read-only (`400`); stored in `<data_dir>/telephony.json` |
| `GET /api/v1/telephony/numbers`, `PUT` / `DELETE /api/v1/telephony/numbers/{number}` | the called number → agent map; `PUT` body `{"agent": "<name>"}` |
| `GET /api/v1/telephony/calls?limit=50&provider=&agent=&status=` | recent calls, newest first: `{calls: [{id, provider, direction, agent, from, to, status, outcome, duration_s, turns, interruptions, latency, summary, …}]}` |
| `GET /api/v1/telephony/calls/{id}` | one call with `timeline`, `turns` (each with `latency: {endpoint_ms, stt_ms, llm_first_token_ms, tts_first_audio_ms, send_ms, total_ms, response_ms}`), `transcript` (`speaker`: `caller` / `agent`), `dtmf`, `transfer_to`, `summary`. `404` once it rotated out |
| `POST /api/v1/telephony/calls` | place a call: `{"provider", "to", "agent", "from", "metadata"}` → `202` with the call record and a `compliance_notice`. `503 telephony_not_configured` names missing credentials; `502 telephony_provider_error` carries the provider's refusal (credentials removed) |
| `POST /api/v1/telephony/calls/{id}/hangup` | end a call |
| `GET /api/v1/telephony/scenarios`, `POST /api/v1/telephony/simulate` | the call simulator: `{"agent", "provider", "scenario", "noise_db"}` → a report `{passed, checks, caller: {utterances, response_ms, agent_audio_s, clears}, latency, outcome, record}`. Runs in real time (10 to 20 s per scenario) |

`GET /api/v1/stats` has `telephony: {active_calls, calls_total, by_outcome}`;
`/metrics` has `bolo_telephony_active_calls` and
`bolo_telephony_calls_total{provider,outcome}`; `GET /api/v1/events` sends
`{"type": "call", "event": "started" | "ended" | "summary" | "dialing", "call": {…}}`.

## Optional components

Pieces bolo can use but does not require: two phonemisers, **uccharan**
(Apache-2.0, part of `bolo-speech[all]`; its 39 languages, see
[models.md](models.md#phonemisers)) and **eSpeak NG**
(GPL-3.0-or-later, never installed by default; every other language), uccharan's
language packs, and
engine plugins (`plugin:<name>`, [plugins.md](plugins.md)), which are listed so
that a client can say what a model needs but are installed as packages.
Find items by `id`; `kind` is `phonemizer`, `language-pack`, `engine-plugin` or `tts-server`.
`POST /api/v1/components/uccharan/install` needs no licence acceptance.

The `language-pack` components are uccharan's packs, one per pack the installed uccharan lists
(35 in uccharan 0.6): `uccharan-de`, `uccharan-en-gb`, `uccharan-kokoro-en-us`, …,
`uccharan-wikipron-hi`, `uccharan-xlit-hi`. Their items add `size_bytes`, `languages` and
`auto_download` (whether bolo fetches the pack by itself the first time a voice needs it: true
for the Apache-2.0 word tables while the `uccharan_auto_download` setting is on). Such a download
is a `pull` job (`model`: the component id) with the usual `job` events, a `component` event
`installed` and a log line. A pack is downloaded from uccharan's release, checked against the
SHA-256 that is part of the installed uccharan, kept in uccharan's cache
(`~/.cache/uccharan/<version>/`; `$UCCHARAN_DATA_DIR` for machines without network access) and
used without a restart. Install one ahead of time with
`POST /api/v1/components/uccharan-de/install`, `POST /api/v1/models/pull` with
`{"id": "uccharan-de"}`, `bolo pull uccharan-de` or `bolo components install uccharan-de`; without
bolo, `python -m uccharan.packs de`.

The `wikipron-*` packs (Wiktionary pronunciations for uccharan's improved mode) and the
`xlit-*` packs (data for the `romanised` option of text-to-speech) are licensed CC BY-SA 4.0:
bolo never downloads them on its own, and installing one needs `{"accept_license": true}`
(`--accept-license` on the command line) like eSpeak NG.

The `tts-server` components are the speech servers of the `tts-server` engine:
`llama-server` (llama.cpp, MIT) and `audio.cpp` (`audiocpp_server`,
Apache-2.0). Their items add `path` (the binary bolo would start, or null),
`servers` (URLs configured for that runtime in `tts_servers`) and `release`
(`{"version","url","sha256"}` of the prebuilt upstream archive bolo installs on
this platform, or null), `flavour` (the build that runs, from bolo's components
folder: `cpu`, `cuda` or `vulkan`; null for a binary from the settings or
`PATH`), `flavours` (the builds installed) and `wanted_flavour` (the one
`device` asks for). `POST /api/v1/components/llama-server/install` (no
licence acceptance needed; optional `{"flavour": "cuda"}`, else the build
`device` asks for) downloads that archive into
`<data_dir>/components/<id>/<version>/<flavour>/`, checks its SHA-256, unpacks it and
checks that the binary starts here; the job and events are as below. `400` when
no release is pinned for the platform (or no build of that flavour), the operator disabled installs or the
server is exposed without an API key. A runtime is not needed for a model that
`tts_servers` points at a running server (its `requires` is then `[]`).

`GET /api/v1/components`

```json
{"components": [{
  "id": "espeak-ng", "name": "eSpeak NG", "installed": false,
  "source": null,                       // "system-executable" | "python-package" | "configured" | null
  "version": null, "license": "GPL-3.0-or-later",
  "license_url": "https://github.com/espeak-ng/espeak-ng/blob/master/COPYING",
  "required_by_count": 82,              // catalog models that need it
  "installable": true, "install_blocked_reason": null,
  "manual_command": "pip install \"bolo-speech[espeak]\"",
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
for a model on an engine plugin; `["llama-server"]` or `["audio.cpp"]` for a model on an external TTS server
that bolo would start; else `[]`), `available` (false while that is
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
| `GET /healthz` | `200 {"status":"ok"}` as soon as the process serves requests. While the GPU is lost (configuration.md, "A lost GPU") still 200, with `{"status":"degraded","gpu":{...},"notice":"GPU LOST (...): reboot to recover; bolo is using the CPU"}` |
| `GET /readyz` | `200 {"status":"ready"}` once every `preload` model is in memory, else `503 {"status":"starting","loading":[…]}` |
| `GET /metrics` | Prometheus text: `bolo_requests_total`, `bolo_request_errors_total`, `bolo_audio_seconds_total{direction}`, `bolo_active_streams`, `bolo_queue_depth`, `bolo_process_*`, and per model `bolo_model_loaded`, `bolo_model_memory_bytes`, `bolo_model_requests_total`, `bolo_latency_ms{metric,quantile}`, `bolo_real_time_factor{quantile}`, `bolo_model_queue_depth`, `bolo_model_mean_batch_size`, `bolo_queue_time_ms{quantile}`, `bolo_model_queue_rejected_total` |
| `GET /` | the web console (single-page app; unknown paths fall back to it). Served with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` so its in-browser recogniser can use WebAssembly threads; turn off with `cross_origin_isolation = false` |
| `GET /docs`, `GET /openapi.json` | interactive API page and schema |
