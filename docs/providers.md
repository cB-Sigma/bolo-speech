# Cloud providers

bolo runs speech models itself. It can also hand a request to a cloud speech API and treat the
answer as if a local model had produced it. That is the `remote` engine: a model with
`engine: remote` names a *provider* (an account at an API) and the provider's own model name.

**Every cloud model is a paid cloud API, and audio (or text) leaves this server.** The catalog
marks such models `cloud: true`, the console labels them, and nothing is ever sent to a provider
unless a request names a cloud model, a cloud model is set as a default, or a model's `fallback`
list names one.

What is on this page was read from each provider's documentation on **2026-11-14**. Nothing was
run against a real account: there were no keys where bolo is developed, and the test-suite uses
stand-ins for the four APIs (`server/tests/fake_providers.py`). The opt-in real tests are
described at the end. Where the documentation left something open, it says so under
"Not verified".

- [Configuring a provider](#configuring-a-provider)
- [Models](#models)
- [What bolo adds on top](#what-bolo-adds-on-top)
- [Request fields, provider by provider](#request-fields-provider-by-provider)
- [Errors](#errors)
- [Fallback](#fallback)
- [Traces, metrics and cost estimates](#traces-metrics-and-cost-estimates)
- [Cloud speech-to-speech (OpenAI Realtime)](#cloud-speech-to-speech-openai-realtime)
- [Keys](#keys)
- [The providers API](#the-providers-api)
- [What was verified, and what was not](#what-was-verified-and-what-was-not)
- [Real tests](#real-tests)

## Configuring a provider

A provider has a name, a `kind`, a base URL, an API key, options and an optional price table.

| `kind` | Speaks | Default `base_url` | Key header |
| --- | --- | --- | --- |
| `openai-compatible` | OpenAI's audio API: OpenAI itself and every server that copies it (another bolo, speaches, vLLM, LocalAI, …) | `https://api.openai.com/v1` | `Authorization: Bearer <key>` |
| `sarvam` | Sarvam AI (Saaras, Bulbul) | `https://api.sarvam.ai` | `api-subscription-key: <key>` |
| `elevenlabs` | ElevenLabs (text to speech, Scribe) | `https://api.elevenlabs.io` | `xi-api-key: <key>` |
| `deepgram` | Deepgram (Nova) | `https://api.deepgram.com` | `Authorization: Token <key>` |

In the configuration file, one table per provider:

```toml
[providers.openai]                 # the names openai, sarvam, elevenlabs and deepgram imply their kind
api_key = "…"

[providers.sarvam]
api_key = "…"

[providers.office-bolo]            # another bolo (or any OpenAI-compatible server) on the network
kind = "openai-compatible"
base_url = "http://10.0.0.5:8000/v1"
# api_key = "…"                    # only if that server asks for one

[providers.openai.prices]          # optional: what you pay, for cost estimates (bolo ships no prices)
"whisper-1"       = { per_minute = 0.006 }
"gpt-4o-mini-tts" = { per_1m_chars = 0.60, per_minute = 0.015 }
"*"               = { per_request = 0.0 }

[providers.office-bolo.options]
timeout_s = 20
```

In the environment:

```bash
BOLO_PROVIDER_OPENAI_API_KEY=…            # BOLO_PROVIDER_<NAME>_API_KEY / _BASE_URL / _KIND
BOLO_PROVIDER_OFFICE_BOLO_KIND=openai-compatible
BOLO_PROVIDER_OFFICE_BOLO_BASE_URL=http://10.0.0.5:8000/v1
BOLO_PROVIDERS='{"deepgram": {"api_key": "…"}}'     # or the whole table as JSON
```

(`_` in the environment name is `-` in the provider's name. The `BOLO_PROVIDER_…` variables add
to, and override, the same provider's table from the file.)

Or at run time, on the console's **Providers** page or with `PUT /api/v1/providers/{name}`; those
are stored in `<data_dir>/providers.json` (see [Keys](#keys)).

```bash
bolo providers list            # name, kind, configured or not, where it came from, address (never a key)
bolo providers test openai     # one cheap request with the stored key
bolo doctor                    # also lists the providers
```

A provider is *configured* when it has a key. An `openai-compatible` provider at an address other
than OpenAI's is configured without one, since self-hosted servers often need none.

Provider options: `timeout_s` (default: the setting `provider_timeout_s`, 60 s),
`ignore_unsupported` (drop request options the provider has no equivalent for, instead of
answering 400), `organization` and `project` (`openai-compatible`: sent as `OpenAI-Organization`
and `OpenAI-Project`), `models` (per-model options for [ad-hoc ids](#any-other-model-providermodel)).

## Models

### Catalog entries

The catalog carries the well-known models of each provider (`server/src/bolo/catalog/cloud.yaml`;
the full table is in [models.md](models.md)). They need no download: an entry is **installed as
soon as its provider is configured**, and `GET /v1/models` lists it from then on. Each entry has

- `engine: remote`, `cloud: true`, and `notice: "paid cloud API; audio leaves this server"` in
  the API;
- the provider's terms in the licence field (`license`, with the link in `license_url`). Terms
  are labelled, as licences are everywhere in bolo; they are never a reason to leave a model out;
- languages and voices as the provider documents them;
- `provider: {name, kind, model, configured, price}` in the API.

| id | Provider model | Task | Notes |
| --- | --- | --- | --- |
| `openai-whisper-1` | `whisper-1` | STT | segment and word timestamps, translation; OpenAI announced its shutdown for 2027-02-26 |
| `openai-gpt-4o-transcribe`, `openai-gpt-4o-mini-transcribe` | same | STT | text only; streamed from OpenAI; shutdown announced for 2027-02-26 |
| `openai-gpt-transcribe` | `gpt-transcribe` | STT | hotwords as `keywords` |
| `openai-tts-1`, `openai-tts-1-hd`, `openai-gpt-4o-mini-tts` | same | TTS | streamed 24 kHz PCM; shutdown announced for 2027-01-06 |
| `sarvam-saaras-v4` | `saaras:v4` | STT | 22 Indian languages and Indian English, hotwords, translation |
| `sarvam-saaras-v3` | `saaras:v3` | STT | modes `transcribe`, `translate`, `verbatim`, `translit`, `codemix` |
| `sarvam-saaras-v4-streaming` | `saaras:v4` | STT | Sarvam's streaming socket (one transcript per utterance) |
| `sarvam-bulbul-v4-flash`, `sarvam-bulbul-v3`, `sarvam-bulbul-v2` | `bulbul:…` | TTS | 11 languages; 222, 37 and 7 voices |
| `elevenlabs-flash-v2.5`, `elevenlabs-multilingual-v2`, `elevenlabs-v3` | `eleven_…` | TTS | the voices of your account, by id or name; the catalog entry lists them once the model has been loaded |
| `elevenlabs-scribe-v3`, `elevenlabs-scribe-v2` | `scribe_…` | STT | word timestamps, hotwords (billed at a surcharge by ElevenLabs) |
| `deepgram-nova-3`, `deepgram-nova-2` | `nova-…` | STT | batch and live (Deepgram's streaming socket, with interim text) |

An entry looks for a provider with the name in its `options.provider` (`openai`, `sarvam`,
`elevenlabs`, `deepgram`). If you called the account something else, the first configured
provider of the same kind at the vendor's own address is used.

### Any other model: `provider:model`

Every model of a configured provider can be used without a catalog entry, as
`<provider name>:<the provider's model name>`:

```bash
curl -s localhost:8000/v1/audio/transcriptions -F file=@call.wav -F model=openai:gpt-transcribe
curl -s localhost:8000/v1/audio/transcriptions -F file=@call.wav -F model=sarvam:saarika:v2.5 -F language=ta
curl -s localhost:8000/v1/audio/speech -d '{"model":"office-bolo:kokoro-v1.0-int8","input":"Hello.","voice":"af_heart"}' \
  -H 'Content-Type: application/json' -o hello.mp3
```

The task comes from the route that is used. Such a model has no voice list (any `voice` is
passed on) and default options; give it options on the provider:

```toml
[providers.office-bolo.options.models."whisper-large-v3"]
response_format = "verbose_json"
word_timestamps = true
```

### Model options

Catalog `options` of an `engine: remote` entry (and of `options.models.<model>` above):

| Option | Meaning |
| --- | --- |
| `provider`, `provider_kind`, `model` | which account, which adapter if the name is not found, the provider's model name |
| `upload_format` | `wav` (default) or `flac` for the audio bolo uploads (16 kHz mono) |
| `max_chunk_s` | longest piece of audio sent in one request; bolo cuts longer recordings at pauses. Default 600 (30 for Sarvam, whose route is meant for 30 s) |
| `workers` | parallel requests to the provider for this model (catalog entries: 4) |
| `timeout_s` | overrides the provider's |
| `ignore_unsupported` | drop request options the provider lacks instead of answering 400 |
| `text_normalizer` | `uccharan` / `light` (bolo prepares the text), `model` (the provider normalises), `none` |
| `default_voice`, `list_voices` | voice when a request names none; read the account's voices when the model loads (ElevenLabs) |
| `fallback` | models that answer instead when this one cannot ([Fallback](#fallback)) |
| `interim_decodes` | `true`: on live audio, decode the growing utterance repeatedly for interim text (every decode is a billed request). Default: the final decode only |
| `request` | extra fields merged into every request to the provider |
| `openai-compatible`: `response_format`, `word_timestamps`, `translate`, `stream`, `prompt`, `hotwords_field`, `instructions`, `speed`, `send_language`, `pcm_sample_rate` | see `bolo/remote/openai_compat.py` |
| `sarvam`: `mode`, `keyterms`, `with_timestamps`, `legacy_translate`, `live`, `language_field`, `enable_preprocessing`, `max_chars` | see `bolo/remote/sarvam.py` |
| `elevenlabs`: `send_language`, `apply_text_normalization`, `provider_sample_rate` | see `bolo/remote/elevenlabs.py` |
| `deepgram`: `language`, `detect_language`, `smart_format`, `hotwords_field`, `live`, `endpointing_ms` | see `bolo/remote/deepgram.py` |

## What bolo adds on top

A cloud model sits behind the same interfaces as a local one, so the code paths around it are the
ones local models use:

- **Decoding and resampling.** Any upload bolo can decode (WAV, MP3, FLAC, Opus, 8 kHz
  telephony, …) is sent to the provider as 16 kHz mono WAV. Speech comes back as PCM or WAV and is
  resampled to the catalog's rate; `response_format` (mp3, opus, flac, wav, pcm, and the telephony
  formats `mulaw` and `alaw`) and `sample_rate` are bolo's own encoders, so every provider offers
  every format.
- **Long recordings.** Audio longer than the provider takes in one request is cut at pauses by
  bolo's VAD and sent in pieces (several at a time), with segment times of the whole recording.
- **Diarization.** `diarize=true` / `diarized_json` cut the recording at bolo's speaker turns and
  send each turn to the provider, so any cloud recogniser gets speaker labels, known speakers and
  overlap marks.
- **Word lists and text normalisation.** Text for a cloud voice is prepared like text for any
  voice that reads text: SSML subset, pronunciation word lists (as spoken forms or respellings),
  and uccharan's normalisation of numbers, dates, money and so on (bolo's light normaliser without
  uccharan). Entries whose provider normalises well itself are set to `text_normalizer: model`
  (the OpenAI voices, ElevenLabs Multilingual v2 and v3); change it per request (`normalize`) or
  per model (`PUT /api/v1/models/{id}/settings`). Word lists apply either way.
- **Sentence-by-sentence synthesis.** bolo sends a sentence at a time, so audio starts after the
  first sentence even where the provider's API is not streamed (Sarvam), and the providers'
  length limits are rarely met. OpenAI's and ElevenLabs' answers are streamed through as they
  arrive.
- **Realtime.** The STT WebSocket, the Realtime API's transcription sessions and the cascade of
  conversation sessions (`STT → language model → TTS`) take cloud models: set
  `default_realtime_stt` / `default_realtime_tts`, or name the model in the session
  (`audio.input.transcription.model`). bolo keeps turn detection, barge-in and the codecs.
  Deepgram (and Sarvam's streaming entry) use the provider's live socket on the STT WebSocket,
  with interim text. Other cloud recognisers get one request per utterance, when bolo's VAD hears
  it end: bolo's usual interim text (repeated decodes of the growing utterance) is switched off
  for cloud models, because every decode would be billed. The model option `interim_decodes`
  switches it back on. Realtime sessions use the batch route too: one request per turn (plus the
  interim decodes, if `interim_decodes` is set).
- **Fallback**, **traces**, **metrics** and **cost estimates**: below.

Not carried over: voice cloning from `reference_audio` (cloud voices answer 400), the
phonemiser choice (cloud voices read text), speaker-conditioned recognition, and interim
transcript deltas while the user is still speaking in Realtime sessions (a cloud recogniser gets
one request per turn; the transcript arrives when the turn ends).

## Request fields, provider by provider

Speech to text (`/v1/audio/transcriptions`, the STT WebSocket, Realtime sessions):

| bolo | openai-compatible | sarvam | elevenlabs | deepgram |
| --- | --- | --- | --- | --- |
| `language` (`hi`, `hi-IN`) | `language` (`hi`) | `language_code` (`hi-IN`; `unknown` when absent; `or` → `od-IN`); **400** for a language Sarvam lacks | `language_code` (`hi`) | `language` as given (`en-IN` stays); none → `detect_language=true` |
| `prompt` | `prompt` | **400** (only the older translate route takes one) | **400** | **400** |
| `hotwords` (comma or line separated) | appended to `prompt`; `keywords[]` for `gpt-transcribe` | `keyterms` (JSON, 50 × 64 characters) with `saaras:v4`; **400** otherwise | `keyterms` | `keyterm` (Nova-3) or `keywords` |
| `timestamp_granularities[]=word` | `verbose_json` + `timestamp_granularities[]` (`whisper-1`); **400** for the `gpt-4o` models | **400**: phrase-level times only (`with_timestamps`, returned as segments) | always returned | always returned |
| `temperature` | `temperature` | not sent | `temperature` | not sent |
| translation (`/v1/audio/translations`) | `/audio/translations` (`whisper-1`); **400** otherwise | `mode=translate` | **400** | **400** |
| `response_format` (json, text, srt, vtt, verbose_json, diarized_json), `stream` | rendered by bolo from the provider's text, segments and words | same | same | same |

Text to speech (`/v1/audio/speech`, the TTS WebSocket, voice agents):

| bolo | openai-compatible | sarvam | elevenlabs |
| --- | --- | --- | --- |
| `voice` | `voice` (catalog list; any value for ad-hoc models) | `speaker` (catalog list; **400** otherwise) | voice id in the path; names are looked up in the account's voices |
| `speed` | `speed` (0.25 to 4.0) | `pace`; **400** outside 0.5 to 2.0 (0.3 to 3.0 for `bulbul:v2`) | `voice_settings.speed` (ElevenLabs answers 400 outside its range) |
| `language` | not sent (`send_language` for another bolo) | `language_code` (`hi-IN`); absent: from the voice, else from the script of the text; **400** outside the 11 languages | `language_code` where the model takes one |
| `response_format`, `sample_rate` | bolo's encoders, from the provider's 24 kHz PCM | from the provider's WAV | from the provider's PCM |
| `reference_audio` | **400** | **400** | **400** (use a cloned voice's id) |
| `normalize`, `wordlists`, `pronunciations`, `input_format` | applied by bolo before the text is sent | same | same |

"**400**" means a clear `invalid_request` that names the parameter. Set `ignore_unsupported` on
the model (or the provider) to have such options dropped instead.

## Errors

| The provider answered | bolo answers | `code` |
| --- | --- | --- |
| 400, 413, 415, 422 | 400, with the provider's message | `invalid_request` |
| 401, 403 | 502 | `provider_auth` |
| 404 | 502 (unknown model or voice, or a wrong address) | `provider_error` |
| 408, 504, or no answer within `provider_timeout_s` | 504 | `provider_timeout` |
| 429 | 429, with the provider's `Retry-After` | `overloaded` |
| other 5xx, a broken connection | 502 | `provider_error` |
| (no provider entry, or no key) | 503 | `provider_not_configured` |

The API key is removed from every message (some providers echo it).

## Fallback

A model can name others that answer in its place when it *cannot answer right now*: its engine
is unavailable (`engine_unavailable`: a missing library, a TTS server that is down), its queue is
full (`overloaded`), or its provider failed (5xx, timeout, rate limit, refused key, not
configured). Requests the model or provider calls wrong (400) never fall back.

```toml
fallbacks = [
  "whisper-base=openai-whisper-1",                 # local first, cloud when the local model is unavailable
  "sarvam-saaras-v4=indicconformer-hi-community,whisper-base",   # cloud first, local when Sarvam is down
  "openai-gpt-4o-mini-tts=kokoro-v1.0-int8",
]
```

Three places, first match wins: a pin through the API
(`PUT /api/v1/models/{id}/settings {"fallback": ["…"]}`; `[]` switches a configured fallback
off, `null` removes the pin), the setting `fallbacks`, and `options.fallback` of a catalog entry.
Fallback models must be of the same task; a name that does not resolve is skipped with a warning.

What a client sees:

- `x-bolo-model: <id>` on every transcription and speech response: the model that answered;
- `x-bolo-fallback: <requested> -> <answered>; reason=<code>` when it was not the one asked for;
- the trace: `model` is the model that answered, and
  `fallback: {requested, answered, attempts: [{model, code, message}]}`;
- `GET /api/v1/stats` → `fallbacks: [{requested, answered, count}]`, and the metric
  `bolo_fallbacks_total{requested, answered}`.

Speech falls back only before the first audio was delivered (afterwards the voice cannot change
in mid-sentence; the request fails as it would without a fallback). A voice the fallback model
does not have is replaced by that model's default voice. Live sockets (the STT WebSocket with a
natively streaming model) do not fall back; batch decodes on the WebSocket and in Realtime
sessions do.

## Traces, metrics and cost estimates

A request served by a cloud model has, on its trace (`GET /api/v1/requests`):

```jsonc
"provider": {"name": "openai", "kind": "openai-compatible", "model": "whisper-1",
             "requests": 3,            // HTTP requests to the provider (pieces of a long recording, sentences)
             "latency_ms": 2140.5,     // time spent waiting for the provider, summed
             "first_byte_ms": 512.0,   // until the first byte of the first answer
             "bytes_sent": 1920132, "bytes_received": 841},
"cost": {"amount": 0.0061, "currency": "USD", "estimated": true}     // only with a price for the model
```

`GET /api/v1/stats` has `providers: [{provider, model, requests, errors, latency_ms_mean,
latency_ms_total, bytes_sent, bytes_received, audio_seconds, chars, estimated_cost: {USD: …},
last_error}]`. `/metrics` has `bolo_provider_requests_total`, `bolo_provider_errors_total`,
`bolo_provider_latency_seconds_total`, `bolo_provider_bytes_sent_total`,
`bolo_provider_bytes_received_total`, `bolo_provider_audio_seconds_total`,
`bolo_provider_characters_total` and `bolo_provider_estimated_cost_total{currency}`, all labelled
`provider` and `model`.

**Cost is an estimate from your own price table**, never a bill. A price is a table of any of
`per_minute`, `per_hour` (of audio: uploaded for speech-to-text, produced for text-to-speech),
`per_1k_chars`, `per_1m_chars` (of text sent to a voice), `per_request`, and `currency` (default
`USD`), keyed by the provider's model name; `"*"` is the price of every other model. Without a
price there is no `cost` field. Token-priced models (the `gpt-4o` family bills by tokens) can only
be approximated by a per-minute or per-character price. `x-bolo-estimated-cost: 0.0061 USD` is
sent on transcription responses.

## Cloud speech-to-speech (OpenAI Realtime)

The Realtime API's `speech-to-speech` pipeline (see [realtime.md](realtime.md)) can be backed by
OpenAI's Realtime API over WebSocket instead of a local llama-server:

```toml
realtime_pipeline = "speech-to-speech"     # or per session: session.pipeline
s2s_model = "openai:gpt-realtime-2.1"      # <provider of kind openai-compatible>:<realtime model>
```

bolo still owns the session the client sees (authentication, the event set, codecs including
G.711, turn detection and barge-in, transcripts of the user from bolo's own recogniser, traces);
the provider hears the audio of each turn and answers. How the provider's protocol maps onto
bolo's backend interface, from the API reference as typed in openai-python 3.51.0:

| | llama-server `/v1/realtime` | OpenAI Realtime |
| --- | --- | --- |
| address | `ws://host/v1/realtime` | `wss://api.openai.com/v1/realtime?model=<model>`, `Authorization: Bearer` |
| input audio | `audio/pcm` at 16 kHz | `audio/pcm` at 24 kHz only: bolo resamples each turn |
| turn detection | none | on by default: bolo sends `audio.input.turn_detection: null` and commits turns itself |
| a response | text and audio together | audio (its text arrives as `response.output_audio_transcript.delta`) **or** text, never both |
| voice | the model's own | `audio.output.voice` (OpenAI's voice names are passed on) |
| tools, `response.cancel`, `response.done` usage | same events | same events |

Not done: Gemini Live (a different protocol: `BidiGenerateContent` over WebSocket with its own
setup, audio and tool messages; it would be a third backend class), cost estimates for
token-priced Realtime usage (the provider's `usage` is passed to the client in `response.done`),
and OpenAI's Realtime *transcription* sessions as a live socket for the STT WebSocket.

## Keys

- Keys are read from the configuration file, the environment, or `PUT /api/v1/providers/{name}`.
- **No route returns a key.** `GET /api/v1/providers` and `/api/v1/info` say `configured` and
  `api_key_set`; the settings dump shows `***`. Options whose name contains `key`, `token`,
  `secret`, `password` or `authorization` are masked the same way.
- Keys are not logged. Messages that came from a provider are scrubbed of the key before they are
  logged, put on a trace or returned.
- Keys travel in request headers only, never in a URL.
- Providers added through the API are stored in `<data_dir>/providers.json`, written with mode
  `0600`. It holds the keys in the clear, like the configuration file: protect the data directory.
- Changing providers through the API is refused on a non-loopback bind without `api_key`, and can
  be switched off (`allow_provider_edit = false`). Providers from the configuration cannot be
  changed through the API at all.

## The providers API

| Call | Result |
| --- | --- |
| `GET /api/v1/providers` | `{providers: [{name, kind, base_url, configured, api_key_set, options, prices, source: "settings"\|"stored", editable, terms_url, stats}], kinds: [{kind, default_base_url, terms_url}], editable, edit_blocked_reason, notice}` |
| `GET /api/v1/providers/{name}` | one provider |
| `PUT /api/v1/providers/{name}` `{kind?, base_url?, api_key?, clear_api_key?, options?, prices?}` | adds or changes a stored provider. `api_key` is write-only: absent or empty keeps the stored key, `clear_api_key: true` removes it. Loaded models of the provider are unloaded so the change takes effect. `400` for a provider that comes from the configuration, an unknown kind or field, or when editing is not allowed |
| `DELETE /api/v1/providers/{name}` | `204`; `400` for a provider from the configuration |
| `POST /api/v1/providers/{name}/test` | one cheap request with the stored key: `{name, ok, message, code, checked: "credentials"\|"reachability", latency_ms}`. `credentials`: the key was accepted (OpenAI-compatible: `GET /models`; ElevenLabs: `GET /v2/voices`; Deepgram: `GET /v1/projects`). `reachability`: the address answered and did not refuse the key, but nothing proves the key (Sarvam documents no free read-only request; a refused key is still reported as a failure) |

Events: `provider` `{name, event: "saved"|"removed"}` on `GET /api/v1/events`.

## What was verified, and what was not

Read on 2026-11-14:

| Provider | Source | What was taken from it |
| --- | --- | --- |
| OpenAI | the API reference as typed in the `openai` Python package 3.51.0 (generated from OpenAI's OpenAPI description); `developers.openai.com/api/docs/guides/realtime-websocket` | routes, fields, model names and announced shutdown dates, the 13 voice names, `pcm` at 24 kHz, speed 0.25 to 4.0, 4096 characters, streaming event names, Realtime session shape (`type: realtime`, `output_modalities`, `audio.input/output.format`, `turn_detection`), the WebSocket address and header |
| Sarvam | `docs.sarvam.ai/api-reference-docs/` (speech-to-text/transcribe, speech-to-text-translate/translate, text-to-speech/convert, the two WebSocket references) | routes, `api-subscription-key`, `saaras:v4` and `saaras:v3`, modes, the 23 and 11 language codes, `keyterms`, phrase-level timestamps, the 30-second note, `bulbul:v2` / `v3` / `v4-flash`, all speaker names, `pace` ranges, sample rates, character limits, the streaming message shapes |
| ElevenLabs | `elevenlabs.io/docs/api-reference/` (text-to-speech/stream, speech-to-text/convert, voices/search), `elevenlabs.io/docs/overview/models` | routes, `xi-api-key`, `output_format` values, body fields, `scribe_v3` / `scribe_v2`, word objects, the voice listing, model ids, the 32 language codes of Flash v2.5 |
| Deepgram | `developers.deepgram.com` (reference: listen-pre-recorded, listen-streaming, manage/projects/list; the pre-recorded guide; models and languages overview) | routes, `Authorization: Token`, the raw-audio upload, query parameters, `keyterm` against `keywords`, the response paths, the live socket's parameters and messages, Nova-3 and Nova-2 language codes |

Not verified, or taken from memory:

- **Nothing was run against a real account.** Request and response shapes are what the
  documentation says; the stand-ins in the tests were written from the same documentation.
- OpenAI's reference site (`platform.openai.com`) refused automated reading; the typed SDK was
  used instead. Which voices `tts-1` and `tts-1-hd` accept is from memory (the catalog lists nine
  for them and all thirteen for `gpt-4o-mini-tts`).
- Sarvam: **Saarika** no longer appears in the reference (recognition is `saaras:v3` and
  `saaras:v4` now), so there is no catalog entry for it; `sarvam:saarika:v2.5` passes the name on.
  The text-to-speech language field is `language_code` in today's reference and was
  `target_language_code` earlier (option `language_field`). bolo's use of the streaming socket
  (raw PCM with `input_audio_codec=pcm_s16le`) follows the reference but the reference's audio
  message names only `audio/wav` as encoding. Sarvam's streaming text-to-speech socket is not
  used.
- ElevenLabs: the limits of `voice_settings.speed`, and the ids of the stock voices (the
  catalog lists no voices; they are read from the account). The 29 languages of Multilingual v2
  are the 32 of Flash v2.5 without Hungarian, Norwegian and Vietnamese, from memory.
- The terms links in the licence field: Sarvam's, ElevenLabs' and Deepgram's pages were opened
  today; OpenAI's (`openai.com/policies/services-agreement/`) refused automated reading and is
  from memory.
- Prices: none are shipped; providers change them.

## Real tests

`server/tests/test_real.py` has one test per provider (`test_cloud_openai`, `test_cloud_sarvam`,
`test_cloud_elevenlabs`, `test_cloud_deepgram`): the connection test, a speak-then-transcribe
round trip with a word-error-rate bound, word timestamps, translation, and Deepgram's live
socket. They are skipped unless both are true:

```bash
export BOLO_TEST_REAL=1
export OPENAI_API_KEY=…        # or BOLO_PROVIDER_OPENAI_API_KEY; likewise SARVAM_API_KEY,
                               # ELEVENLABS_API_KEY, DEEPGRAM_API_KEY
pytest -m real -k cloud
```

They are billed by the provider (a few seconds of audio and a few sentences each).
