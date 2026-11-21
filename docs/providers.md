# Cloud providers

bolo runs speech models itself. It can also hand a request to a cloud speech API and treat the
answer as if a local model had produced it. That is the `remote` engine: a model with
`engine: remote` names a *provider* (an account at an API) and the provider's own model name.

**Every cloud model is a paid cloud API, and audio (or text) leaves this server.** The catalog
marks such models `cloud: true`, the console labels them, and nothing is ever sent to a provider
unless a request names a cloud model, a cloud model is set as a default, or a model's `fallback`
list names one.

What is on this page was read from each provider's documentation on **2026-11-14** (Soniox's on
**2026-11-17**; AssemblyAI's, xAI's, Rime's, Microsoft's for MAI, and ElevenLabs' for Scribe v3 and
Scribe v3 Live on **2026-11-21**). Nothing was run against a real account: there
were no keys where bolo is developed, and the test-suite uses stand-ins written from the same
documentation (`server/tests/fake_providers.py`, `server/tests/fake_cloud_apis.py`). The opt-in real tests are
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
| `soniox` | Soniox (speech-to-text: async API and live WebSocket) | `https://api.soniox.com` (the live socket: `wss://stt-rt.soniox.com`) | `Authorization: Bearer <key>` |
| `assemblyai` | AssemblyAI (speech-to-text: pre-recorded API and Streaming API v3, both with speaker labels) | `https://api.assemblyai.com` (the live socket: `wss://streaming.assemblyai.com/v3/ws`) | `Authorization: <key>` (no prefix) |
| `azure-mai` | Microsoft's MAI models in Azure: MAI-Transcribe-2-Streaming (the realtime socket of a Foundry resource) and MAI-Voice-2.1 (Azure Speech's text-to-speech route) | none: set it to your resource, `https://<resource>.services.ai.azure.com`; the voices need the option `region` (or `tts_url`) | `api-key: <key>` (socket), `Ocp-Apim-Subscription-Key: <key>` (voices) |
| `xai` | xAI (speech-to-speech only: the Voice Agent API, `s2s_model`) | `https://api.x.ai/v1` (`wss://api.x.ai/v1/realtime`) | `Authorization: Bearer <key>` |
| `rime` | Rime (speech-to-speech only: Prism, `s2s_model`) | `https://prism.api.rime.ai/v1` (`wss://prism.api.rime.ai/v1/realtime`) | `Authorization: Bearer <key>` |

In the configuration file, one table per provider:

```toml
[providers.openai]                 # the names openai, sarvam, elevenlabs, deepgram, soniox, assemblyai,
                                   # azure-mai, xai and rime imply their kind
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

[providers.azure-mai]              # Microsoft's MAI models: an Azure resource of your own
api_key = "…"
base_url = "https://my-resource.services.ai.azure.com"   # BOLO_PROVIDER_AZURE_MAI_BASE_URL
options = { region = "eastus", deployments = { "MAI-Transcribe-2-Streaming" = "my-transcribe" } }
```

In the environment:

```bash
BOLO_PROVIDER_OPENAI_API_KEY=…            # BOLO_PROVIDER_<NAME>_API_KEY / _BASE_URL / _KIND
BOLO_PROVIDER_OFFICE_BOLO_KIND=openai-compatible
BOLO_PROVIDER_OFFICE_BOLO_BASE_URL=http://10.0.0.5:8000/v1
BOLO_PROVIDERS='{"deepgram": {"api_key": "…"}}'     # or the whole table as JSON
```

(`_` in the environment name is `-` in the provider's name. The `BOLO_PROVIDER_…` variables add
to, and override, the same provider's table from the file.) Every cloud entry of the catalog
names the variable its key goes in: `provider.key_env` in `GET /api/v1/catalog`
(`BOLO_PROVIDER_ASSEMBLYAI_API_KEY` for the AssemblyAI entries), and the tables in
[models.md](models.md).

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
- `provider: {name, kind, model, configured, price, key_env}` in the API (`key_env`: the
  environment variable of the key, `BOLO_PROVIDER_<NAME>_API_KEY`).

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
| `elevenlabs-scribe-v3` | `scribe_v3` | STT | recorded audio (released 2026-11-10): word timestamps, speaker labels from ElevenLabs (`diarize=true`, up to 32 speakers), hotwords (billed at a surcharge by ElevenLabs); the 101 languages ElevenLabs' documentation names (it measured 102 on FLEURS) |
| `elevenlabs-scribe-v3-live` | `scribe_v3_live` | STT | live (ElevenLabs' realtime socket, released 2026-11-10): interim text, turn prediction, speaker labels on the socket; recorded audio and Realtime turns go to `scribe_v3` |
| `elevenlabs-scribe-v2` | `scribe_v2` | STT | word timestamps, speaker labels, hotwords |
| `elevenlabs-voice-isolator` | `audio-isolation` | denoise | Voice Isolator (`POST /v1/audio-isolation`, 16 kHz PCM up): `denoise=elevenlabs-voice-isolator` on a transcription sends the recording there first and transcribes the isolated voice. One request per recording, so not for live streams. Billed per minute of audio |
| `deepgram-nova-3`, `deepgram-nova-2` | `nova-…` | STT | batch and live (Deepgram's streaming socket, with interim text) |
| `soniox-stt-async-v6` | `stt-async-v6` | STT | recorded audio (Soniox's async API): word timestamps, speaker labels, language identification, translation to English |
| `soniox-stt-rt-v6` | `stt-rt-v6` | STT | live (Soniox's WebSocket, with interim text); recorded audio and Realtime turns go to `stt-async-v6` |
| `assemblyai-stt-3.7` | `assemblyai-stt-3-7` | STT | recorded audio (AssemblyAI's pre-recorded API, released 2026-11-17): 32 languages, word timestamps, speaker labels, language detection, keyterms and prompt |
| `mai-transcribe-2-streaming` | `MAI-Transcribe-2-Streaming` (your deployment) | STT | live (the realtime socket of your Foundry resource; public preview): 60 languages detected as they change, interim text, a final at each pause (bolo commits; MAI detects no speech itself). Recorded audio and Realtime turns go over the same socket with one commit. Text only |
| `mai-voice-2.1`, `mai-voice-2.1-flash` | `MAI-Voice-2.1`, `MAI-Voice-2.1-Flash` | TTS | 97 voices in 23 languages through Azure Speech (public preview); `en-US-Harper` unless the request names one |
| `xai-grok-voice-3.0` | `grok-voice-3.0` | speech-to-speech | `s2s_model` (see [Cloud speech-to-speech](#cloud-speech-to-speech-openai-realtime)) |
| `rime-prism` | `prism` | speech-to-speech | `s2s_model` (see [Cloud speech-to-speech](#cloud-speech-to-speech-openai-realtime)) |
| `assemblyai-stt-3.7-realtime`, `assemblyai-stt-3.7-realtime-agent` | `assemblyai-stt-3-7-realtime`, `…-realtime-agent` | STT | live (AssemblyAI's Streaming API v3, with interim text and AssemblyAI's turn detection) with speaker labels on the socket; the `-agent` model is the one tuned for voice agents. Recorded audio and Realtime turns go to `assemblyai-stt-3-7` |

An entry looks for a provider with the name in its `options.provider` (`openai`, `sarvam`,
`elevenlabs`, `deepgram`, `soniox`, `assemblyai`, `azure-mai`, `xai`, `rime`). If you called the account something else, the first configured
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
| `elevenlabs`: `send_language`, `apply_text_normalization`, `provider_sample_rate`, `async_model`, `speaker_labels`, `num_speakers`, `live`, `live_speaker_labels`, `commit_strategy`, `realtime_url`, `live_request` | see `bolo/remote/elevenlabs.py` |
| `azure-mai`: `deployment` (or the provider option `deployments`), `region`, `tts_url`, `output_format`, `realtime_url`, `commit_silence_s` (0.5), `max_segment_s` (30), `live` | see `bolo/remote/azure_mai.py` and [Microsoft MAI](#microsoft-mai) below (these also work as provider options) |
| `deepgram`: `language`, `detect_language`, `smart_format`, `hotwords_field`, `live`, `endpointing_ms` | see `bolo/remote/deepgram.py` |
| `assemblyai`: `async_model`, `live_model`, `live`, `speaker_labels`, `speakers_expected`, `live_speaker_labels`, `max_speakers`, `language_codes`, `min_turn_silence`, `max_turn_silence`, `mode`, `poll_interval_s`, `poll_timeout_s`, `keep_files`, `realtime_url`, `live_request` | see `bolo/remote/assemblyai.py` and [AssemblyAI](#assemblyai) below (these also work as provider options) |
| `soniox`: `async_model`, `live_model`, `live`, `language_hints`, `language_hints_strict`, `language_identification`, `speaker_diarization`, `provider_diarization`, `context`, `translate`, `translate_to`, `poll_interval_s`, `poll_timeout_s`, `keep_files`, `realtime_url`, `endpoint_delay_ms`, `live_request` | see `bolo/remote/soniox.py` and [Soniox](#soniox) below (these also work as provider options) |

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
  overlap marks. Soniox, AssemblyAI and ElevenLabs label speakers themselves: for their models the
  request is passed on (`enable_speaker_diarization`, `speaker_labels`, `diarize`) and the provider's
  labels are used (see [Soniox](#soniox), [AssemblyAI](#assemblyai), [ElevenLabs Scribe](#elevenlabs-scribe)).
  On the STT WebSocket, AssemblyAI's realtime entries and `elevenlabs-scribe-v3-live` label speakers
  on their live socket too (`diarize=true`).
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
  Deepgram, Soniox (`soniox-stt-rt-v6`), AssemblyAI's realtime entries, `elevenlabs-scribe-v3-live`,
  `mai-transcribe-2-streaming` and Sarvam's streaming entry use the provider's live socket on the STT
  WebSocket, with interim text. Other cloud recognisers get one request per utterance, when bolo's VAD hears
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

| bolo | openai-compatible | sarvam | elevenlabs | deepgram | soniox | assemblyai |
| --- | --- | --- | --- | --- | --- | --- |
| `language` (`hi`, `hi-IN`) | `language` (`hi`) | `language_code` (`hi-IN`; `unknown` when absent; `or` → `od-IN`); **400** for a language Sarvam lacks | `language_code` (`hi`) | `language` as given (`en-IN` stays); none → `detect_language=true` | `language_hints: ["hi"]`; language identification on, and the answer's language is the one most of the text is in | `language_code` (`hi`); none → `language_detection: true` (live: `language_codes`) |
| `prompt` | `prompt` | **400** (only the older translate route takes one) | **400** | **400** | `context.text` | `prompt` (recorded audio) |
| `hotwords` (comma or line separated) | appended to `prompt`; `keywords[]` for `gpt-transcribe` | `keyterms` (JSON, 50 × 64 characters) with `saaras:v4`; **400** otherwise | `keyterms` | `keyterm` (Nova-3) or `keywords` | `context.terms` | `keyterms_prompt` (up to 1000 terms) |
| `timestamp_granularities[]=word` | `verbose_json` + `timestamp_granularities[]` (`whisper-1`); **400** for the `gpt-4o` models | **400**: phrase-level times only (`with_timestamps`, returned as segments) | always returned | always returned | always returned (Soniox's tokens joined into words) | always returned |
| `temperature` | `temperature` | not sent | `temperature` | not sent | not sent | **400** |
| `diarize`, `diarized_json` | bolo's speaker turns | same | same | same | `enable_speaker_diarization`; Soniox's labels (`1`, `2` → `A`, `B`) | `speaker_labels`; AssemblyAI's labels (`A`, `B`); live: `speaker_labels=true` on the socket |
| translation (`/v1/audio/translations`) | `/audio/translations` (`whisper-1`); **400** otherwise | `mode=translate` | **400** | **400** | `translation: {type: one_way, target_language: en}` | **400** |
| `response_format` (json, text, srt, vtt, verbose_json, diarized_json), `stream` | rendered by bolo from the provider's text, segments and words | same | same | same | same | same |

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

### Soniox

Recorded audio goes through Soniox's async API in three steps: the audio is uploaded
(`POST /v1/files`), a transcription is created (`POST /v1/transcriptions`) and polled
(`GET /v1/transcriptions/{id}`, every 0.25 s at first, then up to every 2 s) until it is
`completed`, and the transcript is read (`GET /v1/transcriptions/{id}/transcript`). bolo then
deletes the transcription and the uploaded file at Soniox (set `keep_files` to leave them). The
wait is bounded by `poll_timeout_s` (default: 60 s plus the length of the audio, at least 120 s);
past it the request answers 504 `provider_timeout`. A transcription that ends in `error` answers
400 when Soniox says the audio or the request was at fault, 502 otherwise. Each step is a request
on the trace (`provider.requests`); the deletes are not counted.

- **Words.** Soniox answers in tokens, pieces of words with times in milliseconds; a token that
  starts with a space starts a new word (in Chinese, Japanese and Thai each character is a word).
  Segments end at a change of speaker or at the end of a sentence.
- **Speakers.** With `diarize=true` or `diarized_json`, Soniox is asked to label speakers
  (`enable_speaker_diarization`) and its labels `1`, `2`, … become `A`, `B`, … on segments and
  words; no local diarization model is loaded (the trace says `diarization_model: provider`).
  bolo's own diarizer is used instead when the request names one (`diarization_model`), gives
  known speakers, or asks for overlap separation; `num_speakers`, `min_speakers` and
  `max_speakers` have no Soniox equivalent and are not sent. Recordings longer than
  `max_chunk_s` (an hour by default; Soniox takes up to five) are sent in pieces, and Soniox
  numbers the speakers of each piece on its own: raise `max_chunk_s` for long diarized
  recordings. The model option `speaker_diarization` asks for labels on every request;
  `provider_diarization: false` always uses bolo's diarizer.
- **Languages.** `language` becomes a hint (`language_hints`; `language_hints_strict` makes
  Soniox keep to it). Language identification is on by default (`language_identification`);
  the answer's `language` is the one most of the text is in.
- **Live.** `soniox-stt-rt-v6` opens `wss://stt-rt.soniox.com/transcribe-websocket` with the key
  in the `Authorization` header (never in the start message), sends 16 kHz PCM, and asks for
  Soniox's endpoint detection. Final tokens are held until Soniox's `<end>` or `<fin>` token and
  then handed on as one final text; until then they are shown with the non-final tokens as
  interim text. A provider at another address takes its socket at `<base_url>/transcribe-websocket`
  (or at `realtime_url`). Live speaker labels and per-token languages are not passed on yet.
- **Realtime sessions** use the async route, one transcription per turn: three requests and a
  short wait for every turn. For conversations, a recogniser with a single-request batch route
  (OpenAI, Deepgram) answers sooner; `soniox-stt-rt-v6` on the STT WebSocket is the live option.

### AssemblyAI

Recorded audio goes through AssemblyAI's pre-recorded API in three steps: the WAV is uploaded as
it is (`POST /v2/upload`, `application/octet-stream`), a transcript is requested
(`POST /v2/transcript` with `speech_models: ["assemblyai-stt-3-7"]`) and polled
(`GET /v2/transcript/{id}`, every 0.25 s at first, then up to every 3 s; AssemblyAI allows 60
looks a minute per transcript) until it is `completed`. bolo then deletes the transcript at
AssemblyAI (`keep_files` leaves it). The wait is bounded by `poll_timeout_s` as for Soniox; a
transcript that ends in `error` answers 502 with AssemblyAI's message.

- **Speakers.** With `diarize=true` or `diarized_json`, AssemblyAI is asked for speaker labels
  (`speaker_labels`; the model option `speakers_expected` sends a count) and its utterances
  become segments with their speakers (`A`, `B`, …, as bolo labels them). The trace says
  `diarization_model: provider`; bolo's own diarizer is used when the request names one, gives
  known speakers or asks for overlap separation, exactly as for Soniox.
- **Languages.** `language` is sent as `language_code`; without one, AssemblyAI detects the
  language (`language_detection: true`). STT-3.7 handles 32 languages and switches between them
  within a recording; a language it lacks falls back to AssemblyAI's Universal-2 at AssemblyAI.
- **Filler words.** STT-3.7 keeps them (and repetitions and false starts) by default; set the
  model option `request: {"disfluencies": false}` to drop most of them.
- **Live.** `assemblyai-stt-3.7-realtime` and `-realtime-agent` open
  `wss://streaming.assemblyai.com/v3/ws` with the key in the `Authorization` header and the
  settings as query parameters (`speech_model`, `sample_rate=16000`, `encoding=pcm_s16le`,
  `language_codes` when the session names a language, `speaker_labels=true` when it asks for
  diarization). Audio goes in binary frames of 50 ms to 1 s (AssemblyAI refuses frames under
  20 ms, so short pieces are joined). Each `Turn` replaces the interim text; the turn's last
  message (`end_of_turn` and `turn_is_formatted`) is the final, with its words and speaker. The
  end of input sends `ForceEndpoint`, and the session ends with `Terminate`. The model options
  `min_turn_silence`, `max_turn_silence` and `mode` are passed on; `realtime_url` points at
  another address (AssemblyAI's EU socket is `wss://streaming.eu.assemblyai.com/v3/ws`).
- **Realtime sessions** use the pre-recorded route, one transcript per turn, as for Soniox.

### ElevenLabs Scribe

`elevenlabs-scribe-v3` sends recorded audio to `POST /v1/speech-to-text` (`model_id: scribe_v3`).
With `diarize=true` or `diarized_json`, ElevenLabs is asked to label speakers (`diarize`; the model
option `num_speakers` caps them, up to 32) and its `speaker_0`, `speaker_1`, … become `A`, `B`, … on
segments (one per run of the same speaker) and words; the trace says `diarization_model: provider`.

`elevenlabs-scribe-v3-live` opens `wss://api.elevenlabs.io/v1/speech-to-text/realtime` with the key in
the `xi-api-key` header and `model_id=scribe_v3_live`, `audio_format=pcm_16000`,
`commit_strategy=turn_prediction` (ElevenLabs commits a segment when the model predicts the speaker has
finished; `vad` and `manual` through the model option `commit_strategy`), `include_timestamps=true`,
the session's language and, with `diarize=true`, `diarize=true`. Audio goes as JSON
`input_audio_chunk` messages; the end of input sends one with `commit: true`. `partial_transcript`
is the interim text and `committed_transcript_with_timestamps` the final with its words. With speaker
labels the final waits for the segment's `diarized_transcript` (ElevenLabs sends it after the committed
text, with the final speakers; `speakers_merged` is applied), and is handed on without a speaker if the
next segment is committed first.

### Microsoft MAI

Microsoft's MAI models run in your own Azure resource, so the provider needs its address as
`base_url` (`https://<resource>.services.ai.azure.com`; until then the provider counts as not
configured and a request answers 503 naming `BOLO_PROVIDER_AZURE_MAI_BASE_URL`), the key, and for the
voices the region of the Speech resource (option `region`, or the whole address as `tts_url`).
Both are public previews (Microsoft's preview terms are the licence link).

- **MAI-Transcribe-2-Streaming** (released 2026-10-01) is reached at
  `wss://<resource>.services.ai.azure.com/mai/v1/realtime?intent=transcription` with an `api-key`
  header. bolo configures the session before any audio (`session.type: transcription`, 16 kHz PCM,
  `transcription.model`: the deployment, which is the model's name unless the provider option
  `deployments` maps it; `transcription.language`: the session's language or `null` to detect it;
  `turn_detection` and `noise_reduction` `null`, the only values MAI takes), then sends
  `input_audio_buffer.append` messages. `delta` (finalised text) and `intermediate` (the provisional
  rest) make the interim text. MAI detects no speech itself and finalises only on
  `input_audio_buffer.commit`, so bolo commits when it hears a pause of `commit_silence_s` (0.5 s) after
  speech, at the latest every `max_segment_s` (30 s), and at the end of input; each `completed` is a
  final. Recorded audio goes over the same socket: the whole recording, one commit, the `completed`
  transcript. `prompt`, `hotwords` and `temperature` answer 400 (MAI has no equivalent). A session lasts
  at most an hour.
- **MAI-Voice-2.1** and **MAI-Voice-2.1-Flash** use Azure Speech's text-to-speech route
  (`POST https://<region>.tts.speech.microsoft.com/cognitiveservices/v1`, `Ocp-Apim-Subscription-Key`):
  bolo sends SSML naming the voice with the model as suffix (`es-MX-Valeria:MAI-Voice-2.1-Flash`,
  `xml:lang` from the voice), asks for Microsoft's example format `audio-24khz-160kbitrate-mono-mp3`
  (option `output_format`, for example `raw-24khz-16bit-mono-pcm`) and decodes the answer. The catalog
  lists the 97 managed voices; `speed` other than 1 answers 400.
- **MAI Realtime 1** (Microsoft's speech-to-speech model, announced 2026-11-17) is in limited-access
  preview and has no public API reference yet, so it has no catalog entry.

## Errors

| The provider answered | bolo answers | `code` |
| --- | --- | --- |
| 400, 413, 415, 422 | 400, with the provider's message | `invalid_request` |
| 401, 403 | 502 | `provider_auth` |
| 404 | 502 (unknown model or voice, or a wrong address) | `provider_error` |
| 408, 504, or no answer within `provider_timeout_s` | 504 | `provider_timeout` |
| 429 | 429, with the provider's `Retry-After` (1 s when it sends none, as Soniox documents none) | `overloaded` |
| 402 (Soniox: balance or budget exhausted) | 502 | `provider_error` |
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

Two more Realtime services have catalog entries, named directly in `s2s_model` (their providers
are configured like any other: `BOLO_PROVIDER_XAI_API_KEY`, `BOLO_PROVIDER_RIME_API_KEY`):

```toml
realtime_pipeline = "speech-to-speech"
s2s_model = "xai-grok-voice-3.0"           # or "rime-prism"
```

| | xAI Grok Voice 3.0 (`xai-grok-voice-3.0`) | Rime Prism (`rime-prism`) |
| --- | --- | --- |
| address | `wss://api.x.ai/v1/realtime?model=grok-voice-3.0`, `Authorization: Bearer` | `wss://prism.api.rime.ai/v1/realtime` (no query), `Authorization: Bearer` |
| session | `voice`, `instructions`, `turn_detection: {"type": null}` (bolo commits turns), `audio.input` / `audio.output` `format` (16 kHz up, 24 kHz back), `tools` nested under `function` | only `voice`, `instructions`, `tools` (flat) and `turn_detection: null`: Prism refuses a session with any field it does not know. 16 kHz up, 24 kHz back (fixed) |
| a response | `response.create` with `modalities`; audio with its transcript (`response.output_audio_transcript.delta`), or text (`response.output_text.delta`) | always text (`response.text.delta`) and audio (`response.audio.delta`); bolo leaves the audio out of a text-only response |
| typed turns, tool results | `conversation.item.create` | a typed turn is `response.create` with `prism_input_text`; after tool results bolo waits for Prism's `conversation.item.created` and asks for a `tool_continuation` naming the reply that made the calls |
| voices | `eve` (the entry's default when a client sends an OpenAI voice name), `ara`, … and custom voice ids | `astra` (Prism's default), `aurora`, `brando` and the other Coda voices; English only |
| not passed on | `tool_choice`, `max_output_tokens` (xAI's `response.create` has neither) | `tool_choice` (Prism takes only `auto`), `max_output_tokens` |

Microsoft's MAI Realtime 1 is not offered: it is in limited-access preview without a public API
reference.

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
| `POST /api/v1/providers/{name}/test` | one cheap request with the stored key: `{name, ok, message, code, checked: "credentials"\|"reachability", latency_ms}`. `credentials`: the key was accepted (OpenAI-compatible: `GET /models`; ElevenLabs: `GET /v2/voices`; Deepgram: `GET /v1/projects`; Soniox: `GET /v1/models`). `reachability`: the address answered and did not refuse the key, but nothing proves the key (Sarvam documents no free read-only request; a refused key is still reported as a failure). AssemblyAI: `GET /v2/transcript?limit=1`; xAI: `GET /v1/tts/voices`; Rime and Azure MAI: the WebSocket handshake, closed right after `session.created`, before any configuration or audio (all `credentials`) |

Events: `provider` `{name, event: "saved"|"removed"}` on `GET /api/v1/events`.

## What was verified, and what was not

Read on 2026-11-14:

| Provider | Source | What was taken from it |
| --- | --- | --- |
| OpenAI | the API reference as typed in the `openai` Python package 3.51.0 (generated from OpenAI's OpenAPI description); `developers.openai.com/api/docs/guides/realtime-websocket` | routes, fields, model names and announced shutdown dates, the 13 voice names, `pcm` at 24 kHz, speed 0.25 to 4.0, 4096 characters, streaming event names, Realtime session shape (`type: realtime`, `output_modalities`, `audio.input/output.format`, `turn_detection`), the WebSocket address and header |
| Sarvam | `docs.sarvam.ai/api-reference-docs/` (speech-to-text/transcribe, speech-to-text-translate/translate, text-to-speech/convert, the two WebSocket references) | routes, `api-subscription-key`, `saaras:v4` and `saaras:v3`, modes, the 23 and 11 language codes, `keyterms`, phrase-level timestamps, the 30-second note, `bulbul:v2` / `v3` / `v4-flash`, all speaker names, `pace` ranges, sample rates, character limits, the streaming message shapes |
| ElevenLabs | `elevenlabs.io/docs/api-reference/` (text-to-speech/stream, speech-to-text/convert, voices/search, audio-isolation/convert), `elevenlabs.io/docs/overview/models` | routes, `xi-api-key`, `output_format` values, body fields, `scribe_v3` / `scribe_v2`, word objects, the voice listing, model ids, the 32 language codes of Flash v2.5, the voice isolator's `audio` and `file_format` fields (the reference does not name the format of its answer: bolo decodes WAV, MP3 or anything else its decoder reads) |
| Soniox (read 2026-11-17) | `docs.soniox.com` (API reference: files, transcriptions, models, the WebSocket API, errors; guides: async transcription, models, supported languages, context, manual finalization, endpoint detection, connection keepalive, WebSocket authentication) | the three-step async flow and its routes, `Authorization: Bearer` for REST and the live socket, `stt-async-v6` / `stt-rt-v6` (v5 deprecated for 2026-12-15), the 60 language codes, request fields, token fields (`start_ms`, `end_ms`, `speaker`, `language`, `translation_status`), the start message, `finalize` / `keepalive` / empty-frame messages, `<end>` and `<fin>`, error codes and bodies |
| AssemblyAI (read 2026-11-21) | `assemblyai.com/docs` (pre-recorded: files/upload, transcripts/submit, get, list and delete, select-the-speech-model, the assemblyai-stt-3-7 migration guide, label-speakers, supported-languages; streaming: select-the-speech-model, message-sequence, turn-detection, label-speakers-and-separate-channels, and the AsyncAPI description `api-reference/specs/streaming.yaml`) | the three-step pre-recorded flow and its routes, `Authorization` without a prefix, `assemblyai-stt-3-7` and the two realtime models (released 2026-11-17), the 32 language codes, request fields (`speech_models`, `language_code`, `language_detection`, `speaker_labels`, `speakers_expected`, `keyterms_prompt`, `prompt`), word and utterance fields (times in ms, `speaker`), the socket's address, query parameters, binary frames of 20 ms to 1 s, `ForceEndpoint` / `Terminate`, `Begin` / `Turn` / `Termination` / `Error`, `speaker_label` per turn and `speaker` per final word |
| ElevenLabs Scribe v3 and v3 Live (read 2026-11-21) | `elevenlabs.io/docs` (api-reference/speech-to-text/convert and v-1-speech-to-text-realtime, overview/models, overview/capabilities/speech-to-text, the server-side streaming guide) | `diarize`, `num_speakers`, `speaker_id` on words, `scribe_v3` / `scribe_v3_live`, the realtime socket's address, query parameters (`commit_strategy` with `turn_prediction`, `include_timestamps`, `diarize`), `input_audio_chunk`, the message types (`partial_transcript`, `committed_transcript_with_timestamps`, `diarized_transcript`, `speakers_merged`, the error types), the language list |
| xAI (read 2026-11-21) | `docs.x.ai` (model-capabilities/audio/speech-to-speech, voice, text-to-speech; rest-api-reference/inference/voice; `voice-realtime.ws.json`) | `grok-voice-3.0` (`grok-voice-latest`; 2.0 deprecated, the same endpoint, session and events), the address and header, the session fields and their JSON schema (nested function tools, `turn_detection.type` null for manual turns, PCM rates), `response.create` with `modalities`, every server event name, the languages, `GET /v1/tts/voices` |
| Rime (read 2026-11-21) | `docs.rime.ai` (api-reference/prism/websocket, prism/migrate-from-openai) | the address, Bearer on the upgrade, no query, the strict session fields, 16 kHz in and 24 kHz out, `turn_detection: null`, `response.create` and `prism_cause` (typed turns, tool continuations, acknowledgement by `conversation.item.created`), the beta event names, `response.done` statuses, voices |
| Microsoft MAI (read 2026-11-21) | `learn.microsoft.com/azure/ai-services/speech-service/` (mai-transcribe-2-streaming, mai-transcribe-2-streaming-realtime, mai-voices) | the socket's address, `api-key`, the transcription session, `append` / `commit`, `delta` / `intermediate` / `completed`, the 60 languages; the Speech route, SSML with the model suffix, the output format of Microsoft's example, the 97 voices and 23 languages, the regions |
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
- Soniox: the shape of a whole async transcript was taken from the field list and a two-token example;
  how translated tokens are ordered in an async transcript, which `error_type` values a failed
  transcription carries (bolo answers 400 for `invalid_request`, `audio_decode_error` and
  `invalid_audio`, and 502 for the rest), and whether the live socket refuses a bad key at the
  handshake or with an error message were not shown. No `Retry-After` is documented for 429.
- The terms links in the licence field: Sarvam's, ElevenLabs' and Deepgram's pages were opened
  today; OpenAI's (`openai.com/policies/services-agreement/`) refused automated reading and is
  from memory.
- AssemblyAI: whether the live socket refuses a bad key at the handshake or with an `Error`
  message, and the error bodies beyond the documented `error` field. The 32 languages of
  STT-3.7 are the streaming reference's list (`language_codes`); the pre-recorded page names the
  same count.
- ElevenLabs: whether words in `committed_transcript_with_timestamps` already carry speakers (bolo
  waits for `diarized_transcript`, which the reference says carries the final ones). "102 languages"
  is ElevenLabs' FLEURS figure; its documentation names 101, which the catalog lists.
- xAI: the voice list beyond `eve` and `ara` (it is published in the console and through
  `GET /v1/tts/voices`, not on the page); whether `response.create` with `modalities: ["text", "audio"]`
  also sends `response.output_text.delta` (bolo takes the transcript events for a spoken response).
  The terms link (`x.ai/legal/terms-of-service`) refused automated reading.
- Rime: whether Prism takes the conversation's `function_call` items from a reply as the parent of a
  continuation when several replies made calls (bolo names the last reply).
- Microsoft MAI: whether one key serves both the Foundry socket and the Speech route (give the voices
  a second provider of kind `azure-mai` if not), the shape of `error` events, whether `completed`
  carries a language, and SSML `prosody` with MAI voices (bolo sends none).
- Prices: none are shipped; providers change them.

## Real tests

`server/tests/test_real.py` has one test per provider (`test_cloud_openai`, `test_cloud_sarvam`,
`test_cloud_elevenlabs`, `test_cloud_deepgram`, `test_cloud_soniox`, `test_cloud_assemblyai`): the connection test, a
speak-then-transcribe round trip with a word-error-rate bound, word timestamps, translation, and
Deepgram's and Soniox's live sockets. They are skipped unless both are true:

```bash
export BOLO_TEST_REAL=1
export OPENAI_API_KEY=…        # or BOLO_PROVIDER_OPENAI_API_KEY; likewise SARVAM_API_KEY,
                               # ELEVENLABS_API_KEY, DEEPGRAM_API_KEY, SONIOX_API_KEY,
                               # ASSEMBLYAI_API_KEY
pytest -m real -k cloud
```

They are billed by the provider (a few seconds of audio and a few sentences each).
