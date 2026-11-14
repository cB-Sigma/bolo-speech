# Moving from speaches to bolo

bolo implements the HTTP interface that `speaches` (formerly
`faster-whisper-server`) clients use, so in most setups the switch is:
stop one container, start the other, keep the clients as they are.

This page is specific about what is the same, what differs and what is
missing. It was written against the speaches source at commit `993994f`
(17 April 2026).

## Quick switch

```bash
# speaches                                # bolo
docker run -p 8000:8000 \                 pipx install 'bolo-speech[all]'   # or the container image
  ghcr.io/speaches-ai/speaches:latest     UVICORN_HOST=0.0.0.0 bolo serve
```

Your existing environment keeps working: `API_KEY`, `PRELOAD_MODELS`,
`STT_MODEL_TTL`, `TTS_MODEL_TTL`, `VAD_MODEL_TTL`, `ALLOW_ORIGINS`,
`UVICORN_HOST`, `UVICORN_PORT`, `LOG_LEVEL`, `ENABLE_UI`,
`WHISPER__CPU_THREADS`, `WHISPER__NUM_WORKERS` and `HF_TOKEN` are read by
bolo (see [configuration.md](configuration.md)). One default differs on
purpose: **bolo binds 127.0.0.1**, speaches binds 0.0.0.0.

## Model names

speaches identifies models by Hugging Face repository id. bolo has short
catalog ids, and accepts the speaches ids of the same models as aliases:

| You send | bolo uses |
| --- | --- |
| `Systran/faster-whisper-small` (and `tiny`, `base`, `medium`, `large-v2`, `large-v3`, `*.en`) | `faster-whisper-small`, … |
| `Systran/faster-distil-whisper-large-v3` (and `small.en`, `medium.en`, `large-v2`) | `faster-whisper-distil-large-v3`, … |
| `deepdml/faster-whisper-large-v3-turbo-ct2` | `faster-whisper-large-v3-turbo` |
| `speaches-ai/Kokoro-82M-v1.0-ONNX` | `kokoro-v1.0` |
| `speaches-ai/Kokoro-82M-v1.0-ONNX-int8` | `kokoro-v1.0-int8` |
| `speaches-ai/piper-en_US-amy-low` (any Piper voice in the catalog) | `piper-en-us-amy-low` |
| `whisper-1` | the configured `default_stt` |
| `tts-1` | `default_realtime_tts` (`piper-en-us-amy-low`) |
| `tts-1-hd` | the configured `default_tts` (Kokoro) |

speaches ships a `model_aliases.json` that maps `whisper-1` to
`Systran/faster-whisper-large-v3`. bolo's default for `whisper-1` is
`whisper-base`; to keep the old behaviour set
`BOLO_DEFAULT_STT=faster-whisper-large-v3`.

Any other CTranslate2 Whisper repository (a fine-tune, for example) can be
added with `bolo pull hf:<owner>/<repo>`; it is then addressable by its
repository id, as in speaches.

Kokoro voice names are the same (`af_heart`, `am_michael`, `bf_emma`, …).
For Piper voices speaches uses the voice name as `voice`; in bolo each Piper
voice is its own model with a single voice, so `voice` can be omitted or set
to the speaker name (`amy`).

## Endpoints

| Endpoint | speaches | bolo |
| --- | --- | --- |
| `POST /v1/audio/transcriptions` | yes | yes; same form fields including `stream`, `timestamp_granularities[]`, `hotwords`; `vad_filter` is an explicit option (see below) |
| `POST /v1/audio/translations` | yes | yes (multilingual Whisper models) |
| `POST /v1/audio/speech` | yes | yes; `response_format` mp3, opus, aac, flac, wav, pcm; `stream_format` audio/sse; `sample_rate`; `speed` |
| `GET /v1/models`, `GET /v1/models/{id}` | yes | yes (installed models; same extra fields `language`, `task`) |
| `POST /v1/models/{id}` | download | yes: blocks until downloaded; 200 when downloaded, 201 when it already existed (as speaches does) |
| `DELETE /v1/models/{id}` | yes | yes |
| `GET /v1/registry?task=` | yes | yes (the whole catalog) |
| `GET /v1/audio/models`, `GET /v1/audio/voices` | yes | yes |
| `GET /api/ps`, `POST /api/ps/{id}`, `DELETE /api/ps/{id}` | yes | yes (same status codes: 201, 409, 404) |
| `POST /v1/audio/speech/timestamps` | yes (Silero VAD) | yes; returns `[{"start": ms, "end": ms}]` |
| `GET /health` | yes | yes (also `/healthz`, `/readyz`) |
| `POST /v1/chat/completions` (audio in/out, proxied to an LLM) | yes | yes; `CHAT_COMPLETION_BASE_URL` / `CHAT_COMPLETION_API_KEY` are read (see [voice-agents.md](voice-agents.md)) |
| `WS /v1/realtime` (OpenAI Realtime API) | yes | yes, over WebSocket, with the GA event names |
| `POST /v1/realtime` (WebRTC) | yes | **different route**: OpenAI's current `POST /v1/realtime/calls` (SDP offer in, answer out; extra `webrtc`), plus `POST /v1/realtime/client_secrets`. Tested with a WebRTC library, not with browsers |
| `POST /v1/audio/diarization` | yes (pyannote, needs PyTorch) | **different route**: `POST /api/v1/diarize`, or `diarize=true` / `diarized_json` on transcriptions (ONNX pipeline, no PyTorch) |
| `POST /v1/audio/speech/embedding` | yes | **no** |
| Gradio UI at `/` | yes | replaced by bolo's web console |
| OpenTelemetry export | yes | **no**; bolo has Prometheus `/metrics`, request traces and an SSE event stream instead |

If you depend on one of the rows marked **no**, stay on speaches for that
part, or run both side by side: the two servers can share a host on
different ports.

## Behaviour differences worth knowing

* **Errors.** speaches answers errors with FastAPI's `{"detail": …}`. bolo
  answers `{"error": {"message", "type", "param", "code"}, "detail": …}`:
  the OpenAI envelope, with `detail` kept for code that reads it.
* **Voice-activity filtering.** speaches runs Silero VAD over every upload
  before Whisper. bolo decodes clips that fit the model's window (30 s for
  Whisper) directly, which is faster, and uses VAD to split anything longer.
  Send `vad_filter=true` to force filtering on short clips, for example to
  stop Whisper inventing text on silence.
* **`verbose_json` language.** bolo returns the language name as OpenAI does
  (`"english"`); speaches returns the code (`"en"`).
* **Word timestamps** come from faster-whisper, Parakeet, Zipformer,
  SenseVoice and other CTC/transducer models. The sherpa-onnx Whisper exports
  (`whisper-*` without the `faster-` prefix) and Moonshine have none.
* **Default decoding.** bolo's faster-whisper engine uses greedy decoding
  (`beam_size` 1) and `condition_on_previous_text=false` by default, trading a
  little accuracy for speed and for immunity to repetition loops on long
  audio. Set `options: {beam_size: 5}` on a model to change it.
* **Piper voices** run through sherpa-onnx, not the GPL `piper-tts` package
  that speaches depends on (see
  [models.md](models.md#licences-and-the-espeak-ng-question)). Voices in
  uccharan's 39 languages (English, Hindi and the South Asian languages, Urdu,
  and Spanish, German, French, Italian, Portuguese, Dutch, Polish and the other
  European ones) get their phonemes from uccharan; other languages need
  espeak-ng.
* **Concurrency.** Requests for the same model are queued and micro-batched
  instead of each taking a model handle; when the queue is full bolo answers
  `429` with `Retry-After`.
* **No PyTorch.** bolo never imports torch, which is why the image is small.
  Diarization uses pyannote's segmentation model exported to ONNX with a
  separate speaker-embedding model, not the `pyannote.audio` pipeline, so
  results will differ from speaches'.
* **Piper voices outside uccharan's languages need a component speaches bundles.**
  speaches installs the GPL `piper-tts` package, which contains espeak-ng.
  bolo does not ship espeak-ng: install it once (console button, `bolo
  components install espeak-ng --accept-license`, or `pip install
  'bolo-speech[espeak]'`) if you use such voices. Until then they answer `424
  component_missing`. Piper voices in uccharan's languages use uccharan
  (`bolo-speech[all]`); its word table for a language is downloaded the first
  time a voice needs it.

## Checking a migration

```bash
# the same calls your clients make
curl -s localhost:8000/v1/models | jq '.data[].id'
curl -s localhost:8000/v1/audio/transcriptions -F file=@sample.wav -F model=Systran/faster-whisper-small
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"speaches-ai/Kokoro-82M-v1.0-ONNX","voice":"af_heart","input":"It still works."}' -o out.mp3
```

The compatibility tests in `server/tests/test_speaches_compat.py` and
`test_api_openai.py` are the executable version of this page.
