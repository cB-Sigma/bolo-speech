# Voice agents: the Realtime API and audio in chat completions

bolo can be the whole speech side of a voice agent. You bring a language
model, any server that speaks OpenAI's chat-completions protocol, and bolo
puts speech recognition in front of it and speech synthesis behind it, over
two interfaces that existing clients already use:

* **`WS /v1/realtime`**: the OpenAI Realtime API. Audio in, audio out, turn
  detection and interruption handled by the server.
* **`POST /v1/chat/completions`**: ordinary chat completions that also accept
  `input_audio` parts and can answer with audio.

```
 microphone ──► bolo: VAD ► speech-to-text ──► your LLM (Ollama, llama.cpp, vLLM, hosted…)
 speaker   ◄── bolo: text-to-speech  ◄── sentences, as the tokens arrive ──┘
```

Read this first:

* **bolo is a cascade, not a speech-to-speech model.** The language model
  receives text. It does not hear tone, laughter or hesitation, and it cannot
  choose how something is said.
* **The quality of the conversation is the quality of the model you connect.**
  bolo was developed with a 360M-parameter model on a CPU, which is enough to
  prove that every part works and far too small to hold a useful
  conversation. The latency figures below come from that setup and are **not
  representative**: with a real model on its own hardware, time to first
  token will be very different.
* Implemented against the event set of the generally-available Realtime API
  as typed in the official `openai` Python SDK **3.50.0** (November 2026):
  GA names such as `response.output_audio.delta`, not the older beta names.

## Connect a language model

```bash
# Ollama
ollama pull llama3.2
bolo serve --llm-base-url http://127.0.0.1:11434/v1 --llm-model llama3.2

# llama.cpp:   --llm-base-url http://127.0.0.1:8080/v1
# vLLM:        --llm-base-url http://127.0.0.1:8000/v1 --llm-model <served-model-name>   (run bolo on another port)
# hosted API:  --llm-base-url https://api.example.com/v1 --llm-api-key sk-… --llm-model …
```

or in `bolo.toml` / the environment:

```toml
[llm]
base_url = "http://127.0.0.1:11434/v1"
model = "llama3.2"
# api_key = "…"
# instructions = "You are a helpful voice assistant. Answer briefly, in plain spoken sentences."
```

`BOLO_LLM_BASE_URL`, `BOLO_LLM_MODEL`, `BOLO_LLM_API_KEY` (speaches'
`CHAT_COMPLETION_BASE_URL` and `CHAT_COMPLETION_API_KEY` are honoured too).
`bolo doctor` tells you whether the endpoint is reachable and knows the model;
`GET /api/v1/info` reports it under `capabilities.llm`.

Only this combination was run by the maintainers: llama-cpp-python's
OpenAI-compatible server with SmolLM2-360M-Instruct. **Ollama, llama.cpp's
own server, vLLM and hosted APIs were not tried**; they speak the same
protocol, and reports are welcome.

Which models are used:

| Stage | Model | Change it with |
| --- | --- | --- |
| speech to text | `default_realtime_stt` (`moonshine-tiny-en`) in Realtime sessions, `default_stt` in chat completions | `session.audio.input.transcription.model` / the `transcription_model` body field |
| language model | `llm_model` (if unset, the `model` the client sent is passed on) | configuration |
| text to speech | `default_tts` (`kokoro-v1.0-int8`) | `--default-tts`; the voice with `session.audio.output.voice` / `audio.voice` |

For low latency on CPU use a Piper-format voice as `default_tts` (it needs
the optional eSpeak NG component): first audio after about 150 ms per
sentence instead of 2 to 3 seconds for Kokoro on four cores.

## A working example: bolo + Ollama, with the OpenAI SDK

```python
# pip install "openai[realtime]" sounddevice numpy     (bolo serve --llm-base-url … must be running)
import asyncio, base64
import numpy as np, sounddevice as sd
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

async def main() -> None:
    loop = asyncio.get_running_loop()
    mic: asyncio.Queue[bytes] = asyncio.Queue()
    speaker = sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16")
    speaker.start()
    async with client.realtime.connect(model="gpt-realtime") as conn:
        await conn.session.update(session={
            "type": "realtime",
            "instructions": "You are a friendly voice assistant. Keep answers to one or two sentences.",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 400}},
                      "output": {"voice": "af_heart"}},
        })

        async def send_mic() -> None:
            with sd.RawInputStream(samplerate=24000, channels=1, dtype="int16", blocksize=2400,
                                   callback=lambda data, *_: loop.call_soon_threadsafe(mic.put_nowait, bytes(data))):
                while True:
                    await conn.input_audio_buffer.append(audio=base64.b64encode(await mic.get()).decode())

        task = asyncio.create_task(send_mic())
        async for event in conn:
            if event.type == "conversation.item.input_audio_transcription.completed":
                print("you:", event.transcript)
            elif event.type == "response.output_audio.delta":
                speaker.write(base64.b64decode(event.delta))
            elif event.type == "response.output_audio_transcript.done":
                print("assistant:", event.transcript)
            elif event.type == "input_audio_buffer.speech_started":
                speaker.abort(); speaker.start()          # barge-in: drop what is still queued for playback
            elif event.type == "error":
                print("error:", event.error.message)
        task.cancel()

asyncio.run(main())
```

Status of this script: the protocol exchange is what the test-suite and
`examples/realtime_agent.py` (which uses a file instead of a microphone) run;
the microphone and speaker parts were **not run** (no audio hardware), and
**Ollama was not available** to the maintainers.

## The Realtime API in bolo

Connect to `ws://<host>/v1/realtime?model=<anything>`. With an API key set,
send `Authorization: Bearer <key>` (the SDKs do), or from a browser the
WebSocket subprotocols `realtime` and `openai-insecure-api-key.<key>`.
`?intent=transcription` starts a transcription-only session.

### What is implemented

| Client event | Behaviour |
| --- | --- |
| `session.update` | `type` (`realtime` \| `transcription`), `instructions`, `output_modalities` (`["audio"]` or `["text"]`), `audio.input.format` / `audio.output.format` (`audio/pcm` at 24 kHz, `audio/pcmu`, `audio/pcma`), `audio.input.transcription` (`model`, `language`, `prompt`), `audio.input.turn_detection` (`server_vad` with `threshold`, `silence_duration_ms`, `prefix_padding_ms`, `create_response`, `interrupt_response`; or `null`), `audio.output.voice`, `audio.output.speed`, `tools` (functions), `tool_choice`, `max_output_tokens` |
| `input_audio_buffer.append` / `.commit` / `.clear` | base64 audio in the session's input format; manual commit when turn detection is off |
| `conversation.item.create` | user/assistant/system messages (`input_text`, `input_audio` with base64 audio that is transcribed, `output_text`), `function_call`, `function_call_output` |
| `conversation.item.delete` / `.retrieve` / `.truncate` | `truncate` cuts an assistant message's transcript to the sentences that had started playing before `audio_end_ms`, so the model does not "remember" words the user never heard |
| `response.create` | optional `instructions`, `output_modalities`, `tools`, `tool_choice`, `max_output_tokens`, `input`, `audio.output.voice` for this response |
| `response.cancel`, `output_audio_buffer.clear` | stop the response in progress |

| Server event | Notes |
| --- | --- |
| `session.created`, `session.updated` | the full session object |
| `input_audio_buffer.speech_started` / `.speech_stopped` / `.committed` / `.cleared` | `audio_start_ms` / `audio_end_ms` are positions in the input stream |
| `conversation.item.added`, `conversation.item.done`, `.deleted`, `.retrieved`, `.truncated` | |
| `conversation.item.input_audio_transcription.delta` / `.completed` / `.failed` | sent for every committed utterance. One delta with the whole text, then `completed` with `usage: {"type": "duration"}` |
| `response.created`, `response.output_item.added` / `.done`, `response.content_part.added` / `.done`, `response.done` | `response.done` carries `status` (`completed`, `cancelled` with `status_details.reason` `turn_detected` or `client_cancelled`, `incomplete`, `failed`) and token `usage` when the language model reports it |
| `response.output_audio.delta` / `.done`, `response.output_audio_transcript.delta` / `.done` | audio in the session's output format; transcript deltas are the model's tokens as they arrive |
| `response.output_text.delta` / `.done` | when `output_modalities` is `["text"]` |
| `response.function_call_arguments.delta` / `.done` | when the model calls one of the session's functions (see below) |
| `error` | `{type, code, message, param, event_id}`; the connection stays open |

**Turn taking.** With `server_vad` (the default) bolo detects the end of an
utterance (Silero VAD; `silence_duration_ms` defaults to the server's
`endpoint_silence_ms`, 500), commits it, transcribes it, and starts a
response. If the user starts speaking while a response is being produced it
is cancelled at once (`interrupt_response`, on by default): `response.done`
arrives with `status: "cancelled"`, `reason: "turn_detected"`, and the
assistant message stays in the history only as far as it had been generated.
Stop your playback when you see `input_audio_buffer.speech_started`, and send
`conversation.item.truncate` with how much was actually played if you want
the history to match what the user heard.

**Speech starts before the answer is complete.** The model's tokens are cut
into sentences as they arrive and each sentence is synthesised while the
model keeps writing. Time to first audio is therefore the time to the end of
the model's *first sentence* plus the synthesis of that sentence.

**Function calling.** `tools` of type `function` are passed to the language
model as chat-completions tools. When the model answers with a tool call you
get `response.output_item.added` (a `function_call` item),
`response.function_call_arguments.delta` / `.done`, and `response.done`. Run
the function, send its result as a `function_call_output` item and then
`response.create`. This requires a model and server that support tool calls
in streaming mode. It is tested with a scripted fake model only: **no real
model's tool calling was exercised.**

### What is different from OpenAI, or missing

* A cascade: no speech-to-speech model, no emotional prosody control, no
  `semantic_vad` (accepted, treated as `server_vad`), no noise reduction.
* Input transcription events are always sent (bolo has to transcribe anyway),
  and arrive as one delta rather than incrementally.
* Not implemented: WebRTC and SIP (`POST /v1/realtime/calls`, the
  `output_audio_buffer.*` server events), ephemeral client secrets
  (`/v1/realtime/client_secrets`), MCP tools, image input, `prompt`
  templates, `reasoning`, `tracing`, `truncation` settings, log probabilities,
  DTMF, idle timeouts, rate-limit events, the realtime translation sessions.
  Unsupported settings in `session.update` are ignored unless they would
  change behaviour silently (MCP tools and images are rejected with an error).
* `model` in the URL is echoed back in the session; the language model that
  answers is the operator's `llm_model`.
* Voices are those of bolo's text-to-speech model; OpenAI voice names are
  mapped onto them.

### Latency, and where to look

Every conversation turn is a request trace (`GET /api/v1/requests?kind=agent`,
the console's Latency screen, the `request` events on `/api/v1/events`) with
`transport: "realtime"`, model `voice-agent` and, measured **from the moment
the end of speech is detected**:

| Field | Meaning |
| --- | --- |
| `final_ms` | the transcript is ready |
| `first_token_ms` | the language model's first token arrived |
| `ttfa_ms` | the first audio of the answer was sent |
| `total_ms` | the response finished |
| phases | `decode` (speech-to-text), `llm_first_token`, `synthesize_first` (first token → first audio: the rest of the first sentence, then its synthesis), `synthesize_rest` |
| extras | `stt_model`, `llm_model`, `user_text`, `audio_seconds_out`, `response_id`, `tool_calls` |

`/api/v1/stats` and `/metrics` aggregate them under the model id
`voice-agent`. Add the endpointing silence (500 ms by default) for the delay
a user perceives after they stop talking.

Measured here (three turns each, one at a time, the language model sharing
the same four CPU cores, see `docs/measurements/voice-agent-smollm2.json`):

| Voice | Transcript | First LLM token | First audio |
| --- | --- | --- | --- |
| `piper-en-us-amy-low` | 339 ms | 494 ms | 2,373 ms |
| `kokoro-v1.0-int8` | 253 ms | 361 ms | 3,762 ms |

(medians). **Do not plan with these numbers**: a 360M model on two CPU
threads, competing with recognition and synthesis for the same cores, says
little about your deployment. What they do show is the shape of a turn: the
transcript and the first token are quick, and the time to first audio is
dominated by how long the model needs to finish its first sentence.

## Audio in chat completions

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",                      # replaced by your llm_model
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Answer the question in this recording."},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(open("q.wav", "rb").read()).decode(), "format": "wav"}},
    ]}],
)
print(completion.choices[0].message.audio.transcript)
open("answer.wav", "wb").write(base64.b64decode(completion.choices[0].message.audio.data))
```

What happens:

1. every `input_audio` part is transcribed and replaced by a text part
   (messages that end up text-only are sent as plain strings);
2. `modalities` and `audio` are removed and the request is forwarded to the
   language model with **every other field untouched** (`temperature`,
   `tools`, `response_format`, vendor extensions…); `model` is replaced by
   `llm_model` when that is set;
3. without audio output the model's response is returned as it is. With
   `modalities` containing `"audio"`, the reply text is synthesised and
   returned as `message.audio` (`id`, `data`, `expires_at`, `transcript`)
   with `message.content` set to null, as OpenAI does;
4. with `stream: true` you get chat-completion chunks whose
   `delta.audio` carries `transcript` pieces (the model's tokens) and `data`
   pieces (base64 audio, sentence by sentence, in the requested format;
   `pcm16` is the natural choice), then a final chunk with the finish reason.

Requests with no `input_audio` parts and no audio output are **proxied
unchanged** (streaming included), so a client can use bolo's base URL for
all its chat calls. `audio.format`: `wav`, `mp3`, `flac`, `opus`, `aac`,
`pcm16`. Extensions that are read and not forwarded: `transcription_model`,
`transcription_language`, `speech_model`. `expires_at` is a formality: bolo
does not store audio, so `audio.id` cannot be referenced in a later request.

Errors: `503 llm_not_configured` when no language model is set,
`502 llm_error` when it cannot be reached or answers with an error (for
proxied requests the model's own status and body are passed through),
`400` for malformed audio parts.

## Using bolo from agent frameworks

Both snippets below are **untested**: the frameworks were not installed while
bolo was developed. They rely only on those frameworks' documented ability to
point their OpenAI plugins at another base URL.

**Pipecat**, with bolo's Realtime endpoint as the speech-to-speech service:

```python
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService   # name as of Pipecat's docs; check your version

s2s = OpenAIRealtimeLLMService(
    api_key="not-needed",
    base_url="ws://127.0.0.1:8000/v1/realtime",
    model="gpt-realtime",
)
# pipeline: transport.input() -> s2s -> transport.output()
```

or with bolo as separate STT and TTS services around Pipecat's own LLM step:
see `examples/pipecat_bot.py`.

**LiveKit Agents**:

```python
from livekit.agents import AgentSession
from livekit.plugins import openai

session = AgentSession(
    llm=openai.realtime.RealtimeModel(base_url="http://127.0.0.1:8000/v1", api_key="not-needed", model="gpt-realtime", voice="af_heart"),
)
```

or as separate `openai.STT(base_url=…)` / `openai.TTS(base_url=…)` plugins:
see `examples/livekit_agent.py`. If a framework insists on features bolo does
not have (WebRTC, semantic VAD, MCP), use the separate STT/TTS route.
