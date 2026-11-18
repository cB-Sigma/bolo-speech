# Voice agents: the Realtime API and audio in chat completions

bolo can be the whole speech side of a voice agent. You bring a language
model, any server that speaks OpenAI's chat-completions protocol, and bolo
puts speech recognition in front of it and speech synthesis behind it, over
two interfaces that existing clients already use:

* **`WS /v1/realtime`**: the OpenAI Realtime API. Audio in, audio out, turn
  detection and interruption handled by the server. The same sessions are
  available over **WebRTC** (`POST /v1/realtime/calls`) for browsers.
* **`POST /v1/chat/completions`**: ordinary chat completions that also accept
  `input_audio` parts and can answer with audio.

```
 microphone ──► bolo: VAD ► speech-to-text ──► your LLM (Ollama, llama.cpp, vLLM, hosted…)
 speaker   ◄── bolo: text-to-speech  ◄── sentences, as the tokens arrive ──┘
```

Read this first:

* **By default bolo is a cascade, not a speech-to-speech model.** The language
  model receives text. It does not hear tone, laughter or hesitation, and it
  cannot choose how something is said. Two opt-in pipelines change that
  ([below](#pipelines-cascade-audio-in-language-model-speech-to-speech)): a
  language model that takes audio in, and a speech-to-speech model.
* **The quality of the conversation is the quality of the model you connect.**
  bolo was developed with models of 360M and 500M parameters on a shared
  CPU, which is enough to prove that every part works and far too small to
  hold a useful conversation. The latency figures below come from that setup
  and are **not representative**: with a real model on its own hardware, time
  to first token will be very different.
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

Run by the maintainers, each with a real model: **Ollama** 0.40.16
(`qwen2.5:0.5b`), **llama.cpp's `llama-server`** (the same GGUF file) and
llama-cpp-python's server (SmolLM2-360M-Instruct). **vLLM and hosted APIs
were not tried**; they speak the same protocol, and reports are welcome.

On a machine whose cores are shared with other work, give the language model
fewer threads than cores (Ollama: `PARAMETER num_thread 2` in a Modelfile;
llama-server: `-t 2`) and bolo too (`--threads 2`). With the default of one
thread per core on a busy four-core machine we saw Ollama fall below one
token per second; with two threads the same model answered in about a
second.

Which models are used:

| Stage | Model | Change it with |
| --- | --- | --- |
| speech to text | `default_realtime_stt` (`moonshine-tiny-en`) in Realtime sessions, `default_stt` in chat completions | `session.audio.input.transcription.model` / the `transcription_model` body field |
| language model | `llm_model` (if unset, the `model` the client sent is passed on) | configuration |
| text to speech | `default_realtime_tts` (`piper-en-us-amy-low`) | `default_realtime_tts`; `speech_model` in chat completions; the voice with `session.audio.output.voice` / `audio.voice` (a Kokoro voice such as `af_heart` selects Kokoro); `session.audio.output.phonemizer` / `audio.phonemizer`; `session.audio.output.romanised` / `audio.romanised` (`off`, `auto`, `on`: a language model that answers in romanised Hindi is read in Devanagari; api.md, "Romanised text") |

Agents speak with the low-latency voice by default: Piper `amy` starts in
about 150 ms per sentence (its phonemes come from uccharan, so nothing
optional is needed), against 2 to 3 seconds for Kokoro on this CPU (see
models.md, "Default voices"). Name a Kokoro voice, or set
`default_realtime_tts = "kokoro-v1.0-int8"`, for Kokoro's quality.

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
`examples/realtime_agent.py` (which uses a file instead of a microphone) run,
and that exchange was run against Ollama; the microphone and speaker parts
were **not run** (no audio hardware).

## The Realtime API in bolo

Connect to `ws://<host>/v1/realtime?model=<anything>`. With an API key set,
send `Authorization: Bearer <key>` (the SDKs do), or from a browser the
WebSocket subprotocols `realtime` and `openai-insecure-api-key.<key>`.
`?intent=transcription` starts a transcription-only session.

### What is implemented

| Client event | Behaviour |
| --- | --- |
| `session.update` | `type` (`realtime` \| `transcription`), `instructions`, `output_modalities` (`["audio"]` or `["text"]`), `audio.input.format` / `audio.output.format` (`audio/pcm` at 24 kHz, `audio/pcmu`, `audio/pcma`), `audio.input.transcription` (`model`, `language`, `prompt`), `audio.input.noise_reduction` (`{"type": "near_field"}` or `"far_field"`: the default denoiser before transcription; `null`: the server's default; see below), `audio.input.denoise` (bolo extension: `true`, `false`, a strength between 0 and 1, a model id, or `{"model", "strength"}`; wins over `noise_reduction`), `audio.input.echo_cancellation` (bolo extension: `true`, `false`, a model id or `{"model", "delay_ms"}`; see "Echo cancellation" below), `audio.input.turn_detection` (`server_vad` with `threshold`, `silence_duration_ms`, `prefix_padding_ms`, `create_response`, `interrupt_response`; `semantic_vad` with `eagerness`, see below; or `null`; bolo extensions in either: `vad_model`, the voice-activity detector of the session, any `vad` model, default `default_vad`; `turn_model`, the end-of-turn model `semantic_vad` asks, any `turn` model, default `default_turn`; a model named in either is pulled on first use with `auto_pull`, as anywhere else, and never replaced by another: when it cannot be used (its engine is missing, or it is not installed and `auto_pull` is off) `session.update` answers with an `error` naming the parameter, with the code and message loading the model gives), `audio.output.voice`, `audio.output.speed`, `tools` (functions, and MCP servers: below), `tool_choice`, `max_output_tokens` |
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
| `conversation.item.input_audio_transcription.delta` / `.completed` / `.failed` | sent for every utterance. Deltas arrive **while the user is still speaking** (the utterance so far is re-decoded every `partial_interval_ms` and a word is sent once two decodes agree on it), the rest right after the end of speech. A delta cannot be taken back, so when the final decode revises something already sent (usually punctuation or capitalisation, sometimes a word) the concatenated deltas differ slightly from the `transcript` of `completed`, which is the authoritative text and also has `usage: {"type": "duration"}` |
| `output_audio_buffer.started` / `.stopped` / `.cleared` | WebRTC only: what the outgoing audio track is doing |
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

**Semantic turn detection (optional).** `turn_detection: {"type":
"semantic_vad"}` adds an end-of-turn model: the voice-activity detector
reports a pause after 300 ms of silence and the model says whether the
speaker has finished. If so the turn ends at once; if not, the turn stays
open until the speaker continues or the pause has lasted 1.5 s
(`eagerness: "high"`), 3 s (`"medium"`, `"auto"`) or 6 s (`"low"`). The
`default_turn` setting chooses the model, and a session can choose its own
with the bolo extension `turn_detection.turn_model`:

| id | judges | size | licence | runs in bolo |
| --- | --- | --- | --- | --- |
| `smart-turn-v3` (default), `smart-turn-v3.1`, `smart-turn-v3.0`, `smart-turn-v3-fp32` | the last 8 s of audio; 23 languages | 9 MB (fp32: 32 MB) | BSD-2-Clause | yes (`onnxruntime`) |
| `namo-turn-en`, `namo-turn-multilingual` | the transcript; English, or 23 languages | 139 MB, 343 MB | Apache-2.0 | yes (`onnxruntime`, `tokenizers`) |
| `smart-turn-v2` | up to 16 s of audio; 14 languages | 379 MB | BSD-2-Clause | yes (NumPy, built in; seconds per check on a small CPU) |
| `ten-turn-detection` | the transcript (finished / unfinished / wait); English, Chinese | 4.7 GB | Apache-2.0 with Agora's conditions ⚠ | yes (llama-server; a GPU is recommended) |
| `silence-only` | nothing: the pause decides | none | - | yes (built in) |

LiveKit's turn detector is not run by bolo: its licence allows use only with
the LiveKit Agents framework ([models-unsupported.md](models-unsupported.md)).

Models that read text need words, so when the pause comes the session
decodes the utterance so far with its transcription model (one extra
decode per pause; not done for cloud recognisers, which bill every decode)
and asks the model about that text. Without a transcript, or without the
model (not installed and `auto_pull` off, no `onnxruntime`, or
`silence-only`), `semantic_vad` behaves exactly like `server_vad`;
`GET /api/v1/info` says which under `capabilities.realtime.semantic_vad`
(with `input`: `audio` or `text`). Each check is recorded on the turn's
trace (`turn_detection.checks`: probability and milliseconds, and for a
text model the text and the time the decode took).

What we measured with it (`docs/measurements/semantic-vad-smart-turn.json`;
the built-in English sample through `moonshine-tiny-en`, three runs each, on
a busy 4-vCPU machine):

| Audio | `server_vad` (500 ms) | `semantic_vad` |
| --- | --- | --- |
| the whole sentence | 1 turn; end of turn reported 632 ms after the last word | 1 turn; 738 ms |
| a 0.9 s pause after "... tell me when" (unfinished) | **2 turns**: "Good morning. Could you tell me when?" / "My order will arrive." | **1 turn**, whole sentence; 942 ms |
| a 0.9 s pause after "Could you tell me" | 2 turns | 2 turns: the model takes "Could you tell me" for a finished sentence (probability 0.92) |

One inference took 160 to 260 ms on two threads of that machine. On isolated
clips the model gave 0.99 for the complete English sentence and 0.02 to 0.07
for three mid-sentence cuts. On four Hindi sentences (FLEURS) it was much
less decisive: 0.43 to 0.94 for the complete sentences and 0.05 to 0.86 for
the same recordings cut in half. So: it keeps a hesitating speaker's turn
together when the pause falls mid-phrase, it costs roughly a hundred
milliseconds on finished sentences, and it is a judgement about *sound*, not
grammar. It is off unless a client asks for it. No evaluation on real
conversational recordings was done.

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

**MCP tools.** A `tools` entry of type `mcp` connects the session to a Model
Context Protocol server; bolo calls its tools for the language model, so the
client does not have to. It needs the MCP Python SDK
(`pip install 'bolo-speech[mcp]'`, MIT; 1.x and 2.x both work).

```jsonc
// a server the client names (Streamable HTTP; SSE when the URL ends in /sse)
{"type": "mcp", "server_label": "docs", "server_url": "https://example.com/mcp",
 "allowed_tools": ["search"], "require_approval": "never",
 "headers": {"X-Api-Key": "…"}, "authorization": "…"}     // authorization: sent as a bearer token
// a server the operator configured (setting mcp_servers): the label alone
{"type": "mcp", "server_label": "files"}
```

```toml
# bolo.toml: servers for every session to use by label. A command starts a local server (stdio) per session;
# a session can never start a process itself. require_approval defaults to "never" here.
[mcp_servers.files]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/srv/shared"]
[mcp_servers.search]
url = "https://search.internal/mcp"
headers = { Authorization = "Bearer …" }
allowed_tools = ["search"]
```

What happens, with OpenAI's event names:

1. After `session.updated`, each server is opened: a `conversation.item.added`
   with an `mcp_list_tools` item, `mcp_list_tools.in_progress`, then
   `mcp_list_tools.completed` and `conversation.item.done` with the tools filled
   in (or `mcp_list_tools.failed` and an `error`, code `mcp_list_tools_failed`).
2. The tools are offered to the language model as functions, next to the
   session's own (a name taken already becomes `<server_label>__<name>`).
3. When the model calls one and no approval is needed, the response carries an
   `mcp_call` item: `response.mcp_call_arguments.delta` / `.done`,
   `response.mcp_call.in_progress`, then `.completed` (with `output`) or
   `.failed` (with `error`). bolo then starts a new response by itself, in which
   the model answers with the result. At most eight responses in a row end in MCP
   calls; then bolo stops with an error (`mcp_rounds`).
4. With `require_approval` `"always"` (the default for a session's own server,
   as at OpenAI) or `{"never": {"tool_names": [...]}, "always": {...}}`, the
   response carries an `mcp_approval_request` item instead. Answer with
   `conversation.item.create` and `{"type": "mcp_approval_response",
   "approval_request_id": "…", "approve": true}`: bolo runs the call in a response
   of its own and lets the model go on. With `"approve": false` the model is told
   the call was not allowed.

A server is opened once per session and closed with it; a `session.update`
with other tools closes the servers no longer named. A `server_url` given by a
client makes bolo connect where the client says, so by default it is accepted
only on a loopback bind or when requests need an API key; `realtime_mcp_urls`
`true` or `false` decides it outright (`false`: only the configured servers). Headers are shown as `***` in
`session.updated`. MCP tools work with the `cascade` and `audio-llm` pipelines,
not with a speech-to-speech model. Not implemented: `connector_id` (OpenAI's own
connectors). `capabilities.realtime.mcp` in `/api/v1/info` says whether the SDK
is installed, which servers are configured and whether a session may give a
URL. In the console, the Conversation page has a **Tools (MCP)** panel: tick
configured servers or give one by URL, then follow each server's tools and each
call, and allow or decline calls that need approval. Tested with a protocol-level fake
MCP server over Streamable HTTP and stdio, with mcp 1.30 and 2.3.

### Pipelines: cascade, audio-in language model, speech-to-speech

What answers a conversation turn is the session's *pipeline*: the setting
`realtime_pipeline`, or per session the bolo extension field
`session.pipeline` in `session.update` (`session.created` reports it). It
can change until the first audio or item. In every pipeline bolo keeps the
session itself: authentication and client secrets, the event set,
`audio/pcmu` / `audio/pcma` and resampling, turn detection (`server_vad`,
`semantic_vad`, manual commits), barge-in, conversation items, tool-call
items, user transcripts from its own STT, and traces.

| `pipeline` | Who hears the user | Who answers | Who speaks | Needs |
| --- | --- | --- | --- | --- |
| `cascade` (default) | bolo's STT | the `llm_*` chat-completions model | bolo's TTS | `llm_base_url` |
| `audio-llm` | the model itself (the turn's audio as `input_audio`, WAV) | an audio-understanding model, `audio_llm_*` (falls back to `llm_*`) | bolo's TTS | a chat-completions server that takes `input_audio` |
| `speech-to-speech` | the model itself (16 kHz PCM, streamed while the user speaks; for turn-based models, sent when the turn ends) | a speech-to-speech model | the same model | `s2s_server_url`, or `s2s_model` + llama-server, or `s2s_model = "lfm3-audio-2b-onnx"` + `bolo-speech[s2s]`, or `s2s_model` naming a turn-based model of an engine plugin ; or a full-duplex model (below: it hears everything, all the time, and answers by itself) |

**`audio-llm`.** The latest user turn is sent to the model as audio, earlier
turns as their transcripts (or as audio while a transcript is missing; the
last eight turns' audio is kept). Tool calls work as in the cascade. bolo's
STT still transcribes each turn, in parallel with the response
(`realtime_transcribe_user`), for the transcript events and the history, and
bolo's TTS speaks the answer, so any of bolo's voices (Indic included) can
speak what the model says. llama-server accepts `input_audio` content in
`/v1/chat/completions` for its audio models (tools/server/README.md and
docs/multimodal.md of llama.cpp): Gemma 4 E2B / E4B
(`llama-server -hf ggml-org/gemma-4-E4B-it-GGUF`, Apache-2.0), Ultravox 0.5
(`ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF`; Llama 3.2 licence), Voxtral,
Qwen2.5-Omni and Qwen3-Omni. Whether Gemma 4 understands spoken Hindi well
was **not measured**.

```toml
realtime_pipeline = "audio-llm"
audio_llm_base_url = "http://127.0.0.1:8081/v1"   # llama-server -hf ggml-org/gemma-4-E4B-it-GGUF --port 8081
```

Or let bolo run the model: without `audio_llm_base_url`, an `audio_llm_model` that names a catalog entry of task
`audio-llm` is pulled (pinned files, SHA-256 checked) and served by a llama-server bolo starts on a loopback port with
a random API key and `--parallel audio_llm_parallel`, placed on the GPU or the CPU like the speech-to-speech server,
restarted after a crash and stopped with bolo (`bolo components install llama-server` first). The catalog has
LFM2.5-Audio 1.5B, Ultravox 0.5 (Llama 3.2 1B), Qwen2.5-Omni 3B and 7B, Qwen3-Omni 30B-A3B and Gemma 4 E2B and
E4B, each with its licence ([models.md](models.md#speech-to-speech-and-audio-in-models)). On llama.cpp these hear
but do not speak (the Qwen-Omni Talker does not run there), so bolo's TTS speaks their answers.

```toml
realtime_pipeline = "audio-llm"
audio_llm_model = "ultravox-0.5-llama-3.2-1b-llamacpp"   # 2.2 GB, MIT + Llama 3.2 Community License
```

Only `lfm2.5-audio-1.5b-llamacpp` was run here, on an overloaded CPU: the managed server started and the model
heard the question, but it transcribed it instead of answering (models.md has the details).

**`speech-to-speech`.** The model behind it is a small backend interface in
`bolo/s2s.py` (append the turn's audio, commit, add user text or a tool
result, stream a response of text, audio and tool calls, cancel), so the
session logic does not depend on any one server. The first backend is
llama.cpp's `llama-server` and its WebSocket `/v1/realtime` in conversation
mode, which runs **LFM3-Audio 2B** (and LFM3-Omni) with tool calls: English
conversation only (its speech languages do not include Hindi), CPU-capable,
under the **LFM Open License v1.0** (commercial use only for organisations
below USD 10M yearly revenue; bolo's default pipeline stays the cascade).
Either run llama-server yourself:

```sh
llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF --port 8090 --parallel 2
bolo serve   # with s2s_server_url = "http://127.0.0.1:8090" and realtime_pipeline = "speech-to-speech"
```

or let bolo start one: `bolo components install llama-server`, then the
default `s2s_model = "lfm3-audio-2b-llamacpp"` (the same GGUF files as that
TTS catalog entry, pulled on first use) is served by a llama-server on a
loopback port with a random API key and `--parallel s2s_parallel`, restarted
after a crash with a back-off as TTS servers are. Each Realtime session holds
one upstream session (one llama-server slot).

or run the model **in bolo's own process**, without llama-server:

```sh
pip install "bolo-speech[s2s]"          # onnxruntime and tokenizers
bolo pull lfm3-audio-2b-onnx     # 2.7 GB, every file pinned by SHA-256
bolo serve                       # with s2s_model = "lfm3-audio-2b-onnx" and realtime_pipeline = "speech-to-speech"
```

`lfm3-audio-2b-onnx` is the same LFM3-Audio 2B from Liquid's ONNX export
(`LiquidAI/LFM3-Audio-2B-ONNX`, the 4-bit language model with the float32
audio encoder, depth decoder and codec decoder; the same licence). bolo runs
the four graphs with ONNX Runtime and NumPy (`bolo/s2s_onnx.py`) and follows
the model's own generation procedure: the turn's audio is encoded and
evaluated while the user is still speaking, the answer is interleaved text
and speech (six text tokens, then twelve 80 ms speech frames, each frame from
three passes of the depth decoder), the codec turns the frames into 24 kHz
audio in chunks of about half a second, and tool calls are parsed out of the
text. The session's instructions follow the sentence the model was trained
with (`Respond with interleaved text and audio.`), the tools go into the
system prompt, a barge-in stops the model before its next step, and the
conversation stays in the model's context from turn to turn (it starts over
after 8192 positions, about seven minutes of speech). One copy of the model
(about 3.4 GB of memory) serves every session; sessions take turns step by
step, so two conversations at once each run at half speed. `voice` is not
used: in a conversation the model has one voice. The same catalog entry is
also a text-to-speech voice (`female`, `male`; 18 languages).

Measured on 4 shared CPU cores (Xeon 2.1 GHz, other jobs running, load 2 to
4): first audio 0.8 to 1.2 s after the end of the user's turn, then about 7.5
steps per second where real time needs about 19, so speech comes out at
0.4 to 0.5 times the speed it is spoken (RTF 2.2 to 2.5) and playback
pauses. On that machine it shows that the pipeline works, not that it is
usable; real time needs a CPU about two and a half times as fast (8 or more
modern cores) or a GPU (precision `q4f16`, not tested). What was checked
against llama.cpp: the same tokenisation of the prompts, the same number of
audio tokens for the same recording, and the same answer text to a spoken
question; the two 4-bit quantisations differ, so longer texts drift apart
after some tens of tokens. Without `bolo-speech[s2s]` a speech-to-speech session
answers 424 (`component_missing`) with the command to run, and
`GET /api/v1/info` shows the pipeline as configured but not available.

**Turn-based models in an engine plugin.** A catalog entry with `task: speech-to-speech` (for example bolo-torch's
Qwen2.5-Omni and Qwen3-Omni, see [models.md](models.md#speech-to-speech-and-audio-in-models)) is loaded by bolo's
model manager like any other model (GPU/CPU placement, keep-alive, unloading when idle) and answers over the plugin
protocol's `respond` op ([plugins.md](plugins.md)). Such a model keeps no state: bolo keeps the conversation (the
user's turns as audio, the last eight spoken turns, the model's answers as text, tool calls and their results) and
sends all of it with every response, so one copy serves every session. Unlike llama-server's realtime mode the
turn's audio reaches the model only when the turn ends, so the time to first audio includes the whole prefill.
Text, audio and tool calls (when the model supports tools) stream back as the model produces them; a barge-in stops
it before its next piece. Tested with a fake model in a plugin process.

A cloud model can stand in the same place: `s2s_model = "openai:gpt-realtime-2.1"`
(`<provider>:<model>`, a provider of kind `openai-compatible` from the
`providers` setting) makes **OpenAI's Realtime API** the backend, over
WebSocket. It is a paid cloud API and the conversation's audio leaves this
server. bolo resamples each turn to the 24 kHz that API takes, switches its
server-side turn detection off (bolo detects turns and barge-in itself), and
asks for audio responses whose transcript is the text (that API answers with
audio or text, never both). See [providers.md](providers.md#cloud-speech-to-speech-openai-realtime).
Gemini Live is not supported.

How a turn maps onto the upstream protocol: when bolo's turn detection hears
speech start, the audio from `prefix_padding_ms` before it is sent upstream
and the rest follows as it arrives (llama-server prefills it while the user
speaks); at the end of the turn bolo sends `input_audio_buffer.commit` and
then `response.create`. Barge-in sends `response.cancel` and drops what is
left of the cancelled response. With turn detection off, all appended audio
is forwarded and the client's commit is passed on. User text items and
`function_call_output` items are forwarded; assistant and system messages
created by the client stay in bolo's history only. The model's text becomes
`response.output_audio_transcript.delta` (or `response.output_text.delta`
for `["text"]` sessions), its audio (PCM16 at the model's rate) is resampled
and encoded to the session's output format, and its tool calls become
`function_call` items as in the cascade. Limits: `instructions`, `tools` and
the voice reach the model only when its session opens (the first turn);
`input_audio_buffer.clear` cannot take back audio already forwarded; OpenAI
voice names are not sent (LFM3's voices are `female` and `male`); upstream
token usage is output tokens only. Tested against a scripted fake
llama-server, **not against a real LFM3-Audio model.**

### Echo cancellation

A caller without headphones sends back what the agent says. bolo then hears the agent's own voice as the user
speaking: it cuts the answer short (barge-in) and transcribes the agent's words as a user turn. Browsers cancel
echo on the microphone themselves (`echoCancellation`), but phones, SIP trunks, embedded devices and clients that
send raw audio often do not. `audio.input.echo_cancellation` (a bolo extension) removes it on the server: the
audio the session sends out is the reference, and the input is cleaned before the voice-activity detector, the
recogniser and a speech-to-speech model hear it.

```jsonc
{"type": "session.update", "session": {"audio": {"input": {
  "echo_cancellation": true                          // the server's default canceller (default_echo_cancellation)
  // or "webrtc-aec3", "dtln-aec-128", or {"model": "dtln-aec-128", "delay_ms": 120}, or false
}}}}
```

| Model | What it is | Licence | Install |
| --- | --- | --- | --- |
| `speex-aec` (default) | SpeexDSP's adaptive filter (400 ms long) with its residual-echo suppressor | BSD-3-Clause, MIT binding | `pip install 'bolo-speech[aec]'` (pyaec, 0.2 MB) |
| `webrtc-aec3` | WebRTC's AEC3, the canceller in Chrome; finds the delay itself | BSD-3-Clause, Apache-2.0 SDK | `pip install 'bolo-speech[aec]'` (LiveKit's SDK, 11 MB) |
| `dtln-aec-128` | DTLN-aec, a small LSTM network (1.8 M parameters) | MIT | pulled on first use (7 MB), ONNX Runtime |

The reference is released at the pace the input arrives, which is the pace the client plays it, so it leads its
echo by about the round trip plus the client's buffer. It is dropped when the client stops playing: on barge-in,
`response.cancel` and `output_audio_buffer.clear`. `delay_ms` holds the reference back by a fixed time first, for
a canceller that does not find the delay itself: DTLN-aec wants its reference within a few tens of milliseconds of
the echo. `session.updated` reports the canceller in use, how much audio it has cleaned and how much of that had a
reference (`with_reference_s`). When the canceller cannot be loaded (its package missing), the session goes on
without it and sends an `error` with code `echo_cancellation_unavailable`. The setting `echo_cancellation` turns
it on for every session and phone call that does not say otherwise.

Measured on CPU on a synthetic echo (the 11.8 s English sample through a 60 ms echo path with a 250 ms tail, the
3 s English sample spoken over it from 7 s; echo return loss enhancement while only the agent talks, and the level
of the user's voice while both talk):

| Model | Echo removed | User's voice while both talk | Real-time factor |
| --- | --- | --- | --- |
| `speex-aec` | 36.6 dB | -7.1 dB | 0.026 |
| `webrtc-aec3` | 33.0 dB | -6.1 dB | 0.026 |
| `dtln-aec-128` | 19.7 dB | -1.3 dB | 0.23 |

With a 250 ms echo delay SpeexDSP removes 26.8 dB, AEC3 34.3 dB, and DTLN-aec nothing unless `delay_ms` (240)
brings it back to 33.9 dB. Moonshine tiny transcribed the user's sentence correctly from SpeexDSP's and DTLN-aec's
output while both talked, and not from AEC3's, which suppresses double talk harder in its first seconds. A real
room has more reverberation than this test: expect less.

The console's Conversation page has an Echo cancellation choice next to the denoiser.

### Full-duplex models: Moshi, Hibiki, PersonaPlex, FLM-Audio, Freeze-Omni, MiniCPM-o 4.5

A *full-duplex* model listens and speaks at the same time, frame by frame (Moshi: 80 ms frames, about 200 ms of
latency on a GPU). There are no turns: it decides itself when to answer, keeps listening while it talks, and stops
(or not) when you talk over it. In bolo such a model is the `speech-to-speech` pipeline's model: set `s2s_model` to a
catalog entry with the feature `full-duplex` and every Realtime session (WebSocket or WebRTC) of that pipeline talks
to it. Turn-based models and the other pipelines are unchanged.

```toml
realtime_pipeline = "speech-to-speech"
s2s_model = "moshiko-pytorch-q8"        # or moshika-*, hibiki-*, personaplex-7b-v1 ... (models.md)
```

```bash
bolo components install moshi          # Kyutai's server: PyTorch + CUDA on Linux/Windows, MLX on Apple silicon
bolo pull moshiko-pytorch-q8           # 8.4 GB, CC-BY-4.0, files pinned by SHA-256
bolo serve
```

| Models | Server (component) | Builds | Conversations at once |
| --- | --- | --- | --- |
| Moshi (`moshiko-*`, `moshika-*`: bf16, q8, MLX q4/q8/bf16, candle q8/bf16), Hibiki 1B/2B | Kyutai's `moshi` (MIT; Rust server MIT/Apache-2.0) | `pytorch` (moshi 0.2.13, PyTorch 2.9.1 CUDA 12.8; NVIDIA GPU), `mlx` (moshi_mlx 0.3.0; Apple silicon), `candle` (moshi-backend 0.6.4, `cargo install`; CUDA, Metal or CPU) | PyTorch and MLX: 1 (the server holds one connection; a second one waits, so bolo refuses it with `overloaded`). candle: `duplex_max_sessions` (each one more stream on the GPU; the server ends a connection after 360 s) |
| Hibiki-Zero 3B | Kyutai's `hibiki-zero` (MIT), pinned commit | `pytorch` | 1 |
| PersonaPlex (`personaplex-7b-v1`, `personaplex-rl-seamless`) | NVIDIA's `personaplex` (MIT), pinned commit | `pytorch` (PyTorch 2.4.1 CUDA 12.4) | 1 |
| FLM-Audio (`flm-audio`) | CofeAI's `flm-audio` (Apache-2.0), pinned commit, run from its checkout | `pytorch` (PyTorch 2.6.0 CUDA 12.4) | 1 |
| Freeze-Omni (`freeze-omni`) | Tencent's `freeze-omni` (Freeze-Omni License, non-commercial), pinned commit, run from its checkout; **Socket.IO**, not Kyutai's protocol | `pytorch` (PyTorch 2.2.0 CUDA 12.1, Python 3.10) | `duplex_max_sessions` (its `--max_users` and `--llm_exec_nums`) |
| MiniCPM-o 4.5 (`minicpm-o-4.5`) | `minicpm-o`: an environment with PyTorch 2.8.0 (CUDA 12.8), transformers 4.51.0 and OpenBMB's minicpmo-utils 1.0.6, in which bolo runs its own small server (`bolo/duplex/servers/minicpmo_duplex.py`) around the model's `as_duplex()` API; it speaks Kyutai's protocol. The model's code is **remote code**: it runs only after `bolo models config minicpm-o-4.5 --set allow_remote_code=true`, and bolo checks every pinned file (sha256, no unlisted `.py`) before it starts the server | `pytorch` | 1 |

bolo installs each server in a virtual environment of its own under `<data_dir>/components/<id>/<version>/<build>`
(with `uv` when it is on `PATH`, which fetches Python 3.12; else bolo's own Python, 3.10 to 3.12), starts it on a
loopback port when the first session needs it, gives it the files bolo pulled (`HF_HUB_OFFLINE=1`: the server
downloads nothing), and stops it with bolo. One server process serves every session; a 7 B model takes 8 to 24 GB of
GPU memory, so bolo never starts one process per session. The MLX server keeps the model's state from one connection
to the next, so bolo stops it after each conversation and the next one starts it afresh (a few seconds of loading).
`duplex_server_url` points bolo at a server you run instead (`http://host:8998`, or `https://` for the Rust server's
own certificate). Kyutai's Python servers synchronise CUDA while warming up and PersonaPlex's binds every interface
and asks the Hub for `config.json` at each start; bolo starts the Python servers through a few lines of launcher that
pin the bind address to `127.0.0.1`, answer the Hub from disk and skip the synchronisation on a CPU (`device = "cpu"`,
far from real time for 7 B models). Only Kyutai's servers are touched this way, never their code.

**torch.compile and Triton.** The PyTorch servers compile their kernels with `torch.compile` at the first start,
which needs Triton on the GPU (and a C++ compiler on the CPU). Linux's CUDA and ROCm builds of PyTorch bring Triton;
PyTorch for Windows does not, and Moshi and Hibiki crashed at start there (`TritonMissing`). bolo looks before it
starts a server: without Triton (or without a compiler on the CPU) it starts it with compile off
(`NO_TORCH_COMPILE=1`, `TORCHDYNAMO_DISABLE=1`), and a server whose compile still fails at start is started again
that way. Either way the session gets a `bolo.notice` event ("Moshi speed-up off: Triton is not installed"), the log
and the console's Conversation page say so, and `bolo doctor` lists each server's Triton and speed-up. With compile
off, moshi's loader would turn the q8 checkpoints' scales into bfloat16, which its q8 layers refuse ("Expected
`weight_scb` to have type float"); the launcher keeps them as stored. On Windows `triton-windows` (MIT) can be added:
`bolo components install triton-windows` (or the console's Components page) installs the release that goes with each
environment's torch (3.5 for torch 2.9) into the full-duplex servers and the bolo-torch environments, and bolo always
sets `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` there (inductor's static launcher fails with an `OverflowError` on
Windows). It is optional: on an RTX 3060 laptop the first start took 32 s including the compile, with no real-time
gain over compile off, while Hibiki 1B held real time with compile off (4.6 GB of GPU memory).

**How the session maps onto the Realtime API** (`bolo/duplex/bridge.py`):

| Realtime | Full-duplex model |
| --- | --- |
| `input_audio_buffer.append` / the WebRTC track | every sample goes to the model as it arrives (16 kHz → 24 kHz, Ogg Opus), speech or not; if the client falls more than `duplex_jitter_ms` (200) behind the clock, bolo sends silence so the model keeps running |
| `input_audio_buffer.speech_started` / `speech_stopped`, input items, `input_audio_transcription.*` | from bolo's VAD and STT, as in every pipeline; they do **not** cancel the model's answer |
| `response.created` … `response.done` | one per stretch of the model's speech: it starts at the model's first word or when its audio rises above about -40 dBFS (with 160 ms before), and ends after `duplex_silence_ms` (800) without either |
| `response.output_audio.delta` | the model's audio (24 kHz), in the session's output format; over WebRTC on the outgoing track |
| `response.output_audio_transcript.delta` | the model's text tokens as they come (`response.output_text.delta` with `output_modalities: ["text"]`) |
| `response.cancel`, `output_audio_buffer.clear` | ends the current response as `cancelled` and drops what the model says until its next pause; the model itself keeps running |
| `response.create`, `conversation.item.create` (user text, tool results) | refused (`unsupported`): the model answers speech, by itself |
| `instructions`, `voice` | PersonaPlex: its text prompt and voice prompt (`NATF0` … `VARM4`; OpenAI voice names mean the default `NATF2`); MiniCPM-o 4.5: its system prompt and reference voice (`ht`, `system`, `system-2`); Freeze-Omni: its system prompt; fixed when the conversation opens. The other models take neither |
| the model stops because the caller talked over it (Freeze-Omni's `stop_tts`) | the current response ends as `cancelled` (`turn_detected`) and, over WebRTC, its unplayed audio is dropped |
| `session.duplex` (bolo extension) | sampling options for the Rust server's query string (`text_temperature`, `text_topk`, `audio_temperature`, `audio_topk`, `pad_mult`, `repetition_penalty`, ...); the Python servers ignore them |
| closing the session | closes the model's WebSocket; the server is free for the next conversation |

Use echo cancellation (browsers do by default, or bolo's own: "Echo cancellation" above) or headphones: the model
hears whatever the microphone picks up, its own voice included. Traces are recorded per response (`full_duplex: true`, `started_after_user_ms`).

**Adding a full-duplex model.** A model whose server speaks Kyutai's protocol needs only a catalog entry (task
`speech-to-speech`, engine `duplex`, kind `kyutai`, features `[speech-to-speech, full-duplex]`, options `server`,
`build`, `model_file`, `mimi_file`, `tokenizer_file`, and `config_file`, `voice_param`, `instructions_param` where the
server takes them) and, for a new server, a `Server` in `bolo/duplex/runtime.py` (what to install, pinned, and how to
start it). Anything else implements the interface in `bolo/duplex/base.py` and is returned by
`bolo/duplex/manager.py` for its catalog kind:

```python
class DuplexSession:                  # one live conversation
    input_rate: int                   # the sample rate push() takes (Moshi: 24000)
    frame_samples: int                # the model's step (Moshi: 1920)
    async def push(self, samples): ...           # the caller's audio, any amount, mono float32
    def events(self): ...                        # async iterator: DuplexAudio, DuplexText, ... then one DuplexEnd
    async def close(self): ...                   # end it; events() then ends with DuplexEnd("closed")

class DuplexBackend:                  # one per server
    max_sessions: int
    async def open(self, config: DuplexConfig) -> DuplexSession: ...   # instructions, voice, options
    async def close(self): ...
```

The bridge does the rest (the clock, responses, transcripts, cancellation, WebRTC). Two backends exist:
`bolo/duplex/kyutai.py` (Kyutai's protocol: Moshi, Hibiki, Hibiki-Zero, PersonaPlex, FLM-Audio, and MiniCPM-o 4.5
through bolo's own server, which shows how a model with a Python API but no server gets one) and
`bolo/duplex/freeze_omni.py` (Freeze-Omni's Socket.IO events, with a small Socket.IO client in
`bolo/duplex/socketio.py`); a backend that knows when its model was interrupted yields `DuplexInterrupt`.
`tests/fake_kyutai_server.py` is a stand-in server with the real framing (Kyutai's `sphn` when installed, else
bolo's own Ogg Opus), `tests/fake_freeze_omni_server.py` one built on python-socketio's own server.
What ran: the bridge, both clients and the managed-server path against those fakes, and bolo's Ogg Opus against
Kyutai's `sphn` both ways. **No real Moshi, Hibiki, PersonaPlex, FLM-Audio, Freeze-Omni or MiniCPM-o model was run** (bolo's MiniCPM-o server
was run with a stand-in for the model's duplex object) (the smallest is 4.8 GB and they
want a GPU); [models.md](models.md#full-duplex-models) has a checklist for a first run on a Windows machine with an
NVIDIA GPU.

### Browsers: WebRTC and ephemeral keys

`POST /v1/realtime/calls` is the WebRTC transport of the same sessions
(`pip install "bolo-speech[webrtc]"`, which brings
[aiortc](https://github.com/aiortc/aiortc), BSD-3-Clause; without it the
route answers `503 engine_unavailable` and everything else works). The body
is the browser's SDP offer (`Content-Type: application/sdp`), or a multipart
form with `sdp` and an optional `session` field holding the session
configuration as JSON. The answer is `201` with the SDP answer and a
`Location: /v1/realtime/calls/<call_id>` header;
`POST /v1/realtime/calls/<call_id>/hangup` ends the call. Query parameters
are those of the WebSocket (`model`, `intent=transcription`).

* The caller's microphone is the audio track of the offer (Opus);
  `input_audio_buffer.append` is not used.
* The reply is played on the answer's audio track.
  `response.output_audio.delta` events are **not** sent; instead
  `output_audio_buffer.started` / `.stopped` / `.cleared` say what the track
  is doing. On barge-in bolo also drops the audio it has generated but not
  played yet, so the client has nothing to flush.
* Every other event travels as JSON on the data channel the client creates
  (`oai-events`), in both directions.

```js
// browser: an ephemeral key from your backend, then a call
const pc = new RTCPeerConnection();
pc.ontrack = (e) => { audioElement.srcObject = e.streams[0]; };
pc.addTrack((await navigator.mediaDevices.getUserMedia({ audio: true })).getTracks()[0]);
const events = pc.createDataChannel("oai-events");
events.onmessage = (e) => console.log(JSON.parse(e.data).type);
await pc.setLocalDescription(await pc.createOffer());
const answer = await fetch("https://bolo.example/v1/realtime/calls?model=gpt-realtime", {
  method: "POST", body: pc.localDescription.sdp,
  headers: { "Content-Type": "application/sdp", Authorization: `Bearer ${EPHEMERAL_KEY}` },
});
await pc.setRemoteDescription({ type: "answer", sdp: await answer.text() });
```

**Ephemeral keys.** `POST /v1/realtime/client_secrets` (authenticated with
the server's API key, so call it from your backend) returns
`{"value": "ek_…", "expires_at": …, "session": {…}}`. Body, all optional:
`{"expires_after": {"anchor": "created_at", "seconds": 600}, "session":
{…session configuration…}}`. The key opens Realtime sessions only (WebSocket
or WebRTC) until it expires, the session starts with the configuration it
was minted with, and no other route accepts it. Keys live in the server's
memory: they do not survive a restart and are not shared between replicas.

**Network.** bolo contacts no STUN or TURN server unless
`webrtc_ice_servers` is set (`BOLO_WEBRTC_ICE_SERVERS`, a list of
`stun:` / `turn:` URLs), so out of the box a call connects only when browser
and server can reach each other directly: the same machine, the same
network, or a server with a public address. bolo does not include a TURN
relay. `GET /api/v1/info` reports the transport under
`capabilities.realtime.webrtc`.

Send the offer once the browser has gathered its ICE candidates (wait for
`iceGatheringState === "complete"`, a few hundred milliseconds on a local
network; bolo takes no candidates trickled in later). Chrome hides the
machine's address behind an mDNS name (`….local`) until the page has
microphone permission; bolo cannot always resolve those names, but it learns
the browser's address from the first connectivity check, so the call
connects anyway on the same machine or network.

**In a browser.** The console's Conversation page has a transport switch,
*WebSocket / WebRTC*: with WebRTC the microphone and the agent's voice are
WebRTC audio tracks, the events go over the data channel, and the page shows
the connection, ICE and data-channel states and how much of the agent's
audio has arrived. `examples/webrtc_browser.html` is the same flow as one
self-contained page (no build, no dependencies) to copy into your own
front end.

What was run: the test-suite drives calls with a real aiortc peer (Opus in
both directions, data channel, barge-in, ephemeral keys). With real models
(`moonshine-tiny-en`, a Piper voice, `qwen2.5:0.5b` behind Ollama) three
turns completed over WebRTC and the reply, recorded from the incoming Opus
track and transcribed again, matched the reply text with a word error rate of
0, 0.18 and 0 (`docs/measurements/webrtc-real-models.json`). **Tested in
headless Chromium** (Playwright, `npm run e2e:webrtc` in `ui/`): Chromium's
fake microphone played bolo's `en-short` sample into a real `bolo serve`
(`moonshine-tiny-en`, `piper-en-us-amy-low`, a scripted stand-in language
model), once through the console page and once through the example page; in
both the call connected (0.3 to 1.0 s after the click), the transcript of the
sample came back on the data channel and the answer arrived as sound on the
remote track (`docs/measurements/webrtc-browser.json`). **Real browsers are
still pending**: Chrome and Edge on the Windows laptop are next; Firefox and
Safari are untested, as are a real microphone and speakers, NAT traversal,
TURN and calls across the internet. OpenAI's SIP connector for Realtime
sessions (`sip:...@sip.api.openai.com`) is not implemented; phone calls have
their own path, below.

### Phone calls

The same sessions answer and place phone calls: Twilio Media Streams, Plivo
audio streams, Telnyx media streaming, the Exotel Voicebot applet and a
minimal inbound SIP endpoint each bridge a call into a Realtime session
(transport `telephony` on its traces). On top of what a session does anyway
(turn detection, the pipelines, tools) a call gets named agents, a greeting,
barge-in that clears the audio queued at the provider and truncates the
assistant's message to what was heard, key presses, hang-up and transfer
tools, silence prompts, a longest call length, voicemail handling on
outbound calls and a per-call record with a latency breakdown. See
[telephony.md](telephony.md). If you run your own media gateway instead, the
WebSocket transport with `audio/pcmu` still works as before
(see [indic-telephony.md](indic-telephony.md)).

### What is different from OpenAI, or missing

* A cascade by default; speech-to-speech only through an opt-in pipeline
  (above), full-duplex models included (they answer by themselves: no `response.create`). No emotional prosody control. `noise_reduction` runs bolo's denoiser (GTCRN by default) over each
  user turn before bolo's recogniser hears it; `near_field` and `far_field` are treated alike, the voice-activity
  detector and turn detection hear the original audio, and an audio-in or speech-to-speech model gets the original
  audio too. It can make transcripts worse on clean audio (configuration.md, "Denoising"). `semantic_vad` is an acoustic end-of-turn model added to
  silence detection (above), not OpenAI's.
* Input transcription events are always sent (bolo has to transcribe anyway).
* Not implemented: OpenAI's SIP connector (bolo has its own telephony, [telephony.md](telephony.md)), MCP connectors (`connector_id`; MCP servers by URL work, above), image input, `prompt`
  templates, `reasoning`, `tracing`, `truncation` settings, log probabilities,
  DTMF, idle timeouts, rate-limit events, the realtime translation sessions.
  Unsupported settings in `session.update` are ignored unless they would
  change behaviour silently (connectors and images are rejected with an error).
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

Measured here (three turns each, one at a time, medians, the language model
on the same four CPU cores; files in `docs/measurements/`):

| Language model | Voice | Transcript | First LLM token | First audio | File |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-360M, llama-cpp-python server | `piper-en-us-amy-low` | 339 ms | 494 ms | 2,373 ms | `voice-agent-smollm2.json` |
| SmolLM2-360M, llama-cpp-python server | `kokoro-v1.0-int8` | 253 ms | 361 ms | 3,762 ms | |
| qwen2.5:0.5b, **Ollama** 0.40.16 | `piper-en-us-amy-low` | 167 ms | 408 ms | 2,680 ms (1,426 to 4,396) | `voice-agent-ollama.json` |
| qwen2.5:0.5b, **Ollama** 0.40.16 | `kokoro-v1.0-int8` | 115 ms | 210 ms | 4,926 ms (4,334 to 6,627) | |
| qwen2.5:0.5b, **llama.cpp `llama-server`** | `piper-en-us-amy-low` | 360 ms | 893 ms | 7,456 ms (2,839 to 18,398) | `voice-agent-llama-server.json` |

The Ollama and llama-server rows were taken with two threads each for bolo
and for the model while the machine was busy with unrelated jobs (load
average about 10 on 4 vCPUs); the spread in brackets is that load, and the
llama-server row in particular says nothing about llama.cpp against Ollama.
**Do not plan with these numbers**: a model of half a billion parameters on
two CPU threads, competing with recognition and synthesis for the same cores,
says little about your deployment. What they do show is the shape of a turn: the
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

Pipecat and LiveKit Agents were run against bolo with their stock OpenAI
integrations: Pipecat 1.16.0 and LiveKit Agents 1.8.11 (with
`livekit-plugins-openai` 1.8.11), bolo with `moonshine-tiny-en`, a Piper
voice and `qwen2.5:0.5b` behind Ollama. Neither needed a change to bolo.
The runs had no transport: a wav file played the caller and the reply was
collected (`examples/pipecat_bot.py`, `examples/livekit_agent.py`, results in
`docs/measurements/client-frameworks.json`). A Daily, LiveKit or telephony
transport in front was **not** tried.

| Framework | Integration | Result |
| --- | --- | --- |
| Pipecat | `OpenAIRealtimeLLMService(base_url="ws://…/v1/realtime")` | works: interim and final user transcripts, reply text and audio |
| Pipecat | `OpenAIRealtimeSTTService` (Realtime transcription session) | works: interim and final transcripts |
| Pipecat | `OpenAISTTService` + `OpenAILLMService` + `OpenAITTSService` (HTTP) | works, with two things to know (below) |
| LiveKit | `AgentSession(llm=openai.realtime.RealtimeModel(base_url="http://…/v1"))` | works: transcripts, reply text and audio |
| LiveKit | `openai.STT`, `openai.LLM`, `openai.TTS` with `base_url` (HTTP) | works |
| LiveKit | `openai.STT(use_realtime=True)` | works: interim transcripts while the caller speaks |

**Pipecat**

```python
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

s2s = OpenAIRealtimeLLMService(
    api_key="not-needed",
    base_url="ws://127.0.0.1:8000/v1/realtime",
    settings=OpenAIRealtimeLLMService.Settings(model="gpt-realtime", system_instruction="Answer briefly."),
)
# pipeline: transport.input() -> context_aggregator.user() -> s2s -> transport.output() -> context_aggregator.assistant()
```

Two things to know about Pipecat's HTTP text-to-speech service
(`OpenAITTSService`): it accepts OpenAI's voice names only (`alloy`, `ash`, …;
bolo maps them onto the voices of the configured model), and it gives up on a
sentence when no audio has arrived after three seconds, logging "completed
with no audio" before playing the audio anyway when it comes. With Kokoro on
our busy four cores the first audio of a long sentence took longer than
that; with a Piper voice it did not. For the language model step, point
`OpenAILLMService` at bolo's `/v1` as well (requests without audio are
proxied to your model unchanged) or straight at the model.

**LiveKit Agents**

```python
from livekit.agents import AgentSession
from livekit.plugins import openai, silero

# bolo does the whole turn
session = AgentSession(llm=openai.realtime.RealtimeModel(
    base_url="http://127.0.0.1:8000/v1", api_key="not-needed", model="gpt-realtime", voice="alloy"))

# or LiveKit's turn handling with bolo for speech
session = AgentSession(
    vad=silero.VAD.load(),
    stt=openai.STT(base_url="http://127.0.0.1:8000/v1", api_key="not-needed", model="moonshine-tiny-en", language="en"),
    llm=openai.LLM(base_url="http://127.0.0.1:8000/v1", api_key="not-needed", model="<your model>"),
    tts=openai.TTS(base_url="http://127.0.0.1:8000/v1", api_key="not-needed", model="piper-en-us-amy-low", voice="alloy"),
)
```

If a framework insists on features bolo does not have (image input), use the
separate STT/TTS route.
