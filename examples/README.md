# Examples

Start a server first (`bolo serve`), then run any of these against
`http://127.0.0.1:8000`. Set `BOLO_URL` to point elsewhere and `BOLO_API_KEY`
if the server needs a key.

| File | What it shows | Run by the maintainers? |
| --- | --- | --- |
| `openai_sdk.py` | transcription (plain, verbose, streaming) and speech with the official `openai` Python SDK | yes (the same calls are in the test-suite) |
| `openai_sdk.mjs` | the same with the official `openai` Node SDK | **no** - written against the SDK docs, not executed |
| `curl.sh` | every HTTP endpoint with curl | yes |
| `ws_stt_stream.py` | realtime transcription over the WebSocket, from a file or a microphone | file mode: yes; microphone mode: no (no audio hardware) |
| `ws_tts_stream.py` | streaming synthesis over the WebSocket, text fed incrementally | yes |
| `telephony_mulaw.py` | 8 kHz G.711 μ-law in both directions, RTP-sized 20 ms frames | yes, with synthetic call audio - not against a real phone network |
| `wyoming_client.py` | the Wyoming protocol, as Home Assistant speaks it | yes |
| `home_assistant.md` | wiring bolo into Home Assistant | **no** - no Home Assistant instance was available |
| `realtime_agent.py` | one voice-agent turn over the OpenAI Realtime API (`/v1/realtime`) with the `openai` SDK: audio file in, spoken reply out, latency figures | yes, against a scripted stand-in model; the same exchange was run with small local models behind Ollama and llama.cpp (see [docs/voice-agents.md](../docs/voice-agents.md)) |
| `chat_audio.py` | audio in and audio out through `/v1/chat/completions` | yes, same conditions |
| `pipecat_bot.py` | a Pipecat pipeline with bolo behind it in three ways: its OpenAI Realtime service, its Realtime transcription service, and its HTTP STT/LLM/TTS services. A wav file plays the caller | yes, with Pipecat 1.16.0, real models and a small language model behind Ollama; **not** with a Daily, WebRTC or telephony transport in front |
| `livekit_agent.py` | a LiveKit `AgentSession` with bolo behind it in the same three ways | yes, with LiveKit Agents 1.8.11, same conditions; **not** inside a LiveKit room |

The voice-agent examples need a language model connected to the server
(`bolo serve --llm-base-url … --llm-model …`); how good the conversation is
depends on that model.
