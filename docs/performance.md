# Performance

What was measured, on what, and what the numbers do and do not show. Everything here is CPU only: **no GPU was
available, so nothing on this page was measured on one**. The per-model real-time factors of the
catalog are in [models.md](models.md); how they are scaled to your machine is in
[recommendations.md](recommendations.md).

## The machine and the method

All numbers were measured with bolo's own load generator (`bolo bench`, in process) on **4 vCPUs of an Intel Xeon at 2.1 GHz, 15 GB RAM, no GPU, Linux x86-64, Python 3.12, sherpa-onnx 2.0.1, faster-whisper 1.2.1, 4 inference threads**. The machine is a shared cloud VM that was running other workloads at the same time (load average 2 to 4 before bolo started), so expect better on an idle machine of the same size. Raw results: [docs/measurements/](measurements/).

## Single requests

**Single request** (best of three after warm-up; RTF = processing time / audio duration):

| Model | RTF | Meaning |
| --- | --- | --- |
| `moonshine-tiny-en` | 0.033 | 30× faster than real time |
| `parakeet-tdt-ctc-110m` | 0.030 | 33× |
| `sense-voice-small` | 0.039 | 26× |
| `parakeet-tdt-0.6b-v2` | 0.081 | 12× |
| `faster-whisper-tiny` / `whisper-base` / `whisper-small` | 0.09 / 0.11 / 0.36 | 11× / 9× / 2.7× |
| `whisper-large-v3-turbo` (int8) | 0.30 | 3.3× |
| `piper-en-us-amy-low` (TTS) | 0.033 to 0.049, first audio after 140 to 190 ms | 20 to 30× |
| `kitten-nano-0.8-int8` (TTS) | 0.27, first audio after about 1.0 s | 3.7× |
| `kokoro-v1.0-int8` (TTS) | 0.88, first audio after 2.4 to 3.4 s | barely faster than real time |

Kokoro is the default voice for quality, but on four shared cores it is only just real-time; the float32 export (`kokoro-v1.0`) measured faster than the int8 one, also for first audio. Voice agents, Wyoming and `tts-1` therefore use `piper-en-us-amy-low` by default (`default_realtime_tts`), which starts speaking in about 150 ms and needs no optional component (phonemes from uccharan); see [docs/models.md](models.md#default-voices). Two streaming models were slower than real time here and are not usable live on this hardware: `parakeet-unified-en-streaming-560ms` (RTF 4.7) and, nearly, `nemo-fastconformer-ctc-en-streaming-80ms` (0.76).

## Under load

**Concurrent realtime streams** (each client streams at 1×; p50 / p95 in ms):

| Streams | `zipformer-en-20m-streaming` first partial | final | same, batching off: first partial | final | `moonshine-tiny-en` first partial | final | same, batching off: first partial | final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21 / 22 | 32 / 35 | 22 / 24 | 34 / 64 | 66 / 96 | 161 / 196 | 56 / 65 | 150 / 161 |
| 2 | 28 / 31 | 40 / 44 | 25 / 52 | 48 / 109 | 104 / 182 | 314 / 358 | 83 / 134 | 224 / 338 |
| 4 | 37 / 55 | 54 / 114 | 34 / 144 | 115 / 176 | 262 / 329 | 652 / 792 | 81 / 267 | 684 / 795 |
| 8 | 75 / 86 | 172 / 274 | 448 / 525 | 874 / 2,837 | 514 / 621 | 1,338 / 1,977 | 456 / 924 | 1,719 / 2,386 |

(The Zipformer rows used the 12-second sample, the Moonshine rows the 3-second one.)

**Back-to-back requests** (throughput in seconds of audio per second, 12-second clip):

| Clients | `zipformer-en-20m-streaming` batched / unbatched | `moonshine-tiny-en` batched / unbatched | `faster-whisper-tiny` batched / unbatched |
| --- | --- | --- | --- |
| 1 | 9.8 / 10.3 | 26.6 / 30.2 | 14.1 / 14.0 |
| 2 | 15.3 / 12.1 | 28.1 / 28.5 | 14.0 / 13.6 |
| 4 | 21.5 / 12.2 | 28.5 / 27.4 | 12.2 / 14.0 |
| 8 | 25.7 / 11.3 | 28.3 / 27.5 | 13.4 / 13.8 |

**Thirty minutes of mixed load** (moonshine and whisper-base transcription, Piper and Kitten synthesis; 1 to 8 clients; the machine was also running another project's jobs, load average 6 to 13): 1,494 requests, one refused with `429` by design, none failed. Memory reached 1.69 GB once the four models were loaded and ended at 1.80 GB, most of the rise when the client count went to 8; no unbounded growth. Under 8 clients the p95 was 1.3 s for moonshine and 3.3 s for Piper (whole responses). [docs/measurements/soak-30min.json](measurements/soak-30min.json).

**What batching buys on CPU, honestly:**

- For the **streaming transducer** it is a clear win: with 8 live streams, batched decoding kept the median first partial at 75 ms against 448 ms unbatched, and whole-file throughput more than doubled (25.7× against 11.3× real time). A single stream uses little of the CPU per step, so stepping eight streams in one call shares the cores well.
- For **Moonshine and Whisper it buys almost nothing in throughput** (28× against 27×; 13× against 14×). A single decode already saturates four cores, so a batch of eight takes about eight times as long. It still helps fairness under load: at 8 streams Moonshine's p95 final was 1,977 ms batched against 2,386 ms unbatched, at the price of slower partials when only 2 to 4 streams are active.
- Batching is therefore on by default but never waits for a batch to fill (`batch_wait_ms = 0`): an idle server answers a lone request immediately.
- Four cores carry about 4 live Moonshine streams or 8 live Zipformer streams before latency climbs past half a second. Beyond that the queue bounds latency by refusing work (`429`), which is what you want from a server.

## Diarization, voice agents, synthesis under load

**Speaker diarization** (`pyannote-3.0-eres2net`: pyannote segmentation-3.0 + ERes2Net embeddings, ONNX): on a 21-second clip of two synthetic voices (Kokoro `af_heart` and `am_michael`) taking six turns, the diarization error rate was **5.2%**, all of it speech missed at turn boundaries, with no speaker confusion (0% with a 250 ms collar), at RTF 0.11. Labelling live on the WebSocket (one embedding per utterance) got all six turns right and added about 90 ms per final. On four short public recordings the *number* of speakers was right for the three two-speaker clips and one too many (5 for 4) on the four-speaker one; with the default threshold of other embedding models it over-counts more. Pass `num_speakers` when you know it. That is a smoke test, not a benchmark: no overlapping speech, no noise.

**A voice-agent turn** (Realtime API; `moonshine-tiny-en` → SmolLM2-360M-Instruct on llama-cpp-python, 2 threads, on the same 4 cores → `piper-en-us-amy-low`), measured from the moment the end of speech is detected, median of three turns: transcript after 339 ms, first LLM token after 494 ms, **first audio after 2.4 s**. With Kokoro as the voice the first audio came after 3.8 s. Most of the wait is the tiny model writing its first sentence while it shares the CPU with recognition and synthesis; a language model on its own hardware changes these numbers completely, so read them as "the pipeline works", not as what to expect. The answers of a 360M model are not worth quoting. With `qwen2.5:0.5b` behind **Ollama** on the same busy machine (2 threads each): first token after 408 ms and first audio after 2.7 s with Piper (medians of three turns; `docs/measurements/voice-agent-ollama.json`). Over **WebRTC** with the same models a turn completed end to end and the reply heard on the Opus track transcribed back correctly (`webrtc-real-models.json`). None of these numbers describe a deployment with a real language model on its own hardware.

**Text to speech under load** (`piper-en-us-amy-low`, a 200-character paragraph): first audio after 138 ms with one client, 402 ms with two, 826 ms with four; throughput stays near 27× real time. Synthesis is not batched; requests interleave sentence by sentence.


## The realtime tier

The default realtime model is `moonshine-tiny-en` (27M parameters, MIT); `zipformer-en-20m-streaming` is the true-streaming alternative. One live stream, audio sent at real-time pace through the same code path as the WebSocket, on the machine described above:

| Model | How it streams | First partial (p50 / p95) | Final after end of speech is detected (p50 / p95) | Notes |
| --- | --- | --- | --- | --- |
| `zipformer-en-20m-streaming` | native, incremental | 19 / 21 ms | 22 / 30 ms | lower-case, no punctuation; least accurate of the four, and it can drop the first words of audio that starts abruptly |
| `moonshine-tiny-en` | re-decode + LocalAgreement | 62 / 69 ms | 420 / 465 ms on a 12 s utterance; 161 / 196 ms on a 3 s one | punctuated; the default |
| `whisper-tiny` (sherpa-onnx) | re-decode + LocalAgreement | 95 / 111 ms | 918 / 978 ms | 99 languages |
| `faster-whisper-tiny` | re-decode + LocalAgreement | 570 / 640 ms | 860 / 885 ms | 99 languages, word timestamps |

These are `latency_ms` values: server time from receiving the last audio that contributed to the text until the message is sent. They do **not** include the pause the endpointer waits for before it calls an utterance finished, which is 500 ms by default and usually the larger part of what a user perceives. Measured from the true end of speech to the final text with `moonshine-tiny-en` on a 3-second utterance: **355 ms** with `endpointing=fast` (250 ms), **615 ms** with the default (500 ms), **1,212 ms** with `patient` (900 ms).

Every other speech-to-text model also works on the realtime WebSocket; it is just slower. Word error rates on a LibriSpeech test clip and the full table are in [docs/models.md](models.md).

How it works, and how to tune it: [docs/realtime.md](realtime.md).
