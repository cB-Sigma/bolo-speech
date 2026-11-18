# Realtime speech-to-text

`WS /api/v1/stt/stream` gives you interim and final transcripts for a live
audio stream with **any** speech-to-text model in the catalog. This page
explains how, what latency to expect, and which knobs matter. The wire
format is in [api.md](api.md#ws-apiv1sttstream).

Per-request Whisper options (`repetition_penalty`, `condition_on_previous_text`, `max_repeats`, ...) apply to file
transcription; for streaming, pin them on the model with `PUT /api/v1/models/{id}/settings` (api.md, "Decoding options
and repetition loops"). The repetition guard checks each final against the finals sent before it and never rewrites
one that was sent; partials are not trimmed.

## Two ways to be realtime

**Native streaming** (`"mode": "native"` in the `ready` message). Models
marked `streaming: true` (streaming Zipformer, streaming FastConformer,
streaming Paraformer) have encoders that consume audio a few frames at a time.
Each chunk you send advances the recogniser a little; the text only grows.
Cost per second of audio is constant, so latency does not depend on how long
someone has been talking. When several clients are connected, their decode
steps are advanced together in one batched call.

**Pseudo-streaming** (`"mode": "pseudo"`). Whisper, Moonshine, Parakeet,
SenseVoice and friends decode whole utterances. bolo makes them usable live:

1. a voice-activity detector (Silero) marks where speech starts and ends;
2. while someone is speaking, the utterance so far is re-decoded every
   `partial_interval_ms` (350 ms; less often if a decode takes longer than that);
3. a *LocalAgreement* policy turns the successive hypotheses into stable
   text: a word is committed once two consecutive decodes agree on it, and a
   committed word is never taken back within the utterance. A `partial`
   carries the committed text plus the newest unconfirmed tail, and
   `stable_chars` says where one ends and the other begins;
4. when the detector hears the end of speech, the whole utterance is decoded
   once more and sent as the `final`.

The price is that the cost of a partial grows with the length of the
utterance (the audio is decoded again from its start), so utterances are cut
at `max_utterance_s` (20 s). Small, fast models make this cheap, which is
what the *realtime tier* is for.

## Which model

| You want | Use | Why |
| --- | --- | --- |
| the lowest latency and many concurrent streams, English | `zipformer-en-20m-streaming` | native streaming, 20M parameters; plain lower-case text without punctuation, and noticeably less accurate than the others |
| punctuated, accurate English with sub-second finals | `moonshine-tiny-en` (the default, `model=realtime`) or `moonshine-base-en` | very fast whole-utterance decoding, so pseudo-streaming is cheap |
| many languages | `faster-whisper-tiny` … `-small` with `language=` set | Whisper's fixed 30-second window makes every decode cost about the same, however short the utterance |
| Hindi | `faster-whisper-tiny-hindi` | a Whisper tiny fine-tune |
| Bengali, Russian, Korean, Mandarin, … with true streaming | the `zipformer-*-streaming` models | native streaming |

The default (`default_realtime_stt`) is `moonshine-tiny-en`: on the machine
below it was the best balance of latency and accuracy. Change it with
`--default-realtime-stt`.

## Measured latency

<!-- numbers: docs/measurements/*.json -->
Measured on 4 vCPUs of a shared Intel Xeon at 2.1 GHz (no GPU), one live
stream at real-time pace, values are p50 / p95 of `latency_ms`:

| Model | Mode | First partial | Final | Sample |
| --- | --- | --- | --- | --- |
| `zipformer-en-20m-streaming` | native | 19 / 21 ms | 22 / 30 ms | 12 s, several sentences |
| `moonshine-tiny-en` | pseudo | 62 / 69 ms | 420 / 465 ms | 12 s, several sentences |
| `moonshine-tiny-en` | pseudo | 66 / 96 ms | 161 / 196 ms | 3 s question |
| `whisper-tiny` | pseudo | 95 / 111 ms | 918 / 978 ms | 12 s |
| `faster-whisper-tiny` | pseudo | 570 / 640 ms | 860 / 885 ms | 12 s |

The pattern to take away: a native streaming model answers in tens of
milliseconds regardless of utterance length; Moonshine's final costs about
35 ms per second of speech; Whisper pays for its 30-second window every time.

With more streams (p50 first partial / p50 final, ms):

| Streams | `zipformer-en-20m-streaming` | `moonshine-tiny-en` |
| --- | --- | --- |
| 1 | 21 / 32 | 66 / 161 |
| 2 | 28 / 40 | 104 / 314 |
| 4 | 37 / 54 | 262 / 652 |
| 8 | 75 / 172 | 514 / 1,338 |

The raw results, including the runs with batching switched off, are in
[`docs/measurements/`](measurements/) and discussed in the README. In the
real-speech test (a LibriSpeech clip streamed over the WebSocket) the
Zipformer produced 16 partials with a median latency of 22 ms and a word
error rate of 5.6%; Moonshine produced 16 partials with a median of 160 ms
and no errors, and its stable prefix never retracted.

Caveat for `zipformer-en-20m-streaming`: it is a 20M-parameter model trained
on read speech. It sometimes drops the first words of audio that starts
abruptly, and it returned nothing at all for one of our synthetic 3-second
samples that the other models transcribed perfectly.

## Endpointing: the pause that costs the most

After someone stops talking, three things happen before your application
has the final text:

```
 speech ends ──► endpointing silence ──► final decode ──► network
                 (500 ms by default)     (latency_ms)
```

`latency_ms` on the `final` only covers the decode. On a fast model the
endpointing silence is most of the wait: with the default 500 ms and a
decode of about 150 ms, roughly three quarters of the time from the end of
speech to the final text is the server waiting to be sure the speaker has
finished. That wait is a choice, and it is yours to make:

| Setting | Trailing silence | Use it when |
| --- | --- | --- |
| `endpointing=fast` | 250 ms | turn-taking voice agents, short commands, IVR prompts |
| `endpointing=balanced` (default) | 500 ms | general dictation and conversation |
| `endpointing=patient` | 900 ms | speakers who pause to think, dictating long sentences |
| `endpoint_ms=<n>` | exact (100 to 5000) | you measured your own users |

Server-wide default: `endpoint_silence_ms`. The trade-off is direct:
a shorter threshold returns finals sooner and splits one sentence into
several finals when the speaker pauses mid-sentence (each final is decoded
without the words before the pause, which also costs a little accuracy).

Measured with `moonshine-tiny-en`, from the true end of speech in the audio
to the `final` leaving the server (median of four runs):

| | 3 s utterance | 12 s utterance |
| --- | --- | --- |
| `fast` (250 ms) | 355 ms | 586 ms |
| `balanced` (500 ms) | 615 ms | 1,074 ms |
| `patient` (900 ms) | 1,212 ms | 1,586 ms |

Of the 615 ms in the default case, the decode is about 140 ms: the rest is
the endpointing wait. Neither test clip contains a pause long enough to be
split at 250 ms, so these runs show the gain and not the cost; the unit
tests show the cost (two bursts 400 ms apart become two finals under `fast`
and one under an 800 ms threshold). Raw data:
[`measurements/endpointing-moonshine-tiny-en.json`](measurements/endpointing-moonshine-tiny-en.json).

For native streaming models the detector's verdict and the model's own
endpoint rule both apply, whichever comes first; with `vad=false` only the
model's rule is used and the per-stream setting has no effect.

## Tuning

| Setting | Default | Effect |
| --- | --- | --- |
| `partial_interval_ms` | 350 | pseudo-streaming: how often to re-decode. Lower gives livelier partials and more CPU per stream |
| `agreement` | 2 | hypotheses that must agree before text is shown as stable. 1 disables stabilisation (everything is "stable" at once and may flicker) |
| `endpoint_silence_ms` | 500 | see above |
| `max_utterance_s` | 20 | forced cut for people who never pause |
| `threads`, `workers` | 4, 1 | one worker using all cores keeps single-stream latency lowest |
| `max_batch` | 8 | 1 turns batching off (see the comparison in the README) |
| query `interim=false` | | finals only: no re-decoding at all, the cheapest mode |
| query `vad=false` | | you decide where utterances end, with `{"type":"flush"}` |

## The same thing through the OpenAI Realtime API

`WS /v1/realtime` (see [voice-agents.md](voice-agents.md)) uses the same
machinery, with OpenAI's event names: while the user speaks,
`conversation.item.input_audio_transcription.delta` events carry the words
that have settled (two consecutive decodes agree), and after the end of
speech the remainder and a `completed` event with the whole transcript.
Deltas only ever append. Per-request Whisper options apply to file transcription only; for streaming, pin them on the
model with `PUT /api/v1/models/{id}/settings` (`repetition_penalty`, `condition_on_previous_text`, `max_repeats`
...; api.md, "Decoding options and repetition loops"). The repetition guard runs across utterances here too: a
transcript that only repeats the ones before it comes back empty, and deltas are never trimmed. Measured with a 12-second sample
(`docs/measurements/realtime-transcript-deltas.json`):

| Model | Deltas | Of them before the end of speech | First delta after speech start |
| --- | --- | --- | --- |
| `moonshine-tiny-en` | 9 | 8 | 1.6 s |
| `faster-whisper-tiny` | 12 | 11 | 2.1 s |
| `zipformer-en-20m-streaming` | 10 | 9 | 2.4 s |

The first delta is late by design: a word has to survive a second decode
before it is sent. With the streaming Zipformer the deltas added up exactly
to the final transcript; with Moonshine and Whisper the final decode revised
a word or a punctuation mark that had already been sent, so treat
`completed.transcript` as the text and the deltas as a preview. The native
WebSocket (`/api/v1/stt/stream`) shows unstable text as well and is the
better choice for a live caption display.

`turn_detection: {"type": "semantic_vad"}` adds an optional end-of-turn model
to silence detection in Realtime sessions (what it does and what we measured:
[voice-agents.md](voice-agents.md)). The native WebSocket does not use it.

## Limits, stated plainly

* LocalAgreement is the simple, robust policy from the Whisper-Streaming
  work. Attention-guided policies (AlignAtt, as in SimulStreaming) commit
  text sooner and are on the roadmap, not in this release.
* Partials of pseudo-streaming models lag by one decode plus up to one
  `partial_interval_ms`, and the stable part lags one further decode.
* Whisper models pad every decode to 30 seconds, so their partials are much
  more expensive than Moonshine's. On CPU, do not expect lively partials from
  anything bigger than Whisper tiny or base.
* The streaming models are primed with a second of faint noise when a stream
  opens, because several of them drop the first words of audio that starts
  abruptly. This costs a few tens of milliseconds once per stream.
* If the neural VAD stays silent on speech it was not trained for, an energy
  detector takes over after 1.5 seconds so the stream still produces a
  transcript; the first partials of that utterance are late.
* Echo cancellation needs the audio that was played: it is part of Realtime
  sessions (`audio.input.echo_cancellation`, voice-agents.md, "Echo
  cancellation"), not of this transcription stream. Noise suppression
  (`denoise`) applies here too: see api.md, "Denoising a live stream".
* All latency numbers are server-side. The browser, the network and your
  audio capture buffer come on top.
