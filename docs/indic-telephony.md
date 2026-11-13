# Indian languages and telephony audio

Two things most self-hosted speech servers treat as afterthoughts are
first-class in bolo: Indian languages, and the 8 kHz G.711 audio that phone
networks carry. This page says exactly what works, what was run, and what
was not.

## Indian languages

### Speech to text

| Route | Models | Languages | Status |
| --- | --- | --- | --- |
| Hindi fine-tunes of Whisper | `faster-whisper-tiny-hindi`, `-small-hindi`, `-medium-hindi` (Collabora, CC-BY-4.0) | hi | `tiny` and `small` validated on CPU with synthesised Hindi (0% WER on one sentence); `small` ran slower than real time here (RTF 1.1), `tiny` at 0.15 |
| Multilingual Whisper with `language=hi` (or bn, ta, te, mr, gu, kn, ml, pa, ur, ne, as, or, sa, sd) | `faster-whisper-small` and larger; `whisper-small` and larger | 99 languages | English path validated; Hindi through `faster-whisper-small` is in the opt-in test-suite. Whisper *tiny* was not usable for Hindi in our run (it answered with English-like text); use *small* or larger, or a Hindi fine-tune |
| Meta Omnilingual ASR | `omnilingual-ctc-300m` | 1,600+ including all major Indian languages | runs on CPU (validated with the German test audio in its repository; RTF 0.11); Indian languages not individually checked. Character-level output, no punctuation |
| Dolphin | `dolphin-base`, `dolphin-small` | 40 Eastern languages incl. hi, ur, bn, ta, te, gu, mr, kn, ml, pa, or | `dolphin-base` runs (validated with its own Mandarin test audio); Indian languages not individually checked |
| Bengali streaming | `zipformer-bn-vosk-streaming` | bn | validated with the Bengali test audio in its repository; true streaming |
| IndicConformer (AI4Bharat) | `indicconformer-multi-community` | hi, as, bn, brx, gu, kn, ks, mr | community export of a gated original; validated on CPU with synthesised Hindi (0% WER on one sentence, RTF 0.13). Other languages not checked |

Tell the server the language. Multilingual models detect it otherwise, which
costs time and, on short telephone utterances, accuracy:

```bash
curl localhost:8000/v1/audio/transcriptions -F file=@call.wav -F model=faster-whisper-small -F language=hi
```

```
ws://localhost:8000/api/v1/stt/stream?model=faster-whisper-tiny-hindi&language=hi
```

Every model above works on the realtime WebSocket. Only the Bengali
Zipformer is natively streaming; the others use pseudo-streaming (VAD,
periodic re-decoding, stabilised partials), whose latency is the model's
decode time for the utterance so far. See [realtime.md](realtime.md).

**Code-mixed speech** (Hinglish and friends). Nothing in bolo fights it: text
is passed through as the model emits it, in whatever script. How well it is
recognised depends on the model; Whisper fine-tunes on code-mixed data can be
added with `bolo pull hf:<repo>` if they are in CTranslate2 format. We did not
measure code-mixed accuracy.

**The official AI4Bharat models** are gated on Hugging Face and use an ONNX
layout bolo cannot load yet. bolo never downloads gated models on its own; a
loader that uses your token is on the roadmap. Details in
[models.md](models.md#indian-languages).

### Text to speech

| Language | Models | Needs espeak-ng | Status |
| --- | --- | --- | --- |
| Hindi | `piper-hi-in-rohan-medium`, `-pratham-medium` ⚠, `-priyamvada-medium` ⚠ | yes | all three validated on CPU; `rohan` round-trips through Hindi speech recognition in the test-suite |
| Hindi | `kokoro-v1.0-int8` / `kokoro-v1.0`, voices `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` | yes | `hf_alpha` validated (the built-in Hindi sample was made with it and is recognised with 0% WER by `faster-whisper-tiny-hindi`) |
| Nepali | `piper-ne-np-google-medium`, `-x-low`, `mimic3-ne-google` | yes | `medium` and mimic3 validated (audio produced; not transcribed back) |
| Urdu | `piper-ur-pk-fasih-medium` | yes | validated (audio produced; not transcribed back) |
| Bengali | `coqui-bn-female`, `coqui-bn-male` (character input), `mimic3-bn-multi` | coqui: no; mimic3: yes | `coqui-bn-female` and `mimic3-bn-multi` validated (audio produced) |
| Gujarati | `mimic3-gu-cmu-indic` | yes | validated (audio produced) |
| Telugu, Malayalam | Piper voices exist upstream (`te_IN`, `ml_IN`) | yes | **not available yet**: no converted files; on the roadmap |
| Tamil, Marathi, Kannada, Punjabi, Odia, Assamese | - | - | **no voice in the catalog.** Meta's MMS-TTS covers them but is non-commercial and has no ready-made export |

⚠ = the voice's training data is under a non-commercial licence.

"Audio produced; not transcribed back" means exactly that: the voice loaded,
synthesised a sentence of plausible length and loudness, and nobody on the
team who speaks the language listened to it. Treat those as smoke-tested.

Text handling for Indic scripts:

* sentences are split at the danda (`।`, `॥`) and the Urdu full stop (`۔`,
  `؟`) as well as Latin punctuation, so streaming synthesis starts after the
  first sentence in any of these scripts;
* Hindi numbers are spelled out in the Indian system before synthesis:
  `250000` → `दो लाख पचास हज़ार`, Devanagari digits included. Other Indian
  languages keep their digits (the voice's own front end reads them);
* code-mixed input such as `मेरा order कल deliver होगा` is passed to the voice
  unchanged. How a Hindi voice pronounces Latin-script words depends on the
  voice; we have not evaluated it.

## Telephony

Phone audio is 8 kHz, companded with G.711: μ-law (PCMU) in North America
and Japan, A-law (PCMA) elsewhere, including India. bolo speaks both natively,
in both directions, with no ffmpeg and no extra packages.

### Inbound: caller audio → text

* **Realtime**: `WS /api/v1/stt/stream?encoding=mulaw` (or `alaw`). The sample
  rate defaults to 8000 for these encodings. Send the RTP payloads as binary
  frames as they arrive, typically 160 bytes every 20 ms. The server decodes
  G.711 and resamples to the model's 16 kHz with a stateful resampler, so
  frame boundaries are inaudible.
* **Recordings**: upload raw μ-law/A-law with the right content type, or any
  8 kHz WAV:

  ```bash
  curl localhost:8000/v1/audio/transcriptions -F "file=@call.ulaw;type=audio/basic" -F model=moonshine-tiny-en
  curl localhost:8000/v1/audio/transcriptions -F "file=@call.al;type=audio/x-alaw" -F model=faster-whisper-small -F language=hi
  ```

### Outbound: text → caller audio

* `POST /v1/audio/speech` with `"response_format": "mulaw"` or `"alaw"`
  returns raw 8 kHz G.711, streamed sentence by sentence.
* `WS /api/v1/tts/stream?format=mulaw` does the same over a WebSocket, with
  text fed incrementally (from a language model, say).

Either way you get bytes you can cut into 160-byte RTP payloads.
`bolo.audio.codecs` has the helpers if your gateway is in Python:

```python
from bolo.audio.codecs import Framer, MULAW_SILENCE, frame_bytes, mulaw_decode, mulaw_encode

framer = Framer(frame_bytes())                 # 160 bytes = 20 ms at 8 kHz
for payload in framer.push(chunk_from_bolo):   # exact 20 ms frames, remainder kept for the next chunk
    send_rtp(payload)
last = framer.flush(pad=MULAW_SILENCE)         # pad the tail with digital silence
```

### What was tested

* The codecs are checked bit-for-bit against the reference implementation
  (the C routines behind Python's former `audioop` module) over all 65,536
  input values, in both directions, on Python 3.12.
* The whole path runs in the test-suite with real models: text → Kokoro →
  8 kHz μ-law → 160-byte frames at real-time pace → realtime WebSocket →
  Moonshine → text, asserting a word error rate under 30% (measured: 0%).
  A second test filters the speech to the 300 to 3,400 Hz telephone band
  first (measured: 17% with Moonshine, 11% with the streaming Zipformer; in
  both cases the first words were clipped).
* `examples/telephony_mulaw.py` is that loop as a script.

### What was not tested

* **No real phone network.** No SIP trunk, no carrier, no handset. Real calls
  add narrow-band microphones, packet loss, jitter, echo and background
  noise; accuracy on them will be lower than on synthesised speech passed
  through G.711, and we have no numbers for it.
* bolo is not a SIP or RTP endpoint. It does not parse RTP headers, reorder
  packets, conceal losses or detect DTMF. Put a media gateway in front
  (Asterisk, FreeSWITCH, LiveKit SIP, or a provider's media-stream webhook)
  and forward payloads.
* Models are trained on 16 kHz audio. Upsampled 8 kHz audio has no content
  above 4 kHz, which costs some accuracy with every model. No model in the
  catalog is trained specifically for 8 kHz telephone speech; sherpa-onnx has
  loaders for such models (`telespeech_ctc`, `online_t_one_ctc`) that you can
  use with `bolo pull hf:… --kind …`, untested here.
* G.722, G.729, AMR and Opus-in-RTP are not handled by the WebSocket (Opus
  and AMR files can be uploaded if PyAV or ffmpeg is installed).
