# Indian languages and telephony audio

Two things most self-hosted speech servers treat as afterthoughts are
first-class in bolo: Indian languages, and the 8 kHz G.711 audio that phone
networks carry. This page says exactly what works, what was run, and what
was not.

## Indian languages

At a glance (the sections below have the models and what was measured for each). Licence
classes: **P** permissive (MIT, Apache-2.0, CC-BY, CC0), **NC** non-commercial, **G** gated, **T** needs the PyTorch plugin ([bolo-torch](../plugins/bolo-torch/README.md)).

| Language | STT | TTS |
| --- | --- | --- |
| Hindi | P, run here (also G/T: SraVaani, official IndicConformer) | P (Piper, Kokoro, SYSPIN; some Piper voices NC); NC MMS; G/T Indic Parler-TTS, IndicF5 |
| Bengali, Telugu, Tamil, Malayalam, Marathi, Urdu, Nepali, Gujarati | P, run here | P (some Piper voices NC); NC MMS for most |
| Kannada, Maithili | P, run here | P (SYSPIN); NC MMS |
| Magahi, Chhattisgarhi | Hindi model only | P (SYSPIN); NC MMS (run here) |
| Bhojpuri (not a scheduled language) | Hindi model (41 % WER) | P (SYSPIN) |
| Punjabi, Odia, Assamese | P, run here | NC MMS (run here); G/T Indic Parler-TTS |
| Dogri | P, untested | NC MMS (run here) |
| Sanskrit, Sindhi | P, run here (Sindhi without an error rate) | G/T only (Indic Parler-TTS) |
| Bodo, Kashmiri, Konkani, Manipuri, Santali | P, untested | none (Bodo, Kashmiri: G/T or T only) |


### Speech to text: all 22 scheduled languages

Every scheduled language of India has a model of its own: AI4Bharat's
IndicConformer (Conformer-large, CTC branch, MIT), in an ungated community
ONNX export for sherpa-onnx (`parismitaglobalsolutions/indicconformer-sherpa-onnx`,
pinned to a commit; 198 MB per language, int8). The official AI4Bharat
repositories are gated on Hugging Face; bolo never downloads gated models on
its own.

Each model below was pulled and run here on recordings with known
transcripts. WER and CER are after removing punctuation; small samples (10
utterances, about two minutes, unless stated), so read them as indications.
The real-time factors were taken with 2 threads on a machine busy with other
jobs and are pessimistic.

| Language | Model | Test recordings | WER | CER | RTF |
| --- | --- | --- | --- | --- | --- |
| Assamese (`as`) | `indicconformer-as-community` | FLEURS as_in dev, first 10 utterances | 30.8 % | 14.9 % | 0.15 |
| Bengali (`bn`) | `indicconformer-bn-community` | FLEURS bn_in dev, first 10 utterances | 10.8 % | 2.5 % | 0.11 |
| Bodo (`brx`) | `indicconformer-brx-community` | **not run**: no public test recording found | - | - | - |
| Dogri (`doi`) | `indicconformer-doi-community` | **not run**: no public test recording found | - | - | - |
| Gujarati (`gu`) | `indicconformer-gu-community` | FLEURS gu_in dev, first 10 utterances | 21.7 % | 5.4 % | 0.11 |
| Hindi (`hi`) | `indicconformer-hi-community` | FLEURS hi_in dev, first 10 utterances | 15.2 % | 7.0 % | 0.07 |
| Kannada (`kn`) | `indicconformer-kn-community` | FLEURS kn_in dev, first 9 utterances | 15.1 % | 4.8 % | 0.07 |
| Kashmiri (`ks`) | `indicconformer-ks-community` | **not run**: no public test recording found | - | - | - |
| Konkani (`kok`) | `indicconformer-kok-community` | **not run**: no public test recording found | - | - | - |
| Maithili (`mai`) | `indicconformer-mai-community` | RESPIN Maithili test, 60 utterances | 49.6 % | 11.6 % | 0.06 |
| Malayalam (`ml`) | `indicconformer-ml-community` | FLEURS ml_in dev, first 9 utterances | 31.4 % | 8.8 % | 0.12 |
| Manipuri (`mni`) | `indicconformer-mni-community` | **not run**: no public test recording found | - | - | - |
| Marathi (`mr`) | `indicconformer-mr-community` | FLEURS mr_in dev, first 10 utterances | 19.3 % | 4.2 % | 0.20 |
| Nepali (`ne`) | `indicconformer-ne-community` | FLEURS ne_np dev, first 10 utterances | 30.7 % | 12.7 % | 0.14 |
| Odia (`or`) | `indicconformer-or-community` | FLEURS or_in dev, first 10 utterances | 15.5 % | 4.8 % | 0.16 |
| Punjabi (`pa`) | `indicconformer-pa-community` | FLEURS pa_in dev, first 10 utterances | 15.5 % | 5.3 % | 0.14 |
| Sanskrit (`sa`) | `indicconformer-sa-community` | prathoshap/sushrota-sanskrit-asr-data train, first 12 utterances | 44.2 % | 6.1 % | 0.13 |
| Santali (`sat`) | `indicconformer-sat-community` | **not run**: no public test recording found | - | - | - |
| Sindhi (`sd`) | `indicconformer-sd-community` | FLEURS sd_in dev, first 10 utterances: ran, but the model writes Devanagari and the references are in Arabic script | - | - | 0.11 |
| Tamil (`ta`) | `indicconformer-ta-community` | FLEURS ta_in dev, first 10 utterances | 25.5 % | 10.2 % | 0.06 |
| Telugu (`te`) | `indicconformer-te-community` | FLEURS te_in dev, first 10 utterances | 21.1 % | 10.4 % | 0.07 |
| Urdu (`ur`) | `indicconformer-ur-community` | FLEURS ur_pk dev, first 10 utterances | 18.5 % | 7.5 % | 0.14 |

Things to know:

* **Say which language.** Each model knows one language; pick the model by
  language (`model=indicconformer-mai-community`). Nothing here detects
  Maithili, Bhojpuri or the other Hindi-belt languages reliably, and
  Whisper's language detector does not know most of them.
* Output is in the script the model was trained on: Kashmiri and Urdu in
  Perso-Arabic (right to left), **Sindhi in Devanagari**, Manipuri in Meetei
  Mayek, Santali in Ol Chiki. No punctuation, no capitalisation.
* Bodo, Dogri, Kashmiri, Konkani, Manipuri and Santali are in the catalog as
  **untested**: we found no openly downloadable recordings with transcripts
  to run them on.
* WER overstates the errors where spelling conventions differ. The Maithili
  test set writes postpositions as separate words (`जाड़ में`), the model
  attaches them (`जाड़मे`); the character error rate is the fairer number
  there. Sanskrit was run on verse recitation.
* Smaller exports of the same Tamil, Telugu and Malayalam models, with a
  single-language vocabulary (138 MB): `indicconformer-ta`
  (run on 10 FLEURS utterances: WER 29.0 %, CER 10.3 %, RTF 0.23), `indicconformer-te` (run on 10 FLEURS utterances: WER 21.7 %, CER 10.5 %, RTF 0.26),
  `indicconformer-ml` (run on 9 FLEURS utterances: WER 34.3 %, CER 9.8 %, RTF 0.17).
* All of these work on the realtime WebSocket and in Realtime sessions
  through pseudo-streaming (see [realtime.md](realtime.md)).

```bash
bolo pull indicconformer-mai-community
curl localhost:8000/v1/audio/transcriptions -F file=@call.wav -F model=indicconformer-mai-community
```

### Maithili and Bhojpuri

**Maithili** has a dedicated recogniser, `indicconformer-mai-community`:
11.6 % CER and 49.6 % WER on 60 utterances of the RESPIN test set
(IISc, CC-BY-4.0: read sentences about farming and banking by rural
speakers). Example: reference `गामक चरवाहा जाड़ में बहुत धन प्राप्त कयलक`,
recognised `गामके चरवाहा जाड़मे बहुत धन प्राप्त कयलक`.

**Bhojpuri** has **no dedicated speech-to-text model in the catalog that does
better than the Hindi one.** The practical choice is the Hindi model,
`indicconformer-hi-community`, which we measured on 60 RESPIN Bhojpuri
utterances:

| Model | Audio | WER | CER |
| --- | --- | --- | --- |
| `indicconformer-hi-community` | Hindi (FLEURS, 10 utterances) | 15.2 % | 7.0 % |
| `indicconformer-hi-community` | Hindi (RESPIN, 60 utterances) | 11.5 % | 2.8 % |
| `indicconformer-hi-community` | **Bhojpuri** (RESPIN, 60 utterances) | **40.2 %** | **13.8 %** |
| `indicconformer-hi-community` | Maithili (RESPIN, 60 utterances) | 51.1 % | 18.7 % |
| `indicconformer-mai-community` | Maithili (RESPIN, 60 utterances) | 49.6 % | 11.6 % |

So on Bhojpuri the Hindi model gets about six words in ten right, three to
four times its error on Hindi, and it writes Bhojpuri with Hindi habits
(reference `बायोगैस जीव जंतु के अपघटन से बनत बाटे`, recognised `बायोोगैस जीव जंतु के अपघटन से बनत बांटे`).
That is usable as input for a language model that tolerates noisy text, and
not good enough for transcripts.

We converted the one permissively licensed Bhojpuri recogniser that runs
without PyTorch after conversion, the Whisper-small fine-tune
`Aditya109/whisper-small-bhojpuri-merged` (CC-BY-4.0, trained on IISc's Vaani
data), to CTranslate2 int8 (`tools/voices/convert_whisper.py`) and compared it
with the Hindi model on a second, larger sample: 120 RESPIN Bhojpuri test
utterances (12 at each of 10 offsets; `docs/measurements/bhojpuri-stt.json`):

| Model | WER | CER | RTF (2 threads) |
| --- | --- | --- | --- |
| `indicconformer-hi-community` (Hindi) | **41.2 %** | **12.7 %** | 0.06 |
| Whisper-small Bhojpuri fine-tune (converted here) | 47.7 % | 22.2 % | 0.53 |
| `indicconformer-mai-community` (Maithili) | 66.8 % | 18.2 % | 0.08 |

The fine-tune is worse on this read speech (its card reports 36.4 % WER on
Vaani's own test set, which is spontaneous speech) and nine times slower, so
it is **not** added and the Hindi model stays the Bhojpuri default. IISc's
SraVaani (MIT, gated, TorchScript) reports 20.8 % WER on RESPIN Bhojpuri; it
is in the catalog for the PyTorch engine plugin
([plugins.md](plugins.md)), which does not implement it yet. No permissively licensed Maithili or
Magahi Whisper fine-tune exists that we could find (the one Maithili
fine-tune on Hugging Face states no licence). For Maithili do not use
the Hindi model: it does worse than the Maithili one and rewrites Maithili
grammar into Hindi.

Voices for both languages are published (IISc SYSPIN, below).

### Other speech-to-text models for Indian languages

| Model | Languages | What it is | Status |
| --- | --- | --- | --- |
| `faster-whisper-tiny-hindi`, `-small-hindi`, `-medium-hindi` | hi | Collabora's Hindi fine-tunes of Whisper (CC-BY-4.0) | `tiny` and `small` validated with synthesised Hindi (0 % WER on one sentence); `small` ran slower than real time here (RTF 1.1), `tiny` at 0.15 |
| `faster-whisper-large-v2-hindi` | hi | Collabora, Whisper large-v2 (CC-BY-4.0, 3.1 GB) | **untested** |
| `faster-whisper-base-hinglish` | Hindi-English | Oriserve's Hindi2Hinglish "Swift" (Whisper base, Apache-2.0, 81 MB): writes code-mixed speech in **Latin script** | ran on 10 Hindi FLEURS utterances (RTF 0.17) and produced romanised Hindi such as `Major kin khaana bhoomadhy saagar mein samaan kshetron ki ta`; **no accuracy number**: we had no Hinglish recordings with Latin-script references |
| `faster-whisper-medium-tamil`, `-kannada` | ta, kn | vasista22's Whisper-medium fine-tunes, CTranslate2 builds by a third party (Apache-2.0, 780 MB) | Tamil: run on 3 FLEURS utterances: WER 6.9 %, CER 2.6 %, RTF 1.76. Kannada: run on 3 FLEURS utterances: WER 25.5 %, CER 24.5 %, RTF 1.84 |
| `faster-whisper-medium-telugu`, `-bengali` | te, bn | Whisper-medium fine-tunes (Apache-2.0, 1.5 GB) | Telugu: run on 3 FLEURS utterances: WER 4.3 %, CER 7.0 %, RTF 1.52. Bengali: run on 3 FLEURS utterances: WER 13.2 %, CER 4.1 %, RTF 0.69 |
| `faster-whisper-medium-punjabi` | pa | AI4Bharat IndicWhisper (MIT, 770 MB) | run on 3 FLEURS utterances: WER 48.4 %, CER 38.2 %, RTF 1.21 (3 utterances only; the first came out nearly exact, so most of the error is in the other two, which were not inspected) |
| `nemotron-3.5-streaming-560ms` | hi and 34 non-Indian languages | NVIDIA Nemotron 3.5 ASR (0.6 B parameters, OpenMDW-1.1): the only **natively streaming** Hindi model here, with punctuation. `language=hi` fixes the language | run on 3 FLEURS utterances: WER 20.9 %, CER 6.1 %, RTF 0.31; whole-utterance decoding, streaming latency not measured |
| multilingual Whisper with `language=` | hi, bn, ta, te, mr, gu, kn, ml, pa, ur, ne, as, or, sa, sd | `faster-whisper-small` and larger | English path validated; Whisper *tiny* was not usable for Hindi in our run. Prefer the per-language models above |
| `zipformer-bn-vosk-streaming` | bn | natively streaming Zipformer | validated with the Bengali test audio in its repository |
| `omnilingual-ctc-300m`, `dolphin-base` | many | broad multilingual models | run on CPU with non-Indian test audio only. The Omnilingual CTC model takes no language hint at all, so prefer a per-language model |

The Whisper-medium models are accurate and slow: expect slower than real time
on a few CPU cores. For live use on CPU the IndicConformer models are the
ones to try first.

**Code-mixed speech** (Hinglish and friends). IndicConformer writes English
words in the native script; `faster-whisper-base-hinglish` writes everything
in Latin script. We did not measure code-mixed accuracy for either.

### Text to speech

| Language | Voices | Needs espeak-ng | Status |
| --- | --- | --- | --- |
| Hindi | `piper-hi-in-rohan-medium`, `-pratham-medium` ⚠, `-priyamvada-medium` ⚠; Kokoro voices `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`; `syspin-hi-female`, `-male` | **no** (uccharan; espeak-ng optional) | validated on CPU. On 500 held-out sentences transcribed back by `indicconformer-hi-community`: `rohan` 8.8 % WER with uccharan, 9.5 % with espeak-ng (difference within noise); Kokoro `hf_alpha` (60 sentences) 5.4 % and 4.5 % ([models.md](models.md#phonemisers)) |
| Telugu | `piper-te-in-venkatesh-medium`, `piper-te-in-padmavathi-medium`, `piper-te-in-maya-medium` | yes | **new**, each synthesised a sentence that was transcribed back by the Telugu recogniser: round-trip CER 3.8 % (WER 23.1 %), RTF 0.21; round-trip CER 2.6 % (WER 15.4 %), RTF 0.18; round-trip CER 1.3 % (WER 7.7 %), RTF 0.10 |
| Tamil | `piper-ta-in-rasa-female-medium`, `piper-ta-in-rasa-male-medium` (third-party voices, CC-BY-4.0) | yes | **new**: round-trip CER 0.0 % (WER 0.0 %), RTF 0.11; round-trip CER 3.1 % (WER 10.0 %), RTF 0.15 |
| Malayalam | `piper-ml-in-arjun-medium`, `piper-ml-in-meera-medium` (data licence unclear: check before commercial use) | yes | **new**: round-trip CER 2.2 % (WER 10.0 %), RTF 0.19; round-trip CER 6.7 % (WER 10.0 %), RTF 0.19 |
| Marathi | `piper-mr-in-google-medium` (9 speakers, CC-BY-SA-4.0) | yes | **new**: round-trip CER 6.1 % (WER 33.3 %), RTF 0.13 |
| Bengali | `piper-bn-bd-google-medium` (16 speakers, **new**: round-trip CER 6.0 % (WER 33.3 %), RTF 0.22); `coqui-bn-female`, `coqui-bn-male` (character input); `mimic3-bn-multi` | piper, mimic3: yes; coqui: no | the others validated (audio produced; not transcribed back) |
| Urdu | `piper-ur-pk-fasih-medium`; `piper-ur-pk-aegis-female-medium` (**new**: round-trip CER 2.8 % (WER 27.8 %), RTF 0.14) | yes | `fasih` validated (audio produced) |
| Nepali | `piper-ne-np-chitwan-medium` (CC0 data; **new**: round-trip CER 9.3 % (WER 46.2 %), RTF 0.07); `piper-ne-np-google-medium`, `-x-low`; `mimic3-ne-google` | yes | the others validated (audio produced) |
| Gujarati | `mimic3-gu-cmu-indic` | yes | validated (audio produced) |
| Maithili, Bhojpuri, Kannada, Magahi, Chhattisgarhi; more voices for hi, bn, mr, te, gu and Indian English | `syspin-<lang>-<female\|male>` (IISc SYSPIN, MIT weights, CC-BY-4.0 data; character input) | no | **published** (release `voices-v1`); pull them like any other voice. Round trips: below |
| Punjabi, Odia, Assamese, Dogri; more voices for Maithili, Magahi, Chhattisgarhi, Urdu, Awadhi, Haryanvi and the major languages | `mms-tts-<iso>` (Meta MMS-TTS, **CC-BY-NC-4.0: non-commercial only**; character input, 16 kHz) | no | **published** (release `voices-v1`, Hugging Face `codebucketsolutions/`). Run here from a local conversion, 5 sentences each, transcribed back by the language's IndicConformer: Punjabi 19.4 % WER / 10.3 % CER, Odia 18.5 / 5.7, Assamese 13.3 / 3.2, Dogri 18.8 / 8.0, Maithili 48.6 / 12.9, Urdu 24.2 / 7.5; Magahi 52.9 / 30.5 and Chhattisgarhi 45.7 / 12.3 through the *Hindi* recogniser (there is none of their own). Most Punjabi and Odia errors are a stray syllable added at the start or end (`docs/measurements/mms-tts-round-trip.json`) |
| Sanskrit, Sindhi, Kashmiri, Konkani, Manipuri, Bodo, Santali | - | - | **no voice** that runs without PyTorch: MMS-TTS has none of them; Indic Parler-TTS (Apache-2.0, gated) and Svara (Apache-2.0) cover some and run through the PyTorch engine plugin ([plugins.md](plugins.md); Indic Parler-TTS was run on CPU there, far slower than real time) |

⚠ = the voice's training data is under a non-commercial licence.

The new Piper voices are the Piper project's own files (and two third-party
Tamil voices in the same format), converted for sherpa-onnx by bolo right
after the download; any other Piper voice works by name the same way
(`bolo pull piper:<voice>`, see [models.md](models.md)). "Transcribed back"
means: one FLEURS sentence was synthesised and recognised by that language's
IndicConformer model, whose own error is part of the number. Nobody who
speaks these languages listened. "Audio produced; not transcribed back"
means the voice loaded and synthesised a sentence of plausible length and
loudness: smoke-tested, no more.

**SYSPIN voices for Maithili and Bhojpuri.** IISc's SYSPIN project publishes
studio-quality single-speaker VITS voices as PyTorch checkpoints. bolo's
repository contains the conversion (`tools/voices/export_syspin.py`) and the
workflow that published the converted files as release assets (`voices-v1`).
Before publication, archives packed from voices converted earlier with the
same recipe went through the real pull path and were transcribed back
(`docs/measurements/syspin-voices-archive-pull.json`):

| Voice | Transcribed back by | CER | WER |
| --- | --- | --- | --- |
| `syspin-mai-female` | `indicconformer-mai-community` | 11.9 % | 52.9 % |
| `syspin-mai-male` | `indicconformer-mai-community` | 7.5 % | 41.2 % |
| `syspin-bho-female` | `indicconformer-hi-community` (no Bhojpuri recogniser) | 5.7 % | 13.3 % |

(two sentences each; the recognisers' own error rates are of the same size,
so this says "intelligible", not how good).

**Text cleaning.** The SYSPIN voices were trained on text cleaned by Coqui's
`multilingual_cleaners` (NFC, lower case, `;` and `:` to commas, hyphens to
spaces, brackets and quotes removed). sherpa-onnx does not do that, so text
with capitals, those marks or a precomposed nukta letter (`ज़` as one
character) reached the voices as symbols they never learnt. bolo now applies
the same cleaning as the voices' training (and as the in-browser voices of the
console, whose fixtures the tests share). On the published voices, 8 sentences
each written with such text, transcribed back
(`docs/measurements/syspin-text-cleaning.json`):

| Voice | Without the cleaning | With it |
| --- | --- | --- |
| `syspin-hi-female` (Hindi recogniser) | 24.6 % WER / 8.2 % CER | 14.5 % / 4.2 % |
| `syspin-en-female` (`whisper-base`) | 20.8 % / 12.6 % | 13.9 % / 9.2 % |

These voices read **characters from a closed set**: the Bhojpuri female
voice has no digits and no Latin letters at all, the Maithili female voice
has only some digits. Anything outside the set would be skipped silently, so
bolo refuses instead (`400 unsupported_characters`, naming the characters):
write numbers out in words and transliterate English words into Devanagari.
There is no number normaliser for Maithili or Bhojpuri; a voice entry can opt
into Hindi number words with the option `numbers: hi`, which is an
approximation.

Without these voices, a Hindi voice will read Maithili or Bhojpuri text,
since the script is the same, with Hindi pronunciation. We did not evaluate
that and do not present it as support for either language.

Text handling for Indic scripts:

* sentences are split at the danda (`।`, `॥`) and the Urdu full stop (`۔`,
  `؟`) as well as Latin punctuation, so streaming synthesis starts after the
  first sentence in any of these scripts;
* Hindi numbers are spelled out in the Indian system before synthesis:
  `250000` → `दो लाख पचास हज़ार`, Devanagari digits included. Other Indian
  languages keep their digits: a phoneme-based voice (Piper format) reads
  them through its phonemiser, a character-input voice refuses them (above);
* code-mixed input such as `मेरा order कल deliver होगा` is passed to a
  phoneme-based voice unchanged. How a Hindi voice pronounces Latin-script
  words depends on the voice; we have not evaluated it.

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
* The routes on this page take audio, not calls: they do not parse RTP
  headers, reorder packets, conceal losses or detect DTMF. For whole calls
  use [telephony.md](telephony.md): bolo connects Twilio, Plivo, Telnyx and
  Exotel media streams and has a minimal SIP endpoint (G.711 RTP, RFC 4733
  key presses) for trunks and PBXs. None of that has met a real phone
  network yet either.
* Models are trained on 16 kHz audio. Upsampled 8 kHz audio has no content
  above 4 kHz, which costs some accuracy with every model. No model in the
  catalog is trained specifically for 8 kHz telephone speech; sherpa-onnx has
  loaders for such models (`telespeech_ctc`, `online_t_one_ctc`) that you can
  use with `bolo pull hf:… --kind …`, untested here.
* G.722, G.729, AMR and Opus-in-RTP are not handled by the WebSocket (Opus
  and AMR files can be uploaded if PyAV or ffmpeg is installed).
