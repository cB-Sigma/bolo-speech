# Models

bolo's catalog is a set of YAML files in `server/src/bolo/catalog/`. Each
entry says what the model is, which engine runs it, where its files come
from (a Hugging Face repository pinned to a commit), what licence it carries
and whether the maintainers actually ran it.

```bash
bolo list --all                 # the catalog
bolo list --all -q hindi        # filter
bolo show kokoro-v1.0-int8      # one entry, with its source and voices
bolo pull moonshine-tiny-en     # download
bolo pull hf:owner/repo         # any compatible model that is not in the catalog
bolo recommend --task stt       # the same models, ranked for the machine you are on
```

Which of these suit your machine is a separate question, answered by
`bolo recommend`, the `fit` on every catalog entry and the console's Models
screen: see [recommendations.md](recommendations.md). That is advice only;
nothing in the catalog is hidden or refused because of hardware.

## What "validated" means

`validated: cpu` is set only by `server/tools/validate_models.py`, which
downloads the model, loads it through the server's own code path, runs it
and writes down what it measured. Everything else is `untested`: the entry
was checked against the Hub (the files exist, the size matches) and the
loader for its architecture is exercised by a sibling model, but that exact
model was not run.

How the numbers were produced:

* **Hardware**: 4 vCPUs of an Intel Xeon at 2.1 GHz (a shared cloud VM), no
  GPU, Linux x86-64, Python 3.12, 4 inference threads. The machine was
  shared with other workloads during the runs, so treat the figures as
  indicative and somewhat pessimistic, not as a benchmark.
* **RTF** (real-time factor) = processing time / audio duration, best of
  three runs after a warm-up. 0.05 means twenty times faster than real time.
* **Speech-to-text** models transcribed the `test_wavs` shipped in their own
  model repository (real recordings; WER is given when the repository has
  reference transcripts) or, failing that, speech synthesised by an already
  validated voice (then the WER says more about the pair than about the model).
* **Text-to-speech** models synthesised a sentence in their language. English
  voices were transcribed back by `moonshine-tiny-en` ("round-trip WER"); for
  other languages the check is only that audible speech of plausible length
  came out, unless noted.
* The raw report is in [`docs/measurements/validation.json`](measurements/validation.json).

GPU execution was measured for a few models on one laptop (see
[status.md](status.md#a-windows-11-laptop)); untested for the rest.

## Tiers

| Tier | Meaning | Examples |
| --- | --- | --- |
| `realtime` | small and fast enough for live use on a few CPU cores | Moonshine tiny/base, streaming Zipformer, Whisper tiny, Piper voices, KittenTTS |
| `balanced` | a good default when latency is not critical | Whisper base/small, Parakeet 110M, SenseVoice, Kokoro int8 |
| `accuracy` | the largest models; slow on CPU | Whisper medium/large, Parakeet 0.6B |

`streaming: true` marks models that decode incrementally by design
(transducer/CTC with streaming encoders). Every other speech-to-text model
can still be used on the realtime WebSocket through pseudo-streaming; see
[realtime.md](realtime.md).

## The catalog

<!-- BEGIN GENERATED: models -->
564 models: 261 validated on CPU by the maintainers, 303 untested. 34 more are cloud models (engine `remote`: paid cloud APIs; audio leaves this server; none was run against a real account, see [providers.md](providers.md)).

### Speech to text

#### OpenAI (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `openai-whisper-1` | af, am, ar, as, az, ba +94 | accuracy | pseudo | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |
| `openai-gpt-4o-transcribe` | af, am, ar, as, az, ba +94 | accuracy | pseudo | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |
| `openai-gpt-4o-mini-transcribe` | af, am, ar, as, az, ba +94 | accuracy | pseudo | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |
| `openai-gpt-transcribe` | af, am, ar, as, az, ba +94 | accuracy | pseudo | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |

#### Sarvam (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sarvam-saaras-v4` | hi, bn, kn, ml, mr, or +17 | accuracy | pseudo | none (cloud API) | Sarvam AI Terms of Service (paid cloud API) · [terms](https://www.sarvam.ai/terms-of-service) · needs provider `sarvam` (key: `BOLO_PROVIDER_SARVAM_API_KEY`) | untested |
| `sarvam-saaras-v3` | hi, bn, kn, ml, mr, or +17 | accuracy | pseudo | none (cloud API) | Sarvam AI Terms of Service (paid cloud API) · [terms](https://www.sarvam.ai/terms-of-service) · needs provider `sarvam` (key: `BOLO_PROVIDER_SARVAM_API_KEY`) | untested |
| `sarvam-saaras-v4-streaming` | hi, bn, kn, ml, mr, or +17 | realtime | native | none (cloud API) | Sarvam AI Terms of Service (paid cloud API) · [terms](https://www.sarvam.ai/terms-of-service) · needs provider `sarvam` (key: `BOLO_PROVIDER_SARVAM_API_KEY`) | untested |

#### ElevenLabs (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `elevenlabs-scribe-v3` | af, am, ar, hy, as, ast +95 | accuracy | pseudo | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |
| `elevenlabs-scribe-v3-live` | af, am, ar, hy, as, ast +95 | realtime | native | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |
| `elevenlabs-scribe-v2` | af, am, ar, as, az, ba +94 | accuracy | pseudo | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |

#### Deepgram (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `deepgram-nova-3` | af, ar, hy, as, eu, be +55 | realtime | native | none (cloud API) | Deepgram Terms of Service (paid cloud API) · [terms](https://deepgram.com/terms) · needs provider `deepgram` (key: `BOLO_PROVIDER_DEEPGRAM_API_KEY`) | untested |
| `deepgram-nova-2` | bg, ca, zh, cs, da, nl +27 | realtime | native | none (cloud API) | Deepgram Terms of Service (paid cloud API) · [terms](https://deepgram.com/terms) · needs provider `deepgram` (key: `BOLO_PROVIDER_DEEPGRAM_API_KEY`) | untested |

#### Soniox (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `soniox-stt-async-v6` | af, sq, ar, az, eu, be +54 | accuracy | pseudo | none (cloud API) | Soniox Terms of Service (paid cloud API) · [terms](https://soniox.com/policies/terms-of-service) · needs provider `soniox` (key: `BOLO_PROVIDER_SONIOX_API_KEY`) | untested |
| `soniox-stt-rt-v6` | af, sq, ar, az, eu, be +54 | realtime | native | none (cloud API) | Soniox Terms of Service (paid cloud API) · [terms](https://soniox.com/policies/terms-of-service) · needs provider `soniox` (key: `BOLO_PROVIDER_SONIOX_API_KEY`) | untested |

#### AssemblyAI (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `assemblyai-stt-3.7` | af, ar, yue, ca, da, nl +26 | accuracy | pseudo | none (cloud API) | AssemblyAI Terms of Service (paid cloud API) · [terms](https://www.assemblyai.com/legal/terms-of-service) · needs provider `assemblyai` (key: `BOLO_PROVIDER_ASSEMBLYAI_API_KEY`) | untested |
| `assemblyai-stt-3.7-realtime` | af, ar, yue, ca, da, nl +26 | realtime | native | none (cloud API) | AssemblyAI Terms of Service (paid cloud API) · [terms](https://www.assemblyai.com/legal/terms-of-service) · needs provider `assemblyai` (key: `BOLO_PROVIDER_ASSEMBLYAI_API_KEY`) | untested |
| `assemblyai-stt-3.7-realtime-agent` | af, ar, yue, ca, da, nl +26 | realtime | native | none (cloud API) | AssemblyAI Terms of Service (paid cloud API) · [terms](https://www.assemblyai.com/legal/terms-of-service) · needs provider `assemblyai` (key: `BOLO_PROVIDER_ASSEMBLYAI_API_KEY`) | untested |

#### Microsoft MAI (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mai-transcribe-2-streaming` | af, ar, as, az, bg, bn +54 | realtime | native | none (cloud API) | Microsoft Azure terms; preview (paid cloud API) · [terms](https://azure.microsoft.com/support/legal/preview-supplemental-terms/) · needs provider `azure-mai` (key: `BOLO_PROVIDER_AZURE_MAI_API_KEY`) | untested |

#### Whisper

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `whisper-tiny-mlx` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 74 MB | MIT | **CPU** · RTF 18.100 |
| `whisper-base-mlx` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 144 MB | MIT | **CPU** · RTF 20.000 |
| `faster-whisper-tiny` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 78 MB | MIT | **CPU** · RTF 0.091 · WER 0.0% |
| `faster-whisper-tiny.en` | en | realtime | pseudo | 78 MB | MIT | **CPU** · RTF 0.081 · WER 0.0% |
| `faster-whisper-base` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 148 MB | MIT | **CPU** · RTF 0.152 · WER 0.0% |
| `faster-whisper-base.en` | en | balanced | pseudo | 148 MB | MIT | untested |
| `faster-whisper-small` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 486 MB | MIT | **CPU** · RTF 0.425 · WER 0.0% |
| `faster-whisper-small.en` | en | balanced | pseudo | 486 MB | MIT | untested |
| `faster-whisper-medium` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.5 GB | MIT | untested |
| `faster-whisper-medium.en` | en | accuracy | pseudo | 1.5 GB | MIT | untested |
| `faster-whisper-large-v2` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 3.1 GB | MIT | untested |
| `faster-whisper-large-v3` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 3.1 GB | MIT | untested |
| `faster-whisper-large-v3-turbo` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.6 GB | MIT | untested |
| `faster-whisper-distil-small.en` | en | balanced | pseudo | 336 MB | MIT | **CPU** · RTF 0.212 · WER 0.0% |
| `faster-whisper-distil-medium.en` | en | balanced | pseudo | 792 MB | MIT | untested |
| `faster-whisper-distil-large-v2` | en | accuracy | pseudo | 1.5 GB | MIT | untested |
| `faster-whisper-distil-large-v3` | en | accuracy | pseudo | 1.5 GB | MIT | untested |
| `faster-whisper-tiny-hindi` | hi | realtime | pseudo | 79 MB | CC-BY-4.0 | **CPU** · RTF 0.153 · WER 0.0% |
| `faster-whisper-small-hindi` | hi | balanced | pseudo | 487 MB | CC-BY-4.0 | **CPU** · RTF 1.132 · WER 0.0% |
| `faster-whisper-medium-hindi` | hi | balanced | pseudo | 1.5 GB | CC-BY-4.0 | untested |
| `faster-whisper-base-hinglish` | hi, en | realtime | pseudo | 81 MB | Apache-2.0 | **CPU** · RTF 0.174 |
| `faster-whisper-medium-tamil` | ta | accuracy | pseudo | 778 MB | Apache-2.0 | **CPU** · RTF 1.761 · WER 6.9% · CER 2.6% |
| `faster-whisper-medium-kannada` | kn | accuracy | pseudo | 778 MB | Apache-2.0 | **CPU** · RTF 1.838 · WER 25.5% · CER 24.5% |
| `faster-whisper-medium-telugu` | te | accuracy | pseudo | 1.5 GB | Apache-2.0 | **CPU** · RTF 1.521 · WER 4.3% · CER 7.0% |
| `faster-whisper-medium-bengali` | bn | accuracy | pseudo | 1.5 GB | Apache-2.0 | **CPU** · RTF 0.685 · WER 13.2% · CER 4.1% |
| `faster-whisper-medium-punjabi` | pa | accuracy | pseudo | 773 MB | MIT | **CPU** · RTF 1.213 · WER 48.4% · CER 38.2% |
| `faster-whisper-large-v2-hindi` | hi | accuracy | pseudo | 3.1 GB | CC-BY-4.0 | untested |
| `whisper-tiny` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 104 MB | MIT | **CPU** · RTF 0.140 · WER 4.2% |
| `whisper-tiny.en` | en | realtime | pseudo | 104 MB | MIT | **CPU** · RTF 0.181 · WER 3.1% |
| `whisper-base` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 161 MB | MIT | **CPU** · RTF 0.115 · WER 3.1% |
| `whisper-base.en` | en | balanced | pseudo | 161 MB | MIT | **CPU** · RTF 0.106 · WER 3.1% |
| `whisper-small` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 375 MB | MIT | **CPU** · RTF 0.365 · WER 3.1% |
| `whisper-small.en` | en | balanced | pseudo | 376 MB | MIT | untested |
| `whisper-medium` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 946 MB | MIT | untested |
| `whisper-medium.en` | en | accuracy | pseudo | 946 MB | MIT | untested |
| `whisper-large-v3` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.8 GB | MIT | untested |
| `whisper-large-v3-turbo` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.0 GB | MIT | **CPU** · RTF 0.300 · WER 3.1% |
| `whisper-distil-small.en` | en | balanced | pseudo | 299 MB | MIT | **CPU** · RTF 0.131 · WER 3.1% |
| `whisper-distil-medium.en` | en | balanced | pseudo | 573 MB | MIT | untested |
| `whisper-distil-large-v2` | en | accuracy | pseudo | 983 MB | MIT | untested |
| `whispercpp-tiny` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 78 MB | MIT | ran once here (shared CPU), not measured |
| `whispercpp-tiny-q5_1` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 32 MB | MIT | untested |
| `whispercpp-tiny-q8_0` | en, zh, de, es, ru, ko +94 | realtime | pseudo | 44 MB | MIT | untested |
| `whispercpp-tiny.en` | en | realtime | pseudo | 78 MB | MIT | untested |
| `whispercpp-tiny.en-q5_1` | en | realtime | pseudo | 32 MB | MIT | untested |
| `whispercpp-tiny.en-q8_0` | en | realtime | pseudo | 44 MB | MIT | untested |
| `whispercpp-base` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 148 MB | MIT | ran once here (shared CPU), not measured |
| `whispercpp-base-q5_1` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 60 MB | MIT | untested |
| `whispercpp-base-q8_0` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 82 MB | MIT | untested |
| `whispercpp-base.en` | en | balanced | pseudo | 148 MB | MIT | untested |
| `whispercpp-base.en-q5_1` | en | balanced | pseudo | 60 MB | MIT | untested |
| `whispercpp-base.en-q8_0` | en | balanced | pseudo | 82 MB | MIT | untested |
| `whispercpp-small` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 488 MB | MIT | ran once here (shared CPU), not measured |
| `whispercpp-small-q5_1` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 190 MB | MIT | untested |
| `whispercpp-small-q8_0` | en, zh, de, es, ru, ko +94 | balanced | pseudo | 264 MB | MIT | untested |
| `whispercpp-small.en` | en | balanced | pseudo | 488 MB | MIT | untested |
| `whispercpp-small.en-q5_1` | en | balanced | pseudo | 190 MB | MIT | untested |
| `whispercpp-small.en-q8_0` | en | balanced | pseudo | 264 MB | MIT | untested |
| `whispercpp-medium` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.5 GB | MIT | untested |
| `whispercpp-medium-q5_0` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 539 MB | MIT | untested |
| `whispercpp-medium-q8_0` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 823 MB | MIT | untested |
| `whispercpp-medium.en` | en | accuracy | pseudo | 1.5 GB | MIT | untested |
| `whispercpp-medium.en-q5_0` | en | accuracy | pseudo | 539 MB | MIT | untested |
| `whispercpp-medium.en-q8_0` | en | accuracy | pseudo | 823 MB | MIT | untested |
| `whispercpp-large-v3` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 3.1 GB | MIT | untested |
| `whispercpp-large-v3-q5_0` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.1 GB | MIT | untested |
| `whispercpp-large-v3-turbo` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 1.6 GB | MIT | ran once here (shared CPU), not measured |
| `whispercpp-large-v3-turbo-q5_0` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 574 MB | MIT | ran once here (shared CPU), not measured |
| `whispercpp-large-v3-turbo-q8_0` | en, zh, de, es, ru, ko +94 | accuracy | pseudo | 874 MB | MIT | untested |

#### Parakeet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `parakeet-tdt-ctc-110m-mlx` | en | realtime | pseudo | 459 MB | CC-BY-4.0 | **CPU** · RTF 1.490 |
| `parakeet-tdt-0.6b-v2` | en | accuracy | pseudo | 661 MB | CC-BY-4.0 | **CPU** · RTF 0.081 |
| `parakeet-tdt-0.6b-v3` | bg, hr, cs, da, nl, en +19 | accuracy | pseudo | 671 MB | CC-BY-4.0 | **CPU** · RTF 0.096 |
| `parakeet-tdt-ctc-110m` | en | balanced | pseudo | 458 MB | CC-BY-4.0 | **CPU** · RTF 0.030 |
| `parakeet-tdt-110m` | en | balanced | pseudo | 477 MB | CC-BY-4.0 | untested |
| `parakeet-unified-en-streaming-240ms` | en | realtime | native | 663 MB | CC-BY-4.0 | untested |
| `parakeet-unified-en-streaming-560ms` | en | realtime | native | 663 MB | CC-BY-4.0 | **CPU** · RTF 4.694 |
| `parakeet-unified-en-streaming-1120ms` | en | balanced | native | 663 MB | CC-BY-4.0 | untested |

#### SOV-Transcribe

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sov-transcribe-1-mlx-4bit` | en, fr, de, es, it, pt +1 | accuracy | pseudo | 15.6 GB | Apache-2.0 | untested |
| `sov-transcribe-1-mlx-8bit` | en, fr, de, es, it, pt +1 | accuracy | pseudo | 28.2 GB | Apache-2.0 | untested |
| `sov-transcribe-1` | en, fr, de, es, it, pt +1 | accuracy | pseudo | 53.3 GB | Apache-2.0 · needs plugin torch | untested |

#### SraVaani

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sravaani-1.0` | as, bn, en, gu, hi, kn +8 | accuracy | pseudo | 909 MB | MIT · needs plugin torch · gated | untested |

#### IndicConformer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indicconformer-600m-official` | as, bn, brx, doi, gu, hi +16 | accuracy | pseudo | 2.6 GB | MIT · needs plugin torch · gated | untested |
| `indicconformer-multi-community` | hi, as, bn, brx, gu, kn +2 | balanced | pseudo | 197 MB | MIT | **CPU** · RTF 0.131 · WER 0.0% |
| `indicconformer-mai-community` | mai | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.064 · WER 49.6% · CER 11.6% |
| `indicconformer-hi-community` | hi | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.066 · WER 15.2% · CER 7.0% |
| `indicconformer-as-community` | as | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.150 · WER 30.8% · CER 14.9% |
| `indicconformer-bn-community` | bn | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.106 · WER 10.8% · CER 2.5% |
| `indicconformer-brx-community` | brx | balanced | pseudo | 198 MB | MIT | untested |
| `indicconformer-doi-community` | doi | balanced | pseudo | 198 MB | MIT | untested |
| `indicconformer-gu-community` | gu | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.114 · WER 21.7% · CER 5.4% |
| `indicconformer-kn-community` | kn | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.070 · WER 15.1% · CER 4.8% |
| `indicconformer-kok-community` | kok | balanced | pseudo | 198 MB | MIT | untested |
| `indicconformer-ks-community` | ks | balanced | pseudo | 198 MB | MIT | untested |
| `indicconformer-ml-community` | ml | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.124 · WER 31.4% · CER 8.8% |
| `indicconformer-mni-community` | mni | balanced | pseudo | 198 MB | MIT | untested |
| `indicconformer-mr-community` | mr | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.202 · WER 19.3% · CER 4.2% |
| `indicconformer-ne-community` | ne | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.135 · WER 30.7% · CER 12.7% |
| `indicconformer-or-community` | or | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.155 · WER 15.5% · CER 4.8% |
| `indicconformer-pa-community` | pa | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.139 · WER 15.5% · CER 5.3% |
| `indicconformer-sa-community` | sa | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.127 · WER 44.2% · CER 6.1% |
| `indicconformer-sat-community` | sat | balanced | pseudo | 198 MB | MIT | untested |
| `indicconformer-sd-community` | sd | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.109 |
| `indicconformer-ta-community` | ta | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.060 · WER 25.5% · CER 10.2% |
| `indicconformer-te-community` | te | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.074 · WER 21.1% · CER 10.4% |
| `indicconformer-ur-community` | ur | balanced | pseudo | 198 MB | MIT | **CPU** · RTF 0.143 · WER 18.5% · CER 7.5% |
| `indicconformer-ta` | ta | balanced | pseudo | 138 MB | MIT | **CPU** · RTF 0.228 · WER 29.0% · CER 10.3% |
| `indicconformer-te` | te | balanced | pseudo | 138 MB | MIT | **CPU** · RTF 0.262 · WER 21.7% · CER 10.5% |
| `indicconformer-ml` | ml | balanced | pseudo | 138 MB | MIT | **CPU** · RTF 0.174 · WER 34.3% · CER 9.8% |

#### Reverb

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `reverb-asr` | en | accuracy | pseudo | 1.2 GB | Rev Model Non-Production License (non-commercial) ⚠ · needs plugin torch · gated | **CPU** · RTF 0.050 · WER 0.0% |

#### DiCoW

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `dicow-v3.3` | en | accuracy | pseudo | 3.6 GB | CC-BY-4.0 · needs plugin torch | **CPU** · RTF 0.453 |
| `se-dicow` | en | accuracy | pseudo | 4.4 GB | Apache-2.0 · needs plugin torch | **CPU** · RTF 0.589 |
| `dicow-v3.3-onnx` | en | accuracy | pseudo | 1.5 GB | CC-BY-4.0 | **CPU** · RTF 0.725 |

#### Jais-ASR

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `jais-asr` | ar, en, fr, es, it | accuracy | pseudo | 11.5 GB | Apache-2.0 · needs plugin torch | untested |

#### Whistle

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `whistle` | en, de, fr, es, it, nl +1 | realtime | pseudo | 17 MB | Apache-2.0 | **CPU** · RTF 0.244 · WER 22.8% |

#### Voxtral

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `voxtral-mini-3b-llamacpp` (Voxtral) | en, es, fr, pt, hi, de +2 | accuracy | pseudo | 3.2 GB | Apache-2.0 · runs on llama-server · GPU recommended | **CPU** · RTF 1.980 · WER 2.8% |
| `voxtral-small-24b-llamacpp` (Voxtral) | en, es, fr, pt, hi, de +2 | accuracy | pseudo | 15.7 GB | Apache-2.0 · runs on llama-server · GPU required | untested |

#### Nemotron 3.5 Transcribe

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3.5-transcribe-30b-a4b-iq4nl-llamacpp` (Nemotron 3.5 Transcribe) | cs, nl, en, fr, de, hi +34 | accuracy | pseudo | 20.2 GB | OpenMDW-1.1 · runs on llama-server · GPU recommended | untested |
| `nemotron-3.5-transcribe-30b-a4b-q8-llamacpp` (Nemotron 3.5 Transcribe) | cs, nl, en, fr, de, hi +34 | accuracy | pseudo | 36.5 GB | OpenMDW-1.1 · runs on llama-server · GPU recommended | untested |

#### Canary-Qwen

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `canary-qwen-2.5b-onnx` | en | accuracy | pseudo | 2.9 GB | CC-BY-4.0 | **CPU** · RTF 3.194 · WER 0.0% |

#### SpeechBrain wav2vec 2.0

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `wav2vec2-dvoice-fongbe` | fon | accuracy | pseudo | 1.3 GB | Apache-2.0 | **CPU** · RTF 0.219 |
| `wav2vec2-dvoice-wolof` | wo | accuracy | pseudo | 1.3 GB | Apache-2.0 | **CPU** · RTF 0.219 |
| `wav2vec2-dvoice-darija` | ary | accuracy | pseudo | 1.3 GB | Apache-2.0 | **CPU** · RTF 0.088 |
| `wav2vec2-dvoice-amharic` | am | accuracy | pseudo | 1.3 GB | Apache-2.0 | **CPU** · RTF 0.170 |
| `wav2vec2-commonvoice-14-rw` | rw | accuracy | pseudo | 1.3 GB | Apache-2.0 | **CPU** · RTF 0.202 |

#### Moonshine

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `moonshine-tiny-en` | en | realtime | pseudo | 124 MB | MIT | **CPU** · RTF 0.033 · WER 2.1% |
| `moonshine-base-en` | en | realtime | pseudo | 287 MB | MIT | **CPU** · RTF 0.115 · WER 2.1% |
| `moonshine-v2-tiny-en` | en | realtime | pseudo | 44 MB | MIT | **CPU** · RTF 0.093 |
| `moonshine-v2-base-en` | en | realtime | pseudo | 141 MB | MIT | **CPU** · RTF 0.160 |
| `moonshine-v2-base-es` | es | realtime | pseudo | 65 MB | Moonshine Community License (non-commercial) ⚠ | untested |
| `moonshine-v2-tiny-ja` | ja | realtime | pseudo | 72 MB | Moonshine Community License (non-commercial) ⚠ | untested |
| `moonshine-v2-tiny-ko` | ko | realtime | pseudo | 72 MB | Moonshine Community License (non-commercial) ⚠ | untested |
| `moonshine-v2-base-zh` | zh | realtime | pseudo | 141 MB | Moonshine Community License (non-commercial) ⚠ | untested |
| `moonshine-v2-base-ar` | ar | realtime | pseudo | 141 MB | Moonshine Community License (non-commercial) ⚠ | untested |
| `moonshine-v2-base-uk` | uk | realtime | pseudo | 141 MB | Moonshine Community License (non-commercial) ⚠ | untested |
| `moonshine-v2-base-vi` | vi | realtime | pseudo | 141 MB | Moonshine Community License (non-commercial) ⚠ | untested |

#### NeMo FastConformer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `nemo-fastconformer-ctc-en-streaming-80ms` | en | realtime | native | 132 MB | CC-BY-4.0 | **CPU** · RTF 0.756 · WER 0.0% |
| `nemo-fastconformer-ctc-en-streaming-480ms` | en | realtime | native | 132 MB | CC-BY-4.0 | **CPU** · RTF 0.334 · WER 3.8% |
| `nemo-fastconformer-ctc-en-streaming-1040ms` | en | realtime | native | 132 MB | CC-BY-4.0 | untested |
| `nemo-fastconformer-transducer-en-streaming-80ms` | en | realtime | native | 137 MB | CC-BY-4.0 | **CPU** · RTF 0.640 · WER 1.0% |
| `nemo-fastconformer-transducer-en-streaming-480ms` | en | realtime | native | 137 MB | CC-BY-4.0 | untested |
| `nemo-fastconformer-ctc-en` | en | balanced | pseudo | 458 MB | CC-BY-4.0 | untested |
| `nemo-fastconformer-ctc-en-de-es-fr` | en, de, es, fr | balanced | pseudo | 461 MB | CC-BY-4.0 | untested |
| `nemo-fastconformer-ctc-multi10` | be, de, en, es, fr, hr +4 | balanced | pseudo | 461 MB | CC-BY-4.0 | untested |

#### Canary

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `canary-180m-flash` | en, es, de, fr | balanced | pseudo | 207 MB | CC-BY-4.0 | **CPU** · RTF 0.358 |
| `canary-1b-v2` | en, de, fr, es, it, pt +19 | accuracy | pseudo | 1.4 GB | CC-BY-4.0 | **CPU** · RTF 1.600 |

#### Zipformer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `zipformer-en-20m-streaming` | en | realtime | native | 45 MB | Apache-2.0 | **CPU** · RTF 0.137 · WER 21.5% |
| `zipformer-en-streaming` | en | realtime | native | 147 MB | Apache-2.0 | **CPU** · RTF 0.186 · WER 0.0% |
| `zipformer-zh-en-streaming` | zh, en | realtime | native | 199 MB | Apache-2.0 | untested |
| `zipformer-ko-streaming` | ko | realtime | native | 141 MB | Apache-2.0 | untested |
| `zipformer-multi-zh-hans-streaming` | zh | realtime | native | 76 MB | Apache-2.0 | **CPU** · RTF 0.218 |
| `zipformer-ru-vosk-streaming` | ru | realtime | native | 29 MB | Apache-2.0 | **CPU** · RTF 0.089 |
| `zipformer-bn-vosk-streaming` | bn | realtime | native | 94 MB | Apache-2.0 | **CPU** · RTF 0.102 |
| `zipformer-8lang-streaming` | ar, en, id, ja, ru, th +2 | realtime | native | 339 MB | Apache-2.0 | untested |
| `kroko-en-streaming` | en | realtime | native | 71 MB | CC-BY-SA-4.0 | **CPU** · RTF 0.085 |
| `kroko-es-streaming` | es | realtime | native | 156 MB | CC-BY-SA-4.0 | untested |
| `kroko-fr-streaming` | fr | realtime | native | 71 MB | CC-BY-SA-4.0 | untested |
| `kroko-de-streaming` | de | realtime | native | 71 MB | CC-BY-SA-4.0 | untested |
| `zipformer-en-gigaspeech` | en | balanced | pseudo | 75 MB | Apache-2.0 | **CPU** · RTF 0.028 |
| `zipformer-ru` | ru | balanced | pseudo | 73 MB | Apache-2.0 | untested |
| `zipformer-vi` | vi | balanced | pseudo | 77 MB | Apache-2.0 | untested |
| `zipformer-ko` | ko | balanced | pseudo | 85 MB | Apache-2.0 | untested |

#### SenseVoice

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sense-voice-small` | zh, en, ja, ko, yue | balanced | pseudo | 240 MB | FunASR Model License 1.1 (custom) | **CPU** · RTF 0.039 |

#### Paraformer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `paraformer-zh` | zh, en | balanced | pseudo | 227 MB | FunASR Model License 1.1 (custom) | **CPU** · RTF 0.091 |
| `paraformer-zh-en-streaming` | zh, en | realtime | native | 237 MB | FunASR Model License 1.1 (custom) | untested |

#### Dolphin

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `dolphin-base` | zh, ja, th, ru, ko, id +28 | balanced | pseudo | 104 MB | Apache-2.0 | **CPU** · RTF 0.021 |
| `dolphin-small` | zh, ja, th, ru, ko, id +28 | balanced | pseudo | 250 MB | Apache-2.0 | untested |

#### Omnilingual ASR

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `omnilingual-ctc-300m` | en, hi, bn, ta, te, mr +25 | balanced | pseudo | 365 MB | Apache-2.0 | **CPU** · RTF 0.112 |

#### Nemotron

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3.5-streaming-560ms` | ar, bg, cs, da, de, el +28 | accuracy | native | 682 MB | OpenMDW-1.1 | **CPU** · RTF 0.312 · WER 20.9% · CER 6.1% |

#### Rumble

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `rumble-2-flash` | af, ar, bn, bs, ca, ceb +39 | accuracy | pseudo | 656 MB | Apache-2.0 | **CPU** · RTF 0.463 · WER 0.0% |

#### Qwen3-ASR

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-asr-0.6b` | zh, en, yue, ar, de, fr +24 | accuracy | pseudo | 987 MB | Apache-2.0 | **CPU** · RTF 0.723 · WER 5.6% |
| `qwen3-asr-1.7b` | zh, en, yue, ar, de, fr +24 | accuracy | pseudo | 2.4 GB | Apache-2.0 | **CPU** · RTF 1.820 · WER 5.6% |

#### Cohere Transcribe

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `cohere-transcribe-03-2026` | en, fr, de, it, es, pt +8 | accuracy | pseudo | 2.9 GB | Apache-2.0 | **CPU** · RTF 0.859 · WER 0.0% |

### Text to speech

#### OpenAI (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `openai-tts-1` | af, am, ar, as, az, ba +94 | realtime | - | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |
| `openai-tts-1-hd` | af, am, ar, as, az, ba +94 | balanced | - | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |
| `openai-gpt-4o-mini-tts` | af, am, ar, as, az, ba +94 | balanced | - | none (cloud API) | OpenAI Services Agreement (paid cloud API) · [terms](https://openai.com/policies/services-agreement/) · needs provider `openai` (key: `BOLO_PROVIDER_OPENAI_API_KEY`) | untested |

#### Sarvam (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sarvam-bulbul-v3` | hi, bn, en, gu, kn, ml +5 | balanced | - | none (cloud API) | Sarvam AI Terms of Service (paid cloud API) · [terms](https://www.sarvam.ai/terms-of-service) · needs provider `sarvam` (key: `BOLO_PROVIDER_SARVAM_API_KEY`) | untested |
| `sarvam-bulbul-v2` | hi, bn, en, gu, kn, ml +5 | balanced | - | none (cloud API) | Sarvam AI Terms of Service (paid cloud API) · [terms](https://www.sarvam.ai/terms-of-service) · needs provider `sarvam` (key: `BOLO_PROVIDER_SARVAM_API_KEY`) | untested |
| `sarvam-bulbul-v4-flash` | hi, bn, en, gu, kn, ml +5 | realtime | - | none (cloud API) | Sarvam AI Terms of Service (paid cloud API) · [terms](https://www.sarvam.ai/terms-of-service) · needs provider `sarvam` (key: `BOLO_PROVIDER_SARVAM_API_KEY`) | untested |

#### ElevenLabs (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `elevenlabs-flash-v2.5` | en, ja, zh, de, hi, fr +26 | realtime | - | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |
| `elevenlabs-multilingual-v2` | en, ja, zh, de, hi, fr +23 | balanced | - | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |
| `elevenlabs-v3` | en, ja, zh, de, hi, fr +26 | balanced | - | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |

#### Microsoft MAI (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mai-voice-2.1` | cs, da, de, en, es, fi +17 | accuracy | - | none (cloud API) | Microsoft Azure terms; preview (paid cloud API) · [terms](https://azure.microsoft.com/support/legal/preview-supplemental-terms/) · needs provider `azure-mai` (key: `BOLO_PROVIDER_AZURE_MAI_API_KEY`) | untested |
| `mai-voice-2.1-flash` | cs, da, de, en, es, fi +17 | realtime | - | none (cloud API) | Microsoft Azure terms; preview (paid cloud API) · [terms](https://azure.microsoft.com/support/legal/preview-supplemental-terms/) · needs provider `azure-mai` (key: `BOLO_PROVIDER_AZURE_MAI_API_KEY`) | untested |

#### Kokoro

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kokoro-v1.0-mlx` | en, es, fr, hi, it, pt | balanced | - | 349 MB | Apache-2.0 | **CPU** · RTF 65.600 · round-trip WER 0.0% |
| `kokoro-v1.0-int8` | en, es, fr, hi, it, ja +2 | balanced | - | 172 MB | Apache-2.0 | **CPU** · RTF 0.884 · first audio 2406 ms · round-trip WER 0.0% |
| `kokoro-v1.0` | en, es, fr, hi, it, ja +2 | accuracy | - | 383 MB | Apache-2.0 | **CPU** · RTF 0.759 · first audio 2004 ms · round-trip WER 0.0% |

#### Parler-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-parler-tts` | as, bn, brx, doi, en, gu +15 | accuracy | - | 3.8 GB | Apache-2.0 · needs plugin torch · gated | **CPU** · RTF 22.500 · first audio 39400 ms |
| `parler-tts-tiny-v1` | en | balanced | - | 1.3 GB | Apache-2.0 · needs plugin torch | **CPU** · RTF 4.540 · first audio 11520 ms |

#### F5-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-f5` | as, bn, gu, hi, kn, ml +5 | accuracy | - | 1.4 GB | MIT · needs plugin torch · gated | **CPU** · RTF 119.700 · first audio 211280 ms |
| `f5-tts-v1-base` | en, zh | accuracy | - | 1.3 GB | CC-BY-NC-4.0 ⚠ · needs plugin torch | untested |

#### Veena

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `veena` | en, hi | accuracy | - | 7.6 GB | Apache-2.0 · needs plugin torch | untested |

#### Svara

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `svara-tts-v1` | as, bho, bn, brx, doi, en +13 | accuracy | - | 13.2 GB | Apache-2.0 · needs plugin torch | untested |

#### Indic-Speak

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-speak` | as, bn, brx, doi, en, gu +17 | accuracy | - | 7.1 GB | Indic Open Model License v1.0 ⚠ · needs plugin torch · gated | untested |

#### Spark-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `spark-tts-0.5b` | en, zh | accuracy | - | 3.9 GB | CC-BY-NC-SA-4.0 ⚠ · needs plugin torch | untested |

#### MaskGCT

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `maskgct` | en, zh, ja, ko, fr, de | accuracy | - | 6.6 GB | CC-BY-NC-4.0 ⚠ · needs plugin torch | untested |

#### LFM3-Audio 2B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `lfm3-audio-2b-onnx` | en, ar, da, de, el, es +12 | accuracy | - | 2.7 GB | LFM Open License v1.0 ⚠ | untested |

#### MMS-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mms-tts-pan` | pa | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-ory` | or | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-asm` | as | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-dgo` | doi | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-mai` | mai | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-mag` | mag | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-hne` | hne | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-awa` | awa | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-bgc` | bgc | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-urd` | ur | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | ran once here (shared CPU), not measured |
| `mms-tts-hin` | hi | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-ben` | bn | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-guj` | gu | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-kan` | kn | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-mal` | ml | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-mar` | mr | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-tam` | ta | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-tel` | te | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-eng` | en | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | **CPU** · RTF 0.196 · first audio 590 ms · round-trip WER 5.0% |
| `mms-tts-deu` | de | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-fra` | fr | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-spa` | es | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-rus` | ru | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-ukr` | uk | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-tha` | th | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |

#### Orpheus family on llama.cpp (SNAC in onnxruntime)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `svara-tts-v1-llamacpp` (Svara) | hi, bn, mr, te, kn, ta +13 | accuracy | - | 2.1 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |
| `svara-tts-v1-q8-llamacpp` (Svara) | hi, bn, mr, te, kn, ta +13 | accuracy | - | 3.6 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |
| `veena-llamacpp` (Veena) | hi, en | accuracy | - | 2.5 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |
| `veena-q8-llamacpp` (Veena) | hi, en | accuracy | - | 4.1 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |
| `orpheus-3b-0.1-ft-llamacpp` (Orpheus) | en | accuracy | - | 2.1 GB | Apache-2.0 (Llama 3.2 Community License for the base model) · runs on llama-server · GPU recommended | untested |
| `snortts-indic-v0-llamacpp` (snorTTS Indic) | hi, bn, gu, kn, ml, mr +3 | accuracy | - | 2.4 GB | Apache-2.0 (Llama 3.2 Community License for the base model) · runs on llama-server · GPU recommended | untested |

#### Piper

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `piper-hi-in-pratham-medium` | hi | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · phonemes: uccharan | **CPU** · RTF 0.152 · first audio 238 ms |
| `piper-hi-in-priyamvada-medium` | hi | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · phonemes: uccharan | **CPU** · RTF 0.056 · first audio 96 ms |
| `piper-hi-in-rohan-medium` | hi | realtime | - | 63 MB | See dataset licence: https://www.iitm.ac.in/donlab/indict… · phonemes: uccharan | **CPU** · RTF 0.092 · first audio 164 ms |
| `piper-ne-np-google-medium` | ne | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 International · phonemes: uccharan | **CPU** · RTF 0.040 · first audio 88 ms |
| `piper-ne-np-google-x-low` | ne | realtime | - | 28 MB | Voice: CC-BY-SA-4.0 International · phonemes: uccharan | untested |
| `piper-ur-pk-fasih-medium` | ur | realtime | - | 64 MB | See voice card · phonemes: uccharan | **CPU** · RTF 0.090 · first audio 191 ms |
| `piper-en-us-lessac-medium` | en | realtime | - | 63 MB | See dataset licence: https://www.cstr.ed.ac.uk/projects/b… ⚠ · phonemes: uccharan | **CPU** · RTF 0.126 · first audio 325 ms · round-trip WER 5.0% |
| `piper-en-us-lessac-low` | en | realtime | - | 63 MB | See dataset licence: https://www.cstr.ed.ac.uk/projects/b… ⚠ · phonemes: uccharan | untested |
| `piper-en-us-amy-low` | en | realtime | - | 63 MB | See dataset licence: https://github.com/MycroftAI/mimic3-… · phonemes: uccharan | **CPU** · RTF 0.049 · first audio 189 ms · round-trip WER 0.0% |
| `piper-en-us-amy-medium` | en | realtime | - | 63 MB | See dataset licence: https://github.com/MycroftAI/mimic3-… · phonemes: uccharan | untested |
| `piper-en-us-ryan-medium` | en | realtime | - | 63 MB | Voice: CC BY-NC-SA 4.0 ⚠ · phonemes: uccharan | **CPU** · RTF 0.118 · first audio 259 ms · round-trip WER 0.0% |
| `piper-en-us-ryan-high` | en | realtime | - | 121 MB | Voice: CC BY-NC-SA 4.0 ⚠ · phonemes: uccharan | untested |
| `piper-en-us-hfc-female-medium` | en | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · phonemes: uccharan | untested |
| `piper-en-us-hfc-male-medium` | en | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · phonemes: uccharan | untested |
| `piper-en-us-libritts-r-medium` | en | realtime | - | 79 MB | Voice: CC BY 4.0 · phonemes: uccharan | **CPU** · RTF 0.137 · first audio 309 ms · round-trip WER 0.0% |
| `piper-en-us-ljspeech-medium` | en | realtime | - | 64 MB | Voice: public domain · phonemes: uccharan | untested |
| `piper-en-us-joe-medium` | en | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-en-us-kristin-medium` | en | realtime | - | 64 MB | Voice: public domain · phonemes: uccharan | untested |
| `piper-en-gb-alan-medium` | en | realtime | - | 63 MB | See dataset licence: https://github.com/MycroftAI/mimic3-… · phonemes: uccharan | untested |
| `piper-en-gb-alba-medium` | en | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… · phonemes: uccharan | **CPU** · RTF 0.147 · first audio 380 ms · round-trip WER 15.0% |
| `piper-en-gb-cori-medium` | en | realtime | - | 64 MB | Voice: public domain · phonemes: uccharan | untested |
| `piper-en-gb-jenny-dioco-medium` | en | realtime | - | 63 MB | See dataset licence: https://github.com/dioco-group/jenny… · phonemes: uccharan | untested |
| `piper-en-gb-northern-english-male-medium` | en | realtime | - | 63 MB | Voice: CC-BY-SA 4.0 International · phonemes: uccharan | untested |
| `piper-en-gb-southern-english-female-medium` | en | realtime | - | 77 MB | See voice card · phonemes: uccharan | untested |
| `piper-de-de-thorsten-medium` | de | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | **CPU** · RTF 0.251 · first audio 187 ms |
| `piper-de-de-thorsten-high` | de | realtime | - | 114 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-de-de-kerstin-low` | de | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-es-es-davefx-medium` | es | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | **CPU** · RTF 0.173 · first audio 191 ms |
| `piper-es-es-sharvard-medium` | es | realtime | - | 77 MB | See dataset licence: http://creativecommons.org/licenses/… · phonemes: uccharan | untested |
| `piper-es-mx-ald-medium` | es | realtime | - | 63 MB | See dataset licence: http://unlicense.org · needs espeak-ng | untested |
| `piper-es-ar-daniela-high` | es | realtime | - | 114 MB | Voice: Attribution-ShareAlike 4.0 International · needs espeak-ng | untested |
| `piper-fr-fr-siwis-medium` | fr | realtime | - | 63 MB | Voice: CC-BY 4.0 · phonemes: uccharan | **CPU** · RTF 0.132 · first audio 225 ms |
| `piper-fr-fr-upmc-medium` | fr | realtime | - | 77 MB | See dataset licence: http://creativecommons.org/licenses/… · phonemes: uccharan | untested |
| `piper-fr-fr-tom-medium` | fr | realtime | - | 64 MB | Voice: AGPLv3 · phonemes: uccharan | untested |
| `piper-it-it-paola-medium` | it | realtime | - | 64 MB | See dataset licence: https://huggingface.co/datasets/paol… · phonemes: uccharan | **CPU** · RTF 0.173 · first audio 300 ms |
| `piper-it-it-riccardo-x-low` | it | realtime | - | 28 MB | See dataset licence: https://www.caito.de/2019/01/03/the-… · phonemes: uccharan | untested |
| `piper-pt-br-faber-medium` | pt | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | **CPU** · RTF 0.181 · first audio 287 ms |
| `piper-pt-pt-tugao-medium` | pt | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-nl-nl-mls-medium` | nl | realtime | - | 77 MB | Voice: CC-BY 4.0 · phonemes: uccharan | untested |
| `piper-nl-be-nathalie-medium` | nl | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-pl-pl-gosia-medium` | pl | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-pl-pl-darkman-medium` | pl | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-ru-ru-irina-medium` | ru | realtime | - | 63 MB | Voice: Unknown · needs espeak-ng | **CPU** · RTF 0.233 · first audio 381 ms |
| `piper-ru-ru-denis-medium` | ru | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ru-ru-dmitri-medium` | ru | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ru-ru-ruslan-medium` | ru | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-uk-ua-ukrainian-tts-medium` | uk | realtime | - | 77 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-tr-tr-dfki-medium` | tr | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · phonemes: uccharan | untested |
| `piper-tr-tr-fahrettin-medium` | tr | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-ar-jo-kareem-medium` | ar | realtime | - | 63 MB | See dataset licence: https://github.com/AliMokhammad/arab… · needs espeak-ng | untested |
| `piper-fa-ir-amir-medium` | fa | realtime | - | 64 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-fa-ir-gyro-medium` | fa | realtime | - | 63 MB | See dataset licence: https://github.com/gyroing · needs espeak-ng | untested |
| `piper-zh-cn-huayan-medium` | zh | realtime | - | 63 MB | Voice: Unknown · needs espeak-ng | **CPU** · RTF 0.047 · first audio 160 ms |
| `piper-vi-vn-vais1000-medium` | vi | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… · needs espeak-ng | untested |
| `piper-id-id-news-tts-medium` | id | realtime | - | 63 MB | See dataset licence: https://www.kaggle.com/code/mpwolke/… · needs espeak-ng | untested |
| `piper-sv-se-nst-medium` | sv | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-da-dk-talesyntese-medium` | da | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-no-no-talesyntese-medium` | no | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-fi-fi-harri-medium` | fi | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-cs-cz-jirka-medium` | cs | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-el-gr-rapunzelina-low` | el | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-hu-hu-anna-medium` | hu | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-ro-ro-mihai-medium` | ro | realtime | - | 63 MB | Voice: CC0-1.0 · phonemes: uccharan | untested |
| `piper-sk-sk-lili-medium` | sk | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-sr-rs-serbski-institut-medium` | sr | realtime | - | 77 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-sl-si-artur-medium` | sl | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… · needs espeak-ng | untested |
| `piper-ca-es-upc-ona-medium` | ca | realtime | - | 63 MB | Voice: CC BY-SA 3.0 ES · phonemes: uccharan | untested |
| `piper-cy-gb-gwryw-gogleddol-medium` | cy | realtime | - | 64 MB | See dataset licence: https://git.techiaith.bangor.ac.uk/d… · needs espeak-ng | untested |
| `piper-is-is-bui-medium` | is | realtime | - | 77 MB | See dataset licence: http://hdl.handle.net/20.500.12537/104 · needs espeak-ng | untested |
| `piper-ka-ge-natia-medium` | ka | realtime | - | 63 MB | See LICENSE file · needs espeak-ng | untested |
| `piper-kk-kz-issai-high` | kk | realtime | - | 128 MB | Voice: CC-BY-4.0 · needs espeak-ng | untested |
| `piper-lb-lu-marylux-medium` | lb | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · needs espeak-ng | untested |
| `piper-lv-lv-aivars-medium` | lv | realtime | - | 64 MB | Voice: CC0 1.0 Universal · needs espeak-ng | untested |
| `piper-sw-cd-lanfrica-medium` | sw | realtime | - | 63 MB | See dataset licence: https://lanfrica.com/record/kiswahil… · needs espeak-ng | untested |
| `piper-ne-np-chitwan-medium` | ne | realtime | - | 63 MB | Voice: CC0-1.0 (dataset, per the voice card) · phonemes: uccharan | **CPU** · RTF 0.070 · first audio 402 ms · round-trip WER 46.2% · round-trip CER 9.3% |
| `piper-te-in-padmavathi-medium` | te | realtime | - | 64 MB | Voice: CC-BY-4.0 (IndicVoices-R, per the voice card) · phonemes: uccharan | **CPU** · RTF 0.179 · first audio 1294 ms · round-trip WER 15.4% · round-trip CER 2.6% |
| `piper-te-in-venkatesh-medium` | te | realtime | - | 64 MB | Voice: CC-BY-4.0 (IndicVoices-R, per the voice card) · phonemes: uccharan | **CPU** · RTF 0.211 · first audio 1240 ms · round-trip WER 23.1% · round-trip CER 3.8% |
| `piper-te-in-maya-medium` | te | realtime | - | 63 MB | See dataset licence: https://www.iitm.ac.in/donlab/indict… · phonemes: uccharan | **CPU** · RTF 0.097 · first audio 596 ms · round-trip WER 7.7% · round-trip CER 1.3% |
| `piper-ml-in-arjun-medium` | ml | realtime | - | 63 MB | See dataset licence (IndicTTS Malayalam corpus; the voice… · phonemes: uccharan | **CPU** · RTF 0.191 · first audio 1642 ms · round-trip WER 10.0% · round-trip CER 2.2% |
| `piper-ml-in-meera-medium` | ml | realtime | - | 63 MB | See dataset licence (IndicTTS Malayalam corpus; the voice… · phonemes: uccharan | **CPU** · RTF 0.194 · first audio 1221 ms · round-trip WER 10.0% · round-trip CER 6.7% |
| `piper-mr-in-google-medium` | mr | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 (OpenSLR 64) · phonemes: uccharan | **CPU** · RTF 0.131 · first audio 932 ms · round-trip WER 33.3% · round-trip CER 6.1% |
| `piper-bn-bd-google-medium` | bn | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 (OpenSLR 37) and the CMU Indic licence · phonemes: uccharan | **CPU** · RTF 0.215 · first audio 1012 ms · round-trip WER 33.3% · round-trip CER 6.0% |
| `piper-ur-pk-aegis-female-medium` | ur | realtime | - | 64 MB | Voice: MIT (per the voice card) · phonemes: uccharan | **CPU** · RTF 0.137 · first audio 866 ms · round-trip WER 27.8% · round-trip CER 2.8% |
| `piper-ta-in-rasa-female-medium` | ta | realtime | - | 64 MB | Voice: CC-BY-4.0 (AI4Bharat Rasa data; third-party voice … · phonemes: uccharan | **CPU** · RTF 0.110 · first audio 487 ms · round-trip WER 0.0% · round-trip CER 0.0% |
| `piper-ta-in-rasa-male-medium` | ta | realtime | - | 64 MB | Voice: CC-BY-4.0 (AI4Bharat Rasa data; third-party voice … · phonemes: uccharan | **CPU** · RTF 0.147 · first audio 563 ms · round-trip WER 10.0% · round-trip CER 3.1% |

#### External TTS server: llama-server

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `chatterbox-multilingual-v3-llamacpp` (Chatterbox Multilingual V3) | en, ar, da, de, el, es +17 | accuracy | - | 1.2 GB | MIT · runs on llama-server · GPU recommended | untested |
| `chatterbox-turbo-llamacpp` (Chatterbox Turbo) | en | accuracy | - | 1.0 GB | MIT · runs on llama-server · GPU recommended | **CPU** · RTF 3.063 · first audio 17765 ms · round-trip WER 2.6% · cloned voice: speaker similarity 0.71 |
| `qwen3-tts-1.7b-base-llamacpp` (Qwen3-TTS 1.7B Base) | en, de, es, fr, it, ja +4 | accuracy | - | 1.5 GB | Apache-2.0 · runs on llama-server · GPU recommended | **CPU** · RTF 2.316 · first audio 12043 ms · round-trip WER 0.0% · cloned voice: speaker similarity 0.62 |
| `pocket-tts-en-llamacpp` (Kyutai Pocket TTS (English)) | en | balanced | - | 219 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.716 · first audio 774 ms · round-trip WER 0.0% · cloned voice: speaker similarity 0.57 |
| `pocket-tts-de-llamacpp` (Kyutai Pocket TTS (German)) | de | balanced | - | 219 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.447 · first audio 1439 ms · round-trip WER 8.8% · cloned voice: speaker similarity 0.36 |
| `pocket-tts-es-llamacpp` (Kyutai Pocket TTS (Spanish)) | es | balanced | - | 219 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.463 · first audio 1578 ms · round-trip WER 57.9% · cloned voice: speaker similarity 0.33 |
| `pocket-tts-fr-llamacpp` (Kyutai Pocket TTS (French)) | fr | balanced | - | 673 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.657 · first audio 1864 ms · round-trip WER 50.0% · cloned voice: speaker similarity 0.38 |
| `pocket-tts-it-llamacpp` (Kyutai Pocket TTS (Italian)) | it | balanced | - | 219 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.397 · first audio 1327 ms · round-trip WER 38.9% · cloned voice: speaker similarity 0.38 |
| `pocket-tts-pt-llamacpp` (Kyutai Pocket TTS (Portuguese)) | pt | balanced | - | 219 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.631 · first audio 2262 ms · round-trip WER 41.7% · cloned voice: speaker similarity 0.32 |
| `lfm3-audio-2b-llamacpp` (LFM3-Audio 2B) | en, ar, da, de, el, es +12 | accuracy | - | 1.4 GB | LFM Open License v1.0 ⚠ · runs on llama-server · GPU recommended | untested |
| `soprano-1.1-80m-llamacpp` (Soprano 1.1 80M) | en | balanced | - | 146 MB | Apache-2.0 · runs on llama-server | **CPU** · RTF 1.239 · first audio 6024 ms · round-trip WER 0.0% |
| `tiny-voca-llamacpp` (tiny-voca) | en | realtime | - | 26 MB | Apache-2.0 · runs on llama-server | ran once here (shared CPU), not measured |
| `kokoro-82m-llamacpp` (Kokoro (llama.cpp)) | en | balanced | - | 270 MB | Apache-2.0 · runs on llama-server | ran once here (shared CPU), not measured |
| `zonos3-mini-llamacpp` (ZONOS3-Mini (llama.cpp)) | en, de, es, fr, hi, it +5 | accuracy | - | 2.0 GB | Apache-2.0 · runs on llama-server · GPU recommended | ran once here (shared CPU), not measured |
| `fish-audio-s2-pro-llamacpp` (Fish Audio S2 Pro (llama.cpp)) | - | accuracy | - | 6.3 GB | Fish Audio Research License ⚠ · runs on llama-server · GPU recommended | ran once here (shared CPU), not measured |
| `fish-drama-3-llamacpp` (Fish Audio Drama 3) | - | accuracy | - | 23.5 GB | Fish Audio Community License 1.0 ⚠ · runs on llama-server · GPU required | untested |
| `pocket-tts-multilingual-llamacpp` (Multilingual Pocket TTS (community)) | en, hi, mr, kn, ta, ml +1 | balanced | - | 220 MB | CC-BY-4.0 · runs on llama-server | **CPU** · RTF 0.768 |

#### External TTS server: audio.cpp

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `chatterbox-multilingual-audiocpp` (Chatterbox Multilingual (V3 talker)) | en, ar, da, de, el, es +13 | accuracy | - | 2.1 GB | MIT · runs on audio.cpp · GPU recommended | untested |
| `chatterbox-turbo-audiocpp` (Chatterbox Turbo) | en | accuracy | - | 699 MB | MIT · runs on audio.cpp · GPU recommended | ran once here (shared CPU), not measured |
| `cosyvoice3-0.5b-audiocpp` (Fun-CosyVoice3 0.5B) | en, de, es, fr, it, ja +4 | accuracy | - | 2.3 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `voxcpm2-audiocpp` (VoxCPM2 2B) | en, ar, da, de, el, es +24 | accuracy | - | 3.0 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `magpie-tts-multilingual-357m-audiocpp` (NVIDIA Magpie TTS Multilingual 357M) | en, ar, de, es, fr, hi +5 | accuracy | - | 1.6 GB | NVIDIA Open Model License · runs on audio.cpp · GPU recommended | untested |
| `maya1-audiocpp` (Maya1 3B) | en | accuracy | - | 4.0 GB | Apache-2.0 AND MIT · runs on audio.cpp · GPU required | untested |
| `qwen3-tts-0.6b-base-audiocpp` (Qwen3-TTS 0.6B Base) | en, de, es, fr, it, ja +4 | accuracy | - | 2.0 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | **CPU** · RTF 3.237 · first audio 17611 ms · round-trip WER 0.0% · cloned voice: speaker similarity 0.72 |
| `qwen3-tts-1.7b-customvoice-audiocpp` (Qwen3-TTS 1.7B CustomVoice) | en, de, es, fr, it, ja +4 | accuracy | - | 2.8 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `cilla-lite-audiocpp` (Cilla (Persian)) | fa | balanced | - | 439 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | untested |
| `cilla-mini-audiocpp` (Cilla (Persian)) | fa | realtime | - | 113 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | untested |
| `pocket-tts-en-audiocpp` (Kyutai Pocket TTS (English)) | en | balanced | - | 134 MB | CC-BY-4.0 · runs on audio.cpp | **CPU** · RTF 9.713 · first audio 45069 ms · round-trip WER 0.0% · cloned voice: speaker similarity 0.57 |
| `soprano-1.1-80m-audiocpp` (Soprano 1.1 80M) | en | balanced | - | 123 MB | Apache-2.0 · runs on audio.cpp | untested |
| `supertonic-3-audiocpp` (Supertonic 3) | en, ar, bg, cs, da, de +25 | accuracy | - | 454 MB | BigScience OpenRAIL-M · runs on audio.cpp · GPU recommended | untested |
| `kitten-tts-2-audiocpp` (Kitten TTS 2) | en, ar, de, es, fr, hi +10 | accuracy | - | 3.3 GB | Stellon Labs Community License ⚠ · runs on audio.cpp · GPU recommended | untested |
| `moss-tts-local-v1.5-audiocpp` (MOSS-TTS Local Transformer v1.5) | en, ar, cs, da, de, el +25 | accuracy | - | 7.5 GB | Apache-2.0 · runs on audio.cpp · GPU required | untested |
| `moss-tts-nano-100m-audiocpp` (MOSS-TTS Nano 100M) | en, ar, cs, da, de, el +13 | balanced | - | 193 MB | Apache-2.0 · runs on audio.cpp | **CPU** · RTF 1.051 · first audio 5969 ms · round-trip WER 5.3% · cloned voice: speaker similarity 0.65 |
| `omnivoice-audiocpp` (OmniVoice) | - | accuracy | - | 1.4 GB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · GPU recommended | **CPU** · RTF 37.849 · first audio 210438 ms · round-trip WER 55.2% · cloned voice: speaker similarity 0.12 |
| `voicetut-tts-audiocpp` (VoiceTut-TTS) | ar, en | accuracy | - | 1.4 GB | Apache-2.0 (model card); audio tokenizer under the Boson … ⚠ · runs on audio.cpp · GPU recommended | untested |
| `voxtral-4b-tts-audiocpp` (Voxtral 4B TTS) | en, ar, de, es, fr, hi +3 | accuracy | - | 8.0 GB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · GPU required | untested |
| `fish-audio-s2-pro-audiocpp` (Fish Audio S2 Pro) | - | accuracy | - | 6.3 GB | Fish Audio Research License ⚠ · runs on audio.cpp · GPU required | untested |
| `fish-drama-3-audiocpp` (Fish Audio Drama 3) | - | accuracy | - | 24.6 GB | Fish Audio Community License v1.0 ⚠ · runs on audio.cpp · GPU required | untested |
| `higgs-tts-3-4b-audiocpp` (Higgs TTS 3 4B) | - | accuracy | - | 5.1 GB | Boson Higgs TTS 3 Research and Non-Commercial License ⚠ · runs on audio.cpp · GPU required | untested |
| `indextts-2-audiocpp` (IndexTTS-2) | en, ar, es, ja, zh | accuracy | - | 3.6 GB | bilibili Model Use License ⚠ · runs on audio.cpp · GPU recommended | untested |
| `vibevoice-1.5b-audiocpp` (VibeVoice 1.5B) | en, zh | accuracy | - | 3.2 GB | MIT · runs on audio.cpp · GPU recommended | untested |
| `xtts-v2-audiocpp` (Coqui XTTS v2) | en, ar, cs, de, es, fr +11 | accuracy | - | 568 MB | Coqui Public Model License 1.0.0 ⚠ · runs on audio.cpp · GPU recommended | **CPU** · RTF 1.356 · first audio 6348 ms · round-trip WER 0.0% · cloned voice: speaker similarity 0.61 |
| `outetts-1.0-1b-audiocpp` (OuteTTS 1.0 1B) | en, ar, be, bn, de, es +17 | accuracy | - | 3.0 GB | CC-BY-NC-SA-4.0 ⚠ · runs on audio.cpp · GPU recommended | untested |
| `neutts-2e-audiocpp` (NeuTTS 2E) | en | accuracy | - | 3.0 GB | NeuTTS Open License v1.0 ⚠ · runs on audio.cpp · GPU recommended | untested |
| `sanotts-hi-audiocpp` (sanoTTS (Hindi voice)) | hi | balanced | - | 6 MB | GPL-3.0 · runs on audio.cpp | untested |
| `fireredtts3-instruct-audiocpp` (FireRedTTS3 Instruct) | en, ar, cs, de, el, es +18 | accuracy | - | 4.1 GB | Apache-2.0 · runs on audio.cpp · GPU required | untested |
| `ming-3.0-tts-0.5b-audiocpp` (Ming 3.0 TTS 0.5B) | en, de, es, fr, ja, ko +1 | accuracy | - | 3.0 GB | MIT · runs on audio.cpp · GPU recommended | untested |
| `breeze-tts-2-audiocpp` (BreezeTTS 2) | en, zh | accuracy | - | 5.1 GB | BreezeBlue Research and Non-Commercial License ⚠ · runs on audio.cpp · GPU required | untested |
| `kugelaudio-0-open-audiocpp` (KugelAudio-0-Open) | en, bg, cs, da, de, el +17 | accuracy | - | 9.8 GB | MIT · runs on audio.cpp · GPU required | untested |
| `zonos3-mini-audiocpp` (ZONOS3-Mini) | en, de, es, fr, it, ja +3 | accuracy | - | 1.2 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `dots-tts-soar-audiocpp` (dots.tts SOAR) | - | accuracy | - | 3.0 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `irodori-tts-audiocpp` (Irodori-TTS) | ja | accuracy | - | 1.4 GB | MIT · runs on audio.cpp · GPU recommended | untested |
| `miotts-1.7b-audiocpp` (MioTTS 1.7B) | en, ja | accuracy | - | 2.2 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `gradium-phonon-31m-audiocpp` (Gradium Phonon 31M) | en | balanced | - | 95 MB | Apache-2.0 · runs on audio.cpp | **CPU** · RTF 0.952 · round-trip WER 0.0% |
| `gradium-phonon-40m-multilingual-audiocpp` (Gradium Phonon 40M) | en, de, es, fr, pt | balanced | - | 148 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | **CPU** · RTF 2.190 |
| `gradium-phonon-40m-en-audiocpp` (Gradium Phonon 40M) | en | balanced | - | 132 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | **CPU** · RTF 2.620 |
| `gradium-phonon-40m-fr-audiocpp` (Gradium Phonon 40M) | fr | balanced | - | 126 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | untested |
| `gradium-phonon-40m-de-audiocpp` (Gradium Phonon 40M) | de | balanced | - | 125 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | **CPU** · RTF 0.674 |
| `gradium-phonon-40m-es-audiocpp` (Gradium Phonon 40M) | es | balanced | - | 128 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | untested |
| `gradium-phonon-40m-pt-audiocpp` (Gradium Phonon 40M) | pt | balanced | - | 125 MB | CC-BY-NC-4.0 ⚠ · runs on audio.cpp · gated | untested |
| `audio8-tts-0.6b-audiocpp` (Audio8 TTS Preview 0.6B) | en, de, es, fr, it, ja +5 | accuracy | - | 1.4 GB | Apache-2.0 · runs on audio.cpp · GPU recommended | untested |
| `confucius4-tts-audiocpp` (Confucius4-TTS) | en, de, es, fr, id, it +8 | accuracy | - | 8.2 GB | Apache-2.0 · runs on audio.cpp · GPU required | untested |
| `plapre-3-audiocpp` (Plapre 3 (Danish)) | da | accuracy | - | 1.1 GB | CC-BY-4.0 · runs on audio.cpp · GPU recommended | **CPU** · RTF 3.487 · first audio 29568 ms · cloned voice: speaker similarity 0.50 |
| `echo-tts-audiocpp` (Echo-TTS) | en | accuracy | - | 3.0 GB | CC-BY-NC-SA-4.0 ⚠ · runs on audio.cpp · GPU recommended | untested |
| `lfm3-audio-2b-audiocpp` (LFM3-Audio 2B) | en, ar, da, de, el, es +12 | accuracy | - | 2.2 GB | LFM Open License v1.0 ⚠ · runs on audio.cpp · GPU recommended | untested |
| `indextts-2.5-audiocpp` (IndexTTS-2.5) | en, ar, es, ja, zh | accuracy | - | 3.5 GB | bilibili Model Use License ⚠ · runs on audio.cpp · GPU recommended | untested |

#### External TTS server: tontaube

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `tontaube-v1.1-int4` (TontaubeV1.1) | en, de, es, fr, it, nl +1 | accuracy | - | 1.8 GB | Tontaube Community Model License 1.0 ⚠ · runs on tontaube · GPU recommended | untested |
| `tontaube-v1.1` (TontaubeV1.1) | en, de, es, fr, it, nl +1 | accuracy | - | 6.4 GB | Tontaube Community Model License 1.0 ⚠ · runs on tontaube · GPU required | untested |

#### KittenTTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kitten-nano-0.8-int8` | en | realtime | - | 28 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.266 · first audio 977 ms · round-trip WER 0.0% |
| `kitten-nano-0.8` | en | realtime | - | 60 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.107 · first audio 396 ms · round-trip WER 0.0% |
| `kitten-micro-0.8` | en | realtime | - | 45 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.319 · first audio 1329 ms · round-trip WER 0.0% |
| `kitten-mini-0.8` | en | realtime | - | 82 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 1.112 · first audio 3312 ms · round-trip WER 0.0% |

#### Matcha-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `matcha-en-ljspeech` | en | balanced | - | 78 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.205 · first audio 761 ms · round-trip WER 0.0% |
| `matcha-zh-baker` | zh | balanced | - | 96 MB | Research only (Baker dataset: non-commercial) ⚠ | untested |

#### MeloTTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `melo-tts-en` | en | balanced | - | 175 MB | MIT | **CPU** · RTF 0.278 · first audio 768 ms · round-trip WER 30.0% |

#### VITS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `coqui-bn-female` | bn | balanced | - | 114 MB | Apache-2.0 | **CPU** · RTF 0.574 · first audio 1119 ms |
| `mimic3-gu-cmu-indic` | gu | balanced | - | 76 MB | See voice card (CMU Indic dataset) · phonemes: uccharan | **CPU** · RTF 0.046 · first audio 82 ms |
| `mimic3-bn-multi` | bn | balanced | - | 76 MB | See voice card (OpenSLR Bengali, CC-BY-SA-4.0) · phonemes: uccharan | **CPU** · RTF 0.139 · first audio 282 ms |
| `mimic3-ne-google` | ne | balanced | - | 76 MB | See voice card (OpenSLR Nepali, CC-BY-SA-4.0) · phonemes: uccharan | **CPU** · RTF 0.147 · first audio 248 ms |
| `syspin-mai-female` | mai | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-mai-male` | mai | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-bho-female` | bho | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-bho-male` | bho | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-hi-female` | hi | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-hi-male` | hi | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-bn-female` | bn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-bn-male` | bn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-mr-female` | mr | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-mr-male` | mr | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-te-female` | te | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-te-male` | te | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-kn-female` | kn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-kn-male` | kn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-gu-female` | gu | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-gu-male` | gu | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-mag-female` | mag | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-mag-male` | mag | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-hne-female` | hne | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-hne-male` | hne | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-en-female` | en | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |
| `syspin-en-male` | en | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | untested |

#### tiny-voca

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny-voca-onnx` | en | realtime | - | 27 MB | Apache-2.0 | untested |

#### little-tts

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `little-tts-onnx` | en | realtime | - | 25 MB | CC-BY-NC-4.0 ⚠ | **CPU** · RTF 0.025 · round-trip WER 0.0% |

#### Paradee

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `paradee-8m-onnx` | en | realtime | - | 8 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.084 · round-trip WER 0.0% |

#### PiVo

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `pivo-1.4m-onnx` | en | realtime | - | 9 MB | Apache-2.0 | **CPU** · RTF 0.659 · round-trip WER 0.0% |

#### kale-tts-1

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kale-tts-1-nano-onnx` | en | realtime | - | 29 MB | Apache-2.0 | **CPU** · RTF 0.028 · round-trip WER 0.0% |

#### Kitten TTS 2 Pico

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kitten-tts-2-pico-onnx` | en, de, es, it, pt, ru +2 | realtime | - | 79 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.124 · round-trip WER 0.0% |

#### Kitten TTS 2 Mini

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kitten-tts-2-mini-onnx` | en, de, es, it, pt, ru +9 | realtime | - | 142 MB | Apache-2.0 · phonemes: uccharan | **CPU** · RTF 0.608 · round-trip WER 0.0% |

### Voice-activity detection

#### Silero VAD

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `silero-vad` | - | realtime | - | 2 MB | MIT | **CPU** · RTF 0.006 |
| `silero-vad-v5` | - | realtime | - | 2 MB | MIT | **CPU** · RTF 0.006 |
| `silero-vad-v6` | - | realtime | - | 2 MB | MIT | **CPU** · RTF 0.013 |

#### TEN VAD

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `ten-vad` | - | realtime | - | 0 MB | Apache-2.0 with Agora's additional conditions (own applic… ⚠ | **CPU** · RTF 0.009 |
| `ten-vad-int8` | - | realtime | - | 0 MB | Apache-2.0 with Agora's additional conditions (own applic… ⚠ | **CPU** · RTF 0.017 |

#### MarbleNet VAD

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `marblenet-vad` | - | realtime | - | 0 MB | NVIDIA Open Model License | **CPU** · RTF 0.069 |

#### WebRTC VAD

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `webrtc-vad` | - | realtime | - | 0 MB | BSD-3-Clause (WebRTC) + MIT (webrtcvad-wheels binding) | **CPU** · RTF 0.000 |
| `webrtc-vad-aggressive` | - | realtime | - | 0 MB | BSD-3-Clause (WebRTC) + MIT (webrtcvad-wheels binding) | **CPU** · RTF 0.000 |

#### Energy VAD

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `energy-vad` | - | realtime | - | 0 MB | Apache-2.0 | **CPU** · RTF 0.000 |

### Speaker diarization

#### pyannote + speaker embeddings

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `pyannote-3.0-eres2net` | - | balanced | - | 32 MB | MIT (segmentation) + Apache-2.0 (embedding) | **CPU** · RTF 0.106 · DER 5.2% on a two-voice test clip |
| `pyannote-3.0-titanet-small` | - | balanced | - | 46 MB | MIT (segmentation) + CC-BY-4.0 (embedding) | untested |
| `pyannote-3.0-wespeaker-resnet34` | - | balanced | - | 33 MB | MIT (segmentation) + CC-BY-4.0 (embedding) | untested |
| `reverb-diarization-v1` | - | balanced | - | 36 MB | Rev Model Non-Production License (segmentation, non-comme… ⚠ | **CPU** · RTF 0.258 · DER 5.0% on a two-voice test clip |

#### Nemotron diarization

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3-diarization` | - | balanced | - | 147 MB | OpenMDW-1.1 | **CPU** · RTF 0.039 · DER 24.1% on AMI meeting excerpts |

#### pyannote (PyTorch)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `pyannote-speaker-diarization-3.1` | - | balanced | - | 33 MB | MIT (pipeline, segmentation-3.0) + CC-BY-4.0 (WeSpeaker e… · needs plugin torch · gated | **CPU** · RTF 0.906 · DER 6.0% on a two-voice test clip |
| `pyannote-speaker-diarization-community-1` | - | balanced | - | 33 MB | CC-BY-4.0 · needs plugin torch · gated | **CPU** · RTF 0.876 · DER 6.0% on a two-voice test clip |
| `reverb-diarization-v2-torch` | - | accuracy | - | 450 MB | Rev Model Non-Production License (segmentation, non-comme… ⚠ · needs plugin torch · gated | **CPU** · RTF 16.160 · DER 61.5% on a two-voice test clip |

#### DiariZen

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `diarizen-wavlm-large-s80-md-v2` | - | accuracy | - | 278 MB | CC-BY-NC-4.0 (segmentation) + CC-BY-4.0 (PLDA) ⚠ · needs plugin torch | **CPU** · RTF 12.400 · DER 5.0% on a two-voice test clip |

### End-of-turn detection

#### Smart Turn

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `smart-turn-v3` | ar, bn, da, de, en, es +17 | realtime | - | 9 MB | BSD-2-Clause | **CPU** · RTF 0.059 |
| `smart-turn-v3.1` | ar, bn, da, de, en, es +17 | realtime | - | 9 MB | BSD-2-Clause | **CPU** · RTF 0.070 |
| `smart-turn-v3.0` | ar, bn, da, de, en, es +17 | realtime | - | 9 MB | BSD-2-Clause | **CPU** · RTF 0.043 |
| `smart-turn-v3-fp32` | ar, bn, da, de, en, es +17 | realtime | - | 32 MB | BSD-2-Clause | **CPU** · RTF 0.064 |
| `smart-turn-v2` | en, de, es, fr, hi, it +8 | balanced | - | 379 MB | BSD-2-Clause | **CPU** · RTF 1.588 |

#### Namo Turn Detector

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `namo-turn-en` | en | realtime | - | 139 MB | Apache-2.0 | **CPU** · RTF 0.011 |
| `namo-turn-multilingual` | ar, bn, da, de, en, es +17 | balanced | - | 343 MB | Apache-2.0 | **CPU** · RTF 0.024 |

#### Silence only

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `silence-only` | - | realtime | - | 0 MB | Apache-2.0 | **CPU** · RTF 0.000 |

#### TEN Turn Detection

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `ten-turn-detection` (TEN Turn Detection) | en, zh | accuracy | - | 4.7 GB | Apache-2.0 with Agora's additional conditions (not on end… ⚠ · runs on llama-server · GPU recommended | untested |

### Overlapped-speech separation

#### SepFormer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sepformer-whamr16k` | en | accuracy | - | 113 MB | Apache-2.0 · needs plugin torch | **CPU** · RTF 3.690 |
| `sepformer-whamr` | en | balanced | - | 113 MB | Apache-2.0 · needs plugin torch | **CPU** · RTF 1.440 |
| `sepformer-wsj02mix` | en | balanced | - | 106 MB | Apache-2.0 | **CPU** · RTF 3.507 |
| `sepformer-wsj03mix` | en | balanced | - | 106 MB | Apache-2.0 | **CPU** · RTF 3.597 |
| `sepformer-libri2mix` | en | accuracy | - | 106 MB | Apache-2.0 | **CPU** · RTF 1.860 |
| `sepformer-libri3mix` | en | accuracy | - | 106 MB | Apache-2.0 | **CPU** · RTF 3.823 |

#### PixIT

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `pixit-ami` | en | accuracy | - | 2.6 GB | MIT · needs plugin torch · gated | **CPU** · RTF 5.690 |

#### VoxLock

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `voxlock-int8` | en | balanced | - | 81 MB | Apache-2.0 | **CPU** · RTF 0.460 |

### Denoising

#### ElevenLabs (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `elevenlabs-voice-isolator` | - | accuracy | - | none (cloud API) | ElevenLabs Terms of Use (paid cloud API) · [terms](https://elevenlabs.io/terms-of-use) · needs provider `elevenlabs` (key: `BOLO_PROVIDER_ELEVENLABS_API_KEY`) | untested |

#### GTCRN

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `gtcrn-simple` | - | realtime | - | 1 MB | MIT | **CPU** · RTF 0.060 |

#### DPDFNet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `dpdfnet-baseline` | - | realtime | - | 9 MB | Apache-2.0 | **CPU** · RTF 0.080 |
| `dpdfnet2` | - | realtime | - | 10 MB | Apache-2.0 | **CPU** · RTF 0.163 |
| `dpdfnet4` | - | balanced | - | 12 MB | Apache-2.0 | **CPU** · RTF 0.232 |
| `dpdfnet8` | - | accuracy | - | 15 MB | Apache-2.0 | **CPU** · RTF 0.381 |
| `dpdfnet2-48k` | - | balanced | - | 10 MB | Apache-2.0 | **CPU** · RTF 0.214 |
| `dpdfnet8-48k` | - | accuracy | - | 15 MB | Apache-2.0 | **CPU** · RTF 0.518 |
| `dpdfnet2-8k` | - | realtime | - | 10 MB | Apache-2.0 | **CPU** · RTF 0.148 |
| `dpdfnet8-8k` | - | balanced | - | 15 MB | Apache-2.0 | **CPU** · RTF 0.344 |

#### DeepFilterNet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `deepfilternet3` | - | balanced | - | 16 MB | MIT OR Apache-2.0 | **CPU** · RTF 0.259 |

#### RNNoise

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `rnnoise` | - | realtime | - | 13 MB | BSD-3-Clause | **CPU** · RTF 0.068 |

#### Demucs denoiser

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `demucs-dns48` | - | realtime | - | 75 MB | CC-BY-NC-4.0 ⚠ · needs plugin torch | **CPU** · RTF 0.194 |
| `demucs-dns64` | - | balanced | - | 134 MB | CC-BY-NC-4.0 ⚠ · needs plugin torch | **CPU** · RTF 0.294 |
| `demucs-master64` | - | balanced | - | 134 MB | CC-BY-NC-4.0 ⚠ · needs plugin torch | **CPU** · RTF 0.285 |
| `demucs-valentini-nc` | - | accuracy | - | 243 MB | CC-BY-NC-4.0 ⚠ · needs plugin torch | **CPU** · RTF 3.713 |

#### SepFormer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sepformer-dns4-16k` | - | accuracy | - | 106 MB | Apache-2.0 | **CPU** · RTF 3.783 |

#### MetricGAN+

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `metricgan-plus` | - | balanced | - | 8 MB | Apache-2.0 | **CPU** · RTF 0.006 |

### Language identification

#### VoxLingua107 ECAPA-TDNN

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `voxlingua107-ecapa` | ab, af, am, ar, as, az +101 | realtime | - | 87 MB | Apache-2.0 | **CPU** · RTF 0.106 |
| `voxlingua107-ecapa-int8` | ab, af, am, ar, as, az +101 | realtime | - | 22 MB | Apache-2.0 | **CPU** · RTF 0.135 |

#### Whisper

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `whisper-tiny-lid` | sq, ml, de, hi, uz, zh +93 | realtime | - | 103 MB | MIT | **CPU** · RTF 0.301 |
| `whisper-base-lid` | sq, ml, de, hi, uz, zh +93 | balanced | - | 160 MB | MIT | **CPU** · RTF 0.608 |

#### AmberNet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `ambernet-langid` | ab, af, am, ar, as, az +101 | realtime | - | 117 MB | NGC-Terms-of-Use ⚠ | **CPU** · RTF 0.104 |

#### MMS-LID

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mms-lid-126` | ar, zh, en, es, fr, mg +120 | accuracy | - | 974 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-lid-4017` | ar, zh, en, es, fr, mg +4010 | accuracy | - | 978 MB | CC-BY-NC-4.0 ⚠ | untested |

#### Voxlect

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `voxlect-indic-lid-whisper-small` | as, bn, brx, doi, en, gu +17 | balanced | - | 362 MB | OpenRAIL · needs plugin torch | **CPU** · RTF 0.558 |
| `voxlect-indic-lid-mms-lid-256` | as, bn, brx, doi, en, gu +17 | accuracy | - | 3.9 GB | CC-BY-NC-4.0 ⚠ · needs plugin torch | untested |

#### SpeechBrain ECAPA-TDNN

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `lid-indic-vaani` | en, hi, mr, ta, te | realtime | - | 83 MB | Apache-2.0 · needs plugin torch | **CPU** · RTF 0.130 |

#### mHuBERT-147 SLID

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-slid-mhubert` | as, bn, brx, doi, gu, hi +16 | balanced | - | 378 MB | CC-BY-NC-SA-4.0 ⚠ · needs plugin torch | **CPU** · RTF 0.230 |
| `indic-slid-mhubert-task2` | as, bn, brx, doi, gu, hi +16 | balanced | - | 378 MB | CC-BY-NC-SA-4.0 ⚠ · needs plugin torch | **CPU** · RTF 0.226 |

### Speech to speech (Realtime speech-to-speech pipeline)

#### xAI (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `xai-grok-voice-3.0` | en, ar, bn, zh, fr, de +10 | realtime | native | none (cloud API) | xAI Terms of Service (paid cloud API) · [terms](https://x.ai/legal/terms-of-service) · needs provider `xai` (key: `BOLO_PROVIDER_XAI_API_KEY`) | untested |

#### Rime (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `rime-prism` | en | realtime | native | none (cloud API) | Rime Terms of Service (paid cloud API) · [terms](https://www.rime.ai/terms) · needs provider `rime` (key: `BOLO_PROVIDER_RIME_API_KEY`) | untested |

#### Amazon Nova (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `amazon-nova-2.5-sonic` | en, fr, it, de, es, pt +1 | realtime | native | none (cloud API) | AWS Service Terms (paid cloud API) · [terms](https://aws.amazon.com/service-terms/) · needs provider `bedrock` (key: `BOLO_PROVIDER_BEDROCK_API_KEY`) | untested |

#### Moshi

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `moshiko-pytorch-bf16` | en | realtime | - | 15.8 GB | CC-BY-4.0 | untested |
| `moshiko-pytorch-q8` | en | realtime | - | 8.4 GB | CC-BY-4.0 | untested |
| `moshiko-mlx-q4` | en | realtime | - | 5.2 GB | CC-BY-4.0 | untested |
| `moshiko-mlx-q8` | en | realtime | - | 8.6 GB | CC-BY-4.0 | untested |
| `moshiko-mlx-bf16` | en | realtime | - | 15.8 GB | CC-BY-4.0 | untested |
| `moshiko-candle-q8` | en | realtime | - | 8.6 GB | CC-BY-4.0 | untested |
| `moshiko-candle-bf16` | en | realtime | - | 15.8 GB | CC-BY-4.0 | untested |
| `moshika-pytorch-bf16` | en | realtime | - | 15.8 GB | CC-BY-4.0 | untested |
| `moshika-pytorch-q8` | en | realtime | - | 8.4 GB | CC-BY-4.0 | untested |
| `moshika-mlx-q4` | en | realtime | - | 5.2 GB | CC-BY-4.0 | untested |
| `moshika-mlx-q8` | en | realtime | - | 8.6 GB | CC-BY-4.0 | untested |
| `moshika-mlx-bf16` | en | realtime | - | 15.8 GB | CC-BY-4.0 | untested |
| `moshika-candle-q8` | en | realtime | - | 8.6 GB | CC-BY-4.0 | untested |
| `moshika-candle-bf16` | en | realtime | - | 15.8 GB | CC-BY-4.0 | untested |

#### Hibiki

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `hibiki-1b-pytorch-bf16` | fr, en | realtime | - | 4.0 GB | CC-BY-4.0 | untested |
| `hibiki-2b-pytorch-bf16` | fr, en | realtime | - | 6.0 GB | CC-BY-4.0 | untested |
| `hibiki-1b-mlx-bf16` | fr, en | realtime | - | 4.0 GB | CC-BY-4.0 | untested |
| `hibiki-2b-mlx-bf16` | fr, en | realtime | - | 6.6 GB | CC-BY-4.0 | untested |

#### Hibiki-Zero

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `hibiki-zero-3b-pytorch-bf16` | fr, es, pt, de, en | realtime | - | 6.6 GB | CC-BY-NC-SA-4.0 ⚠ | untested |

#### PersonaPlex

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `personaplex-7b-v1` | en | realtime | - | 17.1 GB | NVIDIA Open Model License · gated | untested |
| `personaplex-rl-seamless` | en | realtime | - | 17.1 GB | CC-BY-NC-4.0 and NVIDIA Open Model License ⚠ · gated | untested |

#### FLM-Audio

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `flm-audio` | en, zh | realtime | - | 16.8 GB | No licence stated (model card: research use only; commerc… ⚠ | untested |

#### Freeze-Omni

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `freeze-omni` | en, zh | realtime | - | 17.6 GB | Freeze-Omni License (academic, research and education onl… ⚠ | untested |

#### MiniCPM-o 4.5

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `minicpm-o-4.5` | en, zh | realtime | - | 20.0 GB | Apache-2.0 | untested |

#### Qwen2.5-Omni 3B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-omni-3b-torch` | en, zh | balanced | - | 12.0 GB | Qwen Research License ⚠ · needs plugin torch | untested |

#### Qwen2.5-Omni 7B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-omni-7b-torch` | en, zh | accuracy | - | 22.4 GB | Apache-2.0 · needs plugin torch | untested |

#### Qwen3-Omni 30B-A3B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-omni-30b-a3b-instruct-torch` | en, zh, fr, de, ru, it +4 | accuracy | - | 70.5 GB | Apache-2.0 · needs plugin torch | untested |

#### GLM-4-Voice

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `glm-4-voice-9b` | zh, en | accuracy | - | 21.1 GB | glm-4-voice License ⚠ · needs plugin torch | untested |

#### MiniCPM-o 2.6

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `minicpm-o-2.6` | en, zh | accuracy | - | 17.4 GB | Apache-2.0 · needs plugin torch | untested |

#### Step-Audio 2

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `step-audio-2-mini` | en, zh | accuracy | - | 17.9 GB | Apache-2.0 · needs plugin torch | untested |

#### Kimi-Audio

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kimi-audio-7b-instruct` | en, zh | accuracy | - | 44.1 GB | MIT · needs plugin torch | untested |

### Audio-in language models (Realtime audio-llm pipeline)

#### LFM2.5-Audio 1.5B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `lfm2.5-audio-1.5b-llamacpp` (LFM2.5-Audio 1.5B) | en | realtime | - | 915 MB | LFM Open License v1.0 ⚠ · runs on llama-server | ran once here (shared CPU), not measured |

#### Ultravox 0.5

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `ultravox-0.5-llama-3.2-1b-llamacpp` (Ultravox 0.5) | en, ar, de, es, fr, hi +9 | realtime | - | 2.2 GB | MIT (adapter) + Llama 3.2 Community License · runs on llama-server | untested |

#### Qwen2.5-Omni 3B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-omni-3b-llamacpp` (Qwen2.5-Omni 3B) | en, zh | balanced | - | 3.6 GB | Qwen Research License ⚠ · runs on llama-server · GPU recommended | untested |

#### Qwen2.5-Omni 7B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-omni-7b-llamacpp` (Qwen2.5-Omni 7B) | en, zh | accuracy | - | 6.2 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |

#### Qwen3-Omni 30B-A3B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-omni-30b-a3b-instruct-llamacpp` (Qwen3-Omni 30B-A3B) | en, zh, ko, ja, de, ru +12 | accuracy | - | 19.9 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |

#### Gemma 4 E2B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma-4-e2b-it-llamacpp` (Gemma 4 E2B) | en | balanced | - | 3.4 GB | Apache-2.0 · runs on llama-server | untested |

#### Gemma 4 E4B

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma-4-e4b-it-llamacpp` (Gemma 4 E4B) | en | accuracy | - | 5.2 GB | Apache-2.0 · runs on llama-server · GPU recommended | untested |

### Speaker embedding

#### ERes2Net

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `eres2net-voxceleb` | - | realtime | - | 26 MB | Apache-2.0 | **CPU** · RTF 0.259 |

#### CAM++

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `campplus-voxceleb` | - | realtime | - | 30 MB | Apache-2.0 | **CPU** · RTF 0.130 |
| `campplus-zh-en` | - | realtime | - | 28 MB | Apache-2.0 | **CPU** · RTF 0.122 |
| `wespeaker-campplus-voxceleb` | - | realtime | - | 29 MB | CC-BY-4.0 | **CPU** · RTF 0.116 |

#### WeSpeaker ResNet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `wespeaker-resnet34-voxceleb` | - | realtime | - | 27 MB | CC-BY-4.0 | **CPU** · RTF 0.126 |

#### TitaNet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `titanet-small` | - | realtime | - | 40 MB | CC-BY-4.0 | **CPU** · RTF 0.068 |

#### ECAPA-TDNN

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `ecapa-tdnn-voxceleb` | - | realtime | - | 42 MB | Apache-2.0 | **CPU** · RTF 0.203 |

### Audio tagging and sound events

#### CED

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `ced-tiny` | - | realtime | - | 6 MB | Apache-2.0 | **CPU** · RTF 0.034 |
| `ced-mini` | - | realtime | - | 10 MB | Apache-2.0 | **CPU** · RTF 0.039 |
| `ced-small` | - | balanced | - | 23 MB | Apache-2.0 | **CPU** · RTF 0.052 |
| `ced-base` | - | accuracy | - | 87 MB | Apache-2.0 | **CPU** · RTF 0.105 |

#### Zipformer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `zipformer-audio-tagging` | - | balanced | - | 68 MB | Apache-2.0 | **CPU** · RTF 0.201 |
| `zipformer-small-audio-tagging` | - | realtime | - | 27 MB | Apache-2.0 | **CPU** · RTF 0.126 |

### Zero-shot audio classification (CLAP)

#### LAION CLAP

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `clap-htsat-unfused` | en | realtime | - | 542 MB | Apache-2.0 | **CPU** · RTF 0.262 |
| `larger-clap-general` | en | balanced | - | 586 MB | Apache-2.0 | **CPU** · RTF 0.380 |
| `larger-clap-music-and-speech` | en | balanced | - | 586 MB | Apache-2.0 | **CPU** · RTF 0.405 |

#### MS-CLAP

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `msclap-2023` | en | realtime | - | 548 MB | MS-PL | **CPU** · RTF 0.295 |

### Speech emotion

#### emotion2vec+

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `emotion2vec-plus-base` | - | balanced | - | 373 MB | FunASR Model License 1.1 (custom) | **CPU** · RTF 0.925 |
| `emotion2vec-plus-large` | - | accuracy | - | 649 MB | FunASR Model License 1.1 (custom) | **CPU** · RTF 0.845 |

### Decisions (typed questions about a recording or a text)

#### Gemini Decision (cloud)

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `gemini-decision-1` | en | realtime | - | none (cloud API) | Gemini API Additional Terms of Service (paid cloud API) · [terms](https://ai.google.dev/gemini-api/terms) · needs provider `gemini` (key: `BOLO_PROVIDER_GEMINI_API_KEY`) | untested |

#### d1-omni

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `d1-omni-600m-llamacpp` (d1-omni) | en | realtime | - | 670 MB | LFM Open License v1.0 ⚠ · runs on llama-server | ran once here (shared CPU), not measured |

#### Clef-Omni

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `clef-omni` | en | accuracy | - | 70.8 GB | Apache-2.0 · needs plugin torch | untested |

### Spoken intent (spoken language understanding)

#### SpeechBrain SLU

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `slu-slurp-hubert` | en | accuracy | - | 411 MB | Apache-2.0 · needs plugin torch | **CPU** · RTF 10.580 |

### Voice conversion

#### OpenVoice v2 tone colour converter

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `openvoice-v2-tone-color-audiocpp` (OpenVoice v2 tone colour converter) | - | realtime | - | 66 MB | MIT · runs on audio.cpp | **CPU** · RTF 2.930 |

#### MeanVC2

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `meanvc2-audiocpp` (MeanVC2) | - | realtime | - | 342 MB | Apache-2.0 · runs on audio.cpp | **CPU** · RTF 3.750 |

#### Chatterbox VC

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `chatterbox-vc-audiocpp` (Chatterbox VC) | - | accuracy | - | 2.1 GB | MIT · runs on audio.cpp · GPU recommended | untested |

#### RVC

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `rvc-audiocpp` (RVC) | - | balanced | - | 1.3 GB | MIT (base models); packaged voices without a stated licence ⚠ · runs on audio.cpp | **CPU** · RTF 6.190 |

#### Seed-VC

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `seed-vc-audiocpp` (Seed-VC) | - | accuracy | - | 3.1 GB | GPL-3.0 · runs on audio.cpp · GPU recommended | untested |

### Forced alignment (word timestamps)

#### Omnilingual ASR

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `omnilingual-ctc-300m-align` | en, hi, bn, ta, te, mr +25 | balanced | - | 365 MB | Apache-2.0 | **CPU** · RTF 0.888 |

#### MMS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mms-300m-forced-aligner` | - | balanced | - | 317 MB | CC-BY-NC-4.0 ⚠ | **CPU** · RTF 1.137 |

### Echo cancellation (voice agents)

#### SpeexDSP

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `speex-aec` | - | realtime | - | 0 MB | BSD-3-Clause (SpeexDSP) + MIT (pyaec binding) | **CPU** · RTF 0.026 |

#### WebRTC AEC3

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `webrtc-aec3` | - | realtime | - | 0 MB | BSD-3-Clause (WebRTC) + Apache-2.0 (LiveKit SDK) | **CPU** · RTF 0.026 |

#### DTLN-aec

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `dtln-aec-128` | - | realtime | - | 7 MB | MIT | **CPU** · RTF 0.233 |

### Wake words and keyword spotting

#### Zipformer keyword spotter

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kws-zipformer-gigaspeech` | en | realtime | - | 18 MB | Apache-2.0 | **CPU** · RTF 0.099 |
| `kws-zipformer-zh-en` | zh, en | realtime | - | 33 MB | Apache-2.0 | **CPU** · RTF 0.083 |

#### openWakeWord

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `openwakeword-hey-jarvis` | en | realtime | - | 4 MB | CC-BY-NC-SA-4.0 ⚠ | **CPU** · RTF 0.098 |
| `openwakeword-alexa` | en | realtime | - | 3 MB | CC-BY-NC-SA-4.0 ⚠ | **CPU** · RTF 0.095 |
| `openwakeword-hey-mycroft` | en | realtime | - | 3 MB | CC-BY-NC-SA-4.0 ⚠ | **CPU** · RTF 0.082 |

#### microWakeWord

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `microwakeword-okay-nabu` | en | realtime | - | 0 MB | Apache-2.0 | **CPU** · RTF 0.004 |
| `microwakeword-hey-jarvis` | en | realtime | - | 0 MB | Apache-2.0 | **CPU** · RTF 0.003 |
| `microwakeword-hey-mycroft` | en | realtime | - | 0 MB | Apache-2.0 | **CPU** · RTF 0.003 |
| `microwakeword-alexa` | en | realtime | - | 0 MB | Apache-2.0 | **CPU** · RTF 0.003 |

### Watermarking

#### AudioSeal

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `audioseal-16bits` | - | realtime | - | 93 MB | MIT | untested |

#### Perth

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `perth-onnx` | - | realtime | - | 56 MB | MIT | **CPU** · RTF 0.033 |

#### WavMark

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `wavmark-onnx` | - | realtime | - | 27 MB | MIT | **CPU** · RTF 2.300 |

⚠ = the licence restricts commercial use. RTF = processing time / audio duration (lower is faster).
<!-- END GENERATED: models -->

## Non-commercial and restricted models, and how each runs

Licences never keep a model out of the catalog: each entry states `license`, `commercial_use` and `gated`
(marked ⚠ in the tables above when commercial use is not plainly allowed), and the defaults stay permissive. The
models below were checked against their model cards and licence files on 2026-11-14; none of them was run here
(`validated: untested`).

| id | Licence (commercial use) | Route | State |
| --- | --- | --- | --- |
| `f5-tts-v1-base` | CC-BY-NC-4.0 weights, MIT code (no) | bolo-torch, `f5-tts` kind with `layout: f5-tts` on the `f5-tts` package (`bolo-torch[f5-original]`); voices are bolo's reference clips or `voices_dir` | catalogued |
| `indic-speak` | Indic Open Model License v1.0, gated (conditional: hosting it as an API for others needs Bodhan AI's approval, size thresholds, use policy) | bolo-torch, `indic-speak` kind running the repository's own `inference.py` (`bolo-torch[indic-speak]`) | catalogued; the loader follows the model card, the gated code was not read |
| `indextts-2.5-audiocpp` | bilibili Model Use License (conditional) | tts-server, audio.cpp `index_tts2` family, `IndexTTS2.5-GGUF` q8_0 | catalogued |
| `spark-tts-0.5b` | CC-BY-NC-SA-4.0 (no) | bolo-torch, `spark-tts` kind: the authors' recipe, BiCodec code from github.com/SparkAudio/Spark-TTS at a pinned commit (bundle `spark-tts`) | runnable, untested (fakes; BiCodec code with random weights) |
| `maskgct` | CC-BY-NC-4.0 (no) | bolo-torch, `maskgct` kind: Amphion's pipeline from github.com/open-mmlab/Amphion at a pinned commit (bundle `maskgct`); its phoneme front end needs GPL-licensed packages installed separately | runnable, untested (fakes; front end and pipeline with tiny random networks) |
| `diarizen-wavlm-large-s80-md-v2` | CC-BY-NC-4.0, PLDA CC-BY-4.0 (no) | bolo-torch, `diarizen` kind: the authors' `DiariZenPipeline` and pyannote.audio fork from github.com/BUTSpeechFIT/DiariZen at a pinned commit (`bolo-torch[diarizen]`, own environment) | measured on CPU (DER 5.0 % on the demo dialogue, RTF 12.4) |
| `mms-lid-126`, `mms-lid-4017` | CC-BY-NC-4.0 (no) | ONNX Runtime, `onnx_lid` kind on Xenova's int8 exports | runnable, not measured (978 MB); "Language identification" below |
| `voxlect-indic-lid-mms-lid-256` | CC-BY-NC-4.0 (no) | bolo-torch, `voxlect-lid` kind (`bolo-torch[voxlect]`) | runnable, not measured (3.9 GB) |
| `indic-slid-mhubert`, `indic-slid-mhubert-task2` | CC-BY-NC-SA-4.0 (no) | bolo-torch, `hf-audio-classification` kind | measured |
| `ambernet-langid` | NVIDIA NGC Terms of Use (not an open licence) | ONNX Runtime, `onnx_lid` kind | measured |
| `breeze-tts-2-audiocpp`, `indextts-2-audiocpp`, `omnivoice-audiocpp`, `voxtral-4b-tts-audiocpp`, `higgs-tts-3-4b-audiocpp`, `fish-audio-s2-pro-audiocpp`, … | non-commercial or conditional | tts-server (audio.cpp, llama-server) | catalogued earlier |

Speech-to-speech: `s2s_model` (default `lfm3-audio-2b-llamacpp`, LFM Open License v1.0: commercial use only
below USD 10M yearly revenue) is used only by the opt-in `speech-to-speech` Realtime pipeline
([voice-agents.md](voice-agents.md#pipelines-cascade-audio-in-language-model-speech-to-speech)).
`lfm3-audio-2b-onnx` is the same model from Liquid's ONNX export, run in bolo's own process with ONNX
Runtime (`pip install "bolo-speech[s2s]"`, no llama-server; the same licence): name it as `s2s_model`, or use it
as a text-to-speech voice. Its files (2.7 GB) are pinned by SHA-256. On 4 shared CPU cores it generates
speech at 0.4 to 0.5 times real time, so it wants a faster CPU or a GPU.

## Speech-to-speech and audio-in models

Three ways to hold a spoken conversation (voice-agents.md, "Pipelines"): bolo's recogniser, a language model and
a voice (`cascade`); a model that hears the user and writes the answer, which a bolo voice speaks (`audio-llm`);
or one model that hears and speaks (`speech-to-speech`). Every entry states its licence; the non-commercial ones
are listed like the rest and marked ⚠. Checked against the model cards, LICENSE files and the Hugging Face API on
2026-11-17.

| id | Pipeline | Licence (commercial use) | Download | Runtime | Duplex | State |
| --- | --- | --- | --- | --- | --- | --- |
| `lfm3-audio-2b-llamacpp` | speech-to-speech | LFM Open License v1.0 (only below USD 10M revenue) | 1.4 GB | llama-server `/v1/realtime` (bolo starts it) | turn-based, audio streamed while the user speaks | runnable, untested |
| `lfm3-audio-2b-onnx` | speech-to-speech | LFM Open License v1.0 (as above) | 2.7 GB | ONNX Runtime in bolo's process | as above | runnable; 0.4-0.5x real time on 4 shared cores |
| `qwen2.5-omni-3b-torch` | speech-to-speech | Qwen Research License (no) | 12.0 GB | bolo-torch (`omni` extra), transformers | turn-based, answer comes back whole | runnable, untested (fakes only) |
| `qwen2.5-omni-7b-torch` | speech-to-speech | Apache-2.0 (yes) | 22.4 GB | bolo-torch, transformers; a 24 GB GPU | turn-based | runnable, untested (fakes only) |
| `qwen3-omni-30b-a3b-instruct-torch` | speech-to-speech | Apache-2.0 (yes) | 70.5 GB | bolo-torch, transformers 4.57+; about 79 GB of GPU memory | turn-based | runnable, untested (fakes only) |
| `moshiko-*`, `moshika-*` (14 builds), `hibiki-*` (5), `personaplex-7b-v1`, `personaplex-rl-seamless`, `flm-audio`, `freeze-omni`, `minicpm-o-4.5` | speech-to-speech | see [Full-duplex models](#full-duplex-models) | 4.0 to 20.0 GB | their authors' servers, which bolo installs and starts | **full-duplex** (listens while it speaks, 80 ms frames) | runnable, untested (fakes only) |
| `glm-4-voice-9b` | speech-to-speech | ⚠ glm-4-voice License: academic use free; **commercial use only after registering with Zhipu** (open.bigmodel.cn/mla/form, then free), "Built with glm-4", Chinese law | 21.1 GB (model, glm-4-voice-tokenizer, glm-4-voice-decoder) | bolo-torch (`glm-4-voice` extra): ChatGLM remote code (opt-in), tokenizer and decoder from github.com/THUDM/GLM-4-Voice at a pinned commit; 24 GB GPU in bfloat16, about 9 GB with `quantization: int4` | turn-based; text and speech interleaved and streamed, first audio after 25 speech tokens (about 2 s of speech) | runnable, untested (fakes only) |
| `minicpm-o-2.6` | speech-to-speech | Apache-2.0 (yes; registration optional) | 17.4 GB (with the voices, Vocos and the ChatTTS tokenizer) | bolo-torch (`minicpm-o` extra): **the whole model is remote code**, run only after the per-model opt-in, pinned files checked by sha256; about 18 GB of GPU memory | turn-based; its streaming API (text and 24 kHz speech piece by piece) or `mode: chat` (whole answer) | runnable, untested (fakes only) |
| `step-audio-2-mini` | speech-to-speech | Apache-2.0 (yes) | 17.9 GB | bolo-torch (`step-audio-2` extra): the model's remote code (opt-in) and token2wav from github.com/stepfun-ai/Step-Audio2 at a pinned commit; a 24 GB NVIDIA GPU | turn-based; text streamed, speech in 25-token pieces (first audio after about 1 s of speech), tool calls | runnable, untested (fakes only) |
| `kimi-audio-7b-instruct` | speech-to-speech | MIT (yes; code derived from Qwen2.5 Apache-2.0) | 44.1 GB (with the pinned glm-4-voice-tokenizer) | bolo-torch (`kimi-audio` extra and flash-attn): remote code (opt-in) and kimia_infer from github.com/MoonshotAI/Kimi-Audio at a pinned commit; NVIDIA GPU only, 24 GB or more | turn-based; the authors' loop run step by step: text streamed, speech in 30-token chunks (first audio after about 2.4 s of speech) | runnable, untested (fakes only) |
| `lfm2.5-audio-1.5b-llamacpp` | audio-llm | LFM Open License v1.0 (only below USD 10M revenue) | 915 MB | llama-server (bolo starts it) | - | **ran here** (see below) |
| `ultravox-0.5-llama-3.2-1b-llamacpp` | audio-llm | MIT adapter + Llama 3.2 Community License (yes, with Meta's conditions) | 2.2 GB | llama-server | - | runnable, untested |
| `qwen2.5-omni-3b-llamacpp` | audio-llm | Qwen Research License (no) | 3.6 GB | llama-server (text out only: llama.cpp has no Talker) | - | runnable, untested |
| `qwen2.5-omni-7b-llamacpp` | audio-llm | Apache-2.0 (yes) | 6.2 GB | llama-server | - | runnable, untested |
| `qwen3-omni-30b-a3b-instruct-llamacpp` | audio-llm | Apache-2.0 (yes) | 19.9 GB | llama-server | - | runnable, untested |
| `gemma-4-e2b-it-llamacpp`, `gemma-4-e4b-it-llamacpp` | audio-llm | Apache-2.0 (yes) | 3.4 / 5.2 GB | llama-server | - | runnable, untested |

What llama.cpp b12375 (bolo's pinned llama-server until b12521) does with audio: its multimodal library has audio encoders
for Ultravox, Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, LFM2/LFM2.5-Audio, Voxtral, Gemma 4, GLM's ASR models,
Granite Speech and others, so these hear `input_audio` in `/v1/chat/completions`; the GGUF headers of the mmproj
files above say `clip.has_audio_encoder = true`. Its realtime conversation mode (`/v1/realtime`, speech out) needs
an audio generation head in the mmproj, which the LFM3-Audio family has; Qwen-Omni's Talker, LFM2.5-Audio's
vocoder (Liquid's own runner) and the others have none on llama.cpp, so on llama-server they are audio-in models.

**What ran here.** `lfm2.5-audio-1.5b-llamacpp` (915 MB) on llama-server b12375 (CPU build), started by bolo for
`audio_llm_model` (`-m`, `-mm`, `-t 4 --parallel 1 -c 8192`, loopback port, random API key), on a 4-core Xeon shared
with other jobs (load 11 to 18). It loaded (llama.cpp marks its audio input experimental) and heard `en-short.wav`
("Good morning. Could you tell me when my order will arrive?", 3 s): with "Perform ASR." it gave the sentence back,
and with a system prompt asking it to answer the customer's question it gave the sentence back again instead of an
answer. So the wiring works end to end, and this model is a poor audio-llm through llama-server's chat endpoint.
First token after 59 to 84 s and the whole reply after 153 to 184 s on that overloaded machine: not a measure of
the model. Nothing else in the table above was downloaded: the Qwen-Omni,
Moshi, Hibiki, PersonaPlex, GLM, MiniCPM, Step-Audio and Kimi weights are 12 to 70 GB and the test machine had about 6 GB of disk and
no GPU. The bolo-torch Qwen-Omni loader, the plugin's `respond` op and bolo's turn-based backend were tested with
fake model objects, and the managed audio-llm server with a stand-in for llama-server.

### Full-duplex models

These hear and speak at the same time; voice-agents.md, "Full-duplex models", explains how a Realtime session runs
them. Each runs in its own server, a component bolo installs (`bolo components install <server>`, pinned versions,
SHA-256 for downloads, a pinned commit for git sources) and starts on a loopback port. Weights are pinned by revision
and SHA-256 like every catalog entry. Code licences: Kyutai's `moshi` and `moshi_mlx` packages MIT, `moshi-backend`
MIT or Apache-2.0, `hibiki-zero` MIT, NVIDIA's `personaplex` MIT, CofeAI's `flm-audio` Apache-2.0, Tencent's
Freeze-Omni code under its own non-commercial licence.

| id | Server, build | Weights licence (commercial use) | Download | Needs |
| --- | --- | --- | --- | --- |
| `moshiko-pytorch-bf16`, `moshika-pytorch-bf16` | `moshi`, pytorch | CC-BY-4.0 (yes, credit Kyutai) | 15.8 GB | NVIDIA GPU, 24 GB |
| `moshiko-pytorch-q8`, `moshika-pytorch-q8` | `moshi`, pytorch (bitsandbytes int8) | CC-BY-4.0 (yes) | 8.4 GB | NVIDIA GPU, about 12 GB (estimate) |
| `moshiko-mlx-q4`, `moshika-mlx-q4` / `-q8` / `moshiko-mlx-bf16`, `moshika-mlx-bf16` | `moshi`, mlx | CC-BY-4.0 (yes) | 5.2 / 8.6 / 15.8 GB | Apple silicon, about 6 / 10 / 18 GB |
| `moshiko-candle-q8`, `moshika-candle-q8` / `-bf16` | `moshi`, candle (`--flavour candle`, built with cargo) | CC-BY-4.0 (yes) | 8.6 / 15.8 GB | CUDA or Metal; the CPU build runs but far below real time |
| `hibiki-1b-pytorch-bf16`, `hibiki-2b-pytorch-bf16` | `moshi`, pytorch | CC-BY-4.0 (yes) | 4.0 / 6.0 GB | NVIDIA GPU; French speech in, English speech and text out |
| `hibiki-1b-mlx-bf16`, `hibiki-2b-mlx-bf16` | `moshi`, mlx | CC-BY-4.0 (yes) | 4.0 / 6.6 GB | Apple silicon |
| `hibiki-zero-3b-pytorch-bf16` | `hibiki-zero`, pytorch | ⚠ CC-BY-NC-SA-4.0 (no) | 6.6 GB | NVIDIA GPU; French, Spanish, Portuguese or German in, English out |
| `personaplex-7b-v1` | `personaplex`, pytorch | NVIDIA Open Model License (yes) | 17.1 GB, gated | NVIDIA GPU, about 24 GB; instructions and 18 voices |
| `personaplex-rl-seamless` | `personaplex`, pytorch | ⚠ CC-BY-NC-4.0 with the NVIDIA Open Model License (no) | 17.1 GB, gated | as above |
| `flm-audio` | `flm-audio`, pytorch (Kyutai's protocol) | ⚠ no licence stated; the model card says research use only (no) | 16.8 GB | NVIDIA GPU, about 24 GB; English and Chinese |
| `freeze-omni` | `freeze-omni`, pytorch (Socket.IO) | ⚠ Freeze-Omni License: academic, research and education only (no); Qwen2-7B-Instruct, which comes with it, Apache-2.0 | 17.6 GB | NVIDIA GPU, about 24 GB; English and Chinese; no text out |
| `minicpm-o-4.5` | `minicpm-o`, pytorch (bolo's server around the model's remote code, opt-in) | Apache-2.0 (yes) | 20.0 GB | NVIDIA GPU, about 24 GB; English and Chinese; it decides once a second whether to speak |

Hibiki's Rust builds (`kyutai/hibiki-1b-rs-q8`, `-rs-q6k`) and other conversions are not in the catalog:
[models-unsupported.md](models-unsupported.md) says why.

**A first run on a Windows laptop with an NVIDIA GPU** (12 GB of GPU memory or more for the q8 model; 24 GB for bf16
and PersonaPlex). Nothing here has run on real weights yet; this is the checklist for doing it:

1. Install bolo with the codec and WebRTC: `pip install "bolo-speech[duplex,webrtc]"` (or `uv tool install
   "bolo-speech[duplex,webrtc]"`), and `uv` (`winget install astral-sh.uv`), which fetches Python 3.12 for the server.
   Free disk: about 6 GB for the server's environment (PyTorch with CUDA) and 8.4 GB for the model.
2. `bolo components install moshi` (the pytorch build: moshi 0.2.13, PyTorch 2.9.1 for CUDA 12.8 from
   download.pytorch.org, bitsandbytes 0.49.2). Check: `bolo components list` shows `moshi installed (bolo, pytorch
   0.2.13)`, and `%USERPROFILE%\.cache\bolo\components\moshi\0.2.13\pytorch\Scripts\python.exe -c "import
   torch; print(torch.cuda.is_available())"` prints `True` (else the NVIDIA driver is too old for CUDA 12.8).
3. `bolo pull moshiko-pytorch-q8` (or `moshika-pytorch-q8` for the female voice).
4. Put `device = "cuda"`, `realtime_pipeline = "speech-to-speech"` and `s2s_model = "moshiko-pytorch-q8"` in
   `bolo.toml` (or set `BOLO_DEVICE`, `BOLO_REALTIME_PIPELINE`, `BOLO_S2S_MODEL`) and run `bolo serve`. The first
   session starts the model's server; loading and warming up
   takes one to a few minutes (`duplex_start_timeout_s`, 900 s). `GET /api/v1/info` shows
   `capabilities.realtime.pipelines.speech-to-speech` with `full_duplex: true` and `available: true`.
5. In the console, open **Conversation**, choose the speech-to-speech pipeline and start a session (WebRTC). Use
   headphones, or keep the browser's echo cancellation on. Say "Hello, who are you?" and wait.
6. What to check: an answer within about a second of your question, in Moshi's voice, with its words appearing as
   the assistant's transcript while it speaks; your own words transcribed as user items; talking over Moshi does
   not cut it off by itself (it decides whether to stop), while the console's stop button (`response.cancel`) ends the
   answer at once; `bolo logs` shows `duplex:moshiko-pytorch-q8` ready, and the server's console lines
   (`frame handled in … ms`: under 80 ms means it keeps up in real time); `nvidia-smi` shows the server's memory;
   **Requests** in the console has one `agent` trace per answer (`full_duplex`, `started_after_user_ms`, time to
   first audio). A second session while one runs is refused with `overloaded` (Kyutai's server holds one).
7. Ending the session frees the server for the next one; stopping bolo stops it. Report back the `frame handled`
   times, the GPU memory and whether answers came in time.

For PersonaPlex: accept its licence on Hugging Face, set `BOLO_HF_TOKEN`, `bolo components install personaplex`,
`bolo pull personaplex-7b-v1`, `s2s_model = "personaplex-7b-v1"`, and give the session `instructions` (its role)
and a `voice` such as `NATM1`. For Hibiki, speak French: it answers with the English translation.
Running the four bolo-torch speech-to-speech models on a Windows laptop with an NVIDIA GPU (what fits in 8, 12,
16 and 24 GB, install, pull, a test request, what to check): plugins/bolo-torch/README.md, "Speech-to-speech models
on a Windows laptop with an NVIDIA GPU".

## Remote code

Some models exist only as Python files in their own Hugging Face repository: transformers imports and runs those
files when a loader passes `trust_remote_code=True`. That code runs inside the plugin's process with everything the
account running bolo may do (read and write files, reach the network, see other models' data). bolo never runs it
unless you opt in, for one model at a time:

```bash
bolo pull minicpm-o-2.6                # the files download either way; bolo pull warns that the model needs remote code
bolo show minicpm-o-2.6                # remote_code: the pinned commit, the files, what the review found
bolo models config minicpm-o-2.6 --set allow_remote_code=true      # the opt-in (or PUT .../settings)
bolo models config minicpm-o-2.6 --unset allow_remote_code         # back to off
```

What the opt-in allows, and nothing more:

* **A reviewed commit.** The catalog entry pins `options.remote_code.revision` (it must be the entry's
  `source.revision`), with a note of what was read (`reviewed`) and the sha256 of every `.py` file and of every JSON
  file with an `auto_map` (`files`). Catalog validation refuses an entry whose pin does not say all of this.
* **Checked before import.** Right before the model library is imported, bolo-torch checks that the folder was
  downloaded at exactly that revision (bolo's `manifest.json`), that each pinned file has its sha256, that no other
  `.py` file is in the folder, and that no `auto_map` names a class in another repository (`org/repo--module.Class`,
  which transformers would download and run). Any difference is refused (`remote_code_refused`) with what to do.
* **Only the operator can opt in.** The setting is per model (`allow_remote_code` in `model-settings.json`); bolo
  sends the plugin your setting and overwrites anything a catalog entry says, so an entry cannot opt in for you.
  `bolo pull`, the pull job's `warnings`, the catalog API (`remote_code.notice`) and every load (a log warning) say
  what runs.

Models in the catalog with remote code: `minicpm-o-2.6` (the whole model), `step-audio-2-mini`, `glm-4-voice-9b`
and `kimi-audio-7b-instruct` (their language models; the rest of their code comes from GitHub, below), and
`sov-transcribe-1` (its model and processor; the repository's `vllm_plugin/` folder is not downloaded), and
`clef-omni` (Cloudflare's `joint_schema_model.py`: its decision head and request encoding). The same
mechanism serves any other model with remote code: add the `remote_code` block to its catalog entry (for
your own entries, `<data_dir>/custom.yaml`). `bolo-torch pin-remote-code <downloaded folder> --revision <commit>`
prints it; read every file it lists before you trust it.

Configuration files that name classes to build (the hyperpyyaml `flow.yaml` of Step-Audio 2's token2wav, GLM-4-Voice's
decoder `config.yaml`) are treated the same way: their sha256 is in the entry's `options.sha256` and checked before
they are read.

### Inference code from GitHub

Some models need code from their authors' GitHub repository that is not a package (a speech tokenizer, a decoder, a
prompt format). bolo-torch carries a pin for each such bundle: the repository, the commit, its licence and the sha256
of every file used (`plugins/bolo-torch/src/bolo_torch/upstream/*.json`). `bolo-torch code install <name>` downloads
exactly those files from `raw.githubusercontent.com` and checks each; a load installs the bundle when it is missing
(`BOLO_TORCH_CODE_AUTO_INSTALL=0` turns that off) and checks every file again before importing anything. The code goes
to `BOLO_TORCH_CODE_DIR` (default `~/.cache/bolo-torch/code`, `%LOCALAPPDATA%\bolo-torch\code` on Windows).
`bolo-torch code list` shows each bundle with its licences.

| Bundle | Repository @ commit | Licence | Used by |
| --- | --- | --- | --- |
| `step-audio-2` | stepfun-ai/Step-Audio2 @ `76e272b56c39` | Apache-2.0 | `step-audio-2-mini` |
| `kimi-audio` | MoonshotAI/Kimi-Audio @ `349251e1d8f4` (`kimia_infer/`) and THUDM/GLM-4-Voice @ `eb00ce9142e8` (`speech_tokenizer/`, its submodule) | MIT per README and pyproject (no LICENSE file; Qwen-derived parts Apache-2.0); Apache-2.0 | `kimi-audio-7b-instruct` |
| `glm-4-voice` | THUDM/GLM-4-Voice @ `eb00ce9142e8` (speech tokenizer, `flow_inference.py`, CosyVoice code) and shivammehta25/Matcha-TTS @ `dd9105b34bf2` (its git submodule) | Apache-2.0 (code; the weights have the glm-4-voice License), MIT | `glm-4-voice-9b` |

## Models outside the catalog

`bolo pull hf:<owner>/<repo>[@revision]` inspects the repository's file names
and registers a custom entry (stored in `<data_dir>/custom.yaml`):

| Layout | Detected as |
| --- | --- |
| `model.bin`, `config.json`, `tokenizer.json` | CTranslate2 Whisper → faster-whisper |
| `*-encoder.onnx`, `*-decoder.onnx`, `*-tokens.txt` | Whisper (sherpa-onnx) |
| `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, `tokens.txt` | transducer (streaming if the name says so; NeMo if the name has `nemo`/`parakeet`) |
| `preprocess.onnx`, `encode*`, `uncached_decode*`, `cached_decode*` | Moonshine |
| `*.onnx`, `*.onnx.json`, `tokens.txt` | VITS in Piper format |
| `model*.onnx`, `voices.bin`, `tokens.txt` | Kokoro or KittenTTS |
| `model*.onnx`, `tokens.txt` | ambiguous: pass `--kind` (`sense_voice`, `paraformer`, `nemo_ctc`, `zipformer_ctc`, …) unless the repository name gives it away |

When both float and int8 exports are present the int8 files are fetched.
Custom models are always `untested`, and their licence is whatever the
repository's card declares. The models must already be in the engine's
format: bolo does not convert PyTorch checkpoints (it has no PyTorch).

### Any Piper voice, by name

```bash
bolo pull piper:te_IN-venkatesh-medium        # <language>_<COUNTRY>-<name>-<quality>
```

fetches the voice from the Piper project's collection
(`rhasspy/piper-voices`: `<voice>.onnx`, `<voice>.onnx.json`, `MODEL_CARD`)
and converts it. The published files are not loadable by sherpa-onnx as they
are: it wants a `tokens.txt` table and a few metadata entries inside the ONNX
file. bolo writes the table from the JSON and appends the metadata to a copy
of the model right after the download (an ONNX file is a protobuf message,
so this needs no machine-learning library), then registers the voice as a
custom entry that needs the optional eSpeak NG component. The same works
over HTTP (`POST /api/v1/models/pull` with `{"id": "piper:…"}`), and a
catalog entry can ask for the step with `source.convert: piper`: that is how
the Telugu, Malayalam, Marathi, Bengali, Urdu and Tamil voices marked
"converted … by bolo" in the table above are defined. Checked here with
ten voices in six languages, each transcribed back by a recogniser of
its language (see [indic-telephony.md](indic-telephony.md)). The licence of a
Piper voice is that of its training data: read its `MODEL_CARD`.

### A directory on your disk

```bash
bolo pull ./my-voice --kind vits --name my-voice --language mai [--option numbers=hi]
```

registers a model directory that is already in an engine's layout (for a
VITS voice: `model.onnx` + `tokens.txt`). Nothing is copied: the store links
to the directory, so the files must stay where they are, and `bolo rm`
removes the registration, not your files. bolo writes one small file,
`manifest.json`, into the directory, so it must be writable. `--kind` is needed when the file
names do not give the layout away; `--option key=value` sets loader options.
This is a command-line feature only: the HTTP API does not register server
paths.

### Models from a plain URL or a release archive

A catalog entry can also name its files by URL instead of a Hugging Face
repository (`source.urls`: `path`, `url`, `size`, `sha256`), and a
`.tar.gz`, `.tgz`, `.tar` or `.zip` archive with `extract: true` is unpacked
into the model directory after its checksum has been verified. Downloads
resume, report progress like any pull and honour the usual proxy
environment variables; the Hugging Face token is sent to Hugging Face only.
Unpacking accepts regular files only and refuses any archive with links or
with a path that would leave the model directory. There is no offline
switch: with `auto_pull` off nothing is downloaded unless you ask.

## Voices converted by this project

Some good voices exist only as PyTorch checkpoints. bolo has no PyTorch, so
the project converts them once, in the open (`tools/voices/` and the "Voices"
workflow), and publishes the results as assets of a GitHub release of this
repository (`https://github.com/cB-Sigma/bolo-speech/releases/tag/voices-v1`) and as one
public repository per voice under `codebucketsolutions/` on Hugging Face (the
catalog pulls from the Hub, pinned to the uploaded commit, with the release
archive as fallback). Each archive
holds `model.onnx`, `tokens.txt`, `voice.json` (source repository and pinned
revision, licence, language, sample rate, character set), `LICENSE` and
`ATTRIBUTION`.

**SYSPIN: published** (release `voices-v1`, 22 voices; the catalog has
their sizes and checksums).

| Voices | Source (all by the SYSPIN project, SPIRE Lab, Indian Institute of Science) | Licence |
| --- | --- | --- |
| `syspin-mai-female`, `syspin-mai-male` (Maithili) | `SYSPIN/vits_Maithili_Female`, `_Male` | weights MIT; SYSPIN corpus CC-BY-4.0 |
| `syspin-bho-female`, `syspin-bho-male` (Bhojpuri) | `SYSPIN/vits_Bhojpuri_*` | same |
| `syspin-hi-*`, `syspin-bn-*`, `syspin-mr-*`, `syspin-te-*`, `syspin-kn-*`, `syspin-gu-*` | `SYSPIN/vits_Hindi_*`, `_Bengali_`, `_Marathi_`, `_Telugu_`, `_Kannada_`, `_Gujarati_` | same |
| `syspin-mag-*` (Magahi), `syspin-hne-*` (Chhattisgarhi), `syspin-en-*` (Indian English) | `SYSPIN/vits_Magahi_*`, `_Chhattisgarhi_`, `_English_` | same |

The pinned revisions are in `tools/voices/syspin.json`. The conversion
changes the file format only (Coqui VITS checkpoint to ONNX); the weights are
not retrained. All are single-speaker VITS voices at 22.05 kHz, about 114 MB
each, with character input and no espeak-ng.

**Meta MMS-TTS (non-commercial): published** (release `voices-v1` and Hugging Face). 18 voices
for Indian languages (`mms-tts-<iso>`: Punjabi, Odia, Assamese, Dogri,
Maithili, Magahi, Chhattisgarhi, Awadhi, Haryanvi, Urdu, Hindi, Bengali,
Gujarati, Kannada, Malayalam, Marathi, Tamil, Telugu), from
`facebook/mms-tts` pinned to a commit, converted with MMS's own VITS
inference code (`tools/voices/export_mms.py`, the recipe sherpa-onnx used for
its English MMS voices). **CC-BY-NC-4.0: non-commercial use only**; each
archive carries the licence text and the attribution, and the entries say
`commercial_use: false`. The conversion was run here for eight languages and each
voice was loaded by bolo and transcribed back (numbers in
[indic-telephony.md](indic-telephony.md) and
`docs/measurements/mms-tts-round-trip.json`); MMS-TTS has no voice for
Sanskrit, Sindhi, Kashmiri, Konkani, Manipuri, Bodo, Santali or Bhojpuri.

## Character-input voices

Voices in Coqui's VITS format (`coqui-bn-*`, the SYSPIN voices) take the
text itself, character by character, from a fixed table. A character that is
not in the table has no sound. Left alone, the engine drops such characters
without a word, so "कमरा 42" would be read as "कमरा". bolo checks the text
against the table first:

* upper-case letters become lower-case when only those are in the table;
* digits the voice has no symbol for are spelled out when a number
  normaliser exists for the language (English and Hindi; a voice entry may
  opt into Hindi number words for a related language with the option
  `numbers: hi`);
* punctuation the voice does not know is replaced by a mark it knows, or
  dropped;
* any letter or digit that is still not in the table stops the request with
  `400 unsupported_characters`, naming the characters.

Checked on the real `coqui-bn-female` voice (83 symbols, Bengali digits but
no ASCII digits and no Latin letters) and on the SYSPIN Maithili and
Bhojpuri voices.

Before that check, the text gets the **cleaning the voice's training text
had** (catalog option `text_cleaner`, or `cleaner` in the voice's
`voice.json`): `multilingual` for the SYSPIN voices (Coqui's
`multilingual_cleaners`: NFC, which also splits precomposed nukta letters,
lower case, `;` and `:` to commas, hyphens to spaces, brackets and double
quotes removed, then `।` to `.` for voices without a danda), `lowercase` for
MMS-TTS. sherpa-onnx does none of this itself. The same front end runs in the
console's in-browser voices and the tests share their fixtures. On the
published Hindi and English SYSPIN voices it lowered the round-trip error on
mixed-case, punctuated text from 24.6 % to 14.5 % and from 20.8 % to 13.9 %
WER (8 sentences each, `docs/measurements/syspin-text-cleaning.json`). Phoneme-based voices (Piper format, Kokoro, KittenTTS) are
not affected: espeak-ng reads digits and foreign words for them.

## Speaker diarization

Two kinds of diarizer are in bolo's core, both without PyTorch (pyannote's own PyTorch pipelines run on the
bolo-torch plugin, [below](#pyannotes-pytorch-pipelines-bolo-torch)):

* **`pyannote-3.0-eres2net`** (the default) and its siblings: pyannote's
  segmentation-3.0 model finds up to three local speakers per 10 s window,
  speaker embeddings (ERes2Net) are clustered across windows, and bolo folds
  spurious clusters into real ones (api.md, "How many speakers"). Run by
  sherpa-onnx.
* **`nemotron-3-diarization`** (opt-in): NVIDIA's Nemotron-3-Diarization
  (September 2026, OpenMDW-1.1, 100M parameters), an end-to-end model. A
  31-layer Transformer reads the log-mel spectrogram and gives, every 10 ms,
  the probability that each of up to 8 speakers is talking; speakers are
  numbered in order of arrival, so there is no clustering step, and two
  speakers can be active at once (overlap). Run by ONNX Runtime
  (`bolo/engines/nemotron.py`), 147 MB download (the int8 graph plus the
  ERes2Net embeddings used for named speakers and live streams).

Choose per request with `diarization_model=nemotron-3-diarization`
(`/v1/audio/transcriptions`) or `model=nemotron-3-diarization`
(`/api/v1/diarize`), or for the whole server with `default_diarization`
(configuration.md). It needs the `onnxruntime` Python package (part of
`bolo-speech[all]` through faster-whisper; otherwise `pip install onnxruntime`).

**When to use Nemotron.** Meetings and conversations where people talk over
each other, telephone audio, and voices that sound alike: there it gets the
number of speakers right far more often and almost never mixes two people up
(measurements below), at a fifth of the cost. It is **not the default**
yet: on clean, turn-taking two-voice dialogues it was no better than
clustering, merged some clearly different voices, and as telephone audio
its result changed from one synthesis of the same dialogue to the next
(below).

### How it runs in bolo

* **Front-end.** NeMo's log-mel features in NumPy (`bolo/audio/features.py`,
  `NemoLogMel`): pre-emphasis 0.97, 25 ms symmetric Hann window inside a
  512-point FFT every 10 ms, zero-padded centred frames, 128 Slaney mel bands,
  natural log with a 2^-24 guard, no normalisation. It works on a whole
  recording or on audio arriving in pieces and gives the same frames either
  way.
* **Chunks and speaker cache.** The ONNX graph is one encoder step: a
  chunk's log-mel frames plus the cached frame embeddings in, speaker logits
  for every frame and the chunk's embeddings out. bolo runs the loop around
  it in NumPy, following transformers' reference implementation: a whole
  recording is cut into 27.2 s chunks with 3.2 s look-ahead; every chunk
  also sees a FIFO queue of recent frames and the Arrival-Order Speaker Cache
  (264 frames chosen per speaker by how clearly they belong to that speaker),
  which keeps a speaker on the same channel across chunks. Recordings of any
  length work, with flat memory. The same loop runs live
  (`NemotronDiarization.stream()`: 1.04 / 0.64 / 0.32 s latency modes), but
  that is not yet connected to bolo's live APIs: a WebSocket session with
  `diarize=true` still labels each utterance with speaker embeddings, also
  when Nemotron is the default model.
* **Turns.** A speaker talks where their probability is above `threshold`
  (0.5); a pause of the same speaker shorter than `min_duration_off` (1 s)
  is closed. Turns of different speakers may overlap and are reported as
  overlaps exactly as with the pyannote pipelines. Further settings
  (catalog `options`): `offset` (hysteresis), `pad_onset`, `pad_offset`,
  `min_duration_on`, `min_speaker_s`. The per-request `threshold` /
  `diarization_threshold` sets the speech threshold.
* **Frame probabilities.** The pipeline keeps the per-frame probabilities,
  one column per final speaker (`DiarizeOptions.activity`, not part of any
  API response), for components that condition recognition on the
  diarization.
* **Speaker counts.** `num_speakers` / `max_speakers` fold the speakers who
  talk least into the speaker the model found next most likely for their
  frames. The model cannot be made to hear more speakers than it does, so
  `min_speakers` has no effect; at most 8 speakers. bolo's clustering
  clean-up (`diarization_min_speaker_s`, `diarization_merge_similarity`) is
  not applied.

**Validation against the reference** (2026-11-13; transformers 5.23 with
PyTorch fp32 as the reference, 24 AMI excerpts of 45 s and two 4.5-minute
recordings made by joining six excerpts of one meeting):

| what | compared with | result |
| --- | --- | --- |
| log-mel front-end | transformers' `NemotronAsrStreamingFeatureExtractor` | max abs. difference 1.6e-4, mean 3.2e-7 (log units); same frame count (the reference adds one masked padding frame); fixed values in `tests/test_nemotron.py` |
| offline probabilities, catalog int8 graph (onnx-community) | PyTorch fp32, same chunking | mean abs. difference 0.0009; 99.95 % of frame-and-speaker decisions at 0.5 identical |
| offline probabilities, our own fp32 export (`server/tools/export_nemotron_onnx.py`) | PyTorch fp32 | 0.00002; 99.998 % |
| offline probabilities, our own int8 export (same tool) | PyTorch fp32 | 0.0014; 99.9 % |
| live loop, 1.04 s mode, audio fed in 0.1 s pieces (2 recordings) | transformers' streaming loop | 0.0008; 99.95 % |

(An earlier measurement gave 0.014 and 98.7 %
for the int8 graph: that was each excerpt in one ONNX pass, without the chunk
loop, against the chunked reference; repeated here, 0.015 and 98.8 %. With
the same chunking the two agree to 0.0009.)

### pyannote's PyTorch pipelines (bolo-torch)

bolo's default diarizer runs pyannote's segmentation-3.0 model exported to ONNX, with its own embedding model,
clustering and clean-up, so its turns are not pyannote's. For pyannote's exact results the
[bolo-torch](../plugins/bolo-torch/README.md) plugin runs the original `pyannote.audio` pipelines as bolo
`diarization` models: `POST /api/v1/diarize` with `model=...`, or `diarization_model=...` with `diarize=true` /
`response_format=diarized_json` on `/v1/audio/transcriptions`.

| id | pipeline | licence | gated | downloads (pinned, sha256) |
| --- | --- | --- | --- | --- |
| `pyannote-speaker-diarization-3.1` | `pyannote/speaker-diarization-3.1`: segmentation-3.0, WeSpeaker ResNet34, agglomerative clustering | MIT (pipeline, segmentation) + CC-BY-4.0 (WeSpeaker embedding, PLDA files) | yes: the pipeline, `pyannote/segmentation-3.0`, and `pyannote/speaker-diarization-community-1` (pyannote.audio 4 loads its PLDA files for every pipeline) | config.yaml, `segmentation-3.0` (5.9 MB), `wespeaker-voxceleb-resnet34-LM` (26.6 MB), community-1's PLDA (0.3 MB) |
| `pyannote-speaker-diarization-community-1` | `pyannote/speaker-diarization-community-1`: segmentation, WeSpeaker embedding, VBx clustering with a PLDA | CC-BY-4.0 (attribution) | yes (it asks for contact details) | the repository's config.yaml, segmentation, embedding and PLDA (32.8 MB) |

* **Setup.** `pip install "bolo-torch[pyannote]"` with PyTorch (plugins/bolo-torch/README.md; the `pyannote`
  extra does not share an environment with `parler`). Accept the conditions on each gated model page, create a read
  token and set `BOLO_HF_TOKEN` (or `HF_TOKEN`) for bolo; without access the load answers `401 model_gated` naming
  the repositories.
* **Requests.** `num_speakers` goes to the pipeline. `min_speakers` / `max_speakers` work as for every diarizer:
  when the pipeline's own count falls outside them, bolo runs it again with that number of speakers (catalog
  options `num_speakers`, `min_speakers`, `max_speakers` are handed to the pipeline on every call). `threshold` is
  not used: pyannote's tuned clustering parameters apply. bolo's speaker clean-up (api.md, "How many speakers") does
  not run: pyannote's speaker count stands. Known speakers, overlap reports and `separate_overlaps` use the pipeline's own speaker
  embedding. Option `exclusive: true` (community-1) returns its exclusive diarization, one speaker at a time.
  `device` is bolo's (`BOLO_TORCH_DEVICE=auto` for CUDA or MPS).
* **Safety.** pyannote builds a pipeline from whatever `config.yaml` names. bolo-torch reads the file itself with
  `yaml.safe_load`, checks its sha256 against the catalog, accepts only pyannote's own pipeline classes and
  clustering names, plain parameter values and sub-models that bolo downloaded at pinned revisions, and gives
  pyannote the checked dictionary, never the file. The same guard covers PixIT. pyannote.audio 4's usage metrics
  (sent to pyannote's servers by default) are switched off unless `PYANNOTE_METRICS_ENABLED` is set.
* **Measured** (2026-11-17, the demo's two-voice dialogue, 27.2 s, reference turns from its script, frame DER without
  collar; 4 vCPU Xeon, 4 threads, torch 2.15.0+cpu, pyannote.audio 4.1.0):

| diarizer | DER | missed | false alarm | confusion | speakers | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `pyannote-speaker-diarization-3.1` (PyTorch) | 6.05 % | 3.8 % | 2.3 % | 0 | 2 | 0.91 |
| `pyannote-speaker-diarization-community-1` (PyTorch) | 6.05 % | 3.8 % | 2.3 % | 0 | 2 | 0.88 |
| `pyannote-3.0-eres2net` (ONNX, default) | 5.87 % | 3.6 % | 2.3 % | 0 | 2 | 0.23 |
| `nemotron-3-diarization` (ONNX) | 5.87 % | 5.1 % | 0.8 % | 0 | 2 | 0.06 |

  The two PyTorch pipelines gave the same seven turns, through bolo as when called directly; on the first 20.8 s
  3.1 scored 4.93 %, the ONNX default 4.93 % and Nemotron 5.22 %. With a 250 ms collar every diarizer is under 1 %.
  One clean synthetic dialogue says nothing about meetings: pyannote reports community-1 ahead of 3.1 on most of its
  benchmarks (AMI SDM 19.9 % against 22.7 %, model card). The machine was shared with other jobs (load average up
  to 17 on 4 cores): the PyTorch RTFs above are from quiet moments; through bolo under load they were 1.0-1.5 (3.1)
  and 4.1 (community-1, 2 threads), and a run at 4 threads
  stalled for minutes in the segmentation LSTM (thread oversubscription; fewer `BOLO_TORCH_THREADS` helps).

### DiariZen (bolo-torch)

`diarizen-wavlm-large-s80-md-v2` (alias `diarizen`) is BUT Speech@FIT's DiariZen v2: a pruned WavLM Large with
Conformer layers labels up to four speakers per 16 s window (windows every 1.6 s), WeSpeaker ResNet34 embeddings of
each local speaker are clustered with VBx and the model's PLDA. The plugin runs the authors' own `DiariZenPipeline`
on their pyannote.audio 3.1.1 fork, both installed from github.com/BUTSpeechFIT/DiariZen (MIT) at a pinned commit in
an environment of their own (the fork replaces pyannote.audio; the install command is the entry's `plugin_install`
and plugins/bolo-torch/README.md). VBx finds the number of speakers: `num_speakers` is not used. Weights
CC-BY-NC-4.0 (not for commercial use), PLDA CC-BY-4.0. On the demo dialogue above, through bolo: DER 5.03 % (missed
3.3 %, false alarm 1.8 %, no confusion, two speakers), but RTF 12.4 on the shared 4-vCPU machine at 4 threads (load
average about 15): a GPU model in practice. The model card reports AMI SDM DER 13.9 % against 22.4 % for pyannote 3.1.

### Measured: Nemotron against clustering

2026-11-13, `docs/measurements/diarization-nemotron.json`. 4 vCPU Xeon @
2.1 GHz, 2 threads, shared with other jobs. Data: the 24 AMI excerpts of
"Overlapping speech" (45 s, single distant microphone, test meetings
ES2004a, IS1009a, TS3003a, EN2002a, 12-65 % overlap) and the synthetic
two-voice mixes of the same section (11 pairs: 4 cross-gender English, 3
same-gender female, 3 same-gender male, 1 Hindi; 2.5 s and 5 s overlap at
0 dB; each also as 8 kHz G.711 μ-law): 44 mixes. DER with pyannote.metrics
4.0, collar 0, overlapped speech scored, pooled over files; for AMI against
the word times (gaps under 0.3 s joined) and against the manual segments.
`pyannote-3.0-eres2net` with its defaults (clustering 0.9 and clean-up);
Nemotron untuned (threshold 0.5, nothing else) and as shipped.

| audio | diarizer | DER | false alarm | missed | confusion | speaker count right | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMI, word-time reference | pyannote-3.0-eres2net | 38.6 % | 4.5 % | 19.9 % | 14.2 % | 7 / 24 | 0.20 |
| | Nemotron, untuned | 34.7 % | 1.6 % | 31.2 % | 1.8 % | 18 / 24 | 0.04 |
| | **Nemotron, as shipped** | **24.1 %** | 3.1 % | **19.0 %** | **2.0 %** | **18 / 24** | **0.04** |
| AMI, manual-segment reference | pyannote-3.0-eres2net | 43.2 % | 1.5 % | 31.2 % | 10.6 % | 6 / 24 | |
| | Nemotron, untuned | 43.0 % | 0.3 % | 41.8 % | 0.8 % | 15 / 24 | |
| | **Nemotron, as shipped** | **33.1 %** | 1.0 % | 31.0 % | 1.1 % | 15 / 24 | |
| synthetic 16 kHz (22 mixes) | pyannote-3.0-eres2net | 3.8 / 27.8 / 17.3 / 1.4 % (cross / same-f / same-m / Hindi) | | | up to 17.3 % | 18 / 22 | 0.19 |
| | Nemotron, as shipped | 0.7 / 0.7 / 0.9 / 1.1 % | | | 0 % | 22 / 22 | 0.04 |
| synthetic 8 kHz μ-law (22 mixes) | pyannote-3.0-eres2net | 30.7 / 46.6 / 41.4 / 25.8 % | | | up to 33.7 % | 5 / 22 | 0.19 |
| | Nemotron, as shipped | 0.9 / 1.0 / 1.4 / 1.3 % | | | ≤ 0.01 % | 22 / 22 | 0.04 |

Where Nemotron got the AMI speaker count wrong it always found too few:
five times it missed a speaker with less than 2.5 s of speech in the
excerpt, once one with about 8.5 s. On two 4.5-minute recordings (six excerpts of one
meeting joined, 4 speakers) it found 4 and 3 speakers (clustering: 7 and 4)
at RTF 0.04 against 0.21-0.23 for clustering.

**Tuning** (the missed speech). Changing the threshold alone moves misses
against false alarms without helping (AMI, word reference: threshold 0.4
DER 33.7 %, 0.5 34.7 %, 0.6 37.1 %). What the model misses is mostly the
short pauses inside a speaker's turn, so the shipped setting closes a
speaker's pauses shorter than 1 s: missed speech 31.2 → 19.0 % (now below
clustering's 19.9 %), false alarm 1.6 → 3.1 %, confusion 1.8 → 2.0 %. Chosen
from a grid over threshold (0.35-0.6), hysteresis, padding (0-0.5 s) and gap
closing (0-2 s) on these 24 excerpts, then by the transcripts below
between three candidates (untuned; pauses under 1 s closed; the same plus
0.2 s added to the end of every turn). The DER-best setting on the grid
(threshold 0.5, 0.2 s end padding, 1 s gap closing: DER 23.4 %; the same
setting was picked when each meeting in turn was left out of the choice)
cost words in the transcripts (cpWER 64.3 % instead of 61.2 %), so padding
is not used. Closing pauses up to 1.5 s gives a slightly lower DER (23.4 %,
false alarm 6.7 %) and was not tried on transcripts. These settings were
chosen on the same excerpts they are reported on.

**Transcripts** (`whisper-base`, `faster-whisper-tiny-hindi` for Hindi; each
diarized turn transcribed on its own, as bolo does; cpWER with MeetEval
0.4.3, Whisper's English normaliser; 95 % intervals by paired bootstrap over
files):

| audio | files | pyannote-3.0-eres2net | Nemotron, as shipped | relative (95 % CI) | files better / worse | worst file |
| --- | --- | --- | --- | --- | --- | --- |
| AMI excerpts | 24 | 69.4 % | **61.2 %** | **-12 % (-16 to -8 %)** | 20 / 3 | +1.3 pt |
| synthetic, all | 44 | 55.5 % | **25.1 %** | **-55 % (-62 to -46 %)** | 33 / 7 | +21.5 pt |
| · 16 kHz cross-gender | 8 | 19.7 % | 23.3 % | +18 % (-0.5 to +45 %) | 1 / 5 | +21.5 pt |
| · 16 kHz same-gender female | 6 | 52.0 % | 20.7 % | -60 % | 5 / 0 | 0.0 pt |
| · 16 kHz same-gender male | 6 | 44.5 % | 19.7 % | -56 % | 6 / 0 | -1.0 pt |
| · 8 kHz cross-gender | 8 | 59.0 % | 29.4 % | -50 % | 8 / 0 | -2.9 pt |
| · 8 kHz same-gender female | 6 | 90.5 % | 22.9 % | -75 % | 6 / 0 | -51.5 pt |
| · 8 kHz same-gender male | 6 | 87.0 % | 29.2 % | -66 % | 6 / 0 | -38.8 pt |
| · Hindi, 16 / 8 kHz (one pair) | 2 + 2 | 23.9 / 56.2 % | 24.4 / 40.3 % | | | |

The 16 kHz cross-gender mixes, which clustering already handled, got worse:
where Nemotron keeps the first speaker active through part of the overlap,
bolo transcribes the overlap with the first speaker's turn and the second
speaker's turn starts late (the worst mix lost most of a sentence). This is
the known limit of cutting the recording at turns; diarization-conditioned
recognition (below) is the way past it.

**Clean two-voice dialogues** (no overlap). 32 dialogues of four lines (about
16 s, 0.6 s pauses) read by pairs of Kokoro voices, 15 cross-gender and 17
same-gender: Nemotron told the two voices apart in 14, clustering in 15. Each
failed on different pairs: Nemotron got five cross-gender pairs wrong that
clustering separated (merged af_heart with am_adam, af_sarah with bm_lewis,
bf_isabella with am_eric; mixed up af_heart with am_michael and bf_alice
with am_onyx; with eight lines instead of four it merged all five), and
separated four same-gender pairs that clustering merged.
PyTorch gives the same probabilities on these files, so this is the model,
not the port: on clean synthetic turn-taking it can hear two voices as one.

These results are for one synthesis of each dialogue, and they are not
stable: Kokoro's output differs slightly every time it reads the same text,
and that is enough to change Nemotron's answer (2026-11-14, af_sky with
af_bella, a same-gender pair that clustering merges; 40 dialogues put
together from five syntheses of each of the four lines, espeak-ng phonemes).
At 16 kHz it found two speakers and gave every line its own speaker in all
40 (DER 3-4 %; one more synthesis outside this set gave 36 %, nearly all
missed speech). As 8 kHz G.711 μ-law it found two speakers in 38 and gave
every line its own speaker in only 31; it missed up to 44 % of the speech
(median 23 %). With longer turns (four lines of about 8 s, 10 dialogues from
two syntheses of each line) it merged or mixed up the two voices at both
rates in all 10. The same two
voices in a conversation with one 2.5 s overlap (the kind of mix measured
above: A, B, A's long sentence with B starting before it ends, A) came out
right in 12 of 12 at both rates, DER 1.4-1.9 %. So what the tables above show
for 8 kHz and same-gender voices holds for conversations with overlap, not
for short clean turn-taking dialogues, where telephone audio makes the result
a matter of chance. `tests/test_real.py` asserts the first and, for the
clean 8 kHz dialogue, only that no third speaker appears.

**Speed.** RTF at 2 threads: 0.04 offline (recordings of 45 s to
4.5 min), against 0.17-0.23 for `pyannote-3.0-eres2net` on the same audio.
The live loop costs much more, because each small chunk attends to the whole
cache: RTF 0.64 (1.04 s latency), 0.93 (0.64 s) and 1.83 (0.32 s, slower
than real time) per stream at 2 threads.

**Default: not changed.** Nemotron is clearly better on the meeting
excerpts (DER 38.6 → 24.1 %, confusion 14.2 → 2.0 %, cpWER -12 %) and on
overlapped, same-gender and telephone audio, but it is not better everywhere:
on clean two-voice dialogues it merged five cross-gender pairs that the
current default separates (the default-settings test
`test_two_voices_are_told_apart` would fail with it), and on 16 kHz
cross-gender overlap mixes the transcripts got worse. The evaluation is
45-second excerpts and synthetic audio, not whole meetings, and the model
files are a third-party conversion (below). It stays opt-in until full
meetings (AMI, NOTSOFAR) and real two-party calls have been measured.

**Licence and provenance.** The weights are NVIDIA's, under the OpenMDW
License Agreement 1.1 (permissive, commercial use allowed). Redistributing the
model files (for example in a container image) requires a copy of the
agreement and the notices that came with the model; a party that sues
claiming the model infringes a patent or copyright loses the licence; the
training data (among it licensed conversational data) is NVIDIA's
representation. The catalog downloads the int8 ONNX conversion from
`onnx-community/Nemotron-3-Diarization-ONNX` (Hugging Face's ONNX community
organisation, not NVIDIA), pinned to commit `353b6f8a` (`model_quantized.onnx`
sha256 `fff7d18c…`, `model_quantized.onnx_data` `002d7483…`). We intend to
publish our own conversion before relying on it: `server/tools/export_nemotron_onnx.py`
makes the same graph from NVIDIA's `model.safetensors` with transformers and
PyTorch on CPU in a few minutes (fp32 397 MB; int8 118 MB with 8-bit
block-wise weights), validated above; it is not hosted yet. Exporting with
NeMo itself was not tried: NeMo and its dependencies need several GB, more
than this machine had free, and the transformers port reproduced the
reference to 0.00002. A directory made by the tool can be registered with
`bolo pull ./dir --kind nemotron_diarization` (the kind is guessed when the
directory name contains "nemotron" and "diar"); it gets the same
post-processing defaults as the catalog entry.

### Diarization-conditioned recognition (DiCoW, bolo-torch)

The diarized transcription above cuts the recording at speaker turns and
transcribes each turn on its own, so where two people talk at once the turn
that started first gets the overlap and the other speaker loses words. A
**speaker-conditioned** model works the other way round: bolo diarizes, then
the model transcribes the whole recording once per speaker, told every 20 ms
whether that speaker is talking alone, someone else is, both are, or nobody
(the *STNO* mask: silence, target, non-target, overlap). No audio is cut and
none is separated.

The first such models are BUT-FIT's **DiCoW v3.3** (`dicow-v3.3`,
CC-BY-4.0) and **SE-DiCoW** (`se-dicow`, Apache-2.0, with self-enrolment): Whisper
large-v3-turbo with frame-level diarization-dependent transformations in front
of the encoder, fine-tuned on AMI, NOTSOFAR-1 and LibriMix. They run in the
bolo-torch plugin (`pip install "bolo-torch[dicow]"`, PyTorch and
transformers 4.55; [plugins.md](plugins.md)). Use one as the `model` of a
diarized transcription (`diarize=true` or `response_format=diarized_json`):

* **Diarizer.** `nemotron-3-diarization`, unless the request names another
  one (`diarization_model`); the catalog option `diarization_model` sets it.
* **Conditioning.** By default the model is told who talks when from the
  diarizer's final speaker turns, 0 or 1 per speaker and 20 ms frame, which
  is what BUT's own inference pipeline feeds it. This works with any
  diarizer, including the clustering ones that give no frame probabilities.
  The catalog option `conditioning: probabilities` passes Nemotron's soft
  per-frame probabilities instead (measured below: not better).
* **Output.** One segment list per speaker, merged in time order; segments
  of different speakers may overlap in time, and a segment is marked
  `overlap` when the diarization has another speaker in it. The text is
  lower-case, without punctuation and with contractions spelled out ("we
  are", "do not"): the normalised style of the model's training references.
  No word timestamps. A speaker with less than 0.2 s of speech is not
  decoded.
* **Not combined with** `separate_overlaps` (400). Without diarization the
  model transcribes like an ordinary Whisper large-v3-turbo.

**Checked against the reference implementation.** On two AMI excerpts (a 2-
and a 4-speaker one, Nemotron's turns) the plugin's STNO masks were identical
to those of `DiCoWPipeline.get_diarization_mask` / `get_stno_mask` from
`BUTSpeechFIT/DiCoW` (0 of 13,500 frame values differ; this needs the turns
put on the 20 ms grid with the reference's own rounding, which an earlier
10 ms grid did not do), and every speaker's transcript was word for word the
same as the reference pipeline's decode of the same model (6 of 6 speaker
streams).

**Measured** (2026-11-14, `docs/measurements/dicow-conditioned.json`; 4 vCPU
Xeon @ 2.1 GHz with AMX, no GPU; bolo's models at 2 threads, the plugin at 2
threads, DiCoW in bfloat16). Data: the 24 AMI excerpts and the 44 synthetic
two-voice mixes of "Measured: Nemotron against clustering" above (the mixes
rebuilt with the same texts, voices and layout; bolo's two pipelines re-run on
exactly these files). cpWER with MeetEval 0.4.3 and Whisper's English
normaliser as above, pooled; 95 % intervals by paired bootstrap over files.
Each system's own hypothesis text is normalised the same way, so DiCoW's
missing punctuation and spelled-out contractions do not count against or for
it.

| audio | files | pyannote + whisper-base (default) | Nemotron + whisper-base | **Nemotron + DiCoW v3.3** | relative to default (95 % CI) | files better / worse | worst file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMI excerpts | 24 | 69.4 % | 61.2 % | **26.0 %** | **-62 % (-66 to -59 %)** | 24 / 0 | -24.0 pt |
| synthetic, all | 44 | 55.4 % | 25.4 % | **6.6 %** | **-88 % (-91 to -84 %)** | 42 / 1 | +2.8 pt |
| · 16 kHz cross-gender | 8 | 20.0 % | 24.4 % | 5.1 % | -75 % | 7 / 0 | 0.0 pt |
| · 16 kHz same-gender female | 6 | 68.7 % | 19.3 % | 7.9 % | -89 % | 5 / 1 | +2.8 pt |
| · 16 kHz same-gender male | 6 | 50.4 % | 21.2 % | 6.3 % | -88 % | 6 / 0 | -19.8 pt |
| · 8 kHz cross-gender | 8 | 46.6 % | 30.1 % | 3.9 % | -92 % | 8 / 0 | -18.6 pt |
| · 8 kHz same-gender female | 6 | 84.3 % | 24.4 % | 3.5 % | -96 % | 6 / 0 | -72.8 pt |
| · 8 kHz same-gender male | 6 | 84.8 % | 27.3 % | 6.9 % | -92 % | 6 / 0 | -67.3 pt |
| · Hindi, 16 / 8 kHz (one pair; whisper-base rows use faster-whisper-tiny-hindi) | 2 + 2 | 23.9 / 59.1 % | 27.3 / 39.2 % | 14.8 / 26.1 % | | 4 / 0 | |

tcpWER (5 s collar, AMI word times as reference) moves with cpWER: 71.6 /
62.9 / 27.2 %, so the segment times are sane. The two left columns
reproduce the AMI numbers of the Nemotron section exactly; the synthetic mixes
were rebuilt (Kokoro is not bit-exact), so those columns differ slightly from
the table above.

Reading:

* **Every AMI excerpt got better, by at least 24 points.** Most of the gain
  needs both halves: the same model conditioned on the clustering diarizer
  (`pyannote-3.0-eres2net`'s turns) reached 47.6 % (-31 %, 23 of 24 better,
  none worse), Nemotron's turns 26.0 %. On the synthetic mixes it removes
  the cases neither diarizer fixed: 8 kHz μ-law and same-gender voices end
  at 3.5-7.9 %. A separate run
  was the control for the bigger recogniser alone (the same weights run
  turn by turn without conditioning): -15 %, not significant, on four of
  these excerpts.
* **Hindi:** one synthetic pair (2 files per rate), 14.8 / 26.1 % against
  23.9 / 59.1 % (`language=hi`); the model was not trained on Hindi
  multi-talker speech. Too small to rely on.
* **Probabilities instead of turns** (`conditioning: probabilities`, AMI):
  26.4 % against 26.0 % (+1.4 %, 95 % CI -2.7 to +5.9 %, 9 files better, 11
  worse). Not better, so the turns stay the default; they also work with
  every diarizer.
* **Limits of the evidence.** 45 s excerpts and ~30 s synthetic mixes, not
  whole meetings or real calls; 24 + 44 files. TS-ASR models are reported to
  be worse than their base model on clean single-speaker speech (research
  note, section 7); that was not measured here, and bolo does not route
  single-speaker audio elsewhere: choose the model per request.

**Speed** (idle machine, four AMI excerpts of 45 s with 3-4 speakers after
one warm-up file; diarization included): pyannote + whisper-base RTF 0.25,
Nemotron + whisper-base 0.13, **Nemotron + DiCoW 0.45 at 2 threads and 0.26
at 4 threads** (bfloat16). The decode is one pass per speaker, so the cost
grows with the number of speakers. In float32 (`dtype: float32`, the default
on CPUs without AMX or AVX512-BF16) the same files took RTF 1.41 at 2 threads
instead of 0.46, with the same accuracy (29.6 against 29.4 % on three
files). Loading the plugin and the model took about 4.5 s with the files in
the page cache; the plugin process holds about 3 GB (bfloat16). The research
note's RTF 2-3 was measured on the same machine type while two other jobs
were running. No GPU was used.

**SE-DiCoW** (`se-dicow`), the same way on the same files: AMI 25.3 %
(-64 %, 95 % CI -67 to -60 %, all 24 excerpts better; tcpWER 26.5 %),
synthetic 7.9 % (43 of 44 better). Against DiCoW v3.3 it is slightly better
on AMI and on the 16 kHz English mixes (3.0-4.8 % against 5.1-7.9 %), and
it failed once: on one 16 kHz Hindi mix it repeated one word for the last
6 s of a speaker's stream (greedy decoding has no fallback against such
loops), which put that pair at 78 %. RTF 0.59 at 2 threads on the four
timing excerpts (one more cross-attention block over each speaker's
self-enrolment window), measured in a run of its own, not the idle timing
run. Its self-enrolment windows and transcripts were identical to BUT's
pipeline on the two reference excerpts.

### DiCoW without PyTorch (`dicow-v3.3-onnx`, core)

The same DiCoW v3.3 also runs in bolo's core, in ONNX Runtime with NumPy and
no PyTorch (`bolo/engines/dicow_onnx.py`, engine `onnxruntime`, kind
`dicow_onnx`). It is used exactly like `dicow-v3.3` (speaker-conditioned,
`nemotron-3-diarization` by default, conditioned on the turns, no
`separate_overlaps`).

**What the port reproduces.** DiCoW's decoder is stock Whisper; what differs
is in the encoder (the FDDT layers that take the STNO mask) and in the model's
own generation code. The model also has a CTC head (joint CTC/attention
decoding), but its published generation config sets `ctc_weight: 0` and the
plugin does not turn it on, so it is not exported. The engine reproduces, step
by step: Whisper's 128-band log-mel (short input padded to 30 s, longer input
processed whole); the STNO masks of the plugin (the same functions; a plugin
test checks they agree); transformers 4.55's long-form loop (30 s windows from
`seek`, the features of a short last window padded with zeros and its mask
with silence); greedy search with the suppress-token list and Whisper's
timestamp rules, plus DiCoW's exception that end-of-text may be the first token
of a window; DiCoW's own `_retrieve_segment` (cutting a window into segments
and moving `seek`, with a different rule from transformers' when a window has
no pair of timestamps); language detection on the first window when no
language is given; and byte-level BPE decoding with transformers' space
clean-up.

**Checked against the plugin.** With the fp32 graphs, on the two reference
excerpts (ES2004a_00355 with 4 speakers, IS1009a_00275 with 2; Nemotron's
turns) every speaker's transcript and every segment time was identical to the
plugin's in float32 (6 of 6 speaker streams); features differ from
transformers' by at most 3e-5, and the token decoder agreed with transformers'
tokenizer on every segment. The int8 graphs change some words: 2 of the 6
streams differ (4 of 6 when every encoder matrix was quantised, for example
"a sign of it rather" for "the same figure out").

**Measured** (2026-11-14, `docs/measurements/dicow-onnx.json`; same machine,
data and scoring as the table above; int8 = `encoder.int8.onnx` +
`decoder.int8.onnx`, dynamic quantisation of every weight matrix):

| system | AMI, 24 excerpts | synthetic, 40 English mixes | synthetic, all 44 |
| --- | --- | --- | --- |
| `dicow-v3.3` (torch plugin, bfloat16; table above) | 26.0 % | 5.5 % | 6.6 % |
| ONNX fp32 | (2 excerpts only) | 5.3 % | 6.6 % |
| **ONNX int8 (`dicow-v3.3-onnx`)** | **26.0 %** (7 files better, 7 worse; 95 % CI -2 to +3 %) | **6.0 %** | **7.8 %** (95 % CI -7 to +47 %) |
| ONNX int8, every MatMul quantised (rejected) | 26.2 % | 8.3 % | 9.9 % |

Reading:

* **int8 matches the plugin on AMI** and is within 0.6 points on the English
  mixes (0.8 points above the fp32 graphs, which match the plugin). Over all
  44 mixes it is 1.2 points worse, mostly from one 8 kHz Hindi mix (+37.5
  points); Hindi was never a strength of the model. The synthetic mixes were
  rebuilt once more for this run (Kokoro is not bit-exact; 34 of 44 files have
  the same length), so the torch column there is not paired on identical audio;
  ONNX fp32 against int8 is.
* **Which matrices stay fp32.** Quantising every weight matrix of the encoder
  cost 2-3 points on the mixes (first attempt, last row). The second
  feed-forward matrix of each encoder layer (`fc2`), whose input has
  outlier channels after the GELU, is kept in fp32; that recovered most of it.
  8-bit block-wise weight-only quantisation (`MatMulNBits`) was as accurate as
  fp32 but no faster than fp32 here.
* **Speed** (four 45 s AMI excerpts, the torch timing files, diarization
  included): RTF 1.0 at 2 threads and 0.73 at 4 threads, measured while other
  jobs shared the 4 vCPUs (load average 2-5), so upper bounds; a decode-only
  rerun of the 4-speaker excerpt at 4 threads took 34 s (RTF 0.76), the
  2-speaker one 16 s (0.35). The plugin in bfloat16 with AMX was faster on an
  idle machine (0.45 / 0.26); in float32, the plugin's default on CPUs without
  AMX or AVX512-BF16, it took 1.41 at 2 threads, slower than int8 here. The
  evaluation process (Nemotron plus int8 DiCoW, 4 speakers) held about 4.7 GB.
* **Small sample.** 24 excerpts of 45 s and 44 synthetic mixes; one machine;
  no idle-machine timing; fp32 was scored on two AMI excerpts only.

**Files.** `server/tools/export_dicow_onnx.py MODEL_DIR OUT_DIR` exports
`BUT-FIT/DiCoW_v3_3` (revision `c34b64d9…`) to `encoder.onnx` (input
features `[batch, 128, 3000]` and STNO mask `[batch, 4, 1500]`; outputs the
decoder's cross-attention keys and values) and `decoder.onnx` (token ids, the
self-attention cache and the cross keys/values; logits of the last position
and the extended cache; one graph for the first and every later step), then
`encoder.int8.onnx` / `decoder.int8.onnx`, plus the tokenizer and config files
and `dicow_onnx.json` (sha256 of every file). It needs PyTorch, transformers
4.55, `onnx` and `onnxruntime`; bolo does not. The int8 export is 1.5 GB (fp32
3.2 GB). Engine options: `encoder` / `decoder` (file names; default the int8
graphs, else the fp32 ones), `batch` (speakers decoded together, 4),
`skip_silent_s` (0.2).

**Hosting.** No public ONNX conversion of DiCoW existed on Hugging Face
(searched 2026-11-14), so bolo publishes its own export as
`codebucketsolutions/dicow-v3.3-onnx`. The repository's "Publish a converted
model" workflow rebuilds the export from the pinned upstream revision and
uploads it only when every file's SHA-256 equals the one recorded in
`docs/measurements/dicow-onnx.json`, so the published files are byte for byte
the ones the measurements above were made with; the catalog entry pins the
upload's revision and is `validated: cpu` on that basis. A local export can
still be registered: `bolo pull ./dicow-onnx` detects the kind from
`dicow_onnx.json` and gives the model the entry's features and options
(speaker-conditioned, `diarization_model: nemotron-3-diarization`;
`server/tests/test_real.py::test_dicow_onnx_transcribes_both_voices_of_an_overlap`
runs one with `BOLO_DICOW_ONNX_DIR` set). The README the script writes keeps the
CC-BY-4.0 attribution and asks users to cite the two DiCoW papers.

## Overlapping speech

When two people talk at once, diarization marks the stretch (`overlap: true`)
but the recogniser hears one mixture and usually writes only the louder
voice. `separate_overlaps=true` (api.md, "Separating overlapping speech")
gives each speaker in an overlap a track of their own first. (A
diarization-conditioned model, "Diarization-conditioned recognition" above,
does far better on the same audio, at a much higher CPU cost.) **It is off by
default and stays opt-in**: in our evaluation it improved the transcript of
16 kHz English two-speaker audio with clearly different voices, but it never
met the cost limit we set for turning it on by default, and it did not help
on telephone audio or on AMI far-field meeting recordings.

**How.** The default separator is `voxlock-int8` (VoxLock, Apache-2.0,
77 MB installed, run by sherpa-onnx 2.0.1, no PyTorch). It is a
*target-speaker* model: given about 3 s of one voice it keeps that voice and
removes the rest. bolo takes each speaker's sample from their own speech
outside overlaps, separates each overlap plus 2 s on each side once per
speaker, and transcribes each speaker's part of that window from their own
track. VoxLock was trained on English (LibriSpeech and AMI), is causal
(streaming-capable, so it leaves some offline quality unused), and is a 2026
model from a single author: the numbers below are ours, not its card's. The
Python binding of sherpa-onnx only takes the voice sample as a file, so each
speaker costs an extractor build (about 1.9 s here) per request.

**Evaluation** (2026-11-13, `docs/measurements/overlap-separation.json`;
4 vCPU Xeon @ 2.1 GHz shared with other jobs, load average 4-7, 2 threads,
so times are pessimistic). Synthetic two-voice mixes made with bolo's own
voices (12 English pairs: 6 cross-gender incl. Piper amy, 3 female and 3
male same-gender pairs; 1 Hindi pair): A alone, B alone, A's long sentence
with B starting before A finishes (1, 2.5 or 5 s of overlap), A alone; B at
0 dB for every overlap length and at -5 / +5 dB for 2.5 s (5 mixes per
pair, about 32 s each, 8 % overlap on average). A telephone copy of every
mix (300-3400 Hz, 8 kHz, G.711 μ-law). AMI: 24 excerpts of 45 s from the
official test meetings ES2004a, IS1009a, TS3003a and EN2002a (single
distant microphone, CC BY 4.0), chosen for the most overlap (12-65 %).
Recognisers: `whisper-base` for English, `faster-whisper-tiny-hindi` for
Hindi; diarization `pyannote-3.0-eres2net`, everything at default settings.
cpWER = concatenated minimum-permutation WER (same definition as MeetEval;
implemented in the scratch scorer, MeetEval itself was not installed);
"overlap WER" = the two sentences involved in the overlap (AMI: reference
words within 0.5 s of an annotated overlap); "solo" = the rest. 95 %
intervals: paired bootstrap over mixes.

| audio | mixes | cpWER without → with | relative (95 % CI) | overlap WER | solo WER change | worst mix | extra RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 kHz English, all | 60 | 29.1 % → 23.2 % | -20 % (-13 to -31 %) | 34.7 % → 25.4 % (-27 %) | -0.2 pt | +3.0 pt | 0.23 |
| · cross-gender pairs | 30 | 14.7 % → 6.1 % | -59 % (-48 to -67 %) | 21.1 % → 7.4 % (-65 %) | 0.0 pt | 0.0 pt | 0.31 |
| · same-gender, male | 15 | 37.8 % → 32.1 % | -15 % (-5 to -30 %) | 39.4 % → 29.7 % (-25 %) | +1.1 pt | +3.0 pt | 0.20 |
| · same-gender, female | 15 | 49.3 % → 48.5 % | -2 % (+0.3 to -5 %) | 57.4 % → 57.0 % | -1.9 pt | +1.0 pt | 0.12 |
| · 1 s overlaps | 12 | 23.4 % → 21.9 % | -6 % (-1 to -25 %) | 25.0 % → 22.6 % (-10 %) | 0.0 pt | +0.9 pt | 0.23 |
| · 5 s overlaps | 12 | 39.6 % → 26.4 % | -33 % (-16 to -59 %) | 48.6 % → 28.4 % (-41 %) | -2.1 pt | 0.0 pt | 0.38 |
| · second voice 5 dB quieter | 12 | 23.5 % → 21.3 % | -9 % (-0.5 to -27 %) | 31.0 % → 26.7 % (-14 %) | +1.8 pt | +3.0 pt | 0.17 |
| 16 kHz Hindi (one pair) | 5 | 19.1 % → 7.5 % | -61 % (-50 to -66 %) | 24.9 % → 6.8 % (-73 %) | 0.0 pt | -3.4 pt | 0.61 |
| 8 kHz μ-law English | 60 | 64.0 % → 64.9 % | +1 % (+4 to -1 %) | 71.3 % → 72.5 % | +0.7 pt | **+26.3 pt** | 0.09 |
| 8 kHz μ-law Hindi | 5 | 54.7 % → 59.5 % | +9 % (+32 to -3 %) | 63.4 % → 64.8 % | +5.1 pt | +14.8 pt | 0.31 |
| AMI far-field meetings | 24 | 72.3 % → 72.5 % | +0.4 % (+4 to -3 %) | 84.0 % → 83.6 % | -0.8 pt | **+14.5 pt** | 0.35 |

Notes. In 13 of the 60 English mixes nothing was separated because
diarization found no overlap (6 of them same-gender female pairs, which it
also tended to merge into one speaker: separation cannot fix that). On telephone
audio diarization found the overlap in fewer than half the mixes, and where
VoxLock ran it often damaged speech (one mix lost 26 points). On AMI the
excerpts are hard for `whisper-base` to begin with (72 % cpWER), in 13 of 24
excerpts at least one speaker had less than 3 s of clean speech for a voice
sample (`partial`),
and separation made no measurable difference overall while one excerpt got
14.5 points worse. The Hindi result is one voice pair (5 mixes) of an
English-trained model: promising, too small to rely on. RTF here is the
extra time of the whole request (separation plus the second decoding)
divided by the audio length (the Hindi recogniser is slower, hence its
0.61); the separation step costs about 1.9 s per
enrolled speaker plus 0.46 s per second of window and speaker, so a
1-minute two-person recording with three overlaps costs roughly 20 s extra
at 2 threads on this machine.

**Decision rule** (set before the evaluation; all must hold to turn it on by
default for a class of audio): (1) cpWER at least 10 % better with the 95 %
interval excluding zero, on AMI and on the synthetic set; (2) overlap WER at
least 20 % better; (3) solo WER no more than 0.5 pt worse; (4) extra RTF at
most 0.15 at 2 threads; (5) no mix more than 5 pt worse.

| audio | (1) | (2) | (3) | (4) | (5) | default |
| --- | --- | --- | --- | --- | --- | --- |
| 16 kHz English, all synthetic | pass | pass | pass | **fail** (0.23) | pass | off |
| 16 kHz English, cross-gender | pass | pass | pass | **fail** (0.31) | pass | off |
| 16 kHz English, same-gender | **fail** (female) | **fail** (female) | **fail** (male +1.1) | **fail** | pass | off |
| 16 kHz Hindi (one pair) | pass | pass | pass | **fail** | pass | off |
| 8 kHz telephone (μ-law) | **fail** | **fail** | **fail** | pass | **fail** | off; refused unless `separation_allow_narrowband=true` |
| AMI far-field meetings | **fail** | **fail** | pass | **fail** | **fail** | off |

Read strictly, rule (1) asks for the gain on AMI as well, and AMI failed, so
no class of audio qualifies for on-by-default; the table shows where the
quality rules alone were met.

So: worth turning on for **16 kHz recordings of two (or a few) clearly
different voices that talk over each other for a second or more**
(interviews, podcasts, two-person calls recorded wideband), if 0.2-0.3 extra
RTF is acceptable. Not for telephone audio, not for distant-microphone
meetings, and of little use when the voices are similar.

**Context and voice-sample length** (12 English pairs, 2.5 s overlap, 0 dB,
other sentences): context 0 / 1 / 2 / 4 s gave cpWER 20.5 / 20.2 / 20.8 /
22.0 % (without separation 28.2 %), so 2 s is kept; a 3 s instead of 10 s
voice sample gave 20.4 % (VoxLock uses the loudest 3 s of the sample
anyway).

**Blind separators through the PyTorch plugin.** `separation_model` may name
a blind separator served by `plugins/bolo-torch`; bolo gives its tracks to
the diarized speakers by voice similarity in the overlap plus each track's
share of the speakers' solo speech in the window. Measured here through bolo:

| separator | mixes (16 kHz English, 0 dB) | cpWER without → with | relative (95 % CI) | overlap WER | worst mix | extra RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `voxlock-int8` (sherpa-onnx) | 36 | 32.8 % → 26.2 % | -20 % (-11 to -34 %) | 38.0 % → 27.9 % | +0.9 pt | 0.26 |
| `sepformer-whamr16k` (bolo-torch, CPU PyTorch) | 36 | 32.8 % → 26.2 % | -20 % (-11 to -34 %) | 38.0 % → 28.0 % | **+8.7 pt** | 0.82 |
| `pixit-ami` (bolo-torch, gated), pipeline defaults ¹ | 36 | 30.4 % → 26.7 % | -12 % (-4 to -23 %) | 34.1 % → 28.1 % | +6.8 pt | 1.44 |
| `pixit-ami`, leakage removal off, 2 speakers ¹ | 36 | 30.4 % → 25.8 % | -15 % (-6 to -26 %) | 34.1 % → 26.7 % | +5.8 pt | 1.85 |

SepFormer matched VoxLock's quality on these mixes but cost three times as
much and failed one mix badly; its weights come from WHAMR! (WSJ0-derived,
an LDC licence: a provenance risk for commercial use).

¹ PixIT rows (2026-11-13, later run): the TTS cache had been cleared, so the
mixes were synthesised again (same design, 12 English pairs × 0 dB × 1 / 2.5 /
5 s overlap, but some sentences and lengths differ), and the baseline is that
run's own; the rows are comparable with each other, not exactly with the rows
above. pyannote.audio 4.0.7 in the plugin; the two PixIT runs shared the
machine for part of the time (load average up to 8.7), so its extra RTF is
pessimistic. PixIT helped (cpWER -12 % with defaults,
-15 % with leakage removal off; overlap WER
34.1 % → 26.7 %), most for
cross-gender pairs (-25 %) and same-gender male
pairs (-26 %), not for same-gender female pairs
(-0 %); Hindi pair (3 mixes) -29 % /
-24 %. It fails the decision rule on cost (extra RTF
1.4-1.8, against 0.15) and on the worst mix (+6.8 / +5.8
pt), and its gain is smaller than VoxLock's at several times the cost. Not
measured for PixIT: the 8 kHz telephone copies and the AMI excerpts (runs
stopped unfinished), and VoxLock on the regenerated mixes. Its pipeline
defaults zero a speaker's track wherever that speaker is judged silent;
`leakage_removal: false` (a bolo-torch option) did better here. A
check with perfect tracks handed over in the wrong order confirmed that
bolo gives each diarized speaker their own track, aligned to the sample.

**SepFormer without PyTorch** (2026-11-21). SpeechBrain's SepFormers trained on
WSJ0-2mix, WSJ0-3mix, Libri2Mix and Libri3Mix run in ONNX Runtime
(`sepformer_onnx`, bolo.engines.speechbrain_onnx): the weights exported
unchanged by `server/tools/export_speechbrain_onnx.py` and published under
`codebucketsolutions/<model>-onnx` with the parity check (the graph's tracks
against SpeechBrain 1.1.2's `separate_batch`: 65 to 97 dB SI-SNR apart, the
same model to float precision). All four are 8 kHz models (SpeechBrain
publishes no 16 kHz version of them); the three-speaker ones are the first
in the catalog. SI-SNR improvement on equal-power mixtures of bolo's three
sample voices (en-long, a man; en-short and hi-short, two women; all starting
together, 3 to 4 s), references band-limited to 8 kHz, best permutation;
RTF through bolo's class, 2 threads, on the shared machine (load average 8
to 15):

| model | voices | card (test set) | man + woman (en) | man + woman (hi) | woman + woman | all three | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sepformer-wsj02mix` | 2 | 22.4 dB | +4.9 | +7.3 | +3.7 | +0.6 | 3.5 |
| `sepformer-libri2mix` | 2 | 20.6 dB | +9.6 | +12.1 | +10.8 | +2.0 | 1.9 |
| `sepformer-wsj03mix` | 3 | 19.8 dB | +6.3 | +7.6 | -1.3 | +2.7 | 3.6 |
| `sepformer-libri3mix` | 3 | 19.0 dB | +5.9 | +11.1 | +7.8 | +12.0 | 3.8 |

A two-voice model given three voices returns two tracks (scored against the
best two voices). The LibriMix models did much better on these voices than
the WSJ0 ones, and `sepformer-libri3mix` is the only one that separated
three voices usefully (+12 dB). Synthetic speech from one TTS engine is not a
recording: this says which model to try first, not what it does on your
audio; it was not run through the cpWER evaluation above. Through
`separate_overlaps` on a three-voice synthetic dialogue the pipeline ran
(`applied`), but the diarizer heard the two women as one speaker, so only
two tracks were used: separation can only help as far as diarization finds
the speakers. The WSJ0 models' training data is LDC-licensed (WSJ0), a
provenance risk for commercial use; LibriMix is built from LibriSpeech
(CC BY 4.0).

## Voice-activity detection

The voice-activity detector finds speech for long-file segmentation (`vad_filter`), for
`POST /v1/audio/speech/timestamps` and for the end of a turn in Realtime sessions. The `default_vad` setting
chooses it for the server (default `silero-vad`); a request to the timestamps route can name another in
`model`, and a Realtime session in `turn_detection.vad_model` (a bolo extension). Every detector takes the same
parameters (threshold, minimum silence, minimum and maximum speech, padding) and gives the same speech segments.

| id | what it is | download | runtime | licence |
| --- | --- | --- | --- | --- |
| `silero-vad` (default) | Silero VAD v4 | 2 MB | sherpa-onnx | MIT |
| `silero-vad-v5`, `silero-vad-v6` | Silero VAD v5, v6.2.3 | 2 MB | sherpa-onnx | MIT |
| `ten-vad`, `ten-vad-int8` | TEN VAD (Agora), sherpa-onnx's export | 0.3 / 0.1 MB | sherpa-onnx | Apache-2.0 **with conditions** (below) ⚠ |
| `marblenet-vad` | NVIDIA Frame-VAD Multilingual MarbleNet v2.0, 91.5 k parameters, third-party ONNX export | 0.4 MB | onnxruntime (`turn` extra) | NVIDIA Open Model License |
| `webrtc-vad`, `webrtc-vad-aggressive` | WebRTC's Gaussian-mixture detector, mode 2 and 3 | none | `webrtcvad` extra | BSD-3-Clause + MIT |
| `energy-vad` | bolo's adaptive energy detector (also the fallback when no neural detector can run) | none | built in | Apache-2.0 |

Downloads are pinned by SHA-256 (Silero v4 and v5 by their repository commit). TEN VAD's licence is
Apache-2.0 plus Agora's conditions: you may deploy it, commercially too, only for your own applications and
their direct end users, not in a way that competes with Agora's offerings or lets third parties build or deploy
applications. That can cover a speech server offered to others; read the licence file (downloaded with the
model) before you use it. MarbleNet's network looks about 1.4 s either side of a frame, so a live stream is
scored with 0.4 s of look-ahead (option `lookahead_s`): its decisions agree with whole-file scoring on 99 % of
frames, and they arrive 0.4 s later than the other detectors' (the table's last but one column).

**Measured** (docs/measurements/vad-detectors.json, `server/tools/measure_detectors.py vad`): bolo's three
sample clips and the console's three cloning samples (synthetic speech from Kokoro and SYSPIN voices, English and
Hindi), each at -20 dBFS, joined with 1 to 3 s of silence into a 74 s stream (54 s of it speech by the reference
below), clean and with added sound: white noise at 10 and 0 dB SNR, pink noise at 5 dB, low rumble (brown noise)
at 0 dB, the speech 30 dB quieter in faint white noise, gliding harmonic tones (a stand-in for music) and clicks
in the gaps between clips. The reference is each clip's 10 ms frames within 40 dB of its peak, with pauses
under 0.3 s filled in. Every detector ran with threshold 0.5, 0.3 s minimum silence, 0.25 s minimum speech and no
padding, fed 20 ms at a time as in a live session. Frame accuracy (share of 10 ms frames classified as the
reference says; 0.73 is what calling everything speech scores):

| detector | clean | white 10 dB | white 0 dB | pink 5 dB | rumble 0 dB | quiet speech | tones in gaps | clicks in gaps | mean of the 7 noisy | start / end reported after | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `energy-vad` | 0.99 | 0.73 | 0.73 | 0.73 | 0.73 | 0.49 | 0.79 | 0.97 | **0.74** | 0.26 s / 0.29 s | 0.0008 |
| `webrtc-vad` | 0.93 | 0.95 | 0.73 | 0.73 | 0.94 | 0.95 | 0.74 | 0.87 | **0.85** | 0.25 s / 0.40 s | 0.0008 |
| `webrtc-vad-aggressive` | 0.94 | 0.94 | 0.73 | 0.87 | 0.95 | 0.38 | 0.76 | 0.89 | **0.79** | 0.27 s / 0.36 s | 0.0008 |
| `silero-vad` | 0.96 | 0.92 | 0.74 | 0.90 | 0.95 | 0.87 | 0.90 | 0.96 | **0.89** | 0.35 s / 0.32 s | 0.0109 |
| `silero-vad-v5` | 0.95 | 0.92 | 0.92 | 0.92 | 0.94 | 0.92 | 0.92 | 0.94 | **0.93** | 0.32 s / 0.38 s | 0.0125 |
| `silero-vad-v6` | 0.93 | 0.93 | 0.92 | 0.93 | 0.93 | 0.93 | 0.92 | 0.93 | **0.93** | 0.30 s / 0.39 s | 0.0146 |
| `ten-vad` | 0.92 | 0.71 | 0.69 | 0.66 | 0.92 | 0.70 | 0.87 | 0.92 | **0.78** | 0.39 s / 0.36 s | 0.0106 |
| `ten-vad-int8` | 0.93 | 0.68 | 0.69 | 0.61 | 0.92 | 0.68 | 0.86 | 0.92 | **0.77** | 0.39 s / 0.36 s | 0.0147 |
| `marblenet-vad` | 0.93 | 0.92 | 0.92 | 0.92 | 0.93 | 0.92 | 0.93 | 0.93 | **0.92** | 0.71 s / 0.73 s | 0.0137 |

"Start / end reported after": median time from the reference start (end) of a clip to the event, in the clean
stream; the end includes the 0.3 s of silence every detector waits for. RTF on one thread of a shared 4-vCPU Xeon.
The JSON has speech recall, false alarms and segment counts per condition.

What it says: the energy detector is perfect on clean synthetic speech and useless in steady noise (it calls
all of it speech). WebRTC is nearly free and copes with moderate noise, but at 0 dB white or pink noise and with
tones it also calls everything speech; mode 3 loses quiet speech. Silero v5 and v6 and MarbleNet hardly moved
across the conditions (0.92 to 0.95); Silero v4, the default, was the best of the neural detectors on clean audio
but fell to 0.74 in white noise at 0 dB, where it missed a third of the speech. TEN VAD missed a third of the
speech in white and pink noise and with quiet speech. So the default stays `silero-vad`, which bolo's endpointing
was tuned with; for noisy lines set `default_vad = "silero-vad-v5"` (or v6, or `marblenet-vad` where 0.4 s more
latency does not matter). The test is small and synthetic (TTS voices, synthetic noise): it shows the direction,
not the size, of the differences on real recordings.

## End-of-turn detection

Realtime sessions with `turn_detection: {"type": "semantic_vad"}` ask an end-of-turn model, at each pause the
voice-activity detector finds, whether the speaker has finished ([voice-agents.md](voice-agents.md), "Semantic
turn detection"). `default_turn` chooses it (default `smart-turn-v3`), a session's `turn_detection.turn_model`
overrides it. Smart Turn v3 listens to the last eight seconds of audio, Smart Turn v2 to up to 16; the Namo models
and TEN Turn Detection read the transcript, which the session decodes at the pause with its transcription model;
`silence-only` is no model (the pause ends the turn). LiveKit's turn detector (licence: only with the LiveKit
Agents framework) is not in the catalog; [models-unsupported.md](models-unsupported.md) says why.

**Smart Turn v2** (`smart-turn-v2`) was published as PyTorch weights only. bolo reads its safetensors file and
computes the network (wav2vec2-base, attention pooling, classifier) in its own NumPy code, so it needs no PyTorch
and no converted or re-hosted file. The reference pads every clip to 16 s; bolo computes the same result over the
real audio only (the padding's contribution to the first layer's normalisation is added analytically and stays one
constant per layer afterwards). On 18 clips (the repository's samples, cuts of them, a 20 s clip, noise) its
probabilities matched Pipecat's model.py run in PyTorch 2.15 with transformers 4.57 to within 4.5e-6. It is heavy:
a wav2vec2-base pass costs several seconds per check on this shared 4-vCPU machine (load average about 13), so it
wants a fast CPU or is a reference rather than a live choice. On the scripted set below it took 5.4 s per check (4
threads, load average 11.3) and was no better than Smart Turn v3.

**TEN Turn Detection** (`ten-turn-detection`) runs on llama-server from a Q4_K_M GGUF that GGUF-my-repo converted
from the official weights (4.7 GB, pinned by SHA-256), started by bolo (`bolo components install llama-server`) or
run by you (setting `tts_servers`). At each pause bolo sends the transcript as the user's message with an empty
system message, as TEN's reference does, asks for one token at temperature 0 with its log-probabilities, and takes
P(finished) + P(wait) over the three labels as the probability that the turn is over (`wait`, "stop talking", also
ends the user's turn). It was tested against a stand-in server only; a 7.6-billion-parameter model does not fit
this machine. To try it on a laptop with a GPU (or 8 GB free for the CPU):

```bash
bolo components install llama-server     # or set llama_server_bin to your own build (CUDA, Metal)
bolo pull ten-turn-detection             # 4.7 GB
BOLO_DEFAULT_TURN=ten-turn-detection BOLO_DEFAULT_REALTIME_STT=moonshine-tiny-en bolo serve
# a Realtime session with turn_detection {"type": "semantic_vad"}: each turn's trace shows the text and the
# probability (turn_detection.checks). Or ask directly:
python server/tools/measure_detectors.py turn --out ten.json ten-turn-detection   # the scripted set (English only)
```

Its licence: Apache-2.0 per the Hugging Face card, plus Agora's conditions in the GitHub licence file (not hosted
on end-user devices; only for your own applications; nothing competing with Agora), hence `commercial_use: false`.

**Measured** (docs/measurements/turn-detectors.json, `server/tools/measure_detectors.py turn`). The repository has
no end-of-turn test data, so the set is scripted for this test: 20 English and 6 Hindi sentences a voice agent
might hear ("Can you book me a table for two at seven tonight?"), each with an unfinished beginning ("Can you book
me a table for"). Audio from Kokoro (two English voices, one Hindi); the unfinished audio is the whole sentence
cut at its quietest 20 ms near the beginning's share of the words (so it has mid-sentence prosody), with 0.2 s of
silence after every clip, as a session hands it over. The text models got the written sentences (finished ones
end in "." or "?") and the same with the final punctuation removed. Accuracy at the 0.5 threshold sessions use,
and AUC (how well the two classes are told apart at any threshold); 40 English and 12 Hindi clips:

| model | judges | English: accuracy | finished / unfinished right | AUC | Hindi: accuracy | AUC | ms per call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `silence-only` | - | 0.50 | 1.00 / 0.00 | 0.50 | 0.50 | 0.50 | 0 |
| `smart-turn-v3` (v3.2, int8, default) | audio | 0.63 | 0.95 / 0.30 | 0.87 | 0.83 | 0.86 | 370 |
| `smart-turn-v3.1` | audio | 0.65 | 0.85 / 0.45 | 0.83 | 0.75 | 0.83 | 330 |
| `smart-turn-v3.0` | audio | 0.70 | 1.00 / 0.40 | 0.92 | 0.50 | 0.78 | 400 |
| `smart-turn-v3-fp32` (v3.2, float32) | audio | 0.80 | 0.95 / 0.65 | 0.88 | 0.83 | 0.81 | 350 |
| `smart-turn-v2` (wav2vec2, bolo's NumPy) | audio | 0.63 | 1.00 / 0.25 | 0.83 | 0.75 | 0.83 | 5,400 |
| `namo-turn-en` | text, as written | **0.98** | 1.00 / 0.95 | 0.97 | - | - | 43 |
| `namo-turn-en` | text, no final punctuation | 0.85 | 0.75 / 0.95 | 0.96 | - | - | 43 |
| `namo-turn-multilingual` | text, as written | 0.48 | 0.45 / 0.50 | 0.46 | 0.42 | 0.44 | 79 |

Milliseconds per call on two threads of the shared 4-vCPU Xeon with its load average at 12.6 (busy; on an idle
machine Smart Turn took 160 to 260 ms, see voice-agents.md); a text model's time does not include the decode the
session makes for it. What it says: every Smart Turn version separates finished from cut-off audio reasonably
(AUC 0.78 to 0.92) but at 0.5 the int8 builds call most cut-off English clips finished; the float32 build was
right on 65 % of them. The English Namo model read the scripted text almost perfectly and an order of magnitude
faster, but it leans on punctuation: without the final "." or "?" a quarter of the finished sentences looked
unfinished, and what a recogniser writes at a pause is neither of those clean cases. The multilingual Namo model
was at chance on this set (its logits stayed close to zero; it reproduces the direction of its model card's four
examples, weakly); do not use it without testing on your own data. The default stays `smart-turn-v3`: it hears
any language without a decode, and the set is too small and too clean to justify a switch; `smart-turn-v3-fp32`
(32 MB) or, for English agents, `namo-turn-en` are worth trying.

**Int8 against float32 on a larger set** (docs/measurements/turn-detectors-large.json, `measure_detectors.py
turn-large`): 60 English and 30 Hindi sentences, in four English and two Hindi Kokoro voices, each whole and cut at
about 35 % and 65 % of its speech (270 clips):

| | English: accuracy | finished / unfinished right | AUC | Hindi: accuracy | finished / unfinished right | AUC | ms per call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smart-turn-v3` (int8) | 0.52 | 0.90 / 0.33 | 0.79 | 0.88 | 0.97 / 0.83 | 0.96 | 290 |
| `smart-turn-v3-fp32` | 0.60 | 0.87 / 0.47 | 0.81 | 0.90 | 0.77 / 0.97 | 0.94 | 288 |

In English the float32 build was right on 18 clips the int8 build got wrong and wrong on 4 it got right (a sign
test gives p < 0.01), better in every voice; in Hindi it was 8 against 6: it caught more cut-off sentences (97 %
against 83 %) but held 23 % of the finished ones open (int8: 3 %), which costs the speaker the full pause before
the answer. It is as fast (the same network; 32 MB instead of 9). Not consistently better, so it is not proposed
as the default; for English agents it is the better Smart Turn.

**In a live session** (docs/measurements/turn-text-realtime.json): a Realtime transcription session with
`moonshine-tiny-en` and `namo-turn-en`, the English sample sent in real time. At each pause the session decoded the
utterance and asked the model (0.5 to 1.3 s of decoding on the busy machine, the model's own call on top). The
recogniser wrote the 60 % cut as "Good morning. could you tell me?", with a question mark, so the model called it
finished (1.00) and the turn was split, as with `server_vad`; Smart Turn kept that turn together
(voice-agents.md). The whole sentence, transcribed "... when my order will arise?", got 0.48 and was held open for
the full 3 s pause. So a text model is only as good as the words and punctuation the recogniser writes at the
pause, and costs a decode per pause.

## Speaker embedding

A speaker embedder turns a stretch of speech into a vector; the cosine of two vectors says how alike the voices
sound. bolo uses one to score cloned voices ([voices.md](voices.md), setting `voice_similarity_model`) and to
answer `POST /v1/audio/speech/embedding` (`model`); diarization pipelines carry their own. Every
`speaker-embedding` entry, and every diarization model (its embedder), is accepted by both.

| id | model | numbers | download | runtime | licence |
| --- | --- | --- | --- | --- | --- |
| `eres2net-voxceleb` | 3D-Speaker ERes2Net, VoxCeleb (the embedder of the default `pyannote-3.0-eres2net`, same file) | 192 | 26 MB | sherpa-onnx | Apache-2.0 |
| `campplus-voxceleb` | 3D-Speaker CAM++, VoxCeleb | 512 | 30 MB | onnxruntime (1.29 or newer), bolo's Kaldi filterbank | Apache-2.0 |
| `campplus-zh-en` | 3D-Speaker CAM++, about 200 k Chinese and English speakers | 192 | 28 MB | sherpa-onnx | Apache-2.0 |
| `wespeaker-campplus-voxceleb` | WeSpeaker CAM++, VoxCeleb | 512 | 29 MB | onnxruntime (1.29 or newer), bolo's Kaldi filterbank | CC-BY-4.0 |
| `wespeaker-resnet34-voxceleb` | WeSpeaker ResNet34, VoxCeleb (the embedder of `pyannote-3.0-wespeaker-resnet34`) | 256 | 27 MB | sherpa-onnx | CC-BY-4.0 |
| `titanet-small` | NVIDIA TitaNet-small (the embedder of `pyannote-3.0-titanet-small`) | 192 | 40 MB | sherpa-onnx | CC-BY-4.0 |
| `ecapa-tdnn-voxceleb` | SpeechBrain ECAPA-TDNN, VoxCeleb 1+2; third-party ONNX export with the front-end inside | 192 | 42 MB | onnxruntime | Apache-2.0 |

**Measured** (docs/measurements/speaker-embedders.json, `server/tools/measure_detectors.py embed`): every pair of
clips scored, same speaker against different speakers. Two sets: (a) the repository's clips (the three samples,
the console's three cloning samples) and three sentences in each of eight Kokoro voices, 30 clips, 37
same-voice and 398 different-voice pairs; (b) real read speech, LibriSpeech dev-clean, 16 speakers with 5
utterances each (the set `tools/calibrate_similarity.py` uses), 160 and 3000 pairs. EER: the equal error rate;
"at 0.5": the share of pairs at or above bolo's "same speaker" line.

| embedder | synthetic: same / different (median) | EER | LibriSpeech: same / different (median) | different p95 | EER | same / different at 0.5 | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eres2net-voxceleb` (and `pyannote-3.0-eres2net`) | 0.83 / 0.08 | 0.3 % | 0.80 / 0.11 | 0.36 | 0.6 % | 99.4 % / 0.1 % | 0.27 |
| `campplus-voxceleb` | 0.85 / 0.10 | 0 % | 0.81 / 0.13 | 0.37 | 0.3 % | 99.4 % / 0.1 % | 0.07 |
| `campplus-zh-en` | 0.91 / 0.25 | 0 % | 0.82 / 0.10 | 0.32 | **0 %** | 100 % / 0.2 % | 0.12 |
| `wespeaker-campplus-voxceleb` | 0.86 / 0.14 | 0.8 % | 0.82 / 0.14 | 0.41 | 0.2 % | 100 % / 0.7 % | 0.07 |
| `wespeaker-resnet34-voxceleb` | 0.92 / 0.53 | 4.8 % | 0.89 / 0.54 | 0.74 | 3.3 % | 100 % / 63 % | 0.15 |
| `titanet-small` | 0.91 / 0.27 | 2.0 % | 0.84 / 0.06 | 0.28 | **0 %** | 100 % / 0.1 % | 0.09 |
| `ecapa-tdnn-voxceleb` | 0.84 / 0.12 | 0 % | 0.76 / 0.10 | 0.31 | 1.3 % | 97.5 % / 0 % | 0.16 |

RTF: embedding time over 732 s of audio on the shared 4-vCPU Xeon (load average about 15, so treat it as
relative). The standalone `eres2net-voxceleb` gave exactly the numbers of the diarization entry it comes from.
CAM++ (all three), TitaNet-small and ECAPA-TDNN told the speakers apart as well as ERes2Net or better; WeSpeaker
ResNet34 separates well but on its own scale (different people around 0.54, so bolo's 0.5 line does not fit it).

The two VoxCeleb CAM++ files first came out close to chance (equal error rate 40 to 43 %) when sherpa-onnx ran
them. The cause is the ONNX Runtime inside sherpa-onnx 2.0.1, version 1.28: the same graph with the same Kaldi
features gives the near-chance embeddings in onnxruntime 1.28.0 and right ones in 1.29, 1.30, 1.31 and 1.32, with
graph optimisations on or off (checked on LibriSpeech dev-clean; ERes2Net and the zh-en CAM++ are not affected).
So these two run in bolo's own ONNX Runtime (1.29 or newer; an older one is refused with a message) with bolo's
Kaldi filterbank (`bolo.audio.features.kaldi_fbank`: 80 bands, Povey window, mean over the recording removed;
WeSpeaker's model gets 16-bit sample values, 3D-Speaker's -1..1), and the numbers above are from that path. The verdicts ("same speaker" at
0.50, "similar" at 0.30) are calibrated for ERes2Net only (`calibrated` in answers is true for
`pyannote-3.0-eres2net` and `eres2net-voxceleb`); the default stays ERes2Net.

## Denoising

`denoise` (configuration.md, "Denoising") runs a speech-enhancement model
before the recogniser: on a transcription, a Realtime session or the live
speech-to-text WebSocket. Every `task: denoise` model of the catalog can be
named; `gtcrn-simple` is the default. Denoisers that were looked at and
cannot run yet are in [models-unsupported.md](models-unsupported.md).

**The default, `gtcrn-simple`**: GTCRN (Rong et al., ICASSP 2024; MIT;
48 thousand parameters, 0.5 MB), the ONNX export sherpa-onnx publishes, at
16 kHz. On one thread of a shared 4-vCPU Xeon it took 0.06 s per second of
audio at light load and 0.12 to 0.13 s with the machine's load average at 17.

**First measurement** (docs/measurements/denoise.json, top level): the
repository's two English samples (47 words of synthetic speech) mixed with
white noise, pink noise, applause and a passing train at 10, 5 and 0 dB SNR,
twelve noisy mixes in all; word error rate over both clips.

| recogniser | clean, off / on | mean of 12 noisy mixes: off | on | on, strength 0.5 |
| --- | --- | --- | --- | --- |
| `moonshine-v2-tiny-en` | 6.4% / 2.1% | 31.7% | 23.4% | 22.9% |
| `whisper-base` | 0% / 0% | 15.4% | 26.8% | 11.7% |

GTCRN at full strength helped the small Moonshine model on 10 of the 12
mixes (applause at 0 dB: 46.8% to 19.1%) and hurt Whisper base on 11 of
them, helping on none (white noise at 5 dB: 17.0% to 76.6%, where Whisper dropped
most of a sentence). Whisper was trained on large amounts of noisy audio
and copes with noise better than with the artefacts enhancement leaves.
Mixing half of the original back (`denoise=0.5`) was the better setting for
both.

### Every denoiser compared

The same test for every runnable denoiser (docs/measurements/denoise.json,
`catalog_comparison`; `server/tools/measure_denoise.py` repeats it against a
running server). Its noise is drawn the same way but not identically to the
first run's, so its "off" error rates differ from the table above (34.4%
against 31.7% for Moonshine, 17.2% against 15.4% for Whisper): compare
denoisers within this table. Word error rate is the mean over the twelve
noisy mixes, at full strength and at strength 0.5; "clean, on" is the
noise-free pair with the denoiser at full strength. CPU time is seconds per
second of audio on one thread (74 s of speech, at the model's own rate; the
PyTorch models in 15 s pieces); the live column is the live stream's
denoiser fed 20 ms chunks (api.md, "Denoising a live stream"). The machine
was shared, with a load average of 12 to 17 throughout: the times are
pessimistic, and the error rates are those of 47 words, so a difference of
two or three points is one word.

| denoiser | licence | rate | download | CPU s per s of audio | live: ms per 20 ms chunk | `moonshine-v2-tiny-en`: on / 0.5 | `whisper-base`: on / 0.5 | clean, on: Moonshine / Whisper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | - | - | - | - | - | 34.4% | 17.2% | 6.4% / 0.0% |
| `gtcrn-simple` | MIT | 16 kHz | 536 KB | 0.051 | 1.8 (p95 3.7) | 23.9% / 22.3% | 22.2% / 11.0% | 2.1% / 0.0% |
| `dpdfnet-baseline` | Apache-2.0 | 16 kHz | 8.7 MB | 0.080 | 2.7 (p95 5.9) | 23.1% / 25.7% | 25.9% / 9.4% | 4.3% / 2.1% |
| `dpdfnet2` | Apache-2.0 | 16 kHz | 10.2 MB | 0.163 | 3.9 (p95 4.5) | 19.7% / 23.6% | 20.2% / 8.9% | 6.4% / 6.4% |
| `dpdfnet4` | Apache-2.0 | 16 kHz | 11.6 MB | 0.232 | 5.8 (p95 9.1) | 21.3% / 23.4% | 19.5% / 8.5% | 6.4% / 2.1% |
| `dpdfnet8` | Apache-2.0 | 16 kHz | 14.5 MB | 0.381 | 16.1 (p95 23.9) | 20.2% / 21.8% | 16.7% / 8.2% | 8.5% / 6.4% |
| `dpdfnet2-48k` | Apache-2.0 | 48 kHz | 10.5 MB | 0.214 | 8.3 (p95 12.8) | 22.2% / 24.5% | 22.7% / 9.4% | 6.4% / 6.4% |
| `dpdfnet8-48k` | Apache-2.0 | 48 kHz | 14.9 MB | 0.518 | 17.6 (p95 25.2) | 20.4% / 22.9% | 15.4% / 9.4% | 4.3% / 0.0% |
| `dpdfnet2-8k` | Apache-2.0 | 8 kHz | 10.2 MB | 0.148 | 3.9 (p95 7.4) | 31.6% / 25.5% | 32.8% / 9.7% | 2.1% / 0.0% |
| `dpdfnet8-8k` | Apache-2.0 | 8 kHz | 14.6 MB | 0.344 | 8.3 (p95 12.3) | 32.3% / 25.9% | 24.3% / 11.7% | 2.1% / 0.0% |
| `deepfilternet3` | MIT OR Apache-2.0 | 48 kHz | 16.1 MB | 0.259 | 5.4 (p95 6.5) | 32.1% / 26.6% | 33.0% / 12.8% | 4.3% / 2.1% |
| `rnnoise` | BSD-3-Clause | 48 kHz | 13.3 MB | 0.068 | 1.7 (p95 2.2) | 38.1% / 27.8% | 39.2% / 13.3% | 4.3% / 0.0% |
| `demucs-dns48` | CC-BY-NC-4.0 ⚠ | 16 kHz | 75.5 MB | 0.194 | 77.4 (p95 103.4) | 31.9% / 26.8% | 36.4% / 9.7% | 6.4% / 0.0% |
| `demucs-dns64` | CC-BY-NC-4.0 ⚠ | 16 kHz | 134.1 MB | 0.294 | 134.7 (p95 167.6) | 24.3% / 23.8% | 20.0% / 10.3% | 6.4% / 0.0% |
| `demucs-master64` | CC-BY-NC-4.0 ⚠ | 16 kHz | 134.1 MB | 0.285 | 133.0 (p95 166.7) | 28.0% / 25.4% | 25.5% / 8.0% | 8.5% / 2.1% |
| `demucs-valentini-nc` | CC-BY-NC-4.0 ⚠ | 16 kHz | 243.3 MB | 3.713 | 1923.4 (p95 2250.7) | 88.5% / 33.2% | 91.0% / 15.2% | 19.1% / 0.0% |
| `sepformer-dns4-16k` | Apache-2.0 | 16 kHz | 105.8 MB | 3.783 | 1594.8 (p95 1846.1) | 28.0% / 26.6% | 21.1% / 11.2% | 6.4% / 0.0% |
| `metricgan-plus` | Apache-2.0 | 16 kHz | 7.6 MB | 0.006 | 3.1 (p95 4.1) | 61.9% / 34.0% | 49.3% / 16.8% | 4.3% / 0.0% |

⚠ = not for commercial use. `elevenlabs-voice-isolator` (cloud) was not run:
no account.

What it shows, on this small synthetic test:

* **At full strength only the largest DPDFNets helped `whisper-base`**, by
  a word or so (`dpdfnet8` 16.7%, `dpdfnet8-48k` 15.4%, against 17.2% off);
  every other denoiser made it worse. **At strength 0.5 every denoiser
  helped it**, the DPDFNets and `demucs-master64` most (8.0% to 9.7%, about
  half the errors of "off").
* **Moonshine gained from most of them at full strength**, the 16 kHz
  DPDFNets most (`dpdfnet2` 19.7%, against 34.4% off and 23.9% for GTCRN).
  RNNoise made it worse (38.1%), and `demucs-valentini-nc` (non-causal,
  trained on VCTK-DEMAND only) wrecked it (88.5%; 33.2% at 0.5).
* The **8 kHz** DPDFNets cut everything above 4 kHz: made for telephone
  audio, they did little for wideband speech at full strength.
* **DeepFilterNet 3 and RNNoise**, tuned for listening, were among the
  weakest at full strength; at 0.5 they helped Whisper less than the others
  (12.8% and 13.3%).
* **SpeechBrain's SepFormer DNS4** (`sepformer-dns4-16k`, Apache-2.0, ONNX Runtime) helped
  Moonshine (28.0%, less than GTCRN) and Whisper at strength 0.5 (11.2%), but it is the slowest denoiser in the
  table after `demucs-valentini-nc` (3.8 s of CPU per second of audio; a live stream cannot keep up).
  **MetricGAN+** (`metricgan-plus`, Apache-2.0, 7.6 MB, the cheapest in CPU time here) was trained to
  please PESQ on VoiceBank-DEMAND and hurt both recognisers badly at full strength (Moonshine 61.9%, Whisper
  49.3%); at 0.5 it did nothing for them. It is labelled for listening, not for use before transcription
  (`features: [listening-only]`): `denoise=true` or `auto` never puts it in front of a recogniser, even when it is
  `default_denoise` (the shipped default, `gtcrn-simple`, is used instead, with a warning in the log); a request,
  a model pin or a pipeline step that names it gets it. Both are SpeechBrain checkpoints exported unchanged to ONNX
  (`server/tools/export_speechbrain_onnx.py`, published under codebucketsolutions/ with parity checks: SepFormer
  52 to 90 dB SI-SNR from SpeechBrain's output, MetricGAN+ about 130 dB). Measured 2026-11-21, load average 6 to 12.
* The **Demucs** denoiser (PyTorch, non-commercial) did no better than the
  DPDFNets and costs more: it has no streaming form here, so a live stream
  enhances every 20 ms chunk with the half second before it, which is
  several times slower than real time. `demucs-valentini-nc` needs 3.7 s of
  CPU per second of audio even for whole recordings.

The default stays `gtcrn-simple`: it is the lightest (0.5 MB, 0.05 s of CPU
per second of audio, 2 ms per live chunk), and at strength 0.5 it helped
both recognisers (Moonshine 22.3%, Whisper 11.0%). For more quality at three
to five times the CPU, `dpdfnet2` or `dpdfnet4` (Apache-2.0, 10 to 12 MB)
did better than GTCRN on both recognisers at full strength and on Whisper at
0.5 (Moonshine at 0.5 was a word worse), and still keep up with a live
stream; none is clearly better enough on 47 words to
replace the default. `bolo recommend --task denoise` lists them all with
this machine's fit: the configured default first, and the denoisers whose
best strength left more than three quarters of the errors (`measured.wer_ratio`
above 0.75 in the catalog: DeepFilterNet 3, RNNoise, `demucs-valentini-nc`)
last. Measure on your own audio before turning denoising on for a model or
the server.

## Language identification

`POST /v1/audio/language` says which language a recording is in, and a
transcription with `model=auto` uses the answer to pick a recogniser
(api.md, "Language identification and routing"; configuration.md,
"Language routing"). These are the `task: lid` entries; each states its
licence, and none is left out for its licence:

| id | Licence (commercial use) | Size | Languages | Runtime | State |
| --- | --- | --- | --- | --- | --- |
| `voxlingua107-ecapa` (default) | Apache-2.0 (yes) | 87 MB | 107 (VoxLingua107) | ONNX Runtime, raw audio in | measured |
| `voxlingua107-ecapa-int8` | Apache-2.0 (yes) | 22 MB | 107 | ONNX Runtime, SpeechBrain filterbanks computed by bolo | measured |
| `ambernet-langid` | NVIDIA NGC Terms of Use (marked no: not an open licence; read the terms) | 117 MB | 107 | ONNX Runtime, raw audio in | measured |
| `whisper-tiny-lid` | MIT (yes) | 103 MB | 99 (Whisper's) | ONNX Runtime on sherpa-onnx's int8 Whisper graphs | measured |
| `whisper-base-lid` | MIT (yes) | 160 MB | 99 | the same | measured |
| `mms-lid-126` | CC-BY-NC-4.0 (no) | 978 MB | 126, Odia among them | ONNX Runtime (Xenova's int8 export of facebook/mms-lid-126) | runnable, not measured (too large for the test machine's 500 MB budget) |
| `mms-lid-4017` | CC-BY-NC-4.0 (no) | 978 MB | 4017, among them Bhojpuri, Maithili, Magahi, Chhattisgarhi, Bodo, Manipuri, Santali, Konkani, Kashmiri | the same | runnable, not measured |
| `voxlect-indic-lid-whisper-small` | OpenRAIL (yes, with use restrictions) | 362 MB | the 22 scheduled languages and English | torch plugin (`bolo-torch[voxlect]`): Whisper small's encoder and Voxlect's head, rebuilt from transformers' classes | measured |
| `voxlect-indic-lid-mms-lid-256` | CC-BY-NC-4.0 (no) | 3.9 GB | the same | torch plugin (`bolo-torch[voxlect]`), MMS-LID-256's wav2vec 2.0 encoder | runnable, not measured (too large) |
| `lid-indic-vaani` | Apache-2.0 (yes) | 83 MB | English, Hindi, Marathi, Tamil, Telugu | torch plugin (`bolo-torch[speechbrain]`), SpeechBrain ECAPA-TDNN | measured |
| `indic-slid-mhubert`, `indic-slid-mhubert-task2` | CC-BY-NC-SA-4.0 (no) | 378 MB each | the 22 scheduled languages (no English) | torch plugin (`bolo-torch[voxlect]`), transformers' HuBERT audio classifier | measured |

Whisper's language detection is used on its own: the encoder hears the
first 30 seconds and the decoder, given only the start-of-transcript
token, scores Whisper's language tokens, which a softmax turns into a
probability per language (what Whisper's own `detect_language` does).
sherpa-onnx has a spoken-language-identification API for the same
models, but it returns only the top language, without probabilities, so
bolo runs the two graphs in ONNX Runtime itself. AI4Bharat publishes no
audio language identifier (IndicLID is for text); the Indian-specific
identifiers on Hugging Face are Voxlect's, a SpeechBrain fine-tune and a
few mHuBERT and wav2vec 2.0 fine-tunes, all above except the ones that
state no licence ([models-unsupported.md](models-unsupported.md)). The
Voxlect models are rebuilt by bolo-torch from transformers' encoder classes
and the checkpoints' weights (the LoRA adapters merged in), since their
own code is not a package.

**Measured** (docs/measurements/lid.json; one thread of a shared 4-vCPU
Xeon, load average 14 to 18). The repository has recorded test speech
only in English and Hindi, so the Indian languages were **synthesised
with bolo's own Indian voices**: three everyday sentences per language
(2.4 to 6.9 s), one voice per language (Piper, Mimic 3, SYSPIN and
MMS-TTS voices). Synthetic speech from one voice is easier than real
recordings; these numbers show which models tell the languages apart, not
their accuracy on your audio.

| model | repository clips (11, en and hi) | Indian clips (39) | hi | bn | ta | te | mr | gu | kn | ml | pa | ur | ne | or | as | RTF (median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `voxlingua107-ecapa` | 11/11 | 33/39 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 | 0.106 |
| `voxlingua107-ecapa-int8` | 11/11 | 33/39 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 | 0.135 |
| `ambernet-langid` | 10/11 | 33/39 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 | 0.104 |
| `whisper-base-lid` | 11/11 | 24/39 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 0/3 | 3/3 | 3/3 | 1/3 | 1/3 | 0/3 | 0/3 | 0.608 |
| `whisper-tiny-lid` | 11/11 | 19/39 | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 | 0/3 | 2/3 | 3/3 | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0.301 |
| `voxlect-indic-lid-whisper-small` | 6/11 | 35/39 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 0.558 |
| `indic-slid-mhubert` | 0/11 (no English) | 31/39 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0.231 |
| `indic-slid-mhubert-task2` | 0/11 (no English) | 31/39 | 0/3 | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0.226 |
| `lid-indic-vaani` | 8/11 | 11/39 (knows 4 of the 13) | 3/3 | - | 3/3 | 3/3 | 2/3 | - | - | - | - | - | - | - | - | 0.130 |

* **Odia** is not among the languages of VoxLingua107, AmberNet or Whisper:
  those models answer Bengali or Assamese. The Voxlect model knew all three
  Odia clips; the mHuBERT ones, which list Odia, still answered Bengali.
* **Urdu and Hindi**: the VoxLingua107, AmberNet and Whisper models heard
  Urdu as Hindi (Whisper base got one of three); the Voxlect and mHuBERT
  models, trained on Indian speech, did the opposite and heard every Hindi
  clip as Urdu, with the Urdu clips right.
  Spoken everyday Urdu and Hindi are close to one language; the difference
  is mostly in the script, which speech does not have. Route both to a
  recogniser that writes the script you want (`language_routes`).
* The two VoxLingua107 graphs agreed on every clip; the int8 one is a
  quarter of the size and, here, a little slower (bolo computes its
  filterbanks in NumPy).
* AmberNet heard a 1961 recording of English (`en-fellow-americans`,
  old and noisy) as Portuguese; the VoxLingua107 model was right at 0.81.
* Whisper's detector confused Gujarati, Kannada, Nepali and Assamese
  with neighbours and with unrelated languages on these short clips, and
  always pays for 30 s of encoder work, so it is the slowest per clip.
  It is the identifier to use when you want exactly Whisper's view of the
  language (it is what a Whisper recogniser would detect by itself).

* The Voxlect model (Whisper small) is the best on the Indian clips apart
  from Hindi, the slowest after Whisper base (it always encodes 15 s), and
  the only one that knows Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri
  and Santali besides MMS-LID-4017; none of those could be tested here
  (there are no voices for them). The mHuBERT models answer with low
  confidence (0.2 to 0.3 for the top language), so routing with
  `lid_min_probability` 0.5 would always fall back to `default_stt`.

`voxlingua107-ecapa` is the default (`default_lid`): permissive licence,
the best results here and about a tenth of a second per second of audio
on one core. Routing with it in front of `whisper-tiny`, with
`indicconformer-ta-community` installed: the Tamil clips went to the
Tamil recogniser (which wrote both clips word for word, punctuation aside, where `whisper-tiny` wrote
broken Tamil), English and Hindi stayed with `whisper-tiny`, and the
`x-bolo-route` header named `indicconformer-hi-community` and
`indicconformer-bn-community` as the recognisers to pull for Hindi and
Bengali. The Odia clip, heard as Bengali, was routed as Bengali: a
language the identifier does not know is routed as its nearest neighbour.

## Sound events and speech emotion

**Taggers** (task `audio-tagging`, `POST /v1/audio/tags`): every model scores each of AudioSet's 527 classes
on its own (sigmoids, not a softmax), so a recording can be speech and music at once. They hear 30 s at most;
bolo cuts longer recordings into 10 s windows and keeps each class's highest probability. CED (Xiaomi,
Apache-2.0) is a ViT distilled from an ensemble; the Zipformer taggers come from icefall (Apache-2.0). All run on
sherpa-onnx in 8 bits, from the exports k2-fsa publishes.

**Measured** (docs/measurements/audio-classifiers.json, `server/tools/measure_classifiers.py`): the 13 test clips
k2-fsa ships with its taggers (113 s) and emotion2vec's own example clip (4 s), one request at a time on the shared
4-vCPU Xeon (load average about 21, so the RTFs are pessimistic and only comparable with each other). The top tag of
four of the clips: 1.wav is a cat, 3.wav a capella singing, 7.wav a smoke alarm, the last an actor's line.

| id | model | RTF | 1.wav | 3.wav | 7.wav | speech clip |
| --- | --- | --- | --- | --- | --- | --- |
| `ced-tiny` | CED tiny (5.5 M) | 0.034 | Animal 0.90 | A capella 0.74 | Smoke detector, smoke alarm 0.77 | Speech 0.70 |
| `ced-mini` | CED mini (9.6 M), the default | 0.039 | Animal 0.91 | A capella 0.72 | Smoke detector, smoke alarm 0.94 | Speech 0.76 |
| `ced-small` | CED small (22 M) | 0.052 | Animal 0.90 | A capella 0.70 | Smoke detector, smoke alarm 0.84 | Speech 0.78 |
| `ced-base` | CED base (86 M) | 0.105 | Animal 0.91 | A capella 0.75 | Smoke detector, smoke alarm 0.85 | Speech 0.84 |
| `zipformer-small-audio-tagging` | Zipformer small (icefall) | 0.126 | Animal 0.95 | Music 0.79 | Smoke detector, smoke alarm 0.79 | Speech 1.00 |
| `zipformer-audio-tagging` | Zipformer (icefall) | 0.201 | Cat 0.95 | Music 0.86 | Smoke detector, smoke alarm 0.37 | Speech 1.00 |

The four CEDs agree with one another and with the Zipformers on every clip's top class; the Zipformers are
more confident (speech at 0.99 against CED's 0.70 to 0.84) and four to six times slower. The browser demo runs CED
tiny and mini with the same features computed in JavaScript (within 0.02 of sherpa-onnx's probabilities on these
clips).

**Emotion** (task `emotion`, `POST /v1/audio/emotion`): emotion2vec+ base and large (Ziyang Ma et al.; FunASR),
nine classes, in ONNX Runtime from third-party exports of the authors' weights. Their licence is the FunASR Model
Open Source License Agreement 1.1: use, copying, changes and sharing are allowed with attribution and the model names
kept; the model is "provided for reference and learning purposes only", the licence ends for anyone who disparages
the software, and later revisions apply automatically. It says nothing about commercial use either way: read it
before relying on it commercially.

| id | RTF | load | emotion2vec's example clip |
| --- | --- | --- | --- |
| `emotion2vec-plus-base` | 0.92 | 4.2 s | angry 1.00 |
| `emotion2vec-plus-large` | 0.84 | 14.0 s | angry 1.00 |

Both call emotion2vec's example (an actor's line) "angry". On the non-speech clips the answer means
nothing (a cat comes out "sad"): run the tagger first, or send speech only. An emotion model guesses how someone
sounds, not how they feel; acted, synthesised and telephone speech and other ways of speaking move it.

**SenseVoice** tags emotions and sounds while it transcribes; bolo returns them with `verbose_json`
([api.md](api.md#speech-emotion-post-v1audioemotion)). On the clips above it said "music" for the singing and
"speech" for the actor's line, and gave no emotion for either (its `EMO_UNKNOWN`).

## Word timestamps

| Models | Word times | Word confidence (`probability`) |
| --- | --- | --- |
| faster-whisper (CTranslate2 Whisper) | from the model: cross-attention alignment of the transcript, after decoding (the text is the same with or without words) | Whisper's word probability |
| whisper.cpp (`whispercpp-*`) | from the model: whisper.cpp's token timestamps (DTW), computed after decoding (the text is the same) | mean of the word's token probabilities |
| transducer and CTC models through sherpa-onnx (Zipformer, Parakeet, NeMo, IndicConformer, SenseVoice, …) | from the model: the token times the decoder reports | mean of the word's token probabilities when the decoder reports token log-probabilities (`ys_log_probs`), else none |
| Whisper through sherpa-onnx (`whisper-*`) | **forced alignment** (below): the stock exports have no cross-attention outputs, so no token times. A catalog entry for an export with them can switch on the model's own with the option `token_timestamps: true` | none |
| Moonshine | **forced alignment**: the runtime exposes no token times | none |
| Voxtral on llama.cpp (`voxtral-*-llamacpp`) and other models that write text only | **forced alignment** of each piece of speech bolo cut the audio into (at most 15 s each for Voxtral) | none |
| cloud recognisers | from the provider when it gives them, else forced alignment | Deepgram's word confidence |

### Forced alignment

Recognisers without word times of their own get them by aligning the transcript to the audio (setting
`align_words`, on by default; header `x-bolo-word-aligner`). The aligner is a character-level CTC model run in ONNX
Runtime: its per-frame scores and the Viterbi path that spells the transcript put each letter on its 20 ms frames
(bolo.alignment). The same aligner answers `POST /v1/audio/alignments` (audio and text in, word times out) and
gives the word times of synthesised speech (`POST /v1/audio/speech/captioned`); docs/api.md, "Word timestamps and
forced alignment".

| Aligner (task `align`) | Model | Languages | Licence |
| --- | --- | --- | --- |
| `omnilingual-ctc-300m-align` (default) | Meta Omnilingual ASR CTC 300M, int8; the same files as the recogniser `omnilingual-ctc-300m` (pulled once) | the scripts of 1,600+ languages, as written | Apache-2.0 |
| `mms-300m-forced-aligner` | ctc-forced-aligner's MMS-300m alignment model (wav2vec 2.0, 1,130 languages), int8 ONNX by onnx-community | any, romanized with uroman (`pip install "bolo-speech[align]"`); Latin script only without it | **CC-BY-NC-4.0 (non-commercial)** |

What was run (CPU, 4 threads): both aligners on the bundled `en-long` (11.8 s) and `hi-short` (Hindi) clips with
their texts. On `en-long` every word start and end of the two aligners is within 0.05 s of the other's; Hindi is
aligned in Devanagari by the Omnilingual model and after romanizing by MMS. Through the HTTP API: `moonshine-tiny-en`
on `en-long` with `timestamp_granularities[]=word` (words from `omnilingual-ctc-300m-align`), and captioned speech
from `kitten-nano-0.8-int8`.

NVIDIA's CTC models in the catalog (Parakeet TDT-CTC, NeMo FastConformer CTC) are not offered as aligners: their
sub-word vocabularies would need the models' SentencePiece tokenizers, which the sherpa-onnx exports do not ship,
and their exports take NeMo's filterbank features rather than audio. They give word times of their own as
recognisers.

With `diarize=true`, words also carry their speaker.

## Cloud providers

The catalog also lists models bolo does not run: the well-known speech models
of **OpenAI**, **Sarvam**, **ElevenLabs** and **Deepgram**, served through the
`remote` engine (`server/src/bolo/catalog/cloud.yaml`; they are in the tables
above under "OpenAI (cloud)", "Sarvam (cloud)" and so on). They differ from
every other entry in four ways:

- **Paid cloud API; audio leaves this server.** Every entry says so in its
  description, the API marks it (`cloud: true`, `notice`), and the console
  labels it and can filter on it. Nothing goes to a provider unless a request,
  a default (`default_stt` …) or a model's `fallback` list names a cloud model.
- **Nothing is downloaded.** An entry is installed once its provider is
  configured (`[providers.<name>]` with an `api_key`, or
  `BOLO_PROVIDER_<NAME>_API_KEY`). `bolo pull` on one only checks that.
- **The licence field carries the provider's terms**, with the link in
  `license_url`. As with licences elsewhere in the catalog, terms are
  labelled, never a reason to leave a model out.
- **None is validated.** No entry was run against a real account; languages,
  voices and limits are what each provider's documentation said on 2026-11-14.

Any other model of a configured provider (and any model of an
`openai-compatible` server, such as another bolo) can be used as
`<provider>:<model>` without an entry.

Diarization, long-recording segmentation, word lists and text normalisation,
telephony formats, Realtime sessions, traces and fallback work on top of
cloud models as they do for local ones. Request-field mapping per provider,
errors, cost estimates, key handling and what was and was not verified are in
[providers.md](providers.md).

## External TTS servers (llama-server, audio.cpp)

Many 2026 text-to-speech models (Chatterbox, Qwen3-TTS, VoxCPM2, Magpie,
CosyVoice3, Maya1, …) are large autoregressive models that bolo does not run
in-process. Two ggml runtimes do, and both have an HTTP server:

* **llama.cpp's `llama-server`** (MIT). `POST /tts` takes `input`, `lang`,
  `voice` (one of the `voices` that `GET /props` lists), `speaker_ref_b64` (a
  reference recording to clone), sampling fields and `response_format` `wav` /
  `pcm` / `pcm16`, and with `stream: true` sends audio as it is generated;
  `POST /v1/audio/speech` is the OpenAI-shaped variant (only `speed` 1.0).
  One model per process: `llama-server -m model.gguf -mm mmproj.gguf`.
  Documented speech families (docs/multimodal/tts.md, build b12521): Qwen3-TTS,
  Pocket TTS, Soprano, LFM3-Audio, Chatterbox (Turbo and Multilingual V3), Fish
  Drama 3, ZONOS3, Fish Audio S2 Pro, Kokoro, tiny-voca, and the song models
  YuE2 and MiniMax Music 3 (not speech: not catalogued). Chatterbox Multilingual's classifier-free guidance runs in
  `llama-tts` but not yet in `llama-server`.
* **audio.cpp's `audiocpp_server`** (Apache-2.0, v0.9.4). `POST
  /v1/audio/speech` with `model` (an id from the server's JSON config),
  `input`, `language`, `voice` (a preset, a reference-library name or a
  model-native voice), `voice_ref` (`{"type": "base64", "data": …}`, WAV, at most
  5 MiB), `reference_text`, `speed` where the model supports it and
  model-specific `options`; WAV by default, raw PCM16 with `stream_format:
  "audio"` for models configured with `mode: "streaming"`. `GET
  /v1/audio/voices?model=` lists voices. About 60 speech-generation families.
* **The Tontaube inference server** (`tontaube serve`, Apache-2.0, v1.1.0), the
  only runtime for TontaubeV1.1. `POST /predict` with `text`, `language`, `tag`
  (style), `voice_audio_b64` or `voice_path`, `controls` (five sliders) and
  `format`, answering base64 audio in JSON; `GET /readyz` is 503 while it
  loads. A Python application rather than a binary: bolo clones the pinned tag
  (`bolo components install tontaube`) and runs `uv run --python 3.12 --frozen
  tontaube serve`. See [voices.md](voices.md#tontaube).

bolo talks to both through one engine, **`tts-server`**
(`server/src/bolo/engines/tts_server.py`). Every catalog entry names its
runtime (`options.runtime`) and what the server must load; the rest of bolo
sees an ordinary voice: `/v1/audio/speech` (all formats, `stream_format: sse`),
the TTS WebSocket, Realtime conversations and `GET /v1/audio/voices` work
unchanged. Audio is streamed back where the runtime streams (llama-server
always, as PCM16) and resampled to the entry's `sample_rate` with bolo's
resampler.

### Two ways to run the server

**Connect** to a server you run (any machine, any GPU build). Name it in the
setting `tts_servers`:

```toml
[tts]
servers = [
  "llama-server=http://127.0.0.1:8080",          # every llama-server model goes here
  "voxcpm2-audiocpp=http://gpu-box:8080",        # one model, by catalog id (wins over the runtime)
]
server_api_key = "…"                             # if your server has --api-key
```

Nothing is downloaded and no binary is needed: the entry counts as installed.
Start llama-server with the model the entry lists (for example
`llama-server -hf ggml-org/chatterbox-GGUF`, or `-m … -mm …` with the files in
the entry's `source`). For audio.cpp, declare the model in the server's config
under the id in the entry's `options.server_model` (the catalog id), with the
entry's `family`:

```json
{"host": "0.0.0.0", "port": 8080, "backend": "cuda", "models": [
  {"id": "voxcpm2-audiocpp", "family": "voxcpm2", "path": "/models/VoxCPM2-GGUF", "task": "tts", "mode": "offline"}]}
```

**Managed**: when no URL is configured, bolo downloads the files the entry
lists (`bolo pull <id>`), starts the binary on a free loopback port when the
model is first used (llama-server with a random `--api-key`; audio.cpp with a
config bolo writes to `<data_dir>/run/tts-server/<id>.json`), waits for `GET
/health` (llama-server answers 503 while it loads), and stops it when the
model is unloaded (`tts_ttl_s`) or bolo exits. A server that dies fails the
requests in flight with `503 engine_unavailable` and is started again on a
later request after a back-off of 1 s, doubling up to 60 s while crashes keep
coming, as engine plugins are. The binary comes from the setting
`llama_server_bin` / `audiocpp_server_bin`, else from

```bash
bolo components install llama-server    # llama.cpp b12521 release (CPU build), SHA-256 checked, 24 MB
bolo components install audio.cpp      # audio.cpp v0.9.4 release (portable CPU build), 70 MB
```

(also `POST /api/v1/components/<id>/install`), else from `PATH`. The pinned
archives are upstream's own release assets for Linux x86-64 (both), Linux
arm64 and macOS (llama.cpp) and macOS arm64 Metal (audio.cpp); their SHA-256
sums were computed from the downloads. **The audio.cpp Linux release is built on
Ubuntu 26.04 and needs glibc 2.43**: on older systems the install stops with
that message, and you build `audiocpp_server` from source or connect to one.
Both prebuilt Linux builds are CPU-only; for a GPU point the `*_bin` setting at
a CUDA/Vulkan build or connect to a server. With `device = "cuda"` bolo adds
`-ngl 99` to llama-server; audio.cpp gets `"backend": "cuda"`.

A missing binary answers `424 component_missing` naming both routes; an
unreachable server, one that did not start (its last log lines are in the
message) or crashed answers `503 engine_unavailable`; what the server refuses
(an unknown voice, an option the model lacks) answers `400` with the server's
message.

### Voices and cloning

* Entries of models that clone voices have `features: ["voice-cloning"]`. A
  **saved voice** ([voices.md](voices.md)) can be named as their `voice`; it is
  sent as the reference below.
  `POST /v1/audio/speech` takes `reference_audio` (base64 or a data URL, up to
  30 s are used) and optionally `reference_text`; bolo decodes it, makes it
  mono 16-bit WAV and sends it as `speaker_ref_b64` (llama-server) or
  `voice_ref` + `reference_text` (audio.cpp). Models that cannot clone answer
  `400`.
* Voices named `heart`, `michael` and `alpha-hi` are **bolo's synthetic
  reference clips** (the built-in samples `en-short`, `en-long`, `hi-short`,
  spoken by Kokoro voices, so no real person's voice is imitated), sent as the
  voice to imitate. Pocket TTS and Qwen3-TTS on llama.cpp need a reference for
  every request, so they have only these.
* `default` is the model's own voice (no reference sent). Voices the server
  lists (`/props`, `/v1/audio/voices`) are added to the model's voices once it
  is loaded; other names are passed through, and OpenAI names (`alloy`, …) that
  the server does not know are dropped instead of being refused.
* `language` reaches multilingual models (`lang` / `language`); the text is
  normalised for that language first, as for every voice.

The reference is sent again for every sentence (llama-server does not cache
it), so a long reference costs time on each sentence: with an 11.8 s clip
Pocket TTS on CPU ran at RTF 1.7-2.3 instead of 0.7 with a 3 s clip.

### A model that is not in the catalog

Add an entry to `<data_dir>/custom.yaml` with `engine: tts-server`, `kind:
tts-server`, `requires: [llama-server]` (or `[audio.cpp]`), a `source` (a
repository and `include`, or `local: /path`) and `options`: `runtime`;
`model_file` and `mmproj_file` (llama-server) or `family`, `model_file`
(optional: the package folder is used without it), `mode`-related `stream`,
`load_options`, `session_options` and `server_model` (audio.cpp); `cloning`,
`send_language`, `language_option`, `language_names`, `speed` (the server
honours `speed`), `request` (extra request fields, audio.cpp: model
`options`), `server_args` and `hardware`. Voices may carry `reference:
<sample id>` (a reference clip) or `server_voice: <name>` (`""` for the
model's own voice). The entries in `catalog/tts-server.yaml` are examples of
each.

### What is catalogued

58 entries (`server/src/bolo/catalog/tts-server.yaml`): 16 for llama-server,
40 for audio.cpp, 2 for Tontaube. Licences never exclude a model; each entry states
`license`, `commercial_use` and `gated`, and `options.hardware` gives the GPU
advice (`optional`, `recommended`, `required`) and an approximate memory need
(weights x 1.3 + 0.5 GB, an estimate, not a measurement). The catalog API
returns it as `hardware`. `options.cpu_only_on` names device families a model cannot run on (`metal: "why"`): it is placed on the CPU there with that reason as its notice (Soprano 1.1 on audio.cpp segfaults on Metal; its llama.cpp entry runs there). `measured.metal` is a speed measured on an Apple silicon GPU, which `bolo recommend` uses for a model placed on Metal.

`options.float_decoder_without_vnni` (a file pattern, for sherpa-onnx Whisper exports) names a float decoder to use
instead of the int8 one on x86 CPUs without AVX-512 VNNI, where ONNX Runtime's int8 matrix products may saturate.
There it is downloaded with the model and loaded; elsewhere it is not downloaded. whisper-large-v3-turbo has it:
on an AMD EPYC 7763 its int8 decoder turned Telugu, Bengali and Tamil into nonsense. A copy pulled before this
option existed keeps the int8 decoder, with a warning in the log, until `bolo pull` fetches the float one.

| Runtime | Entries | Licences (commercial use) |
| --- | --- | --- |
| llama-server | Chatterbox Multilingual V3 (23 languages incl. **Hindi**), Chatterbox Turbo, Qwen3-TTS 1.7B Base, Pocket TTS (en, de, es, fr, it, pt), LFM3-Audio 2B; Soprano 1.1 80M, Fish Drama 3, Kokoro-82M (English voices), tiny-voca, ZONOS3-Mini and Fish Audio S2 Pro (ggml-org's conversions) | MIT, Apache-2.0, CC-BY-4.0 (yes); LFM Open License (only under USD 10M revenue), Fish Audio Community License (revenue threshold) - marked `commercial_use: false` |
| tontaube | TontaubeV1.1 int4 (GPU or x86-64 CPU) and full precision (vLLM, 24 GiB GPU); en, de first, es, fr, it, nl, pt | Tontaube Community Model License 1.0: free below EUR 200,000 group revenue with "Powered by TontaubeV1", research and testing free, a speech-generation service always needs a commercial licence - marked `commercial_use: false` |
| audio.cpp | **Hindi**: Chatterbox Multilingual (V3 talker), VoxCPM2, NVIDIA Magpie Multilingual 357M, MOSS-TTS Local v1.5, FireRedTTS3, Kitten TTS 2, Supertonic 3, sanoTTS (Hindi voice), XTTS v2, Voxtral TTS; and CosyVoice3, Maya1, Qwen3-TTS (0.6B Base, 1.7B CustomVoice), Pocket TTS, Soprano 1.1, Chatterbox Turbo, MOSS-TTS Nano, OmniVoice, Fish Audio S2 Pro, Fish Drama 3, Higgs TTS 3, IndexTTS-2, VibeVoice 1.5B, OuteTTS 1.0, NeuTTS 2E, Ming 3.0 TTS, BreezeTTS 2, KugelAudio, ZONOS3-Mini, dots.tts, Irodori-TTS, MioTTS, Gradium Phonon, Audio8 TTS, Confucius4-TTS, Plapre 3, Echo-TTS, LFM3-Audio | per audio.cpp's `docs/model_licenses.md`: MIT, Apache-2.0, CC-BY-4.0, NVIDIA Open Model License, OpenRAIL-M, GPL-3.0 (yes); CC-BY-NC(-SA), Fish, Boson, BreezeBlue, Coqui (no); bilibili, Stellon, NeuTTS, LFM (conditional, marked `false`) |

Sizes run from 6 MB (sanoTTS) to 25 GB (Drama 3); the table in "The catalog"
above has each one. File lists come from audio.cpp's `model_specs/*.json`
(default package of each family) and were checked against the Hugging Face
listing, pinned to a commit. The output rate the catalog states is what bolo
resamples to; audio.cpp's WAV carries the model's real rate.

**llama-server b12521** (2026-11-18; Kokoro from b12494, tiny-voca from b12497): one short English sentence each,
on CPU, came back as non-silent audio at the catalog's rate from `soprano-1.1-80m-llamacpp` (32 kHz, 1.3 s),
`pocket-tts-en-llamacpp` (24 kHz, 1.4 s), `kokoro-82m-llamacpp` (voices `af_heart` and `bm_george`, 24 kHz) and
`tiny-voca-llamacpp` (voices `F2` and `M2`, 24 kHz). Not run on this build for lack of disk: LFM3-Audio and the
speech-to-text and audio-in models on llama-server (Voxtral, LFM2.5-Audio).

**Added, not yet run** (`options.not_yet_run: true` in the catalog; files, sizes and revisions checked against Hugging
Face, voices and options from llama.cpp's docs/multimodal/tts.md, architectures supported by the pinned b12521):
`zonos3-mini-llamacpp` (ZONOS3-Mini, Apache-2.0, 27 voices of the release, 2.0 GB) and `fish-audio-s2-pro-llamacpp`
(Fish Audio S2 Pro, Fish Audio Research License: not for commercial use, 6.3 GB).

**Gradium Phonon 40M on audio.cpp**: the multilingual checkpoint (`gradium-phonon-40m-multilingual-audiocpp`:
English, French, German, Spanish and Portuguese in one set of weights) and the one-language ones
(`gradium-phonon-40m-en-audiocpp`, `-fr-`, `-de-`, `-es-`, `-pt-`), each about 122 MB. CC BY-NC 4.0, not for
commercial use, and gated: Gradium grants access automatically after a form on Hugging Face, then bolo needs a token
(`hf_token`). audio.cpp v0.9.4's `gradium_phonon` family loads them from the published safetensors, config.json and
tokenizer.model, which is what the entries pull. Run on the 4-vCPU Xeon (2026-11-21, audio.cpp built here, bolo's
test suite running alongside): the multilingual model in all five languages and the English and German ones came back
as non-silent 24 kHz audio that faster-whisper-small read back right (French with one word wrong, German with one
letter). The French, Spanish and Portuguese one-language models are still marked not run (French: access not
granted yet when tried). The 102 voices of
`gradium/phonon-voices` (Apache-2.0) work with every Phonon checkpoint, and each Phonon entry carries those of its
languages (the 31M and the English 40M the 37 English ones, the multilingual 40M all 102) next to the checkpoint's
default voice: bolo puts each voice's file in the model's `embeddings/` folder, where audio.cpp finds it by name. On
the 31M (2026-11-21, audio.cpp v0.9.4 built here) the default voice and four packaged ones, Zoey, Easton, Maeve and the
character Captain Silas Blackwater, came back word for word from faster-whisper-small, each in its own voice.

**Added, not yet run, on audio.cpp: VoiceTut-TTS** (`voicetut-tts-audiocpp`, Egyptian Arabic with English code-switching,
1.35 GB): its author's Q8_0 GGUF on audio.cpp's omnivoice family, with the language id `arz` the author asks for
sent on every request. Tried on the 4-vCPU Xeon (2026-11-21, audio.cpp v0.9.4 built here, a reference clip as the
voice): an Egyptian Arabic and an English sentence each took about 4 to 6 minutes for 2 s and came back as loud noise
("AHHHHHH!" from faster-whisper-small for "Hello, how are you today?"). OmniVoice itself was poor on this CPU too
(above), and the author measured on a GPU, so it stays marked as not run until it is tried there. The model card says
Apache-2.0, but the audio tokenizer it reuses is under the Boson Higgs Audio 2 Community License and OmniVoice moved to
a non-commercial licence, so the entry is marked not for commercial use until that is clear.

**Added, not yet run, on audio.cpp** (`options.not_yet_run: true`): Cilla Mini (`cilla-mini-audiocpp`, Persian, about 113 MB, 18.2 M parameters, audio.cpp's pocket_tts_small family),
from Oncilla AI. CC BY-NC 4.0 (commercial use needs a paid plan from Oncilla AI) and gated behind a terms form, so it
was not downloaded here. audio.cpp v0.9.4 loads that family from the published layout (`model.safetensors`,
`config.yaml`, `tokenizer.model` at the root), which is what the entry pulls. The authors' own runtime normalises
Persian text (their `normalize_fa.py`, pulled with the model) and turns off Pocket TTS's capitalisation and final full
stop; bolo's audio.cpp path does neither, which the first run should check.

**Added, not yet run, on audio.cpp** (`options.not_yet_run: true`): Cilla Lite (`cilla-lite-audiocpp`, Persian, about 440 MB, 100 M parameters, audio.cpp's pocket_tts family),
from Oncilla AI. CC BY-NC 4.0 (commercial use needs a paid plan from Oncilla AI) and gated behind a terms form, so it
was not downloaded here. audio.cpp v0.9.4 loads that family from the published layout (`model.safetensors`,
`config.yaml`, `tokenizer.model` at the root), which is what the entry pulls. The authors' own runtime normalises
Persian text (their `normalize_fa.py`, pulled with the model) and turns off Pocket TTS's capitalisation and final full
stop; bolo's audio.cpp path does neither, which the first run should check.

**Multilingual Pocket TTS on llama.cpp** (`pocket-tts-multilingual-llamacpp`, CC-BY-4.0): Thanushttk's community
fine-tune of Kyutai Pocket TTS, a 6-layer student with one tokenizer for English, Hindi, Marathi, Kannada, Tamil,
Malayalam and Telugu. This project converted it with llama.cpp's own Pocket TTS converter (release b12521) and
publishes the pair at codebucketsolutions/pocket-tts-multilingual-GGUF. One change to the converter was needed: the
fine-tune's tokenizer is a BPE SentencePiece model where Kyutai's are unigram ones, so it is written as llama.cpp's
SentencePiece tokenizer (the patch is in that repository); llama.cpp's token ids then equal SentencePiece's. Run on
the 4-vCPU Xeon (2026-11-21) through bolo's HTTP API with bolo's Hindi reference clip: Hindi, Marathi and Tamil read
back close by faster-whisper-small, Kannada and Malayalam sounding right (the recogniser wrote them in Devanagari),
Telugu not recognised by whisper-small, English word for word; best of three warm Hindi runs at 0.77 of real time
on the busy machine.

**tiny-voca in process** (`tiny-voca-onnx`): the authors' ONNX export on ONNX Runtime, no server, with their
phonemiser dizione (Apache-2.0; bolo carries a copy in `bolo/vendor/dizione`, its data comes with the model). One
sentence per voice came back as non-silent 24 kHz audio, as loud as the llama.cpp conversion's. dizione exists only in
Python (the authors' browser studio runs it under Pyodide); the browser demo has a TypeScript port of bolo's copy
(`ui/src/lib/browser-tts/dizione`), so its Text to speech page offers tiny-voca F2 and M2: the ONNX export on ONNX
Runtime Web, phonemes from the port. The port gives the token ids and stress of the Python exactly on a corpus of 306
texts (numbers, money, dates, times, codes, addresses, homographs, chat spellings, unknown words; the unit test
`dizione.test.ts`, its fixture written by `ui/scripts/parity/dizione_cases.py`) and on 3,000 lines of bolo's docs. The
first use fetches the model (21 MB) and the phonemiser data (5 MB); in headless Chromium on a 4-core CPU a two-clause
sentence came back as 3.5 to 3.8 s of non-silent 24 kHz audio in 2 to 6 s, downloads included.

**In the browser demo, beside tiny-voca**: little-tts (voices 92 and 6097, the tiny-voca path with its own weights;
its phonemiser data is byte for byte tiny-voca's, so the two share that download), Paradee-8M (its four voices, from
uccharan's Kokoro phonemes in the page, American or British by voice), Kitten TTS 2 Pico (two voices each in
English, Hindi, Spanish, French, Italian and Brazilian Portuguese: the languages uccharan's Kokoro profile reads in
the page; the non-English ones marked approximate, as for Kokoro) and kale-tts-1-nano (its ten voices, with Kale's
own JavaScript front end carried in `ui/src/vendor/kale`, Apache-2.0, over its word lists fetched from the model
repository). All from Hugging Face at pinned revisions. In headless Chromium (2026-11-21) one sentence from each of
little-tts 92, Paradee Heart and Isabella, Kitten Pico Eleanor and Priya (Hindi), and kale Kenzie and Jack came back as
non-silent 24 kHz audio (`ui/e2e-demo/tts-small-voices.spec.ts`).

**little-tts in process** (`little-tts-onnx`, TuringLogic; weights CC BY-NC 4.0, not for commercial use): a 4.6
M-parameter English voice trained from scratch on HiFi-TTS, with a DDSP harmonic-plus-noise head. Its ONNX graph takes
the same inputs as tiny-voca's and its phonemiser is dizione at the commit bolo carries, so it runs on the tiny-voca
engine (which reads little-tts's flat `onnx/model.json` as well as tiny-voca's `config.json`, and refuses a graph whose
phoneme table is not dizione's). Run on the 4-vCPU Xeon with 2 threads (2026-11-21): one sentence per voice came back
as non-silent 24 kHz audio (median pitch 210 Hz for voice 92, 121 Hz for 6097) at 0.026 of real time, load 0.3 s, and
faster-whisper-small transcribed both back word for word.

**Paradee-8M in process** (`paradee-8m-onnx`, Sahil Mahendrakar, Apache-2.0): 7.9 M parameters distilled from
Kokoro-82M, four voices in one int8 graph of 8.5 MB (af_heart, am_puck American; bf_isabella, bm_fable British). It
reads Kokoro's phoneme ids, so bolo phonemises it as it does Kokoro (profile `kokoro`: uccharan's port of misaki, or
espeak-ng), each voice in its own accent: a catalog voice with a `language` of its own now reads in that language.
ONNX Runtime 1.32's extended graph fusions fail on this int8 graph with bolo's exact int8 products, so it loads with
the basic ones. Run on the 4-vCPU Xeon with 2 threads (2026-11-21): one sentence per voice, 24 kHz, non-silent, all
four transcribed back word for word by faster-whisper-small, at 0.084 of real time.

**PiVo-1.4M in process** (`pivo-1.4m-onnx`, Empero Research, Apache-2.0; its vocoder derives from Xiph/Opus FARGAN,
BSD-3-Clause): two English voices, Pi and Vo, at 16 kHz from 1.4 M parameters in four ONNX graphs (encoder, a decoder
with a key/value cache, FARGAN's conditioning and a vocoder step run four times per 10 ms frame). bolo's engine
(`engines/pivo.py`) follows the model's reference code and gives the same token ids and the same audio, bit for bit,
on the sentences compared; it differs in two ways a server needs: a word CMUdict lacks is spelled letter by letter
instead of refusing the request, and characters it cannot say are left out. Run on the 4-vCPU Xeon with 2 threads
(2026-11-21, other jobs keeping the load near 12): one sentence per voice, non-silent 16 kHz audio (median pitch 193
Hz for Pi, 142 Hz for Vo), both transcribed back word for word by faster-whisper-small, at 0.66 of real time: the
vocoder is called 400 times per second of audio, so the time goes to ONNX Runtime's per-call cost.

**Kitten TTS 2 Pico and Mini in process** (`kitten-tts-2-pico-onnx`, `kitten-tts-2-mini-onnx`, KittenML, Apache-2.0):
Pico has one 8 MB int8 graph per language (8 languages, 30 voices), Mini one 83 MB graph for 16 languages (bolo offers
15: Japanese needs misaki's Japanese front end, which bolo does not have). Both read Kokoro's phoneme ids with an end
token and a style row per input length from each voice's pack. Kitten's own front end (kittenml) is misaki for
English and misaki's eSpeak NG conversion for the other languages, through the GPL `phonemizer` package and PyTorch;
bolo gives the same kind of phonemes without either: uccharan's `kokoro` profile where it covers the language (English,
Spanish, French, Italian, Portuguese, Hindi), else eSpeak NG (the optional component) asked for tied phonemes
(`a^ɪ`) and converted with misaki's table, then Kitten's folds (its phonemizer v1 for Pico, v2 for Mini). Kitten's
English front end falls back to CMUdict and a small BART for words misaki lacks; uccharan's port of misaki uses its
own fallback. Run on the 4-vCPU Xeon with 2 threads (2026-11-21): Pico at 0.12 of real time, Mini at 0.61; English,
German and French came back word for word from faster-whisper-small, Hindi and Swahili recognisably.

**kale-tts-1-nano in process** (`kale-tts-1-nano-onnx`, Kale Computer, Apache-2.0): ten synthetic English voices
(eight American, two British) designed with Maya-1, 7.5 M parameters in one 14.8 MB int8 graph, distilled from a
Kokoro-82M fine-tune. Its text front end is Kale's own, in JavaScript; bolo runs a Python port of it
(`engines/kale.py`): Kale's technical terms, dizione's text rules (the Python dizione bolo carries; Kale uses its
JavaScript port), Kale's size suffixes, dotted names and letter spelling, sentences with one-word ones joined, then
its checked word list, dizione's homograph picker with Kale's sense rules, stems, spelling and its GRU guesser for
unknown words, and the IPA as Kokoro-native ids. On Kale's 214 test texts in both accents, the port gives the same
sentences and IPA as Kale's released front end (run in Node) on 212; the other two are British homograph readings,
which Kale rewrites from American and bolo leaves American. Run on the 4-vCPU Xeon with 2 threads (2026-11-21): three
voices transcribed back word for word by faster-whisper-small, at 0.028 of real time.

**What was run**: `pocket-tts-en-llamacpp` on the upstream llama-server b12375
release, started by bolo (`bolo components install llama-server`), on the
shared 4-vCPU Xeon with 2 threads: RTF 0.72, first audio 0.77 s, two English
sentences transcribed back word for word by `moonshine-tiny-en`. Cloning from
saved voices (two LibriSpeech speakers) was then measured for all six Pocket TTS
languages, Chatterbox Turbo and Qwen3-TTS 1.7B Base, which are now
`validated: cpu`, with the speaker similarity of each clone: see
[voices.md, "Measured: cloning on CPU"](voices.md#measured-cloning-on-cpu).
Chatterbox Multilingual V3 needed a smaller context to load (now set) and then
took about 5 minutes per short sentence on this CPU, so it stays untested.
On audio.cpp v0.9.4, built from source because the release binary needs
glibc 2.43 (this machine has 2.39), Pocket TTS, MOSS-TTS Nano, XTTS v2, Plapre
3, OmniVoice and Qwen3-TTS 0.6B Base ran (`validated: cpu`; OmniVoice did not
clone); XTTS v2 needed two entry fixes, and ZONOS3-Mini is not in v0.9.4.
Everything else is `validated: untested`: the request fields come from each
project's documentation and server source (read 2026-11-14), and the engine is
tested against fake servers.

## Orpheus-family voices on llama.cpp (Svara, Veena, Orpheus, snorTTS)

Orpheus (Canopy Labs) and the models built on its recipe are stock Llama
language models whose vocabulary was extended with SNAC audio codes, so stock
`llama-server` runs their GGUF files. bolo does the rest itself, without
PyTorch: kind **`orpheus-llamacpp`** of the `tts-server` engine
(`server/src/bolo/engines/orpheus_llamacpp.py`, catalog
`server/src/bolo/catalog/tts-orpheus.yaml`).

1. **Prompt.** bolo tokenises the text with the server (`POST /tokenize`,
   special tokens parsed, no BOS added; identical ids to the Hugging Face
   tokenizer for Svara and Veena) and wraps it in the model's special tokens:
   `orpheus` `[SOH] [BOS] "speaker: text" [EOT] [EOH] [SOAI] [SOS]`, `veena`
   `[SOH] "<spk_kavya> text" [EOH] [SOAI] [SOS]`, `svara` (Kenpath's
   svara-tts-inference) `[BOS] [SOH] [AUDIO] "Hindi (Male): text" [EOH] [EOT]
   [SOAI] [SOS]`.
2. **Generation.** `POST /completion` with the prompt as a token array,
   `stream: true`, `return_tokens: true`; sampling from the model card
   (Svara 0.75 / 0.9 / repetition penalty 1.1, Veena and snorTTS 0.4 / 0.9 /
   1.05, Orpheus 0.6 / 0.8 / 1.3), temperature before top-p, top-k and min-p
   off, the penalty over the whole sequence. It stops at the end-of-speech id
   (bolo closes the stream, and llama-server frees the slot), at llama.cpp's
   own end-of-generation ids, or at a budget of 2 frames per character of text
   plus 12 frames (bolo-torch's 1.3 frames per character cut ordinary Bengali
   and Hindi sentences short in our runs).
3. **Audio.** Every 7 audio tokens are one SNAC frame (1 + 2 + 4 codes, 2048
   samples, 85.3 ms at 24 kHz). Every 4 new frames, bolo decodes them with 3
   frames of left context (whose audio is dropped), holding one frame back
   until the end: the same windows as bolo-torch's `OrpheusSNAC`. PCM streams
   out as it is decoded; first audio needs 5 frames (35 tokens).

The server is managed or operator-run as for every `tts-server` model (see
"External TTS servers" above; managed servers get `-c 4096 -np 1`). With
`tts_servers = ["llama-server=…"]` every llama-server model goes to that one
server, so name these models by id when your server runs another GGUF.
onnxruntime is needed for the decoder: `pip install "bolo-speech[orpheus]"`.

### The SNAC decoder

`server/tools/export_snac_onnx.py` exports the decoder of
`hubertsiuzdak/snac_24khz` (MIT, revision `d73ad17`) itself: weight
normalisation folded, `repeat_interleave` written as expand + flatten (the
tracer otherwise bakes one length in), dynamic batch and length, opset 17; fp16
by onnxconverter-common with the noise op kept in fp32. The decoder's noise
blocks stay (random, as in PyTorch). Files (`snac_onnx.json` records versions
and the verification; the export is byte-identical when repeated with the same
packages: torch 2.15.0+cpu, snac 1.2.1, onnx 1.23.2, onnxconverter-common
1.16.0):

| file | size | sha256 |
| --- | --- | --- |
| `snac_24khz_decoder.onnx` (fp32, used by the catalog) | 52,581,828 | `33d7968c2a635c66ffbd2e9c38491a059c181edf7c03d8ed783a7b10f99cb911` |
| `snac_24khz_decoder_fp16.onnx` | 26,368,793 | `0d9f92693bf05e453365f87aad6052d8939b2575eba64eacad5b60d53a317a41` |

They are published at `codebucketsolutions/snac-24khz-onnx` (rebuilt by the
"Publish a converted model" workflow and uploaded after their SHA-256 sums
matched the ones above); the catalog entries pin that revision, and
`options.snac_path` can point at a local copy instead. With an operator-run server nothing is pulled, so the
decoder is fetched once into `<data_dir>/components/snac/` and checked against
its SHA-256.

**Parity with PyTorch**, with the noise term set to zero on both sides (the
only way to compare sample by sample), on 3 random code sequences and 7 real
Svara generations (Q2_K and Q4_K_M; Hindi, Bengali, Tamil; 27-51 frames):

| | max abs difference | SNR vs PyTorch |
| --- | --- | --- |
| fp32, whole sequence | 6.1e-7 to 2.9e-6 | 112.3-126.7 dB |
| fp16, whole sequence | 4.9e-4 to 7.1e-3 | 49.7-67.4 dB |
| fp32, bolo's streaming windows vs bolo-torch's (same token ids) | 8.1e-7 to 1.6e-6 | 118.6-126.6 dB, same length and chunking |

With the noise on, PyTorch, our fp32 and our fp16 graphs all sit 19-44 dB
above the noise-free output (same size of noise), and `onnx-community/snac_24khz-ONNX`
(a cross-check only) does the same. fp16 is not faster on a CPU (106 ms
against 97 ms for a 7-frame window with 2 threads, about 0.3 x real time with
the overlap), so the catalog uses fp32. int8 is not offered (it is reported to
fail).

### What was run

On the shared 4-vCPU Xeon (no GPU), llama.cpp b12375 CPU release with 3
threads, while other jobs used the machine:

| model, quant | route | languages | RTF | first audio | round trip (IndicConformer; whisper-base for English) |
| --- | --- | --- | --- | --- | --- |
| Svara Q2_K (1.4 GB), Kenpath layout | bolo HTTP, managed server | hi, bn | 11-34 | 5 s | Hindi word for word (2/2); Bengali failed (the model spoke the speaker label, and the old budget cut the sentence). Functional check only |
| Svara Q4_K_M (2.1 GB), Kenpath layout | direct | hi, bn, ta | 11-15 | 5 s | mean CER 14 %: Bengali spoke the speaker label in 2/2, one Tamil word dropped |
| Svara Q4_K_M, Orpheus layout, Svara sampling | direct | hi, bn, ta | 11-16 | 5 s | 6/6 word for word |
| Svara Q4_K_M, the catalog entry as shipped | bolo HTTP, managed server | hi, bn, ta | 14.4-15.7 | 5.0-5.3 s (16 s with the server start) | CER 0 % in 4/6; a dropped first word (bn), one letter (ta); mean CER 6 % |
| Veena Q2_K (1.6 GB) | direct | hi, en | 10-15 | 4.6 s | English word for word; Hindi garbled. Too lossy |
| Veena Q4_K_M (2.5 GB) | direct, and `test_real.py` through bolo | hi, en | 11-15 | 4.6 s | Hindi and English word for word |

That is 3-8 generated tokens per second (5-8 when the machine was quieter) against the 82 that real time needs:
**on a CPU these voices are for offline jobs, not live use**; on a GPU the
estimate in the research note is RTF 0.7-0.9 at Q4 on an RTX 3060 (not
measured). From these runs the Svara entries use the Orpheus layout (the one
mlx-audio and the transformers.js export use for Svara) with Svara's sampling;
`prompt_format: svara` gives Kenpath's layout. The samples are small (2-3
sentences per language and setting); quality was judged by speech recognition
only, not by listening. Orpheus 3B and snorTTS were not run.

### Licences

Licences never exclude a model; each entry states `license`, `commercial_use`
and `gated`. Svara and Veena are Apache-2.0 and ungated. **Svara's training
data includes the IIT Madras IndicTTS corpus; whether its terms allow
commercial use of derived models was not verified.** Orpheus 3B 0.1 and
snorTTS-Indic are labelled Apache-2.0 but are Llama 3.2 3B derivatives
(Orpheus: Llama 3.2 3B Instruct; snorTTS: Canopy's Hindi pretrain, tagged
`llama3.2`), so the Llama 3.2 Community License applies too; Canopy's own
repository is gated (click-through), the Unsloth GGUF used here is not. Veena's
tokenizer has 11 speaker tokens; the card documents 4 voices (kavya, agastya,
maitri, vinaya), which are the catalogued ones.

## Whisper on whisper.cpp

The `whispercpp-*` entries are OpenAI's Whisper in ggml format for whisper.cpp, from
[ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) (MIT), each one file pinned to a commit with its
size and SHA-256: tiny, base, small and medium (also as `.en`, English only), large-v3 and large-v3-turbo, each
full-precision and in whisper.cpp's own 5-bit (`q5_0`/`q5_1`) and 8-bit (`q8_0`) quantisations, which are smaller and
faster on a CPU and a little less accurate. They need `pip install 'bolo-speech[whispercpp]'` (pywhispercpp, MIT):

```bash
bolo pull whispercpp-base-q5_1
curl -s localhost:8000/v1/audio/transcriptions -F model=whispercpp-base-q5_1 -F file=@clip.wav
```

A multilingual model detects the language when the request names none (whisper.cpp's own default would be English);
`.en` models are English only and cannot translate, and large-v3-turbo was not trained for translation. whisper.cpp
gives segment times and, with `timestamp_granularities[]=word`, word times from its token timestamps. The Whisper decoding options it honours are listed in api.md,
"Decoding options and repetition loops". A client that disconnects stops its decode (whisper.cpp's abort callback).

What was run: `whispercpp-tiny` pulled through the catalog and run by `tests/test_real_whispercpp.py` (opt-in): the
decoding options reach whisper.cpp and do not carry over to the next request, silence removal and the ignored-options
header work, and the Hindi sample with no language given came back as Hindi. No speed or accuracy was measured on a
quiet machine, so every entry is marked untested; on a shared four-core machine whisper.cpp was far slower than
faster-whisper on the same audio (api.md, "Decoding options and repetition loops").

## Voxtral speech-to-text on llama.cpp

Mistral's Voxtral (Apache-2.0) puts a Whisper large-v3 style audio encoder in
front of a Mistral language model. llama.cpp runs both (the `voxtral`
projector of libmtmd, in the pinned build b12375: no new llama-server pin was
needed), so bolo serves Voxtral without PyTorch: kind **`voxtral-llamacpp`**
of the `tts-server` engine (`server/src/bolo/engines/voxtral_llamacpp.py`,
catalog `server/src/bolo/catalog/stt-llamacpp.yaml`). The llama-server is
managed or operator-run as for every `tts-server` model (`bolo components
install llama-server`, or `tts_servers = ["voxtral-mini-3b-llamacpp=http://host:port"]`).

| id | files | notes |
| --- | --- | --- |
| `voxtral-mini-3b-llamacpp` | Q4_K_M language model + Q8_0 audio encoder (ggml-org), 3.2 GB | run on CPU, see below |
| `voxtral-small-24b-llamacpp` | Q4_K_M language model + F16 audio encoder (bartowski), 15.7 GB | catalogued, not run; wants a GPU with about 20 GB |

How a request runs:

* **Transcription** uses Voxtral's dedicated transcription mode, the request
  encoding of mistral-common: `[INST][BEGIN_AUDIO]<audio>[/INST]lang:<code>[TRANSCRIBE]`,
  sent to `POST /completion` as a prompt string with the audio as base64 WAV
  (`multimodal_data`). The `language` of the request becomes `lang:<code>`;
  without one the model picks the language itself but does not say which.
  The OpenAI `prompt` is not used: transcription mode takes no text.
* **Translation** (`/v1/audio/translations`, into English) is an instruction
  in the chat format (`[INST]<audio>Translate this audio into English. ...[/INST]`):
  Voxtral follows spoken-language instructions, but it has no dedicated
  translation mode, so treat it as instruction-following rather than a trained
  translation task.
* **Long audio** is cut at pauses by bolo's voice-activity detection into
  pieces of at most 15 s (`max_chunk_s`), one request each. 15 s rather than
  Whisper's 30 s window: on a 28.6 s piece holding three utterances the model
  transcribed the first, then commented on it, and dropped the other two.
  The catalog entries also set `queue_timeout_s: 0`, so that the pieces of a
  long recording, which are queued together, are not refused after the usual
  30 s wait.
* **Timestamps** are not available from the model. `verbose_json` segments
  are those pieces, with their real start and end in the audio; there are no
  word timestamps.
* **Stops.** On llama.cpp the model sometimes goes on after the transcript as
  if answering a chat message ("Here's a breakdown of what you're saying: ..."),
  always after a blank line; a transcript has none, so generation stops at
  `\n\n` (option `stop`). The token budget is 12 tokens per second of audio
  plus 32 (options `tokens_per_s`, `max_new_tokens`). Decoding is greedy unless
  the request sets a temperature.
* The prompts are built by bolo, not by the GGUF's chat template: the GGUF
  files carry another model's template (a Devstral system prompt), so
  `/v1/chat/completions` and llama-server's own `/v1/audio/transcriptions`
  would not use Voxtral's format with them. The media marker is read from
  `GET /props` (llama-server makes it random).
* Managed servers get `-c 4096`: llama-server's default context for this GGUF
  (32 k) would reserve about 4 GB of KV cache.

### What was run

On the shared 4-vCPU Xeon (no GPU, load average 10-16 from other jobs during
every run), llama.cpp b12375 CPU release started by bolo with 4 threads,
`voxtral-mini-3b-llamacpp`:

| sample | audio | RTF | result |
| --- | --- | --- | --- |
| en-short | 3.0 s | 7.1 | "Good morning! Could you tell me when my order will arrive?" (exact) |
| en-long | 11.8 s | 1.98 | "Speech servers are judged by how quickly first sound arrives. A good one starts speaking before it has finished reading, keeps a steady pace under load, and tells you honestly how long each step took." (WER 2.8 %: "the" dropped) |
| hi-short (Hindi) | 4.1 s | 4.9 | "नमस्ते आप कैसे हैं? आज मौसम बहुत अच्छा है।" (exact but for a comma) |
| en-long, translated | 11.8 s | 3.4 | the same English text, with "the" |

Loading took 41.7 s (llama-server start, weights read from disk), and the
llama-server process peaked at 5.7 GB resident. The times varied widely with
the machine's load (the same 3 s sample took RTF 7 in one run and 103 in
another), so take them as an upper bound. Short pieces cost almost as much as
30 s ones, because the encoder pads every piece to a 30 s window.

Not verified end to end: a 46 s recording of the samples with pauses, cut into
15 s pieces through bolo (`test_real.py::test_voxtral_mini_large_on_a_managed_llama_server`),
did not finish on the overloaded machine: generation fell to about 0.1 token
per second and one piece passed the 600 s request timeout (option `timeout_s`).
The cutting itself is covered by the tests with a stand-in server.

Quantisations tried: the community IQ4_XS files (bartowski's, with an
importance matrix, and mradermacher's static one) with the same audio encoder
answered the 3 s question ("To provide an accurate estimate ... I need a bit
more information") instead of transcribing it, in transcription mode and with
a "transcribe" instruction alike; ggml-org's Q4_K_M transcribed it. That is why
the entry uses ggml-org's files although they are 0.2 GB larger.

### NVIDIA Nemotron 3.5 Transcribe on the same client

Nemotron 3.5 Transcribe (OpenMDW-1.1, commercial use allowed; 30B-A4B, a
Mamba-2 and Transformer mixture-of-experts language model behind a speech
encoder with a speaker branch, 40 languages for transcription, translation
between English and 31 of them) runs on llama.cpp from the pinned b12521:
libmtmd has its `nemotron_transcribe` projector and llama.cpp the
`nemotron_h_moe` architecture. The same `voxtral-llamacpp` client sends it,
with the model's own prompts from the catalog entry (`options.prompts`:
`transcribe`, optionally `transcribe_language`, and `translate`, with
`{audio}`, `{language}` and `{language_name}`). They are NVIDIA's chat
template written out for one user turn with reasoning off
(`<|im_start|>assistant\n<think></think>`) and the instructions of its card's
Prompt Guide: a readable transcript ("Audio transcription. Language: as
spoken. ..."), and "Please translate the given speech to <language>.".

| id | files | notes |
| --- | --- | --- |
| `nemotron-3.5-transcribe-30b-a4b-iq4nl-llamacpp` | IQ4_NL language model + F16 audio encoder (maternion), 20.2 GB | about 24 GB of GPU memory |
| `nemotron-3.5-transcribe-30b-a4b-q8-llamacpp` | Q8_0 language model + the same encoder, 36.5 GB | about 44 GB |

**Added, not yet run** (`options.not_yet_run: true`; too large for this
machine's disk). Checked without downloading the weights: the GGUF headers
(read with HTTP range requests) name the architecture `nemotron_h_moe` and the
projector `nemotron_transcribe` with an audio encoder, the sizes and revision
come from the Hugging Face API. IQ4_NL rather than Q4_K_M: the model's matrix
widths (2688, 1856, 3712) are not multiples of 256, so a Q4_K_M file falls back
to larger types for most matrices (mradermacher's is 25.4 GB against 18.7 GB).
NVIDIA's own GGUF repository is for NeMo-Speech.cpp, its encoder file is not a
llama.cpp mmproj, so it is not used. The model's speaker tags and word
timestamps (CTC head) are not used here: segments are the pieces bolo cut the
audio into (at most 30 s each).

## SpeechBrain recognisers for languages without a strong option

SpeechBrain publishes about sixty recognisers. Each was compared, language by language, with what bolo already has
(Whisper, Parakeet, Canary, Omnilingual, the Indian-language models), from SpeechBrain's own test numbers on its
model cards; five were added, for languages where bolo had no strong local recogniser. They run in ONNX Runtime
without PyTorch (`speechbrain_ctc`, bolo.engines.speechbrain_onnx): SpeechBrain's `EncoderASR` (wav2vec 2.0 XLSR-53
fine-tuned, a DNN, a CTC layer) exported unchanged by `server/tools/export_speechbrain_onnx.py asr` and published
under codebucketsolutions/ with parity checks, decoded greedily as SpeechBrain does. On each repository's example
clip the ONNX graph gave the same best path and the same text as SpeechBrain 1.1.2. The clips come without a
reference transcript, so no error rate was measured here: the error rates below are the cards'. All Apache-2.0;
1.28 GB each; 4 threads on the shared 4-vCPU machine (load average 2 to 5).

| model | language | card (test set) | RTF | memory | output on the example clip |
| --- | --- | --- | --- | --- | --- |
| `wav2vec2-dvoice-fongbe` | Fon (fon) | DVoice 2.0: WER 9.0 %, CER 4.0 % | 0.219 | 2137 MB | távo è un xɔ ɔ́ cí glijaà |
| `wav2vec2-dvoice-wolof` | Wolof (wo) | DVoice 2.0: WER 16.1 %, CER 4.8 % | 0.219 | 2208 MB | Naka la ubbi bunt bi rekk mu fekk ruum bi laloo ay niiw |
| `wav2vec2-dvoice-darija` | Moroccan Arabic (ary) | DVoice 2.0: WER 18.3 %, CER 5.9 % | 0.088 | 2103 MB | ممكن تعيطي ليها في لبورطابل |
| `wav2vec2-dvoice-amharic` | Amharic (am) | DVoice 2.0: WER 24.9 %, CER 6.6 % | 0.17 | 2216 MB | ተጋጣሚ ያችን ቡና በ አገር ውስጥ እና በ ኢንተርናሽናል ውድድር ጥሩ ስ ም እንዳለው እናውቃለን |
| `wav2vec2-commonvoice-14-rw` | Kinyarwanda (rw) | Common Voice 14: WER 23.7 %, CER 7.6 % | 0.202 | 2192 MB | ABAMUGARIYE KU RUGAMBA BAFASHWA KUBONA INSIMBURANGINGO |

Why these five: Whisper has no Fon, Wolof or Kinyarwanda and is far from usable on Amharic; it and bolo's other
Arabic recognisers are trained mostly on Modern Standard Arabic, not Darija; the only other local candidate for all
five is `omnilingual-ctc-300m`, one small model for 1,600 languages. The DVoice and Common Voice test sets are read
speech close to the training data: expect more errors on conversations, phone audio and other domains.

Not added, with the reason:

* **English** (LibriSpeech, Switchboard, TED-LIUM, GigaSpeech, Common Voice, Loquacious; CRDNN, Conformer,
  Transformer, Branchformer, wav2vec 2.0, streaming Conformer): SpeechBrain's best LibriSpeech numbers (1.90 / 3.96
  % WER clean / other) are behind Parakeet and Canary, which bolo has.
* **French, German, Spanish, Italian, Portuguese** (Common Voice 14 wav2vec 2.0: 10.2, 8.9, 13.3, 8.4, 22.4 % WER;
  CRDNN, Transformer and transducer models are worse): Whisper large-v3, Canary and Parakeet v3 are as good or
  better on these languages. French MEDIA (an SLU corpus) is a narrow domain.
* **Chinese** (AISHELL, Common Voice): SenseVoice, Paraformer and FireRedASR in bolo are far better on AISHELL.
* **Arabic, Persian, Hindi, Serbian, Mongolian, Italian, French** (Whisper large-v2 and medium fine-tuned on Common
  Voice: Arabic 17.0, Persian 31.8, Hindi 15.3, Serbian 23.6, Mongolian 64.9 % WER): not clearly better than
  Whisper large-v3, Jais-ASR or the Indian-language models bolo has; the Mongolian model is not usable.
* **Korean** (KsponSpeech Conformer, 7.3 % CER on KsponSpeech): an in-domain number for Korean spontaneous speech;
  Korean is well served by Whisper and SenseVoice.
* **Swahili** (DVoice 2.0: 23.2 % WER): Whisper and `omnilingual-ctc-300m` both know Swahili, and the card's test set
  is not comparable with theirs; not clearly better.
* **Kinyarwanda, older release** (`asr-wav2vec2-commonvoice-rw`, Common Voice 7 test): superseded by the Common
  Voice 14 model above.
* Noisy-speech models (`whisper_rescuespeech`, `noisy-whisper-rescuespeech`) are for a German search-and-rescue
  corpus.

## Indian languages

See [indic-telephony.md](indic-telephony.md) for the full picture, with what
was measured for each language. In short:

* **Speech to text for all 22 scheduled languages** through per-language
  IndicConformer models (`indicconformer-<lang>-community`, MIT), 16 of them
  run here on real recordings; Hinglish in Latin script
  (`faster-whisper-base-hinglish`); Whisper fine-tunes for Hindi, Tamil,
  Kannada, Telugu, Bengali and Punjabi; a streaming Bengali Zipformer and
  NVIDIA's streaming Nemotron for Hindi.
* **Text to speech** for Hindi, Bengali, Telugu, Tamil, Malayalam, Marathi,
  Urdu, Nepali, Gujarati, Kannada, Maithili, Bhojpuri, Magahi and
  Chhattisgarhi (the SYSPIN voices are published). Non-commercial MMS-TTS
  voices for Punjabi, Odia, Assamese, Dogri and more are converted and
  published. **No voice** that runs without PyTorch for Sanskrit,
  Sindhi, Kashmiri, Konkani, Manipuri, Bodo and Santali.
* **Bhojpuri** has no speech-to-text model of its own here; the Hindi model
  is the best we measured (41 % word error rate on 120 utterances), better than
  the Whisper-small Bhojpuri fine-tune we converted (48 %).

**AI4Bharat's own repositories** (`ai4bharat/indicconformer_stt_*`,
`ai4bharat/indic-conformer-600m-multilingual`) are **gated** on Hugging Face
(MIT licence, but you must accept terms and use a token). bolo never
downloads a gated model without a token you supply, so the default
recognisers are ungated community exports of the same models, pinned to a
commit and named `-community`. The official multilingual release is listed
too (`indicconformer-600m-official`, gated, for the PyTorch engine plugin). They are third-party conversions: we verified the licence of
the originals and ran the files, not how they were made.

## Phonemisers

Piper-format voices, KittenTTS, Matcha-TTS and Kokoro were trained on phonemes, not letters, and
sherpa-onnx 2.x expects the caller to supply them. bolo has two phonemisers:

| Phonemiser | Licence | Covers | How it gets there |
| --- | --- | --- | --- |
| **uccharan** (default) | Apache-2.0 | Piper and Mimic 3 voices in uccharan's 39 languages (below); Kokoro voices in English (US, British), Spanish, French, Italian, Brazilian Portuguese and Hindi; Kitten and Matcha (US English). For Kokoro English it is a port of misaki, the front end Kokoro was trained with, with misaki's dictionaries | `bolo-speech[uccharan]` (uccharan 1.0 or a later 1.x release), part of `bolo-speech[all]` and the Docker image; `bolo components install uccharan` |
| **espeak-ng** | GPL-3.0-or-later | every language espeak-ng has | optional component, never installed by default (next section) |

uccharan phonemises **English** (US `en-us`, British `en-gb`), **Hindi** and the other South Asian
languages (Bengali, Marathi, Telugu, Nepali, Kannada, Tamil, Malayalam, Assamese, Gujarati,
Bhojpuri, Maithili, Magahi, Punjabi, Odia, Konkani, Sinhala, Sanskrit), **Urdu** and **Sindhi**, and
**Spanish** (Spain), **German**, **Dutch**, **French** (also Belgian `fr-be` and Swiss `fr-ch`),
**Italian**, **Portuguese** (Brazil `pt-br`, Portugal `pt`), **Polish**, **Turkish**, **Czech**,
**Swedish**, **Catalan**, **Romanian**, **Hungarian**, **Finnish**, **Norwegian Bokmål** and
**Danish**. A voice's language is matched by its espeak-ng voice name or its language tag: `es`
and `es-ES` are `es`; `pt-br` / `pt-BR` are `pt-br`; `fr-fr` is `fr`; `no` and `nb` are `nb`;
`en-gb-x-rp` is `en-gb`; Piper's Flemish voices (espeak-ng `nl`) are `nl`. Spanish outside Spain
(espeak-ng `es-419`, Piper's `es_MX` and `es_AR` voices), Canadian French, Mandarin (`cmn`),
Russian and the rest stay with espeak-ng.

With the default setting (`phonemizer = "auto"`) a voice uses uccharan when uccharan covers its
language, and espeak-ng otherwise. That makes **69 catalog voices work without espeak-ng** (all
Piper voices in the languages above: 12 US English, 6 British English, 3 each Hindi, German, French,
Nepali and Telugu, 2 each Spanish, Italian, Polish, Turkish, Urdu, Malayalam and Tamil, and Danish,
Dutch, Flemish, Norwegian, Swedish, Finnish, Czech, Hungarian, Romanian, Catalan, Brazilian and
European Portuguese, Marathi and Bengali; the Mimic 3 voices in Bengali, Gujarati and Nepali; the 4
KittenTTS models; Matcha English) plus Kokoro's English, Spanish, French, Italian, Portuguese and Hindi
voices. The other 24 phoneme-based voices (Piper voices in Russian, Ukrainian, Arabic, Persian, Greek,
Mexican and Argentinian Spanish, Mandarin and others, and Kokoro Japanese) need espeak-ng. The table is
code (`uccharan_tag` and `UCCHARAN_SUPPORT` in `server/src/bolo/text/phonemize.py`); a uccharan older
than 0.6 is used only for the languages it has.

**Language packs.** uccharan's wheel carries American English and Hindi. The word table of every
other language, British English and Kokoro's English dictionaries are packs (`de.ucd`, `en-gb.ucd`,
`kokoro-en-us.ucd` …, 46 KB to 2.1 MB) attached to uccharan's GitHub release. **bolo downloads a pack
the first time a voice needs it**: the download shows as a `pull` job (`GET /api/v1/jobs`, the
events stream) and in the log, is checked against the SHA-256 the installed uccharan lists for it,
and lands in uccharan's cache (`~/.cache/uccharan/<version>/`, or `$XDG_CACHE_HOME/uccharan/…`),
where uccharan checks it again when it loads it. Each download happens once per machine.

* `uccharan_auto_download = false` (configuration.md), or `UCCHARAN_OFFLINE=1` in bolo's
  environment, turns this off: a voice whose pack is missing answers `424 component_missing` with
  the pack's name and the command that fetches it.
* Fetch packs yourself with `bolo pull uccharan-de` (or `bolo components install uccharan-de`,
  `POST /api/v1/models/pull` with `{"id": "uccharan-de"}`, the console's Components screen) or with
  uccharan's own `python -m uccharan.packs de fr ur` (`--all` for every Apache-2.0 pack).
* Offline machines: `python -m uccharan.packs --dir DIR de fr` on a machine with network (or take the
  files from the release page), copy `DIR` over and name it in `UCCHARAN_DATA_DIR`.
* Every pack is listed as a component, `uccharan-<pack>`, with its size and licence
  (`bolo components list`, `bolo doctor`, `GET /api/v1/components`): 35 in uccharan 0.6.
* The **CC BY-SA 4.0** packs, `uccharan-wikipron-*` (Wiktionary pronunciations for improved mode)
  and `uccharan-xlit-*` (romanised text, below), are never downloaded automatically, only when you
  install them with `--accept-license` (`"accept_license": true` over HTTP).

While bolo runs it sets `UCCHARAN_OFFLINE=1` in its own process, so uccharan never downloads
anything behind bolo's back; the packs bolo fetched are the ones uccharan reads.

You can choose, at three levels (a request beats a pin, a pin beats the server setting):

* per request: `phonemizer` in the body of `/v1/audio/speech` (ignored by OpenAI clients, which do not
  send it), `?phonemizer=` on the TTS WebSocket, `session.audio.output.phonemizer` in a Realtime
  session, `audio.phonemizer` in `/v1/chat/completions`;
* per voice, persistently: `bolo models config piper-en-us-amy-low --phonemizer espeak-ng`, or
  `PUT /api/v1/models/{id}/settings` with `{"phonemizer": "espeak-ng"}` (`null` removes the pin);
* server-wide: the `phonemizer` setting (`auto`, `uccharan`, `espeak-ng`).

Values: `uccharan` (= `uccharan:compatible`, reproduces what the voices were trained on),
`uccharan:improved` (uccharan's cleaner readings of homographs, numbers and Hindi schwa deletion;
not what the voices were trained on), `espeak-ng`, `auto`. Asking for espeak-ng when it is not
installed answers `424 component_missing`; asking for uccharan in a language it does not cover
answers `400` with the list of languages it does cover. Each catalog entry lists the choices for
its voice in `phonemizers` (`[{id, available, default, note}]`; for Kokoro per voice).

### Text normalisation, word lists and romanised text

uccharan is also bolo's text normaliser, apart from phonemes. For voices that read text themselves
(`tts-server` models, Orpheus-family voices on llama.cpp, plugin voices) its normalise-only mode
writes numbers, money, dates, times, phone numbers, units, abbreviations, URLs and email addresses
out in words in the voice's language: English, Hindi and the other South Asian languages, Urdu,
Sindhi and, since uccharan 0.6.1, its Latin-script
languages too (Spanish, German, French, …; also currency signs, the decimal comma and letters of
foreign names). uccharan 0.6.0 read digits in English in those languages and 0.6.1 wrote Sinhala
and Sindhi numbers as lookup keys that are no words (0.6.2 writes words; 0.6.4, the minimum `bolo-speech[uccharan]`
asks for, also reads clock times, ordinals and dates and puts currency words in the plural); bolo asks the installed uccharan to write out a few numbers in the
language and, when they do not come back as words of the language's script, gives those text
voices its own light normaliser (digits left for the voice). Voices that read phonemes
are not affected: uccharan's phonemiser reads numbers, dates and money in every one of its
languages. Pronunciation word lists (`/api/v1/wordlists`) set how
names, brands and jargon are said for every kind of voice: as phonemes, as `[word](/ipa/)` markup
or as a respelling, whichever the voice can take. Each voice's text-input record says which
(`GET /api/v1/text-input/{model}`; api.md, "Text preparation, SSML and pronunciation word lists").

**Romanised text** is the third part: an Indian language typed in Latin letters ("mujhe kal
office jana hai") is written in the voice's script ("मुझे कल office जाना है") before the
normaliser and the phonemiser see it; English words are left as English. It applies to every
kind of voice (phonemes, text, characters) because it runs first.

| | |
| --- | --- |
| Modes | `off` (the default), `auto` (when the voice speaks an Indian language and at least 60 % of the letters are Latin), `on` |
| Where it is set | per request (`romanised` in `/v1/audio/speech`, `?romanised=` on the TTS WebSocket, `session.audio.output.romanised`, `audio.romanised` in chat completions, `romanised` on a telephony agent, `bolo speak --romanised`); per model (`PUT /api/v1/models/{id}/settings` with `{"romanised": "auto"}`, or the catalog option `romanised`); server-wide (the `romanised` setting) |
| What it needs | uccharan with its transliterator (bolo asks for 0.6) and the language's pack |
| Packs | `xlit-<language>`; uccharan lists Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Punjabi, Kannada and Malayalam. **CC BY-SA 4.0** (built from the Dakshina and Aksharantar word lists), unlike uccharan itself (Apache-2.0), so a pack is only installed on request: `bolo components install uccharan-xlit-hi --accept-license`, the console's Components screen, `bolo pull uccharan-xlit-hi --accept-license`, or `python -m uccharan.packs xlit-hi` |
| When something is missing | the text is spoken as written and the request's trace says what is missing and how to install it; only a request that itself says `"romanised": "on"` is refused (`424 component_missing`) |

Not converted: words that have a pronunciation of their own (word lists, inline
`pronunciations`, `[word](/ipa/)`), URLs and email addresses. What is not known: how good the
conversion sounds on bolo's voices. The transliterator's accuracy is measured in uccharan's own
documentation; in bolo the option was tested with a stand-in transliterator and against a development
build of uccharan (after 0.3.0) with its Hindi pack, and no listening or round-trip test of converted text was made.

### Measured: uccharan against espeak-ng, voice by voice

**Run 2, 500 sentences per voice (paired).** The 500 held-out sentences per language that
uccharan's own end-to-end evaluation uses (selected the same way from its held-out sets, which were
never used to write its rules), each synthesised through bolo twice, once per phonemiser, and both
results transcribed back: English by `whisper-base` (sherpa-onnx int8, Whisper's English
normaliser), Hindi by `indicconformer-hi-community`. The interval is the 95 % paired bootstrap
interval of WER(uccharan) − WER(espeak-ng) over sentences. 2 threads per process, two voices at a
time on the shared 4-vCPU machine (load average 0.8 to 5.3 for amy and rohan, 3.2 to 7.0 for lessac
and alba, from other jobs). espeak-ng 1.52.0, uccharan 0.1.0, 2026-11-13.

| Voice | WER uccharan | WER espeak-ng | Difference (95 % CI), points | Sentences better / worse with uccharan | First audio (median) uccharan / espeak-ng |
| --- | ---: | ---: | --- | --- | --- |
| `piper-en-us-amy-low` | 2.89 % | 2.65 % | +0.24 (−0.25 … +0.81) | 20 / 25 | 166 / 167 ms |
| `piper-en-us-lessac-medium` | 3.67 % | 3.94 % | −0.27 (−1.05 … +0.44) | 48 / 51 | 371 / 370 ms |
| `piper-en-gb-alba-medium` | 6.52 % | 6.65 % | −0.13 (−1.16 … +0.87) | 71 / 70 | 398 / 397 ms |
| `piper-hi-in-rohan-medium` | 8.83 % | 9.45 % | −0.62 (−1.46 … +0.25) | 56 / 52 | 221 / 211 ms |

**Every interval contains zero: no voice is measurably better or worse with either phonemiser.**
A check of how much of the difference is noise: the two phonemisers give identical phonemes for
377 of the 500 English sentences (347 for alba) and 494 of 500 Hindi ones, and even on those
sentences the error counts differ (amy 82 against 75 errors, rohan 274 against 294), because VITS
samples its output and the recogniser reacts to small differences. Differences of this size
between the phonemisers are therefore within what the synthesis itself varies. Where the
phonemes do differ, the sentences that came out worse with uccharan involve names (Gabor,
Tatoeba, Walid, Janos), a few vowel choices (`beyond`, `realize`), and contractions that espeak-ng
joins with the next word (`wouldn't have`); the list is in the measurement file, for uccharan's
maintainers.

**Run 1, 60 sentences** (the last 60 lines of the same held-out sets, not paired: one run per
phonemiser, 2 threads, busier machine). It is superseded by run 2 and kept because the first
version of this page quoted it; its differences were within noise too (60 sentences are about
600 words, one word is 0.17 points).

| Voice | Phonemiser | WER | CER | First audio (median / p90) |
| --- | --- | ---: | ---: | --- |
| `piper-en-us-amy-low` | uccharan | 2.99 % | 1.26 % | 135 / 279 ms |
| | uccharan:improved | 2.99 % | 1.26 % | 130 / 229 ms |
| | espeak-ng | 2.61 % | 1.15 % | 148 / 263 ms |
| `piper-en-us-lessac-medium` | uccharan | 3.36 % | 1.52 % | 145 / 248 ms |
| | espeak-ng | 2.43 % | 1.00 % | 252 / 395 ms |
| `piper-en-gb-alba-medium` | uccharan | 5.04 % | 2.89 % | 152 / 245 ms |
| | espeak-ng | 4.48 % | 2.67 % | 182 / 324 ms |
| `piper-hi-in-rohan-medium` | uccharan | 7.65 % | 2.68 % | 146 / 204 ms |
| | uccharan:improved | 8.22 % | 3.01 % | 153 / 265 ms |
| | espeak-ng | 9.07 % | 2.68 % | 170 / 238 ms |
| `kitten-nano-0.8-int8` | uccharan | 1.68 % | 0.82 % | 890 / 1512 ms |
| | espeak-ng | 1.87 % | 0.78 % | 1198 / 1928 ms |
| `kokoro-v1.0-int8`, `af_heart` | uccharan (misaki port) | 2.99 % | 1.22 % | 2373 / 4102 ms |
| | espeak-ng | 2.99 % | 1.82 % | 2518 / 4636 ms |
| | neither (bundled lexicon only) | 6.72 % | 6.93 % | 2361 / 3968 ms |
| `kokoro-v1.0-int8`, `hf_alpha` | uccharan | 5.38 % | 1.77 % | 2856 / 4042 ms |
| | espeak-ng | 4.53 % | 1.44 % | 3018 / 4469 ms |

The one clear result of run 1 stands: Kokoro English without a phonemiser (the bundled lexicon
skips words it does not know) has more than twice the error. Kitten and Kokoro were not part of
run 2. `uccharan:improved` changed 12 of the 60 English sentences (for example "OK" read as
"okay") without changing the error rate. Raw rows of both runs:
`docs/measurements/phonemizer-ab.json`.

## Default voices

Two defaults, measured on the same machine (4 threads unless noted; details in
`docs/measurements/tts-first-audio.json`):

* `default_tts` = `kokoro-v1.0-int8`: the quality voice (54 voices, 8 languages). Requests that name
  no model, `tts-1-hd` and `gpt-4o-mini-tts` use it.
* `default_realtime_tts` = `piper-en-us-amy-low`: the low-latency voice, for voice agents (Realtime
  API, chat completions with audio), Wyoming, `tts-1` and `model: "realtime"`. With uccharan it needs
  no optional component; its licence allows commercial use (`lessac` does not), and it is 16 kHz,
  which suits telephony. A voice that only Kokoro has (`af_heart`) still selects Kokoro.

Time to first audio through bolo for replies of two sentences (about 150 characters), first request
after loading and the median / p90 of 19 more. The machine was heavily loaded by other jobs during
these runs (load average 4.4 to 7.3 on 4 vCPUs), so the absolute numbers are pessimistic; compare rows.

| Voice | Threads | First chunk | Warm-up on load | Cold | Median / p90 | Round-trip WER |
| --- | --- | --- | --- | --- | --- | --- |
| `kokoro-v1.0-int8` (before) | 4 | 120 chars | no | 2484 ms | 3312 / 5704 ms | 1.35 % |
| `kokoro-v1.0-int8` | 4 | 120 chars | yes | 3872 ms | 2755 / 4642 ms | |
| `kokoro-v1.0-int8` (now) | 4 | 48 chars | yes | 3055 ms | 2325 / 3338 ms | 1.89 % |
| `kokoro-v1.0-int8` | 2 | 48 chars | yes | 1887 ms | 2327 / 2842 ms | |
| `kokoro-v1.0` (float32) | 4 | 48 chars | yes | 1233 ms | 1572 / 2018 ms | |
| `piper-en-us-amy-low` | 4 | 120 chars | yes | 308 ms | 402 / 646 ms | 1.35 % |
| `piper-en-us-amy-low` | 2 | 120 chars | yes | 113 ms | 148 / 232 ms | |

What this says:

* **Kokoro is slow to start on this CPU whatever we do.** sherpa-onnx alone needs 1.4 to 2.2 s to
  say "Hello." with the int8 model (best of three, under the same load), so a large part is fixed
  cost per call, not length. A shorter first chunk (the catalog now sets `first_chunk_chars: 48`
  for Kokoro, so the first piece ends at a clause or about 48 characters) cut the median by about
  15 % and the p90 by about 30 % without a measurable change in the round-trip error.
* **The int8 export is not the fast one**: the float32 `kokoro-v1.0` started speaking about a third
  sooner in the same conditions (it is 383 MB instead of 172 MB, and uses more memory). If first
  audio matters more than download size, set `default_tts = "kokoro-v1.0"`.
* **Threads**: on a loaded machine 2 threads were as fast as 4 for Kokoro and much faster for Piper;
  on an idle machine 4 is the default. Leave `threads` at its default unless you measured otherwise.
* **Warm-up on load** (`warm_up: true` in a catalog entry's options: one short synthesis right
  after loading) did not show a reliable gain in these noisy runs and is off by default.
* **Piper `amy` starts speaking in about 0.15 s** with the same round-trip error, which is why it is
  the realtime default; Kokoro is the voice to choose for quality and for its 54 voices.

## Licences and the espeak-ng question

bolo is Apache-2.0 and contains no third-party code. What you install next to
it, and what you download, carry their own terms.

### Engines

| Engine | Licence | Notes |
| --- | --- | --- |
| sherpa-onnx | Apache-2.0 | bundles ONNX Runtime (MIT). **From version 2.0 its binaries no longer contain espeak-ng** (see below) |
| faster-whisper / CTranslate2 | MIT / MIT | |
| useful-moonshine-onnx | MIT | optional |
| onnxruntime | MIT | runs the optional end-of-turn model (Smart Turn v3, BSD-2-Clause); installed with faster-whisper anyway |
| aiortc | BSD-3-Clause | optional (`bolo-speech[webrtc]`): the WebRTC transport of the Realtime API; brings PyAV |
| pywhispercpp / whisper.cpp | MIT / MIT | optional |
| cactus-compute / libcactus | Apache-2.0 / Apache-2.0 | optional (`bolo-speech[cactus]`): runs Whistle's `.cact` weights in process; the package downloads its engine library on first use, and bolo turns its anonymous usage counts off (`CACTUS_TELEMETRY=0`) |
| kokoro-onnx | MIT, **but depends on `phonemizer` (GPL-3.0-or-later)** and loads espeak-ng in process | optional, deliberately not in `bolo-speech[all]` |
| piper-tts (`piper1-gpl`) | GPL-3.0 | **not used and not supported**: Piper-format voices run through sherpa-onnx instead |
| llama.cpp (`llama-server`) | MIT | optional, a separate process: the `tts-server` engine starts it or connects to one (`bolo components install llama-server` downloads the upstream release) |
| audio.cpp (`audiocpp_server`) | Apache-2.0 | optional, a separate process, as llama-server. Model weights keep their own licences (audio.cpp's `docs/model_licenses.md`) |

Codec libraries: `soundfile` bundles libsndfile (LGPL-2.1+) with mpg123 and
LAME (LGPL), `soxr` is LGPL-2.1+, PyAV bundles FFmpeg libraries (LGPL-2.1+ in
the published wheels). These are dynamically loaded libraries used through
their public interfaces.

### espeak-ng

Most small neural voices (every Piper-format voice, KittenTTS, Matcha-TTS
English, Kokoro for languages other than English and Mandarin) are trained on
phonemes produced by **espeak-ng, which is GPL-3.0-or-later**. Something has
to turn text into those phonemes at run time. What we found:

* The original Piper runtime moved to GPL-3.0 (`piper1-gpl`, the `piper-tts`
  package). bolo does not depend on it.
* sherpa-onnx used to link espeak-ng into its binaries (through
  piper-phonemize). **sherpa-onnx 2.0 removed that**: the library's own
  runtime warnings say espeak-ng "is no longer used" and that callers must
  pass phonemes in (`GenerationConfig.phoneme_codepoints`) or use a lexicon.
  We verified on the 2.0.1 wheel that the `data_dir` options are ignored.
  bolo requires sherpa-onnx ≥ 2.0 for this reason.
* So bolo has a phonemiser front end (`bolo/text/phonemize.py`): uccharan
  (Apache-2.0) where it covers the language, espeak-ng otherwise (section
  "Phonemisers" above).

**bolo does not ship espeak-ng.** It is not in the wheel, not in
`bolo-speech[all]`, and not in the Docker images unless you build them with
`--build-arg WITH_ESPEAK=1`. It is an *optional component*, installed only
when you ask for it. Voices that read phonemes have `requires: ["phonemizer"]`
in the catalog files; in the API their `requires` names the phonemiser they
would use here (`["uccharan"]` for the 69 in uccharan's languages, `["espeak-ng"]` for the
24 others, marked "needs espeak-ng" in the tables above), and they show as
unavailable until it is installed; using one answers `424 component_missing`
with the install instructions instead of producing silence.

Ways to install it, any one of:

| Route | What happens |
| --- | --- |
| the console (Models → Optional components, or the notice next to a voice) | asks you to accept the licence, then the same as the next row |
| `bolo components install espeak-ng --accept-license`, or `POST /api/v1/components/espeak-ng/install` with `{"accept_license": true}` | `pip install espeakng-loader` (or `uv pip install`) into the server's own Python environment; a running server starts using it without a restart |
| `pip install "bolo-speech[espeak]"` | the same package, installed by you |
| `apt install espeak-ng` / `brew install espeak-ng` | the system executable |

`bolo components list`, `bolo doctor` and `GET /api/v1/components` show
whether it is installed and from where. The phonemiser looks, in order, for:
an explicit setting (`BOLO_ESPEAK_LIBRARY`), the `espeak-ng` executable on
`PATH` (a separate process: text in, phonemes out), then the library inside
the `espeakng-loader` package (loaded into the server process with `ctypes`).
`BOLO_ESPEAK=off` disables all of them.

Installing through the API is refused when the server listens on a
non-loopback address without an API key, when the operator set
`allow_component_install = false`, and when the environment is read-only or
has neither pip nor uv; `install_blocked_reason` says which, and
`manual_command` gives the command for your kind of installation.

Things to know:

* `espeakng-loader` is a third-party wheel (espeak-ng 1.52 built for Linux
  x86-64/aarch64, macOS and Windows). Its package metadata declares no
  licence; the library inside is GPL-3.0-or-later.
* **What was compared:** most measurements here used the library from that
  wheel (espeak-ng 1.52.0). The executable route was also run with a real
  `espeak-ng` 1.53.0 binary: three of four test sentences (English, German)
  gave identical phoneme strings, the Hindi one differed in how one doubled
  consonant is written, and `piper-en-us-amy-low`, `kitten-nano-0.8-int8` and
  `piper-hi-in-rohan-medium` each round-tripped through speech recognition
  with 0 % word error either way (`docs/measurements/espeak-executable.json`).
  Other espeak-ng versions and other languages were not compared. The
  one-click install was run for real in a clean environment (no espeak-ng
  anywhere): a Piper voice answered `component_missing`, the install job
  finished, and the same server process then synthesised with that voice.
* bolo's phonemiser reproduces Piper's conventions (NFD-normalised IPA,
  punctuation kept) on top of stock espeak-ng. Piper itself uses a patched
  espeak-ng that reports clause terminators; bolo splits clauses itself
  instead. Round-trip tests show intelligible speech, but prosody at clause
  boundaries may differ slightly from the reference Piper runtime.
* We are not lawyers. Running a GPL program as a separate process and
  exchanging text with it is generally understood not to extend the GPL to
  the caller; loading a GPL library into your process is a different
  situation if you distribute the combination. bolo itself stays Apache-2.0
  because it distributes neither; what you install and redistribute is your
  decision. If it matters to you, use the system executable, or stay with
  voices that need no espeak-ng at all.

Voices that never need espeak-ng: every English and Hindi voice that reads
phonemes (through uccharan), **Kokoro's Mandarin voices** (lexicon),
**MeloTTS**, **MMS-TTS** (non-commercial licence), the **Coqui VITS**
Bengali voices and the SYSPIN voices.

### Models

The licence column in the table above is the model publisher's statement,
copied when the entry was written. Things to watch:

* **Piper-format voices** inherit the licence of their training data, which
  varies voice by voice (CC0, CC-BY, CC-BY-SA, CC-BY-NC-SA, research-only
  corpora, or simply "see the dataset page"). Entries marked ⚠ are
  non-commercial as far as the voice card says. Where the card only links to
  a dataset licence, the entry says so and `commercial_use` is left true:
  check before you ship.
* **Moonshine**: English models are MIT; the non-English models are under the
  Moonshine Community License, which is non-commercial.
* **MMS-TTS** is CC-BY-NC-4.0: non-commercial use only (`commercial_use: false`). bolo lists
  non-commercial and gated models because the choice is yours; the licence, `commercial_use` and
  `gated` are on every catalog entry and can be filtered on.
* **SenseVoice / Paraformer** use Alibaba's FunASR model licence (custom).
* **Kroko** community models are CC-BY-SA.
* **Parakeet, Canary, NeMo FastConformer** are CC-BY-4.0.
* **DiCoW v3.3** (`dicow-v3.3`) is CC-BY-4.0: commercial use allowed with attribution to BUT Speech@FIT
  (the model card asks for the DiCoW and SE-DiCoW papers to be cited); **SE-DiCoW** (`se-dicow`) is
  Apache-2.0. Both are fine-tuned from Whisper large-v3-turbo (MIT) on AMI and NOTSOFAR-1 (CC BY 4.0) and
  LibriMix / LibriSpeech mixtures; LibriMix's noisy variants contain WHAM! noise (CC-BY-NC-4.0) and the
  cards do not say which variants were used (unverified). Both need the bolo-torch plugin.
* **Nemotron-3-Diarization** (`nemotron-3-diarization`) is under the OpenMDW License Agreement 1.1:
  commercial use allowed; whoever redistributes the model files must include the agreement and the
  model's notices; suing over patents or copyright in the model ends the licence. The catalog uses a
  third-party ONNX conversion (onnx-community); see "Speaker diarization" above.
* **TEN VAD** (`ten-vad`, `ten-vad-int8`) is Apache-2.0 with Agora's additional conditions: deployment only for
  your own applications and their direct end users, not in a way that competes with Agora's offerings or lets
  third parties develop or deploy applications; derivative works stay under the same terms. `commercial_use` is
  false because of those conditions. The files are the ones sherpa-onnx publishes (the ONNX file in TEN's
  repository lacks the metadata sherpa-onnx reads).
* **MarbleNet** (`marblenet-vad`) is under the NVIDIA Open Model License: commercial use allowed; suing over the
  model ends the licence; redistributors keep NVIDIA's notice. The catalog uses a third-party ONNX export of
  NVIDIA's .nemo checkpoint.
* **WebRTC VAD** (`webrtc-vad`) is BSD-3-Clause code in an MIT-licensed binding (`webrtcvad-wheels`); nothing is
  downloaded.
* LiveKit's turn detector and TEN Turn Detection are not in the catalog: see
  [models-unsupported.md](models-unsupported.md) for their licences and why bolo does not run them.
* **Namo Turn Detector** (`namo-turn-en`, `namo-turn-multilingual`) is Apache-2.0.
* **Gated** entries (`gated: true`: Indic Parler-TTS, IndicF5, SraVaani, AI4Bharat's official
  IndicConformer) are never downloaded without a Hugging Face token; using one without a token
  answers `401 model_gated` with the model page to accept the terms on. All four also need the
  PyTorch engine plugin bolo-torch ([plugins.md](plugins.md)); SraVaani and the official
  IndicConformer are not implemented in bolo-torch 0.1.0. PixIT (`pixit-ami`) is gated too, on two
  repositories.
