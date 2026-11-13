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
```

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

GPU execution is untested for every model.

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
253 models: 108 validated on CPU by the maintainers, 145 untested. 18 more are described but not yet published (see "Voices converted by this project").

### Speech to text

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

#### Whisper

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
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

#### Parakeet

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `parakeet-tdt-0.6b-v2` | en | accuracy | pseudo | 661 MB | CC-BY-4.0 | **CPU** · RTF 0.081 |
| `parakeet-tdt-0.6b-v3` | bg, hr, cs, da, nl, en +19 | accuracy | pseudo | 671 MB | CC-BY-4.0 | **CPU** · RTF 0.096 |
| `parakeet-tdt-ctc-110m` | en | balanced | pseudo | 458 MB | CC-BY-4.0 | **CPU** · RTF 0.030 |
| `parakeet-tdt-110m` | en | balanced | pseudo | 477 MB | CC-BY-4.0 | untested |
| `parakeet-unified-en-streaming-240ms` | en | realtime | native | 663 MB | CC-BY-4.0 | untested |
| `parakeet-unified-en-streaming-560ms` | en | realtime | native | 663 MB | CC-BY-4.0 | **CPU** · RTF 4.694 |
| `parakeet-unified-en-streaming-1120ms` | en | balanced | native | 663 MB | CC-BY-4.0 | untested |

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

### Text to speech

#### Parler-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-parler-tts` | as, bn, en, gu, hi, kn +11 | accuracy | - | 3.8 GB | Apache-2.0 · needs plugin torch · gated | untested |

#### F5-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-f5` | as, bn, gu, hi, kn, ml +5 | accuracy | - | 1.4 GB | MIT · needs plugin torch · gated | untested |

#### Veena

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `veena` | en, hi | accuracy | - | 7.6 GB | Apache-2.0 · needs plugin torch | untested |

#### Svara

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `svara-tts-v1` | as, bho, bn, brx, doi, en +13 | accuracy | - | 13.2 GB | Apache-2.0 · needs plugin torch | untested |

#### MMS-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mms-tts-pan` | pa | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-ory` | or | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-asm` | as | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-dgo` | doi | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-mai` | mai | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-mag` | mag | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-hne` | hne | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-awa` | awa | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-bgc` | bgc | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-urd` | ur | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-hin` | hi | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-ben` | bn | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-guj` | gu | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-kan` | kn | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-mal` | ml | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-mar` | mr | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-tam` | ta | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-tel` | te | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | not yet published |
| `mms-tts-eng` | en | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | **CPU** · RTF 0.196 · first audio 590 ms · round-trip WER 5.0% |
| `mms-tts-deu` | de | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-fra` | fr | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-spa` | es | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-rus` | ru | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-ukr` | uk | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-tha` | th | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |

#### Piper

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `piper-hi-in-pratham-medium` | hi | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · phonemes: uccharan | **CPU** · RTF 0.152 · first audio 238 ms |
| `piper-hi-in-priyamvada-medium` | hi | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · phonemes: uccharan | **CPU** · RTF 0.056 · first audio 96 ms |
| `piper-hi-in-rohan-medium` | hi | realtime | - | 63 MB | See dataset licence: https://www.iitm.ac.in/donlab/indict… · phonemes: uccharan | **CPU** · RTF 0.092 · first audio 164 ms |
| `piper-ne-np-google-medium` | ne | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 International · needs espeak-ng | **CPU** · RTF 0.040 · first audio 88 ms |
| `piper-ne-np-google-x-low` | ne | realtime | - | 28 MB | Voice: CC-BY-SA-4.0 International · needs espeak-ng | untested |
| `piper-ur-pk-fasih-medium` | ur | realtime | - | 64 MB | See voice card · needs espeak-ng | **CPU** · RTF 0.090 · first audio 191 ms |
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
| `piper-de-de-thorsten-medium` | de | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | **CPU** · RTF 0.251 · first audio 187 ms |
| `piper-de-de-thorsten-high` | de | realtime | - | 114 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-de-de-kerstin-low` | de | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-es-es-davefx-medium` | es | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | **CPU** · RTF 0.173 · first audio 191 ms |
| `piper-es-es-sharvard-medium` | es | realtime | - | 77 MB | See dataset licence: http://creativecommons.org/licenses/… · needs espeak-ng | untested |
| `piper-es-mx-ald-medium` | es | realtime | - | 63 MB | See dataset licence: http://unlicense.org · needs espeak-ng | untested |
| `piper-es-ar-daniela-high` | es | realtime | - | 114 MB | Voice: Attribution-ShareAlike 4.0 International · needs espeak-ng | untested |
| `piper-fr-fr-siwis-medium` | fr | realtime | - | 63 MB | Voice: CC-BY 4.0 · needs espeak-ng | **CPU** · RTF 0.132 · first audio 225 ms |
| `piper-fr-fr-upmc-medium` | fr | realtime | - | 77 MB | See dataset licence: http://creativecommons.org/licenses/… · needs espeak-ng | untested |
| `piper-fr-fr-tom-medium` | fr | realtime | - | 64 MB | Voice: AGPLv3 · needs espeak-ng | untested |
| `piper-it-it-paola-medium` | it | realtime | - | 64 MB | See dataset licence: https://huggingface.co/datasets/paol… · needs espeak-ng | **CPU** · RTF 0.173 · first audio 300 ms |
| `piper-it-it-riccardo-x-low` | it | realtime | - | 28 MB | See dataset licence: https://www.caito.de/2019/01/03/the-… · needs espeak-ng | untested |
| `piper-pt-br-faber-medium` | pt | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | **CPU** · RTF 0.181 · first audio 287 ms |
| `piper-pt-pt-tugao-medium` | pt | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-nl-nl-mls-medium` | nl | realtime | - | 77 MB | Voice: CC-BY 4.0 · needs espeak-ng | untested |
| `piper-nl-be-nathalie-medium` | nl | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-pl-pl-gosia-medium` | pl | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-pl-pl-darkman-medium` | pl | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ru-ru-irina-medium` | ru | realtime | - | 63 MB | Voice: Unknown · needs espeak-ng | **CPU** · RTF 0.233 · first audio 381 ms |
| `piper-ru-ru-denis-medium` | ru | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ru-ru-dmitri-medium` | ru | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ru-ru-ruslan-medium` | ru | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-uk-ua-ukrainian-tts-medium` | uk | realtime | - | 77 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-tr-tr-dfki-medium` | tr | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-tr-tr-fahrettin-medium` | tr | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ar-jo-kareem-medium` | ar | realtime | - | 63 MB | See dataset licence: https://github.com/AliMokhammad/arab… · needs espeak-ng | untested |
| `piper-fa-ir-amir-medium` | fa | realtime | - | 64 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-fa-ir-gyro-medium` | fa | realtime | - | 63 MB | See dataset licence: https://github.com/gyroing · needs espeak-ng | untested |
| `piper-zh-cn-huayan-medium` | zh | realtime | - | 63 MB | Voice: Unknown · needs espeak-ng | **CPU** · RTF 0.047 · first audio 160 ms |
| `piper-vi-vn-vais1000-medium` | vi | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… · needs espeak-ng | untested |
| `piper-id-id-news-tts-medium` | id | realtime | - | 63 MB | See dataset licence: https://www.kaggle.com/code/mpwolke/… · needs espeak-ng | untested |
| `piper-sv-se-nst-medium` | sv | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-da-dk-talesyntese-medium` | da | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-no-no-talesyntese-medium` | no | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-fi-fi-harri-medium` | fi | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-cs-cz-jirka-medium` | cs | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-el-gr-rapunzelina-low` | el | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-hu-hu-anna-medium` | hu | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-ro-ro-mihai-medium` | ro | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-sk-sk-lili-medium` | sk | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-sr-rs-serbski-institut-medium` | sr | realtime | - | 77 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-sl-si-artur-medium` | sl | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… · needs espeak-ng | untested |
| `piper-ca-es-upc-ona-medium` | ca | realtime | - | 63 MB | Voice: CC BY-SA 3.0 ES · needs espeak-ng | untested |
| `piper-cy-gb-gwryw-gogleddol-medium` | cy | realtime | - | 64 MB | See dataset licence: https://git.techiaith.bangor.ac.uk/d… · needs espeak-ng | untested |
| `piper-is-is-bui-medium` | is | realtime | - | 77 MB | See dataset licence: http://hdl.handle.net/20.500.12537/104 · needs espeak-ng | untested |
| `piper-ka-ge-natia-medium` | ka | realtime | - | 63 MB | See LICENSE file · needs espeak-ng | untested |
| `piper-kk-kz-issai-high` | kk | realtime | - | 128 MB | Voice: CC-BY-4.0 · needs espeak-ng | untested |
| `piper-lb-lu-marylux-medium` | lb | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · needs espeak-ng | untested |
| `piper-lv-lv-aivars-medium` | lv | realtime | - | 64 MB | Voice: CC0 1.0 Universal · needs espeak-ng | untested |
| `piper-sw-cd-lanfrica-medium` | sw | realtime | - | 63 MB | See dataset licence: https://lanfrica.com/record/kiswahil… · needs espeak-ng | untested |
| `piper-ne-np-chitwan-medium` | ne | realtime | - | 63 MB | Voice: CC0-1.0 (dataset, per the voice card) · needs espeak-ng | **CPU** · RTF 0.070 · first audio 402 ms · round-trip WER 46.2% · round-trip CER 9.3% |
| `piper-te-in-padmavathi-medium` | te | realtime | - | 64 MB | Voice: CC-BY-4.0 (IndicVoices-R, per the voice card) · needs espeak-ng | **CPU** · RTF 0.179 · first audio 1294 ms · round-trip WER 15.4% · round-trip CER 2.6% |
| `piper-te-in-venkatesh-medium` | te | realtime | - | 64 MB | Voice: CC-BY-4.0 (IndicVoices-R, per the voice card) · needs espeak-ng | **CPU** · RTF 0.211 · first audio 1240 ms · round-trip WER 23.1% · round-trip CER 3.8% |
| `piper-te-in-maya-medium` | te | realtime | - | 63 MB | See dataset licence: https://www.iitm.ac.in/donlab/indict… · needs espeak-ng | **CPU** · RTF 0.097 · first audio 596 ms · round-trip WER 7.7% · round-trip CER 1.3% |
| `piper-ml-in-arjun-medium` | ml | realtime | - | 63 MB | See dataset licence (IndicTTS Malayalam corpus; the voice… · needs espeak-ng | **CPU** · RTF 0.191 · first audio 1642 ms · round-trip WER 10.0% · round-trip CER 2.2% |
| `piper-ml-in-meera-medium` | ml | realtime | - | 63 MB | See dataset licence (IndicTTS Malayalam corpus; the voice… · needs espeak-ng | **CPU** · RTF 0.194 · first audio 1221 ms · round-trip WER 10.0% · round-trip CER 6.7% |
| `piper-mr-in-google-medium` | mr | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 (OpenSLR 64) · needs espeak-ng | **CPU** · RTF 0.131 · first audio 932 ms · round-trip WER 33.3% · round-trip CER 6.1% |
| `piper-bn-bd-google-medium` | bn | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 (OpenSLR 37) and the CMU Indic licence · needs espeak-ng | **CPU** · RTF 0.215 · first audio 1012 ms · round-trip WER 33.3% · round-trip CER 6.0% |
| `piper-ur-pk-aegis-female-medium` | ur | realtime | - | 64 MB | Voice: MIT (per the voice card) · needs espeak-ng | **CPU** · RTF 0.137 · first audio 866 ms · round-trip WER 27.8% · round-trip CER 2.8% |
| `piper-ta-in-rasa-female-medium` | ta | realtime | - | 64 MB | Voice: CC-BY-4.0 (AI4Bharat Rasa data; third-party voice … · needs espeak-ng | **CPU** · RTF 0.110 · first audio 487 ms · round-trip WER 0.0% · round-trip CER 0.0% |
| `piper-ta-in-rasa-male-medium` | ta | realtime | - | 64 MB | Voice: CC-BY-4.0 (AI4Bharat Rasa data; third-party voice … · needs espeak-ng | **CPU** · RTF 0.147 · first audio 563 ms · round-trip WER 10.0% · round-trip CER 3.1% |

#### Kokoro

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `kokoro-v1.0-int8` | en, es, fr, hi, it, ja +2 | balanced | - | 172 MB | Apache-2.0 | **CPU** · RTF 0.884 · first audio 2406 ms · round-trip WER 0.0% |
| `kokoro-v1.0` | en, es, fr, hi, it, ja +2 | accuracy | - | 383 MB | Apache-2.0 | **CPU** · RTF 0.759 · first audio 2004 ms · round-trip WER 0.0% |

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
| `mimic3-gu-cmu-indic` | gu | balanced | - | 76 MB | See voice card (CMU Indic dataset) · needs espeak-ng | **CPU** · RTF 0.046 · first audio 82 ms |
| `mimic3-bn-multi` | bn | balanced | - | 76 MB | See voice card (OpenSLR Bengali, CC-BY-SA-4.0) · needs espeak-ng | **CPU** · RTF 0.139 · first audio 282 ms |
| `mimic3-ne-google` | ne | balanced | - | 76 MB | See voice card (OpenSLR Nepali, CC-BY-SA-4.0) · needs espeak-ng | **CPU** · RTF 0.147 · first audio 248 ms |
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

### Voice-activity detection

#### Silero VAD

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `silero-vad` | - | realtime | - | 2 MB | MIT | **CPU** · RTF 0.006 |
| `silero-vad-v5` | - | realtime | - | 2 MB | MIT | **CPU** · RTF 0.006 |

### Speaker diarization

#### pyannote + speaker embeddings

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `pyannote-3.0-eres2net` | - | balanced | - | 32 MB | MIT (segmentation) + Apache-2.0 (embedding) | **CPU** · RTF 0.106 · DER 5.2% on a two-voice test clip |
| `pyannote-3.0-titanet-small` | - | balanced | - | 46 MB | MIT (segmentation) + CC-BY-4.0 (embedding) | untested |
| `pyannote-3.0-wespeaker-resnet34` | - | balanced | - | 33 MB | MIT (segmentation) + CC-BY-4.0 (embedding) | untested |

### End-of-turn detection

#### Smart Turn

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `smart-turn-v3` | ar, bn, da, de, en, es +17 | realtime | - | 9 MB | BSD-2-Clause | untested |

⚠ = the licence restricts commercial use. RTF = processing time / audio duration (lower is faster).
<!-- END GENERATED: models -->

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
repository (`https://github.com/cb-Sigma/bolo/releases/tag/voices-v1`) and,
once the workflow has a Hugging Face token, as one public repository per voice
under `codebucketsolutions/` (the catalog then pulls from the Hub, pinned to
the uploaded commit, with the release archive as fallback). Each archive
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

**Meta MMS-TTS (non-commercial): converted, not yet published.** 18 voices
for Indian languages (`mms-tts-<iso>`: Punjabi, Odia, Assamese, Dogri,
Maithili, Magahi, Chhattisgarhi, Awadhi, Haryanvi, Urdu, Hindi, Bengali,
Gujarati, Kannada, Malayalam, Marathi, Tamil, Telugu), from
`facebook/mms-tts` pinned to a commit, converted with MMS's own VITS
inference code (`tools/voices/export_mms.py`, the recipe sherpa-onnx used for
its English MMS voices). **CC-BY-NC-4.0: non-commercial use only**; each
archive carries the licence text and the attribution, and the entries say
`commercial_use: false`. They are `status: pending` until the workflow
publishes them. The conversion was run here for eight languages and each
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

## Word timestamps

| Models | Word times |
| --- | --- |
| faster-whisper (CTranslate2 Whisper) | yes (cross-attention alignment) |
| transducer and CTC models through sherpa-onnx (Zipformer, Parakeet, NeMo, IndicConformer, SenseVoice, …) | yes, from the token times the decoder reports |
| Whisper through sherpa-onnx (`whisper-*`) | **no** with the exports in the catalog: token times need a model exported with cross-attention outputs. sherpa-onnx 2.0.1 supports such exports, and a catalog entry for one can switch them on with the option `token_timestamps: true`; we checked that the stock `whisper-tiny` export reports none |
| Moonshine | **no**: the runtime exposes no token times, and we did not add a forced aligner |

A cheap forced aligner (CTC alignment of the transcript against a small CTC
model) was considered for both and not built: sherpa-onnx's offline
recogniser does not expose the per-frame posteriors such an alignment needs,
so it would mean running a CTC model's ONNX graph and NeMo-compatible feature
extraction ourselves, and we had no reference word timings here to check the
result against.

With `diarize=true`, words also carry their speaker. For word times with
Whisper, use the `faster-whisper-*` entries.

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
  voices for Punjabi, Odia, Assamese, Dogri and more are converted and wait
  for publication. **No voice** that runs without PyTorch for Sanskrit,
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
| **uccharan** (default) | Apache-2.0 | Piper and Kokoro voices in English (US, British) and Hindi; Kitten and Matcha (US English). For Kokoro English it is a port of misaki, the front end Kokoro was trained with, with misaki's dictionaries built in | `bolo[uccharan]`, part of `bolo[all]` and the Docker image; `bolo components install uccharan` |
| **espeak-ng** | GPL-3.0-or-later | every language espeak-ng has | optional component, never installed by default (next section) |

With the default setting (`phonemizer = "auto"`) a voice uses uccharan when uccharan covers its
language, and espeak-ng otherwise. That makes **26 catalog voices work out of the box** (15 US
English and 6 British English Piper voices, 3 Hindi Piper voices, the 4 KittenTTS models, Matcha
English) plus Kokoro's English and Hindi voices; the other 67 phoneme-based voices (Piper voices
in other languages, Mimic 3 voices) still need espeak-ng. The coverage table is data
(`UCCHARAN_SUPPORT` in `server/src/bolo/text/phonemize.py`); it grows when uccharan gains a language.

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

### Measured: uccharan against espeak-ng, voice by voice

60 held-out sentences per language (the last 60 lines of uccharan's held-out evaluation sets,
which were never used to write its rules), synthesised through bolo with each phonemiser and
transcribed back: English by `whisper-base` (sherpa-onnx int8, Whisper's English normaliser),
Hindi by `indicconformer-hi-community` (punctuation stripped, NFC). Time to first audio is the
median per sentence through bolo, 2 threads on the shared 4-vCPU machine (busy: other jobs ran at
the same time, so compare rows with each other rather than with other pages). espeak-ng 1.52.0
from the `espeakng-loader` library; uccharan 0.1.0; 2026-11-13.

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

Reading it: with 60 sentences (about 600 words) one word is 0.17 points, so differences under about
one point are within noise: uccharan and espeak-ng are equivalent on these voices, as uccharan's own
500-sentence evaluation found. The clear result is Kokoro English: without a phonemiser (the
bundled lexicon skips words it does not know) the error is more than twice as high. Time to first
audio is the same or lower with uccharan (it runs in process). `uccharan:improved` changed 12 of the
60 English sentences (for example "OK" read as "okay") without changing the error rate here.
Raw rows: `docs/measurements/phonemizer-ab.json`.

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
| aiortc | BSD-3-Clause | optional (`bolo[webrtc]`): the WebRTC transport of the Realtime API; brings PyAV |
| pywhispercpp / whisper.cpp | MIT / MIT | optional |
| kokoro-onnx | MIT, **but depends on `phonemizer` (GPL-3.0-or-later)** and loads espeak-ng in process | optional, deliberately not in `bolo[all]` |
| piper-tts (`piper1-gpl`) | GPL-3.0 | **not used and not supported**: Piper-format voices run through sherpa-onnx instead |

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
`bolo[all]`, and not in the Docker images unless you build them with
`--build-arg WITH_ESPEAK=1`. It is an *optional component*, installed only
when you ask for it. Voices that read phonemes have `requires: ["phonemizer"]`
in the catalog files; in the API their `requires` names the phonemiser they
would use here (`["uccharan"]` for English and Hindi, `["espeak-ng"]` for the
67 others, marked "needs espeak-ng" in the tables above), and they show as
unavailable until it is installed; using one answers `424 component_missing`
with the install instructions instead of producing silence.

Ways to install it, any one of:

| Route | What happens |
| --- | --- |
| the console (Models → Optional components, or the notice next to a voice) | asks you to accept the licence, then the same as the next row |
| `bolo components install espeak-ng --accept-license`, or `POST /api/v1/components/espeak-ng/install` with `{"accept_license": true}` | `pip install espeakng-loader` (or `uv pip install`) into the server's own Python environment; a running server starts using it without a restart |
| `pip install 'bolo[espeak]'` | the same package, installed by you |
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
* **TEN VAD** is not in the catalog: its licence adds field-of-use conditions
  to Apache-2.0, and the ONNX file in its repository does not load in
  sherpa-onnx without added metadata. The loader kind (`ten_vad`) exists for
  people who have a compatible file.
* **Gated** entries (`gated: true`: Indic Parler-TTS, IndicF5, SraVaani, AI4Bharat's official
  IndicConformer) are never downloaded without a Hugging Face token; using one without a token
  answers `401 model_gated` with the model page to accept the terms on. All four also need the
  PyTorch engine plugin ([plugins.md](plugins.md)), which is not written yet.
