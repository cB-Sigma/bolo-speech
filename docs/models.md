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
277 models: 116 validated on CPU by the maintainers, 161 untested.

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
| `indic-parler-tts` | as, bn, brx, doi, en, gu +15 | accuracy | - | 3.8 GB | Apache-2.0 · needs plugin torch · gated | **CPU** · RTF 22.500 · first audio 39400 ms |
| `parler-tts-tiny-v1` | en | balanced | - | 1.3 GB | Apache-2.0 · needs plugin torch | **CPU** · RTF 4.540 · first audio 11520 ms |

#### F5-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `indic-f5` | as, bn, gu, hi, kn, ml +5 | accuracy | - | 1.4 GB | MIT · needs plugin torch · gated | **CPU** · RTF 119.700 · first audio 211280 ms |

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
| `mms-tts-pan` | pa | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-ory` | or | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-asm` | as | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-dgo` | doi | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-mai` | mai | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-mag` | mag | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-hne` | hne | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-awa` | awa | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-bgc` | bgc | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-urd` | ur | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
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

#### Nemotron diarization

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3-diarization` | - | balanced | - | 147 MB | OpenMDW-1.1 | **CPU** · RTF 0.039 · DER 24.1% on AMI meeting excerpts |

### End-of-turn detection

#### Smart Turn

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `smart-turn-v3` | ar, bn, da, de, en, es +17 | realtime | - | 9 MB | BSD-2-Clause | untested |

### Overlapped-speech separation

#### SepFormer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `sepformer-whamr16k` | en | accuracy | - | 113 MB | Apache-2.0 · needs plugin torch | **CPU** · RTF 3.690 |
| `sepformer-whamr` | en | balanced | - | 113 MB | Apache-2.0 · needs plugin torch | **CPU** · RTF 1.440 |

#### PixIT

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `pixit-ami` | en | accuracy | - | 2.6 GB | MIT · needs plugin torch · gated | **CPU** · RTF 5.690 |

#### VoxLock

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `voxlock-int8` | en | balanced | - | 81 MB | Apache-2.0 | **CPU** · RTF 0.460 |

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
repository (`https://github.com/cb-Sigma/bolo/releases/tag/voices-v1`) and as one
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

Two kinds of diarizer are in the catalog, both without PyTorch:

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
`bolo[all]` through faster-whisper; otherwise `pip install onnxruntime`).

**When to use Nemotron.** Meetings and conversations where people talk over
each other, telephone audio, and voices that sound alike: there it gets the
number of speakers right far more often and almost never mixes two people up
(measurements below), at a fifth of the cost. It is **not the default**
yet: on clean, turn-taking two-voice dialogues it was no better than
clustering and merged some clearly different voices (below).

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

(The research note `docs/_research-overlap-v2.md` measured 0.014 and 98.7 %
for the int8 graph: that was each excerpt in one ONNX pass, without the chunk
loop, against the chunked reference; repeated here, 0.015 and 98.8 %. With
the same chunking the two agree to 0.0009.)

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
recognition (research note, section 7) is the way past it.

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

## Overlapping speech

When two people talk at once, diarization marks the stretch (`overlap: true`)
but the recogniser hears one mixture and usually writes only the louder
voice. `separate_overlaps=true` (api.md, "Separating overlapping speech")
gives each speaker in an overlap a track of their own first. **It is off by
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
* **Nemotron-3-Diarization** (`nemotron-3-diarization`) is under the OpenMDW License Agreement 1.1:
  commercial use allowed; whoever redistributes the model files must include the agreement and the
  model's notices; suing over patents or copyright in the model ends the licence. The catalog uses a
  third-party ONNX conversion (onnx-community); see "Speaker diarization" above.
* **TEN VAD** is not in the catalog: its licence adds field-of-use conditions
  to Apache-2.0, and the ONNX file in its repository does not load in
  sherpa-onnx without added metadata. The loader kind (`ten_vad`) exists for
  people who have a compatible file.
* **Gated** entries (`gated: true`: Indic Parler-TTS, IndicF5, SraVaani, AI4Bharat's official
  IndicConformer) are never downloaded without a Hugging Face token; using one without a token
  answers `401 model_gated` with the model page to accept the terms on. All four also need the
  PyTorch engine plugin bolo-torch ([plugins.md](plugins.md)); SraVaani and the official
  IndicConformer are not implemented in bolo-torch 0.1.0. PixIT (`pixit-ami`) is gated too, on two
  repositories.
