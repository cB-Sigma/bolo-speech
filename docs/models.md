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
225 models: 108 validated on CPU by the maintainers, 117 untested. 22 more are described but not yet published (see "Voices converted by this project").

### Speech to text

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

#### IndicConformer

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
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

#### Nemotron

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3.5-streaming-560ms` | ar, bg, cs, da, de, el +28 | accuracy | native | 682 MB | OpenMDW-1.1 | **CPU** · RTF 0.312 · WER 20.9% · CER 6.1% |

### Text to speech

#### Piper

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `piper-hi-in-pratham-medium` | hi | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · needs espeak-ng | **CPU** · RTF 0.152 · first audio 238 ms |
| `piper-hi-in-priyamvada-medium` | hi | realtime | - | 63 MB | See dataset licence: http://creativecommons.org/licenses/… ⚠ · needs espeak-ng | **CPU** · RTF 0.056 · first audio 96 ms |
| `piper-hi-in-rohan-medium` | hi | realtime | - | 63 MB | See dataset licence: https://www.iitm.ac.in/donlab/indict… · needs espeak-ng | **CPU** · RTF 0.092 · first audio 164 ms |
| `piper-ne-np-google-medium` | ne | realtime | - | 77 MB | Voice: CC-BY-SA-4.0 International · needs espeak-ng | **CPU** · RTF 0.040 · first audio 88 ms |
| `piper-ne-np-google-x-low` | ne | realtime | - | 28 MB | Voice: CC-BY-SA-4.0 International · needs espeak-ng | untested |
| `piper-ur-pk-fasih-medium` | ur | realtime | - | 64 MB | See voice card · needs espeak-ng | **CPU** · RTF 0.090 · first audio 191 ms |
| `piper-en-us-lessac-medium` | en | realtime | - | 63 MB | See dataset licence: https://www.cstr.ed.ac.uk/projects/b… ⚠ · needs espeak-ng | **CPU** · RTF 0.126 · first audio 325 ms · round-trip WER 5.0% |
| `piper-en-us-lessac-low` | en | realtime | - | 63 MB | See dataset licence: https://www.cstr.ed.ac.uk/projects/b… ⚠ · needs espeak-ng | untested |
| `piper-en-us-amy-low` | en | realtime | - | 63 MB | See dataset licence: https://github.com/MycroftAI/mimic3-… · needs espeak-ng | **CPU** · RTF 0.049 · first audio 189 ms · round-trip WER 0.0% |
| `piper-en-us-amy-medium` | en | realtime | - | 63 MB | See dataset licence: https://github.com/MycroftAI/mimic3-… · needs espeak-ng | untested |
| `piper-en-us-ryan-medium` | en | realtime | - | 63 MB | Voice: CC BY-NC-SA 4.0 ⚠ · needs espeak-ng | **CPU** · RTF 0.118 · first audio 259 ms · round-trip WER 0.0% |
| `piper-en-us-ryan-high` | en | realtime | - | 121 MB | Voice: CC BY-NC-SA 4.0 ⚠ · needs espeak-ng | untested |
| `piper-en-us-hfc-female-medium` | en | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-en-us-hfc-male-medium` | en | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… ⚠ · needs espeak-ng | untested |
| `piper-en-us-libritts-r-medium` | en | realtime | - | 79 MB | Voice: CC BY 4.0 · needs espeak-ng | **CPU** · RTF 0.137 · first audio 309 ms · round-trip WER 0.0% |
| `piper-en-us-ljspeech-medium` | en | realtime | - | 64 MB | Voice: public domain · needs espeak-ng | untested |
| `piper-en-us-joe-medium` | en | realtime | - | 63 MB | Voice: CC0-1.0 · needs espeak-ng | untested |
| `piper-en-us-kristin-medium` | en | realtime | - | 64 MB | Voice: public domain · needs espeak-ng | untested |
| `piper-en-gb-alan-medium` | en | realtime | - | 63 MB | See dataset licence: https://github.com/MycroftAI/mimic3-… · needs espeak-ng | untested |
| `piper-en-gb-alba-medium` | en | realtime | - | 63 MB | See dataset licence: https://creativecommons.org/licenses… · needs espeak-ng | **CPU** · RTF 0.147 · first audio 380 ms · round-trip WER 15.0% |
| `piper-en-gb-cori-medium` | en | realtime | - | 64 MB | Voice: public domain · needs espeak-ng | untested |
| `piper-en-gb-jenny-dioco-medium` | en | realtime | - | 63 MB | See dataset licence: https://github.com/dioco-group/jenny… · needs espeak-ng | untested |
| `piper-en-gb-northern-english-male-medium` | en | realtime | - | 63 MB | Voice: CC-BY-SA 4.0 International · needs espeak-ng | untested |
| `piper-en-gb-southern-english-female-medium` | en | realtime | - | 77 MB | See voice card · needs espeak-ng | untested |
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
| `kitten-nano-0.8-int8` | en | realtime | - | 28 MB | Apache-2.0 · needs espeak-ng | **CPU** · RTF 0.266 · first audio 977 ms · round-trip WER 0.0% |
| `kitten-nano-0.8` | en | realtime | - | 60 MB | Apache-2.0 · needs espeak-ng | **CPU** · RTF 0.107 · first audio 396 ms · round-trip WER 0.0% |
| `kitten-micro-0.8` | en | realtime | - | 45 MB | Apache-2.0 · needs espeak-ng | **CPU** · RTF 0.319 · first audio 1329 ms · round-trip WER 0.0% |
| `kitten-mini-0.8` | en | realtime | - | 82 MB | Apache-2.0 · needs espeak-ng | **CPU** · RTF 1.112 · first audio 3312 ms · round-trip WER 0.0% |

#### Matcha-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `matcha-en-ljspeech` | en | balanced | - | 78 MB | Apache-2.0 · needs espeak-ng | **CPU** · RTF 0.205 · first audio 761 ms · round-trip WER 0.0% |
| `matcha-zh-baker` | zh | balanced | - | 96 MB | Research only (Baker dataset: non-commercial) ⚠ | untested |

#### MeloTTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `melo-tts-en` | en | balanced | - | 175 MB | MIT | **CPU** · RTF 0.278 · first audio 768 ms · round-trip WER 30.0% |

#### MMS-TTS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `mms-tts-eng` | en | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | **CPU** · RTF 0.196 · first audio 590 ms · round-trip WER 5.0% |
| `mms-tts-deu` | de | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-fra` | fr | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-spa` | es | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-rus` | ru | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-ukr` | uk | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |
| `mms-tts-tha` | th | balanced | - | 114 MB | CC-BY-NC-4.0 ⚠ | untested |

#### VITS

| id | languages | tier | streaming | download | licence | validated |
| --- | --- | --- | --- | --- | --- | --- |
| `coqui-bn-female` | bn | balanced | - | 114 MB | Apache-2.0 | **CPU** · RTF 0.574 · first audio 1119 ms |
| `mimic3-gu-cmu-indic` | gu | balanced | - | 76 MB | See voice card (CMU Indic dataset) · needs espeak-ng | **CPU** · RTF 0.046 · first audio 82 ms |
| `mimic3-bn-multi` | bn | balanced | - | 76 MB | See voice card (OpenSLR Bengali, CC-BY-SA-4.0) · needs espeak-ng | **CPU** · RTF 0.139 · first audio 282 ms |
| `mimic3-ne-google` | ne | balanced | - | 76 MB | See voice card (OpenSLR Nepali, CC-BY-SA-4.0) · needs espeak-ng | **CPU** · RTF 0.147 · first audio 248 ms |
| `syspin-mai-female` | mai | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-mai-male` | mai | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-bho-female` | bho | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-bho-male` | bho | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-hi-female` | hi | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-hi-male` | hi | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-bn-female` | bn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-bn-male` | bn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-mr-female` | mr | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-mr-male` | mr | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-te-female` | te | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-te-male` | te | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-kn-female` | kn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-kn-male` | kn | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-gu-female` | gu | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-gu-male` | gu | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-mag-female` | mag | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-mag-male` | mag | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-hne-female` | hne | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-hne-male` | hne | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-en-female` | en | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |
| `syspin-en-male` | en | balanced | - | 114 MB | MIT (weights, IISc SYSPIN); training data SYSPIN corpus, … | not yet published |

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

Some good, permissively licensed voices exist only as PyTorch checkpoints.
bolo has no PyTorch, so the project converts them once, in the open
(`tools/voices/` and the "Voices" workflow), and publishes the results as
assets of a GitHub release of this repository
(`https://github.com/cb-Sigma/bolo/releases/tag/voices-v1`). Each archive
holds `model.onnx`, `tokens.txt`, `voice.json` (source repository and pinned
revision, licence, language, sample rate, character set), `LICENSE` and
`ATTRIBUTION`.

**Status: not yet published.** The 22 catalog entries below carry
`status: pending`: they are listed here, hidden from `bolo list` and the
console, and answer `model_not_published` when pulled. Once the workflow has
run, `tools/voices/update_catalog.py` writes the checksums into the catalog.
What was run so far: packing, the sherpa-onnx smoke synthesis, the index and
the catalog update on three voices converted earlier with the same recipe
(Maithili female and male, Bhojpuri female), then the complete pull path
against a local web server and a transcription of the result. The export
step of the script itself (PyTorch) and the workflow were **not** run.

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
Bhojpuri voices. Phoneme-based voices (Piper format, Kokoro, KittenTTS) are
not affected: espeak-ng reads digits and foreign words for them.

## Word timestamps

| Models | Word times |
| --- | --- |
| faster-whisper (CTranslate2 Whisper) | yes (cross-attention alignment) |
| transducer and CTC models through sherpa-onnx (Zipformer, Parakeet, NeMo, IndicConformer, SenseVoice, …) | yes, from the token times the decoder reports |
| Whisper through sherpa-onnx (`whisper-*`) | **no** with the exports in the catalog: token times need a model exported with cross-attention outputs. sherpa-onnx 2.0.1 supports such exports, and a catalog entry for one can switch them on with the option `token_timestamps: true`; we checked that the stock `whisper-tiny` export reports none |
| Moonshine | **no**: the runtime exposes no token times, and we did not add a forced aligner |

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
  Urdu, Nepali and Gujarati. Maithili, Bhojpuri, Kannada and more are
  prepared but not yet published (previous section). **No voice** for
  Punjabi, Odia, Assamese, Sanskrit, Sindhi, Kashmiri, Konkani, Dogri,
  Manipuri, Bodo and Santali.
* **Bhojpuri** has no speech-to-text model of its own here; the Hindi model
  is the fallback, at about 40 % word error rate in our run.

**AI4Bharat's own repositories** (`ai4bharat/indicconformer_stt_*`,
`ai4bharat/indic-conformer-600m-multilingual`) are **gated** on Hugging Face
(MIT licence, but you must accept terms and use a token). bolo never
downloads a gated model without a token you supply, so the catalog uses
ungated community exports of the same models, pinned to a commit and named
`-community`. They are third-party conversions: we verified the licence of
the originals and ran the files, not how they were made.

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
* So bolo has a small phonemiser front end (`bolo/text/phonemize.py`) that
  hands text to espeak-ng and gets phonemes back.

**bolo does not ship espeak-ng.** It is not in the wheel, not in
`bolo[all]`, and not in the Docker images unless you build them with
`--build-arg WITH_ESPEAK=1`. It is an *optional component*, installed only
when you ask for it. Models that need it have `requires: ["espeak-ng"]` in the
catalog (marked "needs espeak-ng" in the tables above, 93 entries) and show as
unavailable until it is there; using one answers `424 component_missing` with
the install instructions instead of producing silence.

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

Voices that never need espeak-ng: **Kokoro's English and Mandarin voices**
(the default text-to-speech model; they use the bundled lexicons, and words
missing from the lexicon are skipped, so quality is better with espeak-ng
installed), **MeloTTS**, **MMS-TTS** (non-commercial licence), the
**Coqui VITS** Bengali voices and the SYSPIN voices.

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
* **MMS-TTS** is CC-BY-NC-4.0.
* **SenseVoice / Paraformer** use Alibaba's FunASR model licence (custom).
* **Kroko** community models are CC-BY-SA.
* **Parakeet, Canary, NeMo FastConformer** are CC-BY-4.0.
* **TEN VAD** is not in the catalog: its licence adds field-of-use conditions
  to Apache-2.0, and the ONNX file in its repository does not load in
  sherpa-onnx without added metadata. The loader kind (`ten_vad`) exists for
  people who have a compatible file.
* Nothing in the default catalog is gated.
