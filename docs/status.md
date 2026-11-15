# What has been tested, and what has not

bolo is alpha software, developed and measured on one kind of machine. This page states plainly what was run and
what was only written against documentation, so that you do not find out the hard way. Each feature's own page
has the details; the links say where.

## The machine everything was measured on

4 vCPUs of an Intel Xeon at 2.1 GHz (AVX-512, AMX), 15 GB RAM, Linux x86-64, **no GPU**; a shared cloud VM that
was running other workloads at the same time. Every real-time factor, latency and memory figure in the
documentation comes from it ([performance.md](performance.md)), from small samples, with the conditions next to
the number.

## Not run at all

- **GPUs.** No GPU was available. `device = "cuda"` is passed to faster-whisper and sherpa-onnx, the torch plugin
  takes `BOLO_TORCH_DEVICE`, and there is a `Dockerfile.gpu`, but none of it has been run and **no number in the
  documentation was measured on a GPU**. The hardware recommendations for GPU machines are estimates from formulas, not
  measurements ([recommendations.md](recommendations.md#how-uncertain-this-is)). GPU/CPU placement, keep-alive
  eviction of GPU-resident models, several GPUs, llama.cpp `-ngl` and `--tensor-split`, the ONNX graph cut and
  bolo-torch's `device_map` splits are tested with a fake NVML of made-up GPUs and llama-server command lines that
  are built but not run; the cut graph was run with both halves on the CPU
  ([configuration.md](configuration.md#gpu-and-cpu-placement)).
- **Apple Silicon and macOS, Linux arm64, Windows.** Not run. The dependencies publish wheels for macOS and arm64,
  so it should work there; Windows is unknown.
- **Python 3.10 and 3.11.** The package targets 3.10 and newer; the test-suite was run on 3.12, 3.13 and 3.14.
- **A phone network.** No real call was made and no telephony account was used. Twilio, Plivo, Telnyx and Exotel
  are implemented from the providers' documentation as read on 2026-11-14 and tested against fake providers that
  speak those protocols; the SIP endpoint is tested against the test-suite's own SIP user agent, not against a
  trunk, a PBX or a softphone. [telephony.md](telephony.md#what-was-verified-and-what-was-not) lists, provider by
  provider, what was read and what was assumed. The G.711 paths are tested end to end with synthesised speech.
- **Cloud accounts.** Nothing was run against OpenAI, Sarvam, ElevenLabs or Deepgram: there were no keys where
  bolo is developed. The adapters follow each provider's documentation as read on 2026-11-14 and are tested
  against stand-ins written from the same documentation
  ([providers.md](providers.md#what-was-verified-and-what-was-not)). Cost estimates use prices you enter; none
  are shipped.
- **The live demo on real devices.** The demo's end-to-end tests run in headless Chromium on a desktop CPU (the
  storage crash was reproduced and fixed in WebKit). WebGPU, the GPU/CPU split of a model, and the limits for phones
  and small devices are tested with a faked device; nothing was run on a GPU, an iPhone or an Android phone.
- **Home Assistant and browsers.** The Wyoming flows Home Assistant uses were run with the official `wyoming`
  client against a real server with real models, not with Home Assistant itself. The WebRTC transport of the
  Realtime API is tested with the aiortc library (also with real models), not with Chrome, Firefox or Safari, and
  not across NAT or a TURN relay. Pipecat and LiveKit Agents were run against bolo, but without a transport in
  front (a wav file played the caller).
- **The GPU image, the systemd unit and the Kubernetes manifest.** The CPU image was built and run (health check,
  synthesis, transcription) and CI builds and smoke-tests it on every push; the compose file was run. The systemd
  unit passes `systemd-analyze verify` but was not run under systemd, the Kubernetes manifest was not applied to
  a cluster, and the GPU image was not built ([deploy/README.md](../deploy/README.md)).
- Engines `moonshine` (useful-moonshine-onnx), `whisper.cpp` and `kokoro-onnx` have plugins that were written
  against their documentation and not run.

## Run, but only a little

- **Most of the catalog.** About two thirds of the local catalog entries are marked `untested` (the exact counts
  head the table in [models.md](models.md#the-catalog)): their files were checked on the Hub and a sibling of the
  same architecture was run, but that model was not. Entries marked `validated: cpu` were downloaded and run by
  `server/tools/validate_models.py` on the machine above.
- **Accuracy.** Word and character error rates come from a handful of clips or, for the phonemiser comparison,
  500 sentences per voice. They show that the pipelines work, not how good the models are; for that, see the
  model publishers' evaluations.
- **Voice agents with a capable language model.** The conversation path was run end to end with models of 360M
  and 500M parameters on a shared CPU, through llama-cpp-python's server, Ollama and llama.cpp's `llama-server`;
  the answers of such models are poor and the turn latency is not representative of a real deployment. Function
  calling was exercised with a scripted fake model, not a real one. vLLM and hosted APIs were not tried
  ([voice-agents.md](voice-agents.md)).
- **The audio-in and speech-to-speech pipelines.** The `audio-llm` pipeline and the llama-server route of
  `speech-to-speech` are tested against stand-in servers. The in-process speech-to-speech model
  (`lfm3-audio-2b-onnx`) was run here: on 4 shared CPU cores it generates speech at 0.4 to 0.5 times real time,
  too slow to hold a conversation ([voice-agents.md](voice-agents.md#pipelines-cascade-audio-in-language-model-speech-to-speech)).
- **Diarization accuracy.** Clustering diarization: one clip of two synthetic voices, speaker counts on 18 public
  recordings, overlap handling on three synthetic mixes. Nemotron diarization and diarization-conditioned
  recognition (DiCoW): 24 AMI far-field meeting excerpts and 44 synthetic two-voice mixes
  ([models.md](models.md#speaker-diarization-1)). No telephone recordings, no noisy rooms; known-speaker naming was
  tested with synthetic voices only.
- **The torch plugin.** Indic Parler-TTS, IndicF5, the SepFormer and PixIT separators and DiCoW were run on CPU
  (all slower than real time except DiCoW with bfloat16); Veena and Svara were not run beyond their SNAC decoder;
  Indic-Speak, the original F5-TTS checkpoints and the other newly catalogued kinds are unit-tested with fakes
  only ([plugins.md](plugins.md#the-pytorch-plugin-bolo-torch)).
- **Published conversions.** The 22 SYSPIN and 18 MMS-TTS voices were published by the "Voices" workflow; eight
  MMS voices were run here from local conversions, the published MMS files were not re-run. `dicow-v3.3-onnx` is
  validated on a local export whose files have the same SHA-256 sums as the published ones; the Orpheus-family
  llama.cpp voices are untested from their published files.
- **Romanised text.** bolo's side of the `romanised` option is tested against a stand-in transliterator and
  with uccharan 0.4.0 and its Hindi pack on one Piper Hindi voice (one request, converted and spoken). How the converted text sounds on bolo's voices was not evaluated
  ([api.md](api.md#text-preparation-ssml-and-pronunciation-word-lists)).
- **Indian languages.** Sixteen of the 22 recognisers were run on 10 to 60 real utterances each; six have no
  public test recordings and are untested. Nobody who speaks the languages reviewed the synthesised audio
  ([indic-telephony.md](indic-telephony.md)).
- **Long runs and hostile input.** A 30-minute mixed-load soak was run ([performance.md](performance.md));
  memory growth over days and hostile input were not examined.
