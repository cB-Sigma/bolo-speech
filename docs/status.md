# What has been tested, and what has not

bolo is alpha software, developed and measured on one kind of machine, and checked on one Windows laptop. This page states plainly what was run and
what was only written against documentation, so that you do not find out the hard way. Each feature's own page
has the details; the links say where.

## The reference machine

4 vCPUs of an Intel Xeon at 2.1 GHz (AVX-512, AMX), 15 GB RAM, Linux x86-64, **no GPU**; a shared cloud VM that
was running other workloads at the same time. Every CPU real-time factor, latency and memory figure in the
documentation comes from it ([performance.md](performance.md)), from small samples, with the conditions next to
the number, unless it names the laptop below.

## A Windows 11 laptop

Windows, the GPU runtimes and the live demo in a desktop browser were also measured on one laptop: an AMD Ryzen 7
5800H (8 cores, AVX2, no VNNI), an NVIDIA RTX 3060 Laptop GPU with 6 GB next to AMD Radeon integrated graphics,
Windows 11, bolo 0.6.1 and 0.6.2. Every GPU figure in the documentation comes from this one machine; it says what
ran there, not what other GPUs or drivers do.

- **Native Windows, CPU.** The server's test suite passes. `server/tools/validate_models.py` ran 19 of the 23 core
  models and 56 of the 92 further models; their median real-time factor was 0.31 to 0.60 times the reference
  machine's (faster), the slowest being int8 faster-whisper at 1.02 to 1.04 times. 33 models cannot be run by the
  tool on any machine: they have no test audio or validation sentence (mostly Indian languages), or they are gated.
- **CUDA on Windows** (`onnxruntime-gpu` 1.32, CTranslate2 4.9; the recipe is in
  [configuration.md](configuration.md#cuda-on-windows)). faster-whisper ran 6 to 19 times faster on the GPU than on
  the CPU. Nemotron diarization: real-time factor 0.0019 on CUDA, 0.0226 on the CPU. Pocket TTS on llama.cpp's
  prebuilt Windows CUDA build: real-time factor 0.032, first audio after 47 ms (CPU build 0.168, Vulkan build
  0.049). `bolo ps` showed the models at `100% GPU`. sherpa-onnx's CUDA wheel cannot load inside bolo on Windows
  (Error 1114), so sherpa-onnx models run on the CPU there.
- **DirectML** (`onnxruntime-directml` 1.24.4). Nemotron diarization: real-time factor 0.0055 on the RTX 3060 with
  plain `dml` (the high-performance adapter), 0.0131 on the Radeon (`dml:0`), 0.0226 on the CPU. `smart-turn-v3`
  fails on DirectML, so bolo runs it on the CPU.
- **WSL2 on the same laptop.** CPU speed within about 10 % of native Windows. CUDA works there with ONNX Runtime and
  faster-whisper. llama-server built from source with CUDA
  ([configuration.md](configuration.md#a-gpu-build-of-llama-server-on-linux)) ran Pocket TTS at real-time factor
  0.034, first audio after 41 ms, the same as native Windows.
- **Vulkan llama-server on Windows** works once AMD's switchable-graphics layer is disabled, which bolo does itself
  since 0.6.2 ([configuration.md](configuration.md#windows)).
- **A lost GPU.** In long runs at about 92 °C the RTX 3060 dropped out twice. bolo now notices a lost GPU and moves
  its models to the CPU; that detection has not yet been triggered on real hardware.
- **The live demo** works in Chrome and Edge 157 there. For the small recognisers WebAssembly was 2 to 10 times
  faster than WebGPU (Moonshine Tiny: real-time factor 0.097 on WebAssembly, 0.94 on WebGPU). Chrome on Windows
  ignores the request for the high-performance GPU (crbug 369219127). MOSS-TTS-Nano cloning in the browser:
  real-time factor 0.69, speaker similarity 0.70. Memory stayed flat over 20 reloads.
- **Voice cloning.** Saved voices work over the command line and HTTP on Windows; Pocket TTS cloned on the CUDA
  build of llama-server.

- **Two GPUs, AMD and NVIDIA.** The integrated Radeon (GCN 5) ran compute through DirectML (Nemotron diarization
  above, `dml:0`) and through WebGPU in the browser. As a hybrid laptop: bolo picks the high-performance adapter for
  plain `dml`, lists the NVIDIA card once, never counts the integrated GPU as the best GPU, and the Vulkan build
  works with the AMD layer disabled.

Not covered by the laptop: Apple Silicon (Metal, CoreML); ROCm (which does not support that integrated Radeon) and
discrete Radeon cards; a discrete GPU under native Linux (only WSL2); the features for several NVIDIA GPUs (models
or layers spread over `cuda:0` and `cuda:1`, the `gpus` setting), which need two NVIDIA cards; arm64 Linux; phones;
GPU runs of the catalog models not named above; and cloud providers with real accounts.

## Not run at all

- **GPUs other than one laptop's.** One GPU machine was used: the laptop [above](#a-windows-11-laptop), through
  CUDA, DirectML and Vulkan on its NVIDIA GPU and DirectML and WebGPU on its integrated Radeon, natively and under
  WSL2. The torch plugin on a GPU, `Dockerfile.gpu`, a discrete GPU under native Linux, ROCm, discrete Radeon cards,
  Metal and several NVIDIA GPUs have not been run. The hardware recommendations for GPU machines are
  estimates from formulas, not measurements ([recommendations.md](recommendations.md#how-uncertain-this-is)).
  Apart from whole models placed on the laptop's GPU, GPU/CPU placement, keep-alive eviction of GPU-resident
  models, several NVIDIA GPUs, llama.cpp `-ngl` and `--tensor-split`, the ONNX graph cut and bolo-torch's `device_map` splits are tested with a fake NVML of made-up
  GPUs and llama-server command lines that are built but not run; the cut graph was run with both halves on the CPU
  ([configuration.md](configuration.md#gpu-and-cpu-placement)).
- **Apple Silicon and macOS, Linux arm64.** Not run. The dependencies publish wheels for macOS and arm64,
  so it should work there.
- **Windows: one laptop, plus CI.** Windows 11 was run natively on the laptop [above](#a-windows-11-laptop), on
  its CPU and with CUDA, DirectML and Vulkan, and WSL2 on the same laptop. The server's test suite, the bolo-torch plugin's fast tests, `bolo doctor` and `bolo serve` (answering `/healthz`)
  run on every push on `windows-latest` (CPU, Python 3.12, the win_amd64 wheels of sherpa-onnx,
  CTranslate2/faster-whisper, PyAV, soundfile and uccharan). The pinned Windows audio.cpp builds, the
  restart and service recipes (PowerShell loop, NSSM, WinSW) and the junction fallback for local model folders on a
  machine without Developer Mode have not been run (docs/configuration.md, "Windows").
- **Python 3.10 and 3.11.** The package targets 3.10 and newer; the test-suite was run on 3.12, 3.13 and 3.14.
- **A phone network.** No real call was made and no telephony account was used. Twilio, Plivo, Telnyx and Exotel
  are implemented from the providers' documentation as read on 2026-11-14 and tested against fake providers that
  speak those protocols; the SIP endpoint is tested against the test-suite's own SIP user agent, not against a
  trunk, a PBX or a softphone. [telephony.md](telephony.md#what-was-verified-and-what-was-not) lists, provider by
  provider, what was read and what was assumed. The G.711 paths are tested end to end with synthesised speech.
- **Cloud accounts.** Nothing was run against OpenAI, Sarvam, ElevenLabs, Deepgram or Soniox: there were no keys
  where bolo is developed. The adapters follow each provider's documentation as read on 2026-11-14 (Soniox on
  2026-11-17) and are tested against stand-ins written from the same documentation
  ([providers.md](providers.md#what-was-verified-and-what-was-not)). Cost estimates use prices you enter; none
  are shipped.
- **The live demo on real devices.** The demo's end-to-end tests run in headless Chromium on a desktop CPU (the
  storage crash was reproduced and fixed in WebKit). It was run in Chrome and Edge on the laptop
  [above](#a-windows-11-laptop), WebGPU included. The GPU/CPU split of a model and the limits for phones and small
  devices are tested with a faked device; nothing was run on an iPhone or an Android phone.
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
  `server/tools/validate_models.py` on the reference machine.
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
- **Voice cloning: measured on CPU for 14 models.** Pocket TTS (six languages), Chatterbox Turbo and Qwen3-TTS 1.7B
  Base on llama.cpp, and Pocket TTS, MOSS-TTS Nano, XTTS v2, Plapre 3, OmniVoice (ran, did not clone) and Qwen3-TTS
  0.6B Base on audio.cpp (built from source) cloned two LibriSpeech speakers through bolo's own code path, scored
  with the speaker-similarity model ([voices.md](voices.md#measured-cloning-on-cpu)). The other 25 audio.cpp
  cloning entries, the torch plugin's cloning models, Chatterbox Multilingual (too slow here) and Tontaube were not
  run. The similarity thresholds are calibrated on LibriSpeech read speech only, the voice checks (noise, clipping,
  more than one person) were tuned on the same data, and no bolo-side watermark is applied.
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
