# AMD GPUs (ROCm)

bolo runs models on AMD GPUs through ROCm, AMD's GPU compute stack, on Linux (and llama-server's HIP build on
Windows). `device = "rocm"` (also `hip`, or `rocm:N` for GPU N) asks for it; `device = "auto"` picks it on a machine
with an AMD GPU and no NVIDIA GPU.

**Untested on real hardware.** bolo's CI has no AMD GPU. What is tested is the selection logic: which device,
provider and build each engine gets, against fake `/sys/class/drm` trees, provider lists and hardware descriptions
(`server/tests/test_amd_apple_devices.py`, `plugins/bolo-torch/tests/test_runtime.py`). Nobody has run a model on an
AMD GPU through bolo yet. Reports are welcome.

## What runs where

| engine | on an AMD GPU | what to install |
|---|---|---|
| ONNX Runtime engines (Nemotron diarization, DiCoW ONNX, the Orpheus SNAC decoder, end-of-turn models, in-process speech-to-speech) | MIGraphX execution provider, else the older ROCm provider | `onnxruntime-migraphx` (MIT; Linux x86-64) in place of `onnxruntime`, with ROCm and MIGraphX installed |
| llama-server voices, audio LLMs, speech-to-speech | llama.cpp's HIP build | `bolo components install llama-server --flavour rocm` (or with `device = "rocm"`) |
| audio.cpp | its `hip` backend | a HIP build of your own (`audiocpp_server_bin`) |
| bolo-torch plugin (PyTorch models) | a ROCm build of PyTorch (HIP serves `torch.cuda`) | torch from `https://download.pytorch.org/whl/rocm6.4` (or a newer ROCm index) |
| whisper.cpp | a HIP build of pywhispercpp | `GGML_HIP=1 pip install pywhispercpp --no-binary pywhispercpp` |
| sherpa-onnx | the CPU: there is no ROCm build of sherpa-onnx | nothing; placed on the CPU with a notice |
| faster-whisper (CTranslate2) | the CPU: CTranslate2 has no ROCm build on PyPI | nothing |

Each of these is decided per model, as for CUDA ([configuration.md](configuration.md#gpu-and-cpu-placement)): a model
whose runtime cannot reach the GPU goes on the CPU with a notice in `bolo ps` and `GET /api/v1/ps`.

## Install

1. ROCm itself, from AMD's instructions for your distribution (`amdgpu-install --usecase=rocm`, or the TheRock
   packages). Check with `rocminfo` (it lists your GPU as an `Agent`) and `amd-smi` (or `rocm-smi`). Your user needs
   to be in the `render` and `video` groups.
2. bolo: `pip install "bolo-speech[all]"`.
3. ONNX Runtime with MIGraphX. `onnxruntime-migraphx` installs the same `onnxruntime` module as `onnxruntime`, so
   replace it rather than adding it:

   ```
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install onnxruntime-migraphx
   ```

   It needs MIGraphX from ROCm (`apt install migraphx` on Ubuntu with AMD's repository). `onnxruntime-rocm` (the ROCm
   provider, last released for ONNX Runtime 1.22) works too; bolo prefers MIGraphX when a build has both.
4. llama-server's HIP build: `bolo components install llama-server --flavour rocm`. The pinned release (llama.cpp
   b12375, built against ROCm 10.0) covers gfx908 to gfx1201: Instinct MI100 to MI350, Radeon RX 5000 to 9000 and the
   Radeon 780M / 890M / 8060S APUs. On Linux it needs the ROCm runtime libraries (`libamdhip64`, hipBLAS, rocBLAS) on
   the machine; on Windows it brings its own HIP DLLs and needs a recent Adrenalin driver. It goes in its own folder
   (`components/llama-server/<version>/rocm`), next to the CPU and Vulkan builds.
5. The torch plugin, if you use PyTorch models: install a ROCm build of torch first, then the plugin:

   ```
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
   pip install "bolo-torch[parler]"
   ```

6. `device = "rocm"` in `bolo.toml` (or `BOLO_DEVICE=rocm`), or leave `device = "auto"`.

Without ROCm an AMD GPU can still run llama-server's Vulkan build (`device = "vulkan"`, Mesa's RADV driver); nothing
else in bolo uses it.

## What bolo doctor says

`bolo doctor` (and `GET /api/v1/hardware`) lists AMD GPUs from the amdgpu driver's counters in `/sys/class/drm`: name
(`product_name`), VRAM total and used, and busy percent. It does not need ROCm for that. It then looks for ROCm
(`/opt/rocm/.info/version`, `ROCM_PATH`, or `rocminfo` / `amd-smi` / `rocm-smi` / `hipconfig` on `PATH`):

- with ROCm, the GPU's backend is `rocm` and doctor suggests `device = "rocm"`;
- without it, a note says that only a Vulkan build of llama-server can use the GPU, and how to get ROCm.

It also reports the ONNX Runtime build (`the AMD MIGraphX build`), its providers, the llama-server build
(`rocm build`), the torch plugin's `ROCm build: yes/no`, and pywhispercpp's GPU back ends. `bolo recommend` counts the
GPU for the engines that can use it and says why the others cannot ("sherpa-onnx has no ROCm build").

An APU (Radeon 780M and the like) has a small VRAM carve-out (512 MB to 1 GB) and works from the machine's memory:
bolo treats a GPU with less than 2 GB of VRAM as unified memory and plans with the machine's available memory, as on
Apple Silicon.

### On Windows

There is no sysfs on Windows: `bolo doctor` lists a Radeon through DXGI, as a DirectML adapter (`dml:N`), and adds a
line saying what ROCm means for it. AMD's HIP runtime for Windows covers the Radeon RX 6000, 7000 and 9000 series and
Radeon PRO W6000/W7000 (RDNA 2 to 4); for those, doctor says that llama-server's ROCm (HIP) build can use the card
(`bolo components install llama-server --flavour rocm`; the archive brings its own HIP runtime DLLs and needs a recent
Adrenalin driver). For other Radeons (integrated graphics such as the Radeon 780M, RX 5000 and older) it says
"ROCm: not available on Windows for this GPU". Either way the ONNX Runtime engines use the GPU through DirectML
(`device = "dml"`, with `onnxruntime-directml`) and llama-server through its Vulkan build (`device = "vulkan"`).
MIGraphX, ROCm builds of PyTorch and the rest of ROCm are Linux only for bolo. Doctor matches the card's name; it
does not ask the driver, and none of this has been run on a Radeon by bolo's maintainers.

## Several AMD GPUs

`rocm:N` numbers GPUs as HIP does: PCI order, after `HIP_VISIBLE_DEVICES` (else `ROCR_VISIBLE_DEVICES`). With
`HIP_VISIBLE_DEVICES=1`, the second card is `rocm:0`. The memory figures are the whole device's: the amdgpu driver
keeps no per-process figure in sysfs, so `bolo ps` measures a load by the device's used memory.

## Known gaps

- Nothing here has been run on an AMD GPU by the maintainers. The llama-server HIP archives are pinned by SHA-256 and
  size, but bolo has not started them.
- sherpa-onnx and faster-whisper have no ROCm builds; their models run on the CPU.
- The `gpus` setting and NVML-based per-process memory are NVIDIA only.
- Windows: only llama-server's HIP build, on RDNA 2 and newer Radeon cards (see "On Windows"). ONNX Runtime engines
  use an AMD GPU there through DirectML (`device = "dml"`).
