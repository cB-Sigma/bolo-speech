/**
 * The small audio models the page runs next to the recogniser: denoisers, spoken-language identifiers,
 * voice-activity and end-of-turn detectors, and speaker embedders. Each keeps the id and the licence of the
 * server's catalog entry it mirrors (server/src/bolo/catalog/*.yaml), and its files are fetched at the revision
 * the server pins, from a host that allows a page to read them (Hugging Face, raw GitHub, jsDelivr), or from the
 * demo's own site when the only copy is a GitHub release (which a page cannot read).
 *
 * `device` is where Auto runs it. Measured in headless Chromium (docs/measurements/browser-audio-models.json):
 * every one of these models is too small to keep a GPU busy, so WebAssembly is the default; WebGPU is one choice
 * away for the ones ONNX Runtime can place there.
 */

export type AudioTask = 'denoise' | 'lid' | 'vad' | 'turn' | 'embed'

/** Where a model's bytes come from. */
export type AudioFiles =
  | { from: 'hub'; repo: string; revision: string; files: Record<string, string> }
  | { from: 'url'; files: Record<string, string> }
  /** Copied into the demo's own site by the Pages workflow (`models/` next to the page), pinned by SHA-256. */
  | { from: 'site'; files: Record<string, string>; origin: string; sha256: Record<string, string> }
  /** An npm package from jsDelivr, at an exact version. */
  | { from: 'npm'; pkg: string; version: string; files: Record<string, string> }

export type AudioRunner =
  | 'gtcrn'
  | 'dpdfnet'
  | 'deepfilternet3'
  | 'rnnoise'
  | 'voxlingua'
  | 'whisper-lid'
  | 'silero-v4'
  | 'silero-v5'
  | 'ten-vad'
  | 'energy'
  | 'smart-turn'
  | 'speaker-embedding'

export interface AudioModel {
  /** The server's catalog id. */
  id: string
  task: AudioTask
  label: string
  runner: AudioRunner
  files: AudioFiles | null
  /** Bytes fetched on first use. */
  bytes: number
  /** Estimated memory while loaded. */
  peakBytes: number
  /** The model's own sample rate (audio is resampled to it and back). */
  rate: number
  license: string
  /** Not for commercial use, or under conditions worth reading first. */
  licenseNote?: string
  commercial?: boolean
  source: string
  blurb: string
  /** Runs on ONNX Runtime and may be placed on WebGPU. */
  webgpu: boolean
  /** Where Auto runs it (measured; see the module comment). */
  device: 'wasm' | 'webgpu'
  /** What the measurement said, for the picker. */
  measured?: string
}

const SHERPA_RELEASE = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/'

export const AUDIO_MODELS: AudioModel[] = [
  // -- denoisers (server/src/bolo/catalog/denoise.yaml) -----------------------------------------------------------
  {
    id: 'gtcrn-simple',
    task: 'denoise',
    label: 'GTCRN',
    runner: 'gtcrn',
    files: {
      from: 'site',
      files: { model: 'models/gtcrn_simple.onnx' },
      origin: `${SHERPA_RELEASE}speech-enhancement-models/gtcrn_simple.onnx`,
      sha256: { model: 'e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534' },
    },
    bytes: 535_638,
    peakBytes: 40e6,
    rate: 16000,
    license: 'MIT',
    source: 'https://github.com/Xiaobin-Rong/gtcrn',
    blurb: 'The server’s default denoiser: 0.5 MB, 16 kHz, frame by frame (sherpa-onnx’s export).',
    webgpu: true,
    device: 'wasm',
  },
  ...(['baseline', 'dpdfnet2', 'dpdfnet4', 'dpdfnet8'] as const).map((name, i): AudioModel => ({
    id: name === 'baseline' ? 'dpdfnet-baseline' : name,
    task: 'denoise',
    label: name === 'baseline' ? 'DPDFNet baseline' : `DPDFNet-${name.slice(-1)}`,
    runner: 'dpdfnet',
    files: {
      from: 'hub',
      repo: 'Ceva-IP/DPDFNet',
      revision: 'dd6818d00f50c836fed43a6243ebe49116de5964',
      files: { model: `onnx/${name}.onnx` },
    },
    bytes: [8_722_707, 10_178_747, 11_633_337, 14_542_517][i] as number,
    peakBytes: 90e6,
    rate: 16000,
    license: 'Apache-2.0',
    source: 'https://github.com/ceva-ip/DPDFNet',
    blurb:
      name === 'baseline'
        ? 'Ceva’s DPDFNet without its dual-path blocks: the lightest of the family, 16 kHz.'
        : `Ceva’s DPDFNet with ${name.slice(-1)} dual-path blocks: stronger and slower, 16 kHz.`,
    webgpu: true,
    device: 'wasm',
  })),
  {
    id: 'deepfilternet3',
    task: 'denoise',
    label: 'DeepFilterNet 3',
    runner: 'deepfilternet3',
    files: {
      from: 'hub',
      repo: 'kimtos-labs/denoiser-dfn3',
      revision: '888f33c41851d7168ae31c3637c323b9e2f3c2a4',
      files: { model: 'denoiser_model.onnx' },
    },
    bytes: 16_104_687,
    peakBytes: 120e6,
    rate: 48000,
    license: 'MIT OR Apache-2.0',
    source: 'https://github.com/Rikorose/DeepFilterNet',
    blurb: 'Full band (48 kHz), 10 ms frames, torchDF’s single streaming graph. Heavier: 100 model runs a second.',
    webgpu: true,
    device: 'wasm',
  },
  {
    id: 'rnnoise',
    task: 'denoise',
    label: 'RNNoise',
    runner: 'rnnoise',
    files: { from: 'npm', pkg: '@shiguredo/rnnoise-wasm', version: '2025.1.5', files: { module: 'dist/rnnoise.js' } },
    bytes: 4_827_304,
    peakBytes: 30e6,
    rate: 48000,
    license: 'BSD-3-Clause',
    licenseNote: 'Xiph’s library (BSD-3-Clause), built for WebAssembly by Shiguredo (Apache-2.0).',
    source: 'https://gitlab.xiph.org/xiph/rnnoise',
    blurb: 'The classic: a tiny recurrent network in C, 48 kHz. Its own WebAssembly build, not ONNX Runtime.',
    webgpu: false,
    device: 'wasm',
  },
  // -- spoken-language identification (lid.yaml) ------------------------------------------------------------------
  {
    id: 'voxlingua107-ecapa-int8',
    task: 'lid',
    label: 'VoxLingua107 ECAPA (8-bit)',
    runner: 'voxlingua',
    files: {
      from: 'hub',
      repo: 'doanhlt/cabin-lid-ecapa-voxlingua107-onnx',
      revision: '44f3fb4facf01d97d1c3342807eaf15150b271d2',
      files: { model: 'ecapa.int8pc.onnx', labels: 'labels.txt' },
    },
    bytes: 21_937_026,
    peakBytes: 120e6,
    rate: 16000,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa',
    blurb: '107 languages, among them 15 Indian ones (not Odia). The server’s best measured identifier, 22 MB.',
    webgpu: true,
    device: 'wasm',
  },
  // -- voice-activity detection (vad.yaml) ------------------------------------------------------------------------
  {
    id: 'energy-vad',
    task: 'vad',
    label: 'Energy detector',
    runner: 'energy',
    files: null,
    bytes: 0,
    peakBytes: 0,
    rate: 16000,
    license: 'Apache-2.0',
    source: 'https://github.com/cB-Sigma/bolo',
    blurb: 'No model: loudness above the room’s noise floor. Nothing to download; fooled by noise.',
    webgpu: false,
    device: 'wasm',
  },
  {
    id: 'silero-vad',
    task: 'vad',
    label: 'Silero VAD v4',
    runner: 'silero-v4',
    files: {
      from: 'hub',
      repo: 'csukuangfj/vad',
      revision: 'fba88cd2e921609e7675c3aaf51e0b9b295da4bc',
      files: { model: 'silero_vad.onnx' },
    },
    bytes: 1_807_522,
    peakBytes: 30e6,
    rate: 16000,
    license: 'MIT',
    source: 'https://github.com/snakers4/silero-vad',
    blurb: 'The server’s default detector (v4), 1.8 MB.',
    webgpu: true,
    device: 'wasm',
  },
  {
    id: 'silero-vad-v5',
    task: 'vad',
    label: 'Silero VAD v5',
    runner: 'silero-v5',
    files: {
      from: 'hub',
      repo: 'csukuangfj/vad',
      revision: 'fba88cd2e921609e7675c3aaf51e0b9b295da4bc',
      files: { model: 'silero_vad_v5.onnx' },
    },
    bytes: 2_313_101,
    peakBytes: 30e6,
    rate: 16000,
    license: 'MIT',
    source: 'https://github.com/snakers4/silero-vad',
    blurb: 'Silero v5: the best of the server’s detectors on noisy audio (with v6).',
    webgpu: true,
    device: 'wasm',
  },
  {
    id: 'silero-vad-v6',
    task: 'vad',
    label: 'Silero VAD v6',
    runner: 'silero-v5',
    files: {
      from: 'url',
      files: {
        model:
          'https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx',
      },
    },
    bytes: 2_327_524,
    peakBytes: 30e6,
    rate: 16000,
    license: 'MIT',
    source: 'https://github.com/snakers4/silero-vad',
    blurb: 'Silero v6 (release 6.2.3), the file of the Silero repository. Same interface as v5.',
    webgpu: true,
    device: 'wasm',
  },
  {
    id: 'ten-vad',
    task: 'vad',
    label: 'TEN VAD',
    runner: 'ten-vad',
    files: {
      from: 'hub',
      repo: 'TEN-framework/ten-vad',
      revision: 'bda8ffc78b1846c5c7cbd38f04e52deff49de707',
      files: { module: 'lib/Web/ten_vad.js', wasm: 'lib/Web/ten_vad.wasm' },
    },
    bytes: 288_958,
    peakBytes: 20e6,
    rate: 16000,
    license: 'Apache-2.0 with Agora’s additional conditions (own applications only; no use competing with Agora)',
    licenseNote: 'Not for commercial use as the server catalog reads it: only in your own applications.',
    commercial: false,
    source: 'https://github.com/TEN-framework/ten-vad',
    blurb: 'Agora’s TEN VAD in its own WebAssembly build (the server runs the same model through sherpa-onnx).',
    webgpu: false,
    device: 'wasm',
  },
  // -- end of turn (turn.yaml) --------------------------------------------------------------------------------------
  ...(
    [
      [
        'smart-turn-v3',
        'Smart Turn v3.2',
        'smart-turn-v3.2-cpu.onnx',
        8_679_182,
        'The server’s default (v3.2, 8-bit).',
      ],
      ['smart-turn-v3.1', 'Smart Turn v3.1', 'smart-turn-v3.1-cpu.onnx', 8_679_180, 'The previous release, 8-bit.'],
      ['smart-turn-v3.0', 'Smart Turn v3.0', 'smart-turn-v3.0.onnx', 8_757_193, 'The first v3 release, 8-bit.'],
      [
        'smart-turn-v3-fp32',
        'Smart Turn v3.2 (32-bit)',
        'smart-turn-v3.2-gpu.onnx',
        32_411_198,
        'v3.2 in full precision: the most accurate on the server’s English set (0.80 against 0.63), 32 MB.',
      ],
    ] as const
  ).map(([id, label, file, bytes, blurb]): AudioModel => ({
    id,
    task: 'turn',
    label,
    runner: 'smart-turn',
    files: {
      from: 'hub',
      repo: 'pipecat-ai/smart-turn-v3',
      revision: 'f766f81d3cfdf7737ac64aad813d91bbfd56bf93',
      files: { model: file },
    },
    bytes,
    peakBytes: bytes * 3 + 60e6,
    rate: 16000,
    license: 'BSD-2-Clause',
    source: 'https://huggingface.co/pipecat-ai/smart-turn-v3',
    blurb: `Pipecat’s Smart Turn: hears the last 8 s and says whether the speaker has finished. ${blurb}`,
    webgpu: true,
    device: 'wasm',
  })),
  // -- speaker embeddings (speaker-embedding.yaml; the embedders of POST /v1/audio/speech/embedding) -----------
  ...(
    [
      [
        'eres2net-voxceleb',
        '3D-Speaker ERes2Net (VoxCeleb)',
        '3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx',
        26_485_263,
        'Apache-2.0',
        'https://github.com/modelscope/3D-Speaker',
        'The server’s default embedder, the one its same / similar / different thresholds were measured with. 192 numbers per voice.',
      ],
      [
        'campplus-voxceleb',
        '3D-Speaker CAM++ (VoxCeleb)',
        '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx',
        29_596_978,
        'Apache-2.0',
        'https://github.com/modelscope/3D-Speaker',
        'Quick; the embedder the cloning page scores voices with. 512 numbers per voice.',
      ],
      [
        'campplus-zh-en',
        '3D-Speaker CAM++ (Chinese and English)',
        '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx',
        28_281_164,
        'Apache-2.0',
        'https://github.com/modelscope/3D-Speaker',
        'Trained on about 200 thousand Chinese and English speakers. 192 numbers per voice.',
      ],
      [
        'wespeaker-campplus-voxceleb',
        'WeSpeaker CAM++ (VoxCeleb)',
        'wespeaker_en_voxceleb_CAM++.onnx',
        29_292_684,
        'CC-BY-4.0',
        'https://github.com/wenet-e2e/wespeaker',
        'CAM++ trained by the WeSpeaker project. 512 numbers per voice.',
      ],
      [
        'wespeaker-resnet34-voxceleb',
        'WeSpeaker ResNet34 (VoxCeleb)',
        'wespeaker_en_voxceleb_resnet34.onnx',
        26_534_365,
        'CC-BY-4.0',
        'https://github.com/wenet-e2e/wespeaker',
        'The embedder pyannote 3.1 uses. 256 numbers per voice.',
      ],
    ] as const
  ).map(([id, label, file, bytes, license, source, blurb]): AudioModel => ({
    id,
    task: 'embed',
    label,
    runner: 'speaker-embedding',
    files: {
      from: 'hub',
      repo: 'csukuangfj/speaker-embedding-models',
      revision: '0743f301363dec56491a490f6d6cbc9d67f9a3bf',
      files: { model: file },
    },
    bytes,
    peakBytes: 200e6,
    rate: 16000,
    license,
    source,
    blurb,
    webgpu: true,
    device: 'wasm',
  })),
  {
    id: 'ecapa-tdnn-voxceleb',
    task: 'embed',
    label: 'SpeechBrain ECAPA-TDNN (VoxCeleb)',
    runner: 'speaker-embedding',
    files: {
      from: 'hub',
      repo: 'Pawls2theWall/equivl-ecapa-voxceleb-onnx',
      revision: '4393ad2afd9af3621bcf602ac49e380c9de8306b',
      files: { model: 'model_fp16.onnx' },
    },
    bytes: 42_324_080,
    peakBytes: 260e6,
    rate: 16000,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb',
    blurb:
      'SpeechBrain’s speaker model in a third-party ONNX export with its filterbank inside the graph. 192 numbers per voice.',
    webgpu: true,
    device: 'wasm',
  },
]

export function audioModel(id: string): AudioModel {
  const m = AUDIO_MODELS.find((x) => x.id === id)
  if (!m) throw new Error(`Unknown audio model ${id}`)
  return m
}

export function audioModelsFor(task: AudioTask): AudioModel[] {
  return AUDIO_MODELS.filter((m) => m.task === task)
}

/** Server features left out of the page, and why (shown next to the choices). */
export const NOT_IN_THE_PAGE: { task: AudioTask; label: string; why: string }[] = [
  {
    task: 'denoise',
    label: 'DPDFNet 48 kHz and 8 kHz builds',
    why: 'the page hears 16 kHz audio: the 16 kHz builds are the ones that fit',
  },
  { task: 'denoise', label: 'Demucs denoiser', why: 'PyTorch only (bolo-torch plugin), and not for commercial use' },
  { task: 'denoise', label: 'ElevenLabs Voice Isolator', why: 'a cloud service: audio would leave the page' },
  { task: 'lid', label: 'MMS-LID 126 / 4017', why: '978 MB each and not for commercial use: too large for a tab' },
  { task: 'lid', label: 'AmberNet', why: 'NGC Terms of Use, marked not for commercial use' },
  {
    task: 'vad',
    label: 'WebRTC VAD',
    why: 'no WebAssembly build pinned by the server; its energy-like rules add little here',
  },
  { task: 'vad', label: 'MarbleNet VAD', why: 'needs NeMo’s 80-band front end, not ported yet' },
  {
    task: 'turn',
    label: 'Namo Turn Detector',
    why: '136 MB (English) to 309 MB, and it needs a decoded transcript at every pause',
  },
  {
    task: 'embed',
    label: 'NVIDIA TitaNet-small',
    why: 'needs NeMo’s filterbank front end, not ported to the page yet',
  },
]
