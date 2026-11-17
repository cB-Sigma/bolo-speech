/**
 * Models the in-page speaker diarization can use, fetched from the Hugging
 * Face Hub at pinned revisions on first use. Licences as their authors state
 * them; heavier models that need a bolo server are listed so the visitor
 * knows they exist.
 */

export interface DiarizeModel {
  id: string
  label: string
  role: 'segmentation' | 'embedding'
  repo: string
  revision: string
  /** transformers.js models: the precision; ONNX Runtime models: the file. */
  runner: { kind: 'transformers'; dtype: 'fp32' | 'q8' } | { kind: 'onnx'; file: string }
  bytes: number
  /** Estimated memory while it runs. */
  peakBytes: number
  license: string
  licenseNote?: string
  source: string
  blurb: string
  /** Shown next to heavy choices: works on the CPU, better on a laptop or with WebGPU. */
  heavy?: string
}

export const SEGMENTATION: DiarizeModel = {
  id: 'pyannote-segmentation-3.0',
  label: 'pyannote segmentation 3.0',
  role: 'segmentation',
  repo: 'onnx-community/pyannote-segmentation-3.0',
  revision: '733a93b6473d019a773298e08cefa686894b1854',
  runner: { kind: 'transformers', dtype: 'fp32' },
  bytes: 5_987_000,
  peakBytes: 80e6,
  license: 'MIT',
  licenseNote:
    'The original (pyannote/segmentation-3.0) asks for contact details before download; this ONNX copy does not.',
  source: 'https://huggingface.co/pyannote/segmentation-3.0',
  blurb: 'Who talks in each 10 s window, up to three at once, overlaps included.',
}

export const EMBEDDINGS: DiarizeModel[] = [
  {
    id: 'wespeaker-resnet34-lm',
    label: 'WeSpeaker ResNet34-LM',
    role: 'embedding',
    repo: 'onnx-community/wespeaker-voxceleb-resnet34-LM',
    revision: '6a61a1833ff2583aabeba044f5c8221f00b67ceb',
    runner: { kind: 'transformers', dtype: 'fp32' },
    bytes: 26_536_000,
    peakBytes: 200e6,
    license: 'CC-BY-4.0',
    source: 'https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM',
    blurb: 'Voice embeddings trained on VoxCeleb (mostly English). The one pyannote 3.1 uses.',
  },
  {
    id: 'campplus-voxceleb',
    label: '3D-Speaker CAM++ (VoxCeleb)',
    role: 'embedding',
    repo: 'csukuangfj/speaker-embedding-models',
    revision: '0743f301363dec56491a490f6d6cbc9d67f9a3bf',
    runner: { kind: 'onnx', file: '3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx' },
    bytes: 29_597_000,
    peakBytes: 200e6,
    license: 'Apache-2.0',
    source: 'https://www.modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k/summary',
    blurb: 'Alibaba 3D-Speaker CAM++ trained on VoxCeleb; quick, in the ONNX form sherpa-onnx publishes.',
  },
  {
    id: 'wavlm-base-plus-sv',
    label: 'WavLM Base Plus SV',
    role: 'embedding',
    repo: 'Xenova/wavlm-base-plus-sv',
    revision: 'e61029603001bd11295c36d878698708bf59190f',
    runner: { kind: 'transformers', dtype: 'q8' },
    bytes: 101_700_000,
    peakBytes: 600e6,
    license: 'MIT',
    licenseNote: 'Microsoft UniLM (MIT); the original model card names no licence of its own.',
    source: 'https://huggingface.co/microsoft/wavlm-base-plus-sv',
    blurb: 'A self-supervised speech model fine-tuned for speaker verification. Slower, 8-bit.',
    heavy: 'laptop or desktop: about 100 MB and several times slower than the others on a phone',
  },
]

/** Diarizers bolo runs on a server that the page does not run (yet): listed with what they need. */
export const SERVER_DIARIZERS = [
  {
    label: 'NVIDIA Nemotron-3-Diarization',
    license: 'OpenMDW-1.1',
    size: '120 MB (int8)',
    needs: 'needs a bolo server: its streaming speaker cache and 128-band features are not in the page yet',
    source: 'https://huggingface.co/nvidia/Nemotron-3-Diarization',
  },
  {
    label: 'DiCoW v3.3 (speaker-conditioned Whisper)',
    license: 'CC-BY-4.0',
    size: '1.5 GB (ONNX)',
    needs: 'needs a bolo server or a desktop with a large GPU: too large for a browser tab',
    source: 'https://huggingface.co/BUT-FIT/DiCoW_v3_3',
  },
  {
    label: "pyannote's PyTorch pipelines (speaker-diarization-3.1, community-1)",
    license: 'MIT + CC-BY-4.0 · gated',
    size: '33 MB',
    needs:
      'needs a bolo server with the torch engine plugin (PyTorch, pyannote.audio); accept the conditions on Hugging Face and set BOLO_HF_TOKEN',
    source: 'https://huggingface.co/pyannote/speaker-diarization-community-1',
  },
]

export const DEFAULT_EMBEDDING = EMBEDDINGS[0] as DiarizeModel

export function embeddingModel(id: string | null | undefined): DiarizeModel {
  return EMBEDDINGS.find((m) => m.id === id) ?? DEFAULT_EMBEDDING
}

export type DiarizeWorkerRequest =
  | {
      type: 'load'
      segmentation: DiarizeModel
      embedding: DiarizeModel
      threads: number
      /** Origin serving model files in Hub layout (tests); defaults to huggingface.co. */
      modelHost?: string
      ortWasmBase?: string
    }
  | { type: 'run'; id: number; audio: Float32Array; speakers: number | null; threshold: number }

export type DiarizeWorkerResponse =
  | { type: 'progress'; files: { file: string; loaded: number; total: number }[] }
  | { type: 'ready'; loadMs: number; cacheKeys: string[]; bytes: number }
  | { type: 'step'; id: number; done: number }
  | {
      type: 'result'
      id: number
      segments: { speaker: number; start: number; end: number }[]
      ms: number
    }
  | { type: 'error'; id?: number; message: string }
