/**
 * Language models the voice conversation can run in the page (transformers.js,
 * ONNX Runtime Web), from ~0.4 GB to over 1 GB, at pinned revisions. 4-bit
 * weights with 16-bit activations on WebGPU when the GPU has shader-f16; the
 * 4-bit fp32 file on the CPU otherwise (bigger and slow: labelled).
 */
export interface LlmModel {
  id: string
  label: string
  repo: string
  revision: string
  params: string
  /** Download per precision, bytes. */
  bytes: { q4f16: number; q4: number }
  license: string
  licenseNote?: string
  source: string
  blurb: string
  /** Needs WebGPU to be usable (the CPU would take tens of seconds per reply). */
  gpuOnly?: boolean
}

export const LLM_MODELS: LlmModel[] = [
  {
    id: 'smollm2-360m',
    label: 'SmolLM2 360M Instruct',
    repo: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    revision: 'a10cc1512eabd3dde888204e902eca88bddb4951',
    params: '0.36B',
    bytes: { q4f16: 273e6, q4: 388e6 },
    license: 'Apache-2.0',
    source: 'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct',
    blurb: 'English only. The smallest: quick even on the CPU, short and simple replies.',
  },
  {
    id: 'qwen2.5-0.5b',
    label: 'Qwen2.5 0.5B Instruct',
    repo: 'onnx-community/Qwen2.5-0.5B-Instruct',
    revision: '22942cb7d7ba4cc81bb4673549ca4d4614469b5e',
    params: '0.5B',
    bytes: { q4f16: 483e6, q4: 786e6 },
    license: 'Apache-2.0',
    source: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
    blurb: 'Many languages (some Hindi). A good first choice with WebGPU.',
  },
  {
    id: 'gemma-3-1b',
    label: 'Gemma 3 1B IT',
    repo: 'onnx-community/gemma-3-1b-it-ONNX',
    revision: 'a58439f40017d3b99c7d378ff525e54e0ba08ebf',
    params: '1B',
    bytes: { q4f16: 763e6, q4: 859e6 },
    license: 'Gemma Terms of Use',
    licenseNote: "Google's own terms, with a prohibited-use policy; not an open-source licence.",
    source: 'https://huggingface.co/google/gemma-3-1b-it',
    blurb: 'Better replies than the smaller ones; needs WebGPU to answer quickly.',
    gpuOnly: true,
  },
  {
    id: 'llama-3.2-1b',
    label: 'Llama 3.2 1B Instruct',
    repo: 'onnx-community/Llama-3.2-1B-Instruct-ONNX',
    revision: '82ac69368916173bad498ad390290faab92d4fb0',
    params: '1.2B',
    bytes: { q4f16: 1090e6, q4: 1693e6 },
    license: 'Llama 3.2 Community License',
    licenseNote: "Meta's licence: attribution, an acceptable-use policy, and limits above 700M monthly users.",
    source: 'https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct',
    blurb: 'Good English replies; about 1.1 GB, laptop or desktop with WebGPU.',
    gpuOnly: true,
  },
  {
    id: 'qwen2.5-1.5b',
    label: 'Qwen2.5 1.5B Instruct',
    repo: 'onnx-community/Qwen2.5-1.5B-Instruct',
    revision: 'd54957e3af1be04546436e874a60fe386041b244',
    params: '1.5B',
    bytes: { q4f16: 1222e6, q4: 1788e6 },
    license: 'Apache-2.0',
    source: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct',
    blurb: 'The best replies here; about 1.2 GB, needs WebGPU and a laptop or desktop.',
    gpuOnly: true,
  },
]

export const DEFAULT_LLM = LLM_MODELS[0] as LlmModel

export function llmModel(id: string | null | undefined): LlmModel {
  return LLM_MODELS.find((m) => m.id === id) ?? DEFAULT_LLM
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export type LlmWorkerRequest =
  | {
      type: 'load'
      repo: string
      revision: string
      dtype: 'q4f16' | 'q4'
      device: 'webgpu' | 'wasm'
      threads: number
      modelHost?: string
    }
  | { type: 'generate'; id: number; messages: ChatMessage[]; maxTokens: number }
  | { type: 'stop' }

export type LlmWorkerResponse =
  | { type: 'progress'; files: { file: string; loaded: number; total: number }[] }
  | { type: 'ready'; loadMs: number; cacheKeys: string[]; bytes: number; device: 'webgpu' | 'wasm' }
  | { type: 'token'; id: number; text: string }
  | { type: 'done'; id: number; text: string; tokens: number; ms: number }
  | { type: 'error'; id?: number; message: string }
