/**
 * Models the in-browser mode can run, fetched from the Hugging Face Hub on first use.
 * Keep docs/ui-server-notes.md in step when adding one (name, licence, source).
 */
export interface BrowserModel {
  id: string
  label: string
  repo: string
  /** transformers.js quantisation to request. */
  dtype: 'q8' | 'fp32' | 'q4'
  approxBytes: number
  languages: string[]
  license: string
  source: string
}

export const BROWSER_MODELS: BrowserModel[] = [
  {
    id: 'moonshine-tiny',
    label: 'Moonshine Tiny',
    repo: 'onnx-community/moonshine-tiny-ONNX',
    dtype: 'q8',
    approxBytes: 28_200_000,
    languages: ['en'],
    license: 'MIT',
    source: 'https://huggingface.co/onnx-community/moonshine-tiny-ONNX',
  },
]

export type BrowserDevice = 'webgpu' | 'wasm'

export interface BrowserSttConfig {
  /** Origin serving model files in Hub layout; defaults to huggingface.co. */
  modelHost?: string
  /** Directory serving the ONNX Runtime `.wasm`/`.mjs` files; defaults to the jsDelivr copy of the pinned version. */
  ortWasmBase?: string
  /** Force a device instead of auto-detecting. */
  device?: BrowserDevice
}

const KEY = 'bolo.browserStt'

/** Deployment overrides (air-gapped mirrors, tests) read from local storage. */
export function readBrowserSttConfig(): BrowserSttConfig {
  try {
    const raw = globalThis.localStorage?.getItem(KEY)
    return raw ? (JSON.parse(raw) as BrowserSttConfig) : {}
  } catch {
    return {}
  }
}

export type WorkerRequest =
  | { type: 'load'; repo: string; dtype: string; device: BrowserDevice; threads: number; config: BrowserSttConfig }
  | { type: 'transcribe'; id: number; audio: Float32Array }

export interface FileProgress {
  file: string
  loaded: number
  total: number
}

export type WorkerResponse =
  | { type: 'progress'; files: FileProgress[] }
  | { type: 'ready'; device: BrowserDevice; loadMs: number }
  | { type: 'result'; id: number; text: string; ms: number }
  | { type: 'error'; id?: number; message: string }
