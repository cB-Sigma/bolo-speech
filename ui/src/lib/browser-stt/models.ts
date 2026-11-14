/**
 * Models the in-browser mode can run, fetched from the Hugging Face Hub on first use.
 * Keep docs/ui-server-notes.md in step when adding one (name, licence, source).
 */
export interface BrowserModel {
  id: string
  label: string
  family: 'Moonshine' | 'Whisper'
  repo: string
  /** transformers.js quantisation to request. */
  dtype: 'q8' | 'fp32' | 'q4'
  /** Bytes fetched on first use (weights, tokenizer and configuration), as listed on the Hub. */
  approxBytes: number
  params: number
  languages: string[]
  /** One model for many languages: the language is chosen (or guessed) per utterance. */
  multilingual: boolean
  license: string
  source: string
  /** Short honest description for pickers. */
  blurb: string
}

/** Languages in Whisper's tokenizer (multilingual checkpoints up to large-v2). */
export const WHISPER_LANGUAGES =
  'en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su'.split(
    ' ',
  )

export const BROWSER_MODELS: BrowserModel[] = [
  {
    id: 'moonshine-tiny',
    label: 'Moonshine Tiny',
    family: 'Moonshine',
    repo: 'onnx-community/moonshine-tiny-ONNX',
    dtype: 'q8',
    approxBytes: 32_100_000,
    params: 27_000_000,
    languages: ['en'],
    multilingual: false,
    license: 'MIT',
    source: 'https://huggingface.co/onnx-community/moonshine-tiny-ONNX',
    blurb: 'English only. The fastest: built for live use.',
  },
  {
    // Fine-tuned for Hindi by ARTPARK-IISc (Apache-2.0); the ONNX conversion is a community upload.
    id: 'whisper-tiny-hindi',
    label: 'Whisper Tiny Hindi',
    family: 'Whisper',
    repo: 'astronova001/whisper-tiny-hindi-ONNX',
    dtype: 'q8',
    approxBytes: 124_000_000,
    params: 39_000_000,
    languages: ['hi'],
    multilingual: false,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/ARTPARK-IISc/whisper-tiny-vaani-hindi',
    blurb:
      'Hindi only. Whisper Tiny fine-tuned on about 700 hours of Hindi; the only one here that is usable for Hindi.',
  },
  {
    id: 'whisper-tiny-q8',
    label: 'Whisper Tiny',
    family: 'Whisper',
    repo: 'onnx-community/whisper-tiny',
    dtype: 'q8',
    approxBytes: 43_600_000,
    params: 39_000_000,
    languages: WHISPER_LANGUAGES,
    multilingual: true,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/onnx-community/whisper-tiny',
    blurb: 'Multilingual. Fine for major European languages; in our Hindi test it got nothing usable.',
  },
  {
    id: 'whisper-base-q8',
    label: 'Whisper Base',
    family: 'Whisper',
    repo: 'onnx-community/whisper-base',
    dtype: 'q8',
    approxBytes: 79_700_000,
    params: 74_000_000,
    languages: WHISPER_LANGUAGES,
    multilingual: true,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/onnx-community/whisper-base',
    blurb: 'Multilingual, a step up from Tiny. Still poor in Indian languages: a rough guess at best.',
  },
]

export const DEFAULT_BROWSER_MODEL = BROWSER_MODELS[0] as BrowserModel

/**
 * The model to suggest for a language: a model made for that language if there
 * is one, otherwise the larger multilingual model.
 */
export function suggestedModelFor(language: string): BrowserModel | null {
  return (
    BROWSER_MODELS.find((m) => !m.multilingual && m.languages.includes(language)) ??
    BROWSER_MODELS.find((m) => m.id === 'whisper-base-q8' && m.languages.includes(language)) ??
    null
  )
}

export function browserModel(id: string | null | undefined): BrowserModel {
  return BROWSER_MODELS.find((m) => m.id === id) ?? DEFAULT_BROWSER_MODEL
}

export type BrowserDevice = 'webgpu' | 'wasm'

export interface BrowserSttConfig {
  /** Origin serving model files in Hub layout; defaults to huggingface.co. */
  modelHost?: string
  /** Directory serving the ONNX Runtime `.wasm`/`.mjs` files; defaults to the jsDelivr copy of the pinned version. */
  ortWasmBase?: string
  /** Force a device instead of auto-detecting. */
  device?: BrowserDevice
  /** Seconds without any progress before the load is reported as stuck (default 40). */
  stallSeconds?: number
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

/** What the loader is doing right now. */
export type LoadStage = 'runtime' | 'model' | 'compiling' | 'warmup'

export type WorkerRequest =
  | {
      type: 'load'
      repo: string
      family: BrowserModel['family']
      dtype: string
      device: BrowserDevice
      threads: number
      /** Leave fetching the runtime binary entirely to the library (no progress reporting). */
      plainRuntimeLoad: boolean
      config: BrowserSttConfig
    }
  | {
      type: 'transcribe'
      id: number
      audio: Float32Array
      /** BCP-47 code, or `auto` to guess (multilingual models only). */
      language?: string
    }

export interface FileProgress {
  file: string
  loaded: number
  total: number
}

/** One network request made while loading, for the diagnostics panel. */
export interface FetchRecord {
  url: string
  /** HTTP status, or 0 when the request failed before a response. */
  status: number
  /** Bytes, from Content-Length or counted while streaming; null when unknown. */
  bytes: number | null
  /** Time to response headers (or to the end of the body for the runtime binary). */
  ms: number
  error?: string
}

export type WorkerResponse =
  | { type: 'progress'; files: FileProgress[] }
  | { type: 'stage'; stage: LoadStage; loaded?: number; total?: number }
  | { type: 'fetch'; record: FetchRecord }
  | {
      type: 'ready'
      device: BrowserDevice
      loadMs: number
      /** Cache Storage keys of the model's files, and their size (for the downloads list). */
      cacheKeys?: string[]
      bytes?: number
    }
  | { type: 'result'; id: number; text: string; ms: number; language: string | null }
  | { type: 'error'; id?: number; message: string }
