import type { SttWord } from '@/lib/api/types'
import type { PlaceableModel } from '@/lib/placement'
import type { GpuPower } from '@/lib/webgpu'

/**
 * Models the in-browser mode can run, fetched from the Hugging Face Hub on first use.
 * Keep docs/ui-server-notes.md in step when adding one (name, licence, source).
 *
 * Every model that runs in a browser is listed, light or heavy: each says
 * what it costs (download, memory, WebGPU) and the visitor decides.
 */
export interface BrowserModel {
  id: string
  label: string
  family: 'Moonshine' | 'Whisper' | 'Wav2Vec2'
  repo: string
  /** transformers.js quantisation to request. */
  dtype: 'q8' | 'fp32' | 'q4' | 'q4f16' | 'fp16'
  /** Bytes fetched on first use (weights, tokenizer and configuration), as listed on the Hub. */
  approxBytes: number
  /** Estimated peak memory while it runs, bytes. */
  peakBytes: number
  params: number
  languages: string[]
  /** One model for many languages: the language is chosen (or guessed) per utterance. */
  multilingual: boolean
  /** Whisper checkpoints trained on English only take no language or task tokens. */
  englishOnly?: boolean
  /** Whisper fine-tunes that must be given this language token whatever is spoken (Hinglish output). */
  forceLanguage?: string
  /** Only runs with WebGPU (16-bit weights). */
  requiresWebGpu?: boolean
  /**
   * Writes no punctuation or capitals of its own (CTC models, some fine-tunes), as the server's catalog says
   * (no `punctuation` feature): the page's punctuation model restores them in the languages it knows.
   */
  noPunctuation?: boolean
  /** Runs on the CPU too, but is meant for WebGPU: on the CPU it is slow. */
  prefersWebGpu?: boolean
  license: string
  /** Licence or access terms worth knowing before use (non-commercial, gated original, …). */
  licenseNote?: string
  source: string
  /** Picker group. */
  group: 'English' | 'Hindi and Indian languages' | 'Many languages' | 'Other languages'
  /** Short honest description for pickers. */
  blurb: string
}

/** Languages in Whisper's tokenizer (multilingual checkpoints up to large-v2). */
export const WHISPER_LANGUAGES =
  'en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su'.split(
    ' ',
  )
/** large-v3 added Cantonese. */
const WHISPER_V3_LANGUAGES = [...WHISPER_LANGUAGES, 'yue']

/** Peak memory estimate: the weights as loaded plus the runtime and activations. */
const peak = (bytes: number, factor = 2.2) => Math.round(bytes * factor + 120e6)

const HUB = 'https://huggingface.co/'
const MOONSHINE_LICENCE =
  'Moonshine Community License: free for research and for organisations under USD 1M revenue; others need a licence from Useful Sensors.'

function moonshineLanguage(
  size: 'tiny' | 'base',
  code: string,
  name: string,
  bytes: number,
  license = 'Moonshine Community License',
): BrowserModel {
  const repo = `onnx-community/moonshine-${size}-${code}-ONNX`
  return {
    id: `moonshine-${size}-${code}`,
    label: `Moonshine ${size === 'tiny' ? 'Tiny' : 'Base'} ${name}`,
    family: 'Moonshine',
    repo,
    dtype: 'q8',
    approxBytes: bytes,
    peakBytes: peak(bytes),
    params: size === 'tiny' ? 27_000_000 : 61_000_000,
    languages: [code],
    multilingual: false,
    license,
    licenseNote: license === 'MIT' ? undefined : MOONSHINE_LICENCE,
    source: `${HUB}${repo}`,
    group: 'Other languages',
    blurb: `${name} only. Moonshine trained for one language by Useful Sensors.`,
  }
}

export const BROWSER_MODELS: BrowserModel[] = [
  {
    id: 'moonshine-tiny',
    label: 'Moonshine Tiny',
    family: 'Moonshine',
    repo: 'onnx-community/moonshine-tiny-ONNX',
    dtype: 'q8',
    approxBytes: 32_100_000,
    peakBytes: peak(32_100_000),
    params: 27_000_000,
    languages: ['en'],
    multilingual: false,
    license: 'MIT',
    source: 'https://huggingface.co/onnx-community/moonshine-tiny-ONNX',
    group: 'English',
    blurb: 'English only. The fastest: built for live use.',
  },
  {
    // Fine-tuned for Hindi by ARTPARK-IISc (Apache-2.0); the ONNX conversion is a community upload.
    id: 'whisper-tiny-hindi',
    noPunctuation: true,
    label: 'Whisper Tiny Hindi',
    family: 'Whisper',
    repo: 'astronova001/whisper-tiny-hindi-ONNX',
    dtype: 'q8',
    approxBytes: 124_000_000,
    peakBytes: peak(124_000_000),
    params: 39_000_000,
    languages: ['hi'],
    multilingual: false,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/ARTPARK-IISc/whisper-tiny-vaani-hindi',
    group: 'Hindi and Indian languages',
    blurb: 'Hindi only. Whisper Tiny fine-tuned on about 700 hours of Hindi; small and quick, usable for Hindi.',
  },
  {
    id: 'whisper-tiny-q8',
    label: 'Whisper Tiny',
    family: 'Whisper',
    repo: 'onnx-community/whisper-tiny',
    dtype: 'q8',
    approxBytes: 43_600_000,
    peakBytes: peak(43_600_000),
    params: 39_000_000,
    languages: WHISPER_LANGUAGES,
    multilingual: true,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/onnx-community/whisper-tiny',
    group: 'Many languages',
    blurb: 'Multilingual. Fine for major European languages; in our Hindi test it got nothing usable.',
  },
  {
    id: 'whisper-base-q8',
    label: 'Whisper Base',
    family: 'Whisper',
    repo: 'onnx-community/whisper-base',
    dtype: 'q8',
    approxBytes: 79_700_000,
    peakBytes: peak(79_700_000),
    params: 74_000_000,
    languages: WHISPER_LANGUAGES,
    multilingual: true,
    license: 'Apache-2.0',
    source: 'https://huggingface.co/onnx-community/whisper-base',
    group: 'Many languages',
    blurb: 'Multilingual, a step up from Tiny. Still poor in Indian languages: a rough guess at best.',
  },
  {
    id: 'moonshine-base',
    label: 'Moonshine Base',
    family: 'Moonshine',
    repo: 'onnx-community/moonshine-base-ONNX',
    dtype: 'q8',
    approxBytes: 67_100_000,
    peakBytes: peak(67_100_000),
    params: 61_000_000,
    languages: ['en'],
    multilingual: false,
    license: 'MIT',
    source: `${HUB}onnx-community/moonshine-base-ONNX`,
    group: 'English',
    blurb: 'English only. More accurate than Tiny and still quick enough for live use on a laptop.',
  },
  {
    id: 'whisper-tiny-en',
    label: 'Whisper Tiny (English)',
    family: 'Whisper',
    repo: 'onnx-community/whisper-tiny.en',
    dtype: 'q8',
    approxBytes: 45_200_000,
    peakBytes: peak(45_200_000),
    params: 39_000_000,
    languages: ['en'],
    multilingual: false,
    englishOnly: true,
    license: 'Apache-2.0',
    source: `${HUB}onnx-community/whisper-tiny.en`,
    group: 'English',
    blurb: 'English only. The English-trained Tiny: a little better than the multilingual one in English.',
  },
  {
    id: 'whisper-base-en',
    label: 'Whisper Base (English)',
    family: 'Whisper',
    repo: 'onnx-community/whisper-base.en',
    dtype: 'q8',
    approxBytes: 81_300_000,
    peakBytes: peak(81_300_000),
    params: 74_000_000,
    languages: ['en'],
    multilingual: false,
    englishOnly: true,
    license: 'Apache-2.0',
    source: `${HUB}onnx-community/whisper-base.en`,
    group: 'English',
    blurb: 'English only.',
  },
  {
    id: 'distil-small-en',
    label: 'Distil-Whisper Small (English)',
    family: 'Whisper',
    repo: 'onnx-community/distil-small.en',
    dtype: 'q8',
    approxBytes: 176_000_000,
    peakBytes: peak(176_000_000),
    params: 166_000_000,
    languages: ['en'],
    multilingual: false,
    englishOnly: true,
    license: 'MIT',
    source: `${HUB}distil-whisper/distil-small.en`,
    group: 'English',
    blurb: 'English only. Whisper Small distilled to a shallow decoder: close to Small, much faster.',
  },
  {
    id: 'whisper-small-en',
    label: 'Whisper Small (English)',
    family: 'Whisper',
    repo: 'onnx-community/whisper-small.en',
    dtype: 'q8',
    approxBytes: 253_000_000,
    peakBytes: peak(253_000_000),
    params: 244_000_000,
    languages: ['en'],
    multilingual: false,
    englishOnly: true,
    license: 'Apache-2.0',
    source: `${HUB}onnx-community/whisper-small.en`,
    group: 'English',
    prefersWebGpu: true,
    blurb: 'English only. Accurate; slow on a phone.',
  },
  {
    id: 'distil-medium-en',
    label: 'Distil-Whisper Medium (English)',
    family: 'Whisper',
    repo: 'distil-whisper/distil-medium.en',
    dtype: 'q8',
    approxBytes: 406_000_000,
    peakBytes: peak(406_000_000),
    params: 394_000_000,
    languages: ['en'],
    multilingual: false,
    englishOnly: true,
    license: 'MIT',
    source: `${HUB}distil-whisper/distil-medium.en`,
    group: 'English',
    prefersWebGpu: true,
    blurb: 'English only. Near Whisper Medium in English at a fraction of its decoding time. Heavy for phones.',
  },
  {
    id: 'whisper-small',
    label: 'Whisper Small',
    family: 'Whisper',
    repo: 'onnx-community/whisper-small',
    dtype: 'q8',
    approxBytes: 253_000_000,
    peakBytes: peak(253_000_000),
    params: 244_000_000,
    languages: WHISPER_LANGUAGES,
    multilingual: true,
    license: 'Apache-2.0',
    source: `${HUB}onnx-community/whisper-small`,
    group: 'Many languages',
    prefersWebGpu: true,
    blurb: 'Multilingual. The smallest Whisper that gets Indian languages partly right. Slow on a phone.',
  },
  {
    id: 'whisper-medium',
    label: 'Whisper Medium',
    family: 'Whisper',
    repo: 'onnx-community/whisper-medium-ONNX',
    dtype: 'q4',
    approxBytes: 685_000_000,
    peakBytes: peak(685_000_000, 2),
    params: 769_000_000,
    languages: WHISPER_LANGUAGES,
    multilingual: true,
    license: 'Apache-2.0',
    source: `${HUB}onnx-community/whisper-medium-ONNX`,
    group: 'Many languages',
    prefersWebGpu: true,
    blurb: 'Multilingual, 4-bit weights. Much better in Indian languages than Small; a laptop with WebGPU is advised.',
  },
  {
    id: 'whisper-large-v3-turbo',
    label: 'Whisper Large v3 Turbo',
    family: 'Whisper',
    repo: 'onnx-community/whisper-large-v3-turbo',
    dtype: 'q4',
    approxBytes: 763_000_000,
    peakBytes: peak(763_000_000, 2),
    params: 809_000_000,
    languages: WHISPER_V3_LANGUAGES,
    multilingual: true,
    license: 'MIT',
    source: `${HUB}openai/whisper-large-v3-turbo`,
    group: 'Many languages',
    prefersWebGpu: true,
    blurb:
      'Multilingual, the best Whisper that fits a tab (4-bit weights). Needs a laptop; with WebGPU it is usable live.',
  },
  {
    id: 'whisper-large-v3-turbo-f16',
    label: 'Whisper Large v3 Turbo (WebGPU, 16-bit)',
    family: 'Whisper',
    repo: 'onnx-community/whisper-large-v3-turbo',
    dtype: 'q4f16',
    approxBytes: 567_000_000,
    peakBytes: peak(567_000_000, 1.6),
    params: 809_000_000,
    languages: WHISPER_V3_LANGUAGES,
    multilingual: true,
    license: 'MIT',
    source: `${HUB}openai/whisper-large-v3-turbo`,
    group: 'Many languages',
    requiresWebGpu: true,
    blurb: 'The same model with 16-bit activations: smaller and faster, but only on a GPU with 16-bit shaders.',
  },
  {
    id: 'whisper-hindi-small',
    noPunctuation: true,
    label: 'Whisper Small Hindi (IIT Madras)',
    family: 'Whisper',
    repo: 'onnx-community/whisper-hindi-small-ONNX',
    dtype: 'q4',
    approxBytes: 304_000_000,
    peakBytes: peak(304_000_000, 2),
    params: 244_000_000,
    languages: ['hi'],
    multilingual: false,
    license: 'Apache-2.0',
    source: `${HUB}vasista22/whisper-hindi-small`,
    group: 'Hindi and Indian languages',
    prefersWebGpu: true,
    blurb:
      'Hindi only. Whisper Small fine-tuned on Hindi by SPRING Lab, IIT Madras (4-bit weights): more accurate than Tiny Hindi, much slower.',
  },
  {
    id: 'whisper-hinglish-swift',
    noPunctuation: true,
    label: 'Whisper Hindi to Hinglish (Swift)',
    family: 'Whisper',
    repo: 'Hirecentive-D3l/Whisper-Hindi2Hinglish-Swift-ONNX',
    dtype: 'q8',
    approxBytes: 108_000_000,
    peakBytes: peak(108_000_000),
    params: 74_000_000,
    languages: ['hi'],
    multilingual: false,
    forceLanguage: 'en',
    license: 'Apache-2.0',
    source: `${HUB}Oriserve/Whisper-Hindi2Hinglish-Swift`,
    group: 'Hindi and Indian languages',
    blurb:
      'Hindi and Hindi-English speech written out in Latin letters (Hinglish), by Oriserve; trained on noisy Indian audio.',
  },
  {
    id: 'indicwav2vec-hindi',
    noPunctuation: true,
    label: 'IndicWav2Vec Hindi (AI4Bharat)',
    family: 'Wav2Vec2',
    repo: 'onnx-community/indicwav2vec-hindi-ONNX',
    dtype: 'q8',
    approxBytes: 317_000_000,
    peakBytes: peak(317_000_000),
    params: 315_000_000,
    languages: ['hi'],
    multilingual: false,
    license: 'Apache-2.0',
    licenseNote: 'The original checkpoint on the Hub asks you to accept its terms; this ONNX copy does not.',
    source: `${HUB}ai4bharat/indicwav2vec-hindi`,
    group: 'Hindi and Indian languages',
    blurb: 'Hindi only. A character model (CTC) from AI4Bharat: no punctuation, never invents text that was not said.',
  },
  {
    id: 'wav2vec2-base-960h',
    label: 'Wav2Vec2 Base (English)',
    family: 'Wav2Vec2',
    repo: 'onnx-community/wav2vec2-base-960h-ONNX',
    dtype: 'q8',
    approxBytes: 95_800_000,
    peakBytes: peak(95_800_000),
    params: 95_000_000,
    languages: ['en'],
    multilingual: false,
    noPunctuation: true,
    license: 'Apache-2.0',
    source: `${HUB}facebook/wav2vec2-base-960h`,
    group: 'English',
    blurb:
      'English only. Meta’s character model (CTC) trained on LibriSpeech read speech: capitals only, no punctuation, which the page’s punctuation model adds.',
  },
  moonshineLanguage('tiny', 'ar', 'Arabic', 32_100_000),
  moonshineLanguage('tiny', 'zh', 'Chinese', 32_100_000),
  moonshineLanguage('base', 'zh', 'Chinese', 67_100_000),
  moonshineLanguage('tiny', 'ja', 'Japanese', 32_100_000),
  moonshineLanguage('base', 'ja', 'Japanese', 67_100_000),
  moonshineLanguage('tiny', 'ko', 'Korean', 32_100_000),
  moonshineLanguage('base', 'ko', 'Korean', 67_100_000),
  moonshineLanguage('tiny', 'uk', 'Ukrainian', 129_000_000),
  moonshineLanguage('base', 'uk', 'Ukrainian', 246_000_000),
  moonshineLanguage('tiny', 'vi', 'Vietnamese', 129_000_000),
  moonshineLanguage('tiny', 'fr', 'French', 32_100_000, 'MIT'),
]

export const DEFAULT_BROWSER_MODEL = BROWSER_MODELS[0] as BrowserModel

/** Picker groups in order, each with its models. */
export function modelsByGroup(): { group: BrowserModel['group']; models: BrowserModel[] }[] {
  const order: BrowserModel['group'][] = ['English', 'Hindi and Indian languages', 'Many languages', 'Other languages']
  return order.map((group) => ({ group, models: BROWSER_MODELS.filter((m) => m.group === group) }))
}

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

/** The graphs a recogniser is made of, as transformers.js names its sessions. */
export function sttModules(m: Pick<BrowserModel, 'family'>): string[] {
  return m.family === 'Wav2Vec2' ? ['model'] : ['encoder_model', 'decoder_model_merged']
}

/**
 * Recognisers up to this size start on the CPU on Auto. Measured on a Windows laptop (Ryzen 7 5800H, RTX 3060
 * Laptop GPU, Chrome and Edge, 2.7 s of English, WebAssembly with threads vs WebGPU, real-time factor):
 * Moonshine Tiny 0.097 vs 0.94, Moonshine Base 0.17 vs 1.55, Whisper Base (English) 0.99 vs 1.92, with the same
 * transcripts; loading took 1.4 to 1.9 s on the CPU and 11 to 12 s on WebGPU. These models are too small to keep
 * a GPU busy: each decoding step is a round trip. WebGPU stays one choice away in the picker.
 */
export const SMALL_RECOGNISER_PARAMS = 100_000_000

export function isSmallRecogniser(m: Pick<BrowserModel, 'params' | 'requiresWebGpu' | 'prefersWebGpu'>): boolean {
  return m.params <= SMALL_RECOGNISER_PARAMS && !m.requiresWebGpu && !m.prefersWebGpu
}

/** A recogniser in the terms of lib/placement.ts. */
export function placeableStt(m: BrowserModel): PlaceableModel {
  return {
    id: m.id,
    modules: sttModules(m),
    bytes: m.approxBytes,
    peakBytes: m.peakBytes,
    requiresWebGpu: m.requiresWebGpu,
    gpuFirst: !isSmallRecogniser(m),
    cpuWhy:
      'a small recogniser: faster on the CPU than on WebGPU (measured on a laptop: Moonshine Tiny 0.10 vs 0.94 real-time factor)',
    splitText: m.family === 'Wav2Vec2' ? undefined : 'encoder on WebGPU, decoder on the CPU',
  }
}

export interface BrowserSttConfig {
  /** Origin serving model files in Hub layout; defaults to huggingface.co. */
  modelHost?: string
  /** Directory serving the ONNX Runtime `.wasm`/`.mjs` files; defaults to the jsDelivr copy of the pinned version. */
  ortWasmBase?: string
  /** Base of npm packages loaded at run time (RNNoise's WebAssembly build); defaults to jsDelivr. */
  npmBase?: string
  /** Directory of the demo's own model copies (GTCRN); defaults to `models/` next to the demo's page. */
  modelsBase?: string
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
      englishOnly?: boolean
      forceLanguage?: string
      device: BrowserDevice
      /** Device per graph when the model is split (encoder on WebGPU, decoder on the CPU). */
      devices?: Record<string, BrowserDevice>
      threads: number
      /** Leave fetching the runtime binary entirely to the library (no progress reporting). */
      plainRuntimeLoad: boolean
      config: BrowserSttConfig
      /** Which GPU WebGPU asks for (the device panel's choice); high performance when absent. */
      power?: GpuPower
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
  | { type: 'result'; id: number; text: string; ms: number; language: string | null; words?: SttWord[] }
  | { type: 'error'; id?: number; message: string }
