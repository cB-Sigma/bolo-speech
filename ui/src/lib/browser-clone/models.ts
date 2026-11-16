/**
 * Voice-cloning models the page can run, fetched from the Hugging Face Hub
 * at pinned revisions on first use (nothing is hosted with the demo).
 * Licences as their authors state them on the model cards.
 */

export type CloneFamily = 'pocket' | 'chatterbox' | 'moss'

export type CloneGroup = 'Small: phones too' | 'Medium: a laptop' | 'Large: a desktop, WebGPU advised'

export interface CloneModel {
  id: string
  label: string
  family: CloneFamily
  repo: string
  revision: string
  /** Pocket TTS: the language bundle's directory in the repository. */
  bundle?: string
  /** Chatterbox: transformers.js dtype on the CPU and on WebGPU. */
  dtype?: { wasm: string; webgpu: string }
  /** Chatterbox multilingual: the text starts with a language tag such as `[hi]`. */
  languageTag?: boolean
  /** Languages it speaks (ISO 639-1). */
  languages: string[]
  /** Bytes downloaded the first time. */
  bytes: number
  /** Estimated memory while it runs. */
  peakBytes: number
  /** Runs on the CPU but is meant for WebGPU (several times slower than real time without it). */
  prefersWebGpu?: boolean
  /** Maximum of the runtime's WebAssembly memory for this model, in 64 KiB pages (ONNX Runtime sessions we create). */
  memoryPages?: number
  group: CloneGroup
  license: string
  licenseNote?: string
  /** How the reference voice is taken in, in words. */
  reference: string
  /** Needs the words of the reference recording. None of the models offered here does. */
  needsTranscript: boolean
  /** Said next to the model: a watermark, or the lack of one. */
  watermark?: string
  /** Plays while it is still generating. */
  streams: boolean
  /** Takes only the first this many seconds of the reference. */
  maxReferenceSeconds?: number
  /** What it needs, when the default wording (WebGPU advised, or quick on the CPU) does not fit. */
  needs?: string
  source: string
  blurb: string
}

const POCKET_REPO = 'KevinAHM/pocket-tts-onnx'
const POCKET_REV = '58a6d00cf13d239b6748cb0769f35c580a8f606c'
const POCKET_REFERENCE =
  'The recording goes through Mimi, the model’s own audio encoder: one vector per 80 ms becomes the start of the model’s context. No transcript needed.'

const pocket = (
  id: string,
  label: string,
  bundle: string,
  language: string,
  big: boolean,
  extra: Partial<CloneModel> = {},
): CloneModel => ({
  id,
  label,
  family: 'pocket',
  repo: POCKET_REPO,
  revision: POCKET_REV,
  bundle,
  languages: [language],
  bytes: big ? 394_054_000 : 165_233_000,
  peakBytes: big ? 1_150e6 : 520e6,
  memoryPages: big ? 65_536 : 32_768,
  group: big ? 'Medium: a laptop' : 'Small: phones too',
  license: 'CC-BY-4.0',
  licenseNote: 'Kyutai Pocket TTS weights (CC BY 4.0); ONNX export by KevinAHM (code Apache-2.0).',
  reference: POCKET_REFERENCE,
  needsTranscript: false,
  streams: true,
  source: 'https://huggingface.co/kyutai/pocket-tts',
  blurb: big
    ? '24-layer version: about four times the work per frame of the small ones, slower than real time on most CPUs.'
    : 'Kyutai’s 100M-parameter model, 8-bit. Starts speaking within a second and plays as it goes.',
  ...extra,
})

const MOSS_REPO = 'OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX'
const MOSS_REV = 'f52645cb467506d8e18e746ddd59482685b74e58'
const MOSS_CODEC_REPO = 'OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX'
const MOSS_CODEC_REV = 'ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae'

/** A file of the MOSS-TTS-Nano bundle: the language model's repository or the codec's. */
export interface MossFile {
  name: string
  repo: string
  revision: string
  path: string
  bytes: number
}

const mossFile = (name: string, codec: boolean, path: string, bytes: number): MossFile => ({
  name,
  repo: codec ? MOSS_CODEC_REPO : MOSS_REPO,
  revision: codec ? MOSS_CODEC_REV : MOSS_REV,
  path,
  bytes,
})

/** What the page fetches for MOSS-TTS-Nano (sizes as the Hub lists them at the pinned revisions). */
export const MOSS_FILES: MossFile[] = [
  mossFile('manifest', false, 'browser_poc_manifest.json', 503_354),
  mossFile('meta', false, 'tts_browser_onnx_meta.json', 4_487),
  mossFile('tokenizer', false, 'tokenizer.model', 470_897),
  mossFile('prefill', false, 'moss_tts_prefill.onnx', 283_305),
  mossFile('decode', false, 'moss_tts_decode_step.onnx', 291_483),
  mossFile('frame', false, 'moss_tts_local_fixed_sampled_frame.onnx', 471_262),
  mossFile('globalData', false, 'moss_tts_global_shared.data', 440_813_568),
  mossFile('localData', false, 'moss_tts_local_shared.data', 229_678_080),
  mossFile('codecMeta', true, 'codec_browser_onnx_meta.json', 17_036),
  mossFile('encoder', true, 'moss_audio_tokenizer_encode.onnx', 815_775),
  mossFile('encoderData', true, 'moss_audio_tokenizer_encode.data', 44_507_136),
  mossFile('codecDecoder', true, 'moss_audio_tokenizer_decode_step.onnx', 351_400),
  mossFile('codecDecoderData', true, 'moss_audio_tokenizer_decode_shared.data', 44_198_912),
]

export const CLONE_MODELS: CloneModel[] = [
  pocket('pocket-en', 'Pocket TTS · English', 'english_2026-04', 'en', false),
  pocket('pocket-de', 'Pocket TTS · German', 'german', 'de', false),
  pocket('pocket-es', 'Pocket TTS · Spanish', 'spanish', 'es', false),
  pocket('pocket-it', 'Pocket TTS · Italian', 'italian', 'it', false),
  pocket('pocket-pt', 'Pocket TTS · Portuguese', 'portuguese', 'pt', false),
  pocket('pocket-fr-24l', 'Pocket TTS · French (24 layers)', 'french_24l', 'fr', true, {
    blurb:
      'French comes only as the 24-layer model: about four times the work per frame of the small ones, slower than real time on most CPUs.',
  }),
  {
    id: 'chatterbox-turbo',
    label: 'Chatterbox Turbo · English',
    family: 'chatterbox',
    repo: 'ResembleAI/chatterbox-turbo-ONNX',
    revision: 'd21799bd0354adb85e348b8a0442a8405110a2cf',
    dtype: { wasm: 'q4', webgpu: 'q4f16' },
    languages: ['en'],
    bytes: 717_700_000,
    peakBytes: 1_700e6,
    prefersWebGpu: true,
    group: 'Large: a desktop, WebGPU advised',
    license: 'MIT',
    reference:
      'One pass of its speech encoder: speech tokens of the recording (the prompt), a 192-dim speaker vector and mel features for the decoder. No transcript needed.',
    needsTranscript: false,
    watermark:
      'Resemble AI’s Python package adds its PerTh watermark to every output. This page runs the ONNX export, and the watermark is not applied in the browser: the audio carries no watermark.',
    streams: false,
    source: 'https://huggingface.co/ResembleAI/chatterbox-turbo',
    blurb: 'Resemble AI’s 350M-parameter Turbo, 4-bit. Natural and close to the voice; the whole clip comes at once.',
  },
  {
    id: 'chatterbox-multilingual',
    label: 'Chatterbox Multilingual · 23 languages',
    family: 'chatterbox',
    repo: 'BricksDisplay/chatterbox-multilingual-ONNX-q4',
    revision: '171d2d625bf424fd39847c10d4ebdd6612ea81f6',
    dtype: { wasm: 'fp32', webgpu: 'fp32' },
    languageTag: true,
    languages: [
      'ar',
      'da',
      'de',
      'el',
      'en',
      'es',
      'fi',
      'fr',
      'he',
      'hi',
      'it',
      'ja',
      'ko',
      'ms',
      'nl',
      'no',
      'pl',
      'pt',
      'ru',
      'sv',
      'sw',
      'tr',
      'zh',
    ],
    bytes: 828_000_000,
    peakBytes: 2_200e6,
    prefersWebGpu: true,
    group: 'Large: a desktop, WebGPU advised',
    license: 'MIT',
    licenseNote:
      'Resemble AI Chatterbox Multilingual (MIT); 4-bit single-file ONNX by ipsilondev (BricksDisplay), from onnx-community’s export.',
    reference:
      'One pass of its speech encoder: speech tokens of the recording (the prompt), a 192-dim speaker vector and mel features for the decoder. No transcript needed.',
    needsTranscript: false,
    watermark:
      'Resemble AI’s Python package adds its PerTh watermark to every output. This page runs the ONNX export, and the watermark is not applied in the browser: the audio carries no watermark.',
    streams: false,
    source: 'https://huggingface.co/ResembleAI/chatterbox',
    blurb:
      'Resemble AI’s 500M-parameter multilingual model, 4-bit: Hindi and 22 more languages. Very slow without WebGPU. Japanese, Chinese and Hebrew need text preparation this page does not do yet.',
  },
  {
    id: 'moss-tts-nano',
    label: 'MOSS-TTS-Nano 100M · 20 languages',
    family: 'moss',
    repo: MOSS_REPO,
    revision: MOSS_REV,
    languages: [
      'ar',
      'cs',
      'da',
      'de',
      'el',
      'en',
      'es',
      'fa',
      'fr',
      'he',
      'hu',
      'it',
      'ja',
      'ko',
      'pl',
      'pt',
      'ru',
      'sv',
      'tr',
      'zh',
    ],
    bytes: MOSS_FILES.reduce((a, f) => a + f.bytes, 0),
    // Measured in Chromium: 1.3 GB of WebAssembly memory once loaded, 2.5 GB after taking in a 12 s voice.
    peakBytes: 2_600e6,
    memoryPages: 65_536,
    group: 'Medium: a laptop',
    license: 'Apache-2.0',
    licenseNote:
      'MOSS-TTS-Nano and MOSS-Audio-Tokenizer-Nano by the OpenMOSS team and MOSI.AI (Apache-2.0), their official ONNX exports; this page’s driver is written from their Apache-2.0 reference code.',
    reference:
      'The recording goes through the model’s own audio codec (MOSS-Audio-Tokenizer-Nano, 48 kHz): 16 codes per 80 ms become the voice prompt. It takes the first 12 s (its encoder’s memory grows with the square of the length). No transcript needed.',
    needsTranscript: false,
    maxReferenceSeconds: 12,
    streams: true,
    needs:
      'a laptop or desktop with memory to spare (about 2.5 GB); on four CPU cores it takes 1.1 to 1.2 s per second of speech in Chromium, so long text may pause now and then',
    source: 'https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano',
    blurb:
      'OpenMOSS’s 100M-parameter model in full precision with its 48 kHz codec (played here at 24 kHz, like the other models): closer to the voice than Pocket TTS, and plays as it goes. Its own number-to-words step (WeTextProcessing, Python) has no browser version, so this page writes numbers out with uccharan first, in the languages uccharan reads (see Numbers).',
  },
]

export const CLONE_GROUPS: CloneGroup[] = ['Small: phones too', 'Medium: a laptop', 'Large: a desktop, WebGPU advised']

export const DEFAULT_CLONE_MODEL = CLONE_MODELS[0] as CloneModel

export function cloneModel(id: string | null | undefined): CloneModel {
  return CLONE_MODELS.find((m) => m.id === id) ?? DEFAULT_CLONE_MODEL
}

/** Download-list id of a cloning model. */
export const cloneDownloadId = (m: Pick<CloneModel, 'id'>) => `clone-${m.id}`

/** Pocket TTS files of a bundle, relative to the repository root. */
export function pocketFiles(m: CloneModel): { name: string; path: string }[] {
  const dir = `onnx/${m.bundle}/`
  return [
    ['bundle', 'bundle.json'],
    ['tokenizer', 'tokenizer.model'],
    ['bos', 'bos_before_voice.npy'],
    ['encoder', 'mimi_encoder.onnx'],
    ['textConditioner', 'text_conditioner.onnx'],
    ['main', 'flow_lm_main_int8.onnx'],
    ['flow', 'flow_lm_flow_int8.onnx'],
    ['decoder', 'mimi_decoder_int8.onnx'],
  ].map(([name, file]) => ({ name: name as string, path: dir + file }))
}

/** Chatterbox files transformers.js fetches for a dtype (used for the download list's keys). */
export function chatterboxFiles(m: CloneModel, dtype: string): string[] {
  const suffix = dtype === 'fp32' ? '' : `_${dtype}`
  const external = m.repo.startsWith('ResembleAI/')
  const graphs = ['embed_tokens', 'speech_encoder', 'language_model', 'conditional_decoder'].flatMap((g) => [
    `onnx/${g}${suffix}.onnx`,
    ...(external ? [`onnx/${g}${suffix}.onnx_data`] : []),
  ])
  return ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', ...graphs]
}

/** Languages by name, for the pickers. */
export const LANGUAGE_NAMES: Record<string, string> = {
  ar: 'Arabic',
  cs: 'Czech',
  da: 'Danish',
  de: 'German',
  el: 'Greek',
  en: 'English',
  es: 'Spanish',
  fa: 'Persian',
  fi: 'Finnish',
  fr: 'French',
  he: 'Hebrew',
  hi: 'Hindi',
  hu: 'Hungarian',
  it: 'Italian',
  ja: 'Japanese',
  ko: 'Korean',
  ms: 'Malay',
  nl: 'Dutch',
  no: 'Norwegian',
  pl: 'Polish',
  pt: 'Portuguese',
  ru: 'Russian',
  sv: 'Swedish',
  sw: 'Swahili',
  tr: 'Turkish',
  zh: 'Chinese',
}

/** Models from the brief that the page does not run, and why. */
export const NOT_OFFERED = [
  {
    label: 'NeuTTS Air (Neuphonic)',
    license: 'Apache-2.0',
    size: '3.0 GB (one fp32 file)',
    why: 'The only ONNX export is a single 3 GB fp32 graph: over what a browser can hand ONNX Runtime in one piece, and it also needs the NeuCodec encoder, which has no ONNX export.',
  },
  {
    label: 'OuteTTS 1.0 0.6B (OuteAI)',
    license: 'Apache-2.0',
    size: '580 MB to 2.4 GB plus the DAC codec',
    why: 'A cloned voice is a "speaker profile": word-aligned DAC codes and pitch and energy figures of the reference, made by Python tools (Whisper word timings, the DAC encoder) that have no browser port yet.',
  },
]
