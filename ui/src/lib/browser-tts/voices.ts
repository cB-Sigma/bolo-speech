import { isDemo } from '@/lib/api/config'
import type { NumberLanguage } from './numbers'
import { TINY_VOCA_BASE } from './tiny-voca'

/**
 * Voices the console can run in the browser.
 *
 * - SYSPIN voices (IISc, character input): prepared by ui/scripts/voices/build.py
 *   (same ids as ui/scripts/voices/voices.json) and published next to the
 *   live demo under `voices/`, 8-bit; the original 32-bit weights are also
 *   offered, straight from their Hugging Face copy.
 * - Piper voices (phonemes from uccharan): straight from rhasspy/piper-voices
 *   on Hugging Face at a pinned revision, as published (32-bit). Each keeps
 *   the licence of its training data, from its model card.
 * - Kokoro-82M (phonemes from uccharan): onnx-community's ONNX export, in
 *   8-, 16- or 32-bit, with the voice's style vector.
 */
export interface BrowserVoice {
  id: string
  kind: 'coqui' | 'piper' | 'kokoro' | 'tiny-voca'
  /** BCP-47 tag. */
  language: string
  /** Name of the language in its own script, and in English. */
  nativeName: string
  englishName: string
  gender: 'female' | 'male' | null
  /** Short label for pickers. */
  label: string
  /** Bytes fetched on first use (model and settings); the index served with the voices is preferred. */
  approxBytes: number
  /** Estimated peak memory while it speaks, bytes. */
  peakBytes: number
  /** A short natural sentence or two to start with. */
  sample: string
  /** Language used to spell out numbers; null when the phonemiser does it (or nobody does). */
  numbers: NumberLanguage | null
  /** In-browser recogniser for the round trip, when there is one for this language. */
  roundTrip: { model: string; language: string } | null
  /** Who made the voice, and under which terms. */
  credit: string
  license: string
  dataLicense: string
  source: string
  /** Honest notes shown next to the voice. */
  notes: string[]
  /** Text to phonemes, for voices that read phonemes. */
  phonemizer: 'uccharan' | 'dizione' | null
  /** uccharan's language for the phonemes (Piper and Kokoro voices). */
  phonemeLanguage?: string
  /** Weights as downloaded: '8-bit', '16-bit', '32-bit'. */
  precision: string
  /** The phonemes are not the ones the voice was trained with: said plainly next to it. */
  approximate?: string
  /** Where the files come from; `published` is the demo's own voices directory. */
  files:
    | { from: 'published' }
    | { from: 'remote'; model: string; config: string; configFormat: 'bolo' | 'piper' }
    | { from: 'kokoro'; model: string; style: string }
    | { from: 'tiny-voca'; model: string; config: string }
  /** Id under which its download is listed (Kokoro voices share one model per precision). */
  downloadId: string
  downloadLabel: string
}

interface Language {
  language: string
  nativeName: string
  englishName: string
  sample: string
  numbers: NumberLanguage | null
  roundTrip: BrowserVoice['roundTrip']
}

const EN_TRIP = { model: 'moonshine-tiny', language: 'en' }
const HI_TRIP = { model: 'whisper-tiny-hindi', language: 'hi' }
const L = (
  language: string,
  nativeName: string,
  englishName: string,
  sample: string,
  numbers: NumberLanguage | null = null,
  roundTrip: BrowserVoice['roundTrip'] = null,
): Language => ({ language, nativeName, englishName, sample, numbers, roundTrip })

/** Languages in picker order. */
const LANGUAGES: Record<string, Language> = {
  hi: L('hi', 'हिन्दी', 'Hindi', 'नमस्ते! आप कैसे हैं? आज मौसम बहुत अच्छा है।', 'hi', HI_TRIP),
  mai: L('mai', 'मैथिली', 'Maithili', 'प्रणाम! अहाँ केहन छी? आइ मौसम बड्ड नीक अछि।', 'hi'),
  bho: L('bho', 'भोजपुरी', 'Bhojpuri', 'प्रणाम! रउआ कइसन बानी? आज मौसम बहुत बढ़िया बा।', 'hi'),
  mag: L('mag', 'मगही', 'Magahi', 'प्रणाम! अपने कइसन ही? आज मौसम बड़ी बढ़िया हे।', 'hi'),
  hne: L('hne', 'छत्तीसगढ़ी', 'Chhattisgarhi', 'जय जोहार! तुमन कइसे हव? आज मौसम बड़ सुग्घर हे।', 'hi'),
  mr: L('mr', 'मराठी', 'Marathi', 'नमस्कार! तुम्ही कसे आहात? आज हवामान खूप छान आहे.', 'mr'),
  bn: L('bn', 'বাংলা', 'Bengali', 'নমস্কার! আপনি কেমন আছেন? আজ আবহাওয়া খুব ভালো।'),
  gu: L('gu', 'ગુજરાતી', 'Gujarati', 'નમસ્તે! તમે કેમ છો? આજે હવામાન ખૂબ સરસ છે.'),
  te: L('te', 'తెలుగు', 'Telugu', 'నమస్కారం! మీరు ఎలా ఉన్నారు? ఈ రోజు వాతావరణం చాలా బాగుంది.'),
  kn: L('kn', 'ಕನ್ನಡ', 'Kannada', 'ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.', 'kn'),
  ml: L('ml', 'മലയാളം', 'Malayalam', 'നമസ്കാരം! സുഖമാണോ? ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്.'),
  ne: L('ne', 'नेपाली', 'Nepali', 'नमस्ते! तपाईंलाई कस्तो छ? आज मौसम धेरै राम्रो छ।'),
  ur: L('ur', 'اردو', 'Urdu', 'السلام علیکم! آپ کیسے ہیں؟ آج موسم بہت اچھا ہے۔'),
  'en-IN': L(
    'en-IN',
    'English',
    'English (India)',
    'Hello! How are you today? The weather is lovely this morning.',
    'en',
    EN_TRIP,
  ),
  'en-US': L(
    'en-US',
    'English',
    'English (US)',
    'Hello! This voice runs entirely in your browser. Nothing you type is sent anywhere.',
    null,
    EN_TRIP,
  ),
  'en-GB': L('en-GB', 'English', 'English (UK)', 'Hello! This voice runs entirely in your browser.', null, EN_TRIP),
  'es-ES': L('es-ES', 'Español (España)', 'Spanish (Spain)', '¡Hola! ¿Qué tal? Hoy hace muy buen tiempo.'),
  'es-MX': L('es-MX', 'Español (México)', 'Spanish (Mexico)', '¡Hola! ¿Cómo estás? Hoy hace muy buen tiempo.'),
  fr: L('fr', 'Français', 'French', 'Bonjour ! Comment allez-vous ? Il fait très beau aujourd’hui.'),
  de: L('de', 'Deutsch', 'German', 'Hallo! Wie geht es dir? Heute ist das Wetter sehr schön.'),
  it: L('it', 'Italiano', 'Italian', 'Ciao! Come stai? Oggi il tempo è molto bello.'),
  'pt-BR': L('pt-BR', 'Português (Brasil)', 'Portuguese (Brazil)', 'Olá! Tudo bem? Hoje o tempo está ótimo.'),
  'pt-PT': L('pt-PT', 'Português (Portugal)', 'Portuguese (Portugal)', 'Olá! Como está? Hoje está um ótimo dia.'),
  nl: L('nl', 'Nederlands', 'Dutch', 'Hallo! Hoe gaat het? Het is vandaag mooi weer.'),
  'nl-BE': L('nl-BE', 'Nederlands (België)', 'Dutch (Belgium)', 'Hallo! Hoe gaat het? Het is vandaag mooi weer.'),
  pl: L('pl', 'Polski', 'Polish', 'Cześć! Jak się masz? Dzisiaj jest piękna pogoda.'),
  cs: L('cs', 'Čeština', 'Czech', 'Ahoj! Jak se máš? Dnes je krásné počasí.'),
  tr: L('tr', 'Türkçe', 'Turkish', 'Merhaba! Nasılsın? Bugün hava çok güzel.'),
  ro: L('ro', 'Română', 'Romanian', 'Bună! Ce mai faci? Astăzi vremea este foarte frumoasă.'),
  hu: L('hu', 'Magyar', 'Hungarian', 'Szia! Hogy vagy? Ma nagyon szép az idő.'),
  fi: L('fi', 'Suomi', 'Finnish', 'Hei! Mitä kuuluu? Tänään on todella kaunis sää.'),
  sv: L('sv', 'Svenska', 'Swedish', 'Hej! Hur mår du? Idag är det väldigt fint väder.'),
  nb: L('nb', 'Norsk bokmål', 'Norwegian', 'Hei! Hvordan har du det? I dag er det veldig fint vær.'),
  da: L('da', 'Dansk', 'Danish', 'Hej! Hvordan har du det? I dag er vejret rigtig dejligt.'),
  ca: L('ca', 'Català', 'Catalan', 'Hola! Com estàs? Avui fa molt bon temps.'),
}

const lang = (key: string): Language => {
  const l = LANGUAGES[key]
  if (!l) throw new Error(`unknown voice language ${key}`)
  return l
}

/** Peak memory estimate: the weights expanded to 32-bit plus the runtime and a long sentence's activations. */
const peakOf = (fp32Bytes: number) => Math.round(fp32Bytes * 1.6 + 350e6)

const SYSPIN_CREDIT = 'SYSPIN project, SPIRE Lab, Indian Institute of Science, Bengaluru'
const SYSPIN_SIZE = 36_530_000
const SYSPIN_FP32 = 114_220_000
const CLOSED_SET =
  'Reads a closed set of characters: anything outside it is skipped, so numbers are spelt out first and other scripts are left out.'
const HINDI_NUMBERS =
  'Numbers are read with their Hindi names, which is close but not always how a speaker would say them.'
const NO_NUMBERS = 'Numbers are not spelt out for this language yet: write them as words.'
const NO_DANDA = 'Its character set has no danda (।); full stops are used in its place.'

/** The Hugging Face copy of each SYSPIN voice's original (32-bit) export, at a pinned revision. */
const SYSPIN_HUB: Record<string, [string, string]> = {
  'syspin-mai-female': ['bolo-syspin-maithili-female', 'fb0fc6ae735ec754a77634e93f8792b28367d8ea'],
  'syspin-mai-male': ['bolo-syspin-maithili-male', 'fd04c59ef17c3236119f45f60582abaec5d4cdfe'],
  'syspin-bho-female': ['bolo-syspin-bhojpuri-female', '773cf5f69d1cce9fff75dc4500540f424a2950e8'],
  'syspin-bho-male': ['bolo-syspin-bhojpuri-male', 'bb7608eed52bdbf1f51a719317d37083bd93e4eb'],
  'syspin-hi-female': ['bolo-syspin-hindi-female', 'c48b729dfae118ebce7086d9695918edd02f8fa0'],
  'syspin-hi-male': ['bolo-syspin-hindi-male', '776013ac9d2a3adb5c330aa3adbc9af7fda5ad20'],
  'syspin-bn-female': ['bolo-syspin-bengali-female', 'c344ee1197e62d6be373cb9e14634a94931bdf12'],
  'syspin-bn-male': ['bolo-syspin-bengali-male', '8e898d53ded194dfa90d357de0b8c8e3c47d4905'],
  'syspin-mr-female': ['bolo-syspin-marathi-female', '78c765d9d729b20a1297e53ed8f9f658062f0925'],
  'syspin-mr-male': ['bolo-syspin-marathi-male', 'daee212b8f9d7bdf9bf8c39f348dacad36e2faaa'],
  'syspin-te-female': ['bolo-syspin-telugu-female', '24682cdcc245b8930f177cc57108baca7103e0fb'],
  'syspin-te-male': ['bolo-syspin-telugu-male', '6650e4cb5e553e80af13e5b079ffaf86c30ac19b'],
  'syspin-kn-female': ['bolo-syspin-kannada-female', 'a35d24ac60139a2051745b4899a5335aabab6fab'],
  'syspin-kn-male': ['bolo-syspin-kannada-male', 'f5e28d5e8340d7cf7bd05264dd34b33ca0bdb9f6'],
  'syspin-gu-female': ['bolo-syspin-gujarati-female', '4d12dcb2dcf397ec033dfc1f94ce246ac57833dd'],
  'syspin-gu-male': ['bolo-syspin-gujarati-male', '8b07836d2b27b01d45457a15205c11cc51e49cd2'],
  'syspin-mag-female': ['bolo-syspin-magahi-female', '89a217a0090dad01c584488b8650d0660b126443'],
  'syspin-mag-male': ['bolo-syspin-magahi-male', '50eeb4a7637d20a0681ac2ac84d06e9cfbf51e2e'],
  'syspin-hne-female': ['bolo-syspin-chhattisgarhi-female', '7465910e8ac2a2edcd6a227fd1ac1f3a5e123ec6'],
  'syspin-hne-male': ['bolo-syspin-chhattisgarhi-male', 'db5ab9743979328d262f724b966a68ec40d8fa2e'],
  'syspin-en-female': ['bolo-syspin-english-female', '6718ed2eb53b763b0a30d76a6895f15770c36928'],
  'syspin-en-male': ['bolo-syspin-english-male', '5a3593d507688effe6f8d70f5d118dd20fc18082'],
}

const SYSPIN_REPO: Record<string, string> = {
  hi: 'Hindi',
  mai: 'Maithili',
  bho: 'Bhojpuri',
  mag: 'Magahi',
  hne: 'Chhattisgarhi',
  mr: 'Marathi',
  bn: 'Bengali',
  gu: 'Gujarati',
  te: 'Telugu',
  kn: 'Kannada',
  'en-IN': 'English',
}

/** A SYSPIN voice in two forms: the demo's 8-bit build and the original 32-bit export. */
function syspin(code: string, gender: 'female' | 'male', notes: string[] = []): BrowserVoice[] {
  const l = lang(code)
  const short = code === 'en-IN' ? 'en' : code
  const id = `syspin-${short}-${gender}`
  const repo = `vits_${SYSPIN_REPO[code]}_${gender === 'female' ? 'Female' : 'Male'}`
  const name = code === 'en-IN' ? 'Indian English' : l.englishName
  const numberNotes = l.numbers === 'hi' && code !== 'hi' ? [HINDI_NUMBERS] : l.numbers ? [] : [NO_NUMBERS]
  const common = {
    kind: 'coqui' as const,
    language: l.language,
    nativeName: l.nativeName,
    englishName: l.englishName,
    gender,
    sample: l.sample,
    numbers: l.numbers,
    roundTrip: l.roundTrip,
    credit: SYSPIN_CREDIT,
    license: 'MIT',
    dataLicense: 'CC-BY-4.0 (SYSPIN corpus)',
    source: `https://huggingface.co/SYSPIN/${repo}`,
    notes: [CLOSED_SET, ...numberNotes, ...notes],
    phonemizer: null,
  }
  const [hub, revision] = SYSPIN_HUB[id] ?? ['', 'main']
  return [
    {
      ...common,
      id,
      label: `SYSPIN ${name}, ${gender}`,
      approxBytes: SYSPIN_SIZE,
      peakBytes: peakOf(SYSPIN_FP32),
      precision: '8-bit',
      files: { from: 'published' },
      downloadId: id,
      downloadLabel: `SYSPIN ${name}, ${gender}`,
    },
    {
      ...common,
      id: `${id}-fp32`,
      label: `SYSPIN ${name}, ${gender} (original 32-bit)`,
      approxBytes: SYSPIN_FP32,
      peakBytes: peakOf(SYSPIN_FP32),
      precision: '32-bit',
      notes: [
        ...common.notes,
        'The original weights, unchanged: three times the download of the 8-bit build, which differs from it by less than anyone has been able to hear in our checks.',
      ],
      files: {
        from: 'remote',
        model: `https://huggingface.co/codebucketsolutions/${hub}/resolve/${revision}/model.onnx`,
        config: `${id}/voice.json`,
        configFormat: 'bolo',
      },
      downloadId: `${id}-fp32`,
      downloadLabel: `SYSPIN ${name}, ${gender} (32-bit)`,
    },
  ]
}

const PIPER_REVISION = '6b9cfaacb4fb2e02a9621227e3c1a48bec9710c3'
const PIPER_BASE = `https://huggingface.co/rhasspy/piper-voices/resolve/${PIPER_REVISION}/`
/** espeak voice names in Piper configs that uccharan reads under another name, and how close that is. */
const PIPER_LANGUAGE: Record<string, [string, string | undefined]> = {
  'es-419': ['es', 'Latin American Spanish voice given Castilian (Spain) phonemes from uccharan: close, not exact.'],
}

function piper(
  key: string,
  dir: string,
  code: string,
  gender: 'female' | 'male' | null,
  dataLicense: string,
  bytes: number,
  espeak: string,
  rate: number,
): BrowserVoice {
  const l = lang(code)
  const [, name = key, quality = 'medium'] = key.split('-')
  const pretty = name.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
  const [phonemeLanguage, approximate] = PIPER_LANGUAGE[espeak] ?? [espeak, undefined]
  const id = `piper-${key.toLowerCase().replace(/_/g, '-')}`
  const file = `${PIPER_BASE}${encodeURI(dir)}/${encodeURIComponent(key)}.onnx`
  return {
    id,
    kind: 'piper',
    language: l.language,
    nativeName: l.nativeName,
    englishName: l.englishName,
    gender,
    label: `Piper ${pretty} (${quality.replace('_', '-')})`,
    approxBytes: bytes,
    peakBytes: peakOf(bytes),
    sample: l.sample,
    numbers: null,
    roundTrip: l.roundTrip,
    credit: `Piper voice "${name}" from the Rhasspy / Open Home Foundation piper-voices collection; phonemes by uccharan`,
    license: 'MIT (piper-voices)',
    dataLicense,
    source: `https://huggingface.co/rhasspy/piper-voices/tree/${PIPER_REVISION}/${dir}`,
    notes: [
      'Piper voices read phonemes. They come from uccharan, a permissively licensed phonemiser that runs in the page; no eSpeak NG is involved, so a word can come out slightly differently from how the voice was trained.',
      `${rate / 1000} kHz, Piper's "${quality.replace('_', '-')}" quality tier, downloaded as published (32-bit).`,
      ...(approximate ? [approximate] : []),
    ],
    phonemizer: 'uccharan',
    phonemeLanguage,
    precision: '32-bit',
    approximate,
    files: { from: 'remote', model: file, config: `${file}.json`, configFormat: 'piper' },
    downloadId: id,
    downloadLabel: `Piper ${pretty}, ${l.englishName}`,
  }
}

const KOKORO_REVISION = '1939ad2a8e416c0acfeecc08a694d14ef25f2231'
const KOKORO_BASE = `https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/${KOKORO_REVISION}/`
/** Kokoro's weights in each precision: file and size. */
interface KokoroPrecision {
  file: string
  bytes: number
  label: string
  note?: string
}

export const KOKORO_PRECISIONS: Record<string, KokoroPrecision> = {
  q8: { file: 'onnx/model_quantized.onnx', bytes: 92_361_116, label: '8-bit' },
  fp16: {
    file: 'onnx/model_fp16.onnx',
    bytes: 163_234_740,
    label: '16-bit',
    note: 'Computed in 16 bits on the CPU, which not every browser does quickly.',
  },
  fp32: { file: 'onnx/model.onnx', bytes: 325_532_232, label: '32-bit (original)' },
}

const KOKORO_APPROX =
  'Approximate: Kokoro learnt this language from eSpeak NG phonemes; uccharan writes them in the same style but not identically, so some words sound off.'

function kokoro(
  style: string,
  code: string,
  gender: 'female' | 'male',
  approximate = true,
  precision = 'q8',
): BrowserVoice {
  const l = lang(code)
  const p = (KOKORO_PRECISIONS[precision] ?? KOKORO_PRECISIONS.q8) as KokoroPrecision
  const name = style.split('_')[1] ?? style
  const phonemeLanguage =
    { 'en-US': 'en-us', 'en-GB': 'en-gb', hi: 'hi', 'es-ES': 'es', fr: 'fr', it: 'it', 'pt-BR': 'pt-br' }[code] ??
    'en-us'
  const id = `kokoro-${style.replace('_', '-')}${precision === 'q8' ? '' : `@${precision}`}`
  return {
    id,
    kind: 'kokoro',
    language: l.language,
    nativeName: l.nativeName,
    englishName: l.englishName,
    gender,
    label: `Kokoro ${name.replace(/^\w/, (c) => c.toUpperCase())}`,
    approxBytes: p.bytes + 524_288,
    peakBytes: Math.round(325_532_232 * 1.5 + 300e6),
    sample: l.sample,
    numbers: null,
    roundTrip: l.roundTrip,
    credit: 'Kokoro-82M by hexgrad (ONNX export by onnx-community); phonemes by uccharan',
    license: 'Apache-2.0',
    dataLicense: 'Apache-2.0 (Kokoro-82M; trained on permissively licensed and synthetic audio)',
    source: 'https://huggingface.co/hexgrad/Kokoro-82M',
    notes: [
      approximate
        ? KOKORO_APPROX
        : 'English phonemes come from uccharan with misaki’s dictionaries, the front end Kokoro was trained with.',
      `${p.label} weights (${Math.round(p.bytes / 1e6)} MB), shared by every Kokoro voice of that precision. 24 kHz.`,
      ...(p.note ? [p.note] : []),
    ],
    phonemizer: 'uccharan',
    phonemeLanguage,
    precision: p.label,
    approximate: approximate ? KOKORO_APPROX : undefined,
    files: { from: 'kokoro', model: `${KOKORO_BASE}${p.file}`, style: `${KOKORO_BASE}voices/${style}.bin` },
    downloadId: `kokoro-82m@${precision}`,
    downloadLabel: `Kokoro-82M (${p.label})`,
  }
}

/** Kokoro voices offered, per language (style file, language, gender, whether its phonemes are approximate). */
const KOKORO_VOICES: [string, string, 'female' | 'male', boolean][] = [
  ['af_heart', 'en-US', 'female', false],
  ['af_bella', 'en-US', 'female', false],
  ['am_michael', 'en-US', 'male', false],
  ['bf_emma', 'en-GB', 'female', false],
  ['bm_george', 'en-GB', 'male', false],
  ['hf_alpha', 'hi', 'female', true],
  ['hm_omega', 'hi', 'male', true],
  ['ef_dora', 'es-ES', 'female', true],
  ['em_alex', 'es-ES', 'male', true],
  ['ff_siwis', 'fr', 'female', true],
  ['if_sara', 'it', 'female', true],
  ['im_nicola', 'it', 'male', true],
  ['pf_dora', 'pt-BR', 'female', true],
  ['pm_alex', 'pt-BR', 'male', true],
]

function tinyVoca(speaker: 'F2' | 'M2'): BrowserVoice {
  const l = lang('en-US')
  return {
    id: `tiny-voca-${speaker.toLowerCase()}`,
    kind: 'tiny-voca',
    language: l.language,
    nativeName: l.nativeName,
    englishName: l.englishName,
    gender: speaker === 'F2' ? 'female' : 'male',
    label: `tiny-voca ${speaker}`,
    approxBytes: 21_386_372 + 5_400_000,
    peakBytes: 400e6,
    sample: 'Hello! This is tiny voca, a small voice that runs in your browser. The meeting moved to 3:30 pm.',
    numbers: null,
    roundTrip: l.roundTrip,
    credit: 'tiny-voca by Muhammed Nazeem (n4ze3m), the authors’ ONNX export; phonemes by their dizione under Pyodide',
    license: 'Apache-2.0',
    dataLicense: 'Apache-2.0 (trained from scratch on about 18 hours of synthetic speech); dizione Apache-2.0',
    source: 'https://huggingface.co/n4ze3m/tiny-voca',
    notes: [
      'A 4.5 M-parameter English voice (21 MB), 24 kHz. Numbers, dates, money and units are read out by its phonemiser.',
      'The phonemiser, dizione, is Python: it runs under Pyodide (CPython on WebAssembly), about 15 MB fetched once from jsDelivr.',
    ],
    phonemizer: 'dizione',
    precision: '32-bit',
    files: { from: 'tiny-voca', model: `${TINY_VOCA_BASE}onnx/model.onnx`, config: `${TINY_VOCA_BASE}config.json` },
    downloadId: 'tiny-voca',
    downloadLabel: 'tiny-voca (and its phonemiser)',
  }
}

export const BROWSER_VOICES: BrowserVoice[] = [
  ...syspin('hi', 'female'),
  ...syspin('hi', 'male'),
  ...syspin('mai', 'male'),
  ...syspin('mai', 'female'),
  ...syspin('bho', 'female', [NO_DANDA]),
  ...syspin('bho', 'male', [NO_DANDA]),
  ...syspin('mag', 'female'),
  ...syspin('mag', 'male'),
  ...syspin('hne', 'female'),
  ...syspin('hne', 'male'),
  ...syspin('mr', 'female'),
  ...syspin('mr', 'male'),
  ...syspin('bn', 'female'),
  ...syspin('bn', 'male'),
  ...syspin('gu', 'female'),
  ...syspin('gu', 'male'),
  ...syspin('te', 'female'),
  ...syspin('te', 'male'),
  ...syspin('kn', 'female'),
  ...syspin('kn', 'male'),
  ...syspin('en-IN', 'female', [
    'Reads letters, not phonemes: spelling is pronounced the way the voice learnt it, so unusual words and names can come out wrong.',
  ]),
  ...syspin('en-IN', 'male', [
    'Reads letters, not phonemes: spelling is pronounced the way the voice learnt it, so unusual words and names can come out wrong.',
  ]),
  {
    ...piper(
      'en_US-amy-low',
      'en/en_US/amy/low',
      'en-US',
      'female',
      'Amy dataset from Mycroft mimic3-voices; its licence is not stated per voice',
      27_080_000,
      'en-us',
      16000,
    ),
    label: 'Piper Amy (low), US English',
    precision: '8-bit',
    notes: [
      'Piper voices read phonemes. They come from uccharan, a permissively licensed phonemiser that runs in the page; no eSpeak NG is involved.',
      '16 kHz, the "low" quality tier: small and quick rather than polished. Shrunk to 8 bits by bolo and published with the demo.',
    ],
    files: { from: 'published' },
  },
  piper(
    'en_US-lessac-medium',
    'en/en_US/lessac/medium',
    'en-US',
    'female',
    'Lessac data: research and non-commercial use (Blizzard 2013 licence)',
    63206179,
    'en-us',
    22050,
  ),
  piper(
    'en_US-ryan-medium',
    'en/en_US/ryan/medium',
    'en-US',
    'male',
    'CC-BY-NC-SA-4.0 (non-commercial)',
    63227450,
    'en-us',
    22050,
  ),
  piper('en_GB-alba-medium', 'en/en_GB/alba/medium', 'en-GB', 'female', 'CC-BY-4.0', 63227455, 'en-gb-x-rp', 22050),
  piper(
    'en_GB-northern_english_male-medium',
    'en/en_GB/northern_english_male/medium',
    'en-GB',
    'male',
    'CC-BY-SA-4.0',
    63206141,
    'en-gb-x-rp',
    22050,
  ),
  piper(
    'hi_IN-priyamvada-medium',
    'hi/hi_IN/priyamvada/medium',
    'hi',
    'female',
    'CC-BY-NC-SA-4.0 (non-commercial)',
    63521023,
    'hi',
    22050,
  ),
  piper(
    'hi_IN-rohan-medium',
    'hi/hi_IN/rohan/medium',
    'hi',
    'male',
    'IIT Madras Indic TTS licence (see its terms)',
    62955085,
    'hi',
    22050,
  ),
  piper(
    'hi_IN-pratham-medium',
    'hi/hi_IN/pratham/medium',
    'hi',
    'male',
    'CC-BY-NC-SA-4.0 (non-commercial)',
    63521020,
    'hi',
    22050,
  ),
  piper('ne_NP-chitwan-medium', 'ne/ne_NP/chitwan/medium', 'ne', null, 'CC0', 63227452, 'ne', 22050),
  piper('te_IN-padmavathi-medium', 'te/te_IN/padmavathi/medium', 'te', 'female', 'CC-BY-4.0', 63521024, 'te', 22050),
  piper('te_IN-venkatesh-medium', 'te/te_IN/venkatesh/medium', 'te', 'male', 'CC-BY-4.0', 63521023, 'te', 22050),
  piper(
    'te_IN-maya-medium',
    'te/te_IN/maya/medium',
    'te',
    'female',
    'IIT Madras Indic TTS licence (see its terms)',
    62955084,
    'te',
    22050,
  ),
  piper('ml_IN-meera-medium', 'ml/ml_IN/meera/medium', 'ml', 'female', 'see the model card', 62955089, 'ml', 22050),
  piper('ml_IN-arjun-medium', 'ml/ml_IN/arjun/medium', 'ml', 'male', 'see the model card', 62955088, 'ml', 22050),
  piper('ur_PK-aegis_female-medium', 'ur/ur_PK/aegis_female/medium', 'ur', 'female', 'MIT', 63520825, 'ur', 22050),
  piper('ur_PK-fasih-medium', 'ur/ur_PK/fasih/medium', 'ur', 'male', 'MIT', 64835260, 'ur', 22050),
  piper('es_ES-davefx-medium', 'es/es_ES/davefx/medium', 'es-ES', 'male', 'CC0', 63227442, 'es', 22050),
  piper('es_MX-ald-medium', 'es/es_MX/ald/medium', 'es-MX', 'male', 'Unlicense', 63206172, 'es-419', 22050),
  piper('de_DE-thorsten-medium', 'de/de_DE/thorsten/medium', 'de', 'male', 'CC0', 63227444, 'de', 22050),
  piper('de_DE-kerstin-medium', 'de/de_DE/kerstin/medium', 'de', 'female', 'CC0', 63227443, 'de', 22050),
  piper('nl_NL-alex-medium', 'nl/nl_NL/alex/medium', 'nl', 'male', 'CC0', 63227446, 'nl', 22050),
  piper('nl_BE-nathalie-medium', 'nl/nl_BE/nathalie/medium', 'nl-BE', 'female', 'CC0', 63227446, 'nl', 22050),
  piper('fr_FR-siwis-medium', 'fr/fr_FR/siwis/medium', 'fr', 'female', 'CC-BY-4.0', 63227442, 'fr', 22050),
  piper('fr_FR-tom-medium', 'fr/fr_FR/tom/medium', 'fr', 'male', 'AGPL-3.0', 63515997, 'fr', 44100),
  piper('it_IT-paola-medium', 'it/it_IT/paola/medium', 'it', 'female', 'CC0', 63227441, 'it', 22050),
  piper('it_IT-riccardo-x_low', 'it/it_IT/riccardo/x_low', 'it', 'male', 'see the model card', 28134952, 'it', 16000),
  piper('pt_BR-faber-medium', 'pt/pt_BR/faber/medium', 'pt-BR', 'male', 'CC0', 63227450, 'pt-br', 22050),
  piper('pt_BR-cadu-medium', 'pt/pt_BR/cadu/medium', 'pt-BR', 'male', 'CC0', 63227449, 'pt-br', 22050),
  piper('pt_PT-tugão-medium', 'pt/pt_PT/tugão/medium', 'pt-PT', 'male', 'CC0', 63227450, 'pt', 22050),
  piper('pl_PL-gosia-medium', 'pl/pl_PL/gosia/medium', 'pl', 'female', 'CC0', 63206108, 'pl', 22050),
  piper('pl_PL-darkman-medium', 'pl/pl_PL/darkman/medium', 'pl', 'male', 'CC0', 63227441, 'pl', 22050),
  piper(
    'tr_TR-dfki-medium',
    'tr/tr_TR/dfki/medium',
    'tr',
    'male',
    'CC-BY-NC-SA-4.0 (non-commercial)',
    63206254,
    'tr',
    22050,
  ),
  piper('cs_CZ-jirka-medium', 'cs/cs_CZ/jirka/medium', 'cs', 'male', 'CC0', 63227449, 'cs', 22050),
  piper('cs_CZ-kasandra-medium', 'cs/cs_CZ/kasandra/medium', 'cs', 'female', 'CC-BY-4.0', 63514270, 'cs', 22050),
  piper('sv_SE-lisa-medium', 'sv/sv_SE/lisa/medium', 'sv', 'female', 'see the model card', 63518277, 'sv', 22050),
  piper('sv_SE-nst-medium', 'sv/sv_SE/nst/medium', 'sv', null, 'CC0', 63108683, 'sv', 22050),
  piper('ca_ES-upc_ona-medium', 'ca/ca_ES/upc_ona/medium', 'ca', 'female', 'CC-BY-SA-3.0-ES', 63206169, 'ca', 22050),
  piper('ro_RO-mihai-medium', 'ro/ro_RO/mihai/medium', 'ro', 'male', 'CC0', 63227444, 'ro', 22050),
  piper('hu_HU-anna-medium', 'hu/hu_HU/anna/medium', 'hu', 'female', 'CC0', 63227442, 'hu', 22050),
  piper('hu_HU-imre-medium', 'hu/hu_HU/imre/medium', 'hu', 'male', 'CC0', 63227442, 'hu', 22050),
  piper('fi_FI-harri-medium', 'fi/fi_FI/harri/medium', 'fi', 'male', 'CC0', 63227440, 'fi', 22050),
  piper('no_NO-talesyntese-medium', 'no/no_NO/talesyntese/medium', 'nb', null, 'CC0', 63227447, 'nb', 22050),
  piper('da_DK-talesyntese-medium', 'da/da_DK/talesyntese/medium', 'da', null, 'CC0', 63206172, 'da', 22050),
  ...KOKORO_VOICES.map(([style, code, gender, approx]) => kokoro(style, code, gender, approx)),
  tinyVoca('F2'),
  tinyVoca('M2'),
]

export const DEFAULT_BROWSER_VOICE = BROWSER_VOICES[0] as BrowserVoice

/**
 * The voice for an id. Kokoro voices take a precision after `@`
 * (`kokoro-af-heart@fp16`); without one they are 8-bit.
 */
export function browserVoice(id: string | null | undefined): BrowserVoice {
  const found = BROWSER_VOICES.find((v) => v.id === id)
  if (found) return found
  const [base, precision] = (id ?? '').split('@')
  const k = KOKORO_VOICES.find(([style]) => `kokoro-${style.replace('_', '-')}` === base)
  if (k && precision && KOKORO_PRECISIONS[precision]) return kokoro(k[0], k[1], k[2], k[3], precision)
  return DEFAULT_BROWSER_VOICE
}

/** The voice's id without a precision suffix (what the picker lists). */
export function baseVoiceId(id: string): string {
  return id.split('@')[0] ?? id
}

/** Languages in picker order, each with its voices. */
export function voicesByLanguage(): {
  language: string
  nativeName: string
  englishName: string
  voices: BrowserVoice[]
}[] {
  const groups = new Map<
    string,
    { language: string; nativeName: string; englishName: string; voices: BrowserVoice[] }
  >()
  const order = Object.values(LANGUAGES).map((l) => l.englishName)
  for (const v of [...BROWSER_VOICES].sort((a, b) => order.indexOf(a.englishName) - order.indexOf(b.englishName))) {
    const key = v.englishName
    if (!groups.has(key))
      groups.set(key, { language: v.language, nativeName: v.nativeName, englishName: v.englishName, voices: [] })
    groups.get(key)?.voices.push(v)
  }
  return [...groups.values()]
}

/** "मैथिली · Maithili"; English names alone. */
export function languageLabel(v: Pick<BrowserVoice, 'nativeName' | 'englishName'>): string {
  return v.nativeName === v.englishName || v.englishName.startsWith(v.nativeName)
    ? v.englishName
    : `${v.nativeName} · ${v.englishName}`
}

export interface BrowserTtsConfig {
  /** Directory that serves `index.json` and `<voice>/…` (default: next to the demo, or the public demo). */
  voicesBase?: string
  /** Origin serving Hugging Face files in Hub layout, in place of huggingface.co (mirrors, tests). */
  hubHost?: string
}

/** A Hugging Face URL, moved to the configured mirror when there is one. */
export function hubUrl(url: string): string {
  const host = readBrowserTtsConfig().hubHost
  return host ? url.replace(/^https:\/\/huggingface\.co\//, host.replace(/\/?$/, '/')) : url
}

const KEY = 'bolo.browserTts'

/** Public copy of the voices, used by consoles that do not ship them (the one served by `bolo serve`). */
export const PUBLIC_VOICES = 'https://cb-sigma.github.io/bolo-speech/voices/'

export function readBrowserTtsConfig(): BrowserTtsConfig {
  try {
    const raw = globalThis.localStorage?.getItem(KEY)
    return raw ? (JSON.parse(raw) as BrowserTtsConfig) : {}
  } catch {
    return {}
  }
}

/** Absolute URL of the voices directory, ending in a slash. */
export function voicesBase(): string {
  const configured = readBrowserTtsConfig().voicesBase
  const base = configured ?? (isDemo ? './voices/' : PUBLIC_VOICES)
  return new URL(base.replace(/\/?$/, '/'), globalThis.location?.href ?? 'http://localhost/').href
}

/** One voice's entry in the published index.json (written by build.py). */
export interface VoiceIndexEntry {
  id: string
  bytes: number
  files: Record<string, { bytes: number; sha256: string }>
}

export interface VoiceIndex {
  format: number
  voices: VoiceIndexEntry[]
  missing?: string[]
}
