import { isDemo } from '@/lib/api/config'
import type { NumberLanguage } from './numbers'

/**
 * Voices the console can run in the browser. The files are prepared by
 * ui/scripts/voices/build.py (same ids as ui/scripts/voices/voices.json) and
 * published next to the live demo under `voices/`.
 */
export interface BrowserVoice {
  id: string
  kind: 'coqui' | 'piper'
  /** BCP-47 tag. */
  language: string
  /** Name of the language in its own script, and in English. */
  nativeName: string
  englishName: string
  gender: 'female' | 'male'
  /** Short label for pickers. */
  label: string
  /** Bytes fetched on first use (model and settings) as published; the index served with the voices is preferred. */
  approxBytes: number
  /** A short natural sentence or two to start with. */
  sample: string
  /** Language used to spell out numbers; null when the phonemiser does it. */
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
  phonemizer: 'uccharan' | null
}

const SYSPIN_CREDIT = 'SYSPIN project, SPIRE Lab, Indian Institute of Science, Bengaluru'
const SYSPIN_SIZE = 36_530_000
const CLOSED_SET =
  'Reads a closed set of characters: anything outside it is skipped, so numbers are spelt out first and other scripts are left out.'

function syspin(
  id: string,
  language: string,
  nativeName: string,
  englishName: string,
  gender: 'female' | 'male',
  repo: string,
  sample: string,
  numbers: NumberLanguage,
  roundTrip: BrowserVoice['roundTrip'],
  notes: string[] = [],
): BrowserVoice {
  return {
    id,
    kind: 'coqui',
    language,
    nativeName,
    englishName,
    gender,
    label: `SYSPIN ${englishName}, ${gender}`,
    approxBytes: SYSPIN_SIZE,
    sample,
    numbers,
    roundTrip,
    credit: SYSPIN_CREDIT,
    license: 'MIT',
    dataLicense: 'CC-BY-4.0 (SYSPIN corpus)',
    source: `https://huggingface.co/SYSPIN/${repo}`,
    notes: [CLOSED_SET, ...notes],
    phonemizer: null,
  }
}

const HINDI_NUMBERS =
  'Numbers are read with their Hindi names, which is close but not always how a speaker would say them.'

export const BROWSER_VOICES: BrowserVoice[] = [
  syspin(
    'syspin-hi-female',
    'hi',
    'हिन्दी',
    'Hindi',
    'female',
    'vits_Hindi_Female',
    'नमस्ते! आप कैसे हैं? आज मौसम बहुत अच्छा है।',
    'hi',
    { model: 'whisper-tiny-hindi', language: 'hi' },
  ),
  syspin(
    'syspin-mai-female',
    'mai',
    'मैथिली',
    'Maithili',
    'female',
    'vits_Maithili_Female',
    'प्रणाम! अहाँ केहन छी? आइ मौसम बड्ड नीक अछि।',
    'hi',
    null,
    [HINDI_NUMBERS],
  ),
  syspin(
    'syspin-mai-male',
    'mai',
    'मैथिली',
    'Maithili',
    'male',
    'vits_Maithili_Male',
    'प्रणाम! अहाँ केहन छी? आइ मौसम बड्ड नीक अछि।',
    'hi',
    null,
    [HINDI_NUMBERS],
  ),
  syspin(
    'syspin-bho-female',
    'bho',
    'भोजपुरी',
    'Bhojpuri',
    'female',
    'vits_Bhojpuri_Female',
    'प्रणाम! रउआ कइसन बानी? आज मौसम बहुत बढ़िया बा।',
    'hi',
    null,
    [HINDI_NUMBERS, 'Its character set has no danda (।); full stops are used in its place.'],
  ),
  syspin(
    'syspin-bho-male',
    'bho',
    'भोजपुरी',
    'Bhojpuri',
    'male',
    'vits_Bhojpuri_Male',
    'प्रणाम! रउआ कइसन बानी? आज मौसम बहुत बढ़िया बा।',
    'hi',
    null,
    [HINDI_NUMBERS, 'Its character set has no danda (।); full stops are used in its place.'],
  ),
  syspin(
    'syspin-mr-female',
    'mr',
    'मराठी',
    'Marathi',
    'female',
    'vits_Marathi_Female',
    'नमस्कार! तुम्ही कसे आहात? आज हवामान खूप छान आहे.',
    'mr',
    null,
  ),
  syspin(
    'syspin-kn-female',
    'kn',
    'ಕನ್ನಡ',
    'Kannada',
    'female',
    'vits_Kannada_Female',
    'ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.',
    'kn',
    null,
  ),
  {
    ...syspin(
      'syspin-en-female',
      'en-IN',
      'English',
      'English (India)',
      'female',
      'vits_English_Female',
      'Hello! How are you today? The weather is lovely this morning.',
      'en',
      { model: 'moonshine-tiny', language: 'en' },
    ),
    label: 'SYSPIN Indian English, female',
    notes: [
      'Reads letters, not phonemes: spelling is pronounced the way the voice learnt it, so unusual words and names can come out wrong.',
      CLOSED_SET,
    ],
  },
  {
    id: 'piper-en-us-amy-low',
    kind: 'piper',
    language: 'en-US',
    nativeName: 'English',
    englishName: 'English (US)',
    gender: 'female',
    label: 'Piper Amy (low), US English',
    approxBytes: 27_080_000,
    sample: 'Hello! This voice runs entirely in your browser. Nothing you type is sent anywhere.',
    numbers: null,
    roundTrip: { model: 'moonshine-tiny', language: 'en' },
    credit: 'Piper voice "amy" by the Rhasspy/Open Home Foundation project; phonemes by uccharan',
    license: 'MIT (piper-voices)',
    dataLicense: 'Amy dataset from Mycroft mimic3-voices; its licence is not stated per voice',
    source: 'https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/low',
    notes: [
      'Piper voices read phonemes. They come from uccharan, a permissively licensed phonemiser that runs in the page; no eSpeak NG is involved.',
      '16 kHz, the "low" quality tier: small and quick rather than polished.',
    ],
    phonemizer: 'uccharan',
  },
]

export const DEFAULT_BROWSER_VOICE = BROWSER_VOICES[0] as BrowserVoice

export function browserVoice(id: string | null | undefined): BrowserVoice {
  return BROWSER_VOICES.find((v) => v.id === id) ?? DEFAULT_BROWSER_VOICE
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
  for (const v of BROWSER_VOICES) {
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
}

const KEY = 'bolo.browserTts'

/** Public copy of the voices, used by consoles that do not ship them (the one served by `bolo serve`). */
export const PUBLIC_VOICES = 'https://cb-sigma.github.io/bolo/voices/'

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
