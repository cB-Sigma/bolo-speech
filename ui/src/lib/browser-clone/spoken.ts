/**
 * The spoken form of the text a cloning model is given. The models read text as written, so "10:30", "25 €"
 * or "3. Mai" reach them as digits and signs they say badly or skip (MOSS-TTS-Nano's own number-to-words step,
 * WeTextProcessing, is a Python package). Before generating, the page runs the text through uccharan's
 * normaliser for the chosen language, which writes numbers, dates, times, money, units, ordinals and common
 * abbreviations out in words, in the languages uccharan has; the others are left as they are.
 */
import { LANGUAGE_NAMES } from './models'

/** Page language → the uccharan language whose normaliser writes it out. */
export const NORMALISER_LANGUAGE: Readonly<Record<string, string>> = {
  ca: 'ca',
  cs: 'cs',
  da: 'da',
  de: 'de',
  en: 'en-us',
  es: 'es',
  fi: 'fi',
  fr: 'fr',
  hi: 'hi',
  hu: 'hu',
  it: 'it',
  nl: 'nl',
  no: 'nb',
  pl: 'pl',
  pt: 'pt',
  ro: 'ro',
  sv: 'sv',
  tr: 'tr',
}

/** Languages of the cloning models that uccharan has no normaliser for: their text goes to the model as written. */
export const NOT_NORMALISED = ['ar', 'el', 'fa', 'he', 'ja', 'ko', 'ms', 'ru', 'sw', 'zh'] as const

/** Writes `text` out in words in uccharan language `tag`. */
export type Normaliser = (text: string, tag: string) => Promise<string>

export interface Spoken {
  /** What the model is given. */
  text: string
  /** The uccharan language used, or null when the text went as written. */
  tag: string | null
  /** Why the text went as written: the language has no normaliser, or the normaliser failed (`error`). */
  skipped?: 'language' | 'failed'
  error?: string
}

/**
 * uccharan's normaliser on a loaded module. Improved mode keeps brand names whole ("PayPal", not
 * "Pay Pal"), which suits a model that reads spelling; the rest is the normalise-only mode of text voices.
 */
export function uccharanNormaliser(mod: {
  normalize: (text: string, o: { lang: string; improved: boolean }) => Promise<{ text: string }>
}): Normaliser {
  return async (text, tag) => (await mod.normalize(text, { lang: tag, improved: true })).text
}

/** The page's normaliser: uccharan and its language data are fetched the first time it runs. */
export const pageNormaliser: Normaliser = async (text, tag) => {
  const { loadUccharan } = await import('@/lib/uccharan')
  return uccharanNormaliser(await loadUccharan())(text, tag)
}

/** The text in words for language `language`, or as written where uccharan does not read the language. */
export async function spokenText(
  text: string,
  language: string,
  normalise: Normaliser = pageNormaliser,
): Promise<Spoken> {
  const tag = NORMALISER_LANGUAGE[language] ?? null
  if (!tag) return { text, tag: null, skipped: 'language' }
  try {
    const out = (await normalise(text, tag)).replace(/\s+/g, ' ').trim()
    return { text: out || text, tag }
  } catch (err) {
    return { text, tag: null, skipped: 'failed', error: err instanceof Error ? err.message : String(err) }
  }
}

/** True when the text has something a normaliser would write out (digits, currency or other signs). */
export const hasNumbers = (text: string) => /[\p{N}\p{Sc}%‰°§#@&+=/]/u.test(text)

/** "Numbers" in a model's facts: which of its languages get them written out in words. */
export function numbersFact(languages: string[]): string {
  const name = (l: string) => LANGUAGE_NAMES[l] ?? l
  const done = languages.filter((l) => NORMALISER_LANGUAGE[l])
  const left = languages.filter((l) => !NORMALISER_LANGUAGE[l])
  if (!done.length)
    return 'given to the model as written (digits and signs); uccharan has no normaliser for this language'
  const words = `written out in words by uccharan before the model reads them (numbers, dates, times, money, ordinals)`
  if (!left.length) return languages.length === 1 ? words : `${words}, in all its languages`
  return `${words} in ${done.map(name).join(', ')}; in ${left.map(name).join(', ')} they go to the model as written (uccharan has no normaliser for them)`
}
