import { languageName } from './format'

/**
 * Languages of India (and close neighbours) that speech models commonly cover,
 * with their own names, in the order the console lists them.
 */
export const INDIC_LANGUAGES: { code: string; native: string }[] = [
  { code: 'hi', native: 'हिन्दी' },
  { code: 'bn', native: 'বাংলা' },
  { code: 'ta', native: 'தமிழ்' },
  { code: 'te', native: 'తెలుగు' },
  { code: 'mr', native: 'मराठी' },
  { code: 'gu', native: 'ગુજરાતી' },
  { code: 'kn', native: 'ಕನ್ನಡ' },
  { code: 'ml', native: 'മലയാളം' },
  { code: 'pa', native: 'ਪੰਜਾਬੀ' },
  { code: 'ur', native: 'اردو' },
  { code: 'ne', native: 'नेपाली' },
  { code: 'or', native: 'ଓଡ଼ିଆ' },
  { code: 'as', native: 'অসমীয়া' },
]

const NATIVE = new Map(INDIC_LANGUAGES.map((l) => [l.code, l.native]))

/** Primary subtag: `hi-IN` → `hi`. */
export const baseLanguage = (code: string): string => code.toLowerCase().split(/[-_]/)[0] ?? code

export function isIndic(code: string): boolean {
  return NATIVE.has(baseLanguage(code))
}

/** "हिन्दी · Hindi" for languages with a listed native name, otherwise the English name. */
export function languageLabel(code: string): string {
  const native = NATIVE.get(baseLanguage(code))
  const english = languageName(code)
  return native ? `${native} · ${english}` : english
}

/** Whether a model's language list covers `code` (`multi` covers everything). */
export function supportsLanguage(languages: readonly string[], code: string): boolean {
  if (!code || code === 'auto') return true
  const want = baseLanguage(code)
  return languages.some((l) => l === 'multi' || baseLanguage(l) === want)
}

/**
 * Splits a model's languages into Indian languages (in the fixed order above),
 * English, and the rest sorted by English name.
 */
export function groupLanguages(languages: readonly string[]): { indic: string[]; english: string[]; other: string[] } {
  const set = new Set(languages.filter((l) => l && l !== 'multi'))
  const indic = INDIC_LANGUAGES.map((l) => l.code).filter((c) => set.has(c))
  const english = set.has('en') ? ['en'] : []
  const other = [...set]
    .filter((c) => c !== 'en' && !NATIVE.has(c))
    .sort((a, b) => languageName(a).localeCompare(languageName(b)))
  return { indic, english, other }
}

/** Search text for a language: code, English name and native name. */
export function languageSearchText(code: string): string {
  return `${code} ${languageName(code)} ${NATIVE.get(baseLanguage(code)) ?? ''}`.toLowerCase()
}
