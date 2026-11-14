/**
 * Romanised text for text-to-speech: an Indian language typed in Latin letters is written in the voice's
 * script before it is read (`romanised` on a speech request). The server announces what it can do in
 * `capabilities.romanised`; a server that says nothing has no such option.
 */
import type { RomanisedRecord, ServerInfo } from '@/lib/api/types'

export type RomanisedMode = 'off' | 'auto' | 'on'
export const ROMANISED_MODES: RomanisedMode[] = ['off', 'auto', 'on']

/** Languages whose voices can be handed romanised text (the server's list of Indian-language codes). */
const INDIC = new Set(
  'as awa bho bn brx doi gu hi hne kn kok ks mag mai ml mni mr ne or pa raj sa sat sd si ta te ur'.split(' '),
)

export function isIndicLanguage(language: string | null | undefined): boolean {
  return INDIC.has((language ?? '').trim().toLowerCase().replace('_', '-').split('-')[0] ?? '')
}

/** What the server says about the option, or null when it has none. */
export function romanisedOffered(info: ServerInfo | undefined) {
  return info?.capabilities?.romanised ?? null
}

export const ROMANISED_LABEL: Record<RomanisedMode, string> = {
  off: 'off',
  auto: 'automatic',
  on: 'always',
}

export const ROMANISED_HINT: Record<RomanisedMode, string> = {
  off: 'The text is read as written.',
  auto: "Written in the voice's script when the text is mostly Latin letters.",
  on: "Latin-letter words of the language are always written in the voice's script.",
}

/** Why choosing a mode would do nothing for this language on this server, or null when it would work. */
export function romanisedBlocked(info: ServerInfo | undefined, language: string | null | undefined): string | null {
  const cap = romanisedOffered(info)
  if (!cap) return 'This server has no romanised-text option.'
  if (cap.available === false) return cap.reason ?? 'The server has no transliterator installed.'
  const base = (language ?? '').toLowerCase().split('-')[0] ?? ''
  if (cap.languages && !cap.languages.includes(base)) return `The server has no romanised-text pack for ${base}.`
  if (cap.installed && !cap.installed.includes(base))
    return `The pack for ${base} is not installed (Models → Components: uccharan-xlit-${base}).`
  return null
}

/** One line for the trace view: what happened to romanised text in a request. */
export function romanisedSummary(r: RomanisedRecord | null | undefined): string | null {
  if (!r) return null
  const how = r.mode === 'auto' ? 'automatic' : r.mode === 'on' ? 'always' : r.mode
  const from = r.source === 'request' ? 'asked for by the request' : "the voice's setting"
  if (r.applied) {
    const words = typeof r.words === 'number' ? `${r.words} ${r.words === 1 ? 'word' : 'words'} ` : ''
    return `Romanised text: ${words}written in the voice's script (${how}, ${from})`
  }
  if (r.notice) return `Romanised text (${how}): ${r.notice}`
  return `Romanised text (${how}): nothing to convert`
}
