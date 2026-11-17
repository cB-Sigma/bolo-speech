import type { CatalogModel } from './api/types'
import { languageName } from './format'

/** The most voices one blend may mix (the server's limit). */
export const MAX_BLEND_PARTS = 8

/** One voice of a blend and its weight as typed (any positive number; the server normalises). */
export interface BlendPart {
  voice: string
  weight: number
}

/** Models that speak blends of their own voices (Kokoro): the catalog lists `voice-blending`. */
export function canBlend(model: Pick<CatalogModel, 'features'> | null | undefined): boolean {
  return Boolean(model?.features?.includes('voice-blending'))
}

function fmtWeight(w: number): string {
  const text = w.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
  return text || '0'
}

/** Weights scaled to sum to 1; parts without a voice or with a weight of 0 or less are left out. */
export function normalised(parts: BlendPart[]): BlendPart[] {
  const usable = parts.filter((p) => p.voice && Number.isFinite(p.weight) && p.weight > 0)
  const total = usable.reduce((s, p) => s + p.weight, 0)
  if (!total) return []
  // The same voice twice adds up, as on the server.
  const merged = new Map<string, number>()
  for (const p of usable) merged.set(p.voice, (merged.get(p.voice) ?? 0) + p.weight / total)
  return [...merged].map(([voice, weight]) => ({ voice, weight }))
}

/** `af_bella(0.7)+af_sky(0.3)`: the blend as the server spells it (normalised weights), or '' when it is empty. */
export function blendString(parts: BlendPart[]): string {
  return normalised(parts)
    .map((p) => `${p.voice}(${fmtWeight(p.weight)})`)
    .join('+')
}

/** `af_bella(0.7)+af_sky(0.3)` or `af_bella+af_sky` back into parts; null when it cannot be read. */
export function parseBlend(value: string): BlendPart[] | null {
  const parts: BlendPart[] = []
  for (const raw of value.split('+')) {
    const m = /^\s*([^()+\s]+)\s*(?:\(\s*([^()]*?)\s*\))?\s*$/.exec(raw)
    if (!m) return null
    const weight = m[2] === undefined ? 1 : Number(m[2])
    if (!Number.isFinite(weight) || weight <= 0) return null
    parts.push({ voice: m[1] as string, weight })
  }
  return parts.length ? parts : null
}

/** What `x-bolo-language` says: the language spoken and how it was decided. */
export interface LanguageDecision {
  language: string
  /** `request`, `detected`, `voice`, `default` or `model`. */
  source: string
  detected: string | null
  script: string | null
  matched: string | null
  voice: string | null
  /** The named voice speaks another language than the text. */
  mismatch: boolean
  /** False when the guessed language is not one the model offers. */
  offered: boolean
}

/** Reads `hi-IN; source=detected; detected=hi; script=Devanagari; voice=hf_alpha`; null for an empty header. */
export function parseLanguageHeader(value: string | null | undefined): LanguageDecision | null {
  if (!value || !value.trim()) return null
  const [first, ...rest] = value.split(';').map((s) => s.trim())
  const fields: Record<string, string> = {}
  for (const r of rest) {
    const i = r.indexOf('=')
    if (i > 0) fields[r.slice(0, i).trim()] = r.slice(i + 1).trim()
  }
  return {
    language: first ?? '',
    source: fields.source ?? 'unknown',
    detected: fields.detected ?? null,
    script: fields.script ?? null,
    matched: fields.matched ?? null,
    voice: fields.voice ?? null,
    mismatch: fields.mismatch === 'true',
    offered: fields.offered !== 'false',
  }
}

const SOURCE_TEXT: Record<string, string> = {
  request: 'the language you chose',
  detected: 'detected from the text',
  voice: 'the voice decided',
  default: 'the default (no sure guess)',
  model: 'the model has one language',
}

/** One line for the decision: "hi-IN, detected from the text (Devanagari), voice hf_alpha". */
export function languageSummary(d: LanguageDecision): string {
  let source = SOURCE_TEXT[d.source] ?? d.source
  if (d.source === 'default' && d.detected && d.detected !== 'unsure' && !d.offered) {
    source = `the default (the text looks like ${languageName(d.detected)}, which this model does not offer)`
  }
  const script = d.script ? ` (${d.script})` : ''
  return `${d.language}, ${source}${script}${d.voice ? `, voice ${d.voice}` : ''}`
}

/** A warning worth showing, or null: a named voice reading text in another language. */
export function languageWarning(d: LanguageDecision | null): string | null {
  if (!d) return null
  if (d.mismatch) {
    const text = d.detected && d.detected !== 'unsure' ? languageName(d.detected) : 'another language'
    return `The text looks like ${text}, but the voice${d.voice ? ` ${d.voice}` : ''} speaks ${languageName(d.language)}. If it sounds wrong, choose a voice of the text's language, or "From the text".`
  }
  return null
}
