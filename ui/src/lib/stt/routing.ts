import { BROWSER_MODELS, type BrowserModel } from '@/lib/browser-stt/models'
import { languageName } from '@/lib/format'

/** Below this probability the identifier is not trusted (the server's `lid_min_probability`). */
export const LID_MIN_PROBABILITY = 0.5

export interface LidDecision {
  /** Most likely languages first, with their probability. */
  ranked: [string, number][]
  /** Seconds heard, and the time it took (ms). */
  seconds: number
  ms: number
  lid: string
  /** The language the recogniser is told, or null to let it decide. */
  language: string | null
  /** The recogniser that answers. */
  model: string
  reason: 'kept' | 'routed' | 'recommended' | 'unsure' | 'unknown'
  /** A recogniser for the language that is not downloaded: suggested, never fetched by itself. */
  recommended: string | null
  /** In words. */
  summary: string
  /** As the server's `x-bolo-route` header would say it. */
  header: string
}

/** Whisper's code for a language some identifiers name differently. */
const SAME: Record<string, string> = { iw: 'he', jw: 'jv', nn: 'no', nb: 'no' }

/**
 * Where a transcription goes once the language is known (the server's routing for `model=auto`): the current
 * recogniser when it speaks the language; else a downloaded recogniser made for it (or a multilingual one); else
 * the current one, with a recogniser for the language suggested. Below the minimum probability nothing changes.
 */
export function routeByLanguage(
  ranked: [string, number][],
  current: BrowserModel,
  downloaded: (id: string) => boolean,
  meta: { seconds: number; ms: number; lid: string },
): LidDecision {
  const [rawLang, prob] = ranked[0] ?? ['', 0]
  const lang = SAME[rawLang] ?? rawLang
  const base = { ranked, ...meta, recommended: null as string | null }
  const header = (model: string, reason: string, extra = '') =>
    `auto -> ${model}; reason=${reason}; language=${lang || 'unknown'}; probability=${prob.toFixed(2)}; lid=${meta.lid}${extra}`
  if (!lang || prob < LID_MIN_PROBABILITY)
    return {
      ...base,
      language: null,
      model: current.id,
      reason: 'unsure',
      summary: `Not sure enough (${Math.round(prob * 100)}% for ${lang ? languageName(lang) : 'any language'}): ${current.label} decides by itself.`,
      header: header(current.id, 'unsure'),
    }
  const speaks = (m: BrowserModel) => m.languages.includes(lang)
  if (speaks(current))
    return {
      ...base,
      language: current.multilingual ? lang : null,
      model: current.id,
      reason: 'kept',
      summary: `${languageName(lang)}: ${current.label} speaks it${current.multilingual ? ' and is told the language' : ''}.`,
      header: header(current.id, 'kept'),
    }
  const candidates = [
    ...BROWSER_MODELS.filter((m) => speaks(m) && !m.multilingual && !m.requiresWebGpu),
    ...BROWSER_MODELS.filter((m) => speaks(m) && m.multilingual && !m.requiresWebGpu),
  ]
  const ready = candidates.find((m) => downloaded(m.id))
  if (ready)
    return {
      ...base,
      language: ready.multilingual ? lang : null,
      model: ready.id,
      reason: 'routed',
      summary: `${languageName(lang)}: sent to ${ready.label}, downloaded already, which speaks it.`,
      header: header(ready.id, 'routed'),
    }
  const suggestion = candidates[0] ?? null
  if (!suggestion)
    return {
      ...base,
      language: null,
      model: current.id,
      reason: 'unknown',
      summary: `${languageName(lang)}: no recogniser in the page speaks it; ${current.label} answers.`,
      header: header(current.id, 'no-recogniser'),
    }
  return {
    ...base,
    language: null,
    model: current.id,
    reason: 'recommended',
    recommended: suggestion.id,
    summary: `${languageName(lang)}: ${current.label} does not speak it. ${suggestion.label} does; it is not downloaded, so it is only suggested.`,
    header: header(current.id, 'default', `; recommended=${suggestion.id}`),
  }
}
