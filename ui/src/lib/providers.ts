import type { Provider, ProviderKind, ProviderUpdate } from './api/types'
import { priceRows, pricesFromRows, type PriceRow } from './cost'

export const PROVIDER_KIND_LABEL: Record<string, string> = {
  'openai-compatible': 'OpenAI-compatible',
  sarvam: 'Sarvam',
  elevenlabs: 'ElevenLabs',
  deepgram: 'Deepgram',
}

export const PROVIDER_KIND_NOTE: Record<string, string> = {
  'openai-compatible': 'OpenAI, or any server that speaks its audio API (another bolo, speaches, vLLM, LocalAI).',
  sarvam: 'Indian-language recognition (Saaras) and voices (Bulbul).',
  elevenlabs: 'Voices and Scribe recognition.',
  deepgram: 'Nova recognition, batch and live.',
}

export const PROVIDER_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/
export const DOCS_PROVIDERS = 'https://github.com/cb-Sigma/bolo/blob/main/docs/providers.md'

/** The name the catalog's entries of a kind look for. */
export function catalogName(kind: string): string {
  return kind === 'openai-compatible' ? 'openai' : kind
}

/** What the add/edit dialog holds. `apiKey` starts empty every time: a stored key is never read back. */
export interface ProviderForm {
  name: string
  kind: string
  baseUrl: string
  apiKey: string
  clearKey: boolean
  prices: PriceRow[]
}

export function providerForm(existing: Provider | null, kinds: ProviderKind[]): ProviderForm {
  if (existing) {
    return {
      name: existing.name,
      kind: existing.kind,
      baseUrl: existing.base_url,
      apiKey: '',
      clearKey: false,
      prices: priceRows(existing.prices),
    }
  }
  return { name: '', kind: kinds[0]?.kind ?? 'openai-compatible', baseUrl: '', apiKey: '', clearKey: false, prices: [] }
}

/** The body for `PUT /api/v1/providers/{name}`. The key is part of it only when one was typed. */
export function providerUpdate(form: ProviderForm): { body: ProviderUpdate; problem: string | null } {
  const { prices, problem } = pricesFromRows(form.prices)
  const body: ProviderUpdate = { kind: form.kind, prices }
  const url = form.baseUrl.trim()
  if (url) {
    if (!/^https?:\/\//i.test(url)) return { body, problem: 'The address must start with http:// or https://.' }
    body.base_url = url
  }
  if (form.apiKey) body.api_key = form.apiKey
  else if (form.clearKey) body.clear_api_key = true
  return { body, problem }
}
