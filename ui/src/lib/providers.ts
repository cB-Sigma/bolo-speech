import type { Provider, ProviderKind, ProviderUpdate } from './api/types'
import { priceRows, pricesFromRows, type PriceRow } from './cost'

export const PROVIDER_KIND_LABEL: Record<string, string> = {
  'openai-compatible': 'OpenAI-compatible',
  sarvam: 'Sarvam',
  elevenlabs: 'ElevenLabs',
  deepgram: 'Deepgram',
  soniox: 'Soniox',
  assemblyai: 'AssemblyAI',
  'azure-mai': 'Microsoft MAI (Azure)',
  xai: 'xAI',
  rime: 'Rime',
  bedrock: 'Amazon Bedrock',
}

export const PROVIDER_KIND_NOTE: Record<string, string> = {
  'openai-compatible': 'OpenAI, or any server that speaks its audio API (another bolo, speaches, vLLM, LocalAI).',
  sarvam: 'Indian-language recognition (Saaras) and voices (Bulbul).',
  elevenlabs: 'Voices and Scribe recognition.',
  deepgram: 'Nova recognition, batch and live.',
  soniox: 'Recognition of recordings and live audio, with speaker labels.',
  assemblyai: 'STT-3.7 recognition of recordings and live audio, with speaker labels.',
  'azure-mai':
    'MAI-Transcribe-2-Streaming and MAI-Voice-2.1 in your Azure resource: set the base URL to the resource, and the option region for the voices.',
  xai: 'Grok Voice 3.0 for speech-to-speech sessions (s2s_model).',
  rime: 'Prism for speech-to-speech sessions (s2s_model).',
  bedrock:
    'Nova 2.5 Sonic for speech-to-speech sessions (s2s_model). The key is <access key id>:<secret access key>; needs bolo-speech[nova].',
}

export const PROVIDER_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/
export const DOCS_PROVIDERS = 'https://github.com/cB-Sigma/bolo-speech/blob/main/docs/providers.md'

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
