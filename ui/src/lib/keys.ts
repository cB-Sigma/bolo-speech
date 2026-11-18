import type { ApiKeyCreate, ApiKeyInfo } from './api/types'

export const KEY_SCOPES = ['read', 'stt', 'tts', 'realtime', 'manage', 'admin'] as const

export const SCOPE_LABEL: Record<string, string> = {
  read: 'Read',
  stt: 'Speech to text',
  tts: 'Text to speech',
  realtime: 'Realtime and agents',
  manage: 'Manage models',
  admin: 'Admin',
}

export const PERIOD_LABEL: Record<string, string> = { day: 'per day', month: 'per month', total: 'in all' }

export const DOCS_KEYS = 'https://github.com/cb-Sigma/bolo/blob/main/docs/api.md#api-keys-apiv1keys'

/** What the create/edit dialog holds; numbers stay text until they are sent. */
export interface KeyForm {
  name: string
  scopes: string[]
  rateLimit: string
  burst: string
  quota: string
  quotaPeriod: string
  expiresInDays: string
}

export function keyForm(existing?: ApiKeyInfo | null): KeyForm {
  if (!existing) {
    return {
      name: '',
      scopes: ['stt', 'tts'],
      rateLimit: '',
      burst: '',
      quota: '',
      quotaPeriod: 'day',
      expiresInDays: '',
    }
  }
  return {
    name: existing.name,
    scopes: [...existing.scopes],
    rateLimit: existing.rate_limit == null ? '' : String(existing.rate_limit),
    burst: existing.burst == null ? '' : String(existing.burst),
    quota: existing.quota == null ? '' : String(existing.quota),
    quotaPeriod: existing.quota_period,
    expiresInDays: '',
  }
}

function positive(text: string, label: string, integer: boolean): { value: number | null; problem: string | null } {
  const t = text.trim()
  if (!t) return { value: null, problem: null }
  const n = Number(t)
  if (!Number.isFinite(n) || n <= 0 || (integer && !Number.isInteger(n))) {
    return { value: null, problem: `${label} must be a positive ${integer ? 'whole number' : 'number'}, or empty.` }
  }
  return { value: n, problem: null }
}

/** The body for `POST /api/v1/keys` (or `PATCH`: the same fields, an empty limit removes it). */
export function keyBody(form: KeyForm): { body: ApiKeyCreate; problem: string | null } {
  const rate = positive(form.rateLimit, 'The rate limit', false)
  const burst = positive(form.burst, 'The burst', true)
  const quota = positive(form.quota, 'The quota', true)
  const expires = positive(form.expiresInDays, 'The expiry', false)
  const body: ApiKeyCreate = {
    name: form.name.trim(),
    scopes: form.scopes,
    rate_limit: rate.value,
    burst: burst.value,
    quota: quota.value,
    quota_period: form.quotaPeriod,
  }
  if (expires.value != null) body.expires_in_days = expires.value
  const problem =
    (!body.name ? 'Give the key a name.' : null) ??
    (body.name.length > 80 ? 'The name is longer than 80 characters.' : null) ??
    (form.scopes.length === 0 ? 'Pick at least one scope.' : null) ??
    rate.problem ??
    burst.problem ??
    quota.problem ??
    expires.problem
  return { body, problem }
}

/** A short line for a key's limits: "60/min · 5,000 per day". */
export function limitsText(k: Pick<ApiKeyInfo, 'rate_limit' | 'quota' | 'quota_period'>): string {
  const parts: string[] = []
  if (k.rate_limit != null) parts.push(`${k.rate_limit}/min`)
  if (k.quota != null)
    parts.push(`${k.quota.toLocaleString('en-US')} ${PERIOD_LABEL[k.quota_period] ?? k.quota_period}`)
  return parts.length ? parts.join(' · ') : 'No limits'
}

/** How much of the quota this period has used, 0 to 1 (null without a quota). */
export function quotaUsed(k: Pick<ApiKeyInfo, 'quota' | 'usage'>): number | null {
  if (!k.quota) return null
  return Math.min(1, k.usage.period_requests / k.quota)
}

export function keyState(k: Pick<ApiKeyInfo, 'revoked_at' | 'active'>): 'active' | 'revoked' | 'expired' {
  return k.revoked_at ? 'revoked' : k.active ? 'active' : 'expired'
}
