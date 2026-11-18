import { describe, expect, it } from 'vitest'
import type { ApiKeyInfo } from './api/types'
import { keyBody, keyForm, keyState, limitsText, quotaUsed } from './keys'

const key: ApiKeyInfo = {
  id: 'key_1',
  name: 'app',
  hint: 'bolo_abcd…wxyz',
  scopes: ['stt'],
  created_at: '2026-11-18T00:00:00Z',
  expires_at: null,
  revoked_at: null,
  active: true,
  rate_limit: 60,
  burst: null,
  quota: 5000,
  quota_period: 'day',
  usage: {
    requests: 10,
    period: '2026-11-18',
    period_requests: 1250,
    quota_remaining: 3750,
    by_scope: { stt: 10 },
    rejected: 0,
    last_used_at: null,
  },
}

describe('API key forms', () => {
  it('builds the body and checks the numbers', () => {
    const form = { ...keyForm(), name: ' mobile ', rateLimit: '60', quota: '5000' }
    expect(keyBody(form)).toEqual({
      body: {
        name: 'mobile',
        scopes: ['stt', 'tts'],
        rate_limit: 60,
        burst: null,
        quota: 5000,
        quota_period: 'day',
      },
      problem: null,
    })
    expect(keyBody({ ...form, burst: '1.5' }).problem).toMatch(/burst/)
    expect(keyBody({ ...form, scopes: [] }).problem).toMatch(/scope/)
    expect(keyBody({ ...form, name: '' }).problem).toMatch(/name/)
    expect(keyBody({ ...form, expiresInDays: '30' }).body.expires_in_days).toBe(30)
  })

  it('round-trips an existing key', () => {
    const form = keyForm(key)
    expect(form.rateLimit).toBe('60')
    expect(form.burst).toBe('')
    expect(keyBody(form).body.quota).toBe(5000)
  })

  it('describes limits and usage', () => {
    expect(limitsText(key)).toBe('60/min · 5,000 per day')
    expect(limitsText({ rate_limit: null, quota: null, quota_period: 'day' })).toBe('No limits')
    expect(quotaUsed(key)).toBe(0.25)
    expect(quotaUsed({ ...key, quota: null })).toBeNull()
    expect(keyState(key)).toBe('active')
    expect(keyState({ ...key, revoked_at: 'x' })).toBe('revoked')
    expect(keyState({ ...key, active: false })).toBe('expired')
  })
})
