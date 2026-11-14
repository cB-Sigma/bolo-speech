import { describe, expect, it } from 'vitest'
import { describePrice, fmtCost, parseCostHeader, priceRows, pricesFromRows, providerTotals } from './cost'
import { providerForm, providerUpdate } from './providers'
import type { Provider, ProviderStats } from './api/types'

describe('cost helpers', () => {
  it('formats small amounts with enough digits', () => {
    expect(fmtCost({ amount: 0.0002, currency: 'USD' })).toBe('0.00020 USD')
    expect(fmtCost({ amount: 0.0456, currency: 'INR' })).toBe('0.0456 INR')
    expect(fmtCost({ amount: 12.5, currency: 'EUR' })).toBe('12.50 EUR')
    expect(fmtCost({ amount: 0, currency: 'USD' })).toBe('0.00 USD')
    expect(fmtCost(null)).toBe('–')
  })
  it('parses the estimated-cost header', () => {
    expect(parseCostHeader('0.0061 USD')).toEqual({ amount: 0.0061, currency: 'USD', estimated: true })
    expect(parseCostHeader('1e-05 inr')?.currency).toBe('INR')
    expect(parseCostHeader(null)).toBeNull()
    expect(parseCostHeader('free')).toBeNull()
  })
  it('describes a price and sums provider statistics', () => {
    expect(describePrice({ per_minute: 0.006 })).toBe('0.006 USD per minute of audio')
    expect(describePrice({ per_hour: 30, per_request: 1, currency: 'INR' })).toBe(
      '30 INR per hour of audio + 1 INR per request',
    )
    expect(describePrice(null)).toBeNull()
    expect(describePrice({})).toBeNull()
    const s = (over: Partial<ProviderStats>): ProviderStats =>
      ({ requests: 0, errors: 0, estimated_cost: {}, ...over }) as ProviderStats
    expect(
      providerTotals({
        a: s({ requests: 3, errors: 1, estimated_cost: { USD: 0.01 } }),
        b: s({ requests: 2, estimated_cost: { USD: 0.02, INR: 4 } }),
      }),
    ).toEqual({ requests: 5, errors: 1, cost: { USD: 0.03, INR: 4 } })
    expect(providerTotals(undefined)).toEqual({ requests: 0, errors: 0, cost: {} })
  })
  it('turns a price table into rows and back', () => {
    const prices = {
      'whisper-1': { per_minute: 0.006 },
      '*': { per_request: 0.001, per_1k_chars: 0.02, currency: 'EUR' },
    }
    const rows = priceRows(prices)
    expect(rows).toHaveLength(3)
    expect(pricesFromRows(rows)).toEqual({ prices, problem: null })
    expect(pricesFromRows([{ model: '', unit: 'per_minute', amount: '', currency: 'USD' }])).toEqual({
      prices: {},
      problem: null,
    })
    expect(pricesFromRows([{ model: 'm', unit: 'per_minute', amount: 'cheap', currency: 'USD' }]).problem).toMatch(
      /not a number/,
    )
    expect(pricesFromRows([{ model: '', unit: 'per_minute', amount: '1', currency: 'USD' }]).problem).toMatch(
      /model name/,
    )
    expect(
      pricesFromRows([
        { model: 'm', unit: 'per_minute', amount: '1', currency: 'USD' },
        { model: 'm', unit: 'per_hour', amount: '2', currency: 'INR' },
      ]).problem,
    ).toMatch(/two currencies/)
  })
})

describe('the provider form', () => {
  const stored: Provider = {
    name: 'sarvam',
    kind: 'sarvam',
    base_url: 'https://api.sarvam.ai',
    configured: true,
    api_key_set: true,
    options: {},
    prices: {},
    source: 'stored',
    editable: true,
    terms_url: null,
  }
  it('never starts with a key, and sends one only when it was typed', () => {
    const form = providerForm(stored, [])
    expect(form.apiKey).toBe('')
    const kept = providerUpdate(form)
    expect(kept.problem).toBeNull()
    expect(kept.body).toEqual({ kind: 'sarvam', prices: {}, base_url: 'https://api.sarvam.ai' })
    expect('api_key' in kept.body).toBe(false)
    expect(providerUpdate({ ...form, apiKey: 'sv-new' }).body.api_key).toBe('sv-new')
    expect(providerUpdate({ ...form, clearKey: true }).body).toMatchObject({ clear_api_key: true })
    // typing a key wins over "remove the stored key"
    expect(providerUpdate({ ...form, apiKey: 'k', clearKey: true }).body.clear_api_key).toBeUndefined()
  })
  it('starts a new provider from the first kind and checks the address', () => {
    const form = providerForm(null, [{ kind: 'deepgram', default_base_url: 'https://api.deepgram.com', terms_url: '' }])
    expect(form).toMatchObject({ name: '', kind: 'deepgram', baseUrl: '', apiKey: '' })
    expect(providerUpdate(form).body).toEqual({ kind: 'deepgram', prices: {} })
    expect(providerUpdate({ ...form, baseUrl: 'api.example.com' }).problem).toMatch(/http/)
  })
})
