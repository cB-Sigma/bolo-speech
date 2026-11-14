import type { Cost, Price, ProviderStats } from './api/types'

/** The units a price can be given in, in the order the editor offers them. */
export const PRICE_UNITS = ['per_minute', 'per_hour', 'per_1k_chars', 'per_1m_chars', 'per_request'] as const
export type PriceUnit = (typeof PRICE_UNITS)[number]

export const PRICE_UNIT_LABEL: Record<PriceUnit, string> = {
  per_minute: 'per minute of audio',
  per_hour: 'per hour of audio',
  per_1k_chars: 'per 1,000 characters',
  per_1m_chars: 'per million characters',
  per_request: 'per request',
}

/** An estimated amount, with enough digits that a fraction of a cent still shows. */
export function fmtCost(cost: Cost | null | undefined): string {
  if (!cost || !Number.isFinite(cost.amount)) return '–'
  const a = cost.amount
  const digits = a === 0 ? 2 : a < 0.01 ? 5 : a < 1 ? 4 : 2
  return `${a.toFixed(digits)} ${cost.currency}`
}

/** `"0.0061 USD"` (the `x-bolo-estimated-cost` header) → a cost, or null. */
export function parseCostHeader(value: string | null | undefined): Cost | null {
  const m = /^\s*([0-9.eE+-]+)\s+([A-Za-z]{3})\s*$/.exec(value ?? '')
  if (!m) return null
  const amount = Number(m[1])
  return Number.isFinite(amount) ? { amount, currency: (m[2] as string).toUpperCase(), estimated: true } : null
}

/** A price row as text: `0.006 USD per minute of audio`. */
export function describePrice(price: Price | null | undefined): string | null {
  if (!price) return null
  const currency = price.currency ?? 'USD'
  const parts = PRICE_UNITS.filter((u) => typeof price[u] === 'number').map(
    (u) => `${price[u]} ${currency} ${PRICE_UNIT_LABEL[u]}`,
  )
  return parts.length ? parts.join(' + ') : null
}

/** Totals over a provider's models: requests, errors and estimated spend by currency. */
export function providerTotals(stats: Record<string, ProviderStats> | undefined): {
  requests: number
  errors: number
  cost: Record<string, number>
} {
  const out = { requests: 0, errors: 0, cost: {} as Record<string, number> }
  for (const s of Object.values(stats ?? {})) {
    out.requests += s.requests
    out.errors += s.errors
    for (const [currency, amount] of Object.entries(s.estimated_cost ?? {})) {
      out.cost[currency] = (out.cost[currency] ?? 0) + amount
    }
  }
  return out
}

export interface PriceRow {
  model: string
  unit: PriceUnit
  amount: string
  currency: string
}

/** The price table as editable rows (one row per model and unit). */
export function priceRows(prices: Record<string, Price> | undefined): PriceRow[] {
  const rows: PriceRow[] = []
  for (const [model, price] of Object.entries(prices ?? {})) {
    for (const unit of PRICE_UNITS) {
      const v = price[unit]
      if (typeof v === 'number') rows.push({ model, unit, amount: String(v), currency: price.currency ?? 'USD' })
    }
  }
  return rows
}

/**
 * Editable rows back into a price table. Rows without a model or an amount are dropped; a row whose amount is
 * not a number, or two currencies for one model, is an error (returned as `problem`).
 */
export function pricesFromRows(rows: PriceRow[]): { prices: Record<string, Price>; problem: string | null } {
  const prices: Record<string, Price> = {}
  const currencies = new Map<string, string>()
  for (const row of rows) {
    const model = row.model.trim()
    if (!model && !row.amount.trim()) continue
    if (!model) return { prices, problem: 'Every price needs a model name (or * for all other models).' }
    const amount = Number(row.amount)
    if (row.amount.trim() === '' || !Number.isFinite(amount) || amount < 0)
      return { prices, problem: `The price of ${model} is not a number.` }
    const currency = (row.currency.trim() || 'USD').toUpperCase()
    const seen = currencies.get(model)
    if (seen && seen !== currency) return { prices, problem: `${model} has prices in two currencies; use one.` }
    currencies.set(model, currency)
    const entry = (prices[model] ??= {})
    entry[row.unit] = amount
    if (currency !== 'USD') entry.currency = currency
  }
  return { prices, problem: null }
}
