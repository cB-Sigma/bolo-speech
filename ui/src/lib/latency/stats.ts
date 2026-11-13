/** Linear-interpolated percentile (same definition as numpy's default). `p` in 0..100. */
export function percentile(values: readonly number[], p: number): number {
  const xs = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
  if (xs.length === 0) return NaN
  if (xs.length === 1) return xs[0] as number
  const rank = (Math.min(100, Math.max(0, p)) / 100) * (xs.length - 1)
  const lo = Math.floor(rank)
  const hi = Math.ceil(rank)
  const a = xs[lo] as number
  const b = xs[hi] as number
  return a + (b - a) * (rank - lo)
}

export interface Summary {
  n: number
  p50: number
  p95: number
  p99: number
  min: number
  max: number
  mean: number
}

export function summarize(values: readonly number[]): Summary {
  const xs = values.filter((v) => Number.isFinite(v))
  if (xs.length === 0) return { n: 0, p50: NaN, p95: NaN, p99: NaN, min: NaN, max: NaN, mean: NaN }
  return {
    n: xs.length,
    p50: percentile(xs, 50),
    p95: percentile(xs, 95),
    p99: percentile(xs, 99),
    min: Math.min(...xs),
    max: Math.max(...xs),
    mean: xs.reduce((a, b) => a + b, 0) / xs.length,
  }
}

export interface Bin {
  x0: number
  x1: number
  count: number
}

/** "Nice" step (1, 2, 2.5, 5 × 10^n) that yields at most `maxBins` bins over `span`. */
export function niceStep(span: number, maxBins: number): number {
  if (!(span > 0) || maxBins <= 0) return 1
  const raw = span / maxBins
  const mag = 10 ** Math.floor(Math.log10(raw))
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (m * mag >= raw) return m * mag
  }
  return 10 * mag
}

/** Fixed-width histogram from zero with nice bin edges. Values above `cap` land in the last bin. */
export function histogram(values: readonly number[], maxBins = 24, cap?: number): Bin[] {
  const xs = values.filter((v) => Number.isFinite(v) && v >= 0)
  if (xs.length === 0) return []
  const hi = cap ?? Math.max(...xs)
  const step = niceStep(Math.max(hi, 1), maxBins)
  const n = Math.max(1, Math.ceil(hi / step + 1e-9))
  const bins: Bin[] = Array.from({ length: n }, (_, i) => ({ x0: i * step, x1: (i + 1) * step, count: 0 }))
  for (const v of xs) {
    const i = Math.min(n - 1, Math.floor(v / step))
    ;(bins[i] as Bin).count += 1
  }
  return bins
}

export type BudgetState = 'good' | 'warning' | 'bad' | 'unknown'

/** Within target → good; up to `warnFactor`× target → warning; beyond → bad. */
export function budgetState(measuredMs: number | null | undefined, targetMs: number, warnFactor = 1.5): BudgetState {
  if (measuredMs == null || !Number.isFinite(measuredMs) || !(targetMs > 0)) return 'unknown'
  if (measuredMs <= targetMs) return 'good'
  if (measuredMs <= targetMs * warnFactor) return 'warning'
  return 'bad'
}

export interface Split {
  totalMs: number
  serverMs: number
  networkMs: number
  /** 0..1 */
  networkShare: number
  serverShare: number
}

/**
 * Split an end-to-end latency measured in the browser into the part the
 * server reports for itself (`latency_ms`) and the remainder, which is
 * network transit plus client scheduling. Clamped so neither part is negative.
 */
export function splitLatency(endToEndMs: number, serverMs: number): Split {
  const total = Math.max(0, endToEndMs)
  const server = Math.min(total, Math.max(0, serverMs))
  const network = total - server
  return {
    totalMs: total,
    serverMs: server,
    networkMs: network,
    networkShare: total > 0 ? network / total : 0,
    serverShare: total > 0 ? server / total : 0,
  }
}

/**
 * Highest concurrency whose latency stays within `targetMs`, scanning upward
 * and stopping at the first breach (a later recovery is not trusted).
 * Returns null when even the lowest level misses the target.
 */
export function concurrencyCeiling(
  points: readonly { concurrency: number; latencyMs: number | null | undefined }[],
  targetMs: number,
): number | null {
  const sorted = [...points].sort((a, b) => a.concurrency - b.concurrency)
  let best: number | null = null
  for (const p of sorted) {
    if (p.latencyMs == null || !Number.isFinite(p.latencyMs) || p.latencyMs > targetMs) break
    best = p.concurrency
  }
  return best
}

export interface Breakdown {
  totalMs: number
  /** Time the model spent, as reported by whoever ran it. */
  modelMs: number
  /** Network round trip attributed to this result. */
  networkMs: number
  /** What is left: waiting for the end-of-speech decision, queueing and client scheduling. */
  waitMs: number
}

/**
 * Three-way attribution of the delay between the end of speech and the final
 * transcript. Model time is taken first, then the measured network round
 * trip, and the remainder is endpointing and scheduling. Nothing goes negative
 * and the parts always sum to the total.
 */
export function breakdown(
  endToEndMs: number,
  modelMs: number | null | undefined,
  rttMs: number | null | undefined,
): Breakdown {
  const total = Math.max(0, Number.isFinite(endToEndMs) ? endToEndMs : 0)
  const model = Math.min(total, Math.max(0, modelMs ?? 0))
  const network = Math.min(total - model, Math.max(0, rttMs ?? 0))
  return { totalMs: total, modelMs: model, networkMs: network, waitMs: total - model - network }
}

export function median(values: readonly number[]): number {
  return percentile(values, 50)
}
