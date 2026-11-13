import { describe, expect, it } from 'vitest'
import {
  breakdown,
  budgetState,
  concurrencyCeiling,
  histogram,
  median,
  niceStep,
  percentile,
  splitLatency,
  summarize,
} from './stats'

describe('percentile', () => {
  it('interpolates linearly like numpy', () => {
    const xs = [10, 20, 30, 40]
    expect(percentile(xs, 0)).toBe(10)
    expect(percentile(xs, 50)).toBe(25)
    expect(percentile(xs, 100)).toBe(40)
    expect(percentile(xs, 95)).toBeCloseTo(38.5)
    expect(percentile([15, 20, 35, 40, 50], 40)).toBeCloseTo(29)
  })
  it('ignores order and non-finite values', () => {
    expect(percentile([3, NaN, 1, Infinity, 2], 50)).toBe(2)
  })
  it('handles empty and single inputs', () => {
    expect(percentile([], 50)).toBeNaN()
    expect(percentile([7], 99)).toBe(7)
  })
  it('clamps p', () => {
    expect(percentile([1, 2, 3], 150)).toBe(3)
    expect(percentile([1, 2, 3], -5)).toBe(1)
  })
  it('median is p50', () => {
    expect(median([5, 1, 3])).toBe(3)
  })
})

describe('summarize', () => {
  it('reports the usual figures', () => {
    const s = summarize(Array.from({ length: 101 }, (_, i) => i))
    expect(s).toMatchObject({ n: 101, p50: 50, p95: 95, p99: 99, min: 0, max: 100, mean: 50 })
  })
  it('is NaN-safe when empty', () => {
    expect(summarize([]).n).toBe(0)
    expect(summarize([]).p95).toBeNaN()
  })
})

describe('histogram', () => {
  it('uses nice bin edges from zero and counts every value', () => {
    const bins = histogram([5, 12, 18, 31, 49, 50, 99], 5)
    expect(bins[0]?.x0).toBe(0)
    expect(bins.every((b) => b.x1 - b.x0 === 20)).toBe(true)
    expect(bins.reduce((a, b) => a + b.count, 0)).toBe(7)
    expect(bins.map((b) => b.count)).toEqual([3, 1, 2, 0, 1])
  })
  it('folds outliers into the last bin when capped', () => {
    const bins = histogram([10, 20, 5000], 10, 100)
    expect(bins[bins.length - 1]?.count).toBe(1)
    expect(bins[bins.length - 1]?.x1).toBeLessThanOrEqual(110)
  })
  it('returns nothing for no data', () => {
    expect(histogram([])).toEqual([])
  })
  it('niceStep picks 1/2/2.5/5 multiples', () => {
    expect(niceStep(100, 5)).toBe(20)
    expect(niceStep(950, 4)).toBe(250)
    expect(niceStep(7, 10)).toBe(1)
    expect(niceStep(0.3, 3)).toBeCloseTo(0.1)
  })
})

describe('budgetState', () => {
  it('goes good, warning, bad as latency grows past the target', () => {
    expect(budgetState(250, 300)).toBe('good')
    expect(budgetState(300, 300)).toBe('good')
    expect(budgetState(301, 300)).toBe('warning')
    expect(budgetState(450, 300)).toBe('warning')
    expect(budgetState(451, 300)).toBe('bad')
  })
  it('is unknown without a measurement', () => {
    expect(budgetState(null, 300)).toBe('unknown')
    expect(budgetState(NaN, 300)).toBe('unknown')
    expect(budgetState(100, 0)).toBe('unknown')
  })
})

describe('network / server split', () => {
  it('attributes the remainder to the network', () => {
    const s = splitLatency(400, 300)
    expect(s.networkMs).toBe(100)
    expect(s.networkShare).toBeCloseTo(0.25)
    expect(s.serverShare).toBeCloseTo(0.75)
  })
  it('never goes negative when the server reports more than was measured', () => {
    const s = splitLatency(200, 260)
    expect(s.serverMs).toBe(200)
    expect(s.networkMs).toBe(0)
  })
  it('handles zero', () => {
    expect(splitLatency(0, 0).networkShare).toBe(0)
  })
})

describe('breakdown', () => {
  it('splits into model, network and waiting, summing to the total', () => {
    const b = breakdown(700, 220, 30)
    expect(b).toEqual({ totalMs: 700, modelMs: 220, networkMs: 30, waitMs: 450 })
    expect(b.modelMs + b.networkMs + b.waitMs).toBe(b.totalMs)
  })
  it('clamps parts that exceed the total', () => {
    expect(breakdown(100, 250, 40)).toEqual({ totalMs: 100, modelMs: 100, networkMs: 0, waitMs: 0 })
    expect(breakdown(100, 80, 40)).toEqual({ totalMs: 100, modelMs: 80, networkMs: 20, waitMs: 0 })
  })
  it('treats missing parts as zero', () => {
    expect(breakdown(300, null, undefined)).toEqual({ totalMs: 300, modelMs: 0, networkMs: 0, waitMs: 300 })
  })
})

describe('concurrencyCeiling', () => {
  const pts = (xs: [number, number | null][]) => xs.map(([concurrency, latencyMs]) => ({ concurrency, latencyMs }))
  it('finds the last level under the target', () => {
    expect(
      concurrencyCeiling(
        pts([
          [1, 100],
          [2, 180],
          [4, 290],
          [8, 600],
        ]),
        300,
      ),
    ).toBe(4)
  })
  it('stops at the first breach even if a later level recovers', () => {
    expect(
      concurrencyCeiling(
        pts([
          [1, 100],
          [2, 400],
          [4, 200],
        ]),
        300,
      ),
    ).toBe(1)
  })
  it('returns null when nothing meets the target, and sorts its input', () => {
    expect(concurrencyCeiling(pts([[1, 500]]), 300)).toBeNull()
    expect(
      concurrencyCeiling(
        pts([
          [8, 900],
          [1, 100],
          [2, 200],
        ]),
        300,
      ),
    ).toBe(2)
  })
  it('treats a missing measurement as a breach', () => {
    expect(
      concurrencyCeiling(
        pts([
          [1, 100],
          [2, null],
          [4, 100],
        ]),
        300,
      ),
    ).toBe(1)
  })
})
