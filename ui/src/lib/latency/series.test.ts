import { describe, expect, it } from 'vitest'
import type { RequestTrace } from '@/lib/api/types'
import { bucketize, filterTraces, metricValue } from './series'

const base = Date.parse('2026-11-13T10:00:00.000Z')
function trace(offsetMs: number, p: Partial<RequestTrace> = {}): RequestTrace {
  return {
    id: `r_${offsetMs}`,
    kind: 'stt',
    transport: 'websocket',
    model: 'm1',
    started_at: new Date(base + offsetMs).toISOString(),
    status: 'ok',
    error: null,
    audio_seconds: 2,
    chars: null,
    language: 'en',
    phases: [],
    first_partial_ms: 100,
    final_ms: 200,
    ttfa_ms: null,
    total_ms: 200,
    rtf: 0.1,
    batch_size: 1,
    text_preview: null,
    ...p,
  }
}

describe('bucketize', () => {
  it('groups by time bucket and takes percentiles within each', () => {
    const traces = [
      trace(0, { first_partial_ms: 100 }),
      trace(500, { first_partial_ms: 300 }),
      trace(1500, { first_partial_ms: 50 }),
    ]
    const s = bucketize(traces, 'first_partial', base, base + 3000, 3)
    expect(s.map((p) => p.n)).toEqual([2, 1, 0])
    expect(s[0]?.p50).toBe(200)
    expect(s[1]?.p95).toBe(50)
    expect(s[2]?.p50).toBeNull()
    expect(s[1]?.t).toBe(base + 1000)
  })
  it('skips failures, missing metrics and out-of-range traces', () => {
    const traces = [trace(100, { status: 'error' }), trace(200, { first_partial_ms: null }), trace(-50), trace(5000)]
    expect(bucketize(traces, 'first_partial', base, base + 1000, 2).every((p) => p.n === 0)).toBe(true)
  })
})

describe('filterTraces / metricValue', () => {
  it('filters by model, kind and time', () => {
    const traces = [trace(0), trace(10, { model: 'm2' }), trace(20, { kind: 'tts', ttfa_ms: 80 })]
    expect(filterTraces(traces, { model: 'm2' })).toHaveLength(1)
    expect(filterTraces(traces, { kind: 'tts' })).toHaveLength(1)
    expect(filterTraces(traces, { from: base + 5 })).toHaveLength(2)
  })
  it('reads the right field per metric', () => {
    const t = trace(0, { ttfa_ms: 70 })
    expect(metricValue(t, 'first_partial')).toBe(100)
    expect(metricValue(t, 'final')).toBe(200)
    expect(metricValue(t, 'ttfa')).toBe(70)
  })
})
