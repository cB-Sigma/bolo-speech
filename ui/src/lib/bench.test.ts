import { describe, expect, it } from 'vitest'
import type { BenchPoint } from './api/types'
import { ceilingFor, defaultMetric, parseLevels, toCsv, toJson, type BenchRun } from './bench'

const point = (c: number, p95: number): BenchPoint => ({
  concurrency: c,
  ok: 10 * c,
  errors: 0,
  first_partial_ms: { p50: p95 / 3, p95: p95 / 2 },
  final_ms: { p50: p95 / 1.5, p95 },
  ttfa_ms: null,
  rtf: { p50: 0.05, p95: 0.08 },
  throughput_x_realtime: 12.5 * c,
})
const run: BenchRun = {
  id: 'x',
  startedAt: Date.parse('2026-11-13T10:00:00Z'),
  request: {
    kind: 'stt',
    model: 'moonshine-tiny-en',
    concurrency: [1, 2, 4],
    duration_s: 10,
    sample: 'builtin:en,short',
  },
  points: [point(1, 150), point(2, 280), point(4, 520)],
}

describe('parseLevels', () => {
  it('accepts commas, spaces and ranges; sorts and de-duplicates', () => {
    expect(parseLevels('1, 2, 4, 8')).toEqual([1, 2, 4, 8])
    expect(parseLevels('8 2  2;1')).toEqual([1, 2, 8])
    expect(parseLevels('1, 4-6, 16')).toEqual([1, 4, 5, 6, 16])
    expect(parseLevels('6–4')).toEqual([4, 5, 6])
  })
  it('drops junk, zero and values over the cap', () => {
    expect(parseLevels('0, -3, abc, 2.5, 4, 9999')).toEqual([4])
    expect(parseLevels('')).toEqual([])
  })
})

describe('ceilingFor', () => {
  it('uses p95 of the chosen metric', () => {
    expect(ceilingFor(run.points, 'final_ms', 300)).toBe(2)
    expect(ceilingFor(run.points, 'first_partial_ms', 300)).toBe(4)
    expect(ceilingFor(run.points, 'final_ms', 100)).toBeNull()
    expect(ceilingFor(run.points, 'ttfa_ms', 300)).toBeNull()
  })
  it('picks a sensible default metric per task', () => {
    expect(defaultMetric('stt')).toBe('final_ms')
    expect(defaultMetric('tts')).toBe('ttfa_ms')
  })
})

describe('export', () => {
  it('writes one CSV row per level with a header, quoting where needed', () => {
    const lines = toCsv(run).trimEnd().split('\n')
    expect(lines).toHaveLength(4)
    expect(lines[0]?.split(',')).toContain('throughput_x_realtime')
    expect(lines[1]).toBe('moonshine-tiny-en,stt,"builtin:en,short",10,1,10,0,50,75,100,150,,,0.05,0.08,12.5')
  })
  it('writes JSON that parses back to the same points', () => {
    const parsed = JSON.parse(toJson(run)) as { started_at: string; points: BenchPoint[]; request: { model: string } }
    expect(parsed.started_at).toBe('2026-11-13T10:00:00.000Z')
    expect(parsed.points).toEqual(run.points)
    expect(parsed.request.model).toBe('moonshine-tiny-en')
  })
})
