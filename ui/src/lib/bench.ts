import type { BenchPoint, BenchRequest } from './api/types'
import { concurrencyCeiling } from './latency/stats'

export interface BenchRun {
  id: string
  /** epoch ms */
  startedAt: number
  request: BenchRequest
  points: BenchPoint[]
}

export type BenchMetric = 'first_partial_ms' | 'final_ms' | 'ttfa_ms'

export const BENCH_METRIC_LABEL: Record<BenchMetric, string> = {
  first_partial_ms: 'First partial',
  final_ms: 'Final',
  ttfa_ms: 'Time to first audio',
}

export function defaultMetric(kind: 'stt' | 'tts'): BenchMetric {
  return kind === 'tts' ? 'ttfa_ms' : 'final_ms'
}

/** Parse "1, 2, 4-8, 16" into a sorted list of unique positive integers. */
export function parseLevels(input: string, max = 256): number[] {
  const out = new Set<number>()
  for (const part of input.split(/[\s,;]+/).filter(Boolean)) {
    const range = /^(\d+)\s*[-–]\s*(\d+)$/.exec(part)
    if (range) {
      const a = Number(range[1])
      const b = Number(range[2])
      for (let v = Math.min(a, b); v <= Math.max(a, b) && out.size < 64; v++) if (v >= 1 && v <= max) out.add(v)
    } else if (/^\d+$/.test(part)) {
      const v = Number(part)
      if (v >= 1 && v <= max) out.add(v)
    }
  }
  return [...out].sort((a, b) => a - b)
}

export function ceilingFor(points: readonly BenchPoint[], metric: BenchMetric, targetMs: number): number | null {
  return concurrencyCeiling(
    points.map((p) => ({ concurrency: p.concurrency, latencyMs: p[metric]?.p95 })),
    targetMs,
  )
}

const csvCell = (v: unknown): string => {
  if (v == null) return ''
  const s = String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function toCsv(run: BenchRun): string {
  const head = [
    'model',
    'kind',
    'sample',
    'duration_s',
    'concurrency',
    'ok',
    'errors',
    'first_partial_p50_ms',
    'first_partial_p95_ms',
    'final_p50_ms',
    'final_p95_ms',
    'ttfa_p50_ms',
    'ttfa_p95_ms',
    'rtf_p50',
    'rtf_p95',
    'throughput_x_realtime',
  ]
  const rows = run.points.map((p) => [
    run.request.model,
    run.request.kind,
    run.request.sample,
    run.request.duration_s,
    p.concurrency,
    p.ok,
    p.errors,
    p.first_partial_ms?.p50,
    p.first_partial_ms?.p95,
    p.final_ms?.p50,
    p.final_ms?.p95,
    p.ttfa_ms?.p50,
    p.ttfa_ms?.p95,
    p.rtf?.p50,
    p.rtf?.p95,
    p.throughput_x_realtime,
  ])
  return [head, ...rows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n'
}

export function toJson(run: BenchRun): string {
  return (
    JSON.stringify(
      { started_at: new Date(run.startedAt).toISOString(), request: run.request, points: run.points },
      null,
      2,
    ) + '\n'
  )
}

const KEY = 'bolo.benchRuns'

export function loadRuns(): BenchRun[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]') as BenchRun[]
    return Array.isArray(raw) ? raw.filter((r) => r && Array.isArray(r.points)) : []
  } catch {
    return []
  }
}

export function saveRuns(runs: BenchRun[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(runs.slice(0, 12)))
  } catch {
    // storage full or unavailable: history just is not kept
  }
}

export function download(filename: string, content: string | Uint8Array, type: string): void {
  const blob = new Blob([content as BlobPart], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
