import type { RequestTrace } from '@/lib/api/types'
import { percentile } from './stats'

export type Metric = 'first_partial' | 'final' | 'ttfa'

export const METRIC_LABEL: Record<Metric, string> = {
  first_partial: 'First partial',
  final: 'Final',
  ttfa: 'Time to first audio',
}

export function metricValue(t: RequestTrace, metric: Metric): number | null {
  const v = metric === 'first_partial' ? t.first_partial_ms : metric === 'final' ? t.final_ms : t.ttfa_ms
  return v != null && Number.isFinite(v) ? v : null
}

export interface SeriesPoint {
  /** Bucket start, epoch ms. */
  t: number
  n: number
  p50: number | null
  p95: number | null
}

/** Percentiles of a metric over fixed time buckets covering [from, to). */
export function bucketize(
  traces: readonly RequestTrace[],
  metric: Metric,
  from: number,
  to: number,
  buckets: number,
): SeriesPoint[] {
  const width = (to - from) / buckets
  const groups: number[][] = Array.from({ length: buckets }, () => [])
  for (const tr of traces) {
    if (tr.status !== 'ok') continue
    const v = metricValue(tr, metric)
    if (v == null) continue
    const at = Date.parse(tr.started_at)
    if (!(at >= from && at < to)) continue
    ;(groups[Math.min(buckets - 1, Math.floor((at - from) / width))] as number[]).push(v)
  }
  return groups.map((g, i) => ({
    t: from + i * width,
    n: g.length,
    p50: g.length ? percentile(g, 50) : null,
    p95: g.length ? percentile(g, 95) : null,
  }))
}

/** Traces matching the filters, newest first (the feed delivers them in completion order). */
export function filterTraces(
  traces: readonly RequestTrace[],
  opts: { model?: string | null; from?: number; to?: number; kind?: string | null },
): RequestTrace[] {
  return traces
    .filter((t) => {
      if (opts.model && t.model !== opts.model) return false
      if (opts.kind && t.kind !== opts.kind) return false
      const at = Date.parse(t.started_at)
      if (opts.from != null && at < opts.from) return false
      if (opts.to != null && at > opts.to) return false
      return true
    })
    .sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at))
}
