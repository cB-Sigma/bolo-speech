import type { RequestTrace } from '@/lib/api/types'

type Listener = (trace: RequestTrace) => void

const listeners = new Set<Listener>()
const recent: RequestTrace[] = []
let seq = 1

/** Real traces of work done by the in-browser recogniser during this visit, newest first. */
export function browserTraces(): readonly RequestTrace[] {
  return recent
}

export function onBrowserTrace(listener: Listener): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export interface BrowserTraceInput {
  model: string
  /** `performance.now()` at which the measured span began. */
  startedAtPerf: number
  phases: { name: string; ms: number }[]
  audioSeconds?: number | null
  firstPartialMs?: number | null
  finalMs?: number | null
  rtf?: number | null
  text?: string | null
  status?: 'ok' | 'error'
  error?: string | null
  batchSize?: number | null
}

/** Record one measured span in the same shape the server uses for its request traces. */
export function recordBrowserTrace(input: BrowserTraceInput): RequestTrace {
  let cursor = 0
  const phases = input.phases
    .filter((p) => p.ms > 0)
    .map((p) => {
      const phase = { name: p.name, start_ms: Math.round(cursor * 10) / 10, dur_ms: Math.round(p.ms * 10) / 10 }
      cursor += p.ms
      return phase
    })
  const trace: RequestTrace = {
    id: `b_${(seq++).toString().padStart(4, '0')}`,
    kind: 'stt',
    transport: 'browser',
    model: input.model,
    started_at: new Date(Date.now() - (performance.now() - input.startedAtPerf)).toISOString(),
    status: input.status ?? 'ok',
    error: input.error ?? null,
    audio_seconds: input.audioSeconds ?? null,
    chars: null,
    language: 'en',
    phases,
    first_partial_ms: input.firstPartialMs != null ? Math.round(input.firstPartialMs) : null,
    final_ms: input.finalMs != null ? Math.round(input.finalMs) : null,
    ttfa_ms: null,
    total_ms: Math.round(cursor),
    rtf: input.rtf != null ? Math.round(input.rtf * 1000) / 1000 : null,
    batch_size: input.batchSize ?? 1,
    text_preview: input.text ? input.text.slice(0, 80) : null,
  }
  recent.unshift(trace)
  if (recent.length > 500) recent.length = 500
  for (const l of listeners) l(trace)
  return trace
}

/** Test helper. */
export function clearBrowserTraces(): void {
  recent.length = 0
  seq = 1
}
