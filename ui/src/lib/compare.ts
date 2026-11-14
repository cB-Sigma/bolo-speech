import { api, ApiError, type SpeechBody } from './api/client'
import type { CatalogModel, Cost, RequestTrace } from './api/types'
import { parseCostHeader } from './cost'

/** What one model made of the shared input. */
export interface CompareResult {
  model: string
  status: 'running' | 'ok' | 'error'
  error?: string
  errorCode?: string | null
  /** Speech to text: the transcript. */
  text?: string
  /** Text to speech: the audio, as an object URL the caller revokes, and its length. */
  audioUrl?: string
  audioSeconds?: number | null
  /** Wall-clock time in the browser, request to last byte. */
  clientMs?: number
  /** Text to speech: until the first bytes of audio arrived in the browser. */
  firstByteMs?: number | null
  /** Server-side numbers from the request's trace. */
  serverMs?: number | null
  ttfaMs?: number | null
  rtf?: number | null
  /** Time spent waiting for the cloud provider, when the model is a cloud model. */
  providerMs?: number | null
  cost?: Cost | null
  /** The model that answered, when a fallback took over. */
  answeredBy?: string | null
  fallbackReason?: string | null
  requestId?: string | null
}

export const MIN_COMPARE = 2
export const MAX_COMPARE = 6

/** `"a -> b; reason=provider_error"` → the reason. */
export function fallbackReason(header: string | null | undefined): string | null {
  return /reason=([\w-]+)/.exec(header ?? '')?.[1] ?? null
}

/** Whether a compared model costs money per request (a cloud model), for the "local: no API cost" label. */
export function isPaid(m: Pick<CatalogModel, 'cloud' | 'engine'> | undefined): boolean {
  return Boolean(m && (m.cloud ?? m.engine === 'remote'))
}

function fromHeaders(model: string, headers: Headers): Partial<CompareResult> {
  const answered = headers.get('x-bolo-model')
  const fallback = headers.get('x-bolo-fallback')
  return {
    requestId: headers.get('x-request-id'),
    answeredBy: answered && answered !== model && fallback ? answered : null,
    fallbackReason: fallbackReason(fallback),
    cost: parseCostHeader(headers.get('x-bolo-estimated-cost')),
  }
}

/** The server's own numbers for a request. A trace that has rotated out (or an older server) reads as nothing. */
async function fromTrace(requestId: string | null | undefined, signal?: AbortSignal): Promise<Partial<CompareResult>> {
  if (!requestId) return {}
  let trace: RequestTrace
  try {
    trace = await api.requestTrace(requestId, signal)
  } catch {
    return {}
  }
  const out: Partial<CompareResult> = {
    serverMs: trace.total_ms ?? null,
    ttfaMs: trace.ttfa_ms,
    rtf: trace.rtf,
    providerMs: trace.provider?.latency_ms ?? null,
  }
  if (trace.cost) out.cost = trace.cost
  if (trace.fallback) {
    out.answeredBy = trace.fallback.answered
    out.fallbackReason = trace.fallback.attempts[0]?.code ?? null
  }
  if (trace.kind === 'tts' && trace.audio_seconds != null) out.audioSeconds = trace.audio_seconds
  return out
}

function failure(model: string, err: unknown, t0: number): CompareResult {
  return {
    model,
    status: 'error',
    error: err instanceof Error ? err.message : 'Something went wrong.',
    errorCode: err instanceof ApiError ? err.code : null,
    clientMs: performance.now() - t0,
  }
}

/** Transcribe one clip with one model. Never throws (except when aborted): a failure is a result too. */
export async function compareTranscription(
  model: string,
  audio: Blob,
  language: string | undefined,
  signal?: AbortSignal,
): Promise<CompareResult> {
  const t0 = performance.now()
  try {
    const res = await api.transcribeWithHeaders(audio, model, language, signal)
    const clientMs = performance.now() - t0
    const head = fromHeaders(model, res.headers)
    return { model, status: 'ok', text: res.text, clientMs, ...head, ...(await fromTrace(head.requestId, signal)) }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    return failure(model, err, t0)
  }
}

/** Speak one text with one model. The audio comes back as a WAV object URL. */
export async function compareSpeech(
  model: string,
  body: Omit<SpeechBody, 'model' | 'response_format'>,
  signal?: AbortSignal,
): Promise<CompareResult> {
  const t0 = performance.now()
  try {
    const res = await api.speech({ ...body, model, response_format: 'wav' }, signal)
    const firstByteMs = performance.now() - t0
    const blob = await res.blob()
    const clientMs = performance.now() - t0
    const head = fromHeaders(model, res.headers)
    // The trace is stored when the server has sent the last byte: give it a moment to land.
    await new Promise((r) => setTimeout(r, 60))
    return {
      model,
      status: 'ok',
      audioUrl: URL.createObjectURL(new Blob([blob], { type: 'audio/wav' })),
      clientMs,
      firstByteMs,
      ...head,
      ...(await fromTrace(head.requestId, signal)),
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    return failure(model, err, t0)
  }
}

/** The fastest successful result, by the server's time when every result has one, else the browser's. */
export function fastest(results: CompareResult[]): string | null {
  const ok = results.filter((r) => r.status === 'ok')
  if (ok.length < 2) return null
  const key = ok.every((r) => r.serverMs != null) ? 'serverMs' : 'clientMs'
  return [...ok].sort((a, b) => (a[key] ?? Infinity) - (b[key] ?? Infinity))[0]?.model ?? null
}

/** Models a comparison can use: installed, usable, of the task, runnable. */
export function comparable(models: CatalogModel[], task: 'stt' | 'tts'): CatalogModel[] {
  return models.filter((m) => m.task === task && m.installed && m.available !== false && m.runnable !== false)
}
