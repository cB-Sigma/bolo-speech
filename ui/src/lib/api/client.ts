import { apiUrl, authHeaders, getTransport } from './config'
import type {
  BenchRequest,
  BenchResult,
  Catalog,
  Component,
  Job,
  RequestTrace,
  Sample,
  ServerInfo,
  Stats,
  Task,
} from './types'

export class ApiError extends Error {
  readonly status: number
  readonly body: unknown
  constructor(status: number, message: string, body?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
  /** Machine-readable error code, when the server sent one (`error.code` or `code`). */
  get code(): string | null {
    const b = this.body as { code?: unknown; error?: { code?: unknown } } | undefined
    const c = b?.error?.code ?? b?.code
    return typeof c === 'string' ? c : null
  }
  get unauthorized(): boolean {
    return this.status === 401 || this.status === 403
  }
}

/** Raised when the server cannot be reached at all (as opposed to an HTTP error). */
export class NetworkError extends Error {
  constructor(message = 'Server unreachable') {
    super(message)
    this.name = 'NetworkError'
  }
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === 'object') {
    const b = body as Record<string, unknown>
    const detail = b.detail ?? b.message ?? b.error
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object' && typeof (detail as Record<string, unknown>).message === 'string') {
      return (detail as Record<string, string>).message as string
    }
  }
  return fallback
}

export interface RequestOptions {
  method?: string
  query?: Record<string, string | number | boolean | undefined | null>
  json?: unknown
  body?: BodyInit
  signal?: AbortSignal
  headers?: Record<string, string>
}

export async function request(path: string, opts: RequestOptions = {}): Promise<Response> {
  const headers: Record<string, string> = { ...authHeaders(), ...opts.headers }
  let body = opts.body
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(opts.json)
  }
  let res: Response
  try {
    res = await getTransport().fetch(apiUrl(path, opts.query), {
      method: opts.method ?? 'GET',
      headers,
      body,
      signal: opts.signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new NetworkError(err instanceof Error ? err.message : undefined)
  }
  if (!res.ok) {
    let parsed: unknown
    try {
      parsed = await res.clone().json()
    } catch {
      parsed = undefined
    }
    throw new ApiError(res.status, errorMessage(parsed, `${res.status} ${res.statusText}`.trim()), parsed)
  }
  return res
}

async function json<T>(path: string, opts?: RequestOptions): Promise<T> {
  const res = await request(path, opts)
  if (res.status === 204) return undefined as T
  const text = await res.text()
  return (text ? JSON.parse(text) : undefined) as T
}

export interface CatalogQuery {
  task?: Task | ''
  q?: string
  installed?: boolean
}

export const api = {
  info: (signal?: AbortSignal) => json<ServerInfo>('api/v1/info', { signal }),
  catalog: (q: CatalogQuery = {}, signal?: AbortSignal) =>
    json<Catalog>('api/v1/catalog', { query: { task: q.task, q: q.q, installed: q.installed }, signal }),
  pull: (id: string) => json<{ job_id: string }>('api/v1/models/pull', { method: 'POST', json: { id } }),
  job: <R = unknown>(jobId: string, signal?: AbortSignal) =>
    json<Job<R>>(`api/v1/jobs/${encodeURIComponent(jobId)}`, { signal }),
  load: (id: string) => json<unknown>(`api/v1/models/${encodeURIComponent(id)}/load`, { method: 'POST' }),
  unload: (id: string) => json<unknown>(`api/v1/models/${encodeURIComponent(id)}/unload`, { method: 'POST' }),
  remove: (id: string) => json<void>(`api/v1/models/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  stats: (signal?: AbortSignal) => json<Stats>('api/v1/stats', { signal }),
  requests: (q: { limit?: number; kind?: string; model?: string } = {}, signal?: AbortSignal) =>
    json<{ requests: RequestTrace[] }>('api/v1/requests', { query: q, signal }),
  /** Optional components. Older servers have no such route: that reads as an empty list. */
  components: async (signal?: AbortSignal): Promise<{ components: Component[] }> => {
    try {
      return await json<{ components: Component[] }>('api/v1/components', { signal })
    } catch (err) {
      if (err instanceof ApiError && (err.status === 404 || err.status === 501)) return { components: [] }
      throw err
    }
  },
  installComponent: (id: string) =>
    json<{ job_id: string }>(`api/v1/components/${encodeURIComponent(id)}/install`, {
      method: 'POST',
      json: { accept_license: true },
    }),
  samples: (signal?: AbortSignal) => json<{ samples: Sample[] }>('api/v1/samples', { signal }),
  bench: (req: BenchRequest) => json<{ job_id: string }>('api/v1/bench', { method: 'POST', json: req }),
  benchJob: (jobId: string, signal?: AbortSignal) => api.job<BenchResult>(jobId, signal),

  /** Round-trip time of a tiny request, used as the network share of latency. */
  async ping(signal?: AbortSignal): Promise<number> {
    const t0 = performance.now()
    await request('healthz', { signal })
    return performance.now() - t0
  },

  /** Fetch a sample (or any server-relative URL) as bytes. */
  async bytes(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
    const res = await request(url, { signal })
    return res.arrayBuffer()
  },

  /** One-shot synthesis through the OpenAI-compatible route. */
  async speech(
    body: { model: string; input: string; voice?: string; response_format?: string },
    signal?: AbortSignal,
  ): Promise<Response> {
    return request('v1/audio/speech', { method: 'POST', json: body, signal })
  },

  /** One-shot transcription through the OpenAI-compatible route. */
  async transcribe(file: Blob, model: string, language?: string, signal?: AbortSignal): Promise<{ text: string }> {
    const form = new FormData()
    form.set('file', file, 'audio.wav')
    form.set('model', model)
    form.set('response_format', 'json')
    if (language) form.set('language', language)
    const res = await request('v1/audio/transcriptions', { method: 'POST', body: form, signal })
    return res.json() as Promise<{ text: string }>
  },
}
