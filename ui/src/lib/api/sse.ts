import { apiUrl, authHeaders, getTransport } from './config'
import type { ServerEvent } from './types'

/** Incremental parser for a `text/event-stream` body. Returns complete `data:` payloads. */
export class SseParser {
  private buffer = ''
  private data: string[] = []

  push(chunk: string): string[] {
    this.buffer += chunk
    const out: string[] = []
    let idx: number
    while ((idx = this.buffer.search(/\r?\n/)) !== -1) {
      const line = this.buffer.slice(0, idx)
      this.buffer = this.buffer.slice(idx + (this.buffer[idx] === '\r' ? 2 : 1))
      if (line === '') {
        if (this.data.length) out.push(this.data.join('\n'))
        this.data = []
      } else if (line.startsWith('data:')) {
        this.data.push(line.slice(5).replace(/^ /, ''))
      }
      // comments (":"), "event:", "id:" and "retry:" lines are ignored
    }
    return out
  }
}

export type StreamStatus = 'connecting' | 'open' | 'reconnecting' | 'unauthorized' | 'closed'

export interface EventStreamOptions {
  onEvent: (ev: ServerEvent) => void
  onStatus?: (status: StreamStatus) => void
  /** Reconnect backoff bounds in ms. */
  minDelay?: number
  maxDelay?: number
}

/**
 * Subscribes to `/api/v1/events` with fetch streaming (so the API key can be
 * sent as a header) and reconnects with exponential backoff.
 */
export function openEventStream(opts: EventStreamOptions): () => void {
  const min = opts.minDelay ?? 500
  const max = opts.maxDelay ?? 8000
  let stopped = false
  let delay = min
  let controller: AbortController | null = null
  let timer: ReturnType<typeof setTimeout> | undefined

  const run = async (first: boolean) => {
    if (stopped) return
    opts.onStatus?.(first ? 'connecting' : 'reconnecting')
    controller = new AbortController()
    try {
      const res = await getTransport().fetch(apiUrl('api/v1/events'), {
        headers: { Accept: 'text/event-stream', ...authHeaders() },
        signal: controller.signal,
      })
      if (res.status === 401 || res.status === 403) {
        opts.onStatus?.('unauthorized')
        return // wait for a new key; restart() is called by the owner
      }
      if (!res.ok || !res.body) throw new Error(`events: ${res.status}`)
      opts.onStatus?.('open')
      delay = min
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      const parser = new SseParser()
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        for (const payload of parser.push(decoder.decode(value, { stream: true }))) {
          try {
            opts.onEvent(JSON.parse(payload) as ServerEvent)
          } catch {
            // ignore malformed payloads
          }
        }
      }
    } catch {
      // fall through to reconnect
    }
    if (stopped) return
    opts.onStatus?.('reconnecting')
    timer = setTimeout(() => void run(false), delay)
    delay = Math.min(max, delay * 2)
  }

  void run(true)
  return () => {
    stopped = true
    clearTimeout(timer)
    controller?.abort()
    opts.onStatus?.('closed')
  }
}
