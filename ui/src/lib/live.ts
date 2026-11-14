import type { QueryClient } from '@tanstack/react-query'
import { openEventStream, type StreamStatus } from './api/sse'
import type { Job, RequestTrace, ServerEvent, Stats } from './api/types'
import { Store } from './store'

export interface StatsPoint {
  /** epoch ms */
  t: number
  stats: Stats
}

export interface LiveState {
  status: StreamStatus
  /** Last time any event arrived (epoch ms). */
  lastEventAt: number | null
  stats: Stats | null
  history: StatsPoint[]
  requests: RequestTrace[]
  jobs: Record<string, Job>
  logs: { t: number; level: string; message: string }[]
}

const HISTORY_MS = 15 * 60_000
const MAX_REQUESTS = 500

/**
 * The live feed: one SSE subscription shared by every screen. Keeps a rolling
 * window of stats samples and request traces in memory.
 */
export class LiveFeed {
  readonly store = new Store<LiveState>({
    status: 'connecting',
    lastEventAt: null,
    stats: null,
    history: [],
    requests: [],
    jobs: {},
    logs: [],
  })
  private stop: (() => void) | null = null
  private queryClient: QueryClient | null = null

  start(queryClient: QueryClient): void {
    this.queryClient = queryClient
    this.restart()
  }

  restart(): void {
    this.stop?.()
    this.stop = openEventStream({
      onEvent: (ev) => this.onEvent(ev),
      onStatus: (status) => {
        if (status !== 'closed') this.store.patch({ status })
      },
    })
  }

  dispose(): void {
    this.stop?.()
    this.stop = null
  }

  /** Merge traces fetched over HTTP (initial load) with those already streamed. */
  seedRequests(traces: RequestTrace[]): void {
    const s = this.store.get()
    const seen = new Set(s.requests.map((r) => r.id))
    const merged = [...s.requests, ...traces.filter((t) => !seen.has(t.id))]
    merged.sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at))
    this.store.patch({ requests: merged.slice(0, MAX_REQUESTS) })
  }

  private onEvent(ev: ServerEvent): void {
    const now = Date.now()
    const s = this.store.get()
    switch (ev.type) {
      case 'stats': {
        const { type: _type, ...stats } = ev
        void _type
        const history = [...s.history, { t: now, stats }]
        const cutoff = now - HISTORY_MS
        let drop = 0
        while (drop < history.length && (history[drop] as StatsPoint).t < cutoff) drop++
        this.store.patch({ stats, history: drop ? history.slice(drop) : history, lastEventAt: now })
        break
      }
      case 'request': {
        const { type: _type, ...trace } = ev
        void _type
        if (s.requests.some((r) => r.id === trace.id)) break
        this.store.patch({ requests: [trace, ...s.requests].slice(0, MAX_REQUESTS), lastEventAt: now })
        break
      }
      case 'job': {
        const { type: _type, ...job } = ev
        void _type
        this.store.patch({ jobs: { ...s.jobs, [job.id]: job }, lastEventAt: now })
        if (job.kind === 'pull') {
          if (job.state === 'done' || job.state === 'error')
            void this.queryClient?.invalidateQueries({ queryKey: ['catalog'] })
        }
        break
      }
      case 'model':
        void this.queryClient?.invalidateQueries({ queryKey: ['catalog'] })
        this.store.patch({ lastEventAt: now })
        break
      case 'component':
        void this.queryClient?.invalidateQueries({ queryKey: ['components'] })
        void this.queryClient?.invalidateQueries({ queryKey: ['catalog'] })
        this.store.patch({ lastEventAt: now })
        break
      case 'wordlist':
        void this.queryClient?.invalidateQueries({ queryKey: ['wordlists'] })
        this.store.patch({ lastEventAt: now })
        break
      case 'log':
        this.store.patch({
          logs: [{ t: now, level: ev.level, message: ev.message }, ...s.logs].slice(0, 200),
          lastEventAt: now,
        })
        break
      default:
        break
    }
  }
}

export const live = new LiveFeed()
