import { diag } from '@/lib/diagnostics'
import { idleSeconds, noteActivity, releaseModel, residentModels } from '@/lib/downloads'
import { Store } from '@/lib/store'

/**
 * Keep-alive for the models this page holds, like a server's: a model that
 * has been idle for the set number of minutes is unloaded (its worker ends
 * and its memory goes back to the device). "Keep loaded" exempts one model
 * until it is unloaded by hand. The choice is kept in this browser only.
 */

export interface KeepAliveState {
  /** Unload after this many idle minutes; null: never. */
  minutes: number | null
  /** Models kept loaded until they are unloaded by hand. */
  pinned: string[]
}

const KEY = 'bolo.keepAlive'
export const DEFAULT_KEEP_ALIVE_MINUTES = 5
/** Choices offered (minutes; null: never). */
export const KEEP_ALIVE_CHOICES: (number | null)[] = [1, 2, 5, 10, 15, 30, 60, null]

function read(): KeepAliveState {
  try {
    const raw = globalThis.localStorage?.getItem(KEY)
    if (raw) {
      const v = JSON.parse(raw) as Partial<KeepAliveState>
      return {
        minutes:
          v.minutes === null
            ? null
            : typeof v.minutes === 'number' && v.minutes > 0
              ? v.minutes
              : DEFAULT_KEEP_ALIVE_MINUTES,
        pinned: [],
      }
    }
  } catch {
    // unreadable: defaults
  }
  return { minutes: DEFAULT_KEEP_ALIVE_MINUTES, pinned: [] }
}

export const keepAlive = new Store<KeepAliveState>(read())

keepAlive.subscribe(() => {
  try {
    globalThis.localStorage?.setItem(KEY, JSON.stringify({ minutes: keepAlive.get().minutes }))
  } catch {
    // blocked: this visit only
  }
})

/** Reload from storage (tests). */
export function reloadKeepAlive(): void {
  keepAlive.set(read())
}

export function setKeepAliveMinutes(minutes: number | null): void {
  keepAlive.patch({ minutes })
}

export function keepLoaded(id: string): void {
  const { pinned } = keepAlive.get()
  if (!pinned.includes(id)) keepAlive.patch({ pinned: [...pinned, id] })
}

export function isPinned(id: string): boolean {
  return keepAlive.get().pinned.includes(id)
}

/** Unload by hand: also drops "Keep loaded". */
export function unloadModel(id: string): boolean {
  const { pinned } = keepAlive.get()
  if (pinned.includes(id)) keepAlive.patch({ pinned: pinned.filter((p) => p !== id) })
  return releaseModel(id)
}

/** Seconds a model stays loaded once idle; null: kept loaded. */
export function keepAliveSeconds(id: string): number | null {
  const { minutes, pinned } = keepAlive.get()
  if (minutes == null || pinned.includes(id)) return null
  return minutes * 60
}

/** Unload every model that has been idle longer than its keep-alive. Returns the ids unloaded. */
export function sweepIdle(now = Date.now()): string[] {
  const out: string[] = []
  for (const r of residentModels()) {
    if (r.busy || r.loading) {
      // Idle time counts from the end of the last request.
      noteActivity(r.id, now)
      continue
    }
    const ttl = keepAliveSeconds(r.id)
    if (ttl == null) continue
    if (idleSeconds(r.id, now) >= ttl) {
      releaseModel(r.id)
      out.push(r.id)
    }
  }
  if (out.length) diag.event('model', `unloaded after ${keepAlive.get().minutes} idle min: ${out.join(', ')}`)
  // Pins of models no longer loaded mean nothing.
  const loaded = new Set(residentModels().map((r) => r.id))
  const { pinned } = keepAlive.get()
  if (pinned.some((p) => !loaded.has(p))) keepAlive.patch({ pinned: pinned.filter((p) => loaded.has(p)) })
  return out
}

let timer: ReturnType<typeof setInterval> | null = null

/** Check for idle models every few seconds (started once by the demo). */
export function startKeepAlive(everyMs = 5000): () => void {
  if (timer) clearInterval(timer)
  timer = setInterval(() => sweepIdle(), everyMs)
  return () => {
    if (timer) clearInterval(timer)
    timer = null
  }
}
