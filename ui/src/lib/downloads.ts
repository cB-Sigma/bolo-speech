import { diag } from '@/lib/diagnostics'
import { Store } from '@/lib/store'

/**
 * The models this browser has downloaded for the in-page recogniser and
 * synthesiser, kept as a small list in local storage.
 *
 * Why a list and not a look at Cache Storage: in WebKit (every browser on
 * iPhone and iPad, and Safari) `Cache.keys()` reads every stored response
 * body into the browser's networking process. The page used to call it once
 * per known model at start-up, so the memory it needed grew with every model
 * downloaded: with five or so it was more than a phone allows, and the page
 * was closed on every visit. Here nothing is read from the cache to find out
 * what is in it; a model's files are deleted by their exact keys, and
 * "Clear all" deletes whole caches.
 */

export type ModelKind = 'tts' | 'stt' | 'diarize' | 'llm' | 'clone' | 'audio'

export interface DownloadEntry {
  id: string
  kind: ModelKind
  label: string
  /** Bytes on disk (as downloaded). */
  bytes: number
  /** Cache Storage bucket that holds the files. */
  cache: string
  /** Exact cache keys (URLs) of the model's own files. */
  keys: string[]
  addedAt: number
  lastUsed: number
}

export interface DownloadsState {
  entries: DownloadEntry[]
  /** Keep the downloads under this many bytes by removing the least recently used ones; null = no limit. */
  capBytes: number | null
  /** Models removed by the limit most recently (for a short notice). */
  evicted: string[]
}

const KEY = 'bolo.downloads'
const LOADING_KEY = 'bolo.loadingModel'

/** Choices offered for the storage limit, in bytes (null: no limit). */
export const CAP_CHOICES: (number | null)[] = [null, 250e6, 500e6, 1e9, 2e9, 4e9]

function read(): DownloadsState {
  try {
    const raw = globalThis.localStorage?.getItem(KEY)
    if (raw) {
      const s = JSON.parse(raw) as Partial<DownloadsState>
      return {
        entries: Array.isArray(s.entries) ? s.entries.filter((e) => e && typeof e.id === 'string') : [],
        capBytes: typeof s.capBytes === 'number' && s.capBytes > 0 ? s.capBytes : null,
        evicted: [],
      }
    }
  } catch {
    // unreadable: start empty
  }
  return { entries: [], capBytes: null, evicted: [] }
}

export const downloads = new Store<DownloadsState>(read())

downloads.subscribe(() => {
  const { entries, capBytes } = downloads.get()
  try {
    globalThis.localStorage?.setItem(KEY, JSON.stringify({ entries, capBytes }))
  } catch {
    // storage full or blocked: the list lives for this visit only
  }
})

/** Re-read the list (tests, and another tab having changed it). */
export function reloadDownloads(): void {
  downloads.set(read())
}

/** Where an in-page model's weights live while it is loaded. */
export type Backend = 'webgpu' | 'wasm' | 'split'

/** One model held in the page right now, as the Loaded models view shows it. */
export interface Resident {
  id: string
  label: string
  task: 'stt' | 'tts' | 'diarize' | 'llm' | 'vad' | 's2s' | 'denoise' | 'lid' | 'turn' | 'embed'
  backend: Backend
  /** Estimated memory it takes while loaded (a page cannot measure one model's share), bytes. */
  bytes: number
  /** Of `bytes`, the part placed on WebGPU. */
  gpuBytes: number
  /** A request is running. */
  busy: boolean
  /** Still downloading or compiling. */
  loading: boolean
  /** How the model is placed, in words ("encoder on WebGPU, decoder on the CPU"). */
  placement?: string
}

interface Holder {
  /** Id of the model it has loaded or is loading; null when idle. */
  current: () => string | null
  release: () => void
  /** The loaded model for the Loaded models view; null when idle. */
  describe?: () => Resident | null
}

const holders = new Map<string, Holder>()

/**
 * Everything that holds a model in the page registers here (the recogniser, the synthesiser, diarization, the
 * language model, …) so a model can be unloaded before its files are deleted, listed as loaded, and unloaded
 * when it has been idle too long. `key` names the holder; a later registration with the same key replaces it.
 */
export function registerHolder(key: string, holder: Holder): void {
  holders.set(key, holder)
}

/** Ids of the models loaded (or loading) right now. */
export function residentIds(): string[] {
  return [...holders.values()].map((h) => h.current()).filter((x): x is string => x != null)
}

/** The models loaded (or loading) right now, described by their holders. */
export function residentModels(): Resident[] {
  return [...holders.values()]
    .map((h) => (h.current() == null ? null : (h.describe?.() ?? null)))
    .filter((x): x is Resident => x != null)
}

/** Unload one model, if it is loaded. Returns whether something was released. */
export function releaseModel(id: string): boolean {
  let released = false
  for (const h of holders.values())
    if (h.current() === id) {
      h.release()
      released = true
    }
  return released
}

/** Unload every in-page model (workers terminated, memory returned). */
export function releaseAllModels(): void {
  for (const h of holders.values()) h.release()
}

/**
 * Terminate the model workers when the page is going away for good (reload, another site, closing the tab), so
 * their WebAssembly memory and WebGPU devices are handed back right then instead of whenever the browser tears
 * the old page down. A page kept in the back/forward cache (`persisted`) keeps its models.
 */
export function releaseModelsOnPageHide(target: Pick<Window, 'addEventListener'> | undefined = globalThis): void {
  target?.addEventListener('pagehide', (e) => {
    if (!(e as PageTransitionEvent).persisted) releaseAllModels()
  })
}

const lastActive = new Map<string, number>()

/** A model was just used (or finished loading): its idle time starts again. */
export function noteActivity(id: string, now = Date.now()): void {
  lastActive.set(id, now)
}

/** Seconds since the model was last used in this visit (0 when never noted). */
export function idleSeconds(id: string, now = Date.now()): number {
  const t = lastActive.get(id)
  return t == null ? 0 : Math.max(0, (now - t) / 1000)
}

export function isDownloaded(id: string): boolean {
  return downloads.get().entries.some((e) => e.id === id)
}

export function downloadedIds(kind: ModelKind): string[] {
  return downloads
    .get()
    .entries.filter((e) => e.kind === kind)
    .map((e) => e.id)
}

export function totalBytes(entries: DownloadEntry[] = downloads.get().entries): number {
  return entries.reduce((a, e) => a + e.bytes, 0)
}

async function deleteKeys(cacheName: string, keys: string[]): Promise<void> {
  if (typeof caches === 'undefined' || keys.length === 0) return
  try {
    const cache = await caches.open(cacheName)
    for (const k of keys) await cache.delete(k)
  } catch {
    // no Cache Storage here: nothing to remove
  }
}

/**
 * Note a model as downloaded (after it loaded). Keys of an earlier copy that
 * are no longer used (a voice republished with new files) are deleted. Then
 * the storage limit is applied.
 */
export async function recordDownload(
  input: Omit<DownloadEntry, 'addedAt' | 'lastUsed'> & {
    /** Add these keys to the ones already listed (a shared model whose voices each bring a file). */
    merge?: boolean
  },
): Promise<string[]> {
  const { merge, ...entry } = input
  const now = Date.now()
  noteActivity(entry.id, now)
  const s = downloads.get()
  const previous = s.entries.find((e) => e.id === entry.id)
  const keys = merge
    ? [...new Set([...(previous?.keys ?? []), ...entry.keys])]
    : entry.keys.length
      ? entry.keys
      : (previous?.keys ?? [])
  const next: DownloadEntry = {
    ...entry,
    keys,
    bytes: entry.bytes || previous?.bytes || 0,
    addedAt: previous?.addedAt ?? now,
    lastUsed: now,
  }
  downloads.patch({ entries: [...s.entries.filter((e) => e.id !== entry.id), next] })
  if (previous && previous.cache === next.cache) {
    const stale = previous.keys.filter((k) => !keys.includes(k))
    await deleteKeys(previous.cache, stale)
  }
  return enforceCap()
}

/** Mark a model as just used (for least-recently-used removal). */
export function touchDownload(id: string): void {
  noteActivity(id)
  const s = downloads.get()
  if (!s.entries.some((e) => e.id === id)) return
  downloads.patch({ entries: s.entries.map((e) => (e.id === id ? { ...e, lastUsed: Date.now() } : e)) })
}

/** Unload the model if it is loaded and delete its files. */
export async function deleteDownload(id: string): Promise<void> {
  const entry = downloads.get().entries.find((e) => e.id === id)
  for (const h of holders.values()) if (h.current() === id) h.release()
  if (entry) await deleteKeys(entry.cache, entry.keys)
  downloads.patch({ entries: downloads.get().entries.filter((e) => e.id !== id) })
  diag.event('storage', `removed ${id}`)
}

/** Caches that hold model files (not the page itself, which the service worker keeps). */
export const MODEL_CACHES = ['bolo-voices', 'transformers-cache']

/** Unload everything and delete every downloaded model, listed or not. */
export async function clearAllDownloads(): Promise<void> {
  releaseAllModels()
  if (typeof caches !== 'undefined') {
    for (const name of MODEL_CACHES) {
      try {
        await caches.delete(name)
      } catch {
        // ignore
      }
    }
  }
  downloads.patch({ entries: [], evicted: [] })
  diag.event('storage', 'removed every downloaded model')
}

/** Set (or remove, with null) the storage limit and apply it at once. */
export function setStorageCap(bytes: number | null): Promise<string[]> {
  downloads.patch({ capBytes: bytes })
  return enforceCap()
}

/**
 * Which downloads to remove to get under `cap`: least recently used first,
 * never one that is loaded right now.
 */
export function chooseEvictions(entries: DownloadEntry[], cap: number | null, protect: string[]): string[] {
  if (cap == null) return []
  let total = totalBytes(entries)
  const out: string[] = []
  for (const e of [...entries].sort((a, b) => a.lastUsed - b.lastUsed)) {
    if (total <= cap) break
    if (protect.includes(e.id)) continue
    out.push(e.id)
    total -= e.bytes
  }
  return out
}

/** Apply the storage limit; returns the ids removed. */
export async function enforceCap(): Promise<string[]> {
  const s = downloads.get()
  const victims = chooseEvictions(s.entries, s.capBytes, residentIds())
  for (const id of victims) await deleteDownload(id)
  if (victims.length) {
    downloads.patch({ evicted: victims })
    diag.event('storage', `storage limit: removed ${victims.join(', ')}`)
  }
  return victims
}

export interface StorageUse {
  usage: number | null
  quota: number | null
  persisted: boolean | null
}

/** What the browser says this site stores, and may store. */
export async function storageUse(): Promise<StorageUse> {
  const storage = globalThis.navigator?.storage
  let usage: number | null = null
  let quota: number | null = null
  let persisted: boolean | null = null
  try {
    const e = await storage?.estimate?.()
    usage = e?.usage ?? null
    quota = e?.quota ?? null
  } catch {
    // not offered
  }
  try {
    persisted = (await storage?.persisted?.()) ?? null
  } catch {
    // not offered
  }
  return { usage, quota, persisted }
}

/**
 * A load that never finished: set before a model starts loading, cleared when
 * it is ready, fails or is unloaded. Still set at the next start-up means the
 * page was closed mid-load, most likely by the browser for using too much
 * memory; the page then loads nothing by itself and offers to clear downloads.
 */
export function markLoading(kind: ModelKind, id: string): void {
  // The visitor asked for a model again: the notice about the last attempt has done its job.
  if (interrupted.get()) interrupted.set(null)
  try {
    globalThis.localStorage?.setItem(LOADING_KEY, JSON.stringify({ kind, id, at: Date.now() }))
  } catch {
    // ignore
  }
}

export function markSettled(id?: string): void {
  try {
    const raw = globalThis.localStorage?.getItem(LOADING_KEY)
    if (!raw) return
    if (id && (JSON.parse(raw) as { id?: string }).id !== id) return
    globalThis.localStorage?.removeItem(LOADING_KEY)
  } catch {
    // ignore
  }
}

export interface InterruptedLoad {
  kind: ModelKind
  id: string
  at: number
}

function readInterrupted(): InterruptedLoad | null {
  try {
    const raw = globalThis.localStorage?.getItem(LOADING_KEY)
    if (!raw) return null
    const v = JSON.parse(raw) as Partial<InterruptedLoad>
    return typeof v.id === 'string' &&
      (v.kind === 'tts' ||
        v.kind === 'stt' ||
        v.kind === 'diarize' ||
        v.kind === 'llm' ||
        v.kind === 'clone' ||
        v.kind === 'audio')
      ? { kind: v.kind, id: v.id, at: Number(v.at) || 0 }
      : null
  } catch {
    return null
  }
}

/**
 * The load that was under way when the page last closed. It stays until the
 * visitor dismisses it or loads a model again (the page may reload once by
 * itself on a first visit, for isolation, and must not lose it).
 */
export const interrupted = new Store<InterruptedLoad | null>(readInterrupted())

export function dismissInterrupted(): void {
  interrupted.set(null)
  markSettled()
}

/** Re-read the interrupted-load marker (tests). */
export function reloadInterrupted(): void {
  interrupted.set(readInterrupted())
}

/**
 * Make room for a download of `bytes`: remove the least recently used models
 * (never a loaded one) until the browser's quota has that much free, or
 * nothing more can go. Returns the ids removed.
 */
export async function freeSpaceFor(bytes: number): Promise<string[]> {
  const { usage, quota } = await storageUse()
  if (quota == null) return []
  let free = quota - (usage ?? 0)
  const removed: string[] = []
  const protect = residentIds()
  for (const e of [...downloads.get().entries].sort((a, b) => a.lastUsed - b.lastUsed)) {
    if (free >= bytes) break
    if (protect.includes(e.id)) continue
    await deleteDownload(e.id)
    removed.push(e.id)
    free += e.bytes
  }
  return removed
}
