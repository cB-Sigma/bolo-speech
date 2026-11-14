import { readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { isPhoneLike } from './browser-runtime'
import {
  chooseEvictions,
  clearAllDownloads,
  dismissInterrupted,
  deleteDownload,
  downloads,
  enforceCap,
  isDownloaded,
  markLoading,
  markSettled,
  recordDownload,
  registerHolder,
  reloadInterrupted,
  interrupted,
  setStorageCap,
  touchDownload,
  type DownloadEntry,
} from './downloads'

/** A stand-in for Cache Storage that records what was asked of it. */
function fakeCaches() {
  const stores = new Map<string, Map<string, string>>()
  const calls: string[] = []
  const api = {
    open: vi.fn(async (name: string) => {
      if (!stores.has(name)) stores.set(name, new Map())
      const m = stores.get(name) as Map<string, string>
      return {
        delete: async (k: string) => {
          calls.push(`delete ${name} ${k}`)
          return m.delete(k)
        },
        keys: async () => {
          calls.push(`keys ${name}`)
          return [...m.keys()].map((url) => ({ url }))
        },
        put: async (k: string, v: string) => void m.set(k, v),
      }
    }),
    delete: vi.fn(async (name: string) => {
      calls.push(`drop ${name}`)
      return stores.delete(name)
    }),
  }
  return { api, stores, calls }
}

const entry = (id: string, bytes: number, lastUsed: number): DownloadEntry => ({
  id,
  kind: 'tts',
  label: id,
  bytes,
  cache: 'bolo-voices',
  keys: [`https://x/${id}/model.onnx`],
  addedAt: 0,
  lastUsed,
})

describe('downloads list', () => {
  let fake: ReturnType<typeof fakeCaches>
  beforeEach(() => {
    fake = fakeCaches()
    vi.stubGlobal('caches', fake.api)
    downloads.set({ entries: [], capBytes: null, evicted: [] })
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    localStorage.clear()
    registerHolder('tts', { current: () => null, release: () => undefined })
    registerHolder('stt', { current: () => null, release: () => undefined })
  })

  it('records a model with its size and keys, and keeps the list in local storage', async () => {
    await recordDownload({ id: 'a', kind: 'tts', label: 'A', bytes: 36e6, cache: 'bolo-voices', keys: ['k1', 'k2'] })
    expect(isDownloaded('a')).toBe(true)
    expect(JSON.parse(localStorage.getItem('bolo.downloads') ?? '{}').entries[0]).toMatchObject({
      id: 'a',
      bytes: 36e6,
      keys: ['k1', 'k2'],
    })
  })

  it('a republished voice replaces its old files: the keys no longer used are deleted', async () => {
    await recordDownload({ id: 'a', kind: 'tts', label: 'A', bytes: 1, cache: 'bolo-voices', keys: ['m?v=1', 'c?v=1'] })
    await recordDownload({ id: 'a', kind: 'tts', label: 'A', bytes: 1, cache: 'bolo-voices', keys: ['m?v=2', 'c?v=2'] })
    expect(fake.calls).toEqual(['delete bolo-voices m?v=1', 'delete bolo-voices c?v=1'])
  })

  it('deleting a model unloads it first and removes only its own files, without listing the cache', async () => {
    const release = vi.fn()
    registerHolder('tts', { current: () => 'a', release })
    downloads.set({ entries: [entry('a', 10, 1), entry('b', 10, 2)], capBytes: null, evicted: [] })
    await deleteDownload('a')
    expect(release).toHaveBeenCalledTimes(1)
    expect(fake.calls).toEqual(['delete bolo-voices https://x/a/model.onnx'])
    expect(downloads.get().entries.map((e) => e.id)).toEqual(['b'])
  })

  it('clear all unloads both engines and drops the model caches whole', async () => {
    const tts = vi.fn()
    const stt = vi.fn()
    registerHolder('tts', { current: () => 'a', release: tts })
    registerHolder('stt', { current: () => 'm', release: stt })
    downloads.set({ entries: [entry('a', 10, 1)], capBytes: null, evicted: [] })
    await clearAllDownloads()
    expect(tts).toHaveBeenCalled()
    expect(stt).toHaveBeenCalled()
    expect(fake.calls).toEqual(['drop bolo-voices', 'drop transformers-cache'])
    expect(downloads.get().entries).toEqual([])
  })

  it('the storage limit removes the least recently used models, never a loaded one', async () => {
    expect(chooseEvictions([entry('a', 100, 1), entry('b', 100, 2), entry('c', 100, 3)], 150, [])).toEqual(['a', 'b'])
    expect(chooseEvictions([entry('a', 100, 1), entry('b', 100, 2), entry('c', 100, 3)], 150, ['a'])).toEqual([
      'b',
      'c',
    ])
    expect(chooseEvictions([entry('a', 100, 1)], null, [])).toEqual([])

    downloads.set({ entries: [entry('a', 100, 1), entry('b', 100, 2)], capBytes: null, evicted: [] })
    touchDownload('a')
    registerHolder('stt', { current: () => null, release: () => undefined })
    expect(await setStorageCap(150)).toEqual(['b'])
    expect(downloads.get().evicted).toEqual(['b'])
    expect(await enforceCap()).toEqual([])
  })

  it('a new download over the limit pushes out the oldest', async () => {
    downloads.set({ entries: [entry('old', 300e6, 1)], capBytes: 400e6, evicted: [] })
    const removed = await recordDownload({
      id: 'new',
      kind: 'stt',
      label: 'New',
      bytes: 200e6,
      cache: 'transformers-cache',
      keys: ['u'],
    })
    expect(removed).toEqual(['old'])
    expect(downloads.get().entries.map((e) => e.id)).toEqual(['new'])
  })

  it('a load that never finished is reported at the next start until dismissed', () => {
    markLoading('stt', 'whisper-base-q8')
    reloadInterrupted()
    expect(interrupted.get()).toMatchObject({ kind: 'stt', id: 'whisper-base-q8' })
    // Still there after another reload, until dismissed.
    reloadInterrupted()
    expect(interrupted.get()).toMatchObject({ id: 'whisper-base-q8' })
    dismissInterrupted()
    reloadInterrupted()
    expect(interrupted.get()).toBeNull()
    markLoading('tts', 'v')
    markSettled('other')
    markSettled('v')
    reloadInterrupted()
    expect(interrupted.get()).toBeNull()
  })
})

describe('phones keep one model runtime', () => {
  it('treats iPhone, iPad, Android and small-memory devices as phones', () => {
    const nav = (userAgent: string, deviceMemory?: number, maxTouchPoints = 0) =>
      isPhoneLike({ userAgent, maxTouchPoints, deviceMemory })
    expect(nav('Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/141.0 Mobile Safari/537.36')).toBe(
      true,
    )
    expect(nav('Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15')).toBe(true)
    expect(nav('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15', 8, 5)).toBe(true)
    expect(nav('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/141.0 Safari/537.36', 4)).toBe(true)
    expect(nav('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/141.0 Safari/537.36', 8)).toBe(false)
    expect(nav('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/141.0 Safari/537.36')).toBe(false)
  })
})

describe('no listing of Cache Storage', () => {
  // In WebKit, Cache.keys() reads every stored body into memory: with a few models downloaded that alone was more
  // than a phone allows. Nothing in the page may call it on the model caches.
  it('no source file calls keys() on a cache', () => {
    const root = path.resolve(import.meta.dirname, '..')
    const files: string[] = []
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const p = path.join(dir, name)
        if (statSync(p).isDirectory()) {
          if (name !== 'vendor') walk(p)
        } else if (/\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name)) files.push(p)
      }
    }
    walk(root)
    const offenders = files.filter((f) =>
      /(\bcache\w*|caches\.open\([^)]*\)\))\.keys\(\)/.test(readFileSync(f, 'utf8')),
    )
    expect(offenders.map((f) => path.relative(root, f))).toEqual([])
  })
})
