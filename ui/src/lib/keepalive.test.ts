import { beforeEach, describe, expect, it } from 'vitest'
import { noteActivity, registerHolder, residentModels, type Resident } from '@/lib/downloads'
import {
  DEFAULT_KEEP_ALIVE_MINUTES,
  keepAlive,
  keepAliveSeconds,
  keepLoaded,
  reloadKeepAlive,
  setKeepAliveMinutes,
  sweepIdle,
  unloadModel,
} from '@/lib/keepalive'
import { processorText } from '@/demo/backend'

function fakeHolder(key: string, over: Partial<Resident> = {}) {
  let loaded: string | null = over.id ?? key
  const released: string[] = []
  registerHolder(key, {
    current: () => loaded,
    release: () => {
      if (loaded) released.push(loaded)
      loaded = null
    },
    describe: () =>
      loaded
        ? {
            id: loaded,
            label: loaded,
            task: 'stt',
            backend: 'wasm',
            bytes: 100e6,
            gpuBytes: 0,
            busy: false,
            loading: false,
            ...over,
          }
        : null,
  })
  return { released, load: (id: string) => (loaded = id) }
}

describe('in-browser keep-alive', () => {
  beforeEach(() => {
    localStorage.clear()
    reloadKeepAlive()
  })

  it('defaults to 5 minutes and remembers the choice', () => {
    expect(keepAlive.get().minutes).toBe(DEFAULT_KEEP_ALIVE_MINUTES)
    setKeepAliveMinutes(15)
    reloadKeepAlive()
    expect(keepAlive.get().minutes).toBe(15)
    setKeepAliveMinutes(null)
    reloadKeepAlive()
    expect(keepAlive.get().minutes).toBeNull()
  })

  it('unloads a model idle past its keep-alive, not one in use or kept loaded', () => {
    const t0 = 1_000_000
    const a = fakeHolder('ka-a', { id: 'a' })
    const b = fakeHolder('ka-b', { id: 'b', busy: true })
    const c = fakeHolder('ka-c', { id: 'c' })
    for (const id of ['a', 'b', 'c']) noteActivity(id, t0)
    keepLoaded('c')
    expect(keepAliveSeconds('c')).toBeNull()
    expect(sweepIdle(t0 + 299_000)).toEqual([])
    expect(sweepIdle(t0 + 301_000)).toEqual(['a'])
    expect(a.released).toEqual(['a'])
    expect(b.released).toEqual([])
    expect(c.released).toEqual([])
    // Unloading by hand drops "Keep loaded".
    unloadModel('c')
    expect(c.released).toEqual(['c'])
    expect(keepAlive.get().pinned).toEqual([])
    expect(residentModels().map((r) => r.id)).toEqual(['b'])
    b.load('b')
    unloadModel('b')
  })

  it('writes the processor column by bytes', () => {
    expect(processorText(0, 100)).toBe('100% CPU')
    expect(processorText(100, 100)).toBe('100% GPU')
    expect(processorText(75, 100)).toBe('25%/75% CPU/GPU')
  })
})
