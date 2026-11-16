import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { downloads, reloadDownloads, residentModels } from '@/lib/downloads'
import { BrowserCloner, chooseDevice, setCloneWorkerFactories, SimilarityScorer } from './engine'
import { FakeCloneWorker, FakeEmbedWorker } from './fake-workers'
import { cloneModel } from './models'
import { CLONE_RATE } from './protocol'
import { cosine } from './similarity'

const OPTS = { language: 'en', temperature: 0.7, lsdSteps: 1, exaggeration: 0.5 }
const voice = Float32Array.from({ length: CLONE_RATE * 6 }, (_, i) => 0.2 * Math.sin(i / 20))

describe('voice-cloning engine (scripted workers)', () => {
  let created: FakeCloneWorker[]
  beforeEach(() => {
    localStorage.clear()
    reloadDownloads()
    created = []
    setCloneWorkerFactories(
      () => {
        const w = new FakeCloneWorker()
        created.push(w)
        return w
      },
      () => new FakeEmbedWorker(),
    )
  })
  afterEach(() => setCloneWorkerFactories(null, null))

  it('loads on first use, streams audio, and lists the model as downloaded and loaded', async () => {
    const c = new BrowserCloner()
    const pieces: Float32Array[] = []
    const r = await c.generate('k1', voice, 'Hello there, this is a test.', OPTS, (a) => pieces.push(a))
    expect(pieces.length).toBe(2)
    expect(r.audio.length).toBe(pieces[0]!.length + pieces[1]!.length)
    expect(r.firstAudioMs).not.toBeNull()
    expect(r.voiceMs).toBeGreaterThan(0)
    expect(r.rtf).toBeGreaterThan(0)
    expect(c.store.get().status).toBe('ready')
    expect(downloads.get().entries.map((e) => [e.id, e.kind])).toContainEqual(['clone-pocket-en', 'clone'])
    expect(residentModels().map((m) => m.id)).toContain('clone-pocket-en')
    // The same voice is not taken in again.
    const again = await c.generate('k1', voice, 'Again.', OPTS, () => undefined)
    expect(again.voiceMs).toBe(0)
    c.dispose()
  })

  it('unloads the current model on switch', async () => {
    const c = new BrowserCloner()
    await c.load()
    c.setModel('chatterbox-turbo')
    expect(c.store.get().status).toBe('idle')
    expect(residentModels().map((m) => m.id)).not.toContain('clone-pocket-en')
    await c.load()
    expect(created).toHaveLength(2)
    expect(residentModels().map((m) => m.id)).toContain('clone-chatterbox-turbo')
    c.dispose()
  })

  it('stops a generation when asked', async () => {
    const c = new BrowserCloner()
    const p = c.generate('k', voice, 'Some longer text to speak here.', OPTS, () => c.cancel())
    const r = await p
    expect(r.cancelled).toBe(true)
    c.dispose()
  })

  it('recovers from running out of memory: retries on one CPU thread', async () => {
    setCloneWorkerFactories(
      () => new FakeCloneWorker('oom'),
      () => new FakeEmbedWorker(),
    )
    const c = new BrowserCloner()
    await c.load()
    expect(c.store.get().status).toBe('ready')
    expect(c.store.get().threads).toBe(1)
    expect(c.store.get().device).toBe('wasm')
    c.dispose()
  })

  it('scores similarity with the embedding worker', async () => {
    const s = new SimilarityScorer()
    const a = await s.embed(voice.subarray(0, 16_000 * 3))
    const b = await s.embed(voice.subarray(16_000, 16_000 * 4))
    const silence = await s.embed(new Float32Array(16_000))
    expect(cosine(a, b)).toBeGreaterThan(0.99)
    expect(cosine(a, silence)).toBeLessThan(cosine(a, b))
    expect(downloads.get().entries.map((e) => e.id)).toContain('clone-similarity-campplus')
    s.dispose()
  })

  it('places big models on WebGPU on Auto only when there is an adapter', () => {
    expect(chooseDevice('wasm', cloneModel('chatterbox-turbo'))).toBe('wasm')
    expect(chooseDevice('webgpu', cloneModel('pocket-en'))).toBe('webgpu')
    // jsdom has no WebGPU adapter.
    expect(chooseDevice('auto', cloneModel('chatterbox-turbo'))).toBe('wasm')
    expect(chooseDevice('auto', cloneModel('moss-tts-nano'))).toBe('wasm')
  })
})
