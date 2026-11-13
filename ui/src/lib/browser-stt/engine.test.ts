import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { diagnostics } from '@/lib/diagnostics'
import { fmtBytes } from '@/lib/format'
import { FakeSpeechWorker } from '@/mock/fake-stt-worker'
import { BrowserSttEngine, describeLoad, setSpeechWorkerFactory } from './engine'
import { browserModel, BROWSER_MODELS, suggestedModelFor } from './models'

describe('browser model registry', () => {
  it('lists an English model, a Hindi model and two multilingual ones, each with size and licence', () => {
    expect(BROWSER_MODELS.map((m) => m.id)).toEqual([
      'moonshine-tiny',
      'whisper-tiny-hindi',
      'whisper-tiny-q8',
      'whisper-base-q8',
    ])
    for (const m of BROWSER_MODELS) {
      expect(m.approxBytes).toBeGreaterThan(20e6)
      expect(m.license).toMatch(/MIT|Apache-2\.0/)
      expect(m.source).toMatch(/^https:\/\/huggingface\.co\//)
    }
    const whisper = browserModel('whisper-base-q8')
    for (const code of ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur', 'ne', 'as']) {
      expect(whisper.languages, code).toContain(code)
    }
    // Odia is not in Whisper's language list.
    expect(whisper.languages).not.toContain('or')
  })
  it('suggests the language-specific model when there is one, otherwise the larger multilingual model', () => {
    expect(suggestedModelFor('hi')?.id).toBe('whisper-tiny-hindi')
    expect(suggestedModelFor('ta')?.id).toBe('whisper-base-q8')
    expect(suggestedModelFor('en')?.id).toBe('moonshine-tiny')
    expect(suggestedModelFor('or')).toBeNull()
  })
  it('falls back to the default for an unknown id', () => {
    expect(browserModel('nope').id).toBe('moonshine-tiny')
  })
})

describe('BrowserSttEngine with a scripted worker', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('WebAssembly', { validate: () => true })
    diagnostics.set({ fetches: [], errors: [], events: [] })
  })
  afterEach(() => {
    setSpeechWorkerFactory(null)
    vi.useRealTimers()
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it('walks through runtime, model, compile and warm-up with bytes, then is ready', async () => {
    setSpeechWorkerFactory(() => new FakeSpeechWorker({ loadMs: 1000 }))
    const engine = new BrowserSttEngine()
    const seen: string[] = []
    engine.store.subscribe(() => {
      const text = describeLoad(engine.store.get(), fmtBytes)
      if (text && seen[seen.length - 1] !== text) seen.push(text)
    })
    const done = engine.load()
    await vi.advanceTimersByTimeAsync(1100)
    await done
    expect(engine.store.get().status).toBe('ready')
    expect(engine.store.get().stage).toBeNull()
    expect(seen[0]).toMatch(/^Downloading the runtime/)
    expect(seen).toContain('Downloading the runtime · 7.1 MB of 14.3 MB')
    expect(seen.some((s) => /^Downloading the model · [\d.]+ MB of [\d.]+ MB$/.test(s))).toBe(true)
    expect(seen).toContain('Compiling the model')
    expect(seen[seen.length - 1]).toBe('Warming up')
    // Every request is on record for the diagnostics panel.
    expect(diagnostics.get().fetches.map((f) => f.status)).toEqual([200, 200])
    expect(
      diagnostics
        .get()
        .events.map((e) => e.message)
        .join('|'),
    ).toContain('ready after')
  })

  it('flags a load that stops moving, and the single-threaded retry clears it', async () => {
    localStorage.setItem('bolo.browserStt', JSON.stringify({ stallSeconds: 3 }))
    setSpeechWorkerFactory(() => new FakeSpeechWorker({ loadMs: 1000, stall: true }))
    const engine = new BrowserSttEngine()
    void engine.load().catch(() => undefined)
    await vi.advanceTimersByTimeAsync(2000)
    expect(engine.store.get().stalled).toBeNull()
    await vi.advanceTimersByTimeAsync(3000)
    expect(engine.store.get().status).toBe('loading')
    expect(engine.store.get().stalled).toMatchObject({ stage: 'compiling' })
    expect(diagnostics.get().errors[0]?.message).toMatch(/no progress for \d+ s while compiling the model/)

    const retry = engine.retrySafe()
    await vi.advanceTimersByTimeAsync(1100)
    await retry
    expect(engine.store.get()).toMatchObject({
      status: 'ready',
      threads: 1,
      device: 'wasm',
      safeMode: true,
      stalled: null,
    })
  })

  it('reports the worker’s own error message when a requested load fails', async () => {
    setSpeechWorkerFactory(
      () => new FakeSpeechWorker({ loadMs: 500, fail: 'CompileError: WebAssembly.instantiate(): bad magic' }),
    )
    const engine = new BrowserSttEngine()
    const result = engine.load().catch((e: Error) => e.message)
    await vi.advanceTimersByTimeAsync(600)
    expect(await result).toBe('CompileError: WebAssembly.instantiate(): bad magic')
    expect(engine.store.get()).toMatchObject({
      status: 'error',
      error: 'CompileError: WebAssembly.instantiate(): bad magic',
    })
  })

  it('after an out-of-memory error it retries once on one thread before reporting anything', async () => {
    setSpeechWorkerFactory(() => new FakeSpeechWorker({ loadMs: 500, fail: 'RangeError: Out of memory' }))
    const engine = new BrowserSttEngine()
    const result = engine.load().then(
      () => 'loaded',
      (e: Error) => e.message,
    )
    await vi.advanceTimersByTimeAsync(2000)
    expect(await result).toBe('loaded')
    expect(engine.store.get()).toMatchObject({ status: 'ready', safeMode: true, threads: 1, error: null })
  })

  it('keeps quiet when a speculative preload fails, but records it', async () => {
    setSpeechWorkerFactory(() => new FakeSpeechWorker({ loadMs: 500, fail: 'TypeError: Failed to fetch' }))
    const engine = new BrowserSttEngine()
    void engine.load({ background: true }).catch(() => undefined)
    await vi.advanceTimersByTimeAsync(600)
    expect(engine.store.get()).toMatchObject({ status: 'idle', error: null })
    expect(
      diagnostics
        .get()
        .errors.map((e) => e.message)
        .join('|'),
    ).toContain('Failed to fetch')
  })

  it('switching model unloads the current one and passes the language through', async () => {
    setSpeechWorkerFactory(() => new FakeSpeechWorker({ loadMs: 100 }))
    const engine = new BrowserSttEngine()
    const first = engine.load()
    await vi.advanceTimersByTimeAsync(150)
    await first
    engine.setModel('whisper-tiny-hindi')
    expect(engine.store.get()).toMatchObject({ status: 'idle', loadMs: null })
    expect(engine.store.get().model.id).toBe('whisper-tiny-hindi')
    const result = engine.transcribe(new Float32Array(16000), { language: 'hi' })
    await vi.advanceTimersByTimeAsync(300)
    expect((await result).text).toContain('नमस्ते')
    expect((await result).language).toBe('hi')
  })
})
