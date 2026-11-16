import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { device } from '@/lib/device'
import { reloadDownloads } from '@/lib/downloads'
import { BrowserCloner, chooseDevice, gpuFailures, setCloneWorkerFactories } from './engine'
import { FakeCloneWorker, FakeEmbedWorker, WEBGPU_RUN_ERROR } from './fake-workers'
import { cloneModel } from './models'
import { CLONE_RATE } from './protocol'
import { failOnRuntimeError } from './runtime-errors'

const OPTS = { language: 'en', temperature: 0.7, lsdSteps: 1, exaggeration: 0.5 }
const voice = Float32Array.from({ length: CLONE_RATE * 3 }, (_, i) => 0.2 * Math.sin(i / 20))

/** Chatterbox Turbo on Auto with a WebGPU adapter: the case seen in Chrome and Edge on Windows. */
function cloner(mode: string, opts: ConstructorParameters<typeof BrowserCloner>[0] = {}) {
  const created: { worker: FakeCloneWorker; device: string }[] = []
  setCloneWorkerFactories(
    () => {
      const w = new FakeCloneWorker(mode)
      const post = w.postMessage.bind(w)
      w.postMessage = (req) => {
        if (req.type === 'load') created.push({ worker: w, device: req.device })
        post(req)
      }
      return w
    },
    () => new FakeEmbedWorker(),
  )
  const c = new BrowserCloner(opts)
  c.setModel('chatterbox-turbo')
  return { c, created }
}

describe('voice cloning: errors reach the page, WebGPU falls back to the CPU', () => {
  const before = device.get()
  beforeEach(() => {
    localStorage.clear()
    reloadDownloads()
    device.set({
      ...before,
      webgpu: {
        available: true,
        f16: true,
        maxBufferSize: 2 ** 31,
        maxStorageBufferBindingSize: 2 ** 30,
        adapter: 'amd',
      },
    })
  })
  afterEach(() => {
    setCloneWorkerFactories(null, null)
    device.set(before)
  })

  it('a failed WebGPU run is reported and made again on the CPU, with a note', async () => {
    const { c, created } = cloner('gpu-fail')
    const r = await c.generate('k', voice, 'Hello there.', OPTS, () => undefined)
    expect(created.map((x) => x.device)).toEqual(['webgpu', 'wasm'])
    expect(r.fallback).toBe(`generation failed on WebGPU: ${WEBGPU_RUN_ERROR}`)
    expect(r.audio.length).toBeGreaterThan(0)
    expect(c.store.get().device).toBe('wasm')
    expect(c.store.get().notice).toMatch(/^Generation failed on WebGPU: failed to call OrtRun\(\).*Made on the CPU/)
    // The failed worker was let go.
    expect(created[0]!.worker.onmessage).toBeNull()
    // Auto remembers it for this model in this browser; choosing a placement again clears the visit's choice.
    expect(gpuFailures()['chatterbox-turbo']).toContain('BufferManager::Download')
    expect(chooseDevice('auto', cloneModel('chatterbox-turbo'))).toBe('wasm')
    const again = await c.generate('k', voice, 'Again.', OPTS, () => undefined)
    expect(again.fallback).toBeUndefined()
    expect(created).toHaveLength(2)
    c.dispose()
  })

  it('a model that will not load on WebGPU is loaded again in a new worker on the CPU, its download size said first', async () => {
    const { c, created } = cloner('gpu-load-fail')
    const r = await c.generate('k', voice, 'Hello there.', OPTS, () => undefined)
    expect(created.map((x) => x.device)).toEqual(['webgpu', 'wasm'])
    // The WebGPU worker (whose runtime is the WebGPU build) was let go, not reused for the CPU.
    expect(created[0]!.worker.onmessage).toBeNull()
    expect(created[1]!.worker).not.toBe(created[0]!.worker)
    expect(r.audio.length).toBeGreaterThan(0)
    expect(c.store.get().device).toBe('wasm')
    expect(c.store.get().notice).toMatch(/^Loading on WebGPU failed: .*separate 725 MB download/)
    c.dispose()
  })

  it('an error the worker cannot catch fails the request instead of leaving it waiting', async () => {
    const { c, created } = cloner('gpu-crash')
    const r = await c.generate('k', voice, 'Hello there.', OPTS, () => undefined)
    expect(r.fallback).toContain('Uncaught Error: failed to call OrtRun()')
    expect(created.map((x) => x.device)).toEqual(['webgpu', 'wasm'])
    c.dispose()
  })

  it('a WebGPU run that never comes back is caught by the stall watchdog', async () => {
    const { c, created } = cloner('gpu-hang', { stallMs: { webgpu: 150, wasm: 5000 } })
    const r = await c.generate('k', voice, 'Hello there.', OPTS, () => undefined)
    expect(r.fallback).toMatch(/^generation failed on WebGPU: stalled: no progress for \d+ s$/)
    expect(created.map((x) => x.device)).toEqual(['webgpu', 'wasm'])
    c.dispose()
  })

  it('when the CPU fails too, the page gets both errors and no endless wait', async () => {
    const { c, created } = cloner('hang', { stallMs: { webgpu: 100, wasm: 150 } })
    const failed = c.generate('k', voice, 'Hello there.', OPTS, () => undefined)
    await expect(failed).rejects.toThrow(/^generation failed on WebGPU: stalled: .*; on the CPU: stalled: no progress/)
    expect(created).toHaveLength(2)
    expect(c.store.get().notice).toMatch(/On the CPU it failed too/)
    expect(c.store.get().busy).toBe(false)
  })

  it('on the CPU an error is reported as it is, without a retry', async () => {
    const { c, created } = cloner('hang', { stallMs: { wasm: 120 } })
    c.setPlacement('wasm')
    await expect(c.generate('k', voice, 'Hi.', OPTS, () => undefined)).rejects.toThrow(/^stalled: no progress/)
    expect(created).toHaveLength(1)
    expect(c.store.get().notice).toBeNull()
  })

  it('Stop lets go of a worker that does not answer, keeping what was made', async () => {
    const { c } = cloner('hang', { stallMs: { wasm: 60_000 }, cancelGraceMs: 50 })
    c.setPlacement('wasm')
    const p = c.generate('k', voice, 'Hi.', OPTS, () => undefined)
    await vi.waitFor(() => expect(c.store.get().busy).toBe(true))
    await new Promise((r) => setTimeout(r, 80)) // the voice is taken in, the generation is under way
    c.cancel()
    const r = await p
    expect(r.cancelled).toBe(true)
    expect(c.store.get().status).toBe('idle')
  })
})

describe('failOnRuntimeError', () => {
  it('fails as soon as transformers.js logs a failed run, and puts the console back', async () => {
    const log = { error: vi.fn() }
    const original = log.error
    const never = new Promise<string>(() => undefined)
    const run = failOnRuntimeError(() => never, log)
    log.error(`An error occurred during model execution: "Error: ${WEBGPU_RUN_ERROR}".`)
    log.error('Inputs given to model:', {})
    await expect(run).rejects.toThrow(`Error: ${WEBGPU_RUN_ERROR}`)
    expect(log.error).toBe(original)
    expect(original).toHaveBeenCalledTimes(2)
  })

  it('passes results and other errors through', async () => {
    const log = { error: vi.fn() }
    await expect(failOnRuntimeError(async () => 7, log)).resolves.toBe(7)
    await expect(failOnRuntimeError(async () => Promise.reject(new Error('nope')), log)).rejects.toThrow('nope')
  })
})
