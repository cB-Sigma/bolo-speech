/// <reference lib="webworker" />
import type * as Ort from 'onnxruntime-web/wasm'
import { resample } from '@/lib/audio/resample'
import { cachedDownload, lowerMemoryMaximum, ortCdnBase, RUNTIME_CACHE } from '@/lib/browser-runtime'
import { setOrtGpuPower } from '@/lib/webgpu'
import { mixWithOriginal, resampleAligned } from './dsp'
import { makeDenoiser, makeEmbedder, makeLid, makeTagger, makeTurn, makeVad, type Runner } from './runners'
import { AUDIO_CACHE, type AudioWorkerRequest, type AudioWorkerResponse } from './protocol'

/**
 * One small audio model in its own worker: a denoiser, a language identifier, a voice-activity or end-of-turn
 * detector, or a speaker embedder (src/lib/browser-audio/models.ts). ONNX Runtime Web runs the ones that are ONNX
 * graphs (on WebAssembly, or WebGPU when asked); RNNoise and TEN VAD bring their own WebAssembly builds.
 */

const scope = self as unknown as DedicatedWorkerGlobalScope
const post = (msg: AudioWorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer)
/** 256 MiB: the largest of these models is 32 MB. */
const MEMORY_PAGES = 4096

let runner: Runner | null = null
let chain: Promise<void> = Promise.resolve()

async function loadOrt(req: Extract<AudioWorkerRequest, { type: 'load' }>): Promise<typeof Ort> {
  const gpu = req.device === 'webgpu'
  const ort = gpu
    ? ((await import('onnxruntime-web/webgpu')) as unknown as typeof Ort)
    : await import('onnxruntime-web/wasm')
  const wasm = ort.env.wasm
  wasm.numThreads = req.threads
  wasm.proxy = false
  if (gpu) setOrtGpuPower(ort.env as Parameters<typeof setOrtGpuPower>[0], req.power)
  const base = req.ortWasmBase?.replace(/\/?$/, '/') ?? ortCdnBase(ort.env.versions.web ?? '')
  const variant = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded'
  const paths = { mjs: `${base}${variant}.mjs`, wasm: `${base}${variant}.wasm` }
  wasm.wasmPaths = { ...paths }
  try {
    const glue = await cachedDownload(paths.mjs, RUNTIME_CACHE, {}, { contentType: 'text/javascript' })
    const lowered = lowerMemoryMaximum(new TextDecoder().decode(glue.data), MEMORY_PAGES)
    if (lowered)
      wasm.wasmPaths = { ...paths, mjs: URL.createObjectURL(new Blob([lowered], { type: 'text/javascript' })) }
    const { data } = await cachedDownload(paths.wasm, RUNTIME_CACHE, {}, { contentType: 'application/wasm' })
    wasm.wasmBinary = data
  } catch {
    // ONNX Runtime fetches the files itself and reports the real error if this fails too.
  }
  return ort
}

async function load(req: Extract<AudioWorkerRequest, { type: 'load' }>): Promise<void> {
  const started = performance.now()
  await runner?.release()
  runner = null
  const files = new Map<string, { file: string; loaded: number; total: number }>()
  let bytes = 0
  const fetched: Record<string, ArrayBuffer> = {}
  await Promise.all(
    Object.entries(req.urls).map(async ([key, url]) => {
      const name =
        url
          .replace(/[?#].*$/, '')
          .split('/')
          .pop() ?? key
      const { data } = await cachedDownload(url, AUDIO_CACHE, {
        progress: (loaded, total) => {
          files.set(key, { file: name, loaded, total })
          post({ type: 'progress', files: [...files.values()] })
        },
        record: (record) => post({ type: 'fetch', record }),
      })
      bytes += data.byteLength
      fetched[key] = data
    }),
  )
  const needsOrt = !['rnnoise', 'ten-vad', 'energy'].includes(req.model.runner)
  const ort = needsOrt ? await loadOrt(req) : null
  const session = async (data: ArrayBuffer): Promise<Ort.InferenceSession> => {
    if (!ort) throw new Error('no runtime')
    const opts = (providers: string[]): Ort.InferenceSession.SessionOptions => ({
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    })
    if (req.device !== 'webgpu') return ort.InferenceSession.create(new Uint8Array(data), opts(['wasm']))
    // Operators WebGPU lacks run on the CPU within the same session.
    return ort.InferenceSession.create(new Uint8Array(data), opts(['webgpu', 'wasm']))
  }
  const ctx = { ort, session, files: fetched, urls: req.urls, model: req.model, features: req.features }
  const task = req.model.task
  runner =
    task === 'denoise'
      ? await makeDenoiser(ctx)
      : task === 'vad'
        ? await makeVad(ctx)
        : task === 'turn'
          ? await makeTurn(ctx)
          : task === 'embed'
            ? await makeEmbedder(ctx)
            : task === 'tag'
              ? await makeTagger(ctx)
              : await makeLid(ctx)
  await runner.warmup?.()
  post({
    type: 'ready',
    loadMs: performance.now() - started,
    device: req.device,
    bytes,
    cacheKeys: Object.values(req.urls),
  })
}

async function handle(req: AudioWorkerRequest): Promise<void> {
  if (req.type === 'load') return load(req)
  if (!runner) throw new Error('The model is not loaded')
  const started = performance.now()
  const r = runner
  if (req.type === 'denoise') {
    if (!r.denoise) throw new Error('not a denoiser')
    const rate = r.rate
    // Whole factors (16 to 48 kHz) keep the samples in step; anything else goes through the general resampler.
    const convert = (x: Float32Array, a: number, b: number) =>
      Number.isInteger(Math.max(a, b) / Math.min(a, b)) ? resampleAligned(x, a, b) : resample(x, a, b)
    const input = req.rate === rate ? req.audio : convert(req.audio, req.rate, rate)
    const enhanced = await r.denoise(input)
    const back = req.rate === rate ? enhanced : convert(enhanced, rate, req.rate)
    const fitted = new Float32Array(req.audio.length)
    fitted.set(back.subarray(0, fitted.length))
    const out = mixWithOriginal(req.audio, fitted, req.strength)
    post({ type: 'audio', id: req.id, audio: out, ms: performance.now() - started }, [out.buffer])
  } else if (req.type === 'vad') {
    if (!r.vad) throw new Error('not a voice-activity detector')
    if (req.reset) r.reset?.()
    const probs = await r.vad(req.audio)
    post({ type: 'probs', id: req.id, probs, frame: r.frame ?? 512, ms: performance.now() - started }, [probs.buffer])
  } else if (req.type === 'turn') {
    if (!r.turn) throw new Error('not an end-of-turn model')
    const value = await r.turn(req.audio)
    post({ type: 'value', id: req.id, value, ms: performance.now() - started })
  } else if (req.type === 'embed') {
    if (!r.embed) throw new Error('not a speaker embedder')
    const vector = await r.embed(req.audio)
    post({ type: 'vector', id: req.id, vector, ms: performance.now() - started }, [vector.buffer])
  } else if (req.type === 'lid') {
    if (!r.lid) throw new Error('not a language identifier')
    const ranked = await r.lid(req.audio)
    post({ type: 'langs', id: req.id, ranked: ranked.slice(0, req.top ?? 5), ms: performance.now() - started })
  } else if (req.type === 'tag') {
    const tag = r.tag
    const labels = r.labels ?? []
    if (!tag) throw new Error('not a sound-event tagger')
    const top = req.top ?? 10
    const rank = (p: Float32Array): [string, number][] =>
      Array.from(p, (v, i): [string, number] => [labels[i] ?? String(i), v])
        .sort((a, b) => b[1] - a[1])
        .slice(0, top)
    // As the server (bolo.classify): at most 30 s at once; longer recordings in 10 s windows, each class keeping
    // its highest probability in any of them.
    const spans = (size: number, hop: number): [number, number][] => {
      const n = req.audio.length
      if (n <= size) return [[0, n]]
      const out: [number, number][] = []
      for (let a = 0; a < n - Math.floor(size / 4); a += hop) out.push([a, Math.min(n, a + size)])
      return out
    }
    const R = 16000
    const long = req.audio.length > 30 * R
    const whole = spans(long ? 10 * R : 30 * R, 10 * R)
    let best: Float32Array | null = null
    for (const [a, b] of whole) {
      const p = await tag(req.audio.subarray(a, b))
      if (!best) best = p
      else for (let i = 0; i < best.length; i++) best[i] = Math.max(best[i] as number, p[i] ?? 0)
    }
    const windows: { start: number; end: number; ranked: [string, number][] }[] = []
    if (req.window) {
      const size = Math.round(req.window * R)
      for (const [a, b] of spans(size, size))
        windows.push({ start: a / R, end: b / R, ranked: rank(await tag(req.audio.subarray(a, b))) })
    }
    post({
      type: 'tags',
      id: req.id,
      ranked: rank(best ?? new Float32Array()),
      windows,
      ms: performance.now() - started,
    })
  }
}

scope.onmessage = (e: MessageEvent<AudioWorkerRequest>) => {
  const req = e.data
  chain = chain
    .then(() => handle(req))
    .catch((err: unknown) =>
      post({
        type: 'error',
        id: 'id' in req ? req.id : undefined,
        message: err instanceof Error ? err.message : String(err),
      }),
    )
}
