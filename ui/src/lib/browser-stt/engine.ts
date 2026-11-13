import { Store } from '@/lib/store'
import {
  claimRuntime,
  isOutOfMemory,
  releaseOtherRuntimes,
  releaseRuntime,
  requireOneRuntime,
} from '@/lib/browser-runtime'
import { diag } from '@/lib/diagnostics'
import { recordBrowserTrace } from './traces'
import { detectSupport, probeWebGpu } from '@/lib/support'
import {
  browserModel,
  BROWSER_MODELS,
  DEFAULT_BROWSER_MODEL,
  readBrowserSttConfig,
  type BrowserDevice,
  type BrowserModel,
  type FileProgress,
  type LoadStage,
  type WorkerRequest,
  type WorkerResponse,
} from './models'

export type EngineStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface EngineState {
  status: EngineStatus
  model: BrowserModel
  device: BrowserDevice | null
  threads: number
  files: FileProgress[]
  /** 0..1 over all model files seen so far. */
  progress: number
  /** What the loader is doing; null when not loading. */
  stage: LoadStage | null
  /** Bytes of the ONNX Runtime binary fetched so far, when it is being downloaded. */
  runtime: { loaded: number; total: number } | null
  /** Set when nothing has moved for a while: which stage, and for how long. */
  stalled: { stage: LoadStage | null; seconds: number } | null
  /** The last load was started in the reduced mode (one thread, WebAssembly only). */
  safeMode: boolean
  loadMs: number | null
  /** Whether the current model's weights are in the browser cache; null until checked. */
  cached: boolean | null
  /** Ids of all models whose weights are in the browser cache. */
  cachedIds: string[]
  error: string | null
}

export interface LoadOptions {
  /** One thread, WebAssembly only, and the runtime fetched the plain way. */
  safe?: boolean
  /** A speculative preload: a failure is recorded quietly instead of becoming the visible error. */
  background?: boolean
}

export interface TranscribeResult {
  text: string
  ms: number
  language: string | null
}

/** Whether the model's weights are already in the browser's Cache Storage. */
export async function isModelCached(model: BrowserModel): Promise<boolean> {
  try {
    if (typeof caches === 'undefined') return false
    const cache = await caches.open('transformers-cache')
    const keys = await cache.keys()
    return keys.some((k) => k.url.includes(model.repo) && k.url.endsWith('.onnx'))
  } catch {
    return false
  }
}

/** Minimal surface of a Worker that the engine relies on (the tests substitute a scripted one). */
export interface SpeechWorker {
  onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null
  onerror: ((e: ErrorEvent) => void) | null
  postMessage(message: WorkerRequest, transfer?: Transferable[]): void
  terminate(): void
}

let workerFactory: (() => SpeechWorker | Promise<SpeechWorker>) | null = null

/** Replace the worker (mock build and unit tests). */
export function setSpeechWorkerFactory(factory: (() => SpeechWorker | Promise<SpeechWorker>) | null): void {
  workerFactory = factory
}

const STAGE_TEXT: Record<LoadStage, string> = {
  runtime: 'downloading the runtime',
  model: 'downloading the model',
  compiling: 'compiling the model',
  warmup: 'warming up',
}

/**
 * Owns the web worker that runs speech recognition in the page via ONNX
 * Runtime Web. The worker (and the ~1 MB of runtime JS behind it) is only
 * created when `load()` is first called.
 */
export class BrowserSttEngine {
  readonly store: Store<EngineState>
  private worker: SpeechWorker | null = null
  private loading: Promise<void> | null = null
  private nextId = 1
  private pending = new Map<number, { resolve: (r: TranscribeResult) => void; reject: (e: Error) => void }>()
  private watchdog: ReturnType<typeof setInterval> | null = null
  private lastMovement = 0
  private generation = 0

  constructor(model: BrowserModel = DEFAULT_BROWSER_MODEL) {
    this.store = new Store<EngineState>({
      status: 'idle',
      model,
      device: null,
      threads: 1,
      files: [],
      progress: 0,
      stage: null,
      runtime: null,
      stalled: null,
      safeMode: false,
      loadMs: null,
      cached: null,
      cachedIds: [],
      error: null,
    })
    void this.refreshCached()
  }

  /** Re-read which models are in the browser cache. */
  async refreshCached(): Promise<void> {
    const flags = await Promise.all(BROWSER_MODELS.map((m) => isModelCached(m)))
    const cachedIds = BROWSER_MODELS.filter((_, i) => flags[i]).map((m) => m.id)
    this.store.patch({ cachedIds, cached: cachedIds.includes(this.store.get().model.id) })
  }

  get busy(): boolean {
    return this.pending.size > 0
  }

  /** Switch to another model. The current one is unloaded; nothing is downloaded until `load()`. */
  setModel(id: string): void {
    const model = browserModel(id)
    if (model.id === this.store.get().model.id) return
    this.dispose()
    this.store.patch({ model, cached: this.store.get().cachedIds.includes(model.id), loadMs: null, error: null })
  }

  load(opts: LoadOptions = {}): Promise<void> {
    if (this.loading) return this.loading
    const generation = ++this.generation
    const loading = this.doLoad(opts, generation)
      .catch(async (err: unknown) => {
        if (generation !== this.generation) throw err
        const message = err instanceof Error ? err.message : String(err)
        if (!isOutOfMemory(message) || opts.safe) throw err
        // Out of memory: release the synthesiser's runtime if it is loaded, keep one at a time from now on,
        // and try once more on one thread before reporting the error.
        this.worker?.terminate()
        this.worker = null
        requireOneRuntime()
        const freed = releaseOtherRuntimes('stt')
        diag.error(
          'model',
          `out of memory while loading (${message}); retrying on one thread${freed.length ? ` after releasing ${freed.join(', ')}` : ''}`,
        )
        await new Promise((r) => setTimeout(r, 250))
        if (generation !== this.generation) throw err
        return this.doLoad({ ...opts, safe: true }, generation)
      })
      .catch((err: unknown) => {
        if (generation !== this.generation) throw err
        releaseRuntime('stt')
        this.loading = null
        this.stopWatchdog()
        this.worker?.terminate()
        this.worker = null
        const message = err instanceof Error ? err.message : String(err)
        const s = this.store.get()
        diag.error('model', `${s.stage ? `${STAGE_TEXT[s.stage]}: ` : ''}${message}`)
        if (opts.background) {
          // Nobody asked for this load yet: stay quiet, and let an explicit start report the problem.
          this.store.patch({ status: 'idle', stage: null, runtime: null, stalled: null, progress: 0, files: [] })
        } else {
          this.store.patch({ status: 'error', error: message, stalled: null })
        }
        throw err
      })
    this.loading = loading
    return loading
  }

  /** Throw the current attempt away and load again on one WebAssembly thread. */
  retrySafe(): Promise<void> {
    this.dispose()
    return this.load({ safe: true })
  }

  private moved(): void {
    this.lastMovement = performance.now()
    if (this.store.get().stalled) this.store.patch({ stalled: null })
  }

  private startWatchdog(): void {
    this.stopWatchdog()
    const limit = (readBrowserSttConfig().stallSeconds ?? 40) * 1000
    this.lastMovement = performance.now()
    this.watchdog = setInterval(() => {
      const s = this.store.get()
      if (s.status !== 'loading') return
      const quiet = performance.now() - this.lastMovement
      if (quiet >= limit) {
        const seconds = Math.round(quiet / 1000)
        if (!s.stalled)
          diag.error('model', `no progress for ${seconds} s while ${s.stage ? STAGE_TEXT[s.stage] : 'starting'}`)
        this.store.patch({ stalled: { stage: s.stage, seconds } })
      }
    }, 1000)
  }

  private stopWatchdog(): void {
    if (this.watchdog) clearInterval(this.watchdog)
    this.watchdog = null
  }

  private async doLoad(opts: LoadOptions, generation: number): Promise<void> {
    const support = detectSupport()
    if (!support.wasm) throw new Error('This browser does not support WebAssembly.')
    const config = readBrowserSttConfig()
    const safe = Boolean(opts.safe)
    const { model } = this.store.get()
    this.store.patch({
      status: 'loading',
      error: null,
      files: [],
      progress: 0,
      stage: 'runtime',
      runtime: null,
      stalled: null,
      safeMode: safe,
    })
    this.startWatchdog()
    const device: BrowserDevice = safe ? 'wasm' : (config.device ?? ((await probeWebGpu()) ? 'webgpu' : 'wasm'))
    // Threads need SharedArrayBuffer, i.e. a cross-origin-isolated page.
    const threads = !safe && support.wasmThreads ? Math.max(1, Math.min(4, support.hardwareConcurrency)) : 1
    if (generation !== this.generation) throw new Error('Load cancelled')
    this.store.patch({ device, threads })
    diag.event(
      'model',
      `loading ${model.id} on ${device}, ${threads} thread${threads === 1 ? '' : 's'}${safe ? ' (single-threaded retry)' : ''}`,
    )

    // On WebKit only one model runtime fits in the page: this releases the synthesiser's, if it is loaded.
    claimRuntime('stt', () => this.dispose())
    const worker: SpeechWorker = workerFactory
      ? await workerFactory()
      : (new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as SpeechWorker)
    if (generation !== this.generation) {
      worker.terminate()
      throw new Error('Load cancelled')
    }
    this.worker = worker
    await new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The speech worker failed to start'))
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.moved()
          this.store.patch({ files: msg.files, progress: total > 0 ? loaded / total : 0 })
        } else if (msg.type === 'stage') {
          const s = this.store.get()
          if (s.stage !== msg.stage) diag.event('model', STAGE_TEXT[msg.stage])
          if (s.stage !== msg.stage || msg.loaded !== s.runtime?.loaded) this.moved()
          this.store.patch({
            stage: msg.stage,
            runtime:
              msg.stage === 'runtime' && msg.loaded != null ? { loaded: msg.loaded, total: msg.total ?? 0 } : s.runtime,
          })
        } else if (msg.type === 'fetch') {
          diag.fetch(msg.record)
          if (msg.record.error || msg.record.status >= 400 || msg.record.status === 0)
            diag.error('fetch', `${msg.record.url}: ${msg.record.error ?? `HTTP ${msg.record.status}`}`)
        } else if (msg.type === 'ready') {
          this.stopWatchdog()
          recordBrowserTrace({
            model: model.id,
            startedAtPerf: performance.now() - msg.loadMs,
            phases: [{ name: 'load', ms: msg.loadMs }],
            text: `Model ${this.store.get().cached ? 'loaded from cache' : 'downloaded'} and compiled (${msg.device === 'webgpu' ? 'WebGPU' : `WebAssembly, ${threads} thread${threads === 1 ? '' : 's'}`})`,
          })
          diag.event('model', `ready after ${Math.round(msg.loadMs)} ms`)
          const cachedIds = [...new Set([...this.store.get().cachedIds, model.id])]
          this.store.patch({
            status: 'ready',
            device: msg.device,
            loadMs: msg.loadMs,
            progress: 1,
            cached: true,
            cachedIds,
            stage: null,
            stalled: null,
          })
          resolve()
        } else if (msg.type === 'result') {
          this.pending.get(msg.id)?.resolve({ text: msg.text, ms: msg.ms, language: msg.language })
          this.pending.delete(msg.id)
        } else if (msg.type === 'error') {
          if (msg.id != null) {
            this.pending.get(msg.id)?.reject(new Error(msg.message))
            this.pending.delete(msg.id)
          } else {
            reject(new Error(msg.message))
          }
        }
      }
      const req: WorkerRequest = {
        type: 'load',
        repo: model.repo,
        family: model.family,
        dtype: model.dtype,
        device,
        threads,
        plainRuntimeLoad: safe,
        config,
      }
      worker.postMessage(req)
    })
  }

  /** Transcribe 16 kHz mono audio. Requests run one at a time in the worker. */
  async transcribe(audio: Float32Array, opts: { language?: string } = {}): Promise<TranscribeResult> {
    await this.load()
    const worker = this.worker
    if (!worker) throw new Error('Speech worker is not running')
    const id = this.nextId++
    const copy = audio.slice()
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const req: WorkerRequest = { type: 'transcribe', id, audio: copy, language: opts.language }
      worker.postMessage(req, [copy.buffer])
    })
  }

  /** Jobs submitted to the worker and not yet answered. */
  get pendingCount(): number {
    return this.pending.size
  }

  dispose(): void {
    this.generation += 1
    releaseRuntime('stt')
    this.stopWatchdog()
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    for (const p of this.pending.values()) p.reject(new Error('Engine disposed'))
    this.pending.clear()
    this.store.patch({
      status: 'idle',
      progress: 0,
      files: [],
      stage: null,
      runtime: null,
      stalled: null,
      error: null,
    })
  }
}

let shared: BrowserSttEngine | null = null

export function sharedEngine(): BrowserSttEngine {
  shared ??= new BrowserSttEngine()
  return shared
}

/** Human description of where a load is, with bytes where they are known. */
export function describeLoad(
  s: Pick<EngineState, 'status' | 'stage' | 'runtime' | 'files'> & { model: { approxBytes: number } },
  fmt: (bytes: number) => string,
  noun = 'model',
): string {
  if (s.status !== 'loading') return ''
  if (s.stage === 'runtime') {
    return s.runtime && s.runtime.total > 0
      ? `Downloading the runtime · ${fmt(s.runtime.loaded)} of ${fmt(s.runtime.total)}`
      : 'Downloading the runtime'
  }
  if (s.stage === 'model') {
    const loaded = s.files.reduce((a, f) => a + f.loaded, 0)
    const seen = s.files.reduce((a, f) => a + f.total, 0)
    // Files announce their size as they start; until most have, the listed size is the better denominator.
    const total = seen >= s.model.approxBytes * 0.9 ? seen : s.model.approxBytes
    return loaded > 0
      ? `Downloading the ${noun} · ${fmt(loaded)} of ${fmt(Math.max(total, loaded))}`
      : `Downloading the ${noun} · ${fmt(total)}`
  }
  if (s.stage === 'compiling') return `Compiling the ${noun}`
  if (s.stage === 'warmup') return 'Warming up'
  return 'Starting'
}
