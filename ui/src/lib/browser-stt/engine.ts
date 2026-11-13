import { Store } from '@/lib/store'
import { recordBrowserTrace } from './traces'
import { detectSupport, probeWebGpu } from '@/lib/support'
import {
  BROWSER_MODELS,
  readBrowserSttConfig,
  type BrowserDevice,
  type BrowserModel,
  type FileProgress,
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
  /** 0..1 over all files seen so far. */
  progress: number
  loadMs: number | null
  cached: boolean | null
  error: string | null
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

/**
 * Owns the web worker that runs speech recognition in the page via ONNX
 * Runtime Web. The worker (and the ~1 MB of runtime JS behind it) is only
 * created when `load()` is first called.
 */
export class BrowserSttEngine {
  readonly store: Store<EngineState>
  private worker: Worker | null = null
  private loading: Promise<void> | null = null
  private nextId = 1
  private pending = new Map<
    number,
    { resolve: (r: { text: string; ms: number }) => void; reject: (e: Error) => void }
  >()

  constructor(model: BrowserModel = BROWSER_MODELS[0] as BrowserModel) {
    this.store = new Store<EngineState>({
      status: 'idle',
      model,
      device: null,
      threads: 1,
      files: [],
      progress: 0,
      loadMs: null,
      cached: null,
      error: null,
    })
    void isModelCached(model).then((cached) => this.store.patch({ cached }))
  }

  get busy(): boolean {
    return this.pending.size > 0
  }

  load(): Promise<void> {
    if (this.loading) return this.loading
    this.loading = this.doLoad().catch((err: unknown) => {
      this.loading = null
      this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      throw err
    })
    return this.loading
  }

  private async doLoad(): Promise<void> {
    const support = detectSupport()
    if (!support.wasm) throw new Error('This browser does not support WebAssembly.')
    const config = readBrowserSttConfig()
    const device: BrowserDevice = config.device ?? ((await probeWebGpu()) ? 'webgpu' : 'wasm')
    // Threads need SharedArrayBuffer, i.e. a cross-origin-isolated page.
    const threads = support.wasmThreads ? Math.max(1, Math.min(4, support.hardwareConcurrency)) : 1
    const { model } = this.store.get()
    this.store.patch({ status: 'loading', device, threads, error: null, files: [], progress: 0 })

    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    this.worker = worker
    await new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The speech worker failed to start'))
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.store.patch({ files: msg.files, progress: total > 0 ? loaded / total : 0 })
        } else if (msg.type === 'ready') {
          recordBrowserTrace({
            model: model.id,
            startedAtPerf: performance.now() - msg.loadMs,
            phases: [{ name: 'load', ms: msg.loadMs }],
            text: `Model ${this.store.get().cached ? 'loaded from cache' : 'downloaded'} and compiled (${msg.device === 'webgpu' ? 'WebGPU' : `WebAssembly, ${threads} thread${threads === 1 ? '' : 's'}`})`,
          })
          this.store.patch({ status: 'ready', device: msg.device, loadMs: msg.loadMs, progress: 1, cached: true })
          resolve()
        } else if (msg.type === 'result') {
          this.pending.get(msg.id)?.resolve({ text: msg.text, ms: msg.ms })
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
      const req: WorkerRequest = { type: 'load', repo: model.repo, dtype: model.dtype, device, threads, config }
      worker.postMessage(req)
    })
  }

  /** Transcribe 16 kHz mono audio. Requests run one at a time in the worker. */
  async transcribe(audio: Float32Array): Promise<{ text: string; ms: number }> {
    await this.load()
    const worker = this.worker
    if (!worker) throw new Error('Speech worker is not running')
    const id = this.nextId++
    const copy = audio.slice()
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      const req: WorkerRequest = { type: 'transcribe', id, audio: copy }
      worker.postMessage(req, [copy.buffer])
    })
  }

  /** Jobs submitted to the worker and not yet answered. */
  get pendingCount(): number {
    return this.pending.size
  }

  dispose(): void {
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    for (const p of this.pending.values()) p.reject(new Error('Engine disposed'))
    this.pending.clear()
    this.store.patch({ status: 'idle', progress: 0, files: [] })
  }
}

let shared: BrowserSttEngine | null = null

export function sharedEngine(): BrowserSttEngine {
  shared ??= new BrowserSttEngine()
  return shared
}
