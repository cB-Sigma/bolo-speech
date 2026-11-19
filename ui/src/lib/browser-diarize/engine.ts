import {
  claimRuntime,
  isOutOfMemory,
  releaseOtherRuntimes,
  releaseRuntime,
  requireOneRuntime,
} from '@/lib/browser-runtime'
import { readBrowserSttConfig } from '@/lib/browser-stt/models'
import { peakFor } from '@/lib/device'
import { diag } from '@/lib/diagnostics'
import {
  markLoading,
  markSettled,
  noteActivity,
  recordDownload,
  registerHolder,
  touchDownload,
  type Resident,
} from '@/lib/downloads'
import { Store } from '@/lib/store'
import { detectSupport } from '@/lib/support'
import {
  DEFAULT_EMBEDDING,
  embeddingModel,
  SEGMENTATION,
  type DiarizeModel,
  type DiarizeWorkerRequest,
  type DiarizeWorkerResponse,
} from './models'
import type { Segment } from './pipeline'

export interface DiarizerState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  embedding: DiarizeModel
  files: { file: string; loaded: number; total: number }[]
  progress: number
  /** 0..1 through the current run. */
  running: number | null
  loadMs: number | null
  error: string | null
}

export interface DiarizeWorker {
  onmessage: ((e: MessageEvent<DiarizeWorkerResponse>) => void) | null
  onerror: ((e: ErrorEvent) => void) | null
  postMessage(message: DiarizeWorkerRequest, transfer?: Transferable[]): void
  terminate(): void
}

let factory: (() => DiarizeWorker) | null = null
/** Replace the worker (tests). */
export function setDiarizeWorkerFactory(f: (() => DiarizeWorker) | null): void {
  factory = f
}

/** Download-list id of a diarizer: the embedding model's (the segmentation model comes with it). */
export const diarizerDownloadId = (m: DiarizeModel) => `diarize-${m.id}`

/**
 * Owns the worker that runs speaker diarization in the page. One embedding
 * model at a time; switching unloads the current one. Listed in Downloaded
 * models and Loaded models like the recogniser and the voices.
 */
export class BrowserDiarizer {
  readonly store = new Store<DiarizerState>({
    status: 'idle',
    embedding: DEFAULT_EMBEDDING,
    files: [],
    progress: 0,
    running: null,
    loadMs: null,
    error: null,
  })
  private worker: DiarizeWorker | null = null
  private loading: Promise<void> | null = null
  private pending = new Map<
    number,
    { resolve: (r: { segments: Segment[]; ms: number }) => void; reject: (e: Error) => void }
  >()
  private nextId = 1
  private generation = 0

  constructor() {
    registerHolder('diarize', {
      current: () => (this.store.get().status === 'idle' ? null : diarizerDownloadId(this.store.get().embedding)),
      release: () => this.dispose(),
      describe: () => this.describe(),
    })
  }

  describe(): Resident | null {
    const s = this.store.get()
    if (s.status === 'idle' || s.status === 'error') return null
    const id = diarizerDownloadId(s.embedding)
    return {
      id,
      label: `${SEGMENTATION.label} + ${s.embedding.label}`,
      task: 'diarize',
      backend: 'wasm',
      bytes: peakFor(id, SEGMENTATION.peakBytes + s.embedding.peakBytes),
      gpuBytes: 0,
      busy: this.pending.size > 0,
      loading: s.status === 'loading',
    }
  }

  setEmbedding(id: string): void {
    const m = embeddingModel(id)
    if (m.id === this.store.get().embedding.id) return
    this.dispose()
    this.store.patch({ embedding: m, error: null, loadMs: null })
  }

  load(): Promise<void> {
    if (this.loading) return this.loading
    const generation = ++this.generation
    const run = (threads: number) => this.doLoad(threads, generation)
    const support = detectSupport()
    const threads = support.wasmThreads ? Math.max(1, Math.min(4, support.hardwareConcurrency)) : 1
    this.loading = run(threads)
      .catch(async (err: unknown) => {
        if (generation !== this.generation) throw err
        const message = err instanceof Error ? err.message : String(err)
        if (!isOutOfMemory(message) || threads === 1) throw err
        // Out of memory: let the other in-page models go, keep one runtime at a time, retry on one thread.
        this.worker?.terminate()
        this.worker = null
        requireOneRuntime()
        const freed = releaseOtherRuntimes('diarize')
        diag.error(
          'diarize',
          `out of memory (${message}); retrying on one thread${freed.length ? ` after releasing ${freed.join(', ')}` : ''}`,
        )
        return run(1)
      })
      .catch((err: unknown) => {
        if (generation === this.generation) {
          this.loading = null
          this.worker?.terminate()
          this.worker = null
          releaseRuntime('diarize')
          markSettled()
          this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
        }
        throw err
      })
    return this.loading
  }

  private async doLoad(threads: number, generation: number): Promise<void> {
    const embedding = this.store.get().embedding
    const id = diarizerDownloadId(embedding)
    this.store.patch({ status: 'loading', files: [], progress: 0, error: null })
    claimRuntime('diarize', () => this.dispose())
    markLoading('diarize', id, { label: `Speaker diarization: ${embedding.label}`, store: this.store })
    diag.event('diarize', `loading ${SEGMENTATION.id} + ${embedding.id}, ${threads} thread${threads === 1 ? '' : 's'}`)
    const worker =
      factory?.() ??
      (new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as DiarizeWorker)
    if (generation !== this.generation) {
      worker.terminate()
      throw new Error('Load cancelled')
    }
    this.worker = worker
    const cfg = readBrowserSttConfig()
    await new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The diarization worker failed to start'))
      worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.store.patch({ files: msg.files, progress: total ? loaded / total : 0 })
        } else if (msg.type === 'ready') {
          markSettled()
          void recordDownload({
            id,
            kind: 'diarize',
            label: `Speaker diarization: ${embedding.label}`,
            bytes: msg.bytes || SEGMENTATION.bytes + embedding.bytes,
            cache: 'transformers-cache',
            keys: msg.cacheKeys,
          })
          diag.event('diarize', `ready after ${Math.round(msg.loadMs)} ms`)
          this.store.patch({ status: 'ready', progress: 1, loadMs: msg.loadMs })
          resolve()
        } else if (msg.type === 'step') {
          this.store.patch({ running: msg.done })
        } else if (msg.type === 'result') {
          this.pending.get(msg.id)?.resolve({ segments: msg.segments, ms: msg.ms })
          this.pending.delete(msg.id)
        } else if (msg.type === 'error') {
          if (msg.id != null) {
            this.pending.get(msg.id)?.reject(new Error(msg.message))
            this.pending.delete(msg.id)
          } else reject(new Error(msg.message))
        }
      }
      worker.postMessage({
        type: 'load',
        segmentation: SEGMENTATION,
        embedding,
        threads,
        modelHost: cfg.modelHost,
        ortWasmBase: cfg.ortWasmBase,
      })
    })
  }

  /** Who spoke when in 16 kHz mono audio. */
  async run(
    audio: Float32Array,
    opts: { speakers: number | null; threshold: number },
  ): Promise<{ segments: Segment[]; ms: number }> {
    await this.load()
    const worker = this.worker
    if (!worker) throw new Error('The diarization worker is not running')
    const id = diarizerDownloadId(this.store.get().embedding)
    touchDownload(id)
    const reqId = this.nextId++
    const copy = audio.slice()
    this.store.patch({ running: 0 })
    try {
      return await new Promise((resolve, reject) => {
        this.pending.set(reqId, { resolve, reject })
        worker.postMessage({ type: 'run', id: reqId, audio: copy, ...opts }, [copy.buffer])
      })
    } finally {
      noteActivity(id)
      this.store.patch({ running: null })
    }
  }

  dispose(): void {
    this.generation += 1
    releaseRuntime('diarize')
    if (this.store.get().status === 'loading') markSettled(diarizerDownloadId(this.store.get().embedding))
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    for (const p of this.pending.values()) p.reject(new Error('Unloaded'))
    this.pending.clear()
    this.store.patch({ status: 'idle', files: [], progress: 0, running: null, error: null })
  }
}

let shared: BrowserDiarizer | null = null
export function sharedDiarizer(): BrowserDiarizer {
  shared ??= new BrowserDiarizer()
  return shared
}
