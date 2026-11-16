import {
  claimRuntime,
  isOutOfMemory,
  releaseOtherRuntimes,
  releaseRuntime,
  requireOneRuntime,
} from '@/lib/browser-runtime'
import { readBrowserSttConfig, type FileProgress } from '@/lib/browser-stt/models'
import { device as deviceStore, pageMemoryBytes, peakFor, rememberMeasured } from '@/lib/device'
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
import { cloneDownloadId, cloneModel, DEFAULT_CLONE_MODEL, type CloneModel } from './models'
import type {
  CloneWorkerRequest,
  CloneWorkerResponse,
  EmbedWorkerRequest,
  EmbedWorkerResponse,
  GenerateOptions,
} from './protocol'
import { CLONE_RATE } from './protocol'
import { SIMILARITY_DOWNLOAD_ID, SIMILARITY_MODEL } from './similarity'

export type Placement = 'auto' | 'webgpu' | 'wasm'

/** Out of memory, as the browser or ONNX Runtime (C++ `std::bad_alloc` inside WebAssembly) reports it. */
export const isOom = (message: string | null | undefined) => isOutOfMemory(message) || /bad_alloc/i.test(message ?? '')

export interface ClonerState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  model: CloneModel
  placement: Placement
  /** Where the loaded model runs. */
  device: 'webgpu' | 'wasm' | null
  stage: string | null
  files: FileProgress[]
  progress: number
  loadMs: number | null
  threads: number
  busy: boolean
  error: string | null
  /** The last error was the browser running out of memory. */
  outOfMemory: boolean
}

export interface GenerateResult {
  /** All the audio, 24 kHz mono. */
  audio: Float32Array
  /** Time from the request to the first audio, ms. */
  firstAudioMs: number | null
  /** Time to take in the reference (0 when it was already in), ms. */
  voiceMs: number
  /** Generation time, ms (reference excluded). */
  ms: number
  /** Real-time factor: generation time / audio length. */
  rtf: number
  cancelled: boolean
}

export interface WorkerLike<Req, Res> {
  onmessage: ((e: MessageEvent<Res>) => void) | null
  onerror: ((e: ErrorEvent) => void) | null
  postMessage(message: Req, transfer?: Transferable[]): void
  terminate(): void
}

type CloneWorker = WorkerLike<CloneWorkerRequest, CloneWorkerResponse>
type EmbedWorker = WorkerLike<EmbedWorkerRequest, EmbedWorkerResponse>

let cloneFactory: (() => CloneWorker) | null = null
let embedFactory: (() => EmbedWorker) | null = null

/** Replace the workers (tests). */
export function setCloneWorkerFactories(clone: (() => CloneWorker) | null, embed: (() => EmbedWorker) | null): void {
  cloneFactory = clone
  embedFactory = embed
}

/**
 * Where to run a model. Auto: Chatterbox on WebGPU when there is an adapter; Pocket TTS on the CPU, where its
 * frame loop (small steps with large state tensors passed back each time) is quicker than a round trip to the GPU.
 */
export function chooseDevice(placement: Placement, model: CloneModel): 'webgpu' | 'wasm' {
  if (placement !== 'auto') return placement
  const gpu = deviceStore.get().webgpu?.available ?? false
  return gpu && model.family === 'chatterbox' ? 'webgpu' : 'wasm'
}

/**
 * Owns the worker that runs one voice-cloning model. Switching models
 * unloads the current one. Listed in Downloaded models and Loaded models; an
 * out-of-memory failure releases the other in-page models and retries once
 * on one thread on the CPU.
 */
export class BrowserCloner {
  readonly store = new Store<ClonerState>({
    status: 'idle',
    model: DEFAULT_CLONE_MODEL,
    placement: 'auto',
    device: null,
    stage: null,
    files: [],
    progress: 0,
    loadMs: null,
    threads: 1,
    busy: false,
    error: null,
    outOfMemory: false,
  })
  private worker: CloneWorker | null = null
  private loading: Promise<void> | null = null
  private generation = 0
  private nextId = 1
  private voiceKey: string | null = null
  private pending = new Map<number, (msg: CloneWorkerResponse) => void>()
  private activeId: number | null = null

  constructor() {
    registerHolder('clone', {
      current: () => (this.store.get().status === 'idle' ? null : cloneDownloadId(this.store.get().model)),
      release: () => this.dispose(),
      describe: () => this.describe(),
    })
  }

  describe(): Resident | null {
    const s = this.store.get()
    if (s.status === 'idle' || s.status === 'error') return null
    const id = cloneDownloadId(s.model)
    const bytes = peakFor(id, s.model.peakBytes)
    const gpu = s.device === 'webgpu'
    return {
      id,
      label: `Voice cloning: ${s.model.label}`,
      task: 'tts',
      backend: gpu ? 'webgpu' : 'wasm',
      bytes,
      gpuBytes: gpu ? Math.round(bytes * 0.7) : 0,
      busy: s.busy,
      loading: s.status === 'loading',
      placement: gpu ? 'language model on WebGPU, audio codec on the CPU' : 'all on the CPU',
    }
  }

  setModel(id: string): void {
    const m = cloneModel(id)
    if (m.id === this.store.get().model.id) return
    this.dispose()
    this.safe = false
    this.store.patch({ model: m, error: null, loadMs: null, outOfMemory: false })
  }

  setPlacement(p: Placement): void {
    if (p === this.store.get().placement) return
    this.dispose()
    this.store.patch({ placement: p })
  }

  load(): Promise<void> {
    if (this.loading) return this.loading
    const generation = ++this.generation
    const support = detectSupport()
    const threads = support.wasmThreads && !this.safe ? Math.max(1, Math.min(4, support.hardwareConcurrency)) : 1
    const first = this.safe ? 'wasm' : chooseDevice(this.store.get().placement, this.store.get().model)
    this.loading = this.doLoad(threads, first, generation)
      .catch(async (err: unknown) => {
        if (generation !== this.generation) throw err
        const message = err instanceof Error ? err.message : String(err)
        if (!isOom(message)) throw err
        // Out of memory: let the other in-page models go, keep one runtime at a time, retry on one thread.
        this.terminate()
        this.safe = true
        requireOneRuntime()
        const freed = releaseOtherRuntimes('clone')
        diag.error(
          'clone',
          `out of memory (${message}); retrying on one CPU thread${freed.length ? ` after releasing ${freed.join(', ')}` : ''}`,
        )
        return this.doLoad(1, 'wasm', generation)
      })
      .catch((err: unknown) => {
        if (generation === this.generation) {
          const message = err instanceof Error ? err.message : String(err)
          this.loading = null
          this.terminate()
          releaseRuntime('clone')
          markSettled()
          this.store.patch({ status: 'error', stage: null, error: message, outOfMemory: isOom(message) })
        }
        throw err
      })
    return this.loading
  }

  private terminate(): void {
    this.worker?.terminate()
    this.worker = null
    this.voiceKey = null
    for (const resolve of this.pending.values()) resolve({ type: 'error', message: 'Unloaded' })
    this.pending.clear()
  }

  private async doLoad(threads: number, device: 'webgpu' | 'wasm', generation: number): Promise<void> {
    const model = this.store.get().model
    const id = cloneDownloadId(model)
    this.store.patch({
      status: 'loading',
      stage: 'runtime',
      files: [],
      progress: 0,
      error: null,
      outOfMemory: false,
      threads,
    })
    claimRuntime('clone', () => this.dispose())
    markLoading('clone', id)
    diag.event('clone', `loading ${model.id} on ${device}, ${threads} thread${threads === 1 ? '' : 's'}`)
    const worker =
      cloneFactory?.() ??
      (new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as CloneWorker)
    if (generation !== this.generation) {
      worker.terminate()
      throw new Error('Load cancelled')
    }
    this.worker = worker
    const cfg = readBrowserSttConfig()
    await new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The cloning worker failed to start'))
      worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.store.patch({ files: msg.files, progress: total ? loaded / total : 0 })
        } else if (msg.type === 'stage') this.store.patch({ stage: msg.stage })
        else if (msg.type === 'note') diag.event('clone', msg.text)
        else if (msg.type === 'ready') {
          markSettled()
          void recordDownload({
            id,
            kind: 'clone',
            label: `Voice cloning: ${model.label}`,
            bytes: msg.bytes,
            cache: 'transformers-cache',
            keys: msg.cacheKeys,
          })
          void pageMemoryBytes().then((memoryBytes) => rememberMeasured(id, { loadMs: msg.loadMs, memoryBytes }))
          diag.event('clone', `ready after ${Math.round(msg.loadMs)} ms on ${msg.device}`)
          this.store.patch({ status: 'ready', stage: null, progress: 1, loadMs: msg.loadMs, device: msg.device })
          resolve()
        } else if (msg.type === 'error' && msg.id == null) reject(new Error(msg.message))
        else if ('id' in msg && msg.id != null) this.pending.get(msg.id)?.(msg)
      }
      worker.postMessage({
        type: 'load',
        model,
        device,
        threads,
        modelHost: cfg.modelHost,
        ortWasmBase: cfg.ortWasmBase,
        memoryPages: model.memoryPages ?? 16_384,
      })
    })
  }

  private call(
    req:
      | Omit<Extract<CloneWorkerRequest, { type: 'voice' }>, 'id'>
      | Omit<Extract<CloneWorkerRequest, { type: 'generate' }>, 'id'>,
    onAudio?: (a: Float32Array) => void,
  ) {
    const worker = this.worker
    if (!worker) throw new Error('The cloning model is not loaded')
    const id = this.nextId++
    return {
      id,
      done: new Promise<Extract<CloneWorkerResponse, { type: 'voice' | 'done' }>>((resolve, reject) => {
        this.pending.set(id, (msg) => {
          if (msg.type === 'audio') return onAudio?.(msg.samples)
          this.pending.delete(id)
          if (msg.type === 'voice' || msg.type === 'done') resolve(msg)
          else if (msg.type === 'error') reject(new Error(msg.message))
        })
        if (req.type === 'voice') worker.postMessage({ ...req, id }, [req.audio.buffer])
        else worker.postMessage({ ...req, id })
      }),
    }
  }

  /**
   * Speak `text` in the voice of `reference` (24 kHz mono, identified by `key`; it is taken in again only when
   * the key changes). Audio is handed to `onAudio` as it comes.
   */
  async generate(
    key: string,
    reference: Float32Array,
    text: string,
    options: GenerateOptions,
    onAudio: (a: Float32Array) => void,
  ): Promise<GenerateResult> {
    try {
      return await this.generateOnce(key, reference, text, options, onAudio)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      if (!isOom(message) || this.safe) throw err
      // Out of memory while generating: unload everything else in the page, then load this model again on one
      // CPU thread and try once more.
      this.dispose()
      requireOneRuntime()
      const freed = releaseOtherRuntimes('clone')
      diag.error(
        'clone',
        `out of memory while generating (${message}); retrying on one CPU thread${freed.length ? ` after releasing ${freed.join(', ')}` : ''}`,
      )
      this.safe = true
      return this.generateOnce(key, reference, text, options, onAudio)
    }
  }

  /** After running out of memory: one thread, the CPU, until the model is changed. */
  private safe = false

  private async generateOnce(
    key: string,
    reference: Float32Array,
    text: string,
    options: GenerateOptions,
    onAudio: (a: Float32Array) => void,
  ): Promise<GenerateResult> {
    await this.load()
    const id = cloneDownloadId(this.store.get().model)
    touchDownload(id)
    this.store.patch({ busy: true })
    const started = performance.now()
    let voiceMs = 0
    try {
      if (this.voiceKey !== key) {
        await this.call({ type: 'voice', key, audio: reference.slice() }).done
        voiceMs = performance.now() - started
        this.voiceKey = key
      }
      const genStart = performance.now()
      let first: number | null = null
      const parts: Float32Array[] = []
      const c = this.call({ type: 'generate', key, text, options }, (a) => {
        first ??= performance.now() - started
        parts.push(a)
        onAudio(a)
      })
      this.activeId = c.id
      await c.done
      const ms = performance.now() - genStart
      const n = parts.reduce((a, p) => a + p.length, 0)
      const audio = new Float32Array(n)
      let o = 0
      for (const p of parts) {
        audio.set(p, o)
        o += p.length
      }
      const wasCancelled = this.cancelledIds.has(this.activeId ?? -1)
      return {
        audio,
        firstAudioMs: first,
        voiceMs,
        ms,
        rtf: n ? ms / 1000 / (n / CLONE_RATE) : 0,
        cancelled: wasCancelled,
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      if (isOom(message)) {
        // The worker is in an unknown state after this: unload so the next attempt starts clean.
        this.dispose()
        this.store.patch({ status: 'error', error: message, outOfMemory: true })
      }
      throw err
    } finally {
      this.activeId = null
      noteActivity(id)
      this.store.patch({ busy: false })
    }
  }

  private cancelledIds = new Set<number>()

  /** Stop the generation under way (what was made so far is kept). */
  cancel(): void {
    if (this.activeId == null) return
    this.cancelledIds.add(this.activeId)
    this.worker?.postMessage({ type: 'cancel', id: this.activeId })
  }

  dispose(): void {
    this.generation += 1
    releaseRuntime('clone')
    if (this.store.get().status === 'loading') markSettled(cloneDownloadId(this.store.get().model))
    this.terminate()
    this.loading = null
    this.store.patch({ status: 'idle', stage: null, files: [], progress: 0, busy: false, device: null, error: null })
  }
}

export interface ScorerState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  progress: number
  error: string | null
}

/** Speaker embeddings for the similarity score, in their own small worker (holder 'clone-similarity'). */
export class SimilarityScorer {
  readonly store = new Store<ScorerState>({ status: 'idle', progress: 0, error: null })
  private worker: EmbedWorker | null = null
  private loading: Promise<void> | null = null
  private pending = new Map<number, { resolve: (v: Float32Array) => void; reject: (e: Error) => void }>()
  private nextId = 1

  constructor() {
    registerHolder('clone-similarity', {
      current: () => (this.store.get().status === 'idle' ? null : SIMILARITY_DOWNLOAD_ID),
      release: () => this.dispose(),
      describe: () => {
        const s = this.store.get()
        if (s.status === 'idle' || s.status === 'error') return null
        return {
          id: SIMILARITY_DOWNLOAD_ID,
          label: `Speaker similarity: ${SIMILARITY_MODEL.label}`,
          task: 'diarize',
          backend: 'wasm',
          bytes: peakFor(SIMILARITY_DOWNLOAD_ID, SIMILARITY_MODEL.peakBytes),
          gpuBytes: 0,
          busy: this.pending.size > 0,
          loading: s.status === 'loading',
        }
      },
    })
  }

  load(): Promise<void> {
    if (this.loading) return this.loading
    this.store.patch({ status: 'loading', progress: 0, error: null })
    claimRuntime('clone-similarity', () => this.dispose())
    markLoading('clone', SIMILARITY_DOWNLOAD_ID)
    const worker =
      embedFactory?.() ??
      (new Worker(new URL('./embed-worker.ts', import.meta.url), { type: 'module' }) as unknown as EmbedWorker)
    this.worker = worker
    const cfg = readBrowserSttConfig()
    const support = detectSupport()
    this.loading = new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The embedding worker failed to start'))
      worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.store.patch({ progress: total ? loaded / total : 0 })
        } else if (msg.type === 'ready') {
          markSettled()
          void recordDownload({
            id: SIMILARITY_DOWNLOAD_ID,
            kind: 'clone',
            label: `Speaker similarity: ${SIMILARITY_MODEL.label}`,
            bytes: msg.bytes,
            cache: 'transformers-cache',
            keys: msg.cacheKeys,
          })
          this.store.patch({ status: 'ready', progress: 1 })
          resolve()
        } else if (msg.type === 'embedding') {
          this.pending.get(msg.id)?.resolve(msg.vector)
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
        threads: support.wasmThreads ? Math.max(1, Math.min(2, support.hardwareConcurrency)) : 1,
        modelHost: cfg.modelHost,
        ortWasmBase: cfg.ortWasmBase,
      })
    }).catch((err: unknown) => {
      this.loading = null
      this.worker?.terminate()
      this.worker = null
      releaseRuntime('clone-similarity')
      markSettled()
      this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      throw err
    })
    return this.loading
  }

  /** Embedding of 16 kHz mono audio. */
  async embed(audio16k: Float32Array): Promise<Float32Array> {
    await this.load()
    const worker = this.worker
    if (!worker) throw new Error('The embedding model is not loaded')
    touchDownload(SIMILARITY_DOWNLOAD_ID)
    const id = this.nextId++
    const copy = audio16k.slice()
    try {
      return await new Promise<Float32Array>((resolve, reject) => {
        this.pending.set(id, { resolve, reject })
        worker.postMessage({ type: 'embed', id, audio: copy }, [copy.buffer])
      })
    } finally {
      noteActivity(SIMILARITY_DOWNLOAD_ID)
    }
  }

  dispose(): void {
    releaseRuntime('clone-similarity')
    if (this.store.get().status === 'loading') markSettled(SIMILARITY_DOWNLOAD_ID)
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    for (const p of this.pending.values()) p.reject(new Error('Unloaded'))
    this.pending.clear()
    this.store.patch({ status: 'idle', progress: 0, error: null })
  }
}

let cloner: BrowserCloner | null = null
let scorer: SimilarityScorer | null = null
export function sharedCloner(): BrowserCloner {
  cloner ??= new BrowserCloner()
  return cloner
}
export function sharedScorer(): SimilarityScorer {
  scorer ??= new SimilarityScorer()
  return scorer
}
