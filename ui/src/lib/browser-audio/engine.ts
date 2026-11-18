import { isDemo } from '@/lib/api/config'
import { readBrowserSttConfig } from '@/lib/browser-stt/models'
import { claimRuntime, releaseRuntime } from '@/lib/browser-runtime'
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
import { readGpuPower } from '@/lib/webgpu'
import { Store } from '@/lib/store'
import { detectSupport } from '@/lib/support'
import { SIMILARITY_FEATURES } from '@/lib/browser-clone/similarity'
import { audioModel, type AudioModel, type AudioTask } from './models'
import { AUDIO_CACHE, type AudioWorkerRequest, type AudioWorkerResponse } from './protocol'

/**
 * The small audio models in the page (src/lib/browser-audio/models.ts), one worker per task: the denoiser, the
 * language identifier, the voice-activity detector, the end-of-turn model and the speaker embedder can be loaded
 * together, each a few MB. Every one registers with the downloads list (so it can be deleted and is unloaded
 * first), the Loaded models view, the keep-alive and the one-runtime-at-a-time rule of phones and WebKit.
 */

export type AudioDevice = 'wasm' | 'webgpu'
/** The visitor's choice of device: Auto follows the model's measured default. */
export type AudioDeviceChoice = 'auto' | AudioDevice

export interface AudioEngineState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  model: AudioModel | null
  device: AudioDevice | null
  progress: number
  loadMs: number | null
  error: string | null
  /** Time of the last request, ms. */
  lastMs: number | null
}

export interface AudioWorkerLike {
  postMessage(msg: AudioWorkerRequest, transfer?: Transferable[]): void
  terminate(): void
  onmessage: ((e: MessageEvent<AudioWorkerResponse>) => void) | null
  onerror: ((e: ErrorEvent) => void) | null
}

let workerFactory: (() => AudioWorkerLike) | null = null
/** Tests swap the worker for a fake. */
export function setAudioWorkerFactory(f: (() => AudioWorkerLike) | null): void {
  workerFactory = f
}

const HUB = 'https://huggingface.co/'
const NPM = 'https://cdn.jsdelivr.net/npm/'

/** Where the public demo is published: the console served by a bolo server reads the demo's copies from there. */
const PUBLIC_DEMO = 'https://cb-sigma.github.io/bolo-speech/'

/** Base URL of the demo's own copies of models (`models/` next to the demo's page). */
function siteBase(): string {
  const here = (globalThis.location?.href ?? 'http://localhost/').split('#')[0] as string
  return new URL(isDemo ? './' : PUBLIC_DEMO, here).href
}

/** Absolute URL of each of a model's files, by role. */
export function modelUrls(m: AudioModel): Record<string, string> {
  const f = m.files
  if (!f) return {}
  const cfg = readBrowserSttConfig()
  const hub = (cfg.modelHost ?? HUB).replace(/\/?$/, '/')
  const npm = (cfg.npmBase ?? NPM).replace(/\/?$/, '/')
  const out: Record<string, string> = {}
  for (const [key, path] of Object.entries(f.files)) {
    if (f.from === 'hub') out[key] = `${hub}${f.repo}/resolve/${f.revision}/${path}`
    else if (f.from === 'npm') out[key] = `${npm}${f.pkg}@${f.version}/${path}`
    else if (f.from === 'site')
      out[key] = cfg.modelsBase
        ? `${cfg.modelsBase.replace(/\/?$/, '/')}${path.replace(/^models\//, '')}`
        : new URL(path, siteBase()).href
    // Raw GitHub files: through the model host when a test points it elsewhere.
    else out[key] = cfg.modelHost ? `${hub}raw/${path.replace(/^https:\/\/[^/]+\//, '')}` : path
  }
  return out
}

const TASK_NOUN: Record<AudioTask, string> = {
  denoise: 'Denoiser',
  lid: 'Language identification',
  vad: 'Voice activity',
  turn: 'End of turn',
  embed: 'Speaker embeddings',
  tag: 'Sound events',
}

const FRAME_BY_FRAME = new Set(['gtcrn', 'dpdfnet', 'deepfilternet3', 'silero-v4', 'silero-v5'])

export function downloadId(m: AudioModel): string {
  return `audio-${m.id}`
}

export class AudioModelEngine {
  readonly store: Store<AudioEngineState>
  readonly task: AudioTask
  private worker: AudioWorkerLike | null = null
  private loading: Promise<void> | null = null
  private loadingKey = ''
  private pending = new Map<number, { resolve: (v: AudioWorkerResponse) => void; reject: (e: Error) => void }>()
  private nextId = 1

  constructor(task: AudioTask) {
    this.task = task
    this.store = new Store<AudioEngineState>({
      status: 'idle',
      model: null,
      device: null,
      progress: 0,
      loadMs: null,
      error: null,
      lastMs: null,
    })
    registerHolder(this.holderKey, {
      current: () => {
        const s = this.store.get()
        return s.status === 'idle' || !s.model ? null : downloadId(s.model)
      },
      release: () => this.dispose(),
      describe: () => this.describe(),
    })
  }

  private get holderKey(): string {
    return `audio-${this.task}`
  }

  describe(): Resident | null {
    const s = this.store.get()
    if (!s.model || s.status === 'idle' || s.status === 'error') return null
    const bytes = peakFor(downloadId(s.model), s.model.peakBytes)
    return {
      id: downloadId(s.model),
      label: `${TASK_NOUN[this.task]}: ${s.model.label}`,
      task: this.task,
      backend: s.device ?? 'wasm',
      bytes,
      gpuBytes: s.device === 'webgpu' ? bytes : 0,
      busy: this.pending.size > 0,
      loading: s.status === 'loading',
    }
  }

  /** Load `id` on `device` (Auto: the model's measured default). Loading another model replaces this one. */
  load(id: string, choice: AudioDeviceChoice = 'auto'): Promise<void> {
    const model = audioModel(id)
    const support = detectSupport()
    const device: AudioDevice =
      choice === 'auto' ? model.device : choice === 'webgpu' && model.webgpu && support.webgpu ? 'webgpu' : 'wasm'
    const key = `${id}@${device}`
    const s = this.store.get()
    if (this.loading && this.loadingKey === key) return this.loading
    if (s.status === 'ready' && s.model?.id === id && s.device === device) return Promise.resolve()
    this.dispose()
    this.loadingKey = key
    if (!model.files) {
      // Nothing to load (the energy detector runs in the page itself).
      this.store.patch({ status: 'ready', model, device: 'wasm', progress: 1, loadMs: 0, error: null })
      this.loading = Promise.resolve()
      return this.loading
    }
    this.store.patch({ status: 'loading', model, device, progress: 0, error: null, loadMs: null })
    claimRuntime(this.holderKey, () => this.dispose())
    markLoading('audio', downloadId(model))
    const worker =
      workerFactory?.() ??
      (new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as AudioWorkerLike)
    this.worker = worker
    const urls = modelUrls(model)
    const cfg = readBrowserSttConfig()
    const started = performance.now()
    this.loading = new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The audio worker failed to start'))
      worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.store.patch({ progress: total ? loaded / total : 0 })
        } else if (msg.type === 'fetch') {
          diag.fetch(msg.record)
        } else if (msg.type === 'ready') {
          markSettled()
          diag.event(
            'audio',
            `${model.label} ready on ${msg.device === 'webgpu' ? 'WebGPU' : 'WebAssembly'} after ${Math.round(msg.loadMs)} ms`,
          )
          void recordDownload({
            id: downloadId(model),
            kind: 'audio',
            label: `${TASK_NOUN[this.task]}: ${model.label}`,
            bytes: msg.bytes,
            cache: AUDIO_CACHE,
            keys: msg.cacheKeys,
          })
          noteActivity(downloadId(model))
          this.store.patch({ status: 'ready', progress: 1, loadMs: performance.now() - started, device: msg.device })
          resolve()
        } else if (msg.type === 'error') {
          if (msg.id != null) {
            this.pending.get(msg.id)?.reject(new Error(msg.message))
            this.pending.delete(msg.id)
          } else reject(new Error(msg.message))
        } else if ('id' in msg) {
          this.pending.get(msg.id)?.resolve(msg)
          this.pending.delete(msg.id)
        }
      }
      const support = detectSupport()
      worker.postMessage({
        type: 'load',
        model,
        urls,
        device,
        // Models run frame by frame do a few hundred small runs a second: threads cost more than they give.
        threads: FRAME_BY_FRAME.has(model.runner)
          ? 1
          : support.wasmThreads
            ? Math.max(1, Math.min(2, support.hardwareConcurrency))
            : 1,
        ortWasmBase: cfg.ortWasmBase,
        power: readGpuPower(),
        features:
          model.task === 'embed'
            ? {
                host: (cfg.modelHost ?? HUB).replace(/\/?$/, '/'),
                repo: SIMILARITY_FEATURES.repo,
                revision: SIMILARITY_FEATURES.revision,
              }
            : undefined,
      })
    }).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err)
      diag.error('audio', `${model.label}: ${message}`)
      this.worker?.terminate()
      this.worker = null
      this.loading = null
      releaseRuntime(this.holderKey)
      markSettled()
      this.store.patch({ status: 'error', error: message })
      throw err
    })
    return this.loading
  }

  private async request<T extends AudioWorkerResponse>(
    msg: AudioWorkerRequest & { id: number },
    transfer: Transferable[] = [],
  ): Promise<T> {
    await this.loading
    const worker = this.worker
    const model = this.store.get().model
    if (!worker || !model) throw new Error('The model is not loaded')
    touchDownload(downloadId(model))
    try {
      const reply = await new Promise<AudioWorkerResponse>((resolve, reject) => {
        this.pending.set(msg.id, { resolve, reject })
        worker.postMessage(msg, transfer)
      })
      if ('ms' in reply) this.store.patch({ lastMs: reply.ms })
      return reply as T
    } finally {
      noteActivity(downloadId(model))
    }
  }

  /** Denoised audio, same rate and length; `strength` mixes it with the original (1: denoised only). */
  async denoise(audio: Float32Array, rate: number, strength = 1): Promise<{ audio: Float32Array; ms: number }> {
    const copy = audio.slice()
    const r = await this.request<Extract<AudioWorkerResponse, { type: 'audio' }>>(
      { type: 'denoise', id: this.nextId++, audio: copy, rate, strength },
      [copy.buffer],
    )
    return { audio: r.audio, ms: r.ms }
  }

  /** Speech probability per frame of a stretch of 16 kHz audio (the detector keeps its state between calls). */
  async vad(audio: Float32Array, reset = false): Promise<{ probs: Float32Array; frame: number; ms: number }> {
    const copy = audio.slice()
    const r = await this.request<Extract<AudioWorkerResponse, { type: 'probs' }>>(
      { type: 'vad', id: this.nextId++, audio: copy, reset },
      [copy.buffer],
    )
    return { probs: r.probs, frame: r.frame, ms: r.ms }
  }

  /** Probability that the speaker has finished, from the end of `audio` (16 kHz). */
  async turn(audio: Float32Array): Promise<{ value: number; ms: number }> {
    const copy = audio.slice()
    const r = await this.request<Extract<AudioWorkerResponse, { type: 'value' }>>(
      { type: 'turn', id: this.nextId++, audio: copy },
      [copy.buffer],
    )
    return { value: r.value, ms: r.ms }
  }

  async embed(audio: Float32Array): Promise<{ vector: Float32Array; ms: number }> {
    const copy = audio.slice()
    const r = await this.request<Extract<AudioWorkerResponse, { type: 'vector' }>>(
      { type: 'embed', id: this.nextId++, audio: copy },
      [copy.buffer],
    )
    return { vector: r.vector, ms: r.ms }
  }

  /** The most likely languages of 16 kHz audio, with their probability. */
  async identify(audio: Float32Array, top = 5): Promise<{ ranked: [string, number][]; ms: number }> {
    const copy = audio.slice()
    const r = await this.request<Extract<AudioWorkerResponse, { type: 'langs' }>>(
      { type: 'lid', id: this.nextId++, audio: copy, top },
      [copy.buffer],
    )
    return { ranked: r.ranked, ms: r.ms }
  }

  /** Sound events in 16 kHz audio, the `top` most likely, whole and (with `window` seconds) window by window. */
  async tag(
    audio: Float32Array,
    top = 10,
    window?: number,
  ): Promise<Omit<Extract<AudioWorkerResponse, { type: 'tags' }>, 'type' | 'id'>> {
    const copy = audio.slice()
    const r = await this.request<Extract<AudioWorkerResponse, { type: 'tags' }>>(
      { type: 'tag', id: this.nextId++, audio: copy, top, window },
      [copy.buffer],
    )
    return { ranked: r.ranked, windows: r.windows, ms: r.ms }
  }

  dispose(): void {
    releaseRuntime(this.holderKey)
    const s = this.store.get()
    if (s.status === 'loading' && s.model) markSettled(downloadId(s.model))
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    this.loadingKey = ''
    for (const p of this.pending.values()) p.reject(new Error('Unloaded'))
    this.pending.clear()
    this.store.patch({ status: 'idle', progress: 0, error: null })
  }
}

const engines = new Map<AudioTask, AudioModelEngine>()

/** The page's one engine for `task`. */
export function audioEngine(task: AudioTask): AudioModelEngine {
  let e = engines.get(task)
  if (!e) {
    e = new AudioModelEngine(task)
    engines.set(task, e)
  }
  return e
}
