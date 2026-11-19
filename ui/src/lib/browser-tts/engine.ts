import {
  claimRuntime,
  isOutOfMemory,
  releaseOtherRuntimes,
  releaseRuntime,
  reportPageMemory,
  requireOneRuntime,
  TTS_MEMORY_PAGES,
} from '@/lib/browser-runtime'
import { device as deviceInfo, ensureDeviceProbe, peakFor, rememberMeasured } from '@/lib/device'
import { placementChoice, rememberSpeed, resolvePlacement, type PlaceableModel, type Placement } from '@/lib/placement'
import { diag } from '@/lib/diagnostics'
import {
  downloadedIds,
  downloads,
  isDownloaded,
  markLoading,
  markSettled,
  recordDownload,
  registerHolder,
  type Resident,
  touchDownload,
} from '@/lib/downloads'
import { readBrowserSttConfig, type BrowserDevice, type FileProgress, type LoadStage } from '@/lib/browser-stt/models'
import { Store } from '@/lib/store'
import { detectSupport } from '@/lib/support'
import { readGpuPower } from '@/lib/webgpu'
import { VOICE_CACHE, type TtsWorkerRequest, type TtsWorkerResponse } from './protocol'
import type { VoiceConfig } from './text'
import {
  browserVoice,
  BROWSER_VOICES,
  hubUrl,
  DEFAULT_BROWSER_VOICE,
  voicesBase,
  type BrowserVoice,
  type VoiceIndex,
} from './voices'

export type TtsEngineStatus = 'idle' | 'loading' | 'ready' | 'error'

/**
 * State of the in-browser synthesiser. The load-related fields have the same
 * shape as the recogniser's, so the same status chip, progress and trouble
 * components show both.
 */
export interface TtsEngineState {
  status: TtsEngineStatus
  voice: BrowserVoice
  /** `model` is what the shared load components read: label and size. */
  model: { id: string; label: string; approxBytes: number }
  device: BrowserDevice | null
  /** Where the loaded voice runs: WebGPU, the CPU, or its two halves split between them. */
  placement: Placement | null
  /** Why Auto chose it, or why it fell back. */
  placementWhy: string
  /** Share of the weights on WebGPU, 0..1. */
  gpuShare: number
  threads: number
  files: FileProgress[]
  progress: number
  stage: LoadStage | null
  runtime: { loaded: number; total: number } | null
  stalled: { stage: LoadStage | null; seconds: number } | null
  safeMode: boolean
  loadMs: number | null
  cached: boolean | null
  cachedIds: string[]
  error: string | null
  config: VoiceConfig | null
  /** What the deployment publishes (sizes, which voices exist); null until read or when absent. */
  index: VoiceIndex | null
  /** Why the index could not be read, if it could not. */
  indexError: string | null
  /** The blend of Kokoro voices spoken instead of the voice's own style, or null. */
  blend?: string | null
}

export interface SynthResult {
  audio: Float32Array
  sampleRate: number
  ms: number
  tokens: number
  phonemes?: string
  missing: string[]
}

export interface TtsWorker {
  onmessage: ((e: MessageEvent<TtsWorkerResponse>) => void) | null
  onerror: ((e: ErrorEvent) => void) | null
  postMessage(message: TtsWorkerRequest, transfer?: Transferable[]): void
  terminate(): void
}

let workerFactory: (() => TtsWorker | Promise<TtsWorker>) | null = null

/** Replace the worker (unit tests). */
export function setTtsWorkerFactory(factory: (() => TtsWorker | Promise<TtsWorker>) | null): void {
  workerFactory = factory
}

const STAGE_TEXT: Record<LoadStage, string> = {
  runtime: 'downloading the runtime',
  model: 'downloading the voice',
  compiling: 'compiling the voice',
  warmup: 'warming up',
}

function modelOf(voice: BrowserVoice, index: VoiceIndex | null, bytes?: number) {
  const published = voice.files.from === 'published' ? index?.voices.find((v) => v.id === voice.id) : undefined
  return { id: voice.id, label: voice.label, approxBytes: bytes ?? published?.bytes ?? voice.approxBytes }
}

/**
 * Whether a voice has been downloaded, from the list in local storage. The
 * browser cache itself is not listed: in WebKit that reads every stored file
 * into memory (see lib/downloads.ts).
 */
interface Halves {
  front: string
  back: string
  /** Share of the weights in the front half, by bytes. */
  frontShare: number
}

/** Whether the published index has the voice's graph cut in two (front and back halves). */
export function hasHalves(voice: BrowserVoice, index: VoiceIndex | null): boolean {
  const f = index?.voices.find((v) => v.id === voice.id)?.files
  return Boolean(voice.files.from === 'published' && f?.['front.onnx'] && f['back.onnx'])
}

/** A voice in the terms of lib/placement.ts. Kokoro starts on WebGPU; the small VITS voices on the CPU. */
export function placeableVoice(voice: BrowserVoice, located?: { halves?: unknown }): PlaceableModel {
  return {
    id: voice.downloadId,
    modules: ['model'],
    bytes: voice.approxBytes,
    peakBytes: voice.peakBytes,
    gpuFirst: voice.kind === 'kokoro',
    cutHalves: Boolean(located?.halves),
    splitText: 'front half on WebGPU, back half on the CPU',
  }
}

export function isVoiceCached(id: string): boolean {
  return isDownloaded(id)
}

/**
 * Owns the worker that runs one voice at a time in the page. Nothing is
 * downloaded until `load()` (or the first synthesis) asks for it.
 */
export class BrowserTtsEngine {
  readonly store: Store<TtsEngineState>
  private worker: TtsWorker | null = null
  private loading: Promise<void> | null = null
  private nextId = 1
  private pending = new Map<number, { resolve: (r: SynthResult) => void; reject: (e: Error) => void }>()
  private blends = new Map<number, { resolve: () => void; reject: (e: Error) => void }>()
  private watchdog: ReturnType<typeof setInterval> | null = null
  private lastMovement = 0
  private generation = 0
  private indexRead: Promise<VoiceIndex | null> | null = null
  private peakMemory: number | null = null

  constructor(voice: BrowserVoice = DEFAULT_BROWSER_VOICE) {
    this.store = new Store<TtsEngineState>({
      status: 'idle',
      voice,
      model: modelOf(voice, null),
      device: null,
      placement: null,
      placementWhy: '',
      gpuShare: 0,
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
      config: null,
      index: null,
      indexError: null,
    })
    this.refreshCached()
    downloads.subscribe(() => this.refreshCached())
    registerHolder('tts', {
      current: () => (this.store.get().status === 'idle' ? null : this.store.get().voice.downloadId),
      release: () => this.dispose(),
      describe: () => this.describe(),
    })
  }

  /** The loaded voice for the Loaded models view (memory is an estimate). */
  describe(): Resident | null {
    const s = this.store.get()
    if (s.status === 'idle' || s.status === 'error') return null
    const bytes = peakFor(s.voice.downloadId, s.voice.peakBytes)
    const backend = s.placement ?? s.device ?? 'wasm'
    return {
      id: s.voice.downloadId,
      label: s.voice.downloadLabel,
      task: 'tts',
      backend,
      bytes,
      gpuBytes: Math.round(bytes * s.gpuShare),
      placement:
        [
          backend === 'split' ? 'front half on WebGPU, back half on the CPU' : '',
          backend === 'webgpu' ? 'operators WebGPU lacks run on the CPU' : '',
          s.placementWhy,
        ]
          .filter(Boolean)
          .join(' · ') || undefined,
      busy: this.pending.size > 0,
      loading: s.status === 'loading',
    }
  }

  /** Read the published index (sizes, which voices this deployment has). Cached for the page's lifetime. */
  readIndex(): Promise<VoiceIndex | null> {
    this.indexRead ??= (async () => {
      const url = `${voicesBase()}index.json`
      try {
        const res = await fetch(url, { cache: 'no-cache' })
        if (!res.ok) throw new Error(res.status === 404 ? 'not published with this build' : `HTTP ${res.status}`)
        const index = (await res.json()) as VoiceIndex
        const s = this.store.get()
        this.store.patch({ index, indexError: null, model: modelOf(s.voice, index) })
        return index
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        diag.error('voices', `${url}: ${message}`)
        this.store.patch({ indexError: message })
        return null
      }
    })()
    return this.indexRead
  }

  /** Which voices are downloaded, from the downloads list. */
  refreshCached(): void {
    const listed = downloadedIds('tts')
    const cachedIds = BROWSER_VOICES.filter((v) => listed.includes(v.downloadId)).map((v) => v.id)
    const current = this.store.get().voice
    if (listed.includes(current.downloadId) && !cachedIds.includes(current.id)) cachedIds.push(current.id)
    const s = this.store.get()
    if (s.cachedIds.join() === cachedIds.join() && s.cached === cachedIds.includes(s.voice.id)) return
    this.store.patch({ cachedIds, cached: cachedIds.includes(s.voice.id) })
  }

  /** Switch voice. The current one is unloaded; nothing is downloaded until `load()`. */
  setVoice(id: string): void {
    const voice = browserVoice(id)
    if (voice.id === this.store.get().voice.id) return
    this.dispose()
    const s = this.store.get()
    this.store.patch({
      voice,
      model: modelOf(voice, s.index),
      cached: s.cachedIds.includes(voice.id),
      loadMs: null,
      error: null,
      config: null,
    })
  }

  load(opts: { safe?: boolean } = {}): Promise<void> {
    if (this.loading) return this.loading
    const generation = ++this.generation
    const loading = this.doLoad(Boolean(opts.safe), generation)
      .catch(async (err: unknown) => {
        if (generation !== this.generation) throw err
        const message = err instanceof Error ? err.message : String(err)
        if (!isOutOfMemory(message) || opts.safe) throw err
        // Out of memory: free everything else the page runs in WebAssembly, keep one runtime at a time from
        // now on, and try once more the leanest way (one thread). Only then is the error shown.
        this.worker?.terminate()
        this.worker = null
        requireOneRuntime()
        const freed = releaseOtherRuntimes('tts')
        diag.error(
          'voice',
          `out of memory while loading (${message}); retrying on one thread${freed.length ? ` after releasing ${freed.join(', ')}` : ''}`,
        )
        this.store.patch({ safeMode: true })
        await new Promise((r) => setTimeout(r, 250))
        if (generation !== this.generation) throw err
        return this.doLoad(true, generation)
      })
      .catch((err: unknown) => {
        if (generation !== this.generation) throw err
        this.loading = null
        this.stopWatchdog()
        this.worker?.terminate()
        this.worker = null
        releaseRuntime('tts')
        markSettled()
        const message = err instanceof Error ? err.message : String(err)
        const s = this.store.get()
        diag.error('voice', `${s.stage ? `${STAGE_TEXT[s.stage]}: ` : ''}${message}`)
        this.store.patch({ status: 'error', error: message, stalled: null })
        throw err
      })
    this.loading = loading
    return loading
  }

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
          diag.error('voice', `no progress for ${seconds} s while ${s.stage ? STAGE_TEXT[s.stage] : 'starting'}`)
        this.store.patch({ stalled: { stage: s.stage, seconds } })
      }
    }, 1000)
  }

  private stopWatchdog(): void {
    if (this.watchdog) clearInterval(this.watchdog)
    this.watchdog = null
  }

  private async doLoad(safe: boolean, generation: number): Promise<void> {
    const support = detectSupport()
    if (!support.wasm) throw new Error('This browser does not support WebAssembly.')
    const { voice } = this.store.get()
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
    const base = voicesBase()
    const located = await this.locate(voice, base)
    const bytes = located.bytes
    const threads = !safe && support.wasmThreads ? Math.max(1, Math.min(4, support.hardwareConcurrency)) : 1
    if (generation !== this.generation) throw new Error('Load cancelled')
    await Promise.race([ensureDeviceProbe(), new Promise((r) => setTimeout(r, 3000))])
    const resolved = resolvePlacement(
      placeableVoice(voice, located),
      safe ? 'wasm' : placementChoice(voice.downloadId),
      deviceInfo.get(),
    )
    const halves = resolved.placement === 'split' ? located.halves : undefined
    const gpu: 'webgpu' | 'wasm' = resolved.placement === 'wasm' ? 'wasm' : 'webgpu'
    this.store.patch({
      device: gpu,
      threads,
      placement: halves ? 'split' : gpu,
      placementWhy: resolved.why,
      gpuShare: halves ? halves.frontShare : gpu === 'webgpu' ? 1 : 0,
    })
    diag.event(
      'voice',
      `loading ${voice.id}, ${threads} thread${threads === 1 ? '' : 's'}${safe ? ' (single-threaded retry)' : ''}`,
    )
    // On WebKit and phones only one model runtime fits in the page: this releases the recogniser's, if it is loaded.
    claimRuntime('tts', () => this.dispose())
    markLoading('tts', voice.downloadId, { label: voice.downloadLabel, store: this.store })
    const worker: TtsWorker = workerFactory
      ? await workerFactory()
      : (new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as TtsWorker)
    if (generation !== this.generation) {
      worker.terminate()
      throw new Error('Load cancelled')
    }
    this.worker = worker
    await new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The speech worker failed to start'))
      worker.onmessage = (e: MessageEvent<TtsWorkerResponse>) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          const loaded = msg.files.reduce((a, f) => a + f.loaded, 0)
          this.moved()
          this.store.patch({ files: msg.files, progress: total > 0 ? loaded / total : 0 })
        } else if (msg.type === 'stage') {
          const s = this.store.get()
          if (s.stage !== msg.stage) diag.event('voice', STAGE_TEXT[msg.stage])
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
        } else if (msg.type === 'note') {
          diag.event('voice', msg.text)
        } else if (msg.type === 'ready') {
          this.stopWatchdog()
          diag.event(
            'voice',
            `ready after ${Math.round(msg.loadMs)} ms${msg.memoryBytes != null ? ` · runtime memory ${(msg.memoryBytes / 2 ** 20).toFixed(0)} MiB` : ''}`,
          )
          void reportPageMemory('after loading the voice')
          markSettled()
          rememberMeasured(voice.downloadId, { loadMs: msg.loadMs, memoryBytes: msg.memoryBytes })
          void recordDownload({
            id: voice.downloadId,
            kind: 'tts',
            label: voice.downloadLabel,
            bytes,
            cache: VOICE_CACHE,
            keys: msg.cacheKeys ?? [],
            // Kokoro voices share their model: each voice adds its own small style file.
            merge: voice.kind === 'kokoro',
          })
          if (msg.device === 'wasm' && this.store.get().placement !== 'wasm')
            this.store.patch({ device: 'wasm', placement: 'wasm', gpuShare: 0, placementWhy: 'WebGPU failed here' })
          this.store.patch({
            status: 'ready',
            loadMs: msg.loadMs,
            progress: 1,
            cached: true,
            cachedIds: [...new Set([...this.store.get().cachedIds, voice.id])],
            model: modelOf(voice, this.store.get().index, bytes),
            stage: null,
            stalled: null,
            config: msg.config,
          })
          resolve()
        } else if (msg.type === 'audio') {
          const rate = this.store.get().config?.sample_rate ?? 22050
          this.pending.get(msg.id)?.resolve({
            audio: msg.audio,
            sampleRate: rate,
            ms: msg.ms,
            tokens: msg.tokens,
            phonemes: msg.phonemes,
            missing: msg.missing,
          })
          this.pending.delete(msg.id)
          if (msg.memoryBytes != null && msg.memoryBytes > (this.peakMemory ?? 0)) {
            this.peakMemory = msg.memoryBytes
            diag.event('voice', `runtime memory grew to ${(msg.memoryBytes / 2 ** 20).toFixed(0)} MiB`)
          }
        } else if (msg.type === 'styled') {
          this.blends.get(msg.id)?.resolve()
          this.blends.delete(msg.id)
        } else if (msg.type === 'error') {
          if (msg.id != null && this.blends.has(msg.id)) {
            this.blends.get(msg.id)?.reject(new Error(msg.message))
            this.blends.delete(msg.id)
          } else if (msg.id != null) {
            this.pending.get(msg.id)?.reject(new Error(msg.message))
            this.pending.delete(msg.id)
          } else {
            reject(new Error(msg.message))
          }
        }
      }
      const req: TtsWorkerRequest = {
        type: 'load',
        base,
        voice: voice.id,
        kind: voice.kind,
        model: located.model,
        config: located.config,
        configFormat: located.configFormat,
        style: located.style,
        phonemizer: voice.phonemizer,
        phonemeLanguage: voice.phonemeLanguage,
        uccharanBase: `${base}uccharan/`,
        threads,
        device: gpu,
        halves: halves ? { front: halves.front, back: halves.back } : undefined,
        ortWasmBase: readBrowserSttConfig().ortWasmBase,
        plainRuntimeLoad: safe,
        memoryPages: TTS_MEMORY_PAGES,
        power: readGpuPower(),
      }
      worker.postMessage(req)
    })
  }

  /**
   * Where a voice's files are: the demo's own voices directory (checked
   * against the published index, with a version so a republished voice is not
   * read from a stale cache), or a pinned Hugging Face URL.
   */
  private async locate(
    voice: BrowserVoice,
    base: string,
  ): Promise<{
    model: string
    config: string | null
    configFormat: 'bolo' | 'piper' | null
    style?: string
    bytes: number
    /** The graph cut in two at build time, when published: front for WebGPU, back for the CPU. */
    halves?: Halves
  }> {
    const files = voice.files
    if (files.from === 'tiny-voca')
      return { model: hubUrl(files.model), config: hubUrl(files.config), configFormat: null, bytes: voice.approxBytes }
    if (files.from === 'kokoro')
      return {
        model: hubUrl(files.model),
        config: null,
        configFormat: null,
        style: hubUrl(files.style),
        bytes: voice.approxBytes,
      }
    if (files.from === 'remote' && files.configFormat === 'piper')
      return {
        model: hubUrl(files.model),
        config: hubUrl(files.config),
        configFormat: 'piper',
        bytes: voice.approxBytes,
      }
    const index = await this.readIndex()
    // A remote model with a published configuration (the 32-bit SYSPIN voices) needs its 8-bit sibling's settings.
    const publishedId = files.from === 'remote' ? files.config.split('/')[0] : voice.id
    const published = index?.voices.find((v) => v.id === publishedId)
    if (index && !published) throw new Error(`${voice.label} is not published with this build of the demo.`)
    if (!index || !published)
      throw new Error(`The voices could not be found (${this.store.get().indexError ?? 'no index'}).`)
    const f = published.files
    const v = `?v=${encodeURIComponent((f['model.onnx']?.sha256 ?? '').slice(0, 12) + (f['voice.json']?.sha256 ?? '').slice(0, 6))}`
    if (files.from === 'remote')
      return {
        model: files.model,
        config: `${base}${files.config}${v}`,
        configFormat: 'bolo',
        bytes: voice.approxBytes,
      }
    const front = f['front.onnx']
    const back = f['back.onnx']
    return {
      model: `${base}${voice.id}/model.onnx${v}`,
      config: `${base}${voice.id}/voice.json${v}`,
      configFormat: 'bolo',
      bytes: published.bytes,
      halves:
        front && back
          ? {
              front: `${base}${voice.id}/front.onnx?v=${front.sha256.slice(0, 12)}`,
              back: `${base}${voice.id}/back.onnx?v=${back.sha256.slice(0, 12)}`,
              frontShare: front.bytes / (front.bytes + back.bytes),
            }
          : undefined,
    }
  }

  /** Synthesise one prepared chunk. Requests run one at a time in the worker. */
  async synthesize(text: string): Promise<SynthResult> {
    await this.load()
    const worker = this.worker
    const cfg = this.store.get().config
    if (!worker || !cfg) throw new Error('The voice is not loaded')
    touchDownload(this.store.get().voice.downloadId)
    const id = this.nextId++
    const { noise_scale, length_scale, noise_scale_w } = cfg.inference
    const { placement, voice } = this.store.get()
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        resolve: (r) => {
          // Speeds measured here let Auto choose between WebGPU and the CPU next time.
          const seconds = r.audio.length / r.sampleRate
          if (placement && seconds >= 1) rememberSpeed(voice.downloadId, placement, r.ms / 1000 / seconds)
          resolve(r)
        },
        reject,
      })
      worker.postMessage({ type: 'synth', id, text, scales: [noise_scale, length_scale, noise_scale_w] })
    })
  }

  /**
   * Speak a blend of Kokoro voices from now on (the loaded voice must be one of them, the heaviest: its language
   * is the one read), or the loaded voice's own style again with null. Returns the blend as the server spells it.
   */
  async setBlend(parts: { voice: BrowserVoice; weight: number }[] | null): Promise<void> {
    await this.load()
    const worker = this.worker
    if (!worker) throw new Error('The voice is not loaded')
    const styles = (parts ?? []).map((p) => {
      if (p.voice.files.from !== 'kokoro') throw new Error(`${p.voice.label} is not a Kokoro voice`)
      return { url: hubUrl(p.voice.files.style), weight: p.weight }
    })
    const id = this.nextId++
    await new Promise<void>((resolve, reject) => {
      this.blends.set(id, { resolve, reject })
      worker.postMessage({ type: 'style', id, styles })
    })
    this.store.patch({ blend: parts && parts.length ? parts.map((p) => `${p.voice.id}(${p.weight})`).join('+') : null })
  }

  /** Stop answering outstanding requests (the worker finishes the one it is on). */
  cancelPending(): void {
    for (const p of this.pending.values()) p.reject(new Error('Stopped'))
    this.pending.clear()
  }

  dispose(): void {
    this.generation += 1
    releaseRuntime('tts')
    if (this.store.get().status === 'loading') markSettled(this.store.get().voice.downloadId)
    this.peakMemory = null
    this.stopWatchdog()
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    this.cancelPending()
    for (const b of this.blends.values()) b.reject(new Error('Unloaded'))
    this.blends.clear()
    this.store.patch({
      status: 'idle',
      progress: 0,
      files: [],
      stage: null,
      runtime: null,
      stalled: null,
      error: null,
      blend: null,
    })
  }
}

let shared: BrowserTtsEngine | null = null

export function sharedTtsEngine(): BrowserTtsEngine {
  shared ??= new BrowserTtsEngine()
  return shared
}
