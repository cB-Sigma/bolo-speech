import {
  claimRuntime,
  isOutOfMemory,
  releaseOtherRuntimes,
  releaseRuntime,
  requireOneRuntime,
} from '@/lib/browser-runtime'
import { readBrowserSttConfig } from '@/lib/browser-stt/models'
import { device, ensureDeviceProbe } from '@/lib/device'
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
import { readGpuPower } from '@/lib/webgpu'
import {
  DEFAULT_LLM,
  llmModel,
  type ChatMessage,
  type LlmModel,
  type LlmWorkerRequest,
  type LlmWorkerResponse,
} from './models'

/** Something that streams a chat reply: the in-page model, or the visitor's own endpoint. */
export interface ChatBackend {
  reply(messages: ChatMessage[], onToken: (t: string) => void, signal: AbortSignal): Promise<string>
}

export interface LlmState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  model: LlmModel
  device: 'webgpu' | 'wasm' | null
  progress: number
  error: string | null
  loadMs: number | null
}

export interface LlmWorker {
  onmessage: ((e: MessageEvent<LlmWorkerResponse>) => void) | null
  onerror: ((e: ErrorEvent) => void) | null
  postMessage(m: LlmWorkerRequest): void
  terminate(): void
}

let factory: (() => LlmWorker) | null = null
export function setLlmWorkerFactory(f: (() => LlmWorker) | null): void {
  factory = f
}

/** Owns the worker that runs one language model in the page. */
export class BrowserLlm implements ChatBackend {
  readonly store = new Store<LlmState>({
    status: 'idle',
    model: DEFAULT_LLM,
    device: null,
    progress: 0,
    error: null,
    loadMs: null,
  })
  private worker: LlmWorker | null = null
  private loading: Promise<void> | null = null
  private generation = 0
  private nextId = 1
  private pending = new Map<
    number,
    { onToken: (t: string) => void; resolve: (t: string) => void; reject: (e: Error) => void }
  >()

  constructor() {
    registerHolder('llm', {
      current: () => (this.store.get().status === 'idle' ? null : `llm-${this.store.get().model.id}`),
      release: () => this.dispose(),
      describe: () => this.describe(),
    })
  }

  describe(): Resident | null {
    const s = this.store.get()
    if (s.status === 'idle' || s.status === 'error') return null
    const bytes = Math.round((s.device === 'webgpu' ? s.model.bytes.q4f16 : s.model.bytes.q4) * 1.3 + 150e6)
    return {
      id: `llm-${s.model.id}`,
      label: s.model.label,
      task: 'llm',
      backend: s.device ?? 'wasm',
      bytes,
      gpuBytes: s.device === 'webgpu' ? bytes : 0,
      busy: this.pending.size > 0,
      loading: s.status === 'loading',
    }
  }

  setModel(id: string): void {
    const m = llmModel(id)
    if (m.id === this.store.get().model.id) return
    this.dispose()
    this.store.patch({ model: m, error: null, loadMs: null })
  }

  load(): Promise<void> {
    if (this.loading) return this.loading
    const generation = ++this.generation
    const support = detectSupport()
    const threads = support.wasmThreads ? Math.max(1, Math.min(4, support.hardwareConcurrency)) : 1
    this.loading = this.doLoad(threads, generation)
      .catch(async (err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)
        if (generation !== this.generation || !isOutOfMemory(message)) throw err
        // Out of memory: the other in-page models go first, then one more try.
        this.worker?.terminate()
        this.worker = null
        requireOneRuntime()
        const freed = releaseOtherRuntimes('llm')
        diag.error('llm', `out of memory (${message}); retrying after releasing ${freed.join(', ') || 'nothing'}`)
        return this.doLoad(1, generation)
      })
      .catch((err: unknown) => {
        if (generation === this.generation) {
          this.loading = null
          this.worker?.terminate()
          this.worker = null
          releaseRuntime('llm')
          markSettled()
          this.store.patch({ status: 'error', error: err instanceof Error ? err.message : String(err) })
        }
        throw err
      })
    return this.loading
  }

  private async doLoad(threads: number, generation: number): Promise<void> {
    const m = this.store.get().model
    await Promise.race([ensureDeviceProbe(), new Promise((r) => setTimeout(r, 3000))])
    const gpu = device.get().webgpu
    const onGpu = Boolean(gpu?.available && gpu.f16)
    const id = `llm-${m.id}`
    this.store.patch({ status: 'loading', progress: 0, error: null, device: onGpu ? 'webgpu' : 'wasm' })
    claimRuntime('llm', () => this.dispose())
    markLoading('llm', id)
    const worker =
      factory?.() ?? (new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }) as unknown as LlmWorker)
    if (generation !== this.generation) {
      worker.terminate()
      throw new Error('Load cancelled')
    }
    this.worker = worker
    await new Promise<void>((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message || 'The language model worker failed to start'))
      worker.onmessage = (e) => {
        const msg = e.data
        if (msg.type === 'progress') {
          const total = msg.files.reduce((a, f) => a + f.total, 0)
          this.store.patch({ progress: total ? msg.files.reduce((a, f) => a + f.loaded, 0) / total : 0 })
        } else if (msg.type === 'ready') {
          markSettled()
          void recordDownload({
            id,
            kind: 'llm',
            label: `${m.label} (${onGpu ? '4-bit, WebGPU' : '4-bit, CPU'})`,
            bytes: msg.bytes || (onGpu ? m.bytes.q4f16 : m.bytes.q4),
            cache: 'transformers-cache',
            keys: msg.cacheKeys,
          })
          this.store.patch({ status: 'ready', progress: 1, loadMs: msg.loadMs, device: msg.device })
          resolve()
        } else if (msg.type === 'token') {
          this.pending.get(msg.id)?.onToken(msg.text)
        } else if (msg.type === 'done') {
          this.pending.get(msg.id)?.resolve(msg.text)
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
        repo: m.repo,
        revision: m.revision,
        dtype: onGpu ? 'q4f16' : 'q4',
        device: onGpu ? 'webgpu' : 'wasm',
        threads,
        modelHost: readBrowserSttConfig().modelHost,
        power: readGpuPower(),
      })
    })
  }

  async reply(messages: ChatMessage[], onToken: (t: string) => void, signal: AbortSignal): Promise<string> {
    await this.load()
    const worker = this.worker
    if (!worker) throw new Error('The language model is not running')
    const id = this.nextId++
    touchDownload(`llm-${this.store.get().model.id}`)
    const onAbort = () => worker.postMessage({ type: 'stop' })
    signal.addEventListener('abort', onAbort)
    try {
      return await new Promise<string>((resolve, reject) => {
        this.pending.set(id, { onToken, resolve, reject })
        worker.postMessage({ type: 'generate', id, messages, maxTokens: 160 })
      })
    } finally {
      signal.removeEventListener('abort', onAbort)
      noteActivity(`llm-${this.store.get().model.id}`)
    }
  }

  dispose(): void {
    this.generation += 1
    releaseRuntime('llm')
    if (this.store.get().status === 'loading') markSettled(`llm-${this.store.get().model.id}`)
    this.worker?.terminate()
    this.worker = null
    this.loading = null
    for (const p of this.pending.values()) p.reject(new Error('Unloaded'))
    this.pending.clear()
    this.store.patch({ status: 'idle', progress: 0, error: null })
  }
}

let shared: BrowserLlm | null = null
export function sharedLlm(): BrowserLlm {
  shared ??= new BrowserLlm()
  return shared
}

export interface Endpoint {
  /** Base URL of an OpenAI-compatible API, e.g. https://api.example.com/v1 */
  url: string
  key: string
  model: string
}

const ENDPOINT_KEY = 'bolo.llmEndpoint'

/** The visitor's own endpoint, kept only in this browser's local storage. */
export function readEndpoint(): Endpoint {
  try {
    const v = JSON.parse(globalThis.localStorage?.getItem(ENDPOINT_KEY) ?? '{}') as Partial<Endpoint>
    return { url: v.url ?? '', key: v.key ?? '', model: v.model ?? '' }
  } catch {
    return { url: '', key: '', model: '' }
  }
}

export function saveEndpoint(e: Endpoint): void {
  try {
    globalThis.localStorage?.setItem(ENDPOINT_KEY, JSON.stringify(e))
  } catch {
    // blocked: this visit only
  }
}

/** Streams a reply from an OpenAI-compatible /chat/completions endpoint (server-sent events). */
export class EndpointLlm implements ChatBackend {
  private readonly endpoint: Endpoint

  constructor(endpoint: Endpoint) {
    this.endpoint = endpoint
  }

  async reply(messages: ChatMessage[], onToken: (t: string) => void, signal: AbortSignal): Promise<string> {
    const { url, key, model } = this.endpoint
    const res = await fetch(`${url.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', ...(key ? { Authorization: `Bearer ${key}` } : {}) },
      body: JSON.stringify({ model, messages, stream: true, max_tokens: 200 }),
    })
    if (!res.ok || !res.body) throw new Error(`The endpoint answered HTTP ${res.status}`)
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    let text = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim()
        buf = buf.slice(nl + 1)
        if (!line.startsWith('data:')) continue
        const data = line.slice(5).trim()
        if (data === '[DONE]') return text
        try {
          const t = (JSON.parse(data) as { choices?: { delta?: { content?: string } }[] }).choices?.[0]?.delta?.content
          if (t) {
            text += t
            onToken(t)
          }
        } catch {
          // a keep-alive or a partial line
        }
      }
    }
    return text
  }
}
