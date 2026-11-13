import type {
  BenchPoint,
  BenchRequest,
  CatalogModel,
  Job,
  ModelStats,
  Percentiles,
  RequestTrace,
  Sample,
  ServerInfo,
  Stats,
} from '@/lib/api/types'
import { INSTALL_COMMAND } from '@/lib/api/config'
import { STT_RATE } from '@/lib/audio/capture'
import { resample } from '@/lib/audio/resample'
import { decodeWav } from '@/lib/audio/wav'
import { isModelCached, sharedEngine } from '@/lib/browser-stt/engine'
import { BROWSER_MODELS, type BrowserModel } from '@/lib/browser-stt/models'
import { browserTraces, onBrowserTrace, recordBrowserTrace } from '@/lib/browser-stt/traces'
import { percentile } from '@/lib/latency/stats'
import longUrl from '../mock/assets/en-ask-not.wav?url'
import shortUrl from '../mock/assets/en-fellow-americans.wav?url'
import snapshot from './catalog-snapshot.json'

/**
 * The live demo's stand-in for a server. Nothing here is simulated: the
 * catalog is a build-time snapshot of a real server's catalog plus the
 * models that can run in this page, and every statistic and trace comes from
 * work the in-browser recogniser actually did during this visit.
 */

export const NEEDS_SERVER = 'This needs a bolo server. The live demo runs entirely in your browser.'

const SAMPLE_FILES: Record<string, string> = { 'en-ask-not': longUrl, 'en-fellow-americans': shortUrl }

export const DEMO_SAMPLES: Sample[] = [
  {
    id: 'en-fellow-americans',
    kind: 'audio',
    language: 'en',
    seconds: 2.7,
    url: '/api/v1/samples/en-fellow-americans.wav',
    text: 'And so my fellow Americans,',
  },
  {
    id: 'en-ask-not',
    kind: 'audio',
    language: 'en',
    seconds: 11.0,
    url: '/api/v1/samples/en-ask-not.wav',
    text: 'And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.',
  },
]

function browserEntry(m: BrowserModel, installed: boolean, loaded: boolean, progress: number | null): CatalogModel {
  return {
    id: m.id,
    aliases: [m.repo],
    task: 'stt',
    engine: 'onnxruntime-web',
    family: 'Moonshine',
    languages: m.languages,
    tier: 'realtime',
    streaming: false,
    sample_rate: 16000,
    size_bytes: m.approxBytes,
    params: 27_000_000,
    license: m.license,
    commercial_use: true,
    gated: false,
    voices: [],
    features: ['partials', 'in-browser'],
    installed,
    loaded,
    pulling: progress != null ? { job_id: 'browser-load', progress } : null,
    description:
      'Moonshine Tiny, 8-bit quantised, running in this page with ONNX Runtime Web. Downloaded once from the Hugging Face Hub, then kept in the browser cache.',
    source_url: m.source,
    validated: 'untested',
    measured: null,
    runtime: 'browser',
  }
}

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

const needsServer = (): Response => json({ detail: `${NEEDS_SERVER} Install: ${INSTALL_COMMAND}` }, 501)

export function isApiUrl(url: string): boolean {
  return /\/(api\/v1|v1\/(models|audio)|healthz|readyz|metrics)(\/|\?|$)/.test(
    new URL(url, 'http://demo.local/').pathname,
  )
}

function pathOf(url: string): { path: string; query: URLSearchParams } {
  const u = new URL(url, 'http://demo.local/')
  const i = u.pathname.search(/\/(api\/v1|v1|healthz|readyz|metrics)(\/|$)/)
  return { path: i >= 0 ? u.pathname.slice(i) : u.pathname, query: u.searchParams }
}

export class DemoBackend {
  private readonly started = performance.now()
  private jobs = new Map<string, Job>()
  private jobSeq = 1
  private benchRunning = false

  info(): ServerInfo {
    return {
      name: 'bolo',
      version: `${snapshot.server_version} demo`,
      engines: [{ id: 'onnxruntime-web', version: null, available: typeof WebAssembly === 'object', reason: null }],
      device: 'browser',
      cpu_count: navigator.hardwareConcurrency ?? 1,
      platform: 'web',
      auth_required: false,
      data_dir: 'browser cache',
    }
  }

  async catalog(): Promise<CatalogModel[]> {
    const engine = sharedEngine()
    const e = engine.store.get()
    const cached = e.cached ?? (await isModelCached(e.model))
    const browser = BROWSER_MODELS.map((m) =>
      browserEntry(m, cached || e.status === 'ready', e.status === 'ready', e.status === 'loading' ? e.progress : null),
    )
    const server = (snapshot.models as unknown as CatalogModel[]).map((m) => ({
      ...m,
      installed: false,
      loaded: false,
      pulling: null,
      runtime: 'server' as const,
    }))
    return [...browser, ...server]
  }

  stats(): Stats {
    const now = Date.now()
    const engine = sharedEngine()
    const e = engine.store.get()
    const all = browserTraces()
    const recent = all.filter((t) => now - Date.parse(t.started_at) < 5 * 60_000 && t.status === 'ok')
    const nums = (xs: (number | null)[]) => xs.filter((x): x is number => x != null)
    const pct = (xs: number[]): Percentiles | null =>
      xs.length === 0
        ? null
        : {
            p50: Math.round(percentile(xs, 50)),
            p95: Math.round(percentile(xs, 95)),
            p99: Math.round(percentile(xs, 99)),
          }
    const rtfs = nums(recent.map((t) => t.rtf))
    const models: ModelStats[] =
      e.status === 'ready'
        ? [
            {
              id: e.model.id,
              task: 'stt',
              loaded: true,
              // A page cannot measure this; zero is shown as unavailable.
              memory_bytes: 0,
              idle_s: 0,
              ttl_s: null,
              active: engine.pendingCount > 0 ? 1 : 0,
              requests: all.filter((t) => t.final_ms != null).length,
              latency_ms: {
                first_partial: pct(nums(recent.map((t) => t.first_partial_ms))),
                final: pct(nums(recent.map((t) => t.final_ms))),
                ttfa: null,
              },
              rtf: rtfs.length ? { p50: percentile(rtfs, 50), p95: percentile(rtfs, 95) } : null,
            },
          ]
        : []
    return {
      uptime_s: (performance.now() - this.started) / 1000,
      active_streams: engine.pendingCount > 0 ? 1 : 0,
      queue_depth: Math.max(0, engine.pendingCount - 1),
      // A page cannot read its own CPU or memory use; the console shows these as unavailable in the demo.
      system: { cpu_percent: 0, rss_bytes: 0, threads: e.threads },
      totals: {
        requests: all.filter((t) => t.final_ms != null).length,
        errors: all.filter((t) => t.status !== 'ok').length,
        audio_seconds_in: all.reduce((a, t) => a + (t.audio_seconds ?? 0), 0),
        audio_seconds_out: 0,
      },
      models,
    }
  }

  private events(signal?: AbortSignal | null): Response {
    const enc = new TextEncoder()
    let cleanup = () => {}
    const body = new ReadableStream<Uint8Array>({
      start: (controller) => {
        const send = (ev: unknown) => {
          try {
            controller.enqueue(enc.encode(`data: ${JSON.stringify(ev)}\n\n`))
          } catch {
            cleanup()
          }
        }
        send({ type: 'stats', ...this.stats() })
        const timer = setInterval(() => send({ type: 'stats', ...this.stats() }), 1000)
        const off = onBrowserTrace((t) => send({ type: 'request', ...t }))
        const offEngine = sharedEngine().store.subscribe(() => {
          const s = sharedEngine().store.get().status
          if (s === 'ready') send({ type: 'model', id: sharedEngine().store.get().model.id, event: 'loaded' })
          if (s === 'idle') send({ type: 'model', id: sharedEngine().store.get().model.id, event: 'unloaded' })
        })
        cleanup = () => {
          clearInterval(timer)
          off()
          offEngine()
        }
        signal?.addEventListener('abort', () => {
          cleanup()
          try {
            controller.close()
          } catch {
            // already closed
          }
        })
      },
      cancel: () => cleanup(),
    })
    return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
  }

  /** Download and compile the browser model, reported as a pull job. */
  private pull(id: string): Response {
    const model = BROWSER_MODELS.find((m) => m.id === id)
    if (!model) return needsServer()
    const engine = sharedEngine()
    const job: Job = {
      id: `j_${this.jobSeq++}`,
      kind: 'pull',
      model: id,
      state: 'running',
      progress: 0,
      bytes_done: 0,
      bytes_total: model.approxBytes,
      error: null,
    }
    this.jobs.set(job.id, job)
    const off = engine.store.subscribe(() => {
      const e = engine.store.get()
      job.progress = e.progress
      job.bytes_done = e.files.reduce((a, f) => a + f.loaded, 0)
      job.bytes_total = e.files.reduce((a, f) => a + f.total, 0) || model.approxBytes
    })
    engine
      .load()
      .then(() => {
        job.state = 'done'
        job.progress = 1
      })
      .catch((err: unknown) => {
        job.state = 'error'
        job.error = err instanceof Error ? err.message : String(err)
      })
      .finally(off)
    return json({ job_id: job.id }, 202)
  }

  /**
   * A real sweep in the page. "Concurrency" here means how many utterances
   * are handed to the single recogniser worker at once; they are decoded one
   * after another, so latency grows with queueing. That is not how a server
   * with several workers behaves, and the console says so.
   */
  private bench(req: BenchRequest): Response {
    if (req.kind !== 'stt' || !BROWSER_MODELS.some((m) => m.id === req.model)) return needsServer()
    if (this.benchRunning) return json({ detail: 'A benchmark is already running in this page.' }, 409)
    const sampleId = req.sample.replace(/^builtin:/, '')
    const file = SAMPLE_FILES[sampleId]
    if (!file) return json({ detail: `Unknown sample: ${req.sample}` }, 404)
    const levels = [...new Set(req.concurrency)].filter((c) => c > 0 && c <= 16).sort((a, b) => a - b)
    if (levels.length === 0) return json({ detail: 'concurrency must list levels between 1 and 16' }, 422)
    const job: Job<{ points: BenchPoint[] }> = {
      id: `j_${this.jobSeq++}`,
      kind: 'bench',
      model: req.model,
      state: 'running',
      progress: 0,
      error: null,
      result: { points: [] },
    }
    this.jobs.set(job.id, job as Job)
    this.benchRunning = true
    const run = async () => {
      const engine = sharedEngine()
      await engine.load()
      const wav = decodeWav(await (await fetch(file)).arrayBuffer())
      const audio = resample(wav.samples, wav.sampleRate, STT_RATE)
      const seconds = audio.length / STT_RATE
      for (const [i, c] of levels.entries()) {
        const latencies: number[] = []
        const decodes: number[] = []
        let done = 0
        const levelStart = performance.now()
        // Repeat rounds of `c` simultaneous submissions until the level's time budget is spent.
        do {
          const t0 = performance.now()
          const results = await Promise.all(
            Array.from({ length: c }, async () => {
              const r = await engine.transcribe(audio)
              return { total: performance.now() - t0, ms: r.ms, text: r.text }
            }),
          )
          for (const r of results) {
            latencies.push(r.total)
            decodes.push(r.ms)
            done += 1
            recordBrowserTrace({
              model: req.model,
              startedAtPerf: t0,
              phases: [
                { name: 'queue', ms: Math.max(0, r.total - r.ms) },
                { name: 'decode', ms: r.ms },
              ],
              audioSeconds: seconds,
              finalMs: r.total,
              rtf: r.ms / 1000 / seconds,
              text: r.text,
              batchSize: c,
            })
          }
        } while (performance.now() - levelStart < req.duration_s * 1000)
        const wall = (performance.now() - levelStart) / 1000
        const rtfs = decodes.map((ms) => ms / 1000 / seconds)
        job.result?.points.push({
          concurrency: c,
          ok: done,
          errors: 0,
          first_partial_ms: null,
          final_ms: { p50: Math.round(percentile(latencies, 50)), p95: Math.round(percentile(latencies, 95)) },
          ttfa_ms: null,
          rtf: {
            p50: Math.round(percentile(rtfs, 50) * 1000) / 1000,
            p95: Math.round(percentile(rtfs, 95) * 1000) / 1000,
          },
          throughput_x_realtime: Math.round(((done * seconds) / wall) * 100) / 100,
        })
        job.progress = (i + 1) / levels.length
      }
      job.state = 'done'
    }
    run()
      .catch((err: unknown) => {
        job.state = 'error'
        job.error = err instanceof Error ? err.message : String(err)
      })
      .finally(() => {
        this.benchRunning = false
      })
    return json({ job_id: job.id }, 202)
  }

  async handle(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const method = (init?.method ?? 'GET').toUpperCase()
    const { path, query } = pathOf(url)
    if (path === '/healthz' || path === '/readyz') return json({ status: 'ok' })
    if (path === '/api/v1/info') return json(this.info())
    if (path === '/api/v1/events') return this.events(init?.signal)
    if (path === '/api/v1/stats') return json(this.stats())
    if (path === '/api/v1/catalog') {
      const task = query.get('task')
      return json({ models: (await this.catalog()).filter((m) => !task || m.task === task) })
    }
    if (path === '/api/v1/requests') {
      const limit = Number(query.get('limit') ?? 100)
      return json({ requests: browserTraces().slice(0, limit) as RequestTrace[] })
    }
    if (path === '/api/v1/components') {
      const server = snapshot.models as unknown as CatalogModel[]
      return json({
        components: [
          {
            id: 'espeak-ng',
            name: 'eSpeak NG',
            installed: false,
            source: null,
            version: null,
            license: 'GPL-3.0-or-later',
            license_url: 'https://github.com/espeak-ng/espeak-ng/blob/master/COPYING',
            required_by_count: server.filter((m) => m.requires?.includes('espeak-ng')).length,
            installable: false,
            install_blocked_reason: NEEDS_SERVER,
            manual_command: '',
            description: 'Text-to-phoneme front end that some voices need. Not shipped with bolo.',
          },
        ],
      })
    }
    if (path === '/api/v1/samples') return json({ samples: DEMO_SAMPLES })
    const sample = /^\/api\/v1\/samples\/([\w-]+)\.wav$/.exec(path)
    if (sample) {
      const file = SAMPLE_FILES[sample[1] as string]
      return file ? fetch(file) : json({ detail: 'Unknown sample' }, 404)
    }
    if (path === '/api/v1/models/pull' && method === 'POST') {
      return this.pull((JSON.parse(String(init?.body ?? '{}')) as { id: string }).id)
    }
    const job = /^\/api\/v1\/jobs\/([\w-]+)$/.exec(path)
    if (job) {
      const j = this.jobs.get(job[1] as string)
      return j ? json(j) : json({ detail: 'Unknown job' }, 404)
    }
    const action = /^\/api\/v1\/models\/([^/]+)\/(load|unload)$/.exec(path)
    if (action && method === 'POST') {
      const id = decodeURIComponent(action[1] as string)
      if (!BROWSER_MODELS.some((m) => m.id === id)) return needsServer()
      if (action[2] === 'load') await sharedEngine().load()
      else sharedEngine().dispose()
      return json({ id, loaded: action[2] === 'load' })
    }
    const del = /^\/api\/v1\/models\/([^/]+)$/.exec(path)
    if (del && method === 'DELETE') {
      const id = decodeURIComponent(del[1] as string)
      if (!BROWSER_MODELS.some((m) => m.id === id)) return needsServer()
      sharedEngine().dispose()
      await caches.delete('transformers-cache').catch(() => false)
      sharedEngine().store.patch({ cached: false })
      return new Response(null, { status: 204 })
    }
    if (path === '/api/v1/bench' && method === 'POST')
      return this.bench(JSON.parse(String(init?.body ?? '{}')) as BenchRequest)
    return needsServer()
  }
}
