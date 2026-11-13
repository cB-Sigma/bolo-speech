import type {
  BenchPoint,
  BenchRequest,
  CatalogModel,
  Component,
  Job,
  ModelStats,
  Percentiles,
  Phase,
  RequestTrace,
  ServerEvent,
  ServerInfo,
  Stats,
} from '@/lib/api/types'
import { percentile } from '@/lib/latency/stats'
import { seedCatalog, type HintSegment } from './catalog'

/** Deterministic PRNG so screenshots and tests are repeatable. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Profile {
  /** ms for the first partial / final at no load; ttfa for TTS. */
  partial: number
  final: number
  rtf: number
  /** Concurrent streams this model sustains before queueing. */
  capacity: number
}

const TIER_PROFILE: Record<string, Profile> = {
  'stt:realtime': { partial: 95, final: 120, rtf: 0.06, capacity: 4 },
  'stt:balanced': { partial: 230, final: 340, rtf: 0.16, capacity: 3 },
  'stt:accuracy': { partial: 620, final: 980, rtf: 0.55, capacity: 1 },
  'tts:realtime': { partial: 55, final: 55, rtf: 0.04, capacity: 6 },
  'tts:balanced': { partial: 150, final: 150, rtf: 0.18, capacity: 3 },
  'tts:accuracy': { partial: 900, final: 900, rtf: 0.8, capacity: 1 },
}

export function profileOf(m: Pick<CatalogModel, 'task' | 'tier' | 'id'>): Profile {
  const p = TIER_PROFILE[`${m.task}:${m.tier}`] ?? (TIER_PROFILE['stt:balanced'] as Profile)
  // Small per-model offset so models in one tier are not identical.
  let h = 0
  for (const c of m.id) h = (h * 31 + c.charCodeAt(0)) >>> 0
  const k = 0.85 + ((h % 100) / 100) * 0.4
  return { partial: p.partial * k, final: p.final * k, rtf: p.rtf * k, capacity: p.capacity }
}

type Listener = (ev: ServerEvent) => void

export interface MockFlags {
  /** Start with eSpeak NG already installed. */
  espeak?: boolean
  /** The server refuses to install components (shows the manual path). */
  noInstall?: boolean
  auth?: boolean
  offline?: boolean
  /** Freeze traffic (for stable test snapshots). */
  quiet?: boolean
}

export const MOCK_KEY = 'bolo-demo'

/** In-memory stand-in for a bolo server: catalog state, jobs, simulated traffic and stats. */
export class MockServer {
  readonly rand = mulberry32(20261113)
  models: CatalogModel[] = seedCatalog()
  jobs = new Map<string, Job>()
  traces: RequestTrace[] = []
  flags: MockFlags
  /** Text the next STT stream is expected to contain (set by sample fetches and TTS). */
  hint: { segments: HintSegment[]; expires: number } | null = null
  private listeners = new Set<Listener>()
  private started = Date.now() - 4 * 3600_000 - 17 * 60_000
  private jobSeq = 1
  private reqSeq = 4200
  private totals = { requests: 18230, errors: 41, audio_seconds_in: 96412.5, audio_seconds_out: 20311.2 }
  private perModel = new Map<string, { requests: number; lastUsed: number; active: number }>()
  private timers: ReturnType<typeof setInterval>[] = []
  private pendingTimeouts = new Set<ReturnType<typeof setTimeout>>()
  liveStreams = 0
  private espeak = false

  constructor(flags: MockFlags = {}) {
    this.flags = flags
    this.espeak = Boolean(flags.espeak)
    this.applyComponents()
    for (const m of this.models) {
      this.perModel.set(m.id, { requests: Math.floor(this.rand() * 4000), lastUsed: Date.now(), active: 0 })
    }
    // Back-fill a few minutes of history so the dashboard is never empty.
    const now = Date.now()
    for (let t = now - 240_000; t < now; t += 300 + this.rand() * 900) this.simulateRequest(t, false)
    if (!flags.quiet) {
      this.timers.push(setInterval(() => this.tick(), 1000))
      this.scheduleTraffic()
    }
  }

  dispose(): void {
    this.timers.forEach(clearInterval)
    this.pendingTimeouts.forEach(clearTimeout)
    this.listeners.clear()
  }

  private later(fn: () => void, ms: number): void {
    const id = setTimeout(() => {
      this.pendingTimeouts.delete(id)
      fn()
    }, ms)
    this.pendingTimeouts.add(id)
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  emit(ev: ServerEvent): void {
    for (const l of this.listeners) l(ev)
  }

  info(): ServerInfo {
    return {
      name: 'bolo',
      version: '0.1.0',
      engines: [
        { id: 'sherpa-onnx', version: '1.12.14', available: true, reason: null },
        { id: 'onnxruntime', version: '1.23.0', available: true, reason: null },
        { id: 'ctranslate2', version: null, available: false, reason: 'not installed (pip install bolo[whisper])' },
      ],
      device: 'cpu',
      cpu_count: 4,
      platform: 'linux-x86_64',
      auth_required: Boolean(this.flags.auth),
      data_dir: '/home/bolo/.cache/bolo',
    }
  }

  model(id: string): CatalogModel | undefined {
    return this.models.find((m) => m.id === id || m.aliases.includes(id))
  }

  /** Load factor 0..~1.6: a slow wave plus occasional bursts. */
  load(at = Date.now()): number {
    const t = (at - this.started) / 1000
    const wave = 0.45 + 0.25 * Math.sin(t / 37) + 0.12 * Math.sin(t / 7.3)
    // Phase chosen so a fresh page starts calm and meets its first burst after about a minute and a half.
    const burst = Math.sin(t / 61 + 5) > 0.93 ? 0.7 : 0
    return Math.max(0.05, wave + burst)
  }

  private scheduleTraffic(): void {
    const next = () => {
      this.simulateRequest(Date.now(), true)
      this.later(next, 250 + this.rand() * (1400 - 900 * Math.min(1, this.load())))
    }
    this.later(next, 400)
  }

  private jitter(base: number, spread = 0.25): number {
    // Log-normal-ish: mostly near base with a long right tail.
    const u = this.rand()
    const tail = u > 0.95 ? 1 + (u - 0.95) * 18 : 1
    return base * (1 - spread / 2 + this.rand() * spread) * tail
  }

  /** Build a trace as the server would report it. */
  makeTrace(
    m: CatalogModel,
    at: number,
    opts: { audioSeconds?: number; chars?: number; text?: string; transport?: string } = {},
  ): RequestTrace {
    const p = profileOf(m)
    const load = this.load(at)
    const pressure = 1 + Math.max(0, load - 0.55) ** 2 * 3.5
    const queue = Math.max(0, this.jitter(2 + Math.max(0, load - 0.7) * 160, 0.8))
    const phases: Phase[] = []
    let cursor = 0
    const add = (name: string, dur: number) => {
      const d = Math.max(0.5, Math.round(dur * 10) / 10)
      phases.push({ name, start_ms: Math.round(cursor * 10) / 10, dur_ms: d })
      cursor += d
    }
    const id = `r_${(this.reqSeq++).toString(36)}`
    if (m.task === 'tts') {
      const chars = opts.chars ?? Math.round(40 + this.rand() * 180)
      const audio = chars / 15
      const first = this.jitter(p.partial * pressure)
      const rest = Math.max(0, audio * 1000 * p.rtf * pressure - first)
      add('queue', queue)
      add('normalize', this.jitter(3))
      add('synthesize_first', first)
      const ttfa = cursor
      add('synthesize_rest', rest)
      add('encode', this.jitter(4 + audio))
      return {
        id,
        kind: 'tts',
        transport: opts.transport ?? (this.rand() > 0.5 ? 'websocket' : 'http'),
        model: m.id,
        started_at: new Date(at).toISOString(),
        status: 'ok',
        error: null,
        audio_seconds: Math.round(audio * 100) / 100,
        chars,
        language: m.languages[0] ?? null,
        phases,
        first_partial_ms: null,
        final_ms: null,
        ttfa_ms: Math.round(ttfa),
        total_ms: Math.round(cursor),
        rtf: Math.round((cursor / 1000 / audio) * 1000) / 1000,
        batch_size: 1,
        text_preview: opts.text ?? 'Your order will arrive tomorrow before six…',
      }
    }
    const audio = opts.audioSeconds ?? Math.round((1.5 + this.rand() * 9) * 10) / 10
    const decode = this.jitter(p.final * pressure * 0.82)
    add('queue', queue)
    add('resample', this.jitter(2))
    add('vad', this.jitter(9))
    const beforeDecode = cursor
    add('decode', decode)
    add('finalize', this.jitter(p.final * 0.1))
    const batch = load > 0.7 ? 1 + Math.floor(this.rand() * 4) : 1
    return {
      id,
      kind: 'stt',
      transport: opts.transport ?? (this.rand() > 0.3 ? 'websocket' : 'http'),
      model: m.id,
      started_at: new Date(at).toISOString(),
      status: 'ok',
      error: null,
      audio_seconds: audio,
      chars: null,
      language: m.languages[0] === 'multi' ? 'en' : (m.languages[0] ?? 'en'),
      phases,
      first_partial_ms: Math.round(beforeDecode + decode * (p.partial / p.final) * 0.9),
      final_ms: Math.round(cursor),
      ttfa_ms: null,
      total_ms: Math.round(cursor),
      rtf: Math.round(p.rtf * pressure * (0.8 + this.rand() * 0.4) * 1000) / 1000,
      batch_size: batch,
      text_preview: opts.text ?? PREVIEWS[Math.floor(this.rand() * PREVIEWS.length)] ?? null,
    }
  }

  record(trace: RequestTrace, emit = true): void {
    this.traces.unshift(trace)
    if (this.traces.length > 600) this.traces.length = 600
    this.totals.requests += 1
    if (trace.status !== 'ok') this.totals.errors += 1
    if (trace.kind === 'stt') this.totals.audio_seconds_in += trace.audio_seconds ?? 0
    else this.totals.audio_seconds_out += trace.audio_seconds ?? 0
    const pm = this.perModel.get(trace.model)
    if (pm) {
      pm.requests += 1
      pm.lastUsed = Date.now()
    }
    if (emit) this.emit({ type: 'request', ...trace })
  }

  private simulateRequest(at: number, emit: boolean): void {
    const loaded = this.models.filter((m) => m.loaded && m.task !== 'vad')
    if (loaded.length === 0) return
    // Weighted towards the first loaded models so one model dominates, as in practice.
    const idx = Math.min(loaded.length - 1, Math.floor(this.rand() ** 1.7 * loaded.length))
    const m = loaded[idx] as CatalogModel
    const trace = this.makeTrace(m, at)
    if (this.rand() < 0.012) {
      trace.status = 'error'
      trace.error = 'client disconnected before final'
      trace.final_ms = null
    }
    this.record(trace, emit)
  }

  stats(): Stats {
    const now = Date.now()
    const load = this.load(now)
    const loaded = this.models.filter((m) => m.loaded)
    const recent = this.traces.filter((t) => now - Date.parse(t.started_at) < 60_000 && t.status === 'ok')
    const pct = (xs: number[]): Percentiles | null =>
      xs.length === 0
        ? null
        : {
            p50: Math.round(percentile(xs, 50)),
            p95: Math.round(percentile(xs, 95)),
            p99: Math.round(percentile(xs, 99)),
          }
    const nums = (xs: (number | null)[]) => xs.filter((x): x is number => x != null)
    const models: ModelStats[] = loaded.map((m) => {
      const mine = recent.filter((t) => t.model === m.id)
      const pm = this.perModel.get(m.id)
      const rtfs = nums(mine.map((t) => t.rtf))
      const active =
        m.task === 'vad'
          ? 0
          : Math.max(0, Math.round(load * 3 * (m.tier === 'realtime' ? 1.2 : 0.5) + this.rand() - 0.5))
      if (pm) pm.active = active
      return {
        id: m.id,
        task: m.task,
        loaded: true,
        memory_bytes: Math.round(m.size_bytes * 1.35 + 40e6),
        idle_s: active > 0 ? 0 : Math.round(((now - (pm?.lastUsed ?? now)) / 1000) * 10) / 10,
        ttl_s: m.task === 'vad' ? null : 300,
        active,
        requests: pm?.requests ?? 0,
        latency_ms: {
          first_partial: pct(nums(mine.map((t) => t.first_partial_ms))),
          final: pct(nums(mine.map((t) => t.final_ms))),
          ttfa: pct(nums(mine.map((t) => t.ttfa_ms))),
        },
        rtf: rtfs.length ? { p50: round3(percentile(rtfs, 50)), p95: round3(percentile(rtfs, 95)) } : null,
      }
    })
    const active = models.reduce((a, m) => a + m.active, 0) + this.liveStreams
    return {
      uptime_s: (now - this.started) / 1000,
      active_streams: active,
      queue_depth: Math.max(0, Math.round((load - 0.85) * 14 + this.rand() - 0.5)),
      system: {
        cpu_percent: Math.round(Math.min(100 * 4, 28 + load * 190 + this.rand() * 18) * 10) / 10,
        rss_bytes: Math.round(310e6 + models.reduce((a, m) => a + m.memory_bytes, 0) + load * 60e6),
        threads: 14 + active,
      },
      totals: {
        ...this.totals,
        audio_seconds_in: round1(this.totals.audio_seconds_in),
        audio_seconds_out: round1(this.totals.audio_seconds_out),
      },
      models,
    }
  }

  private tick(): void {
    this.emit({ type: 'stats', ...this.stats() })
  }

  // ---- optional components ----------------------------------------------

  /** Recompute `available` on catalog entries from what is installed. */
  private applyComponents(): void {
    for (const m of this.models) {
      if (!m.requires?.length) continue
      m.available = this.espeak
      if (!m.available) m.loaded = false
    }
  }

  components(): Component[] {
    return [
      {
        id: 'espeak-ng',
        name: 'eSpeak NG',
        installed: this.espeak,
        source: this.espeak ? 'python-package' : null,
        version: this.espeak ? '1.52.0' : null,
        license: 'GPL-3.0-or-later',
        license_url: 'https://github.com/espeak-ng/espeak-ng/blob/master/COPYING',
        required_by_count: this.models.filter((m) => m.requires?.includes('espeak-ng')).length,
        installable: !this.flags.noInstall,
        install_blocked_reason: this.flags.noInstall
          ? 'The server was started with component installs disabled.'
          : null,
        manual_command: 'pip install espeakng-loader',
        description: 'Text-to-phoneme front end used by Piper and Kokoro voices.',
      },
    ]
  }

  installComponent(id: string, accept: boolean): Job {
    if (id !== 'espeak-ng') throw new HttpError(404, `Unknown component: ${id}`)
    if (!accept) throw new HttpError(422, 'accept_license must be true')
    if (this.flags.noInstall) throw new HttpError(403, 'The server was started with component installs disabled.')
    const job: Job = {
      id: `j_${this.jobSeq++}`,
      kind: 'install',
      model: id,
      state: 'running',
      progress: 0,
      error: null,
    }
    this.jobs.set(job.id, job)
    const step = () => {
      job.progress = Math.min(1, job.progress + 0.12 + this.rand() * 0.1)
      if (job.progress >= 1) {
        job.state = 'done'
        this.espeak = true
        this.applyComponents()
        this.emit({ type: 'job', ...job })
        this.emit({ type: 'component', id, event: 'installed' })
        return
      }
      this.emit({ type: 'job', ...job })
      this.later(step, 220)
    }
    this.later(step, 220)
    return job
  }

  // ---- model management -------------------------------------------------

  pull(id: string): Job {
    const m = this.model(id)
    if (!m) throw new HttpError(404, `Unknown model: ${id}`)
    if (m.gated) throw new HttpError(403, `${m.id} is gated. Accept the licence at ${m.source_url} and set HF_TOKEN.`)
    if (m.pulling) return this.jobs.get(m.pulling.job_id) as Job
    const job: Job = {
      id: `j_${this.jobSeq++}`,
      kind: 'pull',
      model: m.id,
      state: 'running',
      progress: 0,
      bytes_done: 0,
      bytes_total: m.size_bytes,
      error: null,
    }
    this.jobs.set(job.id, job)
    m.pulling = { job_id: job.id, progress: 0 }
    const rate = 38e6 + this.rand() * 30e6 // bytes per second
    const step = () => {
      const done = Math.min(m.size_bytes, (job.bytes_done ?? 0) + rate * 0.2 * (0.6 + this.rand() * 0.8))
      job.bytes_done = Math.round(done)
      job.progress = done / m.size_bytes
      m.pulling = { job_id: job.id, progress: job.progress }
      if (done >= m.size_bytes) {
        job.state = 'done'
        job.progress = 1
        m.pulling = null
        m.installed = true
        this.emit({ type: 'job', ...job })
        this.emit({ type: 'model', id: m.id, event: 'installed' })
        return
      }
      this.emit({ type: 'job', ...job })
      this.later(step, 200)
    }
    this.later(step, 200)
    return job
  }

  async loadModel(id: string): Promise<void> {
    const m = this.model(id)
    if (!m) throw new HttpError(404, `Unknown model: ${id}`)
    if (!m.installed) throw new HttpError(409, `${m.id} is not installed. Pull it first.`)
    if (m.available === false)
      throw new HttpError(409, `${m.id} needs eSpeak NG, which is not installed.`, 'component_missing')
    await sleep(350 + Math.min(2200, m.size_bytes / 6e5))
    m.loaded = true
    this.emit({ type: 'model', id: m.id, event: 'loaded' })
  }

  unloadModel(id: string): void {
    const m = this.model(id)
    if (!m) throw new HttpError(404, `Unknown model: ${id}`)
    m.loaded = false
    this.emit({ type: 'model', id: m.id, event: 'unloaded' })
  }

  removeModel(id: string): void {
    const m = this.model(id)
    if (!m) throw new HttpError(404, `Unknown model: ${id}`)
    m.loaded = false
    m.installed = false
    this.emit({ type: 'model', id: m.id, event: 'removed' })
  }

  // ---- benchmark -----------------------------------------------------------

  bench(req: BenchRequest): Job {
    const m = this.model(req.model)
    if (!m) throw new HttpError(404, `Unknown model: ${req.model}`)
    if (!m.installed) throw new HttpError(409, `${m.id} is not installed. Pull it first.`)
    const levels = [...new Set(req.concurrency)].filter((c) => c > 0).sort((a, b) => a - b)
    if (levels.length === 0) throw new HttpError(422, 'concurrency must list at least one positive level')
    const job: Job<{ points: BenchPoint[] }> = {
      id: `j_${this.jobSeq++}`,
      kind: 'bench',
      model: m.id,
      state: 'running',
      progress: 0,
      error: null,
      result: { points: [] },
    }
    this.jobs.set(job.id, job as Job)
    const p = profileOf(m)
    // The mock compresses time: each level takes ~0.9 s instead of duration_s.
    const per = 900
    levels.forEach((c, i) => {
      this.later(
        () => {
          const over = c / p.capacity
          const stretch = 1 + over ** 2.6 * 0.9 + (over > 1 ? (over - 1) * 2.2 : 0)
          const noise = () => 0.94 + this.rand() * 0.12
          const pc = (base: number): Percentiles => {
            const p50 = base * stretch * noise()
            return { p50: Math.round(p50), p95: Math.round(p50 * (1.45 + Math.min(1.2, over * 0.5)) * noise()) }
          }
          const rtf = p.rtf * stretch
          const point: BenchPoint = {
            concurrency: c,
            ok: Math.max(1, Math.round((req.duration_s * Math.min(c, p.capacity * 1.15)) / (4 * Math.max(1, rtf)))),
            errors: over > 2.2 ? Math.round((over - 2.2) * 3) : 0,
            first_partial_ms: req.kind === 'stt' ? pc(p.partial) : null,
            final_ms: req.kind === 'stt' ? pc(p.final) : null,
            ttfa_ms: req.kind === 'tts' ? pc(p.partial) : null,
            rtf: { p50: round3(rtf * noise()), p95: round3(rtf * 1.5 * noise()) },
            throughput_x_realtime:
              Math.round(
                (Math.min(c, p.capacity * (1 + 0.15 * Math.tanh(over))) / p.rtf / Math.max(1, stretch ** 0.35)) * 10,
              ) / 10,
          }
          job.result?.points.push(point)
          job.progress = (i + 1) / levels.length
          if (i === levels.length - 1) job.state = 'done'
          this.emit({ type: 'job', ...(job as Job) })
        },
        per * (i + 1),
      )
    })
    return job as Job
  }
}

const PREVIEWS = [
  'can you move my nine thirty meeting to…',
  'turn on the lights in the living room',
  'मुझे अपने ऑर्डर की जानकारी चाहिए',
  'what is the weather like in Pune tomorrow',
  'I would like to check my account balance',
  'set a timer for twelve minutes',
  'हाँ, कल शाम छह बजे ठीक रहेगा',
  'play some music in the kitchen',
]

const round3 = (v: number) => Math.round(v * 1000) / 1000
const round1 = (v: number) => Math.round(v * 10) / 10
export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

export class HttpError extends Error {
  readonly status: number
  readonly code: string | null
  constructor(status: number, message: string, code: string | null = null) {
    super(message)
    this.status = status
    this.code = code
  }
}
