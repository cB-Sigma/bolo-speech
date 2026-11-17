import type {
  BenchPoint,
  BenchRequest,
  CatalogModel,
  Component,
  FitWarning,
  HardwareInfo,
  Job,
  GpuMemory,
  ModelStats,
  Percentiles,
  Phase,
  Provider,
  ProviderList,
  ProviderTest,
  ProviderUpdate,
  RequestTrace,
  ServerEvent,
  ServerInfo,
  SetupPlan,
  SetupUse,
  Stats,
} from '@/lib/api/types'
import { percentile } from '@/lib/latency/stats'
import { seedCatalog, type HintSegment } from './catalog'
import { applyFits, mockHardware, mockSetup, mockWarnings } from './hardware'
import { MockTelephony } from './telephony'

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

const PROVIDER_KINDS = [
  {
    kind: 'openai-compatible',
    default_base_url: 'https://api.openai.com/v1',
    terms_url: 'https://openai.com/policies/services-agreement/',
  },
  { kind: 'sarvam', default_base_url: 'https://api.sarvam.ai', terms_url: 'https://www.sarvam.ai/terms-of-service' },
  {
    kind: 'elevenlabs',
    default_base_url: 'https://api.elevenlabs.io',
    terms_url: 'https://elevenlabs.io/terms-of-use',
  },
  { kind: 'deepgram', default_base_url: 'https://api.deepgram.com', terms_url: 'https://deepgram.com/terms' },
  {
    kind: 'soniox',
    default_base_url: 'https://api.soniox.com',
    terms_url: 'https://soniox.com/policies/terms-of-service',
  },
]

/** uccharan 0.6's packs: name, size, catalog models that need it, installed models that need it. */
const MOCK_PACKS: [string, number, number, number][] = [
  ['ca', 194_000, 1, 0], ['cs', 77_000, 1, 0], ['da', 657_000, 1, 0], ['de', 372_000, 3, 1],
  ['en-gb', 707_000, 8, 1], ['en-us', 727_000, 19, 0], ['fi', 136_000, 1, 0], ['fr', 181_000, 5, 0],
  ['hi', 2_000, 5, 0], ['hu', 44_000, 1, 0], ['it', 188_000, 4, 1], ['kokoro-en-gb', 2_066_000, 2, 0],
  ['kokoro-en-us', 1_914_000, 2, 0], ['nb', 99_000, 1, 0], ['nl', 242_000, 2, 0], ['pl', 239_000, 2, 0],
  ['pt', 215_000, 1, 0], ['pt-br', 170_000, 3, 0], ['ro', 308_000, 1, 0], ['sd', 119_000, 0, 0],
  ['sv', 171_000, 1, 0], ['tr', 230_000, 2, 0], ['ur', 542_000, 2, 0], ['wikipron-en-gb', 728_000, 0, 0],
  ['wikipron-en-us', 741_000, 0, 0], ['wikipron-hi', 321_000, 0, 0], ['xlit-bn', 7_836_000, 0, 0],
  ['xlit-gu', 5_951_000, 0, 0], ['xlit-hi', 8_666_000, 0, 0], ['xlit-kn', 6_651_000, 0, 0],
  ['xlit-ml', 6_378_000, 0, 0], ['xlit-mr', 7_355_000, 0, 0], ['xlit-pa', 3_132_000, 0, 0],
  ['xlit-ta', 7_039_000, 0, 0], ['xlit-te', 7_025_000, 0, 0],
] // prettier-ignore

export interface MockFlags {
  /** Start with eSpeak NG already installed. */
  espeak?: boolean
  /** The server refuses to install components (shows the manual path). */
  noInstall?: boolean
  auth?: boolean
  offline?: boolean
  /** Freeze traffic (for stable test snapshots). */
  quiet?: boolean
  /** Pretend the server has a 12 GB NVIDIA GPU: models placed on it, the largest split between GPU and CPU. */
  gpu?: boolean
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
  /** Keep-alive per loaded model, seconds (-1: kept loaded), as a load with `keep_alive` sets it. */
  private keepAlive = new Map<string, number>()
  private timers: ReturnType<typeof setInterval>[] = []
  private pendingTimeouts = new Set<ReturnType<typeof setTimeout>>()
  liveStreams = 0
  private espeak = false
  /** uccharan's language packs that are on the mock server. */
  private packsInstalled = new Set(['fr', 'en-gb'])
  /** Phone calls: providers, agents, numbers and calls. */
  telephony = new MockTelephony()
  /** The machine the mock runs on, and how fast it is against the reference machine. */
  hardware: HardwareInfo = mockHardware()
  /** Cloud providers: `openai` came with the configuration, others are added through the API. */
  providerList: Provider[] = [
    {
      name: 'openai',
      kind: 'openai-compatible',
      base_url: 'https://api.openai.com/v1',
      configured: true,
      api_key_set: true,
      options: {},
      prices: { 'gpt-4o-transcribe': { per_minute: 0.006 } },
      source: 'settings',
      editable: false,
      terms_url: 'https://openai.com/policies/services-agreement/',
    },
  ]

  constructor(flags: MockFlags = {}) {
    this.flags = flags
    this.espeak = Boolean(flags.espeak)
    this.applyComponents()
    applyFits(this.models, this.hardware)
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
        {
          id: 'ctranslate2',
          version: null,
          available: false,
          reason: 'not installed (pip install bolo-speech[whisper])',
        },
      ],
      device: 'cpu',
      cpu_count: 4,
      platform: 'linux-x86_64',
      auth_required: Boolean(this.flags.auth),
      data_dir: '/home/bolo/.cache/bolo',
      auto_pull: true,
      providers: this.providerList.map((p) => ({ name: p.name, kind: p.kind, configured: p.configured })),
      capabilities: {
        providers: {
          endpoint: '/api/v1/providers',
          kinds: PROVIDER_KINDS.map((k) => k.kind),
          editable: true,
          edit_blocked_reason: null,
          configured: this.providerList.filter((p) => p.configured).map((p) => p.name),
          cost_estimates: true,
        },
        telephony: {
          endpoint: '/api/v1/telephony',
          providers: ['twilio', 'plivo', 'telnyx', 'exotel'],
          configured: ['twilio'],
          exposed: ['twilio'],
          validate_signatures: true,
          agents: this.telephony.agents.length,
          sip: { enabled: false, listening: false, port: 5060 },
        },
        recommendations: {
          hardware: '/api/v1/hardware',
          endpoint: '/api/v1/recommendations',
          setup: '/api/v1/recommendations/setup',
          benchmark: '/api/v1/hardware/benchmark',
          uses: ['live', 'files', 'voiceover'],
          warnings: true,
          restricts: false,
        },
      },
    }
  }

  // --- hardware and recommendations --------------------------------------------------------------------------

  /** The micro-benchmark: from then on real-time factors are scaled by a measurement, not a guess. */
  runBenchmark(): HardwareInfo {
    this.hardware = mockHardware(true)
    applyFits(this.models, this.hardware)
    return this.hardware
  }

  setup(use: SetupUse, language: string | null, commercial: boolean): SetupPlan {
    return mockSetup(this.models, this.hardware, use, language, commercial)
  }

  /** Notices for a pull or load of a model that is unlikely to suit this machine. They never stop it. */
  warnings(id: string): FitWarning[] {
    const m = this.model(id)
    return m ? mockWarnings(m) : []
  }

  // --- cloud providers ---------------------------------------------------------------------------------------

  providers(): ProviderList {
    return {
      providers: this.providerList.map((p) => ({ ...p, stats: {} })),
      kinds: PROVIDER_KINDS,
      editable: true,
      edit_blocked_reason: null,
      notice: 'paid cloud API; audio leaves this server',
    }
  }

  /** Cloud models follow their provider: installed once it is configured. */
  private syncCloudModels(): void {
    for (const m of this.models) {
      if (!m.cloud || !m.provider) continue
      const p = this.providerList.find((x) => x.name === m.provider?.name)
      m.provider = { ...m.provider, configured: Boolean(p?.configured), price: p?.prices[m.provider.model] ?? null }
      m.installed = Boolean(p?.configured)
      if (!m.installed) m.loaded = false
    }
  }

  putProvider(name: string, body: ProviderUpdate): Provider {
    const existing = this.providerList.find((p) => p.name === name)
    if (existing && !existing.editable)
      throw new HttpError(400, `provider '${name}' comes from the configuration; change it there`, 'invalid_request')
    const kind = body.kind ?? existing?.kind ?? (PROVIDER_KINDS.find((k) => k.kind === name)?.kind || '')
    const known = PROVIDER_KINDS.find((k) => k.kind === kind)
    if (!known) throw new HttpError(400, `provider '${name}': kind must be one of the known kinds`, 'invalid_request')
    // The key itself is not kept anywhere a response could read it from: only whether one was given.
    const keySet = body.api_key ? true : body.clear_api_key ? false : (existing?.api_key_set ?? false)
    const base_url = body.base_url ?? (existing?.kind === kind ? existing.base_url : known.default_base_url)
    const next: Provider = {
      name,
      kind,
      base_url,
      configured: keySet || (kind === 'openai-compatible' && base_url !== known.default_base_url),
      api_key_set: keySet,
      options: body.options ?? existing?.options ?? {},
      prices: body.prices ?? existing?.prices ?? {},
      source: 'stored',
      editable: true,
      terms_url: base_url === known.default_base_url ? known.terms_url : null,
    }
    this.providerList = existing
      ? this.providerList.map((p) => (p.name === name ? next : p))
      : [...this.providerList, next]
    this.syncCloudModels()
    this.emit({ type: 'provider', name, event: 'saved' })
    return next
  }

  deleteProvider(name: string): void {
    const existing = this.providerList.find((p) => p.name === name)
    if (!existing) throw new HttpError(404, `unknown provider '${name}'`, 'model_not_found')
    if (!existing.editable)
      throw new HttpError(400, `provider '${name}' comes from the configuration; remove it there`, 'invalid_request')
    this.providerList = this.providerList.filter((p) => p.name !== name)
    this.syncCloudModels()
    this.emit({ type: 'provider', name, event: 'removed' })
  }

  testProvider(name: string): ProviderTest {
    const p = this.providerList.find((x) => x.name === name)
    if (!p) throw new HttpError(404, `unknown provider '${name}'`, 'model_not_found')
    if (!p.configured)
      return {
        name,
        ok: false,
        message: `provider '${name}' has no API key yet`,
        code: 'provider_not_configured',
        checked: null,
        latency_ms: 0.4,
      }
    return {
      name,
      ok: true,
      message: p.kind === 'sarvam' ? 'the address answers and did not refuse the key' : 'the key was accepted',
      code: null,
      checked: p.kind === 'sarvam' ? 'reachability' : 'credentials',
      latency_ms: 180 + Math.round(this.rand() * 120),
    }
  }

  /** What a request to a cloud model adds to its trace. */
  cloudTrace(m: CatalogModel, trace: RequestTrace): RequestTrace {
    if (!m.cloud || !m.provider) return trace
    const price = m.provider.price
    const minutes = (trace.audio_seconds ?? 0) / 60
    const amount = (price?.per_minute ?? 0) * minutes + ((price?.per_1k_chars ?? 0) * (trace.chars ?? 0)) / 1000
    return {
      ...trace,
      provider: {
        name: m.provider.name,
        kind: m.provider.kind,
        model: m.provider.model,
        requests: 1,
        latency_ms: Math.round(trace.total_ms * 0.85),
        first_byte_ms: Math.round(trace.total_ms * 0.4),
        bytes_sent: Math.round((trace.audio_seconds ?? 0) * 32000) + (trace.chars ?? 0),
        bytes_received: m.task === 'tts' ? Math.round((trace.audio_seconds ?? 0) * 48000) : 240,
      },
      ...(price
        ? { cost: { amount: Math.round(amount * 1e6) / 1e6, currency: price.currency ?? 'USD', estimated: true } }
        : {}),
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
      const memory = Math.round(m.size_bytes * 1.35 + 40e6)
      const idle = active > 0 ? 0 : Math.round(((now - (pm?.lastUsed ?? now)) / 1000) * 10) / 10
      const keepAlive = this.keepAliveOf(m)
      // The mock machine has no GPU (see mock/hardware.ts): everything runs on the CPU, as on a server without one,
      // unless the `gpu` flag pretends there is one (see placeOnGpu).
      const placed = this.flags.gpu ? this.placeOnGpu(m, memory, loaded) : null
      return {
        id: m.id,
        task: m.task,
        loaded: true,
        memory_bytes: memory,
        idle_s: idle,
        ttl_s: keepAlive,
        size_vram_bytes: 0,
        ram_bytes: memory,
        vram_measured: false,
        processor: '100% CPU',
        device: 'cpu',
        placement: {
          device: 'cpu',
          requested: 'cpu',
          split: 'whole',
          gpu_bytes: 0,
          cpu_bytes: memory,
          processor: '100% CPU',
          notice: null,
        },
        keep_alive_s: keepAlive,
        expires_in_s: keepAlive < 0 ? null : Math.max(0, Math.round((keepAlive - idle) * 10) / 10),
        stopping: false,
        ...placed,
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
        ram_total_bytes: 16 * 1024 ** 3,
        ram_available_bytes: Math.round(11.2 * 1024 ** 3 - models.reduce((a, m) => a + m.memory_bytes, 0)),
        gpus: this.flags.gpu ? [this.mockGpu(models, load)] : [],
        accelerator_source: this.flags.gpu ? 'nvml' : null,
        models_vram_bytes: models.reduce((a, m) => a + (m.size_vram_bytes ?? 0), 0),
        models_ram_bytes: models.reduce((a, m) => a + m.memory_bytes, 0),
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
    const stats = this.stats()
    // keep-alive: a model idle for longer than its keep-alive is unloaded, as the server's reaper does
    for (const m of stats.models) {
      if (m.expires_in_s === 0 && m.active === 0) this.unloadModel(m.id)
    }
    this.emit({ type: 'stats', ...stats })
  }

  /** With the `gpu` flag: VAD on the CPU, the largest model split 35%/65% CPU/GPU, the rest whole on the GPU. */
  private placeOnGpu(m: CatalogModel, memory: number, loaded: CatalogModel[]): Partial<ModelStats> {
    if (m.task === 'vad') return {}
    const largest = loaded.filter((x) => x.task !== 'vad').sort((a, b) => b.size_bytes - a.size_bytes)[0]
    const split = largest?.id === m.id && loaded.length > 2
    const vram = split ? Math.round(memory * 0.65) : memory
    const processor = split ? '35%/65% CPU/GPU' : '100% GPU'
    return {
      size_vram_bytes: vram,
      ram_bytes: memory - vram,
      vram_measured: true,
      processor,
      device: 'cuda:0',
      placement: {
        device: 'cuda:0',
        requested: 'cuda',
        split: split ? 'layers' : 'whole',
        gpu_bytes: vram,
        cpu_bytes: memory - vram,
        processor,
        notice: split ? `'${m.id}': 19 of 29 layers on cuda:0; the rest on the CPU` : null,
        ...(split ? { gpu_layers: 19, total_layers: 29 } : {}),
      },
    }
  }

  private mockGpu(models: ModelStats[], load: number): GpuMemory {
    const ours = models.reduce((a, m) => a + (m.size_vram_bytes ?? 0), 0)
    const total = 12 * 1024 ** 3
    const others = 0.6 * 1024 ** 3
    return {
      index: 0,
      vendor: 'nvidia',
      name: 'NVIDIA GeForce RTX 3060',
      backend: 'cuda',
      total_bytes: total,
      used_bytes: Math.round(ours + others),
      free_bytes: Math.round(total - ours - others),
      unified: false,
      utilization: Math.round(Math.min(100, load * 70)),
      process_bytes: ours,
      source: 'nvml',
    }
  }

  private keepAliveOf(m: CatalogModel): number {
    return this.keepAlive.get(m.id) ?? (m.task === 'vad' ? -1 : 300)
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
      ...MOCK_PACKS.map(([name, size, needed, mine]): Component => {
        const installed = this.packsInstalled.has(name)
        const builtIn = name === 'en-us' || name === 'hi'
        const shareAlike = /^(wikipron|xlit)-/.test(name)
        const language = name.replace(/^(kokoro|wikipron|xlit)-/, '')
        return {
          id: `uccharan-${name}`,
          name: `uccharan language pack '${name}' (${language})`,
          kind: 'language-pack',
          installed,
          source: installed ? 'uccharan' : null,
          version: installed ? '0.6.6' : null,
          license: shareAlike ? 'CC BY-SA 4.0' : 'Apache-2.0',
          license_url: 'https://github.com/cb-Sigma/uccharan/blob/main/DATA_LICENSES.md',
          built_in: builtIn,
          required_by_count: builtIn ? 0 : needed,
          required_by_installed_count: builtIn ? 0 : mine,
          used_by_count: builtIn ? needed : null,
          removable: installed && !builtIn,
          installable: !installed && !this.flags.noInstall,
          install_blocked_reason: this.flags.noInstall
            ? 'The server was started with component installs disabled.'
            : null,
          manual_command: `python -m uccharan.packs ${name}`,
          size_bytes: size,
          languages: [language],
          description: `uccharan's ${name} pack.`,
        }
      }),
    ]
  }

  removeComponent(id: string): void {
    const name = id.replace(/^uccharan-/, '')
    if (!this.packsInstalled.has(name)) throw new HttpError(400, `cannot remove ${id}: it is not installed`)
    this.packsInstalled.delete(name)
    this.emit({ type: 'component', id, event: 'removed' })
  }

  installComponent(id: string, accept: boolean): Job {
    const pack = MOCK_PACKS.find(([name]) => `uccharan-${name}` === id)
    if (id !== 'espeak-ng' && !pack) throw new HttpError(404, `Unknown component: ${id}`)
    if (id === 'espeak-ng' && !accept) throw new HttpError(422, 'accept_license must be true')
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
        if (pack) this.packsInstalled.add(pack[0])
        else this.espeak = true
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

  async loadModel(id: string, keepAlive?: number): Promise<void> {
    const m = this.model(id)
    if (!m) throw new HttpError(404, `Unknown model: ${id}`)
    if (!m.installed) throw new HttpError(409, `${m.id} is not installed. Pull it first.`)
    if (m.available === false)
      throw new HttpError(409, `${m.id} needs eSpeak NG, which is not installed.`, 'component_missing')
    if (keepAlive !== undefined) {
      this.keepAlive.set(m.id, keepAlive)
      const pm = this.perModel.get(m.id)
      if (pm) pm.lastUsed = Date.now()
    }
    if (m.loaded) return
    await sleep(350 + Math.min(2200, m.size_bytes / 6e5))
    m.loaded = true
    this.emit({ type: 'model', id: m.id, event: 'loaded' })
  }

  unloadModel(id: string): void {
    const m = this.model(id)
    if (!m) throw new HttpError(404, `Unknown model: ${id}`)
    m.loaded = false
    this.keepAlive.delete(m.id)
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
