import type {
  CatalogModel,
  FitWarning,
  HardwareInfo,
  ModelFit,
  Recommendation,
  SetupPlan,
  SetupUse,
} from '@/lib/api/types'

const GIB = 1024 ** 3

/** The machine the mock server "runs on": a small CPU-only VM, like the one the catalog was measured on. */
export function mockHardware(benchmarked = false): HardwareInfo {
  return {
    system: 'linux',
    cpu: {
      model: 'Intel(R) Xeon(R) Processor @ 2.10GHz',
      vendor: 'GenuineIntel',
      architecture: 'x86_64',
      physical_cores: 4,
      logical_cores: 4,
      usable_cores: 4,
      features: ['fma', 'sse4.2', 'avx', 'avx2', 'avx512', 'avx512_vnni', 'amx'],
    },
    memory: { total_bytes: 16 * GIB, available_bytes: 11.2 * GIB, limit_bytes: null, usable_bytes: 16 * GIB },
    disk: { path: '/home/bolo/.cache/bolo', total_bytes: 250 * GIB, free_bytes: 41 * GIB },
    gpus: [],
    gpus_known: true,
    runtimes: {
      onnxruntime: { installed: true, version: '1.23.0', providers: ['CPUExecutionProvider'], gpu: false },
      'sherpa-onnx': { installed: true, version: '1.12.14', gpu: null },
      ctranslate2: { installed: false, version: null, cuda_devices: null, gpu: false },
    },
    container: {
      detected: false,
      kind: null,
      virtual_machine: true,
      hypervisor: null,
      cpu_quota: null,
      memory_limit_bytes: null,
    },
    device: 'cpu',
    threads: 4,
    notes: [],
    detected_at: '2026-11-14T09:00:00Z',
    summary: '4 cores (Intel(R) Xeon(R) Processor @ 2.10GHz, AMX), 16.0 GB RAM, no GPU found',
    speed: benchmarked
      ? {
          factor: 0.94,
          source: 'benchmark',
          spread: [0.75, 1.33],
          note: 'measured by the micro-benchmark (onnxruntime, 4 threads) on 2026-11-14',
          benchmark: { workload: 'onnxruntime', threads: 4, ms: 3.09, reference_ms: 2.9, speed: 0.94 },
        }
      : {
          factor: 1,
          source: 'heuristic',
          spread: [0.5, 2],
          note: 'estimated from the CPU (x86 avx512 vnni, 4 inference threads), not measured; `bolo recommend --benchmark` measures it',
          benchmark: null,
        },
  }
}

const RAM = 16 * GIB

/** A fit like the server's, from what a mock entry says about itself. Deterministic. */
export function mockFit(m: CatalogModel, factor = 1, source: 'benchmark' | 'heuristic' = 'heuristic'): ModelFit {
  const marks = { recommended: false, recommended_for: [] as string[], default_for: [] as string[] }
  if (m.status === 'pending' || m.runnable === false) {
    return {
      level: 'unavailable',
      label: 'not available yet',
      expected_rtf: null,
      rtf_range: null,
      rtf_basis: 'unknown',
      scaled_by: null,
      device: null,
      memory: null,
      reason: 'Its files are not published yet, so it cannot be pulled; hardware is not the issue.',
      notes: [],
      ...marks,
    }
  }
  if (m.cloud) {
    return {
      level: 'cloud',
      label: 'cloud',
      expected_rtf: null,
      rtf_range: null,
      rtf_basis: 'unknown',
      scaled_by: null,
      device: 'cloud',
      memory: null,
      reason: "Runs at the provider, so this machine's hardware does not matter; audio or text leaves this server.",
      notes: [],
      ...marks,
    }
  }
  const need = Math.round(m.size_bytes * 1.2 + 150e6)
  const memory = { needed_bytes: need, available_bytes: RAM, kind: 'ram', basis: 'estimated' }
  // real-time factor on the reference machine, by size: what a model of that many parameters does on four cores
  const params = m.params ?? m.size_bytes / 4
  const reference = m.task === 'vad' ? 0.006 : Math.max(0.03, (params / 1e9) * (m.task === 'tts' ? 10 : 0.5))
  const measured = m.validated === 'cpu'
  const rtf = reference / factor
  const spread: [number, number] = source === 'benchmark' ? [0.75, 1.33] : [0.5, 2]
  const level = rtf <= 0.4 ? 'comfortable' : rtf <= 1 ? 'usable' : rtf <= 5 ? 'slow' : 'unlikely'
  const pace =
    rtf < 1 ? `About ${Math.round(1 / rtf)}x faster than real time` : `About ${Math.round(rtf)}x slower than real time`
  const by = source === 'benchmark' ? 'by the micro-benchmark' : 'by a rough estimate from the CPU'
  return {
    level,
    label: level === 'unlikely' ? 'unlikely to fit' : level,
    expected_rtf: Number(rtf.toPrecision(2)),
    rtf_range: [Number((rtf / spread[1]).toPrecision(2)), Number((rtf / spread[0]).toPrecision(2))],
    rtf_basis: measured ? 'measured' : 'estimated',
    scaled_by: source,
    device: 'cpu',
    memory,
    engine_available: true,
    reason: measured
      ? `${pace} here (real-time factor ~${rtf.toPrecision(2)}, scaled ${by} from the ${reference.toPrecision(2)} measured on the reference machine); needs about ${(need / GIB).toFixed(1)} GB of the 16.0 GB of memory.`
      : `Not measured: ${pace.toLowerCase()} here (real-time factor ~${rtf.toPrecision(2)}), estimated from a measured model of the same architecture, and scaled ${by}; needs about ${(need / GIB).toFixed(1)} GB of the 16.0 GB of memory.`,
    notes: [],
    ...marks,
  }
}

const LEVELS = ['comfortable', 'usable', 'cloud', 'slow', 'unlikely', 'unavailable']
const speaks = (m: CatalogModel, lang: string | null) =>
  !lang || m.languages.length === 0 || m.languages.includes(lang) || m.languages.includes('multi')

function ranked(models: CatalogModel[], task: string, lang: string | null, live: boolean | null): CatalogModel[] {
  const tier = live
    ? { realtime: 0, balanced: 1, accuracy: 2 }
    : ({ accuracy: 0, balanced: 1, realtime: 2 } as Record<string, number>)
  const key = (m: CatalogModel) =>
    LEVELS.indexOf(m.fit?.level ?? 'unavailable') * 100 +
    (m.fit?.rtf_basis === 'measured' ? 0 : 10) +
    (tier[m.tier] ?? 3)
  return models
    .filter((m) => m.task === task && speaks(m, lang))
    .map((m, i) => ({ m, i, k: key(m) }))
    .sort((a, b) => a.k - b.k || a.i - b.i)
    .map((x) => x.m)
}

/** Give every mock entry a fit and the "recommended for this machine" marks, the way the server does. */
export function applyFits(models: CatalogModel[], hw: HardwareInfo): void {
  for (const m of models)
    m.fit = mockFit(m, hw.speed.factor, hw.speed.source === 'benchmark' ? 'benchmark' : 'heuristic')
  const tasks = [...new Set(models.map((m) => m.task))]
  for (const task of tasks) {
    const langs = new Set<string | null>([null])
    for (const m of models) if (m.task === task) for (const l of m.languages) if (l !== 'multi') langs.add(l)
    for (const lang of langs) {
      const good = ranked(models, task, lang, null).filter(
        (m) => m.fit?.level === 'comfortable' || m.fit?.level === 'usable',
      )
      const families = new Set<string>()
      const picks = good.filter((m) => !families.has(m.family) && families.add(m.family)).slice(0, 3)
      const mark = lang ?? '*'
      picks.forEach((m, i) => {
        if (!m.fit) return
        m.fit.recommended_for = [...(m.fit.recommended_for ?? []), mark]
        if (i === 0) m.fit.default_for = [...(m.fit.default_for ?? []), mark]
        if (mark === '*') m.fit.recommended = true
      })
    }
  }
}

/** The notices a pull or load carries. They never stop it. */
export function mockWarnings(m: CatalogModel): FitWarning[] {
  const fit = m.fit
  if (!fit || (fit.level !== 'slow' && fit.level !== 'unlikely')) return []
  const what =
    fit.level === 'unlikely'
      ? 'is unlikely to fit or to run well on this machine'
      : 'is expected to be slower than real time on this machine'
  return [
    {
      code: 'hardware_fit',
      model: m.id,
      level: fit.level,
      message: `'${m.id}' ${what}. ${fit.reason} This is a recommendation, not a restriction.`,
    },
  ]
}

const row = (m: CatalogModel, rank: number, recommended: boolean, isDefault: boolean): Recommendation => ({
  rank,
  id: m.id,
  task: m.task,
  family: m.family,
  engine: m.engine,
  tier: m.tier,
  languages: m.languages,
  size_bytes: m.size_bytes,
  installed: m.installed,
  license: m.license,
  commercial_use: m.commercial_use !== false,
  cloud: Boolean(m.cloud),
  validated: m.validated,
  quality: null,
  recommended,
  default: isDefault,
  fit: m.fit as ModelFit,
})

const USES: Record<SetupUse, { label: string; tasks: [string, boolean, boolean][] }> = {
  live: {
    label: 'Live conversation',
    tasks: [
      ['stt', true, false],
      ['tts', true, false],
      ['vad', true, true],
    ],
  },
  files: {
    label: 'Transcription of files',
    tasks: [
      ['stt', false, false],
      ['diarization', false, true],
    ],
  },
  voiceover: { label: 'Voice-over', tasks: [['tts', false, false]] },
}

export function mockSetup(
  models: CatalogModel[],
  hw: HardwareInfo,
  use: SetupUse,
  language: string | null,
  commercial: boolean,
): SetupPlan {
  const plan = USES[use]
  const picks = plan.tasks.map(([task, live, optional]) => {
    const list = ranked(models, task, language, live).filter(
      (m) => !m.cloud && (!commercial || m.commercial_use !== false),
    )
    const good = list.filter((m) => m.fit?.level === 'comfortable' || m.fit?.level === 'usable')
    const chosen = good[0] ?? list.find((m) => m.fit?.level === 'slow') ?? null
    return {
      task,
      optional,
      realtime: live,
      model: chosen ? row(chosen, 1, good.includes(chosen), true) : null,
      alternatives: good
        .filter((m) => m !== chosen)
        .slice(0, 2)
        .map((m, i) => row(m, i + 2, true, false)),
      note: chosen ? null : 'No model for this task is expected to run on this machine as it is.',
    }
  })
  const chosen = picks.map((p) => p.model).filter((m): m is Recommendation => m != null)
  const pull = chosen.filter((m) => !m.installed)
  return {
    use,
    label: plan.label,
    language,
    commercial,
    local_only: true,
    hardware: hw.summary,
    speed: hw.speed,
    picks,
    pull: pull.map((m) => m.id),
    download_bytes: pull.reduce((a, m) => a + m.size_bytes, 0),
  }
}
