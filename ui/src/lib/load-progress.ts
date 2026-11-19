/** The stages of a staged load, in order, as the downloads tray lists them. */
export const LOAD_STEPS = [
  { id: 'runtime', label: 'Runtime' },
  { id: 'model', label: 'Model' },
  { id: 'compiling', label: 'Compile' },
  { id: 'warmup', label: 'Warm up' },
] as const

/** What a staged in-page load reports: the recogniser's and the voices' engines (runtime, model, compile, warm-up). */
export interface StagedLoad {
  status: string
  progress?: number
  stage?: string | null
  runtime?: { loaded: number; total: number } | null
}

/**
 * One 0-100 figure for a load: the stages weighted by how long they usually take (the runtime's download, the
 * model's files, compiling, warming up). A load that reports no stage is its files' progress alone.
 */
export function overallPercent(s: StagedLoad): number {
  if (s.status === 'ready') return 100
  if (s.status !== 'loading') return 0
  const files = Math.min(1, s.progress ?? 0)
  if (s.stage === undefined) return files * 100
  const runtime = s.runtime && s.runtime.total > 0 ? Math.min(1, s.runtime.loaded / s.runtime.total) : 0
  switch (s.stage) {
    case 'runtime':
      return 2 + runtime * 28
    case 'model':
      return 30 + files * 50
    case 'compiling':
      return 85
    case 'warmup':
      return 95
    default:
      return 1
  }
}

/** Human description of where a load is, with bytes where they are known. */
export function describeLoad(
  s: {
    status: string
    stage?: string | null
    runtime?: { loaded: number; total: number } | null
    files: { loaded: number; total: number }[]
    model: { approxBytes: number }
  },
  fmt: (bytes: number) => string,
  noun = 'model',
): string {
  if (s.status !== 'loading') return ''
  if (s.stage === 'runtime') {
    return s.runtime && s.runtime.total > 0
      ? `Downloading the runtime · ${fmt(s.runtime.loaded)} of ${fmt(s.runtime.total)}`
      : 'Downloading the runtime'
  }
  if (s.stage === 'model') {
    const loaded = s.files.reduce((a, f) => a + f.loaded, 0)
    const seen = s.files.reduce((a, f) => a + f.total, 0)
    // Files announce their size as they start; until most have, the listed size is the better denominator.
    const total = seen >= s.model.approxBytes * 0.9 ? seen : s.model.approxBytes
    return loaded > 0
      ? `Downloading the ${noun} · ${fmt(loaded)} of ${fmt(Math.max(total, loaded))}`
      : `Downloading the ${noun} · ${fmt(total)}`
  }
  if (s.stage === 'compiling') return `Compiling the ${noun}`
  if (s.stage === 'warmup') return 'Warming up'
  return 'Starting'
}
