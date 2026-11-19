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
