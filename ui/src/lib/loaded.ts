import type { GpuMemory, ModelStats, Stats } from './api/types'

/** One loaded model as the "Loaded models" panel shows it. */
export interface LoadedRow {
  id: string
  task: string
  size: number
  vram: number
  ram: number
  /** "100% GPU", "100% CPU", "35%/65% CPU/GPU", "remote", "cloud". */
  processor: string
  device: string
  onGpu: boolean
  /** On the GPU, but how much memory it took could not be measured (Windows under WDDM): `vram` is the plan. */
  vramUnknown: boolean
  /** Seconds left when the stats were sampled; null: kept loaded. */
  expiresIn: number | null
  keepAlive: number | null
  stopping: boolean
  active: number
  requests: number
  notice: string | null
  /** The GPU the model's memory is on, when one is reported. */
  gpu: GpuMemory | null
  /** CUDA numbers of the GPUs it is on (several when a llama.cpp model is spread over them). */
  gpuIndexes: number[]
  /** Seconds since its last request. */
  idle: number
  /** `size` is an estimate (the browser demo). */
  estimated: boolean
}

function gpuFor(device: string, gpus: GpuMemory[]): GpuMemory | null {
  const [family = '', index = '0'] = device.toLowerCase().split(':')
  const backend =
    family === 'cuda' || family === 'gpu' ? 'cuda' : ['coreml', 'mps', 'metal'].includes(family) ? 'metal' : ''
  if (!backend) return null
  const idx = Number.parseInt(index, 10) || 0
  return gpus.find((g) => g.backend === backend && g.index === idx) ?? null
}

/** Loaded models from a stats sample. Older servers (and the browser demo) send no placement: 100% CPU. */
export function loadedRows(stats: Stats | null): LoadedRow[] {
  if (!stats) return []
  const gpus = stats.system.gpus ?? []
  return stats.models
    .filter((m) => m.loaded)
    .map((m: ModelStats): LoadedRow => {
      const device = m.device ?? 'cpu'
      const vram = m.size_vram_bytes ?? 0
      const ttl = m.keep_alive_s ?? m.ttl_s
      const expiresIn =
        m.expires_in_s !== undefined ? m.expires_in_s : ttl == null || ttl < 0 ? null : Math.max(0, ttl - m.idle_s)
      return {
        id: m.id,
        task: m.task,
        size: m.memory_bytes,
        vram,
        ram: m.ram_bytes ?? Math.max(0, m.memory_bytes - vram),
        processor: m.processor ?? '100% CPU',
        device,
        onGpu: vram > 0 || m.vram_source === 'plan',
        vramUnknown: m.vram_source === 'plan',
        expiresIn,
        keepAlive: ttl ?? null,
        stopping: m.stopping ?? false,
        active: m.active,
        requests: m.requests,
        notice: m.placement?.notice ?? null,
        gpu: gpuFor(device, gpus),
        gpuIndexes: m.placement?.gpus ?? [],
        idle: m.idle_s,
        estimated: m.memory_estimated ?? false,
      }
    })
    .sort((a, b) => b.size - a.size || a.id.localeCompare(b.id))
}

/** Seconds left after `elapsedMs` since the sample; null stays null ("kept loaded"). */
export function remaining(expiresIn: number | null, elapsedMs: number): number | null {
  if (expiresIn == null) return null
  return Math.max(0, expiresIn - elapsedMs / 1000)
}

/** "0:42", "12:05", "1:02:03" */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}

/** "4:59", "1:02:03", "Kept loaded", "Unloading…" */
export function countdown(seconds: number | null, stopping = false): string {
  if (stopping) return 'Unloading…'
  if (seconds == null) return 'Kept loaded'
  const s = Math.max(0, Math.ceil(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}

/** CPU and GPU shares of a model by bytes, in percent (for the split bar). */
export function split(row: Pick<LoadedRow, 'vram' | 'size' | 'processor'>): { gpu: number; cpu: number } {
  if (row.processor === '100% GPU') return { gpu: 100, cpu: 0 }
  const m = /^(\d+)%\/(\d+)% CPU\/GPU$/.exec(row.processor)
  if (m) return { cpu: Number(m[1]), gpu: Number(m[2]) }
  if (row.size > 0 && row.vram > 0) {
    const gpu = Math.round((100 * row.vram) / row.size)
    return { gpu, cpu: 100 - gpu }
  }
  return { gpu: 0, cpu: 100 }
}

export interface MachineTotals {
  cpuPercent: number
  rss: number
  ramUsed: number | null
  ramTotal: number | null
  gpus: {
    /** Its number, when the machine has several GPUs. */
    index: number
    label: string
    /** false: the `gpus` setting leaves it out. */
    allowed: boolean
    used: number | null
    total: number | null
    bolo: number | null
    busy: number | null
    unified: boolean
  }[]
}

export function machineTotals(stats: Stats | null): MachineTotals | null {
  if (!stats) return null
  const sys = stats.system
  const total = sys.ram_total_bytes ?? null
  return {
    cpuPercent: sys.cpu_percent,
    rss: sys.rss_bytes,
    ramTotal: total,
    ramUsed: total != null && sys.ram_available_bytes != null ? total - sys.ram_available_bytes : null,
    gpus: (sys.gpus ?? []).map((g, _, all) => ({
      index: g.index,
      // several GPUs (often the same model twice): say which is which
      label: all.length > 1 ? `GPU ${g.index} · ${g.name ?? g.vendor}` : (g.name ?? `${g.vendor} GPU ${g.index}`),
      allowed: g.allowed !== false,
      used: g.used_bytes,
      total: g.total_bytes,
      bolo: g.process_bytes,
      busy: g.utilization,
      unified: g.unified,
    })),
  }
}
