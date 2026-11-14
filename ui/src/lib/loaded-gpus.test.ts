import { describe, expect, it } from 'vitest'
import type { GpuMemory, ModelStats, Stats } from '@/lib/api/types'
import { loadedRows, machineTotals } from '@/lib/loaded'

const GB = 1024 ** 3

function gpu(index: number, over: Partial<GpuMemory> = {}): GpuMemory {
  return {
    index,
    vendor: 'nvidia',
    name: 'NVIDIA GeForce RTX 3090',
    backend: 'cuda',
    total_bytes: 24 * GB,
    free_bytes: 20 * GB,
    used_bytes: 4 * GB,
    unified: false,
    utilization: null,
    process_bytes: 3 * GB,
    source: 'nvml',
    ...over,
  }
}

function stats(gpus: GpuMemory[], models: ModelStats[] = []): Stats {
  return {
    uptime_s: 1,
    active_streams: 0,
    queue_depth: 0,
    system: { cpu_percent: 1, rss_bytes: GB, threads: 4, gpus },
    totals: { requests: 0, errors: 0, audio_seconds_in: 0, audio_seconds_out: 0 },
    models,
  }
}

describe('several GPUs in the Loaded models panel', () => {
  it('gives each GPU its own bar, numbered, and marks the ones bolo may not use', () => {
    const totals = machineTotals(stats([gpu(0), gpu(1, { allowed: false })]))
    expect(totals?.gpus.map((g) => [g.index, g.label, g.allowed])).toEqual([
      [0, 'GPU 0 · NVIDIA GeForce RTX 3090', true],
      [1, 'GPU 1 · NVIDIA GeForce RTX 3090', false],
    ])
    // one GPU: its name alone, as before
    expect(machineTotals(stats([gpu(0)]))?.gpus[0]?.label).toBe('NVIDIA GeForce RTX 3090')
  })

  it('lists the GPUs a spread model is on', () => {
    const spread: ModelStats = {
      id: 'orpheus-3b',
      task: 'tts',
      loaded: true,
      memory_bytes: 3 * GB,
      size_vram_bytes: 3 * GB,
      processor: '100% GPU',
      device: 'cuda:0',
      idle_s: 0,
      ttl_s: 300,
      active: 0,
      requests: 0,
      latency_ms: { first_partial: null, final: null, ttfa: null },
      rtf: null,
      placement: {
        device: 'cuda:0',
        requested: 'cuda',
        split: 'layers',
        gpu_bytes: 3 * GB,
        cpu_bytes: 0,
        processor: '100% GPU',
        notice: null,
        gpus: [0, 1],
      },
    }
    const [row] = loadedRows(stats([gpu(0), gpu(1)], [spread]))
    expect(row?.gpuIndexes).toEqual([0, 1])
    expect(row?.gpu?.index).toBe(0)
  })
})
