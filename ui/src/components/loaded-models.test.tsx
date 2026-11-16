import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LoadedModelsPanel } from '@/components/loaded-models'
import { api } from '@/lib/api/client'
import type { ModelStats, Stats } from '@/lib/api/types'
import { live } from '@/lib/live'
import { countdown, loadedRows, machineTotals, remaining, split } from '@/lib/loaded'

const GB = 1024 ** 3
const MB = 1024 ** 2

function model(over: Partial<ModelStats>): ModelStats {
  return {
    id: 'm',
    task: 'stt',
    loaded: true,
    memory_bytes: 100 * MB,
    idle_s: 10,
    ttl_s: 300,
    active: 0,
    requests: 3,
    latency_ms: { first_partial: null, final: null, ttfa: null },
    rtf: null,
    ...over,
  }
}

// A 3060-shaped machine: one model whole on the GPU, one split llama.cpp model, one on the CPU kept loaded.
const stats: Stats = {
  uptime_s: 100,
  active_streams: 0,
  queue_depth: 0,
  system: {
    cpu_percent: 23.4,
    rss_bytes: 2 * GB,
    threads: 30,
    ram_total_bytes: 32 * GB,
    ram_available_bytes: 20 * GB,
    gpus: [
      {
        index: 0,
        vendor: 'nvidia',
        name: 'NVIDIA GeForce RTX 3060',
        backend: 'cuda',
        total_bytes: 12 * GB,
        free_bytes: 6 * GB,
        used_bytes: 6 * GB,
        unified: false,
        utilization: 41,
        process_bytes: 5 * GB,
        source: 'nvml',
      },
    ],
  },
  totals: { requests: 0, errors: 0, audio_seconds_in: 0, audio_seconds_out: 0 },
  models: [
    model({
      id: 'whisper-large-v3-turbo',
      memory_bytes: 1.6 * GB,
      size_vram_bytes: 1.5 * GB,
      processor: '100% GPU',
      device: 'cuda:0',
      keep_alive_s: 300,
      expires_in_s: 240,
    }),
    model({
      id: 'orpheus-3b',
      task: 'tts',
      memory_bytes: 3 * GB,
      size_vram_bytes: 1.95 * GB,
      processor: '35%/65% CPU/GPU',
      device: 'cuda:0',
      keep_alive_s: 300,
      expires_in_s: 61,
      placement: {
        device: 'cuda:0',
        requested: 'cuda',
        split: 'layers',
        gpu_bytes: 1.95 * GB,
        cpu_bytes: 1.05 * GB,
        processor: '35%/65% CPU/GPU',
        notice: "'orpheus-3b': 19 of 29 layers on cuda:0",
        gpu_layers: 19,
        total_layers: 29,
      },
    }),
    model({ id: 'silero-vad', task: 'vad', memory_bytes: 2 * MB, keep_alive_s: -1, expires_in_s: null }),
    model({ id: 'gone', loaded: false }),
  ],
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('loaded-model helpers', () => {
  it('turns stats into rows, biggest first, with the GPU each one is on', () => {
    const rows = loadedRows(stats)
    expect(rows.map((r) => r.id)).toEqual(['orpheus-3b', 'whisper-large-v3-turbo', 'silero-vad'])
    expect(rows[0]).toMatchObject({ processor: '35%/65% CPU/GPU', onGpu: true, expiresIn: 61 })
    expect(rows[0]?.gpu?.name).toBe('NVIDIA GeForce RTX 3060')
    expect(rows[0]?.notice).toMatch(/19 of 29 layers/)
    expect(rows[2]).toMatchObject({ processor: '100% CPU', onGpu: false, expiresIn: null, gpu: null })
    expect(loadedRows(null)).toEqual([])
  })
  it('keeps a model placed on the GPU there when its memory could not be measured (Windows WDDM)', () => {
    const wddm = model({
      id: 'kokoro',
      memory_bytes: 400 * MB,
      size_vram_bytes: 300 * MB,
      vram_measured: false,
      vram_source: 'plan',
      processor: '100% GPU',
      device: 'cuda:0',
    })
    const [row] = loadedRows({ ...stats, models: [wddm] })
    expect(row).toMatchObject({ processor: '100% GPU', onGpu: true, vramUnknown: true })
    const [measured] = loadedRows({ ...stats, models: [{ ...wddm, vram_measured: true, vram_source: 'device' }] })
    expect(measured).toMatchObject({ vramUnknown: false, onGpu: true })
  })
  it('reads older servers and the demo, which send no placement', () => {
    const [row] = loadedRows({ ...stats, system: { cpu_percent: 1, rss_bytes: 1, threads: 1 }, models: [model({})] })
    expect(row).toMatchObject({ processor: '100% CPU', device: 'cpu', expiresIn: 290, vram: 0 })
  })
  it('counts down and splits', () => {
    expect(countdown(299.2)).toBe('5:00')
    expect(countdown(61)).toBe('1:01')
    expect(countdown(3725)).toBe('1:02:05')
    expect(countdown(null)).toBe('Kept loaded')
    expect(countdown(10, true)).toBe('Unloading…')
    expect(remaining(61, 2500)).toBe(58.5)
    expect(remaining(1, 5000)).toBe(0)
    expect(remaining(null, 5000)).toBeNull()
    expect(split({ processor: '35%/65% CPU/GPU', vram: 0, size: 0 })).toEqual({ cpu: 35, gpu: 65 })
    expect(split({ processor: '100% GPU', vram: 0, size: 0 })).toEqual({ cpu: 0, gpu: 100 })
    expect(split({ processor: '100% CPU', vram: 0, size: 10 })).toEqual({ cpu: 100, gpu: 0 })
  })
  it('sums up the machine', () => {
    const t = machineTotals(stats)
    expect(t).toMatchObject({ cpuPercent: 23.4, ramUsed: 12 * GB, ramTotal: 32 * GB })
    expect(t?.gpus).toMatchObject([
      { label: 'NVIDIA GeForce RTX 3060', used: 6 * GB, total: 12 * GB, bolo: 5 * GB, busy: 41, unified: false },
    ])
  })
})

describe('LoadedModelsPanel', () => {
  const show = () => {
    live.store.patch({ stats, history: [{ t: Date.now(), stats }] })
    return render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <LoadedModelsPanel />
      </QueryClientProvider>,
    )
  }

  it('shows each loaded model with its placement, a countdown and the machine totals', () => {
    show()
    const rows = screen.getAllByTestId('loaded-model')
    expect(rows.map((r) => r.dataset.model)).toEqual(['orpheus-3b', 'whisper-large-v3-turbo', 'silero-vad'])
    expect(within(rows[0]!).getByTestId('processor')).toHaveTextContent('35%/65% CPU/GPU')
    expect(within(rows[0]!).getByTestId('countdown')).toHaveTextContent('1:01')
    expect(within(rows[2]!).getByTestId('countdown')).toHaveTextContent('Kept loaded')
    expect(within(rows[1]!).getByLabelText(/share of NVIDIA GeForce RTX 3060 memory/)).toBeInTheDocument()
    expect(screen.getByTestId('machine-totals')).toHaveTextContent('23%')
    expect(screen.getByTestId('gpu-total')).toHaveTextContent('NVIDIA GeForce RTX 3060 · 41% busy')
    // a model already kept loaded has nothing to keep
    expect(within(rows[2]!).getByRole('button', { name: 'Keep loaded: silero-vad' })).toBeDisabled()
  })

  it('marks a GPU model whose memory could not be measured instead of calling it CPU', () => {
    const wddm = {
      ...stats,
      models: [
        model({
          id: 'kokoro',
          memory_bytes: 400 * MB,
          size_vram_bytes: 300 * MB,
          vram_measured: false,
          vram_source: 'plan',
          processor: '100% GPU',
          device: 'cuda:0',
        }),
      ],
    }
    live.store.patch({ stats: wddm, history: [{ t: Date.now(), stats: wddm }] })
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <LoadedModelsPanel />
      </QueryClientProvider>,
    )
    const [row] = screen.getAllByTestId('loaded-model')
    expect(within(row!).getByTestId('processor')).toHaveTextContent('100% GPU ?')
    expect(within(row!).getByTestId('model-memory')).toHaveTextContent('VRAM unknown')
  })

  it('ticks the countdown down between samples', () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'Date'] })
    show()
    act(() => vi.advanceTimersByTime(3000))
    const row = screen.getAllByTestId('loaded-model')[0]!
    expect(within(row).getByTestId('countdown')).toHaveTextContent('0:58')
  })

  it('unloads and keeps loaded through the API', async () => {
    const unload = vi.spyOn(api, 'unload').mockResolvedValue({})
    const load = vi.spyOn(api, 'load').mockResolvedValue({})
    show()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Unload orpheus-3b' }))
    expect(unload).toHaveBeenCalledWith('orpheus-3b')
    await user.click(screen.getByRole('button', { name: 'Keep loaded: whisper-large-v3-turbo' }))
    expect(load).toHaveBeenCalledWith('whisper-large-v3-turbo', -1)
  })

  it('says when nothing is loaded', () => {
    const empty = { ...stats, models: [] }
    live.store.patch({ stats: empty, history: [] })
    render(
      <QueryClientProvider client={new QueryClient()}>
        <LoadedModelsPanel />
      </QueryClientProvider>,
    )
    expect(screen.getByText('No models loaded')).toBeInTheDocument()
  })
})
