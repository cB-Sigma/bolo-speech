import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LoadedModelsPanel } from '@/components/loaded-models'
import type * as Config from '@/lib/api/config'
import type { Stats } from '@/lib/api/types'
import { device } from '@/lib/device'
import { live } from '@/lib/live'

// The live demo: models run in the browser tab, and the page cannot read its own CPU or memory use.
vi.mock('@/lib/api/config', async (original) => ({
  ...(await original<typeof Config>()),
  isDemo: true,
}))

const model = (over: Partial<Stats['models'][number]>): Stats['models'][number] => ({
  id: 'moonshine-tiny',
  task: 'stt',
  loaded: true,
  memory_bytes: 190e6,
  memory_estimated: true,
  idle_s: 42,
  ttl_s: 300,
  keep_alive_s: 300,
  expires_in_s: 258,
  active: 0,
  requests: 2,
  latency_ms: { first_partial: null, final: null, ttfa: null },
  rtf: null,
  ...over,
})

function show(models: Stats['models']) {
  const stats: Stats = {
    uptime_s: 1,
    active_streams: 0,
    queue_depth: 0,
    system: { cpu_percent: 0, rss_bytes: 0, threads: 4 },
    totals: { requests: 0, errors: 0, audio_seconds_in: 0, audio_seconds_out: 0 },
    models,
  }
  live.store.patch({ stats, history: [{ t: Date.now(), stats }] })
  render(
    <QueryClientProvider client={new QueryClient()}>
      <LoadedModelsPanel />
    </QueryClientProvider>,
  )
}

describe('LoadedModelsPanel in the demo', () => {
  beforeEach(() => {
    device.patch({
      platform: 'desktop',
      memoryGb: 8,
      cores: 8,
      webgpu: {
        available: true,
        f16: true,
        maxBufferSize: 2 ** 31,
        maxStorageBufferBindingSize: 2 ** 30,
        adapter: 'apple metal-3',
      },
    })
  })

  it('shows the device totals, the estimate, idle time and the keep-alive control', () => {
    show([model({})])
    const totals = screen.getByTestId('machine-totals')
    expect(within(totals).getByTestId('browser-memory')).toHaveTextContent(/190 MB of about .* estimate/)
    expect(within(totals).getByTestId('browser-device')).toHaveTextContent('8 GB memory reported')
    expect(within(totals).getByTestId('browser-gpu')).toHaveTextContent('apple metal-3')
    // An Apple GPU shares memory with the CPU: the panel says what a split does and does not buy.
    expect(within(totals).getByTestId('shared-memory-note')).toHaveTextContent(/not with total memory/)
    expect(within(totals).getByRole('combobox', { name: /Unload models after/ })).toHaveTextContent('5 min')
    const row = screen.getByTestId('loaded-model')
    expect(within(row).getByTestId('processor')).toHaveTextContent('100% CPU')
    expect(within(row).getByTestId('model-memory')).toHaveTextContent('(estimate)')
    expect(within(row).getByTestId('countdown')).toHaveTextContent('4:18')
    expect(within(row).getByTestId('idle')).toHaveTextContent('idle 0:42')
    expect(within(row).getByRole('button', { name: 'Keep loaded: moonshine-tiny' })).toBeEnabled()
    expect(within(row).getByRole('button', { name: 'Unload moonshine-tiny' })).toBeEnabled()
  })

  it('shows a split model by bytes and a kept-loaded one without a countdown', () => {
    show([
      model({
        id: 'whisper-base-q8',
        memory_bytes: 400e6,
        size_vram_bytes: 300e6,
        processor: '25%/75% CPU/GPU',
        device: 'WebGPU + WASM CPU',
        keep_alive_s: -1,
        ttl_s: null,
        expires_in_s: null,
      }),
    ])
    const row = screen.getByTestId('loaded-model')
    expect(within(row).getByTestId('processor')).toHaveTextContent('25%/75% CPU/GPU')
    expect(within(row).getByTestId('model-memory')).toHaveTextContent('WebGPU 300 MB')
    expect(within(row).getByTestId('countdown')).toHaveTextContent('Kept loaded')
    expect(within(row).getByRole('button', { name: 'Keep loaded: whisper-base-q8' })).toBeDisabled()
  })
})
