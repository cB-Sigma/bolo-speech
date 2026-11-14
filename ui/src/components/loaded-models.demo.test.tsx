import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { LoadedModelsPanel } from '@/components/loaded-models'
import type * as Config from '@/lib/api/config'
import { live } from '@/lib/live'

// The live demo: models run in the browser tab, and the page cannot read its own CPU or memory use.
vi.mock('@/lib/api/config', async (original) => ({
  ...(await original<typeof Config>()),
  isDemo: true,
}))

describe('LoadedModelsPanel in the demo', () => {
  it('lists the browser model, says why there are no totals, and offers Unload', () => {
    const stats = {
      uptime_s: 1,
      active_streams: 0,
      queue_depth: 0,
      system: { cpu_percent: 0, rss_bytes: 0, threads: 4 },
      totals: { requests: 0, errors: 0, audio_seconds_in: 0, audio_seconds_out: 0 },
      models: [
        {
          id: 'moonshine-tiny-browser',
          task: 'stt' as const,
          loaded: true,
          memory_bytes: 0,
          idle_s: 0,
          ttl_s: null,
          active: 0,
          requests: 2,
          latency_ms: { first_partial: null, final: null, ttfa: null },
          rtf: null,
        },
      ],
    }
    live.store.patch({ stats, history: [{ t: Date.now(), stats }] })
    render(
      <QueryClientProvider client={new QueryClient()}>
        <LoadedModelsPanel />
      </QueryClientProvider>,
    )
    expect(screen.getByTestId('machine-totals')).toHaveTextContent(/runs? in this browser tab/)
    const row = screen.getByTestId('loaded-model')
    expect(within(row).getByTestId('processor')).toHaveTextContent('100% CPU')
    expect(within(row).getByTestId('countdown')).toHaveTextContent('Kept loaded')
    expect(within(row).getByRole('button', { name: 'Unload moonshine-tiny-browser' })).toBeEnabled()
  })
})
