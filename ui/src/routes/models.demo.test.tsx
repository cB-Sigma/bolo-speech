import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import type * as Config from '@/lib/api/config'
import { setServerBase, setTransport } from '@/lib/api/config'
import { MockServer } from '@/mock/engine'
import { handleRequest } from '@/mock/http'
import { createMockWebSocket } from '@/mock/ws'
import ModelsPage from './models'

// The live demo build: there is no server, so "this machine" would be the visitor's browser.
vi.mock('@/lib/api/config', async (original) => ({
  ...(await original<typeof Config>()),
  isDemo: true,
}))

let server: MockServer
let sent: string[]

beforeEach(() => {
  server = new MockServer({ quiet: true })
  sent = []
  setServerBase('http://bolo.test/')
  setTransport({
    fetch: (input, init) => {
      sent.push(typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url)
      return handleRequest(server, input, init)
    },
    WebSocket: createMockWebSocket(server),
  })
})
afterEach(() => {
  cleanup()
  server.dispose()
  setServerBase(null)
})

describe('Models page in the live demo', () => {
  it('says nothing about hardware fit, even if the data were there', async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <TooltipProvider>
          <MemoryRouter>
            <ModelsPage />
          </MemoryRouter>
        </TooltipProvider>
      </QueryClientProvider>,
    )
    await screen.findAllByTestId('model-card')
    expect(server.models.every((m) => m.fit)).toBe(true) // the stand-in server does report fits
    expect(screen.queryByTestId('fit-line')).toBeNull()
    expect(screen.queryByTestId('recommended-badge')).toBeNull()
    expect(screen.queryByTestId('hardware-card')).toBeNull()
    expect(screen.queryByTestId('first-run-pick')).toBeNull()
    expect(screen.queryByTestId('filter-unlikely')).toBeNull()
    expect(screen.queryByTestId('sort-models')).toBeNull()
    expect(sent.some((url) => url.includes('/api/v1/hardware') || url.includes('/recommendations'))).toBe(false)
  })
})
