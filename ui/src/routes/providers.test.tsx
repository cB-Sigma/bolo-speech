import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { api } from '@/lib/api/client'
import { setServerBase, setTransport } from '@/lib/api/config'
import { MockServer } from '@/mock/engine'
import { handleRequest } from '@/mock/http'
import { createMockWebSocket } from '@/mock/ws'
import ProvidersPage from './providers'

let server: MockServer
let sent: { url: string; method: string; body: string }[]

beforeEach(() => {
  server = new MockServer({ quiet: true })
  sent = []
  setServerBase('http://bolo.test/')
  setTransport({
    fetch: (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      sent.push({ url, method: init?.method ?? 'GET', body: typeof init?.body === 'string' ? init.body : '' })
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

const wrap = (ui: React.ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TooltipProvider>
        <MemoryRouter>{ui}</MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  )

describe('Providers page', () => {
  it('lists providers with the notice, and never shows a key', async () => {
    wrap(<ProvidersPage />)
    const row = await screen.findByTestId('provider-openai')
    expect(screen.getByTestId('cloud-notice')).toHaveTextContent('Paid cloud API; audio leaves this server')
    expect(within(row).getByTestId('provider-state-openai')).toHaveTextContent('Configured')
    expect(row).toHaveTextContent('from the configuration')
    expect(row).toHaveTextContent('https://api.openai.com/v1')
    // a provider from the configuration can be tested but not edited or removed here
    expect(within(row).queryByRole('button', { name: /edit/i })).toBeNull()
    await userEvent.click(within(row).getByRole('button', { name: /test/i }))
    expect(await within(row).findByTestId('provider-test-openai')).toHaveTextContent('The key was accepted')
  })

  it('adds a provider: the key goes out once and is not in anything that comes back', async () => {
    const user = userEvent.setup()
    wrap(<ProvidersPage />)
    await screen.findByTestId('provider-openai')
    await user.click(screen.getByRole('button', { name: /add provider/i }))
    const dialog = await screen.findByRole('dialog')
    await user.type(within(dialog).getByLabelText('Name'), 'sarvam')
    const key = within(dialog).getByTestId('provider-key')
    expect(key).toHaveAttribute('type', 'password')
    await user.type(key, 'sv-secret-123')
    await user.click(within(dialog).getByRole('button', { name: /add provider/i }))
    const row = await screen.findByTestId('provider-sarvam')
    const put = sent.find((r) => r.method === 'PUT')
    expect(put?.url).toContain('/api/v1/providers/sarvam')
    expect(JSON.parse(put?.body ?? '{}')).toMatchObject({ api_key: 'sv-secret-123' })
    expect(within(row).getByTestId('provider-state-sarvam')).toHaveTextContent('Configured')
    expect(JSON.stringify(await api.providers())).not.toContain('sv-secret-123')
    expect(document.body.innerHTML).not.toContain('sv-secret-123')
    // its cloud models are installed now
    const models = (await api.catalog()).models
    expect(models.find((m) => m.id === 'sarvam-saaras-v4')).toMatchObject({ installed: true, cloud: true })

    // editing: the key field starts empty, and saving without typing sends no key
    await user.click(within(row).getByRole('button', { name: /edit/i }))
    const edit = await screen.findByRole('dialog')
    expect(within(edit).getByTestId('provider-key')).toHaveValue('')
    expect(within(edit).getByTestId('provider-key')).toHaveAttribute('placeholder', expect.stringMatching(/stored/))
    await user.click(within(edit).getByRole('button', { name: /^save$/i }))
    await waitFor(() => expect(sent.filter((r) => r.method === 'PUT')).toHaveLength(2))
    expect('api_key' in JSON.parse(sent.filter((r) => r.method === 'PUT')[1]?.body ?? '{}')).toBe(false)
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect((await api.providers()).providers.find((p) => p.name === 'sarvam')?.api_key_set).toBe(true)
  })
})
