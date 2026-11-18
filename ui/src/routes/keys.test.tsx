import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { getApiKey, setApiKey, setServerBase, setTransport } from '@/lib/api/config'
import type { ApiKeyInfo } from '@/lib/api/types'
import KeysPage from './keys'

let keys: ApiKeyInfo[]
let sent: { url: string; method: string; body: string }[]

function info(name: string, scopes: string[]): ApiKeyInfo {
  return {
    id: `key_${name}`,
    name,
    hint: 'bolo_abcd…wxyz',
    scopes,
    created_at: '2026-11-18T00:00:00Z',
    expires_at: null,
    revoked_at: null,
    active: true,
    rate_limit: 60,
    burst: null,
    quota: 100,
    quota_period: 'day',
    usage: {
      requests: 40,
      period: '2026-11-18',
      period_requests: 40,
      quota_remaining: 60,
      by_scope: {},
      rejected: 3,
      last_used_at: null,
    },
  }
}

const reply = (body: unknown, status = 200) =>
  Promise.resolve(new Response(body === undefined ? null : JSON.stringify(body), { status }))

beforeEach(() => {
  keys = []
  sent = []
  setApiKey(null)
  setServerBase('http://bolo.test/')
  setTransport({
    fetch: (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const method = init?.method ?? 'GET'
      const body = typeof init?.body === 'string' ? init.body : ''
      sent.push({ url, method, body })
      if (url.endsWith('api/v1/keys') && method === 'GET') {
        return reply({ auth: keys.length > 0, root_key: false, you: null, scopes: {}, periods: [], data: keys })
      }
      if (url.endsWith('api/v1/keys') && method === 'POST') {
        const req = JSON.parse(body) as { name: string; scopes: string[] }
        const k = info(req.name, req.scopes)
        keys.push(k)
        return reply({ ...k, key: 'bolo_secret_value' }, 201)
      }
      return reply({}, 404)
    },
  })
})
afterEach(() => {
  cleanup()
  setServerBase(null)
  setApiKey(null)
})

const wrap = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TooltipProvider>
        <MemoryRouter>
          <KeysPage />
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  )

describe('Keys page', () => {
  it('creates the first key with the admin scope, shows it once and keeps it for the console', async () => {
    const user = userEvent.setup()
    wrap()
    expect(await screen.findByText('Authentication is off')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /create key/i }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByLabelText('Name')).toHaveValue('console')
    await user.type(within(dialog).getByLabelText('Requests per minute'), '60')
    await user.click(within(dialog).getByRole('button', { name: 'Create key' }))
    expect(await screen.findByTestId('new-key')).toHaveTextContent('bolo_secret_value')
    const post = sent.find((r) => r.method === 'POST')
    expect(JSON.parse(post?.body ?? '{}')).toMatchObject({ name: 'console', scopes: ['admin'], rate_limit: 60 })
    expect(getApiKey()).toBe('bolo_secret_value')
    await user.click(screen.getByRole('button', { name: 'Done' }))
    const row = await screen.findByTestId('key-key_console')
    expect(row).toHaveTextContent('60/min · 100 per day')
    expect(row).toHaveTextContent('40 of 100 per day')
    expect(row).not.toHaveTextContent('bolo_secret_value')
  })
})
