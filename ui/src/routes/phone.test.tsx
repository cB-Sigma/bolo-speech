import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { setServerBase, setTransport } from '@/lib/api/config'
import { visibleNav } from '@/lib/nav'
import { MockServer } from '@/mock/engine'
import { handleRequest } from '@/mock/http'
import { createMockWebSocket } from '@/mock/ws'
import PhonePage from './phone'

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

describe('Phone page', () => {
  it('is in the navigation only when the server has telephony', () => {
    expect(visibleNav({ wordLists: false, telephony: true }).map((n) => n.to)).toContain('/phone')
    expect(visibleNav({ wordLists: false }).map((n) => n.to)).not.toContain('/phone')
  })

  it('shows each provider with the address to paste, and no credential', async () => {
    wrap(<PhonePage />)
    const twilio = await screen.findByTestId('phone-provider-twilio')
    expect(within(twilio).getByTestId('phone-provider-state-twilio')).toHaveTextContent('Ready')
    expect(twilio).toHaveTextContent('https://voice.example.org/telephony/twilio/voice')
    expect(twilio).toHaveTextContent("'A call comes in' = Webhook, HTTP POST")
    expect(within(twilio).getByRole('button', { name: /copy twilio a call comes in/i })).toBeInTheDocument()
    const plivo = screen.getByTestId('phone-provider-plivo')
    expect(within(plivo).getByTestId('phone-provider-state-plivo')).toHaveTextContent('Not set up')
    expect(plivo).toHaveTextContent('telephony.providers.plivo.auth_token is not set')
    expect(screen.getByTestId('phone-compliance')).toHaveTextContent('DND')
    expect(document.body.innerHTML).not.toContain('***')
    expect(screen.queryByTestId('phone-unvalidated')).toBeNull()
  })

  it('adds an agent and maps a number to it', async () => {
    const user = userEvent.setup()
    wrap(<PhonePage />)
    const configured = await screen.findByTestId('phone-agent-front-desk')
    expect(configured).toHaveTextContent('from the configuration')
    expect(within(configured).queryByRole('button', { name: /edit/i })).toBeNull()
    await user.click(screen.getByRole('button', { name: /add agent/i }))
    const dialog = await screen.findByRole('dialog')
    await user.type(within(dialog).getByLabelText('Name'), 'orders')
    await user.type(within(dialog).getByLabelText('Greeting'), 'Orders, hello.')
    await user.type(within(dialog).getByLabelText('Transfer number'), 'reception')
    expect(within(dialog).getByRole('alert')).toHaveTextContent('international form')
    expect(within(dialog).getByRole('button', { name: /^add agent$/i })).toBeDisabled()
    await user.clear(within(dialog).getByLabelText('Transfer number'))
    await user.type(within(dialog).getByLabelText('Transfer number'), '+14155550199')
    await user.click(within(dialog).getByRole('button', { name: /^add agent$/i }))
    const row = await screen.findByTestId('phone-agent-orders')
    expect(row).toHaveTextContent('Greets with “Orders, hello.”')
    expect(row).toHaveTextContent('transfers to +14155550199')
    const put = sent.find((r) => r.method === 'PUT')
    expect(put?.url).toContain('/api/v1/telephony/agents/orders')
    expect(JSON.parse(put?.body ?? '{}')).toMatchObject({
      greeting: 'Orders, hello.',
      transfer_number: '+14155550199',
      instructions: null,
      max_duration_s: 600,
    })

    await user.type(screen.getByLabelText('Number'), '+14155550142')
    await user.click(screen.getByRole('button', { name: /map number/i }))
    await waitFor(() => expect(screen.getByTestId('phone-numbers')).toHaveTextContent('+14155550142'))
    const mapped = sent.filter((r) => r.method === 'PUT')[1]
    expect(decodeURIComponent(mapped?.url ?? '')).toContain('/api/v1/telephony/numbers/+14155550142')
    expect(JSON.parse(mapped?.body ?? '{}')).toEqual({ agent: 'front-desk' })
  })

  it('lists calls and opens one with its latency, transcript and timeline', async () => {
    const user = userEvent.setup()
    wrap(<PhonePage />)
    const row = await screen.findByTestId('phone-call-call_000001_00001')
    expect(row).toHaveTextContent('+14155550123 → +14155550100')
    expect(row).toHaveTextContent('Caller hung up')
    expect(row).toHaveTextContent('1,061 ms')
    await user.click(within(row).getByRole('button'))
    const detail = await screen.findByTestId('call-detail')
    expect(detail).toHaveTextContent('A caller asked about opening hours')
    const turns = within(detail).getByTestId('call-turns')
    expect(turns).toHaveTextContent('interrupted by the caller')
    expect(within(turns).getByRole('img')).toHaveAccessibleName(/End of speech 500 ms, Speech to text 90 ms/)
    expect(within(detail).getByTestId('call-transcript')).toHaveTextContent('What time do you close today?')
    expect(within(detail).getByTestId('call-transcript')).toHaveTextContent('(cut off by the caller)')
    expect(within(detail).getByTestId('call-timeline')).toHaveTextContent('barge in')
  })

  it('simulates a call and reports how it went', async () => {
    const user = userEvent.setup()
    wrap(<PhonePage />)
    await screen.findByTestId('phone-provider-twilio')
    await user.click(screen.getByRole('button', { name: /simulate a call/i }))
    const report = await screen.findByTestId('simulation-report')
    expect(report).toHaveTextContent('The call went as expected')
    expect(report).toHaveTextContent('1,090 ms')
    expect(report).toHaveTextContent('Passed: agent spoke')
    const post = sent.find((r) => r.method === 'POST')
    expect(post?.url).toContain('/api/v1/telephony/simulate')
    expect(JSON.parse(post?.body ?? '{}')).toEqual({ agent: 'front-desk', provider: 'twilio', scenario: 'basic' })
    await waitFor(() => expect(screen.getByTestId('phone-calls').querySelectorAll('tbody tr')).toHaveLength(2))
  })
})
