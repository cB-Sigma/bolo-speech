import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { api } from '@/lib/api/client'
import { setServerBase, setTransport } from '@/lib/api/config'
import { compareSpeech, compareTranscription, comparable, fastest, fallbackReason } from '@/lib/compare'
import { MockServer } from '@/mock/engine'
import { handleRequest } from '@/mock/http'
import { createMockWebSocket } from '@/mock/ws'
import ComparePage from './compare'

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

describe('comparing models', () => {
  const audio = new Blob([new Uint8Array(64)], { type: 'audio/wav' })

  it('returns text, timings from the trace and an estimated cost for a cloud model', async () => {
    const local = await compareTranscription('moonshine-tiny-en', audio, undefined)
    const cloud = await compareTranscription('openai-gpt-4o-transcribe', audio, 'en')
    expect(local).toMatchObject({ status: 'ok', answeredBy: null })
    expect(local.text).toBeTruthy()
    expect(local.serverMs).toBeGreaterThan(0)
    expect(local.cost ?? null).toBeNull()
    expect(cloud.status).toBe('ok')
    expect(cloud.cost).toMatchObject({ currency: 'USD', estimated: true })
    expect(cloud.cost?.amount).toBeGreaterThan(0)
    expect(cloud.providerMs).toBeGreaterThan(0)
    expect(fastest([local, cloud])).toBeTruthy()
  })

  it('reports a failure as a result, not an exception', async () => {
    const r = await compareTranscription('sarvam-saaras-v4', audio, 'hi') // its provider is not configured
    expect(r).toMatchObject({ status: 'error', errorCode: 'provider_not_configured' })
    expect(fastest([r])).toBeNull()
  })

  it('compares speech and offers only usable models', async () => {
    const r = await compareSpeech('openai-gpt-4o-mini-tts', { input: 'Hello there.' })
    expect(r.status).toBe('ok')
    expect(r.audioUrl).toMatch(/^blob:/)
    expect(r.firstByteMs).toBeGreaterThan(0)
    const models = (await api.catalog()).models
    const stt = comparable(models, 'stt').map((m) => m.id)
    expect(stt).toContain('openai-gpt-4o-transcribe')
    expect(stt).not.toContain('sarvam-saaras-v4')
    expect(fallbackReason('a -> b; reason=provider_error')).toBe('provider_error')
    expect(fallbackReason(null)).toBeNull()
  })

  it('runs a comparison from the page and marks the cloud model', async () => {
    const user = userEvent.setup()
    wrap(<ComparePage />)
    const panel = await screen.findByTestId('compare-models')
    // one local and one cloud model are preselected
    await waitFor(() => expect(within(panel).getByLabelText('openai-gpt-4o-transcribe')).toBeChecked())
    expect(screen.getByTestId('compare-cloud-notice')).toHaveTextContent('openai')
    await waitFor(() => expect(screen.getByTestId('compare-run')).toBeEnabled())
    await user.upload(
      screen.getByLabelText('Audio file to compare'),
      new File([new Uint8Array(64)], 'call.wav', { type: 'audio/wav' }),
    )
    await user.click(screen.getByTestId('compare-run'))
    const card = await screen.findByTestId('compare-result-openai-gpt-4o-transcribe')
    await waitFor(() => expect(within(card).getByTestId('compare-text')).toBeInTheDocument())
    expect(within(card).getByTestId('cloud-mark')).toBeInTheDocument()
    expect(card).toHaveTextContent(/Estimated cost\s*0\.\d+ USD/)
    const others = screen.getAllByTestId(/^compare-result-/).filter((c) => c !== card)
    expect(others).toHaveLength(1)
    await waitFor(() => expect(others[0]).toHaveTextContent('none (runs here)'))
  })
})
