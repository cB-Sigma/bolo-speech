import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ThemeProvider } from '@/hooks/use-theme'
import { setServerBase, setTransport } from '@/lib/api/config'
import { setPref } from '@/lib/prefs'
import { MockServer } from '@/mock/engine'
import { handleRequest } from '@/mock/http'
import { createMockWebSocket } from '@/mock/ws'
import ModelsPage from './models'

let server: MockServer
let sent: { url: string; method: string; body: string }[]

beforeEach(() => {
  server = new MockServer({ quiet: true })
  sent = []
  setPref('modelsView', 'cards')
  setPref('modelsSort', 'default')
  setPref('pickDismissed', false)
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

const wrap = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TooltipProvider>
        <MemoryRouter>
          <ThemeProvider>
            <ModelsPage />
            <Toaster />
          </ThemeProvider>
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  )

const cardIds = () => screen.getAllByTestId('model-card').map((c) => c.querySelector('.num')?.textContent ?? '')
const card = (id: string) =>
  screen.getAllByTestId('model-card').find((c) => c.querySelector('.num')?.textContent === id) as HTMLElement

describe('Models page: fit for this machine', () => {
  it('shows the hardware card, with the estimate named as one until the benchmark has run', async () => {
    const user = userEvent.setup()
    wrap()
    const hw = await screen.findByTestId('hardware-card')
    await within(hw).findByText(/Intel\(R\) Xeon\(R\) Processor @ 2\.10GHz · 4 usable cores/)
    expect(hw).toHaveTextContent('none found')
    expect(hw).toHaveTextContent('it never restricts what you can pull or load')
    expect(within(hw).getByTestId('hardware-speed')).toHaveTextContent(
      '1.00× the reference machine (a rough estimate from the CPU)',
    )
    expect(hw).toHaveTextContent('rougher until the benchmark has run')
    await user.click(within(hw).getByTestId('run-benchmark'))
    await waitFor(() =>
      expect(within(hw).getByTestId('hardware-speed')).toHaveTextContent(
        '0.94× the reference machine (measured by the benchmark)',
      ),
    )
    expect(sent.some((r) => r.method === 'POST' && r.url.includes('/api/v1/hardware/benchmark'))).toBe(true)
    expect(hw).not.toHaveTextContent('rougher until')
  })

  it('puts a fit line on every card and marks a few as recommended, hiding none', async () => {
    wrap()
    await screen.findAllByTestId('model-card')
    const total = server.models.length
    expect(screen.getAllByTestId('model-card')).toHaveLength(total)
    expect(screen.getAllByTestId('fit-line')).toHaveLength(total)
    // the model that cannot run well here is on the page like any other, and says so
    const heavy = card('indic-parler-tts')
    expect(within(heavy).getByTestId('fit-line')).toHaveAttribute('data-level', 'unlikely')
    expect(within(heavy).getByTestId('fit-line')).toHaveTextContent('Unlikely to fit here')
    expect(within(card('moonshine-tiny-en')).getByTestId('fit-line')).toHaveTextContent(/Comfortable here · .*measured/)
    expect(within(card('openai-gpt-4o-transcribe')).getByTestId('fit-line')).toHaveTextContent(
      'Cloud: this machine does not matter',
    )
    const badges = screen.getAllByTestId('recommended-badge')
    expect(badges.length).toBeGreaterThan(0)
    expect(badges.length).toBeLessThanOrEqual(3 * new Set(server.models.map((m) => m.task)).size)
    expect(within(heavy).queryByTestId('recommended-badge')).toBeNull()
    expect(screen.getByTestId('filter-unlikely')).toHaveAttribute('aria-checked', 'false')
    expect(screen.queryByTestId('unlikely-hidden')).toBeNull()
  })

  it('hides the unlikely ones only when the labelled toggle is switched on, and says how many', async () => {
    const user = userEvent.setup()
    wrap()
    await screen.findAllByTestId('model-card')
    const total = server.models.length
    await user.click(screen.getByTestId('filter-unlikely'))
    await waitFor(() => expect(screen.getAllByTestId('model-card')).toHaveLength(total - 1))
    expect(cardIds()).not.toContain('indic-parler-tts')
    const note = screen.getByTestId('unlikely-hidden')
    expect(note).toHaveTextContent('1 unlikely to fit this machine hidden by your filter')
    await user.click(within(note).getByRole('button', { name: 'show them' }))
    await waitFor(() => expect(screen.getAllByTestId('model-card')).toHaveLength(total))
    expect(cardIds()).toContain('indic-parler-tts')
  })

  it('"best for this machine" reorders and keeps every model', async () => {
    wrap()
    await screen.findAllByTestId('model-card')
    const before = cardIds()
    expect(screen.getByTestId('sort-models')).toHaveTextContent('Sort: installed first')
    setPref('modelsSort', 'fit')
    await waitFor(() => expect(screen.getByTestId('sort-models')).toHaveTextContent('Sort: best for this machine'))
    const after = cardIds()
    expect([...after].sort()).toEqual([...before].sort())
    expect(after).not.toEqual(before)
    // within text to speech: a recommended voice leads, the one that cannot run well here is last
    const tts = server.models.filter((m) => m.task === 'tts').map((m) => m.id)
    const order = after.filter((id) => tts.includes(id))
    expect(order[order.length - 1]).toBe('indic-parler-tts')
    expect(within(card(order[0] as string)).getByTestId('recommended-badge')).toBeInTheDocument()
  })

  it('explains the fit in the details and still offers the model', async () => {
    const user = userEvent.setup()
    const heavy = server.models.find((m) => m.id === 'indic-parler-tts')
    if (heavy) heavy.gated = false
    wrap()
    await screen.findAllByTestId('model-card')
    await user.click(within(card('indic-parler-tts')).getAllByRole('button')[0] as HTMLElement)
    const section = await screen.findByTestId('fit-section')
    expect(section).toHaveTextContent('On this machine')
    expect(section).toHaveTextContent('Unlikely to fit')
    expect(within(section).getByTestId('fit-reason')).toHaveTextContent(/slower than real time here/)
    expect(within(section).getByTestId('fit-notice')).toHaveTextContent(
      'That is a recommendation, not a restriction: you can still pull and load it.',
    )
    expect(section).toHaveTextContent('Expected real-time factor')
    // the pull button is there and works: the warning rides along, nothing is refused
    await user.keyboard('{Escape}')
    await user.click(screen.getByTestId('pull-indic-parler-tts'))
    await waitFor(() =>
      expect(sent.some((r) => r.method === 'POST' && r.url.endsWith('/api/v1/models/pull'))).toBe(true),
    )
    expect(await screen.findByText('Pulling indic-parler-tts anyway')).toBeInTheDocument()
    expect(await screen.findByText(/is unlikely to fit or to run well on this machine/)).toBeInTheDocument()
    expect(server.models.find((m) => m.id === 'indic-parler-tts')?.pulling).not.toBeNull()
  })

  it('pulls a model that suits the machine without a warning', async () => {
    const user = userEvent.setup()
    wrap()
    await screen.findAllByTestId('model-card')
    await user.click(screen.getByTestId('pull-whisper-small'))
    expect(await screen.findByText('Pulling whisper-small')).toBeInTheDocument()
    expect(screen.queryByText('Pulling whisper-small anyway')).toBeNull()
    expect(server.warnings('whisper-small')).toEqual([])
  })

  it('offers "Pick for me" on a server with nothing installed and pulls the proposal in one click', async () => {
    const user = userEvent.setup()
    for (const m of server.models) {
      if (!m.cloud) {
        m.installed = false
        m.loaded = false
      }
    }
    wrap()
    const banner = await screen.findByTestId('first-run-pick')
    expect(banner).toHaveTextContent('No models installed yet')
    await user.click(within(banner).getByTestId('first-run-pick-button'))
    const dialog = await screen.findByTestId('pick-dialog')
    const stt = await within(dialog).findByTestId('pick-stt')
    const tts = within(dialog).getByTestId('pick-tts')
    expect(within(dialog).getByTestId('pick-vad')).toHaveTextContent('optional')
    expect(within(dialog).getByTestId('pick-hardware')).toHaveTextContent(
      'a rough estimate until the benchmark has run',
    )
    expect(within(stt).getByTestId('fit-line')).toHaveTextContent(/Comfortable here|Usable here/)
    expect(sent.some((r) => r.url.includes('/api/v1/recommendations/setup?use=live&language=en'))).toBe(true)
    const plan = server.setup('live', 'en', false)
    expect(stt).toHaveTextContent(plan.picks[0]?.model?.id ?? 'missing')
    expect(tts).toHaveTextContent(plan.picks[1]?.model?.id ?? 'missing')
    const button = within(dialog).getByTestId('pick-pull')
    expect(button).toHaveTextContent(`Pull ${plan.pull.length} models`)
    await user.click(button)
    await waitFor(() => {
      const pulls = sent.filter((r) => r.method === 'POST' && r.url.endsWith('/api/v1/models/pull'))
      expect(pulls.map((r) => (JSON.parse(r.body) as { id: string }).id).sort()).toEqual([...plan.pull].sort())
    })
    await waitFor(() => expect(screen.queryByTestId('pick-dialog')).toBeNull())
  })

  it('does not nag: the first-run prompt goes away when dismissed, and is absent once something is installed', async () => {
    const user = userEvent.setup()
    wrap()
    await screen.findAllByTestId('model-card')
    expect(screen.queryByTestId('first-run-pick')).toBeNull() // the mock server starts with models installed
    expect(await screen.findByTestId('pick-for-me')).toBeInTheDocument() // still one click away on the hardware card
    cleanup()
    for (const m of server.models) if (!m.cloud) m.installed = false
    wrap()
    const banner = await screen.findByTestId('first-run-pick')
    await user.click(within(banner).getByRole('button', { name: 'Not now' }))
    await waitFor(() => expect(screen.queryByTestId('first-run-pick')).toBeNull())
  })

  it('shows nothing about fit on a server that does not report it', async () => {
    for (const m of server.models) delete m.fit
    const info = server.info.bind(server)
    server.info = () => {
      const doc = info()
      delete doc.capabilities?.recommendations
      return doc
    }
    wrap()
    await screen.findAllByTestId('model-card')
    expect(screen.queryByTestId('fit-line')).toBeNull()
    expect(screen.queryByTestId('hardware-card')).toBeNull()
    expect(screen.queryByTestId('filter-unlikely')).toBeNull()
    expect(screen.queryByTestId('sort-models')).toBeNull()
    expect(sent.some((r) => r.url.includes('/api/v1/hardware'))).toBe(false)
  })
})
