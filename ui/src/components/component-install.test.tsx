import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { ComponentNotice, componentSentence, ComponentsCard } from '@/components/component-install'
import { missingComponent } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import { setServerBase, setTransport } from '@/lib/api/config'
import type { CatalogModel, Component } from '@/lib/api/types'
import { TtsSession } from '@/lib/tts/session'
import { MockServer } from '@/mock/engine'
import { handleRequest } from '@/mock/http'
import { createMockWebSocket } from '@/mock/ws'

let server: MockServer
function install(flags = {}) {
  server = new MockServer({ quiet: true, ...flags })
  setServerBase('http://bolo.test/')
  setTransport({ fetch: (input, init) => handleRequest(server, input, init), WebSocket: createMockWebSocket(server) })
}
const wrap = (ui: React.ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {ui}
    </QueryClientProvider>,
  )

beforeEach(() => install())
afterEach(() => {
  cleanup()
  server.dispose()
  setServerBase(null)
})

const espeak = async (): Promise<Component> => (await api.components()).components[0]!
const model = async (id: string): Promise<CatalogModel> => (await api.catalog()).models.find((m) => m.id === id)!

describe('components API (mock)', () => {
  it('reports eSpeak NG as optional, GPL-licensed and needed by some voices', async () => {
    const c = await espeak()
    expect(c).toMatchObject({
      id: 'espeak-ng',
      installed: false,
      source: null,
      license: 'GPL-3.0-or-later',
      installable: true,
      required_by_count: 2,
    })
    expect(await model('kokoro-82m')).toMatchObject({ requires: ['espeak-ng'], available: false, loaded: false })
    expect(await model('piper-en-us-lessac')).toMatchObject({ requires: [], available: true })
  })

  it('refuses to install without licence acceptance, then installs and frees the voices', async () => {
    const res = await handleRequest(server, 'http://bolo.test/api/v1/components/espeak-ng/install', {
      method: 'POST',
      body: '{}',
    })
    expect(res.status).toBe(422)
    const { job_id } = await api.installComponent('espeak-ng')
    let job = await api.job(job_id)
    expect(job.kind).toBe('install')
    for (let i = 0; i < 40 && job.state !== 'done'; i++) {
      await new Promise((r) => setTimeout(r, 100))
      job = await api.job(job_id)
    }
    expect(job.state).toBe('done')
    expect(await espeak()).toMatchObject({ installed: true, source: 'python-package' })
    expect((await model('kokoro-82m')).available).toBe(true)
  })

  it('answers synthesis for an unavailable voice with component_missing, over both transports', async () => {
    const err = await api.speech({ model: 'kokoro-82m', input: 'hi' }).catch((e: unknown) => e)
    expect(err).toMatchObject({ status: 409, code: 'component_missing' })
    const s = new TtsSession({ model: 'kokoro-82m', text: 'hi', silent: true })
    s.start()
    await waitFor(() => expect(s.store.get().status).toBe('error'))
    expect(s.store.get().errorCode).toBe('component_missing')
  })

  it('treats a server without the route as having no components', async () => {
    setTransport({ fetch: async () => new Response('{"detail":"Not found"}', { status: 404 }) })
    expect(await api.components()).toEqual({ components: [] })
  })
})

describe('missingComponent', () => {
  const c = { id: 'espeak-ng', name: 'eSpeak NG', installed: false } as Component
  it('names the unmet requirement only for unavailable models', () => {
    expect(missingComponent({ available: false, requires: ['espeak-ng'] } as CatalogModel, [c])?.name).toBe('eSpeak NG')
    expect(missingComponent({ available: true, requires: ['espeak-ng'] } as CatalogModel, [c])).toBeNull()
    expect(missingComponent({ requires: ['espeak-ng'] } as CatalogModel, [c])).toBeNull() // older server: no `available`
    expect(missingComponent(null, [c])).toBeNull()
  })
  it('still names it before the component list has loaded', () => {
    expect(missingComponent({ available: false, requires: ['espeak-ng'] } as CatalogModel, undefined)).toMatchObject({
      name: 'eSpeak NG',
      installable: false,
    })
  })
})

describe('ComponentNotice and ComponentsCard', () => {
  it('explains the component in one sentence: what it is, optional, separate, licence', async () => {
    const c = await espeak()
    expect(componentSentence(c)).toMatch(/eSpeak NG is .*optional.*not part of bolo.*GPL-3\.0-or-later\.$/)
    wrap(<ComponentNotice component={c} subject="kokoro-82m" />)
    expect(screen.getByRole('note')).toHaveTextContent('kokoro-82m needs eSpeak NG')
    expect(screen.getByRole('button', { name: 'Install eSpeak NG' })).toBeInTheDocument()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('asks for confirmation with the licence before anything is installed', async () => {
    wrap(<ComponentNotice component={await espeak()} subject="kokoro-82m" />)
    await userEvent.click(screen.getByRole('button', { name: 'Install eSpeak NG' }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveTextContent('Python package is installed into the server')
    expect(screen.getByRole('link', { name: /GPL-3\.0-or-later/ })).toHaveAttribute(
      'href',
      expect.stringContaining('espeak-ng'),
    )
    expect((await espeak()).installed).toBe(false)
    await userEvent.click(screen.getByRole('button', { name: /Accept the licence and install/ }))
    await waitFor(async () => expect((await espeak()).installed).toBe(true), { timeout: 5000 })
  })

  it('shows the reason and the manual command when the server cannot install', async () => {
    server.dispose()
    install({ noInstall: true })
    wrap(<ComponentNotice component={await espeak()} subject="kokoro-82m" />)
    expect(screen.getByRole('note')).toHaveTextContent('component installs disabled')
    expect(screen.getByLabelText('Manual install command')).toHaveTextContent('pip install')
    expect(screen.queryByRole('button', { name: 'Install eSpeak NG' })).not.toBeInTheDocument()
  })

  it('lists status, licence and how many models need it', async () => {
    wrap(<ComponentsCard />)
    const row = await screen.findByTestId('component-espeak-ng')
    expect(row).toHaveTextContent('Not installed')
    expect(row).toHaveTextContent('needed by 2 models')
    expect(screen.getByRole('link', { name: 'GPL-3.0-or-later' })).toBeInTheDocument()
  })
})
