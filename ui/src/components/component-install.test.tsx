import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
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
    await waitFor(async () => expect((await espeak()).installed).toBe(true))
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

describe('language packs', () => {
  beforeEach(() => localStorage.removeItem('bolo.languagePacks.open'))

  it('are one collapsed row with a summary, opened with View and remembered', async () => {
    const { unmount } = wrap(<ComponentsCard />)
    const summary = await screen.findByTestId('language-packs-summary')
    expect(summary).toHaveTextContent('2 of 35 installed · 3 needed by your models, 2 missing')
    expect(screen.queryByTestId('language-pack-list')).not.toBeInTheDocument()
    expect(screen.queryByTestId('component-uccharan-de')).not.toBeInTheDocument()
    expect(screen.getByTestId('component-espeak-ng')).toBeInTheDocument()
    const toggle = screen.getByTestId('language-packs-toggle')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await userEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByTestId('language-pack-list')).toBeInTheDocument()
    unmount()
    wrap(<ComponentsCard />)
    expect(await screen.findByTestId('language-pack-list')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Hide/ }))
    expect(screen.queryByTestId('language-pack-list')).not.toBeInTheDocument()
  })

  it('lists needed packs first, filters, searches and pages eight at a time', async () => {
    localStorage.setItem('bolo.languagePacks.open', '1')
    wrap(<ComponentsCard />)
    const list = await screen.findByTestId('language-pack-list')
    // "Needed" is the default while models here need packs: missing ones first
    let rows = within(list).getAllByRole('listitem')
    expect(rows.map((r) => r.dataset.testid)).toEqual([
      'component-uccharan-de',
      'component-uccharan-it',
      'component-uccharan-en-gb',
    ])
    expect(rows[0]).toHaveTextContent('needed by 3 models (1 installed)')
    expect(screen.getByTestId('install-needed-packs')).toHaveTextContent('Install needed (2)')

    await userEvent.click(screen.getByTestId('packs-filter-all'))
    rows = within(list).getAllByRole('listitem')
    expect(rows).toHaveLength(8)
    expect(screen.getByTestId('packs-range')).toHaveTextContent('1–8 of 35')
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: 'Next page' }))
    expect(screen.getByTestId('packs-range')).toHaveTextContent('9–16 of 35')
    for (let i = 0; i < 3; i++) await userEvent.click(screen.getByRole('button', { name: 'Next page' }))
    expect(screen.getByTestId('packs-range')).toHaveTextContent('33–35 of 35')
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled()
    // built-in packs say so, last
    expect(screen.getByTestId('component-uccharan-hi')).toHaveTextContent('built in, not needed · used by 5 models')

    await userEvent.type(screen.getByRole('searchbox', { name: 'Search language packs' }), 'german')
    expect(screen.getByTestId('packs-range')).toHaveTextContent('1–1 of 1')
    expect(within(list).getAllByRole('listitem')[0]).toHaveTextContent('German')

    await userEvent.clear(screen.getByRole('searchbox'))
    await userEvent.click(screen.getByTestId('packs-filter-installed'))
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((r) => r.dataset.testid),
    ).toEqual(['component-uccharan-en-gb', 'component-uccharan-fr'])
  })

  it('installs every needed pack, and removes one', async () => {
    localStorage.setItem('bolo.languagePacks.open', '1')
    wrap(<ComponentsCard />)
    await userEvent.click(await screen.findByTestId('install-needed-packs'))
    await waitFor(() =>
      expect(screen.getByTestId('language-packs-summary')).toHaveTextContent('4 of 35 installed · 3 needed'),
    )
    expect(screen.getByTestId('language-packs-summary')).not.toHaveTextContent('missing')
    expect(screen.queryByTestId('install-needed-packs')).not.toBeInTheDocument()
    await userEvent.click(screen.getByTestId('remove-uccharan-de'))
    await waitFor(() =>
      expect(screen.getByTestId('language-packs-summary')).toHaveTextContent(
        '3 of 35 installed · 3 needed by your models, 1 missing',
      ),
    )
  })
})

describe('PyTorch builds', () => {
  const torch: Component = {
    id: 'torch-cuda',
    name: 'PyTorch, CUDA build',
    kind: 'pytorch',
    installed: false,
    source: null,
    version: '2.15.0+cpu',
    license: 'BSD-3-Clause',
    license_url: null,
    required_by_count: 40,
    installable: true,
    install_blocked_reason: null,
    manual_command: 'bolo torch use cuda',
    description: 'The CUDA build of PyTorch for the bolo-torch plugin.',
    size_bytes: 2_500_000_000,
    gpu: 'NVIDIA GeForce RTX 4060 Laptop GPU',
    envs: [
      {
        plugins: ['plugin:torch'],
        python: 'C:\\bolo-env\\Scripts\\python.exe',
        torch: '2.15.0+cpu',
        build: 'cpu',
        family: 'cpu',
        label: 'CPU build, GPU not used',
      },
    ],
  }

  it("shows each environment's build and installs the GPU version after a confirmation", async () => {
    const asked: string[] = []
    setTransport({
      fetch: async (input, init) => {
        const url = String(input)
        if (url.endsWith('api/v1/components'))
          return new Response(JSON.stringify({ components: [torch] }), {
            headers: { 'content-type': 'application/json' },
          })
        if (url.includes('api/v1/components/torch-cuda/install')) {
          asked.push(url)
          return new Response(JSON.stringify({ job_id: 'job-1' }), {
            status: 202,
            headers: { 'content-type': 'application/json' },
          })
        }
        if (url.includes('api/v1/jobs/job-1'))
          return new Response(JSON.stringify({ id: 'job-1', state: 'done', progress: 1 }), {
            headers: { 'content-type': 'application/json' },
          })
        return handleRequest(server, input, init)
      },
      WebSocket: createMockWebSocket(server),
    })
    wrap(<ComponentsCard bare detailed kind="pytorch" />)
    const row = await screen.findByTestId('component-torch-cuda')
    expect(within(row).getByText(/2\.15\.0\+cpu: CPU build, GPU not used/)).toBeTruthy()
    await userEvent.click(within(row).getByRole('button', { name: /Install GPU version/ }))
    const dialog = screen.getByTestId('install-dialog')
    expect(within(dialog).getByText(/about 2\.5 GB to download/)).toBeTruthy()
    expect(componentSentence(torch)).toContain('use the NVIDIA GeForce RTX 4060 Laptop GPU instead of the CPU')
    await userEvent.click(within(dialog).getByTestId('install-confirm'))
    await waitFor(() => expect(asked).toHaveLength(1))
  })
})
