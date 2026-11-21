import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { setApiKey, setServerBase, setTransport } from '@/lib/api/config'
import type { Pipeline, StepType } from '@/lib/pipelines'
import PipelinesPage from './pipelines'

let sent: { url: string; method: string; body: unknown }[]
let saved: Pipeline[]

const stepType = (t: Partial<StepType> & { type: string }): StepType => ({
  kind: null,
  group: 'model',
  help: '',
  inputs: [],
  options: [],
  outputs: {},
  live: false,
  live_only: false,
  ...t,
})

const TYPES: StepType[] = [
  stepType({
    type: 'stt',
    kind: 'stt',
    help: 'Speech to text.',
    inputs: [{ name: 'audio', type: 'audio', required: true, default: 'audio', help: '' }],
    options: [{ name: 'language', type: 'text', help: '', required: false, choices: [] }],
    outputs: { text: 'text', transcript: 'transcript' },
  }),
  stepType({
    type: 'template',
    group: 'text',
    help: 'Build a text from references.',
    options: [{ name: 'template', type: 'text', help: '', required: true, choices: [] }],
    outputs: { text: 'text' },
    live: true,
  }),
  stepType({
    type: 'respond',
    group: 'action',
    help: "Let the session's language model answer.",
    options: [{ name: 'instructions', type: 'text', help: '', required: false, choices: [] }],
    outputs: { text: 'text' },
    live: true,
    live_only: true,
  }),
]

const LIVE: Pipeline = { id: 'desk', name: 'Desk', mode: 'live', steps: [{ type: 'respond' }] }

const TEMPLATE: Pipeline = {
  id: 'meeting-notes',
  name: 'Meeting notes',
  steps: [
    { id: 'stt', type: 'stt' },
    { type: 'template', options: { template: 'Said: {{steps.stt.text}}' } },
  ],
}

const reply = (body: unknown, status = 200) =>
  Promise.resolve(new Response(body === undefined ? null : JSON.stringify(body), { status }))

function validation(p: Pipeline) {
  const errors = p.steps.some((s) => s.type === 'template' && !s.options?.template)
    ? [{ path: 'steps[1].options.template', message: 'a template step needs template' }]
    : []
  return {
    valid: errors.length === 0,
    errors,
    warnings: [],
    pipeline: p,
    steps: p.steps.map((s, i) => ({
      id: s.id ?? s.type,
      type: s.type,
      model: s.type === 'stt' ? 'fake-stt' : null,
      inputs: {},
      defaults: s.type === 'stt' ? { audio: ['{{input.audio}}'] } : {},
      outputs: {},
      path: `steps[${i}]`,
    })),
  }
}

beforeEach(() => {
  sent = []
  saved = []
  setApiKey(null)
  setServerBase('http://bolo.test/')
  setTransport({
    fetch: (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      const method = init?.method ?? 'GET'
      let body: unknown = init?.body
      if (typeof body === 'string') body = JSON.parse(body)
      if (body instanceof FormData) body = Object.fromEntries([...body.entries()].map(([k, v]) => [k, String(v)]))
      sent.push({ url, method, body })
      if (url.includes('pipelines/step-types')) return reply({ data: TYPES })
      if (url.includes('pipelines/templates')) return reply({ data: [TEMPLATE, LIVE] })
      if (url.endsWith('v1/pipelines/simulate')) {
        const turns = JSON.parse((body as { turns: string }).turns) as string[]
        return reply({
          object: 'pipeline.simulation',
          ended: null,
          transcript: [],
          ms: 5,
          turns: turns.map((t) => ({
            user: t,
            agent: `You said: ${t}`,
            run: {
              id: 'run_x',
              status: 'ok',
              steps: [
                {
                  id: 'respond',
                  type: 'respond',
                  model: null,
                  status: 'ok',
                  ms: 3,
                  output: { text: `You said: ${t}` },
                  error: null,
                  depth: 0,
                },
              ],
              actions: [{ action: 'respond', text: `You said: ${t}` }],
            },
          })),
        })
      }
      if (url.includes('pipelines/validate')) return reply(validation(body as Pipeline))
      if (url.includes('pipelines/runs')) return reply({ data: [] })
      if (url.endsWith('v1/pipelines/run') && method === 'POST') {
        return reply({
          id: 'run_1',
          object: 'pipeline.run',
          pipeline: null,
          version: null,
          status: 'ok',
          created_at: '2026-11-21T00:00:00Z',
          started_at: null,
          finished_at: null,
          ms: 120,
          input: {},
          steps: [
            {
              id: 'stt',
              type: 'stt',
              model: 'fake-stt',
              status: 'ok',
              ms: 80,
              output: { text: 'hello' },
              error: null,
              depth: 0,
            },
            {
              id: 'template',
              type: 'template',
              model: null,
              status: 'ok',
              ms: 0,
              output: { text: 'Said: hello' },
              error: null,
              depth: 0,
            },
          ],
          outputs: { text: 'Said: hello' },
          error: null,
          files: [],
        })
      }
      if (url.endsWith('api/v1/pipelines') && method === 'POST') {
        const doc = { ...(body as Pipeline), version: 1 }
        saved.push(doc)
        return reply(doc, 201)
      }
      if (url.endsWith('api/v1/pipelines')) return reply({ data: saved })
      if (url.includes('api/v1/catalog')) return reply({ models: [] })
      if (url.includes('api/v1/info')) return reply({ name: 'bolo', version: '1', engines: [], auto_pull: true })
      return reply({}, 404)
    },
  })
})
afterEach(() => {
  cleanup()
  setServerBase(null)
})

const wrap = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TooltipProvider>
        <MemoryRouter>
          <PipelinesPage />
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  )

describe('Pipelines page', () => {
  it('starts from a template, shows problems inline, runs a test and saves', async () => {
    const user = userEvent.setup()
    wrap()
    expect(await screen.findByText('No steps yet')).toBeInTheDocument()
    await user.click(screen.getByTestId('pipeline-template'))
    await user.click(await screen.findByRole('option', { name: 'Meeting notes' }))
    const first = await screen.findByTestId('pipeline-step-0')
    expect(within(first).getByText('Speech to text.')).toBeInTheDocument()
    expect(await within(first).findByText('fake-stt')).toBeInTheDocument() // the model the server resolved
    expect(await screen.findByTestId('pipeline-validity')).toHaveTextContent('Valid')

    // emptying the template's text is caught on its step
    const second = screen.getByTestId('pipeline-step-1')
    await user.clear(within(second).getByLabelText('template'))
    expect(await within(second).findByText(/a template step needs template/)).toBeInTheDocument()
    expect(screen.getByTestId('pipeline-validity')).toHaveTextContent('1 problem')
    await user.type(within(second).getByLabelText('template'), 'Said: {{{{steps.stt.text}}')
    await waitFor(() => expect(screen.getByTestId('pipeline-validity')).toHaveTextContent('Valid'))

    await user.type(screen.getByLabelText('Run text'), 'hello')
    await user.click(screen.getByTestId('pipeline-run'))
    const steps = await screen.findByTestId('pipeline-run-steps')
    expect(within(steps).getByTestId('run-step-template')).toHaveTextContent('Said: hello')
    expect(within(steps).getByTestId('run-step-stt')).toHaveTextContent('80 ms')
    const run = sent.find((r) => r.url.endsWith('v1/pipelines/run'))
    expect(JSON.parse((run?.body as { pipeline: string }).pipeline).steps).toHaveLength(2)

    await user.click(screen.getByRole('button', { name: /save/i }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(saved[0]).toMatchObject({ id: 'meeting-notes', steps: [{ id: 'stt', type: 'stt' }, { type: 'template' }] })
  }, 20000)

  it('adds, reorders and removes steps', async () => {
    const user = userEvent.setup()
    wrap()
    await user.click(await screen.findByTestId('pipeline-add-step'))
    await user.click(await screen.findByRole('option', { name: /^stt/ }))
    await user.click(screen.getByTestId('pipeline-add-step'))
    await user.click(await screen.findByRole('option', { name: /^template/ }))
    expect(screen.getByTestId('pipeline-step-1')).toHaveTextContent('template')
    await user.click(within(screen.getByTestId('pipeline-step-1')).getByRole('button', { name: 'Move up' }))
    expect(screen.getByTestId('pipeline-step-0')).toHaveTextContent('template')
    await user.click(within(screen.getByTestId('pipeline-step-0')).getByRole('button', { name: 'Remove step' }))
    expect(screen.getByTestId('pipeline-step-0')).toHaveTextContent('stt')
    expect(screen.queryByTestId('pipeline-step-1')).toBeNull()
  })

  it('simulates a live pipeline turn by turn and offers only per-turn steps', async () => {
    const user = userEvent.setup()
    wrap()
    await user.click(await screen.findByTestId('pipeline-template'))
    await user.click(await screen.findByRole('option', { name: 'Desk' }))
    expect(screen.getByTestId('pipeline-mode')).toHaveTextContent('each turn')
    await user.click(screen.getByTestId('pipeline-add-step'))
    expect(screen.queryByRole('option', { name: /^stt/ })).toBeNull() // not a per-turn step
    expect(screen.getByRole('option', { name: /^respond/ })).toBeInTheDocument()
    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.getByTestId('pipeline-validity')).toHaveTextContent('Valid'))
    await user.type(screen.getByLabelText('Caller turns'), 'hello{Enter}my bill')
    await user.click(screen.getByTestId('pipeline-simulate'))
    const turns = await screen.findByTestId('pipeline-sim-turns')
    expect(turns).toHaveTextContent('Agent: You said: hello')
    expect(turns).toHaveTextContent('Agent: You said: my bill')
    const sim = sent.find((r) => r.url.endsWith('v1/pipelines/simulate'))
    expect(JSON.parse((sim?.body as { turns: string }).turns)).toEqual(['hello', 'my bill'])
  }, 20000)
})
