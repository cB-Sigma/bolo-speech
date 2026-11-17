import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { setServerBase, setTransport } from '@/lib/api/config'
import { VoiceCompare } from './voice-compare'

let sent: { name: string; model: string | null }[] = []
let model = 'pyannote-3.0-eres2net'

beforeEach(() => {
  sent = []
  model = 'pyannote-3.0-eres2net'
  setServerBase('http://bolo.test/')
  setTransport({
    fetch: (async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname
      const json = (b: unknown) => new Response(JSON.stringify(b), { headers: { 'Content-Type': 'application/json' } })
      if (path === '/api/v1/catalog') return json({ models: [] })
      if (path === '/v1/audio/speech/embedding') {
        const form = init?.body as FormData
        const name = (form.get('file') as File).name
        sent.push({ name, model: form.get('model') as string | null })
        const embedding = name === 'a.wav' ? [1, 0, 0] : [0.8, 0.6, 0]
        return json({ object: 'list', data: [{ object: 'embedding', embedding, index: 0 }], model })
      }
      return new Response('{}', { status: 404 })
    }) as typeof fetch,
  })
})
afterEach(() => setServerBase(null))

const wrap = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <VoiceCompare />
    </QueryClientProvider>,
  )

describe('VoiceCompare', () => {
  it('embeds both recordings and shows their cosine with a verdict', async () => {
    const user = userEvent.setup()
    wrap()
    const run = screen.getByTestId('compare-run')
    expect(run).toBeDisabled()
    await user.upload(
      within(screen.getByTestId('compare-a')).getByLabelText('First voice: recording'),
      new File(['x'], 'a.wav'),
    )
    await user.upload(
      within(screen.getByTestId('compare-b')).getByLabelText('Second voice: recording'),
      new File(['y'], 'b.wav'),
    )
    await user.click(run)
    const result = await screen.findByTestId('compare-result')
    expect(result).toHaveTextContent('0.80')
    expect(result).toHaveTextContent('same speaker')
    expect(sent.map((s) => s.name).sort()).toEqual(['a.wav', 'b.wav'])
    expect(sent.every((s) => s.model === null)).toBe(true)
  })

  it('gives no verdict for a model the thresholds were not measured with', async () => {
    model = 'other-embedder'
    const user = userEvent.setup()
    wrap()
    await user.upload(screen.getByLabelText('First voice: recording'), new File(['x'], 'a.wav'))
    await user.upload(screen.getByLabelText('Second voice: recording'), new File(['y'], 'b.wav'))
    await user.click(screen.getByTestId('compare-run'))
    await waitFor(() => expect(screen.getByTestId('compare-result')).toHaveTextContent('cosine similarity'))
    expect(screen.getByTestId('compare-result')).toHaveTextContent('another model has another scale')
  })
})
