import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { newVoiceForm } from '@/lib/api/client'
import { setServerBase, setTransport } from '@/lib/api/config'
import type { SavedVoice, ServerInfo } from '@/lib/api/types'
import { dataUrlToBlob } from './new-voice-dialog'
import { scoreText, scoreTone, voicesAnnounced } from './use-voices'
import { VoiceLibrary } from './voice-library'

/** A stand-in for the server's voice routes: enough of `/api/v1/voices` for the library. */
function fakeServer() {
  const voices: SavedVoice[] = []
  const forms: FormData[] = []
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  const make = (name: string, consent: SavedVoice['consent']['basis']): SavedVoice => ({
    id: name.toLowerCase(),
    object: 'voice',
    name,
    description: null,
    language: 'en',
    transcript: null,
    default_model: null,
    created_at: '2026-11-16T00:00:00Z',
    consent: { basis: consent, statement: 'This is my voice.', by: { auth: 'none' }, at: '2026-11-16T00:00:00Z' },
    reference: { sample_rate: 24000, duration_s: 9.5 },
    warnings: [],
    scores: {},
    audio_url: `/api/v1/voices/${name.toLowerCase()}/audio`,
  })
  const fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(String(input))
    const path = url.pathname.replace(/^\//, '')
    const method = init?.method ?? 'GET'
    if (path === 'api/v1/info') return json({ capabilities: { voices: { endpoint: '/api/v1/voices' } } })
    if (path === 'api/v1/voices' && method === 'GET') {
      return json({ voices, cloning_models: ['pocket-tts-en-llamacpp'], consent: {}, similarity: {} })
    }
    if (path === 'api/v1/voices' && method === 'POST') {
      const form = init?.body as FormData
      forms.push(form)
      if (!form.get('consent')) return json({ error: { code: 'consent_required', message: 'consent' } }, 400)
      const v = make(String(form.get('name')), form.get('consent') as 'self')
      v.warnings = ['the reference is short (3.0 s); most models clone better from 5 to 30 s of speech']
      if (form.get('score_model')) {
        v.score = { model: String(form.get('score_model')), similarity: 0.66, verdict: 'same speaker' }
      }
      voices.push(v)
      return json(v, 201)
    }
    const m = /^api\/v1\/voices\/([^/]+)(\/score)?$/.exec(path)
    if (m && method === 'DELETE') {
      voices.splice(
        voices.findIndex((v) => v.id === m[1]),
        1,
      )
      return json({ id: m[1], deleted: true })
    }
    if (m?.[2]) {
      const v = voices.find((x) => x.id === m[1])!
      const s = { model: url.searchParams.get('model')!, similarity: 0.21, verdict: 'different' as const }
      v.scores[s.model] = s
      return json(s)
    }
    return json({ error: { message: 'not found' } }, 404)
  }
  return { voices, forms, make, fetch }
}

let server: ReturnType<typeof fakeServer>
beforeEach(() => {
  server = fakeServer()
  setServerBase('http://bolo.test/')
  setTransport({ fetch: server.fetch as typeof fetch })
})
afterEach(() => {
  cleanup()
  setServerBase(null)
})

const wrap = (ui: React.ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TooltipProvider>{ui}</TooltipProvider>
    </QueryClientProvider>,
  )

describe('voice library helpers', () => {
  it('reads the capability and formats scores', () => {
    const info = (caps?: Record<string, unknown>) => ({ capabilities: caps }) as unknown as ServerInfo
    expect(voicesAnnounced(undefined)).toBeNull()
    expect(voicesAnnounced(info({ voices: {} }))).toBe(true)
    expect(voicesAnnounced(info({ providers: {} }))).toBe(false)
    expect(scoreText({ model: 'm', similarity: 0.6612, verdict: 'same speaker' })).toBe('0.66 · same speaker')
    expect(scoreText({ model: 'm', error: 'cannot clone' })).toBe('cannot clone')
    expect(scoreText(undefined)).toBe('not scored')
    expect(scoreTone({ model: 'm', similarity: 0.4, verdict: 'similar' })).toBe('warning')
  })

  it('builds the creation form with the consent statement', () => {
    const form = newVoiceForm({
      name: 'Priya',
      consent: 'permission',
      files: [{ blob: new Blob(['x']), name: 'a.wav' }],
      scoreModel: 'pocket-tts-en-llamacpp',
    })
    expect(form.get('consent')).toBe('permission')
    expect(form.get('score_model')).toBe('pocket-tts-en-llamacpp')
    expect((form.get('file') as File).name).toBe('a.wav')
  })

  it('turns a data URL back into a blob', async () => {
    const blob = dataUrlToBlob('data:audio/wav;base64,UklGRg==')
    expect(blob.type).toBe('audio/wav')
    expect(new TextDecoder().decode(await blob.arrayBuffer())).toBe('RIFF')
  })
})

describe('VoiceLibrary', () => {
  it('lists voices, uses one and deletes one after a confirmation', async () => {
    server.voices.push(server.make('Asha', 'permission'))
    let selected: string | null = null
    wrap(
      <VoiceLibrary
        model="pocket-tts-en-llamacpp"
        canUse
        selected={null}
        onSelect={(id) => {
          selected = id
        }}
      />,
    )
    const row = await screen.findByTestId('voice-asha')
    expect(within(row).getByText("I have the speaker's permission")).toBeInTheDocument()
    expect(within(row).getByTestId('voice-score-asha')).toHaveTextContent('not scored')
    await userEvent.click(within(row).getByTestId('use-voice-asha'))
    expect(selected).toBe('asha')
    await userEvent.click(within(row).getByRole('button', { name: 'Score Asha with pocket-tts-en-llamacpp' }))
    await waitFor(() => expect(screen.getByTestId('voice-score-asha')).toHaveTextContent('0.21 · different'))
    await userEvent.click(within(row).getByRole('button', { name: 'Delete Asha' }))
    await userEvent.click(within(row).getByRole('button', { name: 'Confirm deleting Asha' }))
    await waitFor(() => expect(screen.queryByTestId('voice-asha')).not.toBeInTheDocument())
  })

  it('says which models can use the voices when this one cannot clone', async () => {
    wrap(<VoiceLibrary model="kokoro-82m" canUse={false} selected={null} onSelect={() => undefined} />)
    await waitFor(() =>
      expect(screen.getByTestId('voice-library-hint')).toHaveTextContent(
        'kokoro-82m does not clone voices. Choose one that does to use these: pocket-tts-en-llamacpp.',
      ),
    )
    expect(await screen.findByText('No saved voices yet')).toBeInTheDocument()
  })

  it('saves a voice only with a consent statement ticked, and shows the score', async () => {
    const user = userEvent.setup()
    wrap(<VoiceLibrary model="pocket-tts-en-llamacpp" canUse selected={null} onSelect={() => undefined} />)
    await user.click(await screen.findByRole('button', { name: /New voice/ }))
    const dialog = await screen.findByTestId('new-voice-dialog')
    const save = within(dialog).getByTestId('new-voice-save')
    await user.type(within(dialog).getByTestId('new-voice-name'), 'Priya')
    await user.upload(
      within(dialog).getByTestId('new-voice-file'),
      new File(['RIFF'], 'priya.wav', { type: 'audio/wav' }),
    )
    expect(within(dialog).getByTestId('new-voice-source')).toHaveTextContent('priya.wav')
    expect(save).toBeDisabled() // no consent yet
    await user.click(within(dialog).getByRole('radio', { name: 'This is my voice' }))
    expect(save).toBeDisabled() // chosen, not confirmed
    await user.click(within(dialog).getByTestId('new-voice-confirm'))
    expect(save).toBeEnabled()
    await user.click(save)
    expect(await within(dialog).findByTestId('new-voice-score')).toHaveTextContent(
      'pocket-tts-en-llamacpp: 0.66 · same speaker',
    )
    expect(within(dialog).getByTestId('new-voice-result')).toHaveTextContent('the reference is short')
    const form = server.forms[0]!
    expect(form.get('consent')).toBe('self')
    expect(form.get('score_model')).toBe('pocket-tts-en-llamacpp')
    await user.click(within(dialog).getByRole('button', { name: 'Done' }))
    expect(await screen.findByTestId('voice-priya')).toBeInTheDocument()
  })

  it('opens the save dialog with a recording handed over by the clone panel', async () => {
    const pending = { blob: new Blob(['RIFF']), name: 'Recorded here.wav', seconds: 7.5, transcript: 'hello there' }
    wrap(
      <VoiceLibrary
        model="pocket-tts-en-llamacpp"
        canUse
        selected={null}
        onSelect={() => undefined}
        pending={pending}
      />,
    )
    const dialog = await screen.findByTestId('new-voice-dialog')
    expect(within(dialog).getByTestId('new-voice-source')).toHaveTextContent('Recorded here.wav · 7.5 s')
    expect(within(dialog).getByDisplayValue('hello there')).toBeInTheDocument()
  })
})
