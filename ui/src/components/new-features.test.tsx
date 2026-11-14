import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { installDescription, needsLicenceAcceptance } from '@/components/component-install'
import { PipelinePicker } from '@/components/conversation/pipeline-picker'
import { SpeakerResult } from '@/components/stt/speaker-result'
import { SpeakersPanel } from '@/components/stt/speakers-panel'
import { TooltipProvider } from '@/components/ui/tooltip'
import { WordListEditor } from '@/components/wordlists/editor'
import { OverrideText } from '@/components/wordlists/override-text'
import { WordListSelect } from '@/components/wordlists/wordlist-select'
import { api } from '@/lib/api/client'
import type { CatalogModel, Component } from '@/lib/api/types'
import { pipelineOptions } from '@/lib/capabilities'
import type { DiarizedResult } from '@/lib/diarize'
import { TtsSession } from '@/lib/tts/session'

const wrap = (ui: ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <TooltipProvider>{ui}</TooltipProvider>
    </QueryClientProvider>,
  )

afterEach(() => vi.restoreAllMocks())

const result: DiarizedResult = {
  text: 'hello there',
  duration: 10,
  segments: [
    { start: 0, end: 4, text: 'hello', speaker: 'A' },
    { start: 3, end: 6, text: 'there', speaker: 'B', overlap: true, separated: true },
  ],
  separation: {
    model: 'voxlock-int8',
    status: 'skipped',
    reason: 'no_voice_sample',
    detail: null,
    windows: [],
  },
}

describe('SpeakerResult', () => {
  it('shows speakers, overlap and separation marks, and the separation note', () => {
    wrap(<SpeakerResult result={result} />)
    expect(screen.getByTestId('separation-note')).toHaveAttribute('data-status', 'skipped')
    expect(screen.getByTestId('separation-note')).toHaveTextContent(/clean speech/)
    const segs = within(screen.getByTestId('speaker-segments')).getAllByRole('listitem')
    expect(segs).toHaveLength(2)
    expect(within(segs[1]!).getByTestId('overlap-badge')).toBeInTheDocument()
    expect(within(segs[1]!).getByTestId('separated-badge')).toBeInTheDocument()
    expect(within(segs[0]!).queryByTestId('overlap-badge')).toBeNull()
    expect(screen.getByLabelText('Who spoke when')).toHaveTextContent('1 overlapping')
  })
  it('says when nothing was transcribed', () => {
    wrap(<SpeakerResult result={{ text: '', segments: [] }} />)
    expect(screen.getByText('No speech found')).toBeInTheDocument()
  })
})

const model = (over: Partial<CatalogModel>): CatalogModel =>
  ({
    id: 'm',
    aliases: [],
    task: 'stt',
    engine: 'fake',
    family: 'F',
    languages: ['en'],
    tier: 'realtime',
    streaming: false,
    sample_rate: 16000,
    size_bytes: 1,
    params: null,
    license: 'MIT',
    gated: false,
    voices: [],
    features: [],
    installed: true,
    loaded: false,
    pulling: null,
    description: '',
    source_url: '',
    validated: 'cpu',
    ...over,
  }) as CatalogModel

describe('SpeakersPanel', () => {
  const catalog = [
    model({ id: 'pyannote-3.0-eres2net', task: 'diarization' }),
    model({ id: 'nemotron-3-diarization', task: 'diarization' }),
    model({ id: 'voxlock-int8', task: 'separation' }),
  ]
  it('sends the chosen options and shows the speaker-attributed result', async () => {
    const spy = vi.spyOn(api, 'transcribeSegments').mockResolvedValue(result)
    const user = userEvent.setup()
    wrap(
      <SpeakersPanel
        info={undefined}
        catalog={catalog}
        model={model({ id: 'whisper-base' })}
        language="auto"
        samples={[]}
      />,
    )
    const run = screen.getByTestId('speakers-run')
    expect(run).toBeDisabled()
    const file = new File([new Uint8Array(64)], 'call.wav', { type: 'audio/wav' })
    await user.upload(screen.getByLabelText('Recording to transcribe with speaker labels'), file)
    await user.click(screen.getByRole('radio', { name: 'Exactly' }))
    await user.click(screen.getByTestId('separate-toggle'))
    await user.click(run)
    await screen.findByTestId('speaker-result')
    expect(spy).toHaveBeenCalledWith(
      expect.anything(),
      'call.wav',
      expect.objectContaining({
        model: 'whisper-base',
        diarize: true,
        diarizer: 'pyannote-3.0-eres2net',
        speakers: { mode: 'exact', n: 2 },
        separate: true,
        separationModel: 'voxlock-int8',
      }),
    )
  })
  it('explains why separation is off without speaker labels', async () => {
    const user = userEvent.setup()
    wrap(<SpeakersPanel info={undefined} catalog={catalog} model={model({})} language="en" samples={[]} />)
    await user.click(screen.getByTestId('diarize-toggle'))
    expect(screen.getByTestId('separate-toggle')).toBeDisabled()
    expect(screen.getByText('Needs speaker labels.')).toBeInTheDocument()
  })
})

describe('PipelinePicker', () => {
  const options = pipelineOptions({
    capabilities: {
      realtime: {
        pipelines: {
          cascade: { available: true },
          'audio-llm': { configured: false },
          'speech-to-speech': { model: 'lfm', url: 'http://x' },
        },
      },
    },
  } as never)
  it('lets the user pick an available pipeline and greys out the rest', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    wrap(<PipelinePicker options={options} value="cascade" onChange={onChange} />)
    const radios = screen.getAllByRole('radio')
    expect(radios[1]).toBeDisabled()
    expect(screen.getByTestId('pipeline-speech-to-speech')).toHaveTextContent('LFM Open License')
    await user.click(radios[2]!)
    expect(onChange).toHaveBeenCalledWith('speech-to-speech')
  })
  it('locks every choice once audio has started', () => {
    wrap(<PipelinePicker options={options} value="cascade" onChange={() => undefined} locked />)
    expect(screen.getAllByRole('radio').every((r) => (r as HTMLInputElement).disabled)).toBe(true)
    expect(screen.getByText(/Locked for this conversation/)).toBeInTheDocument()
  })
})

describe('WordListEditor', () => {
  it('edits rows, saves TSV and shows warnings next to their rows', async () => {
    vi.spyOn(api, 'wordList').mockResolvedValue({
      name: 'brand',
      bytes: 10,
      updated: '2026-11-14T00:00:00Z',
      languages: ['en-us'],
      entries: [{ word: 'bolo', lang: 'en-us', ipa: 'ˈboʊloʊ', say: null, match: 'word', note: null }],
    })
    const put = vi.spyOn(api, 'putWordList').mockResolvedValue({
      name: 'brand',
      bytes: 10,
      updated: '',
      languages: ['en-us'],
      entries: 2,
      warnings: [{ line: 3, message: 'odd entry' }],
    })
    const user = userEvent.setup()
    wrap(<WordListEditor name="brand" helperModels={[]} onDeleted={() => undefined} />)
    expect(await screen.findByDisplayValue('bolo')).toBeInTheDocument()
    expect(screen.getByTestId('wordlist-save')).toBeDisabled()
    await user.click(screen.getByTestId('wordlist-add'))
    await user.type(screen.getByLabelText('Word, entry 1'), 'MH12')
    expect(screen.getByTestId('wordlist-save')).toBeDisabled() // needs ipa or say
    await user.type(screen.getByLabelText('Say, entry 1'), 'em aitch twelve')
    await user.click(screen.getByTestId('wordlist-save'))
    await waitFor(() => expect(put).toHaveBeenCalled())
    const tsv = put.mock.calls[0]![1]
    expect(tsv.split('\n')[1]).toBe('MH12\ten-us\t\tem aitch twelve\tword\t')
    expect(await screen.findByText('odd entry')).toBeInTheDocument()
  })
})

describe('word-list widgets', () => {
  it('orders the chosen lists', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    wrap(<WordListSelect names={['a', 'b']} value={['a', 'b']} onChange={onChange} />)
    await user.click(screen.getByTestId('wordlist-select'))
    await user.click(screen.getByRole('button', { name: 'Move b up' }))
    expect(onChange).toHaveBeenCalledWith(['b', 'a'])
  })
  it('draws IPA overrides as chips', () => {
    wrap(<OverrideText text="Call [bolo](/ˈboʊloʊ/) now" />)
    expect(screen.getByTestId('override-chip')).toHaveTextContent('bolo/ˈboʊloʊ/')
  })
})

describe('components and synthesis options', () => {
  const comp = (over: Partial<Component>): Component =>
    ({ id: 'x', name: 'X', license: 'MIT', installed: false, ...over }) as Component
  it('asks for licence acceptance only for copyleft components', () => {
    expect(needsLicenceAcceptance(comp({ id: 'espeak-ng', license: 'GPL-3.0-or-later' }))).toBe(true)
    expect(needsLicenceAcceptance(comp({ id: 'llama-server', license: 'MIT' }))).toBe(false)
    expect(needsLicenceAcceptance(comp({ id: 'uccharan-xlit-hi', license: 'CC BY-SA 4.0' }))).toBe(true)
    expect(installDescription(comp({ id: 'uccharan-xlit-hi', kind: 'language-pack' }))).toMatch(/uccharan's release/)
  })
  it('describes a binary install for speech servers', () => {
    const c = comp({
      id: 'llama-server',
      kind: 'tts-server',
      release: { version: 'b12375', url: 'https://github.com/x/y.tar.gz', sha256: 'a' },
    })
    expect(installDescription(c)).toMatch(/b12375 release from github.com, checks its SHA-256/)
    expect(installDescription(comp({ kind: 'phonemizer' }))).toMatch(/Python package/)
  })
  it('sends cloning and SSML over HTTP', () => {
    expect(TtsSession.transportFor({ transport: 'websocket' })).toBe('websocket')
    expect(TtsSession.transportFor({ transport: 'websocket', referenceAudio: 'data:x' })).toBe('http')
    expect(TtsSession.transportFor({ inputFormat: 'ssml' })).toBe('http')
  })
})
