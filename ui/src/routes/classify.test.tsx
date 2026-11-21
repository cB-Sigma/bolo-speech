import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { audioFiles, filesFromDrop, parseLabels, type DropEntry } from '@/lib/classify'
import type { AudioClassifier, CatalogModel } from '@/lib/api/types'

const models = [{ id: 'clap-htsat-unfused', task: 'audio-classify', features: ['zero-shot'], installed: true }]
const trained = [
  {
    id: 'voicemail',
    labels: ['person', 'voicemail'],
    encoder: 'clap-htsat-unfused',
    examples: { person: 30, voicemail: 30 },
    cross_validation: { accuracy: 0.95, folds: 5, per_class: {}, confusion: [] },
    warnings: [],
  },
] as unknown as AudioClassifier[]
const audioClassify = vi.fn()
const trainClassifier = vi.fn()

vi.mock('@/hooks/use-api', () => ({
  useInfo: () => ({ data: { capabilities: { audio_classify: { default_model: 'clap-htsat-unfused' } } } }),
  useModels: (task: string) => ({ models: (task === 'audio-classify' ? models : []) as unknown as CatalogModel[] }),
  describeError: (e: unknown) => String(e),
}))
vi.mock('@/lib/api/client', () => ({
  api: {
    classifiers: () => Promise.resolve(trained),
    audioClassify: (...a: unknown[]) => audioClassify(...a),
    trainClassifier: (...a: unknown[]) => trainClassifier(...a),
    deleteClassifier: () => Promise.resolve(),
  },
}))
vi.mock('@/components/catalog-select', () => ({
  CatalogSelect: ({ testId }: { testId: string }) => <div data-testid={testId} />,
}))
vi.mock('@/components/voices/voice-compare', () => ({
  ClipSlot: ({ onChange, testId }: { onChange: (c: unknown) => void; testId: string }) => (
    <button data-testid={testId} onClick={() => onChange({ blob: new Blob(['x']), name: 'a.wav', seconds: 1 })}>
      clip
    </button>
  ),
}))
vi.mock('@/components/layout/app-shell', () => ({
  Page: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

function wav(name: string): File {
  return new File(['x'], name, { type: name.endsWith('.wav') ? 'audio/wav' : '' })
}

describe('classify helpers', () => {
  it('reads labels from lines and commas, without blanks or repeats', () => {
    expect(parseLabels('hold music\n a dog barking, speech \n\nspeech')).toEqual([
      'hold music',
      'a dog barking',
      'speech',
    ])
  })

  it('keeps audio files only', () => {
    const files = [wav('a.wav'), wav('b.flac'), wav('notes.txt'), wav('.DS_Store')]
    expect(audioFiles(files).map((f) => f.name)).toEqual(['a.wav', 'b.flac'])
  })

  it('walks dropped folders', async () => {
    const file = (name: string): DropEntry => ({ isFile: true, isDirectory: false, name, file: (ok) => ok(wav(name)) })
    let reads = 0
    const dir: DropEntry = {
      isFile: false,
      isDirectory: true,
      name: 'voicemail',
      createReader: () => ({
        // two batches, then none: readEntries is called until it gives nothing
        readEntries: (ok) => ok(reads++ === 0 ? [file('1.wav'), file('x.txt')] : reads === 2 ? [file('2.mp3')] : []),
      }),
    }
    const dt = { items: [{ webkitGetAsEntry: () => dir }], files: [] } as unknown as DataTransfer
    expect((await filesFromDrop(dt)).map((f) => f.name)).toEqual(['1.wav', '2.mp3'])
  })
})

describe('Classify page', () => {
  it('classifies against typed labels and shows the scores', async () => {
    audioClassify.mockResolvedValue({
      object: 'audio.classification',
      model: 'clap-htsat-unfused',
      classifier: null,
      duration: 3,
      label: 'music',
      score: 0.9,
      prompt: 'This is a sound of {}.',
      scores: [
        { label: 'music', score: 0.9, similarity: 0.4 },
        { label: 'speech', score: 0.1, similarity: 0.1 },
      ],
    })
    const { default: ClassifyPage } = await import('./classify')
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ClassifyPage />
      </QueryClientProvider>,
    )
    await userEvent.click(screen.getByTestId('classify-clip'))
    const labels = screen.getByTestId('classify-labels')
    await userEvent.clear(labels)
    await userEvent.type(labels, 'music{enter}speech')
    await userEvent.click(screen.getByTestId('classify-run'))
    await waitFor(() => expect(screen.getAllByTestId('classify-score')).toHaveLength(2))
    expect(audioClassify.mock.calls[0]?.[2]).toMatchObject({ labels: ['music', 'speech'] })
    expect(screen.getByTestId('classify-result')).toHaveTextContent('similarity 0.40')
    expect(await screen.findByTestId('classify-list')).toHaveTextContent('voicemail')
  })

  it('refuses to train until every class has a name and clips', async () => {
    const { default: ClassifyPage, trainingExamples } = await import('./classify')
    expect(trainingExamples([{ label: 'a', files: [wav('1.wav')] }])).toMatch(/two classes/)
    expect(
      trainingExamples([
        { label: 'a', files: [wav('1.wav')] },
        { label: 'b', files: [] },
      ]),
    ).toMatch(/no clips/)
    const ok = trainingExamples([
      { label: 'a', files: [wav('1.wav'), wav('2.wav')] },
      { label: 'b', files: [wav('3.wav')] },
    ])
    expect(Array.isArray(ok) && ok.map((e) => e.label)).toEqual(['a', 'a', 'b'])
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ClassifyPage />
      </QueryClientProvider>,
    )
    await userEvent.click(screen.getByTestId('classify-train'))
    expect(screen.getByTestId('classify-train-error')).toHaveTextContent('two classes')
    expect(trainClassifier).not.toHaveBeenCalled()
  })
})
