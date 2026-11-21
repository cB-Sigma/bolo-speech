import { useQuery, useQueryClient } from '@tanstack/react-query'
import { FolderOpen, GraduationCap, Loader2, Plus, Tags, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { CatalogSelect } from '@/components/catalog-select'
import { Page } from '@/components/layout/app-shell'
import { Empty, Panel } from '@/components/panel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ClipSlot, type Clip } from '@/components/voices/voice-compare'
import { describeError, useInfo, useModels } from '@/hooks/use-api'
import { api } from '@/lib/api/client'
import type { AudioClassification, AudioClassifier, ClassScore } from '@/lib/api/types'
import { IntentPanels } from '@/components/classify/intent-panels'
import { autoPull, intentOffered } from '@/lib/capabilities'
import { audioFiles, filesFromDrop, folderName, parseLabels } from '@/lib/classify'
import { cn } from '@/lib/utils'

type Mode = 'labels' | 'classifier'

/** One class of the training form: its label and its example clips. */
export interface DraftClass {
  label: string
  files: File[]
}

/** The examples of the training form as the request wants them, or why they cannot be sent yet. */
export function trainingExamples(classes: DraftClass[]): { label: string; file: File }[] | string {
  const named = classes.filter((c) => c.files.length || c.label.trim())
  if (named.length < 2) return 'Add at least two classes, each with its clips'
  const labels = named.map((c) => c.label.trim())
  if (labels.some((l) => !l)) return 'Every class needs a name'
  if (new Set(labels).size !== labels.length) return 'Two classes have the same name'
  const empty = named.find((c) => !c.files.length)
  if (empty) return `“${empty.label.trim()}” has no clips`
  return named.flatMap((c) => c.files.map((file) => ({ label: c.label.trim(), file })))
}

function ScoreBars({ scores }: { scores: ClassScore[] }) {
  return (
    <ol className="grid gap-1.5">
      {scores.map((s) => (
        <li key={s.label} className="grid gap-0.5 text-[13px]" data-testid="classify-score">
          <span className="flex justify-between gap-3">
            <span className="truncate">{s.label}</span>
            <span className="shrink-0 text-muted-foreground tabular-nums">
              {(s.score * 100).toFixed(1)}%{s.similarity !== undefined && ` · similarity ${s.similarity.toFixed(2)}`}
            </span>
          </span>
          <span className="h-1.5 rounded bg-muted">
            <span className="block h-1.5 rounded bg-primary" style={{ width: `${Math.round(s.score * 100)}%` }} />
          </span>
        </li>
      ))}
    </ol>
  )
}

/** A class of the training form: a name, and clips dropped (files or whole folders) or chosen. */
function ClassDrop({
  index,
  value,
  onChange,
  onRemove,
  disabled,
}: {
  index: number
  value: DraftClass
  onChange: (c: DraftClass) => void
  onRemove: () => void
  disabled?: boolean
}) {
  const folder = useRef<HTMLInputElement>(null)
  const files = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const add = (picked: File[]) => {
    const audio = audioFiles(picked)
    if (!audio.length) return
    const label = value.label.trim() || folderName(audio) || ''
    onChange({ label, files: [...value.files, ...audio] })
  }
  return (
    <div
      className={cn(
        'grid gap-2 rounded-lg border border-dashed border-border p-2.5',
        over && 'border-primary bg-primary/5',
      )}
      data-testid={`classify-class${index + 1}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        if (!disabled) void filesFromDrop(e.dataTransfer).then(add)
      }}
    >
      <div className="flex items-center gap-2">
        <Input
          value={value.label}
          onChange={(e) => onChange({ ...value, label: e.target.value })}
          placeholder={`Class ${index + 1} (e.g. voicemail)`}
          aria-label={`Name of class ${index + 1}`}
          disabled={disabled}
        />
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Remove class ${index + 1}`}
          onClick={onRemove}
          disabled={disabled}
        >
          <Trash2 />
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span data-testid={`classify-class${index + 1}-count`}>
          {value.files.length ? `${value.files.length} clips` : 'Drop a folder or clips here, or'}
        </span>
        <Button variant="outline" size="sm" onClick={() => folder.current?.click()} disabled={disabled}>
          <FolderOpen /> Folder
        </Button>
        <Button variant="outline" size="sm" onClick={() => files.current?.click()} disabled={disabled}>
          Files
        </Button>
        {value.files.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => onChange({ ...value, files: [] })} disabled={disabled}>
            Clear
          </Button>
        )}
        <input
          ref={folder}
          type="file"
          className="hidden"
          // a whole folder: every audio file in it
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
          multiple
          onChange={(e) => {
            add(Array.from(e.target.files ?? []))
            e.target.value = ''
          }}
        />
        <input
          ref={files}
          type="file"
          className="hidden"
          accept="audio/*"
          multiple
          data-testid={`classify-class${index + 1}-files`}
          onChange={(e) => {
            add(Array.from(e.target.files ?? []))
            e.target.value = ''
          }}
        />
      </div>
    </div>
  )
}

/** Classification against labels you write (zero-shot, CLAP: `POST /v1/audio/classify`), and classifiers trained
 * here from your own clips (`/v1/audio/classifiers`). */
export default function ClassifyPage() {
  const info = useInfo()
  const qc = useQueryClient()
  const clapModels = useModels('audio-classify').models
  const speakerModels = useModels('speaker-embedding').models
  const taggers = useModels('audio-tagging').models
  const zeroShot = clapModels.filter((m) => !m.features.includes('trained'))
  const list = useQuery({ queryKey: ['classifiers'], queryFn: ({ signal }) => api.classifiers(signal) })
  const trained = list.data ?? []
  const defaultId = info.data?.capabilities?.audio_classify?.default_model ?? null

  const [mode, setMode] = useState<Mode>('labels')
  const [clip, setClip] = useState<Clip | null>(null)
  const [labels, setLabels] = useState('a person speaking\nmusic\na dog barking\nsilence')
  const [model, setModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [classifier, setClassifier] = useState('')
  const [windowS, setWindowS] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AudioClassification | null>(null)

  const [name, setName] = useState('')
  const [encoder, setEncoder] = useState('')
  const [head, setHead] = useState<'logistic' | 'mlp'>('logistic')
  const [classes, setClasses] = useState<DraftClass[]>([
    { label: '', files: [] },
    { label: '', files: [] },
  ])
  const [training, setTraining] = useState(false)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [made, setMade] = useState<AudioClassifier | null>(null)

  const run = async () => {
    if (!clip) {
      setError('Upload or record a clip first')
      return
    }
    const picked = parseLabels(labels)
    if (mode === 'labels' && picked.length < 2) {
      setError('Write at least two labels')
      return
    }
    if (mode === 'classifier' && !classifier) {
      setError('Choose a trained classifier')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const w = Number(windowS)
      const body = await api.audioClassify(clip.blob, clip.name, {
        ...(mode === 'labels'
          ? { labels: picked, model: model || undefined, prompt: prompt.trim() ? prompt : undefined }
          : { classifier }),
        window: w > 0 ? w : undefined,
      })
      setResult(body)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const train = async () => {
    const examples = trainingExamples(classes)
    if (typeof examples === 'string') {
      setTrainError(examples)
      return
    }
    if (!name.trim()) {
      setTrainError('Give the classifier a name')
      return
    }
    setTraining(true)
    setTrainError(null)
    try {
      const body = await api.trainClassifier(name.trim(), examples, {
        encoder: encoder || undefined,
        head,
        replace: trained.some((c) => c.id === name.trim().toLowerCase()),
      })
      setMade(body)
      setClassifier(body.id)
      await qc.invalidateQueries({ queryKey: ['classifiers'] })
      await qc.invalidateQueries({ queryKey: ['catalog'] })
    } catch (err) {
      setTrainError(describeError(err))
    } finally {
      setTraining(false)
    }
  }

  const remove = async (id: string) => {
    try {
      await api.deleteClassifier(id)
      if (classifier === id) setClassifier('')
      await qc.invalidateQueries({ queryKey: ['classifiers'] })
      await qc.invalidateQueries({ queryKey: ['catalog'] })
    } catch (err) {
      setTrainError(describeError(err))
    }
  }

  const encoders = [...zeroShot, ...speakerModels, ...taggers]

  return (
    <Page title="Classify">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,6fr)_minmax(0,6fr)]">
        <Panel
          title="Classify a clip"
          description="Against labels you write (zero-shot), or with a classifier trained on your clips"
          bodyClassName="grid gap-3"
        >
          <ClipSlot label="Clip" clip={clip} onChange={setClip} disabled={busy} testId="classify-clip" />
          <Select value={mode} onValueChange={(v) => setMode(v as Mode)} disabled={busy}>
            <SelectTrigger className="h-9" aria-label="Classify with" data-testid="classify-mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper">
              <SelectItem value="labels">My labels (zero-shot)</SelectItem>
              <SelectItem value="classifier">A trained classifier</SelectItem>
            </SelectContent>
          </Select>
          {mode === 'labels' ? (
            <>
              <Textarea
                value={labels}
                onChange={(e) => setLabels(e.target.value)}
                placeholder="One label per line, in plain words: hold music, an angry voice, a dog barking"
                aria-label="Labels"
                rows={4}
                disabled={busy}
                data-testid="classify-labels"
              />
              <CatalogSelect
                models={zeroShot}
                value={model}
                onChange={setModel}
                label="CLAP model"
                defaultLabel={defaultId ? `Server default (${defaultId})` : 'Server default'}
                disabled={busy}
                autoPull={autoPull(info.data)}
                testId="classify-model"
              />
              <Input
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Prompt, {} marks the label (default: This is a sound of {}.)"
                aria-label="Prompt"
                disabled={busy}
              />
            </>
          ) : (
            <Select value={classifier} onValueChange={setClassifier} disabled={busy || !trained.length}>
              <SelectTrigger className="h-9" aria-label="Trained classifier" data-testid="classify-classifier">
                <SelectValue placeholder={trained.length ? 'Choose a classifier' : 'No trained classifiers yet'} />
              </SelectTrigger>
              <SelectContent position="popper">
                {trained.map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {c.id} ({c.labels.join(', ')})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={windowS}
              onChange={(e) => setWindowS(e.target.value)}
              placeholder="Window, s (optional)"
              aria-label="Window in seconds"
              inputMode="decimal"
              className="w-44"
              disabled={busy}
            />
            <Button size="sm" onClick={() => void run()} disabled={busy} data-testid="classify-run">
              {busy ? <Loader2 className="animate-spin" /> : <Tags />} Classify
            </Button>
          </div>
        </Panel>
        <Panel
          title="Scores"
          description={
            result
              ? result.classifier
                ? `${result.classifier} on ${result.model}`
                : `${result.model} · ${result.prompt ?? ''}`
              : 'Zero-shot scores sum to 1 over your labels: add a catch-all label to notice when none fits'
          }
          data-testid="classify-result"
        >
          {error && (
            <Alert variant="destructive" className="mb-3" data-testid="classify-error">
              <AlertTitle>Not classified</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {result ? (
            <div className="grid gap-4">
              <ScoreBars scores={result.scores} />
              {result.segments && (
                <ol className="grid gap-1 text-xs" data-testid="classify-segments">
                  {result.segments.map((s) => (
                    <li key={s.start} className="flex justify-between gap-3">
                      <span className="text-muted-foreground tabular-nums">
                        {s.start.toFixed(1)}–{s.end.toFixed(1)} s
                      </span>
                      <span className="truncate">
                        {s.label} ({(s.score * 100).toFixed(0)}%)
                      </span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          ) : (
            <Empty icon={Tags} title="Nothing classified yet" className="py-8">
              Upload or record a clip, write the labels you want to tell apart, and classify.
            </Empty>
          )}
        </Panel>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,6fr)_minmax(0,6fr)]">
        <Panel
          title="Train a classifier"
          description="A few dozen clips per class; trained in seconds on the server's CPU and saved as a local model"
          bodyClassName="grid gap-3"
        >
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_9rem]">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Name, e.g. voicemail"
              aria-label="Classifier name"
              disabled={training}
              data-testid="classify-train-name"
            />
            <Select value={encoder || '__default'} onValueChange={(v) => setEncoder(v === '__default' ? '' : v)}>
              <SelectTrigger className="h-9" aria-label="Encoder" disabled={training}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper">
                <SelectItem value="__default">Encoder: {defaultId ?? 'server default'}</SelectItem>
                {encoders.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.id} ({m.task})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={head} onValueChange={(v) => setHead(v as 'logistic' | 'mlp')}>
              <SelectTrigger className="h-9" aria-label="Head" disabled={training}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper">
                <SelectItem value="logistic">Logistic</SelectItem>
                <SelectItem value="mlp">Small MLP</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {classes.map((c, i) => (
            <ClassDrop
              key={i}
              index={i}
              value={c}
              disabled={training}
              onChange={(v) => setClasses((cs) => cs.map((x, j) => (j === i ? v : x)))}
              onRemove={() => setClasses((cs) => cs.filter((_, j) => j !== i))}
            />
          ))}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={training}
              onClick={() => setClasses((cs) => [...cs, { label: '', files: [] }])}
            >
              <Plus /> Class
            </Button>
            <Button size="sm" onClick={() => void train()} disabled={training} data-testid="classify-train">
              {training ? <Loader2 className="animate-spin" /> : <GraduationCap />} Train
            </Button>
          </div>
          {trainError && (
            <Alert variant="destructive" data-testid="classify-train-error">
              <AlertTitle>Not trained</AlertTitle>
              <AlertDescription>{trainError}</AlertDescription>
            </Alert>
          )}
          {made && (
            <div className="grid gap-1 text-[13px]" data-testid="classify-trained">
              <span className="font-medium">
                {made.id}: cross-validated accuracy{' '}
                {made.cross_validation ? `${(made.cross_validation.accuracy * 100).toFixed(1)}%` : 'unknown'}
              </span>
              {made.cross_validation && (
                <span className="text-muted-foreground">
                  {Object.entries(made.cross_validation.per_class)
                    .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
                    .join(' · ')}
                </span>
              )}
              {made.warnings.map((w) => (
                <span key={w} className="text-warning-text">
                  {w}
                </span>
              ))}
            </div>
          )}
        </Panel>
        <Panel title="Trained classifiers" description="Also listed under Models, task audio-classify">
          {trained.length ? (
            <ul className="grid gap-2 text-[13px]" data-testid="classify-list">
              {trained.map((c) => (
                <li key={c.id} className="flex items-center justify-between gap-3">
                  <span className="min-w-0">
                    <span className="font-medium">{c.id}</span>{' '}
                    <span className="text-muted-foreground">
                      {Object.entries(c.examples)
                        .map(([k, v]) => `${k} (${v})`)
                        .join(', ')}{' '}
                      on {c.encoder}
                      {c.cross_validation && ` · ${(c.cross_validation.accuracy * 100).toFixed(0)}%`}
                    </span>
                  </span>
                  <span className="flex shrink-0 gap-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setMode('classifier')
                        setClassifier(c.id)
                      }}
                    >
                      Use
                    </Button>
                    <Button variant="ghost" size="icon" aria-label={`Delete ${c.id}`} onClick={() => void remove(c.id)}>
                      <Trash2 />
                    </Button>
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <Empty icon={GraduationCap} title="No trained classifiers" className="py-8">
              Name two or more classes, drop a folder of clips on each, and train.
            </Empty>
          )}
        </Panel>
        {intentOffered(info.data) && <IntentPanels />}
      </div>
    </Page>
  )
}
